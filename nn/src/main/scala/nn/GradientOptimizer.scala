package nn

import dimwit.*
import dimwit.Conversions.given
import dimwit.autodiff.Grad
import dimwit.jax.Jax
import dimwit.jax.Jit

/** Gradient optimizer interface with functional state management.
  *
  * This API provides the following two styles of usage:
  *
  * 1. **Simple iterator API**
  *    {{{
  *    val optimizer = GradientDescent(lr = 0.1)
  *    optimizer.iterate(initParams)(gradientFunction).take(1000).foreach(...)
  *    }}}
  *
  * 2. **Functional state threading with foldLeft** for minibatch training
  *    {{{
  *    val optimizer = GradientDescent(lr = 0.1)
  *    val (finalParams, finalState) = batches.foldLeft((initParams, optimizer.init(initParams))):
  *      case ((params, state), batch) =>
  *        val grads = Autodiff.grad(loss(batch))(params)
  *        optimizer.update(grads, params, state)
  *    }}}
  */
trait GradientOptimizer:
  type State[_]

  // Core API
  def init[Params: ToPyTree: FloatTensorTree](params: Params): State[Params]
  def update[Params: ToPyTree: FloatTensorTree](gradients: Grad[Params], params: Params, state: State[Params]): (Params, State[Params])

  // Convenience: iterator with fixed gradient function
  def iterateWithState[Params: ToPyTree: FloatTensorTree](init: Params)(df: Params => Grad[Params]): Iterator[(Params, State[Params])] =
    Iterator.iterate((init, this.init(init))): (params, state) =>
      val grads = df(params)
      update(grads, params, state)

  def iterate[Params: ToPyTree: FloatTensorTree](init: Params)(df: Params => Grad[Params]): Iterator[Params] =
    iterateWithState(init)(df).map(_._1)

case class GradientDescent(learningRate: Tensor0[Float]) extends GradientOptimizer:
  import dimwit.Conversions.given

  type State[P] = Unit // Stateless optimizer

  def init[Params: ToPyTree: FloatTensorTree](params: Params): Unit = ()

  def update[Params: ToPyTree: FloatTensorTree](gradients: Grad[Params], params: Params, state: Unit): (Params, Unit) =
    val paramTree = summon[FloatTensorTree[Params]]
    val newParams = paramTree.zipMap(
      gradients.value,
      params,
      [T <: Tuple] => (n: Labels[T]) ?=> (g: Tensor[T, Float], p: Tensor[T, Float]) => p - g.scale(learningRate)
    )
    (newParams, ())

case class Lion(learningRate: Tensor0[Float], weightDecay: Tensor0[Float] = Tensor0(0.0f), beta1: Tensor0[Float] = Tensor0(0.9f), beta2: Tensor0[Float] = Tensor0(0.99f)) extends GradientOptimizer:

  type State[P] = P // momentum state has same structure as params

  def init[Params: ToPyTree: FloatTensorTree](params: Params): Params =
    val paramTree = summon[FloatTensorTree[Params]]
    paramTree.map(
      params,
      [T <: Tuple] =>
        (n: Labels[T]) ?=>
          (t: Tensor[T, Float]) =>
            Tensor(t.shape).fill(0f)
    )

  def update[Params: ToPyTree: FloatTensorTree](gradients: Grad[Params], params: Params, momentums: Params): (Params, Params) =
    val paramTree = summon[FloatTensorTree[Params]]
    // the direction (1 or -1)
    // is determined by the sign of the momentum + gradient
    val updateDirection = paramTree.zipMap(
      gradients.value,
      momentums,
      [T <: Tuple] =>
        (n: Labels[T]) ?=>
          (grad: Tensor[T, Float], momentum: Tensor[T, Float]) =>
            (momentum *! beta1 + grad *! (1f - beta1)).sign
    )

    val updatedParams = paramTree.zipMap(
      updateDirection,
      params,
      [T <: Tuple] =>
        (n: Labels[T]) ?=>
          (updateDir: Tensor[T, Float], param: Tensor[T, Float]) =>
            param - updateDir *! learningRate - param *! weightDecay
    )

    val newMomentums = paramTree.zipMap(
      gradients.value,
      momentums,
      [T <: Tuple] =>
        (n: Labels[T]) ?=>
          (g: Tensor[T, Float], m: Tensor[T, Float]) =>
            m *! beta2 + g *! (1f - beta2)
    )

    (updatedParams, newMomentums)

case class AdamState[P](momentums: P, velocities: P)

/** Implements the Adam optimization algorithm.
  *
  * @see [[https://arxiv.org/abs/1412.6980 Adam: A Method for Stochastic Optimization]]
  */
case class Adam(
    learningRate: Tensor0[Float],
    beta1: Tensor0[Float] = Tensor0(0.9f),
    beta2: Tensor0[Float] = Tensor0(0.999f),
    epsilon: Tensor0[Float] = Tensor0(1e-8f)
) extends GradientOptimizer:

  type State[P] = AdamState[P]

  def init[Params: ToPyTree: FloatTensorTree](params: Params): State[Params] =
    val paramTree = summon[FloatTensorTree[Params]]
    val zeros = paramTree.map(
      params,
      [T <: Tuple] =>
        (n: Labels[T]) ?=>
          (t: Tensor[T, Float]) =>
            Tensor(t.shape).fill(0f)
    )
    AdamState(zeros, zeros)

  def update[Params: ToPyTree: FloatTensorTree](
      gradients: Grad[Params],
      params: Params,
      state: State[Params]
  ): (Params, State[Params]) =
    val paramTree = summon[FloatTensorTree[Params]]
    val (momentums, velocities) = (state.momentums, state.velocities)

    val newMomentums = paramTree.zipMap(
      gradients.value,
      momentums,
      [T <: Tuple] =>
        (n: Labels[T]) ?=>
          (g: Tensor[T, Float], m: Tensor[T, Float]) =>
            m *! beta1 + g *! (1f - beta1)
    )

    val newVelocities = paramTree.zipMap(
      gradients.value,
      velocities,
      [T <: Tuple] =>
        (n: Labels[T]) ?=>
          (g: Tensor[T, Float], v: Tensor[T, Float]) =>
            v *! beta2 + (g * g) *! (1f - beta2)
    )

    val updatedParams = paramTree.zipMap(
      params,
      newMomentums,
      newVelocities,
      [T <: Tuple] =>
        (n: Labels[T]) ?=>
          (p: Tensor[T, Float], m: Tensor[T, Float], v: Tensor[T, Float]) =>
            p - (m *! learningRate) / (v.sqrt +! epsilon)
    )

    (updatedParams, AdamState(newMomentums, newVelocities))

/** Implements the AdamW algorithm (Adam with decoupled weight decay).
  *
  * This implementation follows the logic described in "Decoupled Weight Decay Regularization"
  * where weight decay is performed directly on parameters rather than added to gradients.
  *
  * @see [[https://arxiv.org/abs/1711.05101 Decoupled Weight Decay Regularization]]
  *
  * @param learningRate The step size.
  * @param weightDecayFactor The coefficient for weight decay (lambda).
  */
case class AdamW(
    val adam: Adam,
    val weightDecayFactor: Tensor0[Float]
) extends GradientOptimizer:

  type State[P] = adam.State[P]

  def init[Params: ToPyTree: FloatTensorTree](params: Params): State[Params] = adam.init(params)

  def update[Params: ToPyTree: FloatTensorTree](
      gradients: Grad[Params],
      params: Params,
      state: State[Params]
  ): (Params, State[Params]) =
    // Tie weight decay to learning rate
    val `λ'` = weightDecayFactor
    val λ = `λ'` * adam.learningRate
    val weightDecay = WeightDecay(λ)
    val (newParams, newState) = adam.update(gradients, weightDecay(params), state)
    (newParams, newState)
