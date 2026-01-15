package nn

import dimwit.*
import dimwit.Conversions.given
import dimwit.autodiff.Grad
import dimwit.jax.Jax
import dimwit.jax.Jit
import me.shadaj.scalapy.py
import scala.annotation.meta.param

trait GradientOptimizer[P]:
  type State
  def initState(params: P): State
  def updateState(state: State, grads: Grad[P]): State
  def updateParams(params: P, state: State, grads: Grad[P]): P

object GradientDescent:
  def createFor[P: FloatTensorTree](p: P)(learningRate: Float) =
    GradientDescent[P](learningRate)

case class GradientDescent[P: FloatTensorTree](learningRate: Float) extends GradientOptimizer[P]:
  opaque type EmptyState = Unit
  object EmptyState:
    given ToPyTree[EmptyState] = summon[ToPyTree[Unit]]
  type State = EmptyState
  def initState(params: P): State = ()
  def updateState(state: State, grads: Grad[P]): State = ()
  def updateParams(params: P, state: State, grads: Grad[P]): P =
    val paramTree = summon[FloatTensorTree[P]]
    paramTree.zipMap(
      grads,
      params,
      [T <: Tuple] =>
        (n: Labels[T]) ?=>
          (grad: Tensor[T, Float], param: Tensor[T, Float]) =>
            param - grad *! learningRate
    )

object Lion:
  def createFor[P: FloatTensorTree](p: P)(learningRate: Float, weightDecay: Float = 0.0f, beta1: Float = 0.9f, beta2: Float = 0.99f) =
    Lion[P](learningRate, weightDecay, beta1, beta2)

case class Lion[P: FloatTensorTree](learningRate: Float, weightDecay: Float = 0.0f, beta1: Float = 0.9f, beta2: Float = 0.99f) extends GradientOptimizer[P]:

  case class LionState(momentums: P)
  type State = LionState

  def initState(params: P): State =
    val paramTree = summon[FloatTensorTree[P]]
    LionState(
      paramTree.map(
        params,
        [T <: Tuple] =>
          (n: Labels[T]) ?=>
            (t: Tensor[T, Float]) =>
              Tensor.zeros(t.shape, VType[Float])
      )
    )

  def updateState(state: State, grads: Grad[P]): State =
    val paramTree = summon[FloatTensorTree[P]]
    LionState(
      paramTree.zipMap(
        grads,
        state.momentums,
        [T <: Tuple] =>
          (n: Labels[T]) ?=>
            (grad: Tensor[T, Float], momentum: Tensor[T, Float]) =>
              (momentum *! beta1 + grad *! (1f - beta1)).sign
      )
    )

  def updateParams(params: P, state: State, grads: Grad[P]): P =
    val paramTree = summon[FloatTensorTree[P]]
    paramTree.zipMap(
      state.momentums,
      params,
      [T <: Tuple] =>
        (n: Labels[T]) ?=>
          (updateDir: Tensor[T, Float], param: Tensor[T, Float]) =>
            param - updateDir *! learningRate - param *! weightDecay
    )
