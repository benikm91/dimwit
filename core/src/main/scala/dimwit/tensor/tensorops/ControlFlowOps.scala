package dimwit.tensor.tensorops

import dimwit.OnError
import dimwit.jax.Jax
import dimwit.python.PyIndex.itemAt
import dimwit.tensor.Axis
import dimwit.tensor.DType.Bool
import dimwit.tensor.DType.Int32
import dimwit.tensor.Label
import dimwit.tensor.Labels
import dimwit.tensor.ShapeTypeHelpers.AxisRemover
import dimwit.tensor.Tensor
import dimwit.tensor.Tensor0
import dimwit.tensor.ShapeTypeHelpers.SharedAxisRemover
import dimwit.tensor.tensorops.FunctionalOps.PrependAxis
import dimwit.tensor.tensorops.FunctionalOps.ZipVmap.ShapesOf
import dimwit.tensor.tensorops.FunctionalOps.ZipVmap.TensorsOf
import dimwit.tensor.tensorops.FunctionalOps.ZipVmap.ValuesOf
import dimwit.tensor.tensorops.FunctionalOps.ZipVmap.slicesOf
import dimwit.tensortree.TensorTree
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

/** Structured control flow, e.g. `scan` or `whileLoop`. The body of a loop is traced once,
  * so that a loop does not unroll under `jit`, however many iterations it runs.
  */
private[dimwit] object ControlFlowOps:

  /** Loops over the axis `L` of `xs`, like `jax.lax.scan`: the body `f` gets the carry and the slice of `xs`
    * at each position along `L`, and returns the next carry and an output. The outputs are stacked along `L`.
    *
    * The body is traced once, so it may not change the shape of the carry. It works under `jit` and is differentiable.
    *
    * @param axis The axis to loop over.
    * @param init The initial carry, any tensor tree.
    * @param xs The tensor to loop over.
    * @param f The body, from the carry and the slice of `xs` to the next carry and an output.
    * @return The final carry, and the outputs with `L` prepended to every tensor.
    */
  def scan[L: Label, Carry, T <: Tuple, V, Y](axis: Axis[L])(init: Carry, xs: Tensor[T, V])(using
      ev: AxisRemover[T, L],
      sliceLabels: Labels[ev.RemainingAxes]
  )(f: (Carry, Tensor[ev.RemainingAxes, V]) => (Carry, Y))(using
      carryTree: TensorTree[Carry],
      yTree: TensorTree[Y],
      prependAxis: PrependAxis[L, Y],
      ysTree: TensorTree[prependAxis.Out]
  ): (Carry, prependAxis.Out) =
    scanAlong[L, Carry, Tensor[ev.RemainingAxes, V], Y, prependAxis.Out](
      init,
      Jax.jnp.moveaxis(xs.jaxValue, ev.index, 0),
      List(xs.shape.dimensions(ev.index))
    )(Tensor[ev.RemainingAxes, V](_))(f)

  /** Loops over the axis `L` of all tensors of `xs` together, like `jax.lax.scan`: the body `f` gets the carry and
    * the tuple of their slices at each position along `L`, and returns the next carry and an output.
    * The outputs are stacked along `L`.
    *
    * The body is traced once, so it may not change the shape of the carry. It works under `jit` and is differentiable.
    *
    * @param axis The axis to loop over. Every tensor of `xs` must have it, with the same extent.
    * @param init The initial carry, any tensor tree.
    * @param xs The tuple of tensors to loop over.
    * @param f The body, from the carry and the tuple of slices of `xs` to the next carry and an output.
    * @return The final carry, and the outputs with `L` prepended to every tensor.
    */
  def scan[L: Label, Carry, Inputs <: Tuple, Y](axis: Axis[L])(init: Carry, xs: Inputs)(using
      ev: SharedAxisRemover[ShapesOf[Inputs], L]
  )(f: (Carry, TensorsOf[ev.RemainingAxes, ValuesOf[Inputs]]) => (Carry, Y))(using
      carryTree: TensorTree[Carry],
      yTree: TensorTree[Y],
      prependAxis: PrependAxis[L, Y],
      ysTree: TensorTree[prependAxis.Out]
  ): (Carry, prependAxis.Out) =
    val tensors = xs.toList.map(_.asInstanceOf[Tensor[?, ?]])
    scanAlong[L, Carry, TensorsOf[ev.RemainingAxes, ValuesOf[Inputs]], Y, prependAxis.Out](
      init,
      py.Dynamic.global.tuple(tensors.zip(ev.indices).map((t, i) => Jax.jnp.moveaxis(t.jaxValue, i, 0)).toPythonProxy),
      tensors.zip(ev.indices).map((t, i) => t.shape.dimensions(i))
    )(slices => slicesOf[ShapesOf[Inputs], L, ValuesOf[Inputs]](slices.as[Seq[Jax.PyDynamic]], ev))(f)

  /** The scan over the axis `L` of the tensors with the extents `extents`, moved to the front in the pytree `leading`. */
  private def scanAlong[L: Label, Carry, X, Y, Ys](init: Carry, leading: py.Any, extents: List[Int])(
      sliceOf: Jax.PyDynamic => X
  )(f: (Carry, X) => (Carry, Y))(using
      carryTree: TensorTree[Carry],
      yTree: TensorTree[Y],
      ysTree: TensorTree[Ys]
  ): (Carry, Ys) =
    val axisName = summon[Label[L]].name
    require(
      extents.distinct.sizeIs == 1,
      s"All tensors scanned over Axis[$axisName] must have the same extent along it, but they have ${extents.mkString(", ")}"
    )
    val body = (carryPy: Jax.PyDynamic, xPy: Jax.PyDynamic) =>
      OnError.traceStack:
        val carry = carryTree.fromPyTree(carryPy)
        val (next, y) = f(carry, sliceOf(xPy))
        requireSameShapes(carry, next)
        // the Python tuple (carry, y) that jax.lax.scan expects
        py.Dynamic.global.tuple(Seq(carryTree.toPyTree(next), yTree.toPyTree(y)).toPythonProxy): py.Any
    val result = Jax.jax_helper.scan(body, carryTree.toPyTree(init), leading)
    (carryTree.fromPyTree(result.itemAt(0)), ysTree.fromPyTree(result.itemAt(1)))

  /** Loops over the indices `lower until upper`, like `jax.lax.fori_loop`: the body gets the index and the carry,
    * and returns the next carry.
    *
    * The index is a traced tensor, e.g. to slice or set at it. The body is traced once, so it may not change
    * the shape of the carry. Since the bounds are static, the loop works under `jit` and is differentiable.
    *
    * @param lower The first index.
    * @param upper The index after the last one.
    * @param init The initial carry, any tensor tree.
    * @param body The body, from the index and the carry to the next carry.
    * @return The final carry.
    */
  def foriLoop[Carry](lower: Int, upper: Int)(init: Carry)(body: (Tensor0[Int32], Carry) => Carry)(using
      carryTree: TensorTree[Carry]
  ): Carry =
    val bodyPy = (indexPy: Jax.PyDynamic, carryPy: Jax.PyDynamic) =>
      OnError.traceStack:
        val carry = carryTree.fromPyTree(carryPy)
        val next = body(Tensor[EmptyTuple, Int32](indexPy), carry)
        requireSameShapes(carry, next)
        carryTree.toPyTree(next)
    carryTree.fromPyTree(Jax.jax_helper.fori_loop(lower, upper, bodyPy, carryTree.toPyTree(init)))

  /** Loops while `condition` holds, like `jax.lax.while_loop`: the body gets the carry and returns the next carry.
    *
    * The number of iterations may depend on the values, e.g. to stop once a computation has converged.
    * The body is traced once, so it may not change the shape of the carry.
    * JAX cannot reverse-differentiate a while loop, so `Autodiff.grad` through it fails; forward mode (`Autodiff.jacFwd`) works.
    *
    * @param init The initial carry, any tensor tree.
    * @param condition Whether to run the body (again) on the carry.
    * @param body The body, from the carry to the next carry.
    * @return The first carry for which `condition` does not hold.
    */
  def whileLoop[Carry](init: Carry)(condition: Carry => Tensor0[Bool])(body: Carry => Carry)(using
      carryTree: TensorTree[Carry]
  ): Carry =
    val conditionPy = (carryPy: Jax.PyDynamic) =>
      OnError.traceStack:
        condition(carryTree.fromPyTree(carryPy)).jaxValue
    val bodyPy = (carryPy: Jax.PyDynamic) =>
      OnError.traceStack:
        val carry = carryTree.fromPyTree(carryPy)
        val next = body(carry)
        requireSameShapes(carry, next)
        carryTree.toPyTree(next)
    carryTree.fromPyTree(Jax.jax_helper.while_loop(conditionPy, bodyPy, carryTree.toPyTree(init)))

  /** Evaluates `ifTrue` or `ifFalse` depending on `predicate`, like `jax.lax.cond`.
    * Unlike Scala's `if`, the predicate may be traced, e.g. under `jit` or in a loop body.
    * Both branches are traced, so they must return the same shapes.
    *
    * @param predicate Which branch to evaluate.
    * @param ifTrue The result if `predicate` holds, any tensor tree.
    * @param ifFalse The result otherwise.
    */
  def cond[T](predicate: Tensor0[Bool])(ifTrue: => T)(ifFalse: => T)(using tree: TensorTree[T]): T =
    // JAX traces both branches, one after the other; the second one is checked against the first one
    var firstBranch: Option[T] = None
    def branch(value: => T) = (_: Jax.PyDynamic) =>
      OnError.traceStack:
        val result = value
        firstBranch match
          case None        => firstBranch = Some(result)
          case Some(first) => requireSameShapes(first, result, "The branches of cond must return the same shapes")
        tree.toPyTree(result)
    tree.fromPyTree(Jax.jax_helper.cond(predicate.jaxValue, branch(ifTrue), branch(ifFalse)))

  /** Requires that the tensor trees `before` and `after` have the same shapes and value types, naming the first tensor that differs. */
  private def requireSameShapes[T](before: T, after: T, problem: String = "The body of a loop must not change the shape of the carry")(using
      tree: TensorTree[T]
  ): Unit =
    leaves(before).zip(leaves(after)).foreach:
      case ((path, b), (_, a)) =>
        val name = if path.isEmpty then "the result" else s"`$path`"
        require(
          b.shape.dimensions == a.shape.dimensions && b.dtype == a.dtype,
          s"$problem, but $name is ${b.shape} of ${b.dtype} and then ${a.shape} of ${a.dtype}"
        )

  /** The tensors of the tree `t`, with their paths (e.g. `cache.keys`). */
  private def leaves[T](t: T)(using tree: TensorTree[T]): List[(String, Tensor[?, ?])] =
    val result = List.newBuilder[(String, Tensor[?, ?])]
    tree.foreachWithName(t, [S <: Tuple, V] => (_: Labels[S]) ?=> (path: String, tensor: Tensor[S, V]) => result += ((path, tensor)): Unit)
    result.result()
