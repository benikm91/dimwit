package dimwit.tensor.tensorops

import dimwit.jax.Jax
import dimwit.tensor.Axis
import dimwit.tensor.DType.Int32
import dimwit.tensor.Label
import dimwit.tensor.Labels
import dimwit.tensor.ShapeTypeHelpers.AxisIndex
import dimwit.tensor.ShapeTypeHelpers.AxisIndices
import dimwit.tensor.ShapeTypeHelpers.UnwrapAxes
import dimwit.tensor.Tensor
import dimwit.tensor.Tensor1
import dimwit.tensor.TensorOps.IsFloating
import dimwit.tensor.TensorOps.IsNumber
import me.shadaj.scalapy.py.SeqConverters
import dimwit.tensor.tensorops.FunctionalOps.vapply

/** Operations along an axis that keep all axes of the tensor (unlike reductions, which remove them). */
object AlongAxisOps:

  extension [T <: Tuple: Labels, V](t: Tensor[T, V])

    /** rolls the elements of `t` along the specified axis by `shift` positions. */
    def roll[L: Label](axis: Axis[L], shift: Int)(using AxisIndex[T, L]): Tensor[T, V] =
      t.vapply(axis)(Tensor1.roll(shift))

  extension [T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])

    /** Returns a tensor of indices that would sort `t` along the specified axes */
    def argsort[Inputs <: Tuple](axes: Inputs)(using ev: AxisIndices[T, UnwrapAxes[Inputs]]): Tensor[T, Int32] = Tensor(Jax.jnp.argsort(t.jaxValue, axis = ev.indices.toPythonProxy))
    def argsort[L: Label](axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, Int32] =
      t.vapply(axis)(Tensor1.argsort)

    /** sorts the tensor `t` along the specified axis */
    def sort[L: Label](axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
      t.vapply(axis)(Tensor1.sort)

    /** computes the cumulative sum of the tensor `t` along the specified axis. */
    def cumsum[L: Label](axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
      t.vapply(axis)(Tensor1.cumsum)

    /** computes the cumulative product of the tensor `t` along the specified axis. */
    def cumprod[L: Label](axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
      t.vapply(axis)(Tensor1.cumprod)

    /** computes the cumulative maximum of the tensor `t` along the specified axis. */
    def cummax[L: Label](axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
      t.vapply(axis)(Tensor1.cummax)

    /** computes the cumulative minimum of the tensor `t` along the specified axis. */
    def cummin[L: Label](axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
      t.vapply(axis)(Tensor1.cummin)

    /** computes the discrete difference of the tensor `t` along the specified axis, reducing that axis' size by one. */
    def diff[L: Label](axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
      t.vapply(axis)(Tensor1.diff)

  extension [T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V])

    /** computes the cumulative log-sum-exp of the tensor `t` along the specified axis. */
    def logcumsumexp[L: Label](axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
      t.vapply(axis)(Tensor1.logcumsumexp)

    /** computes the softmax of `t` along the specified axis. */
    def softmax[L: Label](axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
      t.vapply(axis)(Tensor1.softmax)

    /** computes the log-softmax of `t` along the specified axis. */
    def logSoftmax[L: Label](axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
      t.vapply(axis)(Tensor1.logSoftmax)
