package dimwit.tensor.tensorops

import dimwit.jax.Jax
import dimwit.tensor.Axis
import dimwit.tensor.DType.Int32
import dimwit.tensor.Label
import dimwit.tensor.Labels
import dimwit.tensor.ShapeTypeHelpers.AxisIndex
import dimwit.tensor.Tensor
import dimwit.tensor.Tensor1
import dimwit.tensor.ValueTypeClasses.IsFloating
import dimwit.tensor.ValueTypeClasses.IsNumber
import dimwit.tensor.tensorops.FunctionalExtensions.vapply

/** Functions on a Tensor1 that keep its axis, exported as e.g. `Tensor1.softmax`. */
private[dimwit] object AlongAxisTensor1Ops:

  // ---------------------------------------------------------
  // Functions from a Tensor1 to a Tensor1.
  // They are lifted to any tensor shape with `vapply` by the functions in [[AlongAxisOps]].
  // ---------------------------------------------------------

  /** sorts the Tensor1 `t`. */
  def sort[L: Label, V: IsNumber](t: Tensor1[L, V]): Tensor1[L, V] =
    Tensor(Jax.jnp.sort(t.jaxValue, axis = 0))

  /** returns the indices that would sort the Tensor1 `t`. */
  def argsort[L: Label, V: IsNumber](t: Tensor1[L, V]): Tensor1[L, Int32] =
    Tensor(Jax.jnp.argsort(t.jaxValue, axis = 0))

  /** computes the cumulative sum of the Tensor1 `t`. */
  def cumsum[L: Label, V: IsNumber](t: Tensor1[L, V]): Tensor1[L, V] =
    Tensor(Jax.jnp.cumsum(t.jaxValue, axis = 0))

  /** computes the cumulative product of the Tensor1 `t`. */
  def cumprod[L: Label, V: IsNumber](t: Tensor1[L, V]): Tensor1[L, V] =
    Tensor(Jax.jnp.cumprod(t.jaxValue, axis = 0))

  /** computes the cumulative maximum of the Tensor1 `t`. */
  def cummax[L: Label, V: IsNumber](t: Tensor1[L, V]): Tensor1[L, V] =
    Tensor(Jax.lax.cummax(t.jaxValue, axis = 0))

  /** computes the cumulative minimum of the Tensor1 `t`. */
  def cummin[L: Label, V: IsNumber](t: Tensor1[L, V]): Tensor1[L, V] =
    Tensor(Jax.lax.cummin(t.jaxValue, axis = 0))

  /** computes the cumulative log-sum-exp of the Tensor1 `t`, i.e. a numerically stable `log(cumsum(exp(t)))`. */
  def logcumsumexp[L: Label, V: IsFloating](t: Tensor1[L, V]): Tensor1[L, V] =
    Tensor(Jax.lax.cumlogsumexp(t.jaxValue, axis = 0))

  /** computes the discrete difference of the Tensor1 `t`, reducing its size by one. */
  def diff[L: Label, V: IsNumber](t: Tensor1[L, V]): Tensor1[L, V] =
    Tensor(Jax.jnp.diff(t.jaxValue, axis = 0))

  /** rolls the elements of the Tensor1 `t` by `shift` positions; elements shifted beyond the end re-appear at the start. */
  def roll[L: Label, V](shift: Int)(t: Tensor1[L, V]): Tensor1[L, V] =
    Tensor(Jax.jnp.roll(t.jaxValue, shift = shift, axis = 0))

  /** computes the softmax of the Tensor1 `t`. */
  def softmax[L: Label, V: IsFloating](t: Tensor1[L, V]): Tensor1[L, V] =
    Tensor(Jax.jnn.softmax(t.jaxValue, axis = 0))

  /** computes the log of the softmax of the Tensor1 `t`, more stable than `softmax(t).log`. */
  def logSoftmax[L: Label, V: IsFloating](t: Tensor1[L, V]): Tensor1[L, V] =
    Tensor(Jax.jnn.log_softmax(t.jaxValue, axis = 0))

/** Operations along an axis that keep all axes of the tensor (unlike reductions, which remove them), exported as e.g. `Tensor.softmax(t, Axis[A])`. */
private[dimwit] object AlongAxisOps:

  /** rolls the elements of `t` along the specified axis by `shift` positions; elements shifted beyond the end re-appear at the start. */
  def roll[T <: Tuple: Labels, V, L: Label](t: Tensor[T, V], axis: Axis[L], shift: Int)(using AxisIndex[T, L]): Tensor[T, V] =
    t.vapply(axis)(AlongAxisTensor1Ops.roll(shift))

  /** returns the indices that would sort `t` along the specified axis. */
  def argsort[T <: Tuple: Labels, V: IsNumber, L: Label](t: Tensor[T, V], axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, Int32] =
    t.vapply(axis)(AlongAxisTensor1Ops.argsort)

  /** sorts `t` along the specified axis. */
  def sort[T <: Tuple: Labels, V: IsNumber, L: Label](t: Tensor[T, V], axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
    t.vapply(axis)(AlongAxisTensor1Ops.sort)

  /** computes the cumulative sum of `t` along the specified axis. */
  def cumsum[T <: Tuple: Labels, V: IsNumber, L: Label](t: Tensor[T, V], axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
    t.vapply(axis)(AlongAxisTensor1Ops.cumsum)

  /** computes the cumulative product of `t` along the specified axis. */
  def cumprod[T <: Tuple: Labels, V: IsNumber, L: Label](t: Tensor[T, V], axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
    t.vapply(axis)(AlongAxisTensor1Ops.cumprod)

  /** computes the cumulative maximum of `t` along the specified axis. */
  def cummax[T <: Tuple: Labels, V: IsNumber, L: Label](t: Tensor[T, V], axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
    t.vapply(axis)(AlongAxisTensor1Ops.cummax)

  /** computes the cumulative minimum of `t` along the specified axis. */
  def cummin[T <: Tuple: Labels, V: IsNumber, L: Label](t: Tensor[T, V], axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
    t.vapply(axis)(AlongAxisTensor1Ops.cummin)

  /** computes the discrete difference of `t` along the specified axis, reducing that axis' size by one. */
  def diff[T <: Tuple: Labels, V: IsNumber, L: Label](t: Tensor[T, V], axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
    t.vapply(axis)(AlongAxisTensor1Ops.diff)

  /** computes the cumulative log-sum-exp of `t` along the specified axis. */
  def logcumsumexp[T <: Tuple: Labels, V: IsFloating, L: Label](t: Tensor[T, V], axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
    t.vapply(axis)(AlongAxisTensor1Ops.logcumsumexp)

  /** computes the softmax of `t` along the specified axis. */
  def softmax[T <: Tuple: Labels, V: IsFloating, L: Label](t: Tensor[T, V], axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
    t.vapply(axis)(AlongAxisTensor1Ops.softmax)

  /** computes the log-softmax of `t` along the specified axis. */
  def logSoftmax[T <: Tuple: Labels, V: IsFloating, L: Label](t: Tensor[T, V], axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
    t.vapply(axis)(AlongAxisTensor1Ops.logSoftmax)

/** Extension methods for operations along an axis, e.g. `t.softmax(Axis[A])`. */
private[dimwit] object AlongAxisExtensions:

  extension [T <: Tuple: Labels, V](t: Tensor[T, V])

    /** rolls the elements of `t` along the specified axis by `shift` positions. */
    def roll[L: Label](axis: Axis[L], shift: Int)(using AxisIndex[T, L]): Tensor[T, V] =
      AlongAxisOps.roll(t, axis, shift)

  extension [T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])

    /** Returns a tensor of indices that would sort `t` along the specified axis */
    def argsort[L: Label](axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, Int32] =
      AlongAxisOps.argsort(t, axis)

    /** sorts the tensor `t` along the specified axis */
    def sort[L: Label](axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
      AlongAxisOps.sort(t, axis)

    /** computes the cumulative sum of the tensor `t` along the specified axis. */
    def cumsum[L: Label](axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
      AlongAxisOps.cumsum(t, axis)

    /** computes the cumulative product of the tensor `t` along the specified axis. */
    def cumprod[L: Label](axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
      AlongAxisOps.cumprod(t, axis)

    /** computes the cumulative maximum of the tensor `t` along the specified axis. */
    def cummax[L: Label](axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
      AlongAxisOps.cummax(t, axis)

    /** computes the cumulative minimum of the tensor `t` along the specified axis. */
    def cummin[L: Label](axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
      AlongAxisOps.cummin(t, axis)

    /** computes the discrete difference of the tensor `t` along the specified axis, reducing that axis' size by one. */
    def diff[L: Label](axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
      AlongAxisOps.diff(t, axis)

  extension [T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V])

    /** computes the cumulative log-sum-exp of the tensor `t` along the specified axis. */
    def logcumsumexp[L: Label](axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
      AlongAxisOps.logcumsumexp(t, axis)

    /** computes the softmax of `t` along the specified axis. */
    def softmax[L: Label](axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
      AlongAxisOps.softmax(t, axis)

    /** computes the log-softmax of `t` along the specified axis. */
    def logSoftmax[L: Label](axis: Axis[L])(using AxisIndex[T, L]): Tensor[T, V] =
      AlongAxisOps.logSoftmax(t, axis)
