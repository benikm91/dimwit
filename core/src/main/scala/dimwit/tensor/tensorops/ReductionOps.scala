package dimwit.tensor.tensorops

import dimwit.jax.Jax
import dimwit.tensor.Axis
import dimwit.tensor.DType.Int32
import dimwit.tensor.Label
import dimwit.tensor.Labels
import dimwit.tensor.ShapeTypeHelpers.AxesRemover
import dimwit.tensor.ShapeTypeHelpers.AxisRemover
import dimwit.tensor.ShapeTypeHelpers.UnwrapAxes
import dimwit.tensor.Tensor
import dimwit.tensor.Tensor0
import dimwit.tensor.ValueTypeClasses.IsFloating
import dimwit.tensor.ValueTypeClasses.IsNumber
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.readwrite.Writer

private[dimwit] object ReductionOps:

  // the axes reduced over are removed from the result, without an axis `t` is reduced to a scalar.

  /** sums `t` (over all elements, or) along the specified axis or axes. */
  def sum[T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V]): Tensor0[V] = Tensor0(Jax.jnp.sum(t.jaxValue))
  def sum[T <: Tuple: Labels, V: IsNumber, L: Label](t: Tensor[T, V], axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.sum(t.jaxValue, axis = ev.index))
  def sum[T <: Tuple: Labels, V: IsNumber, Inputs <: Tuple](t: Tensor[T, V], axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.sum(t.jaxValue, axis = ev.indices.toPythonProxy))

  /** takes the maximum of `t` (over all elements, or) along the specified axis or axes. */
  def max[T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V]): Tensor0[V] = Tensor0(Jax.jnp.max(t.jaxValue))
  def max[T <: Tuple: Labels, V: IsNumber, L: Label](t: Tensor[T, V], axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.max(t.jaxValue, axis = ev.index))
  def max[T <: Tuple: Labels, V: IsNumber, Inputs <: Tuple](t: Tensor[T, V], axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.max(t.jaxValue, axis = ev.indices.toPythonProxy))

  /** takes the minimum of `t` (over all elements, or) along the specified axis or axes. */
  def min[T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V]): Tensor0[V] = Tensor0(Jax.jnp.min(t.jaxValue))
  def min[T <: Tuple: Labels, V: IsNumber, L: Label](t: Tensor[T, V], axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.min(t.jaxValue, axis = ev.index))
  def min[T <: Tuple: Labels, V: IsNumber, Inputs <: Tuple](t: Tensor[T, V], axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.min(t.jaxValue, axis = ev.indices.toPythonProxy))

  /** returns the index of the maximum of `t` along the specified axis or axes. */
  def argmax[T <: Tuple: Labels, V: IsNumber, L: Label](t: Tensor[T, V], axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, Int32] = Tensor(Jax.jnp.argmax(t.jaxValue, axis = ev.index))
  def argmax[T <: Tuple: Labels, V: IsNumber, Inputs <: Tuple](t: Tensor[T, V], axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, Int32] = Tensor(Jax.jnp.argmax(t.jaxValue, axis = ev.indices.toPythonProxy))

  /** returns the index of the minimum of `t` along the specified axis or axes. */
  def argmin[T <: Tuple: Labels, V: IsNumber, L: Label](t: Tensor[T, V], axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, Int32] = Tensor(Jax.jnp.argmin(t.jaxValue, axis = ev.index))
  def argmin[T <: Tuple: Labels, V: IsNumber, Inputs <: Tuple](t: Tensor[T, V], axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, Int32] = Tensor(Jax.jnp.argmin(t.jaxValue, axis = ev.indices.toPythonProxy))

  /** computes the mean of `t` (over all elements, or) along the specified axis or axes. */
  def mean[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor0[V] = Tensor0(Jax.jnp.mean(t.jaxValue))
  def mean[T <: Tuple: Labels, V: IsFloating, L: Label](t: Tensor[T, V], axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.mean(t.jaxValue, axis = ev.index))
  def mean[T <: Tuple: Labels, V: IsFloating, Inputs <: Tuple](t: Tensor[T, V], axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.mean(t.jaxValue, axis = ev.indices.toPythonProxy))

  /** computes the standard deviation of `t` (over all elements, or) along the specified axis or axes. */
  def std[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor0[V] = Tensor0(Jax.jnp.std(t.jaxValue))
  def std[T <: Tuple: Labels, V: IsFloating, L: Label](t: Tensor[T, V], axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.std(t.jaxValue, axis = ev.index))
  def std[T <: Tuple: Labels, V: IsFloating, Inputs <: Tuple](t: Tensor[T, V], axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.std(t.jaxValue, axis = ev.indices.toPythonProxy))

  /** computes the median of `t` (over all elements, or) along the specified axis or axes. */
  def median[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor0[V] = Tensor0(Jax.jnp.median(t.jaxValue))
  def median[T <: Tuple: Labels, V: IsFloating, L: Label](t: Tensor[T, V], axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.median(t.jaxValue, axis = ev.index))
  def median[T <: Tuple: Labels, V: IsFloating, Inputs <: Tuple](t: Tensor[T, V], axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.median(t.jaxValue, axis = ev.indices.toPythonProxy))

  /** computes the mean of `t`, ignoring NaN values, (over all elements, or) along the specified axis or axes. */
  def nanmean[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor0[V] = Tensor0(Jax.jnp.nanmean(t.jaxValue))
  def nanmean[T <: Tuple: Labels, V: IsFloating, L: Label](t: Tensor[T, V], axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.nanmean(t.jaxValue, axis = ev.index))
  def nanmean[T <: Tuple: Labels, V: IsFloating, Inputs <: Tuple](t: Tensor[T, V], axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.nanmean(t.jaxValue, axis = ev.indices.toPythonProxy))

  /** computes the median of `t`, ignoring NaN values, (over all elements, or) along the specified axis or axes. */
  def nanmedian[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor0[V] = Tensor0(Jax.jnp.nanmedian(t.jaxValue))
  def nanmedian[T <: Tuple: Labels, V: IsFloating, L: Label](t: Tensor[T, V], axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.nanmedian(t.jaxValue, axis = ev.index))
  def nanmedian[T <: Tuple: Labels, V: IsFloating, Inputs <: Tuple](t: Tensor[T, V], axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.nanmedian(t.jaxValue, axis = ev.indices.toPythonProxy))

  /** computes the `q`th quantile of `t` (over all elements, or) along the specified axis or axes. */
  def quantile[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V], q: Tensor0[V]): Tensor0[V] = Tensor0(Jax.jnp.quantile(t.jaxValue, q.jaxValue))
  def quantile[T <: Tuple: Labels, V: IsFloating, L: Label](t: Tensor[T, V], q: Tensor0[V], axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.quantile(t.jaxValue, q.jaxValue, axis = ev.index))
  def quantile[T <: Tuple: Labels, V: IsFloating, Inputs <: Tuple](t: Tensor[T, V], q: Tensor0[V], axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.quantile(t.jaxValue, q.jaxValue, axis = ev.indices.toPythonProxy))

private[dimwit] object ReductionExtensions:

  extension [T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])

    /** sums the tensor `t` along the specified axes, returning a new tensor with those axes removed. */
    def sum[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = ReductionOps.sum(t, axes)
    def sum[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = ReductionOps.sum(t, axis)
    def sum: Tensor0[V] = ReductionOps.sum(t)

    /** takes the maximum of the tensor `t` along the specified axes, returning a new tensor with those axes removed. */
    def max[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = ReductionOps.max(t, axes)
    def max[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = ReductionOps.max(t, axis)
    def max: Tensor0[V] = ReductionOps.max(t)

    /** takes the minimum of the tensor `t` along the specified axes, returning a new tensor with those axes removed. */
    def min[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = ReductionOps.min(t, axes)
    def min[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = ReductionOps.min(t, axis)
    def min: Tensor0[V] = ReductionOps.min(t)

    /** argument of the maximum of the tensor `t` along the specified axes, returning a new tensor with those axes removed. */
    def argmax[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, Int32] = ReductionOps.argmax(t, axes)
    def argmax[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, Int32] = ReductionOps.argmax(t, axis)

    /** argument of the minimum of the tensor `t` along the specified axes, returning a new tensor with those axes removed. */
    def argmin[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, Int32] = ReductionOps.argmin(t, axes)
    def argmin[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, Int32] = ReductionOps.argmin(t, axis)

  // ---------------------------------------------------------
  // IsFloat operations (IsFloat or IsInt)
  // ---------------------------------------------------------

  extension [T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V])

    /** computes the mean of the tensor `t` along the specified axes, returning a new tensor with those axes removed. */
    def mean[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = ReductionOps.mean(t, axes)
    def mean[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = ReductionOps.mean(t, axis)
    def mean: Tensor0[V] = ReductionOps.mean(t)

    /** computes the mean of the tensor `t` along the specified axes, returning a new tensor with those axes removed. */
    def std[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = ReductionOps.std(t, axes)
    def std[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = ReductionOps.std(t, axis)
    def std: Tensor0[V] = ReductionOps.std(t)

    /** computes the qth quantile of the tensor `t` along the specified axes, returning a new tensor with those axes removed. */
    def quantile[Inputs <: Tuple](q: Tensor0[V], axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = ReductionOps.quantile(t, q, axes)
    def quantile[L: Label](q: Tensor0[V], axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = ReductionOps.quantile(t, q, axis)
    def quantile(q: Tensor0[V]): Tensor0[V] = ReductionOps.quantile(t, q)

    /** computes the median of the tensor `t` along the specified axes, returning a new tensor with those axes removed. */
    def median: Tensor0[V] = ReductionOps.median(t)
    def median[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = ReductionOps.median(t, axis)
    def median[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = ReductionOps.median(t, axes)

    /** computes the mean of the tensor `t` along the specified axes, ignoring na values and returning a new tensor with those axes removed. */
    def nanmean: Tensor0[V] = ReductionOps.nanmean(t)
    def nanmean[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = ReductionOps.nanmean(t, axis)
    def nanmean[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = ReductionOps.nanmean(t, axes)

    /** computes the median of the tensor `t` along the specified axes, ignoring na values and returning a new tensor with those axes removed. */
    def nanmedian[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = ReductionOps.nanmedian(t, axes)
    def nanmedian[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = ReductionOps.nanmedian(t, axis)
    def nanmedian: Tensor0[V] = ReductionOps.nanmedian(t)
