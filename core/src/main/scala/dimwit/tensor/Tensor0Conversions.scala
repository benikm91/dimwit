package dimwit.tensor

import dimwit.tensor.ValueTypeClasses.IsBoolean
import dimwit.tensor.ValueTypeClasses.IsFloating
import dimwit.tensor.ValueTypeClasses.IsInteger

/** Conversions from Scala scalars to a `Tensor0` of the tensor's value type, e.g. for `t *! 2.0f`.
  *
  * Users never import them: the `Tensor` companion object exports them, so the compiler finds them automatically.
  */
private[dimwit] object Tensor0Conversions:

  given boolean2BooleanTensor[V: IsBoolean]: Conversion[Boolean, Tensor0[V]] with
    def apply(value: Boolean): Tensor0[V] = Tensor0(VType[V])(value)

  given byte2IntegerTensor[V: IsInteger]: Conversion[Byte, Tensor0[V]] with
    def apply(value: Byte): Tensor0[V] = Tensor0(VType[V])(value)

  given short2IntegerTensor[V: IsInteger]: Conversion[Short, Tensor0[V]] with
    def apply(value: Short): Tensor0[V] = Tensor0(VType[V])(value)

  given int2IntegerTensor[V: IsInteger]: Conversion[Int, Tensor0[V]] with
    def apply(value: Int): Tensor0[V] = Tensor0(VType[V])(value)

  given int2FloatingTensor[V: IsFloating]: Conversion[Int, Tensor0[V]] with
    def apply(value: Int): Tensor0[V] = Tensor0(VType[V])(value.toFloat)

  given long2IntegerTensor[V: IsInteger]: Conversion[Long, Tensor0[V]] with
    def apply(value: Long): Tensor0[V] = Tensor0(VType[V])(value)

  given float2FloatingTensor[V: IsFloating]: Conversion[Float, Tensor0[V]] with
    def apply(value: Float): Tensor0[V] = Tensor0(VType[V])(value)

  given double2FloatingTensor[V: IsFloating]: Conversion[Double, Tensor0[V]] with
    def apply(value: Double): Tensor0[V] = Tensor0(VType[V])(value)
