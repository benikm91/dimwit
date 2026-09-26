package dimwit.tensor

import dimwit.tensor.DType.Bool
import dimwit.tensor.ValueTypeClasses.IsFloating
import dimwit.tensor.ValueTypeClasses.IsNumber
import dimwit.tensor.tensorops.ElementWiseOps

/** Operators with a Scala scalar on the left, e.g. `2.0f *! t` or `3 < t0`.
  *
  * `t op scalar` works through the implicit conversions in [[dimwit.Conversions]], which convert the scalar to a
  * `Tensor0` of the tensor's value type. `scalar op t` cannot use them, because Scala does not convert the receiver of
  * a method call. These extension methods mirror every operator `t op scalar` by asking for the very same conversion,
  * so both orders compile for exactly the same scalar and value types, and in both the scalar takes the precision of
  * the tensor. Named methods like `elementEquals` or `and` are deliberately not mirrored: `2.0f.approxEquals(t)` reads
  * as a method of the scalar.
  *
  * There is one extension block per Scala scalar type. A single generic `extension [S](scalar: S)` would also match
  * e.g. `"-" * 3` and hide the `*` that `String` gets through `StringOps`.
  */
private[dimwit] object ValueExtensions:

  extension (scalar: Boolean)

    // operators with a Tensor0
    def +[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Boolean, Tensor0[V]]): Tensor0[V] = ElementWiseOps.add(toTensor0(scalar), t)
    def -[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Boolean, Tensor0[V]]): Tensor0[V] = ElementWiseOps.subtract(toTensor0(scalar), t)
    def *[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Boolean, Tensor0[V]]): Tensor0[V] = ElementWiseOps.multiply(toTensor0(scalar), t)
    def %[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Boolean, Tensor0[V]]): Tensor0[V] = ElementWiseOps.mod(toTensor0(scalar), t)
    def /[V: IsFloating](t: Tensor0[V])(using toTensor0: Conversion[Boolean, Tensor0[V]]): Tensor0[V] = ElementWiseOps.divide(toTensor0(scalar), t)
    def <[V](t: Tensor0[V])(using toTensor0: Conversion[Boolean, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.less(toTensor0(scalar), t)
    def <=[V](t: Tensor0[V])(using toTensor0: Conversion[Boolean, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.lessEqual(toTensor0(scalar), t)
    def >[V](t: Tensor0[V])(using toTensor0: Conversion[Boolean, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.greater(toTensor0(scalar), t)
    def >=[V](t: Tensor0[V])(using toTensor0: Conversion[Boolean, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.greaterEqual(toTensor0(scalar), t)
    def ===[V](t: Tensor0[V])(using toTensor0: Conversion[Boolean, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.arrayEqual(toTensor0(scalar), t)

    // broadcasting operators
    def +![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Boolean, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.add_!(toTensor0(scalar), t)
    def -![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Boolean, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.subtract_!(toTensor0(scalar), t)
    def *![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Boolean, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.multiply_!(toTensor0(scalar), t)
    def %![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Boolean, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.mod_!(toTensor0(scalar), t)
    def /![T <: Tuple, V: IsFloating](t: Tensor[T, V])(using toTensor0: Conversion[Boolean, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.divide_!(toTensor0(scalar), t)
    def `<!`[T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Boolean, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.less_!(toTensor0(scalar), t)
    def <=![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Boolean, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.lessEqual_!(toTensor0(scalar), t)
    def >![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Boolean, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.greater_!(toTensor0(scalar), t)
    def >=![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Boolean, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.greaterEqual_!(toTensor0(scalar), t)
    def ===![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Boolean, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor0[Bool] = ElementWiseOps.arrayEqual_!(toTensor0(scalar), t)

  extension (scalar: Byte)

    // operators with a Tensor0
    def +[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Byte, Tensor0[V]]): Tensor0[V] = ElementWiseOps.add(toTensor0(scalar), t)
    def -[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Byte, Tensor0[V]]): Tensor0[V] = ElementWiseOps.subtract(toTensor0(scalar), t)
    def *[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Byte, Tensor0[V]]): Tensor0[V] = ElementWiseOps.multiply(toTensor0(scalar), t)
    def %[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Byte, Tensor0[V]]): Tensor0[V] = ElementWiseOps.mod(toTensor0(scalar), t)
    def /[V: IsFloating](t: Tensor0[V])(using toTensor0: Conversion[Byte, Tensor0[V]]): Tensor0[V] = ElementWiseOps.divide(toTensor0(scalar), t)
    def <[V](t: Tensor0[V])(using toTensor0: Conversion[Byte, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.less(toTensor0(scalar), t)
    def <=[V](t: Tensor0[V])(using toTensor0: Conversion[Byte, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.lessEqual(toTensor0(scalar), t)
    def >[V](t: Tensor0[V])(using toTensor0: Conversion[Byte, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.greater(toTensor0(scalar), t)
    def >=[V](t: Tensor0[V])(using toTensor0: Conversion[Byte, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.greaterEqual(toTensor0(scalar), t)
    def ===[V](t: Tensor0[V])(using toTensor0: Conversion[Byte, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.arrayEqual(toTensor0(scalar), t)

    // broadcasting operators
    def +![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Byte, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.add_!(toTensor0(scalar), t)
    def -![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Byte, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.subtract_!(toTensor0(scalar), t)
    def *![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Byte, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.multiply_!(toTensor0(scalar), t)
    def %![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Byte, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.mod_!(toTensor0(scalar), t)
    def /![T <: Tuple, V: IsFloating](t: Tensor[T, V])(using toTensor0: Conversion[Byte, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.divide_!(toTensor0(scalar), t)
    def `<!`[T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Byte, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.less_!(toTensor0(scalar), t)
    def <=![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Byte, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.lessEqual_!(toTensor0(scalar), t)
    def >![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Byte, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.greater_!(toTensor0(scalar), t)
    def >=![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Byte, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.greaterEqual_!(toTensor0(scalar), t)
    def ===![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Byte, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor0[Bool] = ElementWiseOps.arrayEqual_!(toTensor0(scalar), t)

  extension (scalar: Short)

    // operators with a Tensor0
    def +[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Short, Tensor0[V]]): Tensor0[V] = ElementWiseOps.add(toTensor0(scalar), t)
    def -[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Short, Tensor0[V]]): Tensor0[V] = ElementWiseOps.subtract(toTensor0(scalar), t)
    def *[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Short, Tensor0[V]]): Tensor0[V] = ElementWiseOps.multiply(toTensor0(scalar), t)
    def %[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Short, Tensor0[V]]): Tensor0[V] = ElementWiseOps.mod(toTensor0(scalar), t)
    def /[V: IsFloating](t: Tensor0[V])(using toTensor0: Conversion[Short, Tensor0[V]]): Tensor0[V] = ElementWiseOps.divide(toTensor0(scalar), t)
    def <[V](t: Tensor0[V])(using toTensor0: Conversion[Short, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.less(toTensor0(scalar), t)
    def <=[V](t: Tensor0[V])(using toTensor0: Conversion[Short, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.lessEqual(toTensor0(scalar), t)
    def >[V](t: Tensor0[V])(using toTensor0: Conversion[Short, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.greater(toTensor0(scalar), t)
    def >=[V](t: Tensor0[V])(using toTensor0: Conversion[Short, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.greaterEqual(toTensor0(scalar), t)
    def ===[V](t: Tensor0[V])(using toTensor0: Conversion[Short, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.arrayEqual(toTensor0(scalar), t)

    // broadcasting operators
    def +![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Short, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.add_!(toTensor0(scalar), t)
    def -![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Short, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.subtract_!(toTensor0(scalar), t)
    def *![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Short, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.multiply_!(toTensor0(scalar), t)
    def %![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Short, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.mod_!(toTensor0(scalar), t)
    def /![T <: Tuple, V: IsFloating](t: Tensor[T, V])(using toTensor0: Conversion[Short, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.divide_!(toTensor0(scalar), t)
    def `<!`[T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Short, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.less_!(toTensor0(scalar), t)
    def <=![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Short, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.lessEqual_!(toTensor0(scalar), t)
    def >![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Short, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.greater_!(toTensor0(scalar), t)
    def >=![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Short, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.greaterEqual_!(toTensor0(scalar), t)
    def ===![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Short, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor0[Bool] = ElementWiseOps.arrayEqual_!(toTensor0(scalar), t)

  extension (scalar: Int)

    // operators with a Tensor0
    def +[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Int, Tensor0[V]]): Tensor0[V] = ElementWiseOps.add(toTensor0(scalar), t)
    def -[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Int, Tensor0[V]]): Tensor0[V] = ElementWiseOps.subtract(toTensor0(scalar), t)
    def *[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Int, Tensor0[V]]): Tensor0[V] = ElementWiseOps.multiply(toTensor0(scalar), t)
    def %[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Int, Tensor0[V]]): Tensor0[V] = ElementWiseOps.mod(toTensor0(scalar), t)
    def /[V: IsFloating](t: Tensor0[V])(using toTensor0: Conversion[Int, Tensor0[V]]): Tensor0[V] = ElementWiseOps.divide(toTensor0(scalar), t)
    def <[V](t: Tensor0[V])(using toTensor0: Conversion[Int, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.less(toTensor0(scalar), t)
    def <=[V](t: Tensor0[V])(using toTensor0: Conversion[Int, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.lessEqual(toTensor0(scalar), t)
    def >[V](t: Tensor0[V])(using toTensor0: Conversion[Int, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.greater(toTensor0(scalar), t)
    def >=[V](t: Tensor0[V])(using toTensor0: Conversion[Int, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.greaterEqual(toTensor0(scalar), t)
    def ===[V](t: Tensor0[V])(using toTensor0: Conversion[Int, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.arrayEqual(toTensor0(scalar), t)

    // broadcasting operators
    def +![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Int, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.add_!(toTensor0(scalar), t)
    def -![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Int, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.subtract_!(toTensor0(scalar), t)
    def *![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Int, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.multiply_!(toTensor0(scalar), t)
    def %![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Int, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.mod_!(toTensor0(scalar), t)
    def /![T <: Tuple, V: IsFloating](t: Tensor[T, V])(using toTensor0: Conversion[Int, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.divide_!(toTensor0(scalar), t)
    def `<!`[T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Int, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.less_!(toTensor0(scalar), t)
    def <=![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Int, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.lessEqual_!(toTensor0(scalar), t)
    def >![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Int, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.greater_!(toTensor0(scalar), t)
    def >=![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Int, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.greaterEqual_!(toTensor0(scalar), t)
    def ===![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Int, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor0[Bool] = ElementWiseOps.arrayEqual_!(toTensor0(scalar), t)

  extension (scalar: Long)

    // operators with a Tensor0
    def +[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Long, Tensor0[V]]): Tensor0[V] = ElementWiseOps.add(toTensor0(scalar), t)
    def -[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Long, Tensor0[V]]): Tensor0[V] = ElementWiseOps.subtract(toTensor0(scalar), t)
    def *[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Long, Tensor0[V]]): Tensor0[V] = ElementWiseOps.multiply(toTensor0(scalar), t)
    def %[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Long, Tensor0[V]]): Tensor0[V] = ElementWiseOps.mod(toTensor0(scalar), t)
    def /[V: IsFloating](t: Tensor0[V])(using toTensor0: Conversion[Long, Tensor0[V]]): Tensor0[V] = ElementWiseOps.divide(toTensor0(scalar), t)
    def <[V](t: Tensor0[V])(using toTensor0: Conversion[Long, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.less(toTensor0(scalar), t)
    def <=[V](t: Tensor0[V])(using toTensor0: Conversion[Long, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.lessEqual(toTensor0(scalar), t)
    def >[V](t: Tensor0[V])(using toTensor0: Conversion[Long, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.greater(toTensor0(scalar), t)
    def >=[V](t: Tensor0[V])(using toTensor0: Conversion[Long, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.greaterEqual(toTensor0(scalar), t)
    def ===[V](t: Tensor0[V])(using toTensor0: Conversion[Long, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.arrayEqual(toTensor0(scalar), t)

    // broadcasting operators
    def +![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Long, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.add_!(toTensor0(scalar), t)
    def -![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Long, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.subtract_!(toTensor0(scalar), t)
    def *![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Long, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.multiply_!(toTensor0(scalar), t)
    def %![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Long, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.mod_!(toTensor0(scalar), t)
    def /![T <: Tuple, V: IsFloating](t: Tensor[T, V])(using toTensor0: Conversion[Long, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.divide_!(toTensor0(scalar), t)
    def `<!`[T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Long, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.less_!(toTensor0(scalar), t)
    def <=![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Long, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.lessEqual_!(toTensor0(scalar), t)
    def >![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Long, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.greater_!(toTensor0(scalar), t)
    def >=![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Long, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.greaterEqual_!(toTensor0(scalar), t)
    def ===![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Long, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor0[Bool] = ElementWiseOps.arrayEqual_!(toTensor0(scalar), t)

  extension (scalar: Float)

    // operators with a Tensor0
    def +[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Float, Tensor0[V]]): Tensor0[V] = ElementWiseOps.add(toTensor0(scalar), t)
    def -[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Float, Tensor0[V]]): Tensor0[V] = ElementWiseOps.subtract(toTensor0(scalar), t)
    def *[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Float, Tensor0[V]]): Tensor0[V] = ElementWiseOps.multiply(toTensor0(scalar), t)
    def %[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Float, Tensor0[V]]): Tensor0[V] = ElementWiseOps.mod(toTensor0(scalar), t)
    def /[V: IsFloating](t: Tensor0[V])(using toTensor0: Conversion[Float, Tensor0[V]]): Tensor0[V] = ElementWiseOps.divide(toTensor0(scalar), t)
    def <[V](t: Tensor0[V])(using toTensor0: Conversion[Float, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.less(toTensor0(scalar), t)
    def <=[V](t: Tensor0[V])(using toTensor0: Conversion[Float, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.lessEqual(toTensor0(scalar), t)
    def >[V](t: Tensor0[V])(using toTensor0: Conversion[Float, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.greater(toTensor0(scalar), t)
    def >=[V](t: Tensor0[V])(using toTensor0: Conversion[Float, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.greaterEqual(toTensor0(scalar), t)
    def ===[V](t: Tensor0[V])(using toTensor0: Conversion[Float, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.arrayEqual(toTensor0(scalar), t)

    // broadcasting operators
    def +![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Float, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.add_!(toTensor0(scalar), t)
    def -![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Float, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.subtract_!(toTensor0(scalar), t)
    def *![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Float, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.multiply_!(toTensor0(scalar), t)
    def %![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Float, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.mod_!(toTensor0(scalar), t)
    def /![T <: Tuple, V: IsFloating](t: Tensor[T, V])(using toTensor0: Conversion[Float, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.divide_!(toTensor0(scalar), t)
    def `<!`[T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Float, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.less_!(toTensor0(scalar), t)
    def <=![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Float, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.lessEqual_!(toTensor0(scalar), t)
    def >![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Float, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.greater_!(toTensor0(scalar), t)
    def >=![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Float, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.greaterEqual_!(toTensor0(scalar), t)
    def ===![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Float, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor0[Bool] = ElementWiseOps.arrayEqual_!(toTensor0(scalar), t)

  extension (scalar: Double)

    // operators with a Tensor0
    def +[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Double, Tensor0[V]]): Tensor0[V] = ElementWiseOps.add(toTensor0(scalar), t)
    def -[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Double, Tensor0[V]]): Tensor0[V] = ElementWiseOps.subtract(toTensor0(scalar), t)
    def *[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Double, Tensor0[V]]): Tensor0[V] = ElementWiseOps.multiply(toTensor0(scalar), t)
    def %[V: IsNumber](t: Tensor0[V])(using toTensor0: Conversion[Double, Tensor0[V]]): Tensor0[V] = ElementWiseOps.mod(toTensor0(scalar), t)
    def /[V: IsFloating](t: Tensor0[V])(using toTensor0: Conversion[Double, Tensor0[V]]): Tensor0[V] = ElementWiseOps.divide(toTensor0(scalar), t)
    def <[V](t: Tensor0[V])(using toTensor0: Conversion[Double, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.less(toTensor0(scalar), t)
    def <=[V](t: Tensor0[V])(using toTensor0: Conversion[Double, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.lessEqual(toTensor0(scalar), t)
    def >[V](t: Tensor0[V])(using toTensor0: Conversion[Double, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.greater(toTensor0(scalar), t)
    def >=[V](t: Tensor0[V])(using toTensor0: Conversion[Double, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.greaterEqual(toTensor0(scalar), t)
    def ===[V](t: Tensor0[V])(using toTensor0: Conversion[Double, Tensor0[V]]): Tensor0[Bool] = ElementWiseOps.arrayEqual(toTensor0(scalar), t)

    // broadcasting operators
    def +![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Double, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.add_!(toTensor0(scalar), t)
    def -![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Double, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.subtract_!(toTensor0(scalar), t)
    def *![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Double, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.multiply_!(toTensor0(scalar), t)
    def %![T <: Tuple, V: IsNumber](t: Tensor[T, V])(using toTensor0: Conversion[Double, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.mod_!(toTensor0(scalar), t)
    def /![T <: Tuple, V: IsFloating](t: Tensor[T, V])(using toTensor0: Conversion[Double, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = ElementWiseOps.divide_!(toTensor0(scalar), t)
    def `<!`[T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Double, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.less_!(toTensor0(scalar), t)
    def <=![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Double, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.lessEqual_!(toTensor0(scalar), t)
    def >![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Double, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.greater_!(toTensor0(scalar), t)
    def >=![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Double, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, Bool] = ElementWiseOps.greaterEqual_!(toTensor0(scalar), t)
    def ===![T <: Tuple, V](t: Tensor[T, V])(using toTensor0: Conversion[Double, Tensor0[V]], bc: Broadcast[EmptyTuple, T, V]): Tensor0[Bool] = ElementWiseOps.arrayEqual_!(toTensor0(scalar), t)
