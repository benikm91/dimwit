package dimwit.tensor.tensorops

import dimwit.jax.Jax
import dimwit.tensor.DType.Bool
import dimwit.tensor.DType.Float32
import dimwit.tensor.DType.Int32
import dimwit.tensor.Labels
import dimwit.tensor.Tensor
import dimwit.tensor.Tensor0
import dimwit.tensor.ValueTypeClasses.IsBoolean
import dimwit.tensor.ValueTypeClasses.IsFloating
import dimwit.tensor.ValueTypeClasses.IsInteger
import dimwit.tensor.ValueTypeClasses.IsNumber
import dimwit.tensor.VType
import dimwit.tensor.Broadcast

private[dimwit] object ElementWiseOps:

  /** Tensors of the same type have the same labels, but their extents can still differ.
    * JAX would then silently broadcast axes of extent 1, so operations without `!` check that the extents match.
    */
  private[dimwit] def requireSameShape(t1: Tensor[?, ?], t2: Tensor[?, ?]): Unit =
    require(t1.shape.dimensions == t2.shape.dimensions, s"Shape mismatch: ${t1.shape} vs ${t2.shape}")

  // ---------------------------------------------------------
  // General operations on any tensor type
  // ---------------------------------------------------------
  /** Elementwise maximum of two tensors. */
  def maximum[T <: Tuple: Labels, V](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] =
    requireSameShape(t1, t2)
    Tensor(Jax.jnp.maximum(t1.jaxValue, t2.jaxValue))

  /** Elementwise minimum of two tensors. */
  def minimum[T <: Tuple: Labels, V](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] =
    requireSameShape(t1, t2)
    Tensor(Jax.jnp.minimum(t1.jaxValue, t2.jaxValue))

  /** Elementwise `<` of two tensors of the same shape. */
  def less[T <: Tuple: Labels, V](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, Bool] =
    requireSameShape(t1, t2)
    Tensor(Jax.jnp.less(t1.jaxValue, t2.jaxValue))

  /** Elementwise `<=` of two tensors of the same shape. */
  def lessEqual[T <: Tuple: Labels, V](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, Bool] =
    requireSameShape(t1, t2)
    Tensor(Jax.jnp.less_equal(t1.jaxValue, t2.jaxValue))

  /** Elementwise `>` of two tensors of the same shape. */
  def greater[T <: Tuple: Labels, V](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, Bool] =
    requireSameShape(t1, t2)
    Tensor(Jax.jnp.greater(t1.jaxValue, t2.jaxValue))

  /** Elementwise `>=` of two tensors of the same shape. */
  def greaterEqual[T <: Tuple: Labels, V](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, Bool] =
    requireSameShape(t1, t2)
    Tensor(Jax.jnp.greater_equal(t1.jaxValue, t2.jaxValue))

  /** Checks full array equality, returns true if all elements are equal. */
  def arrayEqual[T <: Tuple: Labels, V](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor0[Bool] =
    requireSameShape(t1, t2)
    Tensor0(Jax.jnp.array_equal(t1.jaxValue, t2.jaxValue))

  /** Elementwise equality of two tensors of the same shape. */
  def equal[T <: Tuple: Labels, V](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, Bool] =
    requireSameShape(t1, t2)
    Tensor(Jax.jnp.equal(t1.jaxValue, t2.jaxValue))

  /** Performs element-wise addition of two tensors of the same shape and type. */
  def add[T <: Tuple: Labels, V: IsNumber](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] =
    requireSameShape(t1, t2)
    Tensor(Jax.jnp.add(t1.jaxValue, t2.jaxValue))

  /** Returns a new tensor with each element negated. */
  def negate[T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.negative(t.jaxValue))

  /** Subtracts one tensor from another of the same shape and type, returning a new tensor. */
  def subtract[T <: Tuple: Labels, V: IsNumber](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] =
    requireSameShape(t1, t2)
    Tensor(Jax.jnp.subtract(t1.jaxValue, t2.jaxValue))

  /** Multiplies two tensors of the same shape and type element-wise, returning a new tensor. */
  def multiply[T <: Tuple: Labels, V: IsNumber](
      t1: Tensor[T, V],
      t2: Tensor[T, V]
  ): Tensor[T, V] =
    requireSameShape(t1, t2)
    Tensor(Jax.jnp.multiply(t1.jaxValue, t2.jaxValue))

  /** Multiplies each element of `t` by the scalar `s`. */
  def scale[T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V], s: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.multiply(t.jaxValue, s.jaxValue))

  /** Computes the element-wise remainder of `t1 / t2`, matching Python's `%` operator (the result takes the sign of the divisor). */
  def mod[T <: Tuple: Labels, V: IsNumber](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] =
    requireSameShape(t1, t2)
    Tensor(Jax.jnp.mod(t1.jaxValue, t2.jaxValue))

  // ---------------------------------------------------------
  // Operations on Floating tensors
  // ---------------------------------------------------------
  /** Divides two tensors of the same shape and type element-wise, returning a new tensor. */
  def divide[T <: Tuple: Labels, V: IsFloating](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] =
    requireSameShape(t1, t2)
    Tensor(Jax.jnp.divide(t1.jaxValue, t2.jaxValue))

  // ---------------------------------------------------------
  // IsBoolean operations
  // ---------------------------------------------------------
  /** Elementwise logical AND of two tensors of the same shape and type. */
  def logicalAnd[T <: Tuple: Labels, V: IsBoolean](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] =
    requireSameShape(t1, t2)
    Tensor(Jax.jnp.logical_and(t1.jaxValue, t2.jaxValue))

  /** Elementwise logical OR of two tensors of the same shape and type. */
  def logicalOr[T <: Tuple: Labels, V: IsBoolean](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] =
    requireSameShape(t1, t2)
    Tensor(Jax.jnp.logical_or(t1.jaxValue, t2.jaxValue))

  /** Elementwise logical XOR of two tensors of the same shape and type. */
  def logicalXor[T <: Tuple: Labels, V: IsBoolean](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] =
    requireSameShape(t1, t2)
    Tensor(Jax.jnp.logical_xor(t1.jaxValue, t2.jaxValue))

  /** Elementwise logical NOT of a tensor. */
  def logicalNot[T <: Tuple: Labels, V: IsBoolean](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.logical_not(t.jaxValue))

  // ---------------------------------------------------------
  // Broadcasting variants: like the operations above, but `t1` and `t2` are first broadcast to their common shape,
  // which is the one of the two shapes containing all axes of the other.
  // ---------------------------------------------------------

  /** Like [[maximum]], but broadcasts `t1` and `t2` to their common shape first. */
  def maximum_![T1 <: Tuple, T2 <: Tuple, V](t1: Tensor[T1, V], t2: Tensor[T2, V])(using bc: Broadcast[T1, T2, V]): Tensor[bc.Out, V] = bc.applyTo(t1, t2)(maximum)

  /** Like [[minimum]], but broadcasts `t1` and `t2` to their common shape first. */
  def minimum_![T1 <: Tuple, T2 <: Tuple, V](t1: Tensor[T1, V], t2: Tensor[T2, V])(using bc: Broadcast[T1, T2, V]): Tensor[bc.Out, V] = bc.applyTo(t1, t2)(minimum)

  /** Like [[less]], but broadcasts `t1` and `t2` to their common shape first. */
  def less_![T1 <: Tuple, T2 <: Tuple, V](t1: Tensor[T1, V], t2: Tensor[T2, V])(using bc: Broadcast[T1, T2, V]): Tensor[bc.Out, Bool] = bc.applyTo(t1, t2)(less)

  /** Like [[lessEqual]], but broadcasts `t1` and `t2` to their common shape first. */
  def lessEqual_![T1 <: Tuple, T2 <: Tuple, V](t1: Tensor[T1, V], t2: Tensor[T2, V])(using bc: Broadcast[T1, T2, V]): Tensor[bc.Out, Bool] = bc.applyTo(t1, t2)(lessEqual)

  /** Like [[greater]], but broadcasts `t1` and `t2` to their common shape first. */
  def greater_![T1 <: Tuple, T2 <: Tuple, V](t1: Tensor[T1, V], t2: Tensor[T2, V])(using bc: Broadcast[T1, T2, V]): Tensor[bc.Out, Bool] = bc.applyTo(t1, t2)(greater)

  /** Like [[greaterEqual]], but broadcasts `t1` and `t2` to their common shape first. */
  def greaterEqual_![T1 <: Tuple, T2 <: Tuple, V](t1: Tensor[T1, V], t2: Tensor[T2, V])(using bc: Broadcast[T1, T2, V]): Tensor[bc.Out, Bool] = bc.applyTo(t1, t2)(greaterEqual)

  /** Like [[arrayEqual]], but broadcasts `t1` and `t2` to their common shape first. */
  def arrayEqual_![T1 <: Tuple, T2 <: Tuple, V](t1: Tensor[T1, V], t2: Tensor[T2, V])(using bc: Broadcast[T1, T2, V]): Tensor0[Bool] =
    val (bt1, bt2) = bc.broadcast(t1, t2)
    arrayEqual(bt1, bt2)(using bc.labelsOut)

  /** Like [[equal]], but broadcasts `t1` and `t2` to their common shape first. */
  def equal_![T1 <: Tuple, T2 <: Tuple, V](t1: Tensor[T1, V], t2: Tensor[T2, V])(using bc: Broadcast[T1, T2, V]): Tensor[bc.Out, Bool] = bc.applyTo(t1, t2)(equal)

  /** Like [[add]], but broadcasts `t1` and `t2` to their common shape first. */
  def add_![T1 <: Tuple, T2 <: Tuple, V: IsNumber](t1: Tensor[T1, V], t2: Tensor[T2, V])(using bc: Broadcast[T1, T2, V]): Tensor[bc.Out, V] = bc.applyTo(t1, t2)(add)

  /** Like [[subtract]], but broadcasts `t1` and `t2` to their common shape first. */
  def subtract_![T1 <: Tuple, T2 <: Tuple, V: IsNumber](t1: Tensor[T1, V], t2: Tensor[T2, V])(using bc: Broadcast[T1, T2, V]): Tensor[bc.Out, V] = bc.applyTo(t1, t2)(subtract)

  /** Like [[multiply]], but broadcasts `t1` and `t2` to their common shape first. */
  def multiply_![T1 <: Tuple, T2 <: Tuple, V: IsNumber](t1: Tensor[T1, V], t2: Tensor[T2, V])(using bc: Broadcast[T1, T2, V]): Tensor[bc.Out, V] = bc.applyTo(t1, t2)(multiply)

  /** Like [[mod]], but broadcasts `t1` and `t2` to their common shape first. */
  def mod_![T1 <: Tuple, T2 <: Tuple, V: IsNumber](t1: Tensor[T1, V], t2: Tensor[T2, V])(using bc: Broadcast[T1, T2, V]): Tensor[bc.Out, V] = bc.applyTo(t1, t2)(mod)

  /** Like [[divide]], but broadcasts `t1` and `t2` to their common shape first. */
  def divide_![T1 <: Tuple, T2 <: Tuple, V: IsFloating](t1: Tensor[T1, V], t2: Tensor[T2, V])(using bc: Broadcast[T1, T2, V]): Tensor[bc.Out, V] = bc.applyTo(t1, t2)(divide)

  /** Like [[logicalAnd]], but broadcasts `t1` and `t2` to their common shape first. */
  def logicalAnd_![T1 <: Tuple, T2 <: Tuple, V: IsBoolean](t1: Tensor[T1, V], t2: Tensor[T2, V])(using bc: Broadcast[T1, T2, V]): Tensor[bc.Out, V] = bc.applyTo(t1, t2)(logicalAnd)

  /** Like [[logicalOr]], but broadcasts `t1` and `t2` to their common shape first. */
  def logicalOr_![T1 <: Tuple, T2 <: Tuple, V: IsBoolean](t1: Tensor[T1, V], t2: Tensor[T2, V])(using bc: Broadcast[T1, T2, V]): Tensor[bc.Out, V] = bc.applyTo(t1, t2)(logicalOr)

  /** Like [[logicalXor]], but broadcasts `t1` and `t2` to their common shape first. */
  def logicalXor_![T1 <: Tuple, T2 <: Tuple, V: IsBoolean](t1: Tensor[T1, V], t2: Tensor[T2, V])(using bc: Broadcast[T1, T2, V]): Tensor[bc.Out, V] = bc.applyTo(t1, t2)(logicalXor)

  // ---------------------------------------------------------
  // Unary operations
  // ---------------------------------------------------------

  /** elementwise absolute value of `t`. */
  def abs[T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.abs(t.jaxValue))

  /** elementwise sign (-1, 0 or 1) of `t`. */
  def sign[T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.sign(t.jaxValue))

  /** clips the elements of `t` to the range [`min`, `max`]. */
  def clip[T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V], min: Tensor0[V], max: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.clip(t.jaxValue, min.jaxValue, max.jaxValue))

  /** raises each element of `t` to the power `n`. */
  def pow[T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V], n: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.power(t.jaxValue, n.jaxValue))

  // Elementwise operations on floating tensors

  /** elementwise square root of `t`. */
  def sqrt[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.sqrt(t.jaxValue))

  /** elementwise exponential of `t`. */
  def exp[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.exp(t.jaxValue))

  /** elementwise natural logarithm of `t`. */
  def log[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.log(t.jaxValue))

  /** elementwise sine of `t`. */
  def sin[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.sin(t.jaxValue))

  /** elementwise cosine of `t`. */
  def cos[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.cos(t.jaxValue))

  /** elementwise hyperbolic tangent of `t`. */
  def tanh[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.tanh(t.jaxValue))

  /** elementwise inverse sine of `t`. */
  def arcsin[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.arcsin(t.jaxValue))

  /** elementwise inverse cosine of `t`. */
  def arccos[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.arccos(t.jaxValue))

  /** elementwise inverse tangent of `t`. */
  def arctan[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.arctan(t.jaxValue))

  /** elementwise floor of `t`. */
  def floor[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.floor(t.jaxValue))

  /** elementwise ceiling of `t`. */
  def ceil[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.ceil(t.jaxValue))

  /** elementwise rounding to the nearest integer of `t`. */
  def round[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.round(t.jaxValue))

  /** elementwise test for NaN. */
  def isnan[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, Bool] = Tensor(Jax.jnp.isnan(t.jaxValue))

  /** elementwise test for finiteness (not NaN and not ±inf). */
  def isfinite[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, Bool] = Tensor(Jax.jnp.isfinite(t.jaxValue))

  /** replaces NaN by `nan`, +inf by `posInf` and -inf by `negInf`.
    * By default, ±inf become the largest/smallest finite value of the dtype.
    */
  def nanToNum[T <: Tuple: Labels, V](t: Tensor[T, V])(using
      IsFloating[V]
  )(
      nan: Tensor0[V],
      posInf: Tensor0[V] = IsFloating[V].maxFinite,
      negInf: Tensor0[V] = IsFloating[V].minFinite
  ): Tensor[T, V] =
    Tensor(Jax.jnp.nan_to_num(t.jaxValue, nan = nan.jaxValue, posinf = posInf.jaxValue, neginf = negInf.jaxValue))

  /** elementwise sigmoid activation of `t`. */
  def sigmoid[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnn.sigmoid(t.jaxValue))

  /** elementwise ReLU activation of `t`. */
  def relu[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnn.relu(t.jaxValue))

  /** elementwise GELU activation of `t`. */
  def gelu[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnn.gelu(t.jaxValue))

  /** returns true if all elements of `t` and `other` are equal within `tolerance`. */
  def approxEquals[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V], other: Tensor[T, V], tolerance: Float = 1e-6f): Tensor0[Bool] =
    all(approxElementEquals(t, other, tolerance))

  /** compares `t` and `other` elementwise within `tolerance`. */
  def approxElementEquals[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V], other: Tensor[T, V], tolerance: Float = 1e-6f): Tensor[T, Bool] =
    requireSameShape(t, other)
    Tensor(
      Jax.jnp.isclose(
        t.jaxValue,
        other.jaxValue,
        atol = tolerance,
        rtol = tolerance
      )
    )

  /** Like [[approxEquals]], but broadcasts `t1` and `t2` to their common shape first. */
  def approxEquals_![T1 <: Tuple, T2 <: Tuple, V: IsFloating](t1: Tensor[T1, V], t2: Tensor[T2, V], tolerance: Float = 1e-6f)(using bc: Broadcast[T1, T2, V]): Tensor0[Bool] =
    all(approxElementEquals_!(t1, t2, tolerance))

  /** Like [[approxElementEquals]], but broadcasts `t1` and `t2` to their common shape first. */
  def approxElementEquals_![T1 <: Tuple, T2 <: Tuple, V: IsFloating](t1: Tensor[T1, V], t2: Tensor[T2, V], tolerance: Float = 1e-6f)(using bc: Broadcast[T1, T2, V]): Tensor[bc.Out, Bool] =
    bc.applyTo(t1, t2)((a, b) => approxElementEquals(a, b, tolerance))

  // Operations on boolean tensors

  /** returns true if all elements of `t` are true, false otherwise */
  def all[T <: Tuple: Labels, V: IsBoolean](t: Tensor[T, V]): Tensor0[V] = Tensor0(Jax.jnp.all(t.jaxValue))

  /** returns true if any element of `t` is true, false otherwise */
  def any[T <: Tuple: Labels, V: IsBoolean](t: Tensor[T, V]): Tensor0[V] = Tensor0(Jax.jnp.any(t.jaxValue))

private[dimwit] object ElementWiseExtensions:

  import ElementWiseOps.{add, add_!, divide, divide_!, equal, equal_!, greater, greater_!, greaterEqual, greaterEqual_!, less, less_!, lessEqual, lessEqual_!, logicalAnd, logicalAnd_!, logicalNot, logicalOr, logicalOr_!, logicalXor, logicalXor_!, mod, mod_!, multiply, multiply_!, negate, subtract, subtract_!}

  // extension methods for comparisons
  extension [T <: Tuple: Labels, V](t: Tensor[T, V])

    def <(other: Tensor[T, V]): Tensor[T, Bool] = less(t, other)
    def <=(other: Tensor[T, V]): Tensor[T, Bool] = lessEqual(t, other)
    def >(other: Tensor[T, V]): Tensor[T, Bool] = greater(t, other)
    def >=(other: Tensor[T, V]): Tensor[T, Bool] = greaterEqual(t, other)

    /** Like [[<]], but broadcasts both sides to their common shape first.
      *
      * Must be written backticked (``a `<!` b``) or dotted (`a.<!(b)`): bare infix `a <! b` does not parse, because the
      * lexer reads `<!` as the start of an XML literal. The other broadcasting comparisons have no such restriction.
      */
    def `<!`[O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, Bool] = less_!(t, other)

    /** Like [[<=]], but broadcasts both sides to their common shape first. */
    def <=![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, Bool] = lessEqual_!(t, other)

    /** Like [[>]], but broadcasts both sides to their common shape first. */
    def >![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, Bool] = greater_!(t, other)

    /** Like [[>=]], but broadcasts both sides to their common shape first. */
    def >=![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, Bool] = greaterEqual_!(t, other)

    /** Checks full array equality, returns true if all elements are equal */
    def ===(other: Tensor[T, V]): Tensor0[Bool] = ElementWiseOps.arrayEqual(t, other)

    /** Like [[===]], but broadcasts both sides to their common shape first. */
    def ===![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor0[Bool] = ElementWiseOps.arrayEqual_!(t, other)

    /** Elementwise equality, returns a tensor of bools indicating which elements are equal */
    def elementEquals(other: Tensor[T, V]): Tensor[T, Bool] = equal(t, other)

    /** Like [[elementEquals]], but broadcasts both sides to their common shape first. */
    def elementEquals_![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, Bool] = equal_!(t, other)

    /** Casts the elements of this tensor to a tensor of type Bool. */
    def asBool: Tensor[T, Bool] = t.asType(VType[Bool])

    /** Casts the elements of this tensor to a tensor of the given boolean type.
      * @param vtype the type to cast to
      */
    def asBoolean[NewV: IsBoolean](vtype: VType[NewV]): Tensor[T, NewV] = t.asType(vtype)

    /** Cast the elements of this tensor to a tensor of type Int32. */
    def asInt32: Tensor[T, Int32] = t.asType(VType[Int32])

    /** Casts the elements of this tensor to a tensor of the given integer type.
      *
      * @param vtype - the type to cast to
      */
    def asInt[NewV: IsInteger](vtype: VType[NewV]): Tensor[T, NewV] = t.asType(vtype)

    /** Casts the elements of this tensor to a tensor of type Float32. */
    def asFloat32: Tensor[T, Float32] = t.asType(VType[Float32])

    /** Casts the elements of this tensor to a tensor of the given floating point type.
      *
      * @param vtype - the type to cast to
      */
    def asFloat[NewV: IsFloating](vtype: VType[NewV]): Tensor[T, NewV] = t.asType(vtype)

  // extension methods for the binary operations on two tensors
  extension [T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])

    def +(other: Tensor[T, V]): Tensor[T, V] = add(t, other)
    def -(other: Tensor[T, V]): Tensor[T, V] = subtract(t, other)
    def *(other: Tensor[T, V]): Tensor[T, V] = multiply(t, other)
    def %(other: Tensor[T, V]): Tensor[T, V] = mod(t, other)

  // extension methods for the scalar operations.
  extension [T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])

    def +![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = add_!(t, other)

    def unary_- : Tensor[T, V] = negate(t)
    def -![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = subtract_!(t, other)

    def *![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = multiply_!(t, other)
    def scale(other: Tensor0[V]): Tensor[T, V] = ElementWiseOps.scale(t, other)
    def %![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = mod_!(t, other)

  // extension methods
  extension [T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])
    def abs: Tensor[T, V] = ElementWiseOps.abs(t)
    def sign: Tensor[T, V] = ElementWiseOps.sign(t)
    def clip(min: Tensor0[V], max: Tensor0[V]): Tensor[T, V] = ElementWiseOps.clip(t, min, max)
    def pow(n: Tensor0[V]): Tensor[T, V] = ElementWiseOps.pow(t, n)

  // extension methods on floating tensors
  extension [T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V])

    def /(other: Tensor[T, V]): Tensor[T, V] = divide(t, other)
    def /![O <: Tuple](other: Tensor[O, V])(using join: Broadcast[T, O, V]): Tensor[join.Out, V] = divide_!(t, other)

    def sqrt: Tensor[T, V] = ElementWiseOps.sqrt(t)
    def exp: Tensor[T, V] = ElementWiseOps.exp(t)
    def log: Tensor[T, V] = ElementWiseOps.log(t)
    def sin: Tensor[T, V] = ElementWiseOps.sin(t)
    def cos: Tensor[T, V] = ElementWiseOps.cos(t)
    def tanh: Tensor[T, V] = ElementWiseOps.tanh(t)
    def arcsin: Tensor[T, V] = ElementWiseOps.arcsin(t)
    def arccos: Tensor[T, V] = ElementWiseOps.arccos(t)
    def arctan: Tensor[T, V] = ElementWiseOps.arctan(t)
    def floor: Tensor[T, V] = ElementWiseOps.floor(t)
    def ceil: Tensor[T, V] = ElementWiseOps.ceil(t)
    def round: Tensor[T, V] = ElementWiseOps.round(t)
    def isnan: Tensor[T, Bool] = ElementWiseOps.isnan(t)
    def isfinite: Tensor[T, Bool] = ElementWiseOps.isfinite(t)

    /** replaces NaN by `nan`, +inf by `posInf` and -inf by `negInf`.
      * By default, ±inf become the largest/smallest finite value of the dtype.
      */
    def nanToNum(using
        ev: IsFloating[V]
    )(
        nan: Tensor0[V],
        posInf: Tensor0[V] = IsFloating[V].maxFinite,
        negInf: Tensor0[V] = IsFloating[V].minFinite
    ): Tensor[T, V] =
      ElementWiseOps.nanToNum(t)(using ev)(nan, posInf, negInf)

    // activation functions
    def sigmoid: Tensor[T, V] = ElementWiseOps.sigmoid(t)
    def relu: Tensor[T, V] = ElementWiseOps.relu(t)
    def gelu: Tensor[T, V] = ElementWiseOps.gelu(t)

    def approxEquals(other: Tensor[T, V], tolerance: Float = 1e-6f): Tensor0[Bool] = ElementWiseOps.approxEquals(t, other, tolerance)
    def approxElementEquals(other: Tensor[T, V], tolerance: Float = 1e-6f): Tensor[T, Bool] = ElementWiseOps.approxElementEquals(t, other, tolerance)
    def approxEquals_![O <: Tuple](other: Tensor[O, V], tolerance: Float = 1e-6f)(using bc: Broadcast[T, O, V]): Tensor0[Bool] = ElementWiseOps.approxEquals_!(t, other, tolerance)
    def approxElementEquals_![O <: Tuple](other: Tensor[O, V], tolerance: Float = 1e-6f)(using bc: Broadcast[T, O, V]): Tensor[bc.Out, Bool] = ElementWiseOps.approxElementEquals_!(t, other, tolerance)

  extension [T <: Tuple: Labels, V: IsBoolean](t: Tensor[T, V])

    /** returns true if all elements of the tensor are true, false otherwise */
    def all: Tensor0[V] = ElementWiseOps.all(t)

    /** return true if any element of the tensor is true, false otherwise */
    def any: Tensor0[V] = ElementWiseOps.any(t)

    /** returns a tensor of the same shape with each element negated (logical NOT) */
    def unary_! : Tensor[T, V] = logicalNot(t)

    /** elementwise logical AND with another tensor of the same shape */
    infix def and(other: Tensor[T, V]): Tensor[T, V] = logicalAnd(t, other)

    /** elementwise logical OR with another tensor of the same shape */
    infix def or(other: Tensor[T, V]): Tensor[T, V] = logicalOr(t, other)

    /** elementwise logical XOR with another tensor of the same shape */
    infix def xor(other: Tensor[T, V]): Tensor[T, V] = logicalXor(t, other)

    /** elementwise logical AND with a broadcastable tensor */
    infix def and_![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = logicalAnd_!(t, other)

    /** elementwise logical OR with a broadcastable tensor */
    infix def or_![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = logicalOr_!(t, other)

    /** elementwise logical XOR with a broadcastable tensor */
    infix def xor_![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = logicalXor_!(t, other)
