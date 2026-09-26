package dimwit.tensor

import dimwit.jax.Jax

import scala.annotation.implicitNotFound

/** Type classes on the value type `V` of a `Tensor[T, V]`, e.g. `IsFloating[V]` for floating point tensors. */
object ValueTypeClasses:

  /** Typeclass to map a type V to its corresponding DType.
    */
  sealed trait HasDType[V]:
    def dtype: DType

  /** Typeclass to indicate that a type V is a numeric type
    */
  @implicitNotFound("Operation only valid for Numeric (Int or Float) tensors.")
  sealed trait IsNumber[V]

  @implicitNotFound("Operation only valid for Int or Float tensors.")
  object IsNumber:
    given [V](using ev1: IsFloating[V]): IsNumber[V] = ev1
    given [V](using ev2: IsInteger[V]): IsNumber[V] = ev2

  /** Type class marker for floating point types (Float32, Float64, etc.). */
  @implicitNotFound("Operation only valid for Floating tensors.")
  trait IsFloating[V] extends IsNumber[V], HasDType[V]:
    def dtype: DType

    /** the largest finite value representable by V. */
    def maxFinite: Tensor0[V] = Tensor0(Jax.jnp.array(Jax.jnp.finfo(dtype.jaxType).max, dtype = dtype.jaxType))

    /** the smallest (most negative) finite value representable by V. */
    def minFinite: Tensor0[V] = Tensor0(Jax.jnp.array(Jax.jnp.finfo(dtype.jaxType).min, dtype = dtype.jaxType))

  object IsFloating:
    def apply[V](using ev: IsFloating[V]): IsFloating[V] = ev

  /** Type class marker for integer types */
  @implicitNotFound("Operation only valid for Integer tensors.")
  trait IsInteger[V] extends IsNumber[V], HasDType[V]:
    def dtype: DType

  object IsInteger:
    def apply[V](using ev: IsInteger[V]): IsInteger[V] = ev

  /** Type class marker for Boolean types */
  @implicitNotFound("Operation only valid for Boolean tensors.")
  trait IsBoolean[V] extends HasDType[V]:
    def dtype: DType

  object IsBoolean:
    def apply[V](using ev: IsBoolean[V]): IsBoolean[V] = ev
