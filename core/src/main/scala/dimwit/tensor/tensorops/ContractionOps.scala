package dimwit.tensor.tensorops

import dimwit.jax.Jax
import dimwit.tensor.Axis
import dimwit.tensor.Labels
import dimwit.tensor.ShapeTypeHelpers.AxisIndex
import dimwit.tensor.Tensor
import dimwit.tensor.TupleHelpers.PrimeConcat
import dimwit.tensor.TupleHelpers.Remove
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.readwrite.Writer

import scala.annotation.targetName

/** Provides extension methods for tensor contraction operations,
  * including outer products and dot products.
  */
object ContractionOps:

  extension [T <: Tuple: Labels, V](tensor: Tensor[T, V])

    /** Computes the outer product of this tensor with another tensor.
      * Automatically primes the labels of the resulting tensor to avoid label collisions.
      */
    def outerProduct[OtherShape <: Tuple: Labels](other: Tensor[OtherShape, V])(using
        labels: Labels[PrimeConcat[T, OtherShape]]
    ): Tensor[PrimeConcat[T, OtherShape], V] = Tensor(
      Jax.jnp.tensordot(tensor.jaxValue, other.jaxValue, axes = 0) // generalized outer product
    )

    /** Computes the dot product of this tensor with another tensor along the specified axis.
      * The axis must be present in both tensors and will be contracted (removed) from the resulting tensor.
      *
      * @param axis The axis along which to contract. Must be present in both tensors.
      * @param other The other tensor to contract with.
      */
    def dot[
        ContractAxis,
        OtherShape <: Tuple
    ](axis: Axis[ContractAxis])(other: Tensor[OtherShape, V])(using
        ev: AxisIndex[T, ContractAxis],
        evOther: AxisIndex[OtherShape, ContractAxis]
    )(using
        labelsOut: Labels[PrimeConcat[Remove[T, ContractAxis], Remove[OtherShape, ContractAxis]]]
    ): Tensor[PrimeConcat[Remove[T, ContractAxis], Remove[OtherShape, ContractAxis]], V] =
      val axesTuple1 = Jax.Dynamic.global.tuple(Seq(ev.index).toPythonProxy)
      val axesTuple2 = Jax.Dynamic.global.tuple(Seq(evOther.index).toPythonProxy)
      val axesPair = Jax.Dynamic.global.tuple(Seq(axesTuple1, axesTuple2).toPythonProxy)

      Tensor(Jax.jnp.tensordot(tensor.jaxValue, other.jaxValue, axes = axesPair))

    /** Computes the dot product of this tensor with another tensor along the specified pair of axes.
      * The axes must be present in their respective tensors and will be contracted (removed) from the resulting tensor.
      *
      * @param axis The pair of axes along which to contract. Each axis must be present in its respective tensor.
      * @param other The other tensor to contract with.
      *
      * Example usage:
      * {{{
      * val t1: Tensor[("A", "B", "C"), Float] = ???
      * val t2: Tensor[("D", "E, "F), Float] = ???
      * val result = t1.dot(Axis[A]->Axis[D])(t2)
      * }}}
      */
    @targetName("dotOn")
    def dot[
        ContractAxisA,
        ContractAxisB,
        OtherShape <: Tuple
    ](axisPair: (Axis[ContractAxisA], Axis[ContractAxisB]))(other: Tensor[OtherShape, V])(using
        ev: AxisIndex[T, ContractAxisA],
        evOther: AxisIndex[OtherShape, ContractAxisB]
    )(using
        outLabels: Labels[PrimeConcat[Remove[T, ContractAxisA], Remove[OtherShape, ContractAxisB]]]
    ): Tensor[PrimeConcat[Remove[T, ContractAxisA], Remove[OtherShape, ContractAxisB]], V] =
      val axesTuple1 = Jax.Dynamic.global.tuple(Seq(ev.index).toPythonProxy)
      val axesTuple2 = Jax.Dynamic.global.tuple(Seq(evOther.index).toPythonProxy)
      val axesPair = Jax.Dynamic.global.tuple(Seq(axesTuple1, axesTuple2).toPythonProxy)

      Tensor(Jax.jnp.tensordot(tensor.jaxValue, other.jaxValue, axes = axesPair))
