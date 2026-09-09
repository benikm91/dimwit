package dimwit.sharding

import scala.quoted.*

/** A label for an axis of a device [[Mesh]], declared with `trait X derives MeshLabel`.
  * The mesh-side counterpart of [[dimwit.tensor.Label]].
  */
@scala.annotation.implicitNotFound("""
A mesh axis label ${T} was given or inferred, which does not have a MeshLabel instance.
Ensure that all mesh axis types ${T} are defined with 'derives MeshLabel' (e.g. 'trait X derives MeshLabel')
""")
trait MeshLabel[T]:
  def name: String

object MeshLabel:

  def apply[T](using meshLabel: MeshLabel[T]): MeshLabel[T] = meshLabel

  inline def derived[T]: MeshLabel[T] = ${ derivedMacro[T] }

  private def derivedMacro[T: Type](using Quotes): Expr[MeshLabel[T]] =
    import quotes.reflect.*
    val tpe = TypeRepr.of[T]
    val simpleName = tpe.typeSymbol.name
    '{
      new MeshLabel[T]:
        def name: String = ${ Expr(simpleName) }
    }

@scala.annotation.implicitNotFound("""
A tuple of mesh axis labels ${T} was given or inferred that does not have a valid MeshLabels instance.

Ensure that all of the types in the tuple have a 'derives MeshLabel' clause.
""")
trait MeshLabels[T]:
  def names: List[String]

private class MeshLabelsImpl[T](val names: List[String]) extends MeshLabels[T]

object MeshLabels:

  def apply[T](using labels: MeshLabels[T]): MeshLabels[T] = labels

  given emptyTuple: MeshLabels[EmptyTuple] = new MeshLabelsImpl[EmptyTuple](Nil)

  given lift[A](using v: MeshLabel[A]): MeshLabels[A] = new MeshLabelsImpl[A](List(v.name))

  given consTuple[H, T <: Tuple](using
      head: MeshLabel[H],
      tail: MeshLabels[T]
  ): MeshLabels[H *: T] = new MeshLabelsImpl[H *: T](head.name :: tail.names)
