package dimwit.sharding

import scala.annotation.implicitNotFound

/** Represents an axis of a device [[Mesh]]. The mesh-side counterpart of [[dimwit.tensor.Axis]]. */
final class MeshAxis[A: MeshLabel]:

  def name: String = summon[MeshLabel[A]].name

  def extent(size: Int): MeshAxisExtent[A] = MeshAxisExtent(this, size)
  def ->(size: Int): MeshAxisExtent[A] = this.extent(size)

  override def toString: String = s"MeshAxis($name)"

/** A mesh axis together with the number of devices along it. */
case class MeshAxisExtent[A: MeshLabel](axis: MeshAxis[A], size: Int)

/** Finds the position of a mesh axis in a mesh. */
@implicitNotFound("MeshAxis[${A}] not found in Mesh[${M}]")
trait MeshAxisIndex[M <: Tuple, A]:
  def index: Int

object MeshAxisIndex:

  def apply[M <: Tuple, A](using idx: MeshAxisIndex[M, A]): Int = idx.index

  given found[A, Tail <: Tuple]: MeshAxisIndex[A *: Tail, A] with
    val index = 0

  given search[H, T <: Tuple, A](using next: MeshAxisIndex[T, A]): MeshAxisIndex[H *: T, A] with
    val index = 1 + next.index
