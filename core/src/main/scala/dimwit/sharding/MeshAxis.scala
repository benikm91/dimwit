package dimwit.sharding

/** Represents an axis of a device [[Mesh]]. The mesh-side counterpart of [[dimwit.tensor.Axis]]. */
final class MeshAxis[A: MeshLabel]:

  def name: String = summon[MeshLabel[A]].name

  def extent(size: Int): MeshAxisExtent[A] = MeshAxisExtent(this, size)
  def ->(size: Int): MeshAxisExtent[A] = this.extent(size)

  override def toString: String = s"MeshAxis($name)"

/** A mesh axis together with the number of devices along it. */
case class MeshAxisExtent[A: MeshLabel](axis: MeshAxis[A], size: Int)
