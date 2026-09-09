package dimwit.sharding

import dimwit.hardware.Device
import dimwit.jax.Jax
import me.shadaj.scalapy.py.SeqConverters

import scala.annotation.implicitNotFound

/** A grid of devices, binding each mesh axis label of `M` to a number of devices. */
final class Mesh[M <: Tuple: MeshLabels] private[sharding] (
    val axisSizes: List[Int],
    val devices: Seq[Device]
):

  lazy val axisNames: List[String] = MeshLabels[M].names

  def size: Int = axisSizes.product

  def sizeOf[A](meshAxis: MeshAxis[A])(using ev: MeshAxisIndex[M, A]): Int = axisSizes(ev.index)

  private[dimwit] lazy val jaxMesh: Jax.PyDynamic =
    Jax.jax_helper
      .device_mesh(devices.map(_.toJaxDevice).toPythonProxy, axisSizes.toPythonProxy, axisNames.toPythonProxy)
      .as[Jax.PyDynamic]

  override def toString: String =
    axisNames.zip(axisSizes).map((name, size) => s"$name -> $size").mkString("Mesh(", ", ", ")")

object Mesh:

  private[sharding] type ExtractLabel[Extent] = Extent match
    case MeshAxisExtent[a] => a

  private[sharding] type ExtractLabels[Extents <: Tuple] = Tuple.Map[Extents, ExtractLabel]

  def apply[Extents <: Tuple](extents: Extents)(using MeshLabels[ExtractLabels[Extents]]): Mesh[ExtractLabels[Extents]] =
    fromTuple(extents)

  def fromTuple[Extents <: Tuple](extents: Extents)(using labels: MeshLabels[ExtractLabels[Extents]]): Mesh[ExtractLabels[Extents]] =
    val sizes = extents.toList.collect:
      case extent: MeshAxisExtent[?] => extent.size
    val required = sizes.product
    val available = Jax.devices
    require(
      available.size >= required,
      s"Mesh ${labels.names.zip(sizes).map((name, size) => s"$name -> $size").mkString("(", ", ", ")")} requires $required devices, but only ${available.size} are available"
    )
    new Mesh(sizes, available.take(required))

type Mesh1[A] = Mesh[Tuple1[A]]
type Mesh2[A, B] = Mesh[(A, B)]
type Mesh3[A, B, C] = Mesh[(A, B, C)]

object Mesh1:
  def apply[A: MeshLabel](extent: MeshAxisExtent[A]): Mesh1[A] = Mesh.fromTuple(Tuple1(extent))

object Mesh2:
  def apply[A: MeshLabel, B: MeshLabel](extent1: MeshAxisExtent[A], extent2: MeshAxisExtent[B]): Mesh2[A, B] =
    Mesh.fromTuple((extent1, extent2))

object Mesh3:
  def apply[A: MeshLabel, B: MeshLabel, C: MeshLabel](
      extent1: MeshAxisExtent[A],
      extent2: MeshAxisExtent[B],
      extent3: MeshAxisExtent[C]
  ): Mesh3[A, B, C] = Mesh.fromTuple((extent1, extent2, extent3))

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
