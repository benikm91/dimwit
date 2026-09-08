package dimwit.sharding

import dimwit.*
import dimwit.hardware.DeviceBackend
import me.shadaj.scalapy.py

import scala.compiletime.testing.typeCheckErrors

/** Mesh axis labels: a different kind than the data axis labels `A`, `B`, ... in the test package. */
trait X derives MeshLabel
trait Y derives MeshLabel

class ShardingSuite extends DimwitTest:

  private val BatchExtent = 8
  private val FeatureExtent = 4
  private val MeshExtent = 4

  private def tensorOf(batch: Int, feature: Int = FeatureExtent): Tensor2[A, B, Float32] =
    Tensor2(Axis[A], Axis[B]).fromArray(Array.tabulate(batch, feature)((i, j) => (i * feature + j).toFloat))

  private val t: Tensor2[A, B, Float32] = tensorOf(BatchExtent)

  private def enoughDevices(): Unit =
    assume(
      DeviceBackend.CPU.devices.size >= MeshExtent,
      s"needs at least $MeshExtent devices, found ${DeviceBackend.CPU.devices.size}"
    )

  private def isFullyReplicated[T <: Tuple: Labels, V](tensor: Tensor[T, V]): Boolean =
    tensor.jaxValue.sharding.is_fully_replicated.as[Boolean]

  private def deviceCount[T <: Tuple: Labels, V](tensor: Tensor[T, V]): Int =
    tensor.jaxValue.sharding.num_devices.as[Int]

  private def mesh1 = Mesh1(MeshAxis[X] -> MeshExtent)

  private def sharded: Tensor2[A |@| X, B, Float32] = t.shard(mesh1, Axis[A] -> MeshAxis[X])

  describe("Mesh"):

    it("binds mesh axis labels to a grid of devices"):
      enoughDevices()
      val mesh = mesh1
      mesh.size shouldBe MeshExtent
      mesh.axisNames shouldBe List("X")
      mesh.axisSizes shouldBe List(MeshExtent)
      mesh.sizeOf(MeshAxis[X]) shouldBe MeshExtent
      mesh.devices.map(_.id).distinct should have size MeshExtent

    it("is built from a tuple of extents, with Mesh1/Mesh2 as special cases"):
      enoughDevices()
      val single: Mesh[Tuple1[X]] = Mesh(MeshAxis[X] -> MeshExtent)
      single.axisNames shouldBe List("X")
      single.axisSizes shouldBe List(MeshExtent)

      val grid: Mesh[(X, Y)] = Mesh((MeshAxis[X] -> 2, MeshAxis[Y] -> 2))
      grid.axisNames shouldBe List("X", "Y")
      grid.size shouldBe 4
      grid.sizeOf(MeshAxis[Y]) shouldBe 2

      Mesh1(MeshAxis[X] -> MeshExtent).axisSizes shouldBe single.axisSizes
      Mesh2(MeshAxis[X] -> 2, MeshAxis[Y] -> 2).axisNames shouldBe grid.axisNames

    it("fails with a clear error when there are not enough devices"):
      val error = intercept[IllegalArgumentException](Mesh1(MeshAxis[X] -> 1000000))
      error.getMessage should include("requires 1000000 devices")

  describe("shard"):

    it("rewrites the axis type and places the shards on the mesh devices"):
      enoughDevices()
      val result: Tensor2[A |@| X, B, Float32] = sharded
      result.axes shouldBe List("A@X", "B")
      result.shape.dimensions shouldBe List(BatchExtent, FeatureExtent)
      deviceCount(result) shouldBe MeshExtent
      isFullyReplicated(result) shouldBe false
      result shouldEqual t

    it("rejects an axis whose extent does not divide over the mesh axis"):
      enoughDevices()
      val error = intercept[IllegalArgumentException](tensorOf(7).shard(mesh1, Axis[A] -> MeshAxis[X]))
      error.getMessage shouldBe "Cannot shard axis A of extent 7 over mesh axis X of size 4: 7 is not divisible by 4."

  describe("a sharded axis is just another axis"):

    it("reducing the sharded axis all-reduces and equals the unsharded sum, bit for bit"):
      enoughDevices()
      val result: Tensor1[B, Float32] = sharded.sum(Axis[A |@| X])
      result shouldEqual t.sum(Axis[A])
      result.axes shouldBe List("B")
      isFullyReplicated(result) shouldBe true

    it("reducing every axis works on both, and is how one loss function serves both runs"):
      enoughDevices()
      sharded.sum shouldEqual t.sum
      sharded.mean shouldEqual t.mean

    it("reducing another axis is local and keeps the mesh annotation"):
      enoughDevices()
      val result: Tensor1[A |@| X, Float32] = sharded.sum(Axis[B])
      result shouldEqual t.sum(Axis[B])
      result.axes shouldBe List("A@X")
      isFullyReplicated(result) shouldBe false
      deviceCount(result) shouldBe MeshExtent

    it("other reductions over the sharded axis work too, with no extra support"):
      enoughDevices()
      sharded.mean(Axis[A |@| X]) shouldEqual t.mean(Axis[A])
      sharded.max(Axis[A |@| X]) shouldEqual t.max(Axis[A])
      sharded.min(Axis[A |@| X]) shouldEqual t.min(Axis[A])

    it("vmap maps over the sharded axis, which stays sharded"):
      enoughDevices()
      val result: Tensor2[A |@| X, B, Float32] = sharded.vmap(Axis[A |@| X])(row => row *! Tensor0(2.0f))
      result shouldEqual t.vmap(Axis[A])(row => row *! Tensor0(2.0f))
      result.axes shouldBe List("A@X", "B")
      deviceCount(result) shouldBe MeshExtent

    it("contracts a replicated tensor against the unsharded axis, staying sharded"):
      enoughDevices()
      val weights = Tensor1(Axis[B]).fromArray(Array.fill(FeatureExtent)(2.0f))
      val result: Tensor1[A |@| X, Float32] = sharded.dot(Axis[B])(weights)
      result shouldEqual t.dot(Axis[B])(weights)
      deviceCount(result) shouldBe MeshExtent

    it("broadcasts a replicated tensor over the axes it does share"):
      enoughDevices()
      val bias = Tensor1(Axis[B]).fromArray(Array.fill(FeatureExtent)(1.0f))
      val result: Tensor2[A |@| X, B, Float32] = sharded -! bias
      result shouldEqual (t -! bias)
      deviceCount(result) shouldBe MeshExtent

  describe("sharded and unsharded tensors do not mix"):

    it("a sharded tensor cannot be combined with an unsharded one"):
      enoughDevices()
      val errors = typeCheckErrors("sharded + t")
      errors should not be empty

    it("the sharded axis cannot be named without its mesh axis"):
      enoughDevices()
      val errors = typeCheckErrors("sharded.sum(Axis[A])")
      errors should not be empty
      errors.head.message should include("not found in Tensor")

  describe("mesh labels are a separate kind from data axis labels"):

    it("a data axis label cannot be used as a mesh axis label"):
      val errors = typeCheckErrors("MeshAxis[A]")
      errors should not be empty
      errors.head.message should include("MeshLabel")

    it("a data axis label cannot be used to build a mesh"):
      val errors = typeCheckErrors("Mesh1(MeshAxis[A] -> 4)")
      errors should not be empty

    it("a mesh axis label cannot be used as a data axis label"):
      val errors = typeCheckErrors("Axis[X]")
      errors should not be empty
      errors.head.message should include("Label")

    it("a data axis label cannot stand on the mesh side of |@|"):
      val errors = typeCheckErrors("summon[Label[A |@| B]]")
      errors should not be empty
