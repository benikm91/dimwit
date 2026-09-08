package dimwit.examples.basic

import dimwit.*
import dimwit.jax.Jax

/** Summing a tensor whose `Batch` axis is split across several devices. */
object ShardedSum:

  trait Batch derives Label
  trait Feature derives Label
  trait X derives MeshLabel

  /** How many CPU devices to run on. `BatchSize` has to be divisible by this. */
  val Devices = 4

  val BatchSize = 8

  val FeatureSize = 4

  @main def runShardedSum(): Unit =
    dimwit.initialize()
    Jax.jax.config.update("jax_num_cpu_devices", Devices)

    val data: Tensor2[Batch, Feature, Float32] =
      Tensor2(Axis[Batch], Axis[Feature]).fromArray(
        Array.tabulate(BatchSize, FeatureSize)((batch, feature) => (batch * FeatureSize + feature).toFloat)
      )
    val reference: Tensor1[Feature, Float32] = data.sum(Axis[Batch])

    val mesh = Mesh1(MeshAxis[X] -> Devices)
    println(s"$mesh on devices ${mesh.devices.map(_.id).mkString(", ")}")

    val sharded = data.shard(mesh, Axis[Batch] -> MeshAxis[X])
    println(s"sharded axes: ${sharded.axes.mkString(", ")} (${BatchSize / Devices} rows per device)")

    // Reducing the sharded axis is the ordinary `sum`; XLA makes it an all-reduce.
    val total: Tensor1[Feature, Float32] = sharded.sum(Axis[Batch |@| X])

    println(s"unsharded sum: $reference")
    println(s"sharded sum:   $total")
    println(s"identical:     ${total == reference}")

    // Naming no axis works on both, so one function can serve a sharded and an unsharded run.
    println(s"total of everything, either way: ${data.sum} / ${sharded.sum}")

    // Reducing an unsharded axis needs no communication; Batch stays spread over X.
    val perRow: Tensor1[Batch |@| X, Float32] = sharded.sum(Axis[Feature])
    println(s"per-row sums, still sharded over X: $perRow")

    val doubled: Tensor2[Batch |@| X, Feature, Float32] =
      sharded.vmap(Axis[Batch |@| X])(row => row *! Tensor0(2.0f))
    println(s"each row doubled, still sharded over X: ${doubled.axes.mkString(", ")}")
    println(s"mean over the sharded axis: ${sharded.mean(Axis[Batch |@| X])}")

    // Things to try, each of which should fail:
    //   sharded.sum(Axis[Batch]) - the sharded axis is called Batch |@| X now (compile error)
    //   sharded + data           - sharded and unsharded tensors do not mix (compile error)
    //   MeshAxis[Batch]          - a data axis label is not a mesh label (compile error)
    //   Axis[X]                  - and a mesh label is not a data axis label (compile error)
    //   BatchSize = 7            - 7 rows do not divide over 4 devices (runtime error)
