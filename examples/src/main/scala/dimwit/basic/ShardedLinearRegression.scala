package dimwit.examples.basic

import dimwit.Conversions.given
import dimwit.*
import dimwit.autodiff.*
import dimwit.jax.Jax
import dimwit.optimizer.GradientDescent
import dimwit.random.Random
import dimwit.stats.Normal

object ShardedLinearRegression:

  val Devices = 4
  val BatchSize = 8192 // must be divisible by Devices
  val FeatureSize = 64
  val Steps = 200
  val LearningRate = 0.05f

  trait Batch derives Label
  trait Feature derives Label

  case class Params(weights: Tensor1[Feature, Float32], bias: Tensor0[Float32])
  object Params:
    def init: Params = Params(Tensor1(Axis[Feature]).fromArray(Array.fill(FeatureSize)(0f)), Tensor0(0f))

  class LinearRegression(params: Params) extends (Tensor1[Feature, Float32] => Tensor0[Float32]):
    def apply(x: Tensor1[Feature, Float32]): Tensor0[Float32] = x.dot(Axis[Feature])(params.weights) + params.bias

  @main def runSingleDevice(): Unit =
    dimwit.initialize()

    val (x, y) = dataset(Random.Key(0))

    def loss(params: Params): Tensor0[Float32] =
      val model = LinearRegression(params)
      val predictions = x.vmap(Axis[Batch])(model)
      // `sum` names no axis, so this line is identical in the sharded run.
      (predictions - y).pow(2f).sum / Tensor0(BatchSize.toFloat)

    val descent = GradientDescent(Tensor0(LearningRate)).iterate(Params.init)(grad(jit(loss)))
    descent.next().bias.item // warm up JIT
    val start = System.nanoTime()
    val params = descent.drop(Steps - 1).next()
    params.bias.item
    report("single device", params, (System.nanoTime() - start) / 1e9)

  @main def runSharded(): Unit =
    dimwit.initialize()
    Jax.jax.config.update("jax_num_cpu_devices", Devices)

    val (x, y) = dataset(Random.Key(0))

    trait X derives MeshLabel
    val mesh = Mesh1(MeshAxis[X] -> Devices)

    // Inputs and targets are split the same way; the parameters stay replicated.
    val shardedX: Tensor2[Batch |@| X, Feature, Float32] = x.shard(mesh, Axis[Batch] -> MeshAxis[X])
    val shardedY: Tensor1[Batch |@| X, Float32] = y.shard(mesh, Axis[Batch] -> MeshAxis[X])

    println(s"$mesh over devices ${mesh.devices.map(_.id).mkString(", ")}")
    println(s"batch ${shardedX.axes.mkString(" x ")}, ${BatchSize / Devices} rows per device")

    def loss(params: Params): Tensor0[Float32] =
      val model = LinearRegression(params)
      // The batch axis is called `Batch |@| X` now; `vmap` treats it like any other.
      val predictions = shardedX.vmap(Axis[Batch |@| X])(model)
      (predictions - shardedY).pow(2f).sum / Tensor0(BatchSize.toFloat)

    val descent = GradientDescent(Tensor0(LearningRate)).iterate(Params.init)(grad(jit(loss)))
    descent.next().bias.item // warm up JIT
    val start = System.nanoTime()
    val params = descent.drop(Steps - 1).next()
    params.bias.item
    report("sharded", params, (System.nanoTime() - start) / 1e9)

  private def dataset(key: Random.Key): (Tensor2[Batch, Feature, Float32], Tensor1[Batch, Float32]) =
    val (xKey, noiseKey) = key.split2()
    val x = Normal.standardNormal(Shape(Axis[Batch] -> BatchSize, Axis[Feature] -> FeatureSize)).sample(xKey)
    val trueWeights = Tensor1(Axis[Feature]).fromArray(Array.tabulate(FeatureSize)(i => (i % 5).toFloat - 2f))
    val noise = Normal.standardNormal(Shape(Axis[Batch] -> BatchSize)).sample(noiseKey) *! Tensor0(0.1f)
    (x, x.dot(Axis[Feature])(trueWeights) + noise)

  private def report(setting: String, params: Params, seconds: Double): Unit =
    println(f"$setting%-14s $Steps steps in $seconds%6.3f s  (${seconds / Steps * 1000}%5.2f ms/step)")
    println(f"${" "}%-14s final bias ${params.bias.item}%+.4f, weights[0] ${params.weights.slice(Axis[Feature].at(0)).item}%+.4f")
