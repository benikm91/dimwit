package examples.basic.aecnn

import dimwit.*
import dimwit.Conversions.given

import examples.timed
import dimwit.stats.Normal
import dimwit.random.Random
import nn.LinearLayer
import nn.ActivationFunctions.relu
import nn.GradientDescent
import dimwit.jax.Jax
import nn.ActivationFunctions.sigmoid
import dimwit.random.Random.Key

import examples.dataset.MNISTLoader

import MNISTLoader.{Sample, TrainSample, TestSample, Height, Width}
import nn.Conv2DLayer
import nn.TransposeConv2DLayer
import nn.Lion
trait Channel derives Label
trait Hidden derives Label
trait Output derives Label

type ReconstructedChannel = Channel

trait EHidden1 derives Label
trait EHidden2 derives Label

trait Latent derives Label

trait DHidden1 derives Label
trait DHidden2 derives Label

trait Batch derives Label

class Encoder(p: Encoder.EncoderParams):

  val layer1 = Conv2DLayer(p.layer1, stride = 2, padding = Padding.SAME)
  val layer2 = Conv2DLayer(p.layer2, stride = 2, padding = Padding.SAME)
  val latentLayer = Conv2DLayer(p.latentLayer, stride = 1, padding = Padding.SAME)

  def apply(v: Tensor3[Height, Width, Channel, Float]): Tensor3[Height, Width, Latent, Float] =
    val h1 = relu(layer1(v))
    val h2 = relu(layer2(h1))
    latentLayer(h2)

object Encoder:
  case class EncoderParams(
      layer1: Conv2DLayer.Params[Height, Width, Channel, EHidden1],
      layer2: Conv2DLayer.Params[Height, Width, EHidden1, EHidden2],
      latentLayer: Conv2DLayer.Params[Height, Width, EHidden2, Latent]
  )

class Decoder(p: Decoder.DecoderParams):

  val layer1 = TransposeConv2DLayer(p.layer1, stride = 2, padding = Padding.SAME)
  val layer2 = TransposeConv2DLayer(p.layer2, stride = 2, padding = Padding.SAME)
  val outputLayer = TransposeConv2DLayer(p.outputLayer, stride = 1, padding = Padding.SAME)

  def apply(v: Tensor3[Height, Width, Latent, Float]): Tensor3[Height, Width, ReconstructedChannel, Float] =
    val h1 = relu(layer1(v))
    val h2 = relu(layer2(h1))
    sigmoid(outputLayer(h2))

object Decoder:
  case class DecoderParams(
      layer1: TransposeConv2DLayer.Params[Height, Width, DHidden1, Latent],
      layer2: TransposeConv2DLayer.Params[Height, Width, DHidden2, DHidden1],
      outputLayer: TransposeConv2DLayer.Params[Height, Width, ReconstructedChannel, DHidden2]
  )

case class AutoencoderCNN(params: AutoencoderCNN.Params):

  val encoder = Encoder(params.encoderParams)
  val decoder = Decoder(params.decoderParams)

  def apply(v: Tensor3[Height, Width, Channel, Float]): (Tensor3[Height, Width, ReconstructedChannel, Float], Tensor3[Height, Width, Latent, Float]) =
    val latent = encoder(v)
    val reconstructed = decoder(latent)
    (reconstructed, latent)

  def loss(original: Tensor3[Height, Width, Channel, Float]): Tensor0[Float] =
    val (reconstructed, _) = apply(original)
    val eps = 1e-5f
    val reconstructionLossPerPixel = -((original * (reconstructed +! eps).log) + ((Tensor0(1f) -! original) * (1f -! reconstructed +! eps).log))
    reconstructionLossPerPixel.mean

object AutoencoderCNN:
  case class Params(
      encoderParams: Encoder.EncoderParams,
      decoderParams: Decoder.DecoderParams
  )
  object Params:
    def apply(params: AutoencoderCNN.Params): Params =
      Params(
        params.encoderParams,
        params.decoderParams
      )

object AutoencoderCNNExample:

  def main(args: Array[String]): Unit =

    val learningRate = 1e-3f

    val numTestSamples = 9728
    val batchSize = 256
    val numSamples = 59904
    val numEpochs = 50

    val initKey = Random.Key(42)

    val (trainX, trainY) = MNISTLoader.createTrainingDataset(maxSamples = Some(numSamples)).get
    val (testX, testY) = MNISTLoader.createTestDataset(maxSamples = Some(numTestSamples)).get

    /*
     * Initialize the model parameters
     * */
    val initKeys = initKey.split(6)
    val encoderParams = Encoder.EncoderParams(
      Conv2DLayer.Params(initKeys(0))(Shape(
        Axis[Height] -> 3,
        Axis[Width] -> 3,
        Axis[Channel] -> 1,
        Axis[EHidden1] -> 8
      )),
      Conv2DLayer.Params(initKeys(1))(Shape(
        Axis[Height] -> 3,
        Axis[Width] -> 3,
        Axis[EHidden1] -> 8,
        Axis[EHidden2] -> 16
      )),
      Conv2DLayer.Params(initKeys(2))(Shape(
        Axis[Height] -> 3,
        Axis[Width] -> 3,
        Axis[EHidden2] -> 16,
        Axis[Latent] -> 8
      ))
    )
    val decoderParams = Decoder.DecoderParams(
      TransposeConv2DLayer.Params(initKeys(3))(Shape(
        Axis[Height] -> 3,
        Axis[Width] -> 3,
        Axis[DHidden1] -> 16,
        Axis[Latent] -> 8
      )),
      TransposeConv2DLayer.Params(initKeys(4))(Shape(
        Axis[Height] -> 3,
        Axis[Width] -> 3,
        Axis[DHidden2] -> 8,
        Axis[DHidden1] -> 16
      )),
      TransposeConv2DLayer.Params(initKeys(5))(Shape(
        Axis[Height] -> 3,
        Axis[Width] -> 3,
        Axis[ReconstructedChannel] -> 1,
        Axis[DHidden2] -> 8
      ))
    )

    val initialParams = AutoencoderCNN.Params(encoderParams, decoderParams)
    val scaledInitialParams = FloatTensorTree[AutoencoderCNN.Params].map(
      initialParams,
      [T <: Tuple] => (n: Labels[T]) ?=> (t: Tensor[T, Float]) => t *! Tensor0(0.1f)
    )

    /*
     * Training loop
     * */

    def loss(trainData: Tensor3[Sample, Height, Width, Float])(params: AutoencoderCNN.Params): Tensor0[Float] =
      val ae = AutoencoderCNN(params)
      trainData
        .vmap(Axis[Sample])(sample => ae.loss(sample.appendAxis(Axis[Channel])))
        .mean

    val batches = trainX.chunk(Axis[TrainSample], numSamples / batchSize)

    // val optimizer = GradientDescent(learningRate = Tensor0(learningRate))
    val optimizer = Lion(learningRate = Tensor0(learningRate))
    type OptState = optimizer.State[AutoencoderCNN.Params]

    def gradientStep(batch: Tensor3[Sample, Height, Width, Float], params: AutoencoderCNN.Params, state: OptState): (AutoencoderCNN.Params, OptState) =
      val grads = Autodiff.grad(loss(batch))(params)
      val (newParams, newState) = optimizer.update(grads, params, state)
      (newParams, newState)

    val jittedGradientStep = jit(gradientStep)

    def trainEpoch(params: AutoencoderCNN.Params, state: OptState): (AutoencoderCNN.Params, OptState) =
      batches.foldLeft((params, state)):
        case ((currentParams, currentState), batch) =>
          jittedGradientStep(batch, currentParams, currentState)

    // run the loop
    val trainTrajectory = Iterator.iterate((scaledInitialParams, optimizer.init(scaledInitialParams))): (currentParams, currentState) =>
      timed("Training"):
        dimwit.gc()
        trainEpoch(currentParams, currentState)

    val trainedParams = trainTrajectory.zipWithIndex
      .map:
        case ((params, state), epoch) => (params, epoch)
      .tapEach:
        case (params, epoch) =>
          timed("Evaluation"):
            val lossValue = loss(testX)(params)
            println(s"Epoch $epoch | Test loss: $lossValue")
      .map((params, _) => params)
      .drop(numEpochs)
      .next()

    /*
     * Evaluation
     * */
    val ae = AutoencoderCNN(trainedParams)

    val reconstructed = testX
      .slice(Axis[TestSample] -> (0 until 64))
      .vmap(Axis[TestSample]): sample =>
        val latent = ae.encoder(sample.appendAxis(Axis[Channel]))
        ae.decoder(latent)
      .split(Axis[TestSample], Axis[Prime[Height]] -> 8, Axis[Prime[Width]] -> 8)

    println(s"Reconstructed shape: ${reconstructed.shape}")
    val img2d = reconstructed.squeeze(Axis[Channel]).rearrange(
      (Axis[Prime[Height] |*| Height], Axis[Prime[Width] |*| Width]),
      (Axis[Prime[Height]] -> 8, Axis[Prime[Width]] -> 8, Axis[Height] -> 28, Axis[Width] -> 28)
    )
    import me.shadaj.scalapy.py
    val plt = py.module("matplotlib.pyplot")
    plt.imshow(img2d.jaxValue, cmap = "gray")
    plt.show()
