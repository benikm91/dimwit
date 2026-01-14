package examples.basic.ae

import dimwit.*
import dimwit.Conversions.given

import examples.basic.timed
import dimwit.stats.Normal
import dimwit.random.Random
import nn.LinearLayer
import nn.ActivationFunctions.relu
import nn.GradientDescent
import dimwit.jax.Jax
import nn.ActivationFunctions.sigmoid
import dimwit.random.Random.Key

import examples.basic.MNISTLoader
import MNISTLoader.{Sample, TrainSample, TestSample, Height, Width}
trait Hidden derives Label
trait Output derives Label

type SourceFeature = Height |*| Width
type ReconstructedFeature = Height |*| Width

trait EHidden1 derives Label
trait EHidden2 derives Label

trait Latent derives Label

trait DHidden1 derives Label
trait DHidden2 derives Label

trait Batch derives Label

type FTensor1[T] = Tensor1[T, Float]

class Encoder(p: Encoder.EncoderParams):
  def apply(v: FTensor1[Height |*| Width]): FTensor1[Latent] =
    val layer1 = LinearLayer(p.layer1)
    val layer2 = LinearLayer(p.layer2)
    val latentLayer = LinearLayer(p.latentLayer)

    val h1 = relu(layer1(v))
    val h2 = relu(layer2(h1))
    val latent = latentLayer(h2)
    latent

object Encoder:
  case class EncoderParams(
      layer1: LinearLayer.Params[Height |*| Width, EHidden1],
      layer2: LinearLayer.Params[EHidden1, EHidden2],
      latentLayer: LinearLayer.Params[EHidden2, Latent]
  )

class Decoder(p: Decoder.DecoderParams):
  def apply(v: FTensor1[Latent]): FTensor1[ReconstructedFeature] =
    val layer1 = LinearLayer(p.layer1)
    val layer2 = LinearLayer(p.layer2)
    val outputLayer = LinearLayer(p.outputLayer)

    val h1 = relu(layer1(v))
    val h2 = relu(layer2(h1))
    val reconstructed = sigmoid(outputLayer(h2))

    reconstructed

object Decoder:
  case class DecoderParams(
      layer1: LinearLayer.Params[Latent, DHidden1],
      layer2: LinearLayer.Params[DHidden1, DHidden2],
      outputLayer: LinearLayer.Params[DHidden2, ReconstructedFeature]
  )

case class AE(params: AE.Params):

  val encoder = Encoder(params.encoderParams)
  val decoder = Decoder(params.decoderParams)

  def apply(v: FTensor1[SourceFeature]): (FTensor1[ReconstructedFeature], FTensor1[Latent]) =
    val latent = encoder(v)
    val reconstructed = decoder(latent)
    (reconstructed, latent)

  def loss(original: FTensor1[Height |*| Width]): Tensor0[Float] =
    val (reconstructed, _) = apply(original)
    val eps = 1e-5f
    val reconLoss = -((original * (reconstructed +! eps).log) + ((Tensor0(1f) -! original) * (1f -! reconstructed +! eps).log)).sum
    reconLoss

object AE:
  case class Params(
      encoderParams: Encoder.EncoderParams,
      decoderParams: Decoder.DecoderParams
  )
  object Params:
    def apply(params: AE.Params): Params =
      Params(
        params.encoderParams,
        params.decoderParams
      )

object AEExample:

  def main(args: Array[String]): Unit =

    val learningRate = 5e-4f

    val numTestSamples = 9728
    val batchSize = 512
    val numSamples = 59904
    val numEpochs = 50
    val latentDim = 20

    val initKey = Random.Key(42)

    val (trainX, trainY) = MNISTLoader.createTrainingDataset(maxSamples = Some(numSamples)).get
    val (testX, testY) = MNISTLoader.createTestDataset(maxSamples = Some(numTestSamples)).get

    /*
     * Initialize the model parameters
     * */
    val initKeys = initKey.split(6)
    val encoderParams = Encoder.EncoderParams(
      LinearLayer.Params[Height |*| Width, EHidden1](initKeys(0))(
        Axis[Height |*| Width] -> (28 * 28),
        Axis[EHidden1] -> 512
      ),
      LinearLayer.Params[EHidden1, EHidden2](initKeys(1))(
        Axis[EHidden1] -> 512,
        Axis[EHidden2] -> 256
      ),
      LinearLayer.Params[EHidden2, Latent](initKeys(2))(
        Axis[EHidden2] -> 256,
        Axis[Latent] -> latentDim
      )
    )
    val decoderParams = Decoder.DecoderParams(
      LinearLayer.Params[Latent, DHidden1](initKeys(3))(
        Axis[Latent] -> 20,
        Axis[DHidden1] -> 256
      ),
      LinearLayer.Params[DHidden1, DHidden2](initKeys(4))(
        Axis[DHidden1] -> 256,
        Axis[DHidden2] -> 512
      ),
      LinearLayer.Params[DHidden2, ReconstructedFeature](initKeys(5))(
        Axis[DHidden2] -> 512,
        Axis[ReconstructedFeature] -> (28 * 28)
      )
    )

    // we need to scale down the initial parameters for
    // better training stability.
    // TODO linear layer et al. should support custom initializers
    // or xavier initialization
    val initialParams = AE.Params(encoderParams, decoderParams)
    val scaledInitialParams = FloatTensorTree[AE.Params].map(
      initialParams,
      [T <: Tuple] => (n: Labels[T]) ?=> (t: Tensor[T, Float]) => t *! Tensor0(0.1f)
    )

    /*
     * Training loop
     * */

    def loss(trainData: Tensor3[Sample, Height, Width, Float])(params: AE.Params): Tensor0[Float] =
      val ae = AE(params)
      trainData
        .vmap(Axis[Sample])(sample => ae.loss(sample.ravel))
        .mean

    val batches = trainX.chunk(Axis[TrainSample], numSamples / batchSize)
    val optimizer = GradientDescent(learningRate)

    def gradientStep(batch: Tensor3[Sample, Height, Width, Float], params: AE.Params): AE.Params =
      val grads = Autodiff.grad(loss(batch))(params)
      optimizer.update(grads, (), params)._2

    // val jittedGradientStep = gradientStep
    val jittedGradientStep = jit(gradientStep, Map("donate_argnums" -> Tuple1(1)))

    def trainEpoch(params: AE.Params): AE.Params =
      batches.foldLeft(params):
        case (batchParams, batch) =>
          jittedGradientStep(batch, batchParams)

    // run the loop
    val trainTrajectory = Iterator.iterate(scaledInitialParams)(currentParams =>
      timed("Training"):
        dimwit.gc()
        trainEpoch(currentParams)
    )
    val trainedParams = trainTrajectory.zipWithIndex
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
    val ae = AE(trainedParams)

    val reconstructed = testX.vmap(Axis[TestSample]): sample =>
      val latent = ae.encoder(sample.ravel)
      ae.decoder(latent)

    reconstructed.chunk(Axis[TestSample], 1).take(6).foreach: img =>
      val img2d = img.slice(Axis[TestSample] -> 0).rearrange(
        (Axis[Height], Axis[Width]),
        (Axis[Height] -> 28, Axis[Width] -> 28)
      )
      import me.shadaj.scalapy.py
      val matplotlib = py.module("matplotlib")
      matplotlib.use("WebAgg")
      val plt = py.module("matplotlib.pyplot")
      plt.imshow(img2d.jaxValue, cmap = "gray")
      plt.show()
