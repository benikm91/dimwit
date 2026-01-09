package examples.basic

import dimwit.*
import dimwit.Conversions.given
import nn.*
import nn.ActivationFunctions.{relu, sigmoid}
import dimwit.random.Random

import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.Try
import java.io.{FileInputStream, DataInputStream, BufferedInputStream}
import java.io.RandomAccessFile
import java.util.Base64
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

def binaryCrossEntropy[L: Label](
    logits: Tensor1[L, Float],
    label: Tensor0[Int]
): Tensor0[Float] =
  val maxLogit = logits.max
  val stableExp = (logits -! maxLogit).exp
  val logSumExp = stableExp.sum.log + maxLogit
  val targetLogit = logits.slice(Axis[L] -> label)
  -(targetLogit - logSumExp)

object MLPClassifierMNist:

  trait Sample derives Label
  trait TrainSample extends Sample derives Label
  trait TestSample extends Sample derives Label
  trait Height derives Label
  trait Width derives Label
  trait Hidden derives Label
  trait Output derives Label

  object MLP:
    case class Params(
        layer1: LinearLayer.Params[Height |*| Width, Hidden],
        layer2: LinearLayer.Params[Hidden, Output]
    )

    object Params:
      def apply(
          layer1Dim: Dim[Height |*| Width],
          layer2Dim: Dim[Hidden],
          outputDim: Dim[Output]
      )(
          paramKey: Random.Key
      ): Params =
        val (key1, key2) = paramKey.split2()
        Params(
          layer1 = LinearLayer.Params(key1)(layer1Dim, layer2Dim),
          layer2 = LinearLayer.Params(key2)(layer2Dim, outputDim)
        )

  case class MLP(params: MLP.Params) extends Function[Tensor2[Height, Width, Float], Tensor0[Int]]:

    private val layer1 = LinearLayer(params.layer1)
    private val layer2 = LinearLayer(params.layer2)

    def logits(
        image: Tensor2[Height, Width, Float]
    ): Tensor1[Output, Float] =
      val hidden = relu(layer1(image.ravel))
      layer2(hidden)

    override def apply(image: Tensor2[Height, Width, Float]): Tensor0[Int] = logits(image).argmax(Axis[Output])

  object MNISTLoader:

    private val pythonLoader = py.eval("lambda b64, shape: __import__('jax').numpy.array(__import__('numpy').frombuffer(__import__('base64').b64decode(b64), dtype=__import__('numpy').uint8).reshape(shape).astype(__import__('numpy').int32))")

    def loadImages[S <: Sample: Label](filename: String, maxImages: Option[Int] = None): Tensor3[S, Height, Width, Int] =
      val file = new RandomAccessFile(filename, "r")
      try
        val magic = file.readInt()
        if magic != 2051 then throw new IllegalArgumentException(s"Invalid magic: $magic")

        val totalImages = file.readInt()
        val rows = file.readInt()
        val cols = file.readInt()

        val numImages = maxImages.map(math.min(_, totalImages)).getOrElse(totalImages)
        val totalPixels = numImages * rows * cols

        println(s"Scala-Loading $numImages images (${rows}x${cols}) from $filename...")

        val pixels = new Array[Byte](totalPixels)
        file.readFully(pixels)

        val shape = Shape(Axis[S] -> numImages, Axis[Height] -> rows, Axis[Width] -> cols)
        Tensor.fromArray(shape)(pixels)

      finally
        file.close()

    def loadLabels[S <: Sample: Label](filename: String, maxLabels: Option[Int] = None): Tensor1[S, Int] =
      val file = new RandomAccessFile(filename, "r")
      try
        val magic = file.readInt()
        if magic != 2049 then throw new IllegalArgumentException(s"Invalid magic for labels: $magic (expected 2049)")

        val totalLabels = file.readInt()
        val numLabels = maxLabels.map(math.min(_, totalLabels)).getOrElse(totalLabels)

        println(s"JAX-Loading $numLabels labels from $filename...")

        val labels = new Array[Byte](numLabels)
        file.readFully(labels)

        val shape = Shape(Axis[S] -> numLabels)
        Tensor.fromArray(shape)(labels)

      finally
        file.close()

    private def createDataset[S <: Sample: Label](imagesFile: String, labelsFile: String, maxSamples: Option[Int] = None): Try[Tuple2[Tensor[(S, Height, Width), Float], Tensor1[S, Int]]] =
      Try:
        val images = loadImages[S](imagesFile, maxSamples)
        val labels = loadLabels[S](labelsFile, maxSamples)
        require(images.shape(Axis[S]) == labels.shape(Axis[S]), s"Number of images and labels must match")
        val imagesFloat = images.asFloat /! 255.0f
        (imagesFloat, labels)

    def createTrainingDataset(dataDir: String = "data", maxSamples: Option[Int] = None): Try[Tuple2[Tensor[(TrainSample, Height, Width), Float], Tensor1[TrainSample, Int]]] =
      val imagesFile = s"$dataDir/train-images-idx3-ubyte"
      val labelsFile = s"$dataDir/train-labels-idx1-ubyte"
      createDataset[TrainSample](imagesFile, labelsFile, maxSamples)

    def createTestDataset(dataDir: String = "data", maxSamples: Option[Int] = None): Try[Tuple2[Tensor[(TestSample, Height, Width), Float], Tensor1[TestSample, Int]]] =
      val imagesFile = s"$dataDir/t10k-images-idx3-ubyte"
      val labelsFile = s"$dataDir/t10k-labels-idx1-ubyte"
      createDataset[TestSample](imagesFile, labelsFile, maxSamples)

  def main(args: Array[String]): Unit =

    val learningRate = 5e-2f
    val numSamples = 59904
    val numTestSamples = 9728
    val batchSize = 512
    val numEpochs = 100
    val (dataKey, trainKey) = Random.Key(42).split2()
    val (initKey, restKey) = trainKey.split2()

    val (trainX, trainY) = MNISTLoader.createTrainingDataset(maxSamples = Some(numSamples)).get
    val (testX, testY) = MNISTLoader.createTestDataset(maxSamples = Some(numTestSamples)).get

    def batchLoss(batchImages: Tensor[(TrainSample, Height, Width), Float], batchLabels: Tensor1[TrainSample, Int])(
        params: MLP.Params
    ): Tensor0[Float] =
      val model = MLP(params)
      val batchSize = batchImages.shape(Axis[TrainSample])
      val losses = (0 until batchSize)
        .map: idx =>
          val image = batchImages.slice(Axis[TrainSample] -> idx)
          val label = batchLabels.slice(Axis[TrainSample] -> idx)
          val logits = model.logits(image)
          binaryCrossEntropy(logits, label)
        .reduce(_ + _)
      losses / batchSize.toFloat

    val initParams = MLP.Params(
      Axis[Height |*| Width] -> 28 * 28,
      Axis[Hidden] -> 128,
      Axis[Output] -> 10
    )(initKey)

    def accuracy[Sample: Label](
        predictions: Tensor1[Sample, Int],
        targets: Tensor1[Sample, Int]
    ): Tensor0[Float] =
      val matches = zipvmap(Axis[Sample])(predictions, targets)(_ === _)
      matches.asFloat.mean

    def gradientStep(
        imageBatch: Tensor[(TrainSample, Height, Width), Float],
        labelBatch: Tensor1[TrainSample, Int],
        params: MLP.Params
    ): MLP.Params =
      val lossBatch = batchLoss(imageBatch, labelBatch)
      val df = Autodiff.grad(lossBatch)
      GradientDescent(df, learningRate).step(params)
    val jitStep = jit(gradientStep)
    def miniBatchGradientDescent(
        imageBatches: Seq[Tensor[(TrainSample, Height, Width), Float]],
        labelBatches: Seq[Tensor1[TrainSample, Int]]
    )(
        params: MLP.Params
    ): MLP.Params =
      imageBatches
        .zip(labelBatches)
        .foldLeft(params):
          case (currentParams, (imageBatch, labelBatch)) =>
            jitStep(imageBatch, labelBatch, currentParams)

    def timed[A](template: String)(block: => A): A =
      val t0 = System.currentTimeMillis()
      val result = block
      println(s"$template took ${System.currentTimeMillis() - t0} ms")
      result

    val trainMiniBatchGradientDescent = miniBatchGradientDescent(
      trainX.chunk(Axis[TrainSample], batchSize),
      trainY.chunk(Axis[TrainSample], batchSize)
    )
    val trainTrajectory = Iterator.iterate(initParams)(currentParams =>
      timed("Training"):
        trainMiniBatchGradientDescent(currentParams)
    )
    def evaluate(
        params: MLP.Params,
        dataX: Tensor[(Sample, Height, Width), Float],
        dataY: Tensor1[Sample, Int]
    ): Tensor0[Float] =
      val model = MLP(params)
      val predictions = dataX.vmap(Axis[Sample])(model)
      accuracy(predictions, dataY)
    val jitEvaluate = jit(evaluate)
    val finalParams = trainTrajectory.zipWithIndex
      .tapEach:
        case (params, epoch) =>
          timed("Evaluation"):
            val testAccuracy = jitEvaluate(params, testX, testY)
            val trainAccuracy = jitEvaluate(params, trainX, trainY)
            println(
              List(
                s"Epoch $epoch",
                f"Test accuracy: ${testAccuracy.item * 100}%.2f%%",
                f"Train accuracy: ${trainAccuracy.item * 100}%.2f%%"
              ).mkString(", ")
            )
      .map((params, _) => params)
      .drop(numEpochs)
      .next()

    println("\nTraining complete!")
