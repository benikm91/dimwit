package examples.basic.mnistcnn

import dimwit.*
import dimwit.Conversions.given
import nn.*
import nn.ActivationFunctions.relu
import dimwit.random.Random
import examples.timed
import examples.dataset.MNISTLoader

// Logits-based Cross Entropy (same as yours)
def binaryCrossEntropy[L: Label](
    logits: Tensor1[L, Float],
    label: Tensor0[Int]
): Tensor0[Float] =
  val maxLogit = logits.max
  val logSumExp = ((logits -! maxLogit).exp.sum + 1e-7f).log + maxLogit
  val targetLogit = logits.slice(Axis[L] -> label)
  logSumExp - targetLogit

object MNistCNN:
  import MNISTLoader.{Sample, TrainSample, Height, Width}

  // New labels for CNN architecture
  trait Channel derives Label
  trait ConvOut derives Label
  trait Hidden derives Label
  trait Output derives Label

  object CNN:
    case class Params(
        conv: Conv2DLayer.Params[Height, Width, Channel, ConvOut],
        fc: LinearLayer.Params[Height |*| Width |*| ConvOut, Output]
    )

    object Params:
      def apply(paramKey: Random.Key): Params =
        val (k1, k2) = paramKey.split2()
        Params(
          conv = Conv2DLayer.Params(k1)(Shape(
            Axis[Height] -> 3,
            Axis[Width] -> 3,
            Axis[Channel] -> 1,
            Axis[ConvOut] -> 16
          )),
          fc = LinearLayer.Params(k2)(
            Axis[Height |*| Width |*| ConvOut] -> 14 * 14 * 16,
            Axis[Output] -> 10
          )
        )

  case class CNN(params: CNN.Params) extends Function[Tensor2[Height, Width, Float], Tensor0[Int]]:
    private val conv = Conv2DLayer(params.conv, stride = 2, padding = Padding.SAME)
    private val fc = LinearLayer(params.fc)

    def logits(image: Tensor2[Height, Width, Float]): Tensor1[Output, Float] =
      // Input: (H, W) -> (H, W, 1)
      val input = image.appendAxis(Axis[Channel])
      // Conv -> ReLU -> Flatten -> Linear
      val features = conv(input)
      fc(features.ravel)

    override def apply(image: Tensor2[Height, Width, Float]): Tensor0[Int] =
      logits(image).argmax(Axis[Output])

  def main(args: Array[String]): Unit =
    // Hyperparameters
    val learningRate = 5e-5f // CNNs often benefit from slightly higher LR than MLPs
    val numSamples = 59904
    val batchSize = 128 // Lowered batch size slightly for GPU memory safety with CNN
    val numEpochs = 50

    val (dataKey, trainKey) = Random.Key(42).split2()
    val (trainX, trainY) = MNISTLoader.createTrainingDataset(maxSamples = Some(numSamples)).get
    val (testX, testY) = MNISTLoader.createTestDataset(maxSamples = Some(9728)).get

    // Initialize Params
    val initParams = CNN.Params(trainKey)

    def batchLoss(batchImages: Tensor[(TrainSample, Height, Width), Float], batchLabels: Tensor1[TrainSample, Int])(
        params: CNN.Params
    ): Tensor0[Float] =
      val model = CNN(params)
      zipvmap(Axis[TrainSample])(batchImages, batchLabels)((img, lbl) =>
        binaryCrossEntropy(model.logits(img), lbl)
      ).mean

    // Standard Training Utilities
    val optimizer = GradientDescent(learningRate = Tensor0(learningRate))

    def gradientStep(
        imageBatch: Tensor[(TrainSample, Height, Width), Float],
        labelBatch: Tensor1[TrainSample, Int],
        params: CNN.Params,
        state: optimizer.State[CNN.Params]
    ): (Tensor0[Float], CNN.Params, optimizer.State[CNN.Params]) =
      val (trainLoss, grads) = Autodiff.valueAndGrad(batchLoss(imageBatch, labelBatch))(params)
      val (newParams, newState) = optimizer.update(grads, params, state)
      (trainLoss, newParams, newState)

    val jitStep = gradientStep

    // Training Loop
    val trainTrajectory = Iterator.iterate((initParams, optimizer.init(initParams))): (params, state) =>
      timed("Training Epoch"):
        val imgBatches = trainX.chunk(Axis[TrainSample], numSamples / batchSize)
        val lblBatches = trainY.chunk(Axis[TrainSample], numSamples / batchSize)
        val (trainLoss, newParams, newState) = imgBatches.zip(lblBatches).foldLeft((Tensor0(0.0f), params, state)):
          case ((avgTrainLoss, p, s), (imgB, lblB)) =>
            val (trainLoss, newParams, newState) = jitStep(imgB, lblB, p, s)
            (0.9f * avgTrainLoss + 0.1f * trainLoss.item, newParams, newState)
        println(f"Average Training Loss: ${trainLoss.item}%.4f")
        val maxWeight = FloatTensorTree[Conv2DLayer.Params[Height, Width, Channel, ConvOut]].foldLeft(
          newParams.conv,
          0.0f
        )([T <: Tuple] =>
          (l: Labels[T]) ?=>
            (acc: Float, t: Tensor[T, Float]) => Math.max(acc, t.max.item)
        )
        println(s"Max weight value: $maxWeight")
        (newParams, newState)

    // Evaluation
    def evaluate(params: CNN.Params, dataX: Tensor[(Sample, Height, Width), Float], dataY: Tensor1[Sample, Int]): Tensor0[Float] =
      val model = CNN(params)
      val predictions = dataX.vmap(Axis[Sample])(model)
      val matches = zipvmap(Axis[Sample])(predictions, dataY)(_ === _)
      matches.asFloat.mean

    val jitEvaluate = jit(evaluate)

    trainTrajectory.zipWithIndex.foreach:
      case ((params, state), epoch) =>
        if epoch % 1 == 0 then
          dimwit.gc()
          val acc = jitEvaluate(params, testX, testY)
          println(f"Epoch $epoch | Test Accuracy: ${acc.item * 100}%.2f%%")
