package examples.complex.nanoGPT

import dimwit.*
import dimwit.Conversions.given

import nn.ActivationFunctions.*
import dimwit.random.Random

// Dimensions
trait Vocab derives Label // 50257
trait Embedding derives Label // 768
trait Context derives Label // 1024
trait EmbeddingMixed derives Label // 3072

trait Batch derives Label

case class LayerNormalizationParams(
    weight: Tensor1[Embedding, Float],
    bias: Tensor1[Embedding, Float]
)

case class LinearLayerParams[In, Out](
    weight: Tensor2[In, Out, Float],
    bias: Tensor1[Out, Float]
)

case class ProjectionLayerParams[In, Out](
    weight: Tensor2[In, Out, Float]
)

trait Head derives Label
trait HeadKey derives Label
trait HeadQuery derives Label
trait HeadValue derives Label

case class HeadsParams[Kind](val weights: Tensor3[Head, Embedding, Kind, Float], val bias: Tensor2[Head, Kind, Float])

case class MultiHeadAttentionParams(
    wq: HeadsParams[HeadQuery],
    wk: HeadsParams[HeadKey],
    wv: HeadsParams[HeadValue],
    proj: LinearLayerParams[Head |*| HeadValue, Embedding]
) derives ToPyTree

case class EmbeddingMixerParams(
    c_fc: LinearLayerParams[Embedding, EmbeddingMixed],
    c_proj: LinearLayerParams[EmbeddingMixed, Embedding]
)

case class TransformerLayerParams(
    ln1: LayerNormalizationParams,
    attn: MultiHeadAttentionParams,
    ln2: LayerNormalizationParams,
    embeddingMixer: EmbeddingMixerParams
)

case class GPT2Params(
    vocabularyEmbeddings: Tensor2[Vocab, Embedding, Float],
    positionalEmbeddings: Tensor2[Context, Embedding, Float],
    layers: List[TransformerLayerParams],
    outputNormalization: LayerNormalizationParams,
    output: ProjectionLayerParams[Embedding, Vocab]
)

object GPT2Params:
  def apply(
      vocabularyEmbeddings: Tensor2[Vocab, Embedding, Float],
      positionalEmbeddings: Tensor2[Context, Embedding, Float],
      layers: List[TransformerLayerParams],
      outputNormalization: LayerNormalizationParams
  ): GPT2Params =
    val outputParams = ProjectionLayerParams(
      vocabularyEmbeddings.transpose // Tying output weights with input embeddings
    )
    GPT2Params(vocabularyEmbeddings, positionalEmbeddings, layers, outputNormalization, outputParams)

case class GPT2(params: GPT2Params) extends (Tensor2[Batch, Context, Int] => Tensor2[Batch, Context, Int]):

  private case class LinearLayer[In: Label, Out: Label](params: LinearLayerParams[In, Out]) extends (Tensor1[In, Float] => Tensor1[Out, Float]):
    override def apply(x: Tensor1[In, Float]): Tensor1[Out, Float] =
      x.dot(Axis[In])(params.weight) + params.bias

  private case class EmbeddingMixer(params: EmbeddingMixerParams) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):
    private val hiddenLayer = LinearLayer(params.c_fc)
    private val outputLayer = LinearLayer(params.c_proj)
    // TODO add dropout

    def apply(in: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
      in.vmap(Axis[Context])(x =>
        val hidden = gelu(hiddenLayer(x))
        outputLayer(hidden)
      )

  private case class ProjectionLayer[In: Label, Out: Label](params: ProjectionLayerParams[In, Out]) extends (Tensor1[In, Float] => Tensor1[Out, Float]):
    def apply(x: Tensor1[In, Float]): Tensor1[Out, Float] =
      x.dot(Axis[In])(params.weight)

  private case class MultiHeadAttention(params: MultiHeadAttentionParams) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):

    private val projection = LinearLayer(params.proj)

    def apply(x: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
      val heads = zipvmap(Axis[Head])(params.wq.weights, params.wq.bias, params.wk.weights, params.wk.bias, params.wv.weights, params.wv.bias):
        attention.tupled(_)(x)
      heads.vmap(Axis[Context])(heads => projection(heads.ravel))

    private def attention(
        wq: Tensor2[Embedding, HeadQuery, Float],
        wqBias: Tensor1[HeadQuery, Float],
        wk: Tensor2[Embedding, HeadKey, Float],
        wkBias: Tensor1[HeadKey, Float],
        wv: Tensor2[Embedding, HeadValue, Float],
        wvBias: Tensor1[HeadValue, Float]
    )(x: Tensor2[Context, Embedding, Float]): Tensor2[Context, HeadValue, Float] =

      trait AttnWeights derives Label

      def causalMasking(attnScores: Tensor2[Context, Prime[Context], Float]): Tensor2[Context, Prime[Context], Float] =
        val ctxLength = attnScores.shape(Axis[Context])
        val causalMask = tril(Tensor(Shape((Axis[Context] -> ctxLength, Axis[Prime[Context]] -> ctxLength))).fill(true))
        where(causalMask, attnScores, Tensor.like(attnScores).fill(Float.NegativeInfinity))

      val queries = x.dot(Axis[Embedding])(wq) +! wqBias
      val keys = x.dot(Axis[Embedding])(wk) +! wkBias
      val values = x.dot(Axis[Embedding])(wv) +! wvBias
      val dk = Tensor0(Math.sqrt(keys.shape(Axis[HeadKey])).toFloat)
      val attnScores = (queries.dot(Axis[HeadQuery ~ HeadKey])(keys) /! dk)
      val attnWeights = causalMasking(attnScores)
        .vmap(Axis[Context])(attnScore => softmax(attnScore).relabelTo(Axis[AttnWeights]))
      val res = attnWeights.dot(Axis[AttnWeights ~ Context])(values)
      res

  private case class LayerNorm(params: LayerNormalizationParams) extends (Tensor1[Embedding, Float] => Tensor1[Embedding, Float]):

    private def standardize(x: Tensor1[Embedding, Float]): Tensor1[Embedding, Float] =
      val x0 = x -! x.mean
      val variance = x0.pow(2).mean
      val epsilon = 1e-6f
      x0 /! (variance + epsilon).sqrt

    def apply(x: Tensor1[Embedding, Float]): Tensor1[Embedding, Float] =
      standardize(x) * params.weight + params.bias

  private case class TransformerLayer(params: TransformerLayerParams) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):
    private val embeddingMixer = EmbeddingMixer(params.embeddingMixer)
    private val multiHeadAttention = MultiHeadAttention(params.attn)
    private val preNormalization = LayerNorm(params.ln1)
    private val postNormalization = LayerNorm(params.ln2)

    def apply(t: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
      var x = t
      x = x + multiHeadAttention(x.vmap(Axis[Context])(preNormalization))
      x = x + embeddingMixer(x.vmap(Axis[Context])(postNormalization))
      x

  private case class TransformerBlock(layers: List[TransformerLayer]) extends (Tensor2[Context, Embedding, Float] => Tensor2[Context, Embedding, Float]):
    override def apply(t: Tensor2[Context, Embedding, Float]): Tensor2[Context, Embedding, Float] =
      layers.foldLeft(t):
        case (t, layer) => layer(t)

  case class Embedder(vocabularyEmbeddings: Tensor2[Vocab, Embedding, Float], positionalEmbeddings: Tensor2[Context, Embedding, Float]):

    def apply(tokens: Tensor1[Context, Int]): Tensor2[Context, Embedding, Float] =
      val embeddings = vocabularyEmbeddings.take(Axis[Vocab])(tokens)
      embeddings + positionalEmbeddings

  case class OutputLayer(normalization: LayerNormalizationParams, projectionParams: ProjectionLayerParams[Embedding, Vocab]) extends (Tensor1[Embedding, Float] => Tensor1[Vocab, Float]):
    private val normalizationLayer = LayerNorm(normalization)
    private val projection = ProjectionLayer(projectionParams)
    override def apply(x: Tensor1[Embedding, Float]): Tensor1[Vocab, Float] =
      projection(normalizationLayer(x))

  private val embedder = Embedder(params.vocabularyEmbeddings, params.positionalEmbeddings)
  private val transformerBlock = TransformerBlock(params.layers.map(TransformerLayer(_)))
  private val outputLayer = OutputLayer(params.outputNormalization, params.output)

  def logits(inputTokens: Tensor2[Batch, Context, Int]): Tensor3[Batch, Context, Vocab, Float] =
    inputTokens.vmap(Axis[Batch]):
      case tokens =>
        val startEmbeddings = embedder(tokens)
        val endEmbeddings = transformerBlock(startEmbeddings)
        endEmbeddings.vmap(Axis[Context])(x => outputLayer(x))

  def probits(inputTokens: Tensor2[Batch, Context, Int]): Tensor3[Batch, Context, Vocab, Float] =
    val x = logits(inputTokens)
    val res = x.vapply(Axis[Vocab])(softmax)
    return res

  def apply(inputTokens: Tensor2[Batch, Context, Int]): Tensor2[Batch, Context, Int] =
    val x = probits(inputTokens)
    val res = x.argmax(Axis[Vocab])
    return res

object GPT2Train:

  import java.io.RandomAccessFile
  import java.nio.channels.FileChannel
  import java.nio.{ByteBuffer, ByteOrder}
  import java.nio.charset.StandardCharsets
  import dimwit.jax.Jax
  import dimwit.tensor.DType
  import me.shadaj.scalapy.py
  import me.shadaj.scalapy.py.SeqConverters

  case class TensorInfo(dtype: String, shape: List[Int], start: Long, end: Long)

  def main(args: Array[String]): Unit =
    trait Data derives Label

    val batchSize = 12
    val blockSize = 1024

    val np = py.module("numpy")
    case class Sample(
        input: Tensor2[Batch, Context, Int],
        target: Tensor2[Batch, Context, Int]
    )
    def createDataset(key: Random.Key, pathToBinaryFile: String): Iterator[Sample] =
      val data = Tensor1.fromPy(Axis[Data], VType[Int])(Jax.jnp.asarray(np.memmap(pathToBinaryFile, dtype = np.uint16, mode = "r")))
      def sliceContextBlockAt(idx: Tensor0[Int]): Tensor1[Context, Int] =
        data
          .dynamicSlice(idx, blockSize)
          .relabelTo(Axis[Context])
      val numDataPoints = data.shape(Axis[Data])
      Iterator.unfold(key):
        case key =>
          val lastValidIdx = numDataPoints - blockSize
          val batchExtent = Axis[Batch] -> batchSize
          val randomBatchIndex = Random.randint(batchExtent, min = 0, max = lastValidIdx)(key)
          val x = randomBatchIndex.vmap(Axis[Batch])(index => sliceContextBlockAt(index))
          val y = randomBatchIndex.vmap(Axis[Batch])(index => sliceContextBlockAt(index + 1))
          Some(Sample(x, y), key.next)

    def createTraintDataset(key: Random.Key): Iterator[Sample] =
      val pathToTrainBinaryFile = "data/nanoGPT/train.bin"
      createDataset(key, pathToTrainBinaryFile)
    def createValDataset(key: Random.Key): Iterator[Sample] =
      val pathToValBinaryFile = "data/nanoGPT/val.bin"
      createDataset(key, pathToValBinaryFile)

    println("Training samples:")
    val trainDataKey = Random.Key(42)
    for sample <- createTraintDataset(trainDataKey).take(5) do
      println(sample.input.shape)
      println(sample.target.shape)

    println("Validation samples:")
    val valDataKey = Random.Key(42)
    for sample <- createValDataset(valDataKey).take(5) do
      println(sample.input.shape)
      println(sample.target.shape)
