package nn

import dimwit.*
import dimwit.random.Random.Key
import dimwit.stats.Normal

object Conv2DLayer:

  case class Params[S1, S2, InChannel, OutChannel](
      kernel: Tensor[S1 *: S2 *: InChannel *: OutChannel *: EmptyTuple, Float]
  )

  object Params:
    given [S1, S2, InChannel, OutChannel]: FloatTensorTree[Params[S1, S2, InChannel, OutChannel]] = FloatTensorTree.derived
    given [S1: Label, S2: Label, InChannel: Label, OutChannel: Label]: ToPyTree[Params[S1, S2, InChannel, OutChannel]] = ToPyTree.derived

    /** Initialize convolutional layer parameters
      *
      * @param paramKey
      *   Random key for parameter initialization
      * @param kernelShape
      *   Shape of the convolutional kernel, e.g., (KernelH, KernelW, InChannel, OutChannel) for 2D conv
      */
    def apply[S1: Label, S2: Label, InChannel: Label, OutChannel: Label](paramKey: Key)(
        kernelShape: Shape[S1 *: S2 *: InChannel *: OutChannel *: EmptyTuple]
    )(using
        executionType: ExecutionType[Float]
    ): Params[S1, S2, InChannel, OutChannel] =
      import Labels.ForConcat.given
      Params(
        kernel = Normal.standardNormal(kernelShape).sample(paramKey)
      )

case class Conv2DLayer[S1: Label, S2: Label, InChannel: Label, OutChannel: Label](
    params: Conv2DLayer.Params[S1, S2, InChannel, OutChannel],
    stride: Int = 1,
    padding: Padding = Padding.SAME
):
  /** Apply convolution to input tensor (like LinearLayer)
    *
    * Input: (Spatial..., InChannel) Output: (Spatial..., OutChannel)
    *
    * Note: The spatial dimensions of the input and kernel must match. For batched inputs, use vmap: inputBatched.vmap(Axis[Batch])(convLayer.apply)
    */
  def apply(x: Tensor[S1 *: S2 *: InChannel *: EmptyTuple, Float]): Tensor[S1 *: S2 *: OutChannel *: EmptyTuple, Float] =
    x.conv2d(params.kernel, stride, padding)
