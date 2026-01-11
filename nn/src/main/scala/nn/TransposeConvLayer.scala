package nn

import dimwit.*
import dimwit.random.Random
import dimwit.random.Random.Key
import dimwit.tensor.{VType, ExecutionType, Labels, Tensor, Shape, Axis, Dim}
import dimwit.tensor.TensorOps.Convolution.Padding
import dimwit.tensor.TupleHelpers.{Last, ReplaceLast}
import dimwit.stats.Normal

object TransposeConvLayer:

  case class Params[InChannels, OutChannels, KernelShape <: Tuple](
      kernel: Tensor[KernelShape, Float]
  )

  object Params:
    given [IC: Label, OC: Label, KS <: Tuple: Labels]: TensorTree[Params[IC, OC, KS]] = TensorTree.derived
    given [IC: Label, OC: Label, KS <: Tuple: Labels]: ToPyTree[Params[IC, OC, KS]] = ToPyTree.derived

    /** Initialize transpose convolutional layer parameters
      *
      * @param paramKey
      *   Random key for parameter initialization
      * @param kernelShape
      *   Shape of the convolutional kernel, e.g., (KernelH, KernelW, InChannels, OutChannels) for 2D transpose conv
      */
    def apply[InChannels: Label, OutChannels: Label, KernelShape <: Tuple: Labels](paramKey: Key)(
        kernelShape: Shape[KernelShape]
    )(using
        executionType: ExecutionType[Float]
    ): Params[InChannels, OutChannels, KernelShape] =
      Params(
        kernel = Normal.standardNormal(kernelShape).sample(paramKey)
      )

case class TransposeConvLayer[InChannels: Label, OutChannels: Label, KernelShape <: Tuple: Labels](
    params: TransposeConvLayer.Params[InChannels, OutChannels, KernelShape],
    stride: Int = 1,
    padding: Padding = Padding.SAME
):
  /** Apply transpose convolution to input tensor
    *
    * Note: For transpose convolution, the input has OutChannels (matching forward conv output) and the output has InChannels (matching forward conv input). This is the adjoint operation to forward convolution.
    *
    * Input: (Spatial..., OutChannels) Output: (Spatial..., InChannels)
    */
  def apply[InputShape <: Tuple: Labels](x: Tensor[InputShape, Float])(using
      inputChannelMatch: Last[InputShape] =:= OutChannels,
      kernelInMatch: dimwit.tensor.TupleHelpers.SecondToLast[KernelShape] =:= InChannels,
      kernelOutMatch: Last[KernelShape] =:= OutChannels,
      outputLabels: Labels[ReplaceLast[InputShape, InChannels]]
  ): Tensor[ReplaceLast[InputShape, InChannels], Float] =
    x.transposeConv(Axis[OutChannels], Axis[InChannels])(params.kernel, stride, padding)
