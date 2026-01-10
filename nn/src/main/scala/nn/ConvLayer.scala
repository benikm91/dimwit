package nn

import dimwit.*
import dimwit.random.Random
import dimwit.random.Random.Key
import dimwit.tensor.{VType, ExecutionType, Labels, Tensor, Shape, Axis, Dim}
import dimwit.tensor.TensorOps.Convolution.Padding
import dimwit.tensor.TupleHelpers.{Last, ReplaceLast}
import dimwit.stats.Normal

object ConvLayer:

  case class Params[InChannels, OutChannels, KernelShape <: Tuple](
      kernel: Tensor[KernelShape, Float]
  )

  object Params:
    given [IC: Label, OC: Label, KS <: Tuple: Labels]: TensorTree[Params[IC, OC, KS]] = TensorTree.derived
    given [IC: Label, OC: Label, KS <: Tuple: Labels]: ToPyTree[Params[IC, OC, KS]] = ToPyTree.derived

    /** Initialize convolutional layer parameters
      *
      * @param paramKey
      *   Random key for parameter initialization
      * @param kernelShape
      *   Shape of the convolutional kernel, e.g., (KernelH, KernelW, InChannels, OutChannels) for 2D conv
      */
    def apply[InChannels: Label, OutChannels: Label, KernelShape <: Tuple: Labels](paramKey: Key)(
        kernelShape: Shape[KernelShape]
    )(using
        executionType: ExecutionType[Float]
    ): Params[InChannels, OutChannels, KernelShape] =
      Params(
        kernel = Normal.standardNormal(kernelShape).sample(paramKey)
      )

case class ConvLayer[InChannels: Label, OutChannels: Label, KernelShape <: Tuple: Labels](
    params: ConvLayer.Params[InChannels, OutChannels, KernelShape],
    stride: Int = 1,
    padding: Padding = Padding.SAME
):
  /** Apply convolution to input tensor (like LinearLayer)
    *
    * Input: (Spatial..., InChannels) Output: (Spatial..., OutChannels)
    */
  def apply[InputShape <: Tuple: Labels](x: Tensor[InputShape, Float])(using
      inputChannelMatch: Last[InputShape] =:= InChannels,
      kernelInMatch: dimwit.tensor.TupleHelpers.SecondToLast[KernelShape] =:= InChannels,
      kernelOutMatch: Last[KernelShape] =:= OutChannels,
      outputLabels: Labels[ReplaceLast[InputShape, OutChannels]]
  ): Tensor[ReplaceLast[InputShape, OutChannels], Float] =
    x.conv(Axis[InChannels], Axis[OutChannels])(params.kernel, stride, padding)
