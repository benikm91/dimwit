package dimwit.tensor.tensorops

import dimwit.python.PyIndex.itemAt
import dimwit.jax.Einops
import dimwit.jax.Jax
import dimwit.tensor.Axis
import dimwit.tensor.AxisAtIndex
import dimwit.tensor.AxisAtIndices
import dimwit.tensor.AxisAtRange
import dimwit.tensor.AxisAtTensorIndex
import dimwit.tensor.AxisAtTupleIndices
import dimwit.tensor.AxisExtent
import dimwit.tensor.DType.Bool
import dimwit.tensor.DType.Int32
import dimwit.tensor.Label
import dimwit.tensor.Labels
import dimwit.tensor.Shape
import dimwit.tensor.ShapeTypeHelpers.AxesMerger
import dimwit.tensor.ShapeTypeHelpers.AxisIndex
import dimwit.tensor.ShapeTypeHelpers.AxisIndices
import dimwit.tensor.ShapeTypeHelpers.DimExtractor
import dimwit.tensor.ShapeTypeHelpers.MergeAxes
import dimwit.tensor.ShapeTypeHelpers.MergeLabels
import dimwit.tensor.ShapeTypeHelpers.UnwrapAxes
import dimwit.tensor.ShapeTypeHelpers.UnwrapDims
import dimwit.tensor.LabelsImpl
import dimwit.tensor.Tensor
import dimwit.tensor.Tensor0
import dimwit.tensor.Tensor1
import dimwit.tensor.TupleHelpers.CheckValid
import dimwit.tensor.TupleHelpers.ComputeMissing
import dimwit.tensor.TupleHelpers.IsPermutation
import dimwit.tensor.TupleHelpers.Remove
import dimwit.tensor.TupleHelpers.RemoveAll
import dimwit.tensor.TupleHelpers.Replace
import dimwit.tensor.TupleHelpers.ReplaceBy
import dimwit.tensor.TupleHelpers.StrictSubset
import dimwit.|+|
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.readwrite.Reader
import me.shadaj.scalapy.readwrite.Writer

import scala.annotation.implicitNotFound
import scala.compiletime.ops
import scala.util.NotGiven

object StructuralOps:

  private object Util:

    type InsertBefore[T <: Tuple, A, B] <: Tuple = T match
      case EmptyTuple => B *: EmptyTuple
      case A *: tail  => B *: A *: tail
      case h *: tail  => h *: InsertBefore[tail, A, B]

    type InsertAfter[T <: Tuple, A, B] <: Tuple = T match
      case EmptyTuple => B *: EmptyTuple
      case A *: tail  => A *: B *: tail
      case h *: tail  => h *: InsertAfter[tail, A, B]

    /** The axis a single slice input refers to. */
    type ExtractLabel[X] = X match
      case AxisAtIndex[l]           => l
      case AxisAtRange[l]           => l
      case AxisAtIndices[l]         => l
      case AxisAtTupleIndices[l, ?] => l
      case AxisAtTensorIndex[l]     => l

    /** The axes that all slice inputs refer to. */
    type ExtractLabels[Inputs <: Tuple] = Tuple.Map[Inputs, ExtractLabel]

    /** The axes that slicing drops, i.e. those selected by a single index rather than by several. */
    type SliceLabels[Inputs <: Tuple] <: Tuple = Inputs match
      case EmptyTuple                    => EmptyTuple
      case AxisAtIndex[l] *: t           => l *: SliceLabels[t]
      case AxisAtTensorIndex[l] *: t     => l *: SliceLabels[t]
      case AxisAtRange[l] *: t           => SliceLabels[t]
      case AxisAtIndices[l] *: t         => SliceLabels[t]
      case AxisAtTupleIndices[l, ?] *: t => SliceLabels[t]

    /** The shape that is left after slicing. */
    type SlicedShape[T <: Tuple, Inputs <: Tuple] = RemoveAll[T, SliceLabels[Inputs]]

    type Swap[T <: Tuple, A, B] <: Tuple = T match
      case EmptyTuple => EmptyTuple
      case A *: tail  => B *: Swap[tail, A, B]
      case B *: tail  => A *: Swap[tail, A, B]
      case h *: tail  => h *: Swap[tail, A, B]

    @implicitNotFound("The axis ${L} is already present in the tensor shape ${T}.")
    sealed trait AxisAbsent[T <: Tuple, L]
    object AxisAbsent:
      given derive[T <: Tuple, L](using NotGiven[Tuple.Contains[T, L] =:= true]): AxisAbsent[T, L] = new AxisAbsent[T, L] {}

  import Util.*

  object TensorWhere:
    /** Returns a new tensor where elements are selected from `x` or `y`
      * depending on the boolean condition.
      *
      * @param condition A tensor of boolean values that determines which elements to select.
      * @param x A tensor from which to select elements when the condition is true.
      * @param y A tensor from which to select elements when the condition is false.
      *
      * @return A new tensor with elements from `x` where the condition is true, and elements from `y` where the condition is false.
      */
    def where[T <: Tuple: Labels, V](
        condition: Tensor[T, Bool],
        x: Tensor[T, V],
        y: Tensor[T, V]
    ): Tensor[T, V] =
      Tensor(Jax.jnp.where(condition.jaxValue, x.jaxValue, y.jaxValue))

  export TensorWhere.where

  /** Returns a new tensor with the upper triangular part of the input tensor,
    * setting elements below the kth diagonal to zero.
    *
    * @param tensor The input tensor from which to extract the upper triangular part.
    * @param kthDiagonal The diagonal above which to set elements to zero.
    *
    * @return A new tensor with the upper triangular part of the input tensor.
    */
  def triu[T <: Tuple: Labels, V](tensor: Tensor[T, V], kthDiagonal: Int = 0): Tensor[T, V] =
    Tensor(Jax.jnp.triu(tensor.jaxValue, k = kthDiagonal))

  /** Returns a new tensor with the lower triangular part of the input tensor,
    * setting elements above the kth diagonal to zero.
    *
    * @param tensor The input tensor from which to extract the lower triangular part.
    * @param kthDiagonal The diagonal below which to set elements to zero.
    *
    * @return A new tensor with the lower triangular part of the input tensor.
    */
  def tril[T <: Tuple: Labels, V](tensor: Tensor[T, V], kthDiagonal: Int = 0): Tensor[T, V] =
    Tensor(Jax.jnp.tril(tensor.jaxValue, k = kthDiagonal))

  /** Stacks a sequence of tensors along a new axis.
    * The new axis is inserted as the first axis of the resulting tensor.
    *
    * @param tensors A sequence of tensors to be stacked. All tensors must have the same shape and type.
    * @param newAxis The new axis to be inserted.
    * @return A new tensor with the stacked tensors.
    */
  def stack[L: Label, T <: Tuple: Labels, V](
      tensors: Seq[Tensor[T, V]],
      newAxis: Axis[L]
  ): Tensor[L *: T, V] =
    require(tensors.nonEmpty, "Cannot stack an empty sequence of tensors")
    val jaxValuesSeq = tensors.map(_.jaxValue).toPythonProxy
    val stackedJaxValue = Jax.jnp.stack(jaxValuesSeq, axis = 0)
    Tensor(stackedJaxValue)

  /** Stacks a sequence of tensors along a new axis, inserting the new axis
    * after the specified existing axis.
    *
    * @param tensors A sequence of tensors to be stacked. All tensors must have the same shape and type.
    * @param newAxis The new axis to be inserted.
    * @param afterAxis The existing axis after which the new axis will be inserted.
    * @return A new tensor with the stacked tensors.
    */
  def stackAfter[NewL, L, T <: Tuple, V](
      tensors: Seq[Tensor[T, V]],
      newAxis: Axis[NewL],
      afterAxis: Axis[L]
  )(using
      axisIndex: AxisIndex[T, L],
      labels: Labels[InsertAfter[T, L, NewL]]
  ): Tensor[InsertAfter[T, L, NewL], V] =
    require(tensors.nonEmpty, "Cannot stack an empty sequence of tensors")
    val axisIdx = axisIndex.index + 1 // we are inserting after the given axis, so shift by 1
    val jaxValuesSeq = tensors.map(_.jaxValue).toPythonProxy
    Tensor(Jax.jnp.stack(jaxValuesSeq, axis = axisIdx))

  /** Concatenates a sequence of tensors along the specified axis, returning a new tensor with the concatenated values.
    *
    * @param tensors A sequence of tensors to be concatenated.
    *                All tensors must have the same shape and type,
    *                except for the dimension corresponding to the concatenation axis.
    * @param concatAxis The axis along which the tensors will be concatenated.
    * @return A new tensor with the concatenated values.
    */
  def concatenate[L: Label, T <: Tuple: Labels, V](
      tensors: Seq[Tensor[T, V]],
      concatAxis: Axis[L]
  )(using
      axisIndex: AxisIndex[T, L]
  ): Tensor[T, V] =
    require(tensors.nonEmpty, "Cannot concatenate an empty sequence of tensors")
    val axisIdx = axisIndex.index
    val jaxValuesSeq = tensors.map(_.jaxValue).toPythonProxy
    val concatenatedJaxValue = Jax.jnp.concatenate(jaxValuesSeq, axis = axisIdx)
    Tensor(concatenatedJaxValue)

  /** Concatenates two tensors along the specified axis,
    * returning a new tensor with the concatenated values.
    *
    * @param t1 The first tensor to be concatenated.
    * @param t2 The second tensor to be concatenated.
    * @param concatAxis The axis along which the tensors will be concatenated.
    * @return A new tensor with the concatenated values.
    */
  def concatenate[L: Label, T <: Tuple: Labels, V](
      t1: Tensor[T, V],
      t2: Tensor[T, V],
      concatAxis: Axis[L]
  )(using
      axisIndex: AxisIndex[T, L]
  ): Tensor[T, V] = concatenate(Seq(t1, t2), concatAxis)

  /** Concatenates two tensors along the common axis, returning a new tensor with the concatenated values.
    */
  def concatenate[T1 <: Tuple, T2 <: Tuple, V](
      t1: Tensor[T1, V],
      t2: Tensor[T2, V]
  )(using
      label: Labels[ConcatShape[T1, T2]],
      concatIndex: ValueOf[ConcatIndex[T1, T2]]
  ): Tensor[ConcatShape[T1, T2], V] =
    val jaxValues = List(t1.jaxValue, t2.jaxValue).toPythonProxy
    val axisIdx: Int = concatIndex.value
    Tensor(Jax.jnp.concatenate(jaxValues, axis = axisIdx))

  /** The shape of two concatenated shapes: they must agree everywhere but in the single axis they are
    * joined along, which becomes the concatenation of the two axes.
    */
  type ConcatShape[T1 <: Tuple, T2 <: Tuple] <: Tuple = T1 match
    case h *: t => ConcatShapeAt[h, t, T2]

  type ConcatShapeAt[H, T1 <: Tuple, T2 <: Tuple] <: Tuple = T2 match
    case H *: t => H *: ConcatShape[T1, t]
    case h *: t => (H |+| h) *: MustEqual[T1, t]

  /** Reduces to `X`, but only if it is also a `Y`. Used to require that the axes behind the
    * concatenated one agree.
    */
  type MustEqual[X <: Tuple, Y <: Tuple] <: Tuple = X match
    case Y => X

  /** The index of the axis that two shapes are concatenated along. */
  type ConcatIndex[T1 <: Tuple, T2 <: Tuple] <: Int = T1 match
    case h *: t => ConcatIndexAt[h, t, T2]

  type ConcatIndexAt[H, T1 <: Tuple, T2 <: Tuple] <: Int = T2 match
    case H *: t => ops.int.S[ConcatIndex[T1, t]]
    case h *: t => 0

  /** One component per split point, plus one for the remainder. */
  type SplitComponents[L, I <: Tuple] = L *: Tuple.Map[I, [_] =>> L]

  /** The axes a (possibly repeatedly) concatenated axis is built from. */
  type Components[L] <: Tuple = L match
    case a |+| b => Tuple.Concat[Components[a], Components[b]]
    case _       => L *: EmptyTuple

  /** The labels of [[Components]], which deconcatenating needs at runtime to name the parts. */
  trait ComponentLabels[L]:
    def labels: List[Label[?]]

  object ComponentLabels extends ComponentLabelsLowPriority:
    given concatenated[A, B](using a: ComponentLabels[A], b: ComponentLabels[B]): ComponentLabels[A |+| B] with
      def labels = a.labels ++ b.labels

  trait ComponentLabelsLowPriority:
    given single[L](using l: Label[L]): ComponentLabels[L] with
      def labels = List(l)

  /** The tensors that splitting the axis `SplitAxis` of a tensor of shape `FullShape` into `Comps` yields. */
  type SplitTensors[Comps <: Tuple, FullShape <: Tuple, SplitAxis, V] <: Tuple = Comps match
    case EmptyTuple => EmptyTuple
    case h *: t     => Tensor[Replace[FullShape, SplitAxis, h], V] *: SplitTensors[t, FullShape, SplitAxis, V]

  private def splitTensors[Comps <: Tuple, FullShape <: Tuple, SplitAxis, V](
      arrays: Seq[Jax.PyDynamic],
      componentLabels: List[Label[?]],
      originalLabels: List[String],
      splitIndex: Int
  ): SplitTensors[Comps, FullShape, SplitAxis, V] =
    val parts = arrays.zip(componentLabels).map: (array, label) =>
      val names = originalLabels.updated(splitIndex, label.name)
      Tensor[FullShape, V](array)(using LabelsImpl(names))
    Tuple.fromArray(parts.toArray).asInstanceOf[SplitTensors[Comps, FullShape, SplitAxis, V]]

  extension [T <: Tuple, V](tensor: Tensor[T, V])

    /** takes a concatenated tensor and splits it into a tuple of tensors along the specified axis,
      *  using the provided dimensions for each component.
      *
      * @param axis The axis along which to deconcatenate the tensor.
      * @param dims A tuple of AxisExtent specifying the sizes of each component along the specified axis.
      * @return A tuple of tensors corresponding to the deconcatenated components.
      *
      * Example usage:
      * {{{
      *   val t : Tensor2[Axis[A], Axis[B |+| C]) = ???
      *   val (partB, partC) = t.deconcatenate(Axis[B |+| C], (Axis[B] -> 2, Axis[C] -> 3)
      * }}}
      */
    def deconcatenate[L, Dims <: Tuple](
        axis: Axis[L],
        dims: Dims
    )(using
        labels: Labels[T],
        axisIndex: AxisIndex[T, L],
        componentLabels: ComponentLabels[L],
        extractor: DimExtractor[Dims]
    ): SplitTensors[Components[L], T, L, V] =
      val orderedSizes = dims.toList.asInstanceOf[List[Any]].map {
        case ae: AxisExtent[?] => ae.size
        case _                 => throw new IllegalArgumentException("Invalid dims format - expected AxisExtent")
      }

      require(orderedSizes.size == componentLabels.labels.size, s"Provided ${orderedSizes.size} sizes but axis has ${componentLabels.labels.size} components")

      val splitIndices = orderedSizes.scanLeft(0)(_ + _).tail.init
      val pyIndices = me.shadaj.scalapy.py.Dynamic.global.list(splitIndices.toPythonProxy)
      val splitArrays = Jax.jnp.split(tensor.jaxValue, pyIndices, axis = axisIndex.index).as[Seq[Jax.PyDynamic]]

      splitTensors[Components[L], T, L, V](splitArrays, componentLabels.labels, summon[Labels[T]].names.toList, axisIndex.index)

    /** Flattens all axes of the tensor into a single axis.
      * The resulting tensor will have a single axis named by concatenating the original axis names with "*".
      *
      * @return a Tensor1 with the merged axis
      */
    def flatten(using labels: Labels[T]): Tensor1[MergeLabels[T], V] =
      given Labels[Tuple1[MergeLabels[T]]] with
        def names = List(summon[Labels[T]].names.mkString("*"))
      Tensor(Jax.jnp.ravel(tensor.jaxValue))

    /** Flattens the specified axes of the tensor into a single axis.
      * The resulting tensor will have the specified axes merged into a single axis named by concatenating the original axis names with "*"
      * The other axes remain unchanged.
      *
      * @param axes the axes to flatten, specified as a tuple of Axis (e.g. (Axis[Ax1], Axis[Ax2]))
      * @return a Tensor with the specified axes merged into a single axis
      */
    def flatten[AxesTuple <: Tuple](
        axes: AxesTuple
    )(using
        merger: AxesMerger[T, UnwrapAxes[AxesTuple]],
        labels: Labels[MergeAxes[T, UnwrapAxes[AxesTuple]]]
    ): Tensor[MergeAxes[T, UnwrapAxes[AxesTuple]], V] =
      val permuted = Jax.jnp.transpose(tensor.jaxValue, merger.permutation.toPythonProxy)

      val originalDims = tensor.shape.dimensions
      val mergedSize = merger.mergeIndices.map(originalDims).product

      val remainingDims = originalDims.zipWithIndex
        .filterNot((d, i) => merger.mergeIndices.contains(i))
        .map(_._1)

      val newDimensions = remainingDims.patch(merger.mergedIndex, Seq(mergedSize), 0)

      Tensor(Jax.jnp.reshape(permuted, newDimensions.toPythonProxy))

    /** Unflattens splitAxis into a new shape specified by newShape. The other axes remain unchanged.
      *
      * The user must ensure that the size of splitAxis matches the product of the dimensions in newShape, otherwise a runtime error will occur.
      *
      * @param splitAxis the axis to unflatten
      * @param newShape the new shape to unflatten into, specified as a Shape
      * @return a Tensor with the specified axis unflattened into the new shape
      */
    def unflatten[SplitL, NewT <: Tuple](
        splitAxis: Axis[SplitL],
        newShape: Shape[NewT]
    )(using
        ev: AxisIndex[T, SplitL],
        labels: Labels[ReplaceBy[T, SplitL, NewT]]
    ): Tensor[ReplaceBy[T, SplitL, NewT], V] =
      val before = tensor.shape.dimensions.take(ev.index)
      val after = tensor.shape.dimensions.drop(ev.index + 1)
      val fullNewShape = before ++ newShape.dimensions ++ after
      Tensor(
        Jax.jnp.reshape(
          tensor.jaxValue,
          py.Dynamic.global.tuple(
            fullNewShape.map(py.Any.from).toPythonProxy
          )
        )
      )

    /** Unflattens the tensor into a new shape specified by newShape.
      *
      * The user must ensure that the size of the tensor matches the product of the dimensions in newShape, otherwise a runtime error will occur.
      *
      * @param newShape the new shape to unflatten into, specified as a Shape
      * @return a Tensor with the new shape
      */
    def unflatten[NewT <: Tuple: Labels](
        newShape: Shape[NewT]
    )(using
        @implicitNotFound("unflatten without axis can only be used on Tensor1 types.")
        ev: T <:< Tuple1[Any] // <--- Ensures this only works on Tensor1
    ): Tensor[NewT, V] =
      val fullNewShape = newShape.dimensions
      Tensor(
        Jax.jnp.reshape(
          tensor.jaxValue,
          py.Dynamic.global.tuple(
            fullNewShape.map(py.Any.from).toPythonProxy
          )
        )
      )

    /** Transposes the tensor according to the specified new order of axes.
      *
      * @param NewOrder A tuple representing the new order of axes for the tensor.
      * @return A new tensor with the axes transposed according to the specified order.
      */
    def transpose[NewOrder <: Tuple](newOrder: NewOrder)(using
        ev: AxisIndices[T, UnwrapAxes[NewOrder]],
        newLabels: Labels[UnwrapAxes[NewOrder]]
    )(using
        allAxesEv: IsPermutation[T, UnwrapAxes[NewOrder]]
    ): Tensor[UnwrapAxes[NewOrder], V] =
      val indices = ev.indices
      Tensor(Jax.jnp.transpose(tensor.jaxValue, indices.toPythonProxy))

    /** Splits the tensor along the specified axis at the given indices, returning a tuple of tensors corresponding to the splits.
      *
      * @param selector of the form Axis[L].at((idx1, idx2, ...)) specifying the axis to split and the indices to split at
      * @return the tuple of tensors resulting from the split
      */
    def split[L: Label, I <: NonEmptyTuple](selector: AxisAtTupleIndices[L, I])(using
        axisIndex: AxisIndex[T, L],
        labels: Labels[T]
    ): SplitTensors[SplitComponents[L, I], T, L, V] = splitAt(selector)

    private def splitAt[L: Label, I <: NonEmptyTuple](selector: AxisAtTupleIndices[L, I])(using
        axisIndex: AxisIndex[T, L],
        labels: Labels[T]
    ): SplitTensors[SplitComponents[L, I], T, L, V] =
      val splitList = selector.indices.toList.asInstanceOf[List[Int]]
      val pyIndices = me.shadaj.scalapy.py.Dynamic.global.list(splitList.toPythonProxy)
      val splitArrays = Jax.jnp.split(tensor.jaxValue, pyIndices, axis = axisIndex.index).as[Seq[Jax.PyDynamic]]
      val componentLabels = List.fill(splitList.size + 1)(summon[Label[L]].asInstanceOf[Label[?]])
      splitTensors[SplitComponents[L, I], T, L, V](splitArrays, componentLabels, labels.names.toList, axisIndex.index)

    /** Splits the tensor along the specified axis at the given index,
      * returning a tuple of two tensors corresponding to the splits.
      *
      * @param selector of the form Axis[L].at(idx) specifying the axis to split and the index to split at
      * @return a tuple of two tensors resulting from the split
      */
    def split[L: Label](selector: AxisAtIndex[L])(using
        axisIndex: AxisIndex[T, L],
        labels: Labels[T]
    ): SplitTensors[SplitComponents[L, Tuple1[Int]], T, L, V] =
      splitAt(AxisAtTupleIndices(selector.axis, Tuple1(selector.index)))

    private def calcPyIndices[Inputs <: Tuple](
        inputs: Inputs,
        targetDims: List[Int]
    ) =

      val PySlice = py.Dynamic.global.slice
      val Colon = PySlice(py.None)
      val rank = tensor.shape.rank
      val indicesBuffer = collection.mutable.ArrayBuffer.fill[py.Any](rank)(Colon)

      val inputList = inputs.toList.asInstanceOf[List[Any]]

      targetDims.zip(inputList).foreach { case (dimIndex, input) =>
        val dimSize = tensor.shape.dimensions(dimIndex)
        input match
          case AxisAtIndex(_, idx) =>
            indicesBuffer(dimIndex) = py.Any.from(idx)
          case AxisAtRange(_, range) =>
            indicesBuffer(dimIndex) = PySlice(range.head, range.last + 1, range.step)
          case AxisAtIndices(_, indices) =>
            indicesBuffer(dimIndex) = indices.map(py.Any.from).toPythonCopy // TODO find out why Copy is needed here
          case AxisAtTupleIndices(_, indices) =>
            indicesBuffer(dimIndex) = indices.toList.asInstanceOf[List[Int]].map(py.Any.from).toPythonCopy
          case AxisAtTensorIndex(_, tensorIdx) =>
            indicesBuffer(dimIndex) = tensorIdx.jaxValue
      }

      Jax.Dynamic.global.tuple(indicesBuffer.toSeq.toPythonProxy)

    /** Unstacks the tensor along the specified axis at the given indices, returning a sequence of tensors corresponding to the splits.
      *
      * @param unstackAxis the axis to split, specified as an Axis (e.g. Axis[Ax1])
      * @return a sequence of tensors resulting from the split, each with the specified axis removed
      */
    def unstack[L: Label](unstackAxis: Axis[L])(using
        labels: Labels[T],
        ev: AxisIndex[T, L],
        labelR: Labels[Remove[T, L]]
    ): Seq[Tensor[Remove[T, L], V]] =
      (0 until tensor.shape.dimensions(ev.index)).map: i =>
        val slicedJax = Jax.jnp.take(tensor.jaxValue, Jax.jnp.array(i), axis = ev.index)
        Tensor[Remove[T, L], V](slicedJax)

    /** splits the tensor into chunks of the specified size along the given axis
      * returning a sequence of tensors corresponding to the chunks.
      */
    def chunk[splitL: Label](splitAxis: Axis[splitL], chunkSize: Int)(using
        labels: Labels[T],
        axisIndex: AxisIndex[T, splitL]
    ): Seq[Tensor[T, V]] =
      val res = Jax.jnp.split(tensor.jaxValue, chunkSize, axis = axisIndex.index).as[Seq[Jax.PyDynamic]]
      res.map(x => Tensor[T, V](x))

    /** Slices the tensor according to the specified inputs,
      * removing the specified labels from the resulting tensor.
      *
      * @param inputs A tuple of inputs specifying how to slice the tensor.
      * @return The sliced tensor with the specified labels removed from its shape.
      */
    def slice[Inputs <: Tuple](
        inputs: Inputs
    )(using
        ev: AxisIndices[T, ExtractLabels[Inputs]],
        labels: Labels[SlicedShape[T, Inputs]]
    ): Tensor[SlicedShape[T, Inputs], V] =
      val pyIndices = tensor.calcPyIndices(inputs, ev.indices)
      Tensor(tensor.jaxValue.itemAt(pyIndices))

    /** Slice the given tensor, specifying the axis and index to slice at.
      *
      * @param selector An AxisAtIndex specifying the axis and index to slice at.
      * @return A sliced tensor with the specified axis removed from its shape.
      */
    def slice[L](
        selector: AxisAtIndex[L]
    )(using
        ev: AxisIndices[T, ExtractLabels[Tuple1[AxisAtIndex[L]]]],
        labels: Labels[Remove[T, L]]
    ): Tensor[Remove[T, L], V] = slice(Tuple1(selector))

    /** Slice the given tensor, specifying the axis and a given range to slice at.
      *
      * @param selector An AxisAtRange specifying the axis and range to slice at.
      * @return A sliced tensor, keeping the sliced axis.
      */
    def slice[L](
        selector: AxisAtRange[L]
    )(using
        ev: AxisIndices[T, ExtractLabels[Tuple1[AxisAtRange[L]]]],
        labels: Labels[T]
    ): Tensor[T, V] = slice(Tuple1(selector))

    /** Slice the given tensor, specifying the axis and a list of indices to slice at.
      *
      * @param selector An AxisAtIndices specifying the axis and indices to slice at.
      * @return A sliced tensor, keeping the sliced axis.
      */
    def slice[L](
        selector: AxisAtIndices[L]
    )(using
        ev: AxisIndices[T, ExtractLabels[Tuple1[AxisAtIndices[L]]]],
        labels: Labels[T]
    ): Tensor[T, V] = slice(Tuple1(selector))

    /** Slice the given tensor, specifying the axis and a tensor of indices to slice at.
      *
      * @param selector An AxisAtTensorIndex specifying the axis and tensor of indices to slice at.
      * @return A sliced tensor with the specified axis removed from its shape.
      */
    def slice[L](
        selector: AxisAtTensorIndex[L]
    )(using
        ev: AxisIndices[T, ExtractLabels[Tuple1[AxisAtTensorIndex[L]]]],
        labels: Labels[Remove[T, L]]
    ): Tensor[Remove[T, L], V] = slice(Tuple1(selector))

    /** Slice the given tensor, specifying the axis and a tuple of indices to slice at.
      *
      * @param selector An AxisAtTupleIndices specifying the axis and tuple of indices to slice at.
      * @return A sliced tensor, keeping the sliced axis.
      */
    def slice[L, U <: NonEmptyTuple](
        selector: AxisAtTupleIndices[L, U]
    )(using
        ev: AxisIndices[T, ExtractLabels[Tuple1[AxisAtTupleIndices[L, U]]]],
        labels: Labels[T]
    ): Tensor[T, V] = slice(Tuple1(selector))

    def take[L1, L2: Label](
        axis: Axis[L1]
    )(
        indices: Tensor1[L2, Int32]
    )(using
        ev: AxisIndex[T, L1],
        labels: Labels[Replace[T, L1, L2]]
    ): Tensor[Replace[T, L1, L2], V] =
      val result = Jax.jnp.take(tensor.jaxValue, indices.jaxValue, axis = ev.index)
      Tensor(result)

    def set[Inputs <: Tuple](
        inputs: Inputs
    )(using
        ev: AxisIndices[T, ExtractLabels[Inputs]],
        labels: Labels[T]
    )(value: Tensor[SlicedShape[T, Inputs], V]): Tensor[T, V] = setAt(inputs, ev.indices, value)

    private def setAt[Inputs <: Tuple, R <: Tuple](inputs: Inputs, indices: List[Int], value: Tensor[R, V])(using
        labels: Labels[T]
    ): Tensor[T, V] =
      val pyIndices = tensor.calcPyIndices(inputs, indices)
      Tensor(tensor.jaxValue.at.itemAt(pyIndices).set(value.jaxValue))

    // Convenience overload for Float
    def set[Inputs <: Tuple](
        inputs: Inputs
    )(using
        ev: AxisIndices[T, ExtractLabels[Inputs]],
        isScalar: SlicedShape[T, Inputs] =:= EmptyTuple,
        labels: Labels[T]
    )(value: Float): Tensor[T, V] =
      val pyIndices = tensor.calcPyIndices(inputs, ev.indices)
      val result = tensor.jaxValue.at.itemAt(pyIndices).set(value)
      Tensor(result)

    // Convenience overload for AxisAtIndex
    def set[L](
        selector: AxisAtIndex[L]
    )(using
        ev: AxisIndices[T, ExtractLabels[Tuple1[AxisAtIndex[L]]]],
        labels: Labels[T]
    )(value: Tensor[Remove[T, L], V]): Tensor[T, V] = setAt(Tuple1(selector), ev.indices, value)

    // Convenience overload for AxisAtRange
    def set[L](
        selector: AxisAtRange[L]
    )(using
        ev: AxisIndices[T, ExtractLabels[Tuple1[AxisAtRange[L]]]],
        labels: Labels[T]
    )(value: Tensor[T, V]): Tensor[T, V] = setAt(Tuple1(selector), ev.indices, value)

    // Convenience overload for AxisAtIndices
    def set[L](
        selector: AxisAtIndices[L]
    )(using
        ev: AxisIndices[T, ExtractLabels[Tuple1[AxisAtIndices[L]]]],
        labels: Labels[T]
    )(value: Tensor[T, V]): Tensor[T, V] = setAt(Tuple1(selector), ev.indices, value)

    // Convenience overload for AxisAtTensorIndex
    def set[L](
        selector: AxisAtTensorIndex[L]
    )(using
        ev: AxisIndices[T, ExtractLabels[Tuple1[AxisAtTensorIndex[L]]]],
        labels: Labels[T]
    )(value: Tensor[Remove[T, L], V]): Tensor[T, V] = setAt(Tuple1(selector), ev.indices, value)

    def rearrange[Axes <: Tuple](newOrder: Axes)(using
        Labels[UnwrapAxes[Axes]]
    )(using
        guard: CheckValid[ComputeMissing[UnwrapAxes[Axes], T, EmptyTuple]]
    ): Tensor[UnwrapAxes[Axes], V] =
      rearrange[Axes, EmptyTuple](newOrder, EmptyTuple)

    // Convenience overloads for a fixed number of dims (to support error messages with single axis)
    inline def rearrange[Axes <: Tuple, L1](newOrder: Axes, d1: AxisExtent[L1])(using
        guard: CheckValid[ComputeMissing[UnwrapAxes[Axes], T, UnwrapDims[Tuple1[AxisExtent[L1]]]]]
    )(using
        newLabels: Labels[UnwrapAxes[Axes]],
        extractor: DimExtractor[Tuple1[AxisExtent[L1]]]
    ): Tensor[UnwrapAxes[Axes], V] =
      rearrange(newOrder, Tuple1(d1))

    inline def rearrange[Axes <: Tuple, L1, L2](newOrder: Axes, d1: AxisExtent[L1], d2: AxisExtent[L2])(using
        guard: CheckValid[ComputeMissing[UnwrapAxes[Axes], T, UnwrapDims[(AxisExtent[L1], AxisExtent[L2])]]]
    )(using
        newLabels: Labels[UnwrapAxes[Axes]],
        extractor: DimExtractor[(AxisExtent[L1], AxisExtent[L2])]
    ): Tensor[UnwrapAxes[Axes], V] =
      rearrange(newOrder, (d1, d2))

    inline def rearrange[Axes <: Tuple, L1, L2, L3](newOrder: Axes, d1: AxisExtent[L1], d2: AxisExtent[L2], d3: AxisExtent[L3])(using
        guard: CheckValid[ComputeMissing[UnwrapAxes[Axes], T, UnwrapDims[(AxisExtent[L1], AxisExtent[L2], AxisExtent[L3])]]]
    )(using
        newLabels: Labels[UnwrapAxes[Axes]],
        extractor: DimExtractor[(AxisExtent[L1], AxisExtent[L2], AxisExtent[L3])]
    ): Tensor[UnwrapAxes[Axes], V] =
      rearrange(newOrder, (d1, d2, d3))

    inline def rearrange[Axes <: Tuple, L1, L2, L3, L4](newOrder: Axes, d1: AxisExtent[L1], d2: AxisExtent[L2], d3: AxisExtent[L3], d4: AxisExtent[L4])(using
        guard: CheckValid[ComputeMissing[UnwrapAxes[Axes], T, UnwrapDims[(AxisExtent[L1], AxisExtent[L2], AxisExtent[L3], AxisExtent[L4])]]]
    )(using
        newLabels: Labels[UnwrapAxes[Axes]],
        extractor: DimExtractor[(AxisExtent[L1], AxisExtent[L2], AxisExtent[L3], AxisExtent[L4])]
    ): Tensor[UnwrapAxes[Axes], V] =
      rearrange(newOrder, (d1, d2, d3, d4))

    def rearrange[Axes <: Tuple, Dims <: Tuple](
        newOrder: Axes,
        dims: Dims
    )(using
        guard: CheckValid[ComputeMissing[UnwrapAxes[Axes], T, UnwrapDims[Dims]]]
    )(using
        newLabels: Labels[UnwrapAxes[Axes]],
        extractor: DimExtractor[Dims]
    ): Tensor[UnwrapAxes[Axes], V] =
      def cleanPatternPrime(pattern: String): String =
        // Support dimwit.Prime by replacing ' with "Prime"
        pattern.replaceAll(
          "'",
          "Prime"
        )
      def createEinopsPattern(fromPattern: String, toPattern: String): String =
        def cleanPatternStar(pattern: String): String =
          // to replace all a*b*c in pattern with (a b c), example:
          // "a*b*c d e f*g h" -> "(a b c) d e (f g) h"
          val regex = raw"([a-zA-Z0-9_]+(\*[a-zA-Z0-9_]+)+)".r
          regex.replaceAllIn(
            pattern,
            _.group(1).split("\\*").mkString("(", " ", ")")
          )
        def cleanPatternPlus(pattern: String): String =
          // Support dimwit.|+| by replacing + with underlines
          val regex = raw"([a-zA-Z0-9_]+(\+[a-zA-Z0-9_]+)+)".r
          regex.replaceAllIn(
            pattern,
            _.group(1).replace("+", "_")
          )
        def cleanPattern(pattern: String): String =
          cleanPatternPlus(cleanPatternStar(cleanPatternPrime(pattern)))
        s"${cleanPattern(fromPattern)} -> ${cleanPattern(toPattern)}"
      val fromPattern = tensor.shape.labels.mkString(" ")
      val toPattern = newLabels.names.mkString(" ")
      val pattern = createEinopsPattern(fromPattern, toPattern)
      val dimSizesMap = extractor.extract(dims)
      val cleanDimSizesMap = dimSizesMap.map { case (k, v) =>
        val newKey = cleanPatternPrime(k)
        (newKey, v)
      }
      Tensor(
        Einops.rearrange(
          tensor.jaxValue,
          pattern,
          kwargsMap = cleanDimSizesMap
        )
      )

    def broadcastTo[O <: Tuple: Labels](newShape: Shape[O])(using
        labels: Labels[T],
        ev: StrictSubset[T, O]
    ): Tensor[O, V] =
      /* Disallow implicit broadcasting where an *existing* axis changes size (implicitly).
       * dimwit broadcasting only adds missing axes, never changes existing ones.
       * 
       * This is a required check to prevent implicit broadcasting across dimwit.
       * If this check is not explicitly present, Jax.jnp.broadcast_to would implicit broadcast.*/
      def disallowImplicitShapeBroadcasting(): Unit =
        val tAxesDims = tensor.axes.zip(tensor.shape.dimensions).toMap
        val newShapeAxesDims = newShape.labels.zip(newShape.dimensions).toMap
        tensor.axes.foreach(axisName =>
          require(
            tAxesDims(axisName) == newShapeAxesDims(axisName),
            s"Broadcasting only adds missing axes. Present axes must have the same size. Axis ${axisName} has size ${tAxesDims(axisName)} in the current tensor but size ${newShapeAxesDims(axisName)} in the target shape."
          )
        )

      disallowImplicitShapeBroadcasting() // Make dimwit coders, good coders :)

      val t = tensor

      val currentNames = summon[Labels[T]].names
      val targetNames = summon[Labels[O]].names

      val targetOrder = targetNames.filter(currentNames.contains)
      val permutation = targetOrder.map(n => currentNames.indexOf(n))

      val alignedJax =
        if permutation != currentNames.indices.toList then Jax.jnp.transpose(t.jaxValue, permutation.toPythonProxy)
        else t.jaxValue

      val currentShapeMap = currentNames.zip(t.shape.dimensions).toMap

      val intermediateShape = targetNames.map { name =>
        currentShapeMap.getOrElse(name, 1)
      }

      val reshapedJax = Jax.jnp.reshape(alignedJax, intermediateShape.toPythonProxy)
      Tensor(Jax.jnp.broadcast_to(reshapedJax, newShape.dimensions.toPythonProxy))

    def relabel[OldLabel: Label, NewLabel: Label](
        rename: (Axis[OldLabel], Axis[NewLabel])
    )(using
        ev: AxisIndex[T, OldLabel],
        newLabels: Labels[Replace[T, OldLabel, NewLabel]]
    ): Tensor[Replace[T, OldLabel, NewLabel], V] = Tensor(tensor.jaxValue)

    def retag[newT <: Tuple](using newLabels: Labels[newT]): Tensor[newT, V] =
      Tensor(tensor.jaxValue)(using newLabels)

    def relabelAll[newT <: Tuple](
        newAxes: newT
    )(using
        newLabels: Labels[UnwrapAxes[newT]],
        @implicitNotFound("Cannot convert tensor of shape ${T} to shape ${newT} due to size mismatch.")
        evSameSize: Tuple.Size[newT] =:= Tuple.Size[T]
    ): Tensor[UnwrapAxes[newT], V] = Tensor[UnwrapAxes[newT], V](tensor.jaxValue)

    def swap[L1: Label, L2: Label](
        axis1: Axis[L1],
        axis2: Axis[L2]
    )(using
        labels: Labels[T],
        axisIndex1: AxisIndex[T, L1],
        axisIndex2: AxisIndex[T, L2]
    ): Tensor[Swap[T, L1, L2], V] =
      given Labels[Swap[T, L1, L2]] with
        def names =
          val originalNames = summon[Labels[T]].names
          val ax1Name = summon[Label[L1]].name
          val ax2Name = summon[Label[L2]].name
          originalNames.map {
            case n if n == ax1Name => ax2Name
            case n if n == ax2Name => ax1Name
            case n                 => n
          }
      Tensor(Jax.jnp.swapaxes(tensor.jaxValue, axisIndex1.index, axisIndex2.index))

    def appendAxis[L: Label](axis: Axis[L])(using labels: Labels[T], ev: AxisAbsent[T, L]): Tensor[Tuple.Concat[T, Tuple1[L]], V] =
      val newShape = tensor.shape.dimensions :+ 1
      Tensor(Jax.jnp.reshape(tensor.jaxValue, newShape.toPythonProxy))

    def prependAxis[L: Label](axis: Axis[L])(using labels: Labels[T], ev: AxisAbsent[T, L]): Tensor[Tuple.Concat[Tuple1[L], T], V] =
      val newShape = 1 +: tensor.shape.dimensions
      Tensor(Jax.jnp.reshape(tensor.jaxValue, newShape.toPythonProxy))

    def squeeze[L: Label](axis: Axis[L])(using
        ev: AxisIndex[T, L],
        labels: Labels[Remove[T, L]]
    ): Tensor[Remove[T, L], V] =
      require(
        tensor.shape.dimensions(ev.index) == 1,
        s"Cannot squeeze axis ${summon[Label[L]].name} of size ${tensor.shape.dimensions(ev.index)}"
      )
      Tensor(Jax.jnp.squeeze(tensor.jaxValue, axis = ev.index))

  extension [L: Label, V](tensor: Tensor1[L, V])
    def roll(shift: Int): Tensor1[L, V] =
      Tensor(Jax.jnp.roll(tensor.jaxValue, shift = shift, axis = 0))
