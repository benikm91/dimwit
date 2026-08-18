package dimwit.tensor

import scala.annotation.implicitNotFound
import scala.compiletime.erasedValue
import scala.compiletime.summonInline

/* Helpers for tracking Tensor Shape types across various operations */
object ShapeTypeHelpers:

  import TupleHelpers.*

  /** Wraps each element of a tuple in an Axis */
  type WrapAxes[T <: Tuple] = Tuple.Map[T, Axis]

  /** Unwraps each Axis in a tuple to get the label types */
  type UnwrapAxes[T <: Tuple] <: Tuple = T match
    case EmptyTuple      => EmptyTuple
    case Axis[a] *: tail => a *: UnwrapAxes[tail]
    case h *: tail       => h *: UnwrapAxes[tail]

  /** Unwrap each AxisExtent in a tuple to get the label types */
  type UnwrapDims[T <: Tuple] <: Tuple = T match
    case EmptyTuple            => EmptyTuple
    case AxisExtent[a] *: tail => a *: UnwrapDims[tail]

  /** Removes axis `X` from every shape of `Shapes`. */
  type RemoveFromAll[Shapes <: Tuple, X] <: Tuple = Shapes match
    case EmptyTuple => EmptyTuple
    case h *: t     => Remove[h, X] *: RemoveFromAll[t, X]

  /** Base trait for tracking an axis in a tensor shape */
  @implicitNotFound("Axis[${Axis}] not found in Tensor[${TensorShape}]")
  trait AxisInTensor[TensorShape <: Tuple, Axis]:
    def index: Int

  /** The index at which an axis sits in a tensor shape.
    *
    * A type class rather than a match type over the shape: implicit search unifies abstract labels,
    * which generic code inside the library relies on, whereas a match type would get stuck on them.
    */
  trait AxisIndex[Shape <: Tuple, Axis] extends AxisInTensor[Shape, Axis]

  object AxisIndex:

    def apply[T <: Tuple, L](using idx: AxisIndex[T, L]): Int = idx.index

    given head[L, Tail <: Tuple]: AxisIndex[L *: Tail, L] with
      val index = 0

    given tail[H, T <: Tuple, L](using
        next: AxisIndex[T, L]
    ): AxisIndex[H *: T, L] with
      val index = 1 + next.index

    given concatRight[A <: Tuple, B <: Tuple, L](using
        sizeA: ValueOf[Tuple.Size[A]],
        idxB: AxisIndex[B, L]
    ): AxisIndex[Tuple.Concat[A, B], L] with
      val index = sizeA.value + idxB.index

    given concatEnd[A <: Tuple, L]: AxisIndex[Tuple.Concat[A, Tuple1[L]], L] with
      val index = -1

  /** Base trait for tracking multiple axes in a tensor shape */
  @implicitNotFound("Axes [${Axes}] not all found in Tensor shape [${TensorShape}]")
  trait AxesInTensor[TensorShape <: Tuple, Axes <: Tuple]:
    def indices: List[Int]

  /** The indices at which several axes sit in a tensor shape */
  sealed trait AxisIndices[T <: Tuple, Axes <: Tuple] extends AxesInTensor[T, Axes]

  object AxisIndices:

    class AxisIndicesImpl[T <: Tuple, Axes <: Tuple](val indices: List[Int]) extends AxisIndices[T, Axes]

    private inline def indicesOfList[InTuple <: Tuple, ToFind <: Tuple]: List[Int] =
      inline erasedValue[ToFind] match
        case _: EmptyTuple     => Nil
        case _: (head *: tail) =>
          summonInline[AxisIndex[InTuple, head]].index :: indicesOfList[InTuple, tail]

    inline given [T <: Tuple, ToFind <: Tuple](using Subset[ToFind, T]): AxisIndices[T, ToFind] = AxisIndicesImpl[T, ToFind](indicesOfList[T, ToFind])

  /** Removes a shared axis from several tensor shapes at once, providing the index of that axis in
    * each of them together with the labels that each shape keeps.
    */
  @implicitNotFound("Axis[${Axis}] not found in all of ${Shapes}")
  trait SharedAxisRemover[Shapes <: Tuple, Axis]:
    def indices: List[Int]
    def shapesLabels: List[List[String]]

  object SharedAxisRemover:

    given empty[Axis]: SharedAxisRemover[EmptyTuple, Axis] with
      def indices = Nil
      def shapesLabels = Nil

    given cons[H <: Tuple, T <: Tuple, Axis](using
        head: AxisIndex[H, Axis],
        tail: SharedAxisRemover[T, Axis],
        headLabels: Labels[Remove[H, Axis]]
    ): SharedAxisRemover[H *: T, Axis] with
      def indices = head.index :: tail.indices
      def shapesLabels = headLabels.names :: tail.shapesLabels

  /** Extracts the dimensions of a tensor shape into a Map of label names to sizes.
    */
  trait DimExtractor[T]:
    def extract(t: T): Map[String, Int]

  object DimExtractor:
    given DimExtractor[EmptyTuple] with
      def extract(t: EmptyTuple) = Map.empty

    given [L, Tail <: Tuple](using
        label: Label[L],
        tailExtractor: DimExtractor[Tail]
    ): DimExtractor[AxisExtent[L] *: Tail] with
      def extract(t: AxisExtent[L] *: Tail) =
        val size = t.head.size
        Map(label.name -> size) ++ tailExtractor.extract(t.tail)

    given single[L](using label: Label[L]): DimExtractor[AxisExtent[L]] with
      def extract(t: AxisExtent[L]) =
        Map(label.name -> t.size)

  import dimwit.|*|

  /** The single label that the axes of `T` collapse into when they are merged. */
  type MergeLabels[T <: Tuple] = T match
    case head *: tail => MergeLabelsRec[tail, head]

  type MergeLabelsRec[T <: Tuple, Acc] = T match
    case EmptyTuple   => Acc
    case head *: tail => MergeLabelsRec[tail, Acc |*| head]

  object MergeLabels:
    given [T <: Tuple: Labels]: Label[MergeLabels[T]] with
      def name = summon[Labels[T]].names.mkString("*")

  /** The shape that remains once the axes `ToMerge` of `S` have been merged into a single axis,
    * which takes the place of the first of them.
    */
  type MergeAxes[S <: Tuple, ToMerge <: Tuple] <: Tuple = ToMerge match
    case head *: tail => Replace[RemoveAll[S, tail], head, MergeLabels[ToMerge]]

  /** The runtime information needed to merge the axes `ToMerge` of a tensor of shape `S`. */
  @implicitNotFound("Cannot merge axes ${ToMerge} in shape ${S}. Ensure all axes exist.")
  trait AxesMerger[S <: Tuple, ToMerge <: Tuple]:
    def permutation: List[Int] // To make axes contiguous
    def mergedIndex: Int // Where the new axis sits in the merged shape
    def mergeIndices: List[Int] // Original indices of axes to be merged

  object AxesMerger:

    given derive[S <: Tuple, ToMerge <: Tuple](using
        indices: AxisIndices[S, ToMerge],
        mergedIdx: AxisIndex[MergeAxes[S, ToMerge], MergeLabels[ToMerge]],
        rank: ValueOf[Tuple.Size[S]]
    ): AxesMerger[S, ToMerge] with

      def mergeIndices = indices.indices

      def permutation: List[Int] =
        val toMerge = indices.indices
        val others = (0 until rank.value).filterNot(toMerge.contains).toList
        // Move all 'toMerge' indices to the position of the first one (the pivot)
        val pivotIdxInS = toMerge.head
        val (pref, suff) = others.partition(_ < pivotIdxInS)
        pref ++ toMerge ++ suff

      def mergedIndex: Int = mergedIdx.index
