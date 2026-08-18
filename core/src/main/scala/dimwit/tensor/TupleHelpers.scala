package dimwit.tensor

import dimwit.Prime
import dimwit.|*|

import scala.annotation.implicitNotFound
import scala.compiletime.ops
import scala.quoted.Expr
import scala.quoted.Quotes
import scala.quoted.Type
import scala.util.NotGiven

/** Type level operations on shape tuples. */
object TupleHelpers:

  /** A tuple of `N` elements, all of type `T`. */
  type TupleNOf[N <: Int, T] <: Tuple = N match
    case 0 => EmptyTuple
    case _ => T *: TupleNOf[ops.int.-[N, 1], T]

  /** `T` without element of `X`. */
  type Remove[T <: Tuple, X] <: Tuple = T match
    case X *: t => t
    case h *: t => h *: Remove[t, X]

  /** `T` without any of the elements of `ToRemove`. */
  type RemoveAll[T <: Tuple, ToRemove <: Tuple] <: Tuple = ToRemove match
    case EmptyTuple => T
    case h *: t     => RemoveAll[Remove[T, h], t]

  /** `T` with `Target` exchanged for `Replacement`. */
  type Replace[T <: Tuple, Target, Replacement] <: Tuple = T match
    case Target *: t => Replacement *: t
    case h *: t      => h *: Replace[t, Target, Replacement]

  /** `T` with `Target` element expanded into all elements of `Replacements`. */
  type ReplaceBy[T <: Tuple, Target, Replacements <: Tuple] <: Tuple = T match
    case Target *: t => Tuple.Concat[Replacements, t]
    case h *: t      => h *: ReplaceBy[t, Target, Replacements]

  /** Evidence that every axis of `S` also occurs in `T`. */
  sealed trait Subset[S <: Tuple, T <: Tuple]

  object Subset:
    given empty[T <: Tuple]: Subset[EmptyTuple, T] with {}

    given head[H, STail <: Tuple, T <: Tuple](using
        evH: SetMember[H, T],
        evT: Subset[STail, T]
    ): Subset[H *: STail, T] with {}

  sealed trait SetMember[K, T <: Tuple]

  object SetMember:
    given found[K, T <: Tuple]: SetMember[K, K *: T] with {}
    given search[K, H, T <: Tuple](using ev: SetMember[K, T]): SetMember[K, H *: T] with {}

  @implicitNotFound("The shape ${S} must be contained in ${T}, and ${T} must not be ${S} itself.")
  sealed trait StrictSubset[S <: Tuple, T <: Tuple]

  object StrictSubset:
    given derive[S <: Tuple, T <: Tuple](using
        ev: Subset[S, T],
        notEq: NotGiven[S =:= T]
    ): StrictSubset[S, T] with {}

  /** That two shapes hold the same axes, up to order.
    *
    * Only the arity is checked here; that each axis of the reordering exists in the source shape is
    * established separately, by the [[dimwit.tensor.ShapeTypeHelpers.AxisIndices]] that the caller needs anyway.
    */
  @implicitNotFound("The shape ${A} is not a valid permutation of ${B}.")
  sealed trait IsPermutation[A <: Tuple, B <: Tuple]

  object IsPermutation:
    given derive[A <: Tuple, B <: Tuple](using Tuple.Size[A] =:= Tuple.Size[B]): IsPermutation[A, B] with {}

  /** Primes (see [[dimwit.Prime]]) every axis of `Incoming` that already occurs in `Fixed`. */
  type PrimeRest[Fixed <: Tuple, Incoming <: Tuple] <: Tuple = Incoming match
    case EmptyTuple => EmptyTuple
    case h *: t     => Tuple.Contains[Fixed, h] match
        case true  => Prime[h] *: PrimeRest[Fixed, t]
        case false => h *: PrimeRest[Fixed, t]

  /** `R1` followed by `R2`, priming the axes of `R2` that would otherwise collide with one of `R1`. */
  type PrimeConcat[R1 <: Tuple, R2 <: Tuple] = Tuple.Concat[R1, PrimeRest[R1, R2]]

  /** Validation of a rearrange pattern, computed by [[ComputeMissing]] and turned into
    * either a compiling program or a readable error by [[CheckValid]].
    */
  sealed trait ValidationResult
  final class AllOk extends ValidationResult
  final class MissingAxis[A, InT <: Tuple] extends ValidationResult

  /** Whether axis `A` can be formed, either because the source shape has it or because its extent was given. */
  type CanForm[A, Source <: Tuple, Ignore <: Tuple] <: Boolean = Tuple.Contains[Source, A] match
    case true  => true
    case false => Tuple.Contains[Ignore, A]

  /** Walks the target axes and reports the first one that cannot be formed from the source. */
  type ComputeMissing[Target <: Tuple, Source <: Tuple, Ignore <: Tuple] <: ValidationResult = Target match
    case EmptyTuple => AllOk
    case h *: t     => CanForm[h, Source, Ignore] match
        case true  => ComputeMissing[t, Source, Ignore]
        case false => ComputeMissingDecomposed[h, t, Source, Ignore]

  /** A combined axis that is not available as a whole is searched for component by component. */
  type ComputeMissingDecomposed[H, Rest <: Tuple, Source <: Tuple, Ignore <: Tuple] <: ValidationResult = H match
    case l |*| r => ComputeMissing[l *: r *: Rest, Source, Ignore]
    case _       => MissingAxis[H, Source]

  /** Turns the outcome of [[ComputeMissing]] into a compile time error carrying the missing axis. */
  sealed trait CheckValid[R <: ValidationResult]

  object CheckValid:
    given ok: CheckValid[AllOk] = new CheckValid[AllOk] {}

    private def failImpl[A: Type, SourceShape <: Tuple: Type](using Quotes): Expr[CheckValid[MissingAxis[A, SourceShape]]] =
      import scala.quoted.quotes.reflect.*
      val name = Type.show[A]
      val sourceShape = Type.show[SourceShape]

      report.errorAndAbort(
        s"""❌ Missing Axis: '$name' in the source shape $sourceShape. There are a few possible reasons:
            |  1. Missing axis $name is not present in the source shape $sourceShape.
            |   👉 New structure must be based on source shape
            |  2. Missing axis $name is present only in flattened form (e.g., $name|*|OtherAxis) in the source shape $sourceShape. This requires additional information to be unflattened.
            |   If you are unflattening (e.g. $name|*|OtherAxis -> $name, OtherAxis), you must provide the size of '$name' explicitly.
            |   👉 Try: .rearrange(newOrder, (Axis[$name] -> size, ...)), where size is the length of $name after the unflattening.
            |""".stripMargin
      )

    inline given fail[A, SourceShape <: Tuple]: CheckValid[MissingAxis[A, SourceShape]] =
      ${ failImpl[A, SourceShape] }
