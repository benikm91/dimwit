package dimwit.tensor

import scala.quoted.*
import scala.compiletime.asMatchable

inline def summonTypeNames[T <: Tuple]: List[String] = ${ summonTypeNamesImpl[T] }

def summonTypeNamesImpl[T <: Tuple: Type](using Quotes): Expr[List[String]] =
  import quotes.reflect.*
  def getNames(tpe: TypeRepr): List[String] =
    if tpe.typeSymbol.isTypeParam then
      report.errorAndAbort(
        s"Cannot extract names from generic type ${tpe.show}. " +
          "Make sure the enclosing method is marked as 'inline'."
      )
    val dealiased = tpe.dealias
    if dealiased <:< TypeRepr.of[EmptyTuple] then Nil
    else
      dealiased.asMatchable match
        case AppliedType(_, List(head, tail)) =>
          head.typeSymbol.name :: getNames(tail)
        case _ => Nil
  Expr(getNames(TypeRepr.of[T].dealias))
