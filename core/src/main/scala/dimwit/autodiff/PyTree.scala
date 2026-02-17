package dimwit.autodiff

import dimwit.tensor.{Tensor, Shape}
import dimwit.jax.Jax
import dimwit.random.Random

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import scala.deriving.*
import scala.compiletime.*
import dimwit.tensor.Labels

trait ToPyTree[P]:
  def toPyTree(p: P): Jax.PyAny
  def fromPyTree(p: Jax.PyAny): P

import scala.deriving.*
import scala.compiletime.*

class ProductToPyTree[P <: Product](
    m: Mirror.ProductOf[P],
    elems: List[ToPyTree[Any]]
) extends ToPyTree[P]:

  def toPyTree(p: P): Jax.PyAny =
    val fields = p.productIterator.toList
    val pyTreeElems = fields.zip(elems).map: (field, tc) =>
      tc.toPyTree(field)
    py.Dynamic.global.tuple(pyTreeElems.toPythonProxy)

  def fromPyTree(pyTree: Jax.PyAny): P =
    val pyTuple = pyTree.as[py.Dynamic]
    val reconstructedArgs = elems.zipWithIndex.map: (tc, index) =>
      val item = pyTuple.bracketAccess(index)
      tc.fromPyTree(item)
    val tupleProduct = Tuple.fromArray(reconstructedArgs.toArray)
    m.fromProduct(tupleProduct)

object ToPyTree:

  def apply[P](using pt: ToPyTree[P]): ToPyTree[P] = pt

  given unitInstance: ToPyTree[Unit] with
    def toPyTree(u: Unit): Jax.PyAny = py.Dynamic.global.None
    def fromPyTree(p: Jax.PyAny): Unit = ()

  // Keep the tensor instance
  given [T <: Tuple: Labels, V]: ToPyTree[Tensor[T, V]] with
    def toPyTree(t: Tensor[T, V]): Jax.PyAny = t.jaxValue
    def fromPyTree(p: Jax.PyAny): Tensor[T, V] = Tensor(p.as[Jax.PyDynamic])

  // Random.Key instance - wraps and unwraps the JAX key
  given ToPyTree[Random.Key] with
    def toPyTree(k: Random.Key): Jax.PyAny = k.jaxKey
    def fromPyTree(p: Jax.PyAny): Random.Key = Random.Key(p.as[Jax.PyDynamic])

  // Tuple instances - these should have lower priority than specific case classes
  given tupleInstance[A, B](using ta: ToPyTree[A], tb: ToPyTree[B]): ToPyTree[(A, B)] with
    def toPyTree(t: (A, B)): Jax.PyAny =
      py.Dynamic.global.tuple(Seq(ta.toPyTree(t._1), tb.toPyTree(t._2)).toPythonProxy)

    def fromPyTree(p: Jax.PyAny): (A, B) =
      val pyTuple = p.as[py.Dynamic]
      val a = ta.fromPyTree(pyTuple.bracketAccess(0))
      val b = tb.fromPyTree(pyTuple.bracketAccess(1))
      (a, b)

  // Handle List[T] -> Python list
  given listInstance[A](using ta: ToPyTree[A]): ToPyTree[List[A]] with
    def toPyTree(l: List[A]): Jax.PyAny =
      val pyItems = l.map(ta.toPyTree)
      py.Dynamic.global.list(pyItems.toPythonProxy)

    def fromPyTree(p: Jax.PyAny): List[A] =
      val pyList = p.as[py.Dynamic]
      val len = py.Dynamic.global.len(pyList).as[Int]
      List.tabulate(len): i =>
        ta.fromPyTree(pyList.bracketAccess(i))

  // Handle String -> Python str (e.g., for Map keys)
  given stringToPyTree: ToPyTree[String] with
    def toPyTree(s: String): Jax.PyAny = py.Dynamic.global.str(s)
    def fromPyTree(p: Jax.PyAny): String = p.as[String]

  // Handle Map[K, V] -> Python dict
  given mapInstance[K, V](using kt: ToPyTree[K], vt: ToPyTree[V]): ToPyTree[Map[K, V]] with
    def toPyTree(m: Map[K, V]): Jax.PyAny =
      val pyItems = m.toList.map { case (k, v) =>
        py.Dynamic.global.tuple(Seq(kt.toPyTree(k), vt.toPyTree(v)).toPythonProxy)
      }
      py.Dynamic.global.dict(pyItems.toPythonProxy)

    def fromPyTree(p: Jax.PyAny): Map[K, V] =
      val itemsList = py.Dynamic.global.list(p.as[py.Dynamic].items())
      val len = py.Dynamic.global.len(itemsList).as[Int]
      List.tabulate(len) { i =>
        val itemTuple = itemsList.bracketAccess(i)
        val k = kt.fromPyTree(itemTuple.bracketAccess(0))
        val v = vt.fromPyTree(itemTuple.bracketAccess(1))
        k -> v
      }.toMap

  inline given derived[P <: Product](using m: Mirror.ProductOf[P]): ToPyTree[P] =
    val elemInstances = summonAll[Tuple.Map[m.MirroredElemTypes, ToPyTree]]
    val elemsList = elemInstances.toList.map(_.asInstanceOf[ToPyTree[Any]])
    new ProductToPyTree[P](m, elemsList)

  // Compile-time reconstruction using field types
  inline def reconstructFields[Types <: Tuple](pyTuple: py.Dynamic, index: Int): Tuple =
    inline erasedValue[Types] match
      case _: EmptyTuple =>
        EmptyTuple
      case _: (head *: tail) =>
        val elem = reconstructField[head](pyTuple.bracketAccess(index))
        val rest = reconstructFields[tail](pyTuple, index + 1)
        elem *: rest

  inline def reconstructField[T](pyElem: py.Dynamic): T =
    inline erasedValue[T] match
      case _: Tensor[?, ?] =>
        // For tensors, delegate to the ToPyTree instance which has the proper type info
        compiletime.summonInline[ToPyTree[T]].fromPyTree(pyElem)
      case _: String =>
        pyElem.as[String].asInstanceOf[T]
      case _: Int =>
        pyElem.as[Int].asInstanceOf[T]
      case _: Float =>
        pyElem.as[Float].asInstanceOf[T]
      case _: Double =>
        pyElem.as[Double].asInstanceOf[T]
      case _ =>
        // For complex types (case classes), try to find ToPyTree instance
        compiletime.summonInline[ToPyTree[T]].fromPyTree(pyElem)

  // Compile-time field conversion
  inline def convertFieldsAtCompileTime[Types <: Tuple](fields: Types): List[Jax.PyAny] =
    inline erasedValue[Types] match
      case _: EmptyTuple =>
        Nil
      case _: (head *: tail) =>
        val headElem = fields.asInstanceOf[head *: tail].head
        val tailElems = fields.asInstanceOf[head *: tail].tail
        val headPy = convertSingleField[head](headElem)
        val tailPy = convertFieldsAtCompileTime[tail](tailElems)
        headPy :: tailPy

  inline def convertSingleField[T](elem: T): Jax.PyAny =
    inline erasedValue[T] match
      case _: Tensor[?, ?] =>
        elem.asInstanceOf[Tensor[?, ?]].jaxValue
      case _: String =>
        py.Dynamic.global.str(elem.asInstanceOf[String])
      case _: Int =>
        py.Dynamic.global.int(elem.asInstanceOf[Int])
      case _: Float =>
        py.Dynamic.global.float(elem.asInstanceOf[Float])
      case _: Double =>
        py.Dynamic.global.float(elem.asInstanceOf[Double])
      case _ =>
        // Use compile-time instance lookup
        compiletime.summonInline[ToPyTree[T]].toPyTree(elem)
