package dimwit.jax

import dimwit.tensor.{Tensor, Shape, Labels}
import dimwit.jax.{Jax, JaxDType}
import dimwit.autodiff.ToPyTree
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

object Jit:

  private def anyToPy(x: Any): py.Any = x match
    case v: py.Any    => v
    case v: Boolean   => py.Any.from(v)
    case v: Int       => py.Any.from(v)
    case v: Long      => py.Any.from(v)
    case v: Float     => py.Any.from(v)
    case v: Double    => py.Any.from(v)
    case v: String    => py.Any.from(v)
    case v: Seq[Any]  => v.map(anyToPy).toPythonProxy
    case v: Map[?, ?] => py.Any.from(v.map { case (k, v) => (k.toString, anyToPy(v)) })
    case v: Product   =>
      val elements = v.productIterator.map(anyToPy).toSeq
      py.Dynamic.global.tuple(elements.toPythonProxy)
    case null => py.None
    case _    => throw new IllegalArgumentException(s"Cannot convert type ${x.getClass} to Python.")

  def jit[T: ToPyTree, R: ToPyTree](
      f: T => R,
      pyKwargs: Map[String, Any] = Map()
  ): T => R =

    val fpy = (pyTreePy: Jax.PyDynamic) =>
      val pyTree = ToPyTree[T].fromPyTree(pyTreePy)
      val result = f(pyTree)
      ToPyTree[R].toPyTree(result)

    val jitted = Jax.jax_helper.jit_fn(fpy, anyToPy(pyKwargs))

    (pyTree: T) =>
      val pyTreePy = ToPyTree[T].toPyTree(pyTree)
      val res = jitted(pyTreePy)
      ToPyTree[R].fromPyTree(res)

  def jit[T1, T2, R](f: (T1, T2) => R)(using t1Tree: ToPyTree[T1], t2Tree: ToPyTree[T2], outTree: ToPyTree[R]): (T1, T2) => R =
    jit(f, Map())
  def jit[T1, T2, R](f: (T1, T2) => R, pyKwargs: Map[String, Any])(using t1Tree: ToPyTree[T1], t2Tree: ToPyTree[T2], outTree: ToPyTree[R]): (T1, T2) => R =
    val fpy = (t1: Jax.PyDynamic, t2: Jax.PyDynamic) =>
      val pyT1 = ToPyTree[T1].fromPyTree(t1)
      val pyT2 = ToPyTree[T2].fromPyTree(t2)
      val result = f(pyT1, pyT2)
      ToPyTree[R].toPyTree(result)

    val jitted = Jax.jax_helper.jit_fn(fpy, anyToPy(pyKwargs))

    (t1: T1, t2: T2) =>
      val pyT1 = ToPyTree[T1].toPyTree(t1)
      val pyT2 = ToPyTree[T2].toPyTree(t2)
      val res = jitted(pyT1, pyT2)
      ToPyTree[R].fromPyTree(res)

  def jit[T1, T2, T3, R](f: (T1, T2, T3) => R)(using t1Tree: ToPyTree[T1], t2Tree: ToPyTree[T2], t3Tree: ToPyTree[T3], outTree: ToPyTree[R]): (T1, T2, T3) => R =
    jit(f, Map())
  def jit[T1, T2, T3, R](f: (T1, T2, T3) => R, pyKwargs: Map[String, Any])(using t1Tree: ToPyTree[T1], t2Tree: ToPyTree[T2], t3Tree: ToPyTree[T3], outTree: ToPyTree[R]): (T1, T2, T3) => R =
    val fpy = (t1: Jax.PyDynamic, t2: Jax.PyDynamic, t3: Jax.PyDynamic) =>
      val pyT1 = ToPyTree[T1].fromPyTree(t1)
      val pyT2 = ToPyTree[T2].fromPyTree(t2)
      val pyT3 = ToPyTree[T3].fromPyTree(t3)
      val result = f(pyT1, pyT2, pyT3)
      ToPyTree[R].toPyTree(result)

    val jitted = Jax.jax_helper.jit_fn(fpy, anyToPy(pyKwargs))

    (t1: T1, t2: T2, t3: T3) =>
      val pyT1 = ToPyTree[T1].toPyTree(t1)
      val pyT2 = ToPyTree[T2].toPyTree(t2)
      val pyT3 = ToPyTree[T3].toPyTree(t3)
      val res = jitted(pyT1, pyT2, pyT3)
      ToPyTree[R].fromPyTree(res)

  def jit[T1, T2, T3, T4, R](f: (T1, T2, T3, T4) => R)(using t1Tree: ToPyTree[T1], t2Tree: ToPyTree[T2], t3Tree: ToPyTree[T3], t4Tree: ToPyTree[T4], outTree: ToPyTree[R]): (T1, T2, T3, T4) => R =
    jit(f, Map())
  def jit[T1, T2, T3, T4, R](f: (T1, T2, T3, T4) => R, pyKwargs: Map[String, Any])(using t1Tree: ToPyTree[T1], t2Tree: ToPyTree[T2], t3Tree: ToPyTree[T3], t4Tree: ToPyTree[T4], outTree: ToPyTree[R]): (T1, T2, T3, T4) => R =
    val fpy = (t1: Jax.PyDynamic, t2: Jax.PyDynamic, t3: Jax.PyDynamic, t4: Jax.PyDynamic) =>
      val pyT1 = ToPyTree[T1].fromPyTree(t1)
      val pyT2 = ToPyTree[T2].fromPyTree(t2)
      val pyT3 = ToPyTree[T3].fromPyTree(t3)
      val pyT4 = ToPyTree[T4].fromPyTree(t4)
      val result = f(pyT1, pyT2, pyT3, pyT4)
      ToPyTree[R].toPyTree(result)

    val jitted = Jax.jax_helper.jit_fn(fpy, anyToPy(pyKwargs))

    (t1: T1, t2: T2, t3: T3, t4: T4) =>
      val pyT1 = ToPyTree[T1].toPyTree(t1)
      val pyT2 = ToPyTree[T2].toPyTree(t2)
      val pyT3 = ToPyTree[T3].toPyTree(t3)
      val pyT4 = ToPyTree[T4].toPyTree(t4)
      val res = jitted(pyT1, pyT2, pyT3, pyT4)
      ToPyTree[R].fromPyTree(res)
