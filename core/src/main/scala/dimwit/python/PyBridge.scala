package dimwit.python

import dimwit.OnError
import dimwit.tensortree.TensorTree
import dimwit.jax.Jax
import dimwit.tensor.ValueTypeClasses.HasDType
import dimwit.tensor.*
import me.shadaj.scalapy.py

object PyBridge:

  def liftPyTensor[T <: Tuple: Labels, V: HasDType](jaxValue: Jax.PyDynamic): Tensor[T, V] = checked(jaxValue, summon[HasDType[V]].dtype)
  def liftPyTensor[T <: Tuple: Labels, V](shape: Shape[T], vtype: VType[V])(jaxValue: Jax.PyDynamic): Tensor[T, V] = checked(jaxValue, vtype.dtype)

  def liftPyTensor0[V](vtype: VType[V])(jaxValue: Jax.PyDynamic): Tensor0[V] = checked(jaxValue, vtype.dtype)
  def liftPyTensor1[L: Label, V](ax: Axis[L], vtype: VType[V])(jaxValue: Jax.PyDynamic): Tensor1[L, V] = checked(jaxValue, vtype.dtype)

  def toPyTensor[T <: Tuple, V](tensor: Tensor[T, V]): Jax.PyDynamic = tensor.jaxValue

  extension [T <: Tuple: Labels, V](tensor: Tensor[T, V])
    def applyPy(f: py.Dynamic): Tensor[T, V] = checked(f(toPyTensor(tensor)), tensor.dtype)

  // The array is not converted, cast on the Python side (e.g. jnp.astype) if a conversion is intended
  private def checked[T <: Tuple: Labels, V](jaxValue: Jax.PyDynamic, expected: DType): Tensor[T, V] =
    val tensor = new Tensor[T, V](jaxValue)
    require(tensor.dtype == expected, s"Expected a Python array of dtype ${expected.name}, but got ${tensor.dtype.name}.")
    tensor

  // --- liftPyFn: wrap a Python callable for typed Scala use ---

  def liftPyFn[In: TensorTree, Out: TensorTree](pyFunc: py.Dynamic): In => Out =
    (in: In) =>
      TensorTree[Out].fromPyTree(pyFunc(TensorTree[In].toPyTree(in)))

  def liftPyFn[T1: TensorTree, T2: TensorTree, Out: TensorTree](pyFunc: py.Dynamic): (T1, T2) => Out =
    val tupled = liftPyFn[(T1, T2), Out](pyFunc)
    (t1, t2) => tupled((t1, t2))

  def liftPyFn[T1: TensorTree, T2: TensorTree, T3: TensorTree, Out: TensorTree](pyFunc: py.Dynamic): (T1, T2, T3) => Out =
    val tupled = liftPyFn[(T1, T2, T3), Out](pyFunc)
    (t1, t2, t3) => tupled((t1, t2, t3))

  def liftPyFn[T1: TensorTree, T2: TensorTree, T3: TensorTree, T4: TensorTree, Out: TensorTree](pyFunc: py.Dynamic): (T1, T2, T3, T4) => Out =
    val tupled = liftPyFn[(T1, T2, T3, T4), Out](pyFunc)
    (t1, t2, t3, t4) => tupled((t1, t2, t3, t4))

  // --- toPyFn: wrap a Scala function as a Python-callable py.Dynamic ---

  /** Wrap a Scala `In => Out` as a `py.Dynamic` callable that operates on JAX pytrees.
    *
    * This is the "Python-side handle" counterpart of `toJax`: it does the same `fromPyTree` → f → `toPyTree`
    * plumbing but returns the raw `py.Dynamic` instead of re-wrapping back into a typed Scala function.
    *
    * Useful when a Python library (e.g. BlackJAX) needs a Python-callable function.
    */
  def toPyFn[In: TensorTree, Out: TensorTree](f: In => Out): py.Dynamic =
    val ttIn = TensorTree[In]
    val ttOut = TensorTree[Out]
    Jax.jax_helper.wrap_fn { (pyIn: Jax.PyDynamic) =>
      OnError.traceStack:
        ttOut.toPyTree(f(ttIn.fromPyTree(pyIn)))
    }

  // --- toJax: wrap a Scala function through a JAX transform ---

  def toJax[In: TensorTree, Out: TensorTree](jaxTransform: py.Dynamic)(f: In => Out): In => Out =
    val fpy = (pyIn: Jax.PyDynamic) =>
      OnError.traceStack:
        TensorTree[Out].toPyTree(f(TensorTree[In].fromPyTree(pyIn)))
    val transformed = Jax.jax_helper.wrap(jaxTransform, fpy, py.None)
    liftPyFn[In, Out](transformed)
