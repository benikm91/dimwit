package dimwit.python

import dimwit.tensor.*
import dimwit.jax.Jax
import me.shadaj.scalapy.py

object PyBridge:

  def liftPyTensor[T <: Tuple: Labels, V](jaxValue: Jax.PyDynamic): Tensor[T, V] = new Tensor(jaxValue)
  def liftPyTensor[T <: Tuple: Labels, V](shape: Shape[T], vtype: VType[V])(jaxValue: Jax.PyDynamic): Tensor[T, V] = new Tensor(jaxValue)

  def liftPyTensor0[V](vtype: VType[V])(jaxValue: Jax.PyDynamic): Tensor0[V] = new Tensor(jaxValue)
  def liftPyTensor1[L: Label, V](ax: Axis[L], vtype: VType[V])(jaxValue: Jax.PyDynamic): Tensor1[L, V] = new Tensor(jaxValue)

  def toPyTensor[T <: Tuple, V](tensor: Tensor[T, V]): Jax.PyDynamic = tensor.jaxValue

  extension [T <: Tuple: Labels, V](tensor: Tensor[T, V])
    def applyPy(f: py.Dynamic): Tensor[T, V] = liftPyTensor(f(toPyTensor(tensor)))
