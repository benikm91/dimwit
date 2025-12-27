package nn

import shapeful.*
import shapeful.jax.Jax

object ActivationFunctions:

  // TODO rewrite relu, sigmoid to JAX
  
  def sigmoid[T <: Tuple : Labels](t: Tensor[T, Float]): Tensor[T, Float] =
    val ones = Tensor(Of(t)).ones(t.shape)
    val minust = t.scale(-Tensor0(Of(t)).one)
    ones / (ones + (minust).exp)
  
  def relu[T <: Tuple : Labels, V](t: Tensor[T, V]): Tensor[T, V] = 
    val zeros = Tensor(Of(t)).zeros(t.shape)
    maximum(t, zeros)

  def gelu[T <: Tuple : Labels, V](t: Tensor[T, V]): Tensor[T, V] =
    Tensor(Jax.jnn.gelu(t.jaxValue))

  def softmax[L: Label, V](t: Tensor1[L, V]): Tensor1[L, V] =
    Tensor(Jax.jnn.softmax(t.jaxValue, axis = 0))
