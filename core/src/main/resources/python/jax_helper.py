# JAX helper functions for dimwit library

import jax
import jax.numpy as jnp
from jax import vmap

import builtins
builtins.jax = jax
builtins.jnp = jnp

def wrap_fn(f):
    """Wrap a ScalaPy callback as a plain Python callable."""
    def python_wrapper(*a, **kw):
        return f(*a, **kw)
    return python_wrapper

def wrap(jax_transform, f, kwargs=None):
    """
    Generic wrapper that shields a ScalaPy function from JAX introspection,
    then applies the given JAX transform.

    Usage:
        wrap(jax.grad, f)
        wrap(jax.vmap, f, kwargs={"in_axes": 0})
        wrap(jax.jit, f, kwargs={"donate_argnums": (0,)})
        wrap(jax.jacfwd, f)
    """
    if kwargs:
        return jax_transform(wrap_fn(f), **kwargs)
    return jax_transform(wrap_fn(f))

def vmap(f, dims):
    return wrap(jax.vmap, f, kwargs={"in_axes": dims})
           
def zipvmap(f, dims):
    def python_wrapper(*args):
        return f(args)
    return lambda jax_inputs_tuple: jax.vmap(python_wrapper, in_axes=dims)(*jax_inputs_tuple)

def apply_over_axes(f, axis):
    """
    Applies a function `f` over specified axes using JAX's vmap functionality.
    
    Args:
        f: Function that takes one argument (x)
        axis: Axis or tuple of axes to map over
    
    It is wrapped in a Python function to ensure that the function, as otherwise
    jax will crash upon inspection.
    """
                
    # Wrap the ScalaPy function in a pure Python wrapper
    def python_wrapper(x):
        return f(x)
            
    # Create vmap with the wrapper
    return jnp.apply_over_axes(python_wrapper, axis)

def vmap2(f, dims):
    in_axes = (dims, dims) if isinstance(dims, int) else dims
    return wrap(jax.vmap, f, kwargs={"in_axes": in_axes})



def grad(f):
    return wrap(jax.grad, f)

def value_and_grad(f):
    return wrap(jax.value_and_grad, f)

def jacfwd(f):
    return wrap(jax.jacfwd, f)

def jacrev(f):
    return wrap(jax.jacrev, f)

def jacobian(f):
    return wrap(jax.jacobian, f)

def hessian(f):
    return wrap(jax.hessian, f)

def jit(f):
    return wrap(jax.jit, f)

def jit_fn(f, jit_kwargs=None):
    return wrap(jax.jit, f, kwargs=jit_kwargs)


# Structured control flow. The body is traced once, so a loop does not unroll under jit.
# Each Scala callback is wrapped in a Python lambda, which shields it from JAX introspection.

def scan(f, init, xs):
    """jax.lax.scan; f(carry, x) returns the Python tuple (carry, y)."""
    return jax.lax.scan(lambda carry, x: tuple(f(carry, x)), init, xs)

def fori_loop(lower, upper, body, init):
    return jax.lax.fori_loop(lower, upper, lambda i, carry: body(i, carry), init)

def while_loop(cond, body, init):
    return jax.lax.while_loop(lambda carry: cond(carry), lambda carry: body(carry), init)

def cond(pred, true_fn, false_fn):
    """jax.lax.cond without operands: the branches are closures."""
    return jax.lax.cond(pred, lambda: true_fn(None), lambda: false_fn(None))
