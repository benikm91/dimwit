# DimWit - Tensor programming with wit

> Programming is the art of telling another human being what one wants the computer to do.
> 
> Donald Knuth

<p align="center">
  <img src="assets/dimwit-logo.svg" alt="DimWit — Tensor programming with wit" width="360">
</p>


DimWit is a Scala library for numerical and machine learning programs that are easy for humans to understand.
▶ Watch the [ScalaDays 2026 talk](https://www.youtube.com/watch?v=2ZwGhCWXN5g).

## Why DimWit?

- **Strong types and fast execution.** JAX executes fast, but its tensors are untyped. DimWit wraps JAX in Scala and names the dimensions of tensors in their types, e.g. `Tensor2[Batch, Feature, Float32]`, so the code says what each axis means and the compiler checks it.
- **No overhead under `jit`.** DimWit only traces a function once; the compiled XLA program then runs without Scala in the loop. [GPT-2 in DimWit](https://github.com/dimwit-dev/deepwit/tree/main/examples/src/main/scala/deepwit/examples/gpt) (124M parameters, trained from scratch on 10B tokens) trains on par with plain JAX.
- **Names are a shared vocabulary.** Axes like `Batch` or `Feature` name the concepts of the domain. They become a shared vocabulary for your team and your AI agents, whether you discuss code, review it or prompt for it.
- **Explicit, not implicit.** DimWit is more than a one-to-one wrapper of JAX: what JAX does silently, DimWit makes explicit, e.g. broadcasting has its own operators (`x +! bias` instead of `x + bias`) and mixing precisions requires a cast (`x.asType(VType[Float16])`). In practice, this prevents many silent errors that are hard to find.

## Example

A minimal example of what named dimensions add to a tensor program, next to the same program in JAX:

<table>
<tr><th>DimWit</th><th>JAX</th></tr>
<tr>
<td>

```scala
// The axis names are the shared vocabulary
trait Batch derives Label
trait Feature derives Label
trait Output derives Label

// f has a strict signature: mapping a feature vector to an output vector
def f(x: Tensor1[Feature, Float32]): Tensor1[Output, Float32] = ...

// Tensor types carry meaning and are tracked by the compiler
val X: Tensor2[Batch, Feature, Float32] = ...
val Y: Tensor2[Batch, Output, Float32] =
  X.vmap(Axis[Batch])(f) // the axis by name, not position
```

</td>
<td>

```python






def f(x: Tensor) -> Tensor: ...


X: Tensor = ...
Y: Tensor = (
  jax.vmap(f, in_axes=0)(X))
```

</td>
</tr>
</table>

In DimWit, the names state the intent of the code; in JAX, the reader has to infer it from axis positions.

```scala mdoc:invisible
// Compiles the DimWit example above
import dimwit.*

trait Batch derives Label
trait Feature derives Label
trait Output derives Label

val A: Tensor2[Feature, Output, Float32] = Tensor2(Axis[Feature], Axis[Output]).fromArray(Array(Array(0f, -1f, 0f), Array(1f, 2f, 2f)))

def f(x: Tensor1[Feature, Float32]): Tensor1[Output, Float32] =
  x.dot(Axis[Feature])(A)

val X: Tensor2[Batch, Feature, Float32] = Tensor2(Axis[Batch], Axis[Feature]).fromArray(Array(Array(1f, 2f), Array(3f, 4f), Array(5f, 6f)))
val Y: Tensor2[Batch, Output, Float32] =
  X.vmap(Axis[Batch])(f)
```

See our [quickstart guide](docs/quickstart.md) for a more detailed introduction to the core concepts and API and 
check out the [examples](examples/src/main/scala).

## Scope

Like JAX, DimWit is a library for fast numerical programming: tensors, automatic differentiation, `vmap`, `jit` and random numbers.
Everything else is built on top of it:

| Project | Description |
|---|---|
| [DeepWit](https://github.com/dimwit-dev/deepwit) | Type-safe deep learning library with layers, attention and GPT-2. |
| [WitDraw](https://github.com/dimwit-dev/witdraw) | Experimental library for probabilistic programming. |

## Using DimWit as a Library

### Requirements

- Scala **3.8** or newer
- sbt **1.11** or newer

### Installation

Add to your `build.sbt` (minimal complete example):

```scala
ThisBuild / scalaVersion := "3.8.1"

lazy val myProject = (project in file("."))
  .settings(
    name := "my-project",
    libraryDependencies ++= Seq(
      "ch.contrafactus" %% "dimwit-core" % "0.1.0"
    ),
    fork := true
  )
```

### Python Environment Setup

DimWit requires **Python 3.9+** and **JAX** since it uses JAX as the backend for tensor operations via ScalaPy. It also relies on **Einops** for tensor reshaping and manipulation.

The easiest way to set up the Python environment is to install the [uv package manager](https://github.com/astral-sh/uv) and add a `pyproject.toml` to your **project root** (the directory where you run `sbt`):

```toml
[project]
name = "my-project-python-env"
version = "0.1.0"
requires-python = "==3.13.*"
dependencies = [
    "einops>=0.8.1",
    # Use jax[cpu] for CPU-only environments (e.g. development/CI without a GPU)
    # Use jax[cuda12] for NVIDIA GPU support
    "jax[cpu]>=0.8.2",
]
```

DimWit provides the command `dimwit.initialize()` which you can call at the start of your application to automatically set up the Python environment. This will check for the required dependencies and set the necessary environment variables for ScalaPy.

```scala 
import dimwit.*

@main def runApp(): Unit = {
    // Initialize the Python environment for DimWit
    dimwit.initialize()

    println(Tensor0(42.0f)) 
}   
```

Alternatively, you can configure the Python environment manually using these environment variables:

- `DIMWIT_PYTHON_PATH` — path to the Python executable (find it with `which python3` or `uv run python -c "import sys; print(sys.executable)"`)
- `DIMWIT_PYTHON_LIBRARY` — path to the Python shared library (find it with `python3 -c "import ctypes.util; print(ctypes.util.find_library('python3'))"`)
- `DIMWIT_SKIP_SYNC` — set to `true` to skip uv environment sync on startup

## Status 

**Ready to use.** The core concepts are stable and the API is maturing, but breaking changes may still happen until 1.0.

## Contributing

We would love to have you on board! Feel free to open issues or pull requests on GitHub.