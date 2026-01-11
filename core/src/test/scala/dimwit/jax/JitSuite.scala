package dimwit.jax

import dimwit.*
import dimwit.Conversions.given
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import me.shadaj.scalapy.py

class JitSuite extends AnyFunSpec with Matchers:

  it("JIT compilation works correctly"):
    def complexFn(t: Tensor1[A, Float]): Tensor1[A, Float] =
      (0 until 50).foldLeft(t) { (acc, _) => acc * ((acc +! 1f) /! 2f) }

    val jitComplexFn = jit(complexFn)
    val tensor = Tensor.ones(Shape1(Axis[A] -> 5), VType[Float])

    val res = complexFn(tensor)
    val jittedRes = jitComplexFn(tensor)
    noException should be thrownBy (tensor.toString) // tensor is still usable, toString to trigger materialization
    res should approxEqual(jittedRes)

  it("JIT compilation works correctly with donate_argument"):
    def complexFn(t: Tensor1[A, Float]): Tensor1[A, Float] =
      (0 until 50).foldLeft(t) { (acc, _) => acc * ((acc +! 1f) /! 2f) }

    val jitComplexFn = jit(complexFn, Map("donate_argnums" -> Tuple1(0)))
    val tensor = Tensor.ones(Shape1(Axis[A] -> 5), VType[Float])

    val res = complexFn(tensor)
    val jittedRes = jitComplexFn(tensor)
    an[Exception] should be thrownBy (tensor.toString) // tensor is not usable, toString to trigger materialization
    res should approxEqual(jittedRes)

  it("JIT compilation example: Speedup for jitted function"):
    def timeFn[T](fn: T => Any, input: T, runs: Int = 100): Long =
      val start = System.nanoTime()
      for _ <- 0 until runs do fn(input)
      val end = System.nanoTime()
      (end - start) / 1_000_000 // ms

    val tensor = Tensor.ones(Shape1(Axis[A] -> 5), VType[Float])

    def complexFn(t: Tensor1[A, Float]): Tensor1[A, Float] =
      (0 until 50).foldLeft(t) { (acc, _) => acc * ((acc +! 1f) /! 2f) }

    val jitComplexFn = jit(complexFn)

    // pre-compile function as in the test we want to compare only execution time
    val compilationTimeMs = timeFn(jitComplexFn, tensor) // first call includes compilation time

    val regularTimeMs = timeFn(complexFn, tensor)
    val jittedTimeMs = timeFn(jitComplexFn, tensor)

    info(f"Regular execution:             $regularTimeMs%.2f ms")
    info(f"JIT execution:                 $jittedTimeMs%.2f ms")
    info(f"JIT compilation overhead time: $compilationTimeMs%.2f ms")
    info(f"Speedup (wo compile overhead): ${regularTimeMs / jittedTimeMs}%.2f x")
    info(f"Speedup (w compile overhead):  ${regularTimeMs / (jittedTimeMs + compilationTimeMs)}%.2f x")

    jittedTimeMs should be < regularTimeMs
