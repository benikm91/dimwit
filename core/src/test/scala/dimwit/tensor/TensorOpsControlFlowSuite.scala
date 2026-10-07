package dimwit.tensor

import dimwit.*
import dimwit.jax.Jax
import me.shadaj.scalapy.py

object TensorOpsControlFlowSuite:
  trait Time derives Label
  trait Feature derives Label

  case class Cache(keys: Tensor2[Time, Feature, Float32], count: Tensor0[Int32])

class TensorOpsControlFlowSuite extends DimwitTest:

  import TensorOpsControlFlowSuite.*

  // Shape: Time=4, Feature=3, the value at (t, f) is t + f / 10
  val xs = Tensor2(Axis[Time], Axis[Feature]).fromArray(Array.tabulate(4, 3)((t, f) => t + f / 10f))
  val zeros = Tensor1(Axis[Feature]).fromArray(Array(0f, 0f, 0f))

  /** The running sum of `xs` along Time, computed with a Scala fold. */
  def runningSum(xs: Tensor2[Time, Feature, Float32]): (Tensor1[Feature, Float32], Tensor2[Time, Feature, Float32]) =
    val sums = xs.unstack(Axis[Time]).scanLeft(zeros)(_ + _).tail
    (sums.last, stack(sums, Axis[Time]))

  /** The number of operations in the jaxpr of `f`, without those inside loop bodies. */
  def jaxprSize(f: Tensor1[Feature, Float32] => Tensor1[Feature, Float32]): Int =
    val fpy = (x: py.Dynamic) => f(Tensor[Tuple1[Feature], Float32](x)).jaxValue
    py.Dynamic.global.len(Jax.jax.make_jaxpr(Jax.jax_helper.wrap_fn(fpy))(zeros.jaxValue).jaxpr.eqns).as[Int]

  describe("scan over an axis"):

    it("gives the same result as a fold"):
      val (sum, sums) = scan(Axis[Time])(zeros, xs)((carry, x) => (carry + x, carry + x))
      val (expectedSum, expectedSums) = runningSum(xs)
      sum should approxEqual(expectedSum)
      sums should approxEqual(expectedSums)

    it("gives the same result under jit"):
      val jitScan = jit((xs: Tensor2[Time, Feature, Float32]) => scan(Axis[Time])(zeros, xs)((carry, x) => (carry + x, carry + x)))
      val (sum, sums) = jitScan(xs)
      val (expectedSum, expectedSums) = runningSum(xs)
      sum should approxEqual(expectedSum)
      sums should approxEqual(expectedSums)

    it("scans over an axis that is not the first one and prepends it to the outputs"):
      val (sum, sums) = scan(Axis[Time])(zeros, xs.transpose)((carry, x) => (carry + x, carry + x))
      val (expectedSum, expectedSums) = runningSum(xs)
      sum should approxEqual(expectedSum)
      sums.axes shouldBe List("Time", "Feature")
      sums should approxEqual(expectedSums)

    it("scans over a tuple of tensors with a case class carry"):
      val weights = Tensor1(Axis[Time]).fromArray(Array(1f, 2f, 3f, 4f))
      val init = Cache(Tensor(Shape(Axis[Time] -> 1, Axis[Feature] -> 3)).fill(0f), Tensor0(0))
      val (cache, counts) = scan(Axis[Time])(init, (xs, weights)):
        case (cache, (x, weight)) =>
          val keys = cache.keys + (x *! weight).prependAxis(Axis[Time])
          (Cache(keys, cache.count + Tensor0(1)), cache.count)
      val weightedSum = xs.unstack(Axis[Time]).zip(Seq(1f, 2f, 3f, 4f)).map((x, w) => x *! w).reduce(_ + _)
      cache.count.item shouldBe 4
      cache.keys.slice(Axis[Time].at(0)) should approxEqual(weightedSum)
      counts shouldEqual Tensor1(Axis[Time]).fromArray(Array(0, 1, 2, 3))

    it("has the same gradient as the unrolled loop"):
      def scanned(w: Tensor1[Feature, Float32]): Tensor0[Float32] =
        scan(Axis[Time])(zeros, xs)((h, x) => ((h * w + x).tanh, ()))._1.sum
      def unrolled(w: Tensor1[Feature, Float32]): Tensor0[Float32] =
        xs.unstack(Axis[Time]).foldLeft(zeros)((h, x) => (h * w + x).tanh).sum
      val w = Tensor1(Axis[Feature]).fromArray(Array(0.5f, -0.3f, 0.8f))
      scanned(w) should approxEqual(unrolled(w))
      Autodiff.grad(scanned)(w).value should approxEqual(Autodiff.grad(unrolled)(w).value, 1e-5f)

    it("rejects tensors with different extents along the axis"):
      val tooShort = xs.slice(Axis[Time].at(0 until 3))
      the[IllegalArgumentException] thrownBy scan(Axis[Time])(zeros, (xs, tooShort))((c, x) => (c + x._1 + x._2, ())) should have message
        "requirement failed: All tensors scanned over Axis[Time] must have the same extent along it, but they have 4, 3"

    it("rejects a body that changes the shape of the carry, naming the field"):
      val init = Cache(Tensor(Shape(Axis[Time] -> 1, Axis[Feature] -> 3)).fill(0f), Tensor0(0))
      val error = the[Exception] thrownBy scan(Axis[Time])(init, xs): (cache, x) =>
        (Cache(concatenate(cache.keys, x.prependAxis(Axis[Time]), Axis[Time]), cache.count), ())
      error.getMessage should include(
        "The body of a loop must not change the shape of the carry, but `keys` is Shape(Time -> 1, Feature -> 3) of Float32 and then Shape(Time -> 2, Feature -> 3) of Float32"
      )

    it("traces its body once, so the program does not grow with the number of steps"):
      var traces = 0
      def scanned(steps: Int)(x: Tensor1[Feature, Float32]) =
        val (last, _) = scan(Axis[Time])(x, Tensor(Shape(Axis[Time] -> steps)).fill(1.5f)): (x, factor) =>
          traces += 1
          ((x *! factor).sin, ())
        last
      def unrolled(steps: Int)(x: Tensor1[Feature, Float32]) = (0 until steps).foldLeft(x)((x, _) => (x *! 1.5f).sin)

      jaxprSize(scanned(100)) shouldBe jaxprSize(scanned(10))
      jaxprSize(unrolled(100)) should be > jaxprSize(unrolled(10))
      traces = 0
      jit(scanned(100))(zeros)
      traces shouldBe 1

  describe("foriLoop"):

    it("fills a buffer row by row at the traced index"):
      val buffer = jit((xs: Tensor2[Time, Feature, Float32]) =>
        foriLoop(0, 4)(Tensor(xs.shape).fill(0f)): (i, buffer) =>
          buffer.set(Axis[Time].at(i, 1))(xs.slice(Axis[Time].at(i, 1)) *! 2f)
      )(xs)
      buffer should approxEqual(xs *! 2f)

    it("is differentiable"):
      def cube(x: Tensor0[Float32]): Tensor0[Float32] = foriLoop(0, 2)(x)((_, y) => y * x)
      Autodiff.grad(cube)(Tensor0(2f)).value.item shouldBe 12f

  describe("whileLoop"):

    it("stops at the first carry for which the condition does not hold"):
      val (steps, value) = whileLoop((Tensor0(0), Tensor0(1f)))((_, value) => value < Tensor0(100f)):
        case (steps, value) => (steps + Tensor0(1), value * 2f)
      steps.item shouldBe 7
      value.item shouldBe 128f

    it("stops at an iteration that depends on traced values"):
      val stepsUntil = jit((limit: Tensor0[Float32]) =>
        whileLoop((Tensor0(0), Tensor0(1f)))((_, value) => value < limit)((steps, value) => (steps + Tensor0(1), value * 2f))._1
      )
      stepsUntil(Tensor0(100f)).item shouldBe 7
      stepsUntil(Tensor0(1000f)).item shouldBe 10

    it("cannot be reverse-differentiated"):
      def doubled(x: Tensor0[Float32]): Tensor0[Float32] = whileLoop(x)(_ < Tensor0(100f))(_ * 2f)
      val error = the[Exception] thrownBy Autodiff.grad(doubled)(Tensor0(1f))
      error.getMessage should include("Reverse-mode differentiation does not work for lax.while_loop")

  describe("cond"):

    it("evaluates the branch selected by a traced predicate"):
      val absolute = jit((x: Tensor0[Float32]) => cond(x > Tensor0(0f))(x)(-x))
      absolute(Tensor0(3f)).item shouldBe 3f
      absolute(Tensor0(-2f)).item shouldBe 2f

    it("rejects branches with different shapes"):
      val error = the[Exception] thrownBy cond(Tensor0(true))(zeros)(Tensor1(Axis[Feature]).fromArray(Array(0f)))
      error.getMessage should include("The branches of cond must return the same shapes, but the result is Shape(Feature -> 3) of Float32 and then Shape(Feature -> 1) of Float32")
