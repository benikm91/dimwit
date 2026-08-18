package dimwit.autodiff

import dimwit.*
import dimwit.Conversions.given
import dimwit.autodiff.Autodiff.Gradient

/** A parameter tree, declared top level so its Mirror is available. */
case class JacParams(w: Tensor1[A, Float32], b: Tensor1[B, Float32]) derives TensorTree

enum JacMode:
  case Rev, Fwd

/** Runs a jacobian in either mode, so both can be driven through the same test bodies. */
def jacIn[In, Out](mode: JacMode)(f: In => Out)(using
    inTree: TensorTree[In],
    outTree: TensorTree[Out],
    grad: Autodiff.Gradient[In, Out],
    gradTree: TensorTree[grad.Result]
): In => grad.Result =
  mode match
    case JacMode.Rev => Autodiff.jacRev(f)
    case JacMode.Fwd => Autodiff.jacFwd(f)

class AutodiffSuite extends DimwitTest:

  describe("grad"):
    describe("single parameter function"):
      it("d¹, d², d³ of x²"):
        def f(x: Tensor0[Float32]) = x * x
        val df = Autodiff.grad(f)
        val ddf = Autodiff.grad((x: Tensor0[Float32]) => df(x).value)
        val dddf = Autodiff.grad((x: Tensor0[Float32]) => ddf(x).value)

        val x = Tensor0(3.0f)
        df(x) shouldEqual Tensor0(6.0f)
        ddf(x) shouldEqual Tensor0(2.0f)
        dddf(x) shouldEqual Tensor0(0.0f)

      it("d¹ sum(x²)"):
        def f(x: Tensor1[A, Float32]) = (x * x).sum
        val df = Autodiff.grad(f)

        val x = Tensor1(Axis[A]).fromArray(Array(1.0f, 5.0f))
        df(x) shouldEqual Tensor1(Axis[A]).fromArray(Array(2.0f, 10.0f))

      it("d¹ function using vmap"):
        def f(x: Tensor2[A, B, Float32]) = x.vmap(Axis[A])(_.sum).sum
        val df = Autodiff.grad(f)

        val x = Tensor(Shape(Axis[A] -> 2, Axis[B] -> 2)).fill(1f)
        df(x) shouldEqual Tensor.like(x).fill(1f)

    describe("two parameter function"):
      it("d¹/dx and d¹/dy of (x + 2y)²"):
        def f(x: Tensor1[A, Float32], y: Tensor1[A, Float32]) = ((x + (y *! 2.0f)).pow(Tensor0(2.0f))).sum
        val df = Autodiff.grad(f)

        val x = Tensor1(Axis[A]).fromArray(Array(1.0f))
        val y = Tensor1(Axis[A]).fromArray(Array(1.0f))

        val (xGrad, yGrad) = df(x, y).value
        xGrad shouldEqual Tensor1(Axis[A]).fromArray(Array(6.0f))
        yGrad shouldEqual Tensor1(Axis[A]).fromArray(Array(12.0f))

  describe("valueAndGrad"):

    describe("two parameter function"):
      it("d¹/dx and d¹/dy of (x + 2y)²"):
        def f(x: Tensor1[A, Float32], y: Tensor1[A, Float32]) = ((x + (y *! 2.0f)).pow(Tensor0(2.0f))).sum
        val df = Autodiff.grad(f)

        val x = Tensor1(Axis[A]).fromArray(Array(1.0f))
        val y = Tensor1(Axis[A]).fromArray(Array(1.0f))

        val g = Autodiff.valueAndGrad(f)
        val (value, grad) = g(x, y)

        value shouldEqual f(x, y)
        grad shouldEqual df(x, y).value

  describe("jacobian"):
    describe("single parameter function"):
      it("Jacobian of f: R² -> R², f(x) = 2x"):
        def f(x: Tensor1[A, Float32]) = x *! 2.0f
        val jf = Autodiff.jacobian(f)

        val x = Tensor1(Axis[A]).fromArray(Array(1.0f, 1.0f))
        jf(x) should approxEqual(Tensor2.eye(x.extent(Axis[A]), x.vtype) *! 2.0f)

  describe("jacRev / jacFwd"):

    // setup engines to test both modes in the same way
    val engines = List(("jacRev", JacMode.Rev), ("jacFwd", JacMode.Fwd))

    engines.foreach:
      case (modeName, mode) =>
        it(s"$modeName d¹ on f: R² -> R², f(x) = swap(x)"):
          def f(x1: Tensor1[A, Float32], x2: Tensor1[A, Float32]): (Tensor1[A, Float32], Tensor1[A, Float32]) = (x2, x1)
          val df = jacIn(mode)(f.tupled)
          val x1 = Tensor1(Axis[A]).fromArray(Array(1.0f, 0.0f))
          val x2 = Tensor1(Axis[A]).fromArray(Array(0.0f, 1.0f))
          val (x1Grad, x2Grad) = df(x1, x2)
          val (x1_dx1, x1_dx2) = x1Grad
          val (x2_dx1, x2_dx2) = x2Grad
          x1_dx1 should approxEqual(Tensor.like(x1_dx1).fill(0f))
          x1_dx2 should approxEqual(Tensor2.eye(x1.extent(Axis[A]), x1.vtype))
          x2_dx1 should approxEqual(Tensor2.eye(x2.extent(Axis[A]), x2.vtype))
          x2_dx2 should approxEqual(Tensor.like(x2_dx2).fill(0f))

        it(s"$modeName d¹ on f: Tensor1[A] => Tensor1[B]"):
          def f(x: Tensor1[A, Float32]): Tensor1[B, Float32] = x.relabel(Axis[A] -> Axis[B]) *! 2.0f
          val df = jacIn(mode)(f)
          val x = Tensor1(Axis[A]).fromArray(Array(1.0f, 1.0f))
          df(x).axes shouldBe List("B", "A")
          df(x) should approxEqual((Tensor2.eye(x.extent(Axis[A])) *! 2.0f).relabelAll((Axis[B], Axis[A])))

        it(s"$modeName d² on f: R² -> R, f(x1, x2) = sum(x1 * x2)"):
          def f(x1: Tensor1[A, Float32], x2: Tensor1[A, Float32]): Tensor0[Float32] = (x1 * x2).sum
          val df = jacIn(mode)(f.tupled)
          val ddf = jacIn(mode)(df)
          val x1 = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
          val x2 = Tensor1(Axis[A]).fromArray(Array(3.0f, 4.0f))
          val (x1Grad, x2Grad) = ddf(x1, x2)
          val (x1_dx1, x1_dx2) = x1Grad
          val (x2_dx1, x2_dx2) = x2Grad
          x1_dx1 should approxEqual(Tensor.like(x1_dx1).fill(0f))
          x1_dx2 should approxEqual(Tensor2.eye(x1.extent(Axis[A]), x1.vtype) *! Tensor0(1.0f))
          x2_dx1 should approxEqual(Tensor2.eye(x2.extent(Axis[A]), x2.vtype) *! Tensor0(1.0f))
          x2_dx2 should approxEqual(Tensor.like(x2_dx2).fill(0f))

  describe("hessian"):
    describe("single parameter function"):
      it("Hessian of f(x) = x^2"):
        def f(x: Tensor0[Float32]) = x * x
        val hf = Autodiff.hessian(f)

        val x = Tensor0(3.0f)
        hf(x) shouldEqual Tensor0(2.0f)

      it("Hessian of f(x) = sum(x^2)"):
        def f(x: Tensor1[A, Float32]) = (x * x).sum
        val hf = Autodiff.hessian(f)

        val x = Tensor1(Axis[A]).fromArray(Array(1.0f, 5.0f))
        hf(x) should approxEqual(Tensor2.eye(x.extent(Axis[A]), x.vtype) *! 2.0f)

      it("Hessian of f(x1, x2) = sum(x1 * x2)"):
        def f(x1: Tensor1[A, Float32], x2: Tensor1[A, Float32]): Tensor0[Float32] = (x1 * x2).sum
        val hf = Autodiff.hessian(f.tupled)

        val x1 = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
        val x2 = Tensor1(Axis[A]).fromArray(Array(3.0f, 4.0f))
        val (x1Grad, x2Grad) = hf(x1, x2)
        val (x1_dx1, x1_dx2) = x1Grad
        val (x2_dx1, x2_dx2) = x2Grad
        x1_dx1 should approxEqual(Tensor.like(x1_dx1).fill(0f))
        x1_dx2 should approxEqual(Tensor2.eye(x1.extent(Axis[A]), x1.vtype) *! Tensor0(1.0f))
        x2_dx1 should approxEqual(Tensor2.eye(x2.extent(Axis[A]), x2.vtype) *! Tensor0(1.0f))
        x2_dx2 should approxEqual(Tensor.like(x2_dx2).fill(0f))

  describe("jacobian of a function whose input and output axes differ"):

    it("non-square jacobian: Tensor1[A] => Tensor1[B]"):
      def f(x: Tensor1[A, Float32]): Tensor1[B, Float32] = x.relabel(Axis[A] -> Axis[B]) *! 2.0f
      val jf: Tensor1[A, Float32] => Tensor[(B, A), Float32] = Autodiff.jacobian(f)

      val x = Tensor1(Axis[A]).fromArray(Array(1.0f, 1.0f))
      jf(x).axes shouldBe List("B", "A")
      jf(x) should approxEqual((Tensor2.eye(x.extent(Axis[A])) *! 2.0f).relabelAll((Axis[B], Axis[A])))

    it("primes an input axis that collides with an output axis"):
      def f(x: Tensor2[A, B, Float32]): Tensor1[B, Float32] = x.sum(Axis[A])
      val jf: Tensor2[A, B, Float32] => Tensor[(B, A, Prime[B]), Float32] = Autodiff.jacobian(f)

      val x = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 2)).fill(1f)
      val jac = jf(x)
      jac.axes shouldBe List("B", "A", "B'")
      jac.shape(Axis[A]) shouldBe 3
      // d(sum over A)_b / dx(a, b') is 1 exactly when b == b', for every a
      jac.sum shouldEqual Tensor0(6.0f)

    it("jacobian over a tuple input with differing axes"):
      def f(x: Tensor1[A, Float32], y: Tensor1[B, Float32]): Tensor0[Float32] = x.sum * y.sum
      val jf = Autodiff.jacobian(f.tupled)

      val x = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
      val y = Tensor1(Axis[B]).fromArray(Array(3.0f, 4.0f))
      val (dx, dy) = jf(x, y)
      dx should approxEqual(Tensor1(Axis[A]).fromArray(Array(7.0f, 7.0f)))
      dy should approxEqual(Tensor1(Axis[B]).fromArray(Array(3.0f, 3.0f)))

    it("hessian of a scalar loss over two different axes"):
      def f(x1: Tensor1[A, Float32], x2: Tensor1[B, Float32]): Tensor0[Float32] = x1.sum * x2.sum
      val hf = Autodiff.hessian(f.tupled)

      val x1 = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
      val x2 = Tensor1(Axis[B]).fromArray(Array(3.0f, 4.0f))
      val (d1, d2) = hf(x1, x2)
      val (d1_d1, d1_d2) = d1
      val (d2_d1, d2_d2) = d2
      d1_d1 should approxEqual(Tensor.like(d1_d1).fill(0f))
      d1_d2 should approxEqual(Tensor.like(d1_d2).fill(1f))
      d2_d1 should approxEqual(Tensor.like(d2_d1).fill(1f))
      d2_d2 should approxEqual(Tensor.like(d2_d2).fill(0f))

  describe("jacobian of structures that are not tensors or plain tuples"):

    val params = JacParams(
      Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f)),
      Tensor1(Axis[B]).fromArray(Array(3.0f, 4.0f))
    )

    it("differentiates a case class tree into a named tuple of its fields"):
      def f(p: JacParams): JacParams = p
      val jf = Autodiff.jacobian(f)
      val jac = jf(params)

      jac.w.w should approxEqual(Tensor2.eye(params.w.extent(Axis[A])))
      jac.w.b should approxEqual(Tensor.like(jac.w.b).fill(0f))
      jac.b.w should approxEqual(Tensor.like(jac.b.w).fill(0f))
      jac.b.b should approxEqual(Tensor2.eye(params.b.extent(Axis[B])))

    it("takes the hessian of a scalar loss over a case class tree"):
      def loss(p: JacParams): Tensor0[Float32] = (p.w * p.w).sum + (p.b * p.b).sum
      val hf = Autodiff.hessian(loss)
      val hess = hf(params)

      hess.w.w should approxEqual(Tensor2.eye(params.w.extent(Axis[A])) *! 2.0f)
      hess.w.b should approxEqual(Tensor.like(hess.w.b).fill(0f))
      hess.b.b should approxEqual(Tensor2.eye(params.b.extent(Axis[B])) *! 2.0f)

    it("differentiates a case class tree into a tensor over a different axis"):
      def f(p: JacParams): Tensor1[C, Float32] = p.w.relabel(Axis[A] -> Axis[C]) *! p.b.sum
      val jac = Autodiff.jacobian(f)(params)

      jac.w.axes shouldBe List("C", "A")
      jac.b.axes shouldBe List("C", "B")

    it("takes the jacobian of a jacobian over a case class tree"):
      def f(p: JacParams): Tensor0[Float32] = p.w.sum * p.b.sum
      val hess = Autodiff.jacobian(Autodiff.jacobian(f))(params)

      hess.w.w should approxEqual(Tensor.like(hess.w.w).fill(0f))
      hess.w.b should approxEqual(Tensor.like(hess.w.b).fill(1f))
      hess.b.w should approxEqual(Tensor.like(hess.b.w).fill(1f))
      hess.b.b should approxEqual(Tensor.like(hess.b.b).fill(0f))

    it("differentiates a function taking a named tuple"):
      def f(p: (w: Tensor1[A, Float32], b: Tensor1[B, Float32])): Tensor0[Float32] = p.w.sum * p.b.sum
      val jf = Autodiff.jacobian(f)
      val jac = jf((w = params.w, b = params.b))

      jac.w should approxEqual(Tensor.like(jac.w).fill(params.b.sum.item))
      jac.b should approxEqual(Tensor.like(jac.b).fill(params.w.sum.item))

    it("differentiates a function returning a named tuple"):
      def f(x: Tensor1[A, Float32]): (u: Tensor1[A, Float32], v: Tensor1[A, Float32]) =
        (u = x *! 2.0f, v = x *! 3.0f)
      val jf = Autodiff.jacobian(f)

      val x = Tensor1(Axis[A]).fromArray(Array(1.0f, 1.0f))
      val jac = jf(x)
      jac.u should approxEqual(Tensor2.eye(x.extent(Axis[A])) *! 2.0f)
      jac.v should approxEqual(Tensor2.eye(x.extent(Axis[A])) *! 3.0f)

  describe("Complex application"):
    it("case class support"):
      case class Params(w: Tensor1[A, Float32], b: Tensor0[Float32])
      def loss(data: Tensor1[A, Float32])(params: Params): Tensor0[Float32] =
        ((data * params.w).sum + params.b).pow(Tensor0(2.0f))
      val trainData = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
      val dloss = Autodiff.grad(loss(trainData))
      val params = Params(Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f)), Tensor0(3.0f))
      val dParams = dloss(params)
      dParams.value.w shouldEqual Tensor1(Axis[A]).fromArray(Array(16.0f, 32.0f))
