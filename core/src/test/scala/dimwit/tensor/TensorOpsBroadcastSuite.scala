package dimwit.tensor

import dimwit.*
import dimwit.Conversions.given

class TensorOpsBroadcastSuite extends DimwitTest:

  val tA = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))

  val tAB = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(10.0f, 20.0f), Array(30.0f, 40.0f)))

  val tAB2 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(100.0f, 200.0f)))
  val iA = Tensor1(Axis[A]).fromArray(Array(1, 2))
  val iAB = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1, 2), Array(3, 4)))

  describe("Scalar Broadcasting"):

    describe("Int"):
      it("Addition"):
        (5 +! iAB) shouldEqual Tensor.like(iAB).fromArray(Array(6, 7, 8, 9))
        (5 +! iAB) shouldEqual (iAB +! 5)

      it("Subtraction"):
        (5 -! iAB) shouldEqual Tensor.like(iAB).fromArray(Array(4, 3, 2, 1))
        (iAB -! 5) shouldEqual Tensor.like(iAB).fromArray(Array(-4, -3, -2, -1))

      it("Multiplication"):
        (3 *! iAB) shouldEqual Tensor.like(iAB).fromArray(Array(3, 6, 9, 12))
        (3 *! iAB) shouldEqual (iAB *! 3)

      it("No Int Division Supported"):
        "5 /! iAB" shouldNot compile
        "iAB /! 5" shouldNot compile

    describe("Float"):

      it("Addition"):
        (2.0f +! tAB) shouldEqual Tensor.like(tAB).fromArray(Array(12.0f, 22.0f, 32.0f, 42.0f))
        (2.0f +! tAB) shouldEqual (tAB +! 2.0f)

      it("Subtraction"):
        (5.0f -! tAB) shouldEqual Tensor.like(tAB).fromArray(Array(-5.0f, -15.0f, -25.0f, -35.0f))
        (tAB -! 5.0f) shouldEqual Tensor.like(tAB).fromArray(Array(5.0f, 15.0f, 25.0f, 35.0f))

      it("Multiplication"):
        (2.0f *! tAB) shouldEqual Tensor.like(tAB).fromArray(Array(20.0f, 40.0f, 60.0f, 80.0f))
        (2.0f *! tAB) shouldEqual (tAB *! 2.0f)

      it("Division"):
        (2.0f /! tAB) shouldEqual Tensor.like(tAB).fromArray(Array(0.2f, 0.1f, 0.06666667f, 0.05f))
        (tAB /! 2.0f) shouldEqual Tensor.like(tAB).fromArray(Array(5.0f, 10.0f, 15.0f, 20.0f))

  describe("Vector-to-Tensor Broadcasting"):

    describe("Int"):

      it("Addition"):
        (iA +! iAB) shouldEqual Tensor.like(iAB).fromArray(Array(2, 3, 5, 6))
        (iA +! iAB) shouldEqual (iAB +! iA)

      it("Subtraction"):
        (iA -! iAB) shouldEqual Tensor.like(iAB).fromArray(Array(0, -1, -1, -2))
        (iAB -! iA) shouldEqual Tensor.like(iAB).fromArray(Array(0, 1, 1, 2))

      it("Multiplication"):
        (iA *! iAB) shouldEqual Tensor.like(iAB).fromArray(Array(1, 2, 6, 8))
        (iA *! iAB) shouldEqual (iAB *! iA)

      it("No Int Division Supported"):
        "iA /! iAB" shouldNot compile
        "iAB /! iA" shouldNot compile

    describe("Float"):

      it("Addition"):
        (tAB +! tA) should approxEqual(
          Tensor.like(tAB).fromArray(
            Array(11.0f, 21.0f, 32.0f, 42.0f)
          )
        )
        (tAB +! tA) shouldEqual (tA +! tAB)

      it("Subtraction"):
        (tAB -! tA) should approxEqual(
          Tensor.like(tAB).fromArray(
            Array(9.0f, 19.0f, 28.0f, 38.0f)
          )
        )
        (tA -! tAB) shouldEqual Tensor.like(tAB).fromArray(
          Array(-9.0f, -19.0f, -28.0f, -38.0f)
        )

      it("Multiplication"):
        (tAB *! tA) should approxEqual(
          Tensor.like(tAB).fromArray(
            Array(10.0f, 20.0f, 60.0f, 80.0f)
          )
        )
        (tAB *! tA) should approxEqual(tA *! tAB)

      it("Division"):
        (tAB /! tA) should approxEqual(
          Tensor.like(tAB).fromArray(
            Array(10.0f, 20.0f, 15.0f, 20.0f)
          )
        )
        (tA /! tAB) should approxEqual(
          Tensor.like(tAB).fromArray(
            Array(0.1f, 0.05f, 0.06666667f, 0.05f)
          )
        )

  describe("Comparison Broadcasting"):

    def bools(values: Boolean*) = Tensor(iAB.shape).fromArray(values.toArray)

    it("Greater (>!)"):
      (iAB >! iA) shouldEqual bools(false, true, true, true)
      (iA >! iAB) shouldEqual bools(false, false, false, false)

    it("Greater or equal (>=!)"):
      (iAB >=! iA) shouldEqual bools(true, true, true, true)
      (iA >=! iAB) shouldEqual bools(true, false, false, false)

    it("Less (`<!`)"):
      (iAB `<!` iA) shouldEqual bools(false, false, false, false)
      (iA `<!` iAB) shouldEqual bools(false, true, true, true)
      iAB.<!(iA) shouldEqual bools(false, false, false, false)

    it("Less or equal (<=!)"):
      (iAB <=! iA) shouldEqual bools(true, false, false, false)
      (iA <=! iAB) shouldEqual bools(true, true, true, true)

    it("Elementwise equality (elementEquals_!)"):
      iAB.elementEquals_!(iA) shouldEqual bools(true, false, false, false)
      iA.elementEquals_!(iAB) shouldEqual bools(true, false, false, false)

    it("Against a scalar"):
      (iAB >! 2) shouldEqual bools(false, false, true, true)
      (2 >! iAB) shouldEqual bools(true, false, false, false)
      (iAB <=! 2) shouldEqual bools(true, true, false, false)
      (iAB `<!` 2) shouldEqual bools(true, false, false, false)
      (2 `<!` iAB) shouldEqual bools(false, false, true, true)

    it("Floats too"):
      (tAB >! tA) shouldEqual bools(true, true, true, true)
      (tA >=! tAB) shouldEqual bools(false, false, false, false)

    it("Nothing to broadcast between equal shapes"):
      "iAB >! iAB" shouldNot compile
      "iAB <=! iAB" shouldNot compile
      "iAB `<!` iAB" shouldNot compile

  describe("Tensor-to-Tensor Broadcasting (complex)"):

    val tABCD = Tensor(Shape(Axis[A] -> 2, Axis[B] -> 2, Axis[C] -> 2, Axis[D] -> 2)).fromArray(
      Array.range(1, 17).map(_.toFloat)
    )

    it("AB broadcastTo ABCD"):
      val AB = Tensor(Shape(Axis[A] -> 2, Axis[B] -> 2)).fromArray(
        Array.range(1, 5).map(_.toFloat)
      )
      val res = AB.broadcastTo(tABCD.shape)
      res.shape shouldEqual tABCD.shape
      res.slice((Axis[C].at(0), Axis[D].at(0))) should approxEqual(AB)
      res.slice((Axis[C].at(1), Axis[D].at(0))) should approxEqual(AB)
      res.slice((Axis[C].at(0), Axis[D].at(1))) should approxEqual(AB)
      res.slice((Axis[C].at(1), Axis[D].at(1))) should approxEqual(AB)

    it("BC broadcastTo ABCD"):
      val BC = Tensor(Shape(Axis[B] -> 2, Axis[C] -> 2)).fromArray(
        Array.range(1, 5).map(_.toFloat)
      )
      val res = BC.broadcastTo(tABCD.shape)
      res.shape shouldEqual tABCD.shape
      res.slice((Axis[A].at(0), Axis[D].at(0))) should approxEqual(BC)
      res.slice((Axis[A].at(1), Axis[D].at(0))) should approxEqual(BC)
      res.slice((Axis[A].at(0), Axis[D].at(1))) should approxEqual(BC)
      res.slice((Axis[A].at(1), Axis[D].at(1))) should approxEqual(BC)

    it("CD broadcastTo ABCD"):
      val CD = Tensor(Shape(Axis[C] -> 2, Axis[D] -> 2)).fromArray(
        Array.range(1, 5).map(_.toFloat)
      )
      val res = CD.broadcastTo(tABCD.shape)
      res.shape shouldEqual tABCD.shape
      res.slice((Axis[A].at(0), Axis[B].at(0))) should approxEqual(CD)
      res.slice((Axis[A].at(1), Axis[B].at(0))) should approxEqual(CD)
      res.slice((Axis[A].at(0), Axis[B].at(1))) should approxEqual(CD)
      res.slice((Axis[A].at(1), Axis[B].at(1))) should approxEqual(CD)

  describe("Disallow"):

    val tABCD = Tensor(Shape(Axis[A] -> 2, Axis[B] -> 2, Axis[C] -> 2, Axis[D] -> 2)).fromArray(
      Array.range(1, 17).map(_.toFloat)
    )

    it("Broadcasting same tensor"):
      "tAB +! tAB" shouldNot compile
      "tAB + tAB" should compile

    it("Shape broadcasting"):
      // JAX allows this, but we disallow it as it often hides bugs
      // dimwit broadcasting only adds missing axes, never changes shapes of existing axes
      val tAB1 = tA.appendAxis(Axis[B])
      an[IllegalArgumentException] should be thrownBy (tAB1 +! tABCD)

  describe("Operator Precedence"):

    it("multiplication (*!) binds tighter than addition (+!)"):
      val tA = Tensor(Shape1(tAB.shape.extent(Axis[A]))).fill(1f)
      val res = tAB *! Tensor0(2.0f) +! tA
      val correct = (tAB *! Tensor0(2.0f)) +! tA
      val wrong = tAB *! (Tensor0(2.0f) +! tA)
      res should approxEqual(correct)
      res shouldNot approxEqual(wrong)

  describe("Mixed Broadcasting Cases"):

    it("Broadcasting ab + bc to abc"):
      val ab = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))
      val bc = Tensor2(Axis[B], Axis[C]).fromArray(Array(Array(10.0f), Array(20.0f)))
      "ab +! bc" shouldNot compile // TODO add support for this

  describe("Function forms (Tensor.op_!(t1, t2) is t1 op! t2)"):
    val bAB = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(true, false), Array(false, true)))
    val bA = Tensor1(Axis[A]).fromArray(Array(true, false))

    it("arithmetic"):
      Tensor.add_!(tAB, tA) shouldEqual (tAB +! tA)
      Tensor.subtract_!(tA, tAB) shouldEqual (tA -! tAB)
      Tensor.multiply_!(tAB, tA) shouldEqual (tAB *! tA)
      Tensor.divide_!(tAB, tA) shouldEqual (tAB /! tA)
      Tensor.mod_!(iAB, iA) shouldEqual (iAB %! iA)

    it("comparisons"):
      Tensor.less_!(tAB, tA) shouldEqual (tAB `<!` tA)
      Tensor.lessEqual_!(tAB, tA) shouldEqual (tAB <=! tA)
      Tensor.greater_!(tAB, tA) shouldEqual (tAB >! tA)
      Tensor.greaterEqual_!(tAB, tA) shouldEqual (tAB >=! tA)
      Tensor.equal_!(iAB, iA) shouldEqual iAB.elementEquals_!(iA)

    it("logical"):
      Tensor.logicalAnd_!(bAB, bA) shouldEqual (bAB and_! bA)
      Tensor.logicalOr_!(bAB, bA) shouldEqual (bAB or_! bA)
      Tensor.logicalXor_!(bAB, bA) shouldEqual (bAB xor_! bA)

    it("maximum and minimum"):
      val tA25 = Tensor1(Axis[A]).fromArray(Array(15.0f, 35.0f))
      maximum_!(tAB, tA25) shouldEqual Tensor.like(tAB).fromArray(Array(15.0f, 20.0f, 35.0f, 40.0f))
      minimum_!(tA25, tAB) shouldEqual Tensor.like(tAB).fromArray(Array(10.0f, 15.0f, 30.0f, 35.0f))

    it("scale"):
      Tensor.scale(tAB, Tensor0(2.0f)) shouldEqual tAB.scale(Tensor0(2.0f))

    it("approxEquals and approxElementEquals"):
      val nearA = Tensor1(Axis[A]).fromArray(Array(10.0000001f, 40.0000001f))
      tAB.approxElementEquals_!(nearA) shouldEqual Tensor(tAB.shape).fromArray(Array(true, false, false, true))
      Tensor.approxElementEquals_!(tAB, nearA) shouldEqual tAB.approxElementEquals_!(nearA)
      tAB.approxEquals_!(nearA).item shouldBe false
      Tensor.like(tAB).fill(3.0f).approxEquals_!(Tensor0(3.0000001f)).item shouldBe true
      Tensor.approxEquals_!(tAB, Tensor0(10.0f)) shouldEqual tAB.approxEquals_!(Tensor0(10.0f))

  describe("No implicit broadcasting without !"):
    val tAB22 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f)))
    val tAB12 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(10.0f, 20.0f)))

    it("same labels but different extents fail fast"):
      an[IllegalArgumentException] should be thrownBy (tAB22 + tAB12)
      an[IllegalArgumentException] should be thrownBy (tAB22 / tAB12)
      an[IllegalArgumentException] should be thrownBy (tAB22 < tAB12)
      an[IllegalArgumentException] should be thrownBy (tAB22 === tAB12)
      an[IllegalArgumentException] should be thrownBy maximum(tAB22, tAB12)
      an[IllegalArgumentException] should be thrownBy where(tAB22 > tAB22, tAB22, tAB12)
      an[IllegalArgumentException] should be thrownBy tAB22.approxElementEquals(tAB12)

    it("=== and ===!"):
      (tAB === tAB).item shouldBe true
      Tensor.arrayEqual(tAB, tAB) shouldEqual (tAB === tAB)
      (Tensor.like(tAB).fill(3.0f) ===! Tensor0(3.0f)).item shouldBe true
      (tAB ===! tA).item shouldBe false
      Tensor.arrayEqual_!(tAB, tA) shouldEqual (tAB ===! tA)

  describe("Scalar first"):
    it("computes scalar op tensor, not tensor op scalar"):
      (10 -! iAB) shouldEqual (Tensor0(10) -! iAB)
      (10 % Tensor0(3)) shouldEqual Tensor0(1)
      (10 %! iAB) shouldEqual (Tensor0(10) %! iAB)
      (2.0 /! tAB) shouldEqual (Tensor0(2.0f) /! tAB)
      (25.0 `<!` tAB) shouldEqual (Tensor0(25.0f) `<!` tAB)

    it("takes the precision of the tensor"):
      (2.5 *! tAB).dtype shouldBe DType.Float32
      (2.5 * Tensor0(2.0f)).dtype shouldBe DType.Float32
      (2L +! iAB).dtype shouldBe DType.Int32
