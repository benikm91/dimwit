package dimwit.tensor

import dimwit.*

class TensorOpsAlongAxisSuite extends DimwitTest:

  val t2 = Tensor2(
    Axis[A],
    Axis[B]
  ).fromArray(
    Array(
      Array(1.0f, 2.0f, 3.0f),
      Array(4.0f, 5.0f, 6.0f)
    )
  )

  val unsorted = Tensor2(
    Axis[A],
    Axis[B]
  ).fromArray(
    Array(
      Array(1.0f, 3.0f, 2.0f),
      Array(4.0f, 0.0f, 6.0f)
    )
  )

  describe("Along Axis Ops"):
    it("argsort"):
      withClue("JAX default to last axis is not supported."):
        "t2.argsort" shouldNot compile

    it("argsort axis A"):
      val res = t2.argsort(axis = Axis[A])
      res shouldEqual Tensor2(
        Axis[A],
        Axis[B]
      ).fromArray(
        Array(
          Array(0, 0, 0),
          Array(1, 1, 1)
        )
      )

    it("argsort axis B"):
      val res = t2.argsort(axis = Axis[B])
      res shouldEqual Tensor2(
        Axis[A],
        Axis[B]
      ).fromArray(
        Array(
          Array(0, 1, 2),
          Array(0, 1, 2)
        )
      )

    it("sort"):
      val descendingAlongB = Tensor2(Axis[A], Axis[B]).fromArray(
        Array(
          Array(3.0f, 2.0f, 1.0f),
          Array(6.0f, 5.0f, 4.0f)
        )
      )
      withClue("JAX default to last axis is not supported."):
        "descendingAlongB.sort" shouldNot compile

    it("sort axis A"):
      val descendingAlongA = Tensor2(Axis[A], Axis[B]).fromArray(
        Array(
          Array(4.0f, 5.0f, 6.0f),
          Array(1.0f, 2.0f, 3.0f)
        )
      )
      val res = descendingAlongA.sort(axis = Axis[A])
      res shouldEqual Tensor2(
        Axis[A],
        Axis[B]
      ).fromArray(
        Array(
          Array(1.0f, 2.0f, 3.0f),
          Array(4.0f, 5.0f, 6.0f)
        )
      )

    it("sort axis B"):
      val descendingAlongB = Tensor2(Axis[A], Axis[B]).fromArray(
        Array(
          Array(3.0f, 2.0f, 1.0f),
          Array(6.0f, 5.0f, 4.0f)
        )
      )
      val res = descendingAlongB.sort(axis = Axis[B])
      res shouldEqual Tensor2(
        Axis[A],
        Axis[B]
      ).fromArray(
        Array(
          Array(1.0f, 2.0f, 3.0f),
          Array(4.0f, 5.0f, 6.0f)
        )
      )

    it("cumsum"):
      withClue("JAX default to flatten tensor for cumsum without axis is not supported."):
        "t2.cumsum" shouldNot compile

    it("cumsum axis B"):
      val res = t2.cumsum(axis = Axis[B])
      res shouldEqual Tensor.like(res).fromArray(Array(1.0f, 3.0f, 6.0f, 4.0f, 9.0f, 15.0f))

    it("cumsum axis A"):
      val res = t2.cumsum(axis = Axis[A])
      res shouldEqual Tensor.like(res).fromArray(Array(1.0f, 2.0f, 3.0f, 5.0f, 7.0f, 9.0f))

    it("cumprod"):
      withClue("JAX default to flatten tensor for cumprod without axis is not supported."):
        "t2.cumprod" shouldNot compile

    it("cumprod axis B"):
      val res = t2.cumprod(axis = Axis[B])
      res shouldEqual Tensor.like(res).fromArray(Array(1.0f, 2.0f, 6.0f, 4.0f, 20.0f, 120.0f))

    it("diff"):
      withClue("JAX default to last axis is not supported."):
        "t2.diff" shouldNot compile

    it("diff axis B"):
      val res = t2.diff(axis = Axis[B])
      res shouldEqual Tensor.like(res).fromArray(Array(1.0f, 1.0f, 1.0f, 1.0f))

    it("diff axis A"):
      val res = t2.diff(axis = Axis[A])
      res shouldEqual Tensor.like(res).fromArray(Array(3.0f, 3.0f, 3.0f))

    it("cummax"):
      withClue("JAX default to first axis is not supported."):
        "unsorted.cummax" shouldNot compile

    it("cummax axis B"):
      val res = unsorted.cummax(axis = Axis[B])
      res shouldEqual Tensor.like(res).fromArray(Array(1.0f, 3.0f, 3.0f, 4.0f, 4.0f, 6.0f))

    it("cummax axis A"):
      val res = unsorted.cummax(axis = Axis[A])
      res shouldEqual Tensor.like(res).fromArray(Array(1.0f, 3.0f, 2.0f, 4.0f, 3.0f, 6.0f))

    it("cummin"):
      withClue("JAX default to first axis is not supported."):
        "unsorted.cummin" shouldNot compile

    it("cummin axis B"):
      val res = unsorted.cummin(axis = Axis[B])
      res shouldEqual Tensor.like(res).fromArray(Array(1.0f, 1.0f, 1.0f, 4.0f, 0.0f, 0.0f))

    it("cummin axis A"):
      val res = unsorted.cummin(axis = Axis[A])
      res shouldEqual Tensor.like(res).fromArray(Array(1.0f, 3.0f, 2.0f, 1.0f, 0.0f, 2.0f))

    it("logcumsumexp"):
      withClue("JAX default to first axis is not supported."):
        "unsorted.logcumsumexp" shouldNot compile

    it("logcumsumexp axis B"):
      unsorted.logcumsumexp(axis = Axis[B]) should approxEqual(unsorted.exp.cumsum(Axis[B]).log, 1e-5f)

    it("logcumsumexp axis A"):
      unsorted.logcumsumexp(axis = Axis[A]) should approxEqual(unsorted.exp.cumsum(Axis[A]).log, 1e-5f)

    it("Tensor1 functions lift with vapply"):
      unsorted.vapply(Axis[B])(Tensor1.cummax) shouldEqual unsorted.cummax(Axis[B])

    it("softmax"):
      withClue("JAX default to last axis is not supported."):
        "unsorted.softmax" shouldNot compile

    it("softmax axis B"):
      val res = unsorted.softmax(axis = Axis[B])
      res.sum(Axis[B]) should approxEqual(Tensor.like(res.sum(Axis[B])).fill(1.0f), 1e-5f)
      res should approxEqual(unsorted.exp /! unsorted.exp.sum(Axis[B]), 1e-5f)

    it("softmax axis A"):
      val res = unsorted.softmax(axis = Axis[A])
      res should approxEqual(unsorted.exp /! unsorted.exp.sum(Axis[A]), 1e-5f)

    it("logSoftmax"):
      withClue("JAX default to last axis is not supported."):
        "unsorted.logSoftmax" shouldNot compile

    it("logSoftmax axis B"):
      unsorted.logSoftmax(axis = Axis[B]) should approxEqual(unsorted.softmax(Axis[B]).log, 1e-5f)

    it("roll"):
      withClue("JAX default to flatten tensor for roll without axis is not supported."):
        "unsorted.roll(shift = 1)" shouldNot compile

    it("roll with Tensor1.roll through vapply"):
      unsorted.vapply(Axis[B])(Tensor1.roll(1)) shouldEqual unsorted.roll(Axis[B], shift = 1)

  describe("Function forms (Tensor.op(t, axis) is t.op(axis))"):
    it("numeric ops"):
      Tensor.argsort(unsorted, Axis[B]) shouldEqual unsorted.argsort(Axis[B])
      Tensor.sort(unsorted, Axis[A]) shouldEqual unsorted.sort(Axis[A])
      Tensor.cumsum(unsorted, Axis[B]) shouldEqual unsorted.cumsum(Axis[B])
      Tensor.cumprod(unsorted, Axis[B]) shouldEqual unsorted.cumprod(Axis[B])
      Tensor.cummax(unsorted, Axis[B]) shouldEqual unsorted.cummax(Axis[B])
      Tensor.cummin(unsorted, Axis[A]) shouldEqual unsorted.cummin(Axis[A])
      Tensor.diff(unsorted, Axis[B]) shouldEqual unsorted.diff(Axis[B])
      Tensor.roll(unsorted, Axis[B], shift = 1) shouldEqual unsorted.roll(Axis[B], shift = 1)

    it("floating ops"):
      Tensor.logcumsumexp(unsorted, Axis[B]) shouldEqual unsorted.logcumsumexp(Axis[B])
      Tensor.softmax(unsorted, Axis[B]) shouldEqual unsorted.softmax(Axis[B])
      Tensor.logSoftmax(unsorted, Axis[A]) shouldEqual unsorted.logSoftmax(Axis[A])
