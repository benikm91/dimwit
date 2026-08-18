package dimwit.tensor

import dimwit.*
import dimwit.tensor.TupleHelpers.Remove

sealed trait X derives Label
sealed trait Y derives Label
sealed trait Z derives Label

/** A module written entirely against abstract labels: it never sees a concrete axis.
  *
  * The shape it returns is a match type, which cannot reduce here - `L3` is not provably distinct
  * from `L1`. It does not have to: the constraint is declared, and the caller reduces it.
  */
object GenericModule:

  def dropLast[L1: Label, L2: Label, L3: Label](t: Tensor[(L1, L2, L3), Float32])(using
      axisIndex: AxisIndex[(L1, L2, L3), L3],
      labels: Labels[Remove[(L1, L2, L3), L3]]
  ): Tensor[Remove[(L1, L2, L3), L3], Float32] = t.sum(Axis[L3])

class GenericModuleSuite extends DimwitTest:

  describe("a generic module returning a match type shape"):

    val t = Tensor(Shape(Axis[X] -> 2, Axis[Y] -> 3, Axis[Z] -> 4)).fill(1f)

    it("drops the axis the module names, keeping the other two in order"):
      val dropped: Tensor[(X, Y), Float32] = GenericModule.dropLast(t)

      dropped.axes shouldBe List("X", "Y")
      dropped.shape(Axis[X]) shouldBe 2
      dropped.shape(Axis[Y]) shouldBe 3

    it("does not return the remaining axes in the wrong order"):
      "val wrong: Tensor[(Y, X), Float32] = GenericModule.dropLast(t)" shouldNot compile

    it("keeps the values with the axes, not just the labels"):
      val counted = Tensor(Shape(Axis[X] -> 2, Axis[Y] -> 3, Axis[Z] -> 4)).fromFunction { idx =>
        idx(Axis[X]).toFloat
      }
      val dropped: Tensor[(X, Y), Float32] = GenericModule.dropLast(counted)

      // summing 4 entries of Z leaves 4 * the X index, independent of Y
      dropped should approxEqual(Tensor2(Axis[X], Axis[Y]).fromArray(
        Array(
          Array(0.0f, 0.0f, 0.0f),
          Array(4.0f, 4.0f, 4.0f)
        )
      ))
