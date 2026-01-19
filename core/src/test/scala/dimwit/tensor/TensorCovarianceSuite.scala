package dimwit.tensor

import dimwit.*
import dimwit.Conversions.given
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import scala.collection.View.Empty

class TensorCovarianceSuite extends AnyFunSpec with Matchers:

  it("Shape type hierarchy example: Concrete function with supertype parameter"):
    trait Parent derives Label
    trait Child1 extends Parent derives Label
    trait Child2 extends Parent derives Label
    trait NoChild derives Label
    def concreteFunction(t: Tensor1[Parent, Float]): Tensor1[Parent, Float] = t + t
    val child1: Tensor1[Child1, Float] = Tensor(Shape1(Axis[Child1] -> 4)).fill(1f)
    val child2: Tensor1[Child2, Float] = Tensor(Shape1(Axis[Child2] -> 4)).fill(1f)
    val noChild: Tensor1[NoChild, Float] = Tensor(Shape1(Axis[NoChild] -> 4)).fill(1f)

    "concreteFunction(child1)" should compile
    "concreteFunction(child2)" should compile
    "concreteFunction(noChild)" shouldNot compile

  it("Shape type hierarchy example: Generic function with upper-bounded type parameter"):
    trait Parent derives Label
    trait Child1 extends Parent derives Label
    trait Child2 extends Parent derives Label
    def genericFunction[T <: Parent: Label](t: Tensor1[T, Float]): Tensor1[T, Float] = t + t
    val child1: Tensor1[Child1, Float] = Tensor(Shape1(Axis[Child1] -> 4)).fill(1f)
    val child2: Tensor1[Child2, Float] = Tensor(Shape1(Axis[Child2] -> 4)).fill(1f)

    "genericFunction(child1)" should compile
    "genericFunction(child2)" should compile
    "genericFunction(noChild)" shouldNot compile

  describe("Binary-Ops on tensor of different children types must be mapped to common parent type"):
    trait Parent derives Label
    trait Child1 extends Parent derives Label
    trait Child2 extends Parent derives Label

    val child1: Tensor1[Child1, Float] = Tensor(Shape1(Axis[Child1] -> 4)).fill(1f)
    val child2: Tensor1[Child2, Float] = Tensor(Shape1(Axis[Child2] -> 4)).fill(1f)

    it("+"):
      val additionParent = child1 + child2
      additionParent shouldBe a[Tensor1[Parent, Float]]
      additionParent.axes shouldBe List("Parent")

    it("-"):
      val subtractionParent = child1 - child2
      subtractionParent shouldBe a[Tensor1[Parent, Float]]
      subtractionParent.axes shouldBe List("Parent")

    it("*"):
      val multiplicationParent = child1 * child2
      multiplicationParent shouldBe a[Tensor1[Parent, Float]]
      multiplicationParent.axes shouldBe List("Parent")

    it("/"):
      val divisionParent = child1 / child2
      divisionParent shouldBe a[Tensor1[Parent, Float]]
      divisionParent.axes shouldBe List("Parent")

    it("maximum"):
      val maximumParent = maximum(child1, child2)
      maximumParent.shouldBe(a[Tensor1[Parent, Float]])
      maximumParent.axes.shouldBe(List("Parent"))

    it("minimum"):
      val minimumParent = minimum(child1, child2)
      minimumParent.shouldBe(a[Tensor1[Parent, Float]])
      minimumParent.axes.shouldBe(List("Parent"))

    it("where"):
      val mask = Tensor(Shape(Axis[Parent] -> 4)).fill(true)
      val whereParent = where(mask, child1, child2)
      whereParent shouldBe a[Tensor1[Parent, Float]]
      whereParent.axes shouldBe List("Parent")

    describe("exclude comparision operations as comparison does not make sense between different child types"):
      it("<"):
        "child1 < child1" should compile
        "child1 < child2" shouldNot compile
      it("<="):
        "child1 <= child1" should compile
        "child1 <= child2" shouldNot compile
      it(">"):
        "child1 > child1" should compile
        "child1 > child2" shouldNot compile
      it(">="):
        "child1 >= child1" should compile
        "child1 >= child2" shouldNot compile
      it("elementEquals"):
        "child1.elementEquals(child1)" should compile
        "child1.elementEquals(child2)" shouldNot compile
      it("approxElementEquals"):
        "child1.approxElementEquals(child1)" should compile
        "child1.approxElementEquals(child2)" shouldNot compile

    it("check: child + parent => parent"):
      val parent: Tensor1[Parent, Float] = Tensor(Shape1(Axis[Parent] -> 4)).fill(1f)
      val additionParent = child1 + parent
      additionParent shouldBe a[Tensor1[Parent, Float]]
      additionParent.axes shouldBe List("Parent")

  it("Value-types example: Logits cannot be added to Probabilities"):
    trait Classes derives Label

    object MLContext:
      opaque type Logit = Float
      opaque type Prob = Float

      def createLogits[L: Label](s: Shape1[L]): Tensor1[L, Logit] = Tensor(s).fill(0f)
      def createProbs[L: Label](s: Shape1[L]): Tensor1[L, Prob] = Tensor(s).fill(0f)

      // Operation restricted only to Logit 'land'
      def combineLogits[L: Label](a: Tensor1[L, Logit], b: Tensor1[L, Logit]): Tensor1[L, Logit] = a + b
      def combineProbs[L: Label](a: Tensor1[L, Prob], b: Tensor1[L, Prob]): Tensor1[L, Prob] = a * b
      def toProbs[L: Label](logits: Tensor1[L, Logit]): Tensor1[L, Prob] = logits.vmap(Axis[L]) { l => 1.0f / (1.0f + -l.exp) }

    val shape = Shape1(Axis[Classes] -> 10)
    val logits = MLContext.createLogits(shape)
    val probs = MLContext.createProbs(shape)
    val rawFloats = Tensor(shape).fill(1f)

    "MLContext.combineLogits(logits, logits)" should compile
    "MLContext.combineProbs(probs, probs)" should compile
    "MLContext.combineLogits(logits, probs)" shouldNot compile
    "MLContext.combineProbs(logits, probs)" shouldNot compile
    "MLContext.combineLogits(logits, rawFloats)" shouldNot compile
    "MLContext.combineProbs(probs, rawFloats)" shouldNot compile
