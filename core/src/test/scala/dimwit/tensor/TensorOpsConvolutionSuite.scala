package dimwit.tensor

import dimwit.*
import dimwit.Conversions.given
import dimwit.tensor.TensorOps.Convolution.Padding
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

class TensorOpsConvolutionSuite extends AnyFunSpec with Matchers:

  describe("Convolution 1D"):

    it("should perform 1D convolution with correct output shape"):
      trait Batch derives Label
      trait Length derives Label
      trait InChannels derives Label
      trait OutChannels derives Label
      trait KernelSize derives Label

      val input = Tensor.ones(
        Shape(
          Axis[Batch] -> 2,
          Axis[Length] -> 10,
          Axis[InChannels] -> 3
        ),
        VType[Float]
      )

      val kernel = Tensor.ones(
        Shape(
          Axis[KernelSize] -> 3,
          Axis[InChannels] -> 3,
          Axis[OutChannels] -> 4
        ),
        VType[Float]
      )

      val output = input.conv(Axis[InChannels], Axis[OutChannels])(kernel, stride = 1, padding = Padding.SAME)

      // Output shape should be (batch=2, length=10, out_channels=4)
      output.shape(Axis[Batch]) shouldBe 2
      output.shape(Axis[Length]) shouldBe 10
      output.shape(Axis[OutChannels]) shouldBe 4

    it("should perform 1D convolution with stride > 1"):
      trait Batch derives Label
      trait Length derives Label
      trait InChannels derives Label
      trait OutChannels derives Label
      trait KernelSize derives Label

      val input = Tensor.ones(
        Shape(
          Axis[Batch] -> 1,
          Axis[Length] -> 10,
          Axis[InChannels] -> 1
        ),
        VType[Float]
      )

      val kernel = Tensor.ones(
        Shape(
          Axis[KernelSize] -> 3,
          Axis[InChannels] -> 1,
          Axis[OutChannels] -> 1
        ),
        VType[Float]
      )

      val output = input.conv(Axis[InChannels], Axis[OutChannels])(kernel, stride = 2, padding = Padding.VALID)

      output.shape(Axis[Batch]) shouldBe 1
      output.shape(Axis[OutChannels]) shouldBe 1
      // With VALID padding, output length should be reduced
      output.shape(Axis[Length]) should be < 10

  describe("Convolution 2D"):

    it("should perform 2D convolution with correct output shape"):
      trait Batch derives Label
      trait Height derives Label
      trait Width derives Label
      trait InChannels derives Label
      trait OutChannels derives Label
      trait KernelH derives Label
      trait KernelW derives Label

      // Input: (batch=2, height=8, width=8, in_channels=3)
      val input = Tensor.ones(
        Shape(
          Axis[Batch] -> 2,
          Axis[Height] -> 8,
          Axis[Width] -> 8,
          Axis[InChannels] -> 3
        ),
        VType[Float]
      )

      // Kernel: (kernel_h=3, kernel_w=3, in_channels=3, out_channels=16)
      val kernel = Tensor.ones(
        Shape(
          Axis[KernelH] -> 3,
          Axis[KernelW] -> 3,
          Axis[InChannels] -> 3,
          Axis[OutChannels] -> 16
        ),
        VType[Float]
      )

      val output = input.conv(Axis[InChannels], Axis[OutChannels])(kernel, stride = 1, padding = Padding.SAME)

      // Output shape should be (batch=2, height=8, width=8, out_channels=16)
      output.shape(Axis[Batch]) shouldBe 2
      output.shape(Axis[Height]) shouldBe 8
      output.shape(Axis[Width]) shouldBe 8
      output.shape(Axis[OutChannels]) shouldBe 16

    it("should perform 2D convolution with stride=2"):
      trait Batch derives Label
      trait Height derives Label
      trait Width derives Label
      trait InChannels derives Label
      trait OutChannels derives Label
      trait KernelH derives Label
      trait KernelW derives Label

      val input = Tensor.ones(
        Shape(
          Axis[Batch] -> 1,
          Axis[Height] -> 16,
          Axis[Width] -> 16,
          Axis[InChannels] -> 3
        ),
        VType[Float]
      )

      val kernel = Tensor.ones(
        Shape(
          Axis[KernelH] -> 3,
          Axis[KernelW] -> 3,
          Axis[InChannels] -> 3,
          Axis[OutChannels] -> 8
        ),
        VType[Float]
      )

      val output = input.conv(Axis[InChannels], Axis[OutChannels])(kernel, stride = 2, padding = Padding.SAME)

      output.shape(Axis[Batch]) shouldBe 1
      // With stride=2 and SAME padding, spatial dims should be halved
      output.shape(Axis[Height]) shouldBe 8
      output.shape(Axis[Width]) shouldBe 8
      output.shape(Axis[OutChannels]) shouldBe 8

    it("should compute correct convolution values for 2D case"):
      trait Batch derives Label
      trait Height derives Label
      trait Width derives Label
      trait InChannels derives Label
      trait OutChannels derives Label
      trait KH derives Label
      trait KW derives Label

      // Create a simple 3x3 input with known values (1 batch, 1 channel)
      // Pattern:
      // 1 2 3
      // 4 5 6
      // 7 8 9
      val inputShape = Shape(
        Axis[Batch] -> 1,
        Axis[Height] -> 3,
        Axis[Width] -> 3,
        Axis[InChannels] -> 1
      )
      val inputData = Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f)
      val input = Tensor.fromArray(inputShape, VType[Float])(inputData)

      // Create a 2x2 kernel that sums all values (all ones)
      // When convolved, each output element will be the sum of a 2x2 window
      val kernelShape = Shape(
        Axis[KH] -> 2,
        Axis[KW] -> 2,
        Axis[InChannels] -> 1,
        Axis[OutChannels] -> 1
      )
      val kernelData = Array(1.0f, 1.0f, 1.0f, 1.0f)
      val kernel = Tensor.fromArray(kernelShape, VType[Float])(kernelData)

      val output = input.conv(Axis[InChannels], Axis[OutChannels])(kernel, stride = 1, padding = Padding.VALID)

      // With VALID padding and 2x2 kernel, output should be 2x2
      // Output values:
      // Top-left:    1+2+4+5 = 12
      // Top-right:   2+3+5+6 = 16
      // Bottom-left: 4+5+7+8 = 24
      // Bottom-right: 5+6+8+9 = 28
      output.shape(Axis[Batch]).shouldBe(1)
      output.shape(Axis[Height]).shouldBe(2)
      output.shape(Axis[Width]).shouldBe(2)
      output.shape(Axis[OutChannels]).shouldBe(1)

      val expectedShape = Shape(
        Axis[Batch] -> 1,
        Axis[Height] -> 2,
        Axis[Width] -> 2,
        Axis[OutChannels] -> 1
      )
      val expectedData = Array(12.0f, 16.0f, 24.0f, 28.0f)
      val expected = Tensor.fromArray(expectedShape, VType[Float])(expectedData)

      (output === expected).item.shouldBe(true)

  describe("Convolution validation"):

    it("should reject input with insufficient dimensions"):
      trait A derives Label
      trait B derives Label
      trait Out derives Label

      val input = Tensor.ones(Shape(Axis[A] -> 3, Axis[B] -> 3), VType[Float])
      val kernel = Tensor.ones(Shape(Axis[A] -> 2, Axis[B] -> 2, Axis[Out] -> 1), VType[Float])

      an[IllegalArgumentException] should be thrownBy {
        input.conv(Axis[B], Axis[Out])(kernel)
      }
