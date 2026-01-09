package dimwit.tensor

import scala.annotation.targetName
import scala.compiletime.{erasedValue, summonFrom}
import dimwit.jax.Jax
import dimwit.jax.JaxDType
import dimwit.jax.Jax.PyDynamic
import dimwit.tensor.{Label, Labels, ExecutionType, VType}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import dimwit.random.Random
import dimwit.stats.{Normal, Uniform}
import me.shadaj.scalapy.readwrite.Writer
import scala.reflect.ClassTag
import scala.annotation.unchecked.uncheckedVariance
import java.nio.ByteBuffer
import java.util.Base64
import java.nio.ByteOrder

enum Device(val platform: String):
  case CPU extends Device("cpu")
  case GPU extends Device("gpu")
  case Other extends Device("other")

object Device:
  val default: Device = Device.CPU
  extension (device: Device)
    def toJaxDevice: Jax.PyDynamic =
      val devices = Jax.devices(device.platform)
      require(devices.nonEmpty, s"No JAX devices found for platform: ${device.platform}")
      devices.head

class Tensor[+T <: Tuple: Labels, V] private[tensor] (
    val jaxValue: Jax.PyDynamic
):

  lazy val axes: List[String] = shape.labels
  lazy val dtype: DType = JaxDType.fromJaxDtype(jaxValue.dtype)
  lazy val shape: Shape[T] = Shape.fromSeq[T](jaxValue.shape.as[Seq[Int]])
  lazy val vtype: VType[V] = VType(this)

  lazy val device: Device =
    val jaxDevice = Jax.device_get(jaxValue)
    jaxDevice.platform.as[String] match
      case "cpu" => Device.CPU
      case "gpu" => Device.GPU
      case _     => Device.Other

  def asType[V2](vtype: VType[V2]): Tensor[T, V2] = new Tensor(Jax.jnp.astype(jaxValue, JaxDType.jaxDtype(vtype.dtype)))

  def toDevice(newDevice: Device): Tensor[T, V] = new Tensor(jaxValue = Jax.device_put(jaxValue, newDevice.toJaxDevice))

  override def equals(other: Any): Boolean =
    other match
      case that: Tensor[?, ?] => Jax.jnp.array_equal(this.jaxValue, that.jaxValue).item().as[Boolean]
      case _                  => false

  override def hashCode(): Int = jaxValue.block_until_ready().tobytes().hashCode()

  override def toString: String =
    jaxTypeName match
      case Jax.ArrayTypeName =>
        jaxValue.block_until_ready().toString()
      case Jax.BatchTracerName =>
        s"TracerTensor(${shape.toString})"
      case _ => jaxValue.toString()

  def dim[L](axis: Axis[L])(using axisIndex: AxisIndex[T @uncheckedVariance, L]): Dim[L] =
    shape.dim(axis)

  private val jaxTypeName: String = py.Dynamic.global.`type`(jaxValue).`__name__`.as[String]

object Tensor:

  type IndicesOf[T <: Tuple] = Tuple.Map[T, [_] =>> Int]

  def apply[T <: Tuple: Labels, V](jaxValue: Jax.PyDynamic): Tensor[T, V] = new Tensor(jaxValue)
  def randn[T <: Tuple: Labels](shape: Shape[T])(key: Random.Key)(using
      executionType: ExecutionType[Float]
  ): Tensor[T, Float] = Normal.standardNormal(shape).sample(key)

  def fromPy[T <: Tuple: Labels, V](vtype: VType[V])(jaxValue: Jax.PyDynamic): Tensor[T, V] = new Tensor(jaxValue)
  def zeros[T <: Tuple: Labels, V](shape: Shape[T], vtype: VType[V]): Tensor[T, V] = Tensor(Jax.jnp.zeros(shape.dimensions.toPythonProxy, dtype = vtype.dtype.jaxType))
  def ones[T <: Tuple: Labels, V](shape: Shape[T], vtype: VType[V]): Tensor[T, V] = Tensor(Jax.jnp.ones(shape.dimensions.toPythonProxy, dtype = vtype.dtype.jaxType))
  def const[T <: Tuple: Labels, V](shape: Shape[T], vtype: VType[V])(value: V)(using writer: Writer[V]): Tensor[T, V] = Tensor(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = vtype.dtype.jaxType))

  def fromArray[T <: Tuple: Labels](shape: Shape[T], vtype: VType[Float])(values: Array[Float]): Tensor[T, Float] = fromFloatArray(shape)(values)
  def fromArray[T <: Tuple: Labels](shape: Shape[T], vtype: VType[Int])(values: Array[Int]): Tensor[T, Int] = fromIntArray(shape)(values)
  def fromArray[T <: Tuple: Labels](shape: Shape[T])(values: Array[Byte]): Tensor[T, Int] = fromByteArray(shape)(values)
  def fromArray[T <: Tuple: Labels](shape: Shape[T], vtype: VType[Boolean])(values: Array[Boolean]): Tensor[T, Boolean] = fromBooleanArray(shape)(values)

  /** array.toPythonProxy is very inefficient for large arrays, so we use base64 encoding as a workaround */
  private val base64Loader = py.eval("lambda b64, shape, dtype: __import__('jax').numpy.array(__import__('numpy').frombuffer(__import__('base64').b64decode(b64), dtype=dtype).reshape(shape))")

  def fromFloatArray[T <: Tuple: Labels](shape: Shape[T])(values: Array[Float]): Tensor[T, Float] =
    require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
    val floatArr = values.asInstanceOf[Array[Float]]
    val buffer = ByteBuffer.allocate(floatArr.length * 4)
    buffer.order(ByteOrder.LITTLE_ENDIAN)
    buffer.asFloatBuffer().put(floatArr)
    val b64String = Base64.getEncoder.encodeToString(buffer.array())
    Tensor(base64Loader(b64String, shape.dimensions.toPythonProxy, "float32"))

  def fromIntArray[T <: Tuple: Labels](shape: Shape[T])(values: Array[Int]): Tensor[T, Int] =
    require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
    val intArr = values.asInstanceOf[Array[Int]]
    val buffer = ByteBuffer.allocate(intArr.length * 4)
    buffer.order(ByteOrder.LITTLE_ENDIAN)
    buffer.asIntBuffer().put(intArr)
    val b64String = Base64.getEncoder.encodeToString(buffer.array())
    Tensor(base64Loader(b64String, shape.dimensions.toPythonProxy, "int32"))

  def fromByteArray[T <: Tuple: Labels](shape: Shape[T])(values: Array[Byte]): Tensor[T, Int] =
    require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
    val buffer = ByteBuffer.allocate(values.length)
    buffer.order(ByteOrder.LITTLE_ENDIAN)
    buffer.put(values)
    val b64String = Base64.getEncoder.encodeToString(buffer.array())
    Tensor(base64Loader(b64String, shape.dimensions.toPythonProxy, "uint8"))

  def fromBooleanArray[T <: Tuple: Labels](shape: Shape[T])(values: Array[Boolean]): Tensor[T, Boolean] =
    require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
    val boolArr = values.map(b => if b then 1.toByte else 0.toByte)
    val buffer = ByteBuffer.allocate(boolArr.length)
    buffer.order(ByteOrder.LITTLE_ENDIAN)
    buffer.put(boolArr)
    val b64String = Base64.getEncoder.encodeToString(buffer.array())
    Tensor(base64Loader(b64String, shape.dimensions.toPythonProxy, "bool"))

type Tensor0[V] = Tensor[EmptyTuple, V]
type Tensor1[L, V] = Tensor[Tuple1[L], V]
type Tensor2[L1, L2, V] = Tensor[(L1, L2), V]
type Tensor3[L1, L2, L3, V] = Tensor[(L1, L2, L3), V]
type Tensor4[L1, L2, L3, L4, V] = Tensor[(L1, L2, L3, L4), V]

object Tensor0:

  given float2FloatTensor: Conversion[Float, Tensor0[Float]] = (x: Float) => Tensor0(x)
  given int2IntTensor: Conversion[Int, Tensor0[Int]] = (x: Int) => Tensor0(x)
  given int2FloatTensor: Conversion[Int, Tensor0[Float]] = (x: Int) => Tensor0(x.toFloat)
  given boolean2BooleanTensor: Conversion[Boolean, Tensor0[Boolean]] = (x: Boolean) => Tensor0(x)

  def zero[V](vtype: VType[V]): Tensor0[V] = Tensor.zeros(Shape.empty, vtype)
  def one[V](vtype: VType[V]): Tensor0[V] = Tensor.ones(Shape.empty, vtype)
  def const[V](vtype: VType[V])(value: V)(using writer: Writer[V]): Tensor0[V] = Tensor.const(Shape.empty, vtype)(value)

  def randn(key: Random.Key)(using executionType: ExecutionType[Float]): Tensor0[Float] = Normal.standardNormal(Shape.empty).sample(key)
  def apply[V](jaxValue: Jax.PyDynamic): Tensor0[V] = Tensor(jaxValue)
  def apply[V](value: V)(using sv: ExecutionType[V], writer: Writer[V]): Tensor0[V] = Tensor0.const(VType[V])(value)

object Tensor1:

  def fromArray[L: Label, V](axis: Axis[L], vtype: VType[Float])(values: Array[Float]) =
    val dim = (axis -> values.length)
    Tensor.fromArray(Shape(dim), vtype)(values)
  def fromArray[L: Label, V](axis: Axis[L], vtype: VType[Int])(values: Array[Int]) =
    val dim = (axis -> values.length)
    Tensor.fromArray(Shape(dim), vtype)(values)
  def fromArray[L: Label, V](axis: Axis[L], vtype: VType[Boolean])(values: Array[Boolean]) =
    val dim = (axis -> values.length)
    Tensor.fromArray(Shape(dim), vtype)(values)

object Tensor2:

  def fromArray[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2], vtype: VType[Float])(values: Array[Array[Float]]): Tensor2[L1, L2, Float] =
    val dims = (axis1 -> values.length, axis2 -> values.head.length)
    Tensor.fromArray(Shape(dims), vtype)(values.flatten)
  def fromArray[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2], vtype: VType[Int])(values: Array[Array[Int]]): Tensor2[L1, L2, Int] =
    val dims = (axis1 -> values.length, axis2 -> values.head.length)
    Tensor.fromArray(Shape(dims), vtype)(values.flatten)
  def fromArray[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2], vtype: VType[Boolean])(values: Array[Array[Boolean]]): Tensor2[L1, L2, Boolean] =
    val dims = (axis1 -> values.length, axis2 -> values.head.length)
    Tensor.fromArray(Shape(dims), vtype)(values.flatten)

  def eye[L: Label, V](dim: Dim[L], vtype: VType[V]): Tensor2[L, L, V] = Tensor(Jax.jnp.eye(dim._2, dtype = vtype.dtype.jaxType))
  def diag[L: Label, V](diag: Tensor1[L, V]): Tensor2[L, L, V] = Tensor(Jax.jnp.diag(diag.jaxValue))

object Tensor3:

  def fromArray[L1: Label, L2: Label, L3: Label, V](axis1: Axis[L1], axis2: Axis[L2], axis3: Axis[L3], vtype: VType[Float])(
      values: Array[Array[Array[Float]]]
  ): Tensor3[L1, L2, L3, Float] =
    val dims = (axis1 -> values.length, axis2 -> values.head.length, axis3 -> values.head.head.length)
    Tensor.fromArray(Shape(dims), vtype)(values.flatten.flatten)
  def fromArray[L1: Label, L2: Label, L3: Label, V](axis1: Axis[L1], axis2: Axis[L2], axis3: Axis[L3], vtype: VType[Int])(
      values: Array[Array[Array[Int]]]
  ): Tensor3[L1, L2, L3, Int] =
    val dims = (axis1 -> values.length, axis2 -> values.head.length, axis3 -> values.head.head.length)
    Tensor.fromArray(Shape(dims), vtype)(values.flatten.flatten)
  def fromArray[L1: Label, L2: Label, L3: Label, V](axis1: Axis[L1], axis2: Axis[L2], axis3: Axis[L3], vtype: VType[Boolean])(
      values: Array[Array[Array[Boolean]]]
  ): Tensor3[L1, L2, L3, Boolean] =
    val dims = (axis1 -> values.length, axis2 -> values.head.length, axis3 -> values.head.head.length)
    Tensor.fromArray(Shape(dims), vtype)(values.flatten.flatten)
