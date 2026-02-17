package dimwit.sharding

import dimwit.*
import dimwit.tensor.TupleHelpers
import dimwit.jax.Jax
import me.shadaj.scalapy.py

case class Mesh[T <: Tuple](shape: Shape[T], devices: Seq[Device]):
  require(shape.size == devices.size, s"Mesh size ${shape.size} must match number of devices ${devices.size}")

object Mesh:
  def apply[T <: Tuple](shape: Shape[T]): Mesh[T] =
    new Mesh(shape, Jax.devices)

trait Sharding
case class NamedSharding[T <: Tuple, T2 <: Tuple](mesh: Mesh[T], axes: Axes[T2])(using TupleHelpers.Subset[T, T2]) extends Sharding

object Axes:
  def apply[T <: Tuple]: Axes[T] = new AxesImpl[T]()

sealed trait Axes[T]
class AxesImpl[T] extends Axes[T]

case class Device private[dimwit] (private[dimwit] val jaxDevice: py.Dynamic):

  def toJaxDevice: py.Dynamic = jaxDevice

  // def addressableMemories: Seq[Memory] = jaxDevice.addressable_memories
  // def client: PyClient = jaxDevice.client
  // def defaultMemory: Memory = jaxDevice.default_memory
  def deviceKind: String = jaxDevice.device_kind.as[String]
  // def getStreamForExternalReadyEvents: ReadyStream = jaxDevice.get_stream_for_external_ready_events
  def hostId: Int = jaxDevice.host_id.as[Int]
  def id: Int = jaxDevice.id.as[Int]
  // def liveBuffers: Seq[Buffer] = jaxDevice.live_buffers
  def localHardwareId: Int = jaxDevice.local_hardware_id.as[Int]
  // def memory: Seq[Memory] = jaxDevice.memory
  // def memoryStats: Map[String, MemoryStats] = jaxDevice.memory_stats
  def platform: String = jaxDevice.platform.as[String]
  def processIndex: Int = jaxDevice.process_index.as[Int]
