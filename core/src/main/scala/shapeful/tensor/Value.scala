package shapeful.tensor

trait ExecutionType[V]:
  def dtype: DType

object ExecutionType:

  given floatValue: ExecutionType[Float] with
    def dtype: DType = DType.Float32
  given intValue: ExecutionType[Int] with
    def dtype: DType = DType.Int32
  given booleanValue: ExecutionType[Boolean] with
    def dtype: DType = DType.Bool

object Of:
  def apply[V](tensor: Tensor[?, V]): Of[V] = new OfImpl[V](tensor.dtype)
  def apply[A: ExecutionType]: Of[A] = new OfImpl[A](summon[ExecutionType[A]].dtype)
  
sealed trait Of[A]:
  def dtype: DType

class OfImpl[A](val dtype: DType) extends Of[A]
