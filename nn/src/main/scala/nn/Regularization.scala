package nn

import dimwit.*

case class WeightDecay(decayFactor: Tensor0[Float]):

  def apply[Params: ToPyTree: FloatTensorTree](params: Params): Params =
    FloatTensorTree[Params].map(
      params,
      [T <: Tuple] =>
        (n: Labels[T]) ?=>
          (p: Tensor[T, Float]) =>
            p - (p *! decayFactor)
    )
