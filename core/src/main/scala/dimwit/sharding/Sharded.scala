package dimwit.sharding

import dimwit.tensor.Label

/** Marks the data axis `A` as sharded over the mesh axis `M`, as in `Tensor2[Batch |@| X, Feature, Float32]`.
  *
  * `Batch |@| X` is just another axis label, so every operation applies to a sharded tensor
  * unchanged: reducing it all-reduces, reducing or mapping any other axis stays local, and it
  * cannot be combined with an unsharded `Batch` because the names differ.
  *
  * Spelled `|@|` alongside [[dimwit.|*|]] and [[dimwit.|+|]]; a bare `@` is Scala's annotation syntax.
  */
infix trait |@|[A, M]

object `|@|`:
  given [A, M](using label: Label[A], meshLabel: MeshLabel[M]): Label[A |@| M] with
    val name: String = s"${label.name}@${meshLabel.name}"
