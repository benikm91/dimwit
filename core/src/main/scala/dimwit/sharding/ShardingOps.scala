package dimwit.sharding

import dimwit.jax.Jax
import dimwit.tensor.Axis
import dimwit.tensor.Label
import dimwit.tensor.Labels
import dimwit.tensor.ShapeTypeHelpers.AxisReplacer
import dimwit.tensor.Tensor

object ShardingOps:

  extension [T <: Tuple: Labels, V](t: Tensor[T, V])

    /** Splits `t` across `mesh` along one axis, rewriting that axis from `L` to `L |@| A`.
      *
      * The shards are placed one per device with `jax.device_put` under a `NamedSharding`.
      *
      * {{{
      * val mesh = Mesh(MeshAxis[X] -> 4)
      * val sharded: Tensor2[Batch |@| X, Feature, Float32] = t.shard(mesh, Axis[Batch] -> MeshAxis[X])
      * }}}
      *
      * @throws IllegalArgumentException if the extent of `L` is not divisible by the mesh axis size.
      */
    def shard[M <: Tuple, L: Label, A: MeshLabel](mesh: Mesh[M], mapping: (Axis[L], MeshAxis[A]))(using
        replacer: AxisReplacer[T, L, L |@| A],
        meshIndex: MeshAxisIndex[M, A],
        labels: Labels[replacer.NewShape]
    ): Tensor[replacer.NewShape, V] =
      val axisName = summon[Label[L]].name
      val meshAxisName = mapping._2.name
      val extent = t.shape.dimensions(replacer.index)
      val meshSize = mesh.axisSizes(meshIndex.index)
      if extent % meshSize != 0 then
        throw new IllegalArgumentException(
          s"Cannot shard axis $axisName of extent $extent over mesh axis $meshAxisName of size $meshSize: " +
            s"$extent is not divisible by $meshSize."
        )
      val sharding = Jax.jax_helper.named_sharding(mesh.jaxMesh, t.shape.rank, replacer.index, meshAxisName)
      Tensor[replacer.NewShape, V](Jax.device_put(t.jaxValue, sharding.as[Jax.PyDynamic]))

    /** Gathers a sharded axis back across the mesh, rewriting it from `L |@| A` to `L`.
      *
      * Every device ends up holding the whole tensor. Only a sharded axis can be unsharded.
      *
      * {{{
      * val gathered: Tensor2[Batch, Feature, Float32] = sharded.unshard(Axis[Batch |@| X])
      * }}}
      */
    def unshard[L, A](axis: Axis[L |@| A])(using
        replacer: AxisReplacer[T, L |@| A, L],
        labels: Labels[replacer.NewShape]
    ): Tensor[replacer.NewShape, V] =
      Tensor[replacer.NewShape, V](Jax.jax_helper.replicate(t.jaxValue).as[Jax.PyDynamic])
