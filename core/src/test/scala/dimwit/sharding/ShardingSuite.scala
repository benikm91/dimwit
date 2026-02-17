package dimwit.sharding

import dimwit.*
import dimwit.*
import dimwit.Conversions.given
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec

import dimwit.autodiff.Autodiff.Gradient
import dimwit.jax.Jax

class ShardingSuite extends AnyFunSpec with Matchers:

  trait X derives Label
  trait Y derives Label

  it("devices"):
    val devices = Jax.devices
    val mesh = Mesh(Shape(Axis[X] -> 4, Axis[Y] -> 2))
    val sharding = NamedSharding(mesh, Axes[(X, Y)])
    println(sharding)
    true shouldBe true
