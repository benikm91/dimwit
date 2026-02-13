package dimwit

import me.shadaj.scalapy.py
import dimwit.autodiff.ToPyTree

private[dimwit] object MemoryHelper:

  def withLocalCleanup(f: => Unit): Unit =
    py.local:
      f

  def withLocalCleanup[A: ToPyTree](f: => A): A =
    val lifeRaft = me.shadaj.scalapy.py.Dynamic.global.list()
    py.local:
      val res = f
      val pyRes = ToPyTree[A].toPyTree(res)
      lifeRaft.append(pyRes)
    val res = lifeRaft.pop()
    ToPyTree[A].fromPyTree(res)
