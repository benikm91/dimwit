package dimwit.examples.basic

import dimwit.*
import dimwit.autodiff.*
import dimwit.examples.basic.SIRSimulation.*
import dimwit.optimizer.Adam
import dimwit.random.Random
import dimwit.stats.Normal

/** Calibrates the SIR model of [[SIRSimulation]] to an observed epidemic.
  *
  * It finds the infection and recovery rates for which the simulated number of infectious
  * individuals matches the observed one. Since the simulation is a `scan`, it can be
  * differentiated, so the rates are fitted by gradient descent through the whole simulation.
  */
object SIRCalibration:

  /** The unknown parameters, on a log scale so that the rates stay positive
    *
    * @param logInfectionScale The log of the factor by which all infection rates are scaled
    * @param logGamma The log of the recovery rate
    */
  case class Params(logInfectionScale: Tensor0[Float32], logGamma: Tensor0[Float32])

  /** The number of infectious individuals after each time step, simulated with the parameters */
  def infectiousOverTime(
      initial: SIRState,
      beta: Tensor2[Group, Prime[Group], Float32],
      dt: Tensor0[Float32],
      contact: Tensor1[Time, Float32]
  )(params: Params): Tensor1[Time, Float32] =
    val (_, infectious) = simulate(initial, beta *! params.logInfectionScale.exp, params.logGamma.exp, dt, contact)
    infectious.sum(Axis[Group])

  /** The mean squared difference between the simulated and the observed fraction of infectious individuals */
  def loss(
      initial: SIRState,
      beta: Tensor2[Group, Prime[Group], Float32],
      dt: Tensor0[Float32],
      contact: Tensor1[Time, Float32],
      observed: Tensor1[Time, Float32]
  )(params: Params): Tensor0[Float32] =
    val population = initial.N.sum
    ((infectiousOverTime(initial, beta, dt, contact)(params) - observed) /! population).pow(2f).mean

  @main def runSIRCalibration(): Unit =
    dimwit.initialize()

    val initial = exampleInitialState()
    val beta = exampleBeta()
    val dt = Tensor0(0.1f)
    val nSteps = 300
    val contact = Tensor(Shape(Axis[Time] -> nSteps)).fill(1f)

    // Observe an epidemic with known rates, with some noise

    val trueParams = Params(logInfectionScale = Tensor0(0f), logGamma = Tensor0(0.1f).log)
    val noise = Normal.standardNormal(Shape(Axis[Time] -> nSteps)).sample(Random.Key(0)) *! 20f
    val observed = infectiousOverTime(initial, beta, dt, contact)(trueParams) + noise

    // Fit the rates, starting from a wrong guess

    val initialGuess = Params(logInfectionScale = Tensor0(0.5f).log, logGamma = Tensor0(0.2f).log)
    val trainLoss = jit(loss(initial, beta, dt, contact, observed))
    val adam = Adam(learningRate = 0.05f)

    def report(params: Params): String =
      f"infection scale = ${params.logInfectionScale.exp.item}%.3f, gamma = ${params.logGamma.exp.item}%.3f"

    val fitted = adam
      .iterate(initialGuess)(grad(trainLoss))
      .zipWithIndex
      .tapEach:
        case (params, index) =>
          if index % 25 == 0 then println(f"step $index%3d: loss = ${trainLoss(params).item}%.2e, ${report(params)}")
      .map(_._1)
      .drop(150)
      .next()

    println(s"fitted: ${report(fitted)}")
    println(s"true:   ${report(trueParams)}")
