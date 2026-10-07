package dimwit.examples.basic

import dimwit.*

/** A simple SIR (Susceptible-Infectious-Recovered) simulation.
  */
object SIRSimulation:

  trait Time derives Label
  trait Group derives Label

  // Explicit state representation replaces the Compartment dimension
  case class SIRState(
      S: Tensor1[Group, Float32],
      I: Tensor1[Group, Float32],
      R: Tensor1[Group, Float32]
  ):
    lazy val N: Tensor1[Group, Float32] = S + I + R

  /** One step of the simulation, according to the SIR model equations
    *
    * @param state The current state of the system
    * @param beta The infection rate matrix (with entry beta[h, g] controlling how strongly infectious
    * individuals in group h infect susceptible individuals in group g)
    * @param gamma The recovery rate
    * @param dt The time step
    * @return The next state of the system
    */
  def step(
      beta: Tensor2[Group, Prime[Group], Float32],
      gamma: Tensor0[Float32],
      dt: Tensor0[Float32]
  )(state: SIRState): SIRState =
    import state.{S, I, R, N}

    val infectiousFraction = I / N
    val force = infectiousFraction.dot(Axis[Group])(beta)
    val transmissions = S * force.dropPrimes

    val recoveries = I *! gamma

    // compute next state
    val SNext = S - transmissions *! dt
    val INext = I + (transmissions - recoveries) *! dt
    val RNext = R + recoveries *! dt

    SIRState(SNext, INext, RNext)

  /** run the simulation over the time steps of `contact`, starting from the initial state
    *
    * The loop is a `scan`: its body is traced once, so the simulation can be jitted and
    * differentiated, however many steps it runs.
    *
    * @param initial The initial state of the system
    * @param beta @see [[step]]
    * @param gamma @see [[step]]
    * @param dt @see [[step]]
    * @param contact The factor by which contacts, and thus infections, are scaled at each time step
    * (e.g. 1 without and below 1 during an intervention)
    * @return The final state of the system, and the infectious individuals per group after each time step
    */
  def simulate(
      initial: SIRState,
      beta: Tensor2[Group, Prime[Group], Float32],
      gamma: Tensor0[Float32],
      dt: Tensor0[Float32],
      contact: Tensor1[Time, Float32]
  ): (SIRState, Tensor2[Time, Group, Float32]) =
    scan(Axis[Time])(initial, contact): (state, contactFactor) =>
      val next = step(beta *! contactFactor, gamma, dt)(state)
      (next, next.I)

  /** The extent of the group axis in the example: children, adults and elderly */
  val groupExtent = Axis[Group] -> 3

  /** The initial state of the example: a few infectious children and adults */
  def exampleInitialState(): SIRState =
    val initialS = Tensor(Shape(groupExtent)).fromFunction(index =>
      index(Axis[Group]) match
        case 0 => 990f
        case 1 => 1995f
        case 2 => 1500f
    )
    val initialI = Tensor(Shape(groupExtent)).fromFunction(index =>
      index(Axis[Group]) match
        case 0 => 10f
        case 1 => 5f
        case 2 => 0f
    )
    val initialR = Tensor(Shape(groupExtent)).fill(0f)
    SIRState(initialS, initialI, initialR)

  /** The infection rates of the example
    *
    * beta(h, g) controls how strongly infectious individuals in group h
    * infect susceptible individuals in group g.
    */
  def exampleBeta(): Tensor2[Group, Prime[Group], Float32] =
    Tensor(Shape(groupExtent, Axis[Prime[Group]] -> groupExtent.size)).fromFunction(index =>
      (index(Axis[Group]), index(Axis[Prime[Group]])) match
        // infectious children -> susceptible children/adults/elderly
        case (0, 0) => 0.40f
        case (0, 1) => 0.20f
        case (0, 2) => 0.10f

        // infectious adults -> susceptible children/adults/elderly
        case (1, 0) => 0.20f
        case (1, 1) => 0.30f
        case (1, 2) => 0.15f

        // infectious elderly -> susceptible children/adults/elderly
        case (2, 0) => 0.10f
        case (2, 1) => 0.15f
        case (2, 2) => 0.20f

        case (_, _) => throw new IllegalArgumentException("Invalid group indices")
    )

  @main def runSIRSimulation(): Unit =
    dimwit.initialize()

    val initial = exampleInitialState()
    val beta = exampleBeta()
    val gamma = Tensor0(0.1f)
    val dt = Tensor0(0.1f)

    // Run the simulation, without and with an intervention that reduces contacts for a while

    val nSteps = 1000
    val noIntervention = Tensor(Shape(Axis[Time] -> nSteps)).fill(1f)
    val lockdown = Tensor1(Axis[Time]).fromArray(Array.tabulate(nSteps)(t => if t >= 100 && t < 300 then 0.4f else 1f))

    for (name, contact) <- List("No intervention" -> noIntervention, "Lockdown" -> lockdown) do
      val (finalState, infectious) = simulate(initial, beta, gamma, dt, contact)

      // Report the results

      val infectedOverTime: Tensor1[Time, Float32] = infectious.sum(Axis[Group])
      println(s"$name:")
      println(s"  peak of infectious = ${infectedOverTime.max} at time step ${infectedOverTime.argmax(Axis[Time])}")
      println(s"  recovered at the end by group = ${finalState.R}")
