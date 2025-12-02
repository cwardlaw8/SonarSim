# Ethics Discussion – Camille Wardlaw

**Meeting:** Internal team sync  
**Date:** 2026-11-30, 12:00 PM  
**Location:** MIT Stata Center  
**Attendees:** Camille Wardlaw, Manuel Valencia, Demircan Tas

## Open Access vs. Classified Research
- Open-sourcing our acoustic simulator promotes equitable access. Many existing ocean acoustic models require proprietary software, deep expertise, or large compute budgets, which excludes researchers in underfunded institutions and developing regions where conservation stakes are high.
- There is an asymmetry between open academic work and classified military research. Any advances we publish can be absorbed by defense programs, yet military-funded underwater acoustics rarely flow back to civilian science. Open tools may unintentionally subsidize naval capabilities without reciprocal benefit to conservation.

## Downstream Use and Loss of Control
- Once the simulator is public, we cannot control how it is adapted. While our intent is environmental impact assessment and mitigation planning, the same physics implementation could be repurposed for sonar system design, submarine detection training, or underwater surveillance.
- Acknowledging dual-use potential does not necessarily argue against open release, but it requires transparency that the code is domain-agnostic and easily tuned for defense scenarios.

## Irony of Research-Driven Noise
- Demircan highlighted that field data collection needed to validate these models contributes to the very noise pollution we aim to reduce. Research vessels generate engine and propeller noise, and active sonar instruments add acoustic energy to the water column.
- This paradox means conservation-driven measurements risk stressing marine life, especially in sensitive habitats or during breeding/migration. Our simulator supports planners who conduct such surveys, so we should acknowledge that mitigation work can itself introduce disturbance.

## Takeaways
- Open access aligns with scientific equity but creates tension with classified research that does not reciprocate.
- The simulator is inherently dual-use; we should state plainly that we cannot control downstream adoption.
- Conservation modeling and measurement campaigns can add to ocean noise—mitigation efforts should weigh the costs of data collection against the benefits of improved policy and engineering decisions.
