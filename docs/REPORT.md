# Inference-Time Agent Security
Hackathon submission for the Agent Security Hackathon.

Incomplete due to time constraints, but scaffolded out the main ideas and structure.

## Abstract

## Introduction

The adoption and integration of LLM-based AI agents into real-world systems is on the rise, spurred by the rapid advancements in the technology. For example SWE-Bench Lite `[5]`, a benchmark on solving Github issues, has seen a significant increase in solve rates from approximately 10% at the beginning of 2024 to around 40% currently (Oct 2024).

However, the safety of these agents is still a concern as deep learning / LLMs suffer from some common weaknesses. For one, LLMs' apparent planning ability might be attributed to memorization rather than actual planning `[6]`. Some characteristics present in humans are missing from deep learning systems, e.g. building world models `[7-8]` and causal reasoning `[8]`. These mismatches between human and LLM capabilities, as LLMs play an increasing role in the human world, could lead to unintended and potentially harmful behavior, especially in new environments.

On the other end of the spectrum, in high-assurance systems such as medical devices, formal verification is used to ensure the safety of the system. PDDL (Planning Domain Definition Language) still remains as a foundational tool for AI planning. Symbolic reasoning has a benefit of being interpretable, explainable and potentially verifiable, traits which are desired in current deep learning systems. However, formal verification and PDDL require specifying constraints and defining models (a formal definition of the world), which can be time-consuming to construct and not scalable to highly complex real-world systems.

The field of neuro-symbolic AI refers to the integration of symbolic reasoning with deep learning, and has shown promise in combining the strengths of both approaches, especially towards safety. In the robotics setting, `[9]` showed that LLMs can be used to generate linear temporal logic constraints from natural language, which is then used as a monitor for an LLM-based robot agent for violations, during which the LLM-based agent is reprompted. `[1]` showed a hybrid LLM and enumerative search scheme to generate abstractions in DSL, a step towards automating the construction of models. `[2]` used LLMs to generate high-level planning representations and low-level operators while planning with PDDL and outperformed pure LLM-based planners. `[3]` combines LLMs with PDDL planners and showed that this combination generates more safe plans than pure LLM-based planners, even with perturbations to inputs. 

This project explores an automated way to create symbolic models of the world so that it is amenable to neuro-symbolic reasoning, taking the first step towards scalable, verifiable safety of LLM-based agents. It is a framework for a guardrails system that assesses the impact of an agent's actions in multiple stages, and halts the agent if the action is unsafe. The world model is built on the fly to alleviate the load of constructing the model beforehand.

## Overview

- We explicitly implement the guardrails system to build world models and perform causal reasoning, in the form of a hybrid state transition and causal graph model.
- Instead of building the world model beforehand (infeasible for real-world systems due to complexity), we build it incrementally as the agent interacts with the environment and new information is encountered.
- The incremental building of the world model can be error-prone, so we only use the world model as a warning for potential issues -- when this occurs, further reasoning is performed to assess if the agent's action is safe.
- Demo on a mock web environment modeled after WebArena `[4]` 
- Currently most of the reasoning is still LLM-based, but the causal graph model is amenable to PDDL where we can encode the predicates and actions as attributes of nodes and edges, opening up the possibility of symbolic reasoning in future work.
- We perform checks on many levels: 
  - If the action is safe in general
  - If the parameters of the action (tools) are within usual parameter range
  - If the action is likely to affect core variables
  - If the action is likely to affect core variables in a way that exceeds bounds
  - If the next state is new and likely to affect core variables


## Code

- NetworkX for graphical model
- Langchain for LLM related reasoning

### Pseudocode

Note that all the results of reasoning are stored in a world model (`src/models/world_model.py`) in the form of a graph, and retrieval is also from this model

in `src/main.py`:

**Given**: Task, core variables we care about (e.g. money, outbound sensitive data, integrity of filesystem)
- **Reason typical variation of core variables**:
  - Analyze the core variables to determine their typical variation given the task

- **Agent-environment loop**:
  - **Reset** the environment to get the initial observation
  - **While** not done:
    - Agent decides on an action based on the observation
    - **Check** if the action is safe via Safety Module:
      - **If** action is safe:
        - Execute the action in the environment and get the new observation
      - **Else**:
        - Raise error, exit loop

#### **Safety Module** check
Main process in `src/safety_module.py`, reasoning processes in `src/reasoning`
- Check/reason if action is always safe (e.g. no op) relative to the task and core variables
- **Check/reason If** parameters of action is within usual parameter range for the task and core variables
- Reason/Retrieve effective state (a state that represents how likely it is to affect core variables) based on observation
- **If** effective state was determined to be likely to affect core variables:
  - Reason to determine the impact of the action on the core variables
  - **If** impact exceeds bounds:
    - Return False
- Reason/retrieve next effective state
  - **If** next effective state is new:
    - Reason if it could affect core variables

## Discussion

No results yet due to time constraints of the hackathon; Currently still needs more debugging and testing.

### Limitations
Currently, a lot of the safeguard reasoning is still LLM-based, which might suffer from the same flaws as the agent itself. 
We look to increasingly replace this with symbolic reasoning.

## Future work
`suggest potential improvements or future directions for your work. `

`Use this section to include observations, discussions on potential expansions if you had more time, results, and broader implications.`

Future work includes finding automated ways to represent the world model as classes, functions and variables, which are then amenable to existing static / dynamic analysis tools (e.g. call graph analysis, data flow analysis) to assess the impact of an agent's action 

- Encode the predicates and actions as attributes of nodes and edges in the graph so that we can additionally use PDDL for reasoning.
- Include variables internal to system which it affects (e.g. shopping cart items in a web shop environment), which could eventually affect the core variables
- Shortest path (partially implemented) to see how close the agent is to affecting the core variable, as warning. 
- Other graph algos for useful insights
- Mining constraints / models from public data, e.g. from Github issues to know what are negative / dangerous patterns for code environments, conventions and expected behavior (via analyzing assert statements)

### More scenarios to test the robustness of the system
- Unsafe web operation
	- surfing unrelated urls (inward data security)
	- posting sensitive info (outward data security)
	- affecting finances
- Unsafe coding operation
    - agent is given filesystem rights to edit files but could potentially delete important files

## Conclusion
The dream: Mapping LLM-based agents and their environments into models that are formally verifiable.


## References
1. Grand, G., Wong, L., Bowers, M., Olausson, T.X., Liu, M., Tenenbaum, J.B. and Andreas, J., 2023. LILO: Learning interpretable libraries by compressing and documenting code. arXiv preprint arXiv:2310.19791.
2. Wong, L., Mao, J., Sharma, P., Siegel, Z.S., Feng, J., Korneev, N., Tenenbaum, J.B. and Andreas, J., 2023. Learning adaptive planning representations with natural language guidance. arXiv preprint arXiv:2312.08566.
3. Martinez-Suñé, A., 2024. Neuro-Symbolic Approaches for Safe LLM-Based Agents https://www.youtube.com/watch?v=6lFd_GWp-Ds
4. Zhou, S., Xu, F.F., Zhu, H., Zhou, X., Lo, R., Sridhar, A., Cheng, X., Ou, T., Bisk, Y., Fried, D. and Alon, U., 2023. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854.
5. Jimenez, C.E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O. and Narasimhan, K., 2023. Swe-bench: Can language models resolve real-world github issues?. arXiv preprint arXiv:2310.06770.
6. Kambhampati, S., Valmeekam, K., Guan, L., Stechly, K., Verma, M., Bhambri, S., Saldyt, L. and Murthy, A., 2024. LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks. arXiv preprint arXiv:2402.01817.
7. Dalrymple, D., Skalse, J., Bengio, Y., Russell, S., Tegmark, M., Seshia, S., Omohundro, S., Szegedy, C., Goldhaber, B., Ammann, N. and Abate, A., 2024. Towards Guaranteed Safe AI: A Framework for Ensuring Robust and Reliable AI Systems. arXiv preprint arXiv:2405.06624.
8. Lake, B.M., Ullman, T.D., Tenenbaum, J.B. and Gershman, S.J., 2017. Building machines that learn and think like people. Behavioral and brain sciences, 40, p.e253.
9. Yang, Z., Raman, S.S., Shah, A. and Tellex, S., 2024, May. Plug in the safety chip: Enforcing constraints for llm-driven robot agents. In 2024 IEEE International Conference on Robotics and Automation (ICRA) (pp. 14435-14442). IEEE.

## Appendix
