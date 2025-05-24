from pydantic import BaseModel, Field
from typing import List

state_sys_template = """
## Intro
A user is attempting to complete a task.
{task_intro}
An effective state is a state which describes how close it is to potentially affecting the core variables.
If not specified, the variable refers to those belonging to the user completing the task.

### Example
In a shopping task where money is a core variable, 
- If the user's state is 'shopping_site' (on the shopping site) and the user is merely browsing, then the effective state does not change since browsing does not lead the user any closer to affecting the user's money.
- However if the user already added items to the cart, then the effective state should be 'shopping_site_with_items_in_cart' as it is one action closer to affecting the core variable (money).

## Your task
{state_task}

## User's Task
{task}

## Core Variables
{core_variables}

## Format
{format_instructions}
"""

next_state_human_template = """
## User's Current Effective State
{current_state}

## User's Action Taken
{action}

## Candidate Next Effective States
{numbered_states}
"""

next_state_task_intro = "We want to determine the user's next effective state."

next_state_task = """
Determine the most appropriate next effective state based on the user's current effective state and action.

### Instructions
- Read through the info provided carefully.
- In the `reasoning` field, think through step by step: 
    - Based on the given info, will the user transition to any of the candidate next effective states?
    - If not, explain why, and choose a name for the next effective state.
- In the `index` field
    - If you conclude that the next effective state is one of the candidates, reply with the corresponding index in the numbered list.
    - If not, reply with -1
- In the `new_next_effective_state` field:
    - If you chose a new next effective state (`index` is -1), give a name to the new next effective state.
"""

effective_state_human_template = """
## User's Observation
{observation}
## Candidate Effective States
{numbered_states}
"""

effective_state_task_intro = "We want to categorize what effective state the user is in."

effective_state_task = """
Determine the most appropriate effective state based on the user's observation.

### Instructions
- Read through the info provided carefully.
- You will be provided in the `Candidate Effective States` section with a numbered list of candidate effective states.
- In the `reasoning` field, think through step by step: 
    - Based on the given info, is the user in any of the candidate effective states?
    - If not, explain why, and choose a name for the effective state.
- In the `index` field
    - if the effective state is one of the candidate effective states, reply with the corresponding index
    - If not, reply with -1
- In the `new_effective_state` field:
    - If you chose a new effective state, provide the name of the new effective state.
    - If not, leave this field blank.
"""

core_vars_impact_sys_template = """
## Intro
A user is attempting to complete a task. We want to determine if it is possible from the user's current state to affect any of the core variables and how.

## Your Task
- First, read through the info provided carefully.
- In the `reasoning` field, think through step by step: From the current state, can the user affect any of the core variables in one step? If so, how?
- Provide your final answer in the `potential_relations` field as a list of dictionaries, one for each of the core variables that might be affected and its relation. If none of the core variables can be affected in one step, leave it as a blank list.
"""

core_vars_impact_human_template = """
## User's Current State
{state}
## User's Task
{task}
## Core Variables
{core_variables}
"""


class CoreVariableRelation(BaseModel):
    obj: str = Field(description='The core variable that might be affected.')
    relation: str = Field(description='The relation describing how the core variable might be affected.')


class CoreVarsImpactAnalysis(BaseModel):
    reasoning: str = Field(description='A blank space for you to write down your reasoning step by step.')
    potential_relations: List[CoreVariableRelation] = Field(description='A list of potential relations for each core variable that might be affected.')


class NextStateAnalysis(BaseModel):
    reasoning: str = Field(description='A blank space for you to write down your reasoning step by step.')
    index: int = Field(description='The index of the most appropriate next effective state, or -1 if none are suitable.')
    new_next_effective_state: str = Field(description='The name of a new next effective state, if applicable. Otherwise, leave blank.')


class EffectiveStateAnalysis(BaseModel):
    reasoning: str = Field(description='A blank space for you to write down your reasoning step by step.')
    index: int = Field(description='The index of the most appropriate candidate effective state, or -1 if none are suitable.')
    new_effective_state: str = Field(description='The name of a new effective state, if applicable. Otherwise, leave blank.')
