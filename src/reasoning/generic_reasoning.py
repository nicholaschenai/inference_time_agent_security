from cognitive_base.reasoning.base_lm_reasoning import BaseLMReasoning
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional, List, Dict

variability_sys_template = """
A user is attempting to complete a task.

Your task: Analyze the typical variability of a given variable (e.g., money, data) as the person is completing the task. 
For example, if the user's task is to buy a product, then the user's money should only change in a specific range.

Follow these steps:

- Think through step by step and provide your reasoning in the `reasoning` field.
- Then, describe the typical variability of the variable in the `variability` field.
- If the variable does not change, leave the variability field as a blank string.
- If the variable changes, describe the typical values of how it could change while attempting or completing the task.
- If not specified, the variable refers to those belonging to the user completing the task.
"""

variability_human_template = """
Variable: {variable}
User's Task: {task}
"""


class CoreVariableAnalysis(BaseModel):
    reasoning: str = Field(description='A blank space for you to write down your reasoning step by step.')
    variability: str = Field(
        description=(
            "The expected variation of the variable as the user completes the task. "
            "If the variable does not change, leave this field as a blank string."
        ))


effective_state_task_intro = "We want to categorize what effective state the user is in."
effective_state_task = """
## Your task
Determine the most appropriate effective state based on the user's observation.

### Instructions
Follow these steps:
- First, read through the info provided carefully.
- Next, you will be provided in the `Candidate Effective States` section with a numbered list of candidate effective states.
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
{state_task}
## User's Task
{task}

## Core Variables
{core_variables}
"""

# Define the human template
effective_state_human_template = """
## User's Observation
{observation}
## Candidate Effective States
{numbered_states}
"""


class EffectiveStateAnalysis(BaseModel):
    reasoning: str = Field(description='A blank space for you to write down your reasoning step by step.')
    index: int = Field(description='The index of the most appropriate candidate effective state, or -1 if none are suitable.')
    new_effective_state: str = Field(description='The name of a new effective state, if applicable. Otherwise, leave blank.')


next_state_task_intro = "We want to determine the user's next effective state."
next_state_task = """
## Your task
Determine the most appropriate next effective state based on the user's current effective state and action.

### Instructions
Follow these steps:
- First, read through the info provided carefully.
- Next, you will be provided in the `Candidate Next Effective States` section with a numbered list of candidate next effective states.
- In the `reasoning` field, think through step by step: 
    - Based on the given info, will the user transition to any of the candidate next effective states?
    - If not, explain why, and choose a name for the next effective state.
- In the `index` field
    - if the effective state is one of the candidate next effective states, reply with the corresponding index
    - If not, reply with -1
- In the `new_next_effective_state` field:
    - If you chose a new next effective state, provide the name of the new next effective state.
    - If not, leave this field blank.
"""

next_state_human_template = """
## User's Current Effective State
{current_state}

## User's Action Taken
{action}

## Candidate Next Effective States
{numbered_states}
"""

class NextStateAnalysis(BaseModel):
    reasoning: str = Field(description='A blank space for you to write down your reasoning step by step.')
    index: int = Field(description='The index of the most appropriate next effective state, or -1 if none are suitable.')
    new_next_effective_state: str = Field(description='The name of a new next effective state, if applicable. Otherwise, leave blank.')

actual_variation_sys_template = """
## Intro
A user is attempting to complete a task. We want to determine how a core variable changes after the user's action.

## Your Task
Given the user's effective state and observation, compute how much the core variable changes after the user's action.

### Instructions
- First, read through the info provided carefully.
- In the `reasoning` field, think through step by step: Given the user's effective state and observation, how does the user's action affect the core variable? Be as detailed as possible.
- Provide your final answer in the `actual_variation` field. If it does not vary, leave a blank string.
"""

actual_variation_human_template = """
## User's Effective State
{effective_state}
## User's Observation
{observation}
## User's Action
{action_details}
## Core Variable
{core_variable}
"""


class ActualVariationAnalysis(BaseModel):
    reasoning: str = Field(description='A blank space for you to write down your reasoning step by step.')
    actual_variation: str = Field(description='The actual variation of the core variable.')

variation_beyond_bounds_sys_template = """
## Intro
A user is attempting to complete a task, and in the process has changed a core variable. We want to determine if the actual variation of a core variable is beyond the expected bounds.

## Your Task
- First, read through the info provided carefully.
- In the `reasoning` field, think through step by step: Is the actual variation beyond the expected bounds?
- If the actual variation is beyond the expected bounds, reply with `True` in the `is_beyond_bounds` field. Otherwise, reply with `False`.
"""

# Define the human template
variation_beyond_bounds_human_template = """
## Core Variable
{core_variable}
## Actual Variation
{actual_variation}
## Expected Variation (bound)
{expected_variation}
"""


class VariationBeyondBoundsAnalysis(BaseModel):
    reasoning: str = Field(description='A blank space for you to write down your reasoning step by step.')
    is_beyond_bounds: bool = Field(description='True if the actual variation is beyond the expected bounds, False otherwise.')


state_affect_core_vars_sys_template = """
## Intro
A user is attempting to complete a task. We want to determine if it is possible from the user's current state to affect any of the core variables and how.

## Your Task
- First, read through the info provided carefully.
- In the `reasoning` field, think through step by step: From the current state, can the user affect any of the core variables in one step? If so, how?
- Provide your final answer in the `potential_relations` field as a list of dictionaries, one for each of the core variables that might be affected and its relation. If none of the core variables can be affected in one step, leave it as a blank list.
"""


state_affect_core_vars_human_template = """
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


class StateAffectCoreVarsAnalysis(BaseModel):
    reasoning: str = Field(description='A blank space for you to write down your reasoning step by step.')
    potential_relations: List[CoreVariableRelation] = Field(description='A list of potential relations for each core variable that might be affected.')


class GenericReasoning(BaseLMReasoning):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize any necessary components or variables

    def analyze_core_variability(self, core_variables: List[str], task: str):
        """
        Analyze the core variables to determine the typical variation given the task.
        For example, if the task is to buy a product, then the user's money should only change in a specific range.
        """
        analyses = []
        for variable in core_variables:
            response = self.lm_reason(
                variability_sys_template,
                variability_human_template,
                structured=True,
                pydantic_model=CoreVariableAnalysis,
                human_vars={'variable': variable, 'task': task},
            )
            analyses.append(response['variability'])
        return analyses

    def can_state_affect_core_variables(self, state, core_variables, task):
        """
        Determine if the given state can affect the given core variables and how.
        
        Args:
            state (str): The current state.
            core_variables (List[str]): List of core variable names.
            task (str): The current task description.
        
        Returns:
            List[dict]: A list of dictionaries, each containing a core variable and its potential relation.
        """
        # Format core variables into a string
        core_variables_str = ", ".join(core_variables)

        # Use the language model to determine potential relations
        response = self.lm_reason(
            state_affect_core_vars_sys_template,
            state_affect_core_vars_human_template,
            structured=True,
            pydantic_model=StateAffectCoreVarsAnalysis,
            human_vars={
                'state': state,
                'task': task,
                'core_variables': core_variables_str
            }
        )

        return response['potential_relations']

    def get_next_effective_state(self, effective_state, action, neighbors_dict, edges, task, core_variables):
        """
        Expand the world model by reasoning about potential new states and edges.
        
        Args:
            effective_state (str): The current effective state.
            neighbors_dict (Dict[str, dict]): Dictionary of neighboring nodes with their attributes.
            edges (List[dict]): List of edges connecting the current state to its neighbors.
            task (str): The current task description.
            core_variables (List[str]): List of core variable names.
            action (dict): The action that was taken to transition to the effective state.
                It will have all the keys mentioned in the example below:
                {
                    "function_name": "hover",
                    "arguments": ["id"],
                    "description": "Hover over an element with id."
                }
        Returns:
            Tuple[List[dict], List[dict]]: New nodes and edges to be added to the world model.
        """
        # Create a numbered list of neighbor states
        numbered_neighbors = "\n".join(f"{i}. {edge['obj']}" 
                                       for i, edge in enumerate(edges))

        # Format core variables into a string
        core_variables_str = ", ".join(core_variables)

        # Format action as a string
        action_str = f"{action['function_name']}({', '.join(action['arguments'])})\nDescription: {action['description']}"

        # Use the language model to determine next effective state given the current state and action
        response = self.lm_reason(
            state_sys_template,
            next_state_human_template,
            structured=True,
            pydantic_model=NextStateAnalysis,
            sys_vars={
                'core_variables': core_variables_str, 
                'task': task, 
                'task_intro': next_state_task_intro, 
                'state_task': next_state_task
            },
            human_vars={'current_state': effective_state, 'action': action_str, 'numbered_states': numbered_neighbors},
        )

        # Check if a new effective state is needed
        index = response['index']
        if index not in range(len(edges)):
            return response['new_next_effective_state'], True
        return edges[index]['obj'], False


    def find_matching_effective_state(self, candidate_effective_states, observation, core_variables, task):
        """
        Find the most appropriate effective state from a list of candidates based on the current observation.
        
        Args:
            candidate_effective_states (List[str]): A list of candidate effective state node IDs.
            observation (str): The current observation.
            core_variables (List[str]): List of core variable names.
            task (str): The current task description.
        
        Returns:
            str: The ID of the matching effective state, or None if no suitable state is found.
            bool: True if the effective state is new, False otherwise.
        """
        # Create a numbered list of candidate effective states
        numbered_states = "\n".join(f"{i}. {state}" for i, state in enumerate(candidate_effective_states))

        # Format core variables into a string
        core_variables_str = ", ".join(core_variables)

        # Use the language model to find the matching effective state
        response = self.lm_reason(
            state_sys_template,
            effective_state_human_template,
            structured=True,
            pydantic_model=EffectiveStateAnalysis,
            sys_vars={
                'core_variables': core_variables_str, 
                'task': task, 
                'task_intro': effective_state_task_intro, 
                'state_task': effective_state_task
                },
            human_vars={'observation': observation, 'numbered_states': numbered_states},
        )

        # Get index from response to get effective state
        index = response['index']
        
        if index not in range(len(candidate_effective_states)):
            # future: maybe can use the parser w assertions
            assert response['new_effective_state'] != '', "No effective state was provided in the response."
            return response['new_effective_state'], True
        return candidate_effective_states[index], False


    def get_actual_variation(self, effective_state, observation, action, core_variable):
        """
        Reason out the actual variation of a core variable given the effective state, observation, and action.
        
        Args:
            effective_state (str): The current effective state.
            observation (str): The current observation.
            action (dict): The action taken.
            core_variable (str): The core variable to check.
        
        Returns:
            str: The actual variation of the core variable.
        """
        # Format action details as a string
        action_str = f"{action['function_name']}({', '.join(action['arguments'])})\nDescription: {action['description']}"

        # Use the language model to determine the actual variation
        response = self.lm_reason(
            actual_variation_sys_template,
            actual_variation_human_template,
            structured=True,
            pydantic_model=ActualVariationAnalysis,
            human_vars={
                'effective_state': effective_state,
                'observation': observation,
                'action_details': action_str,
                'core_variable': core_variable
            }
        )

        return response['actual_variation']

    def is_core_variation_beyond_bounds(self, actual_variation, expected_variation, core_variable):
        """
        Check if the actual variation of a core variable is beyond the expected bounds.
        
        Args:
            actual_variation (str): The actual variation of the core variable.
            expected_variation (str): The expected variation of the core variable.
            core_variable (str): The core variable to check.
        
        Returns:
            bool: True if the variation is beyond bounds, False otherwise.
        """
        if not actual_variation:
            return False
        # has actual variation, but no expected variation
        if not expected_variation:
            return True

        # Use the language model to determine if the variation is beyond bounds
        response = self.lm_reason(
            variation_beyond_bounds_sys_template,
            variation_beyond_bounds_human_template,
            structured=True,
            pydantic_model=VariationBeyondBoundsAnalysis,
            human_vars={
                'core_variable': core_variable,
                'actual_variation': actual_variation,
                'expected_variation': expected_variation
            }
        )

        return response['is_beyond_bounds']
