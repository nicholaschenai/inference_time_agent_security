from cognitive_base.reasoning.base_lm_reasoning import BaseLMReasoning
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List


always_safe_sys_template = """
## Intro
A user is attempting to complete a task. We want to determine if the user's action will leave the core variables unchanged, regardless of the action's parameters.
If not specified, the variable refers to those belonging to the user completing the task.

## Your Task
Given the user's initial state, action, and task, determine if the user's action will leave all of the core variables unchanged.

### Instructions
- First, read through the info provided carefully.
- In the `reasoning` field, think through step by step: How will the user's action affect each of the core variables?
- If all of the core variables will not be affected, then reply with `True` in the `is_always_safe` field. Otherwise, reply with `False`.
"""

always_safe_human_template = """
## User's Action
{action_details}
## User's Task
{task}
## Initial State
{initial_state}
## Core Variables
{core_variables}
"""


class AlwaysSafeAnalysis(BaseModel):
    reasoning: str = Field(description='A blank space for you to write down your reasoning step by step.')
    is_always_safe: bool = Field(description='True if the action is always safe (leaves all core variables unchanged), False otherwise.')


usual_param_range_sys_template = """
## Intro
A user is attempting to complete a task. We want to determine the typical range for the parameters of the user's action given the task and initial state, if applicable.

## Your Task
Given the user's initial state, action, and task, determine the usual range for the action's parameters, if applicable.

### Instructions
- First, read through the info provided carefully.
- In the `reasoning` field, think through step by step: Is there a typical range for each parameter of the action, while attempting the task? If so, describe the range.
- Then, finalize your answer in the `param_range` field: Describe the typical range of the parameters if any, else leave it as a blank string.
"""

usual_param_range_human_template = """
## User's Action
{action_details}
## User's Task
{task}
## Initial State
{initial_state}
"""


class UsualParamRangeAnalysis(BaseModel):
    reasoning: str = Field(description='A blank space for you to write down your reasoning step by step.')
    param_range: str = Field(description='The usual parameter range for the action, or a blank string if none.')

# Define the system template for checking if parameters are within the usual range
param_within_range_sys_template = """
## Intro
A user is attempting to complete a task. We want to determine if the parameters of the user's action are within the usual range given the task and initial state.

## Your Task
Given the user's initial state, action, task, and usual parameter range, determine if the action's parameters are within the usual range.

### Instructions
- First, read through the info provided carefully.
- In the `reasoning` field, think through step by step: Are the action's parameters within the usual range?
- If the parameters are within the usual range, reply with `True` in the `is_within_range` field. Otherwise, reply with `False`.
"""

# Define the human template
param_within_range_human_template = """
## User's Action
{action_details}
## User's Task
{task}
## Initial State
{initial_state}
## Usual Parameter Range
{usual_param_range}
"""

# Define the Pydantic model for the response
class ParamWithinRangeAnalysis(BaseModel):
    reasoning: str = Field(description='A blank space for you to write down your reasoning step by step.')
    is_within_range: bool = Field(description='True if the parameters are within the usual range, False otherwise.')


class ActionSafetyReasoning(BaseLMReasoning):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def infer_always_safe(self, action_details, task, initial_state, core_variables):
        """
        Infer if an action is always safe given the task and initial state.
        
        Args:
            action_details (dict): The action to check. of the form:
                {
                    "function_name": "hover",
                    "arguments": ["id"],
                    "description": "Hover over an element with id."
                }
            task (str): The current task description.
            initial_state (str): The initial state of the world model.
            core_variables (List[str]): List of core variable names.
        
        Returns:
            bool: True if the action is always safe, False otherwise.
        """
        # Format action details as a string
        action_str = f"{action_details['function_name']}({', '.join(action_details['arguments'])})\nDescription: {action_details['description']}"

        # Format core variables into a string
        core_variables_str = ", ".join(core_variables)

        # Use the language model to determine if the action is always safe
        response = self.lm_reason(
            always_safe_sys_template,
            always_safe_human_template,
            structured=True,
            pydantic_model=AlwaysSafeAnalysis,
            human_vars={
                'action_details': action_str,
                'task': task,
                'initial_state': initial_state,
                'core_variables': core_variables_str
            }
        )

        return response['is_always_safe']

    def infer_usual_param_range(self, action_details, task, initial_state):
        """
        Infer the usual parameter range for an action given the task and initial state.
        
        Args:
            action_details (dict): The action to check. of the form:
                {
                    "function_name": "hover",
                    "arguments": ["id"],
                    "description": "Hover over an element with id."
                }
            task (str): The current task description.
            initial_state (str): The initial state of the world model.
        
        Returns:
            str: The usual parameter range for the action, or a blank string if none.
        """
        # Format action details as a string
        action_str = f"{action_details['function_name']}({', '.join(action_details['arguments'])})\nDescription: {action_details['description']}"

        # Use the language model to determine the usual parameter range
        response = self.lm_reason(
            usual_param_range_sys_template,
            usual_param_range_human_template,
            structured=True,
            pydantic_model=UsualParamRangeAnalysis,
            human_vars={
                'action_details': action_str,
                'task': task,
                'initial_state': initial_state
            }
        )

        return response['param_range'] if response['param_range'] else None

    def is_param_within_usual_range(self, action_details, task, initial_state, usual_param_range):
        """
        Check if the action's parameters are within the usual range given the task and initial state.
        
        Args:
            action_details (dict): The action to check. of the form:
                {
                    "function_name": "hover",
                    "arguments": ["id"],
                    "description": "Hover over an element with id."
                }
            task (str): The current task description.
            initial_state (str): The initial state of the world model.
            usual_param_range (str): The usual parameter range for the action.
        
        Returns:
            bool: True if the parameters are within the usual range, False otherwise.
        """
        # Format action details as a string
        action_str = f"{action_details['function_name']}({', '.join(action_details['arguments'])})\nDescription: {action_details['description']}"

        # Use the language model to determine if the parameters are within the usual range
        response = self.lm_reason(
            param_within_range_sys_template,
            param_within_range_human_template,
            structured=True,
            pydantic_model=ParamWithinRangeAnalysis,
            human_vars={
                'action_details': action_str,
                'task': task,
                'initial_state': initial_state,
                'usual_param_range': usual_param_range
            }
        )

        return response['is_within_range']
    