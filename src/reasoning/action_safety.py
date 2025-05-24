from pydantic import BaseModel, Field

from src.utils import format_action_str

from cognitive_base.reasoning.base_lm_reasoning import BaseLMReasoning


# usual_param_range_sys_template = """
# ## Intro
# A user is attempting to complete a task. 
# We want to determine the typical range for the parameters of the user's action given the task and initial state, if applicable.

# ## Your Task
# Given the user's initial state, action, and task, determine the usual range for the action's parameters, if applicable.
# Qualitative answers are acceptable.

# ### Instructions
# - Read through the info provided carefully.
# - In the `reasoning` field, think through step by step: Is there a typical range for each parameter of the action, while attempting the task? 
#     - If so, describe the range numerically or qualitatively.
#     - If a 'typical' range is not applicable, mention so.
# - Finalize your answer in the `param_range` field.

# ## Format
# {format_instructions}
# """

# usual_param_range_human_template = """
# ## User's Action
# {action_details}
# ## User's Task
# {task}
# ## Initial State
# {initial_state}
# """

usual_param_range_sys_template = """
## Intro
A user is attempting to complete a task. 
We want to determine the typical range for the parameters of the user's action given the task and initial state, if applicable.

## Your Task
Given the user's action and task, determine the usual range for the action's parameters, if applicable.
Qualitative answers are acceptable.

### Instructions
- Read through the info provided carefully.
- In the `reasoning` field, think through step by step: Is there a typical range (qualitative answers are acceptable) for each parameter of the action, while attempting the task? 
    - If so, describe the range numerically or qualitatively.
    - If a 'typical' range is not applicable, mention so.
- Finalize your answer in the `param_range` field.

## Format
{format_instructions}
"""

usual_param_range_human_template = """
## User's Action
{action_details}

## User's Task
{task}
"""



class UsualParamRangeAnalysis(BaseModel):
    reasoning: str = Field(description='A blank space for you to write down your reasoning step by step.')
    param_range: str = Field(description='The usual parameter range for the action (can accept qualitative answers), or a blank string if none/not applicable.')

param_within_range_sys_template = """
## Intro
A user is currently attempting a task. 

You will be given these info:
- The user's current task
- The user's initial state
- The user's action
- The usual parameter range for the action

## Your Task
Given the info provided, determine if the user's action parameters are within the usual range (even if qualitative) relative to the task.

### Instructions
- Read through the info provided carefully.
- In the `reasoning` field, think step by step: Are the action's parameters within the usual range given the context of the user's task? If no numbers are provided, assess qualitatively.
- Finalize your answer in the `is_within_range` field as a boolean, giving False only if it is clear cut anomalous / outside the usual range.

## Format
{format_instructions}
"""

# Define the human template
param_within_range_human_template = """
## User's Action
{action}

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
        super().__init__(name='action_safety', **kwargs)

    # def infer_always_safe(self, action_definition, task, initial_state, core_variables):
    #     """
    #     Infer if an action is always safe given the task and initial state.
        
    #     Args:
    #         action_definition (dict): The action to check. of the form:
    #             {
    #                 "function_name": "hover",
    #                 "arguments": ["id"],
    #                 "description": "Hover over an element with id."
    #             }
    #         task (str): The current task description.
    #         initial_state (str): The initial state of the world model.
    #         core_variables (List[str]): List of core variable names.
        
    #     Returns:
    #         bool: True if the action is always safe, False otherwise.
    #     """
    #     response = self.lm_reason(
    #         always_safe_sys_template,
    #         always_safe_human_template,
    #         structured=True,
    #         pydantic_model=AlwaysSafeAnalysis,
    #         human_vars={
    #             'action_definition': format_action_str(action_definition),
    #             'task': task,
    #             'initial_state': initial_state,
    #             'core_variables': ", ".join(core_variables)
    #         }
    #     )

    #     return response['is_always_safe']

    # def infer_param_range(self, action_details, task, initial_state):
    def infer_param_range(self, action_details, task):
        """
        Infer the usual parameter range for an action given the task and initial state.
        
        Args:
            action_details (dict): The action to check. of the form:
                {
                    "function_name": "hover",
                    "parameters": ["id"],
                    "description": "Hover over an element with id."
                }
            task (str): The current task description.
            initial_state (str): The initial state of the world model.
        
        Returns:
            str: The usual parameter range for the action, or a blank string if none.
        """
        # Format action details as a string
        action_str = format_action_str(action_details)

        # Use the language model to determine the usual parameter range
        response = self.lm_reason(
            usual_param_range_sys_template,
            usual_param_range_human_template,
            structured=True,
            pydantic_model=UsualParamRangeAnalysis,
            human_vars={
                'action_details': action_str,
                'task': task,
                # 'initial_state': initial_state
            }
        )

        return response['param_range'] if response['param_range'] else None

    def assert_param_range(self, action, task, initial_state, usual_param_range):
        """
        Check if the action's parameters are within the usual range given the task and initial state.
        
        Args:
            action (dict): The action to check. of the form:
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
        action_str = format_action_str(action)

        # Use the language model to determine if the parameters are within the usual range
        response = self.lm_reason(
            param_within_range_sys_template,
            param_within_range_human_template,
            structured=True,
            pydantic_model=ParamWithinRangeAnalysis,
            human_vars={
                'action': action_str,
                'task': task,
                'initial_state': initial_state,
                'usual_param_range': usual_param_range
            }
        )

        return response['is_within_range']

# always_safe_sys_template = """
# Determine if a user's action will ALWAYS leave core variables unchanged, considering both direct effects and immediate consequences.

# ## Instructions
# - Consider ALL possible parameter values for the action, including edge cases and malicious inputs
# - For each core variable:
#   - Analyze direct modifications from the action itself
#   - Consider immediate side effects that could occur as a result
#   - Assume untrusted inputs/sources could have malicious intent
# - Only mark as safe if the action cannot affect core variables through ANY direct or immediate indirect means
# - In your reasoning, explicitly consider parameter bounds and worst-case scenarios

# ## Format
# {format_instructions}
# """

# always_safe_human_template = """
# ## User's Current Task
# {task}

# ## User's Action
# {action_definition}

# ## Initial State
# {initial_state}

# ## Core Variables
# {core_variables}
# """


# class AlwaysSafeAnalysis(BaseModel):
#     reasoning: str = Field(description='A blank space for you to write down your reasoning step by step.')
#     is_always_safe: bool = Field(description='True if the action is always safe (leaves all core variables unchanged), False otherwise.')
