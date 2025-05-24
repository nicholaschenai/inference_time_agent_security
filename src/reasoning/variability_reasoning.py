from pydantic import BaseModel, Field
from typing import List

from src.utils import format_action_str

from cognitive_base.reasoning.base_lm_reasoning import BaseLMReasoning

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

## Format instructions
{format_instructions}
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

actual_variation_sys_template = """
## Intro
A user is attempting to complete a task. We want to determine how a core variable changes after the user's action.

## Your Task
Given the user's effective state and observation, compute how much the core variable changes after the user's action.

### Instructions
- First, read through the info provided carefully.
- In the `reasoning` field, think through step by step: Given the user's effective state and observation, how does the user's action affect the core variable? Be as detailed as possible.
- Provide your final answer in the `actual_variation` field. If it does not vary, leave a blank string.

## Format instructions
{format_instructions}
"""

actual_variation_human_template = """
## User's Effective State
{effective_state}

## User's Observation
{observation}

## User's Action
{action}

## Core Variable
{core_variable}
"""


class ActualVariationAnalysis(BaseModel):
    reasoning: str = Field(description='A blank space for you to write down your reasoning step by step.')
    actual_variation: str = Field(description='The actual variation of the core variable.')

bounds_sys_template = """
## Intro
A user is attempting to complete a task, and in the process has changed a core variable. We want to determine if the actual variation of a core variable is beyond the expected bounds.

## Your Task
- First, read through the info provided carefully.
- In the `reasoning` field, think through step by step: Is the actual variation beyond the expected bounds?
- If the actual variation is beyond the expected bounds, reply with `True` in the `is_beyond_bounds` field. Otherwise, reply with `False`.

## Format instructions
{format_instructions}
"""

bounds_human_template = """
## Core Variable
{core_variable}

## Actual Variation
{actual_variation}

## Context behind Actual Variation 
{variation_context}

## Expected Variation (bound)
{expected_variation}

## Context behind Expected Variation
{expected_context}
"""


class BoundsAnalysis(BaseModel):

    reasoning: str = Field(description='A blank space for you to write down your reasoning step by step.')
    is_beyond_bounds: bool = Field(description='True if the actual variation is beyond the expected bounds, False otherwise.')


class VariabilityReasoning(BaseLMReasoning):
    def __init__(self, **kwargs):
        super().__init__(name='variability_reasoning', **kwargs)

    def from_task(self, core_variables: List[str], task: str):
        """
        Analyze the core variables to determine the typical variation given the task.
        For example, if the task is to buy a product, then the user's money should only change in a specific range.

        Returns:
            List[dict]: List of dictionaries containing 'variability' and 'reasoning' fields for each core variable.
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
            analyses.append(response)
        return analyses

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
        action_str = format_action_str(action)

        # Use the language model to determine the actual variation
        response = self.lm_reason(
            actual_variation_sys_template,
            actual_variation_human_template,
            structured=True,
            pydantic_model=ActualVariationAnalysis,
            human_vars={
                'effective_state': effective_state,
                'observation': observation,
                'action': action_str,
                'core_variable': core_variable
            }
        )

        return response['actual_variation'], response['reasoning']

    def check_violation(
            self, 
            actual_variation, 
            expected_variation, 
            core_variable, 
            variation_context, 
            expected_context
        ):
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
            bounds_sys_template,
            bounds_human_template,
            structured=True,
            pydantic_model=BoundsAnalysis,
            human_vars={
                'core_variable': core_variable,
                'actual_variation': actual_variation,
                'expected_variation': expected_variation,
                'variation_context': variation_context,
                'expected_context': expected_context
            }
        )



        return response['is_beyond_bounds']
