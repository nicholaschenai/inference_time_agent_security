from . import state_reasoning_prompts as p
from src.utils import format_action_str

from cognitive_base.reasoning.base_lm_reasoning import BaseLMReasoning


class StateReasoning(BaseLMReasoning):
    def __init__(self, **kwargs):
        super().__init__(name='state_reasoning', **kwargs)

    def next_effective_state(self, effective_state, action, state_edges, task, core_variables):
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
        numbered_neighbors = "\n".join(f"{i}. {edge['obj']}" for i, edge in enumerate(state_edges))

        # Use the language model to determine next effective state given the current state and action
        response = self.lm_reason(
            p.state_sys_template,
            p.next_state_human_template,
            structured=True,
            pydantic_model=p.NextStateAnalysis,
            sys_vars={
                'core_variables':  ", ".join(core_variables), 
                'task': task, 
                'task_intro': p.next_state_task_intro, 
                'state_task': p.next_state_task
            },
            human_vars={
                'current_state': effective_state, 
                'action': format_action_str(action), 
                'numbered_states': numbered_neighbors
            },
        )

        # Check if a new effective state is needed
        index = response['index']
        if index not in range(len(state_edges)):
            return response['new_next_effective_state'], True
        
        return state_edges[index]['obj'], False

    def infer_effective_state(self, candidates, observation, core_variables, task):
        """
        Find the most appropriate effective state from a list of candidates based on the current observation.
        If no effective states match observation, use reasoning to create new effective state
        
        Args:
            candidates (List[str]): A list of candidate effective state node IDs.
            observation (str): The current observation.
            core_variables (List[str]): List of core variable names.
            task (str): The current task description.
        
        Returns:
            str: The ID of the matching effective state, or None if no suitable state is found.
            bool: True if the effective state is new, False otherwise.
        """
        # Create a numbered list of candidate effective states
        numbered_states = "\n".join(f"{i}. {state}" for i, state in enumerate(candidates))

        # Format core variables into a string
        core_variables_str = ", ".join(core_variables)

        # Use the language model to find the matching effective state
        response = self.lm_reason(
            p.state_sys_template,
            p.effective_state_human_template,
            structured=True,
            pydantic_model=p.EffectiveStateAnalysis,
            sys_vars={
                'core_variables': core_variables_str, 
                'task': task, 
                'task_intro': p.effective_state_task_intro, 
                'state_task': p.effective_state_task
            },
            human_vars={'observation': observation, 'numbered_states': numbered_states},
        )

        # Get index from response to get effective state
        index = response['index']
        
        if index not in range(len(candidates)):
            # future: maybe can use the parser w assertions
            assert response['new_effective_state'] != '', "No effective state was provided in the response."
            return response['new_effective_state'], True
        return candidates[index], False
    
    def core_vars_impact(self, state, core_variables, task):
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
            p.core_vars_impact_sys_template,
            p.core_vars_impact_human_template,
            structured=True,
            pydantic_model=p.CoreVarsImpactAnalysis,
            human_vars={
                'state': state,
                'task': task,
                'core_variables': core_variables_str
            }
        )

        return response['potential_relations']
