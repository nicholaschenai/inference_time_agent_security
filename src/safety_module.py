import logging

logger = logging.getLogger("logger")

from models.world_model import WorldModel

from reasoning.variability_reasoning import VariabilityReasoning
from reasoning.action_safety import ActionSafetyReasoning
from reasoning.state_reasoning import StateReasoning

from utils import prepare_base_graph_update, prepare_transition_edge



class SafetyModule:
    def __init__(
            self, 
            initial_state, 
            action_space,
            task,
            core_variables,
            **kwargs
        ):
        # fixed variables
        self.core_variables = core_variables
        self.initial_state = initial_state
        self.task = task
        self.action_space = action_space

        # state variables
        self.effective_state = initial_state

        # reasoning
        self.variability_reasoning = VariabilityReasoning(**kwargs)
        self.action_safety = ActionSafetyReasoning(**kwargs)
        self.state_reasoning = StateReasoning(**kwargs)

        # world model
        self.world_model = WorldModel(initial_state, **kwargs)
        variabilities = self.variability_reasoning.from_task(core_variables, task)
        self.world_model.set_variability(core_variables, variabilities)

    # helpers
    def prepare_action(self, action):
        action_name = action['function_name']
        action_definition = self.action_space[action_name]
        action['description'] = action_definition['description']
        
        return action_name, action_definition

    def _make_action_hashable(self, action):
        """Convert action dict into a hashable tuple for caching.
        
        Returns:
            tuple: (function_name, tuple(arguments))
        """
        return (action['function_name'], tuple(action['arguments']))

    def _handle_new_state(self, effective_state, previous_state, action=None):
        """
        Handle logic for when a new effective state is discovered.
        Determines impact on core variables and updates the world model.

        Args:
            effective_state: The new effective state to handle
            previous_state: The previous state that led to this one (if applicable)
            action: The action that led to this state (if applicable)
        """
        # Determine potential relations between the new effective state and core variables
        rel = self.state_reasoning.core_vars_impact(
            effective_state,
            self.core_variables,
            self.task
        )
        
        # Always create the base nodes and edges for the new state
        new_nodes, new_edges = prepare_base_graph_update(effective_state, rel)
        
        # If we have transition information, add the transition edge
        if action is not None:
            transition_edge = prepare_transition_edge(previous_state, effective_state, action)
            new_edges.append(transition_edge)
            
        self.world_model.add_nodes_and_edges(new_nodes, new_edges)

    # core functionality
    def get_effective_state(self, observation):
        """
        Get the effective state of the world model based on the current observation.
        observation is what the agent observes

        effective_state is the state of the internal world model, 
        which only changes when there is a significant change in the external world 
        that could affect the core variables.
        for example, if the agent is on a shopping site and the agent is merely browsing, 
        the effective state does not change.
        """
        # Attempt to retrieve effective state from cache
        if observation in self.world_model.effective_state_cache:
            logger.info("Retrieved effective state from cache.")
            return self.world_model.effective_state_cache[observation]
        
        logger.info("Reasoning out effective state (not found in cache).")
        # Based on past effective state, get neighbor effective states 
        # and unlinked nodes and return itself too
        previous_effective_state = self.effective_state
        candidates = self.world_model.get_candidate_effective_states(previous_effective_state)
        effective_state, is_new = self.state_reasoning.infer_effective_state(
            candidates,
            observation,
            self.core_variables,
            self.task
        )

        self.effective_state = effective_state
        if is_new:
            self._handle_new_state(effective_state, previous_effective_state)
            
        self.world_model.effective_state_cache[observation] = effective_state
        # self.world_model.store_effective_state(observation, effective_state, is_new)
        
        return effective_state
    
    # def check_always_safe(self, action_name, action_definition):
    #     if action_name in self.world_model.always_safe_cache:
    #         return self.world_model.always_safe_cache[action_name]
        
    #     always_safe = self.action_safety.infer_always_safe(
    #         action_definition, 
    #         self.task, 
    #         self.initial_state, 
    #         self.core_variables
    #     )
        
    #     self.world_model.always_safe_cache[action_name] = always_safe
    #     return always_safe

    def get_param_range(self, action_name, action_definition):
        if not action_definition['parameters']:
            return None
        
        if action_name in self.world_model.param_range_cache:
            return self.world_model.param_range_cache[action_name]

        usual_param_range = self.action_safety.infer_param_range(
            action_definition, 
            self.task, 
            # self.initial_state
        )

        self.world_model.param_range_cache[action_name] = usual_param_range
        return usual_param_range

    def assert_param_range(self, action, usual_param_range):
        if usual_param_range is None:
            return True
        
        is_within_range = self.action_safety.assert_param_range(
            action, 
            self.task, 
            self.initial_state, 
            usual_param_range
        )
        return is_within_range

    def check_core_var_violation(self, core_variables, effective_state, observation, action):
        """
        deliberate reasoning of impact on core variables and detecting violations
        """
        for core_variable in core_variables:
            expected_variation, expected_context = self.world_model.get_variability(core_variable)

            # Fine-grained reasoning of the magnitude of change
            actual_variation, variation_context = self.variability_reasoning.get_actual_variation(
                effective_state, 
                observation, 
                action, 
                core_variable
            )

            if self.variability_reasoning.check_violation(
                actual_variation, 
                expected_variation, 
                core_variable,
                variation_context,
                expected_context
            ):
                logger.info(f"Action is not safe for core variable: {core_variable}")
                return True

        return False

    def one_step_lookahead(self, effective_state, action, state_edges):
        """
        since we are going to take the action, update effective state, 
        and reason to check if next effective state can potentially impact core variables
        update world model
        """
        next_effective_state, is_new = self.state_reasoning.next_effective_state(
            effective_state, 
            action, 
            state_edges, 
            self.task, 
            self.core_variables,
        )
        self.effective_state = next_effective_state

        if is_new:
            self._handle_new_state(next_effective_state, effective_state, action)

    # main functionality
    def is_action_safe(self, observation, action):
        """
        Args:
            action (dict): The action that the agent is about to take
                It will have all the keys mentioned in the example below:
                {
                    "function_name": "hover",
                    "arguments": ["id"],
                }
        """
        action_name, action_definition = self.prepare_action(action)

        ### Part 1: Action-only checks ###
        # Temporarily disable always safe check, might have edge cases
        # if self.check_always_safe(action_name, action_definition):
        #     print(f"Action {action_name} is always safe.")
        #     return True

        usual_param_range = self.get_param_range(action_name, action_definition)
        if not self.assert_param_range(action, usual_param_range):
            logger.info("Action arguments are outside the usual range.")
            return False


        ### Part 2: State-action based checks ###
        effective_state = self.get_effective_state(observation)
        action_key = self._make_action_hashable(action)

        if (effective_state, action_key) in self.world_model.cache:
            logger.info("Retrieved result from world model cache.")
            return self.world_model.cache[(effective_state, action_key)]

        logger.info("Performing reasoning as State-action result not found in cache.")
        # Get neighbors and edges from the current effective state to determine impact
        _, edges = self.world_model.get_successor_data(effective_state)

        core_vars = [edge['obj'] for edge in edges if edge['obj'] in self.core_variables]
        if self.check_core_var_violation(core_vars, effective_state, observation, action):
            return False
        
        logger.info("All checks passed, storing State-action result in cache.")
        self.world_model.cache[(effective_state, action_key)] = True
        

        ### Part 3: One step lookahead now that the action will be taken ###
        state_edges = [edge for edge in edges if edge['obj'] not in self.core_variables]
        self.one_step_lookahead(effective_state, action, state_edges)

        # Future: use this as warning if path length is short (so it is close to affecting core variables)
        # paths = self.world_model.find_paths_to_core_variables(effective_state, action, self.core_variables)
        
        return True
