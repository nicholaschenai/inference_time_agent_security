from models.world_model import WorldModel
from reasoning.generic_reasoning import GenericReasoning
from reasoning.action_safety import ActionSafetyReasoning


class SafetyModule:
    def __init__(self, initial_state, action_space, **kwargs):
        self.world_model = WorldModel(initial_state, **kwargs)
        self.reasoning = GenericReasoning(**kwargs)
        self.action_safety = ActionSafetyReasoning(**kwargs)
        self.core_variables = []
        self.initial_state = initial_state
        self.effective_state = initial_state
        self.task = None
        self.action_space = action_space

    def analyze_core_variability(self, core_variables, task):
        """
        Analyze the core variables to determine the typical variation given the task.
        For example, if the task is to buy a product, then the user's money should only change in a specific range.
        """
        self.core_variables = core_variables
        self.task = task
        variabilities = self.reasoning.analyze_core_variability(core_variables, task)
        self.world_model.set_variability(core_variables, variabilities)

    def get_effective_state(self, observation):
        """
        Get the effective state of the world model based on the observation.
        observation is what the agent observes

        effective_state is the state of the internal world model, which only changes when there is a significant change in the external world that could affect the core variables.
        for example, if the agent is on a shopping site and the agent is merely browsing, the effective state does not change.
        """
        # Attempt to retrieve effective state from cache
        effective_state = self.world_model.query_effective_state_cache(observation)
        if effective_state is not None:
            print("Retrieved effective state from cache.")
            return effective_state

        # If observation is not in effective state cache, attempt to reason effective state
        print("Reasoning effective state as it is not found in cache.")
        
        # Based on past effective state, get neighbor effective states and unlinked nodes and return itself too
        candidate_effective_states = self.world_model.get_candidate_effective_states(self.effective_state)
        
        # Reasoning to see if any of these effective states match observation
        # If no effective states match observation, use reasoning to create new effective state
        effective_state, is_new = self.reasoning.find_matching_effective_state(candidate_effective_states, observation, self.core_variables, self.task)
        self.effective_state = effective_state
        self.world_model.store_effective_state_cache(observation, effective_state)
        if is_new:
            self.world_model.add_nodes_and_edges([{'node_id': effective_state, 'node_type': 'state'}], [])

        return effective_state

    def is_action_safe(self, observation, action):
        """
        Determine if the given action is safe based on the core variables.
        First, attempt to retrieve the result from the cache. If not found,
        perform reasoning, store the result in the cache, and return it.

        Args:
            action (dict): The action that the agent is about to take
                It will have all the keys mentioned in the example below:
                {
                    "function_name": "hover",
                    "arguments": ["id"],
                }
        """
        action['description'] = self.action_space[action['function_name']]['description']
        # Check if action has already been analyzed for always safe given the task and initial_state

        # MEGA NOTE: if no time, focus on out of bounds goto example
        usual_param_range = None
        action_name = action['function_name']
        if action_name in self.world_model.analyzed_actions:
            if action_name in self.world_model.always_safe_actions:
                return True
            if action['arguments']:
                usual_param_range = self.world_model.get_param_range(action_name)
        else:
            always_safe = self.action_safety.infer_always_safe(self.action_space[action_name], self.task, self.initial_state, self.core_variables)
            self.world_model.analyzed_actions.add(action_name)
            if always_safe:
                self.world_model.add_always_safe_action(action_name)
                return True
            if action['arguments']:
                usual_param_range = self.action_safety.infer_usual_param_range(self.action_space[action_name], self.task, self.initial_state)
                self.world_model.store_param_range(action_name, usual_param_range)

        # Check if the action is within the usual range of input params
        # for example, if the action is to goto(url), and the task is to buy a product, 
        # the url should be within the shopping site.
        if usual_param_range is not None:
            if not self.action_safety.is_param_within_usual_range(action, self.task, self.initial_state, usual_param_range):
                print("Action parameters are outside the usual range.")
                return False

        # Get the effective state based on the observation
        effective_state = self.get_effective_state(observation)

        # Query the world model for cached result
        cached_result = self.world_model.query_cache(effective_state, action)
        if cached_result is not None:
            print("Retrieved result from world model cache.")
            return cached_result

        # Perform reasoning if not found in cache
        print("Performing reasoning as result not found in cache.")
        
        # Get neighbors and edges from the current effective state
        neighbors_dict, edges = self.world_model.get_outgoing_neighbors_and_edges(effective_state)

        # Check if core variables are in the neighbors and if there is a violation of core variable bounds
        state_edges = []
        for edge in edges:
            if edge['obj'] in self.core_variables:
                core_variable = edge['obj']
                # Fine-grained reasoning of the magnitude of change
                actual_variation = self.reasoning.get_actual_variation(effective_state, observation, action, core_variable)
                expected_variation = self.world_model.get_variability(core_variable)
                if self.reasoning.is_core_variation_beyond_bounds(actual_variation, expected_variation, core_variable):
                    print(f"Action is not safe for core variable: {core_variable}")
                    return False
            else:
                state_edges.append(edge)

        # Use reasoning module to determine next effective state
        next_effective_state, is_new = self.reasoning.get_next_effective_state(
            effective_state, action, neighbors_dict, state_edges, self.task, self.core_variables
        )
        self.effective_state = next_effective_state

        if is_new:
            new_nodes = [{'node_id': next_effective_state, 'node_type': 'state'}]
            new_edges = [{
                'subject': effective_state,
                'relation': 'transition',
                'obj': next_effective_state,
                'action': action_name
            }]

            # Determine potential relations between the new effective state and core variables
            potential_relations = self.reasoning.can_state_affect_core_variables(next_effective_state, self.core_variables, self.task)
            for potential_relation in potential_relations:
                new_edges.append({
                    'subject': next_effective_state,
                    'relation': potential_relation['relation'],
                    'obj': potential_relation['obj'],
                })

            # Add new nodes and edges to the world model
            self.world_model.add_nodes_and_edges(new_nodes, new_edges)

        # Future: use this as warning if path length is short (so it is close to affecting core variables)
        # paths = self.world_model.find_paths_to_core_variables(effective_state, action, self.core_variables)
        
        # Store the result in the world model
        self.world_model.store_cache(effective_state, action, True)
        return True
