from cognitive_base.utils.database.graph_db.nx_db import NxDb
import networkx as nx


class WorldModel:
    def __init__(self, initial_state, verbose=False, **kwargs):
        # Initialize the graph database
        self.graph_db = NxDb()  # self.graph_db.graph is a networkx graph
        self.verbose = verbose
        self.core_variables = []
        self.cache = {}  # Cache to store safety results
        self.effective_state_cache = {}  # Cache to store effective states
        self.always_safe_actions = set()  # Set of actions that are always safe
        self.analyzed_actions = set()  # Set of actions that have been analyzed for always safe
        self.param_ranges = {}  # Dictionary to store parameter ranges

        self.graph_db.add_node(initial_state, {'node_type': 'state'})

    def set_variability(self, core_variables, variabilities):
        """
        Set the variability of core variables in the world model.
        
        Args:
            core_variables (List[str]): List of core variable names.
            variabilities (List[str]): List of variability information.
        """
        for core_variable, variability in zip(core_variables, variabilities):
            node_id = core_variable
            attributes = {
                'node_type': 'core_variable',
                'variability': variability
            }
            self.graph_db.add_node(node_id, verbose=self.verbose, **attributes)
        self.core_variables = core_variables

    def query_cache(self, observation, action):
        """
        Query the cache to determine if the action is safe given the observation.
        
        Args:
            observation (str): The current observation.
            action (str): The action to check.
        
        Returns:
            bool: True if the action is safe, False otherwise.
        """
        # Check the cache for the specific observation-action pair
        return self.cache.get((observation, action), None)

    def store_cache(self, observation, action, is_safe):
        """
        Store the result of the action safety check in the cache.
        
        Args:
            observation (str): The current observation.
            action (str): The action to store.
            is_safe (bool): The result of the safety check.
        """
        self.cache[(observation, action)] = is_safe

    def add_always_safe_action(self, action):
        """
        Add an action to the set of actions that are always considered safe.
        
        Args:
            action (str): The action to add.
        """
        self.always_safe_actions.add(action)

    def add_nodes_and_edges(self, nodes=None, edges=None):
        """
        Add nodes and edges to the world model.

        Args:
            nodes (List[dict]): List of node dictionaries with 'node_id' and other attributes.
            edges (List[dict]): List of edge dictionaries with 'subject', 'relation', 'object', and other attributes.
        """
        if nodes:
            for node in nodes:
                node_id = node.pop('node_id')
                self.graph_db.add_node(node_id, verbose=self.verbose, **node)

        if edges:
            for edge in edges:
                subject = edge.pop('subject')
                obj = edge.pop('obj')
                relation = edge.pop('relation')
                self.graph_db.add_edge(subject, relation, obj, verbose=self.verbose, **edge)

    def find_paths_to_core_variables(self, action, core_variables):
        """
        Find the shortest paths from the action node to each core variable node.
        
        Args:
            action (str): The action node ID.
            core_variables (List[str]): List of core variable node IDs.
        
        Returns:
            List[List[Tuple[str, str]]]: A list of lists containing node and edge IDs in path order.
        """
        # TODO: not in use, future work. so we know how far we are from affecting core variables
        paths = []
        for core_variable in core_variables:
            try:
                # Find the shortest path from the action node to the core variable node
                path = nx.shortest_path(self.graph_db.graph, source=action, target=core_variable)
                paths.append(path)
            except nx.NetworkXNoPath:
                # No path found, continue to the next core variable
                continue
        return paths

    def get_variability(self, core_variable):
        """
        Get the variability of a core variable.
        
        Args:
            core_variable (str): The core variable node ID.
        
        Returns:
            str: The variability of the core variable.
        """
        return self.graph_db.get_node(core_variable)['variability']

    def query_effective_state_cache(self, observation):
        """
        Query the cache to retrieve the effective state for a given observation.
        
        Args:
            observation (str): The current observation.
        
        Returns:
            str: The effective state if found, None otherwise.
        """
        return self.effective_state_cache.get(observation, None)

    def store_effective_state_cache(self, observation, effective_state):
        """
        Store the effective state in the cache for a given observation.
        
        Args:
            observation (str): The current observation.
            effective_state (str): The effective state to store.
        """
        self.effective_state_cache[observation] = effective_state

    def get_candidate_effective_states(self, previous_effective_state):
        """
        Get candidate effective states based on the previous_effective_state.
        
        Args:
            previous_effective_state (str):  node ID.
        
        Returns:
            List[str]: A list of candidate effective state node IDs.
        """
        candidate_states = set()

        # Add the original node
        candidate_states.add(previous_effective_state)

        # Add neighboring nodes with node_type == 'state'
        neighbors = self.graph_db.graph.successors(previous_effective_state)
        for neighbor in neighbors:
            if self.graph_db.get_node(neighbor)['node_type'] == 'state':
                candidate_states.add(neighbor)

        # Add state nodes with no neighbors
        all_state_nodes = self.graph_db.get_nodes_by_attribute('node_type', 'state')
        for state_node in all_state_nodes:
            if not self.graph_db.graph.successors(state_node):
                candidate_states.add(state_node)

        return list(candidate_states)

    def get_outgoing_neighbors_and_edges(self, node_id):
        """
        Get the outgoing neighbors and the edges connecting them from the given node.
        
        Args:
            node_id (str): The node ID from which to find outgoing neighbors.
        
        Returns:
            Tuple[Dict[str, dict], List[dict]]: A dictionary of neighboring nodes with their attributes and a list of edges.
        """
        neighbors_dict = {}
        edges = []

        # Iterate over the outgoing edges from the node
        for neighbor in self.graph_db.graph.successors(node_id):
            # Get the attributes of the neighbor node
            neighbor_attributes = self.graph_db.get_node(neighbor)
            neighbors_dict[neighbor] = neighbor_attributes

            # Get the edge attributes
            edge_attributes = self.graph_db.graph.get_edge_data(node_id, neighbor)
            edges.append({
                'subject': node_id,
                'relation': edge_attributes.get('relation', ''),
                'obj': neighbor,
                **edge_attributes
            })

        return neighbors_dict, edges

    def get_param_range(self, function_name):
        """
        Retrieve the usual parameter range for a given function name.
        
        Args:
            function_name (str): The name of the function.
        
        Returns:
            str: The usual parameter range for the function.
        """
        return self.param_ranges.get(function_name, "")

    def store_param_range(self, function_name, param_range):
        """
        Store the usual parameter range for a given function name.
        
        Args:
            function_name (str): The name of the function.
            param_range (str): The parameter range to store.
        """
        self.param_ranges[function_name] = param_range
