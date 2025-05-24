import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
import textwrap

from cognitive_base.utils.database.graph_db.nx_db import NxDb


class WorldModel:
    def __init__(self, initial_state, verbose=False, **kwargs):
        # settings
        self.verbose = verbose
        
        # Initialize the graph database
        self.graph_db = NxDb()  # self.graph_db.graph is a networkx graph
        self.graph_db.add_node(initial_state, {'node_type': 'state'})
        
        # params
        self.core_variables = []

        # cache
        self.cache = {}  # Cache to store safety results for (effective_state, action)
        self.effective_state_cache = {}  # Cache to store effective states
        self.always_safe_cache = {}  # Set of actions that are always safe
        self.param_range_cache = {}  # Dictionary to store parameter ranges

    def set_variability(self, core_variables, variability_data):
        """
        Set the variability of core variables in the world model.
        
        Args:
            core_variables (List[str]): List of core variable names.
            variability_data (List[dict]): List of dictionaries containing 'variability' and 'reasoning' fields.
        """
        for core_variable, data in zip(core_variables, variability_data):
            node_id = core_variable
            attributes = {
                'node_type': 'core_variable',
                'variability': data['variability'],
                'reasoning': data['reasoning']
            }
            self.graph_db.add_node(node_id, verbose=self.verbose, **attributes)
        self.core_variables = core_variables

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
                self.graph_db.add_edge(subject, obj, relation, verbose=self.verbose, **edge)

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
        Get the variability and reasoning of a core variable.
        
        Args:
            core_variable (str): The core variable node ID.
        
        Returns:
            tuple: A tuple containing (variability, reasoning) for the core variable.
        """
        node = self.graph_db.get_node(core_variable)
        return node['variability'], node['reasoning']

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
            neighbor_attributes = self.graph_db.get_node(neighbor)
            # node_type = 
            # if not node_type:
            #     print(f"Node {neighbor} has no node_type attribute")
                
            if neighbor_attributes.get('node_type') == 'state':
                candidate_states.add(neighbor)

        # Add state nodes with no neighbors
        all_state_nodes = self.graph_db.get_nodes_by_attribute('node_type', 'state')
        for state_node in all_state_nodes:
            if not self.graph_db.graph.successors(state_node):
                candidate_states.add(state_node)

        return list(candidate_states)

    def get_successor_data(self, node_id):
        """
        Get the outgoing neighbors and the edges connecting them from the given node.
        Also includes the current node as a self-referential edge to represent "no change" state.
        
        Args:
            node_id (str): The node ID from which to find outgoing neighbors.
        
        Returns:
            Tuple[Dict[str, dict], List[dict]]: A dictionary of neighboring nodes with their attributes and a list of edges.
        """
        neighbors_dict = {}
        edges = []

        # Add the current node as a self-referential edge to represent "no change"
        # current_node_attributes = self.graph_db.get_node(node_id)
        # neighbors_dict[node_id] = current_node_attributes
        edges.append({
            'subject': node_id,
            'relation': 'no_change',
            'obj': node_id
        })

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

    def visualize_graph(self, save_path=None, description=None):
        """
        Visualize the world model graph with node colors based on type and edge labels showing relations.
        
        Args:
            save_path (str, optional): Path to save the visualization. If None, displays the plot.
            description (str, optional): Description to append to the title.
        """
        # Create a new figure with a balanced size (too large will make text tiny when rescaled)
        plt.figure(figsize=(10, 10))  # Reduced size for better scaling on web
        
        # Get unique node types for coloring and ensure 'unknown' is included
        node_types = set(nx.get_node_attributes(self.graph_db.graph, 'node_type').values())
        node_types.add('initial_state')  # Add unknown type explicitly
        color_map = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(node_types)))
        type_to_color = dict(zip(node_types, color_map))
        
        # Create node colors list with explicit unknown handling
        node_colors = []
        for node in self.graph_db.graph.nodes():
            node_type = self.graph_db.graph.nodes[node].get('node_type', 'initial_state')
            node_colors.append(type_to_color[node_type])
        
        # Create node labels with name and type, wrap long node names
        node_labels = {}
        for node, attrs in self.graph_db.graph.nodes(data=True):
            node_name = str(node)
            if len(node_name) > 20:  # Wrap long node names
                node_name = textwrap.fill(node_name, width=20, break_long_words=True)
            node_labels[node] = f"{node_name}\n({attrs.get('node_type', 'initial_state')})"
        
        # Create edge labels with text wrapping for long relations
        edge_labels = {}
        edge_colors = []  # List to store edge colors
        for edge in self.graph_db.graph.edges():
            edge_data = self.graph_db.graph.get_edge_data(*edge)
            relation = edge_data.get('relation', '')
            
            # For transition relations, include the action
            if relation == 'transition' and 'action' in edge_data:
                label = f"transition:\n{edge_data['action']}"
                edge_colors.append('darkblue')  # Dark blue for transition edges
            else:
                label = str(relation)
                edge_colors.append('gray')  # Gray for other edges
            
            # Wrap long text with smaller width to prevent overlap
            if len(label) > 25:  # Reduced width for edge labels
                wrapped_label = textwrap.fill(label, width=25, break_long_words=True)
            else:
                wrapped_label = label
            edge_labels[edge] = wrapped_label
        
        # Draw the graph with adjusted layout - increase spacing between nodes
        try:
            # Try Kamada-Kawai layout first as it often gives better spacing
            pos = nx.kamada_kawai_layout(self.graph_db.graph, scale=10.0)
        except:
            # Fallback to spring layout if Kamada-Kawai fails
            pos = nx.spring_layout(self.graph_db.graph, k=5.0, iterations=100, scale=2.0)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph_db.graph, pos, node_color=node_colors, 
                             node_size=4000, alpha=0.7)
        
        # Draw edges with increased curvature and spacing, using edge colors
        edge_list = list(self.graph_db.graph.edges())
        nx.draw_networkx_edges(self.graph_db.graph, pos, edge_color=edge_colors,
                             arrows=True, arrowsize=20,
                             width=1.5,
                             connectionstyle='arc3,rad=0.3',
                             edgelist=edge_list)  # Specify edge list to match colors
        
        # Draw node labels with balanced font size
        nx.draw_networkx_labels(self.graph_db.graph, pos, node_labels, font_size=10)
        
        # Draw edge labels with adjusted position and color
        for edge, label in edge_labels.items():
            edge_data = self.graph_db.graph.get_edge_data(*edge)
            # Set color based on relation type
            color = 'darkblue' if edge_data.get('relation') == 'transition' else 'black'
            
            # Draw each edge label separately to apply different colors
            nx.draw_networkx_edge_labels(self.graph_db.graph, pos,
                                       edge_labels={edge: label},  # Only draw one label at a time
                                       font_size=8,
                                       font_color=color,
                                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5),
                                       rotate=False)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, label=node_type, 
                                    markersize=10)
                         for node_type, color in type_to_color.items()]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        
        # Create title with optional description
        title = "World Model Graph Visualization"
        if description:
            title += f" for {description}"
        plt.title(title, fontsize=14, weight='bold', pad=20)  # Added padding to prevent overlap
        plt.axis('off')
        
        # Adjust layout to prevent legend overlap and ensure all elements are visible
        plt.tight_layout()
        # Add extra space on the right for the legend
        plt.subplots_adjust(right=0.85, top=0.95)  # Adjusted top margin for title
        
        # Add extra margin to ensure no cutoff
        plt.margins(x=0.2, y=0.2)
        
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white', 
                       pad_inches=0.5)  # Added padding to prevent cutoff
            plt.close()
        else:
            plt.show()
