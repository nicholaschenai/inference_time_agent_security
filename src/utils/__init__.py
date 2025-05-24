action_str_template = """
{function_name}({formatted_args})
Description: {description}
"""


def format_action_str(action_details: dict) -> str:
    """
    Format action details into a string representation.

    Args:
        action_details (dict): The action details containing function_name, arguments, and description.
            Example: {
                "function_name": "hover",
                "arguments": ["id"],
                "description": "Hover over an element with id."
            }

    Returns:
        str: A formatted string representation of the action.
    """
    formatted_args = ', '.join(action_details.get('arguments', action_details.get('parameters', [])))
    return action_str_template.format(formatted_args=formatted_args, **action_details)


def prepare_base_graph_update(state, potential_relations):
    """
    Prepare the basic graph updates for a new state and its potential relations.
    
    Args:
        state: The state to add to the graph
        potential_relations: List of potential relations this state has with other entities
    
    Returns:
        tuple: (new_nodes, new_edges) to be added to the graph
    """
    new_nodes = [{'node_id': state, 'node_type': 'state'}]
    new_edges = []

    for potential_relation in potential_relations:
        new_edges.append({
            'subject': state,
            'relation': potential_relation['relation'],
            'obj': potential_relation['obj'],
        })

    return new_nodes, new_edges

def prepare_transition_edge(from_state, to_state, action):
    """
    Prepare the transition edge between two states based on an action.
    
    Args:
        from_state: The source state
        to_state: The destination state
        action: The action that caused the transition
        
    Returns:
        dict: The transition edge
    """
    return {
        'subject': from_state,
        'relation': 'transition',
        'obj': to_state,
        'action': format_action_str(action)
    }

# def prepare_graph_update(effective_state, next_effective_state, action, potential_relations):
#     """
#     Prepare complete graph update including both state and transition information.
    
#     Args:
#         effective_state: The current state
#         next_effective_state: The new state to add
#         action: The action that caused the transition
#         potential_relations: List of potential relations the new state has
        
#     Returns:
#         tuple: (new_nodes, new_edges) to be added to the graph
#     """
#     new_nodes, new_edges = prepare_base_graph_update(next_effective_state, potential_relations)
#     transition_edge = prepare_transition_edge(effective_state, next_effective_state, action)
#     new_edges.append(transition_edge)
    
#     return new_nodes, new_edges
