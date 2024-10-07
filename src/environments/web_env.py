"""
Modeled after WebArena, but simplified
https://github.com/web-arena-x/webarena/tree/main
"""
from .base_env import BaseEnv

webarena_actions_list = [
    {
        "function_name": "click",
        "arguments": ["id"],
        "description": "This action clicks on an element with a specific id on the webpage."
    },
    {
        "function_name": "type",
        "arguments": ["id", "content", "press_enter_after=0|1"],
        "description": 'Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.'
    },
    {
        "function_name": "hover",
        "arguments": ["id"],
        "description": "Hover over an element with id."
    },
    {
        "function_name": "press",
        "arguments": ["key_comb"],
        "description": "Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v)."
    },
    {
        "function_name": "scroll",
        "arguments": ["direction=down|up"],
        "description": "Scroll the page up or down."
    },
    {
        "function_name": "new_tab",
        "arguments": [],
        "description": "Open a new, empty browser tab."
    },
    {
        "function_name": "tab_focus",
        "arguments": ["tab_index"],
        "description": "Switch the browser's focus to a specific tab using its index."
    },
    {
        "function_name": "close_tab",
        "arguments": [],
        "description": "Close the currently active tab."
    },
    {
        "function_name": "goto",
        "arguments": ["url"],
        "description": "Navigate to a specific URL."
    },
    {
        "function_name": "go_back",
        "arguments": [],
        "description": "Navigate to the previously viewed page."
    },
    {
        "function_name": "go_forward",
        "arguments": [],
        "description": "Navigate to the next page (if a previous 'go_back' action was performed)."
    }
]

# Transform the list into a dictionary
webarena_actions = {action["function_name"]: {k: v for k, v in action.items()} for action in webarena_actions_list}

class WebEnvironment(BaseEnv):
    def __init__(self, initial_state, **kwargs):
        # Initialize any necessary components or variables
        self.state = initial_state
        self.action_space = webarena_actions

    def reset(self):
        """
        Reset the environment to an initial state and return the initial observation.
        """
        # Logic to reset the environment
        print("Environment reset.")
        observation = self.get_observation()
        reward = 0  # Initial reward
        done = False  # Initial done state
        info = {}  # Additional info
        return observation, reward, done, info

    def get_observation(self):
        """
        Get the current observation from the environment.
        """
        # Logic to get the current observation based on the state
        print("Getting current observation...")
        return self.state

    def step(self, action):
        """
        Execute the given action in the environment and return the new observation.
        """
        # Logic to update the state based on the action
        print(f"Executing action: {action}")
        self.state = f"state_after_{action}"
        observation = self.get_observation()
        reward = 1  # Example reward
        done = self.state == "terminal_state"  # Example condition for done
        info = {}  # Additional info
        return observation, reward, done, info
