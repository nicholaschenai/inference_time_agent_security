"""
Modeled after WebArena, but simplified
https://github.com/web-arena-x/webarena/tree/main
"""
import logging
import re

from src.utils import format_action_str
from .base_env import BaseEnv

logger = logging.getLogger("logger")

webarena_actions_list = [
    {
        "function_name": "click",
        "parameters": ["id"],
        "description": "This action clicks on an element with a specific id on the webpage."
    },
    {
        "function_name": "type",
        "parameters": ["id", "content", "press_enter_after=0|1"],
        "description": 'Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0.'
    },
    {
        "function_name": "hover",
        "parameters": ["id"],
        "description": "Hover over an element with id."
    },
    {
        "function_name": "press",
        "parameters": ["key_comb"],
        "description": "Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v)."
    },
    {
        "function_name": "scroll",
        "parameters": ["direction=down|up"],
        "description": "Scroll the page up or down."
    },
    {
        "function_name": "new_tab",
        "parameters": [],
        "description": "Open a new, empty browser tab."
    },
    {
        "function_name": "tab_focus",
        "parameters": ["tab_index"],
        "description": "Switch the browser's focus to a specific tab using its index."
    },
    {
        "function_name": "close_tab",
        "parameters": [],
        "description": "Close the currently active tab."
    },
    {
        "function_name": "goto",
        "parameters": ["url"],
        "description": "Navigate to a specific URL."
    },
    {
        "function_name": "go_back",
        "parameters": [],
        "description": "Navigate to the previously viewed page."
    },
    {
        "function_name": "go_forward",
        "parameters": [],
        "description": "Navigate to the next page (if a previous 'go_back' action was performed)."
    }
]

# Transform the list into a dictionary
webarena_actions = {action["function_name"]: {k: v for k, v in action.items()} for action in webarena_actions_list}

class WebEnvironment(BaseEnv):
    def __init__(self, initial_state="shopping_site", **kwargs):
        super().__init__()
        self.initial_state = initial_state
        self.action_space = webarena_actions
        self.cart_items = []  # List to store cart items
        self.current_page = initial_state
        self.selected_item = None  # Track currently selected item
        
    def _extract_price(self, item_id):
        """Extract price from item ID if present."""
        price_match = re.search(r'PRICE_(\d+)dollars', item_id)
        if price_match:
            return int(price_match.group(1))
        return 0
        
    def _get_cart_summary(self):
        """Generate a summary of cart contents and total."""
        if not self.cart_items:
            return "Cart is empty"
            
        summary = ["=== Cart Contents ==="]
        total = 0
        for item in self.cart_items:
            price = self._extract_price(item)
            total += price
            summary.append(f"- {item} (${price})")
        summary.append(f"\nTotal: ${total}")
        return "\n".join(summary)

    def _get_observation(self):
        """
        Get current observation including website state and cart contents.
        """
        website_state = f"=== Current Page: {self.current_page} ==="
        
        # Add selected item info if any
        # if self.selected_item:
        #     website_state.append(f"Currently Selected: {self.selected_item}")
            
        cart_summary = self._get_cart_summary()
        
        return f"{website_state}\n{cart_summary}\n\nAction History:\n" + "\n".join(
            f"{i+1}. {action}" for i, action in enumerate(self.action_history)
        )

    def step(self, action):
        """
        Execute action in the web environment.
        
        Args:
            action: Dict with function_name and arguments
        """
        action_str = format_action_str(action)
        logger.info(f"Executing action: {action_str}")

        if self.state is None:
            self.reset()

        self.action_history.append(action_str)
        
        # Handle different actions
        if action["function_name"] == "goto":
            self.current_page = action["arguments"][0]
            self.selected_item = None  # Clear selection when changing pages
        elif action["function_name"] == "click":
            element_id = action["arguments"][0]
            if "PRICE" in element_id:
                # Select the item
                self.selected_item = element_id
            elif element_id == "add_to_cart_button" and self.selected_item:
                # Add currently selected item to cart
                self.cart_items.append(self.selected_item)
                self.selected_item = None  # Clear selection after adding to cart
        
        observation = self._get_observation()
        reward = 0
        # done = "checkout_button" in action_str
        done = self.state == "terminal_state"

        info = {
            "last_action": action_str,
            "cart_total": sum(self._extract_price(item) for item in self.cart_items)
        }
        
        return observation, reward, done, info

    def reset(self):
        """Reset the environment and clear cart."""
        self.current_page = self.initial_state
        self.cart_items = []
        self.selected_item = None
        observation, _, _, _ = super().reset()
        return observation, 0, False, {}
