import logging

logger = logging.getLogger("logger")

from src.utils import format_action_str

class BaseEnv:
    def __init__(self):
        # Initialize state 
        self.state = None
        self.initial_state = "Initial environment state"
        self.action_space = None  # Should be defined by child classes
        self.observation_space = None  # Should be defined by child classes
        
        # Separate tracking of action history
        self.action_history = []
        
    def _get_formatted_observation(self):
        """
        Creates a formatted string combining initial state and action history.
        This maintains the desired functionality while keeping state simple.
        
        Returns:
            str: A formatted string showing initial state and action history
        """
        if self.state is None:
            return "Environment not initialized"
            
        state_lines = [
            "=== Environment State ===",
            f"Starting from: {self.initial_state}",
        ]
        
        # Add action history if any actions have been taken
        if self.action_history:
            state_lines.append("\nActions taken:")
            for idx, action in enumerate(self.action_history, 1):
                state_lines.append(f"{idx}. {action}")
        
        return "\n".join(state_lines)

    def _get_observation(self):
        """
        Get current observation of the environment.
        In this base implementation, returns a formatted string of state and action history.
        Child classes should override this for their specific observation needs.
        
        Returns:
            observation: The current observation of the environment
        """
        return self._get_formatted_observation()

    def reset(self):
        """
        Reset the environment to initial state and return initial observation.
        
        Returns:
            observation: Initial observation of the environment
        """
        # Reset state (Gym-style)
        self.state = self.initial_state
        # Reset action history
        self.action_history = []
        
        logger.info("Environment reset.")
        return self._get_observation(), 0, False, {}

    def step(self, action):
        """
        Execute action in the environment.
        
        Args:
            action: The action to take in the environment
            
        Returns:
            observation: Current observation
            reward: Reward obtained from the action
            done: Whether the episode has ended
            info: Additional information
        """
        # Format and log action
        action_str = format_action_str(action)
        logger.info(f"Executing action: {action_str}")
        
        # Initialize if needed
        if self.state is None:
            self.reset()
            
        # Track action in history
        self.action_history.append(action_str)
        
        # Update state (in this demo, state doesn't change)
        # Child classes should implement their own state transitions
        
        # Get observation (combines state and history)
        observation = self._get_observation()
        
        # Default reward structure
        reward = 0.0
        
        # Check if episode is done
        done = self.state == "terminal_state"  # Can be overridden by child classes
        
        # Additional info
        info = {
            "last_action": action_str,
            "action_history_length": len(self.action_history)
        }
        
        return observation, reward, done, info
