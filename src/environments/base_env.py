class BaseEnv:
    def __init__(self):
        # Initialize any necessary components or variables
        self.state = None

    def reset(self):
        """
        Reset the environment to an initial state and return the initial observation.
        """
        # Logic to reset the environment
        self.state = "initial_state"
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
        return f"observation_based_on_{self.state}"

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
