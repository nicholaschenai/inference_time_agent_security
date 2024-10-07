from .base_agent import BaseAgent


class WebAgent(BaseAgent):
    def __init__(self, scripted_actions=None, **kwargs):
        super().__init__()
        self.scripted_actions = scripted_actions
        self.current_action_index = 0

    def decide(self, observation):
        """
        Decide on the next action based on the scripted actions list.
        """
        if self.current_action_index < len(self.scripted_actions):
            action = self.scripted_actions[self.current_action_index]
            self.current_action_index += 1
            return action
        return None  # No more actions to perform
