import importlib
import argparse
from safety_module import SafetyModule
from config import get_config

from cognitive_base.utils import lm_cache_init


def main(args):
    # Get configuration for the chosen setting
    config = get_config(args.setting_name)

    # Dynamically import the environment and agent based on the setting
    env_cls = getattr(importlib.import_module(config['environment']), config['env_class'])
    agent_cls = getattr(importlib.import_module(config['agent']), config['agent_class'])

    # Initialize components
    kwargs = vars(args)
    agent = agent_cls(scripted_actions=config['scripted_actions'], **kwargs)
    environment = env_cls(**kwargs, **config)
    safety_module = SafetyModule(action_space=environment.action_space, **kwargs, **config)

    # Example task and core variables
    task = config['task']
    core_variables = config['core_variables']

    # Step 2: Reason about the typical variation of core variables given the task
    safety_module.analyze_core_variability(core_variables, task)

    # Reset the environment to get the initial observation
    observation, reward, done, info = environment.reset()

    # Agent-environment loop
    while not done:
        # Agent decides on an action based on the observation
        action = agent.decide(observation)
        
        if action is None:
            # for purposes of demo, end when scripted actions are exhausted
            break
        # Step 3: Determine if an action affects core variables
        if safety_module.is_action_safe(observation, action):
            # Execute the action in the environment and get the new observation
            print("Action is safe. Executing...")
            observation, reward, done, info = environment.step(action)
        else:
            # Retrieve or reason further if core variables are affected
            print("Action is not safe. Further reasoning required.")
            break  # Exit loop if action is not safe


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the guardrails system with specified settings.")

    parser.add_argument("--model_name", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument('--setting_name', type=str, default="webarena_shopping", help='Name of the setting to use.')
    args = parser.parse_args()

    lm_cache_init('./lm_cache')
    main(args)
