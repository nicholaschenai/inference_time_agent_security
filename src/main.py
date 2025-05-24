import importlib
import argparse
import logging
import os

logger = logging.getLogger("logger")

from safety_module import SafetyModule

from environments.base_env import BaseEnv
from agents.base_agent import BaseAgent

from config import get_config

from cognitive_base.utils.log import setup_logging_n_base_dirs
from cognitive_base.utils.argparsers import get_base_parser


def main(args: argparse.Namespace):
    setup_logging_n_base_dirs(args)
    # Get configuration for the chosen setting
    config = get_config(args.setting_name)

    # Dynamically import the environment and agent based on the setting
    env_cls: type[BaseEnv] = getattr(importlib.import_module(config['environment']), config['env_class'])
    agent_cls: type[BaseAgent] = getattr(importlib.import_module(config['agent']), config['agent_class'])

    # Initialize components
    kwargs = vars(args)
    agent = agent_cls(scripted_actions=config['scripted_actions'], **kwargs)
    environment = env_cls(**kwargs, **config)

    # Safety module initialization: 
    # reason about the typical variation of core variables given the task
    safety_module = SafetyModule(action_space=environment.action_space, **kwargs, **config)

    # Reset the environment to get the initial observation
    observation, _, done, _ = environment.reset()

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
            logger.info("Action is safe. Executing...")
            observation, reward, done, info = environment.step(action)
        else:
            # Retrieve or reason further if core variables are affected
            logger.info("Action is not safe. Further reasoning required.")
            break  # Exit loop if action is not safe

    # Save the world model visualization
    graph_path = os.path.join(args.result_dir, 'world_model_graph.png')
    description = config.get('desc', '')  # Get description from config, empty string if not present
    safety_module.world_model.visualize_graph(save_path=graph_path, description=description)
    logger.info(f"World model graph visualization saved to {graph_path}")


if __name__ == "__main__":
    # Set up argument parser
    parser = get_base_parser()
    parser.set_defaults(model_name='gpt-4o-2024-08-06')
    parser.add_argument('--setting_name', type=str, default="webarena_shopping_exceed", help='Name of the setting to use.')
    args = parser.parse_args()

    main(args)
