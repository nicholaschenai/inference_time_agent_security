# Inference-Time Agent Security

This repo was improved upon from a hackathon submission, and the writeup can be found [here](docs/blogpost.md).

See the next section for info correct as at hackathon deadline.

## Hackathon 
Submission for the Agent Security Hackathon.

Incomplete due to time constraints, but scaffolded out the main ideas and structure.

See the [Report](docs/REPORT.md)

## Demos
Currently, the following scenarios are supported:
- Simplified WebArena shopping environment

## Installation

1. Clone the repository, cd into the directory

2. Install the required dependencies:

```bash
conda env create --file env.yml
conda activate agent_security
```

## Usage

```bash
PYTHONPATH=. python src/main.py --model_name your_model_name --setting_name your_setting_name
```

Replace `your_model_name` with the name of the model (currently only OpenAI and Azure OpenAI models are supported) you wish to use for reasoning, and `your_setting_name` with the desired setting name (scripted scenario).

### Arguments
- `--model_name`: The name of the model to use for reasoning.
- `--verbose`: Enable verbose output.
- `--debug_mode`: Enable debug mode.
- `--setting_name`: The name of the setting to use.

### Example

```bash
PYTHONPATH=. python src/main.py --model_name gpt-4o-mini-2024-07-18 --setting_name webarena_shopping
```

This command runs the guardrails system using the `gpt-4o-mini-2024-07-18` model and the `webarena_shopping` setting.
