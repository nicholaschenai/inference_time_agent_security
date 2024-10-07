# Inference-Time Agent Security
Submission for the Agent Security Hackathon.

Incomplete due to time constraints, but scaffolded out the main ideas and structure.

<<<<<<< HEAD
See the [Report](docs/report.md)
=======
See the [Report](docs/REPORT.md)
>>>>>>> dev

## Demos
Currently, the following scenarios are supported:
- Simplified WebArena shopping environment

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

## Installation

1. Clone the repository


2. Install the required dependencies:
TODO:
    ```bash
    pip install -r requirements.txt
    ```
