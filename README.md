# SafeArena

## About

TBD

## Installation

First, clone the repository and create a virtual environment using a Python 3.10+ version:

```bash
git clone https://github.com/McGill-NLP/safearena.git

cd safearena/
python -m venv venv
source venv/bin/activate
```

Then, install the required packages:

```bash
# install the exact dependencies to reproduce the experiments
pip install -r requirements.txt

# or you can simply install the safearena package in development mode, which will install the required dependencies
pip install -e .

# Install playwright
playwright install
```

## Usage

### Downloading task splits

First, request access to the SafeArena dataset on the Hugging Face Hub. Once you have access, you can log in using the `huggingface_hub` CLI:

```bash
huggingface-cli login
```

Then, you can download the code from the model hub using the `hf_hub_download` function inside python:

```python
from huggingface_hub import hf_hub_download

# Download the safe.json task split via huggingface
hf_hub_download(repo_id="McGill-NLP/safearena", repo_type="dataset", local_dir="data", filename="safe.json")
# Download the harm.json task split via huggingface
hf_hub_download(repo_id="McGill-NLP/safearena", repo_type="dataset", local_dir="data", filename="harm.json")
```

You now have the required task splits in the relative `data/` directory.

### Running experiments

To run an experiment, use the `run_experiment.py` script.

```bash
python safearena/run_experiment.py
```

### Running agent-xray

To visualize the agent's behavior, you can use the `agent_xray.py` tool derived from agentlab:

```bash
python apps/agent_xray.py --results_dir <path_to_results_dir> --port <port>
``
# safarena-release
