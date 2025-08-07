# active-learning-graph
Repository containing graph neural network models and a workflow to run active learning.

## Setup

Need to create a virtual environment and download hte packages.

```
conda create --name py3-11 python=3.11
conda activate py3-11
python -m venv venv
conda deactivate
source venv/bin/activate
pip install --upgrade pip
```

Ok now install the requirements.txt

```
pip install -r requirements.txt
```

Now run the refactored sanity check code.
```
python sanity_check.py
```
