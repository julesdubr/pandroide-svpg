VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

env_name = "MyCartPole-v0"
max_epochs = 100
n_steps = 256
n_envs = 32
eval_interval = 3

test: $(VENV)/bin/activate
	$(PYTHON) tests/test.py

run-a2c: $(VENV)/bin/activate
	$(PYTHON) tests/test_a2c.py env_name=$(env_name) n_envs=$(n_envs) max_epochs=$(max_epochs) n_steps=$(n_steps) eval_interval=$(eval_interval)

run-reinforce: $(VENV)/bin/activate
	$(PYTHON) tests/test_reinforce.py env_name=$(env_name) n_envs=$(n_envs) max_epochs=$(max_epochs) eval_interval=$(eval_interval)

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install wheel
	$(PIP) install torch
	$(PIP) install git+https://github.com/Anidwyd/pandroide-svpg.git@main

clean:
	rm -rf $(VENV)