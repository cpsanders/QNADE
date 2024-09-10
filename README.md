# QNADE
QNADE is an algorithm that calculates the ground-state energy of a quantum many-body system using a Feed Forward Neural Network (FFNN) and Neural Autoregressive Distribution Esitmation (NADE). For information on this projects, motivation,findings, and future work, see `docs.report`.

# Environment
This repository uses `poetry` to manage python dependencies through a virtual environment. To activate the proper virtual environment, first install poetry via the command `pipx install poetry` or `pip install poetry`. Then, run `poetry shell` and `poetry install` to activate the shell and install all project dependencies.

# Running the Application
To calculate the ground state energy of a many-body system using the qnade algorithm, first fill out the config provided in `configs/config.yaml`. All required fields and their descriptions are outlined in `qnade.schemas.QNADEConfig`. To run the algorithm, run `python3 main.py -config <path-to-config>`. This script will return the ground state energy calculation along with a plot showing the loss calculations (energy) and the exact calculated energy of the system for comparison.
