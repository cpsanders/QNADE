"""Run QNADE algorithm."""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from qnade.optimize import optimize_qnade_network
from qnade.schemas import QNADEConfig
from qnade.utils import calculate_tfim_exact

logging.basicConfig(encoding="utf-8", level=logging.INFO)

MAX_QBITS_FOR_EXACT_CALC = 12


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file",
        default="configs/config.yaml",
    )
    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.exists():
        err = f"please enter a valid path. {config_path} does not exist."

    with config_path.open("r") as file:
        config_dict = yaml.safe_load(file)

    config = QNADEConfig(**config_dict)

    energies = optimize_qnade_network(config)
    ground_state_energy = min(energies)

    # plot training data
    plt.figure()
    plt.title(
        "L={}; g={}; QNADE Energy={}".format(
            config.number_of_qubits,
            config.magnetic_field_coupling_strength,
            ground_state_energy,
        ),
    )
    plt.plot(energies)

    if config.number_of_qubits < MAX_QBITS_FOR_EXACT_CALC:
        expected_e = calculate_tfim_exact(
            config.number_of_qubits, config.magnetic_field_coupling_strength
        )
        expected_e_plot = [
            expected_e for i in range(config.number_of_training_iterations)
        ]
        logging.info("Expected Energy: {}".format(expected_e))
        plt.plot(expected_e_plot, "g--")

    plt.xlabel("Iterations")
    plt.ylabel("Energy")
    plt.savefig("Figure.png")
    plt.show()
