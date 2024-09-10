"""QNADE schemas module."""

from pydantic import BaseModel, Field


class QNADEConfig(BaseModel):
    """Configuration parameters needed to run QNADE energy optimization."""

    number_of_qubits: int = Field(..., description="Number of qubits in the system, L.")
    hidden_layer_dimension: int = Field(..., description="Hidden layer dimension in the QNADE network.")
    ising_activation: float = Field(default=1, description="isingsigma_z activation, also known as sigma_z activation (J).")
    magnetic_field_coupling_strength: float = Field(default=1, description="Magnetic field coupling strength, also known as sigma_x activation (B).")
    number_of_training_iterations: int = Field(default=100, description="Number of model training iterations.")
    number_of_training_samples: int = Field(default=1000, description="Number of samples to be generated using the NADE sampling algorithm.")
    learning_rate: float = Field(default=0.01, description="Training learning rate.")
    device: str = Field(default="cpu", description="Training hardware.")
