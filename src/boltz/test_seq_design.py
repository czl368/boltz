
import random
import pickle
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
from pytorch_lightning import Trainer, seed_everything
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.yaml import parse_yaml

from boltz.data.types import Manifest
from boltz.data.write.writer import BoltzWriter
from boltz.model.model import Boltz1

# Default Parameters
DATA_PATH = Path("/Users/christinali/boltz/examples/multimer.yaml")
OUT_DIR = Path("./design_results")
CACHE_DIR = Path("~/.boltz").expanduser()
CCD_DIR = CACHE_DIR / "ccd.pkl"
CHECKPOINT_PATH = None  # If None, will use default model
DEVICES = 1
ACCELERATOR = "gpu"  # Choices: "gpu", "cpu"
MAX_ITERATIONS = 20  # Number of optimization rounds
MUTATION_RATE = 0.1  # Mutation probability per residue
SEED = 42  # Random seed for reproducibility

CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
MODEL_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt"


@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters."""
    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True


def download(cache: Path) -> None:
    """Download the required model checkpoint if not present."""
    model_path = cache / "boltz1_conf.ckpt"
    if not model_path.exists():
        print(f"Downloading Boltz1 model to {model_path}...")
        urllib.request.urlretrieve(MODEL_URL, str(model_path))


def mutate_sequence(sequence: str, mutation_rate: float) -> str:
    """Randomly mutates a protein sequence with a given mutation probability."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    seq_list = list(sequence)

    for i in range(len(seq_list)):
        if random.random() < mutation_rate:  # Apply mutation with given probability
            seq_list[i] = random.choice(amino_acids)  # Replace with random AA
    
    return "".join(seq_list)


def main():
    """Main function to run the sequence optimization loop."""
    
    # Set seed for reproducibility
    seed_everything(SEED)

    # Ensure output directories exist
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Download the model if not found
    download(CACHE_DIR)

    # Load model checkpoint
    checkpoint = CHECKPOINT_PATH if CHECKPOINT_PATH else CACHE_DIR / "boltz1_conf.ckpt"

    # Parse input sequence from FASTA
    fasta_path = DATA_PATH
    with open(CCD_DIR, "rb") as file:
        ccd_dict = pickle.load(file)
    records = parse_yaml(fasta_path, ccd=ccd_dict)
    original_sequence = records.sequences[0]
    best_sequence = original_sequence
    best_plddt = 0.0

    print(f"Starting sequence optimization for {MAX_ITERATIONS} iterations...")

    #for iteration in range(MAX_ITERATIONS):
    for iteration in range(0,1):
        # Mutate the sequence
        new_sequence = mutate_sequence(best_sequence, MUTATION_RATE)

        print(f"Iteration {iteration+1}/{MAX_ITERATIONS}: Predicting pLDDT for mutated sequence")

        # Load Boltz1 model
        model = Boltz1.load_from_checkpoint(checkpoint, strict=True, map_location="cpu")
        model.eval()


        # Create data module
        manifest = Manifest([records.record])
        data_module = BoltzInferenceDataModule(
            manifest=manifest,
            target_dir=OUT_DIR / "structures",
            msa_dir=OUT_DIR / "msa",
            num_workers=2,
        )

        trainer = Trainer(
            accelerator=ACCELERATOR,
            devices=DEVICES,
            precision=32,
        )

        # Run prediction
        predictions = trainer.predict(model, datamodule=data_module, return_predictions=True)

        # Extract pLDDT score
        plddt_score = predictions[0]["plddt"].mean().item()
        print(f"Mutated sequence pLDDT: {plddt_score:.2f}")

        # Update the best sequence if it has the highest pLDDT
        if plddt_score > best_plddt:
            best_plddt = plddt_score
            best_sequence = new_sequence
            print(f"New best sequence found with pLDDT {best_plddt:.2f}")

    # Save the best sequence
    best_sequence_path = OUT_DIR / "best_sequence.fasta"
    with open(best_sequence_path, "w") as f:
        f.write(f">Optimized_Sequence\n{best_sequence}")

    print(f"Optimization complete! Best sequence saved to {best_sequence_path}")


if __name__ == "__main__":
    main()


