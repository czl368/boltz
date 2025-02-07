
import random
import pickle
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import torch
from pytorch_lightning import Trainer, seed_everything
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.parse.fasta import parse_fasta, parse_fasta_update_seq
from boltz.data.parse.yaml import parse_yaml, parse_yaml_update_seq
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.csv import parse_csv
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data import const

from boltz.data.types import MSA, Manifest, Record
from boltz.data.write.writer import BoltzWriter
from boltz.model.model import Boltz1
import numpy as np

# Default Parameters
DATA_PATH = Path("/Users/christinali/boltz_testing/mcmc/1ssc_input.yaml")
OUT_DIR = Path("./design_results")
CACHE_DIR = Path("~/.boltz").expanduser()
CCD_DIR = CACHE_DIR / "ccd.pkl"
CHECKPOINT_PATH = None  # If None, will use default model
DEVICES = 1
ACCELERATOR = "cpu"  # Choices: "gpu", "cpu"
MAX_ITERATIONS = 20  # Number of optimization rounds
MUTATION_RATE = 0.5  # Mutation probability per residue
SEED = 42  # Random seed for reproducibility
use_msa_server = False,
msa_server_url = "https://api.colabfold.com"
msa_pairing_strategy = "greedy"
recycling_steps = 3
sampling_steps = 200
diffusion_samples = 1
write_full_pae = False
write_full_pde = False
step_scale = 1.638
new_sequence = None # New mutated sequence
update_seq = False  # Whether or not to update the sequence
output_format = "mmcif"
#output_format: Literal["pdb", "mmcif"] = "mmcif"

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

# Copied from main.py (boltz)
# Both compute_msa and process_inputs 
def compute_msa(
    data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
) -> None:
    """Compute the MSA for the input data.

    Parameters
    ----------
    data : dict[str, str]
        The input protein sequences.
    target_id : str
        The target id.
    msa_dir : Path
        The msa directory.
    msa_server_url : str
        The MSA server URL.
    msa_pairing_strategy : str
        The MSA pairing strategy.

    """
    if len(data) > 1:
        paired_msas = run_mmseqs2(
            list(data.values()),
            msa_dir / f"{target_id}_paired_tmp",
            use_env=True,
            use_pairing=True,
            host_url=msa_server_url,
            pairing_strategy=msa_pairing_strategy,
        )
    else:
        paired_msas = [""] * len(data)

    unpaired_msa = run_mmseqs2(
        list(data.values()),
        msa_dir / f"{target_id}_unpaired_tmp",
        use_env=True,
        use_pairing=False,
        host_url=msa_server_url,
        pairing_strategy=msa_pairing_strategy,
    )

    for idx, name in enumerate(data):
        # Get paired sequences
        paired = paired_msas[idx].strip().splitlines()
        paired = paired[1::2]  # ignore headers
        paired = paired[: const.max_paired_seqs]

        # Set key per row and remove empty sequences
        keys = [idx for idx, s in enumerate(paired) if s != "-" * len(s)]
        paired = [s for s in paired if s != "-" * len(s)]

        # Combine paired-unpaired sequences
        unpaired = unpaired_msa[idx].strip().splitlines()
        unpaired = unpaired[1::2]
        unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
        if paired:
            unpaired = unpaired[1:]  # ignore query is already present

        # Combine
        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)

        # Dump MSA
        csv_str = ["key,sequence"] + [f"{key},{seq}" for key, seq in zip(keys, seqs)]

        msa_path = msa_dir / f"{name}.csv"
        with msa_path.open("w") as f:
            f.write("\n".join(csv_str))

def process_inputs(  # noqa: C901, PLR0912, PLR0915
    data: list[Path],
    out_dir: Path,
    ccd_path: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    max_msa_seqs: int = 4096,
    use_msa_server: bool = False,
    update_seq: bool = False,
    new_sequence: str = None
) -> None:
    """Process the input data and output directory.

    Parameters
    ----------
    data : list[Path]
        The input data.
    out_dir : Path
        The output directory.
    ccd_path : Path
        The path to the CCD dictionary.
    max_msa_seqs : int, optional
        Max number of MSA sequences, by default 4096.
    use_msa_server : bool, optional
        Whether to use the MMSeqs2 server for MSA generation, by default False.

    Returns
    -------
    BoltzProcessedInput
        The processed input data.

    """
    #click.echo("Processing input data.")
    existing_records = None

    # Check if manifest exists at output path
    manifest_path = out_dir / "processed" / "manifest.json"
    if manifest_path.exists():
        print(f"Found a manifest file at output directory: {out_dir}")

        manifest: Manifest = Manifest.load(manifest_path)
        input_ids = [d.stem for d in data]
        existing_records, processed_ids = zip(
            *[
                (record, record.id)
                for record in manifest.records
                if record.id in input_ids
            ]
        )

        if isinstance(existing_records, tuple):
            existing_records = list(existing_records)

        # Check how many examples need to be processed
        missing = len(input_ids) - len(processed_ids)
        if not missing:
            print("All examples in data are processed. Updating the manifest")
            # Dump updated manifest
            updated_manifest = Manifest(existing_records)
            updated_manifest.dump(out_dir / "processed" / "manifest.json")
            return

        print(f"{missing} missing ids. Preprocessing these ids")
        missing_ids = list(set(input_ids).difference(set(processed_ids)))
        data = [d for d in data if d.stem in missing_ids]
        assert len(data) == len(missing_ids)

    # Create output directories
    msa_dir = out_dir / "msa"
    structure_dir = out_dir / "processed" / "structures"
    processed_msa_dir = out_dir / "processed" / "msa"
    predictions_dir = out_dir / "predictions"

    out_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    processed_msa_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load CCD
    with ccd_path.open("rb") as file:
        ccd = pickle.load(file)  # noqa: S301

    if existing_records is not None:
        print(f"Found {len(existing_records)} records. Adding them to records")

    # Parse input data
    records: list[Record] = existing_records if existing_records is not None else []

    #TODO figure out a way to alter the target --> make a different parse_yaml function
    path = data[0]
    try:
        # Parse data
        if path.suffix in (".fa", ".fas", ".fasta"):
            if update_seq:
                target = parse_fasta_update_seq(path, ccd, new_sequence)
            else:
                target = parse_fasta(path, ccd)
        elif path.suffix in (".yml", ".yaml"):
            if update_seq:
                print("update")
                target = parse_yaml_update_seq(path, ccd, new_sequence)
            else:
                print("not updating")
                target = parse_yaml(path, ccd)
        elif path.is_dir():
            msg = f"Found directory {path} instead of .fasta or .yaml, skipping."
            raise RuntimeError(msg)
        else:
            msg = (
                f"Unable to parse filetype {path.suffix}, "
                "please provide a .fasta or .yaml file."
            )
            raise RuntimeError(msg)

        # Get target id
        target_id = target.record.id

        # Get all MSA ids and decide whether to generate MSA
        to_generate = {}
        prot_id = const.chain_type_ids["PROTEIN"]
        for chain in target.record.chains:
            # Add to generate list, assigning entity id
            if (chain.mol_type == prot_id) and (chain.msa_id == 0):
                entity_id = chain.entity_id
                msa_id = f"{target_id}_{entity_id}"
                to_generate[msa_id] = target.sequences[entity_id]
                chain.msa_id = msa_dir / f"{msa_id}.csv"

            # We do not support msa generation for non-protein chains
            elif chain.msa_id == 0:
                chain.msa_id = -1

        # Generate MSA
        if to_generate and not use_msa_server:
            msg = "Missing MSA's in input and --use_msa_server flag not set."
            raise RuntimeError(msg)

        if to_generate:
            msg = f"Generating MSA for {path} with {len(to_generate)} protein entities."
            print(msg)
            compute_msa(
                data=to_generate,
                target_id=target_id,
                msa_dir=msa_dir,
                msa_server_url=msa_server_url,
                msa_pairing_strategy=msa_pairing_strategy,
            )

            # Parse MSA data
            msas = sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})
            msa_id_map = {}
            for msa_idx, msa_id in enumerate(msas):
                # Check that raw MSA exists
                msa_path = Path(msa_id)
                if not msa_path.exists():
                    msg = f"MSA file {msa_path} not found."
                    raise FileNotFoundError(msg)

                # Dump processed MSA
                processed = processed_msa_dir / f"{target_id}_{msa_idx}.npz"
                msa_id_map[msa_id] = f"{target_id}_{msa_idx}"
                if not processed.exists():
                    # Parse A3M
                    if msa_path.suffix == ".a3m":
                        msa: MSA = parse_a3m(
                            msa_path,
                            taxonomy=None,
                            max_seqs=max_msa_seqs,
                        )
                    elif msa_path.suffix == ".csv":
                        msa: MSA = parse_csv(msa_path, max_seqs=max_msa_seqs)
                    else:
                        msg = f"MSA file {msa_path} not supported, only a3m or csv."
                        raise RuntimeError(msg)

                    msa.dump(processed)

            # Modify records to point to processed MSA
            for c in target.record.chains:
                if (c.msa_id != -1) and (c.msa_id in msa_id_map):
                    c.msa_id = msa_id_map[c.msa_id]

            # Keep record
            records.append(target.record)

            # Dump structure
            struct_path = structure_dir / f"{target.record.id}.npz"
            target.structure.dump(struct_path)

    except Exception as e:
        if len(data) > 1:
            print(f"Failed to process {path}. Skipping. Error: {e}.")
        else:
            raise e

    # Dump manifest
    manifest = Manifest(records)
    manifest.dump(out_dir / "processed" / "manifest.json")


# Simulated Annealing Parameters
INITIAL_TEMPERATURE = 5.0
FINAL_TEMPERATURE = 0.1
ANNEALING_RATE = 0.99  # Decay factor per iteration


MAX_ITERATIONS = 10

def acceptance_probability(old_score, new_score, temperature):
    """Compute acceptance probability using the Metropolis criterion."""
    if new_score > old_score:
        return 1.0
    return np.exp((new_score - old_score) / temperature)



import random

def mutate_sequence_wtih_hotspot(sequence: str, mutation_rate: float, ref_seq: str) -> str:
    """
    Randomly mutates a protein sequence with a given mutation probability.
    Only mutates positions corresponding to 'X' in the reference sequence.

    Args:
        sequence (str): The input protein sequence to be mutated.
        ref_seq (str): The reference sequence specifying positions to mutate ('X').
        mutation_rate (float): Probability of mutation for applicable positions.

    Returns:
        str: The mutated protein sequence.
    """
    if len(sequence) != len(ref_seq):
        raise ValueError("The sequence and ref_seq must have the same length.")

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    seq_list = list(sequence)

    for i in range(len(seq_list)):
        if ref_seq[i] == 'X':  # Only mutate positions marked as 'X' in ref_seq
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
    cyc_pep_seq_idx = list(records.sequences.keys())[-1] # Getting the last sequence
    original_sequence = records.sequences[cyc_pep_seq_idx]
    best_sequence = original_sequence
    best_confidence_score = 0.0
    best_iteration = 1

    ref_seq  = str(original_sequence)
    # just an example
    # starting index 0
    ref_seq = list(ref_seq)
    nonhotspots_pos = [0,2,3,4,6,8,9,10]

    for i in nonhotspots_pos:
        ref_seq[i] = "X"
    
    ref_seq = "".join(ref_seq)
    print(ref_seq,original_sequence)


    temperature = INITIAL_TEMPERATURE
    print(f"Starting sequence optimization for {MAX_ITERATIONS} iterations...")

    
    step = 0
    #for iteration in range(MAX_ITERATIONS):
    for iteration in range(MAX_ITERATIONS):
        #new_sequence = mutate_sequence(best_sequence, MUTATION_RATE)

        if iteration >= 0:
            update_seq = True # TODO can delete or change this later --> ensures that the sequence is updated

        # Mutate the sequence
        #new_sequence = mutate_sequence(best_sequence, MUTATION_RATE)
        new_sequence =mutate_sequence_wtih_hotspot(best_sequence, MUTATION_RATE,ref_seq)

        print("new_sequence: ", new_sequence)

        print(f"Iteration {iteration+1}/{MAX_ITERATIONS}: Predicting confidence_score for mutated sequence")

        # Process the inputs
        use_msa_server = True
        process_inputs(
            data=[DATA_PATH],
            out_dir=OUT_DIR,
            ccd_path=CCD_DIR,
            use_msa_server=use_msa_server,
            msa_server_url=msa_server_url,
            msa_pairing_strategy=msa_pairing_strategy,
            update_seq=update_seq,
            new_sequence=new_sequence
        )

        # Create data module and load processed data
        manifest=Manifest.load(OUT_DIR/ "processed" / "manifest.json")
        data_module = BoltzInferenceDataModule(
            manifest=manifest,
            target_dir=OUT_DIR / "processed" / "structures",
            msa_dir=OUT_DIR / "processed" / "msa",
            num_workers=2,
        )

        # Create directory to store predictions
        cur_pred_dir = OUT_DIR / "predictions" / f"step_{step+1}"
        cur_pred_dir.mkdir(parents=True, exist_ok=True)

        # Create prediction writer
        pred_writer = BoltzWriter(
            data_dir=OUT_DIR / "processed" / "structures",
            output_dir=cur_pred_dir,
            output_format=output_format,
        )

        trainer = Trainer(
            accelerator=ACCELERATOR,
            devices=DEVICES,
            callbacks=[pred_writer],
            precision=32,
        )

        # Load Boltz1 model
        predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
        }
        diffusion_params = BoltzDiffusionParams()
        diffusion_params.step_scale = step_scale

        model = Boltz1.load_from_checkpoint(
            checkpoint, 
            strict=True,
            predict_args = predict_args, 
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params), 
            ema = False)
        model.eval()

        # Run prediction
        # TODO see if you can get gradients
        predictions = trainer.predict(model, datamodule=data_module, return_predictions=True)

        # Extract confidence_score score
        confidence_score = predictions[0]["confidence_score"].mean().item()
        #print(predictions)
        print(f"Mutated sequence confidence_score: {confidence_score:.2f}")
        
        new_confidence_score = confidence_score

        acceptance_prob = acceptance_probability(best_confidence_score, new_confidence_score, temperature)
        if np.random.rand() < acceptance_prob:
            best_confidence_score = new_confidence_score
            best_sequence = new_sequence
            print(f"Accepted new sequence with confidence_score {new_confidence_score:.2f} (T={temperature:.2f})")
            step+=1
        else:
            print(f"Rejected new sequence (confidence_score {new_confidence_score:.2f}), keeping current best.")
            shutil.rmtree(cur_pred_dir)

        temperature *= ANNEALING_RATE  # Anneal temperature


        shutil.rmtree(OUT_DIR / "processed")
        #shutil.rmtree(OUT_DIR / "msa")

    best_sequence_path = OUT_DIR / "best_sequence.fasta"
    with open(best_sequence_path, "w") as f:
        f.write(f">Optimized_Sequence\n{best_sequence}\n")
        f.write(f"Final best confidence_score: {best_confidence_score:.2f}\n")

    print(f"Optimization complete! Best sequence saved to {best_sequence_path}")

'''
        # Update the best sequence if it has the highest pLDDT
        if plddt_score > best_plddt:
            best_plddt = plddt_score
            best_sequence = new_sequence
            best_iteration = iteration + 1
            print(f"New best sequence found with pLDDT {best_plddt:.2f}")
            print(f"New best sequence: {best_sequence}")


        # Remove manifest and msa files
        shutil.rmtree(OUT_DIR / "processed")
        shutil.rmtree(OUT_DIR / "msa")

    # Save the best sequence
    best_sequence_path = OUT_DIR / "best_sequence.fasta"
    with open(best_sequence_path, "w") as f:
        f.write(f">Optimized_Sequence\n{best_sequence}")
        f.write(f"New best sequence found with pLDDT: {best_plddt:.2f}")
        f.write(f"Best iteration: iteration_{best_iteration}")

    print(f"Optimization complete! Best sequence saved to {best_sequence_path}")

'''



if __name__ == "__main__":
    main()


