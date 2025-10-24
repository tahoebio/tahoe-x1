# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
import json

import numpy as np
from datasets import load_dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from tqdm.auto import tqdm


def generate_fingerprints_np(
    smiles_list: list,
    radius: int = 2,
    nBits: int = 2048,
) -> np.ndarray:
    """Generate fingerprints for a list of SMILES strings using a NumPy array
    allocation."""
    print(f"Generating Morgan fingerprints for {len(smiles_list)} SMILES strings.")
    fingerprints = np.zeros((len(smiles_list), nBits), dtype=int)
    for i, smiles in enumerate(tqdm(smiles_list, desc="Generating fingerprints")):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Here you might want to set a row of NaNs or zeros to indicate failure
            fingerprints[i, :] = np.nan
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        ConvertToNumpyArray(fp, fingerprints[i])
    print("Fingerprint generation complete.")
    return fingerprints


if __name__ == "__main__":

    # load tahoe data from hf
    drug_metadata = load_dataset("vevotx/Tahoe-100M", "drug_metadata", split="train")
    print(f"Loaded {len(drug_metadata)} drug entries")

    drugs = drug_metadata["drug"]  # None in two places, 379 unique
    cid = drug_metadata["pubchem_cid"]  # None in two places, 377 unique
    smiles = drug_metadata[
        "canonical_smiles"
    ]  # identical in two places and none in two places

    drug_to_id = {"<pad>": 0}
    chosen_smiles = []
    i = 1

    for drug, smile in zip(drugs, smiles):
        if smile is not None:
            drug_to_id[drug] = i
            i += 1
            chosen_smiles.append(smile)

    print("Length drug to id dict: ", len(drug_to_id), drug_to_id)
    print("Length chosen smiles list: ", len(chosen_smiles), chosen_smiles)

    # save the mapping drugs to id
    with open("drug_to_id.json", "w") as f:
        json.dump(drug_to_id, f)

    fps = np.zeros((len(chosen_smiles) + 1, 2048))  # +1 for the padding token
    fps[1:, :] = generate_fingerprints_np(chosen_smiles)
    print("Final fingerprints shape: ", fps.shape)

    # save fingerprints
    np.save("drug_fps", fps)
