import os
import sys

import torch
from rdkit import Chem
from rdkit.Chem import QED
import rdkit

_contrib_path = os.path.join(rdkit.__path__[0], "Contrib")
if _contrib_path not in sys.path:
    sys.path.insert(0, _contrib_path)
from SA_Score import sascorer

def _clean_smiles (smi):
    # Return canonical SMILES, or None if the string is invalid
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def validity(smiles_list):
    # Returns (validity_rate, list_of_valid_smiles)
    valid = [s for s in smiles_list if s and Chem.MolFromSmiles(s) is not None]
    rate = len(valid) / len(smiles_list) if smiles_list else 0.0
    return rate, valid


def uniqueness(valid_smiles):
    # clean then deduplicate — two strings for the same molecule count as one
    # Returns (uniqueness_rate, list_of_unique_canonical_smiles)
    canonical = [_clean_smiles (s) for s in valid_smiles]
    canonical = [s for s in canonical if s is not None]
    unique = list(dict.fromkeys(canonical))
    rate = len(unique) / len(canonical) if canonical else 0.0
    return rate, unique
def novelty(unique_smiles, reference_set):
    # rate of unique valid molecules not seen in the training set
    if not unique_smiles:
        return 0.0
    novel = [s for s in unique_smiles if s not in reference_set]
    return len(novel) / len(unique_smiles)


def mean_qed(smiles_list):
    # Mean drug-likeness score (QED) over valid molecules using RDKit
    scores = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None:
            scores.append(float(QED.qed(mol)))
    return sum(scores) / len(scores) if scores else 0.0


def mean_sa_score(smiles_list):
    # Mean synthetic accessibility score (economic constraint), range [1, 10]
    scores = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None:
            scores.append(float(sascorer.calculateScore(mol)))
    return sum(scores) / len(scores) if scores else 10.0


def mean_reward(smiles_list, lambda_=0.5):
    # wighted reward
    scores = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None:
            q = float(QED.qed(mol))
            sa_norm = (float(sascorer.calculateScore(mol)) - 1.0) / 9.0 #normalzie SA score to [0,1]
            scores.append(q - lambda_ * sa_norm)
    return sum(scores) / len(scores) if scores else 0.0


# Evaluation functions
def compute_metrics(generated_smiles, reference_set, lambda_=0.5):
    val_rate, valid = validity(generated_smiles)
    uniq_rate, unique = uniqueness(valid)
    nov_rate = novelty(unique, reference_set)

    mr = mean_reward(unique, lambda_)
    return {
        "validity":    val_rate,
        "uniqueness":  uniq_rate,
        "novelty":     nov_rate,
        "mean_qed":    mean_qed(unique),
        "mean_sa":     mean_sa_score(unique),
        "mean_reward": mr,
        "net_reward":  val_rate * mr,   # reward per generation attempt
    }


def mean_mw(smiles_list):
    # Mean molecular weight (Da) over valid molecules
    from rdkit.Chem import Descriptors
    scores = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None:
            scores.append(float(Descriptors.MolWt(mol)))
    return sum(scores) / len(scores) if scores else 0.0


def feasibility_rate(smiles_list, sa_threshold=4.0):
    # Fraction of molecules with SA ≤ threshold; returns (rate, feasible_smiles)
    feasible = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None and sascorer.calculateScore(mol) <= sa_threshold:
            feasible.append(smi)
    rate = len(feasible) / len(smiles_list) if smiles_list else 0.0
    return rate, feasible


def mean_scscore(smiles_list, scorer):
    # Mean SCScore [1-5]; lower = easier to synthesize. Requires scscore package.
    scores = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None:
            _, score = scorer.get_score_from_smi(smi)
            scores.append(float(score))
    return sum(scores) / len(scores) if scores else 5.0


def mean_syba(smiles_list, syba):
    # Mean SYBA score; higher = easier to synthesize. Requires syba package.
    scores = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None:
            scores.append(float(syba.predict(smi=smi)))
    return sum(scores) / len(scores) if scores else 0.0


def print_metrics(metrics, title=""):
    if title:
        print(f"\n{'='*50}")
        print(f"  {title}")
        print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<15} {v:.4f}")
    print()
