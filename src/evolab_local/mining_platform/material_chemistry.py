from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Descriptors, rdDepictor, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D


@dataclass(frozen=True)
class StandardizedSmiles:
    raw_smiles: str
    canonical_smiles: str
    isomeric_smiles: str
    inchi: str | None
    inchi_key: str | None
    formula: str
    molecular_weight: float


def standardize_smiles(smiles: str) -> StandardizedSmiles:
    cleaned = smiles.strip()
    if not cleaned:
        raise ValueError("SMILES is empty.")
    molecule = Chem.MolFromSmiles(cleaned)
    if molecule is None:
        raise ValueError("RDKit could not parse the supplied SMILES.")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    isomeric = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    try:
        inchi = Chem.MolToInchi(molecule)
        inchi_key = Chem.InchiToInchiKey(inchi) if inchi else None
    except Exception:
        inchi = None
        inchi_key = None
    return StandardizedSmiles(
        raw_smiles=cleaned,
        canonical_smiles=canonical,
        isomeric_smiles=isomeric,
        inchi=inchi,
        inchi_key=inchi_key,
        formula=rdMolDescriptors.CalcMolFormula(molecule),
        molecular_weight=round(Descriptors.MolWt(molecule), 4),
    )


def smiles_depiction_svg(smiles: str, *, width: int = 420, height: int = 260) -> str:
    structure = standardize_smiles(smiles)
    molecule = Chem.MolFromSmiles(structure.isomeric_smiles)
    if molecule is None:
        raise ValueError("RDKit could not render the supplied SMILES.")
    rdDepictor.Compute2DCoords(molecule)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
