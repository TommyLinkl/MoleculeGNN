from rdkit import Chem

# Define some SMARTS patterns for common functional groups
functional_groups = {
    'alcohol': '[CX4][OH]',
    'amine': '[NX3][CX4]',
    'carboxylic_acid': 'C(=O)[OH]',
    'ketone': 'C(=O)[CX4]',
    'aldehyde': 'C=O',
    'ester': '[CX3](=O)[OX2H1]',
    'ether': '[CX4][OX2][CX4]',
    'alkene': 'C=C',
    'alkyne': 'C#C',
    'aromatic_ring': 'c1ccccc1',
    'halide': '[F,Cl,Br,I]'
}

def identify_functional_groups(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES string"

    identified_groups = []
    for name, pattern in functional_groups.items():
        fg = Chem.MolFromSmarts(pattern)
        if mol.HasSubstructMatch(fg):
            identified_groups.append(name)

    return identified_groups



for smiles in smiles_list:
    groups = identify_functional_groups(smiles)
    print(f"Molecule: {smiles}")
    print(f"\tFunctional Groups: {', '.join(groups) if groups else 'None'}")
