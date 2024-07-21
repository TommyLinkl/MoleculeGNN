import numpy as np
import matplotlib.pyplot as plt
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
    # 'aromatic_ring': 'c1ccccc1',
    'halide': '[F,Cl,Br,I]'
}

# Dictionary of atomic masses
atomMassDict = {
    1: 1.0078,    # Hydrogen
    2: 4.0026,    # Helium
    3: 6.9410,    # Lithium
    4: 9.0122,    # Beryllium
    5: 10.811,    # Boron
    6: 12.0107,   # Carbon
    7: 14.0067,   # Nitrogen
    8: 15.9994,   # Oxygen
    9: 18.9984,   # Fluorine
    10: 20.1797,  # Neon
}

atomNameDict = {
    1: 'H', 
    2: 'He',
    3: 'Li',
    4: 'Be',
    5: 'B',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
    10: 'Ne',
}


def calculate_molMass(atom_types):
    total_mass = 0.0
    for atom in atom_types:
        if atom in atomMassDict:
            total_mass += atomMassDict[atom]
        else:
            raise ValueError(f"Unknown atom type {atom} encountered. Please update atomMassDict.")
    return total_mass


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


def plot_by_values(property, df):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ref = df[f'ref_{property}'].to_numpy()
    pred = df[f'pred_{property}'].to_numpy()
    
    x_values = np.arange(0, 20, 0.2)
    ax.plot(x_values, x_values, "r-", linewidth=1)
    ax.scatter(ref, pred, marker='o', color='blue', s=4, alpha=0.5)

    if property=='gap': 
        ax.set(xlabel=f"Reference {property} (eV)", ylabel=f"Predicted {property} (eV)")
        upper_line = x_values + 10 * 0.0433634
        lower_line = x_values - 10 * 0.0433634
        ax.fill_between(x_values, lower_line, upper_line, color='gray', alpha=0.3, label='10x chemical accuracy')
        ax.legend()
    else: 
        ax.set(xlabel=f"Reference {property}", ylabel=f"Predicted {property}")
    ax.grid(alpha=0.5)
    fig.tight_layout()
    return fig


def plot_by_molMass(property, df):
    df['molMass'] = df['atom_types'].apply(calculate_molMass)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['molMass'], abs(df[f'pred_{property}'] - df[f'ref_{property}']), marker='o', color='blue', s=4, alpha=0.5)

    # Add an average
    avg_df = df.groupby('molMass').apply(lambda x: (abs(x[f'pred_{property}'] - x[f'ref_{property}'])).mean()).reset_index()
    avg_df.columns = ['molMass', 'avg_error']
    ax.plot(avg_df['molMass'], avg_df['avg_error'], 'r-', label='Average Error')

    if property=='gap':
        ax.set(xlabel='Molecular Mass (amu)', ylabel=f'Gap Error (eV)')

        x_min, x_max = ax.get_xlim()
        x_values = np.linspace(x_min, x_max, 100)
        upper_line = [0.0433634*10]*100
        lower_line = [0]*100
        ax.fill_between(x_values, lower_line, upper_line, color='gray', alpha=0.3, label='10x chemical accuracy')
        ax.legend()
    else: 
        ax.set(xlabel='Molecular Mass (amu)', ylabel=f'{property} Error')
    ax.grid(alpha=0.5)

    fig.tight_layout()
    return fig


def plot_by_heaviestAtom(property, df):
    df['heaviestAtom'] = df.apply(lambda entry: max(entry['atom_types']), axis=1)
    
    # Group by heaviestAtom and prepare data for plotting
    grouped = df.groupby('heaviestAtom')
    bar_width = 1
    group_pos = 0.0
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Iterate over groups and plot clustered bar plots
    for i, (heaviestAtom, group) in enumerate(grouped):        
        bar_positions = [group_pos + pos*bar_width for pos in range(len(group))]
        
        ax.fill_between(bar_positions, np.zeros(len(bar_positions)), abs(group[f'pred_{property}'] - group[f'ref_{property}']), color=plt.get_cmap('tab10')(i % 10), alpha=0.5, label=f'heaviestAtom={atomNameDict[heaviestAtom]}')    # edgecolor='none',
        avg_diff = abs(group[f'pred_{property}'] - group[f'ref_{property}']).mean()
        ax.plot(bar_positions, [avg_diff]*len(bar_positions), ':', color=plt.get_cmap('tab10')(i % 10), label=f'{atomNameDict[heaviestAtom]} average error')
        
        group_pos += len(group)*bar_width + 50*bar_width
    
    ax.set(xlabel='Molecules by heaviest atom', title=f'Plot of {property} error by heaviest atom')
    if property=='gap':
        ax.set(ylabel=f'Gap error (eV)')
    else: 
        ax.set(ylabel=f'{property} error')
    ax.legend(title="Heaviest Atom")
    ax.grid(alpha=0.5)
    fig.tight_layout()
    
    return fig


def plot_by_chemGroups(property, df):
    fig, ax = plt.subplots(figsize=(10, 6))
    all_groups = set([item for sublist in df['functional_groups'] for item in sublist])
    bar_width = 1
    x_offset = 0
    xticks_labels = []
    xticks_positions = []

    for i, group in enumerate(all_groups):
        mask = df['functional_groups'].apply(lambda lst: group in lst)
        group_data = df[mask]

        positions = range(len(group_data))
        bar_positions = [x_offset + pos*bar_width for pos in positions]
        ax.fill_between(bar_positions, np.zeros(len(bar_positions)), abs(group_data[f'pred_{property}'] - group_data[f'ref_{property}']), color=plt.get_cmap('Set1')(i % 10), alpha=0.5, label=group)    # edgecolor='none',

        xticks_labels.append(f'{group}')
        xticks_positions.append(x_offset + len(group_data)*bar_width/2)
        x_offset += len(group_data)*bar_width + 20*bar_width

        avg_diff = abs(group_data[f'pred_{property}'] - group_data[f'ref_{property}']).mean()
        ax.plot(bar_positions, [avg_diff]*len(bar_positions), ':', color=plt.get_cmap('Set1')(i % 10)) # , label=f'{group} average error')

    if property=='gap':
        ax.set(xlabel='Chemical functional groups', ylabel=f'Gap error (eV)')
    else: 
        ax.set(xlabel='Chemical functional groups', ylabel=f'{property} error')
    ax.set_title(f'Plot of {property} error by chemical functional groups')
    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(xticks_labels)
    ax.grid(alpha=0.5)
    ax.legend(title='Functional Groups')

    fig.tight_layout()
    return fig

