# Graph Neural Network (GNN) for Modeling Molecular Properties

**Author:** Tommy Lin

This project implements state-of-the-art GNN models for accurately predicting chemical properties.

## Table of contents

1. [Demo and sample results](#demo)
2. [Code Usage](#codeUsage)
3. [Introduction](#introduction)
4. [GNN Models](#gnnModels)
5. [Dataset](#dataset)
6. [Hyperparameter Tuning](#hyperTuning)

## <a name="demo"></a> Demo and sample results

As an example, I show the performance of a GNN model and the analysis of its predictions of HOMO-LUMO gap for small molecules. As shown below in Figures 1-4, we see that this GNN model gives a decent performance that reproduces the DFT-computed gap energy within chemical accuracy across all metrics. 

<figure>
    <img src="./figures/plot_by_values.png"
         alt="Plot by values"
         width="300">
    <figcaption>Figure 1. The prediction of the HOMO-LUMO gap across the QM9 dataset with the tuned SchNet model. The diagonal shaded grey area is 10 times of the chemical accuracy (10 kcal/mol). </figcaption>
</figure>

Specifically, I analyze the performance by categorizing the molecules. A nearly uniform performance across large-gap and small-gap molecules is observed, while some of the small-gap molecules have larger errors. Performance is fairly similar across molecules with C,N,O,F as the heaviest atom, as well as for molecules with different organic functional groups. 

<figure>
    <img src="./figures/plot_by_molMass.png"
         alt="Plot by mol mass"
         width="400">
    <figcaption>Figure 2. HOMO-LUMO gap energy prediction, for molecules sorted by molecular mass. The shaded grey area is 10 times of the chemical accuracy (10 kcal/mol). </figcaption>
</figure>

<figure>
    <img src="./figures/plot_by_chemGroups.png"
        alt="Plot by chemical functional groups"
        width="500">
    <figcaption>Figure 3. The error of HOMO-LUMO gap predictions from the SchNet model, as grouped by chemical functional groups (color coded) within the QM9 dataset. The average gap error for each chemical groups is plotted as the dotted lines. </figcaption>
</figure>

<figure>
    <img src="./figures/plot_by_heaviestAtom.png"
         alt="Plot by heavist atom"
         width="500">
    <figcaption>Figure 4. The error of HOMO-LUMO gap predictions from the SchNet model, as grouped by the heaviest atom in the molecule within the QM9 dataset. The average gap error for atoms with the heaviest atom is plotted as the dotted lines. </figcaption>
</figure>


## <a name="codeUsage"></a> Code Usage

Installation of required packages (using `pip`):

    pip install -r requirements.txt

After satisfying all the package installation requirements, users can run the code using 

    python main.py [configFilename]

The calculation settings is passed via command line in a YAML file. 

Each configuration file specifies settings for `Job` and for `Model`. There are three available job types as specified in `run_mode`: `Training`, `Hyperparameter`, and `Inference`. Please see the sample config_xxx.yml files in the root directory for reference. Users can also specify whether to load an existing model (from PyTorch .pth files) for further tuning or start from scratch. 

One can also choose to fit to HOMO-LUMO gap `target_index: 4` or the dipole moment `target index: 0` (as well as other molecular properties in the QM9 dataset). 

In the `Model` section, one can specify almost all details of the GNN models, from number of neurons in each structure, to learning rate, batch size and optimizer. Note that for the hyperparameter tuning mode, any setting passed in as a list will be tuned. For the inference mode, care must be taken that the model setting in the config file is the same as the model saved in the .pth file. 

## <a name="introduction"></a> Introduction

Machine learning (ML) models have been shown to have great predictive potentials 
in the chemical space. They complement physics-based computational chemistry 
modeling, accelerate chemical and material discovery, and open up new directions 
for the field. The data structure of molecules and the rotational invariance 
require more general machine learning models, such as Graph neural networks (GNN) with its 
general data structure and message-passing formalism. 

In this project, I implement state-of-the-art GNN models in the context of 
molecular property predictions. The specific molecular property data focus on 
*HOMO-LUMO gap energies* and *dipole moments*. Both properties are important factors for a molecule's 
reactivity and interaction with biological targets. The GNN models are trained on 
the molecular properties of 130k+ small organic molecules. The hyperparameters of 
the models are tuned. The GNN models show great performance that reach well 
beyond the chemical accuracy.

## <a name="gnnModels"></a> GNN Models

Chemistry and materials science applications call for specialized MPNN models that 
handle unevenly spaced data and update edge, node, and global attributes better than general Message Passing Neural Networks (MPNNs). Thus, in this project, I implement two GNN models.

The first one uniquely uses continuous-filter convolutional layers, as opposed to 
the discrete convolutions typically found in GCNs. This enables it to capture local 
correlations in unevenly spaced data, i.e., atomic positions in molecules. This is 
inspired by the [SchNet model](
https://doi.org/10.48550/arXiv.1706.08566) [1]. 

The second model uses a convolutional 
operator to update graph attributes instead of simple linear operations in standard 
GCNs. It also uses a pooling layer to aggregate node information to predict molecular 
properties. This model is similar to the [Materials Graph Network (MEGNet)](https://doi.org/10.1021/acs.chemmater.9b01294) [2]. Both models take into account quantum chemistry 
constraints like rotational equivariance. 

<figure>
    <img src="./figures/SchNet_MEGNET_structure.png"
         alt="Plot by heavist atom"
         width="700">
    <figcaption>Figure 5. (a) The use of continuous filter in the SchNet model. [1] (b) Architecture for the MEGNet model. [2]</figcaption>
</figure>

Note that there are many other GNN models with high performance in the literature. 
[OrbNet](https://doi.org/10.1063/5.0021955) [3], 
[Crystal Graph Convolutional Neural Network 
(CGCNN)](https://doi.org/10.1103/PhysRevLett.120.145301) [4], 
[Atomistic Line Graph Neural Network (ALIGNN)](https://doi.org/10.1038/s41524-021-00650-1) [5], 
Gated Graph Neural Networks (GG-NN), and Deep Tensor Neural Networks are some 
models in the literature with great performance in the chemical and materials domain.

## <a name="dataset"></a> Dataset

The [QM9 dataset](https://doi.org/10.1038/sdata.2014.22) [6] is used for this project. 
QM9 contains a comprehensive collection of data on 134,000 small organic molecules. These molecules were calculated at the 
B3LYP/6-31G(2df,p) level of DFT. For this project, I also extracted two crucial chemical properties among the 19 given in QM9: the HOMO-LUMO gap energies and dipole moments. Both properties are important factors for a molecule's reactivity and interaction with biological targets. 

To facilitate machine learning applications, I  conducted rigorous data cleaning 
and extraction. The molecular representations in SMILES format were converted into graph 
data structures that are compatible with the GNN model coded in PyTorch Geometric. Atoms 
are denoted as nodes and bonds as edges. Node features and edge attributes were 
systematically extracted from the intrinsic chemical properties of the atoms and bonds. 
Edge weights were initialized to uniform due to the absence of a priori information. 
This preprocessing was executed using the RDKit package. The dataset was partitioned into 
training, validation, and test sets in an 80-10-10 ratio. 

## <a name="hyperTuning"></a> Hyperparameter Tuning

Hyperparameter tuning is crucial for optimizing the performance of machine learning 
models. In this project, I use the efficient and effective hyperparameter optimization 
process of Async Successive Halving via RayTune, which has similar performance to 
HyperBand.

The hyperparameter search space is defined over all available parameters (including 
pre-GNN MLP depth, GNN depth, post-GNN MLP depth, cutoff distance, learning rates, batch 
sizes, etc.) and is widely customizable by users. 

The optimal hyperparameters for each GNN model (SchNet and MEGNET) with HOMO-LUMO gap as 
the training target were set as defaults in the code. 


## References

[1] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller. SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. Advances in Neural Information Processing Systems 30, pp. 992-1002 (2017). [Paper](https://doi.org/10.48550/arXiv.1706.08566)

[2] C. Chen, W. Ye, Y. Zuo, C. Zheng, and S.P. Ong. Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals. Chemistry of Materials 2019 31 (9), 3564-3572. [Paper](https://doi.org/10.1021/acs.chemmater.9b01294)

[3] Z. Qiao, M. Welborn, A. Anandkumar, F.R. Manby, T.F. Miller; OrbNet: Deep learning for quantum chemistry using symmetry-adapted atomic-orbital features. J. Chem. Phys. 28 September 2020; 153 (12): 124111. [Paper](https://doi.org/10.1063/5.0021955)

[4] T. Xie and J.C. Grossman. Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties. Phys. Rev. Lett. 2018 120 (4). [Paper](https://doi.org/10.1103/PhysRevLett.120.145301)

[5] K. Choudhary, B. DeCost. Atomistic Line Graph Neural Network for improved materials property predictions. npj Comput Mater 7, 185 (2021). [Paper](https://doi.org/10.1038/s41524-021-00650-1)

[6] R. Ramakrishnan, P. Dral, M. Rupp, et al. Quantum chemistry structures and properties of 134 kilo molecules. Sci Data 1, 140022 (2014). [Paper](https://doi.org/10.1038/sdata.2014.22)