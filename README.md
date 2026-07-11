# KrylovQuantumClassical

[![DOI](https://zenodo.org/badge/DOI/)](https://doi.org/)

KrylovQuantumClassical is the software accompanying the numerical computations presented in the article:

* From phase space to Krylov space, one shell at a time, De Ro, N., Sánchez-Garrido, A., and Sonner, J. (2026). [[preprint](https://arxiv.org/abs/)]

**Please cite this article if you use (a part of) this software for a publication.**

This work develops a framework for classical Krylov complexity and investigates its connection with the well-established notion of quantum Krylov complexity. We derive a semiclassical correspondence between the classical and quantum regimes, which we call the Krylov–Ehrenfest theorem. We also introduce a microcanonical formulation of Krylov complexity, in which Krylov dynamics unfolds within fixed-energy shells. These ideas are illustrated using the Lipkin–Meshkov–Glick (LMG) and Feingold–Peres (FP) collective spin models, both of which possess a well-defined classical limit in the large-spin regime.

<p float="left">
  <img src="https://github.com/nicolasdero/KrylovQuantumClassical/blob/main/Animation/Lanczos_animation.gif?raw=true" height="275" />
  <img src="https://github.com/nicolasdero/KrylovQuantumClassical/blob/main/Animation/K_complexity_evolution.gif?raw=true" height="275" /> 
</p>

## General information

KrylovQuantumClassical provides Python classes implementing the quantum and classical Lipkin–Meshkov–Glick (LMG) and Feingold–Peres (FP) models, together with infinite-temperature and microcanonical implementations of the Lanczos algorithm. The classes are defined in the [`KrylovQuantumClassical`](./KrylovQuantumClassical) folder. Some of the classes rely on [spherical](https://doi.org/10.5281/zenodo.4045222) to perform computationally intensive operations involving spherical harmonics. 

Most examples and numerical reproductions are provided in the [project notebook](./project_notebook.ipynb) which guides the user through the main results.. Note that the parameters are adapted for the notebook to execute in a reasonable execution time, and does not reproduce every figure shown in the paper.

The Lanczos algorithms (both infinite-temperature and microcanonical versions) are implemented in the Wolfram Language to leverage arbitrary-precision floating-point arithmetic, which is not available natively in Python. The software interfaces with the Wolfram Language through [wolframclient](https://pypi.org/project/wolframclient/), which requires access to a local WolframKernel.

The repository has the following structure:

```text
KrylovQuantumClassical/
├── KrylovQuantumClassical/    Python source
├── Animation/                 GIFs
├── project_notebook.ipynb     Jupyter notebook 
├── animation_notebook.ipynb   Jupyter notebook 
├── LanczosAlgorithmLMG.wl     Mathematica script
├── LanczosAlgorithmFP.wl      Mathematica script
├── LanczosAlgorithmMCLMG.wl   Mathematica script
├── LanczosAlgorithmMCFP.wl    Mathematica script
├── environment.yml
├── LICENSE
└── README.md
```

## Installation

To install the software:

Clone the repository

    git clone https://github.com/nicolasdero/KrylovQuantumClassical.git
    cd KrylovQuantumClassical

Create the environment

    conda env create -f environment.yml

Install either Mathematica (v11.3 or later) or the free Wolfram Engine for Developers if a local WolframKernel is not already available. Note that the wolframclient library will fail if a local WolframKernel is not found.

After installation, the user can start `jupyter-notebook`:

    conda activate KrylovQuantumClassical
    jupyter-notebook

and open <code>project_notebook.ipynb</code> to reproduce the examples presented in the paper.

## License

This project is distributed under the MIT License.