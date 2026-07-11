# KrylovQuantumClassical

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19452068.svg)](https://doi.org/10.5281/zenodo.19452068)

KrylovQuantumClassical is the software developed for the numerical computations in the article:

* From phase space to Krylov space, one shell at a time, De Ro, N., Sánchez-Garrido, A., and Sonner, J. (2026). [[preprint](https://arxiv.org/abs/)]

**Please cite this article if you use (a part of) this software for a publication.**

In this article, we study and build the framework of classical Krylov complexity, and analyze its connection with the well-known quantum Krylov complexity. We construct a formal semiclassical relation between the classical and quantum regimes, that we name <i>Krylov-Ehrenfest theorem</i>. We also explore microcanonical Krylov complexity for which the Krylov dynamics unfolds within fixed energy shells. We apply these notions to the Lipkin-Meshkov-Glick (LMG) and Feingold-Peres (FP) models. 

<p float="left">
  <img src="https://github.com/Climdyn/Pydlosky/blob/main/Animation/Lanczos_animation.gif?raw=true" height="320" />
  <img src="https://github.com/Climdyn/Pydlosky/blob/main/Animation/K_complexity_evolution.gif?raw=true" height="320" /> 
</p>

## General information

KrylovQuantumClassical includes various Python classes to define the quantum and classical LMG and FP as well as infinite-temperature ad microcanonical classical and quantum Lanczos algorithm in the [`KrylovQuantumClassical`](./KrylovQuantumClassical) folder. Some of the classes rely on [spherical](https://doi.org/10.5281/zenodo.4045222) to perform efficient heavy computations involving spherical harmonics. 

The study itself is contained in the [project notebook](./project_notebook.ipynb) where the user is guided over the various results obtained. Note that the parameters are adapted for the notebook to execute in a reasonable execution time, and does not reflect all results displayed in the main article.

For using the software, the user needs to create the corresponding [Anaconda](https://www.anaconda.com/) environment with

    conda env create -f environment.yml

The Lanczos algorithm (both infinite-temperature and microcanonical versions) are written in Wolfram Language, for reasons that are detailed in the project notebook. The software uses [WolframKernel](https://pypi.org/project/wolframclient/) which is a Wolfram Language integration in Python. Critically, this softwares requires a local installation of either:
   1. Mathematica (v11.3+)
   2. The free Wolfram Engine for Developers (https://www.wolfram.com/engine/)
The wolframclient library will fail if a local WolframKernel is not found.

Once the installation has been performed, the user can start `jupyter-notebook`:

    conda activate pydlosky
    cd Notebook
    jupyter-notebook

and load the project notebook.
