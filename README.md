# HULC
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[<b>What Matters in Language Conditioned Imitation Learning</b>](https://arxiv.org/pdf/foo.pdf)

[Oier Mees](https://www.oiermees.com/), [Lukas Hermann](http://www2.informatik.uni-freiburg.de/~hermannl/), [Wolfram Burgard](http://www2.informatik.uni-freiburg.de/~burgard)

 We present **HULC** (**H**ierarchical **U**niversal **L**anguage **C**onditioned Policies), an end-to-end model that can 
 learn  a wide variety of language conditioned robot skills from  offline free-form imitation datasets. HULC sets a new state of the art on the challenging CALVIN benchmark, 
 on learning a single 7-DoF policy that can perform long-horizon manipulation tasks in a 3D environment, directly from images, and only specified with natural language.
This code accompanies the paper What Matters in Language Conditioned Imitation Learning, which can be found here. 
We hope the code will be useful as a starting point for further research on language conditioned policy learning and will bring us closer towards general-purpose robots that can relate human language to their perception and actions. 

# Installation
To begin, clone this repository locally
```bash
git clone --recurse-submodules https://github.com/mees/hulc.git
export HULC_ROOT=$(pwd)/hulc

```
Install requirements:
```bash
cd $HULC_ROOT
conda create -n hulc_venv python=3.8  # or use virtualenv
conda activate hulc_venv
sh install.sh
```
If you encounter problems installing pyhash, you might have to downgrade setuptools to a version below 58.

# Download 
## CALVIN Dataset
If you want to train on the [CALVIN](https://github.com/mees/calvin) dataset, choose a split with:
```bash
$ cd $HULC_ROOT/dataset
$ sh download_data.sh D | ABC | ABCD
```
## Language Embeddings
We provide the precomputed embeddings of the different Language Models we evaluate in the paper.
The script assumes the corresponding split has been already downloaded.
```bash
$ cd $HULC_ROOT/dataset
$ sh download_lang_embeddings.sh D | ABC | ABCD
```

## Pre-trained Models
We provide our final models for all three CALVIN splits.
```bash
$ cd $HULC_ROOT/checkpoints
$ sh download_model_weights.sh D | ABC | ABCD
```