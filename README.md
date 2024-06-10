#Signature Extraction Evaluation Framework V2 (SEEF2)

SEEF2 is a framework that is improve from last semester project, it can be used to compare different extraction method for mutational signatures, it has both the pre and post methods implemented, in the pre-process it can generate synthetic data to be used to evaulate with, or just use real data, feature filtering based on a threshold, boostraping, add noise to the data, and add injection to the data, for post-processing it has different clustering methods implemented, can use different silhouette metric, weight to how much silhouette score is contributing to the final score, and a new optimal selection if running multi method extraction run, where it will add the best cluster from each method extraction run and cluster it again. It also has a method to evaulate the extraction method based on the latent space and the known signatures, it can also be used to evaulate the extraction method on real data,



## Table of Contents

- [1 Requirements](#1-usage)
- [2 Installation](#2-installation)
- [3 Running](#3-running)
- [4 License](#4-license)
- [5 Contributers](#5-contributers)
- [6 Acknowledgements](#6-acknowledgements)
- [7 previous work](#7-previous-work)


## 1 Requirements
- python >= 3.11.9
- pandas >=1.5.3
- scikit-learn >=1.2.2
- SigProfilerAssignment >=0.1.4
- SigProfilerMatrixGenerator >=1.2.25
- SigProfilerPlotting >=1.3.22
- numpy >=1.26.4
- pytorch >=2.1.2
- scipy >= 1.13.0
- plotnine >= 0.12.4


## 2 Installation
### 2.1 Code
To download the code do
```shell
    git clone https://github.com/Magnijh/Master_thesis
```
### 2.2 Known Mutational Signatures
To get the COSMIC signatures go to
```shell
    https://cancer.sanger.ac.uk/signatures/downloads/
```

And download the file belonging to **GRCh37** and **SBS** as we work with Single-base substitutions

### 2.3 Real data
To get the real data do
```shell
    wget https://dcc.icgc.org/api/v1/download?fn=/PCAWG/mutational_signatures/Input_Data_PCAWG7_23K_Spectra_DB/Mutation_Catalogs_--_Spectra_of_Individual_Tumours/WGS_PCAWG_2018_02_09.zip
```
Unzip it



## 3 Running

### 3.0 Example file
there is a file called `example.py` in the root folder that can be used to run the framework, it is a example on how to use the framework, and can be used as a template for how to use the framework

```shell
    python example.py
```
but if you to implement your own method you can follow the steps below

### 3.1 Creating synthetic dataset
Creating synthetic dataset you need to use the function `create_dataset` from the file `synthetic_dataset.py` in folder `sigGen`, there is 3 parameter required for the function to run, 
1. number of signature
2. number of how many samples
3. path to Known Mutational Signatures file

Example

```python
create_dataset(5,5,"path/to/sig.txt")
```

### 3.2 Extraction Method 
when creating a Extraction Method there is some requirement both for the input parameters and for the return for it to work with the framework

**input**
1. pandas dataframe
2. number of componets/latent

**output**
1. latents space
2. weights
3. loss

### 3.4 SEEF
for using the framework, it requires 3 parameters, tho it has more parameters that can be used, but the 3 required parameters are
- path to data
- extraction method (remember not to call the function, but give pointer to function)
- output

the framework it handles the rest, it will cluster the data, evaluate the extraction method you only need to tell it when to save the results

## 4 License

This project is licensed under the [MIT License](LICENSE). See the `LICENSE` file for more information.

## 5 Contributers
contributors to this master these are

-Magni Jógvansson Hansen - Mjha19@student.aau.dk

-Nikolai Eriksen Kure - Nkure19@student.aau.dk


## 6 Acknowledgements
This project is part of the Master Thesis project at Aalborg University, Denmark, under the supervision of Assoc. Prof. Daniele Dell'Aglio and Senior Bioinformatician Rasmus Froberg Brøndum.

## 7 previous work
The previous work can be found at
    https://github.com/Magnijh/SEEF