# Spatio-Temporal Deep Learning for Multi-Contaminant Forecasting in Sparse Watershed Networks
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)

> **Note:** All the commands are based on a Unix based system.
> For a different system look for similar commands for it.
## Evaluation metrics on test set
![alt text](figures/05_metrics_table.png)

## Spatial Pollutant maps
![alt text](figures/02_spatial_maps.png)

## Parity maps
![alt text](figures/04_parity_diagrams.png)


## Setup

We are using Python version 3.11.9

```bash
$ python --version
Python 3.11.9
```
### Requirements

```bash
# Clone and install
git clone https://github.com/kagozi/Multi-Contaminant-Forecasting.git
cd Multi-Contaminant-Forecasting
```
### Python virtual environment

**Create** a virtual environment:

```bash
python3 -m venv .venv
```
`.venv` is the name of the folder that would contain the virtual environment.

**Activate** the virtual environment:

```bash
source .venv/bin/activate
```

**Windows**
```bash
source .venv/Scripts/activate
```


```bash
pip install -r requirements.txt
```

## Run full pipeline: preprocess → train → evaluate

```bash
python3 main.py
```