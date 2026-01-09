# ECG-Greedy: Multi-Modal ECG Classification with CWT Scalograms & Phasograms 
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
<!-- [![Paper](https://img.shields.io/badge/ISBI-2025-blue)](https://biomedicalimaging.org/2025/)   -->
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)

> **Note:** All the commands are based on a Unix based system.
> For a different system look for similar commands for it.

## Setup

We are using Python version 3.11.9

```bash
$ python --version
Python 3.11.9
```
### Requirements

```bash
# Clone and install
git clone https://github.com/kagozi/MultiModal-ECG.git
cd MultiModal-ECG
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