#!/bin/bash

python3 symmetric_cavity-easy.py
python3 symmetric_cavity-high-contrast.py
python3 symmetric_cavity.py

python3 SolveMetaMaterial-stab.py 1
python3 SolveMetaMaterial-stab.py 0

python3 unsymmetric-cavity-tnorm.py
