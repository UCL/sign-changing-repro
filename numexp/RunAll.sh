 #!/bin/bash

killall -9 python3
python3 symmetric_cavity-easy.py
killall -9 python3
python3 symmetric_cavity-high-contrast.py
killall -9 python3
python3 symmetric_cavity.py
killall -9 python3
python3 SolveMetaMaterial-stab.py 1
killall -9 python3
python3 SolveMetaMaterial-stab.py 0
killall -9 python3
python3 unsymmetric-cavity-tnorm.py
killall -9 python3
