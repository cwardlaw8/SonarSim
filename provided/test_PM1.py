import numpy as np
import os
import subprocess

print('\nthis scripts tests all functions distributed with PM1')

subprocess.run(['python', 'test_VisualizeNetwork.py'])
input('press a key to continue')

subprocess.run(['python', 'test_SimpleSolverSquaredDiag.py'])
input('press a key to continue')

subprocess.run(['python', 'test_SimpleSolverHeatBar.py'])
input('press a key to continue')

subprocess.run(['python', 'test_Jf_SquaredDiagonal.py'])
input('press a key to continue')

subprocess.run(['python', 'test_Jf_HeatBar.py'])