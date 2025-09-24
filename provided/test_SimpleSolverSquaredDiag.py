import numpy as np
from SimpleSolver import SimpleSolver
from getParam_SquaredDiagonal import getParam_SquaredDiagonal
from eval_f_SquaredDiagonal import eval_f_SquaredDiagonal
from eval_u_Sonar import eval_u_step

print('\n******** test_SimpleSolverSquareDiag.py **************')
print('Running: test_SimpleSolverSquareDiag.py')
print('tests state space model with nonlinear (i.e. squared diagonal) vector field')
print('showing time domain simulation as debugging tool\n')

# select input evaluation functions
eval_u = eval_u_step
# eval_u = 'something else...'

# Example with a two state linear system plus squared diagonal nonlinearity
p, x_start, t_start, t_stop, w_max = getParam_SquaredDiagonal()
eval_f = eval_f_SquaredDiagonal

visualize = True
w = w_max
num_iter = np.ceil((t_stop - t_start) / w)

[X, t] = SimpleSolver(eval_f, x_start, p, eval_u, num_iter, w, visualize, gif_file_name="test_SimpleSolver_SquaredDiag.gif")
