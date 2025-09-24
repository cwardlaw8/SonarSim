import numpy as np
import matplotlib.pyplot as plt
from getParam_SquaredDiagonal import getParam_SquaredDiagonal
from eval_f_SquaredDiagonal import eval_f_SquaredDiagonal
from eval_Jf_SquaredDiagonal import eval_Jf_SquaredDiagonal
from eval_Jf_FiniteDifference import eval_Jf_FiniteDifference
from eval_u_Sonar import eval_u_step

print('\n****** Running: test_Jf_SquaredDiagonal.py *******')
print('tests Squared Diagonal analytical Jacobian function')
print('showing difference from finite difference Jacobian as debugging tool\n')

# Example with two-state linear system plus squared diagonal nonlinearity
p, x, t_start,_,_ = getParam_SquaredDiagonal()
eval_f = eval_f_SquaredDiagonal
eval_Jf = eval_Jf_SquaredDiagonal

# select input evaluation functions
eval_u = eval_u_step
u = eval_u(t_start)

# test Analytical Jacobian function vs general Finite Difference Jacobian
Jf_Analytical = eval_Jf(x, p, u)

dxFDeps = np.sqrt(np.finfo(float).eps)
print("dxFDeps =", dxFDeps)
# p['dxFD'] = dxFDeps   # a good choice if machine precision is not known
# p['dxFD'] = 1e-7      # a conservative choice for double precision machines
# if p['dxFD'] is not specified, eval_Jf_FiniteDifference will default
# to a potentially better value proposed in the solver NITSOL
Jf_FiniteDifference, dxFDnitsol = eval_Jf_FiniteDifference(eval_f, x, p, u)

difference_J_an_FD = np.max(np.abs(Jf_Analytical - Jf_FiniteDifference))

# printing the Jf_Analytical and Jf_FiniteDifference matrices
print('\nJf_Analytical:')
print(Jf_Analytical)
print('\nJf_FiniteDifference:')
print(Jf_FiniteDifference)
print('\nDifference between Analytical and Finite Difference Jacobians:')
print(difference_J_an_FD)

print("dxFDnitsol =", dxFDnitsol)
# plot to study error on Finite Difference Jacobian for different dx
k = 0
dx = []
error = []

for n in np.arange(0, 15, 0.05):
    dx.append(10 ** (-n))
    p['dxFD'] = dx[k]
    Jf_FiniteDifference,_ = eval_Jf_FiniteDifference(eval_f, x, p, u)
    error.append(np.max(np.abs(Jf_Analytical - Jf_FiniteDifference)))
    k += 1

plt.loglog(dx, error)
plt.grid(True)
plt.xlabel('dxFD')
plt.axvline(x=dxFDeps, color='k', linestyle='--', label='sqrt(eps)')
plt.axvline(x=dxFDnitsol, color='g', linestyle='--', label='dxFDnitsol')
plt.legend(['|| J_{FD}-J_{an} ||', 'sqrt(eps)','dxFDnitsol'])
plt.title('Difference between Analytic & Finite Difference Jacobians')
plt.show()

