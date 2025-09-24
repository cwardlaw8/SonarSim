import numpy as np
import matplotlib.pyplot as plt
from getParam import getParam_HeatBar
from eval_f import eval_f_LinearSystem
from eval_Jf_Sonar import eval_Jf_LinearSystem
from eval_u_Sonar import eval_u_step
from eval_Jf_FiniteDifference import eval_Jf_FiniteDifference

print('\n****** Running: test_Jf_HeatBar.py *********')
print('tests Heat Conducting Bar analytical Jacobian function')
print('showing difference from finite difference Jacobian as debugging tool')

# Heat Conducting Bar Example
p, x, t_start,_,_ = getParam_HeatBar(10)
eval_f = eval_f_LinearSystem
eval_Jf = eval_Jf_LinearSystem

# select input evaluation functions
eval_u = eval_u_step
u = eval_u(t_start)

# test Analytical Jacobian function vs general Finite Difference Jacobian
Jf_Analytical = eval_Jf(x, p, u)

dxFD = 0.1  # the function is linear, so you can make this quite large
p['dxFD'] = dxFD
Jf_FiniteDifference,_ = eval_Jf_FiniteDifference(eval_f, x, p, u)
difference_J_an_FD = np.max(np.abs(Jf_Analytical - Jf_FiniteDifference))

# printing the Jf_Analytical and Jf_FiniteDifference matrices
print('\nJf_Analytical:')
print(Jf_Analytical)
print("dxFD =", dxFD)
print('\nJf_FiniteDifference:')
print(Jf_FiniteDifference)
print('\nDifference between Analytical and Finite Difference Jacobians:')
print(difference_J_an_FD)


# plot to study error on Finite Difference Jacobian for different dx
k = 0
dx = []
error = []

for n in np.arange(-17, 5, 0.01):
    dx.append(10 ** (n))
    p['dxFD'] = dx[k]
    Jf_FiniteDifference,_ = eval_Jf_FiniteDifference(eval_f, x, p, u)
    error.append(np.max(np.abs(Jf_Analytical - Jf_FiniteDifference)))
    k += 1

plt.loglog(dx, error)
plt.grid(True)
plt.xlabel('dxFD')
plt.axvline(x=dxFD, color='g', linestyle='--', label='dxFDnitsol')
plt.legend(['|| J_{FD}-J_{an} ||', 'dxFD'])
plt.title('Difference between Analytic & Finite Difference Jacobians')
plt.show()

