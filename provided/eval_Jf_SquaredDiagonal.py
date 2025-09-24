import numpy as np

def eval_Jf_SquaredDiagonal(x, p, u):
    """
    example of function that evaluates the Jacobian of vector field f(x,p,u)
    at state vector x, with vector of inputs u.
    p is a structure containing all model parameters.
    i.e. in this case matrices p.A and p.B and vector p.sqd
    defining the state space model dx/dt = p.A x + sqd(x) + p.B u
    where the i-th component of sqd(x) is p.sqd[i] * (x[i])**2.

    Jf = eval_Jf_SquaredDiagonal(x,p,u)
    """
    Jf = p['A'] + np.diagflat(2 * p['sqd']*x)
    return Jf
