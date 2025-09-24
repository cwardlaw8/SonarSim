import numpy as np

def eval_f_SquaredDiagonal(x, p, u):
    """
    example of function that evaluates the vector field f(x,p,u) 
    at state vector x, and with vector of inputs u.
    p is a structure containing all model parameters
    i.e. in this case: matrices p.A and p.B and vector p.sqd
    corresponding to state space model dx/dt = p.A x + sqd(x) + p.B u
    where the i-th component of sqd(x) is p.sqd[i] * (x[i])**2

    f = eval_f_SquareDiagonal(x,p,u)
    """

    f = (p['A'].dot(x)) + (p['sqd'] * x**2) + (p['B'].dot(u))
    return f
