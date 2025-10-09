import numpy as np
import scipy.sparse as sp

def eval_f_Sonar(x, p, u):
    """
    evaluates the vector field f(x, p, u) 
    at state vector x, and with vector of inputs u.
    p is a structure containing all model parameters
    i.e. in this case: matrices p.A and p.B 
    corresponding to state space model dx/dt = p.A x + p.B u

    f = eval_f_Sonar(x,p,u)
    """
    # Robust multiplication supports sparse B and scalar u
    A = p['A']
    B = p['B']

    Ax = A.dot(x)

    # Allow scalar or length-1 array for u
    if np.isscalar(u) or (isinstance(u, np.ndarray) and u.size == 1):
        u_val = float(u)
        if sp.issparse(B):
            Bu = (B * u_val).toarray()
        else:
            Bu = B * u_val
    else:
        # Fallback: assume compatible shape for B @ u
        Bu = B.dot(u)

    f = Ax + Bu
    return f
