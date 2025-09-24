import numpy as np

def eval_f_Sonar(x, p, u):
    """
    evaluates the vector field f(x, p, u) 
    at state vector x, and with vector of inputs u.
    p is a structure containing all model parameters
    i.e. in this case: matrices p.A and p.B 
    corresponding to state space model dx/dt = p.A x + p.B u

    f = eval_f_Sonar(x,p,u)
    """
    # import pdb; pdb.set_trace()
    f = p['A'].dot(x) + p['B'].dot(u)
    return f