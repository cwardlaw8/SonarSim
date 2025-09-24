import numpy as np

def eval_Jf_Sonar(x, p, u):
    """
    evaluates the Jacobian of the vector field f(x, p, u)
    at state vector x, and with vector of inputs u.
    p is a structure containing all model parameters.
    i.e. in this case matrices p.A and p.B
    corresponding to state space model dx/dt = p.A x + p.B u.
    note that in this case, the Jacobian is trivially just p.A.

    f = eval_Jf_linear(x,p,u)
    """
    Jf = p['A']

    # notice this is a trivial implementation since
    # the linearation of a linear function is the function itself.
    # however it is still important to have a separate function for this Jacobian
    # for consistency accross the code base since 
    # evert eval_f should have its own eval_Jf including the trivial ones
    return Jf
