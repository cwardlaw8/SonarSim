import numpy as np

def getParam_SquaredDiagonal():
    """
    defines the parameters used by eval_f_SquaredDiag 
    which in turns returns the nonlinear vector field f(x,p,u)=p.A x+ sqd(x) + p.B u
    for the nonlinear state space model dx/dt = f(x,p,u) where
    x is the state vector of the system
    u is the vector of inputs to the system
    p is a structure containing all model parameters
    i.e. in this case: matrices p.A and p.B and vector p.sqd
    where the i-th component of sqd(x) is p.sqd(i) * (x(i))^2

    OUTPUTS:
    p.A        dynamical matrix of the state space system (Laplacian discretization)
    p.B        input matrix (one column for each input) in this case single input
    p.sqd      vector of coefficients for the squared terms 
    x_start    initial state at initial time t_start if needed for transient simulations
    x_stop     defines the end of the desired transient simulation 
    max_dt_FE  estimation for the maximum timestep value that can be used
               to avoid instability in an explicit integrator such as Forward Euler

    [p,x_start,t_start,t_stop,max_dt_FE] = getParam_SquaredDiagonal
    """
    # example of a 2 state system with decaying modes
    p = {
        'A': np.array([[-1.0, 0.0], [0.0, -2.0]]),   
        'B': np.array([[20.0], [5.0]]),               
        'sqd': np.array([[-3.0], [-5.0]])             
    }

    x_start = np.array([[5.0], [10.0]])
    t_start = 0
    t_stop = 0.1

    # note for linear systems FE may go unstable for dt>2/|fastest_eigenvalue| = 2/|-2|=1
    # but in nonlinear systems FE could go unstable even for much smaller values...
    max_dt_FE = 0.001

    return p, x_start, t_start, t_stop, max_dt_FE
