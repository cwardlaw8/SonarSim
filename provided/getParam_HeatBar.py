import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

def getParam_HeatBar(N, L=1.0,UseSparseMatrices=True):
    """
    Defines the parameters for vector field f(x,p,u) = p.A x+ p.B u
    corresponding to a 1D hear conducting bar problem
    the full state space model equations are dx/dt = f(x,p,u) where
    x is the state vector of the system
    u is the vector of inputs to the system
    p is a structure containing all model parameters  
    Since f(x,p,u) is a linear vector field it is more efficient 
    to precompute here only once matrices p.A and p.A 
    and the use the generic functions eval_f_LinearSystem eval_Jf_LinearSystem
    every time an evaluation of f(x,p.u) and its Jacobian will be needed

    INPUT:
    N          the number of discretization nodes/states/unknowns along the bar
                including the two terminal nodes 
    L          [optional] total lenght of the bar, if not specified L=1 is used

    OUPUTS:
    p.gamma    coef. related to thermal capacitance per unit length of the bar
    p.km       thermal conductance through metal per unit length of the bar
    p.ka       thermal conductance to air per unit length of the bar
    p.dz       leanth of each bar section
    p.A        dynamical matrix of the state space system (Laplacian discretization)
    p.B        input matrix (one column for each input) in this case single input on the left
    x_start    if needed for typical transient simulations
    x_stop     if needed for typical transient simulations
    max_dt_FE  useful if using Forward Euler for transient simulations

    EXAMPLE:
    [p,x_start,t_start,t_stop,max_dt_FE] = getParam_HeatBar(N);
    """
    # Define material properties parameters
    p = {
        'gamma': 0.1,  # Related to thermal capacitance per unit length of the bar
        'km': 0.1,     # Related to thermal conductance through metal per unit length of the bar
        'ka': 0.1      # Related to thermal conductance to air per unit length of the bar
    }

    p['dz'] = L / (N - 1)   # length of each segment. N includes the two terminal nodes
                            # notice when changing N what stays constant is the total length L
                            # not the length of each discretization section.
                            # results should not depend on the number of sections
                            # for a large enough number of sections
    Cstore = p['gamma'] * p['dz']       # The longer the section, the larger the thermal storage
    Rc = (1.0 / p['km']) * p['dz']      # The longer the section, the larger the thermal resistance
    Rloss = 1.0 / (p['ka'] * p['dz'])   # The longer the section, the smaller the thermal resistance to ambient

    if UseSparseMatrices:
        # p['A'] = sp.csr_matrix((N, N), dtype=float) # allocate space for large sparse dynamic matrix in advance
        p['A'] = sp.lil_matrix((N, N), dtype=float)
    else:
        p['A'] = np.zeros((N, N))

    # Coupling resistors Rc between i and j = i + 1
    for i in range(N - 1):
        j = i + 1
        p['A'][i, i] += 1.0 / Rc
        p['A'][i, j] -= 1.0 / Rc
        p['A'][j, i] -= 1.0 / Rc
        p['A'][j, j] += 1.0 / Rc

    # Leakage resistor Rloss between i and ground
    for i in range(N):
        p['A'][i, i] += 1.0 / Rloss

    if UseSparseMatrices:
        # p['B'] = sp.csr_matrix((N, 1), dtype=float) # makes sure all elements are sparse 
                                                # otherwise matlab will convert them to full when adding
        p['B'] = sp.lil_matrix((N, 1), dtype=float)
    else:
        p['B'] = np.zeros((N,1))
    p['B'][0] = 1.0  # Heat source at the leftmost side of the bar

    p['A'] = -p['A'] / Cstore   # note this will result in a 1/p.dz^2 term in A
                                # also pay attention to the negative sign
                                # otherwise the system is completely unstable
    p['B'] = p['B'] / Cstore    # note this is important to make sure results
                                # will not depend on the number of sections N

    # define also some example parameters for a typical transient simulation     
    x_start = np.zeros((N,1))
    t_start = 0
    if UseSparseMatrices:
        slowest_eigenvalue = np.abs(sp.linalg.eigs(p['A'], k=1, which='SM', return_eigenvectors=False))
        fastest_eigenvalue = np.abs(sp.linalg.eigs(p['A'], k=1, which='LM', return_eigenvectors=False))
    else:
        lambda_ = np.linalg.eigvals(p['A'])
        slowest_eigenvalue = min(np.abs(lambda_))
        fastest_eigenvalue = max(np.abs(lambda_))
    
    # if p['A'] is symmetric
    slowest_eigenvalue = np.real(slowest_eigenvalue)
    fastest_eigenvalue = np.real(fastest_eigenvalue)

    # to see steady state need to wait until the slowest mode settles
    t_stop = slowest_eigenvalue * 2.0  # use this to wait until slowest mode
    max_dt_FE = 1.0 / fastest_eigenvalue  
    return p, x_start, t_start, t_stop, max_dt_FE
