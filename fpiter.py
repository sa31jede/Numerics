import numpy as np

def fpiter(fun, x0, p = 2, tol = 1e-9, nmax = 100000, verbose = True):
    """
    Computes the fixed point of a continuous function fun: R^d -> R^d.
    input:
    ------
    fun  : R^d-valued function on R^d
           a continuous function which governs the iteration process
    x0   : np.array (of shape d×1)
           the initial value for the iteration process
    p    : 1, 2, np.Inf
           determines the norm used to measure the error
    tol  : positive real
           the error tolerance
    nmax : positive integer
           maximum number of iterations
    output:
    -------
    xstar : real
            the fixpoint
    hist  : array
            iteration history 
    """
    
    x0 = np.asarray(x0, dtype = float)
    hist = [x0]
    
    # First step calculation
    x_next = fun(x0)
    hist.append(x_next)
    
    if verbose:
        print(f"x0 = {x0}")
        print(f"x1 = {x_next}")

    theta = 0.0
    
    while len(hist) - 1 < nmax:
        x_curr = hist[-1]
        x_next = fun(x_curr)
        
        step_norm = np.linalg.norm(x_next - x_curr, p)
        prev_step_norm = np.linalg.norm(x_curr - hist[-2], p)
        
        # Avoid division by zero if convergence is instant
        theta = step_norm / prev_step_norm if prev_step_norm > 0 else 0.0
        
        hist.append(x_next)
        
        if verbose:
            print(f"x{len(hist)-1} = {x_next} (theta = {theta:.6e}, step = {step_norm:.6e})")
            
        # Check standard fixed-point error tolerance (step size < tol)
        if step_norm < tol:
            break
            
        # Divergence check
        if theta > 1.0:
            if verbose:
                print(f"Warning: Divergence detected (theta = {theta:.4f} > 1)")
            break

    if verbose:
        if len(hist) - 1 >= nmax:
            print(f"Warning: Reached maximum iterations ({nmax}).")
        elif theta <= 1.0:
            print(f"x* = {hist[-1]}")
            
    return {"xstar": hist[-1], "hist": hist}
