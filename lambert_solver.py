import numpy as np
from scipy import optimize

def lambert_solver(r1, r2, tof, mu=1.32712440018e20, prograde=True, max_iterations=100, tolerance=1e-12):
    """
    Lambert problem solver for both elliptical and hyperbolic transfers.
    
    Parameters:
    -----------
    r1 : array-like
        Initial position vector [x, y, z] in meters
    r2 : array-like
        Final position vector [x, y, z] in meters
    tof : float
        Time of flight in seconds
    mu : float, optional
        Gravitational parameter of the central body (default: Sun's GM)
    prograde : bool, optional
        True for prograde motion, False for retrograde
    max_iterations : int, optional
        Maximum number of iterations for the solver
    tolerance : float, optional
        Convergence tolerance
        
    Returns:
    --------
    v1 : ndarray
        Initial velocity vector
    v2 : ndarray
        Final velocity vector
    a : float
        Semi-major axis of the transfer orbit (positive for elliptical, negative for hyperbolic)
    """
    # Convert inputs to numpy arrays
    r1 = np.array(r1, dtype=float)
    r2 = np.array(r2, dtype=float)
    
    # Calculate magnitudes
    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    
    # Calculate the cosine of the angle between r1 and r2
    cos_dnu = np.dot(r1, r2) / (r1_norm * r2_norm)
    
    # Ensure numerical stability
    cos_dnu = np.clip(cos_dnu, -1.0, 1.0)
    
    # Calculate the transfer angle (dnu)
    if prograde:
        if np.cross(r1, r2)[2] >= 0:  # Check z-component of cross product
            dnu = np.arccos(cos_dnu)
        else:
            dnu = 2*np.pi - np.arccos(cos_dnu)
    else:  # Retrograde
        if np.cross(r1, r2)[2] >= 0:
            dnu = 2*np.pi - np.arccos(cos_dnu)
        else:
            dnu = np.arccos(cos_dnu)
    
    # Calculate the sine of the transfer angle
    sin_dnu = np.sin(dnu)
    
    # Calculate the chord
    c = np.sqrt(r1_norm**2 + r2_norm**2 - 2 * r1_norm * r2_norm * cos_dnu)
    
    # Calculate the semi-perimeter
    s = (r1_norm + r2_norm + c) / 2
    
    # Determine the minimum energy ellipse semi-major axis
    a_min = s / 2
    
    # Calculate the minimum transfer time (parabolic transfer)
    if abs(dnu - np.pi) < 1e-10:  # Check if dnu is approximately equal to pi
        t_min = np.sqrt(2) / 3 * (s**1.5) / np.sqrt(mu)
    else:
        t_min = np.sqrt(s**3 / (8 * mu)) * (np.pi - dnu + np.sin(dnu))
    
    # Define the transfer time equation for the solver
    def time_equation(alpha):
        if alpha > 0:  # Elliptical transfer
            beta = 2 * np.arcsin(np.sqrt(s / (2 * alpha)))
            if dnu > np.pi:
                beta = -beta
            t = np.sqrt(alpha**3 / mu) * (2 * np.arctan(np.sqrt((1 - np.cos(dnu)) / (1 + np.cos(dnu))) * np.tan(beta/2)) - 
                                          beta + np.sin(beta))
        else:  # Hyperbolic transfer
            alpha = -alpha  # Work with positive alpha
            beta = 2 * np.arcsinh(np.sqrt(s / (2 * alpha)))
            if dnu > np.pi:
                beta = -beta
            t = np.sqrt(alpha**3 / mu) * (np.sinh(beta) - beta)
            alpha = -alpha  # Restore the negative value
        
        return t - tof
    
    # Determine initial bounds for the solver
    if tof <= t_min:
        # For transfers faster than parabolic, solution is hyperbolic
        alpha_min = -10 * a_min
        alpha_max = -a_min
    else:
        # For transfers slower than parabolic, try elliptical first
        alpha_min = a_min
        alpha_max = 100 * a_min
    
    # Try to find a solution with elliptical/hyperbolic guess
    try:
        alpha = optimize.brentq(time_equation, alpha_min, alpha_max, maxiter=max_iterations, xtol=tolerance)
    except ValueError:
        # If no solution in range, try the opposite type (elliptical/hyperbolic)
        if tof <= t_min:
            alpha_min = a_min
            alpha_max = 100 * a_min
        else:
            alpha_min = -10 * a_min
            alpha_max = -a_min
            
        try:
            alpha = optimize.brentq(time_equation, alpha_min, alpha_max, maxiter=max_iterations, xtol=tolerance)
        except ValueError:
            raise ValueError("No convergence in Lambert solver. Try different time of flight or positions.")
    
    # Semi-major axis of the transfer orbit
    a = alpha
    
    # Now calculate the velocities
    if a > 0:  # Elliptical
        beta = 2 * np.arcsin(np.sqrt(s / (2 * a)))
        if dnu > np.pi:
            beta = -beta
        
        alpha_param = np.sqrt(mu / a)
        r1r2_prod = r1_norm * r2_norm
        
        sin_dnu2 = np.sin(dnu/2)
        
        A = np.sqrt(r1r2_prod) * np.cos(dnu/2) / np.cos(beta/2)
        B = np.sqrt(r1r2_prod) * sin_dnu2 / np.sin(beta/2)
    else:  # Hyperbolic
        a_abs = abs(a)
        beta = 2 * np.arcsinh(np.sqrt(s / (2 * a_abs)))
        if dnu > np.pi:
            beta = -beta
        
        alpha_param = np.sqrt(mu / a_abs)
        r1r2_prod = r1_norm * r2_norm
        
        sin_dnu2 = np.sin(dnu/2)
        
        A = np.sqrt(r1r2_prod) * np.cos(dnu/2) / np.cosh(beta/2)
        B = np.sqrt(r1r2_prod) * sin_dnu2 / np.sinh(beta/2)
    
    # Calculate the velocity components
    v1 = (B*r2 - A*r1) / (r1_norm * r2_norm * sin_dnu)
    v2 = (B*r1 - A*r2) / (r1_norm * r2_norm * sin_dnu)
    
    # Scale by the parameter
    v1 = alpha_param * v1
    v2 = alpha_param * v2
    
    # For retrograde orbits, flip the velocity components
    if not prograde:
        v1 = v1 * np.array([1, -1, 1])  # Flip y component
        v2 = v2 * np.array([1, -1, 1])  # Flip y component
    
    return v1, v2, a