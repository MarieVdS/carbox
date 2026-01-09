import jax.numpy as jnp

def get_cse_physics(t, config):
    """
    Calculate n, T, and Av for a CSE outflow at time t.
    t is in seconds.
    """
    # Convert time to years for easier power-law scaling if desired
    t_yr = t / (365.25 * 24 * 3600)
    
    # Example: Spherical expansion n ~ r^-2
    # We use (t + t_offset) to avoid division by zero at t=0
    t_offset = 1.0 
    
    density = config.number_density * ((t_yr + t_offset) / t_offset)**-2.0
    temperature = config.temperature * ((t_yr + t_offset) / t_offset)**-0.5
    
    # You can also scale Av based on the column density
    visual_extinction = config.visual_extinction * ((t_yr + t_offset) / t_offset)**-1.0
    
    return density, temperature, visual_extinction