# Run 1 
jax.config.update("jax_debug_nans", False)

PHYSICAL_PARAMS = {
    "number_density": 1.0e4,  # cm^-3
    "temperature": 250.0,  # K
    "cr_rate": 1.0,  # s^-1
    "fuv_field": 1.0,  # Habing units
    "visual_extinction": 2.9643750143703076,  # mag (used if not self-consistent)
    # Self-consistent Av calculation (optional)
    "use_self_consistent_av": False,  # Enable self-consistent Av
    "base_av": 2.0,  # Base Av before column density contribution
    "cloud_radius_pc": 1.0,
    "t_start": 0.0,  # years
    "t_end": 1e2,  # years
    "n_snapshots": 100,  # output timesteps (increased for detail)
    "rtol": 1.0e-5,
    "atol": 1.0e-25,
    "solver": "kvaerno5",  # lowercase required
    "max_steps": 1048576,  # max steps, always use power of 16 (e.g., 4096, 65536)
}


Compiling JAX network...
  Network compiled successfully

Solving ODE system with kvaerno5...
  Time range: 0.00e+00 - 1.00e+02 years
  Snapshots: 100
  Compiling solver (first call)...
  Integration complete in 554.60 seconds
  Steps: 8603 (accepted: 5379, rejected: 3224)

Saving results...
Saved abundances to: results/carbox/orich_cse_abundances.csv
Saved metadata to: results/carbox/orich_cse_metadata.json
Saved summary to: results/carbox/orich_cse_summary.txt

============================================================
Simulation complete! Total time: 622.56 seconds
Output saved to: results/carbox/
============================================================

(same result as with jax.config.update("jax_debug_nans", True))

# Run 2
Changed to fractional abundances... 

