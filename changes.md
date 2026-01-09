# Added caching of the binary executable
At the top of benchmarks/run_cse.py
    Needs to be called right when jax is imported for the first time
Makes a .jax_cache 
Stores the chemical network graph (directed acyclic graph), the solver logic, and jacobian matrix instructions
Recompile is triggered by changing the reaction network, ODE math, solver type, changing jax_debug_nans
Should speed up "jnetwork = network.get_ode()" in carbox/main.py


# Changed saving the output
In carbox/output.py
Pandas gets "fragmented" after the first 100 columns, which slows down the saving process.
File saving sped up by changing how the DataFrame is built 

# Fractional abundances
ODEs need to be rewritten

## initialize_abundances.py
In def initialize_abundances()
Don't multiply by number density to keep fractional abundances

## solver.py
In solve_network, pass density along (in ode_term)

## config.py
in get_physical_params_jax, added density

## output.py
Remove division by number density, abundances already fractional

## network.py
In. JNetwork.__call__

After rates = self.get_rates()
    # This gives you (x_A) for 1-body and (x_A * x_B) for 2-body
    abun_product = self.multiply_rates_by_abundance(rates, abundances)

    # Identify which reactions are 2-body
    # In your get_reactant_multipliers, filler_value indicates an empty slot.
    # If the second column is NOT the filler_value, it's a 2-body reaction.
    filler_value = self.incidence.shape[0] + 1
    is_two_body = (self.reactant_multipliers[:, 1] != filler_value)
    
    # Apply density scaling ONLY to 2-body reactions
    # 1-body: dx/dt = k * x
    # 2-body: dx/dt = k * x_a * x_b * n_tot
    density_scaling = jnp.where(is_two_body, density, 1.0)
    
    final_rates = abun_product * rates * density_scaling

    return self.incidence @ final_rates
