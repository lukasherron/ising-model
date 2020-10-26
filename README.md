# ising-model

Testing the limits of the canonical ensemble

# Initializing Data Directories
* Data directories must be initalized to store the simulated data, the bayesian arrays, and the results of the analysis. The directory structure is as follows (nested bullet points indicate parent/child directories)
    * __Simulated data__: The simulated data requires a directory to store data in (any name is acceptable). Once the data is generated it must be sorted (by hand) by the site size and the temperature. The directories should be nested with file structure and naming convention as follows:
        * __sample_size=03__
            * __temp=01.8_subsites=03__
            * __temp=01.9_subsites=03__
            * __...__
        * __sample_size=04__
            * __temp=01.8_subsites=04__
            * __temp=01.9_subsites=04__
            * __...__
        * __...__
    
    * __Bayesian Arrays__: The bayesian arrays should be stored in directories named as follows, where __temp_arrays__ and __final_arrays__ are the directories that the bayesian arrays will be written to upon running bayesian_arrays.py.
        * __bayesian_arrays__
            * __temp_arrays__
            * __final_arrays__
    * __Data Analysis__: The data analysis directories _must_ be structured and named as follows, where the bayesian arrays generated above must be sorted by sample size into __bayesian_arrays__.
        * __data_analysis__
            * __bayesian_arrays__
                * __sample_size=03__
                * __sample_size=04__
                * __...__
            * __dists__
                * __boundary_energy_dists__
                    * __sample_size=03__
                    * __sample_size=04__
                    * __...__
                * __joint_dists__
                    * __sample_size=03__
                    * __sample_size=04__
                    * __...__
                * __system_energy_dists__
                    * __sample_size=03__
                    * __sample_size=04__
                    * __...__
            * __plots_and_figs__
                * __sample_size=03__
                * __sample_size=04__
                * __...__
                        
# Script Sequence
The scripts should be executed in the following order:
1. __ising_args.py__: The script which executes the simulation procdeure. Is dependent on __ising_canonical_simulator_defs.py__, __ising_microcanonical_defs.py__, and __csv_funcs.py__.

2. __generate_bayesian_array.py__: Finds the bayesian arrays p(E | beta) over a range of beta values.

3. __HPG_analysis.py__: Analyzes simulated data, saving results in the file structure outlined above
            
  
   
