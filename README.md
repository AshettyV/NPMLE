## Code for Heteroskedastic Estimation using NPMLE

To run all experiments, please use the Jupyter notebook `NPMLE_py_test.ipynb` and to look at the implementation of the Frank-Wolfe algorithm, look at the file `npmle_tests_parallel.py`. Each experimental case (subset-of-signals, two-point prior, three-point prior, identical variance, quadratic variance) is listed individually to see one output of our method. All experimental simulation results in `NPMLE_py_tests.ipynb` are saved to the folder `~/NPMLE/results`. All experiments were run in parallel on NYU Torch HPC on 128 CPUs. 

### Adjusting current parameters

To adjust current default parameters, go to the "DEFAULT CONFIGURATION" section in `npmle_tests_parallel.py`. Under the global and case-specific experimental parameters in the `get_default_config()` method, change any of the prior parameters. The current configuration in the file match the ones used in the final paper. 
