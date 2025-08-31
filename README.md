# Entropy in compartmental systems

Shared repository for the manuscript "Information content and maximum entropy of compartmental systems in equilibrium".

## Abstract
Mass-balanced compartmental systems defy classical deterministic entropy measures, since both metric and topological entropy vanish in dissipative dynamics.
By interpreting open compartmental systems as absorbing continuous-time Markov chains that describe the random journey of a single representative particle, we allow established information-theoretic principles to be applied to this particular type of deterministic dynamical systems.
In particular, path entropy quantifies the uncertainty of complete trajectories, while entropy rates measure the average uncertainty of instantaneous transitions. Using Shannon’s information entropy, we derive closed-form expressions for these quantities in equilibrium and extend the maximum entropy principle (MaxEnt) to the problem of model selection in compartmental dynamics.
This information-theoretic framework not only provides a systematic way to address equifinality but also reveals hidden structural properties of complex systems such as the global carbon cycle.

## Code
In order to run the IPython notebooks and the Python code (to reproduce the figures), please install [bgc_md2](https://github.com/MPIBGC-TEE/bgc_md2) **and all its subpackages** by following its provided installation steps. For example, on Linux in a conda environment, call

	git clone --recurse-submodules https://github.com/MPIBGC-TEE/bgc_md2.git
	cd bgc_md2
	./install_developer_conda.sh
	
