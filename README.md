# Entropy in compartmental systems

Shared repository for the manuscript "Information content and maximum entropy of compartmental systems in equilibrium".

## Abstract

Although compartmental dynamical systems are used in many different areas of science, model selection based on the maximum entropy principle (MaxEnt) is challenging because of lacking methods for quantifying the entropy for this type of systems. 
Here, we take advantage of the interpretation of compartmental systems as continuous-time Markov chains to obtain entropy measures that quantify model information content. 
In particular we quantify the uncertainty of a single particle's path as it travels through the system as described by path entropy and entropy rates.
Path entropy measures the uncertainty of the entire path of a traveling particle from its entry into the system until its exit, whereas entropy rates measure the average uncertainty of the instantaneous future of a particle while it is in the system.
We derive explicit formulas for these two types of entropy for compartmental systems in equilibrium based on Shannon information entropy and show how they can be used to solve equifinality problems in the process of model selection by means of MaxEnt.

## Code
In order to run the IPython notebooks and the Python code (to reproduce the figures), please install [bgc_md2](https://github.com/MPIBGC-TEE/bgc_md2) **and all its subpackages** by following its provided installation steps. For example, on Linux in a conda environment, call

	git clone --recurse-submodules https://github.com/MPIBGC-TEE/bgc_md2.git
	cd bgc_md2
	./install_developer_conda.sh
	
	
