# BIOS694_Project
This is the code I used to replicate some of the results from "Semi-Implicit Variational Inference" by Yin and Zhou
The goal was to replicate how SIVI can approximate a Gaussian mixture and to replicate their analysis of the negative binomial data

SRC contains all of the code file that I used. It contains two python files (SIVI_Mixture, SIVI_Neg_Bin) and one R file (Neg_Bin_MCMC). It also contains a README file containing the version of all libraries and packages used in those codes.

Results contains the CSV files with the MCMC sample generated from the Neg_Bin_MCMC file. All figures of importance are also stored there. The data for the negative binomial example is not stored anywhere as it simply copied in the code itself.
