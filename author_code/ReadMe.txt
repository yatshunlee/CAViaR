This zip file contains the data and the codes used in the paper "Caviar: Conditional Autoregressive Value at Risk by Regression Quantile". 

The file "dataCAViaR.txt" is a (3392,3) matrix containing daily log returns for General Motors, IBM and S&P500.

The parent code is CAViaR.m. To run it, you must first change the saving and loading paths in lines 50, 51 and 72 of CAViaR.m and line 50 of CAViaRoptimisation.m. Then simply type CAViaR in the command window. The final output will be figures 1 and 2 and table 1 of the paper. The computation time on a Compaq Evo W6000, with 256MB RAM and 1.7GHZ processor, was about 37 minutes.