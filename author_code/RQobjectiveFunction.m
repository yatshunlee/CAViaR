function output = RQobjectiveFunction(BETA, OUT, MODEL, T, y, THETA, empiricalQuantile)

% ****************************************************************************************************************************************
% *                                                                                                                                      *
% * Codes for the paper "CAViaR: Conditional Autoregressive Value at Risk by Regression Quantile" by Robert Engle and Simone Manganelli  *
% *                                                                                                                                      *
% * By SIMONE MANGANELLI, European Central Bank, Frankfurt am Main.                                                                      *
% * Created on 15 September, 2000. Last modified 20 February 2002.                                                                       *
% *                                                                                                                                      *
% ****************************************************************************************************************************************
% 
% 
% RQobjectiveFunction computes the VaR and the RQ criterion for the vector of parameters BETA, for MODEL i (1:SAV, 2:AS, 3:GARCH, 4:ADAPTIVE) 
% given the number of observations T, the time series y and the confidence level THETA.
%
% If OUT=1, the output is the regression quantile objective function.
% If OUT=2, the output is [VaR, Hit].
%
%*****************************************************************************************

% Initial Conditions
VaR = zeros(T,1);
Hit = VaR;

VaR(1) = -empiricalQuantile;
Hit(1) = (y(1) < -VaR(1)) - THETA;

%
%**********************************************
% Compute the VaR
%

%
%********************************************************************************************
% Model 1: Symmetric Absolute Value.
%
if MODEL == 1

   VaR = SAVloop(THETA, BETA, y, VaR(1)); % Call the C program to compute the VaR loop.
   Hit = (y < -VaR) - THETA;

%
%********************************************************************************************
% Model 2: Asymmetric Slope.
%
elseif MODEL == 2
    
   VaR = ASloop(THETA, BETA, y, VaR(1)); % Call the C program to compute the VaR loop.
   Hit = (y < -VaR) - THETA;

%
%********************************************************************************************
% Model 3: GARCH.
%
elseif MODEL == 3
    
   VaR = GARCHloop(THETA, BETA, y, VaR(1)); % Call the C program to compute the VaR loop.
   Hit = (y < -VaR) - THETA;

%
%********************************************************************************************
% Model 4: Adaptive.
%
elseif MODEL == 4
    
   K = 10; % Set the parameter K that multiplies y(t-1)-f(t-1) in the adaptive model. If K is large, the last term of the adaptive model
           % converges to the Hit function, i.e. the indicator function minus THETA. In this case the optimisation algorithm for the adaptive model
           % has to be changed, because the objective function becomes discontinuous. A possible alternative is to use the Genetic Algorithm.
           % ************* WARNING *********************
           % If you change the value of K, remember to change also the value in the file VarianceCovariance.m, line 124.
           
   VaR = ADAPTIVEloop(THETA, K, BETA, y, VaR(1)); % Call the C program to compute the VaR loop.
   Hit = (y < -VaR) - THETA;

end
%
%**********************************************
% Compute the Regression Quantile criterion.
%
RQ  = -Hit'*(y + VaR);

if RQ == Inf | (RQ ~= RQ) | ~isreal(RQ)
   RQ = 1e+100;
end
%
%**********************************************
% Select the output of the program.
if OUT == 1
    output = RQ;
elseif OUT ==2
    output = [VaR, Hit];
else error('Wrong output selected. Choose OUT = 1 for RQ, or OUT = 2 for [VaR, Hit].')
end