function ImpactCurve = NewsImpactCurve(BETA, MODEL)

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
% Compute the CAViaR news impact curve for given parameter estimate BETA.
%
%*****************************************************************************************


lagVaR = 1.645; % Set an arbitrary lagged VaR.
y = [-10 : .05 : 10]'; % Range for lagged returns.

if MODEL == 1
    ImpactCurve = BETA(1) + BETA(2) * lagVaR + BETA(3) * abs(y);
    
elseif MODEL == 2
    ImpactCurve = BETA(1) + BETA(2) * lagVaR + BETA(3) * abs(y) .* (y>0) + BETA(4) * abs(y) .* (y<0);
    
elseif MODEL == 3
    ImpactCurve = sqrt(BETA(1) + BETA(2) * lagVaR^2 + BETA(3) * y.^2);
    
elseif MODEL == 4
    indicator = exp(10*(y + lagVaR));
    THETA = 0.05;
    ImpactCurve = lagVaR + BETA * (1./(1+indicator) - THETA);
end