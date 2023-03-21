function output = DQtest(OUT, MODEL, T, y, THETA, VaR, Hit, D, gradient)

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
% Compute both the in sample and out of sample DQ test.
% For the in sample DQ test, use the formulae of theorem 4:
% Hit'X * inv(MM') * X'Hit / THETA*(1-THETA), where M = X' - inv(T)*(X'*H * grad(f)) * inv(D) * grad(f)'
%
% If OUT=1, the output is the p-value of the in sample DQ test.
% If OUT=2, the output is the p-value of the out of sample DQ test.
%
%*****************************************************************************************

% Compute the quantile residuals.
residuals = y + VaR;


% Set up the bandwidth for the k-Nearest neighbor estimator.
SortedRes = sort(abs(residuals));
if THETA == 0.01;
    k = 40;
elseif THETA == 0.05
    k = 60;
else error('This program considers only two confidence levels, 1% and 5%')
end

BANDWIDTH = SortedRes(k);

LAGS = 4;				% Number of lagged hits to be included in the test.

%
%********************************************************************************************
% Compute the regressor matrix X.
%
Constant     = ones(T-LAGS, 1);
HIT          = Hit(LAGS+1 : T);
VaRforecast  = VaR(LAGS+1 : T);
ylag         = y(LAGS : T-1);
%VaRforecastlag = VaR(LAGS : T-1);

Z = zeros(T-LAGS, LAGS);
for s = 1:LAGS
   Z(: ,s)              = Hit(s : T - (LAGS + 1 - s));
   %VaRforecastlag(: ,s) = VaR(s : T - (LAGS + 1 - s));            
   %ylag(: ,s)           = y  (s : T - (LAGS + 1 - s));
   %ylagsquare(: ,s)     = y(s : T - (LAGS + 1 - s)).^(2*(LAGS-s+1));
end

Xout = [Constant, VaRforecast, Z]; % Instruments for the out of sample test.
Xin  = [Z]; % Instruments for the in sample test.

XHNABLA = zeros(size(Xin,2),size(gradient,2)); %, size(estimatedParameters,1));
NABLA   = gradient(LAGS+1:T,:); % Exclude the first LAGS rows from the gradient (for comformability reasons).

%
%********************************************************************************************
% Estimate the matrices that enter the In Sample DQ test.
if OUT == 1
    for i = 2:size(Xin,1)
        if abs(residuals(i)) <= BANDWIDTH
            XHNABLA = XHNABLA + (Xin(i,:))'*gradient(i,:);
        end
    end
end

XHNABLA = XHNABLA/(2*BANDWIDTH*T);

M = Xin' - XHNABLA*inv(D)*NABLA';

%
%********************************************************************************************
% Compute the DQ tests.
DQstatIn  = (HIT'*Xin * inv(M*M') * Xin'*HIT) / (THETA*(1-THETA));
DQstatOut = (HIT'*Xout * inv(Xout'*Xout) * Xout'*HIT) / (THETA*(1-THETA));

DQin  = 1 - chi2cdf(DQstatIn, size(Xin, 2)); % Compute the P-value of the in sample DQ test.
DQout = 1 - chi2cdf(DQstatOut, size(Xout, 2)); % Compute the P-value of the out of sample DQ test.

%
%**********************************************
% Select the output of the program.
if OUT == 1
    output = DQin;
elseif OUT == 2
    output = DQout;
else error('Wrong output selected. Choose OUT = 1 for in sample DQ test, or OUT = 2 for out of sample DQ test.')
end