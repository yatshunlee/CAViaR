function [VCmatrix, D, gradient] = VarianceCovariance(BETA, MODEL, T, y, THETA, VaR)

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
% Compute the variance-covariance matrix of the estimated parameters using the formulae of theorems 2 and 3.
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
t=0;


% Initialize matrices.
derivative1 = zeros(T,1);
derivative2 = zeros(T,1);
derivative3 = zeros(T,1);
derivative4 = zeros(T,1);

D = zeros(size(BETA,1));
A = D;

%
%********************************************************************************************
% Model 1: Symmetric Absolute Value.
%
if MODEL == 1
    gradient = zeros(T, 3);

   for i = 2:T
       
       % VaR(i) = BETA(1) + BETA(2) * VaR(i-1) + BETA(3) * abs(y(i-1));
       derivative1(i) = 1 + BETA(2) * derivative1(i-1);
       derivative2(i) = VaR(i-1) + BETA(2) * derivative2(i-1);
       derivative3(i) = BETA(2) * derivative3(i-1) + abs(y(i-1));
       
       gradient(i,:) = [derivative1(i), derivative2(i), derivative3(i)];   
       
       A = A + gradient(i,:)'*gradient(i,:);
       
       if abs(residuals(i)) <= BANDWIDTH
           t=t+1;
           D = D + gradient(i,:)'*gradient(i,:);
       end
   end

%
%********************************************************************************************
% Model 2: Asymmetric Slope.
%
elseif MODEL == 2
    gradient = zeros(T, 4);
    
    for i = 2:T

        % VaR(i) = BETA(1) + BETA(2) * VaR(i-1) + BETA(3) * y(i-1) * (y(i-1)>0) - BETA(4) * y(i-1) * (y(i-1)<0);
        derivative1(i) = 1 + BETA(2)*derivative1(i-1);
        derivative2(i) = VaR(i-1) + BETA(2)*derivative2(i-1);
        derivative3(i) = BETA(2)*derivative3(i-1) + y(i-1)*(y(i-1)>0);
        derivative4(i) = BETA(2)*derivative4(i-1) - y(i-1)*(y(i-1)<0);
        
        gradient(i,:) = [derivative1(i), derivative2(i), derivative3(i), derivative4(i)];   
        
        A = A + gradient(i,:)'*gradient(i,:);
        
        if abs(residuals(i)) <= BANDWIDTH
            t=t+1;
            D = D + gradient(i,:)'*gradient(i,:);
        end
    end

%
%********************************************************************************************
% Model 3: GARCH.
%
elseif MODEL == 3
    gradient = zeros(T, 3);
    
    for i = 2:T
        
        % Indirect GARCH(1,1) model.
        % VaR(i) = sqrt(BETA(1) + BETA(2)*VaR(i-1)^2 + BETA(3)*y(i-1)^2);
        derivative1(i) = (1 + 2*BETA(2)*VaR(i-1)*derivative1(i-1)) / (2*VaR(i));
        derivative2(i) = (VaR(i-1)^2 + 2*BETA(2)*VaR(i-1)*derivative2(i-1)) / (2*VaR(i));
        derivative3(i) = (2*BETA(2)*VaR(i-1)*derivative3(i) + y(i-1)^2) / (2*VaR(i));
        
        gradient(i,:) = [derivative1(i), derivative2(i), derivative3(i)];   
        
        A = A + gradient(i,:)'*gradient(i,:);
        
        if abs(residuals(i)) <= BANDWIDTH
            t=t+1;
            D = D + gradient(i,:)'*gradient(i,:);
        end
    end

%
%********************************************************************************************
% Model 4: Adaptive.
%
elseif MODEL == 4
    
    K = 10; % Set the parameter K that multiplies y(t-1)-f(t-1) in the adaptive model. If K is large, the last term of the adaptive model
           % converges to the Hit function, i.e. the indicator function minus THETA. In this case the optimisation algorithm for the adaptive model
           % has to be changed, because the objective function becomes discontinuous. A possible alternative is to use the Genetic Algorithm.
           % ************* WARNING *********************
           % If you change the value of K, remember to change also the value in the file RQobjectiveFunction.m, line 66.
           
    gradient = zeros(T, 1);
      
    for i = 2:T
        
        % VaR(i) = VaR(i-1) + BETA * (1/(1+indicator(i-1)) - THETA).
        indicator = exp(K*(y(i-1)+ VaR(i-1)));
        if indicator == Inf
            derivative1(i) = derivative1(i-1);
        else derivative1(i) = derivative1(i-1) + (1/(1 + indicator) - THETA) - BETA * (1 + indicator)^(-2) * indicator * K * derivative1(i-1);
        end
        
        gradient(i,1) = derivative1(i);   
        
        A = A + gradient(i,1)'*gradient(i,1);
        
        if abs(residuals(i)) <= BANDWIDTH
            t=t+1;
            D = D + gradient(i,1)'*gradient(i,1);
        end
    end

end


tStdErr=t;   % Check the k-NN bandwidth.
A = A/T;
D = D/(2*BANDWIDTH*T);

VCmatrix = THETA * (1-THETA) * inv(D) * A * inv(D) / T;