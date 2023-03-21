function [BetaHat, output, TABLE] = CAViaROptimisation(MODEL, THETA)

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
% Main program. 
% Loads the data set and computes all the statistics for MODEL CAViaR and for all the portfolios (GM, IBM and S&P).
% 
%************ INPUTS *********************
%
% THETA = confidence level of the Value at Risk
% MODEL: 1 = Symmetric Absolute Value
%		 2 = Asymmetric Slope
%		 3 = GARCH
%		 4 = Adaptive
%
%************ OUTPUT *********************
%
% BetaHat = Optimal vectors (three vectors, one for each portfolio)
% output  = Contains information on standard errors, DQ tests, number of Hits, initial conditions and more.
% TABLE   = Matrix of data that appear in the paper.
%
%*****************************************************************************************

if MODEL ~= [1, 2, 3, 4]
    disp(' ')
    disp('*******************************************************************')
    disp('ERROR! You need to select one of the following models:')
    disp('MODEL=1: Symmetric Absolute Value')
    disp('MODEL=2: Asymmetric Slope')
    disp('MODEL=3: GARCH')
    disp('MODEL=4: Adaptive')
    error('Wrong MODEL selected.')
end

% *****************************************************************************************
% Load the file with real data and run the optimization routine.
% The file contains daily returns (7 April 1986 - 7 April 1999) for GM, IBM and S&P respectively.
% It loads a (T,3) matrix, where T is the number of observations and 3 is the number of portfolios considered.
%
% ***************************** WARNING *****************************
% Remember to change the path to your own directory before running the program.
load j:\papersPublished\caviar\data\dataCAViaR.txt 
% *****************************************************************************************

%
% Define some variables.
ytot        = dataCAViaR;
%ytot        = dataCAViaR(501:size(dataCAViaR,1),:)*100;   	
inSampleObs = 2892;				    % Number of in sample observations.
totalObs    = size(ytot,1);			% Number of total observations.
%y           = dataCAViaR(1:inSampleObs,:);   % Vector of in sample observations.
y = ytot(1:inSampleObs,:);
nSamples    = size(y,2);						  % Number of portfolio time series.


% *****************************************************************************************
% Set parameters for optimisation.
% *****************************************************************************************
REP			  = 5;                % Number of times the optimization algorithm is repeated.
if (MODEL == 1) | (MODEL == 3)
    nInitialVectors = [10000, 3]; % Number of initial vector fed in the uniform random number generator SAV and GARCH models.
    nInitialCond = 10;             % Select the number of initial conditions for the optimisation.
    
elseif MODEL == 2
    nInitialVectors = [100000, 4]; % Number of initial vector fed in the uniform random number generator for AS model.
    nInitialCond = 15;            % Select the number of initial conditions for the optimisation.
    
elseif MODEL == 4 % See the comment in RQobjectivefunction.m lines 66-70 for this model.
    nInitialVectors = [10000, 1]; % Number of initial vector fed in the uniform random number generator for AS model.
    nInitialCond = 5;            % Select the number of initial conditions for the optimisation.
end
MaxFunEvals = 500; % Parameters for the optimisation algorithm. Increase them in case the algorithm does not converge.
MaxIter     = 500;
options = optimset('LargeScale', 'off', 'HessUpdate', 'dfp', 'LineSearchType', 'quadcubic','MaxFunEvals', MaxFunEvals, ...
                    'display', 'off', 'MaxIter', MaxIter, 'TolFun', 1e-10, 'TolX', 1e-10);
warning off

rand('seed', 50)                  % Set the random number generator seed for reproducability (seed used in the paper = 50).

%
% Define some matrices.
VaR            = zeros(totalObs, nSamples);
Hit            = VaR;
DQinSample     = zeros(1, nSamples);
DQoutOfSample  = DQinSample;


%
% Compute the empirical THETA-quantile for y (the in-sample vector of observations).
% Sorting followed by select the 1% / 5% quantile for nSamples
% empiricalQuantile is a nSamples vector
for t = 1:nSamples
   ysort(:, t)          = sortrows(y(1:300, t), 1); 
   empiricalQuantile(t) = ysort(round(300*THETA), t);
end

%
%
%**************************** Start the loop over the portfolios ******************************************
for t = 1 : nSamples
    disp('__________________')
    disp(' ')
    SMPL = ['disp(', '''', ' Sample number: ', int2str(t), '''', ' )']; eval(SMPL)
    disp('__________________')

%   
%   
%**************************** Optimization Routine ******************************************  
    initialTargetVectors = unifrnd(0, 1, nInitialVectors);
      
    RQfval = zeros(nInitialVectors(1), 1);
    for i = 1:nInitialVectors(1)
        RQfval(i) = RQobjectiveFunction(initialTargetVectors(i,:), 1, MODEL, inSampleObs, y(:,t), THETA, empiricalQuantile(t));
    end
    Results          = [RQfval, initialTargetVectors];
    SortedResults    = sortrows(Results,1);
    
    if (MODEL == 1) | (MODEL == 3)
        BestInitialCond  = SortedResults(1:nInitialCond,2:4); % 1 is the RQfval and 2:4 are the coefficients
    elseif MODEL == 2
        BestInitialCond  = SortedResults(1:nInitialCond,2:5);
    elseif MODEL == 4
        BestInitialCond  = SortedResults(1:nInitialCond,2);
    end
    
    for i = 1:size(BestInitialCond,1)
        [Beta(i,:), fval(i,1), exitflag(i,1)] = fminsearch('RQobjectiveFunction', BestInitialCond(i,:), options, 1, MODEL, inSampleObs, y(:,t), THETA, empiricalQuantile(t));
        for it = 1:REP % REP = 5
            [Beta(i,:), fval(i,1), exitflag(i,1)] = fminunc('RQobjectiveFunction', Beta(i,:), options, 1, MODEL, inSampleObs, y(:,t), THETA, empiricalQuantile(t));
            [Beta(i,:), fval(i,1), exitflag(i,1)] = fminsearch('RQobjectiveFunction', Beta(i,:), options, 1, MODEL, inSampleObs, y(:,t), THETA, empiricalQuantile(t));
            if exitflag(i,1) == 1
                break
            end
        end
    end
    SortedFval  = sortrows([fval, Beta, exitflag, BestInitialCond], 1);
    
    
    if (MODEL == 1) | (MODEL == 3)
        BestFval         = SortedFval(1, 1);
        BetaHat(:, t)    = SortedFval(1, 2:4)';
        ExitFlag         = SortedFval(1, 5);
        InitialCond(:,t) = SortedFval(1, 6:8)';
        
    elseif MODEL == 2
        BestFval         = SortedFval(1, 1);
        BetaHat(:, t)    = SortedFval(1, 2:5)';
        ExitFlag         = SortedFval(1, 6);
        InitialCond(:,t) = SortedFval(1, 7:10)';
        
    elseif MODEL == 4
        BestFval         = SortedFval(1, 1);
        BetaHat(:, t)    = SortedFval(1, 2);
        ExitFlag         = SortedFval(1, 3);
        InitialCond(:,t) = SortedFval(1, 4);
        
    end

%**************************** End of Optimization Routine ******************************************
   
%
%
%************************** Compute variables that enter output *****************************

    % Compute VaR and Hit for the estimated parameters of RQ. (Now using totalObs)
	% VaRHit = [VaR, Hit]
	% Var = f(beta)
	% Hit = (y < -VaR) - THETA
    VaRHit  = RQobjectiveFunction(BetaHat(:,t)', 2, MODEL, totalObs, ytot(:,t), THETA, empiricalQuantile(t));
    VaR(:,t) = VaRHit(:,1);
	Hit(:,t) = VaRHit(:,2);
   
    % Compute the percentage of hits in sample and out-of-sample.
    HitInSample(1,t)    = mean(Hit(1:inSampleObs,t) + THETA) * 100;
    HitOutOfSample(1,t) = mean(Hit((inSampleObs + 1):totalObs, t) + THETA) * 100;
   
    % Compute the variance-covariance matrix of the estimated parameters.
    [varCov, D, gradient] = VarianceCovariance(BetaHat(:,t)', MODEL, inSampleObs, y(:, t), THETA, VaR(1:inSampleObs,t));
    standardErrors(:, t) = sqrt(diag(varCov));
    coeffPvalue(:, t)    = normcdf(-abs(BetaHat(:,t) ./ standardErrors(:,t)));
   
    %
    % Compute the DQ test in and out of sample.
    DQinSample(1,t)    = DQtest(1, MODEL, inSampleObs, y(:,t), THETA, VaR(1:inSampleObs,t), Hit(1:inSampleObs,t), D, gradient);
    DQoutOfSample(1,t) = DQtest(2, MODEL, totalObs - inSampleObs, ytot((inSampleObs + 1):totalObs, t), THETA, VaR((inSampleObs + 1):totalObs, t), Hit((inSampleObs + 1):totalObs, t), D, gradient);
   
    RQ(1,t)       = BestFval;
    EXITFLAG(1,t) = ExitFlag;
   
end			% End of the t loop.
%
%
%**************************** End of the loop over the portfolios ******************************************

%
%
%**************************** Store the outputs in the vector 'output' ******************************************
output.BETA                 = BetaHat;
output.VaR              	= VaR;
output.Hit              	= Hit;
output.RQ					= RQ;
output.ExitFlag             = EXITFLAG;
output.HitInSample      	= HitInSample;
output.HitOutOfSample   	= HitOutOfSample;
output.DQinSample         	= DQinSample;
output.DQoutOfSample      	= DQoutOfSample;
output.stdErr               = standardErrors;
output.coeffPvalue			= coeffPvalue;
output.initialConditions    = InitialCond;

%
%
%**************************** Construct the table of the CAViaR paper ******************************************
if (MODEL == 1) | (MODEL == 3)
TABLE = [BetaHat(1,:); standardErrors(1,:); coeffPvalue(1,:);
         BetaHat(2,:); standardErrors(2,:); coeffPvalue(2,:);
         BetaHat(3,:); standardErrors(3,:); coeffPvalue(3,:);
         zeros(3, 3);
         RQ;
         HitInSample; HitOutOfSample
         DQinSample;DQoutOfSample];
         
elseif (MODEL == 2)
TABLE = [BetaHat(1,:); standardErrors(1,:); coeffPvalue(1,:);
         BetaHat(2,:); standardErrors(2,:); coeffPvalue(2,:);
         BetaHat(3,:); standardErrors(3,:); coeffPvalue(3,:);
         BetaHat(4,:); standardErrors(4,:); coeffPvalue(4,:);
         RQ;
         HitInSample; HitOutOfSample
         DQinSample;DQoutOfSample];

elseif (MODEL == 4)
TABLE = [BetaHat(1,:); standardErrors(1,:); coeffPvalue(1,:);
         zeros(9, 3);
         RQ;
         HitInSample; HitOutOfSample
         DQinSample;DQoutOfSample];
    
end