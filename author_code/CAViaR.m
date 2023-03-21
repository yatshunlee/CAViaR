function CAViaR

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
% This is the parent program that calls all the other codes.
% The four models described in the paper [1) Symmetric Absolute Value, 2) Asymmetric Slope, 3) GARCH, 4) Adaptive] are estimated
% for two confidence levels: 5% and 1%.
% 
% The outputs are the tables that appear in the paper. The tables are saved in ASCII format. The whole workspace is saved 
% using the path J:\papers\CAViaR\Results\
%
% ****************** WARNING ******************* 
% Remember to change the saving paths in the lines 50, 51 and 55 of this code, and the loading path in the line 50 of CAViaROptimisation.m
% before running the program.
%
%*****************************************************************************************
tic
THETA = [1, 5];

for t = 1:size(THETA,2) % Loop through the THETA.
   for n = 1:4 % Loop through the models.
       disp(' ')
       y = ['disp(', '''', '*********************************************************', '''', ')'];
       x = ['disp(', '''', '*                                                       *', '''', ')'];
       z = ['disp(', '''', '*            STAGE:    ', 'THETA=', int2str(THETA(t)), '%,   ', 'MODEL=', int2str(n), '              *', '''', ')'];
       eval(y), eval(x), eval(z), eval(x), eval(y)
      
      a = ['[param', int2str(n), '_',int2str(THETA(t)), ', output', int2str(n), '_',int2str(THETA(t)), ', TABLE', int2str(n), '_',int2str(THETA(t)), ...
            		'] = CAViaROptimisation(', int2str(n), ', ' num2str(THETA(t)/100),');'];
      eval(a)
   end
end

% Generate the table in the paper.
b = ['TABLE', int2str(THETA(1)), '= [TABLE', int2str(1), '_',int2str(THETA(1)), ', TABLE', int2str(2), '_',int2str(THETA(1)), ...
        ', TABLE', int2str(3), '_',int2str(THETA(1)), ', TABLE', int2str(4), '_',int2str(THETA(1)), ']'];
c = ['TABLE', int2str(THETA(2)), '= [TABLE', int2str(1), '_',int2str(THETA(2)), ', TABLE', int2str(2), '_',int2str(THETA(2)), ...
        ', TABLE', int2str(3), '_',int2str(THETA(2)), ', TABLE', int2str(4), '_',int2str(THETA(2)), ']'];
eval(b), eval(c)

% Save the tables in ASCII format.
d = ['save J:\papersPublished\CAViaR\Results\TABLE', int2str(THETA(1)), ' TABLE', int2str(THETA(1)), ' -ascii -double'];
e = ['save J:\papersPublished\CAViaR\Results\TABLE', int2str(THETA(2)), ' TABLE', int2str(THETA(2)), ' -ascii -double'];
eval(d), eval(e)

% Generate the plots in the paper.
TIME=[1986+65/250:1/250:1986+65/250+3391/250]';
subplot(2,2,1); plot(TIME, output1_5.VaR(:,1)), axis manual, axis([TIME(1), TIME(end),1,10]), title('5% GM VaR - SAV')
subplot(2,2,2); plot(TIME, output2_5.VaR(:,1)), axis manual, axis([TIME(1), TIME(end),1,10]), title('5% GM VaR - AS')
subplot(2,2,3); plot(TIME, output3_5.VaR(:,1)), axis manual, axis([TIME(1), TIME(end),1,10]), title('5% GM VaR - GARCH')
subplot(2,2,4); plot(TIME, output4_5.VaR(:,1)), axis manual, axis([TIME(1), TIME(end),1,10]), title('5% GM VaR - Adaptive')

figure
y = [-10 : .05 : 10]';
ImpactCurveSAV = NewsImpactCurve(output1_1.BETA(:,3), 1);
subplot(2,2,1); plot(y, ImpactCurveSAV), axis manual, axis([-10,10,1,11]), title('1% S&P500 Impact Curve - SAV')
ImpactCurveAS = NewsImpactCurve(output2_1.BETA(:,3), 2);
subplot(2,2,2); plot(y, ImpactCurveAS), axis manual, axis([-10,10,1,11]), title('1% S&P500 Impact Curve - AS')
ImpactCurveGARCH = NewsImpactCurve(output3_1.BETA(:,3), 3);
subplot(2,2,3); plot(y, ImpactCurveGARCH), axis manual, axis([-10,10,1,11]), title('1% S&P500 Impact Curve - GARCH')
ImpactCurveADAPTIVE = NewsImpactCurve(output4_1.BETA(:,3), 4);
subplot(2,2,4); plot(y, ImpactCurveADAPTIVE), axis manual, axis([-10,10,1,11]), title('1% S&P500 Impact Curve - Adaptive')

% Save the workspace.
save J:\papersPublished\CAViaR\Results\allmodels01_07_50
toc