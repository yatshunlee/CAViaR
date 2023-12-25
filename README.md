# Evaluate CAViaR by Quantile Regression
This is a group project of SDSC6013 Topics in Financial Engineering and Technology. We built a value-at-risk model directly modeling the quantile return directly by referring the paper CAViaR: Conditional Autoregressive Value at Risk by Regression Quantiles by Engle and Manganelli (2004).

## Disclaimer
As I found the original optimization approach is computational costly, I have modified a bit the box constraints as well as the starting approach (the initial guess/start of the estimated parameters). For details, you may want to take a look on the documentation (You may easily change the setting back accordingly in the source code). So, please use this package `caviar` with caution and do not use in production environments.

## Known Issues
- TBC

You are welcome to report bug in https://github.com/yatshunlee/CAViaR-Project/issues. :)

## Quick Summary
We constructed two libraries: `caviar` and `var_tests` to model the value at risk and backtest the VaR estimate. For presentation, we constructed a dashboard to showcase how it can be possibly applied. If you are looking for some inspiration, we strongly suggest to take a look of the documentation below and the code in `notebook-example`.

Libraries:
1. CAViaR Model `caviar`
2. VaR Test `var_tests`

Demo Application:
- [dashboard_caviar](https://youtu.be/1NhIeDSbeXE)

Documentation:
- [Click here](./doc/README.md)

Main Author(s) / Maintainer(s) of `caviar` and `var_tests`:
1. Jasper Lee
