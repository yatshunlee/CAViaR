#include "mex.h"
#include "math.h"

/*****************************************************************************************************************************************
  *                                                                                                                                      *
  * Codes for the paper "CAViaR: Conditional Autoregressive Value at Risk by Regression Quantile" by Robert Engle and Simone Manganelli  *
  *                                                                                                                                      *
  * By SIMONE MANGANELLI, European Central Bank, Frankfurt am Main.                                                                      *
  * Created on 15 September, 2000. Last modified 20 February 2002.                                                                       *
  *                                                                                                                                      *
  ****************************************************************************************************************************************/

/* C loop to compute VaR for the Symmetric Absolute Value (SAV) model.
 * The program takes as inputs the vector of returns (y), the parameters THETA, BETA and the empirical quantile
 * and produces as output a vector of VaR.
 */

void ComputeVaR(double THETA, double BETA[], double y[], double empiricalQuantile, double VaR[], int RowsOfBETA, int RowsOfy)
{
	int i;
	
	/* Initialize output variables */
	VaR[0] = empiricalQuantile;

	/* Start the loop */
	for(i = 1; i < RowsOfy; i++)
		{
         // Symmetric Absolute Value
         VaR[i] = BETA[0] + BETA[1] * VaR[i-1] + BETA[2] * (y[i-1]*(y[i-1]>0) - y[i-1]*(y[i-1]<0));

         }
}


/***********************************************/

/* The gateway function */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *y, *VaR, *BETA;
	double THETA, empiricalQuantile;
	int RowsOfBETA, RowsOfy;

	/* Check for proper number of arguments */
	if (nrhs != 4)
		mexErrMsgTxt("Four inputs required.");
	if (nlhs!=1)
		mexErrMsgTxt("One output required.");

	/* Check appropriate form of other inputs
		.......
	 */

	/* Get the scalar input THETA and empiricalQuantile. */
	THETA             = mxGetScalar(prhs[0]);
	empiricalQuantile = mxGetScalar(prhs[3]);
	
	/* Create a pointer to the input vectors BETA and y */
	BETA = mxGetPr(prhs[1]);
	y = mxGetPr(prhs[2]);

	/* Get the dimension of BETA and y */
	RowsOfBETA = mxGetM(prhs[1]);
	RowsOfy = mxGetM(prhs[2]);

	/* Set the output pointer */
	plhs[0] = mxCreateDoubleMatrix(RowsOfy, 1, mxREAL);

	/* Create a C pointer to a copy of the output vectors */
	VaR = mxGetPr(plhs[0]);

	/* Call the C subroutine */
	ComputeVaR(THETA, BETA, y, empiricalQuantile, VaR, RowsOfBETA, RowsOfy);
}
