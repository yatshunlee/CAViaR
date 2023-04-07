# Author: Lee Yat Shun, Jasper
# Copyright (c) 2023 Lee Yat Shun, Jasper. All rights reserved.

import numpy as np
from scipy.optimize import minimize


def rq_fit(returns, model, quantile, caviar, obj, tol, VaR0):
    """
    following Engle & Manganelli (2004) approach
    :param: returns (np.array): a series of returns
    :param: model (str): a type of CAViaR models
    :param: caviar (callable): a CAViaR function
    :param: obj (callable): RQ criterion
    :param: quantile (float): a value between 0 and 1
    :param: tol (float): a very small positive number. Default is 1e-10
    :param: VaR0 (float): initial estimate of VaR_0
    :returns: estimated beta
    """
    # compute the daily returns as 100 times the difference of the log of the prices.
    returns = np.array(returns)
    
    initial_betas = initialize_betas(returns, model, caviar, obj, quantile, VaR0)
    result = []
    
    # print('Optimizing by simplex method and quasi-newton method...')
    print('Optimizing...')
    
    for m, initial_beta in enumerate(initial_betas):
        print(f'when m = {m+1}')
        beta, loss = optimize(initial_beta, returns, model, quantile, obj, caviar, tol, VaR0)
        result.append(
            {
                'beta': beta,
                'loss': loss
            }
        )

    result = sorted(result, key=lambda x: x['loss'])
    return result[0]['beta']


def initialize_betas(returns, model, caviar, obj, quantile, VaR0):
    """
    :param: returns (np.array): a series of returns
    :param: model (str): a type of CAViaR models
    :param: caviar (callable): a CAViaR function
    :param: obj (callable): RQ criterion
    :param: quantile (float): a value between 0 and 1
    :returns: m betas that produced the lowest RQ criterion as initial values
              for the optimization routine
    """

    """
    For below parameters n, m, p:
    n (int): We generated n vectors using a uniform random number generator between 0 and 1.
    m (int): We computed the regression quantile (RQ) function for each of these vectors and
             selected the m vectors that produced the lowest RQ criterion as initial values
             for the optimization routine
    p (int): Number of betas in the vector
    """
    if model == 'adaptive':
        n = 10 ** 4
        m = 5
        p = 1
    elif model == 'symmetric':
        n = 10 ** 4
        m = 10
        p = 3
    elif model == 'asymmetric':
        n = 10 ** 5
        m = 15
        p = 4
    else:  # IGARCH
        n = 10 ** 4
        m = 10
        p = 3
    
    # for faster version
    n = 1000
    m = 5
    
    print(f'Generating {m} best initial betas out of {n}...')
    random_betas = np.random.uniform(0, 1, (n, p))

    best_initial_betas = []

    # if have time, can try heap instead of sorted list
    for i in range(n):
        loss = obj(random_betas[i], returns, quantile, caviar, VaR0)

        best_initial_betas.append(
            {'loss': loss, 'beta': random_betas[i]}
        )

        best_initial_betas = sorted(best_initial_betas, key=lambda x: x['loss'])

        if len(best_initial_betas) == m+1:
            best_initial_betas.pop()

    return best_initial_betas


def optimize(initial_beta, returns, model, quantile, obj, caviar, tol, VaR0):
    """
    
    """
    current_beta = initial_beta['beta']
    current_loss = initial_beta['loss']
    
    if model == 'igarch':
        bounds = [(1e-10, None)] + [(1e-10, 1) for _ in range(len(current_beta)-1)]
    else:
        bounds = [(None, None)] + [(-1, 1) for _ in range(len(current_beta)-1)]
        
    bounds = [(None, None) for _ in range(len(current_beta))] # try with no bound
    
    count = 0
    print(f'Update {count}:', current_loss)
    
    while True:
        # Minimize the function directly using the L-BFGS-B algorithm
        res = minimize(obj, current_beta, args=(returns, quantile, caviar, VaR0), bounds=bounds, method='L-BFGS-B')
        current_beta = res.x

        loss = res.fun
        
        count += 1   
        print(f'Update {count}:',loss)
        
        if current_loss - loss < tol or count >= 5:
            break
        
        current_loss = loss
            
#         # Minimize the function using the Nelder-Mead algorithm
#         res = minimize(obj, current_beta, args=(returns, quantile, caviar), method='Nelder-Mead', bounds=bounds)
#         current_beta = res.x

#         loss = res.fun
        
#         count += 1   
#         print(f'Update {count}:', loss)
        
#         if current_loss - loss < tol:
#             break
#         else:
#             current_loss = loss

#         # Minimize the function using the BFGS algorithm
#         res = minimize(obj, current_beta, args=(returns, quantile, caviar), method='BFGS', bounds=bounds)
#         current_beta = res.x

#         loss = res.fun
        
#         count += 1   
#         print(f'Update {count}:',loss)
        
#         if current_loss - loss < tol:
#             break
#         else:
#             current_loss = loss
        
    return current_beta, loss