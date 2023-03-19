import numpy as np
from scipy.optimize import minimize


def numeric_fit(returns, model, quantile, caviar, obj, tol):
    """
    following Engle & Manganelli (2004) approach
    """
    initial_betas = initialize_betas(returns, model, obj)
    result = []

    for initial_beta in initial_betas:
        beta, loss = optimize(initial_beta, returns, quantile, obj, caviar, tol)
        result.append(
            {
                'beta': beta,
                'loss': loss
            }
        )

    result = sorted(result, key=lambda x: x['loss'])
    return result['beta']


def initialize_betas(returns, model, obj):
    """
    :param: returns (np.array): a series of returns
    :param: model (str): a type of CAViaR models
    :param: obj (callable): RQ criterion
    """

    """
    For below parameters n, m, p:
    n (int): We generated n vectors using a uniform random number generator between 0 and 1.
    m (int): We computed the regression quantile (RQ) function for each of these vectors and
             selected the m vectors that produced the lowest RQ criterion as initial values
             for the optimization routine
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

    random_betas = np.random.uniform(0, 1, (n, p))

    best_initial_betas = []

    # if have time, can try heap instead of sorted list
    for i in range(n):
        loss = obj(random_betas[i], returns, model)

        best_initial_betas.append(
            {'loss': loss, 'beta': random_betas[i]}
        )

        best_initial_betas = sorted(best_initial_betas, key=lambda x: x['loss'])

        if len(best_initial_betas) == m:
            best_initial_betas.pop()

    return best_initial_betas


def optimize(initial_beta, returns, quantile, obj, caviar, tol):
    beta_new = initial_beta['beta']
    current_loss = initial_beta['loss']

    while True:
        # Minimize the function using the Nelder-Mead algorithm
        res = minimize(obj, beta_new, args=(returns, quantile, caviar), method='Nelder-Mead')
        beta_new = res.x

        loss = res.fun

        if current_loss - loss < tol:
            break
        else:
            current_loss = loss

        # Minimize the function using the BFGS algorithm
        res = minimize(obj, beta_new, args=(returns, quantile, caviar), method='BFGS')
        beta_new = res.x

        loss = res.fun

        if current_loss - loss < tol:
            break
        else:
            current_loss = loss

    return beta_new, loss