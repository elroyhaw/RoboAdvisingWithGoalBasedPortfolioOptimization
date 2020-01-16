import numpy as np
import statsmodels.api as sm
import itertools


def iteratively_reweighted_least_squares(tau, Y, X_c, x_c, X_d, x_d, h, g):
    """
    Iteratively reweighted least squares algorithm in Section 3.3.1

    :param tau: Probability value
    :param Y: Response vector
    :param X_c: Continuous covariates matrix
    :param x_c: Given continuous vector
    :param X_d: Discrete covariates matrix
    :param x_d: Given discrete vector
    :param h: Bandwith vector for continuous covariates
    :param g: Bandwith vector for discrete covariates
    :type tau: float
    :type Y: np.ndarray
    :type X_c: np.ndarray
    :type x_c: np.ndarray
    :type X_d: np.ndarray
    :type x_d: np.ndarray
    :type h: np.ndarray
    :type g: np.ndarray
    :return: Estimated conditional quantile function
    :rtype: float
    """

    # initial guess
    weights = K(X_c, x_c, h) * L(X_d, x_d, g)
    alpha_prev, beta_prev = weighted_least_square(Y, X_c - x_c, weights)

    # first iteration
    K_tau = new_weights(tau, Y, X_c, x_c, X_d, x_d, h, g, alpha_prev, beta_prev)
    alpha_curr, beta_curr = weighted_least_square(Y, X_c - x_c, K_tau)
    first_mae = abs(alpha_curr - alpha_prev)
    threshold = 0.01 * first_mae

    # subsequent iterations
    while abs(alpha_curr - alpha_prev) >= threshold:
        K_tau = new_weights(tau, Y, X_c, x_c, X_d, x_d, h, g, alpha_curr, beta_curr)
        alpha_prev, beta_prev = alpha_curr, beta_curr
        alpha, beta = weighted_least_square(Y, X_c - x_c, K_tau)
        alpha_curr, beta_curr = alpha, beta
    return alpha_curr


def weighted_least_square(Y, X, weights):
    """
    Weighted least squares

    :param Y: Response vector
    :param X: Covariates matrix
    :param weights: Weights vector
    :type Y: np.ndarray
    :type X: np.ndarray
    :type weights: np.ndarray
    :return: Intercept, Slope
    :rtype: float, float
    """

    X = sm.add_constant(X)
    model = sm.WLS(Y, X, weights=weights)
    results = model.fit()
    params = results.params
    return params[0], params[1:]


def new_weights(tau, Y, X_c, x_c, X_d, x_d, h, g, alpha, beta):
    """
    Helper function to compute new weights as in K_\tau in Section 3.3.1

    :param tau: Probability value
    :param Y: Response vector
    :param X_c: Continuous covariates matrix
    :param x_c: Given continuous vector
    :param X_d: Discrete covariates matrix
    :param x_d: Given discrete vector
    :param h: Bandwith vector for continuous covariates
    :param g: Bandwith vector for discrete covariates
    :param alpha: Intercept from previous iteration
    :param beta: Slope from previous iteration
    :type tau: float
    :type Y: np.ndarray
    :type X_c: np.ndarray
    :type x_c: np.ndarray
    :type h: np.ndarray
    :type alpha: float
    :type beta: float
    :return: New weight vector
    :rtype: np.ndarray
    """

    residuals = Y.flatten() - alpha - np.dot(X_c - x_c, np.transpose(beta))

    def transform(residual):
        if residual > 0:
            return tau / residual
        elif residual < 0:
            return (tau - 1) / residual
        else:
            return 0

    theta = np.array(list(map(transform, residuals))).flatten()
    # element-wise multiply; theta is vector, K is vector, L is vector
    return theta * K(X_c, x_c, h) * L(X_d, x_d, g)


def K(X_c, x_c, h):
    """
    Kernel function for continuous covariates

    :param X_c: Continuous covariates matrix
    :param x_c: Given continuous vector
    :param h: Bandwith vector for continuous covariates
    :type X_c: np.ndarray
    :type x_c: np.ndarray
    :type h: np.ndarray
    :return: Continuous kernel vector
    :rtype: np.ndarray
    """

    def k(X, x, h):
        # univariate gaussian kernel
        return np.exp(-0.5 * ((X - x) / h) ** 2) / np.sqrt(2 * np.pi)

    output = k(X_c[:, 0], x_c[0], h[0])
    for i in range(1, X_c.shape[1]):
        output *= k(X_c[:, i], x_c[i], h[i])
    return output / np.prod(h)


def L(X_d, x_d, g):
    """
    Kernel function for discrete covariates

    :param X_d: Continuous discrete matrix
    :param x_d: Given discrete vector
    :param g: Bandwith vector for discrete covariates
    :type X_d: np.ndarray
    :type x_d: np.ndarray
    :type g: np.ndarray
    :return: Discrete kernel vector
    :rtype: np.ndarray
    """

    def l(X, x, g):
        return np.power(g, X != x)

    output = l(X_d[:, 0], x_d[0], g[0])
    for i in range(1, X_d.shape[1]):
        output *= l(X_d[:, i], x_d[i], g[i])
    return output


def check_function(u, tau):
    """
    Piecewise linear check function

    :param u: Function argument
    :param tau: Probability value
    :type u: float
    :type tau: float
    :return: Function output
    :rtype: float
    """

    return u * (tau - (u < 0))


def cross_validation(tau, Y, X_c, X_d, H, G):
    """
    Cross-validation for bandwidth selection in Section 3.3.2

    :param tau: Probability value
    :param Y: Response vector
    :param X_c: Continuous covariates matrix
    :param X_d: Discrete covariates matrix
    :param H: Bandwidth values for continuous covariates
    :param G: Bandwidth values for discrete covariates
    :type tau: float
    :type Y: np.ndarray
    :type X_c: np.ndarray
    :type X_d: np.ndarray
    :type H: dict of str: list
    :type G: dict of str: list
    :return: Optimal bandwidth values for continuous covariates, Optimal bandwidth values for discrete covariates
    :rtype: tuple, tuple
    """

    # H = {h_1: [...], ..., h_p: [...]}, G = {g_1: [...], ..., g_q: [...]}
    hs = list(itertools.product(*H.values()))
    gs = list(itertools.product(*G.values()))
    CV_err_h_g = {}
    for h_idx, h in enumerate(hs):
        for g_idx, g in enumerate(gs):
            CV_err = []
            for i in range(Y.shape[0]):
                # leave out i-th obs as validation
                Y_val = Y[i]
                X_c_val = X_c[i]
                X_d_val = X_d[i]
                # remove i-th obs from training set
                Y_tr = np.delete(Y, i)
                X_c_tr = np.delete(X_c, i, 0)
                X_d_tr = np.delete(X_d, i, 0)
                Y_val_pred = iteratively_reweighted_least_squares(tau, Y_tr, X_c_tr, X_c_val, X_d_tr, X_d_val, h, g)
                CV_err.append(check_function(Y_val - Y_val_pred, tau))
            print(f"Iteration with (h_idx, g_idx) = {(h_idx, g_idx)} error: {np.mean(CV_err)}")
            CV_err_h_g[(h_idx, g_idx)] = np.mean(CV_err)
    min_h_idx, min_g_idx = min(CV_err_h_g, key=CV_err_h_g.get)
    return hs[min_h_idx], gs[min_g_idx]
