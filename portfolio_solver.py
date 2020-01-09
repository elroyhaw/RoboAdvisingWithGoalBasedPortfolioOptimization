import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def compute_portfolio_weights(gamma, mu, sigma):
    ones = np.ones(mu.shape[0])
    sigma_inv = np.linalg.inv(sigma)
    a = ones.T @ sigma_inv @ ones
    b = ones.T @ sigma_inv @ mu
    return 1 / gamma * sigma_inv @ (mu - ((b - gamma) / a) * ones)


def compute_gamma(H, tau, mu, sigma):
    feasible = feasibility_test(H, tau, mu, sigma)
    if not feasible:
        raise InfeasibleProblemException
    ones = np.ones(mu.shape[0])
    sigma_inv = np.linalg.inv(sigma)
    a = ones.T @ sigma_inv @ ones
    b = ones.T @ sigma_inv @ mu
    c = mu.T @ sigma_inv @ mu
    qnorm = norm.ppf(tau)
    c1 = qnorm ** 2 / a - H ** 2 + 2 * H * b / a - (b / a) ** 2
    c2 = 2 * H * c - 2 * H * b ** 2 / a - 2 * b * c / a + 2 * b ** 3 / a ** 2
    c3 = qnorm ** 2 * (c - b ** 2 / a) - c ** 2 + 2 * c * b ** 2 / a - b ** 4 / a ** 2
    roots = np.roots([c1, c2, c3])
    return roots[0] if roots[0] > 0 else roots[1]


def feasibility_test(H, tau, mu, sigma):
    results = optimize(tau, mu, sigma)
    return H < - results.fun


def optimize(tau, mu, sigma):
    ones = np.ones(mu.shape[0])
    qnorm = norm.ppf(tau)

    def objective(w):
        return - (w.T @ mu + qnorm * (w.T @ sigma @ w) ** 0.5)

    def constraint(w):
        return w.T @ ones - 1

    cons = ({'type': 'eq', 'fun': constraint})
    initial_guess = np.array([0.5, 0.5, 0])

    return minimize(objective, initial_guess, method='SLSQP', constraints=cons)


class InfeasibleProblemException(Exception):
    pass
