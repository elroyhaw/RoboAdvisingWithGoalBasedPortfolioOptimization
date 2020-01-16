import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
from nonparametric_conditional_quantile_estimator import iteratively_reweighted_least_squares, cross_validation, \
    check_function
from data_processor import load_china_p2p_data, load_imdb_data, load_assets
from portfolio_solver import compute_gamma, compute_portfolio_weights, InfeasibleProblemException


def estimate_quantiles(tau, h, g, df, response_var, cts_vars, discrete_vars, n):
    estimated_quantiles = {}
    for var in cts_vars:
        estimated_quantiles[var] = [[], []]
    for var in discrete_vars:
        estimated_quantiles[var] = [[], []]

    for i, var in enumerate(cts_vars):
        cts_params = np.mean(df[cts_vars]).to_numpy()
        discrete_params = np.mean(df[discrete_vars]).to_numpy()
        for j in range(1, n):
            val = j * (np.max(df[[var]].to_numpy()) - np.min(df[[var]].to_numpy())) / n + np.min(df[[var]].to_numpy())
            cts_params[i] = val
            estimated_quantile = iteratively_reweighted_least_squares(tau, df[response_var].to_numpy(),
                                                                      df[cts_vars].to_numpy(), cts_params,
                                                                      df[discrete_vars].to_numpy(), discrete_params,
                                                                      np.array(h), np.array(g))
            estimated_quantiles[var][0].append(val)
            estimated_quantiles[var][1].append(estimated_quantile)
    for i, var in enumerate(discrete_vars):
        cts_params = np.mean(df[cts_vars]).to_numpy()
        discrete_params = np.mean(df[discrete_vars]).to_numpy()
        for j in range(1, n):
            val = j * (np.max(df[[var]].to_numpy()) - np.min(df[[var]].to_numpy())) / n + np.min(df[[var]].to_numpy())
            discrete_params[i] = val
            estimated_quantile = iteratively_reweighted_least_squares(tau, df[response_var].to_numpy(),
                                                                      df[cts_vars].to_numpy(), cts_params,
                                                                      df[discrete_vars].to_numpy(), discrete_params,
                                                                      np.array(h), np.array(g))
            estimated_quantiles[var][0].append(val)
            estimated_quantiles[var][1].append(estimated_quantile)

    return estimated_quantiles


def plot_estimated_quantiles(estimated_quantiles, cts_vars, discrete_vars, cts_var_names, discrete_var_names):
    fig, axs = plt.subplots(ncols=len(cts_vars), figsize=(18, 5))
    for i, var in enumerate(cts_vars):
        x, y = estimated_quantiles[var]
        axs[i].plot(x, y)
        axs[i].set_xlabel(cts_var_names[i])
        axs[i].set_ylabel("Estimated Conditional Quantiles")
    plt.savefig(f"results/Estimated_Conditional_Quantile_Cts_{str(datetime.now())}.jpg")
    fig, axs = plt.subplots(ncols=len(discrete_vars), figsize=(18, 5))
    for i, var in enumerate(discrete_vars):
        x, y = estimated_quantiles[var]
        axs[i].plot(x, y)
        axs[i].set_xlabel(discrete_var_names[i])
        axs[i].set_ylabel("Estimated Conditional Quantiles")
    plt.savefig(f"results/Estimated_Conditional_Quantile_Discrete_{str(datetime.now())}.jpg")


def evaluation_1():
    df = load_china_p2p_data()
    N = 100
    df = df.sample(N)
    n = 5
    H = {
        "funded_amnt": [i * (max(df.funded_amnt) - min(df.funded_amnt)) / n for i in range(1, n)],
        "annual_inc": [i * (max(df.annual_inc) - min(df.annual_inc)) / n for i in range(1, n)],
        "dti": [i * (max(df.dti) - min(df.dti)) / n for i in range(1, n)]
    }
    G = {
        "emp_length": [i * (max(df.emp_length) - min(df.emp_length)) / n for i in range(1, n)],
        "purpose": [i * (max(df.purpose) - min(df.purpose)) / n for i in range(1, n)]
    }
    tau = 0.05
    Y = df[['int_rate']].to_numpy()
    X_c = df[['funded_amnt', 'annual_inc', 'dti']].to_numpy()
    X_d = df[['emp_length', 'purpose']].to_numpy()
    h, g = cross_validation(tau, Y, X_c, X_d, H, G)
    estimated_quantiles = \
        estimate_quantiles(tau, h, g,
                           df[['int_rate', 'funded_amnt', 'annual_inc', 'dti', 'emp_length', 'purpose']],
                           ['int_rate'], ['funded_amnt', 'annual_inc', 'dti'], ['emp_length', 'purpose'], 100)

    js = json.dumps(estimated_quantiles)
    with open(f"results/evaluation_1_N_{N}_{str(datetime.now())}.json", "w") as f:
        f.write(js)

    plot_estimated_quantiles(estimated_quantiles, ['funded_amnt', 'annual_inc', 'dti'], ['emp_length', 'purpose'],
                             ['Funded amount', 'Annual income', 'Debt-to-income ratio'],
                             ['Employment length', 'Purpose of loan'])

    return estimated_quantiles


def evaluation_2():
    df = load_china_p2p_data()
    N = 100
    df = df.sample(N)
    n = 5
    H = {
        "funded_amnt": [i * (max(df.funded_amnt) - min(df.funded_amnt)) / n for i in range(1, n)],
        "annual_inc": [i * (max(df.annual_inc) - min(df.annual_inc)) / n for i in range(1, n)],
        "dti": [i * (max(df.dti) - min(df.dti)) / n for i in range(1, n)]
    }
    G = {
        "emp_length": [i * (max(df.emp_length) - min(df.emp_length)) / n for i in range(1, n)],
        "purpose": [i * (max(df.purpose) - min(df.purpose)) / n for i in range(1, n)]
    }
    taus = [0.01, 0.05, 0.1, 0.25, 0.5]
    Y = df[['int_rate']].to_numpy()
    X_c = df[['funded_amnt', 'annual_inc', 'dti']].to_numpy()
    X_d = df[['emp_length', 'purpose']].to_numpy()
    output = []
    for tau in taus:
        h, g = cross_validation(tau, Y, X_c, X_d, H, G)
        output.append([h, g])
    estimated_quantiles = []
    for tau, [h, g] in zip(taus, output):
        estimated_quantile = iteratively_reweighted_least_squares(tau, df[['int_rate']].to_numpy(),
                                                                  df[['funded_amnt', 'annual_inc', 'dti']].to_numpy(),
                                                                  np.mean(df[['funded_amnt', 'annual_inc',
                                                                              'dti']]).to_numpy(),
                                                                  df[['emp_length', 'purpose']].to_numpy(),
                                                                  np.mean(df[['emp_length', 'purpose']]).to_numpy(),
                                                                  np.array(h), np.array(g))
        estimated_quantiles.append(estimated_quantile)

    results = {tau: estimated_quantile for tau, estimated_quantile in zip(taus, estimated_quantiles)}
    js = json.dumps(results)
    with open(f"results/evaluation_2_N_{N}_{str(datetime.now())}.json", "w") as f:
        f.write(js)

    return results


def evaluation_imdb():
    df = load_imdb_data()
    N = 10
    os_vals = []
    for i in range(N):
        print(f"i = {i}/{N} iterations")
        df = df.sample(df.shape[0])
        splitting_idx = int(df.shape[0] * 0.9)
        df_in = df.iloc[:splitting_idx, :]
        df_out = df.iloc[splitting_idx:, :]
        H = {
            "imdb_score": [i * (max(df.imdb_score) - min(df.imdb_score)) / 30 for i in range(1, 10)]
        }
        G = {
            "content_rating": [i * (max(df.content_rating) - min(df.content_rating)) / 20 for i in range(1, 10)]
        }
        tau = 0.9
        h, g = cross_validation(tau, df_in[["gross"]].to_numpy(), df_in[["imdb_score"]].to_numpy(),
                                df_in[["content_rating"]].to_numpy(), H, G)
        os_check_values = []
        for _, row in df_out.iterrows():
            os_check_values.append(
                check_function(row.gross - iteratively_reweighted_least_squares(tau, df_in.gross.to_numpy(),
                                                                                df_in.imdb_score.to_numpy().reshape(
                                                                                    df_in.imdb_score.shape[0], 1),
                                                                                np.array([row.imdb_score]),
                                                                                df_in.content_rating.to_numpy().reshape(
                                                                                    df_in.content_rating.shape[0], 1),
                                                                                np.array([row.content_rating]),
                                                                                np.array(h), np.array(g)), tau))
        os_vals.append(np.mean(os_check_values))
    print(f"Mean: {np.mean(os_vals)}, Std: {np.std(os_vals)}")
    return os_vals


if __name__ == "__main__":
    """
    Getting returns and covariance
    """
    tickers = [1, 2, 3]
    ticker_assets = {1: 'Low-risk asset', 2: 'Mid-risk asset', 3: 'High-risk asset'}
    mu = np.array([0.05, 0.1, 0.25])
    sigma = np.array([[0.03, 0, 0], [0, 0.2, 0.02], [0, 0.02, 0.5]])
    # sigma = np.array([[0.01, 0, 0], [0, 0.2, 0.02], [0, 0.02, 0.6]])
    # tickers = ['^GSPC', '^TYX', 'AAPL']
    # ticker_assets = {'^GSPC': 'S&P500', '^TNX': 'Treasury Yield 10 Years', 'BTC-USD': 'Bitcoin US',
    #                  '^HSI': 'Hang Seng Index', '^DJI': 'Dow Jones', 'AAPL': 'Apple', '^IRX': '13 Week Treasury Bill',
    #                  '^TYX': 'Treasury Yield 30 Years'}
    # mu, sigma = load_assets(tickers)
    print(mu)
    print(sigma)

    """
    Test to get conditional quantile value at 5% quantile
    """
    tau = 0.05
    # d = evaluation_1()
    with open("results/evaluation_1_N_1000.json") as f:
        d = json.load(f)
    gammas = []
    weights = np.array([])
    for k in d.keys():
        for val, H in zip(d[k][0], d[k][1]):
            try:
                gamma = compute_gamma(H, tau, mu, sigma)
                portfolio_weights = compute_portfolio_weights(gamma, mu, sigma)
                print(f"Feature: {k}={val}, H: {H}, gamma: {gamma}, portfolio_weights: {portfolio_weights}")
                gammas.append(gamma)
                weights = np.append(weights, portfolio_weights)
            except InfeasibleProblemException:
                pass
    weights = weights.reshape((len(gammas), len(tickers)))
    fig, axs = plt.subplots(ncols=len(tickers), figsize=(18, 5))
    for i in range(len(tickers)):
        lst = sorted(zip(gammas, weights[:, i]))
        pts = np.array([lst[i] for i in range(0, len(lst), 2)])
        axs[i].scatter(pts[:, 0], pts[:, 1])
        axs[i].set_xlabel('Implied Risk Aversion')
        axs[i].set_ylabel('Allocation')
        axs[i].set_title(ticker_assets[tickers[i]])
    plt.savefig(f"results/Portfolio_Weights_{str(datetime.now())}.jpg")

    """
    Test to get conditional quantile values at various quantiles, with conditioned covariates fixed at mean
    """
    # results = evaluation_2()
    # with open("results/evaluation_2_N_100.json") as f:
    #     results = json.load(f)
    # for tau, H in results.items():
    #     try:
    #         gamma = compute_gamma(H, tau, mu, sigma)
    #         portfolio_weights = compute_portfolio_weights(gamma, mu, sigma)
    #         print(f"H: {H}, tau: {tau}, gamma: {gamma}, portfolio_weights: {portfolio_weights}")
    #     except InfeasibleProblemException:
    #         pass

    """
    Test portfolio with new investor's demographics and goals
    """
    # df = load_china_p2p_data()
    # H = iteratively_reweighted_least_squares(tau, df[['int_rate']].to_numpy(),
    #                                          df[['funded_amnt', 'annual_inc', 'dti']].to_numpy(),
    #                                          np.array([25, 100, 20]),
    #                                          df[['emp_length', 'purpose']].to_numpy(), np.array([1, 1]),
    #                                          np.array([7.8, 82.48400000000001, 18.54]), np.array([5.4, 1.8]))
    # print(f"H: {H}, tau: {tau}")
    # try:
    #     gamma = compute_gamma(H, tau, mu, sigma)
    #     portfolio_weights = compute_portfolio_weights(gamma, mu, sigma)
    #     print(f"H: {H}, tau: {tau}, gamma: {gamma}, portfolio_weights: {portfolio_weights}")
    # except InfeasibleProblemException:
    #     print("Infeasible")
    #     pass
