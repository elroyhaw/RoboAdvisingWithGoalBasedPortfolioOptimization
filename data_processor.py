import pandas as pd
import numpy as np
import pandas_datareader.data as web
from datetime import datetime


def load_china_p2p_data():
    df = pd.read_csv("data/chinaP2PData.csv")
    df.drop(index=0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[["funded_amnt", "emp_length", "home_ownership", "annual_inc", "purpose", "dti", "int_rate"]]
    df.dropna(inplace=True)
    df = df[df.purpose != "other"]
    df.funded_amnt = df.funded_amnt.astype(np.int64)
    df.annual_inc = df.annual_inc.astype(np.float64)
    df.dti = df.dti.astype(np.float64)
    df.int_rate = df.int_rate.apply(lambda x: np.float64(x.replace("%", "")))
    int_rate_map = {int_rate: i for i, int_rate in enumerate(sorted(df.int_rate.unique()))}
    df.int_rate = df.int_rate.apply(lambda x: int_rate_map[x])

    def format_emp_length(emp_length):
        if "+" in emp_length:
            emp_length = emp_length.replace("+ years", "")
        elif "<" in emp_length:
            emp_length = emp_length.replace("< ", "").replace(" year", "")
        else:
            if "1" in emp_length:
                emp_length = emp_length.replace(" year", "")
            else:
                emp_length = emp_length.replace(" years", "")
        return np.int64(emp_length)

    def format_purpose(purpose):
        purposes = ['debt_consolidation', 'home_improvement', 'credit_card', 'vacation',
                    'car', 'medical', 'major_purchase', 'house', 'small_business', 'moving',
                    'renewable_energy']
        purpose_map = dict([(purpose, i) for i, purpose in enumerate(purposes)])
        return purpose_map[purpose]

    df.emp_length = df.emp_length.apply(format_emp_length)
    df.funded_amnt /= 1000
    df.annual_inc /= 1000
    df = df[df.dti >= 0]
    df.purpose = df.purpose.apply(format_purpose)
    # drop anomalies
    df.drop(index=[18534, 17181, 6213], inplace=True)

    return df


def load_imdb_data():
    df = pd.read_csv("data/movie_metadata.csv")
    df = df[(df["title_year"] >= 2011) & (df["title_year"] <= 2013)]
    df = df[["gross", "imdb_score", "content_rating"]]
    df = df[df["gross"] >= 6000000.0]
    df = df[(df["content_rating"] == "PG-13") | (df["content_rating"] == "PG")]
    df["content_rating"] = df["content_rating"].apply(lambda x: 0 if x == "PG-13" else 1)
    df["gross"] /= 10 ** 6
    df.dropna(inplace=True)

    return df


def load_assets(tickers, start_date=None):
    data = []
    for ticker in tickers:
        if start_date:
            adj_close = web.DataReader(ticker,
                                       data_source="yahoo",
                                       start=start_date,
                                       end=datetime.now())['Adj Close']
        else:
            adj_close = web.DataReader(ticker,
                                       data_source="yahoo",
                                       start=datetime(2019, 1, 1),
                                       end=datetime.now())['Adj Close']
        adj_close = adj_close[~adj_close.index.duplicated(keep='first')]
        adj_close.name = ticker
        data.append(adj_close)

    all_adj_close = pd.concat(data, axis=1)
    all_adj_close.dropna(inplace=True)
    returns = all_adj_close.diff() / all_adj_close.shift(1)
    returns.dropna(inplace=True)

    mu = returns.iloc[-1]
    sigma = np.cov(returns, rowvar=False)

    return mu, sigma
