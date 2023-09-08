import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from zipfile import ZipFile
import tempfile
from io import StringIO
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings

disable_warnings(InsecureRequestWarning)


class FFData():
    """A set of methods to get financial data from the FF data library"""

    def __init__(self):
        """Create the request object"""
        self.s = requests.Session()
        self.baseURL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"

    def getFactorDailyData(self):
        """ Get 5-factor data from a specific Fama French data source """
        downloadURL = self.baseURL + 'ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'
        download = self.s.get(url=downloadURL, verify=False)

        with tempfile.TemporaryFile() as tmpf:
            tmpf.write(download.content)
            with ZipFile(tmpf, "r") as zf:
                data = zf.open(zf.namelist()[0]).read().decode()
        df = pd.read_csv(StringIO(data), engine='python', skiprows=3, skipfooter=0,
                         decimal='.', sep=',').rename(columns={'Unnamed: 0': 'Date'})
        df_filtered = df.copy()
        df_filtered.Date = pd.to_datetime(df_filtered.Date, format="%Y%m%d")
        df_filtered = df_filtered.set_index("Date")
        return df_filtered.astype(float)

    def getIndustryDailyData(self, data_source='10_Industry_Portfolios', weighting_scheme='value'):
        """ Get industry data from a specific Fama French data source """
        downloadURL = self.baseURL + 'ftp/%s_daily_CSV.zip' % (data_source)
        download = self.s.get(url=downloadURL, verify=False)

        with tempfile.TemporaryFile() as tmpf:
            tmpf.write(download.content)
            with ZipFile(tmpf, "r") as zf:
                data = zf.open(zf.namelist()[0]).read().decode()
        df = pd.read_csv(StringIO(data), engine='python', skiprows=9, skipfooter=1,
                         decimal='.', sep=',').rename(columns={'Unnamed: 0': 'Date'})
        if weighting_scheme == 'value':
            keep_flag = 'first'  # keep value weighted returns
        else:
            keep_flag = 'last'  # keep equal weighted returns
        df_filtered = df.loc[df[df.Date.str.isnumeric() == True].drop_duplicates(
            subset=['Date'], keep=keep_flag).index]
        df_filtered.Date = pd.to_datetime(df_filtered.Date, format="%Y%m%d")
        df_filtered = df_filtered.set_index("Date")
        return df_filtered.astype(float)


class FredAPI():
    """A set of methods to get economic data from the FRED® and ALFRED® websites hosted by the Economic Research Division of the Federal Reserve Bank of St. Louis. Requests can be customized according to data source, release, category, series, and other preferences."""

    def __init__(self, api_key):
        """Create the request object"""
        self.s = requests.Session()
        self.api_key = api_key
        self.baseURL = "https://api.stlouisfed.org/fred/"

    def getAllReleases(self):
        """ Get all available releases """
        downloadURL = self.baseURL + \
            'releases?api_key=%s&file_type=json' % (self.api_key)
        download = self.s.get(url=downloadURL)
        return pd.DataFrame(download.json()['releases'])

    def getReleaseDates(self, release_id=10):
        """ Get release dates for a specific economic variable """
        downloadURL = self.baseURL + \
            '/release/dates?release_id=%s&api_key=%s&file_type=json' % (
                release_id, self.api_key)
        download = self.s.get(url=downloadURL)
        return pd.DataFrame(download.json()['release_dates'])

    def getReleaseLinkedSeries(self, release_id=10):
        """ Get economic series for a specific release of economic data"""
        downloadURL = self.baseURL + \
            '/release/series?release_id=%s&api_key=%s&file_type=json' % (
                release_id, self.api_key)
        download = self.s.get(url=downloadURL)
        return pd.DataFrame(download.json()['seriess'])

    def getSerieData(self, serie_id='CPIAUCNS', start_date='', end_date='', flag_real=False):
        """ Get observations time serie for a specific economic variable """
        if flag_real == True:
            downloadURL = self.baseURL + '/series/observations?series_id=%s&api_key=%s&observation_start=%s&observation_end=%s&realtime_start=%s&realtime_end=%s&file_type=json' % (
                serie_id, self.api_key, start_date, end_date, start_date, end_date)
        else:
            downloadURL = self.baseURL + '/series/observations?series_id=%s&api_key=%s&observation_start=%s&observation_end=%s&file_type=json' % (
                serie_id, self.api_key, start_date, end_date)
        download = self.s.get(url=downloadURL)
        df = pd.DataFrame(download.json()['observations'])
        df["value"] = pd.to_numeric(df["value"])
        return df


class Backtest():
    """ A set of methods to backtest a 10-industry portfolio following a regime portfolio allocation model  """

    def __init__(self, data_returns, data_trigger, data_allocation, start_date="1951-03-31", end_date="1989-12-31"):
        # convert to datetime
        self.start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
        self.end_date = pd.to_datetime(end_date, format='%Y-%m-%d')
        # subsample data accordingly
        self.returns = data_returns.loc[self.start_date:self.end_date]
        self.trigger = data_trigger.loc[self.start_date:self.end_date]
        self.allocation_model = data_allocation
        self.portfolio = pd.DataFrame(
            index=self.returns.index, columns=self.returns.columns)
        self.portfolio_perf = pd.DataFrame(index=self.returns.index, columns=[
                                           "Value", "Perf", "Turnover"], dtype=float)
        # initiate portfolio composition
        self.current_state = -1
        self.current_portfolio = 1/len(self.allocation_model.columns)
        self.portfolio.loc[self.start_date] = self.current_portfolio
        self.current_value = 100
        self.portfolio_perf.loc[self.start_date, "Value"] = self.current_value
        # start after first period
        self.dates_to_run = self.returns.iloc[1:].index

    def _event(self, dt):
        # check if there is a new state, return 1 if yes, else 0
        self.new_state = self.trigger.loc[dt].values[0]
        if self.new_state != self.current_state:
            return 1
        else:
            return 0

    def _rebalance(self, dt):
        # apply allocation model
        previous_ptf = self.current_portfolio
        current_returns = self.returns.loc[dt]
        new_ptf = self.allocation_model.loc[self.new_state]
        previous_ptf_drifted = previous_ptf * (1 + current_returns)
        # compute new nav and performance
        self.portfolio_perf.loc[dt, "Value"] = self.current_value * \
            previous_ptf_drifted.sum()
        self.portfolio_perf.loc[dt, "Perf"] = (
            current_returns * previous_ptf).sum()
        # also compute turnover of end-of-period weights versus new weights
        self.portfolio_perf.loc[dt, "Turnover"] = np.sum(
            np.abs(new_ptf - (previous_ptf_drifted / previous_ptf_drifted.sum())))
        # save new weights from alloc model
        self.portfolio.loc[dt] = new_ptf

    def _drift(self, dt):
        # compute drifted weights
        previous_ptf = self.current_portfolio
        current_returns = self.returns.loc[dt]
        previous_ptf_drifted = previous_ptf * (1 + current_returns)
        new_ptf = previous_ptf_drifted / previous_ptf_drifted.sum()
        # compute new nav and performance
        self.portfolio_perf.loc[dt, "Value"] = self.current_value * \
            previous_ptf_drifted.sum()
        self.portfolio_perf.loc[dt, "Perf"] = (
            current_returns * previous_ptf).sum()
        self.portfolio_perf.loc[dt, "Turnover"] = np.nan  # no turnover
        # save drifted weights
        self.portfolio.loc[dt] = new_ptf
        return 0

    def compute_stats(self, reference=None):
        # compute statistic analysis of the backtest
        stats = pd.DataFrame(index=["Portfolio", "Benchmark"], columns=[
                             "Return", "Volatility", "Turnover"], dtype=float)
        stats.loc["Portfolio", "Return"] = self.portfolio_perf.Perf.add(
            1).prod() ** (260 / len(self.portfolio_perf)) - 1
        stats.loc["Portfolio", "Volatility"] = self.portfolio_perf.Perf.std() * \
            np.sqrt(260)
        stats.loc["Portfolio", "Sharpe Ratio"] = stats.loc["Portfolio",
                                                           "Return"] / stats.loc["Portfolio", "Volatility"]
        stats.loc["Portfolio", "Turnover"] = self.portfolio_perf.groupby(
            self.portfolio_perf.index.year).sum().Turnover.mean()
        if reference is not None:
            stats.loc["Benchmark", "Return"] = reference.portfolio_perf.Perf.add(
                1).prod() ** (260 / len(reference.portfolio_perf)) - 1
            stats.loc["Benchmark", "Volatility"] = reference.portfolio_perf.Perf.std(
            ) * np.sqrt(260)
            stats.loc["Benchmark", "Sharpe Ratio"] = stats.loc["Benchmark",
                                                               "Return"] / stats.loc["Benchmark", "Volatility"]
            stats.loc["Benchmark", "Turnover"] = reference.portfolio_perf.groupby(
                reference.portfolio_perf.index.year).sum().Turnover.mean()
            stats.loc["Portfolio", "Beta"] = pd.concat([self.portfolio_perf.Perf, reference.portfolio_perf.Perf], axis=1).cov(
            ).iloc[0, 1] / reference.portfolio_perf.Perf.var()
            stats.loc["Portfolio", "Tracking Error"] = (
                self.portfolio_perf.Perf-reference.portfolio_perf.Perf).std() * np.sqrt(260)
            stats.loc["Portfolio", "Information Ratio"] = (
                stats.loc["Portfolio", "Return"] - stats.loc["Benchmark", "Return"]) / stats.loc["Portfolio", "Tracking Error"]
        return stats

    def run(self):
        for dt in tqdm(self.dates_to_run):
            if self._event(dt) == 1:
                self._rebalance(dt)
            else:
                self._drift(dt)
            self.current_state = self.new_state  # assign new state
            # assign new nav
            self.current_value = self.portfolio_perf.loc[dt, "Value"]
            # assign new portfolio
            self.current_portfolio = self.portfolio.loc[dt]


def prepare_data(data_signal, data_returns, cpi_releases, start_date='1988-12-31', end_date='2023-01-31', actual=True):
    """ A function to create inflation features and choose the release date, i.e. actual or end of month """
    if actual == True:
        cpi_releases["real_date"] = pd.to_datetime(
            cpi_releases.date, format="%Y-%m-%d")
        mapping = cpi_releases.merge(data_signal.sort_values('realtime_start').drop_duplicates(subset='date_adjusted', keep='first').sort_values('date_adjusted')[
                                     ["realtime_start", "date_adjusted"]], how='left', right_on='realtime_start', left_on='real_date')[["date_adjusted", "real_date"]].dropna()
        signal = data_signal.sort_values('realtime_start').drop_duplicates(subset='date_adjusted', keep='last').sort_values(
            'date_adjusted')[["date_adjusted", "value"]].merge(mapping, how='left', on='date_adjusted')
    else:
        signal = data_signal.sort_values('realtime_start').drop_duplicates(
            subset='date_adjusted', keep='last').sort_values('date_adjusted')
    signal = signal.loc[(signal.date_adjusted >= start_date)
                        & (signal.date_adjusted <= end_date)]
    signal['value_yoy'] = signal.value / signal.value.shift(12) - 1
    signal['value_yoy_mean'] = signal['value_yoy'].rolling(
        12).mean()  # mean inflation rate of last 12M
    signal['value_yoy_rate'] = signal['value_yoy'] - \
        signal['value_yoy'].shift(12)  # rate of change of inflation

    conditionList = [(signal['value_yoy'] < signal['value_yoy_mean']) & (signal['value_yoy_rate'] < 0),
                     (signal['value_yoy'] < signal['value_yoy_mean']) & (
                         signal['value_yoy_rate'] >= 0),
                     (signal['value_yoy'] >= signal['value_yoy_mean']) & (
                         signal['value_yoy_rate'] < 0),
                     (signal['value_yoy'] >= signal['value_yoy_mean']) & (signal['value_yoy_rate'] >= 0)]
    choiceList = ['Lower & Falling', 'Lower & Rising',
                  'Higher & Falling', 'Higher & Rising']
    choiceList_code = [0, 1, 2, 3]
    signal['Regime'] = np.select(conditionList, choiceList, default=np.nan)
    signal['Regime_code'] = np.select(
        conditionList, choiceList_code, default=np.nan)
    signal.dropna(inplace=True)
    data_returns = data_returns.loc[(data_returns.index >= start_date) & (
        data_returns.index <= end_date)]
    if actual == True:
        prepared_data = data_returns.merge(signal[['real_date', 'Regime', 'Regime_code']]
                                           .set_index('real_date'), how='left', right_index=True, left_index=True) \
            .fillna(method="ffill").dropna()
    else:
        prepared_data = data_returns.merge(signal[['date_adjusted', 'Regime', 'Regime_code']]
                                           .set_index('date_adjusted'), how='left', right_index=True, left_index=True) \
            .fillna(method="ffill").dropna()
    return prepared_data
