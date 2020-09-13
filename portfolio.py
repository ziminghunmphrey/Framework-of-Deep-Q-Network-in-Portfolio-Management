import numpy as np
import pandas as pd


class DataSrc(object):

    def __init__(self, df,
                 start_date,
                 end_date,
                 date_range_s,
                 date_range_e,
                 steps,
                 expan_coe,
                 trade_period,
                 window_length,
                 random_reset):

        self.steps = steps
        self.random_reset = random_reset
        self.trade_period = trade_period
        self.expan_coe = expan_coe
        self.window_length = window_length
        self.df = df
        self.df['code'] = self.df['code'].astype(str)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df[start_date:end_date]
        self.date_set = pd.date_range(date_range_s, date_range_e)
        self.asset_names = ['asset_1', 'asset_2', 'asset_3', 'asset_4', 'asset_5']
        self.features = ['Close', 'High', 'Low', 'Open']

        asset_dict = dict()

        for asset in ['asset_1', 'asset_2', 'asset_3', 'asset_4', 'asset_5']:
            asset_data = self.df[self.df["code"] == asset].reindex(self.date_set).sort_index()
            asset_data['Close'] = asset_data['Close'].fillna(method='pad')
            asset_data['Open'] = asset_data['Open'].fillna(method='pad')
            asset_data['High'] = asset_data['High'].fillna(method='pad')
            asset_data['Low'] = asset_data['Low'].fillna(method='pad')
            asset_data = asset_data.fillna(method='bfill')
            asset_data = asset_data.fillna(method='ffill')
            asset_dict[str(asset)] = asset_data

        length = len(self.date_set)

        asset_data = asset_dict['asset_1']

        V_close = asset_data.ix[:length, 'Close']

        V_high = asset_data.ix[:length, 'High']

        V_low = asset_data.ix[:length, 'Low']

        V_open = asset_data.ix[:length, 'Open']

        state = []
        for asset in ['asset_2', 'asset_3', 'asset_4', 'asset_5']:
            asset_data = asset_dict[str(asset)]
            V_close = np.vstack((V_close, asset_data.ix[:length, 'Close']))
            V_high = np.vstack((V_high, asset_data.ix[:length, 'High']))
            V_low = np.vstack((V_low, asset_data.ix[:length, 'Low']))
            V_open = np.vstack((V_open, asset_data.ix[:length, 'Open']))

        state.append(V_close)

        state.append(V_high)

        state.append(V_low)

        state.append(V_open)

        state_tensor = np.stack(state, axis=2)

        data = state_tensor

        if self.features[3] not in ['High', 'Low', 'Close', 'Open']:
            self.price_columns = self.features[:3]
        else:
            self.price_columns = self.features[:4]

        self.nb_pc = len(self.price_columns)
        self.close_pos = self.price_columns.index('Close')
        self.open_pos = self.price_columns.index('Open')
        self._data = data


    def _step(self):

        data_window = self.data[:, (self.step)*self.trade_period:(self.step)*self.trade_period+
                                self.window_length,:self.nb_pc].copy()

        data_window_ = self.data[:,(self.step + 1) * self.trade_period:(self.step + 1) * self.trade_period +
                                 self.window_length, :self.nb_pc].copy()

        y1 = data_window_[:, -self.trade_period, self.open_pos].copy() / data_window[:, -self.trade_period,
                          self.open_pos].copy()

        y1 = np.concatenate([[1.0], y1])

        last_close_price = data_window[:, -1, self.close_pos].copy()

        data_window[:, :, :self.nb_pc] /= last_close_price[:, np.newaxis, np.newaxis]

        data_window -= 1

        data_window *= 200

        history = data_window

        self.step += 1

        done = bool(self.step >= self.steps)

        return history, y1, done



    def reset(self):

        self.step = 0

        self.idx = np.random.randint(low=(self.steps+1)*self.trade_period + self.window_length, high=self._data.shape[1])

        self.data = self._data[:, self.idx - (self.steps+1)*self.trade_period-self.window_length:self.idx,:self.nb_pc].copy()





# this class is trading process

class PortfolioSim(object):

    def __init__(self, asset_names, steps, trading_cost):
        self.cost = trading_cost
        self.steps = steps
        self.asset_names = asset_names


    def _step(self, w0, y1):

        _p0_pre = self._p0_pre

        _p0 = self._p0

        _w0 = self._w0

        c1 = self.cost * (np.abs(_w0[1:]-w0[1:])).sum()

        w0_ = (y1 * w0) / np.dot(y1, w0)

        p0_pre_=_p0_pre * np.dot(w0, y1)

        p0_ = _p0 * (1 - c1) * np.dot(w0, y1)

        r1 = np.log(p0_ / _p0)

        self._w0 = w0_

        self._p0 = p0_

        self._p0_pre = p0_pre_

        done = bool(p0_ <= 0)

        info = {"reward": r1,
                "portfolio_value": p0_,
                "weights": w0,
                "last_weight":_w0}

        return r1, info, done


    def reset(self):

        self.infos = []

        self._w0 = np.array([1.0] + [0.0] * len(self.asset_names))

        self._p0 = 1.0





class PortfolioEnv(object):

    def __init__(self,
                 df,
                 start_date,
                 end_date,
                 date_range_s,
                 date_range_e,
                 steps,
                 trading_cost,
                 trade_period,
                 expan_coe,
                 window_length):

        self.src = DataSrc(df=df,
                           start_date=start_date,
                           end_date=end_date,
                           date_range_s= date_range_s,
                           date_range_e= date_range_e,
                           steps=steps,
                           trade_period=trade_period,
                           window_length=window_length,
                           expan_coe=expan_coe)

        self.sim = PortfolioSim(asset_names=self.src.asset_names, trading_cost=trading_cost, steps=steps)


    def _step(self, action):

        history, y1, done1 = self.src._step()

        reward, info, done2 = self.sim._step(action, y1)

        info['steps'] = self.src.step

        self.infos.append(info)

        return {'history': history, 'weights': info["weight"]}, reward, done1 or done2, info


    def _reset(self):
        self.sim.reset()

        self.src.reset()

        self.infos = []

        action = self.sim._w0

        observation ,reward, done, info = self._step(action)

        return observation
