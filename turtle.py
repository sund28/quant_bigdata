"""
海龟交易法
初始资金1000万

1.交易标的
沪深300指数

2.开仓
- 多头：突破过去20日最高价[收盘价]
- 空头：突破过去20日最低价[收盘价]
- 仓位：总资金 * (20/10000) / N

* 备注：
 （1）一般海龟开仓用总资金的1%[100BP]为一个头寸单位, 这里为了演示加仓效果，20BP作为一个头寸单位。
 （2）使用ATR代替N, ATR=近20日TR的平均值

3.加仓
- 加仓：开仓后向有利方向移动0.5倍ATR(20日)

4.止盈止损
- 多头：
   （1）突破过去20日最低价
   （2）每次建仓、加仓移动止损线，向不利方向移动2倍ATR止损
- 空头
   （1）突破过去20日最高价
   （2）每次建仓、加仓移动止损线，向不利方向移动2倍ATR止损。
"""
import copy
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from qdatasdk import *
import numpy as np


class Side(Enum):
    """交易方向"""
    # 多
    LONG = 'LONG'
    # 空
    SHORT = 'SHORT'


class Offset(Enum):
    """交易方向"""
    # 开仓
    OPEN = 'OPEN'
    # 平仓
    CLOSE = 'CLOSE'


@dataclass
class Position:
    """
    持仓
    """
    security: str
    date: [str, datetime.date]
    side: Side
    volume: int = 0
    avg_cost: float = 0


@dataclass
class Portfolio:
    """
    账户
    """
    # 总市值
    total_value: float
    # 可用资金
    available_cash: float
    # 持仓：多头持仓或空头持仓, 默认为空。
    position: Position = None


@dataclass
class PortfolioDaily:
    """
    每日账户结算信息
    """
    date: datetime.date
    # 可用资金
    available_cash: float
    # 总市值
    total_value: float
    # 持仓：多头持仓或空头持仓, 默认为空。
    position: Position = None


class Backtest:
    def __init__(self, setting: dict):
        self.start_date = setting.get("start_date")
        self.end_date = setting.get("end_date")
        self.cash = float(setting.get("init_cash"))

        self.security = setting.get("security")
        self.portfolio = Portfolio(total_value=self.cash,
                                   available_cash=self.cash)
        # 设置唐奇安通道时间窗口
        self.window = int(setting.get("window"))
        # 最近一次开仓、加仓的价格：用于浮动止盈止损
        self.last_trade_price = None

        self.trade_dates = self.load_trade_dates()
        self.history_data = self.load_history_bars()
        self.history_data_day = get_bars(self.security, self.start_date, self.end_date, unit="1d", type="index")
        print(len(self.history_data))
        self.current_date = None
        # 每日结算数据
        self.settle_data = []
        self.trades = []

    def load_history_bars(self):
        df = get_bars(self.security, self.start_date, self.end_date, unit="1m", type="index")
        
        #print(df[mask])
        #df_filter = df[mask]
        #print(len(df_filter))

        return df

    def load_trade_dates(self):
        """
        加载交易日数据
        :return: 回测期间所有的交易日
        """
        dates = get_trade_days(self.start_date, self.end_date)
        dates = sorted(dates)
        return dates

    def run(self):
        for date in self.trade_dates:
            self.current_date = datetime.fromisoformat(f'{date} 14:00:00')
            self.run_daily()
            self.current_date = datetime.fromisoformat(f'{date} 14:55:00')
            self.run_daily(close=True)
            self.current_date = datetime.fromisoformat(f'{date} 14:55:00')
            self.settle_daily()

    def run_daily(self, close=False):
        # 获取行情数据
        #bars = self.get_bars(self.current_date, self.window + 1)
        half_hour_window = 8 + 1
        bars = self.get_bars(self.current_date, half_hour_window * 30)
        
        bar_len = len(bars)
        # 如果没有数据，返回
        if bar_len < 8*30:
            logging.debug(f"历史K线数量{len(bars)}<={half_hour_window}，跳过当前交易日{self.current_date}")
            return
                
        bars_30 = bars.iloc[range(0, bar_len, 30)]
        

        last_bar = bars.iloc[-1]
        close_price = last_bar["close"]
        
        position = self.portfolio.position
        if position and close:
            # 加仓或止损
            #signal = self.check_add_or_stop_signal(close_price, self.last_trade_price, atr, position)
            side = position.side
            # if signal == 1:
            #     # 加仓信号
            #     unit = self.calc_unit(self.portfolio.total_value, atr)
            #     self.add_pos(side, close_price, unit)
            # elif signal == -1:
                # 止损平仓信号
            
            self.close_pos(side, close_price)
            position = self.portfolio.position
            return

        
        # 计算r价格 r[7]
        r = bars_30.iloc[-8:-1]["close"] / bars_30.iloc[-9:-2]["close"] - 1

        unit = 1
        if r[0] > 0 and r[7] > 0:
            self.open_pos(side=Side.LONG, price = close_price, volume = unit)
            
        if r[0] <= 0 and r[7] <= 0:
            self.open_pos(side=Side.SHORT, price = close_price, volume = unit)
            

#         # 计算ATR
#         atr = self.calc_atr(bars, self.window)
#         # 检测突破
#         break_signal = self.check_break_signal(bars, close_price, self.window)

        # 如果有持仓： 检测加仓、止盈止损
        position = self.portfolio.position
        
        # 多头开仓信号
#         if break_signal == 1 and (position is None or position.side != Side.LONG):
#             if position is not None and position.side == Side.SHORT:
#                 # 空头平仓
#                 self.close_pos(Side.SHORT, close_price)
#             # 多头开仓
#             unit = self.calc_unit(self.portfolio.total_value, atr)
#             self.open_pos(side=Side.LONG, price=close_price, volume=unit)

#         # 空头开仓
#         elif break_signal == -1 and (position is None or self.portfolio.position.side != Side.SHORT):
#             if position is not None and position.side == Side.LONG:
#                 # 多头平仓
#                 self.close_pos(Side.LONG, close_price)
#             # 空头开仓
#             unit = self.calc_unit(self.portfolio.total_value, atr)
#             self.open_pos(side=Side.SHORT, price=close_price, volume=unit)

    def get_bars(self, end_date, count: int):
        df = self.history_data
        df = df[df.index <= end_date]
        df = df.iloc[-count:]
        return df

    def check_break_signal(self, bars, price, window):
        """
        检测突破信号
        :param bars:
        :param price:
        :param window:
        :return:
        """
        up = max(bars['high'].iloc[-window - 1:-2])
        if price > up:
            # 创新高
            return 1
        down = min(bars['low'].iloc[-window - 1:-2])
        if price < down:
            # 创新低
            return -1
        return 0

    def calc_atr(self, price_list, tr):
        tr_list = [max(price_list['high'].iloc[i] - price_list['low'].iloc[i],
                       abs(price_list['high'].iloc[i] - price_list['close'].iloc[i - 1]),
                       abs(price_list['close'].iloc[i - 1] - price_list['low'].iloc[i])) for i in range(1, tr + 1)]
        atr = np.array(tr_list).mean()
        return atr

    def calc_unit(self, total_value, atr):
        return int(total_value * (20 / 10000) / atr)

    def open_pos(self, side: Side, price, volume):
        available_cash = self.portfolio.available_cash
        if available_cash < price * volume:
            logging.debug(f'开仓失败: 可用资金不足')
            return

        # 更新资金
        self.portfolio.available_cash -= price * volume

        # 更新持仓
        position = Position(security=self.security, date=self.current_date, side=side,
                            volume=volume, avg_cost=price)
        self.portfolio.position = position
        # 记录交易价格：用于止损
        self.last_trade_price = price

        # 记录交易
        trade = dict(security=position.security,
                     date=position.date,
                     side=side,
                     offset=Offset.OPEN,
                     volume=volume,
                     price=price,
                     remark="open")
        logging.debug(f'开仓成功: {trade}')
        self.trades.append(trade)

    def add_pos(self, side: Side, add_price, add_volume):
        position = self.portfolio.position
        available_cash = self.portfolio.available_cash
        # 记录交易价格：用于止损
        self.last_trade_price = add_price

        if available_cash < add_price * add_volume:
            logging.debug(f'加仓失败: 可用资金不足')
            return

        # 更新资金
        self.portfolio.available_cash -= add_price * add_volume
        # 更新持仓
        total_value = position.volume * position.avg_cost + add_price * add_volume
        total_volume = position.volume + add_volume
        position.avg_cost = total_value / total_volume
        position.volume = total_volume
        self.portfolio.position = position

        # 记录交易
        trade = dict(security=position.security,
                     date=self.current_date,
                     side=side,
                     offset=Offset.OPEN,
                     volume=add_volume,
                     price=add_price,
                     remark="add")
        logging.debug(f'加仓成功: {trade}')
        self.trades.append(trade)

    def close_pos(self, side, cover_price):
        position = self.portfolio.position
        if position is None or position.side != side:
            logging.debug(f'平仓失败: 没有可用仓位')
            return

        self.last_trade_price = None

        # 更新资金
        volume = position.volume
        profit = volume * (cover_price - position.avg_cost) * (1 if side == Side.LONG else -1)
        self.portfolio.available_cash += (volume * position.avg_cost + profit)
        self.portfolio.total_value = self.portfolio.available_cash

        # 更新仓位
        self.portfolio.position = None

        # 记录交易
        trade = dict(security=position.security,
                     date=self.current_date,
                     side=side,
                     offset=Offset.CLOSE,
                     volume=volume,
                     price=cover_price,
                     remark="close")
        logging.debug(f'平仓成功: {trade}')
        self.trades.append(trade)

    def check_add_or_stop_signal(self, last_price, last_trade_price, atr, position: Position):
        """
        检测：加仓或止损
        :param last_price:
        :param last_trade_price: 最后一次加仓价
        :param atr:
        :param position:
        :return:
        """
        if last_trade_price is None:
            return 0

        side = position.side
        if (last_price >= last_trade_price + 0.5 * atr and side == Side.LONG) or (
                last_price <= last_trade_price - 0.5 * atr and side == Side.SHORT):
            # 多头加仓或空头加仓
            return 1
        elif (last_price <= last_trade_price - 2 * atr and side == Side.LONG) or (
                last_price >= last_trade_price + 2 * atr and side == Side.SHORT):
            # 多头止损或空头止损
            return -1
        else:
            return 0

    def settle_daily(self):
        portfolio = self.portfolio
        position = portfolio.position
        bars = self.get_bars(self.current_date, 1)

        close = bars.iloc[0]["close"]
        # logging.debug(f"每日结算:{bars.iloc[0].to_dict()}")

        available_cash = self.portfolio.available_cash
        # 账户总市值 = 当日现金可用+持仓总市值
        total_value = available_cash
        if position:
            volume = position.volume
            profit = volume * (close - position.avg_cost) * (1 if position.side == Side.LONG else -1)
            total_value += (volume * position.avg_cost + profit)

        self.portfolio.total_value = total_value
        ps = PortfolioDaily(date=self.current_date,
                            available_cash=available_cash,
                            position=copy.copy(position),
                            total_value=total_value)
        self.settle_data.append(ps)
        logging.debug(f"每日结算:{ps}")
