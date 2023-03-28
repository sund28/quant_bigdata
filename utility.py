import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from qdatasdk import *



def init_logging(log_name, level=logging.INFO):
    """
     初始化日志
    :param log_name: 日志文件名前缀，例如：日志文件名是factor_20230101.log, 其中的factor就是log_name
    :param level: 输出到文件中的日志级别。可选级别： 参考python的内置包logging
    :return:
    """
    # 生成日志存放在log目录下
    log_path = "log"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    date = datetime.now().strftime("%Y%m%d%H%M")

    # 定义日志输出格式
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    logger = logging.getLogger()
    # 删除已经存在的handler: 避免在notebook环境下,init_logging运行多次，挂载多个相同handler,日志重复输出。
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 输出日志到文件中
    log_file = f"{log_path}/{log_name}_{date}.log"
    fh = logging.FileHandler(log_file, mode="w", encoding="utf8")
    fh.setFormatter(formatter)
    # 文件输出信息可以自定义
    fh.setLevel(level)
    logger.addHandler(fh)

    # 输出日志到当前控制台
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    # 默认当前控制台只输出警告、异常信息
    # 注意： 控制台输出的日志如果太多，在jupyter环境下可能导致notebook无法正常加载
    ch.setLevel(logging.WARNING)
    logger.addHandler(ch)
    logger.setLevel(level)


@dataclass
class Position:
    """
    持仓
    """
    security: str
    date: [str, datetime.date]
    avg_cost: float


@dataclass
class SelectedStock:
    """
    每期入选的标的
    """
    security: str
    date: [str, datetime.date]
    factor_value: float
    price: float


@dataclass
class StockRor: #rate of return 
    """
    截至到当前调仓日，成分股的收益统计
    """
    # 标的
    security: str
    # 开仓日期
    open_date: [str, datetime.date]
    # 开仓价
    open_price: float
    # 平仓日期
    close_date: [str, datetime.date]
    # 平仓价
    close_price: float
    # 收益率
    ror: float = 0


@dataclass
class StatsData:
    """
    每个调仓日的统计数据
    """
    # 调仓日
    date: [str, datetime.date]
    # 当期收益
    profit: float
    # 成分股平均收益率
    avg_ror: float
    # 成分股收益率统计
    stock_ror_list: List[StockRor] = field(default_factory=list)


@dataclass
class Portfolio:
    """
    每个组对应的账户
    """
    # 初始净值
    net_value: float = 1
    # 持仓
    positions: List[Position] = field(default_factory=list)
    # 股票池：所有入选股票
    stock_pool: List[SelectedStock] = field(default_factory=list)
    # 所有调仓日的统计数据
    stats_datas: List[StatsData] = field(default_factory=list)




# 定义因子检验类
class Backtest:
    def __init__(self, params: dict, history_data: dict):
        self.params = params

        self.factor_name = params.get("factor")
        self.group_number = params.get("group_number")

        self.trade_dates = self.load_trade_dates(self.params.get("start_date"), self.params.get("end_date"))
        self.adjust_dates = self.init_adjust_dates(self.trade_dates, self.params.get("adjust_days"))

        self.history_data = history_data

        self.portfolios = self.init_portfolios()

        # 单调性得分
        self.ic_list = []

    def init_portfolios(self):
        portfolios = []
        for group in range(0, self.group_number):
            portfolios.append(Portfolio(net_value=1))
        return portfolios

    def init_adjust_dates(self, trade_dates, adjust_days):
        """
         初始化调仓日。
        :return:回测期间所有的调仓日
        """
        days = len(trade_dates)
        return [trade_dates[i] for i in range(0, days) if (i % adjust_days == 0) or (i == days - 1)]
    
    def get_current_bar(self, security, date: datetime): #bar 即 K线
        """
        获取当前交易日K线数据
        :param security: 标的代码
        :param date: 指定日期
        :return: 日K数据, 字典类型
        """
        df = self.history_data.get(security)
        if type(df) == DataFrame and len(df) > 0:
            bars = df[df.index == date]
            if type(bars) == DataFrame and len(bars) > 0:
                return bars.iloc[0]

    def get_last_bar(self, security, date: datetime):
        """
        获取最近一个交易日K线数据
        :param security: 标的代码
        :param date: 指定日期
        :return: 日K数据, 字典类型
        """
        df = self.history_data.get(security)
        if type(df) == DataFrame and len(df) > 0:
            bars = df[df.index <= date]
            if type(bars) == DataFrame and len(bars) > 0:
                return bars.iloc[-1]

    def load_trade_dates(self, start_date, end_date):
        """
        加载交易日数据
        :return: 回测期间所有的交易日
        """
        dates = get_trade_days(start_date, end_date)
        dates = sorted(dates)
        return dates

            
    def calc_m_score(self):
        """
        计算单调性得分
        :param date:
        :return:
        """
        numbers = len(self.portfolios)
        profit_delta_list = []
        for pre_idx in range(0, numbers):
            post_idx = (numbers - pre_idx) - 1
            if pre_idx == post_idx:
                break
            pre = self.portfolios[pre_idx]
            post = self.portfolios[post_idx]
            if len(pre.stats_datas) == 0:
                break

            profit_delta = pre.stats_datas[-1].profit - post.stats_datas[-1].profit
            profit_delta_list.append(profit_delta)
        if len(profit_delta_list) == 0:
            score = 0
        else:
            score = profit_delta_list[0] / profit_delta_list[-1] if profit_delta_list[-1] != 0 else 0
        logging.warning(f"单调性得分 = {score}")

        return score

    def calc_ic(self, date):
        dfs = []
        for portfolio in self.portfolios:
            stock_pool = portfolio.stock_pool
            fdf = pd.DataFrame([asdict(s) for s in stock_pool])
            if len(stock_pool) == 0:
                break
            # 当期的统计数据：包含股票池中成分股的收益
            last_stats = portfolio.stats_datas[-1]
            # 股票池中成分股的收益
            ror_list = [asdict(r) for r in last_stats.stock_ror_list]
            # 上期入选股票收益：转换成DataFrame
            rdf = pd.DataFrame(ror_list)

            # 以股票代码为索引，对齐收益、因子值
            df = pd.concat([fdf, rdf], names=["security"], axis=1)
            # 入选股票池：没有买入的股票收益为0
            df = df.fillna(value=0)
            df = df[["security", "ror", "factor_value"]]
            dfs.append(df)

        # 合并所有组合的收益、因子值
        if len(dfs) > 0:
            df = pd.concat(dfs)
            # 计算因子值、收益相关性
            ic = df["factor_value"].corr(df["ror"])
        else:
            ic = None

        logging.warning(f"计算IC: date={date}, ic={ic}")
        self.ic_list.append({"date": date, "ic": ic})

    def draw_profit(self):
        # 设置plt日志级别
        plt.set_loglevel('WARNING')
        # 设置画布大小
        plt.figure(figsize=(14, 5))

        # 设置X轴数据：日期序列
        dates = [data.date.strftime("%Y-%m-%d") for data in self.portfolios[0].stats_datas]
        # 如果显示的刻度有重叠，可以抽样显示刻度：X轴刻度的索引号
        ticks = np.arange(0, len(dates), 10)
        # 设置X轴刻度
        plt.xticks(ticks, rotation=45)

        # 画图、同时设置legend
        legend = []

        # 第1组、最后1组收益
        first, last = [], []

        for pid in range(0, len(self.portfolios)):
            portfolio = self.portfolios[pid]
            profits = [data.profit for data in portfolio.stats_datas]
            plt.plot(dates, profits, label=f'G{pid + 1}')
            legend.append(f'G{pid + 1}')
            if pid == 0:
                first = profits
            elif pid == len(self.portfolios) - 1:
                last = profits

        deltas = [first[i] - last[i] for i in range(0, len(first))]
        label_name = f'0-portfolio-{self.factor_name}'
        plt.plot(dates, deltas, label=label_name, linestyle="--")
        legend.append(label_name)

        plt.legend(legend)
        plt.show()

    def is_trade(self, side, bar):
        bar = bar.to_dict()
        logging.debug(f"检查交易状态: {side}, {bar}")
        # 检查停牌
        if bar["paused"]:
            logging.info(f"停牌无法交易: {side}, {bar}")
            return False
        if side == "buy":
            # 涨停无法买入:
            if bar["close"] >= bar["up"]:
                logging.info(f"涨停无法交易: {side}, {bar}")
                return False
        elif side == "sell":
            # 跌停无法卖出
            if bar["close"] <= bar["down"]:
                logging.info(f"跌停无法交易: {side}, {bar}")
                return False
        return True

    def calc_profit(self, pid, date):
        """
        计算收益
        :param pid: 账户id(?)
        :param date:
        :return:
        """
        portfolio = self.portfolios[pid]
        positions = portfolio.positions
        net_value = portfolio.net_value
        stock_ror_list = []
        for position in positions:
            security = position.security
            # 如果持仓期间股票退市，就以退市前最后一个交易日收盘价计算收益
            bar = self.get_last_bar(security, date)
            last_price = bar["fq_close"]
            open_price = position.avg_cost
            # 计算个股收益率: 收盘价前复权 /开仓价 - 1
            ror = last_price / open_price - 1
            stock_ror = StockRor(security=security, open_date=position.date, open_price=open_price,
                                 close_date=bar.name, close_price=last_price, ror=ror)
            stock_ror_list.append(stock_ror)

        # 统计收益
        sr_list = [sr.ror for sr in stock_ror_list]
        avg_ror = 0 if len(sr_list) == 0 else sum(sr_list) / len(sr_list)
        portfolio.net_value = net_value * (1 + avg_ror)
        # 累计收益
        acc_profit = portfolio.net_value - 1
        stats_data = StatsData(date=date, profit=acc_profit, avg_ror=avg_ror, stock_ror_list=stock_ror_list)

        # 更新上期收益
        portfolio.stats_datas.append(stats_data)
        self.portfolios[pid] = portfolio

    def close(self, pid, date):
        """
        平仓
        :param pid: 账户ID
        :param date: 日期
        :return:
        """
        self.calc_profit(pid, date)

        portfolio = self.portfolios[pid]
        positions = portfolio.positions
        # 更新持仓：清空持仓股
        unclosed_positions = []
        for position in positions:
            security = position.security
            bar = self.get_last_bar(security, date)
            # 停牌或跌停无法卖出
            if type(bar) == Series and not self.is_trade('sell', bar):
                # 因为下个调仓日要统计收益，所以这里重置持仓价格
                position.avg_cost = bar["fq_close"]
                unclosed_positions.append(position)

        portfolio.positions = unclosed_positions
        self.portfolios[pid] = portfolio

    def grouping(self, target: list, size: int) -> list:
        """
        分组，每组长size
        """
        # 数组元素总数量
        total_size = len(target)
        # 分组后：平均每组包含元素数量
        length = int(total_size / size)
        # 平均分配后：多出来的元素数量
        remainder = total_size % size
        # 分组后的元素
        groups = []
        start_idx = 0
        g_idx = 1 #index of group 
        while start_idx < total_size:
            end_idx = start_idx + length
            if g_idx <= remainder:
                end_idx = end_idx + 1
            groups.append(target[start_idx: end_idx])
            g_idx += 1
            start_idx = end_idx
        return groups

    def group(self, stocks, group_number):
        # 排序：按因子值从小到大
        stocks = sorted(stocks, key=lambda x: x["factor_value"], reverse=False)
        # 数组元素总数量
        total_size = len(stocks)
        # 分组后：平均每组包含数量
        length = int(total_size / group_number)
        # 平均分配后：多出来的数量
        remainder = total_size % group_number
        # 分组后
        groups = []
        start_idx = 0
        g_idx = 1
        while start_idx < total_size:
            end_idx = start_idx + length
            if g_idx <= remainder:
                end_idx = end_idx + 1
            groups.append(stocks[start_idx: end_idx])
            g_idx += 1
            start_idx = end_idx
        return groups

    def filter(self, date):
        """
        过滤掉ST special treatment 
        :param date:
        :return:
        """
        # 查询当日在上市的标的
        df = get_all_securities(date)
        # 过滤掉ST
        stocks = []
        factor = self.factor_name
        for security in list(df.index):
            bar = self.get_current_bar(security, date)
            if type(bar) != Series:
                continue
            if bar["is_st"]:
                continue
            stocks.append({"security": security, "factor_value": bar[factor], "bar": bar})
        return stocks

    def select(self, pid, date, stocks):
        """
        选股
        :param pid: 账户信息
        :param date: 日期
        :param stocks: 候选股票
        :return:
        """
        portfolio = self.portfolios[pid]
        selected_stocks = []
        for stock in stocks:
            bar = stock["bar"]
            if not self.is_trade("buy", bar):
                continue
            selected_stock = SelectedStock(security=stock["security"], date=date, factor_value=bar['close'],
                                           price=bar['fq_close'])
            logging.debug(f"selected stock: date={date}, stock={selected_stock}")
            selected_stocks.append(selected_stock)

        # 更新当期股票池
        portfolio.stock_pool = selected_stocks
        self.portfolios[pid] = portfolio

    def open(self, pid, date, stocks):
        """
        开仓，也叫建仓，即开始交易股票
        :param pid: 账户ID
        :param date: 日期
        :param stocks: 当期入选的股票
        :return:
        """

        # 记录当期入选的股票
        stock_pool = []
        for stock in stocks:
            bar = stock["bar"]
            selected_stock = SelectedStock(security=stock["security"], date=date, factor_value=bar['close'],
                                           price=bar['fq_close'])
            logging.debug(f"selected stock: date={date}, stock={selected_stock}")
            stock_pool.append(selected_stock)

        # 更新入选的股票
        portfolio = self.portfolios[pid]
        portfolio.stock_pool = stock_pool

        # 记录仓位信息
        positions = portfolio.positions
        for stock in stocks:
            bar = stock["bar"]
            if not self.is_trade("buy", bar):
                continue
            position = Position(security=stock["security"], date=date, avg_cost=bar["fq_close"])
            logging.info(f"open position: date={date}, position={position}")
            positions.append(position)

        # 更新账户持仓
        portfolio.positions = positions
        self.portfolios[pid] = portfolio

    def run(self):
        for date in self.trade_dates:
            if date not in self.adjust_dates:
                continue

            # 卖出
            for pid in range(0, len(self.portfolios)):
                self.close(pid, date)

            # 计算IC
            self.calc_ic(date)

            # 选股
            # 1. 过滤ST
            stocks = self.filter(date)
            # 2. 分组
            grouped_stocks = self.group(stocks, self.group_number)

            # 买入
            for pid in range(0, len(self.portfolios)):
                self.open(pid, date, grouped_stocks[pid])
