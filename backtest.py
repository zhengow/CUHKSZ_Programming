# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:09:53 2019

@author: Alan
"""
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import jqdatasdk as jq

def double_ma(data,stock_list,fast_window,slow_window):
    '''signal = [long=+1/short=-1,amount] '''
    close=data(stock_list)['close']
    signal = close
    fast_ma = close
    slow_ma = close
    for stock in stock_list:
        fast_ma[list] = talib.SMA(close.list.values,fast_window)
        slow_ma[list] = talib.SMA(close.list.values,slow_window)
        fm = np.array(fast_ma[list])
        sm = np.array(slow_ma[list])
        nan = np.isnan(slow_ma)
        for i in range(0,np.size(fm)):
            if(nan[i]==True):
                signal[list]=[0]
            else:
                if(fm[i]>sm[i]):
                    signal[list]=[1]
                else:
                    signal[i]=[-1]
    
    return signal

class BacktestEngine(object):
    """
    This class is used to read data,
    process data to standard form.
    """
    
    def __init__(self,start_date,end_date):
        self.data = self.load_data(start_date,end_date)
        self.close = data(self.stock_list)['close']
        self.len = self.close[self.stock_list[0]].size()
        self.open =  data(self.stock_list)['open']
        '''trade_log = [cash,shares,total,signal]'''
        self.trade_log_0 = np.empty([len,2]) #it is cash and balance
        self.trade_log_0[0] = [10000,10000]
        self.trade_log_1 = pd.dataframe(np.empty([self.len,self.stock_list.size()]),columns=self.stock_list) #it is share
        
        
    def load_data(self,start_date,end_date):
        jq.auth("15825675534",'Programming123')
        self.stock_list = ['000300.XSHG', '000001.XSHE']
        return jq.get_price(security=self.stock_list,start_date=start_date,end_date=end_date)
        
        
    def load_strategy(self, strategy_name,parameters):
        self.signal = strategy_name(self.data,self.stock_list,parameters[0],parameters[1])
        
    def run(self):
        for i in range(0,self.len-1):
            if(self.trade_log_0[i][1]<0):
                self.trade_log_0[i+1]=self.trade_log_0[i]
                self.trade_log_1[i+1]=self.trade_log_0[i]
                continue;
            self.trade_log_0[i+1]=self.trade_log_0[i]
            self.trade_log_1[i+1]=self.trade_log_1[i]
            for stock in self.stock_list:
                if self.signal[stock][i]!=self.trade_log_1[stock][i]:
                    self.trade_log_1[stock][i+1]=self.signal[stock][i]
                    self.trade_log_0[i+1][0]=self.trade_log_0[i][0]-self.open[stock][i+1]*(self.trade_log_1[stock][i+1]-self.trade_log_1[stock][i])
            self.trade_log_0[i+1][1]=self.trade_log_0[i+1][0]+self.trade_log_1[stock][i+1]*self.close[stock][i+1]
            
    
    def show(self):
        x = range(0,self.len)
        y = self.trade_log_0[:,1]
        plt.plot(x,y)
        plt.show()
    
if __name__ == '__main__':
    start_date, end_date = '2017-01-01', '2017-12-31'
    Backtester = BacktestEngine(start_date=start_date,end_date=end_date)
    parameters=[10,20]
    Backtester.load_strategy(double_ma,parameters);
    Backtester.run();
    Backtester.show();
    
        
        
