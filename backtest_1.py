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

def double_ma(data,fast_window,slow_window):
    '''signal = [long=+1/short=-1,amount] '''
    close=np.array(data['close'])
    signal = np.empty([np.size(close),2])
    fast_ma = talib.SMA(close,fast_window)
    slow_ma = talib.SMA(close,slow_window)
    nan = np.isnan(slow_ma)
    
    for i in range(0,np.size(close)):
        if(nan[i]==True):
            signal[i]=[0,0]
        else:
            if(fast_ma[i]>slow_ma[i]):
                signal[i]=[1,1]
            else:
                signal[i]=[-1,1]
    
    return signal

class BacktestEngine(object):
    """
    This class is used to read data,
    process data to standard form.
    """
    
    def __init__(self):
        self.data = self.load_data()
        self.close = np.array(self.data['close'])
        self.open = np.array(self.data['open'])
        '''trade_log = [cash,shares,total,signal]'''
        self.trade_log = np.empty([np.size(self.close),4])
        self.trade_log[0] = [10000,0,10000,0]
        
    def load_data(self):
        jq.auth("15825675534",'Programming123')
        return jq.get_price("000001.XSHE", start_date="2017-01-01", end_date="2017-12-31")
        
    def load_strategy(self, strategy_name,parameters):
        self.signal = strategy_name(self.data,parameters[0],parameters[1])
        
    def run(self):
        for i in range(0,np.size(self.close)-1):
            if(self.trade_log[i][2]<0):
                trade_log[i+1]=trade_log[i]
                continue;
            self.trade_log[i+1]=self.trade_log[i]
            if self.signal[i][0]==1 and self.trade_log[i][1]<=0:
                self.trade_log[i+1][1]=self.signal[i][1]
                self.trade_log[i+1][0]=self.trade_log[i][0]-self.open[i+1]*(self.trade_log[i+1][1]-self.trade_log[i][1])
                self.trade_log[i+1][3]=1
            if self.signal[i][0]==-1 and self.trade_log[i][1]>=0:
                self.trade_log[i+1][1]=self.signal[i][1]
                self.trade_log[i+1][0]=self.trade_log[i][0]-self.open[i+1]*(self.trade_log[i+1][1]-self.trade_log[i][1])
                self.trade_log[i+1][3]=-1
            self.trade_log[i+1][2]=self.trade_log[i+1][0]+self.trade_log[i+1][1]*self.close[i+1]
            
    
    def show(self):
        x = range(0,np.size(self.close))
        y = self.trade_log[:,2]
        plt.plot(x,y)
        plt.show()
    
if __name__ == '__main__':
    Backtester = BacktestEngine()
    parameters=[10,20]
    Backtester.load_strategy(double_ma,parameters);
    Backtester.run();
    Backtester.show();
