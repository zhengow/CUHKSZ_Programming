import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import jqdatasdk as jq
import statsmodels.api as sm
import datetime,time
from dateutil import rrule
from dateutil.relativedelta import relativedelta
import investment_science as ins
#from EmQuantAPI import *

#c.start("forcelogin=1")

#选股策略

#计算某月最后一天的函数
def last_day_of_month(any_day):

    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)  # this will never fail
    return next_month - datetime.timedelta(days=next_month.day)
    
#筛选出符合锐臻指标的股票，并选择上市时间大于两年的股票
'''
d1=datetime.date(2013,11,1)
d2=datetime.date(2019,10,31)
months = rrule.rrule(rrule.MONTHLY, dtstart=d1, until=d2).count()
regent_stock=[[] for i in range(months//3)]
for i in range(months//3):
    new_date=d1+relativedelta(months=3*i)
    start_date1=new_date.strftime("%Y-%m-%d")
    report_date=last_day_of_month(new_date-relativedelta(months=2))
    report_date1=report_date.strftime("%Y-%m-%d")
    twoyears=relativedelta(years=2)
    test_month_head=(new_date-relativedelta(months=1)).strftime("%Y-%m-%d")
    test_month_tail=last_day_of_month(new_date-relativedelta(months=1)).strftime("%Y-%m-%d")
#字符串内拼接变量用“++”
    regent_index = c.cps("B_001004","HOLDFUNDNUM,HOLDFUNDNUM,"+report_date1+",1;STMTHOLDTNUM,STMTHOLDTNUM,"+report_date1+"","([HOLDFUNDNUM] / [STMTHOLDTNUM])<1 ","top=max([HOLDFUNDNUM] / [STMTHOLDTNUM],50),sectordate="+start_date1+"")
    IPOdate=c.css(regent_index.Codes,"LISTDATE","")
    for code,date in IPOdate.Data.items():
        date1="".join(date)
        time_tuple = time.strptime(date1, '%Y/%m/%d')
        year, month, day = time_tuple[:3]
        date2 = datetime.date(year, month, day)
        if date2 < new_date-twoyears:
            regent_stock[i].append(code) 
        changePercentage_stock=c.csd(regent_stock[i],"PCTCHANGE",test_month_head,test_month_tail,"period=3,adjustflag=3,curtype=1,order=1,market=CNSESH")     
        changePercentage_index=c.csd("000300.SH","PCTCHANGE",test_month_head,test_month_tail,"period=3,adjustflag=3,curtype=1,order=1,market=CNSESH")
        for a,b in changePercentage_index.Data.items():
            benchmark="".join('%s'% x for x in b[0])
            benchmark1=float(benchmark)
        for code,change in changePercentage_stock.Data.items():
            changepercentage=''.join('%s'% change[0][0])
            if changepercentage=='None':
                regent_stock[i].remove(code)
                continue
            else:
                changepercentage1=float(changepercentage)
            if changepercentage1-benchmark1<-7:
                regent_stock[i].remove(code)

All_dates=[] #包含全部报告日期的列表
for i in range(months//3):
    new_Date=(last_day_of_month(d1+relativedelta(months=3*i-2))).strftime("%Y-%m-%d")
    All_dates.append(new_Date)
waiting_list=[[] for i in range(months//3)]#初筛后指标排序
for j in range(len(regent_stock)):
    holding=c.css(regent_stock[j],"STMTHOLDTNUM,HOLDFUNDNUM","ReportDate=%s,CapitalType=1"%All_dates[j]).Data
    for m,n in holding.items():
        rate=n[1]/n[0]
        n.append(rate)
    holding=sorted(holding.items(),key=lambda x :x[1][2] ,reverse=True)
    for y in range(len(holding)):
        waiting_list[j].append(holding[y][0])
#每期留下排名前十只股票
regent_stock1=[[] for i in range(months//3)]
for i in range(len(waiting_list)):
    regent_stock1[i]=waiting_list[i][:10]
'''
#买入策略

def average_position(data,stock_list,risk_free):
    OPEN=stock_panel['OPEN']
    close=stock_panel['CLOSE']
    signal = close.copy()
    signal[signal!=0]=0
    time1=datetime.date(2013,11,1)
    total_balance=10000000
    
    #first change set a change day 
    for i in range(months//3): #tiaocang cishu
        new_Date=time1+relativedelta(months=3*i) #tiaocang ri
        total_stock=0
        flag = 0
        
        excess_return = close[stock_list[10*i:10*(i+1)]]
        for stock in stock_list[10*i:10*(i+1)]:
            len = close[stock].shape[0]
            r_stock = np.log(np.array(close[stock].iloc[1:len])/np.array(close[stock].iloc[0:len-1]))
            r_stock.append(r_stock[-1])
            excess_return[stock] = r_stock-risk_free
        excess_return = excess_return[excess_return.index<=new_Date]
        j=-1    #the jth stock in allocation
        allocation = ins.mle(5,excess_return)
        w0 = 1-sum(allocation)
        month_riskfree = 0.002
        cash = w0*total_balance*(1+month_riskfree)
        
        for stock in stock_list[10*i:10*(i+1)]:
            j+=1
            for x in pre_dict.Dates:    #all trade day
                x1=datetime.datetime.strptime(x,"%Y/%m/%d").date()
                
                if flag==0 and x1 >= new_Date:    #this day is change day
                    year = str(int(x1.strftime('%Y')))
                    month = str(int(x1.strftime('%m')))
                    day = str(int(x1.strftime('%d')))
                    trade_day=year+'/'+month+'/'+day    #set change day
                    flag = 1
                    
                if x1>=new_Date and x1<new_Date+relativedelta(months=3):    
                    signal[stock][x]=(total_balance*allocation[j])//OPEN[stock][trade_day]
                    
                if x1>=new_Date+relativedelta(months=3):    #next tiaocang ri
                    
                    last_day=pre_dict.Dates[pre_dict.Dates.index(x)-1]
                    value_i=total_balance*allocation[j]//OPEN[stock][trade_day]*close[stock][x]
                    total_stock=value_i+total_stock
                    break
        total_balance=total_stock+cash

    return signal

'''
def double_ma(data,stock_list,fast_window,slow_window):
    #signal = [long=+1/short=-1,amount] 
    close=data['close']
    signal = close.copy()

    for stock in stock_list:
        fm = talib.SMA(close[stock],fast_window)
        sm = talib.SMA(close[stock],slow_window)
        nan = np.isnan(sm)
        for i in range(0,np.size(fm)):
            if(nan[i]==True):
                signal[stock][i]=0
            else:
                if(fm[i]>sm[i]):
                    signal[stock][i]=1
                else:
                    signal[stock][i]=-1    
    return signal

'''

def pick_stock():
    stock_pool=[y for x in regent_stock1 for y in x]
    
    return stock_pool
'''
#面板数据输出
nonRedundant=list(set([y for x in regent_stock1 for y in x]))
pre_dict=c.csd(nonRedundant,"OPEN,CLOSE","2013-11-01","2019-10-31","period=1,adjustflag=3,curtype=1,order=1,market=CNSESH")
open_dict=dict()
for k in pre_dict.Data:
    tmp=[]
    for i in range(len(pre_dict.Dates)):
        tmp.append(pre_dict.Data[k][0][i])
    open_dict[k]=tmp
    
close_dict=dict()
for k in pre_dict.Data:
    tmp=[]
    for i in range(len(pre_dict.Dates)):
        tmp.append(pre_dict.Data[k][1][i])
    close_dict[k]=tmp

df_open=pd.DataFrame(pre_dict.Dates)
for k in open_dict:
    df_open.insert(1,k,open_dict[k])
df_open=df_open.set_index([0])
dict_oc=dict()
dict_oc['OPEN']=df_open

df_close=pd.DataFrame(pre_dict.Dates)
for k in close_dict:
    df_close.insert(1,k,close_dict[k])
df_close=df_close.set_index([0])
dict_oc['CLOSE']=df_close
stock_panel=pd.Panel(dict_oc)
'''


class BacktestEngine(object):

    """
    This class is used to read data,
    process data to standard form.
    """

    def __init__(self,start_date,end_date):

        self.start_date = start_date
        self.end_date = end_date
        
        self.stock_list = pick_stock()
        
        self.data = self.load_data()
        self.close = self.data['CLOSE']
        self.open = self.data['OPEN']
        self.len = np.size(self.close[self.stock_list[0]]) #len is the total trade day

        '''it is cash (column=0) and balance (column = 1)'''
        
        self.trade_log = np.empty([self.len,2]) 
        self.trade_log[0] = [10000000,10000000]

        '''it is record the share that we hold'''
        self.share_log = pd.DataFrame(np.empty([self.len,len(self.nonredundant)]),columns=self.nonredundant) 
        for stock in self.nonredundant:
            self.share_log[stock][0]=0
        
        q = query(macro.MAC_LEND_RATE).filter(macro.MAC_LEND_RATE.currency_id==1,macro.MAC_LEND_RATE.market_id==5,macro.MAC_LEND_RATE.term_id==20,macro.MAC_LEND_RATE.day>=self.start_date,macro.MAC_LEND_RATE.day<=self.end_date)
        tmp = macro.run_query(q)
        tmp = tmp.sort_values(by='day')
        
        i, j = 0, 0
        self.rf = np.zeros(self.len-1)
        
        while i<self.len-1 and j<len(tmp['day']):
            
            day = pd.to_datetime(tmp['day'].iloc[j])
            if(day==self.close.index[i]):
                self.rf[i] = tmp['interest_rate'].iloc[j]/360/100
                i+=1
                j+=1
                last = tmp['interest_rate'].iloc[j]/360
            elif(day<self.close.index[i]):
                self.rf[i] = tmp['interest_rate'].iloc[j]/360/100
                j+=1
            elif(day>self.close.index[i]):
                self.rf[i] = tmp['interest_rate'].iloc[j]/360/100
                i+=1
            
        if(i!=self.len):
            while i < self.len-1:
                self.rf[i] = last
                i+=1
        
    def load_data(self):
        jq.auth("15825675534",'Programming123')
        self.nonredundant=list(set(self.stock_list))
        return stock_panel
        #return jq.get_price(security=self.stock_list,start_date=self.start_date,end_date=self.end_date)

    def load_strategy(self, strategy_name):
        self.strategy_name = strategy_name
        self.signal = strategy_name(self.data,self.stock_list,self.rf)       

    def run(self,fee):
        '''if we have a signal in day i, then we will execute it at day i+1 '''
        for i in range(0,self.len-1):
            total_stock_value = 0                                                            #record total cost in stock
            if(self.trade_log[i][1]<0):                                        #if the balance is less than 0, we bankrupt
                self.trade_log[i+1]=self.trade_log[i]
                self.share_log.iloc[i+1]=self.share_log.iloc[i]
                continue;

            self.trade_log[i+1]=self.trade_log[i]                              #first copy day i's situation
            self.signal.iloc[0]=0
            money_to_trade = (self.signal.iloc[i+1]-self.signal.iloc[i])*self.open.iloc[i+1]
            money_to_trade[np.isnan(money_to_trade)]=0
            transaction_fee = sum(abs(money_to_trade))*fee
            total=sum(money_to_trade)

            self.trade_log[i+1][0]=self.trade_log[i+1][0]-total-transaction_fee
            
            today_signal = self.signal.iloc[i+1]
            today_price = self.close.iloc[i+1]
            today_value = today_signal*today_price
            today_value[np.isnan(today_value)]=0
            total_stock_value = sum(today_value)
                
            self.trade_log[i+1][1]=self.trade_log[i+1][0] + total_stock_value                #balance = cash + value of stocks

    def prints(self):
        r_strategy = np.log(np.array(self.trade_log[1:self.len,1])/np.array(self.trade_log[0:self.len-1,1]))          #strategy return rate
        hs300 = jq.get_price('000300.XSHG',start_date=self.start_date, end_date=self.end_date,fields = ('close','pre_close'))
        r_m = np.log(np.array(hs300.close)/np.array(hs300.pre_close))          #market return rate
        r_m = np.delete(r_m,0)
        
        return_period = sum(r_strategy)                                        #total return rate
        x,y = r_m-self.rf,r_strategy-self.rf                                   #excess return rate
        x = x.reshape(len(x),1)
        c = np.ones((len(x),1))
        X = np.hstack((c,x))
        
        '''CAPM model'''
        
        res = (sm.OLS(y,X)).fit()                                             
        alpha, beta = res.params[0], res.params[1]
        vol = np.std(r_strategy)
        loss_rate = len(r_strategy[r_strategy<0])/len(r_strategy)
        loss_ave = r_strategy[r_strategy<0].mean()
        sharpe_ratio = (r_strategy.mean()-self.rf.mean())/vol*(360**0.5)        #360 suitable?
        performance_strategy = {'阶段收益率:':return_period,'詹森系数(alpha):':alpha,'beta:':beta,'波动率:':vol,'亏损比例:':loss_rate,'平均亏损:':loss_ave,
                           'Sharpe比率:':sharpe_ratio}
        
        print(performance_strategy)

    def c_sharpe(self):
        r_strategy = np.log(np.array(self.trade_log[1:self.len,1])/np.array(self.trade_log[0:self.len-1,1]))
        return_period = sum(r_strategy)
        vol = np.std(r_strategy)
        sr = (r_strategy.mean()-self.rf.mean())/vol*(360**0.5)
        
        return sr

    def show(self):
        x = self.close.index
        y = self.trade_log[:,1]
        plt.plot(x,y)
        plt.show()

    '''parameter optimazation'''
    def optimaze(self, p1_min, p1_max, p1_step, p2_min, p2_max, p2_step):
        max_sr = 0
        res = [p1_min,p2_min]
        for i in range(p1_min,p1_max+1,p1_step):
            for j in range(p2_min,p2_max+1,p2_step):
                parameters = [i,j]
                self.load_strategy(self.strategy_name,parameters)
                self.run()
                sr = self.c_sharpe()
                if sr > max_sr:
                    res[0], res[1] = i, j
                    max_sr = sr
        print("Optimazation Parameters are:",res)
                

if __name__ == '__main__':

    start_date, end_date = '2013-11-01', '2019-10-31' #set start time and end time
    Backtester = BacktestEngine(start_date,end_date)  
    
    Backtester.load_strategy(average_position);   #load strategy to create trade signal
    fee = 0.0001
    Backtester.run(fee);                                 #using signal to create trade log
    Backtester.prints();                              #calculate statistics and print it
    Backtester.show();                                #draw our pnl curve
