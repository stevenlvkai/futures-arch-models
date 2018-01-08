# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:57:36 2018

@author: DELL
"""
import numpy as np
import pandas as pd  
from WindPy import *  
import datetime, time  
#import os 
import numpy as np
# 相关库
from scipy import  stats
import statsmodels.api as sm  # 统计相关的库
import arch  # 条件异方差模型相关的库

def collect_data(index,start_date,end_date):
    #由wind导入数据
    w.start()
    stock=w.wsd(index, "close", start_date,end_date)  
    index_data = pd.DataFrame()  
    index_data['trade_date']=stock.Times 
    index_data[index]=stock.Data[0]  
    a=index_data.dropna(axis=0)#删除空白项
    a = a.set_index('trade_date') #将日期设置为索引
    #取年月日
    year=[]
    month=[]
    day=[]
    n=len(a.index)
    for i in range(n):
        year.append(a.index[i].year)
        month.append(a.index[i].month)
        day.append(a.index[i].day)
        
    a['year']=year
    a['month']=month
    a['day']=day

    #取每月最后一天
    l=pd.DataFrame()
    s1=[]
    s2=[]
    s3=[]
    s4=[]
    m=len(a.index)
    for i in range(m-1):
        if a['month'][i] != a['month'][i+1]:
            s1.append(a[index][i])
            s2.append(a['month'][i])
            s3.append(a['year'][i])
            s4.append(a['day'][i])
    l['year']=s3
    l['month']=s2
    l['day']=s4
    l['close']=s1
    
    #取每月第一天
    f=pd.DataFrame()
    s5=[]
    s6=[]
    s7=[]
    s8=[]
    m=len(a.index)
    for i in range(m-1):
        if a['month'][i+1] != a['month'][i]:
            s5.append(a[index][i+1])
            s6.append(a['month'][i+1])
            s7.append(a['year'][i+1])
            s8.append(a['day'][i+1])
    f['year']=s7
    f['month']=s6
    f['day']=s8
    f['close']=s5
    
    return a, l, f

if __name__=='__main__':    
    index=['I.DCE', 'JM.DCE', 'J.DCE', 'RU.SHF', 'RB.SHF', 'HC.SHF', 'SM.CZC', \
           'SF.CZC', 'BU.SHF', 'MA.CZC', 'V.DCE', 'NI.SHF', 'FG.CZC', 'PB.SHF', \
           'ZN.SHF', 'PP.DCE', 'ZC.CZC', 'L.DCE']
    n=len(index)
    data=[]
    data_last_day=[]
    data_first_day=[]
    for i in range(n):
        a, l, f=collect_data(index[i],'19900101','20171222')
        data.append(a)#将所有数据组合
        data_last_day.append(l)
        data_first_day.append(f)
    
    #计算月收益率
    return_all=[]
    for i in range(n):
        d1=data_first_day[i]
        d2=data_last_day[i]
        k=len(d1.index)
        rt=pd.DataFrame()
        r=[]
        year=[]
        month=[]
        for j in range(k):
            year1=d1['year'][j]
            month1=d1['month'][j]
            close1=d1['close'][j]
            for l in range(k):
                year2=d2['year'][l]
                month2=d2['month'][l]
                close2=d2['close'][l]
                if year1==year2 and month1==month2:
                    r1=(close2-close1)/close1#月末减月初
                    r.append(r1)
                    month.append(month1)
                    year.append(year1)
        rt['year']=year
        rt['month']=month
        rt[index[i]]=r
        return_all.append(rt)
        
def get_coefficient(ind):
    # 读取数据
    characteristic=[]
    df=return_all[ind][index[ind]]
    df.index = pd.to_datetime(df.index)  # 将字符串索引转换成时间索引
    ts = df.values  # 生成pd.Series对象
    t = sm.tsa.stattools.adfuller(ts)  # ADF检验
    
    #是否平稳
    if t[1]<=0.05:
        characteristic.append('是')
    else:
        characteristic.append('否')
        
    #计算AR滞后阶数
    lagnum=sm.tsa.pacf(ts, nlags=20, method='ywunbiased', alpha=None)
    n=len(lagnum)
    lagsatis=[]
    aa=1
    for i in range(n):
        if aa==1:
            if abs(lagnum[i]) > 0.05:
                lagsatis.append(i)
            else:
                aa=aa*(-1)#取连续的大于0.05的阶数
        else:    
            break
    
    #建立AR(8)模型，即均值方程
    lagnumber=lagsatis[-1]#AR的阶数
    order=(lagnumber,0)
    model = sm.tsa.ARMA(ts,order).fit()
    
    #计算残差及残差的平方
    at = ts -  model.fittedvalues
    at2 = np.square(at)
    
    # 我们检验25个自相关系数
    m = 25 
    acf,q,p = sm.tsa.acf(at2,nlags=m,qstat=True)  ## 计算自相关系数 及p-value
    out = np.c_[range(1,26), acf[1:], q, p]
    output=pd.DataFrame(out, columns=['lag', "AC", "Q", "P-value"])
    output = output.set_index('lag')
    b=[x[3] for x in out]#读取p-value
    
    #是否序列具有相关性，具有ARCH效应
    s=0
    
    for i in range(5):
        if b[i]>0.05:
            s=s+1
    if s==0:
        characteristic.append('是')
        #建立ARCH模型
        lagnum1=sm.tsa.pacf(at2, nlags=20, method='ywunbiased', alpha=None)#计算滞后阶数
        #计算ARCH滞后阶数
        n=len(lagnum1)
        lagsatis1=[]
        aa=1
        for i in range(n):
            if aa==1:
                if abs(lagnum1[i]) > 0.05:
                    lagsatis1.append(i)
                else:
                    aa=aa*(-1)#取连续的大于0.05的阶数
            else:    
                break
        pnumber=lagsatis1[-1]#ARCH的阶数
        
        train = ts[:-10]
        
        #建立ARCH模型
        am = arch.arch_model(train,mean='AR',lags=lagnumber,vol='ARCH',p=pnumber) 
        res = am.fit()
        
        res.summary()#回归拟合
        arch_coefficient=res.params#取出系数
        arch_tvalue=res.tvalues#取出t值
        arch_final=pd.DataFrame({'coefficient':arch_coefficient,'tvalue':arch_tvalue})
        
        #建立GARCH模型
        am = arch.arch_model(train,mean='AR',lags=lagnumber,vol='GARCH') 
        res1 = am.fit()
        
        res.summary()
        garch_coefficient=res1.params
        garch_tvalue=res1.tvalues#取出t值
        garch_final=pd.DataFrame({'coefficient':garch_coefficient,'tvalue':garch_tvalue})

        #建立EGARCH模型
        am = arch.arch_model(train,mean='AR',lags=lagnumber,vol='EGARCH',p=1, o=1, q=1, power=1.0) 
        res2 = am.fit()
        
        res2.summary()
        egarch_coefficient=res2.params
        egarch_tvalue=res2.tvalues#取出t值
        egarch_final=pd.DataFrame({'coefficient':egarch_coefficient,'tvalue':egarch_tvalue})


    else:
        characteristic.append('否')
        arch_final=pd.DataFrame()
        garch_final=pd.DataFrame()
        egarch_final=pd.DataFrame()
        
    return characteristic,arch_final,garch_final,egarch_final

if __name__=='__main__':            
    status_all=[]
    arch_all=[]
    garch_all=[]
    egarch_all=[]
    
    for i in range(n):
        # 读取数据
        status, arch_final, garch_final, egarch_final=get_coefficient(i)
        status_all.append(status)
        arch_all.append(arch_final)
        garch_all.append(garch_final)
        egarch_all.append(egarch_final)
       
    #输出dataframe至excel    
    writer = pd.ExcelWriter('D:\\output1.xlsx')
    for i in range(n):
        a=arch_all[i].T
        a.to_excel(writer, 'ARCH', startrow=i*3, startcol=0,index_label=index[i])
    
    
    for i in range(n):
        g=garch_all[i].T
        g.to_excel(writer, 'GARCH', startrow=i*3, startcol=0,index_label=index[i])
    
    
    for i in range(n):
        e=egarch_all[i].T
        e.to_excel(writer, 'EGARCH', startrow=i*3, startcol=0,index_label=index[i])    
    writer.save()


        



