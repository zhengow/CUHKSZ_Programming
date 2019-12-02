# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:53:40 2019

@author: Asus
"""

def mle(risk_aversion,excess_retrun):
    T = excess_return.shape[0]
    lamda = 1/risk_aversion
    Cov_smpl = excess_return.cov()
    Cov_mle = Cov_smpl*(T-1)/T
    Mean_smpl = excess_return.mean()
    x = (Cov_mle**-1)*Mean_smpl;
    w_mle = np.dot(c,m);
    return w_mle
