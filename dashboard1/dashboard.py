import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import streamlit as st
from caviar import CaviarModel
from var_tests import *
import yfinance as yf

st.set_option('deprecation.showPyplotGlobalUse', False)

ticker = None

st.title('CAViaR')

ticker = st.text_input('Ticker:')

df = yf.download(ticker, start='2009-12-01')
df['Return'] = df['Close'].pct_change().dropna()
returns = df['Return']
low_open_log_difference = (df['Low'].apply(lambda x: np.log(x)) - df['Open'].apply(lambda x: np.log(x))).dropna()

in_samples = returns['2010':'2019'] * 100
out_of_samples = returns['2020':] * 100

col1, col2, col3 = st.columns(3)

option_caviar = option_quantile = option_method = None

with col1:
    option_caviar = st.radio(
        'CAViaR Specification:',
        options=['adaptive', 'asymmetric', 'igarch', 'symmetric']
    )
    
with col2:    
    option_quantile = st.radio(
        '1 - VaR Level:',
        options=[0.01, 0.05, 0.1, 0.25, 0.5]
    )
    
with col3:
    option_method = st.radio(
        'Optimization Method:',
        options=['RQ', 'mle']
    )

if not (ticker is None):
    caviar_model = CaviarModel(option_quantile, model=option_caviar, method=option_method)
    caviar_model.fit(in_samples)
    
    VaR = caviar_model.predict(out_of_samples)
    forecast = VaR[-1]
    VaR = VaR[:-1]
    
    binom_out = binomial_test(out_of_samples, VaR, caviar_model.quantile)
    traffic_out = traffic_light_test(out_of_samples, VaR, caviar_model.quantile)
    pof_out = kupiec_pof_test(out_of_samples, VaR, caviar_model.quantile)
    cci_out = christoffersen_test(out_of_samples, VaR)
    dq_out = caviar_model.dq_test(out_of_samples, 'out')
    
    st.table(caviar_model.beta_summary())
    
    st.table(pd.DataFrame({
        'Binomial': [binom_out],
        'Traffic Light': [traffic_out[0]],
        'POF Test': [pof_out],
        'IID Test': [cci_out],
        'DQ Test': [dq_out]
    }))
    
    st.table(pd.DataFrame({
        'Current Price': [df['Close'][-1]],
        'Forecast t+1 (Return%)': [forecast],
        'Forecast t+1 (Price)': [df['Close'][-1] * (1+forecast/100)]
    }))
    
    fig = caviar_model.plot_caviar(out_of_samples, 'out')
    st.pyplot(fig)