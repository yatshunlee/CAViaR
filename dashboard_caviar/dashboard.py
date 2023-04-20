# Usage: streamlit run dashboard.py

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

if ticker:
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
            options=[0.01, 0.05]
        )

    with col3:
        option_method = st.radio(
            'Optimization Method:',
            options=['RQ', 'mle']
        )

    caviar_model = CaviarModel(option_quantile, model=option_caviar, method=option_method)
    caviar_model.fit(in_samples)
    
    if np.isnan(caviar_model.training_loss):
        st.write('Fail to optimize. Try other ticker/specification/quantile/method.')
        
    else:
        VaR = caviar_model.predict(out_of_samples)
        forecast = VaR[-1]
        VaR = VaR[:-1]

        binom_out = binomial_test(out_of_samples, VaR, caviar_model.quantile)
        traffic_out = traffic_light_test(out_of_samples, VaR, caviar_model.quantile)
        pof_out = kupiec_pof_test(out_of_samples, VaR, caviar_model.quantile)
        cci_out = christoffersen_test(out_of_samples, VaR)
        dq_out = caviar_model.dq_test(out_of_samples, 'out')
        
        st.subheader("Statistics [Out-of-Sample]")
        
        if option_caviar == 'adaptive':
            text_with_equation = r'$$f_{t}(\beta_{1}) = f_{t-1}(\beta_{1}) + \beta_{1} \cdot ([1 + \exp(G[y_{t-1} - f_{t-1}(\beta_{1})])]^{-1} - \theta)$$'
        elif option_caviar == 'asymmetric':
            text_with_equation = r'$$f_t(\beta) = \beta_{1} + \beta_{2} f_{t-1}(\beta) + \beta_{3} \cdot max(y_{t-1}, 0) + \beta_{4} \cdot min(y_{t-1}, 0)$$'
        elif option_caviar == 'igarch':
            text_with_equation = r'$$f_t(\beta) = \sqrt{\beta_{1} + \beta_{2} f_{t-1}^2(\beta) + \beta_{3} y_{t-1}^2}$$'
        elif option_caviar == 'symmetric':
            text_with_equation = r'$$f_t(\beta) = \beta_{1} + \beta_{2} f_{t-1}(\beta) + \beta_{3} |y_{t-1}|$$'
        else:
            raise ValueError("specification must be in {'adaptive', 'asymmetric', 'igarch', 'symmetric'}")
            
        st.write('The equation of the chosen specification is:')
        st.write(text_with_equation, unsafe_allow_html=True)
        
        st.table(caviar_model.beta_summary().style.apply(
            lambda s: ['background-color: #ffe6e6']*len(s) if s['pval of beta'] > 0.05
            else ['background-color: #ffffcc']*len(s) if s['pval of beta'] > 0.01 
            else ['background-color: #e6ffe6']*len(s), axis=1
        ))
        
        backtest_df = pd.DataFrame({
            'Binomial': [binom_out],
            'Traffic Light': [traffic_out[0]],
            'POF Test': [pof_out],
            'IID Test': [cci_out],
            'DQ Test': [dq_out]
        })
        
        st.table(backtest_df.style.applymap(
            lambda x: 'background-color: #e6ffe6' if x > 0.05 else 'background-color: #ffe6e6',
            subset = ['Binomial', 'POF Test', 'IID Test', 'DQ Test']
        ).applymap(
            lambda x: 'background-color: #e6ffe6' if x == 'green' else 'background-color: #ffe6e6' if x == 'red' else 'background-color: #ffffcc',
            subset = ['Traffic Light']
        ))
        
        if backtest_df.isna().sum().sum() > 0:
            st.write('<span style="color: gray;">**Note that nan is possibly due to the overflowing error.</span>', unsafe_allow_html=True)
        
        st.subheader("CAViaR Plots [Out-of-Sample]")
        
        fig1 = caviar_model.plot_caviar(out_of_samples, 'out')
        st.pyplot(fig1)
        
        fig2 = caviar_model.plot_news_impact_curve()
        st.pyplot(fig2)
        
        st.subheader("VaR Forecast (day t+1)")
        
        price_df = pd.DataFrame({
            'Last Close Price': [df['Close'][-1]],
            'VaR Forecast t+1 (Return%)': [forecast],
            'VaR Forecast t+1 (Price)': [df['Close'][-1] * (1+forecast/100)]
        })
        st.table(price_df)

else:
    st.write("Please enter some ticker, e.g. SPY, JPM, AAPL, MSFT.")