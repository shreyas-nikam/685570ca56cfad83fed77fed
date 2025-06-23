
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import plotly.express as px

def run_residual_analysis():
    st.header("Residual Analysis")

    # Check if data and variables are available in session state
    if 'data' not in st.session_state or 'dependent_variable' not in st.session_state or 'independent_variables' not in st.session_state:
        st.info("Please upload data and select variables on the Data Input page.")
        return

    df = st.session_state['data']
    dependent_variable = st.session_state['dependent_variable']
    independent_variables = st.session_state['independent_variables']

    # Create the regression formula
    formula = f"{dependent_variable} ~ " + " + ".join(independent_variables)

    try:
        # Fit the regression model
        model = smf.ols(formula, data=df).fit()

        # Calculate residuals and predicted values
        residuals = model.resid
        predicted_values = model.fittedvalues

        # Residual Plot
        st.subheader("Residual Plot")
        fig_resid = px.scatter(x=predicted_values, y=residuals, labels={'x': 'Predicted Values', 'y': 'Residuals'})
        st.plotly_chart(fig_resid)

        # Q-Q Plot
        st.subheader("Q-Q Plot")
        fig_qq = sm.qqplot(residuals, fit=True, line="q")
        st.pyplot(fig_qq.fig)
        plt.close(fig_qq.fig)  # Prevent the plot from displaying twice

    except Exception as e:
        st.error(f"Error fitting the regression model or generating plots: {e}")
