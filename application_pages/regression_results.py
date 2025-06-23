
import streamlit as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

def run_regression_results():
    st.header("Regression Results")

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

        # Display the regression summary
        st.subheader("Regression Summary")
        st.write(model.summary())

        # Display the coefficients in a dataframe
        st.subheader("Coefficients")
        coefficients = pd.DataFrame({
            'Coefficient': model.params,
            'Standard Error': model.bse,
            't-statistic': model.tvalues,
            'p-value': model.pvalues
        })
        st.dataframe(coefficients)

    except Exception as e:
        st.error(f"Error fitting the regression model: {e}")
