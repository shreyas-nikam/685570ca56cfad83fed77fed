
import streamlit as st
import pandas as pd

def run_data_input():
    st.header("Data Input & Variable Selection")

    # File Uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())  # Display the first few rows of the dataframe

            # Variable Selection
            column_names = df.columns.tolist()
            dependent_variable = st.selectbox("Select the dependent variable", options=column_names)
            independent_variables = st.multiselect("Select the independent variable(s)", options=column_names)

            # Store selected variables in session state
            st.session_state['data'] = df
            st.session_state['dependent_variable'] = dependent_variable
            st.session_state['independent_variables'] = independent_variables

            st.success("Data loaded and variables selected!")

        except Exception as e:
            st.error(f"Error loading data: {e}")
    else:
        st.info("Please upload a CSV file to proceed.")
