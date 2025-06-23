
import streamlit as st

st.set_page_config(page_title="Multiple Regression Lab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("Multiple Regression Lab")
st.divider()

st.markdown("""
This lab provides an interactive platform for exploring multiple linear regression models. 
You can upload your own datasets, select dependent and independent variables, and visualize the regression results, 
including coefficients, residuals, and diagnostic plots. The application uses insights from statistical principles to guide users 
in understanding the underlying principles and assumptions.
""")

# Navigation
page = st.sidebar.selectbox(
    label="Navigation",
    options=[
        "Data Input & Variable Selection",
        "Regression Results",
        "Residual Analysis",
        "Scatterplot Matrix"
    ]
)

if page == "Data Input & Variable Selection":
    from application_pages.data_input import run_data_input
    run_data_input()
elif page == "Regression Results":
    from application_pages.regression_results import run_regression_results
    run_regression_results()
elif page == "Residual Analysis":
    from application_pages.residual_analysis import run_residual_analysis
    run_residual_analysis()
elif page == "Scatterplot Matrix":
    from application_pages.scatterplot_matrix import run_scatterplot_matrix
    run_scatterplot_matrix()

st.divider()
st.write("© 2025 QuantUniversity. All Rights Reserved.")
st.caption(
    "The purpose of this demonstration is solely for educational use and illustration. "
    "Any reproduction of this demonstration "
    "requires prior written consent from QuantUniversity."
)
