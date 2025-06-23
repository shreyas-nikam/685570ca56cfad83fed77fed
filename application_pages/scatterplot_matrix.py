
import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def run_scatterplot_matrix():
    st.header("Scatterplot Matrix")

    # Check if data is available in session state
    if 'data' not in st.session_state:
        st.info("Please upload data and select variables on the Data Input page.")
        return

    df = st.session_state['data']

    try:
        # Generate the scatterplot matrix
        st.subheader("Scatterplot Matrix")
        fig = sns.pairplot(df)
        st.pyplot(fig)
        plt.close(fig.fig) #Prevent from displaying twice

    except Exception as e:
        st.error(f"Error generating the scatterplot matrix: {e}")
