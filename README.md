
# Multiple Linear Regression Explorer

This Streamlit application provides an interactive platform for exploring multiple linear regression models. 
Users can upload their own datasets, select dependent and independent variables, and visualize the regression results, 
including coefficients, residuals, and diagnostic plots.

## Instructions

1.  Upload a CSV file using the file uploader on the "Data Input & Variable Selection" page.
2.  Select the dependent and independent variables from the dropdown menus.
3.  Navigate to the "Regression Results" page to view the regression summary and coefficients.
4.  Navigate to the "Residual Analysis" page to view the residual plot and Q-Q plot.
5.  Navigate to the "Scatterplot Matrix" page to view the scatterplot matrix.

## Running the Application

To run the application locally, you can use the following steps:

1.  Make sure you have Python 3.7 or higher installed.
2.  Clone this repository.
3.  Create a virtual environment: `python -m venv venv`
4.  Activate the virtual environment:
    *   On Windows: `venv\Scripts\activate`
    *   On macOS and Linux: `source venv/bin/activate`
5.  Install the dependencies: `pip install -r requirements.txt`
6.  Run the application: `streamlit run app.py`

## Docker

To run the application using Docker, you can use the following steps:

1.  Install Docker.
2.  Clone this repository.
3.  Build the Docker image: `docker build -t multiple-regression-explorer .`
4.  Run the Docker container: `docker run -p 8501:8501 multiple-regression-explorer`

You can then access the application in your browser at `http://localhost:8501`.
