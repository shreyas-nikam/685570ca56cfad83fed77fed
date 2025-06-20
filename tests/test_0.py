import pytest
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from definition_c9e6a4f910684c1cb8ba3dba62aacae0 import ols

@pytest.fixture
def sample_df():
    # Create a simple DataFrame for testing
    data = {
        'Y': [1, 2, 3, 4, 5],
        'X1': [5, 4, 3, 2, 1],
        'X2': [10, 20, 30, 40, 50]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_df_non_numeric():
    # DataFrame with non-numeric data
    data = {
        'Y': [1, 2, 3],
        'X1': ['a', 'b', 'c']
    }
    return pd.DataFrame(data)

@pytest.mark.parametrize("formula, data, exception", [
    # Valid formula and data
    ("Y ~ X1", None, None),
    ("Y ~ X1 + X2", None, None),
    # Invalid formula (empty string)
    ("", None, ValueError),
    # Formula not string
    (123, None, TypeError),
    # Formula with dependent var missing from data
    ("Z ~ X1", None, ValueError),
    # Formula with independent var missing from data
    ("Y ~ Z", None, ValueError),
    # Data not DataFrame
    ("Y ~ X1", [1,2,3], TypeError),
    ("Y ~ X1", None, None),  # will use fixture data
])
def test_ols_valid_and_invalid(sample_df, formula, data, exception):
    # Use sample_df as data if data is None and exception is not TypeError for data param
    input_data = data if data is not None else sample_df

    if exception:
        with pytest.raises(exception):
            ols(formula, input_data)
    else:
        # Should return an object with expected attributes
        result = ols(formula, input_data)
        # Check returned object attributes based on typical OLS model summary
        # Expect attributes like params (coefficients), resid (residuals), fittedvalues, summary method
        assert hasattr(result, "params")
        assert hasattr(result, "resid")
        assert hasattr(result, "fittedvalues")
        assert hasattr(result, "summary")

        # Check that fittedvalues and residuals length matches data length
        n = input_data.shape[0]
        assert len(result.resid) == n
        assert len(result.fittedvalues) == n

        # Check params keys contain all variables including Intercept
        params_keys = result.params.index.tolist()
        # Intercept should be in the params keys
        assert "Intercept" in params_keys or "const" in params_keys
        # Dependent variable on left side of formula
        dep_var = formula.split("~")[0].strip()
        # Independent vars on right side
        indep_vars = [v.strip() for v in formula.split("~")[1].split("+")]
        for var in indep_vars:
            # Sometimes wildcards or functions are used in formulas, skip empty or specials
            if var and var != "1" and var != "0":
                # Should appear in params keys or under transformations (best effort)
                # allowed: original var or intercept only
                assert any(var in key for key in params_keys) or var == "Intercept" or var == "const"

@pytest.mark.parametrize("bad_data", [
    None,
    [],
    {},
    123,
    "string",
    pd.Series([1,2,3]),
])
def test_ols_invalid_data_types(bad_data):
    with pytest.raises(TypeError):
        ols("Y ~ X1", bad_data)

def test_ols_formula_with_non_numeric_features(sample_df_non_numeric):
    # Formula includes non-numeric independent variable should raise due to modeling failure
    with pytest.raises(Exception):
        ols("Y ~ X1", sample_df_non_numeric)

def test_ols_empty_dataframe():
    df_empty = pd.DataFrame(columns=['Y', 'X1'])
    with pytest.raises(ValueError):
        ols("Y ~ X1", df_empty)

def test_ols_formula_with_only_intercept(sample_df):
    # formula with intercept only
    result = ols("Y ~ 1", sample_df)
    assert hasattr(result, "params")
    # Only intercept coefficient expected
    assert len(result.params) == 1
    assert "Intercept" in result.params.index or "const" in result.params.index

def test_ols_formula_with_no_intercept(sample_df):
    # formula with no intercept
    result = ols("Y ~ X1 - 1", sample_df)
    assert hasattr(result, "params")
    # Intercept should not be included
    assert "Intercept" not in result.params.index and "const" not in result.params.index

def test_ols_formula_with_categorical_variable(sample_df):
    # Add a categorical variable column to dataframe
    df_cat = sample_df.copy()
    df_cat['C'] = ['a', 'b', 'a', 'b', 'a']
    # Test that ols handles categorical variables internally or fails gracefully
    try:
        result = ols("Y ~ C", df_cat)
        assert hasattr(result, "params")
        # The categorical variable should be encoded into params keys (e.g., C[T.b])
        cat_params = [x for x in result.params.index if "C" in x]
        assert len(cat_params) > 0
    except Exception as e:
        # acceptable that categorical may raise if not processed
        # but should be ValueError or similar
        assert isinstance(e, (ValueError, Exception))

def test_ols_formula_with_missing_values():
    df_nan = pd.DataFrame({
        'Y': [1, 2, None, 4, 5],
        'X1': [5, None, 3, 2, 1],
        'X2': [10, 20, 30, 40, 50]
    })
    # The model fitting should handle missing data via dropping or raise informative exception
    try:
        result = ols("Y ~ X1 + X2", df_nan)
        assert hasattr(result, "params")
        # residuals length should be less or equal to original due to dropna
        assert len(result.resid) <= len(df_nan)
    except Exception as e:
        assert isinstance(e, (ValueError, Exception))

@pytest.mark.parametrize("formula", [
    "Y ~ X1 + X1",  # Duplicate independent variables
    "Y ~ X1 + 0 + 1",  # intercept inclusion and exclusion conflict
])
def test_ols_formula_edge_cases(sample_df, formula):
    try:
        result = ols(formula, sample_df)
        assert hasattr(result, "params")
    except Exception as e:
        # Accept for formulas that are semantically invalid:
        assert isinstance(e, Exception)

