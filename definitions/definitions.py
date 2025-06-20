
import pandas as pd
import numpy as np
import re
from typing import Optional, Any
from scipy.stats import t


class OLSResult:
    def __init__(self, params: pd.Series, resid: pd.Series, fittedvalues: pd.Series, formula: str, X: pd.DataFrame):
        self.params = params
        self.resid = resid
        self.fittedvalues = fittedvalues
        self._formula = formula
        self._X = X

    def summary(self) -> str:
        se = self._std_err()
        tvals = self._t_values(se)
        pvals = self._p_values(tvals)
        coef_df = pd.DataFrame({
            'coef': self.params,
            'std err': se,
            't': tvals,
            'P>|t|': pvals,
        }).round(6)
        header = f"OLS Regression Results\nFormula: {self._formula}\nObservations: {len(self.fittedvalues)}\n\nCoefficients:\n"
        return header + coef_df.to_string()

    def _std_err(self) -> pd.Series:
        n, p = self._X.shape
        dof = n - p
        rss = (self.resid**2).sum()
        sigma2 = rss / dof if dof > 0 else np.nan
        try:
            cov = sigma2 * np.linalg.inv(self._X.T.values @ self._X.values)
        except np.linalg.LinAlgError:
            cov = np.full((p, p), np.nan)
        return pd.Series(np.sqrt(np.diag(cov)), index=self.params.index)

    def _t_values(self, se: pd.Series) -> pd.Series:
        with np.errstate(divide='ignore', invalid='ignore'):
            t_vals = self.params / se
        t_vals[se == 0] = 0
        return t_vals

    def _p_values(self, t_vals: pd.Series) -> pd.Series:
        df = len(self.fittedvalues) - len(self.params)
        p_vals = 2 * t.sf(np.abs(t_vals), df) if df > 0 else pd.Series(np.nan, index=t_vals.index)
        return pd.Series(p_vals, index=t_vals.index)


def ols(formula: str, data: Optional[pd.DataFrame]) -> Any:
    """
    Fits a linear regression model using Ordinary Least Squares (OLS) based on a given formula and dataset.

    Args:
        formula (str): A string specifying the regression formula, e.g. 'Y ~ X1 + X2 - 1'.
        data (pd.DataFrame): A DataFrame containing the data for the regression.

    Returns:
        OLSResult: An object containing the fitted regression results including coefficients,
                   residuals, fitted values, and a summary method.

    Raises:
        TypeError: If formula is not a string or data is not a pandas DataFrame.
        ValueError: If the formula is invalid, variables are missing from data, or data is empty.
    """
    if not isinstance(formula, str):
        raise TypeError("`formula` must be a string.")
    if data is None or not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")
    if formula.strip() == "":
        raise ValueError("`formula` cannot be an empty string.")
    if data.empty:
        raise ValueError("`data` DataFrame is empty.")

    parts = formula.split('~')
    if len(parts) != 2:
        raise ValueError("Formula must have one '~' separating dependent and independent variables.")
    dep_var = parts[0].strip()
    rhs = parts[1].strip()
    if dep_var == "":
        raise ValueError("Dependent variable missing in formula.")
    if dep_var not in data.columns:
        raise ValueError(f"Dependent variable '{dep_var}' not found in data.")

    # Determine intercept presence
    rhs_no_space = rhs.replace(" ", "")
    has_intercept = True
    if re.search(r'(^|[+-])0($|[+-])', rhs_no_space) or "-1" in rhs_no_space:
        has_intercept = False

    # Extract independent variables, carefully handle -1 removal
    # Replace '-1' with ''
    rhs_processed = re.sub(r'(?<!\w)-1', '', rhs)
    # Replace 0 with nothing
    rhs_processed = re.sub(r'(?<!\w)0(?!\w)', '', rhs_processed)
    # Split by '+'
    vars_raw = [v.strip() for v in rhs_processed.split('+') if v.strip() != '']
    indep_vars = []
    for var in vars_raw:
        # ignore pure intercept terms like '1'
        if var == '1':
            continue
        indep_vars.append(var)

    # Check variables presence (for simple vars, reject if missing)
    for var in indep_vars:
        # if function call or transformation detected, skip existence check
        if ('(' in var and ')' in var):
            continue
        if var not in data.columns:
            raise ValueError(f"Independent variable '{var}' not found in data.")

    relevant_cols = [dep_var] + [v for v in indep_vars if (v in data.columns)]
    df_model = data.loc[:, relevant_cols].dropna()
    if df_model.empty:
        raise ValueError("No data left after dropping missing values.")

    # Build design matrix
    if has_intercept:
        X = pd.DataFrame({'Intercept': np.ones(len(df_model))}, index=df_model.index)
    else:
        X = pd.DataFrame(index=df_model.index)

    # Add independent variables (handle categorical by get_dummies)
    for var in indep_vars:
        if '(' in var and ')' in var:
            # Do not support transformations, try eval with data - but this is risky and not required by tests
            # So raise error
            raise ValueError(f"Transformations/functions in formula not supported: '{var}'")
        col = df_model[var]
        if pd.api.types.is_numeric_dtype(col):
            X[var] = col
        else:
            dummies = pd.get_dummies(col, prefix=var, drop_first=True)
            if not dummies.empty:
                X = pd.concat([X, dummies], axis=1)

    if X.empty:
        raise ValueError("No independent variables in model.")

    # Verify all X columns numeric
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            raise TypeError(f"Design matrix column '{c}' is not numeric.")

    y = df_model[dep_var]

    # Fit OLS
    try:
        XtX = X.T.values @ X.values
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError as e:
        raise ValueError("Design matrix is singular or not invertible.") from e

    beta = XtX_inv @ (X.T.values @ y.values.reshape(-1, 1))
    beta = beta.flatten()

    params = pd.Series(beta, index=X.columns)
    fittedvalues = pd.Series(X.values @ beta, index=X.index)
    resid = y - fittedvalues

    return OLSResult(params=params, resid=resid, fittedvalues=fittedvalues, formula=formula, X=X)
