import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_csv(
    df: pd.DataFrame,
    target_col: str = "pce_1",
    scaler: StandardScaler = None,
    fit_scaler: bool = True,
):
    if fit_scaler:
        scaler = StandardScaler()
        scaler.fit(df[target_col])
    
    df[target_col] = df[target_col].astype(np.float64)
    df.loc[:, target_col] = scaler.transform(df.loc[:, target_col])
    return df, scaler

def inverse_transform(
    y, 
    scaler,
):
    return y * scaler.scale_ + scaler.mean_


