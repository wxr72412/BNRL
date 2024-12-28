import numpy as np

def get_missing_index(df):  # get the index of column of df (dataset)
    """
    param df: df is read from record.dat
                       "Hypovolemia"  "StrokeVolume"  ...  "VentLung"  "Intubation"
            0                1.0             1.0  ...         NaN           NaN
            1                1.0             1.0  ...         NaN           NaN
            2                1.0             1.0  ...         NaN           NaN
            3                0.0             0.0  ...         NaN           NaN
            4                0.0             0.0  ...         NaN           NaN
        return e.g., [2,3] represents the index of the latent variable
    """

    row_mis = df.iloc[0, :].T
    # return the index of nonzero, e.g., array([[4], [5], [6]])
    # print('np.isnan:\n', np.isnan(np.asarray(row_mis)))
    mis_index = np.argwhere(np.isnan(np.asarray(row_mis)))
    mis_index = [int(i) for i in mis_index]
    mis_var = [df.columns.tolist()[i] for i in mis_index]
    return mis_index, mis_var