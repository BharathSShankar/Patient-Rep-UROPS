import pandas as pd

def read_csv_patients(filepath, date_time_cols = None):
    df = pd.read_csv(filepath)
    if date_time_cols:
        df[date_time_cols] = df[date_time_cols].apply(
            lambda x : pd.to_datetime(
                x , format = "%Y-%m-%d %H:%M:%S"
                ).dt.to_pydatetime()
                )
    return df
