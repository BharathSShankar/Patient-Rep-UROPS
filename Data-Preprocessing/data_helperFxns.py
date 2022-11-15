import pandas as pd
from tqdm import tqdm

def read_csv_patients(filepath, date_time_cols = None):
    df = pd.read_csv(filepath)
    if date_time_cols:
        df[date_time_cols] = df[date_time_cols].apply(
            lambda x : pd.to_datetime(
                x , format = "%Y-%m-%d %H:%M:%S"
                ).dt.to_pydatetime()
                )
    return df

def normalise_dates(df, ref, cols):
    tmp = df.join(ref, on = "SUBJECT_ID")
    for i in cols:
        tmp[i] = tmp.apply(
            lambda x: (
                x[i].to_pydatetime() - x['DOB'].to_pydatetime()
                ).total_seconds()/(86400*365), axis = 1)
    tmp = tmp.drop(columns = ["DOB"])
    return tmp


def by_patient_data(df, idx_val, path):
    df.drop_duplicates(inplace = True)
    df = df.sort_values(by = ["SUBJECT_ID", idx_val])
    for i in tqdm(df["SUBJECT_ID"].unique()):
        tmp = df[df["SUBJECT_ID"] == i]
        tmp.to_csv(f"../data/dataByPatient/{path}/{int(i)}.csv", mode = "a")
    return df

def full_process(filepath, date_time_cols, ref, idx_val, path):
    df = read_csv_patients(filepath, date_time_cols)
    df = normalise_dates(df, ref, date_time_cols)
    df = df[df['SUBJECT_ID'].notna()]
    if "VALUE" in df.columns:
        df = df[df['VALUE'].notna()]
    return by_patient_data(df, idx_val, path)