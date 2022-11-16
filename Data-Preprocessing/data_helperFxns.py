import pandas as pd
from tqdm import tqdm
import os

def process_dates_pat(df, date_time_cols = None):
    if date_time_cols:
        df[date_time_cols] = df[date_time_cols].apply(
            lambda x : pd.to_datetime(
                x , format = "%Y-%m-%d %H:%M:%S"
                ).dt.to_pydatetime()
                )
    return df

def normalise_dates(df, ref, cols):
    tmp = df
    for i in cols:
        tmp[i] = tmp.apply(
            lambda x: (
                x[i].to_pydatetime() - ref[x["SUBJECT_ID"]].to_pydatetime()
                ).total_seconds()/(86400*365), axis = 1)
    return tmp

def by_patient_data(df, idx_val, path):
    df = df.sort_values(by = ["SUBJECT_ID", idx_val])
    for i in df["SUBJECT_ID"].unique():
        tmp = df[df["SUBJECT_ID"] == i]
        if os.path.isfile(f"/hpctmp/e0550582/UROPS_Proj/data/dataByPatient/{path}/{int(i)}.csv"):
            tmp.to_csv(f"/hpctmp/e0550582/UROPS_Proj/data/dataByPatient/{path}/{int(i)}.csv", mode = "a", header = False)
        else: 
            tmp.to_csv(f"/hpctmp/e0550582/UROPS_Proj/data/dataByPatient/{path}/{int(i)}.csv")
    return df

def full_process(filepath, date_time_cols, ref, idx_val, path, usecols = None):
    for df in tqdm(pd.read_csv(filepath, chunksize = 100000, usecols = usecols)):
        df = process_dates_pat(df, date_time_cols)
        df = normalise_dates(df, ref, date_time_cols)
        by_patient_data(df, idx_val, path)
        del df