import pandas as pd
import pickle
from data_helperFxns import process_dates_pat


pat_dat = pd.read_csv(
    '../data/patientData/PATIENTS.csv',
    )
pat_dat = process_dates_pat(pat_dat
     ,["DOB","DOD", "DOD_HOSP", "DOD_SSN"])

Sub_DOB_mapping = {}
for ind, row in pat_dat[["SUBJECT_ID", "DOB"]].iterrows():
    Sub_DOB_mapping[row.SUBJECT_ID] = row.DOB
Sub_DOB_mapping = pd.DataFrame.from_dict(
    Sub_DOB_mapping, 
    orient='index', 
    columns = ["DOB"])

x = list(Sub_DOB_mapping.index)
out = {}
for i in x:
    out[i] = out.get(i, 0) + 1
    if out[i] == 2:
        print("Repeat!")
with open('../data/dataByPatient/patientList.pkl', 'wb') as f:
    pickle.dump(sorted(x), f)