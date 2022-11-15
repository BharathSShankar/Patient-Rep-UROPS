from data_helperFxns import *
import pandas as pd

pat_dat = pd.read_csv('../data/patientData/PATIENTS.csv')
pat_dat = process_dates_pat(
    pat_dat,
    ["DOB","DOD", "DOD_HOSP", "DOD_SSN"]
)

Sub_DOB_mapping = {}
for ind, row in pat_dat[["SUBJECT_ID", "DOB"]].iterrows():
    Sub_DOB_mapping[row.SUBJECT_ID] = row.DOB


#full_process('../data/patientData/ADMISSIONS.csv', 
#    ["ADMITTIME", "DISCHTIME", "DEATHTIME", "EDREGTIME", "EDOUTTIME"], 
#    Sub_DOB_mapping, 
#    "ADMITTIME", 
#    "Adm")

full_process('../data/patientData/INPUTEVENTS_CV.csv', 
    ["CHARTTIME"], 
    Sub_DOB_mapping, 
    "CHARTTIME", 
    "Inputs")

full_process('../data/patientData/INPUTEVENTS.csv', 
    ["CHARTTIME"], 
    Sub_DOB_mapping, 
    "CHARTTIME", 
    "Inputs")

full_process('../data/patientData/NOTEEVENTS.csv', 
    ["CHARTTIME"], 
    Sub_DOB_mapping, 
    "CHARTTIME", 
    "Notes")

full_process('../data/patientData/CHARTEVENTS.csv', 
    ["CHARTTIME"], 
    Sub_DOB_mapping, 
    "CHARTTIME", 
    "Chart")

full_process('../data/patientData/MICROBIOLOGYEVENTS.csv', 
    ["CHARTDATE", "CHARTTIME"], 
    Sub_DOB_mapping, 
    "CHARTTIME", 
    "Microbio")

full_process('../data/patientData/MICROBIOLOGYEVENTS.csv', 
    ["CHARTTIME"], 
    Sub_DOB_mapping, 
    "CHARTTIME", 
    "Outputs")

full_process('../data/patientData/LABEVENTS.csv', 
    ["CHARTTIME"], 
    Sub_DOB_mapping, 
    "CHARTTIME", 
    "Labs")

full_process('../data/patientData/CPTEVENTS.csv', 
    ["CHARTDATE"], 
    Sub_DOB_mapping, 
    "CHARTDATE",
     "CPT")

full_process('../data/patientData/DRGCODES', 
    ["STARTDATE", "ENDDATE"], 
    Sub_DOB_mapping, 
    "STARTDATE", 
    "Drugs")
