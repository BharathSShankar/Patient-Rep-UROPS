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

full_process('../data/patientData/ADMISSIONS.csv', 
    ["ADMITTIME", "DISCHTIME", "DEATHTIME", "EDREGTIME", "EDOUTTIME"], 
    Sub_DOB_mapping, 
    "ADMITTIME", 
    "Adm")

full_process('../data/patientData/LABEVENTS.csv', 
    ["CHARTTIME"], 
    Sub_DOB_mapping, 
    "CHARTTIME", 
    "Labs")

full_process('../data/patientData/MICROBIOLOGYEVENTS.csv', 
    ["CHARTTIME"], 
    Sub_DOB_mapping, 
    "CHARTTIME", 
    "Microbio",
    ["SUBJECT_ID", "HADM_ID", "CHARTTIME","SPEC_ITEMID", "ORG_ITEMID", "INTERPRETATION"])

full_process('../data/patientData/CPTEVENTS.csv', 
    ["CHARTDATE"], 
    Sub_DOB_mapping, 
    "CHARTDATE",
     "CPT",
     ["SUBJECT_ID", "HADM_ID", "CHARTDATE", "CPT_NUMBER"])

full_process('/hpctmp/e0550582/UROPS_Proj/data/patientData/INPUTEVENTS_CV.csv', 
    ["STORETIME"], 
    Sub_DOB_mapping, 
    "STORETIME", 
    "Inputs", 
    ["SUBJECT_ID", "HADM_ID", "STORETIME", "ITEMID", "AMOUNT"])

full_process('/hpctmp/e0550582/UROPS_Proj/data/patientData/INPUTEVENTS_MV.csv', 
    ["STORETIME"], 
    Sub_DOB_mapping, 
    "STORETIME", 
    "Inputs",
    ["SUBJECT_ID", "HADM_ID", "STORETIME", "ITEMID", "AMOUNT"])

full_process('/hpctmp/e0550582/UROPS_Proj/data/patientData/NOTEEVENTS.csv', 
    ["CHARTTIME"], 
    Sub_DOB_mapping, 
    "CHARTTIME", 
    "Notes",
    ["SUBJECT_ID", "HADM_ID", "CHARTTIME","DESCRIPTION"])

full_process('/hpctmp/e0550582/UROPS_Proj/data/patientData/OUTPUTEVENTS.csv', 
    ["CHARTTIME"], 
    Sub_DOB_mapping, 
    "CHARTTIME", 
    "Outputs",
    ["SUBJECT_ID", "HADM_ID", "CHARTTIME","ITEMID", "VALUE"])


