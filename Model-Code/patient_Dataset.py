import torch
import os
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

PAT_DATA = pd.read_csv("../data/dataByPatient/PATIENTS.csv")
adm_dat = pd.read_csv("../data/patientData/ADMISSIONS.csv")

eth_list = adm_dat["ETHNICITY"].unique()
lang_list = adm_dat["LANGUAGE"].unique()
mar_list = adm_dat["MARITAL_STATUS"].unique()
ins_list = adm_dat["INSURANCE"].unique()
rel_list = adm_dat["RELIGION"].unique()

eth_Encoder = LabelEncoder().fit(eth_list)
ins_Encoder = LabelEncoder().fit(ins_list)
lang_Encoder = LabelEncoder().fit(lang_list)
rel_Encoder = LabelEncoder().fit(rel_list)
mar_Encoder = LabelEncoder().fit(mar_list)

def getPatientData(patientId : int, directory : str):
    out = {}
    for subDir in os.listdir(directory):
        try:
            out[subDir] = pd.read_csv(
                f"{directory}/{subDir}/{patientId}.csv", engine = "python"
                )
        except FileNotFoundError:
            out[subDir] = None
        except NotADirectoryError:
            pass
    return out

DATA_DIR = "../data/dataByPatient"
PATIENT_LIST = '../data/dataByPatient/patientList.pkl'

class PatientData:

    def __init__(self, patientId):
        self.patientId = patientId
        patDict = getPatientData(patientId, DATA_DIR)
        self.inputs = patDict["Inputs"]
        self.adm = patDict["Adm"]
        self.notes = patDict["Notes"]
        self.labs = patDict["Labs"]
        self.cpt = patDict["CPT"]
        self.microbio = patDict["Microbio"]
        self.outputs = patDict["Outputs"]
    
    def __str__(self) -> str:
        return str(self.patientId)

class PatientDataset(Dataset):

    def __init__(self, patient_details:str, transform):
        with open(patient_details, 'rb') as pickle_file:
            self.patientList = pickle.load(pickle_file)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.patientList)
    
    def __getitem__(self, idx:int):
        patIdx = self.patientList[idx]
        return self.transform(PatientData(patIdx))

def transform(patDat:PatientData):
    dem_data = dict(
        gender = 0 if PAT_DATA[PAT_DATA["SUBJECT_ID"] == patDat.patientId]["GENDER"].iloc[0] == "F" else 1,
        ethnicity = eth_Encoder.transform(np.asarray(patDat.adm.ETHNICITY, dtype=object))[0],
        religion = rel_Encoder.transform(np.asarray(patDat.adm.RELIGION, dtype=object))[0],
        insurance = ins_Encoder.transform(np.asarray(patDat.adm.INSURANCE, dtype=object))[0],
        mar_stat = mar_Encoder.transform(np.asarray(patDat.adm.MARITAL_STATUS, dtype=object))[0],
        lang = lang_Encoder.transform(np.asarray(patDat.adm.LANGUAGE, dtype=object))[0]
    )
    if patDat.inputs is not None:
        input_data = dict(
            time = torch.Tensor(patDat.inputs["STORETIME"].to_numpy()).T,
            var = torch.Tensor(patDat.inputs["ITEMID"].to_numpy()).T,
            val = torch.Tensor(patDat.inputs["AMOUNT"].to_numpy()).T,
        )
    if patDat.outputs is not None:
        output_data = dict(
            time = torch.Tensor(patDat.outputs["CHARTTIME"].to_numpy()).T,
            var = torch.Tensor(patDat.outputs["ITEMID"].to_numpy()).T,
            val = torch.Tensor(patDat.outputs["VALUE"].to_numpy()).T,
        )

    note_data = dict(
        text = patDat.notes["DESCRIPTION"],
        time = patDat.notes["CHARTTIME"]
    )

    cpt_data = dict(
        time = torch.Tensor(patDat.cpt["CHARTDATE"].to_numpy()).T,
        procedure = torch.Tensor(patDat.cpt["ITEMID"].to_numpy()).T,
    ) 

    micro_data = dict(
        time = torch.Tensor(patDat.microbio["CHARTTIME"].to_numpy()).T,
        spec_data = patDat.microbio["SPEC_ITEMID"],
        org_data = patDat.microbio["ORG_ITEMID"],
        inter = patDat.microbio["INTERPRETATION"]
    )

    lab_data = dict(
       time = torch.Tensor(patDat.labs["CHARTTIME"].to_numpy()).T,
        var = torch.Tensor(patDat.labs["ITEMID"].to_numpy()),
        val = torch.Tensor(patDat.labs["FLAG"].to_numpy()).T, 
    )
    
    return dict(
        dem = dem_data,
        inputs = input_data,
        output = output_data,
        note = note_data,
        cpt = cpt_data,
        micro = micro_data,
        lab = lab_data
    )

print(PatientDataset(PATIENT_LIST, transform)[10])