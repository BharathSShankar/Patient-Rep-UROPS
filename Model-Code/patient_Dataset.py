import math
import torch
import os
import pandas as pd
import pickle
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

SENT_MODEL = SentenceTransformer('../Sent_Model')
for param in SENT_MODEL.parameters():
    param.requires_grad = False

PAT_DATA = pd.read_csv("../data/patientData/PATIENTS.csv")
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
                ).fillna(method='bfill')
        except FileNotFoundError:
            out[subDir] = None
        except NotADirectoryError:
            pass
    return out
base = "/hpctmp/e0550582/UROPS_Proj"
DATA_DIR = base + "/data/dataByPatient"
PATIENT_LIST = base + '/data/dataByPatient/patientList.pkl'

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



def transform(patDat:PatientData):
    num_default = torch.Tensor([[1e-6]])
    input_data, output_data, note_data, cpt_data, micro_data, lab_data = dict(
        time = num_default,
        var = num_default,
        val = num_default
    ), dict(
        time = num_default,
        var = num_default,
        val = num_default
    ), dict(
        text = SENT_MODEL.encode([""]),
        time = num_default
    ), dict(
        time = num_default,
        procedure = num_default
    ), dict(
        time = num_default,
        spec_data = num_default,
        org_data = num_default,
        inter =  num_default
    ), dict(
        time = num_default,
        var = num_default,
        val = num_default 
    )

    dem_data = torch.Tensor([
        0 if PAT_DATA[PAT_DATA["SUBJECT_ID"] == patDat.patientId]["GENDER"].iloc[0] == "F" else 1,
        0 if math.isnan(eth_Encoder.transform(np.asarray(patDat.adm.ETHNICITY, dtype=object))[0]) else eth_Encoder.transform(np.asarray(patDat.adm.ETHNICITY, dtype=object))[0],
        0 if math.isnan(lang_Encoder.transform(np.asarray(patDat.adm.LANGUAGE, dtype=object))[0]) else lang_Encoder.transform(np.asarray(patDat.adm.LANGUAGE, dtype=object))[0],
        0 if math.isnan(rel_Encoder.transform(np.asarray(patDat.adm.RELIGION, dtype=object))[0]) else rel_Encoder.transform(np.asarray(patDat.adm.RELIGION, dtype=object))[0],
        0 if math.isnan(ins_Encoder.transform(np.asarray(patDat.adm.INSURANCE, dtype=object))[0]) else ins_Encoder.transform(np.asarray(patDat.adm.INSURANCE, dtype=object))[0],
        0 if math.isnan(mar_Encoder.transform(np.asarray(patDat.adm.MARITAL_STATUS, dtype=object))[0]) else mar_Encoder.transform(np.asarray(patDat.adm.MARITAL_STATUS, dtype=object))[0]
    ])
    if not (patDat.inputs is None):
        input_data = dict(
            time = torch.Tensor([list(patDat.inputs["STORETIME"].fillna(0))]).T,
            var = torch.Tensor([list(patDat.inputs["ITEMID"].fillna(0))]).T,
            val = torch.Tensor([list(patDat.inputs["AMOUNT"].fillna(0))]).T,
        )
    
    if not (patDat.outputs is None):
        output_data = dict(
            time = torch.Tensor([list(patDat.outputs["CHARTTIME"].fillna(0))]).T,
            var = torch.Tensor([list(patDat.outputs["ITEMID"].fillna(0))]).T,
            val = torch.Tensor([list(patDat.outputs["VALUE"].fillna(1))]).T,
        )
    
    if not (patDat.notes is None):
        note_data = dict(
            text = SENT_MODEL.encode(list(patDat.notes["DESCRIPTION"].fillna(""))),
            time = torch.Tensor([list(patDat.notes["CHARTTIME"].fillna(0))]).T,
        )

    if not (patDat.cpt is None):
        cpt_data = dict(
            time = torch.Tensor([list(patDat.cpt["CHARTDATE"].fillna(0))]).T,
            procedure = torch.Tensor([list(patDat.cpt["CPT_NUMBER"].fillna(0))]).T,
        )
    
    micro_mappings = {
        "S" : 1,
        "R" : 2,
        "I" : 3,
        np.nan: 0 
    }

    if not (patDat.microbio is None):
        micro_data = dict(
            time = torch.Tensor([list(patDat.microbio["CHARTTIME"].fillna(0))]).T,
            spec_data = torch.Tensor([list(patDat.microbio["SPEC_ITEMID"].fillna(0))]).T,
            org_data = torch.Tensor([list(patDat.microbio["ORG_ITEMID"].fillna(0))]).T,
            inter = torch.Tensor([[micro_mappings.get(i, 0) for i in patDat.microbio["INTERPRETATION"]]]).T
        )
    
    lab_mappings = {
        'abnormal' : 1,
        'delta': 2
    }
    if not(patDat.labs is None):
        lab_data = dict(
            time = torch.Tensor([list(patDat.labs["CHARTTIME"].fillna(0))]).T,
            var = torch.Tensor([list(patDat.labs["ITEMID"].fillna(0))]).T,
            val = torch.Tensor([[lab_mappings.get(i, 0) for i in patDat.labs["FLAG"].to_numpy()]]).T, 
        )
    
    return dict(
        dem = dem_data,
        inputs = input_data,
        outputs = output_data,
        note = note_data,
        cpt = cpt_data,
        micro = micro_data,
        lab = lab_data
    ), torch.tensor([PAT_DATA[PAT_DATA["SUBJECT_ID"] == patDat.patientId]["EXPIRE_FLAG"].iloc[0]], dtype = torch.float)

class PatientDataset(Dataset):

    def __init__(self, patient_list, transform = transform):
        self.patientList = patient_list
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.patientList)
    
    def __getitem__(self, idx:int):
        patIdx = self.patientList[idx]
        return self.transform(PatientData(patIdx))

