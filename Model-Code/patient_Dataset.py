import torch
import os
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader

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
        patDict = getPatientData(patientId, DATA_DIR)
        self.patDetails = patDict["Patient"]
        self.inputsCV = patDict["InputsCV"]
        self.adm = patDict["Adm"]
        self.chart = patDict["Chart"]
        self.drugs = patDict["Drugs"]
        self.notes = patDict["Notes"]
        self.labs = patDict["Labs"]
        self.cpt = patDict["CPT"]
        self.microbio = patDict["Microbio"]
        self.inputsMV = patDict["InputsMV"]
        self.outputs = patDict["Outputs"]
    
    def __str__(self) -> str:
        return str(self.patDetails["SUBJECT_ID"])

class PatientDataset(Dataset):

    def __init__(self, patient_details:str, transform):
        with open(patient_details, 'rb') as pickle_file:
            self.patientList = pickle.load(pickle_file)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.patientList)
    
    def __getitem__(self, idx:int) -> PatientData:
        patIdx = self.patientList[idx]
        return self.transform(PatientData(patIdx))



