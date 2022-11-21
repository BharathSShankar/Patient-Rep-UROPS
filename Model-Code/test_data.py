import pickle
from fullModel import FullModel
import random
import torch
from torch.utils.data import DataLoader
from patient_Dataset import PatientDataset, transform
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pandas as pd

config = {
  "batch_size": 32,
  "beta": 2,
  "embedDim": 512,
  "finalDim": 120,
  "lr": 1e-05,
  "num_epochs": 10,
  "num_layers": 4,
  "wd": 1e-05
}

PATIENT_LIST = '../data/dataByPatient/patientList.pkl'
with open(PATIENT_LIST, 'rb') as pickle_file:
    patientList = pickle.load(pickle_file) 

random.Random(42).shuffle(patientList)

test_data = patientList[44000:]
test_dataset = PatientDataset(test_data, transform)

def my_collate(batch):
        data_out = {}
        data = [item[0] for item in batch]
        for i in data[0].keys():
            if i == "dem":
                data_out[i] = pad_sequence(
                    [data[k][i] for k in range(len(batch))], batch_first=True
                )
            else:
                data_out[i] = {}
                for j in data[0][i].keys():
                    data_out[i][j] = pad_sequence(
                        [torch.Tensor(data[k][i][j])[:256] for k in range(len(batch))], batch_first=True
                    )
        targets = torch.Tensor([[item[1]] for item in batch])
        return [data_out, targets]

test_loader = DataLoader(
    test_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    collate_fn=my_collate
    )


model = FullModel.load_from_checkpoint(
        "../checkpoint/tune_MIMIC/train_model_58d0d_00000_0_batch_size=16,beta=0,embedDim=512,finalDim=120,lr=0.0000,num_epochs=10,num_layers=4,wd=0.0000_2022-11-19_11-00-39/lightning_logs/version_0/checkpoints/epoch=8-step=22500.ckpt",
        config = config
        )

out = {
    "embeddings" : [],
    "prediction" : [],
    "actual" : []
}

model.eval()
print("beginning test....")
for x, y in tqdm(test_loader):
    emb = model(x).detach()
    out["embeddings"].extend(
        emb
    )
    out["prediction"].extend(
        model.classifier(emb).detach()
    )
    out["actual"].extend(
        y
    )

out_df = pd.DataFrame.from_dict(out)
out_df.to_csv("beta0.csv")
