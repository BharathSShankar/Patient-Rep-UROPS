import pickle
import random

from ray import tune
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from patient_Dataset import *
from fullModel import *
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

PATIENT_LIST = '../data/dataByPatient/patientList.pkl'


def train_model(config, data_dir = PATIENT_LIST):

    with open(data_dir, 'rb') as pickle_file:
            patientList = pickle.load(pickle_file) 

    random.shuffle(patientList)
    
    model = FullModel(config)

    train_list = patientList[:40000]
    val_list = patientList[40000:]
    
    train_dataset = PatientDataset(train_list)
    val_dataset = PatientDataset(val_list)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    callbacks = [TuneReportCallback(metrics, on="validation_end"),
        EarlyStopping(monitor="val_accuracy", min_delta=1e-5, patience=3, verbose=False, mode="max")]

    trainer = pl.Trainer(
            progress_bar_refresh_rate=0,
            max_epochs=config["num_epochs"],
            callbacks=callbacks
        )
    
    trainer.fit(model, train_loader, val_loader)

config = {
    "num_layers": tune.choice([1, 4, 8]),
    "embedDim": tune.choice([512, 768]),
    "finalDim":tune.choice([120, 240]),
    "wd":tune.choice([1e-5, 1e-3, 1e-1]),
    "beta": tune.loguniform(1e-6, 1e2), 
    "lr": tune.loguniform(1e-6, 1e-1),
    "batch_size": tune.choice([16, 32, 64]),
    "num_epochs" : tune.choice([50, 100, 200])
}

trainable = tune.with_parameters(
    train_model,
    data_dir=PATIENT_LIST,
    num_gpus=1)

analysis = tune.run(
    trainable,
    resources_per_trial={
        "cpu": 1,
        "gpu": 1
    },
    metric="loss",
    mode="min",
    config=config,
    name="tune_MIMIC")

print(analysis.best_config)