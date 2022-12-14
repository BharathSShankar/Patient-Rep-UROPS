import pickle
import random
from torch.nn.utils.rnn import pad_sequence
from ray import tune
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from patient_Dataset import *
from fullModel import *
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler

PATIENT_LIST = '/hpctmp/e0550582/UROPS_Proj/data/dataByPatient/patientList.pkl'


def train_model(config, data_dir = PATIENT_LIST, num_gpus = 1):
    dev = torch.device("cuda:0")
    with open(data_dir, 'rb') as pickle_file:
            patientList = pickle.load(pickle_file) 

    random.Random(42).shuffle(patientList)
    
    model = FullModel(config)

    train_list = patientList[:40000]
    val_list = patientList[40000:44000]
    
    train_dataset = PatientDataset(train_list)
    val_dataset = PatientDataset(val_list)

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=my_collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=my_collate
    )

    callbacks = [TuneReportCheckpointCallback(
        metrics={
            "loss": "ptl/val_loss",
            "mean_accuracy": "ptl/val_accuracy"
        },
        filename="checkpoint",
        on="validation_end")]

    trainer = pl.Trainer(
            max_epochs=config["num_epochs"],
            callbacks=callbacks,
            accelerator="gpu"
        )
    
    trainer.fit(model, train_loader, val_loader)

config = {
    "num_layers": tune.choice([4]),
    "embedDim": tune.choice([512]),
    "finalDim":tune.choice([120]),
    "wd":tune.choice([1e-5]),
    "beta": tune.choice([20]), 
    "lr": tune.choice([1e-5]),
    "batch_size": tune.choice([16]),
    "num_epochs" : tune.choice([10])
}

trainable = tune.with_parameters(
    train_model,
    data_dir=PATIENT_LIST,
    num_gpus=1)

asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',
    max_t=10,
    grace_period=4,
    reduction_factor=2,
    brackets=1)

analysis = tune.run(
    trainable,
    resources_per_trial={
        "cpu": 1,
        "gpu": 1
    },
    metric="loss",
    mode="min",
    scheduler=asha_scheduler,
    local_dir = "~/ray_results/checkpoint",
    checkpoint_freq = 1,
    config=config,
    name="tune_MIMIC")

print(analysis.best_config)