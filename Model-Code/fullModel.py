from subModules import *
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import Accuracy

class FullModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.beta = config["beta"]
        self.lr = config["lr"]
        self.num_layers = config["num_layers"]
        self.embedDim = config["embedDim"]
        self.finalDim = config["finalDim"]
        self.wd = config["wd"]

        self.inputEncoder = TransformerEncoderCTE(
            num_layers=self.num_layers, embed_dim=self.embedDim
        )
        self.outputEncoder = TransformerEncoderCTE(
            num_layers=self.num_layers, embed_dim=self.embedDim
        )
        self.labEncoder = TransformerEncoderCTE(
            num_layers=self.num_layers, embed_dim=self.embedDim
        )
        self.microEncoder = TransformerEncoderQuad(
            num_layers=self.num_layers, embed_dim=self.embedDim
        )
        self.noteEncoder = TransformerEncoder(
            encoder = TimeObsEncoder(self.embedDim), num_layers=self.num_layers, embed_dim=self.embedDim
        )
        self.cptEncoder = TransformerEncoder(
            TimeObsEncoder(self.embedDim), num_layers=self.num_layers, embed_dim=self.embedDim
        )
        self.demEncoder = nn.LazyLinear(self.embedDim)

        self.condensor = nn.LazyLinear(self.finalDim)

        self.class_accuracy = Accuracy()

        self.discriminator = MLP()
        self.classifier = MLP()

    def forward(self, patDat):

        inpDat = patDat["inputs"]
        inpDat = self.inputEncoder(inpDat["var"], inpDat["time"], inpDat["val"])

        outDat = patDat["outputs"]
        outDat = self.outputEncoder(outDat["var"], outDat["time"], outDat["val"])

        labDat = patDat["lab"]
        labDat = self.labEncoder(labDat["var"], labDat["time"], labDat["val"])

        noteDat = patDat["note"]
        noteDat = self.noteEncoder(noteDat["text"], noteDat["time"])

        cptDat = patDat["cpt"]
        cptDat = self.cptEncoder(cptDat["procedure"], cptDat["time"])

        microDat = patDat["micro"]
        microDat = self.microEncoder(microDat["spec_data"], microDat["time"], microDat["org_data"], microDat["inter"])
    
        demDat = patDat["dem"]
        demDat = self.demEncoder(demDat)

        rep = torch.cat((demDat, microDat, cptDat, noteDat, labDat, outDat, inpDat), dim = 1)
        return self.condensor(rep)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr = self.lr,
            weight_decay=self.wd
        )
    
    def discriminate(self, val):
        disc_val = self.discriminator(val).to(torch.float)
        fake_val = self.discriminator(torch.rand_like(val)).to(torch.float)
        if torch.randint_like(torch.Tensor(1), 0, 1) == 1:
            return disc_val, torch.ones_like(disc_val).to(torch.float)
        else:
            return fake_val, torch.zeros_like(fake_val).to(torch.float)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        rep = self.forward(x)
        class_val = self.classifier(rep)
        disc_val, is_real = self.discriminate(rep)
        class_loss = F.binary_cross_entropy(class_val, y)
        disc_loss = F.binary_cross_entropy(disc_val, is_real)
        loss = class_loss + self.beta * disc_loss
        acc = self.class_accuracy(class_val, y.to(torch.int))
        self.log("ptl/train_loss", loss.detach().cpu(), on_epoch=False, on_step= True)
        self.log("ptl/train_accuracy", acc.detach().cpu(), on_epoch=False, on_step=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        rep = self.forward(x)
        class_val = self.classifier(rep)
        disc_val, is_real = self.discriminate(rep)
        class_loss = F.binary_cross_entropy(class_val, y)
        disc_loss = F.binary_cross_entropy(disc_val, is_real)
        loss = class_loss + self.beta * disc_loss
        acc = self.class_accuracy(class_val, y.to(torch.int))
        return {"val_loss": loss.detach().cpu(), "val_accuracy": acc.detach().cpu()}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(
            [x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack(
            [x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)
