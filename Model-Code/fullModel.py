from subModules import *
import pytorch_lightning as pl
import torch.nn.functional as F

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
            encoder = NoteEncoder(), num_layers=self.num_layers, embed_dim=self.embedDim
        )
        self.cptEncoder = TransformerEncoder(
            nn.LazyLinear(self.embedDim), num_layers=self.num_layers, embed_dim=self.embedDim
        )
        self.demEncoder = nn.LazyLinear(self.embedDim)

        self.condensor = nn.LazyLinear(self.finalDim)

        self.class_accuracy = pl.metrics.Accuracy()

        self.discriminator = MLP()
        self.classifier = MLP()

    def foward(self, patDat):

        inpDat = patDat["inputs"]
        inpDat = self.inputEncoder(inpDat["var"], inpDat["time"], inpDat["val"])

        outDat = patDat["outputs"]
        outDat = self.outputEncoder(outDat["var"], outDat["time"], outDat["val"])

        labDat = patDat["lab"]
        labDat = self.labEncoder(labDat["var"], labDat["time"], labDat["val"])

        noteDat = patDat["note"]
        labDat = self.noteEncoder(noteDat["text"], noteDat["time"])

        cptDat = patDat["cpt"]
        cptDat = self.cptEncoder(cptDat["procedure"], cptDat["time"])

        microDat = patDat["micro"]
        microDat = self.microEncoder(microDat["spec"], microDat["time"], microDat["org"], microDat["inter"])
    
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
    
    def discriminate(self, x):
        if torch.randint(0, 1) == 1:
            return self.discriminator(x), 1
        else:
            return self.discriminator(torch.rand_like(x)), 0

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        rep = self.forward(x)
        class_val = self.classifier(rep)
        disc_val, is_real = self.discriminate(x)
        class_loss = F.binary_cross_entropy(class_val, y)
        disc_loss = F.binary_cross_entropy(disc_val, is_real)
        loss = class_loss + self.beta * disc_loss
        acc = self.accuracy(class_val, y)
        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", acc)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        rep = self.forward(x)
        class_val = self.classifier(rep)
        disc_val, is_real = self.discriminate(x)
        class_loss = F.binary_cross_entropy(class_val, y)
        disc_loss = F.binary_cross_entropy(disc_val, is_real)
        loss = class_loss + self.beta * disc_loss
        acc = self.accuracy(class_val, y)
        return {"val_loss": loss, "val_accuracy": acc}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(
            [x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack(
            [x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)
