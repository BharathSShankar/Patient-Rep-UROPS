from subModules import *
import pytorch_lightning as pl

class FullModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
