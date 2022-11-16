import torch
from torch import nn, utils
from torch.nn import functional as F
import pytorch_lightning as pl

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

import nltk
from nltk import sent_tokenize
nltk.download('punkt')

SENT_MODEL = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2')

class Time2VecPos(nn.Module):
    
    def __init__(self, embedding_dim = 768, fxn = torch.sin):
        super().__init__()
        self.fxn = fxn
        self.weights = nn.Parameter(torch.rand(1, embedding_dim))
        self.bias = nn.Parameter(torch.rand(1, embedding_dim))
    
    def forward(self, input):
        x = torch.matmul(input, self.weights) + self.bias
        x[1:] = self.fxn(x[1:])
        return x

class InitTriplet(nn.Module):
    
    def __init__(self, embed_dim = 768):
        super().__init__()
        self.var_encoder = nn.Linear(1, embed_dim)
        self.time_encoder = nn.Linear(1, embed_dim)
        self.val_encoder = nn.Linear(1, embed_dim)
    
    def forward(self, var, time, val):
        return self.var_encoder(var) + self.time_encoder(time) + self.val_encoder(val)

class TransformerEncoderUnit(nn.Module):

    def __init__(self, embedding_dim = 768 ,hidden_dim = 3072, nheads = 4):
        super().__init__()
        self.mHeadAttention = nn.MultiheadAttention(embedding_dim, 
                                                    nheads, 
                                                    dropout = 0.2)
        
        self.layerNorm1 = nn.LayerNorm(embedding_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p = 0.2),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU()
        )

        self.layerNorm2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        attn, attnWeights = self.mHeadAttention(x, x, x)
        x = self.layerNorm1(attn + x)
        xOut = self.ffn(x)
        x = self.layerNorm2(x + xOut)
        return x, attnWeights

class AttFusion(nn.Module):

    def __init__(self, ffn_dims, embed_dims = 768):
        super().__init__()
        ffn_list = [nn.Linear(embed_dims, ffn_dims[0])]
        for i in range(len(ffn_dims) - 1):
            ffn_list.append(nn.Linear(ffn_dims[i], ffn_dims[i+1]))
            ffn_list.append(nn.Dropout())
        ffn_list.append(nn.Tanh())
        self.ffn = nn.ModuleList(ffn_list)
        self.u = nn.Parameter(torch.rand(size = (ffn_dims[-1], 1)))
        self.smax = nn.Softmax(dim = 0)
        
    def forward(self, x):
        att = x
        for layer in self.ffn:
            att = layer(att)
        att = torch.matmul(att, self.u)
        att = self.smax(att)
        x = torch.sum(x * att, dim = 0)
        return x

class TransformerEncoderCTE(nn.Module):
    
    def __init__(self, num_layers = 4):
        super().__init__() 
        self.cteModule = InitTriplet()
        self.transformerModules = nn.ModuleList(
            modules=[TransformerEncoderUnit() for i in range(num_layers)]
        )
        self.attFusion = AttFusion((512, 120))

    def forward(self, var, time, val):
        emb = self.cteModule(var, time, val)
        for layer in self.transformerModules:
            emb, wts = layer(emb)
        return self.attFusion(emb)    

class NoteEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sentence_model = SENT_MODEL
    
    def forward(self, x):
        return self.sentence_model.encode(x)

class TimeObsEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.time2vec = Time2VecPos()
    
    def forward(self, x, time):
        emb = self.encoder(x)
        timeEmb = self.time2vec(time)
        return emb + timeEmb
    
class TransformerEncoder(nn.Module):
    def __init__(self, encoder, num_layers = 4):
        super().__init__()
        self.encoder = encoder()
        self.transformerModules = nn.ModuleList(
            modules=[TransformerEncoderUnit() for i in range(num_layers)]
        )
        self.attFusion = AttFusion((512, 120))

    def forward(self, val, time):
        emb = self.encoder(val, time)
        for layer in self.transformerModules:
            emb, wts = layer(emb)
        return self.attFusion(emb)     

class LabQuad(nn.Module):
    
    def __init__(self, embed_dim = 768):
        super().__init__()
        self.spec_encoder = nn.Linear(1, embed_dim)
        self.time_encoder = nn.Linear(1, embed_dim)
        self.org_encoder = nn.Linear(1, embed_dim)
        self.val_encoder = nn.Linear(1, embed_dim)

    def forward(self, spec, time, org, val):
        return self.spec_encoder(spec) + self.time_encoder(time) + self.org_encoder(org) + self.val_encoder(val) 

class TransformerEncoderQuad(nn.Module):
    
    def __init__(self, num_layers = 4):
        super().__init__() 
        self.labQuad = LabQuad()
        self.transformerModules = nn.ModuleList(
            modules=[TransformerEncoderUnit() for i in range(num_layers)]
        )
        self.attFusion = AttFusion((512, 120))

    def forward(self, spec, time, org, val):
        emb = self.labQuad(spec, time, org, val)
        for layer in self.transformerModules:
            emb, wts = layer(emb)
        return self.attFusion(emb)    

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LazyLinear(400),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LazyLinear(200),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LazyLinear(50),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LazyLinear(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)