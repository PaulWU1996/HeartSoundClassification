import os
import glob
import fnmatch
import pandas as pd
import numpy as np
import librosa

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchaudio.models import Conformer

# parent folder of sound files
DATA_DIR="data"

set_a = pd.read_csv(DATA_DIR+'/set_a.csv')
set_a.head()

nb_classes=set_a.label.unique()

print("Number of training examples=", set_a.shape[0], "  Number of classes=", len(set_a.label.unique()))
print (nb_classes)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, pframe, duration=10, sr=22050, nmels=64):
        super(Dataset, self).__init__()
        
        self.pframe = pframe
        self.duration = duration
        self.sr = sr
        self.nmels = nmels
        self.data = {}
        self._prepare_data()

    def _prepare_data(self):
        for i in range(len(self.pframe)):
            metadata = self.pframe.iloc[i]
            sound_file='data/'+metadata.fname
            print ("load file ",sound_file)
            X, sr = librosa.load( sound_file, sr=self.sr, duration=self.duration) 
            dur = librosa.get_duration(y=X, sr=sr)
            # pad audio file same duration
            if (round(dur) < self.duration):
                print ("fixing audio lenght :", metadata.fname)
                X = librosa.util.fix_length(X, size=self.sr*self.duration)
            # extract log mel spectrogram feature of an audio waveform
            feature = self._extract_feat(X, sr, self.nmels)
            # convert label to index
            label = self._extract_label(metadata.label)
            self.data[i] = (feature, label)            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        (feat, label) = self.data[index]
        return feat, label
    
    def _extract_feat(self,waveform, sample_rate, n_mels=32):
        # extract log mel spectrogram feature of an audio waveform
        S = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=n_mels)
        log_S = librosa.power_to_db(S, ref=np.max)
        return log_S.transpose()

    def _extract_label(self,label,CLASSES=nb_classes):
        # convert label to index
        # Map integer value to text labels
        label_to_int = {k:v for v,k in enumerate(CLASSES)}
        tag = np.zeros(len(CLASSES))
        tag[label_to_int[label]]=1
        return tag

class DataInterface(pl.LightningDataModule):
    def __init__(self, metadatas, split=[0.8,0.2],batch_size=32):
        super(DataInterface, self).__init__()
        self.full_dataset = Dataset(metadatas)
        self.split = split
        self.batch_size = batch_size

    def setup(self, stage=None):
        # split data
        n = len(self.full_dataset)
        n_train = int(self.split[0]*n)
        n_val = n - n_train
        self.train, self.val = torch.utils.data.random_split(self.full_dataset, [n_train, n_val])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size) # no need to change back


class HeartSound(nn.Module):
    def __init__(
            self,
            input_dim=64, 
            num_heads=4, 
            ffn_dim=128, 
            num_layers=2, 
            depthwise_conv_kernel_size=31,
            num_classes=4
        ):
        
        super(HeartSound, self).__init__()
        self.conformer = Conformer(
                                input_dim=input_dim, 
                                num_heads=num_heads, 
                                ffn_dim=ffn_dim, 
                                num_layers=num_layers, 
                                depthwise_conv_kernel_size=depthwise_conv_kernel_size
                            )
        self.classifer = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        length = torch.ones(x.shape[0])*x.shape[1]
        x,_ = self.conformer(x,length.to(x.device))
        x = self.classifer(x)
        return x.mean(dim=1)

class HeartSoundModel(pl.LightningModule):
    def __init__(self,
                input_dim=64, 
                num_heads=4, 
                ffn_dim=128, 
                num_layers=2, 
                depthwise_conv_kernel_size=31,
                num_classes=4,
                lr=1e-4,
                batch_size=32):
        super(HeartSoundModel, self).__init__()
        self.model = HeartSound(input_dim, num_heads, ffn_dim, num_layers, depthwise_conv_kernel_size, num_classes)
        self.lr = lr
        self.batch_size = batch_size
        self.loss = nn.CrossEntropyLoss()

        self.acc_cache = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.acc_cache.append(accuracy_from_logits(y_hat, y))
        return loss
    
    def on_validation_epoch_end(self):
        acc = torch.stack(self.acc_cache).mean()
        self.log('accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.acc_cache = []
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


# metrics callback
def accuracy_from_logits(y_hat, y):
    _, predicted = torch.max(y_hat, 1)

    correct = torch.sum(predicted == torch.argmax(y, 1))
    total = y.size(0)
    return correct/total

class MyCallback(pl.Callback):
    def __init__(self):
        super(MyCallback, self).__init__()
        self.best_acc = 0
        self.patience = 15
        self.counter = 0
    
    def on_validation_epoch_end(self, trainer, pl_module):
        acc = torch.stack(pl_module.acc_cache).mean()
        if acc > self.best_acc:
            self.best_acc = acc
            print('\n New best accuracy: {}'.format(acc))
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            trainer.should_stop = True

# Trainer
model = HeartSoundModel()
data = DataInterface(set_a)
myCallback = MyCallback()

# Create the trainer
trainer = pl.Trainer(
        accelerator='auto',
        precision=64,
        max_epochs=200,
        check_val_every_n_epoch=1,
        callbacks=[myCallback],
        fast_dev_run=False,
        
    )

trainer.fit(model, data)




# data.setup()
# feat, label = data.train_dataloader().__iter__().__next__()
# model = HeartSound()
# output = model(feat)


# print(output.shape)

