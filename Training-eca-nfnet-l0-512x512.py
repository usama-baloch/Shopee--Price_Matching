import pandas as pd

import torch
from torch.utils.data import DataLoader


from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import tqdm

from model import ShopeeModel
from GC import Ranger
from dataset import ShopeeDataSet, get_transform
from scheduler import _LRScheduler
import config


def train_fn(model, data_loader, optimizer, schedular, i):

    model.train()
    fin_loss = 0.0

    tk = tqdm(data_loader, desc = "Epoch" + " [TRAIN] " + str(i+1))

    for t, data in enumerate(tk):
        for k, v in data.items():
            data[k] = v.to(config.device)
        
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()

        fin_loss += loss.item()

        tk.set_postfix({"loss": '%.6f' %float(fin_loss / (t+1)),
                        "LR": optimizer.param_groups[0]['lr']})
    
    schedular.step()

    return fin_loss / len(data_loader)


def eval_fn(model, data_loader, i):

    model.eval()
    fin_loss = 0.0

    tk = tqdm(data_loader, desc = "Epoch" + " [VALID] " + str(i+1))

    with torch.no_grad():
        for t, data in enumerate(tk):
            for k, v in data.items():
                data[k] = v.to(config.device)

            _, loss = model(**data) 
            fin_loss += loss.item()
            tk.set_postfix({"loss": '%.6f' %float(fin_loss / (t+1))})
    
        return fin_loss / len(data_loader)
    

def run_training():

    train_data = pd.read_csv(config.train_csv_path)

    labelencoder = LabelEncoder().fit(train_data['label_group'])
    train_data['label_group'] = labelencoder.transform(train_data['label_group'])

    train_df = ShopeeDataSet(train_data, get_transform())

    train_data_loader = DataLoader(train_df,
                                    batch_size=config.BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=config.num_workers,
                                    pin_memory=True,
                                    drop_last=True
                            )
    
    shopee_model = ShopeeModel.to(config.device)
    
    optimizer = Ranger(shopee_model.parameters(), lr = config.SCHEDULER_PARAMS['lr_start'])

    schedular = _LRScheduler(optimizer=optimizer, **config.SCHEDULER_PARAMS)

    for i in range(config.Epochs):

        avg_loss_train = train_fn(shopee_model, train_data_loader, optimizer, schedular, i)

        # to save the model
        torch.save(shopee_model.state_dict(), 'eca_nfnet_l0_512x512_(Ranger).pt')


# run this to start training
run_training()