import pandas as pd
import numpy as np
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

import gc
import cudf 
import cuml
import cupy
from cuml.feature_extraction.text import TfidfVectorizer


import config
from dataset import ShopeeDataSet, get_test_transform
from model import ShopeeModel_TEST
from KNN import get_image_predictions

def read_dataset():
    test_data = pd.read_csv(config.test_csv_path)
    cudf_test_data = cudf.DataFrame(test_data)
    images_path = config.test_images_path + test_data['image']

    return test_data, cudf_test_data, images_path

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_image_embeddings(image_paths, model_name = config.model_name):
    
    embeddings = []
    
    model = ShopeeModel_TEST(model_name=model_name)
    model.eval()
    
    model.load_state_dict(torch.load(config.model_path))
    model.to(config.device)
    
    test_data = ShopeeDataSet(image_paths, transform = get_test_transform())
    test_data_loader = DataLoader(
                                 test_data,
                                 batch_size = config.BATCH_SIZE_VALID,
                                 pin_memory = True,
                                 drop_last = False,
                                 num_workers = 2
                    )
    
    
    with torch.no_grad():
        for img, label in tqdm(test_data_loader):
            img = img.cuda()
            label = label.cuda()
            feature = model(img, label)
            img_emb = feature.detach().cpu().numpy()
            embeddings.append(img_emb)
    
    del model
    image_embeddings = np.concatenate(embeddings)
    print(f'Our Image Embadding is : {image_embeddings.shape}')
    del embeddings
    gc.collect()
    
    return image_embeddings


def get_text_predictions(df, df_cu, max_features = 25_000):
    
    model = TfidfVectorizer(stop_words = 'english', binary = True, max_features = max_features)
    text_embeddings = model.fit_transform(df_cu['title']).toarray()
    preds = []
    CHUNK = 1024*4

    print('Finding similar titles...')
    CTS = len(df)//CHUNK
    if len(df)%CHUNK!=0: CTS += 1
    for j in range( CTS ):

        a = j*CHUNK
        b = (j+1)*CHUNK
        b = min(b,len(df))
        print('chunk',a,'to',b)

        # COSINE SIMILARITY DISTANCE
        cts = cupy.matmul( text_embeddings, text_embeddings[a:b].T).T

        for k in range(b-a):
            IDX = cupy.where(cts[k,]>0.75)[0]
            o = df.iloc[cupy.asnumpy(IDX)].posting_id.values
            preds.append(o)
    
    del model,text_embeddings
    gc.collect()
    return preds

def combine_predictions(row):
    x = np.concatenate([row['image_predictions'], row['text_predictions']])
    return ' '.join( np.unique(x))



# running code from here, you can make it better by using shell script and just running
# it from terminal and passing argument from there.

def main():

    seed_torch(config.seed)
    df, df_cu, image_paths = read_dataset()

    image_embeddings = get_image_embeddings(image_paths.values)
    image_predictions = get_image_predictions(df, image_embeddings, threshold = 0.36)
    text_predictions = get_text_predictions(df, df_cu, max_features = 25_000)
    
    df['image_predictions'] = image_predictions
    df['text_predictions'] = text_predictions
    df['matches'] = df.apply(combine_predictions, axis = 1)
    df[['posting_id', 'matches']].to_csv('results.csv', index = False)


if __name__ ==  '__main__': 
    main()