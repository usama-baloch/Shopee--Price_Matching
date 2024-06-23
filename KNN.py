import numpy as np
import cuml
from cuml.neighbors import NearestNeighbors
from tqdm import tqdm
import gc

def get_image_predictions(test_df, embeddings, threshold=0.0):
    
    if len(test_df) > 3:
        KNN = 50
    else:
        KNN = 3
    
    knn_model = NearestNeighbors(n_neighbors = KNN, metric = 'cosine')
    
    knn_model.fit(embeddings)
    distances, indices = knn_model.kneighbors(embeddings)
    
    predictions = []
    for k in tqdm(range(embeddings.shape[0])):
        idx = np.where(distances[k,] < threshold)[0]
        ids = indices[k, idx]
        posting_ids = test_df['posting_id'].iloc[ids].values
        predictions.append(posting_ids)
    
    del knn_model, distances, indices
    gc.collect()
    
    return predictions

