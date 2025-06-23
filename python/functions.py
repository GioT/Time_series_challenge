import pandas as pd
import numpy as np
import torch

## create sliding windows:
def create_dataset(X_,y_, lookback,v=False):
    """Transform a time series into a prediction dataset
    params df X_: the raw feature dataset
    params df y_: the labels
    params int lookback: Size of window for prediction
    return torch.tensor x,y: the datasets in windows
    """
    X, y = [], []
    for i in range(X_.shape[0]-lookback):
        if v:
            print('X:',i,'-',i+lookback,'; y:',i+lookback)
        feature = X_.iloc[i:i+lookback]
        target  = y_.iloc[i:i+lookback]
        X.append(feature.to_numpy(dtype='float32'))
        y.append(target.to_numpy(dtype='float32'))
    return torch.tensor(X), torch.tensor(y)