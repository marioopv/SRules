
import numpy as np

def new_datasets(X_train, y_train, y_pred_train_rules, dataset):
    # Get indexes
    filter_indices = np.where(np.array(y_pred_train_rules) == None)[0]

    # new X_train
    np_filtered_X_train = np.array(X_train)[filter_indices]
    X_train = np_filtered_X_train
    np_filtered_y_train = np.array(y_train)[filter_indices]
    y_train = np_filtered_y_train

    # new Pandas dataset
    df = dataset.filter(items=filter_indices, axis=0)
    dataset = df

    new_len = len(X_train)
    return X_train, y_train, dataset, new_len