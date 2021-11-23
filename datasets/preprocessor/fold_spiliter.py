import os
import json
from sklearn.model_selection import KFold

def split(data, n_fold, data_path, name):
    kfold = KFold(n_splits=n_fold)
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(data)): 
        train = []
        val = []
        for train_id in train_ids:
            train.append(data[train_id])
        for valid_id in valid_ids:
            val.append(data[valid_id])
        
        try:
            os.mkdir(f"{data_path}/{fold}")
        except FileExistsError:
            pass

        with open(f"{data_path}/{fold}/{name}_train.json", 'w', encoding='utf-8') as f:
            json.dump(train, f, indent=6)
        with open(f"{data_path}/{fold}/{name}_dev.json", 'w', encoding='utf-8') as f:
            json.dump(val, f, indent=6)


if __name__=='__main__':
    data = []
    train_file = './datasets/ESL/ESL_intra_train.json'
    with open(train_file, 'r') as f:
        train = json.load(f)
        data.extend(train)
    
    dev_file = './datasets/ESL/ESL_intra_dev.json'
    with open(dev_file, 'r') as f:
        dev = json.load(f)
        data.extend(dev)
    split(data, 5, './datasets/ESL', 'ESL_intra')

