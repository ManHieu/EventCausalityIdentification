from abc import ABC
from collections import defaultdict
import json
import random
from data_reader import cat_xml_reader, ctb_cat_reader, tbd_tml_reader, tdd_tml_reader, tml_reader, tsvx_reader
from datapoint_formats import get_datapoint
random.seed(1741)
import tqdm
import os
from itertools import combinations
from sklearn.model_selection import train_test_split, KFold


class Proprocessor(object):
    def __init__(self, dataset, type_datapoint, intra=True, inter=False) -> None:
        super().__init__()
        self.dataset = dataset
        self.intra = intra
        self.inter = inter
        self.register_reader(self.dataset)
        self.type_datapoint = type_datapoint
    
    def register_reader(self, dataset):
        if self.dataset == 'MATRES':
            self.reader = tml_reader
        elif self.dataset == 'HiEve':
            self.reader = tsvx_reader
        elif self.dataset == 'TBD':
            self.reader = tbd_tml_reader()
        elif dataset in ['TDD_man', 'TDD_auto']:
            self.reader = tdd_tml_reader
        elif dataset == 'ESL':
            self.reader = cat_xml_reader
        elif dataset == 'Causal-TB':
            self.reader = ctb_cat_reader
        else:
            raise ValueError("We have not supported this dataset {} yet!".format(self.dataset))
    
    def load_dataset(self, dir_name):
        corpus = []
        if self.dataset == 'ESL':
            topic_folders = [t for t in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, t))]
            for topic in tqdm.tqdm(topic_folders):
                topic_folder = os.path.join(dir_name, topic)
                onlyfiles = [f for f in os.listdir(topic_folder) if os.path.isfile(os.path.join(topic_folder, f))]
                for file_name in onlyfiles:
                    file_name = os.path.join(topic, file_name)
                    if file_name.endswith('.xml'):
                        my_dict = self.reader(dir_name, file_name, inter=self.inter, intra=self.intra)
                        if my_dict != None:
                            corpus.append(my_dict)
        else:
            onlyfiles = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
            # i = 0
            for file_name in tqdm.tqdm(onlyfiles):
                # if i == 1:
                #     break
                # i = i + 1
                if self.dataset == 'TDD_man':
                    my_dict = self.reader(dir_name, file_name, type_doc='man')
                elif self.dataset == 'TDD_auto':
                    my_dict = self.reader(dir_name, file_name, type_doc='auto')
                else:
                    my_dict = self.reader(dir_name, file_name)
                
                if my_dict != None:
                    corpus.append(my_dict)
        
        return corpus
    
    def process_and_save(self, save_path, corpus):
        if type(corpus) == list:
            processed_corpus = []
            for my_dict in tqdm.tqdm(corpus):
                processed_corpus.extend(get_datapoint(self.type_datapoint, my_dict))
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(processed_corpus, f, indent=6)
        else:
            processed_corpus = defaultdict(list)
            for key, topic in corpus.items():
                for my_dict in tqdm.tqdm(topic):
                    processed_corpus[key].extend(get_datapoint(self.type_datapoint, my_dict))
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(processed_corpus, f, indent=6)

        return processed_corpus


if __name__ == '__main__':

    dataset = 'ESL'

    if dataset == 'ESL':
        kfold = KFold(n_splits=5)
        processor = Proprocessor(dataset, 'intra_ir_datapoint', intra=True, inter=False)
        corpus_dir = './datasets/ESL/annotated_data/v0.9/'
        corpus = processor.load_dataset(corpus_dir)
        
        _train, test = [], []
        data = defaultdict(list)
        for my_dict in corpus:
            topic = my_dict['doc_id'].split('/')[0]
            data[topic].append(my_dict)

            if '37/' in my_dict['doc_id'] or '41/' in my_dict['doc_id']:
                test.append(my_dict)
            else:
                _train.append(my_dict)

        processed_path = f"./datasets/ESL/ESL_intra_data.json"
        processed_data = processor.process_and_save(processed_path, data)

        random.shuffle(_train)
        for fold, (train_ids, valid_ids) in enumerate(kfold.split(_train)):
            try:
                os.mkdir(f"./datasets/ESL/{fold}")
            except FileExistsError:
                pass

            train = [_train[id] for id in train_ids]
            validate = [_train[id] for id in valid_ids]
        
            processed_path = f"./datasets/ESL/{fold}/ESL_intra_train.json"
            processed_train = processor.process_and_save(processed_path, train)

            processed_path = f"./datasets/ESL/{fold}/ESL_intra_dev.json"
            processed_validate = processor.process_and_save(processed_path, validate)
            
            processed_path = f"./datasets/ESL/{fold}/ESL_intra_test.json"
            processed_test = processor.process_and_save(processed_path, test)

            print(f"Statistic in fold {fold}")
            print("Number datapoints in dataset: {}".format(len(processed_train + processed_validate + processed_test)))
            print("Number training points: {}".format(len(processed_train)))
            print("Number validate points: {}".format(len(processed_validate)))
            print("Number test points: {}".format(len(processed_test)))
    
    if dataset == 'Causal-TB':
        processor = Proprocessor(dataset, 'intra_ir_datapoint')
        corpus_dir = './datasets/Causal-TimeBank/Causal-TimeBank-CAT/'
        corpus = processor.load_dataset(corpus_dir)
        
        random.shuffle(corpus)
        kfold = KFold(n_splits=10)

        for fold, (train_ids, valid_ids) in enumerate(kfold.split(corpus)):
            try:
                os.mkdir(f"./datasets/Causal-TimeBank/{fold}")
            except FileExistsError:
                pass

            train = [corpus[id] for id in train_ids]
            validate = [corpus[id] for id in valid_ids]
        
            processed_path = f"./datasets/Causal-TimeBank/{fold}/Causal-TB_intra_train.json"
            processed_train = processor.process_and_save(processed_path, train)

            processed_path = f"./datasets/Causal-TimeBank/{fold}/Causal-TB_intra_dev.json"
            processed_validate = processor.process_and_save(processed_path, validate)
            
            print(f"Statistic in fold {fold}")
            print("Number datapoints in dataset: {}".format(len(processed_train + processed_validate)))
            print("Number training points: {}".format(len(processed_train)))
            print("Number validate points: {}".format(len(processed_validate)))
    
    

