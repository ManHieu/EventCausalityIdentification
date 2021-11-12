from abc import ABC
import json
import random
from datasets.preprocessor.data_reader import cat_xml_reader, tbd_tml_reader, tdd_tml_reader, tml_reader, tsvx_reader
from datasets.preprocessor.datapoint_formats import get_datapoint
random.seed(1741)
import tqdm
import os
from itertools import combinations
from sklearn.model_selection import train_test_split


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
        processed_corpus = []
        for my_dict in tqdm.tqdm(corpus):
            processed_corpus.extend(get_datapoint(self.type_datapoint, my_dict))
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(processed_corpus, f, indent=6)
        
        return processed_corpus


if __name__ == '__main__':

    dataset = 'ESL'

    if dataset == 'ESL':
        processor = Proprocessor(dataset, 'intra_ir_datapoint', intra=True, inter=False)
        corpus_dir = './datasets/ESL/annotated_data/v0.9/'
        corpus = processor.load_dataset(corpus_dir)
        train, test, validate = [], [], []
        for my_dict in corpus:
            if '37/' in my_dict['doc_id'] or '41/' in my_dict['doc_id']:
                test.append(my_dict)
            else:
                train.append(my_dict)
        train, validate = train_test_split(train, test_size=0.1, train_size=0.9)
    
        processed_path = "./datasets/ESL/ESL_intra_train.json"
        processed_train = processor.process_and_save(processed_path, train)

        processed_path = "./datasets/ESL/ESL_intra_dev.json"
        processed_validate = processor.process_and_save(processed_path, validate)
        
        processed_path = "./datasets/ESL/ESL_intra_test.json"
        processed_test = processor.process_and_save(processed_path, test)
    
    if dataset == 'MATRES': 
        pass
    
    print("Number datapoints in dataset: {}".format(len(processed_train + processed_validate + processed_test)))
    print("Number training points: {}".format(len(processed_train)))
    print("Number validate points: {}".format(len(processed_validate)))
    print("Number test points: {}".format(len(processed_test)))

