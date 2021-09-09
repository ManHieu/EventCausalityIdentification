import json
import tqdm
import os
from itertools import combinations
from datasets.data_reader import i2b2_xml_reader, tbd_tml_reader, tdd_tml_reader, tml_reader, tsvx_reader


class Reader(object):
    def __init__(self, type) -> None:
        super().__init__()
        self.type = type
    
    def read(self, dir_name, file_name):
        if self.type == 'tsvx':
            return tsvx_reader(dir_name, file_name)
        elif self.type == 'tml':
            return tml_reader(dir_name, file_name)
        elif self.type == 'i2b2_xml':
            return i2b2_xml_reader(dir_name, file_name)
        elif self.type == 'tbd_tml':
            return tbd_tml_reader(dir_name, file_name)
        elif self.type == 'tdd_man':
            return tdd_tml_reader(dir_name, file_name, type_doc='man')
        elif self.type == 'tdd_auto':
            return tdd_tml_reader(dir_name, file_name, type_doc='auto')
        else:
            raise ValueError("We have not supported {} type yet!".format(self.type))

def load_dataset(dir_name, type):
    reader = Reader(type)
    onlyfiles = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
    corpus = []
    # i = 0
    for file_name in tqdm.tqdm(onlyfiles):
        # if i == 1:
        #     break
        # i = i + 1
        if type == 'i2b2_xml':
            if file_name.endswith('.xml'):
                my_dict = reader.read(dir_name, file_name)
                if my_dict != None:
                    corpus.append(my_dict)
        else:
            my_dict = reader.read(dir_name, file_name)
            if my_dict != None:
                corpus.append(my_dict)
    return corpus

def get_data_point(my_dict):
    pair_sents = set(combinations(range(len(my_dict['sentences'])), 2))
    data_points = []
    for pair in pair_sents:
        sent_1_id, sent_2_id = pair
        assert sent_1_id < sent_2_id
        sent_1 = my_dict['sentences'][sent_1_id]['tokens']
        sent_2 = my_dict['sentences'][sent_2_id]['tokens']
        
        data_point = {}
        data_point['tokens'] = sent_1 + sent_2

        triggers = []
        for eid, event in my_dict['event_dict'].items():
            if event['sent_id'] == sent_1_id:
                trigger = {
                    'eid': eid,
                    'type': event['class'],
                    'start': event['token_id_list'][0],
                    'end': event['token_id_list'][-1] + 1,
                    'mention': " ".join(data_point['tokens'][event['token_id_list'][0]: event['token_id_list'][-1] + 1]),
                }
                try:
                    assert event['mention'] in trigger['mention'], "{} - {}".format(event['mention'], data_point['tokens'])
                except:
                    print("{} - {}".format(event['mention'], data_point['tokens']))
                triggers.append(trigger)
            if event['sent_id'] == sent_2_id:
                trigger = {
                    'eid': eid,
                    'type': event['class'],
                    'start': event['token_id_list'][0] + len(sent_1),
                    'end': event['token_id_list'][-1] + 1 + len(sent_1),
                    'mention': " ".join(data_point['tokens'][event['token_id_list'][0] + len(sent_1): event['token_id_list'][-1] + 1 + len(sent_1)]),
                }
                try:
                    assert event['mention'] in trigger['mention'], "{} - {}".format(event['mention'], data_point['tokens'])
                except:
                    print("{} - {}".format(event['mention'], data_point['tokens']))
                triggers.append(trigger)
            
        data_point['triggers'] = triggers

        relations = []
        for pair, r_type in my_dict['relation_dict'].items():
            eid1, eid2 = pair
            id = [None, None]
            for i, trigger in enumerate(triggers):
                if trigger['eid'] == eid1:
                    id[0] = i
                if trigger['eid'] == eid2:
                    id[1] = i
            if None not in id:
                relation = {
                    'type': r_type,
                    'head': id[0],
                    'tail': id[1]
                }
                relations.append(relation)
        data_point['relations'] = relations
        data_points.append(data_point)
    return data_points

    
def loader(dataset):
    if dataset == "MATRES":
        print("MATRES Loading .......")
        aquaint_dir_name = "./datasets/MATRES/TBAQ-cleaned/AQUAINT/"
        timebank_dir_name = "./datasets/MATRES/TBAQ-cleaned/TimeBank/"
        platinum_dir_name = "./datasets/MATRES/te3-platinum/"
        validate = load_dataset(aquaint_dir_name, 'tml')
        train = load_dataset(timebank_dir_name, 'tml')
        test = load_dataset(platinum_dir_name, 'tml')

        processed_validate = []
        for my_dict in validate:
            processed_validate.extend(get_data_point(my_dict))
        validate_processed_path = "./datasets/MATRES/MATRES_dev.json"
        with open(validate_processed_path, 'w', encoding='utf-8') as f:
            json.dump(processed_validate, f, indent=6)
        
        processed_train = []
        for my_dict in train:
            processed_train.extend(get_data_point(my_dict))
        train_processed_path = "./datasets/MATRES/MATRES_train.json"
        with open(train_processed_path, 'w', encoding='utf-8') as f:
            json.dump(processed_train, f, indent=6)
        
        processed_test = []
        for my_dict in test:
            processed_test.extend(get_data_point(my_dict))
        test_processed_path = "./datasets/MATRES/MATRES_test.json"
        with open(test_processed_path, 'w', encoding='utf-8') as f:
            json.dump(processed_test, f, indent=6)
        

if __name__ == "__main__":
    loader("MATRES")
