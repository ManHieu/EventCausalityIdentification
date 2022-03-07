import json
import re
from scipy.sparse import data
import random
random.seed(1741)
import tqdm
import os
from itertools import combinations
import networkx as nx
from trankit import Pipeline
from preprocessor_utils import covert_to_doc_id, expand_entity, find_common_lowest_ancestor, id_mapping, remove_special_token
from sklearn.model_selection import train_test_split
from data_reader import i2b2_xml_reader, tbd_tml_reader, tdd_tml_reader, tml_reader, tsvx_reader, cat_xml_reader

p = Pipeline('english', gpu=True)
DATAPOINT = {}


def register_datapoint(func):
    DATAPOINT[str(func.__name__)] = func
    return func


def get_datapoint(type, mydict):
    return DATAPOINT[type](mydict)


@register_datapoint
def intra_ir_datapoint(my_dict):
    """
    Read data for intra identify relation:
    Read all sentence in the doc and relation of trigger pairs in each sentence ([] if no relation) 
    """
    data_points = []

    for sid, sentence in enumerate(my_dict['sentences']):
        _data_points = []

        # parse sentence 
        dep_tree = nx.DiGraph()
        doc_tokens_pasered = []
        dep = p.posdep(sentence['content'])
        doc_tokens_pasered = covert_to_doc_id(dep, sentence['d_span'])
        for token in doc_tokens_pasered:
            dep_tree.add_edge(token['head'], token['id'])
        
        prunted_dep_tree = nx.DiGraph()
        for token in doc_tokens_pasered:
            if token['deprel'].split(':')[0].strip() in ['nsubj', 'obj', 'compound']:
                prunted_dep_tree.add_edge(token['head'], token['id'])

        # reading triggers 
        triggers = []
        for eid, event in my_dict['event_dict'].items():
            if event['sent_id'] == sid:
                trigger = {
                        'eid': eid,
                        'type': event['class'],
                        'start': event['token_id_list'][0],
                        'end': event['token_id_list'][-1] + 1,
                        'mention': " ".join(sentence['tokens'][event['token_id_list'][0]: event['token_id_list'][-1] + 1]),
                        'dspan': event['d_span']
                    }
                try:
                    assert event['mention'] in trigger['mention'], "{} - {}".format(event['mention'], sentence['tokens'])
                except:
                    print("{} - {}".format(event['mention'], sentence['tokens']))
                triggers.append(trigger)
        
        event_pairs = combinations(triggers, 2)
        for ev1, ev2 in event_pairs:
            e1_ids, e2_ids = id_mapping(ev1['dspan'], doc_tokens_pasered, my_dict['doc_content']), id_mapping(ev2['dspan'], doc_tokens_pasered, my_dict['doc_content'])
            
            # get the anscestor of all token as as proxy
            if len(e1_ids) > 1:
                e1_id = find_common_lowest_ancestor(dep_tree, e1_ids)
            else:
                e1_id = e1_ids[0]
            if len(e2_ids) > 1:
                e2_id = find_common_lowest_ancestor(dep_tree, e2_ids)
            else:
                e2_id = e2_ids[0]
            
            # expand trigger
            ev1['mention'] = expand_entity(e1_id, prunted_dep_tree, doc_tokens_pasered)
            ev2['mention'] = expand_entity(e2_id, prunted_dep_tree, doc_tokens_pasered)
            
            # get dependency path
            lowest_ancestor = nx.lowest_common_ancestor(dep_tree, e1_id, e2_id)
            path1 = nx.shortest_path(dep_tree, lowest_ancestor, e1_id)
            path2 = nx.shortest_path(dep_tree, lowest_ancestor, e2_id)
            
            dep_path = []
            if len(path1) > 1:
                dep_path1 = []
                for node in path1:
                    enitity = expand_entity(node, prunted_dep_tree, doc_tokens_pasered, in_path=True)
                    dep_path1.append(enitity)
                dep_path.append(dep_path1)
            
            if len(path2) > 1:
                dep_path2 = []
                for node in path2:
                    enitity = expand_entity(node, prunted_dep_tree, doc_tokens_pasered, in_path=True)
                    dep_path2.append(enitity)
                dep_path.append(dep_path2)

            # reading relations
            eid1, eid2 = ev1['eid'], ev2['eid']
            rel = my_dict['relation_dict'].get((eid1, eid2))
            _rel = my_dict['relation_dict'].get((eid2, eid1))
            if (eid1, eid2) in my_dict['relation_dict'].keys() or (eid2, eid1) in my_dict['relation_dict'].keys():
                relation = {'type': rel, 'head': 0, 'tail': 1} if rel != None else {'type': _rel, 'head': 1, 'tail': 0}
            else:
                relation = None
            
            if (relation == None and random.uniform(0, 1) < 0.75) or relation != None:
                _data_points.append({
                    'tokens': sentence['tokens'],
                    'triggers': [ev1, ev2],
                    'relations': [relation] if relation != None else [], 
                    'dep_path': dep_path
                })

        data_points.extend(_data_points)
    
    return data_points