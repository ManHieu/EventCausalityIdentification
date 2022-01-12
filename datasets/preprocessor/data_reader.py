import json
import re
import torch
torch.manual_seed(1741)
import random
random.seed(1741)
import numpy as np
np.random.seed(1741)
import bs4
import xml.etree.ElementTree as ET
from collections import defaultdict
from constants import *
from preprocessor_utils import *
# from nltk import sent_tokenize
from bs4 import BeautifulSoup as Soup
import csv


# =========================
#       HiEve Reader
# =========================
def tsvx_reader(dir_name, file_name):
    my_dict = {}
    my_dict["doc_id"] = file_name.replace(".tsvx", "") # e.g., article-10901.tsvx
    my_dict["event_dict"] = {}
    my_dict["sentences"] = []
    my_dict["relation_dict"] = {}
    
    # Read tsvx file
    for line in open(dir_name + file_name, encoding='UTF-8'):
        line = line.split('\t')
        if line[0] == 'Text':
            my_dict["doc_content"] = line[1]
        elif line[0] == 'Event':
            end_char = int(line[4]) + len(line[2]) - 1
            my_dict["event_dict"][int(line[1])] = {"mention": line[2], "start_char": int(line[4]), "end_char": end_char} 
            # keys to be added later: sent_id & subword_id
        elif line[0] == 'Relation':
            event_id1 = int(line[1])
            event_id2 = int(line[2])
            rel = hieve_label_dict[line[3]]
            my_dict["relation_dict"][(event_id1, event_id2)] = rel
        else:
            raise ValueError("Reading a file not in HiEve tsvx format...")
    
    # Split document into sentences
    sent_tokenized_text = [str(sent) for sent in nlp(my_dict["doc_content"]).sents]
    sent_span = tokenized_to_origin_span(my_dict["doc_content"], sent_tokenized_text)
    count_sent = 0
    for sent in sent_tokenized_text:
        sent_dict = {}
        sent_dict["sent_id"] = count_sent
        sent_dict["content"] = sent
        sent_dict["sent_start_char"] = sent_span[count_sent][0]
        sent_dict["sent_end_char"] = sent_span[count_sent][1]
        count_sent += 1
        spacy_token = nlp(sent_dict["content"])
        sent_dict["tokens"] = []
        sent_dict["pos"] = []
        # spaCy-tokenized tokens & Part-Of-Speech Tagging
        for token in spacy_token:
            sent_dict["tokens"].append(token.text)
            sent_dict["pos"].append(token.pos_)
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent, sent_dict["tokens"])
        sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])

        # RoBERTa tokenizer
        sent_dict["roberta_subword_to_ID"], sent_dict["roberta_subwords"], \
        sent_dict["roberta_subword_span_SENT"], sent_dict["roberta_subword_map"] = \
        RoBERTa_list(sent_dict["content"], sent_dict["tokens"], sent_dict["token_span_SENT"])
            
        sent_dict["roberta_subword_span_DOC"] = \
        span_SENT_to_DOC(sent_dict["roberta_subword_span_SENT"], sent_dict["sent_start_char"])
        
        sent_dict["roberta_subword_pos"] = []
        for token_id in sent_dict["roberta_subword_map"]:
            if token_id == -1 or token_id is None:
                sent_dict["roberta_subword_pos"].append("None")
            else:
                sent_dict["roberta_subword_pos"].append(sent_dict["pos"][token_id])
        
        my_dict["sentences"].append(sent_dict)
    
    # Add sent_id as an attribute of event
    for event_id, event_dict in my_dict["event_dict"].items():
        my_dict["event_dict"][event_id]["sent_id"] = sent_id = sent_id_lookup(my_dict, event_dict["start_char"], event_dict["end_char"])
        my_dict["event_dict"][event_id]["token_id"] = id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"])
        my_dict["event_dict"][event_id]["roberta_subword_id"] = poss = \
        id_lookup(my_dict["sentences"][sent_id]["roberta_subword_span_DOC"], event_dict["start_char"]) + 1 # sentence is added <s> token
        # try:
        mention = event_dict["mention"]
        sub_word_id = my_dict["sentences"][sent_id]["roberta_subword_to_ID"][poss]
        sub_word = tokenizer.decode([sub_word_id])
        assert sub_word.strip() in mention

    return my_dict

# ========================================
#        MATRES: read relation file
# ========================================
# MATRES has separate text files and relation files
# We first read relation files
eiid_to_event_trigger = {}
eiid_pair_to_label = {} 

def matres_reader(matres_file):
    with open(matres_file, 'r', encoding='UTF-8') as f:
        content = f.read().split('\n')
        for rel in content:
            rel = rel.split("\t")
            fname = rel[0]
            trigger1 = rel[1]
            trigger2 = rel[2]
            eiid1 = int(rel[3])
            eiid2 = int(rel[4])
            tempRel = temp_label_map[rel[5]]

            if fname not in eiid_to_event_trigger:
                eiid_to_event_trigger[fname] = {}
                eiid_pair_to_label[fname] = {}
            eiid_pair_to_label[fname][(eiid1, eiid2)] = tempRel
            if eiid1 not in eiid_to_event_trigger[fname].keys():
                eiid_to_event_trigger[fname][eiid1] = trigger1
            if eiid2 not in eiid_to_event_trigger[fname].keys():
                eiid_to_event_trigger[fname][eiid2] = trigger2

matres_reader(MATRES_timebank)
matres_reader(MATRES_aquaint)
matres_reader(MATRES_platinum)

def tml_reader(dir_name, file_name):
    my_dict = {}
    my_dict["event_dict"] = {}
    my_dict["eID_dict"] = {}
    my_dict["doc_id"] = file_name.replace(".tml", "") 
    # e.g., file_name = "ABC19980108.1830.0711.tml"
    # dir_name = '/shared/why16gzl/logic_driven/EMNLP-2020/MATRES/TBAQ-cleaned/TimeBank/'
    try:
        tree = ET.parse(dir_name + file_name)
    except:
        print(dir_name + file_name)
    root = tree.getroot()
    MY_STRING = str(ET.tostring(root))
    # ================================================
    # Load the lines involving event information first
    # ================================================
    for makeinstance in root.findall('MAKEINSTANCE'):
        instance_str = str(ET.tostring(makeinstance)).split(" ")
        try:
            assert instance_str[3].split("=")[0] == "eventID"
            assert instance_str[2].split("=")[0] == "eiid"
            eiid = int(instance_str[2].split("=")[1].replace("\"", "")[2:])
            eID = instance_str[3].split("=")[1].replace("\"", "")
        except:
            for i in instance_str:
                if i.split("=")[0] == "eventID":
                    eID = i.split("=")[1].replace("\"", "")
                if i.split("=")[0] == "eiid":
                    eiid = int(i.split("=")[1].replace("\"", "")[2:])
        # Not all document in the dataset contributes relation pairs in MATRES
        # Not all events in a document constitute relation pairs in MATRES
        if my_dict["doc_id"] in eiid_to_event_trigger.keys():
            my_dict["event_dict"][eiid] = {"eID": eID}
            my_dict["eID_dict"][eID] = {"eiid": eiid}
        
    # ==================================
    #              Load Text
    # ==================================
    start = MY_STRING.find("<TEXT>") + 6
    end = MY_STRING.find("</TEXT>")
    MY_TEXT = MY_STRING[start:end]
    while MY_TEXT[0] == " ":
        MY_TEXT = MY_TEXT[1:]
    MY_TEXT = MY_TEXT.replace("\\n", " ")
    MY_TEXT = MY_TEXT.replace("\\'", "\'")
    MY_TEXT = MY_TEXT.replace("  ", " ")
    MY_TEXT = MY_TEXT.replace(" ...", "...")
    
    # ========================================================
    #    Load position of events, in the meantime replacing 
    #    "<EVENT eid="e1" class="OCCURRENCE">turning</EVENT>"
    #    with "turning"
    # ========================================================
    while MY_TEXT.find("<") != -1:
        start = MY_TEXT.find("<")
        end = MY_TEXT.find(">")
        if MY_TEXT[start + 1] == "E":
            event_description = MY_TEXT[start:end].split(" ")
            end_mention = MY_TEXT[end:].find("<")
            mention = MY_TEXT[end+1: end_mention+end]
            # eID = (event_description[1].split("="))[1].replace("\"", "")
            # print(eID)
            for item in event_description:
                if item.startswith("eid"):
                    eID = (item.split("="))[1].replace("\"", "")
                if item.startswith("class"):
                    e_class = (item.split("="))[1].replace("\"", "")
            MY_TEXT = MY_TEXT[:start] + MY_TEXT[(end + 1):]
            if eID in my_dict["eID_dict"].keys():
                eiid = my_dict['eID_dict'][eID]['eiid']
                if eiid in eiid_to_event_trigger[my_dict["doc_id"]].keys():
                    assert eiid_to_event_trigger[my_dict["doc_id"]][eiid] == mention, "{}-{}".format(eiid_to_event_trigger[my_dict["doc_id"]][eiid], mention)
                my_dict['event_dict'][eiid]['mention'] = mention
                my_dict["event_dict"][eiid]["start_char"] = start # loading position of events
                end_char = start + len(my_dict["event_dict"][eiid]['mention']) - 1
                my_dict['event_dict'][eiid]['end_char'] = end_char
                my_dict['event_dict'][eiid]['class'] = e_class
        else:
            MY_TEXT = MY_TEXT[:start] + MY_TEXT[(end + 1):]

    # MY_TEXT = content
    
    # =====================================
    # Enter the routine for text processing
    # =====================================
    my_dict["doc_content"] = MY_TEXT
    my_dict["sentences"] = []
    my_dict["relation_dict"] = {}
    sent_tokenized_text = [str(sent) for sent in nlp(my_dict["doc_content"]).sents]
    sent_span = tokenized_to_origin_span(my_dict["doc_content"], sent_tokenized_text)
    count_sent = 0
    for sent in sent_tokenized_text:
        sent_dict = {}
        sent_dict["sent_id"] = count_sent
        sent_dict["content"] = sent
        sent_dict["sent_start_char"] = sent_span[count_sent][0]
        sent_dict["sent_end_char"] = sent_span[count_sent][1]
        count_sent += 1
        spacy_token = nlp(sent_dict["content"])
        sent_dict["tokens"] = []
        sent_dict["pos"] = []
        # spaCy-tokenized tokens & Part-Of-Speech Tagging
        for token in spacy_token:
            sent_dict["tokens"].append(token.text)
            sent_dict["pos"].append(token.pos_)
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent, sent_dict["tokens"])
        sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])

        # RoBERTa tokenizer
        sent_dict["roberta_subword_to_ID"], sent_dict["roberta_subwords"], \
        sent_dict["roberta_subword_span_SENT"], sent_dict["roberta_subword_map"] = \
        RoBERTa_list(sent_dict["content"], sent_dict["tokens"], sent_dict["token_span_SENT"])
            
        sent_dict["roberta_subword_span_DOC"] = \
        span_SENT_to_DOC(sent_dict["roberta_subword_span_SENT"], sent_dict["sent_start_char"])
        
        sent_dict["roberta_subword_pos"] = []
        for token_id in sent_dict["roberta_subword_map"]:
            if token_id == -1 or token_id is None:
                sent_dict["roberta_subword_pos"].append("None")
            else:
                sent_dict["roberta_subword_pos"].append(sent_dict["pos"][token_id])
        
        my_dict["sentences"].append(sent_dict)
        
        # Add sent_id as an attribute of event
    wrong_ev = []
    for event_id, event_dict in my_dict["event_dict"].items():
        # print(event_id)
        # print(event_dict)
        # try:
        if len(event_dict) != 5:
            print(event_dict)
            wrong_ev.append(event_id)
            continue
        assert str(my_dict["doc_content"][event_dict["start_char"]:event_dict["end_char"] + 1]).lower() == str(event_dict["mention"]).lower()
        assert str(my_dict["doc_content"][event_dict["start_char"]:event_dict["end_char"] + 1]).strip() != ""
        # except:
        #     print("doc: ", my_dict["doc_content"][event_dict["start_char"]:event_dict["end_char"]])
        #     print("mention: ", event_dict["mention"])
        #     print(my_dict["doc_id"])
        my_dict["event_dict"][event_id]["sent_id"] = sent_id = sent_id_lookup(my_dict, event_dict["start_char"])
        my_dict["event_dict"][event_id]["token_id"] = id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"])
        # sentence is added <s> token
        my_dict["event_dict"][event_id]["roberta_subword_id"] = poss = id_lookup(my_dict["sentences"][sent_id]["roberta_subword_span_DOC"], event_dict["start_char"]) + 1 
        my_dict["event_dict"][event_id]["token_id_list"] = list_id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"], event_dict["end_char"])
        
        mention = event_dict["mention"]
        sub_word_id = my_dict["sentences"][sent_id]["roberta_subword_to_ID"][poss]
        sub_word = tokenizer.decode([sub_word_id])
        assert sub_word.strip() in mention, f'{my_dict["doc_id"]}:{mention}: {sub_word.strip()}'
    
    for k in wrong_ev:
        my_dict["event_dict"].pop(k)

    if eiid_pair_to_label.get(my_dict['doc_id']) == None:
        return None
    relation_dict = eiid_pair_to_label[my_dict['doc_id']]
    my_dict['relation_dict'] = relation_dict

    return my_dict

# =========================
#       I2B2 Reader
# =========================
def i2b2_xml_reader(dir_name, file_name):
    my_dict = {}
    my_dict = {}
    my_dict["event_dict"] = {}
    my_dict["doc_id"] = file_name.replace(".xml", "") 
    
    try:
        tree = ET.parse(dir_name + file_name)
    except:
        print("Can't load this file: {} T_T". format(dir_name + file_name))
        return None
    root = tree.getroot()
    for event_instance in tree.findall('.//EVENT'):
        eid = event_instance.attrib['id']
        mention = event_instance.attrib['text']
        start_char = event_instance.attrib['start']
        end_char = event_instance.attrib['end']
        modality = event_instance.attrib['modality']
        typ = event_instance.attrib['type']

        my_dict['event_dict'][eid] = {}
        my_dict['event_dict'][eid]['mention'] = mention
        my_dict['event_dict'][eid]['start_char'] = int(start_char)
        my_dict['event_dict'][eid]['end_char'] = int(end_char)
        my_dict['event_dict'][eid]['modality'] = modality
        my_dict['event_dict'][eid]['type'] = typ
    
    my_dict["doc_content"] = root.find("TEXT").text
    my_dict["sentences"] = []
    my_dict['relation_dict'] = {}
    sent_tokenized_text = [str(sent) for sent in nlp(my_dict["doc_content"]).sents]
    sent_span = tokenized_to_origin_span(my_dict['doc_content'], sent_tokenized_text)
    count_sent = 0
    for sent in sent_tokenized_text:
        sent_dict = {}
        sent_dict["sent_id"] = count_sent
        sent_dict["content"] = sent
        sent_dict["sent_start_char"] = sent_span[count_sent][0]
        sent_dict["sent_end_char"] = sent_span[count_sent][1]
        count_sent += 1

        spacy_token = nlp(sent_dict['content'])
        sent_dict["tokens"] = []
        sent_dict["pos"] = []
        # spaCy-tokenized tokens & Part-Of-Speech Tagging
        for token in spacy_token:
            sent_dict["tokens"].append(token.text)
            sent_dict["pos"].append(token.pos_)
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent, sent_dict["tokens"])
        sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])

        # RoBERTa tokenizer
        sent_dict["roberta_subword_to_ID"], sent_dict["roberta_subwords"], \
        sent_dict["roberta_subword_span_SENT"], sent_dict["roberta_subword_map"] = \
        RoBERTa_list(sent_dict["content"], sent_dict["tokens"], sent_dict["token_span_SENT"])
            
        sent_dict["roberta_subword_span_DOC"] = \
        span_SENT_to_DOC(sent_dict["roberta_subword_span_SENT"], sent_dict["sent_start_char"])
        
        sent_dict["roberta_subword_pos"] = []
        for token_id in sent_dict["roberta_subword_map"]:
            if token_id == -1 or token_id is None:
                sent_dict["roberta_subword_pos"].append("None")
            else:
                sent_dict["roberta_subword_pos"].append(sent_dict["pos"][token_id])
        
        my_dict["sentences"].append(sent_dict)

    # Add sent_id as an attribute of event
    for event_id, event_dict in my_dict["event_dict"].items():
        my_dict["event_dict"][event_id]["sent_id"] = sent_id = sent_id_lookup(my_dict, event_dict["start_char"])
        my_dict["event_dict"][event_id]["token_id"] = id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"])
        my_dict["event_dict"][event_id]["roberta_subword_id"] = poss = \
        id_lookup(my_dict["sentences"][sent_id]["roberta_subword_span_DOC"], event_dict["start_char"]) + 1 # sentence is added <s> token
        # try:
        mention = event_dict["mention"]
        sub_word_id = my_dict["sentences"][sent_id]["roberta_subword_to_ID"][poss]
        sub_word = tokenizer.decode([sub_word_id])
        # try:
        #     assert sub_word.strip() in mention
        # except:
        #     print(my_dict["doc_id"])
        #     print("subword: '{}', mention: '{}'".format(sub_word, mention))
    
    for rel_instance in tree.findall('.//TLINK'):
        rid = rel_instance.attrib['id']
        e1_id = rel_instance.attrib['fromID']
        e2_id = rel_instance.attrib['toID']
        rel = rel_instance.attrib['type']
        rel_id = i2b2_label_dict.get(rel)
        if  rel_id != None:
            if e1_id.startswith('E') and e2_id.startswith('E'):
                my_dict['relation_dict'][(e1_id, e2_id)] = rel_id
    
    return my_dict

# =========================
#  TimeBank-densen Reader
# =========================
def tbd_tml_reader(dir_name, file_name):
    my_dict = {}
    my_dict["event_dict"] = {}
    eid_to_eiid = {}
    my_dict["doc_id"] = file_name.replace(".tml", "")
    try:
        xml_dom = Soup(open(dir_name + file_name, 'r', encoding='UTF-8'), 'lxml')
    except:
        print("Can't load this file: {} T_T". format(dir_name + file_name))
        return None

    for item in xml_dom.find_all('makeinstance'):
        eiid = item.attrs['eiid']
        eid = item.attrs['eventid']
        my_dict['event_dict'][eiid] = {}
        my_dict['event_dict'][eiid]['tense'] = item.attrs['tense']
        my_dict['event_dict'][eiid]['aspect'] = item.attrs['aspect']
        my_dict['event_dict'][eiid]['polarity'] = item.attrs.get('polarity')
        eid_to_eiid[eid] = eiid

    raw_content = xml_dom.find('text')
    content = ''
    pointer = 0
    for item in raw_content.contents:

        if type(item) == bs4.element.Tag and item.name == 'event':
            eid = item.attrs['eid']
            eiid = eid_to_eiid[eid]
            my_dict['event_dict'][eiid]['mention'] = item.text
            my_dict['event_dict'][eiid]['class'] = item.attrs['class']
            my_dict['event_dict'][eiid]['start_char'] = pointer
            end_char = pointer + len(my_dict["event_dict"][eiid]['mention']) - 1
            my_dict['event_dict'][eiid]['end_char'] = end_char

        if type(item) == bs4.element.NavigableString:
            content += str(item)
            pointer += len(str(item))
        else:
            content += item.text
            pointer += len(item.text)

    my_dict["doc_content"] = content
    my_dict["sentences"] = []
    sent_tokenized_text = [str(sent) for sent in nlp(my_dict["doc_content"]).sents]
    sent_span = tokenized_to_origin_span(my_dict["doc_content"], sent_tokenized_text)
    count_sent = 0
    for sent in sent_tokenized_text:
        sent_dict = {}
        sent_dict["sent_id"] = count_sent
        sent_dict["content"] = sent
        sent_dict["sent_start_char"] = sent_span[count_sent][0]
        sent_dict["sent_end_char"] = sent_span[count_sent][1]
        count_sent += 1
        spacy_token = nlp(sent_dict["content"])
        sent_dict["tokens"] = []
        sent_dict["pos"] = []
        # spaCy-tokenized tokens & Part-Of-Speech Tagging
        for token in spacy_token:
            sent_dict["tokens"].append(token.text)
            sent_dict["pos"].append(token.pos_)
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent, sent_dict["tokens"])
        sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])

        # RoBERTa tokenizer
        sent_dict["roberta_subword_to_ID"], sent_dict["roberta_subwords"], \
        sent_dict["roberta_subword_span_SENT"], sent_dict["roberta_subword_map"] = \
        RoBERTa_list(sent_dict["content"], sent_dict["tokens"], sent_dict["token_span_SENT"])
            
        sent_dict["roberta_subword_span_DOC"] = \
        span_SENT_to_DOC(sent_dict["roberta_subword_span_SENT"], sent_dict["sent_start_char"])
        
        sent_dict["roberta_subword_pos"] = []
        for token_id in sent_dict["roberta_subword_map"]:
            if token_id == -1 or token_id is None:
                sent_dict["roberta_subword_pos"].append("None")
            else:
                sent_dict["roberta_subword_pos"].append(sent_dict["pos"][token_id])
        
        my_dict["sentences"].append(sent_dict)
        
        # Add sent_id as an attribute of event
    for event_id, event_dict in my_dict["event_dict"].items():
        # print(event_id)
        # print(event_dict)
        my_dict["event_dict"][event_id]["sent_id"] = sent_id = \
        sent_id_lookup(my_dict, event_dict["start_char"])
        my_dict["event_dict"][event_id]["token_id"] = \
        id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"])
        my_dict["event_dict"][event_id]["roberta_subword_id"] = poss = \
        id_lookup(my_dict["sentences"][sent_id]["roberta_subword_span_DOC"], event_dict["start_char"]) + 1 # sentence is added <s> token
        # try:
        mention = event_dict["mention"]
        sub_word_id = my_dict["sentences"][sent_id]["roberta_subword_to_ID"][poss]
        sub_word = tokenizer.decode([sub_word_id])
        assert sub_word.strip() in mention

    my_dict['relation_dict'] = {}
    for item in xml_dom.find_all('tlink'):
        attr:dict = item.attrs
        e1_id = attr.get('eventinstanceid')
        e2_id = attr.get('relatedtoeventinstance')
        if  e1_id != None and  e2_id != None:
            rel = attr.get('reltype')
            rel_id = tbd_label_dict.get(rel)
            my_dict['relation_dict'][(e1_id, e2_id)] = rel_id
            if rel_id == None:
                print(item)
    return my_dict


# =========================
#   TDDiscourse - Reader
# =========================
def tdd_reader(tdd_file):
    mapping = []
    with open(tdd_file, 'r', encoding='UTF-8') as f:
        read_tsv = csv.reader(f, delimiter="\t")
        for item in read_tsv:
            mapping.append(item)
    return mapping

def doc_mapping(corpus):
    mapping = defaultdict(list)
    for rel in corpus:
        _rel = (rel[1], rel[2], rel[3])
        mapping[rel[0]].append(_rel)
    return dict(mapping)
try:
    TDD_man_train = tdd_reader(TDD_man_train)
    TDD_man_test = tdd_reader(TDD_man_test)
    TDD_man_val = tdd_reader(TDD_man_val)
    TDD_man = TDD_man_train + TDD_man_val + TDD_man_test
    TDD_man = doc_mapping(TDD_man)

    TDD_auto_train = tdd_reader(TDD_auto_train)
    TDD_auto_test = tdd_reader(TDD_auto_test)
    TDD_auto_val = tdd_reader(TDD_auto_val)
    TDD_auto = TDD_auto_train + TDD_auto_val + TDD_auto_test
    TDD_auto = doc_mapping(TDD_auto)
except:
    pass


def tdd_tml_reader(dir_name, file_name, type_doc):
    my_dict = {}
    my_dict["event_dict"] = {}
    my_dict["doc_id"] = file_name.replace(".tml", "")
    try:
        xml_dom = Soup(open(dir_name + file_name, 'r', encoding='UTF-8'), 'lxml')
    except:
        print("Can't load this file: {} T_T". format(dir_name + file_name))
        return None
    
    for item in xml_dom.find_all('makeinstance'):
        eid = item.attrs['eventid']
        my_dict['event_dict'][eid] = {}
        my_dict['event_dict'][eid]['tense'] = item.attrs['tense']
        my_dict['event_dict'][eid]['aspect'] = item.attrs['aspect']
        my_dict['event_dict'][eid]['polarity'] = item.attrs.get('polarity')

    raw_content = xml_dom.find('text')
    content = ''
    pointer = 0
    for item in raw_content.contents:
        if type(item) == bs4.element.Tag and item.name == 'event':
            eid = item.attrs['eid']
            my_dict['event_dict'][eid]['mention'] = item.text
            my_dict['event_dict'][eid]['class'] = item.attrs['class']
            my_dict['event_dict'][eid]['start_char'] = pointer
            end_char = pointer + len(my_dict["event_dict"][eid]['mention']) - 1
            my_dict['event_dict'][eid]['end_char'] = end_char
        if type(item) == bs4.element.NavigableString:
            content += str(item)
            pointer += len(str(item))
        else:
            content += item.text
            pointer += len(item.text)
    my_dict["doc_content"] = content

    # print(my_dict["doc_content"])

    my_dict["sentences"] = []
    sent_tokenized_text = [str(sent) for sent in nlp(my_dict["doc_content"]).sents]
    sent_span = tokenized_to_origin_span(my_dict["doc_content"], sent_tokenized_text)
    count_sent = 0
    for sent in sent_tokenized_text:
        sent_dict = {}
        sent_dict["sent_id"] = count_sent
        sent_dict["content"] = sent
        sent_dict["sent_start_char"] = sent_span[count_sent][0]
        sent_dict["sent_end_char"] = sent_span[count_sent][1]
        count_sent += 1
        spacy_token = nlp(sent_dict["content"])
        sent_dict["tokens"] = []
        sent_dict["pos"] = []
        # spaCy-tokenized tokens & Part-Of-Speech Tagging
        for token in spacy_token:
            sent_dict["tokens"].append(token.text)
            sent_dict["pos"].append(token.pos_)
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent, sent_dict["tokens"])
        sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])

        # RoBERTa tokenizer
        sent_dict["roberta_subword_to_ID"], sent_dict["roberta_subwords"], \
        sent_dict["roberta_subword_span_SENT"], sent_dict["roberta_subword_map"] = \
        RoBERTa_list(sent_dict["content"], sent_dict["tokens"], sent_dict["token_span_SENT"])
            
        sent_dict["roberta_subword_span_DOC"] = \
        span_SENT_to_DOC(sent_dict["roberta_subword_span_SENT"], sent_dict["sent_start_char"])
        
        sent_dict["roberta_subword_pos"] = []
        for token_id in sent_dict["roberta_subword_map"]:
            if token_id == -1 or token_id is None:
                sent_dict["roberta_subword_pos"].append("None")
            else:
                sent_dict["roberta_subword_pos"].append(sent_dict["pos"][token_id])
        
        my_dict["sentences"].append(sent_dict)
        
        # Add sent_id as an attribute of event
    for event_id, event_dict in my_dict["event_dict"].items():
        # print(event_id)
        # print(event_dict)
        my_dict["event_dict"][event_id]["sent_id"] = sent_id = \
        sent_id_lookup(my_dict, event_dict["start_char"])
        my_dict["event_dict"][event_id]["token_id"] = \
        id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"])
        my_dict["event_dict"][event_id]["roberta_subword_id"] = poss = \
        id_lookup(my_dict["sentences"][sent_id]["roberta_subword_span_DOC"], event_dict["start_char"]) + 1 # sentence is added <s> token
        # try:
        mention = event_dict["mention"]
        sub_word_id = my_dict["sentences"][sent_id]["roberta_subword_to_ID"][poss]
        sub_word = tokenizer.decode([sub_word_id])
        assert sub_word.strip() in mention
    
    my_dict['relation_dict'] = {}
    if type_doc == 'man':
        rel_list = TDD_man.get(my_dict["doc_id"])
        if rel_list == None:
            return None
    elif type_doc == 'auto':
        rel_list = TDD_auto.get(my_dict["doc_id"])
        if rel_list == None:
            return None
    else:
        raise ValueError("Don't have this corpus!")
    eids = my_dict['event_dict'].keys()

    # print(rel_list)
    for item in rel_list:
        eid1, eid2, rel = item
        # print(item)
        if eid1 in eids and eid2 in eids:
            # print(tdd_label_dict[rel])
            my_dict['relation_dict'][(eid1, eid2)] = tdd_label_dict[rel]
    
    return my_dict

    
def cat_xml_reader(dir_name, file_name, intra=True, inter=False):
    my_dict = {}
    my_dict['event_dict'] = {}
    my_dict['doc_id'] = file_name.replace('.xml', '')

    try:
        # xml_dom = Soup(open(dir_name + file_name, 'r', encoding='UTF-8'), 'xml')
        with open(dir_name + file_name, 'r', encoding='UTF-8') as f:
            doc = f.read()
            xml_dom = Soup(doc, 'lxml')
    except Exception as e:
        print("Can't load this file: {}. Please check it T_T". format(dir_name + file_name))
        print(e)
        return None

    doc_toks = []
    my_dict['doc_tokens'] = {}
    _sent_dict = defaultdict(list)
    _sent_token_span_doc = defaultdict(list)
    for tok in xml_dom.find_all('token'):
        token = tok.text
        t_id = int(tok.attrs['t_id'])
        sent_id = int(tok.attrs['sentence'])
        tok_sent_id = len(_sent_dict[sent_id])

        my_dict['doc_tokens'][t_id] = {
            'token': token,
            'sent_id': sent_id,
            'tok_sent_id': tok_sent_id
        }
        
        doc_toks.append(token)
        _sent_dict[sent_id].append(token)
        _sent_token_span_doc[sent_id].append(t_id)
        assert len(doc_toks) == t_id, f"{len(doc_toks)} - {t_id}"
        assert len(_sent_dict[sent_id]) == tok_sent_id + 1
    
    my_dict['doc_content'] = ' '.join(doc_toks)

    my_dict['sentences'] = []
    for k, v in _sent_dict.items():
        start_token_id = _sent_token_span_doc[k][0]
        start = len(' '.join(doc_toks[0:start_token_id-1]))
        if start != 0:
            start = start + 1 # space at the end of the previous sent
        sent_dict = {}
        sent_dict['sent_id'] = k
        sent_dict['token_span_doc'] = _sent_token_span_doc[k] # from 1
        sent_dict['content'] = ' '.join(v)
        sent_dict['tokens'] = v
        sent_dict['pos'] = []
        for tok in v:
            sent_dict['pos'].append(nlp(tok)[0].pos_)
        sent_dict['d_span'] = (start, start + len(sent_dict['content']))
        assert my_dict['doc_content'][sent_dict['d_span'][0]: sent_dict['d_span'][1]] == sent_dict['content'], f"\n'{sent_dict['content']}' \n '{my_dict['doc_content'][sent_dict['d_span'][0]: sent_dict['d_span'][1]]}'"
        my_dict['sentences'].append(sent_dict)

    if xml_dom.find('markables') == None:
        print(my_dict['doc_id'])
        return None

    for item in xml_dom.find('markables').children:
        if type(item)== bs4.element.Tag and 'action' in item.name:
            eid = int(item.attrs['m_id'])
            e_typ = item.name
            mention_span = [int(anchor.attrs['t_id']) for anchor in item.find_all('token_anchor')]
            mention_span_sent = [my_dict['doc_tokens'][t_id]['tok_sent_id'] for t_id in mention_span]
            
            if len(mention_span) != 0:
                mention = ' '.join(doc_toks[mention_span[0]-1:mention_span[-1]])
                start = len(' '.join(doc_toks[0:mention_span[0]-1]))
                if start != 0:
                    start = start + 1 # space at the end of the previous
                my_dict['event_dict'][eid] = {}
                my_dict['event_dict'][eid]['mention'] = mention
                my_dict['event_dict'][eid]['mention_span'] = mention_span
                my_dict['event_dict'][eid]['d_span'] = (start, start + len(mention))
                my_dict['event_dict'][eid]['token_id_list'] = mention_span_sent
                my_dict['event_dict'][eid]['class'] = e_typ
                my_dict['event_dict'][eid]['sent_id'] = find_sent_id(my_dict['sentences'], mention_span)
                assert my_dict['event_dict'][eid]['sent_id'] != None
                assert my_dict['doc_content'][start:  start + len(mention)] == mention, f"\n'{mention}' \n'{my_dict['doc_content'][start:  start + len(mention)]}'"
    
    my_dict['relation_dict'] = {}
    if intra==True:
        for item in xml_dom.find('relations').children:
            if type(item)== bs4.element.Tag and 'plot_link' in item.name:
                r_id = item.attrs['r_id']
                if item.has_attr('signal'):
                    signal = item.attrs['signal']
                else:
                    signal = ''
                try:
                    r_typ = item.attrs['reltype']
                except:
                    # print(my_dict['doc_id'])
                    # print(item)
                    continue
                cause = item.attrs['causes']
                cause_by = item.attrs['caused_by']
                head = int(item.find('source').attrs['m_id'])
                tail = int(item.find('target').attrs['m_id'])

                assert head in my_dict['event_dict'].keys() and tail in my_dict['event_dict'].keys()
                my_dict['relation_dict'][(head, tail)] = r_typ
                
    if inter==True:
        dir_name = './datasets/ESL/annotated_data/v0.9/'
        inter_dir_name = dir_name.replace('annotated_data', 'evaluation_format/full_corpus') + 'event_mentions_extended/'
        file_name = file_name.replace('.xml.xml', '.xml')
        lines = []
        try:
            with open(inter_dir_name+file_name, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except:
            print("{} is not exit!".format(inter_dir_name+file_name))
        for line in lines:
            rel = line.strip().split('\t')
            r_typ = rel[2]
            head_span, tail_span = get_mention_span(rel[0]), get_mention_span(rel[1])
            # print(head_span, tail_span)
            head, tail = find_m_id(head_span, my_dict['event_dict']), find_m_id(tail_span, my_dict['event_dict'])
            assert head != None and tail != None, f"doc: {inter_dir_name+file_name}, line: {line}, rel: {rel}"

            if r_typ != 'null':
                my_dict['relation_dict'][(head, tail)] = r_typ
    
    return my_dict


def ctb_cat_reader(dir_name, file_name):
    my_dict = {}
    my_dict['event_dict'] = {}
    my_dict['doc_id'] = file_name.replace('.xml', '')
    
    try:
        # xml_dom = Soup(open(dir_name + file_name, 'r', encoding='UTF-8'), 'xml')
        with open(dir_name + file_name, 'r', encoding='UTF-8') as f:
            doc = f.read()
            xml_dom = Soup(doc, 'lxml')
    except Exception as e:
        print("Can't load this file: {}. Please check it T_T". format(dir_name + file_name))
        print(e)
        return None

    doc_toks = []
    my_dict['doc_tokens'] = {}
    _sent_dict = defaultdict(list)
    _sent_token_span_doc = defaultdict(list)
    for tok in xml_dom.find_all('token'):
        token = tok.text
        t_id = int(tok.attrs['id'])
        sent_id = int(tok.attrs['sentence'])
        tok_sent_id = int(tok.attrs['number'])

        my_dict['doc_tokens'][t_id] = {
            'token': token,
            'sent_id': sent_id,
            'tok_sent_id': tok_sent_id
        }
        
        doc_toks.append(token)
        _sent_dict[sent_id].append(token)
        _sent_token_span_doc[sent_id].append(t_id)
        assert len(doc_toks) == t_id, f"{len(doc_toks)} - {t_id}"
        assert len(_sent_dict[sent_id]) == tok_sent_id + 1
    
    my_dict['doc_content'] = ' '.join(doc_toks)

    my_dict['sentences'] = []
    for k, v in _sent_dict.items():
        start_token_id = _sent_token_span_doc[k][0]
        start = len(' '.join(doc_toks[0:start_token_id-1]))
        if start != 0:
            start = start + 1 # space at the end of the previous sent
        sent_dict = {}
        sent_dict['sent_id'] = k
        sent_dict['token_span_doc'] = _sent_token_span_doc[k] # from 1
        sent_dict['content'] = ' '.join(v)
        sent_dict['tokens'] = v
        sent_dict['pos'] = []
        for tok in v:
            sent_dict['pos'].append(nlp(tok)[0].pos_)
        sent_dict['d_span'] = (start, start + len(sent_dict['content']))
        assert my_dict['doc_content'][sent_dict['d_span'][0]: sent_dict['d_span'][1]] == sent_dict['content'], f"\n'{sent_dict['content']}' \n '{my_dict['doc_content'][sent_dict['d_span'][0]: sent_dict['d_span'][1]]}'"
        my_dict['sentences'].append(sent_dict)

    if xml_dom.find('markables') == None:
        print(my_dict['doc_id'])
        return None
    
    for item in xml_dom.find('markables').children:
        if type(item)== bs4.element.Tag and 'event' in item.name:
            eid = int(item.attrs['id'])
            e_typ = item.name
            mention_span = [int(anchor.attrs['id']) for anchor in item.find_all('token_anchor')]
            mention_span_sent = [my_dict['doc_tokens'][t_id]['tok_sent_id'] for t_id in mention_span]
            
            if len(mention_span) != 0:
                mention = ' '.join(doc_toks[mention_span[0]-1:mention_span[-1]])
                start = len(' '.join(doc_toks[0:mention_span[0]-1]))
                if start != 0:
                    start = start + 1 # space at the end of the previous
                my_dict['event_dict'][eid] = {}
                my_dict['event_dict'][eid]['mention'] = mention
                my_dict['event_dict'][eid]['mention_span'] = mention_span
                my_dict['event_dict'][eid]['d_span'] = (start, start + len(mention))
                my_dict['event_dict'][eid]['token_id_list'] = mention_span_sent
                my_dict['event_dict'][eid]['class'] = e_typ
                my_dict['event_dict'][eid]['sent_id'] = find_sent_id(my_dict['sentences'], mention_span)
                assert my_dict['event_dict'][eid]['sent_id'] != None
                assert my_dict['doc_content'][start:  start + len(mention)] == mention, f"\n'{mention}' \n'{my_dict['doc_content'][start:  start + len(mention)]}'"
    
    my_dict['relation_dict'] = {}
    for item in xml_dom.find('relations').children:
        if type(item)== bs4.element.Tag and 'clink' in item.name:
            r_id = item.attrs['id']
            r_typ = 'PRECONDITION'
            head = int(item.find('source').attrs['id'])
            tail = int(item.find('target').attrs['id'])

            assert head in my_dict['event_dict'].keys() and tail in my_dict['event_dict'].keys()
            my_dict['relation_dict'][(head, tail)] = r_typ

    return my_dict


if __name__ == '__main__':
    my_dict = cat_xml_reader('./datasets/ESL/annotated_data/v0.9/1/','1_1ecbplus.xml.xml')
    with open('1_1ecbplus.json', 'w', encoding='utf-8') as f:
        json.dump(my_dict, f, indent=6)
    