from typing import Dict, List, Set
from transformers import RobertaTokenizer
from constants import *
import spacy
import re
import networkx as nx


tokenizer = RobertaTokenizer.from_pretrained('/vinai/hieumdt/pretrained_models/tokenizers/roberta-base', unk_token='<unk>')
nlp = spacy.load("en_core_web_sm")


def RoBERTa_list(content, token_list = None, token_span_SENT = None):
    encoded = tokenizer.encode(content)
    roberta_subword_to_ID = encoded
    # input_ids = torch.tensor(encoded).unsqueeze(0)  # Batch size 1
    # outputs = model(input_ids)
    # last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    roberta_subwords = []
    roberta_subwords_no_space = []
    for index, i in enumerate(encoded):
        r_token = tokenizer.decode([i])
        if r_token != " ":
            roberta_subwords.append(r_token)
            if r_token[0] == " ":
                roberta_subwords_no_space.append(r_token[1:])
            else:
                roberta_subwords_no_space.append(r_token)

    roberta_subword_span = tokenized_to_origin_span(content, roberta_subwords_no_space[1:-1]) # w/o <s> and </s>
    roberta_subword_map = []
    if token_span_SENT is not None:
        roberta_subword_map.append(-1) # "<s>"
        for subword in roberta_subword_span:
            roberta_subword_map.append(token_id_lookup(token_span_SENT, subword[0], subword[1]))
        roberta_subword_map.append(-1) # "</s>" 
        return roberta_subword_to_ID, roberta_subwords, roberta_subword_span, roberta_subword_map
    else:
        return roberta_subword_to_ID, roberta_subwords, roberta_subword_span, -1


def tokenized_to_origin_span(text, token_list):
    token_span = []
    pointer = 0
    for token in token_list:
        while True:
            if token[0] == text[pointer]:
                start = pointer
                end = start + len(token) - 1
                pointer = end + 1
                break
            else:
                pointer += 1
        token_span.append([start, end])
    return token_span


def sent_id_lookup(my_dict, start_char, end_char = None):
    for sent_dict in my_dict['sentences']:
        if end_char is None:
            if start_char >= sent_dict['sent_start_char'] and start_char <= sent_dict['sent_end_char']:
                return sent_dict['sent_id']
        else:
            if start_char >= sent_dict['sent_start_char'] and end_char <= sent_dict['sent_end_char']:
                return sent_dict['sent_id']


def token_id_lookup(token_span_SENT, start_char, end_char):
    for index, token_span in enumerate(token_span_SENT):
        if start_char >= token_span[0] and end_char <= token_span[1]:
            return index


def span_SENT_to_DOC(token_span_SENT, sent_start):
    token_span_DOC = []
    #token_count = 0
    for token_span in token_span_SENT:
        start_char = token_span[0] + sent_start
        end_char = token_span[1] + sent_start
        #assert my_dict["doc_content"][start_char] == sent_dict["tokens"][token_count][0]
        token_span_DOC.append([start_char, end_char])
        #token_count += 1
    return token_span_DOC


def id_lookup(span_SENT, start_char):
    # this function is applicable to RoBERTa subword or token from ltf/spaCy
    # id: start from 0
    token_id = -1
    for token_span in span_SENT:
        token_id += 1
        if token_span[0] <= start_char and token_span[1] >= start_char:
            return token_id
    raise ValueError("Nothing is found. \n span sentence: {} \n start_char: {}".format(span_SENT, start_char))


def list_id_lookup(span_SENT: List[Set[int]], start_char: int, end_char: int):
    list_id = []
    token_id = -1
    for token_span in span_SENT:
        token_id = token_id + 1
    
        if token_span[1] < start_char or token_span[0] > end_char:
            pass
        else:
            list_id.append(token_id)
    assert len(list_id) != 0, "Nothing is found. \n span sentence: {} \n start_char: {} \n end_char: {}".format(span_SENT, start_char, end_char)
    return list_id


def find_sent_id(sentences: List[Dict], mention_span: List[int]):
    """
    Find sentence id of mention (ESL)
    """
    for sent in sentences:
        token_span_doc = sent['token_span_doc']
        if set(mention_span) == set(mention_span).intersection(set(token_span_doc)):
            return sent['sent_id']
    
    return None


def remove_special_token(seq: List[str]) -> List[str]:
    return [re.sub("[^A-Za-z0-9.,?']", " ", token) for token in seq]


def get_mention_span(span: str) -> List[int]:
    span = [int(tok.strip()) for tok in span.split('_')]
    return span


def find_m_id(mention: List[int], eventdict:Dict):
    for m_id, ev in eventdict.items():
        # print(mention, ev['mention_span'])
        if mention == ev['mention_span']:
            return m_id
    
    return None


def covert_to_doc_id(parsed_sentences, span_in_doc=(0, 0)):
    doc = []
    num_previous = 0
    for sent in parsed_sentences['sentences']:
        for token in sent['tokens']:
            token['id'] = token['id'] + num_previous
            if token['head'] != 0:
                token['head'] = token['head'] + num_previous
            else:
                token['head'] = token['head']
    
            token['dspan'] = (token['dspan'][0] + span_in_doc[0], token['dspan'][1] + span_in_doc[0])
            
            doc.append(token)

        num_previous = num_previous + len(sent['tokens'])
    
    return doc


def id_mapping(dspan_entity, doc_token, doc_content):
    span_entity = []
    for token in doc_token:
        if set(range(token['dspan'][0], token['dspan'][1])).issubset(set(range(dspan_entity[0], dspan_entity[1]))) or \
           set(range(dspan_entity[0], dspan_entity[1])).issubset(set(range(token['dspan'][0], token['dspan'][1]))):
            span_entity.append(token['id'])
    
    if len(span_entity) == 0:
        print(f"dspan: {dspan_entity}")
        print(f"entity: {doc_content[dspan_entity[0]:dspan_entity[1]]}")
        print(f"doc_tokens: {doc_token}")
        print(f"doc_content: {doc_content}")
    return span_entity


def find_common_lowest_ancestor(tree, nodes):
    ancestor = nx.lowest_common_ancestor(tree, nodes[0], nodes[1])
    for node in nodes[2:]:
        ancestor = nx.lowest_common_ancestor(tree, ancestor, node)
    return ancestor


def expand_entity(node, prunted_dep_tree, doc_tokens_pasered, in_path=False):
    
    span = []
    try:
        if in_path == False:
            descendants = nx.descendants(prunted_dep_tree, node)
            descendants = list(descendants) + [node]
            for n in descendants:
                for tok in doc_tokens_pasered:
                    if tok['id'] == n:
                        span.append(tok)
        else:
            for tok in doc_tokens_pasered:
                if tok['id'] == node:
                    span.append(tok)
    except:
        for tok in doc_tokens_pasered:
            if tok['id'] == node:
                span.append(tok)
    
    span.sort(key=lambda tok: tok['dspan'][0])
    expand_enity = ' '.join([tok['text'] for tok in span])
    
    return expand_enity


