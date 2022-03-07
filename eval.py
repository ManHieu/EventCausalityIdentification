import json
from typing import List, Tuple
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

from data_modules.input_example import InputExample
from utils.utils import compute_f1



def is_mention(item: Tuple[str, str], gold: List[Tuple[str, str]]):
    for rel in gold:
        ev1, ev2 = rel[0], rel[1]
        if item[0] in ev1 and item[1] in ev2:
            return 1
    
    return 0


def same_mention(mention1: str, mention2: str):
    if mention1 in mention2 or mention2 in mention1:
        return True
    else:
        return False


def is_true(item: Tuple[str, str], gold: Tuple[str, str]):
    if same_mention(item[0], gold[0]) and same_mention(item[1], gold[1]):
        return True
    elif same_mention(item[1], gold[0]) and same_mention(item[0], gold[1]):
        return True
    else:
        return False

def inference(sentence: str):
    if sentence != 'None':
        sents = sentence.strip().split('.')
        rels = []
        for sent in sents:
            if 'cause' in sent:
                head, tails = sent.split('cause')[0], sent.split('cause')[1]
                head = head.strip()
                _rels = [(head, item.strip()) for item in tails.split('and')]
                rels.extend(_rels)
        return rels
    else:
        return []
        
def eval_corpus(resutl_file: str='./test.json'):        
    golds = []
    predicts = []

    with open(resutl_file,'r') as f:
        lines = json.load(f)
        for result in lines:
            predict = result['predicted']
            gold = result['gold']

            predicts.append(predict)
            golds.append(gold)
        
    f1, p, r, tp, n_pred, n_gold = compute_f1(predicts, golds)

    # print(tp)
    # print(n_pred)
    # print(n_gold)
    # print('precision: {}'.format(p))
    # print('recall: {}'.format(r))
    # print('f1: {}'.format(f1))
    return f1, p, r
# CM = confusion_matrix(gold, predict)
# print(classification_report(gold, predict))
# print(CM)

if __name__ == '__main__':
    eval_corpus()