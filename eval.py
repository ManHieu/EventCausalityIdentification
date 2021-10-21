import json
from typing import List, Tuple
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

from data_modules.input_example import InputExample



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
def eval_corpus():        
    tp = 0
    n_gold = 0
    n_pred = 0

    with open('./predictions.json','r') as f:
        lines = json.load(f)
        # print(lines)
        # predicts = defaultdict(list)
        # golds = defaultdict(list)
        golds = []
        predicts = []
        
        for result in lines:
            predict = result['predicted']
            gold = result['gold']

            predict = [item.strip() for item in predict.split('causes')]
            # print(predict)
            gold = [item.strip() for item in gold.split('causes')]
            # print(gold)
            # if 'precondition' in result['predicted']:
            #     predict = predict.split('precondition')
            # else:
            #     predict = predict.split('falling action')
            
            # if 'precondition' in result['gold']:
            #     gold = gold.split('precondition')
            # else:
            #     gold = gold.split('falling action')
            if len(predict) == 2:
                n_pred += 1
            if len(gold) == 2:
                n_gold += 1
            if len(predict) == 2 and len(gold) == 2:
                # print(f"predict: {predict} - gold: {gold} - {is_true(predict, gold)}")
                if is_true(predict, gold):
                    tp = tp + 1
                
    #         predicts[result['sentence']] = inference(predict)
    #         golds[result['sentence']] = inference(gold)

    # print(golds)
    # print(predicts)

    # for key in golds.keys():
    #     predict = predicts[key]
    #     gold = golds[key]
    #     n_gold += len(set([(item[0]+'0', item[1]+'1')for item in gold]))
    #     n_pred += len(set([(item[0]+'0', item[1]+'1')for item in predict]))
    #     for item in predict:
    #         print(key)
    #         print(f"predict: {item}")
    #         print(f"gold: {gold}")
    #         print(f"is mention: {is_mention(item, gold)}")
    #         print("_" * 10)
    #         tp += is_mention(item, gold)

    p = tp/(n_pred + 1e-9)
    r = tp/n_gold
    f1 = 2*p*r/(p+r + 1e-9)

    print(tp)
    print(n_pred)
    print(n_gold)
    print('precision: {}'.format(p))
    print('recall: {}'.format(r))
    print('f1: {}'.format(f1))
    return f1
# CM = confusion_matrix(gold, predict)
# print(classification_report(gold, predict))
# print(CM)

if __name__ == '__main__':
    eval_corpus()