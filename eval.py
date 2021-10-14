import json
from sklearn.metrics import confusion_matrix, classification_report


with open('./predictions.jsonl','r') as f:
    lines = f.readlines()
    # print(lines)
    predict = []
    gold = []
    for line in lines:
        result = json.loads(line)
        if 'precondition' in result['predicted']:
            predict.append(0)
        else:
            predict.append(1)
        
        if 'precondition' in result['gold']:
            gold.append(0)
        else:
            gold.append(1)

CM = confusion_matrix(gold, predict)
print(classification_report(gold, predict))
print(CM)

