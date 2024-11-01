"""
Obtained from original POPE repository: https://github.com/RUCAIBox/POPE

MIT License

Copyright (c) 2024 Yifan Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import json

# TODO: Replace with path to model response and label files
ans_file = 'model_responses.json'
label_file = './coco/coco_pope_adversarial.json'

answers = [json.loads(q) for q in open(ans_file, 'r')]
label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

for answer in answers:
    text = answer['answer']

    # Only keep the first sentence
    if text.find('.') != -1:
        text = text.split('.')[0]

    text = text.replace(',', '')
    words = text.split(' ')
    if 'No' in words or 'not' in words or 'no' in words:
        answer['answer'] = 'no'
    else:
        answer['answer'] = 'yes'

for i in range(len(label_list)):
    if label_list[i] == 'no':
        label_list[i] = 0
    else:
        label_list[i] = 1

pred_list = []
for answer in answers:
    if answer['answer'] == 'no':
        pred_list.append(0)
    else:
        pred_list.append(1)

pos = 1
neg = 0
yes_ratio = pred_list.count(1) / len(pred_list)

TP, TN, FP, FN = 0, 0, 0, 0
for pred, label in zip(pred_list, label_list):
    if pred == pos and label == pos:
        TP += 1
    elif pred == pos and label == neg:
        FP += 1
    elif pred == neg and label == neg:
        TN += 1
    elif pred == neg and label == pos:
        FN += 1

print('TP\tFP\tTN\tFN\t')
print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

precision = float(TP) / float(TP + FP)
recall = float(TP) / float(TP + FN)
f1 = 2*precision*recall / (precision + recall)
acc = (TP + TN) / (TP + TN + FP + FN)
print('Accuracy: {}'.format(acc))
print('Precision: {}'.format(precision))
print('Recall: {}'.format(recall))
print('F1 score: {}'.format(f1))
print('Yes ratio: {}'.format(yes_ratio))
