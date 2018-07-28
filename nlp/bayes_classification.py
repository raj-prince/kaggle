# Load the data

import _pickle as cPickle
from collections import defaultdict
import numpy as np

f = open('./data/sentiment_data.pkl', 'rb')
train_positive, train_negative, test_positive, test_negative = cPickle.load(f)
f.close()

print('Data description ...')
print(len(train_positive), len(train_negative), len(test_positive), len(test_negative))
print('='*120)
print(train_positive[:10])
print('='*120)
print(train_negative[:10])
print('='*120)


import re

from sklearn.metrics import classification_report, confusion_matrix

def review_tokens(review):
    return [token.lower() for token in re.findall('[A-Za-z]+', review)]

# naive bayes

nneg, npos = len(train_negative), len(train_positive)

# P(positive) and P(negative)
pos_prob = npos / (nneg + npos)
neg_prob = nneg / (nneg + npos)

# P(positive/token) and P(negative/token)
def get_doc_freq(train_data):
    dict = defaultdict()
    for data in train_data:
        for token in review_tokens(data):
            if token in dict:
                dict[token] += 1
            else:
                dict[token] = 1
    return dict

# stores the in how many document a token present for each and every labels
# i.e for both of the lable, we already processed in how many document a
# particular word exist.
pos_token_freq = get_doc_freq(train_positive)
neg_token_freq = get_doc_freq(train_negative)

# predict
labels = [True]*len(test_positive) + [False]*len(test_negative)
all_test = np.concatenate((test_positive, test_negative))

predictions = list()
for review in all_test:
    pos, neg = pos_prob, neg_prob
    for token in review_tokens(review):
        pos_cnt = pos_token_freq[token] if token in pos_token_freq else 0
        neg_cnt = neg_token_freq[token] if token in neg_token_freq else 0
        pos = pos * (pos_cnt / npos)
        neg = neg * (neg_cnt / nneg)
    predictions.append(pos > neg)


# evaluate, it evaluate all the parameter precision, recall, f1-score, support.
# confusion matrix: Element(i, j) of the confusion matrix is the number of
# times the actual answer is i and the prediction was j. if k class then (k * k)
#            | Predicted No  | Predicted Yes
# Actual No  | True, Negative | False Positive
# Actual yes | False Negative | True Positive
print(classification_report(labels, predictions))
print('='*120)
print(confusion_matrix(labels, predictions))
print('='*120)