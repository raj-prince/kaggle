# Data downloaded from the below link
# link: https://www.dropbox.com/s/g0ump02avf2x0mo/spell_check?dl=0

# P_w_given_c directly proportion to P_c * P_word_given_c

from autocorrect import spell
from collections import Counter, defaultdict
import re

print(spell('somthing')) # will print 'something'

# load the data
data = open('./data/spell_check').read()
data = ' '.join(data.split())
data = data.lower()

# tokenize
corrected = re.sub("<err targ=(.*?)> (.*?) </err>", '\\1', data)
original = re.sub("<err targ=(.*?)> (.*?) </err>", '\\2', data)

# create original as well as correct form of data.
corrected = re.findall("[a-z0-9'-]+", corrected)
original = re.findall("[a-z0-9'-]+", original)

# print(original[:25])
# print(corrected[:25])

# create vocab
alphabets = "abcdefghijklmnopqrstuvwxyz0123456789'-"
vocab = Counter(corrected)
print(len(vocab))
print(vocab.most_common(10))

# estimate prob_c using vocab
prob_c = defaultdict(float)
for k in vocab:
    prob_c[k] = vocab[k] / len(corrected)

print(prob_c['the']) # => 0.0616
print(prob_c['and']) # => 0.0423
print(prob_c['a']) # => 0.0293

# estimate prob_w_given_c
DEBUG = False

def all_edit_distance_one(word):
    # inserts
    inserts = list()
    for position in range(len(word) + 1):
        for alphabet in alphabets:
            new_word = word[:position] + alphabet + word[position:]
            inserts.append(new_word)

    #deletes
    deletes = list()
    for position in range(len(word)):
        new_word = word[:position] + word[position + 1:]
        deletes.append(new_word)
    if (word == 'an'): print(deletes)

    # replaces
    replaces = list()
    for position in range(len(word)):
        for alphabet in alphabets:
            new_word = word[:position] + alphabet +  word[position + 1:]
            replaces.append(new_word)

    z = list(set(inserts  + deletes + replaces) - set([word]))
    if word == 'an': print(z)
    return z

print(sorted(all_edit_distance_one('the')))

prob_w_given_c = dict()
for word in vocab:
    prob_w_given_c[(word, word)] = 0.001 ** 0
    for error in all_edit_distance_one(word):
        prob_w_given_c[(error, word)] = 0.001 ** 1

print(prob_w_given_c[('thew', 'the')])
print(prob_w_given_c[('thw', 'the')])
print(prob_w_given_c[('the', 'the')])
print(prob_w_given_c[('a', 'an')])
print(prob_w_given_c[('an', 'an')])
print(prob_w_given_c[('and', 'an')])

# predict
def predict(word):
    candidates = [word] + all_edit_distance_one(word)
    scores = dict()
    for candidate in candidates:
        if candidate not in prob_c: continue
        if (word, candidate) not in prob_w_given_c: continue
        scores[candidate] = prob_c[candidate] * prob_w_given_c[(word, candidate)]
        if not scores: return word
        return max(scores.keys(), key=scores.get)

print(predict('thew')) # => 'the'
print(predict('the'))  # => 'the'

# Evaluate

def count_matches(lst1, lst2):
    return sum((xx == yy) for xx, yy in zip(lst1, lst2))

predictions = [predict(xx) for xx in original]

print(f'Total words: {len(corrected)}')
print(f'Total already correct {count_matches(corrected, original)}')
print(f'Total correct after spell correction: {count_matches(corrected, predictions)}')

very_bad_count = 0
bad_count = 0
good_count = 0
for xx, yy, pp in zip(original, corrected, predictions):
    if xx == yy and pp != yy:
        very_bad_count += 1
        if very_bad_count <= 10: print(f'Very Bad: {xx}, {yy}, {pp}')
    if pp == yy and xx != yy:
        good_count += 1
        if good_count <= 10: print(f'Good {xx}, {yy}, {pp}')
    if xx != yy and pp != yy:
        bad_count += 1
        if bad_count <= 10: print(f'Bad {xx}, {yy}, {pp}')

print(f'Words that were correct but made incorrect by the system (very bad): {very_bad_count}')
print(f'Words that were incorrect and were not corrected by the system (bad): {bad_count}')
print(f'Words that were incorrect and were corrected by the system (good!): {good_count}')
