# Data downloaded from the below link.
# link: https://www.dropbox.com/s/gnr5o63y9me6aa1/language_identification?dl=0

import io

# Used to make counter dictionary from the list of words.
from collections import Counter, defaultdict

# Used to create the all measurement charts.
from sklearn.metrics import classification_report, accuracy_score

# load data
file = io.open('./data/language_identification', mode='r', encoding='utf-8')
data = file.read().split('\n')[:-1]

# each line in the data first 2 character is language and remaining is snippets.
languages = [line[:2] for line in data]
text = [line[-100:] for line in data]

# split in train and test
train_xx, train_yy = text[:8000], languages[:8000]
test_xx, test_yy = text[8000:10000], languages[8000:10000]

# helper function for ngram
def get_ngrams(text, n):
    # return a list of ngrams for a given size
    res = []
    len_t = len(text)
    if n > len_t:
        return res
    for i in range(0, len_t - n + 1):
        res.append(text[i : i + n])
    return res

# Test the function
print(train_xx[0])
print(get_ngrams(train_xx[0], 1))

# create list of unique language.
languages = list(set(train_yy))

# create counts of each of the language as dictionary.
language_counts = Counter(train_yy)

print(languages)
print(language_counts)

# sets ngram_models[language][ngram] = 0 for everything
ngram_models = dict([(language, defaultdict(float)) for language in languages])

# create ngram_model
for text, language in zip(train_xx, train_yy):
    ngram_models[language] = Counter(ngram_models[language]) + Counter(get_ngrams(text, 2))

for lang in languages:
    lang_cnt = language_counts[lang]
    for k in ngram_models[lang]:
        ngram_models[lang][k] /= lang_cnt

print(ngram_models['en'])

# scoring and prediction
def predict(text):
    scores = dict([(language, score(text, language)) for language in ngram_models.keys()])
    return max(scores.keys(), key=scores.get)

def score(text, language):
    text_ngrams = get_ngrams(text, 2)
    cnt_score = 0.000000001
    for ngram in text_ngrams:
        cnt_score += ngram_models[language][ngram]
    return cnt_score

# Evaluation
predictions = [predict(text) for text in test_xx]
print(classification_report(test_yy, predictions))
print(accuracy_score(test_yy, predictions))
