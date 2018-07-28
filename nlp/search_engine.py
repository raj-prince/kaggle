import codecs

import re
import sys
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# download data from here: https://www.dropbox.com/sh/k91gn7rws8b03lm/AABAPcRySDRkpzF-WDTvgvH_a
# load the dataset.
text = codecs.open('./data/wiki-600', encoding='utf-8').read()

# find all the positions of heading of wikipedia
# because articles titles are formatted as follows: = Albert Einstein =
starts = [match.span()[0] for match in re.finditer('\n = [^=]', text)]

# load all the articles in a list by using the start position of headings.
articles = list()
for i, start in enumerate(starts):
    end = starts[i + 1] if i + 1 < len(starts) else len(text)
    articles.append(text[start:end])

# calculate the snippets from article (extract first 200 character from articles).
snippets = [' '.join(article[:200].split()) for article in articles]

for snippet in snippets[:20]:
    print(snippet)


term_frequency = defaultdict(dict)

stemmer = PorterStemmer()

# get tokens from the article, lowered, tokenized, and stemmed.
def get_tokens(article):
    article = article.lower()
    tokens_list = word_tokenize(article)
    stemmed_token = [stemmer.stem(word) for word in tokens_list]
    return stemmed_token

# calculate the token frequency and index it for the given article.
def index(id, article):
    tokens = get_tokens(article)
    count = {}
    for token in tokens:
        if token in count:
            count[token] += 1
        else:
            count[token] = 1
    for key in count:
        term_frequency[key][id] = count[key]
    return term_frequency

# index all the article.
for i, article in enumerate(articles):
    if i and i % 10 == 0:
        print(i, end=', ')
    sys.stdout.flush()
    # index(i, article) # do when required

# output: {300: 1, 84: 5, 294: 1}
# means 'einstein' exist 1 time in 300th article, 5 times in 84th article and
# 1 time in 294th article.
print(term_frequency['einstein'])

################## Saving and loading data ###################
import pickle

def picklesave(obj, filename):
    print('Saving ..')
    ff = open(filename, 'wb')
    pickle.dump(obj, ff)
    ff.close()
    print('Done')
    return True

def pickleload(filename):
    print('Loading ..')
    ff = open(filename, 'rb')
    obj = pickle.load(ff)
    ff.close()
    print('Done')
    return obj

# picklesave([snippets, term_frequency], './data/data-600.pdata')
snippets, term_frequency = pickleload('./data/data-600.pdata')

################## Ranking the articles for search #############

import math
import operator

D = len(snippets)

print(term_frequency['einstein'])

def get_tfidf(article_id, token):
    res = term_frequency[token][article_id]
    res = res * math.log10(len(articles) / len(term_frequency[token]))
    return res

def search(query, nresults=10):
    tokens = get_tokens(query)
    scores = defaultdict(float)
    for token in tokens:
        for article, score in term_frequency[token].items():
            scores[article] += get_tfidf(article, token)
    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1))
    return reversed(sorted_scores[-nresults:])

def display_results(query, results):
    print(f'You search for: {query}')
    print('-'*100)
    for id, score in results:
        print(snippets[id])
    print('='*100)

display_results('obama', search('obama'))
display_results('einstein', search('einstein'))



