import nltk

sentence = "The famous algorithm produced accurate results"

grammar = '''
    NP: {<DET>? <ADJ>* <NOUN>*}
    P: {<PREP>}
    V: {<VERB.*>}
    PP: {<PREP> <NOUN>}
    VP: {<VERB> <NOUN|PREP>*}
    '''

tokens = nltk.word_tokenize(sentence)

# It returns list of tupples as (token, token's pos).
tagged_sent = nltk.pos_tag(tokens, tagset='universal')
print(tagged_sent)

# On the basis of grammar we will chunk the tokens in phrase.
cp = nltk.RegexpParser(grammar)

result = cp.parse(tagged_sent)
print(result)
