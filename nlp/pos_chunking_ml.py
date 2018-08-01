# POS chunking usinng Machine learning.

from nltk.corpus import conll2000
import random

conll_data = list(conll2000.chunked_sents())
random.shuffle(conll_data)

# randomly select 80% for training data and remaining for test data.
train_sents = conll_data[:int(len(conll_data) * 0.8)]
test_sents = conll_data[int(len(conll_data) * 0.8 + 1):]

from nltk.stem.porter import PorterStemmer

def features(tokens, index, history):
    # tokens are tagged words of a sentence
    # Index is the index of token for which the features to be extracted
    # history is the previous predicted tags

    stemmer = PorterStemmer()

    # Build the sequence of words for training
    tokens = [('__PREVSEQ2__', '__PREVSEQ2__'),
              ('__PREVSEQ1__', '__PRESEQ1__')] + list(tokens) + [('__END1__', '__END1__'),
                                                                 ('__END2__', '__END2__')]
    history = ['__PREVSEQ2__', '__PREVSEQ2__'] + list(history)

    # shift the index with 2 to point to current token
    index += 2

    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prev2word, prev2pos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    next2word, next2pos = tokens[index + 2]

    return {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,

        'next-word': nextword,
        'next-pos': nextpos,

        'next-next-word': next2word,
        'next-next-pos': next2pos,

        'prev-word': prevword,
        'prev-pos': prevpos,

        'prev-prev-word': prev2word,
        'prev-prev-pos': prev2pos,
    }


from nltk import ChunkParserI, ClassifierBasedTagger
from nltk.chunk import conlltags2tree, tree2conlltags

class FooChunkParser(ChunkParserI):
    def __init__(self, chunked_sents, **kwargs):
        # Transform the trees in IOB annotated sentences [(word, pos, chunk)]
        chunked_sents = [tree2conlltags(sent) for sent in chunked_sents]

        # Make tags compatible with the tagger interface [((word, pos), chunk)]
        def get_tagged_pairs(chunked_sent):
            return [((word, pos), chunk) for word, pos, chunk in chunked_sent]

        chunked_sents = [get_tagged_pairs(sent) for sent in chunked_sents]

        self.feature_detector = features
        self.tagger = ClassifierBasedTagger(
            train=chunked_sents,
            feature_detector=features,
            **kwargs)

    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)
        iob_triplets = [(word, token, chunk) for ((word, token), chunk) in chunks]
        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)

chunker = FooChunkParser(train_sents)

# both type of chunking taggers have almost same accuracy.
# So we prefer machine learnning approach to avoid the manual building of grammar rules effort.
print(chunker.evaluate(test_sents))
