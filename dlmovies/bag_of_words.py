
# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd

# Import BeautifulSoup into your workspace
from bs4 import BeautifulSoup

# for regular expression
import re

# Natural Language Tool Kit.
import nltk

# Import the stop word list
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier

import numpy as np

# one time activity
# Download text data sets, including stop words
def download_nltk():
    nltk.download()

def read_csv(file_name):
    data = pd.read_csv(file_name, header=0, delimiter="\t", quoting=3)
    return data

def review_to_words(raw_review):
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, "lxml").get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join(meaningful_words))

def get_clean_reviews_data(data):
    # no. of reviews
    num_reviews = data["review"].size

    # clean each review and store it in clean_train_reviews
    clean_reviews_data = []
    for i in range(0, num_reviews):
        if (i + 1) % 1000 == 0:
            print(f"Review {i + 1} of {num_reviews}")
        clean_reviews_data.append(review_to_words(data["review"][i]))

    return clean_reviews_data

def get_data_features(data):

    clean_reviews_data = get_clean_reviews_data(data)

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None,
                                 max_features = 5000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    data_features = vectorizer.fit_transform(clean_reviews_data)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    data_features = data_features.toarray()

    # Take a look at the words in the vocabulary
    # vocab = vectorizer.get_feature_names()
    # print(vocab)

    # Sum up the counts of each vocabulary word
    dist = np.sum(data_features, axis=0)

    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    # for tag, count in zip(vocab, dist):
    #     print(f"Word: {tag} and Count: {count}")

    return data_features

def train_using_random_forest(train_data_features, train_data):
    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 100)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    forest = forest.fit(train_data_features, train_data["sentiment"])

    return forest

def predict_data_using_random_forest(forest, test_data_features, test_data):
    # Use the random forest to make sentiment label predictions
    result = forest.predict(test_data_features)

    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    output = pd.DataFrame( data={"id":test_data["id"], "sentiment":result} )

    # Use pandas to write the comma-separated output file
    output.to_csv( "data/Bag_of_Words_model.csv", index=False, quoting=3 )

def main():
    train_data = read_csv("data/labeledTrainData.tsv")
    train_data_features = get_data_features(train_data)
    forest = train_using_random_forest(train_data_features, train_data)

    test_data = read_csv("data/testData.tsv")
    test_data_features = get_data_features(test_data)
    predict_data_using_random_forest(forest, test_data_features, test_data)


if __name__ == '__main__':
    main()