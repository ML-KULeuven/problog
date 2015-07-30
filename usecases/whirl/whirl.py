"""
Whirl ProbLog implementation

Builds a TF-IDF model in the background using NLTK and Scikit-learn, such
that ProbLog can make use of document similarity while reasoning.

From:
W. W. Cohen. Whirl: A word-based information representation language.
Artificial Intelligence, 118(1):163-196, 2000.

Author:
- Wannes Meert
- Anton Dries
"""
from __future__ import print_function

import os, sys
import string
import glob

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append('../../')
from problog.extern import problog_export


# Settings
texts = glob.glob('movies/reviews/*.txt') # input texts for TF-IDF model


# Global variables
vectorizer = None
tfidf = None # TF-IDF model
tokens = None # List of tokens used by TF-IDF
texts_weights = None # TF-IDF weights for texts
if sys.version_info.major == 2:
  punct_table = string.maketrans('','')
else:
  punct_table = str.maketrans('','', string.punctuation)
stemmer = PorterStemmer()


def cleantext(text):
  """Clean string from punctuation and capitals."""
  lowers = text.lower()
  if sys.version_info.major == 2:
    no_punctuation = lowers.translate(punct_table, string.punctuation)
  else:
    no_punctuation = lowers.translate(punct_table)
  return no_punctuation


def tokenize(text):
  """Transform string to list of stemmed tokens."""
  tokens = nltk.word_tokenize(text)
  stemmed = (stemmer.stem(token) for token in tokens)
  return stemmed


def getTFIDF():
  """Return cached TFIDF model."""
  global vectorizer
  global tfidf
  global tokens
  global texts_weights

  if tfidf is None:

    texts_content = []
    for text in texts:
      with open(text, 'r') as ifile:
        texts_content.append(cleantext(ifile.read()))

    vectorizer = CountVectorizer(tokenizer=tokenize, stop_words='english')
    texts_counts = vectorizer.fit_transform(texts_content)
    tokens = vectorizer.get_feature_names()

    tfidf = TfidfTransformer()
    # fit_transform can be replaced by fit for efficiency
    texts_weights = tfidf.fit_transform(texts_counts)

  return tfidf


@problog_export('+str', '+str', '-float')
def similarity(e1, e2):
  """TF-IDF similarity between two documents based on pre-processed texts.
     Expects two text/document encodings.
     Input: two strings.
  """
  tfidf = getTFIDF()
  v1 = tfidf.transform(vectorizer.transform([cleantext(e1)]))
  v2 = tfidf.transform(vectorizer.transform([cleantext(e2)]))
  sim = cosine_similarity(v1,v2)
  #print('similarity({}, {}) = {}'.format(e1,e2,sim))
  return float(sim[0,0])


if __name__ == "__main__":
  # TESTS
  model = getTFIDF()
  print('IDF:')
  #print(model.idf_)
  for idx,w in enumerate(model.idf_):
    print('{:<10}: {}'.format(tokens[idx], w))

  print('\nTexts:')
  #print(texts_weights)
  for r in range(len(texts)):
    for c in range(len(tokens)):
      if texts_weights[r,c] != 0.0:
        print('{}:{} '.format(tokens[c], texts_weights[r,c]), end='')
    print('')

  print('\nText 1:')
  with open(texts[0], 'r') as ifile:
    cleantext = cleantext(ifile.read())
    counts = vectorizer.transform([cleantext])
    w = model.transform(counts)
    print(w)
    print('Should equal:')
    print(texts_weights[0])

  print('\nSimilarity:')
  print('Sim(0,1): {}'.format(cosine_similarity(texts_weights[0], texts_weights[1])))
  print('Sim(0,2): {}'.format(cosine_similarity(texts_weights[0], texts_weights[2])))

  #print(tokens)
