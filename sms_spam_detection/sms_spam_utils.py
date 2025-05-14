# Some of these functions were taken from an older project for programming 2 compulsory 2 made by group 20, refer to the sources for more information.

import string
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from collections import Counter

# Tokenizing input sentence and converting to lowercase
def tokenize(sentence):
    return word_tokenize(sentence.lower())

# Mapping POS tags to WordNet POS formats for lemmatization
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN # Default to noun

# Lemmatizing words with POS tags, removing stop words and non-alphanumeric tokens
def lem(words, pos_tags):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    return [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in zip(words, pos_tags) if word.isalnum() and word not in stop_words] # Keep alphanumeric non-stop words

# Counting the number of words in text after tokenization and lemmatization
def count_words(text):
    tokens = tokenize(text)
    pos_tags = [tag for _, tag in nltk.pos_tag(tokens)]
    lemmatized = lem(tokens, pos_tags)
    return len(lemmatized)

# Getting the top 10 most frequent words from a list of texts
def get_word_freq(texts):
    all_words = []
    for text in texts:
        tokens = tokenize(text)
        pos_tags = [tag for _, tag in nltk.pos_tag(tokens)]
        lemmatized = lem(tokens, pos_tags)
        all_words.extend(lemmatized)
    return Counter(all_words).most_common(10) # Return top 10

# Counting punctuation characters in the text
def count_punctuation(text):
    return sum(1 for char in text if char in string.punctuation)

# Counting uppercase words longer than one character
def count_uppercase_words(text):
    words = text.split()
    return sum(1 for word in words if word.isupper() and len(word) > 1) # Sum punctuation occurrences

# Preprocessing text by removing special characters and applying tokenization, POS tagging, and lemmatization
def preprocess_text(text):
    text = text.replace('>', '').replace('<', '') # Remove special characters
    tokens = tokenize(text)
    pos_tags = [tag for _, tag in nltk.pos_tag(tokens)]
    lemmatized = lem(tokens, pos_tags)
    return ' '.join(lemmatized) # Return processed text
