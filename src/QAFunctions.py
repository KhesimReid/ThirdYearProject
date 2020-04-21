import nltk
from nltk.corpus import wordnet
import re
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess



stop_words = set(stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()

# Function for getting text from all files in the given list.
def ReadTxtFiles(files):
    for fname in files:
        for line in open(fname):
            yield simple_preprocess(line)


# Function for removing stop words from a given list of words.
# It also lemmatizes and case folds the words to ease comparisons.
# def removeStopWords(text):
#     filteredText = []
#     for word in text:
#         if not word in stop_words:
#             filteredText.append(lemmatizer.lemmatize(word.lower()))
#
#     return filteredText


def removeStopWords(text):
    stop_words = set(stopwords.words('english'))
    filteredText = []
    for word in text:
        if not word in stop_words:
            filteredText.append(word)

    return filteredText


def stemWords(text):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    taggedText = nltk.pos_tag(text)
    stemmedText = []

    for tag in taggedText:
        if tag[1].startswith('J'):
            wordnet_tag = wordnet.ADJ
        elif tag[1].startswith('V'):
            wordnet_tag = wordnet.VERB
        elif tag[1].startswith('N'):
            wordnet_tag = wordnet.NOUN
        elif tag[1].startswith('R'):
            wordnet_tag = wordnet.ADV

        stemmedText.append(lemmatizer.lemmatize(tag[0], wordnet_tag))

    return stemmedText

    # lemmatizer = nltk.stem.WordNetLemmatizer()
    # taggedText = nltk.pos_tag(text)
    # stemmedText = []
    # for word, tag in taggedText:
    #     stemmedText.append(lemmatizer.lemmatize(word, tag))
    #
    # return stemmedText


def caseFold(text):
    foldedText = []
    for word in text:
        foldedText.append(word.lower())

    return foldedText


# Function for removing punctuation from a string.
def removePunctuation(text):
    filteredText = re.sub(r'[^\w\s]', '', text)
    return filteredText


# Function for decluttering output
def clearScreen():
    print("\n" * 100)


