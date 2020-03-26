import nltk
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
def removeStopWords(text):
    filteredText = []
    for word in text:
        if not word in stop_words:
            filteredText.append(lemmatizer.lemmatize(word.lower()))

    return filteredText


# Function for removing punctuation from a string.
def removePunctuation(text):
    filteredText = re.sub(r'[^\w\s]', '', text)
    return filteredText


# Function for decluttering output
def clearScreen():
    print("\n" * 100)


