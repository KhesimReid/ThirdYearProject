import os
import nltk
import re
from nltk.corpus import stopwords
from gensim import corpora
from gensim.matutils import softcossim
from gensim.models import WordEmbeddingSimilarityIndex
import gensim.downloader as api
from gensim.utils import simple_preprocess
from smart_open import smart_open
from pathlib import Path

# class ReadTxtFiles(object):
#     def __init__(self, dirname):
#         self.dirname = dirname
#
#     def __iter__(self):
#         for fname in os.listdir(self.dirname):
#             for line in open(os.path.join(self.dirname, fname), encoding='latin'):
#                 yield simple_preprocess(line)


# notebook_path = os.path.abspath("Third Year Project - First Iteration.ipynb")
# fileDirectory = os.path.join(os.path.dirname(notebook_path), "Licenses")
# #fileDirectory = "~/Users/khesim/ThirdYearProject/Licenses"

# directory = input("Please enter the file path for the directory containing files to be examined: \n")
# print("\n")
files = ['Echelon.txt', 'Microsoft.txt', 'Oracle.txt', 'PDF Technologies.txt']

# absolutePath = '~/PycharmProjects/ThirdYearProject/venv/Licenses/'

def ReadTxtFiles(files):
    for fname in files:
        for line in open(fname):
            yield simple_preprocess(line)


dictionary = corpora.Dictionary(ReadTxtFiles(files))
#dictionary = corpora.Dictionary(ReadTxtFiles(absolutePath))

fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')

# Prepare the similarity matrix
similarity_matrix = fasttext_model300.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0,
                                                        nonzero_limit=100)

stop_words = set(stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()


# Functions for removing stop words from a given list of words. It also lemmatizes the words to ease comparisons.
def removeStopWords(text):
    filteredText = []
    for word in text:
        if not word in stop_words:
            filteredText.append(lemmatizer.lemmatize(word.lower()))

    return filteredText


# Function for removing punctuation from a string
def removePunctuation(text):
    filteredText = re.sub(r'[^\w\s]', '', text)
    return filteredText


rawQuestion = input("Please enter the question you need answered: \n")
print("\n")

threshold = 0.35
question = removePunctuation(rawQuestion)
questionWords = question.split()
questionWords = removeStopWords(questionWords)

relevantParagraphs = dict()  # Dictionary to store files and the paragraphs which were sufficiently relevant

# PARAGRAPH SELECTION SECTION
# Loop through all files in directory
# for file in os.listdir(directory):
for file in files:
    # fileName = file.name
    fileName = file
    textFile = open(file, "r")

    paragraphs = textFile.readlines()
    updatedList = []

    for paragraph in paragraphs:
        cleanedText = removePunctuation(paragraph)
        tokens = cleanedText.split()
        tokens = removeStopWords(tokens)

        allText = [tokens, questionWords]

        # Prepare a dictionary and a corpus.
        dictionary = corpora.Dictionary(allText)

        # doc2bow for bag of words vectors
        tokenVec = dictionary.doc2bow(tokens)
        questionVec = dictionary.doc2bow(questionWords)
        cosVal = softcossim(tokenVec, questionVec, similarity_matrix)

        # print(cosVal)
        # cosVal = calculateCosSim(tokens, questionWords)

        if (cosVal >= threshold):
            if (fileName in relevantParagraphs):
                currentList = relevantParagraphs.get(fileName)
                updatedList.append(paragraph)
                newEntry = {fileName: updatedList}
                relevantParagraphs.update(newEntry)
            else:
                firstPara = []
                firstPara.append(paragraph)
                newEntry = {fileName: firstPara}
                relevantParagraphs.update(newEntry)

print("\n")
for key in relevantParagraphs.keys():
    print("File: " + key)
    print("\n")
    for para in relevantParagraphs.get(key):
        print(para)
        print("\n\n")

# Questions where the right answer is in the output:
# How many days notice must a user give before cancelling with PDF Technologies?
# What country is covered by the Oracle agreement?
# In what event is Echelon liable for damages?