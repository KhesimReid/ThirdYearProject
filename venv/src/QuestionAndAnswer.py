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
from allennlp.predictors.predictor import Predictor


from gensim.test.utils import common_texts
from gensim.corpora import Dictionary
from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix

# absolutePath = input("Please enter the file path for the directory containing files to be examined: \n")

files = []
absolutePath = '/Users/khesim/PycharmProjects/ThirdYearProject/venv/Licenses/'

for r, d, f in os.walk(absolutePath):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))


def ReadTxtFiles(files):
    for fname in files:
        for line in open(fname):
            yield simple_preprocess(line)


# dictionary = corpora.Dictionary(ReadTxtFiles(files))
#dictionary = corpora.Dictionary(ReadTxtFiles(absolutePath))

# fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')

# Prepare the similarity matrix
# similarity_matrix = fasttext_model300.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0, nonzero_limit=100)


fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
# model = Word2Vec(fasttext_model300, size=20, min_count=1)  # train word-vectors
termsim_index = WordEmbeddingSimilarityIndex(fasttext_model300)
dictionary = corpora.Dictionary(ReadTxtFiles(files))
bow_corpus = [dictionary.doc2bow(document) for document in common_texts]
similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)  # construct similarity matrix
docsim_index = SoftCosineSimilarity(bow_corpus, similarity_matrix, num_best=10)



stop_words = set(stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()

answerPredictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2018.11.30-charpad.tar.gz")

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

keepGoing = True
# while loop to start here
while(keepGoing):
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
            # cosVal = softcossim(tokenVec, questionVec, similarity_matrix)
            cosVal = similarity_matrix.inner_product(tokenVec, questionVec)

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

    for key in relevantParagraphs.keys():
        print("File: " + key + "\n")
        for para in relevantParagraphs.get(key):
            result = answerPredictor.predict(passage=para, question=rawQuestion)
            print('-> ' + result['best_span_str'])
        print("-------------------------------------------------")

    moreQuestions = input("Please enter 'y' if you have more questions. Enter any other input if you do not: \n" )

    if moreQuestions != 'y':
        keepGoing = False


# while loop to end here