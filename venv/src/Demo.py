import os
import nltk
import re
from operator import itemgetter
from nltk.corpus import stopwords
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess
from allennlp.predictors.predictor import Predictor
from gensim.test.utils import common_texts
from gensim.corpora import Dictionary
from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix

def ReadTxtFiles(files):
    for fname in files:
        for line in open(fname):
            yield simple_preprocess(line)


stop_words = set(stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()
# Function for removing stop words from a given list of words. It also lemmatizes the words to ease comparisons.
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


filePath = input("Please enter the file path for the directory containing files to be examined: \n")

documents = []

# Loop through files in the directory and add them to a list
for root, directory, files in os.walk(filePath):
    for file in files:
        if '.txt' in file:
            documents.append(os.path.join(r, file))


# Generates the similarity matrix of words using the fasttext wiki-news model,
# the documents in gensim's common texts and the words from our documents.
fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
termsim_index = WordEmbeddingSimilarityIndex(fasttext_model300)
dictionary = corpora.Dictionary(ReadTxtFiles(files))
bow_corpus = [dictionary.doc2bow(document) for document in common_texts]
similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)  # construct similarity matrix
docsim_index = SoftCosineSimilarity(bow_corpus, similarity_matrix, num_best=10)


#Loads the allenNLP bidaf-elmo model
answerPredictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2018.11.30-charpad.tar.gz")


documentList = documents.copy()


def getOneAnswer(question):


def getMultiAnswer(question):




print("\nIf you would like to ask a question, just enter the question.")
print("\nIf you would like to change the similarity threshold, please enter 1 and the new threshold.")
print("\nIf you would like to change the directory being queried, please enter 2 and the new directory path.")
print("\nIf you would like to query a specific document, please enter 3 and the file path.")
print("\nIf you would like to be given one best answer, please enter 4.")
print("\nIf you would like to be given multiple answers, please enter 5.")
print("\nIf you would like to exit the program, please enter 6.")
print("\nYou can enter the word 'manual' to see the options again, when asked for your input.")


threshold = 0.4
mode = 1
keepGoing = True

#Main driver code of the program
while(keepGoing):
    userInput = input("Your Input: ")

    if(list(userInput)[0] == 1):
        threshold = list(userInput)[1]
        print("Similarity threshold changed to " + str(threshold))

    elif(list(userInput)[0] == 2):
        documentList.clear()
        for root, directory, files in os.walk(list(userInput)[1]):
            for file in files:
                if '.txt' in file:
                    documentList.append(os.path.join(r, file))
        print(documentList)

    elif(list(userInput)[0] == 3):
        documentList.clear()
        documentList.append(list(userInput)[1])
        print(documentList)

    elif(userInput == 4):
        mode = 1
        print("\nYou will only be given one answer for queries.")

    elif(userInput == 5):
        mode = 2
        print("\nYou will be given all answers with similarity above the threshold.")

    elif(userInput == 6):
        keepGoing = False

    elif(userInput == 'manual'):
        print("\nIf you would like to change the similarity threshold, please enter 1 and the new threshold.")
        print("\nIf you would like to change the directory being queried, please enter 2 and the new directory path.")
        print("\nIf you would like to query a specific document, please enter 3 and the file path.")
        print("\nIf you would like to exit the program, please enter 4.")

    elif(mode == 1):
        getOneAnswer(userInput)

    elif(mode == 2):
        getMultiAnswer(userInput)
