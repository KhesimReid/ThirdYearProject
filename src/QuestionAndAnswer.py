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

filePath = input("Please enter the file path for the directory containing files to be examined: \n")

files = []


for r, d, f in os.walk(filePath):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))


def ReadTxtFiles(files):
    for fname in files:
        for line in open(fname):
            yield simple_preprocess(line)


fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
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


threshold = 0.37

keepGoing = True
while(keepGoing):
    allFilesBool = input("Would you like to query all files? Please enter 'y' for yes or any other character for no. \n")

    if(allFilesBool == 'y'):
        rawQuestion = input("Please enter the question you need answered: \n")

        question = removePunctuation(rawQuestion)
        questionWords = question.split()
        questionWords = removeStopWords(questionWords)

        relevantParagraphs = dict()  # Dictionary to store files and the paragraphs which were sufficiently relevant

        # PARAGRAPH SELECTION SECTION
        # Loop through all files in directory
        for file in files:
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
                simVal = similarity_matrix.inner_product(tokenVec, questionVec, normalized=True)

                if (simVal >= threshold):
                    if (fileName in relevantParagraphs):
                        currentList = relevantParagraphs.get(fileName)
                        pair = [simVal, paragraph]
                        updatedList.append(pair)
                        newEntry = {fileName: updatedList}
                        relevantParagraphs.update(newEntry)
                    else:
                        firstPara = []
                        pair = [simVal, paragraph]
                        firstPara.append(pair)
                        newEntry = {fileName: firstPara}
                        relevantParagraphs.update(newEntry)

                # if (simVal >= threshold):
                #     if (fileName in relevantParagraphs):
                #         currentList = relevantParagraphs.get(fileName)
                #         updatedList.append(paragraph)
                #         newEntry = {fileName: updatedList}
                #         relevantParagraphs.update(newEntry)
                #     else:
                #         firstPara = []
                #         firstPara.append(paragraph)
                #         newEntry = {fileName: firstPara}
                #         relevantParagraphs.update(newEntry)


        # for key in relevantParagraphs.keys():
        #     print("File: " + key + "\n")
        #     for para in relevantParagraphs.get(key):
        #         result = answerPredictor.predict(passage=para, question=rawQuestion)
        #         print(para)
        #         print("\n")
        #         print('-> ' + result['best_span_str'])
        #         print("\n")
        #     print("-------------------------------------------------")

        for key in relevantParagraphs.keys():
            print("File: " + key + "\n")
            documentParas = sorted(relevantParagraphs.get(key), key=itemgetter(0), reverse=True)
            for para in documentParas:
                result = answerPredictor.predict(passage=para[1], question=rawQuestion)
                print("Similarity Score: "+ str(para[0]) + "\n")
                print("Paragraph:\n" + para[1])
                print('Answer Span:\n' + result['best_span_str'])
                print("\n")
            print("-------------------------------------------------")

        moreQuestions = input("Please enter 'y' if you have more questions. Enter any other input if you do not: \n")

        if moreQuestions != 'y':
            keepGoing = False

    else:
        queryFile = input("Please enter the path of the file you wish to query: \n")
        rawQuestion = input("Please enter the question you need answered: \n")

        question = removePunctuation(rawQuestion)
        questionWords = question.split()
        questionWords = removeStopWords(questionWords)

        relevantParagraphs = []

        textFile = open(queryFile, "r")

        paragraphs = textFile.readlines()

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
            simVal = similarity_matrix.inner_product(tokenVec, questionVec, normalized=True)


        #     if (simVal >= threshold):
        #         relevantParagraphs.append(paragraph)

            if (simVal >= threshold):
                pair = [simVal, paragraph]
                relevantParagraphs.append(pair)

        relevantParagraphs = sorted(relevantParagraphs, key=itemgetter(0))
        for para in relevantParagraphs:
            result = answerPredictor.predict(passage=para[1], question=rawQuestion)
            print("Similarity Score: " + str(para[0]) + "\n")
            print("Paragraph:\n" + para[1])
            print('Answer Span:\n' + result['best_span_str'])
            print("\n")
        print("-------------------------------------------------")

        # for para in relevantParagraphs:
        #     result = answerPredictor.predict(passage=para, question=rawQuestion)
        #     print(para)
        #     print("\n")
        #     print('-> ' + result['best_span_str'])
        #     print("\n")
        # print("-------------------------------------------------")

        moreQuestions = input("Please enter 'y' if you have more questions. Enter any other input if you do not: \n")

        if moreQuestions != 'y':
            keepGoing = False