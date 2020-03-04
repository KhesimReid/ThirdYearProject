import os
import pickle
import QAFunctions
from operator import itemgetter
from gensim import corpora
from allennlp.predictors.predictor import Predictor
from gensim.corpora import Dictionary
from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix


similarity_matrix = pickle.load(open('similarity_matrix.sav', 'rb'))

# Loads the allenNLP bidaf-elmo model
print("Loading AllenNLP Predictor Model...\n")

answerPredictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2018.11.30-charpad.tar.gz")

QAFunctions.clearScreen()
filePath = '../Documents'

documentList = []

# Loop through files in the directory and add them to a list
for root, directory, files in os.walk(filePath):
    for file in files:
        if '.txt' in file:
            documentList.append(os.path.join(root, file))


def getMultiAnswer(rawQuestion, documents):
    question = QAFunctions.removePunctuation(rawQuestion)
    questionWords = question.split()
    questionWords = QAFunctions.removeStopWords(questionWords)

    # Dictionary to store files and the paragraphs which were sufficiently relevant
    relevantParagraphs = dict()
    # PARAGRAPH SELECTION SECTION
    # Loop through all files in directory
    for document in documents:
        documentName = document

        try:
            textFile = open(document, "r")
        except FileNotFoundError:
            print("It appears the directory or document you have entered does not exist.")
            print("Please enter the option to change file or directory, then carefully enter the name of the file and try again.")
            return

        paragraphs = textFile.readlines()
        textFile.close()
        updatedList = []

        for paragraph in paragraphs:
            cleanedText = QAFunctions.removePunctuation(paragraph)
            tokens = cleanedText.split()
            tokens = QAFunctions.removeStopWords(tokens)

            allText = [tokens, questionWords]

            # Prepare a dictionary and a corpus.
            dictionary = corpora.Dictionary(allText)

            # doc2bow for bag of words vectors
            tokenVec = dictionary.doc2bow(tokens)
            questionVec = dictionary.doc2bow(questionWords)
            simVal = similarity_matrix.inner_product(tokenVec, questionVec, normalized=True)
            # Using inner product with normalisation has same effect as soft cosine similarity

            if simVal >= threshold:
                if documentName in relevantParagraphs:
                    updatedList = relevantParagraphs.get(documentName)
                    pair = [simVal, paragraph]
                    updatedList.append(pair)
                    newEntry = {documentName: updatedList}
                    relevantParagraphs.update(newEntry)
                else:
                    firstPara = []
                    pair = [simVal, paragraph]
                    firstPara.append(pair)
                    newEntry = {documentName: firstPara}
                    relevantParagraphs.update(newEntry)

    if len(relevantParagraphs) == 0:
        print("SORRY! NO ANSWER AVAILABLE.")
        print("Perhaps you can try asking again with a lower threshold value.")
        print("-------------------------------------------------\n")
    else:
        for key in relevantParagraphs.keys():
            print("File: " + key + "\n")
            documentParas = sorted(relevantParagraphs.get(key), key=itemgetter(0), reverse=True)
            for para in documentParas:
                result = answerPredictor.predict(passage=para[1], question=rawQuestion)
                print("Similarity Score: " + str(para[0]) + "\n")
                print("Paragraph:\n")
                start = 0
                lineLength = len(para[1])
                while lineLength - start >= 185:
                    print(para[1][start:start + 185])
                    start += 185
                print(para[1][start:])

                print('Answer Span:\n' + result['best_span_str'])
                print("\n")
            print("-------------------------------------------------")
        print("END OF OUTPUT\n")


def getBestAnswer(rawQuestion, documents):
    question = QAFunctions.removePunctuation(rawQuestion)
    questionWords = question.split()
    questionWords = QAFunctions.removeStopWords(questionWords)

    # Dictionary to store files and the paragraphs which were sufficiently relevant
    relevantParagraphs = dict()

    # PARAGRAPH SELECTION SECTION
    # Loop through all files in directory
    for document in documents:
        documentName = document
        try:
            textFile = open(document, "r")
        except FileNotFoundError:
            print("It appears the directory or document you have entered does not exist.")
            print("Please enter the option to change file or directory, then carefully enter the name of the file and try again.")
            return


        paragraphs = textFile.readlines()
        textFile.close()
        updatedList = []

        for paragraph in paragraphs:
            cleanedText = QAFunctions.removePunctuation(paragraph)
            tokens = cleanedText.split()
            tokens = QAFunctions.removeStopWords(tokens)

            allText = [tokens, questionWords]

            # Prepare a dictionary and a corpus.
            dictionary = corpora.Dictionary(allText)

            # doc2bow for bag of words vectors
            tokenVec = dictionary.doc2bow(tokens)
            questionVec = dictionary.doc2bow(questionWords)
            simVal = similarity_matrix.inner_product(tokenVec, questionVec, normalized=True)

            if simVal >= threshold:
                if documentName in relevantParagraphs:
                    updatedList = relevantParagraphs.get(documentName)
                    updatedList.append(paragraph)
                    newEntry = {documentName: updatedList}
                    relevantParagraphs.update(newEntry)
                else:
                    firstPara = []
                    firstPara.append(paragraph)
                    newEntry = {documentName: firstPara}
                    relevantParagraphs.update(newEntry)

    if len(relevantParagraphs) == 0:
        print("SORRY! NO ANSWERS AVAILABLE.")
        print("Perhaps you can try asking again with a lower threshold value.")
        print("-------------------------------------------------\n")
    else:
        for key in relevantParagraphs.keys():
            print("File: " + key + "\n")
            documentParas = relevantParagraphs.get(key)
            joinedText = ''.join(documentParas)

            result = answerPredictor.predict(passage=joinedText, question=rawQuestion)
            print("Best Answer: " + result['best_span_str'])
            print("-------------------------------------------------")
        print("END OF OUTPUT\n")


threshold = 0.4
mode = "Best Answer"
keepGoing = True

# QAFunctions.clearScreen()
print("\nDefault Values: \nThreshold: " + str(threshold) + "\nMode: " + mode + "\nDocuments Queried: " + filePath)
print("\n")
print("INSTRUCTIONS")
print("If you would like to ask a question, just enter the question.")
print("If you would like to change the similarity threshold, please enter 1.")
print("If you would like to change the directory being queried, please enter 2.")
print("If you would like to query a specific document, please enter 3.")
print("If you would like to be given one best answer per document, please enter 4.")
print("If you would like to be given multiple answers, please enter 5.")
print("If you would like to exit the program, please enter 6.")
print("You can enter the word 'stats' to see the current settings, when asked for your input.")
print("You can enter the word 'manual' to see the options again, when asked for your input.\n\n")
# input("Press enter when you are ready to begin.")


# Main driver code of the program
while keepGoing:
    # QAFunctions.clearScreen()
    userInput = input("Your Input: ")

    if userInput == '1':
        threshold = float(input("Please enter new threshold (number between 0 and 1): "))
        print("Similarity threshold changed to " + str(threshold))

    elif userInput == '2':
        documentList.clear()
        filePath = input("Please enter the file path of the directory: ")
        filesInDir = os.walk(filePath)  # NEED TRY CATCH HERE FOR os.walk
        # for root, directory, files in os.walk(filePath):
        for root, directory, files in filesInDir:
            for file in files:
                if '.txt' in file:
                    documentList.append(os.path.join(root, file))
        print("You are now querying " + filePath)


    elif userInput == '3':
        documentList.clear()
        filePath = input("Please enter the file path of the document: ")
        documentList.append(filePath)
        print("You are now querying " + filePath)

    elif userInput == '4':
        mode = "Best Answer"
        print("You will only be given one answer for queries.\n")

    elif userInput == '5':
        mode = "Multi Answer"
        print("You will be given answers from all paragraphs with similarity above the threshold.\n")

    elif userInput == '6':
        keepGoing = False

    elif userInput == 'manual':
        print("INSTRUCTIONS")
        print("If you would like to ask a question, just enter the question.")
        print("If you would like to change the similarity threshold, please enter 1.")
        print("If you would like to change the directory being queried, please enter 2.")
        print("If you would like to query a specific document, please enter 3.")
        print("If you would like to be given one best answer per document, please enter 4.")
        print("If you would like to be given multiple answers, please enter 5.")
        print("If you would like to exit the program, please enter 6.")
        print("You can enter the word 'stats' to see the current settings, when asked for your input.")

    elif userInput == "stats":
        print("Mode: " + mode)
        print("Threshold: " + str(threshold))
        print("Documents: " + str(documentList))

    elif mode == "Best Answer":
        getBestAnswer(userInput, documentList)

    elif mode == "Multi Answer":
        getMultiAnswer(userInput, documentList)

