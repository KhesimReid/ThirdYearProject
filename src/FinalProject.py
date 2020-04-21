import os
import pickle
import QAFunctions
from operator import itemgetter
from gensim import corpora
from allennlp.predictors.predictor import Predictor

# Load the similarity matrix from the saved file.
similarity_matrix = pickle.load(open('similarity_matrix.sav', 'rb'))

# Load the allenNLP model
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


# Function for returning answers
def getAnswers(rawQuestion, documents, chosenMode):
    # Pre-process the user's question.
    question = QAFunctions.removePunctuation(rawQuestion)
    questionWords = question.split()
    questionWords = QAFunctions.caseFold(questionWords)
    questionWords = QAFunctions.removeStopWords(questionWords)
    questionWords = QAFunctions.stemWords(questionWords)

    # Dictionary to store files and the paragraphs which were sufficiently relevant
    relevantParagraphs = dict()

    # Loop through all files in directory
    for document in documents:
        documentName = document

        # Ensure that the file exists before proceeding.
        try:
            textFile = open(document, "r")
        except FileNotFoundError:
            print("It appears the directory or document you have entered does not exist.")
            print(
                "Please enter the option to change file or directory, then carefully enter the name of the file and try again.")
            return

        # Split document text into separate paragraphs
        paragraphs = textFile.readlines()
        textFile.close()
        updatedList = []

        # Loop through the paragraphs
        for paragraph in paragraphs:
            # Pre-process the paragraph string
            cleanedText = QAFunctions.removePunctuation(paragraph)
            tokens = cleanedText.split()
            tokens = QAFunctions.caseFold(tokens)
            tokens = QAFunctions.removeStopWords(tokens)
            tokens = QAFunctions.stemWords(tokens)

            allText = [tokens, questionWords]

            # Prepare a dictionary from the paragraph and question words.
            dictionary = corpora.Dictionary(allText)

            # Convert the paragraph words and question words to bag of words vectors
            tokenVec = dictionary.doc2bow(tokens)
            questionVec = dictionary.doc2bow(questionWords)

            # Using inner product with normalisation has same effect as soft cosine similarity.
            simVal = similarity_matrix.inner_product(questionVec, tokenVec, normalized=True)

            # Update the dictionary each time a paragraph is greater than or equal the similarity threshold.
            if simVal >= threshold:
                # For multi-answer mode, also show simVal to know which paragraph is most similar to question.
                # So dictionary entry is list of lists (pairs).
                if chosenMode == 'Multi Answer':
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

                # Only need paragraphs for single answer mode so dictionary entry is list of paragraphs.
                else:
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
        print("SORRY! NO ANSWER AVAILABLE.")
        print("Perhaps you can try asking again with a lower threshold value.")
        print("-------------------------------------------------\n")
    else:
        getSpan(relevantParagraphs, rawQuestion, chosenMode)


# Function for identifying the answer span in the paragraphs.
def getSpan(relevantParagraphs, rawQuestion, chosenMode):
    if chosenMode == 'Multi Answer':
        # Loop through files which contain relevant paragraphs.
        for key in relevantParagraphs.keys():
            print("File: " + key + "\n")

            # Sort the paragraphs so that the most similar ones are shown first.
            documentParas = sorted(relevantParagraphs.get(key), key=itemgetter(0), reverse=True)

            # Print out each paragraph with the answer span highlighted.
            for i, para in enumerate(documentParas, 1):
                result = answerPredictor.predict(passage=para[1], question=rawQuestion)
                answer = result['best_span_str']
                print("Answer #{}".format(i))
                print("Similarity Score: " + str(para[0]) + "\n")

                para[1] = para[1].replace(answer, '\033[44;33m{}\033[m'.format(answer))
                start = 0
                lineLength = len(para[1])
                while lineLength - start >= 185:
                    print(para[1][start:start + 185])
                    start += 185
                print(para[1][start:])

            print("-------------------------------------------------")
        print("END OF OUTPUT\n")

    else:
        # Loop through files which contain relevant paragraphs.
        for key in relevantParagraphs.keys():
            print("File: " + key + "\n")
            documentParas = relevantParagraphs.get(key)
            joinedText = '\n'.join(documentParas) # Put relevant paragraphs into one string to get better answer span.

            result = answerPredictor.predict(passage=joinedText, question=rawQuestion)
            print("Best Answer: ")
            answer = result['best_span_str']

            # Split the joined paragraphs and print the one which contains the answer.
            paras = joinedText.splitlines()
            for para in paras:
                if answer in para:
                    answerPara = para.replace(answer, '\033[44;33m{}\033[m'.format(answer))
                    break

            start = 0
            lineLength = len(answerPara)
            while lineLength - start >= 185:
                print(answerPara[start:start + 185])
                start += 185
            print(answerPara[start:])

            print("-------------------------------------------------\n")
        print("END OF OUTPUT\n")


# Default Settings
threshold = 0.4
mode = "Best Answer"
keepGoing = True

print("\nDefault Values: \nThreshold: " + str(threshold) + "\nMode: " + mode + "\nDocuments Queried: " + filePath)
print("\n")
print("INSTRUCTIONS")
print("If you would like to ask a question, just enter the question.")
print("If you would like to change the similarity threshold, please enter 1.")
print("If you would like to change the directory being queried, please enter 2.")
print("If you would like to query a specific document, please enter 3.")
print("If you would like to be given the best answer per document, please enter 4.")
print("If you would like to be given multiple possible answers, please enter 5.")
print("If you would like to exit the program, please enter 6.")
print("You can enter the word 'stats' to see the current settings, when asked for your input.")
print("You can enter the word 'manual' to see the options again, when asked for your input.\n\n")


# Main driver code of the program
while keepGoing:
    userInput = input("Your Input: ")

    # Change similarity threshold.
    if userInput == '1':
        threshold = float(input("Please enter new threshold (number between 0 and 1): "))
        print("Similarity threshold changed to " + str(threshold))

    # Change directory to be queried.
    elif userInput == '2':
        documentList.clear()
        filePath = input("Please enter the file path of the directory: ")
        filesInDir = os.walk(filePath)
        for root, directory, files in filesInDir:
            for file in files:
                if '.txt' in file:
                    documentList.append(os.path.join(root, file))
        print("You are now querying " + filePath)

    # Change document to be queried.
    elif userInput == '3':
        documentList.clear()
        filePath = input("Please enter the file path of the document: ")
        documentList.append(filePath)
        print("You are now querying " + filePath)

    # Change mode to Best Answer mode.
    elif userInput == '4':
        mode = "Best Answer"
        print("You will only be given one answer for queries.\n")

    # Change mode to Multi Answer mode.
    elif userInput == '5':
        mode = "Multi Answer"
        print("You will be given answers from all paragraphs with similarity above the threshold.\n")

    # Terminate program.
    elif userInput == '6':
        keepGoing = False

    # Display instructions.
    elif userInput == 'manual':
        print("INSTRUCTIONS")
        print("If you would like to ask a question, just enter the question.")
        print("If you would like to change the similarity threshold, please enter 1.")
        print("If you would like to change the directory being queried, please enter 2.")
        print("If you would like to query a specific document, please enter 3.")
        print("If you would like to be given the best answer per document, please enter 4.")
        print("If you would like to be given multiple possible answers, please enter 5.")
        print("If you would like to exit the program, please enter 6.")
        print("You can enter the word 'stats' to see the current settings.")

    # Display current system stats.
    elif userInput == "stats":
        print("Mode: " + mode)
        print("Threshold: " + str(threshold))
        print("Documents: " + str(documentList))

    # Get answer for question.
    else:
        getAnswers(userInput, documentList, mode)


