import os
import pickle
import QAFunctions
from gensim import corpora
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix

allFilesPath = input("Enter the path to the directory you wish to build the similarity matrix with: ")

documents = []

# Loop through files in the directory and add them to a list
for root, directory, files in os.walk(allFilesPath):
    for file in files:
        if '.txt' in file:
            documents.append(os.path.join(root, file))


# Generates the similarity matrix of words using the fasttext wiki-news model,
# the documents in gensim's common texts and the words from our documents.
print("1. Loading Fasttext model...")
fasttext_model300 = pickle.load(open('fasttext_model.sav', 'rb'))
termsim_index = WordEmbeddingSimilarityIndex(fasttext_model300)
print("2. Reading files in directory...")
dictionary = corpora.Dictionary(QAFunctions.ReadTxtFiles(documents))
print("3. Generating similarity matrix...")
similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)  # Construct similarity matrix

# Save similarity matrix to a file to be reused without long load time.
pickle.dump(similarity_matrix, open('similarity_matrix.sav', 'wb'))
print("4. Matrix Saved!")



