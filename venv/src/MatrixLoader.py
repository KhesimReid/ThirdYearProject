import os
import pickle
import gensim.downloader as api
import QAFunctions
from gensim import corpora
from allennlp.predictors.predictor import Predictor
from gensim.test.utils import common_texts
from gensim.corpora import Dictionary
from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix

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
similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)  # construct similarity matrix


pickle.dump(similarity_matrix, open('similarity_matrix.sav', 'wb'))
print("4. Matrix Saved!")



