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
print("Loading Fasttext model...")
fasttext_model300 = pickle.load(open('fasttext_model.sav', 'rb'))
print("here")
termsim_index = WordEmbeddingSimilarityIndex(fasttext_model300)
print("here2")
dictionary = corpora.Dictionary(QAFunctions.ReadTxtFiles(documents))
print("here3")
# bow_corpus = [dictionary.doc2bow(document) for document in common_texts]
# print("here4")
print("Generating similarity matrix...")
similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)  # construct similarity matrix
print("here5")
# docsim_index = SoftCosineSimilarity(bow_corpus, similarity_matrix, num_best=10)
# print("here6")

pickle.dump(similarity_matrix, open('similarity_matrix.sav', 'wb'))
print("Matrix Saved!")



