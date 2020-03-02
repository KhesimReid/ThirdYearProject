import pickle
import gensim.downloader as api

model = api.load('fasttext-wiki-news-subwords-300')
filename = 'fasttext_model.sav'
pickle.dump(model, open(filename, 'wb'))
print("Model Saved!")