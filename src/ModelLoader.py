import pickle
import gensim.downloader as api

# Load the fasttext model then save it to a file, so that it can be used without having to load each time.
print("Saving Fasttext Model...")
model = api.load('fasttext-wiki-news-subwords-300')
filename = 'fasttext_model.sav'
pickle.dump(model, open(filename, 'wb'))
print("Model Saved!")