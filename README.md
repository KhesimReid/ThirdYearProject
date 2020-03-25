# ThirdYearProject
My final year undergraduate project at the University of Manchester. It is a question and answer system built using off the shelf libraries, for the purpose of querying legal documents.

## Installing necessary libraries
Run the following commands in the terminal:
<p>pip install nltk</p>
<p>pip install gensim</p>
<p>pip install allennlp</p>

## Running the code:
When running the code for the absolute first time the following steps must be followed:
<p>1. Run ModelLoader.py</p>
<p>2. Run MatrixLoader.py and provide the path to the 'Training Documents' directory at the input prompt.</p>
<p>3. Run demo.py and follow the prompts.</p>
<br>
<p>On subsequent runs of demo.py, you will only need to complete step 3 above.</p>

## Updating Similarity Matrix:
The similarity matrix can be updated by running MatrixLoader.py and providing the path to the directory you wish to build the matrix with. 
