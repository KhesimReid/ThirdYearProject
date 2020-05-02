# ThirdYearProject
My final year undergraduate project at the University of Manchester. It is a question and answer system built using off the shelf libraries, for the purpose of querying legal documents.

## Installing necessary libraries
Run the following command in the terminal:
<p>pip3 install -r requirements.txt </p>

Next, in order to get the necessary list of nltk stopwords, complete the following steps:
<p>1. Open the Python interpreter by typing 'python' in the terminal.</p>
<p>2. Enter command "import nltk".</p>
<p>3. Enter command "nltk.download('stopwords')"</p>
<p>4. Close the Python interpreter.</p>

## Running the code:
When running the code for the absolute first time the following steps must be followed:
<p>1. Enter command "cd src"</p>
<p>2. Enter command "python3 ModelLoader.py"</p>
<p>3. Enter command "python3 MatrixLoader.py" and provide the path "../Training Documents" at the input prompt. (This may take a few hours to run, in order to build the similarity matrix.)</p>
<p>4. Enter command "python3 FinalProject.py" and follow the prompts.</p>
<br>
<p>On subsequent runs of FinalProject.py, you will only need to complete step 4 above.</p>

## Updating Similarity Matrix:
The similarity matrix can be updated by running MatrixLoader.py and providing the path to the directory you wish to build the matrix with. 
