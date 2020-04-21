import unittest
import src.QAFunctions


class MyTestCase(unittest.TestCase):

    # Test for the regular case stop word removal
    def test_stopword_removal(self):
        testSentence = "the quick brown fox ran past the lazy dog"
        expectedOutput = ['quick', 'brown', 'fox', 'ran', 'past', 'lazy', 'dog']
        tokens = testSentence.split()
        filteredText = src.QAFunctions.removeStopWords(tokens)
        self.assertEqual(expectedOutput, filteredText)

    # Test for stop word removal when the entire list is stop words
    def test_stopword_removal_all_stops(self):
        tokens = ['the', 'and', 'of', 'on', 'is', 'am', 'if', 'as']
        filteredText = src.QAFunctions.removeStopWords(tokens)
        self.assertEqual([], filteredText)

    # Test that the empty list is unchanged
    def test_stop_word_removal_empty(self):
        tokens = []
        filteredText = src.QAFunctions.removeStopWords(tokens)
        self.assertEqual(tokens, filteredText)

    # Test for lower case
    def test_case_fold(self):
        tokens = ['My', 'name', 'is', 'Khesim', 'Reid', 'and', 'this', 'is', 'my', 'Third', 'Year', 'Project.']
        foldedText = src.QAFunctions.caseFold(tokens)
        expectedOutput = ['my', 'name', 'is', 'khesim', 'reid', 'and', 'this', 'is', 'my', 'third', 'year', 'project.']
        self.assertEqual(expectedOutput, foldedText)

    # Tests for all words already lower case
    def test_case_fold_all_lower(self):
        tokens = ['legal', 'discovery', 'is', 'a', 'pre-trial', 'process', 'whereby', 'opposing', 'sides', 'exchange', 'documents', 'relevant', 'to', 'the', 'case.']
        foldedText = src.QAFunctions.caseFold(tokens)
        self.assertEqual(tokens, foldedText)

    # Test for all words being completely upper case
    def test_case_fold_all_upper(self):
        tokens = ['ABRSM', 'SQL', 'AXA', 'SXM', 'USA', 'UK']
        foldedText = src.QAFunctions.caseFold(tokens)
        expectedOutput = ['abrsm', 'sql', 'axa', 'sxm', 'usa', 'uk']
        self.assertEqual(foldedText, expectedOutput)

    # Test for punctuation removal
    def test_punctuation_removal(self):
        sentence = "When I went to the store I spoke to my Mom's friend, Wendy, who's also my bestfriend's aunt."
        expectedOutput = "When I went to the store I spoke to my Moms friend Wendy whos also my bestfriends aunt"
        alteredText = src.QAFunctions.removePunctuation(sentence)
        self.assertEqual(expectedOutput, alteredText)

    # Test for string of just punctuation
    def test_punctuation_removal_only_punctuation(self):
        text = "./'?!"
        alteredText = src.QAFunctions.removePunctuation(text)
        self.assertEqual("", alteredText)

    # Test for string with no punctuation
    def test_punctuation_removal_no_punctuation(self):
        sentence = "She sells seashells by the seashore"
        alteredText = src.QAFunctions.removePunctuation(sentence)
        self.assertEqual(sentence, alteredText)

    # Test for lemmatising
    def test_lemmatisation(self):
        tokens = [ 'running', 'stealing', 'exercising', 'kicks', 'broke']
        expectedOutput = ['run', 'steal', 'exercise', 'kick', 'break']
        lemmatisedText = src.QAFunctions.stemWords(tokens)
        self.assertEqual(expectedOutput, lemmatisedText)

    # Test that lemmatising punctuation does not do anything to the text
    def test_lemmatisation_only_symbols(self):
        text = ['./?!,']
        lemmatisedText = src.QAFunctions.stemWords(text)
        self.assertEqual(text, lemmatisedText)

    # Test for lemmatising empty string
    def test_lemmatisation_empty_string(self):
        tokens = []
        lemmatisedText = src.QAFunctions.stemWords(tokens)
        self.assertEqual([], lemmatisedText)



if __name__ == '__main__':
    unittest.main()
