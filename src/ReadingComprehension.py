from allennlp.predictors.predictor import Predictor

# class PythonPredictor:
#     def __init__(self, config):
#         self.predictor = Predictor.from_path(
#             "https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2018.11.30-charpad.tar.gz"
#         )
#
#     def predict(self, payload):
#         prediction = self.predictor.predict(
#             passage=payload["passage"], question=payload["question"]
#         )
#         return prediction["best_span_str"]


answerPredictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2018.11.30-charpad.tar.gz")
# f = open('Echelon.txt', 'r')
# inputText = f.read()
inputText = "IN NO EVENT WILL ECHELON BE LIABLE FOR ANY DAMAGES, INCLUDING LOSS OF DATA, LOST PROFITS, COST OF COVER OR OTHER SPECIAL, INCIDENTAL, PUNITIVE,CONSEQUENTIAL, OR INDIRECT DAMAGES ARISING FROM THE USE OF THE EXAMPLE SOFTWARE, HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY. THIS LIMITATION WILL APPLY EVEN IF ECHELON HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
inputQuestion = "In what event is Echelon liable for damages?"

result = answerPredictor.predict(passage=inputText, question=inputQuestion)

print(result['best_span_str'])

# Echelon question gives wrong answer when querying entire document but gives correct span when querying correct paragraph from IR
# Oracle gives state (California) instead of country (United States of America)

