import kashgari
from kashgari.tasks.labeling import BiLSTM_Model
from kashgari.corpus import ChineseDailyNerCorpus

test_x, test_y = ChineseDailyNerCorpus.load_data('./data/test.txt')
model = kashgari.utils.load_model('saved_ner_model')
print(test_x[:10])
print(model.predict(test_x[:10]))