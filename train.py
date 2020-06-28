import kashgari
from kashgari.embeddings import BERTEmbedding
from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.tasks.labeling import BiLSTM_CRF_Model
train_x, train_y = ChineseDailyNerCorpus.load_data('./data/train.txt')
valid_x, valid_y = ChineseDailyNerCorpus.load_data('./data/dev.txt')
test_x, test_y  = ChineseDailyNerCorpus.load_data('./data/test.txt')

bert_embed = BERTEmbedding('./chinese_L-12_H-768_A-12',
                           task=kashgari.LABELING,
                           sequence_length=100)

# 还可以选择 `CNN_LSTM_Model`, `BiLSTM_Model`, `BiGRU_Model` 或 `BiGRU_CRF_Model`
model = BiLSTM_CRF_Model(bert_embed)
model.fit(train_x,
          train_y,
          x_validate=valid_x,
          y_validate=valid_y,
          epochs=20,
          batch_size=512)

model.save('saved_ner_model')