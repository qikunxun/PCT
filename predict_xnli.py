import os

os.environ['TF_KERAS'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf
import numpy as np
import random
# import keras as K
from tensorflow import keras as K
from tensorflow.keras.layers import Lambda

from transformers import XLMRobertaTokenizer
from tqdm import tqdm
from bert4keras.models import build_transformer_model
from bert4keras.layers import Loss
from sklearn import metrics

my_seed = 1234
np.random.seed(my_seed)
random.seed(my_seed)
tf.random.set_seed(my_seed)
os.environ['PYTHONHASHSEED'] = str(my_seed)
SEQ_LEN = 128
BATCH_SIZE = 64

pretrained_path = 'PCT/xlm_roberta_base/'
config_path = os.path.join(pretrained_path, 'xlm_roberta_base_config.json')
checkpoint_path = os.path.join(pretrained_path, 'xlm_roberta_base.ckpt')

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)
# strategy = tf.distribute.MirroredStrategy()
print("REPLICAS: ", strategy.num_replicas_in_sync)

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large', pad_token='<s>')
tokenizer.padding_side = 'right'

label_map = {'entailment': 63736, 'contradictory': 157, 'neutral': 11354, 'contradiction': 157}
template = {'en': 'question:{}? Answer:<mask>', 'zh': '问题:{}? 答案:<mask>', 'fr': 'question:{}? réponse:<mask>',
            'es': 'pregunta:{}? Respuesta:<mask>',
            'de': 'frage:{}? Antwort:<mask>', 'el': 'ερώτηση:{}? Απάντηση:<mask>', 'bg': 'въпрос: {}? Отговор:<mask>',
            'ru': 'вопрос:{}? Ответ:<mask>',
            'tr': 'soru:{}? Cevap:<mask>', 'vi': 'câu hỏi:{}? Trả lời:<mask>', 'th': 'คำถาม:{}? คำตอบ:<mask>',
            'hi': 'प्रश्न:{}? उत्तर:<mask>',
            'sw': 'swali: {}? Jibu: <mask>', 'ur': '<mask>:سوال: {}؟ جواب', 'ar': '<mask>:سؤال:{}؟ الجواب'
            }
test_data = {}


def load_data(path, is_training=False):
    global tokenizer
    idx = 0
    indices, labels = [], []
    with open(path, 'r') as reader:
        for line in tqdm(reader.readlines()):
            if not line: continue
            if line.startswith('premise'): continue
            if line.startswith('sentence1'): continue
            array = line.strip().split('\t')
            label = array[2].lower()
            premise = array[0]
            language = 'en'
            if len(array) == 4:
                if array[-1] in template: language = array[-1]
            template_cur = template[language]
            hypothesis = template_cur.format(array[1])
            label_id = label_map[label]
            token_ids = tokenizer.encode(premise, hypothesis, max_length=SEQ_LEN,
                                         truncation='longest_first', padding='max_length')

            if 250001 not in token_ids:
                if language == 'ar' or language == 'ur':
                    template_tmp = template_cur.split('؟')[1]
                    template_ids = tokenizer.encode(template_tmp)
                    length = len(template_ids)
                    token_ids = token_ids[:-length]
                    token_ids.append(9446)
                    token_ids.extend(template_ids[1:])
                else:
                    template_tmp = template_cur.split('?')[1]
                    template_ids = tokenizer.encode(template_tmp)
                    length = len(template_ids)
                    token_ids = token_ids[:-length]
                    token_ids.append(705)
                    token_ids.extend(template_ids[1:])
            target = np.equal(np.array(token_ids), 250001).astype('int')
            assert np.sum(target) == 1
            target = target * label_id
            assert len(token_ids) == SEQ_LEN
            # print(target)
            indices.append(token_ids)
            labels.append(target)
            if len(array) == 4:
                language = array[-1]
                if language in test_data:
                    test_data[language]['token_ids'].append(token_ids)
                    test_data[language]['label'].append(label_id)
                else:
                    test_data[language] = {'token_ids': [token_ids], 'label': [label_id]}
            if idx < 5:
                print("*** Example ***")
                print("guid: %s" % (idx))
                print("tokens: %s" % " ".join(
                    [x for x in tokenizer.decode(token_ids)]))
                print("input_ids: %s" % " ".join([str(x) for x in token_ids]))
                print("label: %s (id = %s)" % (label, str(labels[idx])))

            idx += 1


    indices = np.array(indices)
    labels = np.array(labels)
    return [indices, labels], None


test_path = 'PCT/datasets/xnli/dev_and_test/test_all.txt'

test_x, test_y = load_data(test_path)

with strategy.scope():
    class CrossEntropy(Loss):
        def compute_loss(self, inputs, mask=None):
            y_true, y_pred = inputs
            y_mask = K.backend.cast(K.backend.not_equal(y_true, 0), K.backend.floatx())
            accuracy = K.metrics.sparse_categorical_accuracy(y_true, y_pred)
            accuracy = K.backend.sum(accuracy * y_mask) / (K.backend.sum(y_mask) + 1e-6)
            self.add_metric(accuracy, name='accuracy', aggregation='mean')
            loss = K.backend.sparse_categorical_crossentropy(y_true, y_pred)
            loss = K.backend.sum(loss * y_mask) / (K.backend.sum(y_mask) + 1e-6)
            return loss


    bert = build_transformer_model(
        config_path=config_path,
        # checkpoint_path=checkpoint_path,
        model='xlm_roberta',
        return_keras_model=True,
        sequence_length=SEQ_LEN,
        segment_vocab_size=0,
        with_mlm=True
    )

    y_in = K.layers.Input(shape=(None,))
    y_label = K.layers.Input(shape=(None,))
    context_emb = bert(y_in)
    inputs = [y_in, y_label]

    outputs = CrossEntropy(1)([y_in, context_emb])
    outputs_predict = Lambda(lambda x: K.backend.sum(
        x[0] * K.backend.expand_dims(K.backend.cast(K.backend.equal(x[1], 250001), 'float'), -1), axis=1))(
        [context_emb, inputs[0]])

    model_predict = K.models.Model(inputs, outputs_predict)

    model_predict.summary()
    model_predict.load_weights('PCT/model.hdf5')
for language in test_data:
    input_data = np.array(test_data[language]['token_ids'])
    labels = np.array(test_data[language]['label'])
    total_acc = 0
    for i in range(0, input_data.shape[0], BATCH_SIZE):
        try:
            data_tmp = input_data[i: i + BATCH_SIZE, :]
            label_tmp = labels[i: i + BATCH_SIZE]
            y_pred = model_predict.predict([data_tmp, data_tmp], batch_size=BATCH_SIZE, verbose=1)
            if y_pred.shape[0] != label_tmp.shape[0]:
                y_pred = y_pred[:label_tmp.shape[0], :]
            y_pred = y_pred.argmax(axis=-1)

            acc = metrics.accuracy_score(label_tmp, y_pred)
            print('Language:{}\tAcc: {}'.format(language, acc))
            total_acc += acc * data_tmp.shape[0] / input_data.shape[0]
        except Exception as e:
            print(e)
    print('Language:{}\tTotal Acc: {}'.format(language, total_acc))

