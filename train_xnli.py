import os
os.environ['TF_KERAS'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf
import numpy as np
import random

from tensorflow import keras as K
from tensorflow.keras.layers import Lambda
from transformers import XLMRobertaTokenizer
from tqdm import tqdm
from bert4keras.models import build_transformer_model
# from keras_bert import AdamWarmup, calc_train_steps
from bert4keras.layers import Loss

my_seed = 1234
np.random.seed(my_seed)
random.seed(my_seed)
tf.random.set_seed(my_seed)
os.environ['PYTHONHASHSEED'] = str(my_seed)
SEQ_LEN = 128
BATCH_SIZE = 32
EPOCHS = 2
LR = 5e-5
print(LR, BATCH_SIZE)
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

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', pad_token='<s>')
tokenizer.padding_side = 'right'
special_tokens = ['<unused{}>'.format(i) for i in range(200)]
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

label_map = {'entailment': 63736, 'contradictory': 157, 'neutral': 11354, 'contradiction': 157}

template = {'en': 'question:{}? Answer:<mask>', 'zh': '问题:{}? 答案:<mask>', 'fr': 'question:{}? réponse:<mask>',
            'es': 'pregunta:{}? Respuesta:<mask>',
            'de': 'frage:{}? Antwort:<mask>', 'el': 'ερώτηση:{}? Απάντηση:<mask>', 'bg': 'въпрос: {}? Отговор:<mask>',
            'ru': 'вопрос:{}? Ответ:<mask>',
            'tr': 'soru:{}? Cevap:<mask>', 'vi': 'câu hỏi:{}? Trả lời:<mask>', 'th': 'คำถาม:{}? คำตอบ:<mask>',
            'hi': 'प्रश्न:{}? उत्तर:<mask>',
            'sw': 'swali: {}? Jibu: <mask>', 'ur': '<mask>:سوال: {}؟ جواب', 'ar': '<mask>:سؤال:{}؟ الجواب'
            }

language_list = ['fr', 'es', 'de', 'el', 'bg', 'ru', 'tr', 'ar', 'vi', 'th', 'zh', 'sw', 'hi', 'ur']
test_data = {}


def load_data(path, is_training=False, type='test'):
    global tokenizer
    idx = 0
    indices, labels, indices_target, labels_target = [], [], [], []

    with open(path, 'r') as reader:
        for line in tqdm(reader.readlines()):
            if not line: continue
            if line.startswith('premise'): continue
            if line.startswith('sentence1'): continue
            array = line.strip().split('\t')
            label = array[2].lower()
            premise = array[0]
            hypothesis = template['en'].format(array[1])

            label_id = label_map[label]
            token_ids = tokenizer.encode(premise, hypothesis, max_length=SEQ_LEN,
                                         truncation='longest_first', padding='max_length')

            if 250001 not in token_ids:
                template_tmp = template['en'].split('?')[1]
                template_ids = tokenizer.encode(template_tmp)
                length = len(template_ids)
                token_ids = token_ids[:-length]
                token_ids.append(705)
                token_ids.extend(template_ids[1:])
            assert len(token_ids) == SEQ_LEN
            target = np.equal(np.array(token_ids), 250001).astype('int')
            assert np.sum(target) == 1
            target = target * label_id
            # print(target)
            indices.append(token_ids)
            labels.append(target)
            if is_training:
                flag = random.randint(0, 13)
                language_sample = language_list[flag]
                template_cur = template[language_sample]
                hypothesis_ = template_cur.format(array[1])
                token_ids_ = tokenizer.encode(premise, hypothesis_, max_length=SEQ_LEN,
                                              truncation='longest_first', padding='max_length')
                if 250001 not in token_ids_:
                    if language_sample == 'ur' or language_sample == 'ar':
                        template_tmp = template_cur.split('؟')[1]
                        template_ids = tokenizer.encode(template_tmp)
                        length = len(template_ids)
                        token_ids_ = token_ids_[:-length]

                        token_ids_.append(9446)
                        token_ids_.extend(template_ids[1:])
                    else:
                        template_tmp = template_cur.split('?')[1]
                        template_ids = tokenizer.encode(template_tmp)
                        length = len(template_ids)
                        token_ids_ = token_ids_[:-length]

                        token_ids_.append(705)
                        token_ids_.extend(template_ids[1:])
                target_ = np.equal(np.array(token_ids_), 250001).astype('int')
                assert np.sum(target_) == 1
                target_ = target_ * label_id
                assert len(token_ids_) == SEQ_LEN
                indices_target.append(token_ids_)
                labels_target.append(target_)
            if len(array) == 4 and type == 'test':
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
            # if len(indices) > 10000:
            #     break
    indices = np.array(indices)
    labels = np.array(labels)
    if not is_training:
        return [indices, indices, labels, labels], None
    else:
        indices_target = np.array(indices_target)
        labels_target = np.array(labels_target)
        return [indices, indices_target, labels, labels_target], None


train_path = 'PCT/datasets/xnli/train/multinli.train.en.tsv'
valid_path = 'PCT/datasets/xnli/dev_and_test/valid_all.txt'
test_path = 'PCT/datasets/xnli/dev_and_test/test_all.txt'
#
train_x, train_y = load_data(train_path, is_training=True)
valid_x, valid_y = load_data(valid_path, type='dev')
# test_x, test_y = load_data(test_path)

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
        checkpoint_path=checkpoint_path,
        model='xlm_roberta',
        return_keras_model=True,
        sequence_length=SEQ_LEN,
        segment_vocab_size=0,
        with_mlm=True
    )

    y_in = K.layers.Input(shape=(None,))
    y_in_target = K.layers.Input(shape=(None,))
    y_label = K.layers.Input(shape=(None,))
    y_label_target = K.layers.Input(shape=(None,))
    context_emb = bert(y_in)
    context_emb_target = bert(y_in_target)
    inputs = [y_in, y_in_target, y_label, y_label_target]
    outputs = CrossEntropy(1)([y_label, context_emb])
    outputs_ = CrossEntropy(1)([y_label_target, context_emb_target])
    outputs_predict = Lambda(
        lambda x: K.backend.sum(x[0] * K.backend.expand_dims(K.backend.cast(K.backend.equal(x[1], 250001),
                                                                            'float'), -1), axis=1))(
        [context_emb, inputs[0]])
    outputs_predict_target = Lambda(
        lambda x: K.backend.sum(x[0] * K.backend.expand_dims(K.backend.cast(K.backend.equal(x[1], 250001),
                                                                            'float'), -1), axis=1))(
        [context_emb_target, inputs[1]])
    # total_steps, warmup_steps = calc_train_steps(
    #     num_example=train_x[0].shape[0],
    #     batch_size=BATCH_SIZE,
    #     epochs=EPOCHS,
    #     warmup_proportion=0.1,
    # )
    #
    # optimizer = AdamWarmup(total_steps, warmup_steps, lr=LR, min_lr=LR * 0.1)
    optimizer = K.optimizers.RMSprop(LR)
    model = K.models.Model(inputs, [outputs, outputs_])
    model_predict = K.models.Model(inputs[0], outputs_predict)
    model.add_loss(Lambda(lambda x: K.backend.mean(K.losses.kld(x[0], x[1]) + K.losses.kld(x[1], x[0])))(
        [outputs_predict, outputs_predict_target]))
    model.compile(
        optimizer,
    )
    model.summary()

model.fit(
    train_x, train_y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(valid_x, valid_y),
    # callbacks=[checkpoint]
)
model.save_weights('PCT/model.hdf5')