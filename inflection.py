# -*- coding: utf-8 -*-

import argparse
import codecs
import dynet as dy
import matplotlib 
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import myutil
import numpy as np
from operator import itemgetter
import os, sys
from random import random,shuffle



parser = argparse.ArgumentParser()
parser.add_argument("--datapath", help="path to data", type=str, required=True)
parser.add_argument("--L1", help="transfer languages (split with comma for multiple ones)", type=str, required=False)
parser.add_argument("--L2", help="test languages", type=str, required=True)
parser.add_argument("--mode", help="usage mode", type=str,
    choices=['train','test','test-dev','draw-dev','test-dev-ensemble','test-ensemble','test-two-ensemble','test-three-ensemble',
    'test-all-ensemble'], default='', required=True)
parser.add_argument("--setting", help="data setting", type=str, choices=['original','swap','low',], default='original')
parser.add_argument("--modelpath", help="path to store the models", type=str, default='./models')
parser.add_argument("--figurepath", help="path to store the output attention figures", type=str, default='./figures')
parser.add_argument("--outputpath", help="path to store the inflected outputs on the test set", type=str, default='./outputs')
parser.add_argument("--notest", help="do not use the test set at all", action="store_true")
parser.add_argument("--testset", help="path to different test set", type=str, required=False)
parser.add_argument("--outputfile", help="path to store the inflected outputs", type=str, required=False)
parser.add_argument("--use_hall", help="whether to use a hallucinated dataset (def: False)", action="store_true")
parser.add_argument("--only_hall", help="only use the hallucinated dataset to train (def: False)", action="store_true")
parser.add_argument("--predict_lang", help="use the language discriminator auxiliary task (def: False)", action="store_true")
parser.add_argument("--use_att_reg", help="use attention regularization on the lemma attention (def: False)", action="store_true")
parser.add_argument("--use_tag_att_reg", help="use attention regularization on the tag attention (def: False)", action="store_true")
parser.add_argument("--dynet-mem", help="set dynet memory", default=800, type=int, required=False)
parser.add_argument("--dynet-autobatch", help="use dynet autobatching (def: 1)", default=1, type=int, required=False)
args = parser.parse_args()

if args.L1:
    L1 = args.L1
    L1s = L1.split(',')
else:
    L1 = ''
    L1s = []

L2 = args.L2
DATA_PATH = args.datapath

if not os.path.isdir(DATA_PATH):
    print("Wrong data path, the provided one does not exist")
    sys.exit()
LOW_PATH = os.path.join(DATA_PATH, L2+ "-train")
DEV_PATH = os.path.join(DATA_PATH, L2+ "-dev")
HALL_PATH = os.path.join(DATA_PATH, L2+ "-hall")
TEST_PATH = os.path.join(DATA_PATH, L2+ "-test-covered")
if args.testset:
    TEST_PATH = args.testset

if not os.path.isdir(args.modelpath):
    os.mkdir(args.modelpath)
if not os.path.isdir(args.figurepath):
    os.mkdir(args.figurepath)
if not os.path.isdir(args.outputpath):
    os.mkdir(args.outputpath)

if L1:
    exp_dir = L1+"-"+L2
else:
    exp_dir = L2

MODEL_DIR = os.path.join(args.modelpath, exp_dir)
if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)

FIGURE_DIR = os.path.join(args.figurepath, exp_dir)
if not os.path.isdir(FIGURE_DIR):
    os.mkdir(FIGURE_DIR)

OUTPUT_DIR = os.path.join(args.outputpath, exp_dir)
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


TRAIN=False
TEST = False
TEST_ENSEMBLE = False
TEST_TWO_ENSEMBLE = False
TEST_THREE_ENSEMBLE = False
TEST_ALL_ENSEMBLE = False
TEST_DEV = False
DRAW_DEV = False
TEST_DEV_ENSEMBLE = False

if args.mode == "train":
    TRAIN = True
elif args.mode == "test":
    TEST = True
elif args.mode == "test-dev":
    TEST_DEV = True
elif args.mode == "draw-dev":
    DRAW_DEV = True
elif args.mode == "test-dev-ensemble":
    TEST_DEV_ENSEMBLE = True
elif args.mode == "test-ensemble":
    TEST_ENSEMBLE = True
elif args.mode == "test-two-ensemble":
    TEST_TWO_ENSEMBLE = True
elif args.mode == "test-three-ensemble":
    TEST_THREE_ENSEMBLE = True
elif args.mode == "test-all-ensemble":
    TEST_ALL_ENSEMBLE = True

USE_HALL = False
if args.use_hall:
    USE_HALL = True

ONLY_HALL = False
if args.only_hall:
    ONLY_HALL = True


if args.setting == "original":
    ORIGINAL = True
    SWAP = False
    LOW = False
elif args.setting == "swap":
    ORIGINAL = False
    SWAP = True
    LOW = False
elif args.setting == "low":
    ORIGINAL = False
    SWAP = False
    LOW = True
else:
    ORIGINAL = False
    SWAP = False
    LOW = False

MODEL_NAME = "orig."
if SWAP:
    MODEL_NAME = "swap."
elif LOW:
    MODEL_NAME = "low."

if USE_HALL:
    MODEL_NAME += "hall."
if ONLY_HALL:
    MODEL_NAME += "hallonly."


MAX_PREDICTION_LEN_DEF = 20
if L2 == "kabardian":
    MAX_PREDICTION_LEN_DEF = 25
elif L2 == "tatar":
    MAX_PREDICTION_LEN_DEF = 23
elif L2 == "greek":
    MAX_PREDICTION_LEN_DEF = 30
elif L2 == "latin":
    MAX_PREDICTION_LEN_DEF = 25
elif L2 == "livonian":
    MAX_PREDICTION_LEN_DEF = 22
elif L2 == "bengali":
    MAX_PREDICTION_LEN_DEF = 23
elif L2 == "czech":
    MAX_PREDICTION_LEN_DEF = 30
elif L2 == "lithuanian":
    MAX_PREDICTION_LEN_DEF = 33
elif L2 == "russian":
    MAX_PREDICTION_LEN_DEF = 50
elif L2 == "irish":
    MAX_PREDICTION_LEN_DEF = 37
elif L2 == "quechua":
    MAX_PREDICTION_LEN_DEF = 30
elif L2 == "azeri":
    MAX_PREDICTION_LEN_DEF = 22
elif L2 == "yiddish":
    MAX_PREDICTION_LEN_DEF = 22


LENGTH_NORM_WEIGHT = 0.1
EXTRA_WEIGHT = 0.3
USE_ATT_REG = False
USE_TAG_ATT_REG = False
PREDICT_LANG = False

if args.predict_lang:
    PREDICT_LANG = True
    MODEL_NAME += "lang."
if args.use_att_reg:
    USE_ATT_REG = True
if args.use_tag_att_reg:
    USE_TAG_ATT_REG = True

if USE_HALL:
    low_i, low_o, low_t = myutil.read_data(LOW_PATH)
    dev_i, dev_o, dev_t = myutil.read_data(DEV_PATH)
    if args.notest:
        test_i, test_t = dev_i, dev_t
    else:
        test_i, test_t = myutil.read_test_data(TEST_PATH)
    hall_i, hall_o, hall_t = myutil.read_data(HALL_PATH)
    low_i += hall_i
    low_o += hall_o
    low_t += hall_t
    lids_1 = [0]*len(low_i) 
    high_i, high_o, high_t = [], [], []
    for j,L1 in enumerate(L1s):
        HIGH_PATH = os.path.join(DATA_PATH, L1+ "-train")
        ti, to, tt = myutil.read_data(HIGH_PATH)
        high_i += ti
        high_o += to
        high_t += tt
        lids_1 += [j+1]*len(ti)
    NUM_LANG = len(L1s)+1
elif ONLY_HALL:
    high_i, high_o, high_t = [], [], []
    low_i, low_o, low_t = myutil.read_data(LOW_PATH)
    dev_i, dev_o, dev_t = myutil.read_data(DEV_PATH)
    if args.notest:
        test_i, test_t = dev_i, dev_t
    else:
        test_i, test_t = myutil.read_test_data(TEST_PATH)
    hall_i, hall_o, hall_t = myutil.read_data(HALL_PATH)
    low_i += hall_i
    low_o += hall_o
    low_t += hall_t
else:
    low_i, low_o, low_t = myutil.read_data(LOW_PATH)
    dev_i, dev_o, dev_t = myutil.read_data(DEV_PATH)
    if args.notest:
        test_i, test_t = dev_i, dev_t
    else:
        test_i, test_t = myutil.read_test_data(TEST_PATH)
    high_i, high_o, high_t = [], [], []
    lids_1 = [0]*len(low_i) 
    for j,L1 in enumerate(L1s):
        HIGH_PATH = os.path.join(DATA_PATH, L1+ "-train")
        ti, to, tt = myutil.read_data(HIGH_PATH)
        high_i += ti
        high_o += to
        high_t += tt
        lids_1 += [j+1]*len(ti)
    NUM_LANG = len(L1s)+1


if SWAP:
    if len(dev_i) < len(low_o):
        N = len(dev_i)
        tmp1, tmp2, tmp3 = list(low_i[-N:]), list(low_o[-N:]), list(low_t[-N:])
        low_i = list(low_i[:-N] + dev_i)
        low_o = list(low_o[:-N] + dev_o)
        low_t = list(low_t[:-N] + dev_t)
        dev_i, dev_o, dev_t = tmp1, tmp2, tmp3
    else:
        tmp1, tmp2, tmp3 = list(low_i), list(low_o), list(low_t)
        low_i, low_o, low_t = list(dev_i), list(dev_o), list(dev_t)
        dev_i, dev_o, dev_t = tmp1, tmp2, tmp3

print("Data lengths")
print("transfer-language", len(high_i), len(high_o), len(high_t))
print("test-language", len(low_i), len(low_o), len(low_t))
print("dev", len(dev_i), len(dev_o), len(dev_t))
print("test", len(test_i), len(test_t))

def compute_mixing_weights(l):
    if l == 3:
        K = float(len(high_i))
        N = float(len(low_i))
        M = float(len(dev_i))
        denom = 2*N+M+2*K
        return [(K+N)/denom, (M+K)/denom, N/denom]
    elif l == 2:
        K = float(len(high_i))
        N = float(len(low_i))
        M = float(len(dev_i))
        denom = N+M+2*K
        return [(K+N)/denom, (M+K)/denom]


COPY_THRESHOLD = 0.9
COPY_TASK_PROB = 0.2
STARTING_LEARNING_RATE = 0.1
EPOCHS_TO_HALVE = 6

MULTIPLY = 1
if len(high_i)+len(low_i) < 5000:
    MULTIPLY = 1
    STARTING_LEARNING_RATE = 0.2
    COPY_THRESHOLD = 0.6
    COPY_TASK_PROB = 0.4
    EPOCHS_TO_HALVE = 12

def get_chars(l):
    flat_list = [char for word in l for char in word]
    return list(set(flat_list))

def get_tags(l):
    flat_list = [tag for sublist in l for tag in sublist]
    return list(set(flat_list))

EOS = "<EOS>"
NULL = "<NULL>"

if TRAIN:
    #SOS = "<SOS>"
    characters = get_chars(high_i+high_o+low_i+low_o+dev_i+dev_o+test_i)
    #characters.append(SOS)
    characters.append(EOS)
    if u' '  not in characters:
        characters.append(u' ')

    tags = get_tags(high_t+low_t+dev_t+test_t)
    tags.append(NULL)

    #Store vocabularies for future reference
    myutil.write_vocab(characters, os.path.join(MODEL_DIR, MODEL_NAME+"char.vocab"))
    myutil.write_vocab(tags, os.path.join(MODEL_DIR, MODEL_NAME+"tag.vocab"))
else:
    characters = myutil.read_vocab(os.path.join(MODEL_DIR, MODEL_NAME+"char.vocab"))
    if u' '  not in characters:
        characters.append(u' ')
    tags = myutil.read_vocab(os.path.join(MODEL_DIR, MODEL_NAME+"tag.vocab"))

int2char = list(characters)
char2int = {c:i for i,c in enumerate(characters)}

int2tag = list(tags)
tag2int = {c:i for i,c in enumerate(tags)}

VOCAB_SIZE = len(characters)
TAG_VOCAB_SIZE = len(tags)


LSTM_NUM_OF_LAYERS = 1
EMBEDDINGS_SIZE = 32
STATE_SIZE = 100
ATTENTION_SIZE = 100
MINIBATCH_SIZE = 1
COPY_WEIGHT = 0.8
DROPOUT_PROB = 0.2

print("Characters:",characters)
print("Vocab size:", VOCAB_SIZE)
print("All tags:", tags)
print("Tag vocab size:", TAG_VOCAB_SIZE)


def run_lstm(init_state, input_vecs):
    s = init_state
    out_vectors = []
    for vector in input_vecs:
        s = s.add_input(vector)
        out_vector = s.output()
        out_vectors.append(out_vector)
    return out_vectors


class InflectionModel:
    def __init__(self):
        self.model = dy.Model()

        self.enc_fwd_lstm = dy.CoupledLSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, self.model)
        self.enc_bwd_lstm = dy.CoupledLSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, self.model)

        self.dec_lstm = dy.CoupledLSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE*3+EMBEDDINGS_SIZE, STATE_SIZE, self.model)

        self.input_lookup = self.model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE) )
        self.tag_input_lookup = self.model.add_lookup_parameters((TAG_VOCAB_SIZE, EMBEDDINGS_SIZE) )
        self.attention_w1 = self.model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*2) )
        self.attention_w2 = self.model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*LSTM_NUM_OF_LAYERS*2) )
        self.attention_w3 = self.model.add_parameters( (ATTENTION_SIZE, 5) )
        self.attention_v = self.model.add_parameters( (1, ATTENTION_SIZE))

        self.decoder_w = self.model.add_parameters( (VOCAB_SIZE, STATE_SIZE))
        self.decoder_b = self.model.add_parameters( (VOCAB_SIZE))
        #output_lookup = model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE))
        self.output_lookup = self.input_lookup

        self.enc_tag_lstm = dy.CoupledLSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, self.model)
        self.tag_attention_w1 = self.model.add_parameters( (ATTENTION_SIZE, STATE_SIZE))
        self.tag_attention_w2 = self.model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*LSTM_NUM_OF_LAYERS*2))
        self.tag_attention_v = self.model.add_parameters( (1, ATTENTION_SIZE))

        if PREDICT_LANG:
            self.lang_class_w = self.model.add_parameters((STATE_SIZE*2, NUM_LANG))
            #self.lang_class_w = self.model.add_parameters((STATE_SIZE*2, 1))

    def embed_tags(self, tags):
        tags = [tag2int[t] for t in tags]
        return [self.tag_input_lookup[tag] for tag in tags]

    def embed_sentence(self, sentence):
        sentence = [EOS] + list(sentence) + [EOS]
        sentence = [char2int[c] for c in sentence]
        return [self.input_lookup[char] for char in sentence]


    def self_encode_tags(self, tags):
        vectors = tags
        # Self attention for every tag:
        vectors = run_lstm(self.enc_tag_lstm.initial_state(), tags)
        tag_input_mat = dy.concatenate_cols(vectors)
        out_vectors = []
        for v1 in vectors:
            # tag input mat: [tag_emb x seqlen]
            # v1: [tag_emb]
            unnormalized = dy.transpose(dy.transpose(v1) * tag_input_mat)
            self_att_weights = dy.softmax(unnormalized)
            to_add = tag_input_mat*self_att_weights
            out_vectors.append(v1 + tag_input_mat*self_att_weights)
        return out_vectors


    def encode_tags(self, tags):
        vectors = run_lstm(self.enc_tag_lstm.initial_state(), tags)
        return vectors

    def encode_sentence(self, sentence):
        sentence_rev = list(reversed(sentence))

        fwd_vectors = run_lstm(self.enc_fwd_lstm.initial_state(), sentence)
        bwd_vectors = run_lstm(self.enc_bwd_lstm.initial_state(), sentence_rev)
        bwd_vectors = list(reversed(bwd_vectors))
        vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

        return vectors

    def attend_tags(self, state, w1dt):

        # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
        # w1dt: (attdim x seqlen)
        # w2dt: (attdim x attdim)
        w2dt = self.tag_attention_w2*state
        # att_weights: (seqlen,) row vector
        unnormalized = dy.transpose(self.tag_attention_v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
        att_weights = dy.softmax(unnormalized)
        # context: (encoder_state)
        
        return att_weights

    def attend(self, state, w1dt):
        # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
        # w1dt: (attdim x seqlen)
        # w2dt: (attdim x attdim)
        w2dt = self.attention_w2*state
        # att_weights: (seqlen,) row vector
        unnormalized = dy.transpose(self.attention_v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
        att_weights = dy.softmax(unnormalized)
        return att_weights

    def attend_with_prev(self, state, w1dt, prev_att):
        # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
        # w1dt: (attdim x seqlen)
        # w2dt: (attdim x attdim)
        w2dt = self.attention_w2 * state
        w3dt = self.attention_w3 * prev_att
        # att_weights: (seqlen,) row vector
        unnormalized = dy.transpose(self.attention_v * dy.tanh(dy.colwise_add(dy.colwise_add(w1dt, w2dt), w3dt)))
        att_weights = dy.softmax(unnormalized)
        return att_weights

    def decode(self, vectors, tag_vectors, output, lang_id, weight, teacher_prob=1.0):
        output = [EOS] + list(output) + [EOS]
        output = [char2int[c] for c in output]

        N = len(vectors)

        input_mat = dy.concatenate_cols(vectors)
        w1dt = None

        input_mat = dy.dropout(input_mat, DROPOUT_PROB)

        tag_input_mat = dy.concatenate_cols(tag_vectors)
        tag_w1dt = None

        last_output_embeddings = self.output_lookup[char2int[EOS]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([vectors[-1], tag_vectors[-1], last_output_embeddings]))
        loss = []
        prev_att = dy.zeros(5)

        if USE_ATT_REG:
            total_att = dy.zeros(N)
        if USE_TAG_ATT_REG:
            total_tag_att = dy.zeros(len(tag_vectors))

        for char in output:
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or self.attention_w1 * input_mat
            tag_w1dt = tag_w1dt or self.tag_attention_w1 * tag_input_mat

            state = dy.concatenate(list(s.s()))
            
            tag_att_weights = self.attend_tags(state, tag_w1dt)
            tag_context = tag_input_mat * tag_att_weights

            tag_context2 = dy.concatenate([tag_context,tag_context])

            new_state = state + tag_context2

            att_weights = self.attend_with_prev(new_state, w1dt, prev_att)
            context = input_mat * att_weights
            best_ic = np.argmax(att_weights.vec_value())
            context = input_mat * att_weights
            startt = min(best_ic-2,N-6)
            if startt < 0:
                startt = 0
            endd = startt+5

            if N < 5:
                prev_att = dy.concatenate([att_weights] + [dy.zeros(1)]*(5-N) )
            else:
                prev_att = att_weights[startt:endd]
            if prev_att.dim()[0][0] != 5:
                print(prev_att.dim())

            if USE_ATT_REG:
                total_att = total_att + att_weights
            if USE_TAG_ATT_REG:
                total_tag_att = total_tag_att + tag_att_weights

            vector = dy.concatenate([context, tag_context, last_output_embeddings])
            s = s.add_input(vector)

            s_out = dy.dropout(s.output(), DROPOUT_PROB)

            out_vector = self.decoder_w * s_out + self.decoder_b
            probs = dy.softmax(out_vector)
            if teacher_prob == 1:
                last_output_embeddings = self.output_lookup[char]
            else:
                if random() > teacher_prob:
                    out_char = np.argmax(probs.npvalue())
                    last_output_embeddings = self.output_lookup[out_char]
                else:
                    last_output_embeddings = self.output_lookup[char]


            loss.append(-dy.log(dy.pick(probs, char)))
        loss = dy.esum(loss)*weight

        
        if PREDICT_LANG:
            last_enc_state = vectors[-1]
            adv_state = dy.flip_gradient(last_enc_state)
            pred_lang = dy.transpose(dy.transpose(adv_state)*self.lang_class_w)
            lang_probs = dy.softmax(pred_lang)
            lang_loss_1 = -dy.log(dy.pick(lang_probs, lang_id))

            first_enc_state = vectors[0]
            adv_state2 = dy.flip_gradient(first_enc_state)
            pred_lang2 = dy.transpose(dy.transpose(adv_state2)*self.lang_class_w)
            lang_probs2 = dy.softmax(pred_lang2)
            lang_loss_2 = -dy.log(dy.pick(lang_probs2, lang_id))
            loss += lang_loss_1 + lang_loss_2

        if USE_ATT_REG:
            loss += dy.huber_distance(dy.ones(N),total_att)
        if USE_TAG_ATT_REG:
            loss += dy.huber_distance(dy.ones(len(tag_vectors)), total_tag_att)

        return loss

    def generate(self, in_seq, tag_seq, show_att=False, show_tag_att=False, fn=None):
        dy.renew_cg()
        embedded = self.embed_sentence(in_seq)
        encoded = self.encode_sentence(embedded)
        
        embedded_tags = self.embed_tags(tag_seq)
        #encoded_tags = self.encode_tags(embedded_tags)
        encoded_tags = self.self_encode_tags(embedded_tags)
        
        input_mat = dy.concatenate_cols(encoded)
        tag_input_mat = dy.concatenate_cols(encoded_tags)
        w1dt = None
        tag_w1dt = None

        prev_att = dy.zeros(5)

        tmpinseq = [EOS] + list(in_seq) + [EOS]
        N = len(tmpinseq)

        last_output_embeddings = self.output_lookup[char2int[EOS]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([encoded[-1], encoded_tags[-1], last_output_embeddings]))

        out = ''
        count_EOS = 0
        if show_att:
            attt_weights = []
        if show_tag_att:
            ttt_weights = []
        for i in range(len(in_seq)*2):
            if count_EOS == 2: break
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or self.attention_w1 * input_mat
            tag_w1dt = tag_w1dt or self.tag_attention_w1 * tag_input_mat

            state = dy.concatenate(list(s.s()))
            tag_att_weights = self.attend_tags(state, tag_w1dt)
            tag_context = tag_input_mat * tag_att_weights
            tag_context2 = dy.concatenate([tag_context,tag_context])
            new_state = state + tag_context2
            att_weights = self.attend_with_prev(new_state, w1dt, prev_att)
            best_ic = np.argmax(att_weights.vec_value())
            context = input_mat * att_weights
            startt = min(best_ic-2, N-6)
            if startt < 0:
                startt = 0
            endd = startt+5
            if N < 5:
                prev_att = dy.concatenate([att_weights] + [dy.zeros(1)]*(5-N) )
            else:
                prev_att = att_weights[startt:endd]


            if show_att:
                attt_weights.append(att_weights.npvalue())
            if show_tag_att:
                ttt_weights.append(tag_att_weights.npvalue())

            vector = dy.concatenate([context, tag_context, last_output_embeddings])
            s = s.add_input(vector)
            out_vector = self.decoder_w * s.output() + self.decoder_b
            probs = dy.softmax(out_vector).npvalue()

            next_char = np.argmax(probs)
            last_output_embeddings = self.output_lookup[next_char]
            if int2char[next_char] == EOS:
                count_EOS += 1
                continue

            out += int2char[next_char]

        if (show_att) and len(out) and fn is not None:
            arr = np.squeeze(np.array(attt_weights))[1:-1,1:-1]
            fig, ax = plt.subplots()
            ax = plt.imshow(arr)
            x_positions = np.arange(0,len(attt_weights[0])-2)
            y_positions = np.arange(0,len(out))
            plt.xticks(x_positions, list(in_seq))
            plt.yticks(y_positions, list(out))
            plt.savefig(fn+'-char.png')
            plt.clf()
            plt.close()

        if (show_tag_att) and len(out) and fn is not None:
            arr = np.squeeze(np.array(ttt_weights))[1:-1,:]
            fig, ax = plt.subplots()
            ax = plt.imshow(arr)
            x_positions = np.arange(0,len(ttt_weights[0]))
            y_positions = np.arange(0,len(out))
            plt.xticks(x_positions, list(tag_seq))
            plt.yticks(y_positions, list(out))
            plt.savefig(fn+'-tag.png')
            plt.clf()
            plt.close()


        return out


    def draw_decode(self, in_seq, tag_seq, out_seq, show_att=False, show_tag_att=False, fn=None):
        dy.renew_cg()
        embedded = self.embed_sentence(in_seq)
        encoded = self.encode_sentence(embedded)
        N = len(encoded)
        
        embedded_tags = self.embed_tags(tag_seq)
        encoded_tags = self.self_encode_tags(embedded_tags)

        output = [EOS] + list(out_seq) + [EOS]
        output = [char2int[c] for c in output]

        input_mat = dy.concatenate_cols(encoded)
        w1dt = None

        tag_input_mat = dy.concatenate_cols(encoded_tags)
        tag_w1dt = None

        last_output_embeddings = self.output_lookup[char2int[EOS]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([encoded[-1], encoded_tags[-1], last_output_embeddings]))
        prev_att = dy.zeros(5)

        attt_weights = []
        ttt_weights = []

        for char in output:
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or self.attention_w1 * input_mat
            tag_w1dt = tag_w1dt or self.tag_attention_w1 * tag_input_mat

            state = dy.concatenate(list(s.s()))
            
            tag_att_weights = self.attend_tags(state, tag_w1dt)
            tag_context = tag_input_mat * tag_att_weights

            tag_context2 = dy.concatenate([tag_context,tag_context])

            new_state = state + tag_context2

            att_weights = self.attend_with_prev(new_state, w1dt, prev_att)
            best_ic = np.argmax(att_weights.vec_value())
            context = input_mat * att_weights
            startt = min(best_ic-2,N-6)
            if startt < 0:
                startt = 0
            endd = startt+5

            if N < 5:
                prev_att = dy.concatenate([att_weights] + [dy.zeros(1)]*(5-N) )
            else:
                prev_att = att_weights[startt:endd]
            if prev_att.dim()[0][0] != 5:
                print(prev_att.dim())

            if show_att:
                attt_weights.append(att_weights.npvalue())
            if show_tag_att:
                ttt_weights.append(tag_att_weights.npvalue())


            vector = dy.concatenate([context, tag_context, last_output_embeddings])
            s = s.add_input(vector)

            s_out = dy.dropout(s.output(), DROPOUT_PROB)

            out_vector = self.decoder_w * s_out + self.decoder_b
            probs = dy.softmax(out_vector)
            last_output_embeddings = self.output_lookup[char]

        outputchars = [int2char[c] for c in output[1:-1]]
        
        if (show_att) and fn is not None:
            arr = np.squeeze(np.array(attt_weights))[1:-1,1:-1]
            fig, ax = plt.subplots()
            ax = plt.imshow(arr)
            x_positions = np.arange(0,len(attt_weights[0])-2)
            y_positions = np.arange(0,len(outputchars))
            plt.xticks(x_positions, list(in_seq))
            plt.yticks(y_positions, list(outputchars))
            plt.savefig(fn+'-char.png')
            plt.clf()
            plt.close()

        if (show_tag_att) and fn is not None:
            arr = np.squeeze(np.array(ttt_weights))[1:-1,:]
            fig, ax = plt.subplots()
            ax = plt.imshow(arr)
            x_positions = np.arange(0,len(ttt_weights[0]))
            y_positions = np.arange(0,len(outputchars))
            plt.xticks(x_positions, list(tag_seq))
            plt.yticks(y_positions, list(outputchars))
            plt.savefig(fn+'-tag.png')
            plt.clf()
            plt.close()


        return


    def generate_nbest(self, in_seq, tag_seq, beam_size=4, show_att=False, show_tag_att=False, fn=None):
        dy.renew_cg()
        try:
            embedded = self.embed_sentence(in_seq)
        except:
            return []
        encoded = self.encode_sentence(embedded)
        
        embedded_tags = self.embed_tags(tag_seq)
        #encoded_tags = self.encode_tags(embedded_tags)
        encoded_tags = self.self_encode_tags(embedded_tags)
        

        input_mat = dy.concatenate_cols(encoded)
        tag_input_mat = dy.concatenate_cols(encoded_tags)
        prev_att = dy.zeros(5)

        tmpinseq = [EOS] + list(in_seq) + [EOS]
        N = len(tmpinseq)

        last_output_embeddings = self.output_lookup[char2int[EOS]]
        init_vector = dy.concatenate([encoded[-1], encoded_tags[-1], last_output_embeddings])
        s_0 = self.dec_lstm.initial_state().add_input(init_vector)
        w1dt = self.attention_w1 * input_mat
        tag_w1dt = self.tag_attention_w1 * tag_input_mat

        beam = {0: [(0, s_0.s(), [], prev_att)]}

        i = 1

        nbest = []
         # we'll need this
        last_states = {}

        MAX_PREDICTION_LEN = max(len(in_seq)*1.5,MAX_PREDICTION_LEN_DEF)

        # expand another step if didn't reach max length and there's still beams to expand
        while i < MAX_PREDICTION_LEN and len(beam[i - 1]) > 0:
            # create all expansions from the previous beam:
            next_beam_id = []
            for hyp_id, hypothesis in enumerate(beam[i - 1]):
                # expand hypothesis tuple
                #prefix_seq, prefix_prob, prefix_decoder, prefix_context, prefix_tag_context = hypothesis
                prefix_prob, prefix_decoder, prefix_seq, prefix_att = hypothesis
                
                if i > 1:
                    last_hypo_symbol = prefix_seq[-1]
                else:
                    last_hypo_symbol = EOS

                # cant expand finished sequences
                if last_hypo_symbol == EOS and i > 3:
                    continue
                # expand from the last symbol of the hypothesis
                last_output_embeddings = self.output_lookup[char2int[last_hypo_symbol]]

                # Perform the forward step on the decoder
                # First, set the decoder's parameters to what they were in the previous step
                if (i == 1):
                    s = self.dec_lstm.initial_state().add_input(init_vector)
                else:
                    s = self.dec_lstm.initial_state(prefix_decoder)

                state = dy.concatenate(list(s.s()))
                tag_att_weights = self.attend_tags(state, tag_w1dt)
                tag_context = tag_input_mat * tag_att_weights
                tag_context2 = dy.concatenate([tag_context,tag_context])
                new_state = state + tag_context2

                att_weights = self.attend_with_prev(new_state, w1dt, prefix_att)
                best_ic = np.argmax(att_weights.vec_value())
                startt = min(best_ic-2, N-6)
                if startt < 0:
                    startt = 0
                endd = startt+5
                if N < 5:
                    prev_att = dy.concatenate([att_weights] + [dy.zeros(1)]*(5-N) )
                else:
                    prev_att = att_weights[startt:endd]
                if prev_att.dim()[0][0] != 5:
                    print(prev_att.dim())
                context = input_mat * att_weights

                vector = dy.concatenate([context, tag_context, last_output_embeddings])
                s_0 = s.add_input(vector)
                out_vector = self.decoder_w * s_0.output() + self.decoder_b
                probs = dy.softmax(out_vector).npvalue()

                # Add length norm
                length_norm = np.power(5 + i, LENGTH_NORM_WEIGHT)/(np.power(6,LENGTH_NORM_WEIGHT))
                probs = probs/length_norm


                last_states[hyp_id] = s_0.s()

                # find best candidate outputs
                n_best_indices = myutil.argmax(probs, beam_size)
                for index in n_best_indices:
                    this_score = prefix_prob + np.log(probs[index])
                    next_beam_id.append((this_score, hyp_id, index, prev_att))
                next_beam_id.sort(key=itemgetter(0), reverse=True)
                next_beam_id = next_beam_id[:beam_size]

            # Create the most probable hypotheses
            # add the most probable expansions from all hypotheses to the beam
            new_hypos = []
            for item in next_beam_id:
                hypid = item[1]
                this_prob = item[0]
                char_id = item[2]
                next_sentence = beam[i - 1][hypid][2] + [int2char[char_id]]
                new_hyp = (this_prob, last_states[hypid], next_sentence, item[3])
                new_hypos.append(new_hyp)
                if next_sentence[-1] == EOS or i == MAX_PREDICTION_LEN-1:
                    if ''.join(next_sentence) != "<EOS>" and ''.join(next_sentence) != "<EOS><EOS>" and ''.join(next_sentence) != "<EOS><EOS><EOS>":
                        nbest.append(new_hyp)

            beam[i] = new_hypos
            i += 1
            if len(nbest) > 0:
                nbest.sort(key=itemgetter(0), reverse=True)
                nbest = nbest[:beam_size]
            if len(nbest) == beam_size and (len(new_hypos) == 0 or (nbest[-1][0] >= new_hypos[0][0])):
                break

        return nbest


    def get_loss(self, input_sentence, input_tags, output_sentence, lang_id, weight=1, tf_prob=1.0):
        embedded = self.embed_sentence(input_sentence)
        encoded = self.encode_sentence(embedded)
        #encoded = dy.dropout(encoded, DROPOUT_PROB)
        embedded_tags = self.embed_tags(input_tags)
        #encoded_tags = self.encode_tags(enc_tag_lstm, embedded_tags)
        encoded_tags = self.self_encode_tags(embedded_tags)

        return self.decode(encoded, encoded_tags, output_sentence, lang_id, weight, tf_prob)



def ensemble_generate_nbest(inf_models, ensemble_weights, in_seq, tag_seq, beam_size=4):
    dy.renew_cg()
    n_models = len(inf_models)
    embedded = {}
    encoded = {}
    embedded_tags = {}
    encoded_tags = {}
    input_mat = {}
    tag_input_mat = {}
    prev_att = {}
    for i in range(n_models):
        embedded[i] = inf_models[i].embed_sentence(in_seq)
        encoded[i] = inf_models[i].encode_sentence(embedded[i])
        embedded_tags[i] = inf_models[i].embed_tags(tag_seq)
        #encoded_tags[i] = inf_models[i].encode_tags(embedded_tags[i])
        encoded_tags[i] = inf_models[i].self_encode_tags(embedded_tags[i])
        input_mat[i] = dy.concatenate_cols(encoded[i])
        tag_input_mat[i] = dy.concatenate_cols(encoded_tags[i])
        prev_att[i] = dy.zeros(5)

    tmpinseq = [EOS] + list(in_seq) + [EOS]
    N = len(tmpinseq)

    last_output_embeddings = {}
    init_vector = {}
    s_0 = {}
    w1dt = {}
    tag_w1dt = {}
    for i in range(n_models):
        last_output_embeddings[i] = inf_models[i].output_lookup[char2int[EOS]]
        init_vector[i] = dy.concatenate([encoded[i][-1], encoded_tags[i][-1], last_output_embeddings[i]])
        s_0[i] = inf_models[i].dec_lstm.initial_state().add_input(init_vector[i])
        w1dt[i] = inf_models[i].attention_w1 * input_mat[i]
        tag_w1dt[i] = inf_models[i].tag_attention_w1 * tag_input_mat[i]

    beam = {0: [(0, [s_0[i].s() for i in range(n_models)] , [], [prev_att[i] for i in range(n_models)] )]}

    i = 1

    nbest = []
     # we'll need this
    last_states = {}

    MAX_PREDICTION_LEN = max(len(in_seq)*1.5,MAX_PREDICTION_LEN_DEF)

    # expand another step if didn't reach max length and there's still beams to expand
    while i < MAX_PREDICTION_LEN and len(beam[i - 1]) > 0:
        # create all expansions from the previous beam:
        next_beam_id = []
        for hyp_id, hypothesis in enumerate(beam[i - 1]):
            # expand hypothesis tuple
            #prefix_seq, prefix_prob, prefix_decoder, prefix_context, prefix_tag_context = hypothesis
            prefix_prob, prefix_decoders, prefix_seq, prefix_atts = hypothesis
            
            if i > 1:
                last_hypo_symbol = prefix_seq[-1]
            else:
                last_hypo_symbol = EOS

            # cant expand finished sequences
            if last_hypo_symbol == EOS and i > 3:
                continue
            # expand from the last symbol of the hypothesis
            last_output_embeddings = {}
            for k in range(n_models):
                last_output_embeddings[k] = inf_models[k].output_lookup[char2int[last_hypo_symbol]]

            # Perform the forward step on the decoder
            # First, set the decoder's parameters to what they were in the previous step
            s = {}
            if (i == 1):
                for k in range(n_models):
                    s[k] = inf_models[k].dec_lstm.initial_state().add_input(init_vector[k])
            else:
                for k in range(n_models):
                    s[k] = inf_models[k].dec_lstm.initial_state(prefix_decoders[k])

            s_0 = {}
            probs = {}
            state = {}
            tag_att_weights = {}
            tag_context = {}
            tag_context2 = {}
            new_state = {}
            att_weights = {}
            prev_att2 = {}
            context = {}
            vector = {}
            out_vector = {}
            for k in range(n_models):
                state[k] = dy.concatenate(list(s[k].s()))
                tag_att_weights[k] = inf_models[k].attend_tags(state[k], tag_w1dt[k])
                tag_context[k] = tag_input_mat[k] * tag_att_weights[k]
                tag_context2[k] = dy.concatenate([tag_context[k],tag_context[k]])
                new_state[k] = state[k] + tag_context2[k]

                att_weights[k] = inf_models[k].attend_with_prev(new_state[k], w1dt[k], prefix_atts[k])
                best_ic = np.argmax(att_weights[k].vec_value())
                startt = min(best_ic-2, N-6)
                if startt < 0:
                    startt = 0
                endd = startt+5
                if N < 5:
                    prev_att2[k] = dy.concatenate([att_weights[k]] + [dy.zeros(1)]*(5-N) )
                else:
                    prev_att2[k] = att_weights[k][startt:endd]
                if prev_att2[k].dim()[0][0] != 5:
                    print(prev_att2[k].dim())
                context[k] = input_mat[k] * att_weights[k]

                vector[k] = dy.concatenate([context[k], tag_context[k], last_output_embeddings[k]])
                s_0[k] = s[k].add_input(vector[k])
                out_vector[k] = inf_models[k].decoder_w * s_0[k].output() + inf_models[k].decoder_b
                
                probs[k] = dy.softmax(out_vector[k]).npvalue()

                # Add length norm
                length_norm = np.power(5 + i, LENGTH_NORM_WEIGHT)/(np.power(6,LENGTH_NORM_WEIGHT))
                probs[k] = probs[k]/length_norm

                if k == 0:
                    last_states[hyp_id] = []    
                last_states[hyp_id].append(s_0[k].s())

            # Combine the ensemble probabilities
            ensemble_probs = probs[0]
            for k in range(1,n_models):
                ensemble_probs += probs[k]
            # find best candidate outputs
            n_best_indices = myutil.argmax(ensemble_probs, beam_size)
            for index in n_best_indices:
                this_score = prefix_prob + np.log(ensemble_probs[index]/float(n_models))
                next_beam_id.append((this_score, hyp_id, index, [prev_att2[k] for k in range(n_models)]))
            next_beam_id.sort(key=itemgetter(0), reverse=True)
            next_beam_id = next_beam_id[:beam_size]

        # Create the most probable hypotheses
        # add the most probable expansions from all hypotheses to the beam
        new_hypos = []
        for item in next_beam_id:
            hypid = item[1]
            this_prob = item[0]
            char_id = item[2]
            next_sentence = beam[i - 1][hypid][2] + [int2char[char_id]]
            new_hyp = (this_prob, last_states[hypid], next_sentence, item[3])
            new_hypos.append(new_hyp)
            if next_sentence[-1] == EOS or i == MAX_PREDICTION_LEN-1:
                if ''.join(next_sentence) != "<EOS>" and ''.join(next_sentence) != "<EOS><EOS>" and ''.join(next_sentence) != "<EOS><EOS><EOS>":
                    nbest.append(new_hyp)

        beam[i] = new_hypos
        i += 1
        if len(nbest) > 0:
            nbest.sort(key=itemgetter(0), reverse=True)
            nbest = nbest[:beam_size]
        if len(nbest) == beam_size and (len(new_hypos) == 0 or (nbest[-1][0] >= new_hypos[0][0])):
            break

    return nbest


def test_beam_ensemble(inf_models, weights, beam_size=4, fn=None):
    ks = list(range(len(test_i)))
    with codecs.open(fn, 'w', 'utf-8') as outf:
        for j,k in enumerate(ks):
            out = ensemble_generate_nbest(inf_models, weights, test_i[k], test_t[k], beam_size)
            if len(out):
                word = ''.join([c for c in out[0][2] if c != EOS])
                out1 = ''.join(out[0][2][1:-1])
            elif out:
                word = ''.join([c for c in out[0][2] if c != EOS])
            else:
                print("no out")
                word = ''.join(test_i[k])
            outf.write(''.join(test_i[k]) + '\t' + word + '\t' + ';'.join(test_t[k]) + '\n')

    return



def eval_dev_beam_ensemble(inf_models, weights, beam_size=4, K=100, epoch=0):
    if K == "all":
        K = len(dev_i)
    ks = list(range(len(dev_i)))
    shuffle(ks)
    ks = ks[:K]
    outs = []
    levs = []
    correct = 0.0
    for j,k in enumerate(ks):
        out = ensemble_generate_nbest(inf_models, weights, dev_i[k], dev_t[k], beam_size)
        if len(out):
            word = ''.join([c for c in out[0][2] if c != EOS])
            out1 = ''.join(out[0][2][1:-1])
        elif out:
            word = ''.join([c for c in out[0][2] if c != EOS])
        else:
            print("no out")
            word = ''.join(dev_i[k])
        outs.append(word)
        lev = myutil.edit_distance(word, dev_o[k])
        levs.append(lev)
        if list(out1) == dev_o[k]:
            correct += 1

    accuracy = correct/float(K)
    avg_edit = np.average(np.array(levs))
    return accuracy, avg_edit



def test_beam(inf_model, beam_size=4, fn=None):
    ks = list(range(len(test_i)))
    correct = 0.0
    with codecs.open(fn, 'w', 'utf-8') as outf:
        for j,k in enumerate(ks):
            out = inf_model.generate_nbest(test_i[k], test_t[k], beam_size)
            if len(out):
                word = ''.join([c for c in out[0][2] if c != EOS])
                out1 = ''.join(out[0][2][1:-1])
            elif out:
                word = ''.join([c for c in out[0][2] if c != EOS])
            else:
                word = ''.join(test_i[k])
            outf.write(''.join(test_i[k]) + '\t' + word + '\t' + ';'.join(test_t[k]) + '\n')
        
    return 




def draw_decode(inf_model, K=20):
    for k in range(K):
        filename = os.path.join(FIGURE_DIR, str(k))
        inf_model.draw_decode(dev_i[k], dev_t[k], dev_o[k], show_att=True, show_tag_att=True, fn=filename)
    return


def eval_dev_beam(inf_model, beam_size=4, K=100, epoch=0):
    if K == "all":
        K = len(dev_i)
    ks = list(range(len(dev_i)))
    shuffle(ks)
    ks = ks[:K]
    outs = []
    levs = []
    correct = 0.0
    for j,k in enumerate(ks):
        out = inf_model.generate_nbest(dev_i[k], dev_t[k], beam_size)
        if len(out):
            word = ''.join([c for c in out[0][2] if c != EOS])
            out1 = ''.join(out[0][2][1:-1])
        elif out:
            word = ''.join([c for c in out[0][2] if c != EOS])
        else:
            word = ''.join(dev_i[k])
        outs.append(word)
        lev = myutil.edit_distance(word, dev_o[k])
        levs.append(lev)
        if list(out1) == dev_o[k]:
            correct += 1

    accuracy = correct/float(K)
    avg_edit = np.average(np.array(levs))
    return accuracy, avg_edit


def eval_dev_greedy(inf_model, K=100, epoch=0):
    if K == "all":
        K = len(dev_i)
    ks = list(range(len(dev_i)))
    shuffle(ks)
    ks = ks[:K]
    outs = []
    levs = []
    correct = 0.0
    for j,k in enumerate(ks):
        out = inf_model.generate(dev_i[k], dev_t[k])
        outs.append(out)
        lev = myutil.edit_distance(out, dev_o[k])
        levs.append(lev)
        if list(out) == dev_o[k]:
            correct += 1

    accuracy = correct/float(K)
    avg_edit = np.average(np.array(levs))
    return accuracy, avg_edit

def eval_dev_copy_greedy(inf_model, K=40, epoch=0):
    if K == "all":
        K = len(dev_i)
    ks = list(range(len(dev_i)))
    shuffle(ks)
    ks = ks[:K]
    outs = []
    levs = []
    correct = 0.0
    for j,k in enumerate(ks):
        out = inf_model.generate(dev_i[k], [NULL])
        outs.append(out)
        lev = myutil.edit_distance(out, dev_i[k])
        levs.append(lev)
        if list(out) == dev_i[k]:
            correct += 1

    accuracy = correct/float(K)
    avg_edit = np.average(np.array(levs))
    return accuracy, avg_edit



def train_simple_attention_with_tags(inf_model, inputs, tags, outputs, lang_ids=None, finetune=False, trainer=None, prev_acc=None, prev_edd=None):
    indexes = list(range(len(inputs)))
    tasks = [0,1,2]
    burnin_pairs = [(j,t) for j in indexes for t in tasks[:2]]
    total_burnin_pairs = len(burnin_pairs)
    train_pairs = [(j,t) for j in indexes for t in tasks]
    total_train_pairs = len(train_pairs)
    finetune_pairs = [(j,t) for j in indexes for t in tasks[1:]]
    total_finetune_pairs = len(finetune_pairs)
    final_finetune_pairs = [(j,t) for j in indexes for t in [1]]
    total_final_finetune_pairs = len(finetune_pairs)

    learning_rate = STARTING_LEARNING_RATE
    trainer = trainer or dy.SimpleSGDTrainer(inf_model.model, learning_rate)
    epochs_since_improv = 0
    halvings = 0
    #trainer.set_clip_threshold(-1.0)
    #trainer.set_sparse_updates(True if args.SPARSE == 1 else False)

    prev_acc = prev_acc or 0.0
    prev_edd = prev_edd or 100
    if lang_ids == None:
        lang_ids = np.zeros(len(burnin_pairs))

    if not finetune:
        # Learn to copy -- burn in
        MINIBATCH_SIZE = 10
        for i in range(100):
            shuffle(burnin_pairs)
            total_loss = 0.0
            batch = []
            dy.renew_cg()
            for j,t in burnin_pairs:
                #task 0 is copy input
                if t == 0:
                    loss = inf_model.get_loss(inputs[j], [NULL], inputs[j], lang_ids[j])
                # task 1 is copy output with tag
                else:
                    loss = inf_model.get_loss(outputs[j], tags[j], outputs[j], lang_ids[j])
                batch.append(loss)
                if len(batch) == MINIBATCH_SIZE or j == total_burnin_pairs:
                    loss = dy.esum(batch)/len(batch)
                    total_loss += loss.value()
                    loss.backward()
                    trainer.update()
                    batch = []
                    dy.renew_cg()
            if i % 1 == 0:
                trainer.status()
                print("Epoch "+str(i)+" : "+ str(total_loss))
                acc, edd = eval_dev_copy_greedy(inf_model, 'all', i)
                print("\t COPY Accuracy: "+ str(acc) + " average edit distance: " + str(edd))
            if edd < prev_edd:
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME+"edd.model"))
            if (acc > prev_acc and edd < prev_edd) or (acc >= prev_acc and edd < prev_edd) or (acc > prev_acc and edd <= prev_edd):
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME+"both.model"))
                epochs_since_improv = 0
            else:
                epochs_since_improv += 1
            if acc > prev_acc:
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME+"acc.model"))
                epochs_since_improv = 0
            if acc > prev_acc:
                prev_acc = acc
            if edd < prev_edd:
                prev_edd = edd
            if epochs_since_improv > EPOCHS_TO_HALVE:
                print("Restarting the trainer with half the learning rate!")
                learning_rate = learning_rate/2
                halvings += 1
                if halvings == 1:
                    break
                trainer.restart(learning_rate)
                epochs_since_improv = 0
                inf_model.model.populate(os.path.join(MODEL_DIR, MODEL_NAME+"acc.model"))
            if acc > COPY_THRESHOLD:
                print("Accuracy good enough, breaking")
                break

        
        # epochs
        # We don't care for the performance on the copy burnin task
        halvings = 0
        prev_acc = 0.0
        prev_edd = 100
        MINIBATCH_SIZE = 1
        for i in range(40):
            shuffle(train_pairs)
            total_loss = 0.0
            batch = []
            weight = 0.0
            dy.renew_cg()
            for j,t in train_pairs:
                if (t == 0 or t==1):
                    if random() > COPY_TASK_PROB:
                        continue
                if t == 0:
                    loss = inf_model.get_loss(inputs[j], [NULL], inputs[j], lang_ids[j], COPY_WEIGHT, 0.8)
                    weight += COPY_WEIGHT
                elif t == 1:
                    loss = inf_model.get_loss(outputs[j], tags[j], outputs[j], lang_ids[j], COPY_WEIGHT, 0.8)
                    weight += COPY_WEIGHT
                elif t == 2:
                    loss = inf_model.get_loss(inputs[j], tags[j], outputs[j], lang_ids[j], 1, 0.8)
                    weight += 1.0
                batch.append(loss)
                if len(batch) == MINIBATCH_SIZE  or j == total_train_pairs:
                    loss = dy.esum(batch)/weight
                    total_loss += loss.value()
                    loss.backward()
                    trainer.update()
                    batch = []
                    dy.renew_cg()
                    weight = 0.0
            if i % 1 == 0:
                trainer.status()
                print("Epoch ", i, " : ", total_loss)
                acc, edd = eval_dev_copy_greedy(inf_model, 20, 100+i)
                print("\t COPY Accuracy: ", acc, " average edit distance: ", edd)
                acc, edd = eval_dev_greedy(inf_model, 100, 100+i)
                print("\t TASK Accuracy: ", acc, " average edit distance: ", edd)
            if acc > prev_acc:
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME+"acc.model"))
                epochs_since_improv = 0
            if edd < prev_edd:
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME+"edd.model"))
            if (acc > prev_acc and edd < prev_edd) or (acc >= prev_acc and edd < prev_edd) or (acc > prev_acc and edd <= prev_edd):
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME+"both.model"))
                epochs_since_improv = 0
            else:
                epochs_since_improv += 1
            if acc > prev_acc:
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME+"acc.model"))
                epochs_since_improv = 0
            if acc > prev_acc:
                prev_acc = acc
            if edd < prev_edd:
                prev_edd = edd
            if epochs_since_improv > EPOCHS_TO_HALVE:
                print("Restarting the trainer with half the learning rate!")
                halvings += 1
                if halvings == 2:
                    break
                learning_rate = learning_rate/2
                trainer.restart(learning_rate)
                epochs_since_improv = 0
                inf_model.model.populate(os.path.join(MODEL_DIR, MODEL_NAME+"acc.model"))
            if acc > 0.9 and epochs_since_improv == 4:
                print("Accuracy good enough, breaking")
                break

                
    else:
        MINIBATCH_SIZE = 1
        if learning_rate < 0.05:
            learning_rate = 0.05
        trainer.restart(learning_rate)
        halvings = 0
        for i in range(40):
            shuffle(finetune_pairs)
            total_loss = 0.0
            weight = 0.0
            batch = []
            dy.renew_cg()
            for j,t in finetune_pairs:
                if t == 1:
                    if random() > COPY_TASK_PROB:
                        continue
                    loss = inf_model.get_loss(outputs[j], tags[j], outputs[j], lang_ids[j], COPY_WEIGHT, 0.5)
                    weight += COPY_WEIGHT
                elif t == 2:
                    loss = inf_model.get_loss(inputs[j], tags[j], outputs[j], lang_ids[j], 1, 0.5)
                    weight += 1
                batch.append(loss)
                if len(batch) == MINIBATCH_SIZE or j == total_finetune_pairs:
                    loss = dy.esum(batch)/weight
                    total_loss += loss.value()
                    loss.backward()
                    trainer.update()
                    batch = []
                    dy.renew_cg()
                    weight = 0.0
            if i % 1 == 0:
                print("Epoch ", i, " : ", total_loss)
                trainer.status()
                acc, edd = eval_dev_copy_greedy(inf_model, 20, 140+i)
                print("\t COPY Accuracy: ", acc, " average edit distance: ", edd)
                acc, edd = eval_dev_greedy(inf_model, "all", 140+i)
                print("\t TASK Accuracy: ", acc, " average edit distance: ", edd)
            if acc > prev_acc:
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME+"acc.model"))
            if edd < prev_edd:
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME+"edd.model"))
            if (acc > prev_acc and edd < prev_edd) or (acc >= prev_acc and edd < prev_edd) or (acc > prev_acc and edd <= prev_edd):
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME+"both.model"))
                epochs_since_improv = 0
            else:
                epochs_since_improv += 1
            if acc > prev_acc:
                prev_acc = acc
                epochs_since_improv = 0
            if edd < prev_edd:
                prev_edd = edd
            if epochs_since_improv > EPOCHS_TO_HALVE:
                print("Restarting the trainer with half the learning rate!")
                halvings += 1
                if halvings == 2:
                    break
                learning_rate = learning_rate/2
                trainer.restart(learning_rate)
                epochs_since_improv = 0
                inf_model.model.populate(os.path.join(MODEL_DIR, MODEL_NAME+"acc.model"))

        halvings = 0
        for i in range(40):
            shuffle(indexes)
            total_loss = 0.0
            batch = []
            dy.renew_cg()
            for j,t in final_finetune_pairs:
                loss = inf_model.get_loss(inputs[j], tags[j], outputs[j], lang_ids[j], 1, 0.5)
                batch.append(loss)
                if len(batch) == MINIBATCH_SIZE or j == total_final_finetune_pairs:
                    loss = dy.esum(batch)/len(batch)
                    total_loss += loss.value()
                    loss.backward()
                    trainer.update()
                    batch = []
                    dy.renew_cg()
            if i % 1 == 0:
                print("Epoch ", i, " : ", total_loss)
                trainer.status()
                acc, edd = eval_dev_copy_greedy(inf_model, 20, 160+i)
                print("\t COPY Accuracy: ", acc, " average edit distance: ", edd)
                acc, edd = eval_dev_greedy(inf_model, "all", 160+i)
                print("\t TASK Accuracy: ", acc, " average edit distance: ", edd)
            if acc > prev_acc:
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME+"acc.model"))
            if edd < prev_edd:
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME+"edd.model"))
            if (acc > prev_acc and edd < prev_edd) or (acc >= prev_acc and edd < prev_edd) or (acc > prev_acc and edd <= prev_edd):
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME+"both.model"))
                epochs_since_improv = 0
            else:
                epochs_since_improv += 1
            if acc > prev_acc:
                prev_acc = acc
                epochs_since_improv = 0
            if edd < prev_edd:
                prev_edd = edd
            if epochs_since_improv > EPOCHS_TO_HALVE:
                print("Restarting the trainer with half the learning rate!")
                halvings += 1
                if halvings == 3:
                    break
                learning_rate = learning_rate/2
                trainer.restart(learning_rate)
                epochs_since_improv = 0
                inf_model.model.populate(os.path.join(MODEL_DIR, MODEL_NAME+"acc.model"))

    return trainer, prev_acc, prev_edd


# equivalent of main
if TRAIN:
    inflection_model = InflectionModel()
    if PREDICT_LANG:
        if ORIGINAL or SWAP:
            #lids_1 = [0]*MULTIPLY*len(low_i) + [1]*len(high_i)
            trainer, best_acc, best_edd = train_simple_attention_with_tags(inflection_model, MULTIPLY*low_i+high_i, MULTIPLY*low_t+high_t, MULTIPLY*low_o+high_o, lids_1)
            print("Best dev accuracy after pre-training: ", best_acc)
            print("Best dev lev distance after pre-training: ", best_edd)
            lids_2 = [0]*len(low_i)
            trainer2, best_acc, best_edd = train_simple_attention_with_tags(inflection_model, low_i, low_t, low_o, lids_2, True, trainer, best_acc, best_edd)
            print("Best dev accuracy after finetuning: ", best_acc)
            print("Best dev lev distance after finetuning: ", best_edd)
        elif LOW:
            lids_1 = [0]*len(low_i)
            trainer, best_acc, best_edd = train_simple_attention_with_tags(inflection_model, low_i, low_t, low_o, lids_1)
            print("Best dev accuracy after pre-training: ", best_acc)
            print("Best dev lev distance after pre-training: ", best_edd)
            lids_2 = [0]*len(low_i)
            trainer2, best_acc, best_edd = train_simple_attention_with_tags(inflection_model, low_i, low_t, low_o, lids_2, True, trainer, best_acc, best_edd)
            print("Best dev accuracy after finetuning: ", best_acc)
            print("Best dev lev distance after finetuning: ", best_edd)

    else:
        if ORIGINAL or SWAP:
            trainer, best_acc, best_edd = train_simple_attention_with_tags(inflection_model, MULTIPLY*low_i+high_i, MULTIPLY*low_t+high_t, MULTIPLY*low_o+high_o)
            print("Best dev accuracy after pre-training: ", best_acc)
            print("Best dev lev distance after pre-training: ", best_edd)
            trainer2, best_acc, best_edd = train_simple_attention_with_tags(inflection_model, low_i, low_t, low_o, None, True, trainer, best_acc, best_edd)
            print("Best dev accuracy after finetuning: ", best_acc)
            print("Best dev lev distance after finetuning: ", best_edd)
        elif LOW:
            trainer, best_acc, best_edd = train_simple_attention_with_tags(inflection_model, low_i, low_t, low_o)
            print("Best dev accuracy after pre-training: ", best_acc)
            print("Best dev lev distance after pre-training: ", best_edd)
            trainer2, best_acc, best_edd = train_simple_attention_with_tags(inflection_model, low_i, low_t, low_o, None, True, trainer, best_acc, best_edd)
            print("Best dev accuracy after finetuning: ", best_acc)
            print("Best dev lev distance after finetuning: ", best_edd)


elif TEST_DEV:
    inflection_model = InflectionModel()
    inflection_model.model.populate(os.path.join(MODEL_DIR, MODEL_NAME+"acc.model"))
    #acc, edd = eval_dev_greedy(enc_fwd_lstm, enc_bwd_lstm, dec_lstm, "all", "test")
    acc, edd = eval_dev_beam(inflection_model, 8, "all", "test") # it was 8 beams
    print("Best dev accuracy at test: ", acc)
    print("Best dev lev distance at test: ", edd)

elif DRAW_DEV:
    inflection_model = InflectionModel()
    inflection_model.model.populate(os.path.join(MODEL_DIR, MODEL_NAME+"acc.model"))
    draw_decode(inflection_model)


elif TEST_DEV_ENSEMBLE:
    inflection_model1 = InflectionModel()
    inflection_model1.model.populate(os.path.join(MODEL_DIR, MODEL_NAME+"acc.model"))
    inflection_model2 = InflectionModel()
    inflection_model2.model.populate(os.path.join(MODEL_DIR, MODEL_NAME+"edd.model"))
    acc, edd = eval_dev_beam_ensemble([inflection_model1, inflection_model2], [0.5, 0.5], 8, "all", "test")
    print("Best dev accuracy at test: ", acc)
    print("Best dev lev distance at test: ", edd)



elif TEST:
    inflection_model = InflectionModel()
    inflection_model.model.populate(os.path.join(MODEL_DIR, MODEL_NAME+"acc.model"))
    if args.outputfile:
        test_beam(inflection_model, 8, args.outputfile)
    else:
        test_beam(inflection_model, 8, os.path.join(OUTPUT_DIR,MODEL_NAME+"test.output"))


elif TEST_ENSEMBLE:
    inflection_model1 = InflectionModel()
    inflection_model1.model.populate(os.path.join(MODEL_DIR, MODEL_NAME+"acc.model"))
    inflection_model2 = InflectionModel()
    inflection_model2.model.populate(os.path.join(MODEL_DIR, MODEL_NAME+"edd.model"))
    inflection_model3 = InflectionModel()
    inflection_model3.model.populate(os.path.join(MODEL_DIR, MODEL_NAME+"both.model"))
    test_beam_ensemble([inflection_model1, inflection_model2, inflection_model3], [0.34, 0.33, 0.33], 8, os.path.join(OUTPUT_DIR,MODEL_NAME+"test.ensemble.output"))


elif TEST_TWO_ENSEMBLE:
    mixing_weights = compute_mixing_weights(2)
    inflection_model1 = InflectionModel()
    inflection_model1.model.populate(os.path.join(MODEL_DIR,"orig.acc.model"))
    inflection_model2 = InflectionModel()
    inflection_model2.model.populate(os.path.join(MODEL_DIR, "swap.acc.model"))
    test_beam_ensemble([inflection_model1, inflection_model2], mixing_weights, 8, os.path.join(OUTPUT_DIR,"test.two_ensemble.output"))
    
    

elif TEST_THREE_ENSEMBLE:
    mixing_weights = compute_mixing_weights(3)
    inflection_model1 = InflectionModel()
    inflection_model1.model.populate(os.path.join(MODEL_DIR,"orig.acc.model"))
    inflection_model2 = InflectionModel()
    inflection_model2.model.populate(os.path.join(MODEL_DIR,"swap.acc.model"))
    inflection_model3 = InflectionModel()
    inflection_model3.model.populate(os.path.join(MODEL_DIR,"low.both.model"))
    test_beam_ensemble([inflection_model1, inflection_model2, inflection_model3], mixing_weights, 8, os.path.join(OUTPUT_DIR,"test.three_ensemble.output"))
    

elif TEST_ALL_ENSEMBLE:
    mixing_weights = compute_mixing_weights(2)
    inflection_model1 = InflectionModel()
    inflection_model1.model.populate(os.path.join(MODEL_DIR,"orig.acc.model"))
    inflection_model2 = InflectionModel()
    inflection_model2.model.populate(os.path.join(MODEL_DIR,"swap.acc.model"))
    inflection_model3 = InflectionModel()
    inflection_model3.model.populateos.path.join((MODEL_DIR,"orig.edd.model"))
    inflection_model4 = InflectionModel()
    inflection_model4.model.populate(os.path.join(MODEL_DIR,"swap.edd.model"))
    test_beam_ensemble([inflection_model1, inflection_model2, inflection_model3, inflection_model4], mixing_weights+mixing_weights, 8, os.path.join(OUTPUT_DIR,"test.all_ensemble.output"))
    
