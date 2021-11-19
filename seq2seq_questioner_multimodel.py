from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import csv
import pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import optim
import torch.nn.functional as F
import time
import math
import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AGENT_TYPE = 'seq2seq'
BATCH_SIZE = 1
MAX_INPUT_LENGTH = 160
WORD_EMBEDDING_SIZE = 256
ACTION_EMBEDDING_SIZE = 32
TARGET_EMBEDDING_SIZE = 32
HIDDEN_SIZE = 512
BIDIRECTIONAL = False
DROPOUT_RATIO= 0.5
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0005
FEAT_DIR = "./data/json_feat_2.1.0/"
AUGMENT_DIR = "./data/generated_2.1.0/"
MAX_LENGTH = 4

SOS_token = 0
EOS_token = 1

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s, idx=0):
    # skip tasks, trials and subgoals
    if idx >= 2:
        return s
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split() if word in lang.word2index]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(lang, pair):
    input_tensor = tensorFromSentence(lang, pair[0])
    target_tensor = tensorFromSentence(lang, pair[1])
    task = pair[2]
    trial = pair[3]
    sg = pair[4]
    return (input_tensor, target_tensor, task, trial, sg)

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def readLangs(lang1, lang2, fn, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(fn, encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s, idx) for idx, s in enumerate(l.split('\t'))] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def organizeTrainQA(train_csv, train_txt, nec_q):
    # oraganize training questions and answers as a txt file
    # train_csv: the csv file that contains the Alfred instructions and questions asked by users
    # nec_q: whether to use Turker's response about whether the question is necessary
    with open(train_csv, "r") as inputcsv:
        csvreader = csv.reader(inputcsv, delimiter=',')
        header = next(csvreader)
        instr_idx = header.index("instruction")
        task_idx = header.index("task")
        trial_idx = header.index("trial")
        sg_idx = header.index("subgoal_idx")
        nec_idx = header.index("necessary")
        qtype_idx = header.index("question_type")
        q_idx = header.index("question")
        noun_st_idx = header.index("noun1")
        noun_end_idx = header.index("noun2")
        rows = list(csvreader)
        
    num_train = len(rows)
    task2q = {}

    for i in range(num_train):
        row = rows[i]
        question = row[q_idx]
        noun_names = row[noun_st_idx:noun_end_idx+1]
        noun_names_valid = [n for n in noun_names if not n == "none"]
        nec_question = row[nec_idx]
        qtype = row[qtype_idx]

        # skip other questions
        if qtype == "other" or len(noun_names_valid) == 0:
            continue

        # handle unnecessary question
        if nec_question == "FALSE":
            qtype = "none"
            noun = np.random.choice(noun_names_valid)
        
        # handle valid questions that are necessary
        else:
            # for direction question, randomly choose noun
            if qtype == "direction":
                noun = np.random.choice(noun_names_valid)
            elif qtype == "location":
                noun = question[:-1].split()[-1]
            else:
                noun = question[:-1].split()[3]
        
        question = qtype + " " + noun
        instr = row[instr_idx]
        task_name = row[task_idx]
        trial = row[trial_idx]
        subgoal = row[sg_idx]
        key = task_name + "$" + trial + "$" + subgoal

        if key not in task2q:
            task2q[key] = [instr]
        
        # only include unique questions
        if question not in task2q[key]:
            task2q[key].append(question)
            
    with open(train_txt, "w") as f:
        for ke in task2q:
            task, trial, subgoal = ke.split("$")
            instr = task2q[ke][0]
            for question in task2q[ke][1:]:
                f.write(instr + "\t" + question + "\t" + task + "\t" + trial + "\t" + subgoal + "\n")

def prepareQAData(lang1, lang2, fn, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, fn, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    return input_lang, output_lang, pairs

def prepareDataTest(input_csv_fn):
    # prepare the data for the whole test set
    with open(input_csv_fn, "r") as inputcsv:
        csvreader = csv.reader(inputcsv, delimiter=',')
        header = next(csvreader)
        instr_idx = header.index("task_instr")
        rows = list(csvreader)
        
    num_test = len(rows)
    sentences = []
    words = ["location", "appearance", "direction", "none"]
        
    for i in range(num_test):
        row = rows[i]
        instr = row[instr_idx]
        if instr == "none":
            continue
        sentences.append(normalizeString(instr))
    
    lang = Lang("english")
    for w in words:
        lang.addWord(w)
    for s in sentences:
        lang.addSentence(s)
    
    return lang

class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, 
                            dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, self.num_layers, 
                            batch_first=True, dropout=dropout_ratio, 
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions
        )

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a 
            list of lengths for dynamic batching. '''
        embeds = self.embedding(inputs)   # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        decoder_init = nn.Tanh()(self.encoder2decoder(h_t))

        ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
        ctx = self.drop(ctx)
        return ctx,decoder_init,c_t  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)

class SoftDotAttention(nn.Module):
    '''Soft Dot Attention. 
    '''

    def __init__(self, dim, ctx_dim=None):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        dim_ctx = dim if ctx_dim is None else ctx_dim
        self.linear_in = nn.Linear(dim, dim_ctx, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim + dim_ctx, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax 
            attn.data.masked_fill_(mask, -float('inf'))              
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, h), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn

class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, input_action_size, output_action_size, embedding_size, hidden_size, 
                      dropout_ratio, feature_size=512):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_action_size, embedding_size)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.attention_layer = SoftDotAttention(hidden_size)
        self.decoder2action = nn.Linear(hidden_size, output_action_size)

    def forward(self, action, feature, h_0, c_0, ctx, ctx_mask=None):
        ''' Takes a single step in the decoder LSTM (allowing sampling).

        action: batch x target sequence len
        feature: batch x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''
        action_embeds = self.embedding(action)   # (batch, input seq length, embedding_size)
        action_embeds = torch.flatten(action_embeds, start_dim=1) # (batch, input seq length * embedding_size)
        concat_input = torch.cat((action_embeds, feature), 1) # (batch, input seq length*embedding_size + feature_size)
        drop = self.drop(concat_input)
        h_1,c_1 = self.lstm(drop, (h_0,c_0))
        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)        
        logit = self.decoder2action(h_tilde)
        return h_1,c_1,alpha,logit

def train(input_tensor, target_tensor, f_t, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    # input_tensor: subgoal instructions (batch, seq_len)
    # f_t: image features for the subgoal (batch, feature size)
    # target_tensor: target questions (batch, question_len)
    encoder.init_state(input_tensor)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)
    seq_lengths = torch.from_numpy(np.array([input_length]))
    # seq_lengths = Variable(seq_lengths, requires_grad=False).long().cuda()
    loss = 0
    ctx, h_t, c_t = encoder(input_tensor, seq_lengths)
    decoder_input = torch.tensor([[SOS_token]], device=device)

    # teacher forcing
    for di in range(target_length):
        h_t, c_t, alpha, logit = decoder(decoder_input, f_t, h_t, c_t, ctx)
        loss += criterion(logit, target_tensor[:, di])
        decoder_input = target_tensor[:, di].unsqueeze(0)  # Teacher forcing
        
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(lang, pairs, encoder, decoder, n_iters, print_every=1000, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    training_pairs = [tensorsFromPair(lang, random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.CrossEntropyLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = torch.unsqueeze(torch.squeeze(training_pair[0]), 0)
        target_tensor = torch.unsqueeze(torch.squeeze(training_pair[1]), 0)
        task = training_pair[2]
        trial = training_pair[3]
        sg = training_pair[4]

        f_t = extractFeature(training_pair)
        loss = train(input_tensor, target_tensor, f_t, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

def inferQuestion(lang, encoder, decoder, instr, feat):
    output_words = evaluate(lang, encoder, decoder, instr, feat, max_length=MAX_LENGTH)
    if "EOS" in output_words:
        output_words.remove("EOS")
    output_sentence = ' '.join(output_words)
    
    return output_sentence

def evaluate(lang, encoder, decoder, sentence, f_t, max_length):
    with torch.no_grad():
        input_tensor = torch.unsqueeze(torch.squeeze(tensorFromSentence(lang, sentence)), 0)
        input_length = input_tensor.size(1)
        encoder.init_state(input_tensor)
        seq_lengths = torch.from_numpy(np.array([input_length]))
        ctx, h_t, c_t = encoder(input_tensor, seq_lengths)

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoded_words = []

        for di in range(max_length):
            h_t, c_t, alpha, logit = decoder(decoder_input, f_t, h_t, c_t, ctx)
            topv, topi = logit.data.topk(1)
            if topi.item() == EOS_token:
                # decoded_words.append('EOS')
                break
            else:
                decoded_words.append(lang.index2word[topi.item()])

            decoder_input = topi.detach()

        return decoded_words

def extractFeature(pair):
    # load features from file and extract for a subgoal
    task = pair[2]
    trial = pair[3]
    sg = pair[4]

    splits = ["train", "valid_seen", "valid_unseen"]
    ft_dirs = [FEAT_DIR + split + "/" + task + "/" + trial + "/feat_conv.pt" for split in splits]
    
    for idx, ft_dir in enumerate(ft_dirs):
        if os.path.exists(ft_dir):
            ft_dir_final = ft_dir
            split_final = splits[idx]

    aug_json_dir = AUGMENT_DIR + split_final + "/" + task + "/" + trial + "/traj_data.json"
    # features for all low level actions (low action number + 1, feature size, w, h)
    f_t_all_low = torch.load(ft_dir_final)
    with open(aug_json_dir, "r") as f:
        traj_data = json.load(f)
    
    # get the feature at the beginning of the subgoal 
    low_act = traj_data["plan"]["low_actions"]
    found_sg = False

    for low_idx, la in enumerate(low_act):
        if int(la["high_idx"]) == int(sg):
            f_t_sg = f_t_all_low[low_idx, :, :, :]
            found_sg = True
            break

    if not found_sg:
        print("sg %s not found!!" % sg)
        print(aug_json_dir)
    
    avg_pool = nn.AvgPool2d(7)
    f_t = avg_pool(f_t_sg)
    f_t = torch.unsqueeze(torch.squeeze(f_t), 0).to(device)
    return f_t

def evaluateRandomly(pairs, lang, encoder, decoder, num_pairs=10, print_pairs=True):
    succ_pairs = 0
    succ_qtypes = 0
    succ_nouns = 0
    asked_q = 0
    for i in range(num_pairs):
        pair = random.choice(pairs)
        f_t = extractFeature(pair)
        output_words = evaluate(lang, encoder, decoder, pair[0], f_t, max_length=MAX_LENGTH)
        output_sentence = ' '.join(output_words)

        if print_pairs:
            print('>', pair[0])
            print('=', pair[1])
            print('<', output_sentence)
            print('')

        if pair[1] == output_sentence:
            succ_pairs += 1
        
        if pair[1].split(" ")[0] == output_sentence.split(" ")[0]:
            succ_qtypes += 1

        if len(output_sentence.split(" ")) > 1 and pair[1].split(" ")[1] == output_sentence.split(" ")[1]:
            succ_nouns += 1

        if not output_sentence.split(" ")[0] == "none":
            asked_q += 1

    print("Accuracy: %f" % (succ_pairs/num_pairs))
    print("Question type acc: %f" % (succ_qtypes/num_pairs))
    print("Noun acc: %f" % (succ_nouns/num_pairs))
    print("Asking percentage: %f" % (asked_q/num_pairs))

def pretrainQuestioner():
    validation_ratio = 0.1
    num_epoch = 10

    # prepare training and validation data
    input_csv_fn = "./data/dialfred_human_qa.csv"
    output_txt_fn = "./data/instr-question.txt"
    organizeTrainQA(input_csv_fn, output_txt_fn, nec_q=True)

    _, _, pairs = prepareQAData('instr', 'question', output_txt_fn, reverse=False)
    perm_pairs = np.random.permutation(len(pairs))
    valid_end_idx = np.floor(validation_ratio*len(pairs)).astype(int)
    pairs_np = np.array(pairs)
    valid_pairs = pairs_np[perm_pairs[0:valid_end_idx]].tolist()
    training_pairs = pairs_np[perm_pairs[valid_end_idx:]].tolist()
    
    test_csv_fn = "./data/hdc_input_augmented.csv"
    lang = prepareDataTest(test_csv_fn)

    enc_hidden_size = HIDDEN_SIZE//2 if BIDIRECTIONAL else HIDDEN_SIZE
    encoder1 = EncoderLSTM(lang.n_words, WORD_EMBEDDING_SIZE, enc_hidden_size, 
                    DROPOUT_RATIO, bidirectional=BIDIRECTIONAL).to(device)
    attn_decoder1 = AttnDecoderLSTM(lang.n_words, lang.n_words,
                    ACTION_EMBEDDING_SIZE, HIDDEN_SIZE, DROPOUT_RATIO).to(device)
    
    model_fn = "./logs/pretrained_questioner.pt"

    # train questioner and save the model
    trainIters(lang, training_pairs, encoder1, attn_decoder1, n_iters=num_epoch*len(training_pairs), print_every=100)
    torch.save({"encoder": encoder1.state_dict(), "decoder": attn_decoder1.state_dict()}, model_fn)

    # check the validity of the inferred questions
    evaluateRandomly(valid_pairs, lang, encoder1, attn_decoder1, num_pairs=len(valid_pairs))

if __name__ == "__main__":
    pretrainQuestioner()
