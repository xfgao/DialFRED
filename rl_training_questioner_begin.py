import os
from itertools import count
import time
import math
import os
import json
import matplotlib.pyplot as plt
import unicodedata
import string
import re
import random
import csv
import pickle
from io import open
import logging

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical

from alfred.gen import constants
from alfred.env.thor_env import ThorEnv
from alfred.nn.enc_visual import FeatureExtractor
from alfred.utils import data_util, model_util
from alfred.utils.eval_util import *
from alfred.data.zoo.alfred import AlfredDataset
from alfred.eval.eval_subgoals import *
from seq2seq_questioner_multimodel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ITER = 1000
LEARNING_RATE = 0.0001
BATCH_SIZE = 1
WEIGHT_DECAY = 0.0005
DROPOUT_RATIO= 0.5
BIDIRECTIONAL = False
WORD_EMBEDDING_SIZE = 256
ACTION_EMBEDDING_SIZE = 32
TARGET_EMBEDDING_SIZE = 32
HIDDEN_SIZE = 512
MAX_INPUT_LENGTH = 160
MAX_LENGTH = 2
REWARD_INVALID = -0.1
REWARD_QUESTION = -0.05
REWARD_TIME = -0.01
REWARD_SUC = 1.0

SOS_token = 0
EOS_token = 1

class Critic(nn.Module):
    def __init__(self, rnn_dim = 512, dropout=0.5):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(rnn_dim, rnn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def trainIters(args, lang, dataset, encoder, decoder, critic, performer, extractor, split_id, \
    n_iters, print_every=1, save_every=100):
    start = time.time()
    env = ThorEnv(x_display=0)
    obj_predictor = FeatureExtractor(archi='maskrcnn', device=device,
        checkpoint="./logs/pretrained/maskrcnn_model.pth", load_heads=True)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    num_subgoals = len(dataset.jsons_and_keys)
    actor_losses = []
    critic_losses = []
    all_rewards = []
    succ = []
    all_query = []
    all_instr = []
    all_pws = []
    num_q = []
    sg_pairs = []

    # record the dataset index for the original instruction without qas
    data_instruct_list = []
    for i in range(len(dataset.jsons_and_keys)):
        traj_data, traj_key = dataset.jsons_and_keys[i]
        if traj_data["repeat_idx"] == 0:
            data_instruct_list.append(i)
    print("dataset length", len(data_instruct_list))

    for it in range(0, n_iters):
        log_probs = []
        log_prob = 0
        values = []
        rewards = []
        masks = []
        current_num_q = 0
        entropy = 0

        # first sample a subgoal and get the instruction and image feature
        dataset_idx = np.random.choice(data_instruct_list)
        task_json = dataset.jsons_and_keys[dataset_idx]
        turk_annos = task_json[0]["turk_annotations"]["anns"]
        subgoal_idxs = [sg['high_idx'] for sg in task_json[0]['plan']['high_pddl']]
        # ignore the last subgoal which is often the padding one
        subgoal_idxs = subgoal_idxs[:-1]
        subgoal_idx = np.random.choice(subgoal_idxs)
        task, trial = task_json[0]['task'].split("/")
        pair = (None, None, task, trial, subgoal_idx)
        sg_pairs.append([task, trial, subgoal_idx])
        f_t = extractFeature(pair).to(device)
        orig_instr = normalizeString(turk_annos[0]["high_descs"][subgoal_idx])
        all_instr.append(orig_instr)
        input_tensor = torch.unsqueeze(torch.squeeze(tensorFromSentence(lang, orig_instr)), 0).to(device)
        input_length = input_tensor.size(1)

        # infer question based on the instruction
        encoder.init_state(input_tensor)
        seq_lengths = torch.from_numpy(np.array([input_length]))
        ctx, h_t, c_t = encoder(input_tensor, seq_lengths)
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoded_words = []
        for di in range(MAX_LENGTH):
            h_t, c_t, alpha, logit = decoder(decoder_input, f_t, h_t, c_t, ctx)
            # record the value V(s)
            if di == 0:
                value = critic(h_t)
            dist = Categorical(F.softmax(logit, dim=-1))
            selected_word = dist.sample()
            decoded_words.append(lang.index2word[selected_word.item()])
            decoder_input = selected_word.detach().to(device)
            log_prob += dist.log_prob(selected_word).unsqueeze(0)/MAX_LENGTH
            entropy += dist.entropy().mean()/MAX_LENGTH

        log_probs.append(log_prob)
        values.append(value)
        # record the value V(s')
        next_value = critic(h_t)

        # set up the trial_idx to the one with the correct qas
        repeat_idx = -1
        if decoded_words[0] == "appearance":
            query = "<<app>> " + decoded_words[1]
        elif decoded_words[0] == "location":
            query = "<<loc>> " + decoded_words[1]
        elif decoded_words[0] == "direction":
            query = "<<dir>> "
        elif decoded_words[0] == "none":
            query = "none"
            repeat_idx = 0
        else:
            query = "<<invalid>>"
            repeat_idx = 0
        
        all_query.append(query)
        if repeat_idx == -1:
            for r_idx in range(1, 6):
                if r_idx >= len(turk_annos) or subgoal_idx >= len(turk_annos[r_idx]["high_descs"]):
                    break
                instr_qa = turk_annos[r_idx]["high_descs"][subgoal_idx]
                if query in instr_qa:
                    repeat_idx = r_idx
                    break
        
        reward = 0
        # for invalid query, give a negative reward and use the instruction only
        if query == "<<invalid>>" or repeat_idx == -1:
            reward += REWARD_INVALID
            repeat_idx = 0
        # for asking question, we add a small negative reward
        elif not query == "none":
            reward += REWARD_QUESTION
            current_num_q += 1

        trial_uid = "pad:" + str(repeat_idx) + ":" + str(subgoal_idx)
        dataset_idx_qa =  repeat_idx + dataset_idx

        # perform ET rollout
        with torch.no_grad():
            log_entry = evaluate_subgoals(env, performer, dataset, extractor, trial_uid, dataset_idx_qa, args, obj_predictor)

        if log_entry['success']:
            reward += REWARD_SUC
            done = 1.0
        else:
            done = 0.0
            reward += REWARD_TIME

        succ.append(done)
        all_rewards.append(reward)
        all_pws.append(log_entry['success_spl'])
        rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
        masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

        returns = compute_returns(next_value, rewards, masks)
        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.tensor([values], dtype=torch.float, device=device, requires_grad=True)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        critic_optimizer.step()

        actor_losses.append(actor_loss.detach().cpu())
        critic_losses.append(critic_loss.detach().cpu())
        num_q.append(current_num_q)

        if it % print_every == 0:
            logging.info("task, trial, subgoals: %s" % sg_pairs[-1])
            logging.info("instruction: %s" % all_instr[-1])
            logging.info("questions: %s" % all_query[-1])
            logging.info("number of questions: %s" % num_q[-1])
            logging.info('%s (%d %d%%) actor loss %.4f, critic loss %.4f, reward %.4f, SR %.4f, PWSR %.4f' % \
                (timeSince(start, (it+1) / n_iters), (it+1), (it+1) / n_iters * 100, \
                np.mean(actor_losses), np.mean(critic_losses), np.mean(all_rewards), np.mean(succ), np.mean(all_pws)))
        
        if it % save_every == 0:
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(), 
                "critic": critic.state_dict()
                }, "./logs/questioner_rl/questioner_begin_"+split_id+".pt")
            with open("./logs/questioner_rl/questioner_begin_"+split_id+".pkl", "wb") as pkl_f:
                pickle.dump([all_rewards, succ, all_query, all_instr, sg_pairs, num_q, all_pws], pkl_f)
        
    env.stop()

def trainModel():
    np.random.seed(0)
    data_split = "seen"
    train_id = 1
    logging.basicConfig(filename='./logs/rl_begin_'+ data_split + str(train_id) + '.log', level=logging.INFO)

    # load pretrained questioner
    test_csv_fn = "./data/hdc_input_augmented.csv"
    lang = prepareDataTest(test_csv_fn)
    enc_hidden_size = HIDDEN_SIZE//2 if BIDIRECTIONAL else HIDDEN_SIZE
    encoder = EncoderLSTM(lang.n_words, WORD_EMBEDDING_SIZE, enc_hidden_size, \
        DROPOUT_RATIO, bidirectional=BIDIRECTIONAL).to(device)
    decoder = AttnDecoderLSTM(lang.n_words, lang.n_words, \
        ACTION_EMBEDDING_SIZE, HIDDEN_SIZE, DROPOUT_RATIO).to(device)
    pretrain_questioner_fn = "./logs/pretrained_questioner.pt"
    checkpt = torch.load(pretrain_questioner_fn)
    encoder.load_state_dict(checkpt["encoder"])
    decoder.load_state_dict(checkpt["decoder"])
    critic = Critic().to(device)

    # load dataset and pretrained performer
    data_name = "lmdb_augmented_human_subgoal"
    model_path = "./logs/et_augmented_human_subgoal/latest.pth"
    model_args = model_util.load_model_args(model_path)
    model_args.debug = False
    model_args.no_model_unroll = False
    model_args.no_teacher_force = False
    model_args.smooth_nav = False
    model_args.max_steps = 1000
    model_args.max_fails = 10
    dataset = AlfredDataset(data_name, "valid_"+data_split, model_args, "lang")
    performer, extractor = load_agent(model_path, dataset.dataset_info, device)
    dataset.vocab_translate = performer.vocab_out

    trainIters(model_args, lang, dataset, encoder, decoder, critic, performer, extractor, \
        split_id=data_split + str(train_id), 1000, print_every=10, save_every=100)

def evalIters(args, lang, dataset, encoder, decoder, critic, performer, extractor, split_id, \
    print_every=1, save_every=100):
    start = time.time()
    env = ThorEnv(x_display=0)
    obj_predictor = FeatureExtractor(archi='maskrcnn', device=device,
        checkpoint="./logs/pretrained/maskrcnn_model.pth", load_heads=True)

    num_subgoals = len(dataset.jsons_and_keys)
    all_rewards = []
    succ = []
    all_query = []
    all_pws = []
    all_instr = []
    sg_pairs = []
    num_q = []
    it = 0
    
    # record the dataset index for the original instruction without qas
    data_instruct_list = []
    for i in range(len(dataset.jsons_and_keys)):
        traj_data, traj_key = dataset.jsons_and_keys[i]
        if traj_data["repeat_idx"] == 0:
            data_instruct_list.append(i)
    print("dataset length", len(data_instruct_list))
    # rough estimation of total number of subgoals
    n_iters = len(data_instruct_list) * 4

    # first sample a subgoal and get the instruction and image feature
    for dataset_idx in data_instruct_list:
        task_json = dataset.jsons_and_keys[dataset_idx]
        turk_annos = task_json[0]["turk_annotations"]["anns"]
        subgoal_idxs = [sg['high_idx'] for sg in task_json[0]['plan']['high_pddl']]
        # ignore the last subgoal which is often the padding one
        subgoal_idxs = subgoal_idxs[:-1]
        for subgoal_idx in subgoal_idxs:
            current_num_q = 0
            task, trial = task_json[0]['task'].split("/")
            pair = (None, None, task, trial, subgoal_idx)
            sg_pairs.append([task, trial, subgoal_idx])
            f_t = extractFeature(pair).to(device)
            orig_instr = normalizeString(turk_annos[0]["high_descs"][subgoal_idx]).lower().replace(",", "").replace(".", "")
            all_instr.append(orig_instr)
            input_tensor = torch.unsqueeze(torch.squeeze(tensorFromSentence(lang, orig_instr)), 0).to(device)
            input_length = input_tensor.size(1)

            # infer question based on the instruction
            encoder.init_state(input_tensor)
            seq_lengths = torch.from_numpy(np.array([input_length]))
            ctx, h_t, c_t = encoder(input_tensor, seq_lengths)
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoded_words = []
            for di in range(MAX_LENGTH):
                h_t, c_t, alpha, logit = decoder(decoder_input, f_t, h_t, c_t, ctx)
                # record the value V(s)
                if di == 0:
                    value = critic(h_t)
                dist = Categorical(F.softmax(logit, dim=-1))
                selected_word = dist.sample()
                decoded_words.append(lang.index2word[selected_word.item()])
                decoder_input = selected_word.detach().to(device)

            # set up the trial_idx to the one with the correct qas
            repeat_idx = -1
            if decoded_words[0] == "appearance":
                query = "<<app>> " + decoded_words[1]
            elif decoded_words[0] == "location":
                query = "<<loc>> " + decoded_words[1]
            elif decoded_words[0] == "direction":
                query = "<<dir>> "
            elif decoded_words[0] == "none":
                query = "none"
                repeat_idx = 0
            else:
                query = "<<invalid>>"
                repeat_idx = 0
            
            all_query.append(query)
            if repeat_idx == -1:
                for r_idx in range(1, 6):
                    if r_idx >= len(turk_annos) or subgoal_idx >= len(turk_annos[r_idx]["high_descs"]):
                        break
                    instr_qa = turk_annos[r_idx]["high_descs"][subgoal_idx]
                    if query in instr_qa:
                        repeat_idx = r_idx
                        break
            
            reward = 0
            # for invalid query, give a negative reward and use the instruction only
            if query == "<<invalid>>" or repeat_idx == -1:
                reward += REWARD_INVALID
                repeat_idx = 0
            # for asking question, we add a small negative reward
            elif not query == "none":
                reward += REWARD_QUESTION
                current_num_q += 1

            trial_uid = "pad:" + str(repeat_idx) + ":" + str(subgoal_idx)
            dataset_idx_qa =  repeat_idx + dataset_idx

            # perform ET rollout
            with torch.no_grad():
                log_entry = evaluate_subgoals(env, performer, dataset, extractor, trial_uid, dataset_idx_qa, args, obj_predictor)

            if log_entry['success']:
                reward += REWARD_SUC
                done = 1.0
            else:
                done = 0.0
                reward += REWARD_TIME

            succ.append(done)
            all_rewards.append(reward)
            all_pws.append(log_entry['success_spl'])
            num_q.append(current_num_q)

            if it % print_every == 0:
                logging.info("task, trial, subgoals: %s" % sg_pairs[-1])
                logging.info("instruction: %s" % all_instr[-1])
                logging.info("questions: %s" % all_query[-1])
                logging.info("number of questions: %s" % num_q[-1])
                logging.info('%s (%d %d%%) reward %.4f, SR %.4f, PWSR %.4f' % \
                    (timeSince(start, (it+1) / n_iters), (it+1), (it+1) / n_iters * 100, \
                    np.mean(all_rewards), np.mean(succ), np.mean(all_pws)))
            
            if it % save_every == 0:
                with open("./logs/questioner_rl/eval_questioner_begin_"+split_id+".pkl", "wb") as pkl_f:
                    pickle.dump([all_rewards, succ, all_query, all_instr, sg_pairs, num_q, all_pws], pkl_f)
            
            it += 1
        
    env.stop()

def evalModel():
    np.random.seed(0)
    data_split = "unseen"
    train_id = 1
    logging.basicConfig(filename='./logs/rl_begin_eval_'+ data_split + str(train_id) + '.log', level=logging.INFO)

    # load pretrained questioner
    test_csv_fn = "./data/hdc_input_augmented.csv"
    lang = prepareDataTest(test_csv_fn)
    enc_hidden_size = HIDDEN_SIZE//2 if BIDIRECTIONAL else HIDDEN_SIZE
    encoder = EncoderLSTM(lang.n_words, WORD_EMBEDDING_SIZE, enc_hidden_size, \
        DROPOUT_RATIO, bidirectional=BIDIRECTIONAL).to(device)
    decoder = AttnDecoderLSTM(lang.n_words, lang.n_words, \
        ACTION_EMBEDDING_SIZE, HIDDEN_SIZE, DROPOUT_RATIO).to(device)
    critic = Critic().to(device)
    finetune_questioner_fn = "./logs/pretrained_questioner.pt"

    checkpt = torch.load(finetune_questioner_fn)
    encoder.load_state_dict(checkpt["encoder"])
    decoder.load_state_dict(checkpt["decoder"])
    critic.load_state_dict(checkpt["critic"])

    # load dataset and pretrained performer
    data_name = "lmdb_augmented_human_subgoal"
    model_path = "./logs/lmdb_augmented_human_subgoal/latest.pth"
    model_args = model_util.load_model_args(model_path)
    model_args.debug = False
    model_args.no_model_unroll = False
    model_args.no_teacher_force = False
    model_args.smooth_nav = False
    model_args.max_steps = 1000
    model_args.max_fails = 10
    dataset = AlfredDataset(data_name, "valid_"+data_split, model_args, "lang")
    performer, extractor = load_agent(model_path, dataset.dataset_info, device)
    dataset.vocab_translate = performer.vocab_out

    evalIters(model_args, lang, dataset, encoder, decoder, critic, performer, extractor, \
        split_id=data_split + str(train_id), print_every=1, save_every=10)

if __name__ == '__main__':
    trainModel()
    # evalModel()
