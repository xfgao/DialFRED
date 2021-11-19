import os
from itertools import count
import time
import math
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
from hdc.utils import *

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

def def_value():
	return None

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

# extract resnet feature from current observation
def extractFeatureOnline(env, extractor):
    event = env.last_event
    feat = get_observation(event, extractor)
    avg_pool = nn.AvgPool2d(7)
    feat = torch.unsqueeze(torch.squeeze(avg_pool(feat)), 0)
    return feat


def trainIters(args, lang, dataset, encoder, decoder, critic, performer, extractor, all_ans, n_iters, split_id, max_steps, print_every=1, save_every=100):
    start = time.time()
    env = ThorEnv(x_display=0)
    obj_predictor = FeatureExtractor(archi='maskrcnn', device=device,
        checkpoint="./logs/pretrained/maskrcnn_model.pth", load_heads=True)

    loc_ans, app_ans, dir_ans = all_ans
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
    object_found = []
    sg_pairs = []
    num_q = []
    all_pws = []
    all_decoded_words = []
    # assume we only have one instruction
    instr_id = 0

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
        current_query = []
        current_object_found = []
        all_log = []
        current_num_q = 0
        entropy = 0
        t_agent = 0
        num_fails = 0
        episode_end = False
        current_d_words = []

        # first sample a subgoal and get the instruction and image feature
        dataset_idx = np.random.choice(data_instruct_list)
        task_json = dataset.jsons_and_keys[dataset_idx]
        turk_annos = task_json[0]["turk_annotations"]["anns"]
        subgoal_idxs = [sg['high_idx'] for sg in task_json[0]['plan']['high_pddl']]
        # ignore the last subgoal which is often the padding one
        subgoal_idxs = subgoal_idxs[:-1]
        subgoal_idx = np.random.choice(subgoal_idxs)
        
        # set up the performer for expect actions first
        trial_uid = "pad:" + str(0) + ":" + str(subgoal_idx)
        dataset_idx_qa = 0 + dataset_idx
        init_states = evaluate_subgoals_start_qa(
            env, performer, dataset, extractor, trial_uid, dataset_idx_qa, args, obj_predictor)
        _, _, _, init_failed, _ = init_states

        task, trial = task_json[0]['task'].split("/")
        pair = (None, None, task, trial, subgoal_idx)
        sg_pairs.append([task, trial, subgoal_idx])
        orig_instr = normalizeString(turk_annos[0]["high_descs"][subgoal_idx]).lower().replace(",", "").replace(".", "")
        qa = ""
        reward = 0
        all_instr.append(orig_instr)
        interm_states = None
        pws = 0.0
        t_agent_old = 0
        dialog = ""
        
        while True:
            # use online feature extractor instead
            f_t = extractFeatureOnline(env, extractor)
            input_tensor = torch.unsqueeze(torch.squeeze(tensorFromSentence(lang, orig_instr)), 0).to(device)
            input_length = input_tensor.size(1)

            # infer question based on the instruction
            encoder.init_state(input_tensor)
            seq_lengths = torch.from_numpy(np.array([input_length]))
            ctx, h_t, c_t = encoder(input_tensor, seq_lengths)
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoded_words = []

            # decode the question
            for di in range(MAX_LENGTH):
                h_t, c_t, alpha, logit = decoder(decoder_input, f_t, h_t, c_t, ctx)
                # record the value V(s)
                value = critic(h_t)
                dist = Categorical(F.softmax(logit, dim=-1))
                selected_word = dist.sample()
                dword = lang.index2word[selected_word.item()]
                decoded_words.append(dword)
                decoder_input = selected_word.detach().to(device)
                log_prob += dist.log_prob(selected_word).unsqueeze(0)/MAX_LENGTH
                entropy += dist.entropy().mean()/MAX_LENGTH
                if dword == "EOS":
                    break

            # set up query and answer
            repeat_idx = -1
            ans = ""
            current_d_words.append(decoded_words)
            # for appearance answer, we can directly use the saved ones
            if decoded_words[0] == "appearance":
                query = "<<app>> " + decoded_words[1]
                if task in app_ans and trial in app_ans[task] and subgoal_idx in app_ans[task][trial] and 0 in app_ans[task][trial][subgoal_idx]:
                    ans_sg = app_ans[task][trial][subgoal_idx][0]
                    if decoded_words[1] in ans_sg and ans_sg[decoded_words[1]]["ans"] is not None:
                        ans += ans_sg[decoded_words[1]]["ans"]
                    else:
                        ans += "invalid"
                else:
                    logging.info("invalid answer for %s, %s, %s" % (task, trial, subgoal_idx))
                    ans += "invalid"
            # for location answer, we need to construct a new one using current metadata
            elif decoded_words[0] == "location":
                query = "<<loc>> " + decoded_words[1]
                if task in loc_ans and trial in loc_ans[task] and subgoal_idx in loc_ans[task][trial] and 0 in loc_ans[task][trial][subgoal_idx]:
                    ans_sg = loc_ans[task][trial][subgoal_idx][0]
                    if decoded_words[1] in ans_sg:
                        obj_id = ans_sg[decoded_words[1]]["obj_id"]
                        event = env.last_event
                        metadata = event.metadata
                        odata = get_obj_data(metadata, obj_id)
                        if odata is None:
                            ans += "invalid"
                        else:
                            oname = decoded_words[1]
                            recs = odata["parentReceptacles"]
                            rel_ang = get_obj_direction(metadata, odata)
                            ans += objLocAns(oname, rel_ang, recs)
                    else:
                        ans += "invalid"
                else:
                    ans += "invalid"

            elif decoded_words[0] == "direction":
                query = "<<dir>> "
                if task in dir_ans and trial in dir_ans[task] and subgoal_idx in dir_ans[task][trial]:
                    target_pos = dir_ans[task][trial][subgoal_idx]["target_pos"]
                    event = env.last_event
                    cur_metadata = event.metadata
                    targ_metadata = {'agent':{'position':target_pos}}
                    rel_ang, rel_pos = get_agent_direction(cur_metadata, targ_metadata)
                    ans += dirAns(rel_ang, rel_pos)
                else:
                    ans += "invalid"
            elif decoded_words[0] == "none":
                query = "none"
            else:
                query = "<<invalid>>"
            
            # for invalid query, give a negative reward and use the instruction only
            if "invalid" in query or "invalid" in ans:
                reward += REWARD_INVALID
                current_object_found.append(False)
                qa = ""
            # for asking each valid question, we add a small negative reward
            elif not query == "none":
                reward += REWARD_QUESTION
                current_object_found.append(True)
                current_num_q += 1
                qa = query + " " + ans
            # for not asking question, there is no penalty
            else:
                current_object_found.append(True)
                qa = ""

            current_query.append(query + " " + ans)
            # a penalty for each time step
            qa = qa.lower().replace(",", "").replace(".", "")
            dialog += " " + qa 

            # performer rollout for some steps
            with torch.no_grad():
                log_entry, interm_states = evaluate_subgoals_middle_qa(env, performer, dataset, extractor, \
                    trial_uid, dataset_idx_qa, args, obj_predictor, init_states, interm_states, qa, num_rollout=5, use_mc=True, mc_thres=0.7)

            if log_entry['success']:
                reward += REWARD_SUC
                done = 1.0
                pws = log_entry['success_spl']
            else:
                done = 0.0

            t_agent, _, num_fails, _, mc_lists, episode_end, _ = interm_states
            reward += REWARD_TIME * (t_agent - t_agent_old)
            t_agent_old = t_agent
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))
            if done or t_agent > args.max_steps or num_fails > args.max_fails or episode_end or init_failed or len(current_query) > 100:
                break
        
        succ.append(done)
        all_rewards.append(reward)
        all_pws.append(pws)
        all_log.append(log_entry)
        # record the next value V(s')
        next_value = critic(h_t)
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
        object_found.append(current_object_found)
        all_query.append(current_query)
        num_q.append(current_num_q)
        all_decoded_words.append(current_d_words)

        if it % print_every == 0:
            logging.info("task, trial, subgoals: %s" % sg_pairs[-1])
            logging.info("instruction: %s" % all_instr[-1])
            logging.info("questions: %s" % all_query[-1])
            logging.info("number of questions: %s" % num_q[-1])
            logging.info('%s (%d %d%%) actor loss %.4f, critic loss %.4f, reward %.4f, SR %.4f, pws %.4f' % \
                (timeSince(start, (it+1) / n_iters), (it+1), (it+1) / n_iters * 100, \
                np.mean(actor_losses), np.mean(critic_losses), np.mean(all_rewards), np.mean(succ), np.mean(all_pws)))
        
        if it % save_every == 0:
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(), 
                "critic": critic.state_dict()
                }, "./logs/questioner_rl/questioner_anytime_"+split_id+".pt")
            with open("./logs/questioner_rl/questioner_anytime_"+split_id+".pkl", "wb") as pkl_f:
                pickle.dump([all_rewards, succ, all_query, all_instr, sg_pairs, num_q, all_pws], pkl_f)
            
        
    env.stop()

def trainModel():
    data_split = "seen"
    train_id = 1
    logging.basicConfig(filename='./logs/rl_anytime_'+ data_split + str(train_id) + '.log', level=logging.INFO)
    np.random.seed(0)

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
    data_name = "lmdb_augmented_human"
    model_path = "./logs/et_augmented_human/latest.pth"
    model_args = model_util.load_model_args(model_path)
    model_args.debug = False
    model_args.no_model_unroll = False
    model_args.no_teacher_force = False
    model_args.smooth_nav = False
    model_args.max_steps = 1000
    model_args.max_fails = 10
    dataset = AlfredDataset(data_name, "valid_"+data_split, model_args, "lang")
    dataset.vocab_in.name = "lmdb_augmented_human"
    performer, extractor = load_agent(model_path, dataset.dataset_info, device)
    dataset.vocab_translate = performer.vocab_out

    # load answers
    loc_ans_fn = "./data/answers/loc_augmented.pkl"
    app_ans_fn = "./data/answers/appear_augmented.pkl"
    dir_ans_fn = "./data/answers/direction_augmented.pkl"
    with open(loc_ans_fn, "rb") as f:
        loc_ans = pickle.load(f)
    with open(app_ans_fn, "rb") as f:
        app_ans = pickle.load(f)
    with open(dir_ans_fn, "rb") as f:
        dir_ans = pickle.load(f)
    all_ans = [loc_ans, app_ans, dir_ans]
    trainIters(model_args, lang, dataset, encoder, decoder, critic, performer, extractor, all_ans, 1000, split_id=data_split+str(train_id), max_steps=1000, print_every=1, save_every=10)

def evalIters(args, lang, dataset, encoder, decoder, critic, performer, extractor, all_ans, split_id, max_steps, print_every=1, save_every=100):
    start = time.time()
    env = ThorEnv(x_display=0)
    obj_predictor = FeatureExtractor(archi='maskrcnn', device=device,
        checkpoint="./logs/pretrained/maskrcnn_model.pth", load_heads=True)

    loc_ans, app_ans, dir_ans = all_ans
    num_subgoals = len(dataset.jsons_and_keys)
    all_rewards = []
    succ = []
    all_query = []
    all_instr = []
    object_found = []
    sg_pairs = []
    num_q = []
    all_pws = []
    # assume we only have one instruction
    instr_id = 0
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
            current_query = []
            current_object_found = []
            all_log = []
            current_num_q = 0
            t_agent = 0
            num_fails = 0
            episode_end = False        
            # set up the performer for expect actions first
            trial_uid = "pad:" + str(0) + ":" + str(subgoal_idx)
            dataset_idx_qa = 0 + dataset_idx
            init_states = evaluate_subgoals_start_qa(
                env, performer, dataset, extractor, trial_uid, dataset_idx_qa, args, obj_predictor)
            _, _, _, init_failed, _ = init_states

            task, trial = task_json[0]['task'].split("/")
            pair = (None, None, task, trial, subgoal_idx)
            sg_pairs.append([task, trial, subgoal_idx])
            orig_instr = normalizeString(turk_annos[0]["high_descs"][subgoal_idx]).lower().replace(",", "").replace(".", "")
            qa = ""
            reward = 0
            all_instr.append(orig_instr)
            interm_states = None
            pws = 0.0
            t_agent_old = 0

            while True:
                # use online feature extractor instead
                f_t = extractFeatureOnline(env, extractor)
                dialog = orig_instr
                input_tensor = torch.unsqueeze(torch.squeeze(tensorFromSentence(lang, dialog)), 0).to(device)
                input_length = input_tensor.size(1)

                # infer question based on the instruction
                encoder.init_state(input_tensor)
                seq_lengths = torch.from_numpy(np.array([input_length]))
                ctx, h_t, c_t = encoder(input_tensor, seq_lengths)
                decoder_input = torch.tensor([[SOS_token]], device=device)
                decoded_words = []

                # decode the question
                for di in range(MAX_LENGTH):
                    h_t, c_t, alpha, logit = decoder(decoder_input, f_t, h_t, c_t, ctx)
                    # record the value V(s)
                    value = critic(h_t)
                    dist = Categorical(F.softmax(logit, dim=-1))
                    selected_word = dist.sample()
                    decoded_words.append(lang.index2word[selected_word.item()])
                    decoder_input = selected_word.detach().to(device)

                # set up query and answer
                repeat_idx = -1
                ans = ""
                # for appearance answer, we can directly use the saved ones
                if decoded_words[0] == "appearance":
                    query = "<<app>> " + decoded_words[1]
                    if task in app_ans and trial in app_ans[task] and subgoal_idx in app_ans[task][trial] and 0 in app_ans[task][trial][subgoal_idx]:
                        ans_sg = app_ans[task][trial][subgoal_idx][0]
                        if decoded_words[1] in ans_sg and ans_sg[decoded_words[1]]["ans"] is not None:
                            ans += ans_sg[decoded_words[1]]["ans"]
                        else:
                            ans += "invalid"
                    else:
                        logging.info("invalid answer for %s, %s, %s" % (task, trial, subgoal_idx))
                        ans += "invalid"
                # for location answer, we need to construct a new one using current metadata
                elif decoded_words[0] == "location":
                    query = "<<loc>> " + decoded_words[1]
                    if task in loc_ans and trial in loc_ans[task] and subgoal_idx in loc_ans[task][trial] and 0 in loc_ans[task][trial][subgoal_idx]:
                        ans_sg = loc_ans[task][trial][subgoal_idx][0]
                        if decoded_words[1] in ans_sg:
                            obj_id = ans_sg[decoded_words[1]]["obj_id"]
                            event = env.last_event
                            metadata = event.metadata
                            odata = get_obj_data(metadata, obj_id)
                            if odata is None:
                                ans += "invalid"
                            else:
                                oname = decoded_words[1]
                                recs = odata["parentReceptacles"]
                                rel_ang = get_obj_direction(metadata, odata)
                                ans += objLocAns(oname, rel_ang, recs)
                        else:
                            ans += "invalid"
                    else:
                        ans += "invalid"

                elif decoded_words[0] == "direction":
                    query = "<<dir>> "
                    if task in dir_ans and trial in dir_ans[task] and subgoal_idx in dir_ans[task][trial]:
                        target_pos = dir_ans[task][trial][subgoal_idx]["target_pos"]
                        event = env.last_event
                        cur_metadata = event.metadata
                        targ_metadata = {'agent':{'position':target_pos}}
                        rel_ang, rel_pos = get_agent_direction(cur_metadata, targ_metadata)
                        ans += dirAns(rel_ang, rel_pos).lower()
                    else:
                        ans += "invalid"
                elif decoded_words[0] == "none" or decoded_words[0] == "EOS":
                    query = "none"
                else:
                    query = "<<invalid>>"
                
                # for invalid query, give a negative reward and use the instruction only
                if "invalid" in query or "invalid" in ans:
                    reward += REWARD_INVALID
                    current_object_found.append(False)
                    qa = ""
                # for asking each question, we add a small negative reward
                elif not query == "none":
                    reward += REWARD_QUESTION
                    current_object_found.append(True)
                    current_num_q += 1
                    qa = query + " " + ans
                # for not asking question, there is no penalty
                else:
                    current_object_found.append(True)
                    qa = ""

                current_query.append(query + " " + ans)
                qa = qa.lower().replace(",", "").replace(".", "")

                # performer rollout for some steps
                with torch.no_grad():
                    log_entry, interm_states = evaluate_subgoals_middle_qa(env, performer, dataset, extractor, \
                        trial_uid, dataset_idx_qa, args, obj_predictor, init_states, interm_states, qa, use_mc=True, num_rollout=5, mc_thres=0.7)
                
                if log_entry['success']:
                    reward += REWARD_SUC
                    done = 1.0
                    pws = log_entry['success_spl']
                else:
                    done = 0.0

                t_agent, _, num_fails, _, mc_lists, episode_end, _ = interm_states
                # a penalty for each time step
                reward += REWARD_TIME * (t_agent - t_agent_old)
                t_agent_old = t_agent
                if done or t_agent > args.max_steps or num_fails > args.max_fails or episode_end or init_failed or len(current_query) > 100:
                    break
        
            succ.append(done)
            all_rewards.append(reward)
            all_pws.append(pws)
            all_log.append(log_entry)
            # record the next value V(s')
            object_found.append(current_object_found)
            all_query.append(current_query)
            num_q.append(current_num_q)

            if it % print_every == 0:
                logging.info("task, trial, subgoals: %s" % sg_pairs[-1])
                logging.info("instruction: %s" % all_instr[-1])
                logging.info("questions: %s" % all_query[-1])
                logging.info("number of questions: %s" % num_q[-1])
                logging.info('%s (%d %d%%) reward %.4f, SR %.4f, pws %.4f' % \
                    (timeSince(start, (it+1) / n_iters), (it+1), (it+1) / n_iters * 100, \
                    np.mean(all_rewards), np.mean(succ), np.mean(all_pws)))
            
            if it % save_every == 0:
                with open("./logs/questioner_rl/eval_questioner_anytime_"+split_id+".pkl", "wb") as pkl_f:
                    pickle.dump([all_rewards, succ, all_query, all_instr, sg_pairs, num_q, all_pws], pkl_f)
            
            it += 1
        
    env.stop()

def evalModel():
    np.random.seed(0)
    data_split = "unseen"
    train_id = 1
    logging.basicConfig(filename='./logs/rl_anytime_eval_'+ data_split + str(train_id) + '.log', level=logging.INFO)

    # load pretrained questioner
    test_csv_fn = "./data/hdc_input_augmented.csv"
    lang = prepareDataTest(test_csv_fn)
    enc_hidden_size = HIDDEN_SIZE//2 if BIDIRECTIONAL else HIDDEN_SIZE
    encoder = EncoderLSTM(lang.n_words, WORD_EMBEDDING_SIZE, enc_hidden_size, \
        DROPOUT_RATIO, bidirectional=BIDIRECTIONAL).to(device)
    decoder = AttnDecoderLSTM(lang.n_words, lang.n_words, \
        ACTION_EMBEDDING_SIZE, HIDDEN_SIZE, DROPOUT_RATIO).to(device)
    critic = Critic().to(device)
    finetune_questioner_fn = "./logs/questioner_anytime_seen" + str(train_id)+ ".pt"
    checkpt = torch.load(finetune_questioner_fn)
    encoder.load_state_dict(checkpt["encoder"])
    decoder.load_state_dict(checkpt["decoder"])
    critic.load_state_dict(checkpt["critic"])

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
    dataset.vocab_in.name = "lmdb_augmented_human_subgoal"

    # load answers
    loc_ans_fn = "./data/answers/loc_augmented.pkl"
    app_ans_fn = "./data/answers/appear_augmented.pkl"
    dir_ans_fn = "./data/answers/direction_augmented.pkl"
    with open(loc_ans_fn, "rb") as f:
        loc_ans = pickle.load(f)
    with open(app_ans_fn, "rb") as f:
        app_ans = pickle.load(f)
    with open(dir_ans_fn, "rb") as f:
        dir_ans = pickle.load(f)
    all_ans = [loc_ans, app_ans, dir_ans]
    evalIters(model_args, lang, dataset, encoder, decoder, critic, performer, extractor, all_ans, split_id=data_split + str(train_id), max_steps=1000, print_every=1, save_every=10)

if __name__ == '__main__':
	trainModel() 
    # evalModel()