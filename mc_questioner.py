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
# max output length
MAX_LENGTH = 2
MC_THRESHOLD = 0.5

SOS_token = 0
EOS_token = 1

def evalMC(args, dataset, performer, extractor, split_id, print_every=10, save_every=10):
    start = time.time()
    env = ThorEnv(x_display=0)
    obj_predictor = FeatureExtractor(archi='maskrcnn', device=device,
        checkpoint="./logs/pretrained/maskrcnn_model.pth", load_heads=True)

    succ = []
    all_query = []
    all_instr = []
    sg_pairs = []
    all_pws = []

    # record the dataset index for the original instruction without qas
    data_instruct_list = []
    for i in range(len(dataset.jsons_and_keys)):
        traj_data, traj_key = dataset.jsons_and_keys[i]
        if traj_data["repeat_idx"] == 0:
            data_instruct_list.append(i)
    print("dataset length", len(data_instruct_list))
    
    n_iters = len(data_instruct_list)
    for it in range(n_iters):
        dataset_idx = data_instruct_list[it]
        task_json = dataset.jsons_and_keys[dataset_idx]
        turk_annos = task_json[0]["turk_annotations"]["anns"]
        subgoal_idxs = [sg['high_idx'] for sg in task_json[0]['plan']['high_pddl']]
        # ignore the last subgoal which is often the padding one
        subgoal_idxs = subgoal_idxs[:-1]
        subgoal_idx = np.random.choice(subgoal_idxs)
        for subgoal_idx in subgoal_idxs:
            task, trial = task_json[0]['task'].split("/")
            sg_pairs.append([task, trial, subgoal_idx])

            repeat_idx = 0
            trial_uid = "pad:" + str(repeat_idx) + ":" + str(subgoal_idx)
            dataset_idx_qa = repeat_idx + dataset_idx

            # perform the first ET rollout to get model confusion
            with torch.no_grad():
                log_entry, all_mc = evaluate_subgoals_mc(env, performer, \
                    dataset, extractor, trial_uid, dataset_idx_qa, args, obj_predictor)
            
            # if the model is confused, ask a random question
            if len(all_mc) > 0 and np.min(all_mc) < MC_THRESHOLD:
                repeat_idx = np.random.randint(1, 4)
                trial_uid = "pad:" + str(repeat_idx) + ":" + str(subgoal_idx)
                dataset_idx_qa = repeat_idx + dataset_idx
                with torch.no_grad():
                    log_entry, _ = evaluate_subgoals_mc(env, performer, \
                        dataset, extractor, trial_uid, dataset_idx_qa, args, obj_predictor)
                all_query.append(repeat_idx)
            else:
                all_query.append("none")


            if log_entry['success']:
                done = 1.0
            else:
                done = 0.0

            succ.append(done)
            all_pws.append(log_entry['success_spl'])
            if it % print_every == 0:
                logging.info("task, trial, subgoals: %s" % sg_pairs[-1])
                logging.info("instruction: %s" % all_instr[-1])
                logging.info("questions: %s" % all_query[-1])
                logging.info("number of questions: %s" % num_q[-1])
                logging.info('%s (%d %d%%) SR %.4f PWSR %.4f' % \
                    (timeSince(start, (it+1) / n_iters), (it+1), (it+1) / n_iters * 100, np.mean(succ), np.mean(all_pws)))
            
            if it % save_every == 0:
                with open("./logs/questioner_rl/question_mc_"+split_id+".pkl", "wb") as pkl_f:
                    pickle.dump([succ, all_query, sg_pairs, all_pws], pkl_f)

def main():
    np.random.seed(0)
    data_split = "seen"
    train_id = 1
    logging.basicConfig(filename='./logs/mc_eval'+ data_split + str(train_id) + '.log', level=logging.INFO)

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
    evalMC(model_args, dataset, performer, extractor, split_id=data_split + str(train_id), \
        print_every=1, save_every=10)

if __name__ == '__main__':
	main()