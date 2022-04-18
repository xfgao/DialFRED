#!/usr/bin/env python
# coding: utf-8

import stanza
import os
import json
import re
import pickle
import copy
import revtok
import numpy as np
import shutil
from tqdm import tqdm
from utils import *
from vocab import Vocab
import torch


verb_table = {
    "PickupObject": ["pick up", 'grab', 'take'],
    "PutObject":['place', 'put', 'leave'],
    "OpenObject": ['open', 'open up'],
    "CloseObject": ['close'],
    "CoolObject":['cool', 'chill'],
    "HeatObject": ['heat', 'cook', 'heat up'],
    "ToggleObjectOn": ["turn on", 'switch on', 'power on'],
    "ToggleObjectOff": ["turn off", 'switch off', 'power off'],
    "ToggleObject": ["toggle"],
    "SliceObject": ["slice", 'cut'],
    "CleanObject": ["clean", "wash"],
    'GotoLocation' : ["go to", 'move to']
}

def backupJson(gen_data_dir, sp):
	# backup the original json files
    sp_dir = gen_data_dir + sp + "/"
    task_fds = os.listdir(sp_dir)
    for task in task_fds:
        task_dir = sp_dir + task + "/"
        trial_fds = os.listdir(task_dir)
        for trial in trial_fds:
            new_fn = task_dir + trial + "/traj_data_backup.json"
            old_fn = task_dir + trial + "/traj_data.json"
            shutil.copy(old_fn, new_fn)

def recoverJson(gen_data_dir, sp):
	# recover the json files from the backup
    sp_dir = gen_data_dir + sp + "/"
    task_fds = os.listdir(sp_dir)
    for task in task_fds:
        task_dir = sp_dir + task + "/"
        trial_fds = os.listdir(task_dir)
        for trial in trial_fds:
            # recover from the original json file backup
            old_fn = task_dir + trial + "/traj_data_backup.json"
            new_fn = task_dir + trial + "/traj_data.json"
            shutil.copy(old_fn, new_fn)

def modifyData(json_dir, split_id):
    # modify the traj json file based on the saved augmented data pickle
    with open('./data/augmented_data_' + split_id + '.pickle', "rb") as f:
        data_augmented, new_high_idx, tasks, trials, person_id, new_planner_actions = pickle.load(f)
    
    sp_dir = json_dir + split_id + "/"
    num_data = len(data_augmented)
    for i in tqdm(range(num_data)):
        task = tasks[i]
        trial = trials[i]
        instr_id = person_id[i]
        aug_data = data_augmented[i]
        h_idx = new_high_idx[i].tolist()
        plan_act = new_planner_actions[i]
        fn = sp_dir + task + "/" + trial + "/traj_data_backup.json"
        with open(fn, "r") as f:
            traj_data_old = json.load(f)
        
        traj_data = copy.copy(traj_data_old)
        low2high = {}

        # modify images to high index mapping
        for idx, im in enumerate(h_idx):
            if idx >= len(traj_data["images"]):
                break
            traj_data["images"][idx]["high_idx"] = h_idx[idx]
            low_idx = traj_data["images"][idx]["low_idx"]
            low2high[low_idx] = h_idx[idx]
        
        # modify high pddl
        traj_data["plan"]["high_pddl"] = []
        for sg_idx in range(len(aug_data)):
            sg_pddl = {}
            new_discrete = {"parameter": [{"action": aug_data[sg_idx][1][low_idx], "args": [aug_data[sg_idx][2][low_idx],]} for low_idx in range(len(aug_data[sg_idx][1]))]}
            new_discrete["action"] = [aug_data[sg_idx][1][low_idx] for low_idx in range(len(aug_data[sg_idx][1]))]
            sg_pddl["discrete_action"] = new_discrete
            sg_pddl["planner_action"] = {"action":[plan_act[sg_idx][low_idx]["action"] for low_idx in range(len(plan_act[sg_idx]))], "parameter":plan_act[sg_idx]}
            sg_pddl["high_idx"] = sg_idx
            traj_data["plan"]["high_pddl"].append(sg_pddl)

        # modify low level action to high index mapping
        for low_idx in range(len(traj_data["plan"]["low_actions"])):
            traj_data["plan"]["low_actions"][low_idx]["high_idx"] = low2high[low_idx]
        
        # only keep 1 subgoal annotation for now
        anns_first_save = copy.copy(traj_data["turk_annotations"]["anns"][0])
        traj_data["turk_annotations"]["anns"] = []
        traj_data["turk_annotations"]["anns"].append(anns_first_save)
        traj_data["turk_annotations"]["anns"][0]["high_descs"] = [aug_data[sg_idx][0] for sg_idx in range(len(aug_data))]

        fn_new = sp_dir + task + "/" + trial + "/traj_data_augmented_backup.json"
        with open(fn_new, "w") as f:
            json.dump(traj_data, f, sort_keys=True, indent=4)

if __name__ == "__main__":
    np.random.seed(1)
    json_dir = "./data/generated_2.1.0/"
    splits = ["valid_seen", "valid_unseen", "train"]

    for split_id in splits:
        print(split_id)
        backupJson(json_dir, split_id)
        modifyData(json_dir, split_id)    