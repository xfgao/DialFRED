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

def idx2startend(idx_list):
    # convert a list of high/low action indexes to subgoal/action start and end lists
    len_idx = len(idx_list)
    idx_reverse = copy.copy(idx_list)
    idx_reverse.reverse()
    min_idx = min(idx_list)
    max_idx = max(idx_list)
    sg_st = []
    st_end = []

    for i in range(min_idx, max_idx+1):
        sg_st.append(idx_list.index(i))
        sg_end.append(len_idx - idx_reverse.index(i) - 1)
    
    return sg_st, st_end

# combine navigation and one low level actions to form a new subgoal              
def navigation_and_simple_one_step(s, a, high_index, low_index):
    short_phrases = copy.deepcopy(s)
    action_sequence = copy.deepcopy(a)
    total_length = len(action_sequence)
    h_new = np.array(copy.copy(high_index))
    l_new = np.array(copy.copy(low_index))
    
    data_augmented = []
    i = 0
    # for i in range(total_length-1):
    while i < total_length-1:
        if len(action_sequence[i+1][0]) > 1 and action_sequence[i][0][0] =='GotoLocation':
            verbs = action_sequence[i][0] + [action_sequence[i+1][0][0]]
            nouns = action_sequence[i][1] + [action_sequence[i+1][1][0]]

            same_high_idx = np.where(h_new == i+1)[0]
            first_low_idx = l_new[same_high_idx][0]
            first_low_idx = np.where(l_new == first_low_idx)[0]
            h_new[first_low_idx] = i
            
            next_verbs = [action_sequence[i+1][0][1:]]
            next_nouns = [action_sequence[i+1][1][1:]]
            for j in range(len(nouns)):
                tmp = nouns[j].split('|')
                if len(tmp) == 1:
                    noun = tmp[0].lower()
                elif len(tmp) == 4:
                    noun = tmp[0].lower()
                else:
                    noun = tmp[-1].lower().split('_')[0]
                nouns[j] = noun

            v = action_sequence[i+1][0][0]
            n = action_sequence[i+1][1][0]

            num_verbs = len(verb_table[v])
            verb = verb_table[v][np.random.randint(num_verbs)]
            tmp = n.split('|')
            
            if len(tmp) == 1:
                noun = tmp[0].lower()
            elif len(tmp) == 4:
                noun = tmp[0].lower()
            else:
                noun = tmp[-1].lower().split('_')[0]
            sen = verb.lower() + ' the ' + noun + '. '
            data_augmented.append((short_phrases[i].replace(".",'') + ', and '+ sen, 
                                verbs, nouns))
            
            # add the remaining low level actions in the splitted subgoals 
            next_sen = ""
            for next_v, next_n in zip(next_verbs[0], next_nouns[0]):
                num_verbs = len(verb_table[next_v])
                verb = verb_table[next_v][np.random.randint(num_verbs)]
                tmp = next_n.split('|')
                if len(tmp) == 1:
                    noun = tmp[0].lower()
                elif len(tmp) == 4:
                    noun = tmp[0].lower()
                else:
                    noun = tmp[-1].lower().split('_')[0]
                next_sen += verb.lower() + ' the ' + noun + ', '
            
            next_sen = next_sen[:-2] + "."
            data_augmented.append((next_sen.replace(".",'')+".", 
                                next_verbs, next_nouns))
            i += 1
        else:
            data_augmented.append((short_phrases[i].replace(".",'')+".", action_sequence[i][0], action_sequence[i][1]))
        
        i += 1

    if i == total_length-1:
        data_augmented.append((short_phrases[i].replace(".",'')+".", action_sequence[i][0], action_sequence[i][1]))

    return data_augmented, h_new

def combine(short_phrases, action_sequence, length, high_index, low_index):
    total_length = len(short_phrases)
    data_augmented = []
    h_new = np.array(copy.copy(high_index))
    sg_new_cnt = 0
    for i in range(0, total_length-length+1, length):
        string = ""
        verbs = []
        nouns = []
        string_ignore_goto = ""
        for j in range(i, i+length):
            orig_sg_idx = np.where(h_new == j)[0]
            h_new[orig_sg_idx] = sg_new_cnt
            if j == i:
                tmp =  short_phrases[j].capitalize().replace(".","")
                string += tmp
                if not tmp.lower().startswith("gotolocation"):
                    string_ignore_goto += tmp
            else:
                tmp = ', and ' + short_phrases[j].lower()
                string += tmp
                if not tmp.lower().startswith("gotolocation"):
                    string_ignore_goto += tmp
            verbs.extend(action_sequence[j][0])
            short_nouns = []
            for n in action_sequence[j][1]:
                tmp = n.split('|')
                if len(tmp) == 1:
                        noun = tmp[0].lower()
                elif len(tmp) == 4:
                        noun = tmp[0].lower()
                else:
                        noun = tmp[-1].lower().split('_')[0]
                short_nouns.append(noun)
            
            nouns.extend(short_nouns)
        
        data_augmented.append((string.replace(".",'')+".", verbs, nouns))
        sg_new_cnt += 1
    
    for k in range(j+1, total_length):
        data_augmented.append((short_phrases[k].replace(".",'')+".", action_sequence[k][0], action_sequence[k][1]))

    return data_augmented, h_new

def choose(lst, length):
    res = []
    for i in range(len(lst)-length):
        tmp = []
        for j in range(length):
            tmp.append([lst[i+j]])
        res.append(tmp)
    return res

def split_data(action_sequence, verb_table, high_index, low_index, plan):
    data_augmented = []
    total_length = len(action_sequence)
    new_sg_cnt = 0
    # the new high level subgoal index for split cases are just the original low level index in action_sequence
    h_new = np.array(copy.copy(low_index))
    for i in range(total_length):
        verbs = action_sequence[i][0]
        nouns = action_sequence[i][1]
        tmp_augmented = {}
        for idx, (v,n) in enumerate(zip(verbs, nouns)):
            
            tmp_augmented[idx] = []
            all_verbs = verb_table[v]
            num_verbs = len(all_verbs)
            verb = all_verbs[np.random.randint(num_verbs)]
            if v == "PutObject":
                if n.lower() in put_in_list:
                    verb += " in"
                else:
                    verb += " on"
            tmp = n.split('|')
            try:
                if len(tmp) == 1:
                    noun = tmp[0].lower()
                elif len(tmp) == 4:
                    noun = tmp[0].lower()
                else:
                    noun = tmp[-1].lower().split('_')[0]
            except:
                print("error")
                return None
            sen = verb.capitalize() + ' the ' + noun + '. '
            tmp_augmented[idx].append((sen, [v], [noun]))

            data_augmented.append((sen, [v], [noun]))
            new_sg_cnt += 1
    
    plan_new = []
    for sg_plan in plan:
        for low_plan in sg_plan:
            plan_new.append([low_plan])

    return data_augmented, h_new, plan_new
    
def genActionPhrase(json_dir, split_id):
    with open('./data/splits_oct21.json', 'r') as f:
        splits = json.load(f)

    train_path = os.path.join(json_dir, split_id)
    train_path = [ os.path.join(train_path, a)  for a in os.listdir(train_path)]
    action_sequences = []
    phrases = []
    tasks = []
    trials = []
    person_id = []
    high_index = []
    low_index = []
    # the planner actions field for each low level action
    planner_actions = []

    # parse the subgoal instructions from all Turkers and get verbs and nouns 
    for path in tqdm(train_path):
        trajs = os.listdir(path)
        for traj in trajs:
            with open(os.path.join(path, traj, 'traj_data_backup.json'), 'r') as f:
                data = json.load(f)
            
            # sample whether some high level actions should be splitted
            split_high = np.random.randint(2)
            if split_high:
                split_high_list = ["CoolObject", "HeatObject", "CleanObject", "GotoLocation"]
            else:
                split_high_list = ["GotoLocation"]

            task_id = path.split("/")[-1]
            trial_id = traj
            floorPlan = data['scene']['floor_plan']
            object_poses = data['scene']['object_poses']
            actions = data['plan']['low_actions']
            init_action = data['scene']['init_action']
            object_toggles  = data['scene']['object_toggles']
            dirty_and_empty = data['scene']['dirty_and_empty']
            images_info = data["images"]
            high_idx = []
            low_idx = []
            sg_st = []
            sg_end = []
            low_cnt = -1
            gotoState = False
            old_low_action = "None"
            old_high = data['plan']['high_pddl'][0]['discrete_action']['action']
            for im_info in images_info:
                h = im_info["high_idx"]
                high_idx.append(h)
                high_action = data['plan']['high_pddl'][h]['discrete_action']['action']
                l = im_info["low_idx"]
                low_action = data['plan']['low_actions'][l]['discrete_action']['action']
                added = False
                # use some tricks to make the low index keep the same for low actions under some subgoal
                if not low_action == old_low_action:
                    old_low_action = low_action
                    
                    if not gotoState:
                        low_cnt += 1
                        added = True

                    if high_action in split_high_list:
                        gotoState = True
                        if high_action == old_high:
                            pass
                        else:
                            old_high = high_action
                            if not added:
                                low_cnt += 1
                    else:
                        if gotoState:
                            low_cnt += 1
                        gotoState = False

                low_idx.append(low_cnt)

            pddl = data['plan']['high_pddl']
            low_actions = data['plan']['low_actions']
            
            for idx, item in enumerate(data['plan']['high_pddl']):
                if data['plan']['high_pddl'][idx]['discrete_action']['action'] == 'NoOp':
                    continue
            
                v = data['plan']['high_pddl'][idx]['discrete_action']['action']

                # handle the goto, heat, cool and clean cases
                if v in split_high_list:
                    noun = data['plan']['high_pddl'][idx]['discrete_action']['args'][-1]
                    for i in range(len(data['turk_annotations']['anns'])):
                        syn_len = len(verb_table[v])
                        data['turk_annotations']['anns'][i]['high_descs'][idx] = verb_table[v][np.random.randint(syn_len)] + " the " + data['plan']['high_pddl'][idx]['discrete_action']['args'][-1]
                    
                    found = False
                    todelete = []
                    for i in range(len(data['plan']['low_actions'])):
                        if data['plan']['low_actions'][i]['high_idx'] == idx and found is not True:
                            data['plan']['low_actions'][i]['action'] = v
                            data['plan']['low_actions'][i]['args'] = [noun]
                            
                            found = True
                        elif data['plan']['low_actions'][i]['high_idx'] == idx and found is True:
                            todelete.append(i)
                            
                    for i in reversed(range(len(data['plan']['low_actions']))):
                        if i in todelete:
                            data['plan']['low_actions'].pop(i)
                
                                
            turk_annotations =  data['turk_annotations']['anns']
            low_actions = data['plan']['low_actions']
            
            for personi in range(len(data['turk_annotations']['anns'])):
                annos = data['turk_annotations']['anns'][personi]['high_descs']
                action_sequence = []
                short_phrases = []
                planner_action = []
                for annoi, anno in enumerate(annos):
                    sentence = anno
                    verbs = []
                    nouns = []
                    plan_act = []
                    for entry in data['plan']['low_actions']:
                        if entry['high_idx'] == annoi:
                            if 'receptacleObjectId' in entry['api_action']:
                                obj = entry['api_action']['receptacleObjectId']
                            elif 'objectId' in entry['api_action']:
                                obj = entry['api_action']['objectId']
                            else:
                                obj = None
                            if obj is not None:
                                tmp = obj.split('|')
                                if len(tmp) == 4:
                                    obj = tmp[0]
                                elif len(tmp) == 5:
                                    obj = tmp[-1].split('_')[0]
                            if 'args' not in entry or len(entry['args'][-1]) == 0:
                                nouns.append(obj)
                            else:
                                nouns.append(entry['args'][-1])
                            if 'action' not in entry:
                                verbs.append(entry['api_action']['action'])
                            else:
                                verbs.append(entry['action'])
                            
                            # print(data['plan']['high_pddl'][annoi]['discrete_action'])
                            if "Move" in entry['api_action']['action'] or "Look" in entry['api_action']['action'] or "Rotate" in entry['api_action']['action']:
                                plan_act.append(data['plan']['high_pddl'][annoi]['planner_action'])
                            elif data['plan']['high_pddl'][annoi]['discrete_action']['action'] in split_high_list:
                                plan_act.append(data['plan']['high_pddl'][annoi]['planner_action'])
                            else:
                                plan_act.append(entry['api_action'])
                    action_sequence.append((verbs, nouns))
                    short_phrases.append(sentence)
                    planner_action.append(plan_act)
                
                planner_actions.append(planner_action)
                action_sequences.append(action_sequence)
                phrases.append(short_phrases)
                tasks.append(task_id)
                trials.append(trial_id)
                person_id.append(personi)
                high_index.append(high_idx)
                low_index.append(low_idx)
    
    return action_sequences, phrases, tasks, trials, person_id, high_index, low_index, planner_actions

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
        for idx, im in enumerate(traj_data["images"]):
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

# combine low level actions based on templates 
def combine_template_sg(a, verb_table, h_idx, l_idx, plan):
    d, h, plan = split_data(a, verb_table, h_idx, l_idx, plan)

    phrase_seq = []
    verb_seq = []
    noun_seq = []
    plan_seq = []
    for idx, low_data in enumerate(d):
        phrase = low_data[0]
        verbs = low_data[1]
        nouns = low_data[2]
        phrase_seq.append(phrase)
        verb_seq += verbs
        noun_seq += nouns
        plan_seq += (plan[idx])
    
    seq_len = len(verb_seq)
    d_new = []
    h_new = copy.copy(h)
    plan_new = []

    i = 0
    new_sg_idx = 0
    while i < seq_len:
        if i < seq_len - 2:
            sub_seq_v3 = tuple(verb_seq[i:(i+3)])
            sub_seq_n3 = tuple(noun_seq[i:(i+3)])
        else:
            sub_seq_v3 = None

        if i < seq_len - 1:
            sub_seq_v2 = tuple(verb_seq[i:(i+2)])
            sub_seq_n2 = tuple(noun_seq[i:(i+2)])
        else:
            sub_seq_v2 = None

        foundTemplate = False
        # match templates with length 3 first
        if sub_seq_v3 in sg_verb_templates and np.random.rand() > 0.2:
            template_idx = sg_verb_templates.index(sub_seq_v3)
            noun_req = sg_same_noun_idx[template_idx]
            if len(noun_req) == 2 and sub_seq_n3[noun_req[0]] == sub_seq_n3[noun_req[1]]:
                foundTemplate = True
            elif len(noun_req) == 0:
                foundTemplate = True
            else:
                foundTemplate = False
            
            if foundTemplate:
                new_phrases = ""
                for j in range(i, i+3):
                    if j == i+2:
                        new_phrases += phrase_seq[j]
                    else:
                        new_phrases += phrase_seq[j].replace(". ", "") + ", "
                    h_new[np.where(h_new == j)[0]] = copy.copy(new_sg_idx)
                
                # whether to use a non-naive instruction
                if np.random.randint(2) == 0:
                    new_phrases = sg_instr_template[template_idx]
                    for n_idx in range(3):
                        new_phrases = new_phrases.replace("noun"+str(n_idx), sub_seq_n3[n_idx])

                d_new.append((new_phrases, sub_seq_v3, sub_seq_n3))
                plan_new.append(plan_seq[i:(i+3)])
                i += 3

        # try templates with length 2
        if not foundTemplate and sub_seq_v2 in sg_verb_templates and np.random.rand() > 0.2:
            template_idx = sg_verb_templates.index(sub_seq_v2)
            noun_req = sg_same_noun_idx[template_idx]
            if len(noun_req) == 2 and sub_seq_n2[noun_req[0]] == sub_seq_n2[noun_req[1]]:
                foundTemplate = True
            elif len(noun_req) == 0:
                foundTemplate = True
            else:
                foundTemplate = False
            
            if foundTemplate:
                new_phrases = ""
                for j in range(i, i+2):
                    if j == i+1:
                        new_phrases += phrase_seq[j]
                    else:
                        new_phrases += phrase_seq[j].replace(". ", "") + ", "
                    h_new[np.where(h_new == j)[0]] = copy.copy(new_sg_idx)
                
                # whether to use a non-naive instruction
                if np.random.randint(2) == 0:
                    new_phrases = sg_instr_template[template_idx]
                    for n_idx in range(2):
                        new_phrases = new_phrases.replace("noun"+str(n_idx), sub_seq_n2[n_idx])

                d_new.append((new_phrases, sub_seq_v2, sub_seq_n2))
                plan_new.append(plan_seq[i:(i+2)])
                i += 2

        # combine goto and one low level action
        if not foundTemplate and sub_seq_v2 is not None and verb_seq[i] == "GotoLocation" and np.random.rand() > 0.2:
            foundTemplate = True
            template_idx = 0
            new_phrases = ""
            for j in range(i, i+2):
                if j == i+1:
                    new_phrases += phrase_seq[j]
                else:
                    new_phrases += phrase_seq[j].replace(". ", "") + ", "
                h_new[np.where(h_new == j)[0]] = copy.copy(new_sg_idx)
            
            if np.random.randint(2) == 0:
                new_phrases = phrase_seq[i+1]

            d_new.append((new_phrases, sub_seq_v2, sub_seq_n2))
            plan_new.append(plan_seq[i:(i+2)])
            i += 2
        
        # if no match, split into low level actions
        if not foundTemplate:
            d_new.append((phrase_seq[i], (verb_seq[i],), (noun_seq[i],)))
            plan_new.append([plan_seq[i]])
            h_new[np.where(h_new == i)[0]] = copy.copy(new_sg_idx)
            i += 1

        new_sg_idx += 1

    return d_new, h_new, plan_new, h, d

def augmentData(json_dir, split_id):
    # split, combine subgoals or combine navigation and low level actions
    # output:
    #   data_augmented: each entry [(phrase, verbs, nouns), ...] is for instructions and action labels from one user of one task 
    #   new_high_idx: the high subgoal idx for all images 
    #   tasks: the task label corresponds to the augmented data
    #   trials: the trial label corresponds to the augmented data
    #   person_id: the user instruction index corresponds to the augmented data
    action_sequences, phrases, tasks, trials, person_id, high_index, low_index, planner_actions = genActionPhrase(json_dir, split_id)
    
    # sample augmented data
    data_augmented = []
    new_high_idx = []
    new_planner_actions = []
    h_splits = []
    d_splits = []
    for task, trial, a, p, h_idx, l_idx, plan in zip(tasks, trials, action_sequences, phrases, high_index, low_index, planner_actions):
        # if not task == "pick_cool_then_place_in_recep-TomatoSliced-None-CounterTop-10" or not trial == "trial_T20190909_062026_099967":
            # continue
        d, h_new, plan_new, h_split, d_split = combine_template_sg(a, verb_table, h_idx, l_idx, plan)           
        
        data_augmented.append(d)
        new_high_idx.append(h_new)
        new_planner_actions.append(plan_new)
        h_splits.append(h_split)
        d_splits.append(d_split)

    # sanity check
    for i in range(len(data_augmented)):
        if not len(data_augmented[i])-1 == new_high_idx[i][-1]:
            new_high_idx[i][np.where(new_high_idx[i] == new_high_idx[i][-1])[0]] = -1
            new_high_idx[i][np.where(new_high_idx[i] == -1)[0]] = np.max(new_high_idx[i])

    with open('./data/augmented_data_' + split_id + '.pickle', 'wb') as handle:
        pickle.dump([data_augmented, new_high_idx, tasks, trials, person_id, new_planner_actions], handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    np.random.seed(1)
    json_dir = "./data/generated_2.1.0/"
    splits = ["valid_seen", "valid_unseen", "train"]

    for split_id in splits:
        backupJson(json_dir, split_id)
        augmentData(json_dir, split_id)
        modifyData(json_dir, split_id)
    