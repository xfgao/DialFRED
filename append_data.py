import os
import sys
import csv
import json
import copy
import numpy as np
import pickle5 as pickle
import shutil
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import logging
import time
import splitter
import torch
from itertools import chain, combinations, combinations_with_replacement

from tqdm import tqdm
from vocab import Vocab

logging.basicConfig(filename='modify_data.log', level=logging.INFO)
np.random.seed(0)

def def_value():
	return None

def all_subsets(ss):
    return chain(*map(lambda x: combinations_with_replacement(ss, x), range(0, len(ss)+1)))

def prepHdcAns(answer_dir, answer_pck_fn):
	# transform the csv Turker answers to a pickle
	# answer_pkl = {"direction": directionAns, "appear": appearanceAns, "location": locationAns, "obj": objAns}
	# directionAns: task:trial:subgoal:answer
	# appearanceAns: task:trial:subgoal:instr_id:objectname:answer
	# locAns: task:trial:subgoal:instr_id:objectname:answer
	# objAns: task:trial:subgoal:instr_id:objectname:answer
	
	ans_types = ["direction", "appear", "loc", "obj"]
	turker_ans = {key:{} for key in ans_types}
	human_ans_fn = answer_dir + "mergedQA.csv"
	with open(human_ans_fn, "r") as inputcsv:
		csvreader = csv.reader(inputcsv, delimiter=',')
		header = next(csvreader)

		samp_instr_idx = header.index("Input.task_instr")
		url_idx = header.index("Input.video_url")
		high_idx = header.index("Input.subgoal_idx")
		q_start_idx = header.index("Answer.question-template.appearance_clarification1")
		q_end_idx = header.index("Answer.question-template.object_clarification1")
		q_other_idx = header.index("Answer.question-template.others")
		noun_start_idx = header.index("Input.noun1")
		answer_idx = header.index("Answer.answer")
		instr_start_idx = header.index("Input.task_instr0")

		for row in csvreader:
			url = row[url_idx]
			task, trial = url.split("_trial")
			task = task.split("/")[-1]
			trial = "trial" + trial.replace(".mp4", "")
			subgoal = int(row[high_idx])
			answer_user = row[answer_idx]
			all_instr = row[instr_start_idx:(instr_start_idx+8)]
			instruction = row[samp_instr_idx]

			if instruction in all_instr:
				instr_idx = all_instr.index(instruction)
			else:
				instr_idx = 6

			for idx in range(q_start_idx, q_other_idx):
				if row[idx] == "TRUE":
					question_header = header[idx].split(".")[-1]
					q_type = question_header[:-1]
					sel_noun_idx = int(question_header[-1])-1
					break
			
			# skip other qas
			if idx >= q_other_idx:
				continue
			
			if q_type == "appearance_clarification":
				ans_t = "appear"
				sel_noun = row[noun_start_idx+sel_noun_idx]
			elif q_type == "location_clarification":
				ans_t = "loc"
				sel_noun = row[noun_start_idx+sel_noun_idx]
			elif q_type == "direction_clarification":
				ans_t = "direction"
				sel_noun = None
			else:
				ans_t = "obj"
				sel_noun = row[noun_start_idx+sel_noun_idx]
			
			if task not in turker_ans[ans_t]:
				turker_ans[ans_t][task] = {}
			
			if trial not in turker_ans[ans_t][task]:
				turker_ans[ans_t][task][trial] = {}

			if ans_t is not "direction":
				if subgoal not in turker_ans[ans_t][task][trial]:
					turker_ans[ans_t][task][trial][subgoal] = {}

				if instr_idx not in turker_ans[ans_t][task][trial][subgoal]:
					turker_ans[ans_t][task][trial][subgoal][instr_idx] = {}

				if sel_noun not in turker_ans[ans_t][task][trial][subgoal][instr_idx]:
					turker_ans[ans_t][task][trial][subgoal][instr_idx][sel_noun] = ""
				
				turker_ans[ans_t][task][trial][subgoal][instr_idx][sel_noun] += " " + answer_user
			else:
				if subgoal not in turker_ans[ans_t][task][trial]:
					turker_ans[ans_t][task][trial][subgoal] = ""
				turker_ans[ans_t][task][trial][subgoal] += " " + answer_user
	
	ans_pck_fn = answer_dir + answer_pck_fn
	with open(ans_pck_fn,"wb") as f:
		pickle.dump(turker_ans, f, protocol=pickle.HIGHEST_PROTOCOL)

def recoverJson(gen_data_dir):
	# recover the json files from the backup
	splits = ["train", "valid_seen", "valid_unseen"]
	for sp in splits:
		sp_dir = gen_data_dir + sp + "/"
		task_fds = os.listdir(sp_dir)
		for task in task_fds:
			task_dir = sp_dir + task + "/"
			trial_fds = os.listdir(task_dir)
			for trial in trial_fds:
				# recover from the original json file backup
				old_fn = task_dir + trial + "/traj_data_augmented_backup.json"
				new_fn = task_dir + trial + "/traj_data.json"
				shutil.copy(old_fn, new_fn)

def appendQAOracle(gen_data_dir, answer_dir, old_vocab_fn, new_vocab_fn, augmented_data=False):
	ans_types = ["direction", "appear", "loc"]
	# question_to_ask == 0 means not asking question
	qtype2idx = {"location":1, "appearance":2, "direction":3}
	if augmented_data:
		aug_str = "_augmented"
	else:
		aug_str = ""
	answers = {}
	for ans_t in ans_types:
		ans_fn = answer_dir + ans_t + aug_str + ".pkl"
		with open(ans_fn, "rb") as f:
			ans = pickle.load(f)
		answers[ans_t] = ans

	# create a new vocab
	old_vocab = torch.load(old_vocab_fn)
	new_vocab = copy.copy(old_vocab)

	whole_voc_list = ['<<pad>>', '<<seg>>', '<<stop>>', '<<mask>>']
	splits = ["train", "valid_seen", "valid_unseen"]
	# splits = ["valid_seen"]
	task_cnt = 0
	success_ratio_cnt = [0, 0]
	num_q_all = []

	# save the original traj data file for backup and modify the file
	for sp in splits:
		print(sp)
		sp_dir = gen_data_dir + sp + "/"
		task_fds = os.listdir(sp_dir)
		for task in tqdm(task_fds):
			task_dir = sp_dir + task + "/"
			trial_fds = os.listdir(task_dir)
			for trial in trial_fds:
				# first backup the original json file
				old_fn = task_dir + trial + "/traj_data_augmented_backup.json"
				new_fn = task_dir + trial + "/traj_data.json"
				with open(old_fn, "r") as f:
					traj_data = json.load(f)
				
				allsg_object_set = []
				new_data = copy.deepcopy(traj_data)
				num_sg = len(new_data["turk_annotations"]["anns"][0]["high_descs"])
				# some tasks/trials cannot be found
				if task not in answers["loc"] or trial not in answers["loc"][task]:
					task_cnt += 1
					logging.info("Unable to find answers for split %s task %s %s" % (sp, task, trial))
					# with open(new_fn, "w") as f:
					# 	json.dump(new_data, f, sort_keys=True, indent=4)
					continue

				for sg_idx in range(num_sg):
					object_set = set()
					loc_dict = None
					if sg_idx in answers["loc"][task][trial] and 0 in answers["loc"][task][trial][sg_idx]:
						loc_dict = answers["loc"][task][trial][sg_idx][0]
						object_set.update(list(loc_dict.keys()))

					appear_dict = None
					if sg_idx in answers["appear"][task][trial] and 0 in answers["appear"][task][trial][sg_idx]:
						appear_dict = answers["appear"][task][trial][sg_idx][0]
						object_set.update(list(appear_dict.keys()))
					
					allsg_object_set.append(object_set)

				# enumerate all objects for 1 question and sample 1 object for 2 questions or more
				all_question_to_ask = list(all_subsets([1, 2, 3]))
				all_question_to_ask_sg = []
				all_object_to_ask_sg = []
				num_turkers = 0
				for sg_idx in range(num_sg):
					q_to_ask_sg = []
					object_to_ask_sg = []
					# skip the failure matched cases 
					if len(allsg_object_set[sg_idx]) == 0:
						all_question_to_ask_sg.append(q_to_ask_sg)
						all_object_to_ask_sg.append(object_to_ask_sg)
						continue
					for q_idx, q in enumerate(all_question_to_ask):
						if len(q) == 1 and not 3 in q:
							for app_times in range(len(allsg_object_set[sg_idx])):
								q_to_ask_sg.append(q)
								object_to_ask_sg.append([list(allsg_object_set[sg_idx])[app_times]])
						
						else:
							q_to_ask_sg.append(q)
							object_to_ask_sg_temp = []
							for obj_idx in range(len(q)):
								object_to_ask_sg_temp.append(np.random.choice(list(allsg_object_set[sg_idx])))
							object_to_ask_sg.append(object_to_ask_sg_temp)
							
					all_question_to_ask_sg.append(q_to_ask_sg)
					all_object_to_ask_sg.append(object_to_ask_sg)
					if len(q_to_ask_sg) > num_turkers:
						num_turkers = len(q_to_ask_sg)

				for tk_new in range(1, num_turkers):
					new_data["turk_annotations"]["anns"].append(copy.deepcopy(new_data["turk_annotations"]["anns"][0]))

				for tk_idx in range(num_turkers):
					for sg_idx in range(num_sg):
						num_q = 0
						if len(all_question_to_ask_sg[sg_idx]) > tk_idx:
							questions_to_ask = all_question_to_ask_sg[sg_idx][tk_idx]
							select_objs = all_object_to_ask_sg[sg_idx][tk_idx]
							# dummy variable to pass if condition
							question_to_ask = [1,2,3]
						else:
							questions_to_ask = []
							select_obj = ["None"]

						object_set = set()
						loc_dict = None
						if sg_idx in answers["loc"][task][trial] and 0 in answers["loc"][task][trial][sg_idx]:
							loc_dict = answers["loc"][task][trial][sg_idx][0]
							object_set.update(list(loc_dict.keys()))

						appear_dict = None
						if sg_idx in answers["appear"][task][trial] and 0 in answers["appear"][task][trial][sg_idx]:
							appear_dict = answers["appear"][task][trial][sg_idx][0]
							object_set.update(list(appear_dict.keys()))

						dir_dict = None
						if sg_idx in answers["direction"][task][trial] and answers["direction"][task][trial][sg_idx] is not None:
							dir_dict = answers["direction"][task][trial][sg_idx]['ans']

						new_ans = ""
						if len(question_to_ask) > 0:
							for sg_q_idx in range(len(questions_to_ask)):
								question_to_ask = [questions_to_ask[sg_q_idx]]
								select_obj = select_objs[sg_q_idx]
								
								if select_obj in object_set or 3 in question_to_ask:
									if 1 in question_to_ask and select_obj in object_set:
										new_ans += " <<loc>> " + select_obj + " <<ans>> " + str(loc_dict[select_obj]["ans"])
									if 2 in question_to_ask and select_obj in object_set:
										new_ans += " <<app>> " + select_obj + " <<ans>> " + str(appear_dict[select_obj]["ans"])
									if 3 in question_to_ask:
										new_ans += " <<dir>> " + str(dir_dict)
						
						num_q_all.append(num_q)
						whole_ans = copy.deepcopy(traj_data["turk_annotations"]["anns"][0]["high_descs"][sg_idx].lower()) + new_ans.lower()
						new_data["turk_annotations"]["anns"][tk_idx]["high_descs"][sg_idx] = copy.deepcopy(whole_ans)
						vocab_data = copy.deepcopy(whole_ans)
						# split based on regular expression
						vocab_data = re.split('[^a-zA-Z0-9_<>\']', vocab_data)
						# split the sentence and remove spaces, but keep the comma and period
						vocab_data = list(filter(("").__ne__, vocab_data))
						vocab_data = list(filter((" ").__ne__, vocab_data))
						whole_voc_list += vocab_data

				# print(success_ratio_cnt[0]/success_ratio_cnt[1])
				with open(new_fn, "w") as f:
					json.dump(new_data, f, sort_keys=True, indent=4)

				task_cnt += 1

	# print(success_ratio_cnt[0]/success_ratio_cnt[1])
	new_vocab["word"] = Vocab(whole_voc_list)
	torch.save(new_vocab, new_vocab_fn)

def main():
	gen_data_dir = "./data/generated_2.1.0/"
	answer_dir = "./data/answers/"
	old_vocab_fn = "./files/human.vocab"
	
	# recoverJson(gen_data_dir)
	vocab_fn = "./files/augmented_human.vocab"

	# append both oracle questions and answers, mode: all, selected, random (either randomly ask 1 question or no question), random_num (ask random number of questions)
	appendQAOracle(gen_data_dir, answer_dir, old_vocab_fn, vocab_fn, augmented_data=True)

if __name__ == "__main__":
	main()
