import numpy as np
import time
import splitter

nouns_not_nec = ["bottom", "up", "down", "forward", "forwards", "left", "right", "middle", "front", "back", "end", "room", "second", "seconds", "step", "steps", "moment", "moments", "minute", "side", "one", "walk"]
put_in_list = ["box", "bathtubbasin", "bowl", "cabinet", "cup", "drawer", "fridge", "garbagecan", "microwave", "mug", "pan", "pot", "safe", "sinkbasin", "sink"]

# verbs of low level actions that can be combined to form new subgoals
sg_verb_templates = [('OpenObject', 'PutObject', 'CloseObject'), ('OpenObject', 'PickupObject', 'CloseObject'), ('PickupObject', 'GotoLocation'), 
	('PickupObject', 'GotoLocation', 'PutObject'), ('PickupObject', 'GotoLocation', 'SliceObject')]

# index of nouns that are required to be the same
sg_same_noun_idx = [(0, 2), (0, 2), (), (1, 2), (1, 2)]

# non-naive instruction templates for combined subgoals
sg_instr_template = ["Put into the noun0", "Pick up the noun1", "Take the noun0 to the noun1", "Put the noun0 into the noun1", "Cut the noun2 with a knife"]
split_cache = {"saltshaker":"salt shaker", "tvstand":"tv stand"}


def viewDictionary(d, levels, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(levels[indent]) + ": "+ str(key))
        if isinstance(value, dict):
            viewDictionary(value, levels, indent+1)
        else:
            print('\t' * (indent+1) + str(levels[indent+1])+ ": " + str(value))

def logDictionary(f, d, levels, indent=0):
	for key, value in d.items():
		f.write('\t' * indent + str(levels[indent]) + ": "+ str(key) + "\n")
		if isinstance(value, dict):
			logDictionary(f, value, levels, indent+1)
		else:
			f.write('\t' * (indent+1) + str(levels[indent+1])+ ": " + str(value) + "\n")

def get_obj_data(metadata, obj_id):
	objects = metadata["objects"]
	objdata = None
	for o in objects:
		if o["objectId"] == obj_id:
			objdata = o
			break
	
	return objdata

def get_agent_direction(metadata_current, metadata_target, pos_threshold=0.1):
	# given metadata at the beginning of the current subgoal and the next subgoal, get the relative angle the agent should turn to
	current_position = metadata_current['agent']['position']
	current_rotation = metadata_current['agent']['rotation']
	target_position = metadata_target['agent']['position']
	x_offset = target_position['x'] - current_position['x']
	z_offset = target_position['z'] - current_position['z']
	pos_offset = np.sqrt(x_offset**2 + z_offset**2)
	if pos_offset < pos_threshold:
		return 0, pos_offset

	agent_angle = current_rotation['y'] # 0 to 360, zero at z axle
	absolute_angle = np.arctan2(x_offset, z_offset)*180/np.pi # -180 to 180, zero at z axle
	relative_angle = absolute_angle - agent_angle
	relative_angle =  relative_angle % 360 # reduce the angle
	relative_angle = (relative_angle + 360) % 360 # force it to be the positive remainder, so that 0 <= angle < 360

	return relative_angle, pos_offset

def get_obj_direction(metadata, objdata):
	# given event metadata and object data, get the relative angle between object and agent
	agent_position = metadata['agent']['position']
	agent_rotation = metadata['agent']['rotation']
	obj_position = objdata['position']
	x_offset = obj_position['x'] - agent_position['x']
	z_offset = obj_position['z'] - agent_position['z']

	agent_angle = agent_rotation['y'] # 0 to 360, zero at z axle
	absolute_angle = np.arctan2(x_offset, z_offset)*180/np.pi # -180 to 180, zero at z axle
	relative_angle = absolute_angle - agent_angle
	relative_angle =  relative_angle % 360 # reduce the angle
	relative_angle = (relative_angle + 360) % 360 # force it to be the positive remainder, so that 0 <= angle < 360

	return relative_angle

def dirAns(relative_angle, relative_pos, pos_threshold=0.1):
	# no need to move if the position and rotation offset is small 
	if relative_pos < pos_threshold and (relative_angle < 45 or relative_angle >= 315):
	# if relative_pos < pos_threshold:
		dir_ans = "You don't need to move."
	else:
		dir_ans = "You should "
		if relative_angle < 45 or relative_angle >= 315:
			dir_ans += "go straight ahead."
		elif relative_angle < 135:
			dir_ans += "turn right." 
		elif relative_angle < 225:
			dir_ans += "turn around." 
		else:
			dir_ans += "turn left." 
	
	return dir_ans

def objLocAns(obj_name, relative_angle, containers):
	# given relative_angle (0 <= angle < 360) and object name and the container that contains the object, output a direction answer
	dir_ans = "The %s is " % obj_name

	# add the answer to describe the object location relative to the agent
	if relative_angle < 22.5 or relative_angle >= 337.5:
		dir_ans += "in front of you"
	elif relative_angle < 67.5:
		dir_ans += "to your front right" 
	elif relative_angle < 112.5:
		dir_ans += "to your right" 
	elif relative_angle < 157.5:
		dir_ans += "to your rear right" 
	elif relative_angle < 202.5:
		dir_ans += "behind you" 
	elif relative_angle < 247.5:
		dir_ans += "to your rear left"
	elif relative_angle < 292.5:
		dir_ans += "to your left" 
	else:
		dir_ans += "to your front left" 

	if containers is not None and len(containers) > 0:
		# take the first container name of the object
		container = containers[0].split("|")[0].lower()
		if container not in split_cache:
			container_split = splitter.split(container)
			obj_phrase = " ".join(container_split)
			split_cache[container] = obj_phrase
		else:
			obj_phrase = split_cache[container]

		if container in put_in_list:
			dir_ans += " in the %s" % obj_phrase
		else:
			dir_ans += " on the %s" % obj_phrase
	
	dir_ans += "."

	return dir_ans