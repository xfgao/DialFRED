import os 
import numpy as np
import json
from alfred.env.thor_env import ThorEnv
import pickle
from tqdm import tqdm

# get the ground-truth metadata by running the action sequences
def setup_scene(env, traj_data):
    '''
    intialize the scene and agent from the task info
    '''
    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    object_toggles = traj_data['scene']['object_toggles']

    scene_name = 'FloorPlan%d' % scene_num
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    # initialize to start position
    env.step(dict(traj_data['scene']['init_action']))

def generate_submission(input_json):
    # generate a sample submission file
    env = ThorEnv(x_display=0)
    with open(input_json, "r") as f:
        json_data = json.load(f)

    setup_scene(env, json_data)
    dummy_action = {
        "action": "RotateLeft",
        "forceAction": True
    }
    num_subgoal = len(json_data["turk_annotations"]["anns"][0]["high_descs"])
    meta_data = []

    # execute actions and save the metadata for each subgoal
    for sg_idx in range(num_subgoal):
        event = env.step(dummy_action)
        m_data = event.metadata
        m_data["pose_discrete"] = event.pose_discrete
        meta_data.append(m_data)
    
    save_path_full = os.path.join(os.environ['DF_ROOT'] + "/misc/" +input_json[-9:])
    with open(save_path_full, "w") as f:
        json.dump(meta_data, f, sort_keys=True, indent=4)   

def main():
    # path to testset json file 
    input_json = os.environ['DF_ROOT'] + "/testset/dialfred_testset_final/0001.json"
    generate_submission(input_json)


if __name__ == "__main__":
    main()