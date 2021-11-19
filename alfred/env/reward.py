from alfred.gen.utils.game_util import get_object
import copy
import numpy as np

class BaseAction(object):
    '''
    base class for API actions
    '''

    def __init__(self, gt_graph, env, rewards, strict=True):
        self.gt_graph = gt_graph # for navigation
        self.env = env
        self.rewards = rewards
        self.strict = strict

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        reward, done = self.rewards['neutral'], True
        return reward, done

class CombinedAction(object):
    '''
    Class for combined actions
    '''
    def __init__(self, gt_graph, env, rewards_all_action, strict=True):
        self.gt_graph = gt_graph # for navigation
        self.env = env
        self.rewards_all_action = rewards_all_action
        self.strict = strict
        self.opposite_action_pairs = {"PutObject": "PickupObject", "OpenObject": "CloseObject", "CloseObject": "OpenObject", "PickupObject": "PutObject", "GotoLocation":"GotoLocation"}
    
    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        subgoal = expert_plan[goal_idx]['planner_action']
        
        # make sure this is a combined subgoal
        assert("parameter" in subgoal)
        subgoal_actions = subgoal['action']
        subgoal_param = subgoal['parameter']
        subgoal_param_reverse = copy.copy(subgoal_param)
        subgoal_param_reverse.reverse()
        subgoal_param_to_check = []
        firstGoto = True
        objActions = {}
        # print("reward subgoal: ", subgoal_actions)

        # ignore some of the identical/opposite actions if there are multiple ones targeted at the same object
        for idx, low_param in enumerate(subgoal_param_reverse):
            # only keep the last goto
            if low_param["action"] == "GotoLocation" and firstGoto:
                subgoal_param_to_check.insert(0, low_param)
                firstGoto = False
            
            if not low_param["action"] == "GotoLocation":
                o_id = low_param["objectId"]
                if o_id not in objActions:
                    objActions[o_id] = []
                
                if low_param["action"] not in objActions[o_id] and \
                    (low_param["action"] not in self.opposite_action_pairs or self.opposite_action_pairs[low_param["action"]] not in objActions[o_id]):
                    objActions[o_id].insert(0, low_param["action"])
                    subgoal_param_to_check.insert(0, low_param)
        
        # if len(subgoal_param) > 2:
        #     print("original:", subgoal_param)
        #     print("new:", subgoal_param_to_check)

        # aggregate the reward for each low level action
        total_reward = 0
        total_done = True
        # done_list = []
        # action_list = []
        for low_idx, low_p in enumerate(subgoal_param):
            if low_p in subgoal_param_to_check:
                action_type = low_p["action"]
                # action_list.append(action_type)
                action = get_action(action_type, self.gt_graph, self.env, self.rewards_all_action, self.strict)
                reward, done = action.get_reward(state, prev_state, expert_plan, goal_idx, low_idx)
                # done_list.append(done)
                total_reward += reward
                total_done = total_done and done
        
        # if len(done_list) > 1 and np.all(done_list):
        #     print(action_list, done_list)

        return total_reward, total_done

class GotoLocationAction(BaseAction):
    '''
    MoveAhead, Rotate, Lookup
    '''

    valid_actions = {'MoveAhead', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown', 'Teleport', 'TeleportFull'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx, low_idx=None):
        # if state.metadata['lastAction'] not in self.valid_actions:
        #     reward, done = self.rewards['invalid_action'], False
        #     return reward, done
        
        # to make it compatible with both original subgoals and combined subgoals
        if low_idx is None:
            subgoal = expert_plan[goal_idx]['planner_action']
        else:
            subgoal = expert_plan[goal_idx]['planner_action']["parameter"][low_idx]
        
        curr_pose = state.pose_discrete
        prev_pose = prev_state.pose_discrete
        tar_pose = tuple([int(i) for i in subgoal['location'].split('|')[1:]])

        prev_actions, _ = self.gt_graph.get_shortest_path(prev_pose, tar_pose)
        curr_actions, _ = self.gt_graph.get_shortest_path(curr_pose, tar_pose)

        prev_distance = len(prev_actions)
        curr_distance = len(curr_actions)
        reward = (prev_distance - curr_distance) * 0.2 # distance reward factor?

        # [DEPRECATED] Old criteria which requires the next subgoal object to be visible
        # Consider navigation a success if we can see the target object in the next step from here.
        # assert len(expert_plan) > goal_idx + 1
        # next_subgoal = expert_plan[goal_idx + 1]['planner_action']
        # next_goal_object = get_object(next_subgoal['objectId'], state.metadata)
        # done = (next_goal_object['visible'] and curr_distance < self.rewards['min_reach_distance'])

        done = curr_distance < self.rewards['min_reach_distance']

        if done:
            reward += self.rewards['positive']

        return reward, done


class PickupObjectAction(BaseAction):
    '''
    PickupObject
    '''

    valid_actions = {'PickupObject', 'OpenObject', 'CloseObject'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx, low_idx=None):
        # if state.metadata['lastAction'] not in self.valid_actions:
        #     reward, done = self.rewards['invalid_action'], False
        #     return reward, done

        if low_idx is None:
            subgoal = expert_plan[goal_idx]['planner_action']
        else:
            subgoal = expert_plan[goal_idx]['planner_action']["parameter"][low_idx]
        reward, done = self.rewards['neutral'], False
        inventory_objects = state.metadata['inventoryObjects']
        if len(inventory_objects):
            inv_object_id = state.metadata['inventoryObjects'][0]['objectId']
            goal_object_id = subgoal['objectId']
            reward, done = (self.rewards['positive'], True) if inv_object_id == goal_object_id else (self.rewards['negative'], False)
        return reward, done


class PutObjectAction(BaseAction):
    '''
    PutObject
    '''

    valid_actions = {'PutObject', 'OpenObject', 'CloseObject'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx, low_idx=None):
        # if state.metadata['lastAction'] not in self.valid_actions:
        #     reward, done = self.rewards['invalid_action'], False
        #     return reward, done

        if low_idx is None:
            subgoal = expert_plan[goal_idx]['planner_action']
        else:
            subgoal = expert_plan[goal_idx]['planner_action']["parameter"][low_idx]
        reward, done = self.rewards['neutral'], False
        target_object_id = subgoal['objectId']
        if 'receptacleObjectId' not in subgoal:
            print(subgoal)
        recep_object = get_object(subgoal['receptacleObjectId'], state.metadata)
        if recep_object is not None:
            is_target_in_recep = target_object_id in recep_object['receptacleObjectIds']
            reward, done = (self.rewards['positive'], True) if is_target_in_recep else (self.rewards['negative'], False)
        return reward, done


class OpenObjectAction(BaseAction):
    '''
    OpenObject
    '''

    valid_actions = {'OpenObject'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx, low_idx=None):
        # if state.metadata['lastAction'] not in self.valid_actions:
        #     reward, done = self.rewards['invalid_action'], False
        #     return reward, done

        if low_idx is None:
            subgoal = expert_plan[goal_idx]['planner_action']
        else:
            subgoal = expert_plan[goal_idx]['planner_action']["parameter"][low_idx]
        reward, done = self.rewards['neutral'], False
        target_recep = get_object(subgoal['objectId'], state.metadata)
        if target_recep is not None:
            is_target_open = target_recep['isOpen']
            reward, done = (self.rewards['positive'], True) if is_target_open else (self.rewards['negative'], False)
        return reward, done


class CloseObjectAction(BaseAction):
    '''
    CloseObject
    '''

    valid_actions = {'CloseObject'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx, low_idx=None):
        # if state.metadata['lastAction'] not in self.valid_actions:
        #     reward, done = self.rewards['invalid_action'], False
        #     return reward, done

        if low_idx is None:
            subgoal = expert_plan[goal_idx]['planner_action']
        else:
            subgoal = expert_plan[goal_idx]['planner_action']["parameter"][low_idx]
        reward, done = self.rewards['negative'], False
        target_recep = get_object(subgoal['objectId'], state.metadata)
        if target_recep is not None:
            is_target_closed = not target_recep['isOpen']
            reward, done = (self.rewards['positive'], True) if is_target_closed else (self.rewards['negative'], False)
        return reward, done


class ToggleObjectAction(BaseAction):
    '''
    ToggleObjectOn, ToggleObjectOff
    '''

    valid_actions = {'ToggleObjectOn', 'ToggleObjectOff'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx, low_idx=None):
        # if state.metadata['lastAction'] not in self.valid_actions:
        #     reward, done = self.rewards['invalid_action'], False
        #     return reward, done

        if low_idx is None:
            subgoal = expert_plan[goal_idx]['planner_action']
        else:
            subgoal = expert_plan[goal_idx]['planner_action']["parameter"][low_idx]
        reward, done = self.rewards['neutral'], False
        target_toggle = get_object(subgoal['objectId'], state.metadata)
        if target_toggle is not None:
            is_target_toggled = target_toggle['isToggled']
            reward, done = (self.rewards['positive'], True) if is_target_toggled else (self.rewards['negative'], False)
        return reward, done


class ToggleObjectOnAction(BaseAction):
    '''
    ToggleObjectOn, ToggleObjectOff
    '''

    valid_actions = {'ToggleObjectOn', 'ToggleObjectOff'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx, low_idx=None):
        # if state.metadata['lastAction'] not in self.valid_actions:
        #     reward, done = self.rewards['invalid_action'], False
        #     return reward, done

        if low_idx is None:
            subgoal = expert_plan[goal_idx]['planner_action']
        else:
            subgoal = expert_plan[goal_idx]['planner_action']["parameter"][low_idx]
        reward, done = self.rewards['neutral'], False
        target_toggle = get_object(subgoal['objectId'], state.metadata)
        if target_toggle is not None:
            is_target_toggled = target_toggle['isToggled']
            reward, done = (self.rewards['positive'], True) if is_target_toggled else (self.rewards['negative'], False)
        return reward, done

class ToggleObjectOffAction(BaseAction):
    '''
    ToggleObjectOn, ToggleObjectOff
    '''

    valid_actions = {'ToggleObjectOn', 'ToggleObjectOff'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx, low_idx=None):
        # if state.metadata['lastAction'] not in self.valid_actions:
        #     reward, done = self.rewards['invalid_action'], False
        #     return reward, done

        if low_idx is None:
            subgoal = expert_plan[goal_idx]['planner_action']
        else:
            subgoal = expert_plan[goal_idx]['planner_action']["parameter"][low_idx]
        reward, done = self.rewards['neutral'], False
        target_toggle = get_object(subgoal['objectId'], state.metadata)
        if target_toggle is not None:
            is_target_toggled = target_toggle['isToggled']
            reward, done = (self.rewards['positive'], True) if not is_target_toggled else (self.rewards['negative'], False)
        return reward, done


class SliceObjectAction(BaseAction):
    '''
    SliceObject
    '''

    valid_actions = {'SliceObject', 'OpenObject', 'CloseObject'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx, low_idx=None):
        # if state.metadata['lastAction'] not in self.valid_actions:
        #     reward, done = self.rewards['invalid_action'], False
        #     return reward, done

        if low_idx is None:
            subgoal = expert_plan[goal_idx]['planner_action']
        else:
            subgoal = expert_plan[goal_idx]['planner_action']["parameter"][low_idx]
        reward, done = self.rewards['neutral'], False
        target_object = get_object(subgoal['objectId'], state.metadata)
        if target_object is not None:
            is_target_sliced = target_object['isSliced']
            reward, done = (self.rewards['positive'], True) if is_target_sliced else (self.rewards['negative'], False)
        return reward, done


class CleanObjectAction(BaseAction):
    '''
    CleanObject
    '''

    valid_actions = {'PutObject', 'PickupObject', 'ToggleObjectOn', 'ToggleObjectOff'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx, low_idx=None):
        # if state.metadata['lastAction'] not in self.valid_actions:
        #     reward, done = self.rewards['invalid_action'], False
        #     return reward, done

        if low_idx is None:
            subgoal = expert_plan[goal_idx]['planner_action']
        else:
            subgoal = expert_plan[goal_idx]['planner_action']["parameter"][low_idx]
        reward, done = self.rewards['neutral'], False
        clean_object = get_object(subgoal['cleanObjectId'], state.metadata)
        if clean_object is not None:
            is_obj_clean = clean_object['objectId'] in self.env.cleaned_objects
            reward, done = (self.rewards['positive'], True) if is_obj_clean else (self.rewards['negative'], False)
        return reward, done


class HeatObjectAction(BaseAction):
    '''
    HeatObject
    '''

    valid_actions = {'OpenObject', 'CloseObject', 'PickupObject', 'PutObject', 'ToggleObjectOn', 'ToggleObjectOff'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx, low_idx=None):
        # if state.metadata['lastAction'] not in self.valid_actions:
        #     reward, done = self.rewards['invalid_action'], False
        #     return reward, done

        reward, done = self.rewards['neutral'], False
        # (+1) GotoLocation/(GotoLocation, OpenObject) -> (+2) OpenObject/PutObject -> (+3) PutObject/... (get the objectId from the PutObject action)
        if "PutObject" in expert_plan[goal_idx+1]['planner_action']['action']:
            next_put_goal_idx = goal_idx+1
            put_obj_idx = expert_plan[next_put_goal_idx]['planner_action']['action'].index("PutObject")
        elif "PutObject" in expert_plan[goal_idx+2]['planner_action']['action']:
            next_put_goal_idx = goal_idx+2
            put_obj_idx = expert_plan[next_put_goal_idx]['planner_action']['action'].index("PutObject")
        elif goal_idx+3 < len(expert_plan) and "PutObject" in expert_plan[goal_idx+3]['planner_action']['action']:
            next_put_goal_idx = goal_idx+3
            put_obj_idx = expert_plan[next_put_goal_idx]['planner_action']['action'].index("PutObject")
        else:
            print(expert_plan[goal_idx:])
            return reward, done

        if next_put_goal_idx < len(expert_plan):
            heat_object_id = expert_plan[next_put_goal_idx]['planner_action']['parameter'][put_obj_idx]['objectId']
            heat_object = get_object(heat_object_id, state.metadata)
            is_obj_hot = heat_object['objectId'] in self.env.heated_objects
            reward, done = (self.rewards['positive'], True) if is_obj_hot else (self.rewards['negative'], False)
        return reward, done


class CoolObjectAction(BaseAction):
    '''
    CoolObject
    '''

    valid_actions = {'OpenObject', 'CloseObject', 'PickupObject', 'PutObject'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx, low_idx=None):
        # if state.metadata['lastAction'] not in self.valid_actions:
        #     reward, done = self.rewards['invalid_action'], False
        #     return reward, done

        reward, done = self.rewards['neutral'], False
        # (+1) GotoLocation/(GotoLocation, OpenObject) -> (+2) OpenObject/PutObject -> (+3) PutObject/... (get the objectId from the PutObject action)
        if "PutObject" in expert_plan[goal_idx+1]['planner_action']['action']:
            next_put_goal_idx = goal_idx+1
            put_obj_idx = expert_plan[next_put_goal_idx]['planner_action']['action'].index("PutObject")
        elif "PutObject" in expert_plan[goal_idx+2]['planner_action']['action']:
            next_put_goal_idx = goal_idx+2
            put_obj_idx = expert_plan[next_put_goal_idx]['planner_action']['action'].index("PutObject")
        elif goal_idx+3 < len(expert_plan) and "PutObject" in expert_plan[goal_idx+3]['planner_action']['action']:
            next_put_goal_idx = goal_idx+3
            put_obj_idx = expert_plan[next_put_goal_idx]['planner_action']['action'].index("PutObject")
        else:
            print(expert_plan[goal_idx:])
            return reward, done
            # raise("The put action after cool is not found.")

        if next_put_goal_idx < len(expert_plan):
            cool_object_id = expert_plan[next_put_goal_idx]['planner_action']['parameter'][put_obj_idx]['objectId']
            cool_object = get_object(cool_object_id, state.metadata)
            is_obj_cool = cool_object['objectId'] in self.env.cooled_objects
            reward, done = (self.rewards['positive'], True) if is_obj_cool else (self.rewards['negative'], False)

        return reward, done


def get_action(action_type, gt_graph, env, reward_config, strict):
    # decide whether the action is a combined action
    if type(action_type) == list:
        return CombinedAction(gt_graph, env, reward_config, strict)
    else:
        action_type_str = action_type + "Action"
        if action_type_str in globals():
            action = globals()[action_type_str]
            return action(gt_graph, env, reward_config[action_type_str], strict)
        else:
            raise Exception("Invalid action_type %s" % action_type_str)