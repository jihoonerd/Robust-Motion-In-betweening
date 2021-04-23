import json

def from_pose_to_json(joint_names, pose, filename):
    json_out = {}
    json_out['pose_arr'] = pose.tolist()
    json_out['joint_names'] = joint_names

    with open(filename, 'w') as outfile:
        json.dump(json_out, outfile)


def from_root_to_json(root, filename):
    json_out = {}
    json_out['root_pos'] = root.tolist()

    with open(filename, 'w') as outfile:
        json.dump(json_out, outfile)


def from_local_q_to_json(joint_names, local_q, filename):
    json_out = {}
    json_out['local_q_arr'] = local_q.tolist()
    json_out['joint_names'] = joint_names

    with open(filename, 'w') as outfile:
        json.dump(json_out, outfile)