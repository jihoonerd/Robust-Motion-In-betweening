import json

def from_pose_to_json(joint_names, pose, filename):
    json_out = {}
    json_out['pose_arr'] = pose.tolist()
    json_out['joint_names'] = joint_names

    with open(filename, 'w') as outfile:
        json.dump(json_out, outfile)