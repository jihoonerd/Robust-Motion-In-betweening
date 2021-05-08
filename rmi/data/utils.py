import json

def write_json(filename, local_q, root_pos, joint_names):
    json_out = {}
    json_out['local_q_arr'] = local_q.tolist()
    json_out['root_pos'] = root_pos.tolist()
    json_out['joint_names'] = joint_names
    with open(filename, 'w') as outfile:
        json.dump(json_out, outfile)
