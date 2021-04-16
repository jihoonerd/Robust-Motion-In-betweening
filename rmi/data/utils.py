import pandas as pd


def from_pose_to_json(joint_names, pose, filename):
    df = pd.DataFrame(pose)
    df.index = joint_names
    df.columns = ['axis1', 'axis2', 'axis3']
    df.to_json(filename, orient='index')
