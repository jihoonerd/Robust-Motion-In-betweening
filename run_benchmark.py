import os
import pathlib
import pickle as pkl
import zipfile
from datetime import datetime

import torch
import yaml
from torch.utils.data import DataLoader

from rmi.data.lafan1_dataset import LAFAN1Dataset
from rmi.lafan1 import benchmarks, extract, utils
from rmi.model.network import Decoder, InputEncoder, LSTMNetwork
from rmi.model.positional_encoding import PositionalEncoding
from rmi.model.skeleton import (Skeleton, amass_offsets, sk_joints_to_remove,
                                sk_offsets, sk_parents)


# Load configuration from yaml
config = yaml.safe_load(open('./config/config_base.yaml', 'r').read())


# STATS
# the train/test set actors as in the paper

# Extract stats
if config['data']['dataset'] == 'LAFAN':
    train_actors = ["subject1", "subject2", "subject3", "subject4"]
elif config['data']['dataset'] in ['HumanEva', 'PosePrior']:
    train_actors = ["subject1", "subject2"]
elif config['data']['dataset'] in ['HUMAN4D']:
    train_actors = ["subject1", "subject2", "subject3", "subject4", "subject5", "subject6", "subject7"]
else:
    ValueError("Invalid Dataset")

out_path = config['test']['processed_data_dir']
bvh_folder = config['test']['data_dir']

print('Retrieving statistics...')
stats_file = os.path.join(out_path, 'train_stats.pkl')
if not os.path.exists(stats_file):
    x_mean, x_std, offsets = extract.get_train_stats(bvh_folder, train_actors)
    with open(stats_file, 'wb') as f:
        pkl.dump({
            'x_mean': x_mean,
            'x_std': x_std,
            'offsets': offsets,
        }, f, protocol=pkl.HIGHEST_PROTOCOL)
else:
    print('Preprocessed file found! Reusing stats file: ' + stats_file)
    with open(stats_file, 'rb') as f:
        stats = pkl.load(f)
    x_mean = stats['x_mean']
    x_std = stats['x_std']
    offsets = stats['offsets']

# Set device to use
gpu_id = config['device']['gpu_id']
device = torch.device("cpu")

# Prepare Directory
time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
saved_weight_path = config['test']['saved_weight_path']
result_path = os.path.join('results', time_stamp)
result_gif_path = os.path.join(result_path, 'gif')
pathlib.Path(result_gif_path).mkdir(parents=True, exist_ok=True)
result_pose_path = os.path.join(result_path, 'pose_json')

training_frames = config['test']['test_frames']
print("Predicting Frames: ", training_frames)

# Load Skeleton
offset = sk_offsets if config['data']['dataset'] == 'LAFAN' else amass_offsets
skeleton = Skeleton(offsets=offset, parents=sk_parents, device=device)
skeleton.remove_joints(sk_joints_to_remove)

# Load and preprocess data. It utilizes LAFAN1 utilities
processed_data_dir= config['test']['processed_data_dir']
pathlib.Path(processed_data_dir).mkdir(parents=True, exist_ok=True)
lafan_dataset_test = LAFAN1Dataset(lafan_path=config['data']['data_dir'], processed_data_dir=processed_data_dir, train=False, device=device, window=config['test']['test_window'], dataset=config['data']['dataset'])
lafan_data_loader_test = DataLoader(lafan_dataset_test, batch_size=len(lafan_dataset_test), shuffle=False, num_workers=config['data']['data_loader_workers'])

# Extract dimension from processed data
root_v_dim = lafan_dataset_test.root_v_dim
local_q_dim = lafan_dataset_test.local_q_dim
contact_dim = lafan_dataset_test.contact_dim

# Initializing networks
state_in = root_v_dim + local_q_dim + contact_dim
state_encoder = InputEncoder(input_dim=state_in)
state_encoder.to(device)
state_encoder.load_state_dict(torch.load(os.path.join(saved_weight_path, 'state_encoder.pkl'), map_location=device))

offset_in = root_v_dim + local_q_dim
offset_encoder = InputEncoder(input_dim=offset_in)
offset_encoder.to(device)
offset_encoder.load_state_dict(torch.load(os.path.join(saved_weight_path, 'offset_encoder.pkl'), map_location=device))

target_in = local_q_dim
target_encoder = InputEncoder(input_dim=target_in)
target_encoder.to(device)
target_encoder.load_state_dict(torch.load(os.path.join(saved_weight_path, 'target_encoder.pkl'), map_location=device))

# LSTM
lstm_in = state_encoder.out_dim * 3
lstm = LSTMNetwork(input_dim=lstm_in, hidden_dim=lstm_in, device=device)
lstm.to(device)
lstm.load_state_dict(torch.load(os.path.join(saved_weight_path, 'lstm.pkl'), map_location=device))

# Decoder
decoder = Decoder(input_dim=lstm_in, out_dim=state_in)
decoder.to(device)
decoder.load_state_dict(torch.load(os.path.join(saved_weight_path, 'decoder.pkl'), map_location=device))

pe = PositionalEncoding(dimension=256, max_len=training_frames, device=device)

print("MODELS LOADED WITH SAVED WEIGHTS")

state_encoder.eval()
offset_encoder.eval()
target_encoder.eval()
lstm.eval()
decoder.eval()

for i_batch, sampled_batch in enumerate(lafan_data_loader_test):

    current_batch_size = len(sampled_batch['global_pos'])

    global_pos = sampled_batch['global_pos'].to(device)

    with torch.no_grad():
        # state input
        local_q = sampled_batch['local_q'].to(device)
        root_v = sampled_batch['root_v'].to(device)
        contact = sampled_batch['contact'].to(device)
        # offset input
        root_p_offset = sampled_batch['root_p_offset'].to(device)
        local_q_offset = sampled_batch['local_q_offset'].to(device)
        local_q_offset = local_q_offset.view(current_batch_size, -1)
        # target input
        target = sampled_batch['q_target'].to(device)
        target = target.view(current_batch_size, -1)
        # root pos
        root_p = sampled_batch['root_p'].to(device)
        # global pos
        global_pos = sampled_batch['global_pos'].to(device)

        lstm.init_hidden(current_batch_size)

        root_pred_list = []
        local_q_pred_list = []
        contact_pred_list = []
        pos_next_list = []
        local_q_next_list = []
        root_p_next_list = []
        contact_next_list = []
            
        
        for t in range(training_frames):
            # root pos
            if t  == 0:
                root_p_t = root_p[:,t+10]
                root_v_t = root_v[:,t+10]
                local_q_t = local_q[:,t+10]
                local_q_t = local_q_t.view(local_q_t.size(0), -1)
                contact_t = contact[:,t+10]
            else:
                root_p_t = root_pred  # Be careful about dimension
                root_v_t = root_v_pred[0]
                local_q_t = local_q_pred[0]
                contact_t = contact_pred
                
            assert root_p_offset.shape == root_p_t.shape

            # state input
            state_input = torch.cat([local_q_t, root_v_t, contact_t], -1)
            # offset input
            root_p_offset_t = root_p_offset - root_p_t
            local_q_offset_t = local_q_offset - local_q_t
            offset_input = torch.cat([root_p_offset_t, local_q_offset_t], -1)
            # target input
            target_input = target
            
            h_state = state_encoder(state_input)
            h_offset = offset_encoder(offset_input)
            h_target = target_encoder(target_input)
            
            # Use positional encoding
            tta = training_frames - t
            h_state = pe(h_state, tta)
            h_offset = pe(h_offset, tta)
            h_target = pe(h_target, tta)

            offset_target = torch.cat([h_offset, h_target], dim=1)

            # lstm
            h_in = torch.cat([h_state, offset_target], dim=1).unsqueeze(0)
            h_out = lstm(h_in)
        
            # decoder
            h_pred, contact_pred = decoder(h_out)
            local_q_v_pred = h_pred[:,:,:target_in]
            local_q_pred = local_q_v_pred + local_q_t

            local_q_pred_ = local_q_pred.view(local_q_pred.size(0), local_q_pred.size(1), -1, 4)
            local_q_pred_ = local_q_pred_ / torch.norm(local_q_pred_, dim = -1, keepdim = True)

            root_v_pred = h_pred[:,:,target_in:]
            root_pred = root_v_pred + root_p_t

            # root, q, contact prediction
            if root_pred.size(1) == 1:
                root_pred = root_pred[0]
            else:
                root_pred = root_pred.squeeze()

            if local_q_pred_.size(1) == 1:
                local_q_pred_ = local_q_pred_[0]
            else:                
                local_q_pred_ = local_q_pred_.squeeze() # (N, 22, 4)

            root_pred_list.append(root_pred)
            local_q_pred_list.append(local_q_pred_)
            
            if contact_pred.size(1) == 1:
                contact_pred = contact_pred[0]
            else:
                contact_pred = contact_pred.squeeze()
            contact_pred_list.append(contact_pred)

            # For loss
            pos_next_list.append(global_pos[:, t+1+10])
            local_q_next_list.append(local_q[:,t+1+10].view(local_q.size(0), -1))
            root_p_next_list.append(root_p[:,t+1+10])
            contact_next_list.append(contact[:,t+1+10])


        root_pred_stack = torch.stack(root_pred_list, dim=1)
        local_q_pred_stack = torch.stack(local_q_pred_list, dim=1)
        contact_pred_stack = torch.stack(contact_pred_list, dim=1)
        pos_preds, rot_preds = skeleton.forward_kinematics_with_rotation(local_q_pred_stack, root_pred_stack)

        pos_next_stack = torch.stack(pos_next_list, dim=1)
        root_p_next_list = torch.stack(root_p_next_list, dim=1)
        local_q_next_list = torch.stack(local_q_next_list, dim=1)
        contact_next_list = torch.stack(contact_next_list, dim=1)

        local_q_next_list_ = local_q_next_list.reshape(local_q_next_list.size(0), training_frames, lafan_dataset_test.num_joints, 4)
        local_q_next_list_normalized = torch.nn.functional.normalize(local_q_next_list_, p=2.0, dim=-1)
        
        pos_gts, rot_gts = skeleton.forward_kinematics_with_rotation(local_q_next_list_normalized, root_p_next_list)

        norm_preds_pos = (pos_preds.reshape(pos_preds.size(0), training_frames, -1).permute(0,2,1) - x_mean) / x_std
        norm_gts_pos = (pos_next_stack.reshape(pos_next_stack.size(0), training_frames, -1).permute(0,2,1) - x_mean) / x_std

        norm_preds_quat = rot_preds[:,:,skeleton.has_children(),:]
        norm_gts_quat = rot_gts[:,:,skeleton.has_children(),:]

        l2p = torch.mean(torch.sqrt(torch.sum((norm_preds_pos - norm_gts_pos)**2, dim=1))).item()
        l2q = torch.mean(torch.sqrt(torch.sum((norm_preds_quat - norm_gts_quat)**2, dim=(2,3)))).item()

        pred_quaternions = rot_preds
        npss_pred = pred_quaternions[:,:,skeleton.has_children()].reshape(pred_quaternions.shape[0],pred_quaternions.shape[1], -1)
        npss_gt = rot_gts[:,:,skeleton.has_children()].reshape(rot_gts.shape[0],rot_gts.shape[1], -1)
        
        npss = benchmarks.npss(npss_gt, npss_pred).item()

        print('l2p:', l2p)
        print('l2q:', l2q)
        print('npss:', npss)