import os
import pathlib
from datetime import datetime

import imageio
import numpy as np
import torch
import yaml
from kpt.model.skeleton import TorchSkeleton
from PIL import Image
from pymo.parsers import BVHParser
from torch.utils.data import DataLoader

from rmi.data.lafan1_dataset import LAFAN1Dataset
from rmi.data.utils import write_json
from rmi.model.network import Decoder, InputEncoder, LSTMNetwork
from rmi.model.positional_encoding import PositionalEncoding
from rmi.vis.pose import plot_pose
from rmi.model.skeleton import (Skeleton, sk_joints_to_remove, sk_offsets,
                                sk_parents, joint_names)


def test():
    # Load configuration from yaml
    config = yaml.safe_load(open('./config/config_base.yaml', 'r').read())

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

    training_frames = config['test']['training_frames']

    # Load Skeleton
    skeleton = Skeleton(offsets=sk_offsets, parents=sk_parents, device=device)
    skeleton.remove_joints(sk_joints_to_remove)

     # Load and preprocess data. It utilizes LAFAN1 utilities
    lafan_dataset_test = LAFAN1Dataset(lafan_path=config['data']['data_dir'], processed_data_dir=config['data']['processed_data_dir'], train=False, device=device, start_seq_length=30, cur_seq_length=30, max_transition_length=30)
    lafan_data_loader_test = DataLoader(lafan_dataset_test, batch_size=config['model']['batch_size'], shuffle=False, num_workers=config['data']['data_loader_workers'])

    inference_batch_index = config['test']['inference_batch_index']

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

    pe = PositionalEncoding(dimension=256, max_len=lafan_dataset_test.max_transition_length, device=device)

    print("MODELS LOADED WITH SAVED WEIGHTS")

    state_encoder.eval()
    offset_encoder.eval()
    target_encoder.eval()
    lstm.eval()
    decoder.eval()
    
    for i_batch, sampled_batch in enumerate(lafan_data_loader_test):
        img_gt = []
        img_pred = []
        img_integrated = []

        current_batch_size = len(sampled_batch['global_pos'])

        global_pos = sampled_batch['global_pos'].to(device)
        pose_stack = [global_pos[inference_batch_index, 0+9].numpy()]

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
            
            for t in range(training_frames):
                # root pos
                if t  == 0:
                    root_p_t = root_p[:,t+9]
                    root_v_t = root_v[:,t+9]
                    local_q_t = local_q[:,t+9]
                    local_q_t = local_q_t.view(local_q_t.size(0), -1)
                    contact_t = contact[:,t+9]
                else:
                    root_p_t = root_pred  # Be careful about dimension
                    root_v_t = root_v_pred[0]
                    local_q_t = local_q_pred[0]
                    contact_t = contact_pred[0]
                    
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

                # FK
                root_pred = root_pred.squeeze()
                local_q_pred_ = local_q_pred_.squeeze() # (seq, joint, 4)
                pos_pred, _ = skeleton.forward_kinematics_with_rotation(local_q_pred_.unsqueeze(1), root_pred.unsqueeze(1))
                
                # Exporting
                root_pred_t = root_pred[inference_batch_index].numpy()
                local_q_pred_t = local_q_pred_[inference_batch_index].numpy()

                start_pose = global_pos[inference_batch_index, 0+9].numpy()
                in_between_pose = pose_stack.pop(0)
                assert len(pose_stack) == 0
                pose_stack.append(pos_pred[inference_batch_index, 0].numpy())
                in_between_true = global_pos[inference_batch_index, t+9].numpy()
                target_pose = global_pos[inference_batch_index, training_frames-1+9].numpy()

                pose_path = os.path.join(result_pose_path, f"{i_batch}")
                pathlib.Path(pose_path).mkdir(parents=True, exist_ok=True)

                if t == 0: # root_pose[0] only root check
                    write_json(filename=os.path.join(pose_path, f'start.json'), local_q=sampled_batch['local_q'][inference_batch_index][0].numpy(), root_pos=start_pose[0], joint_names=joint_names)
                    write_json(filename=os.path.join(pose_path, f'target.json'), local_q=sampled_batch['local_q'][inference_batch_index][-1].numpy(), root_pos=target_pose[0], joint_names=joint_names)

                write_json(filename=os.path.join(pose_path, f'{t:05}.json'), local_q=local_q_pred_t, root_pos=root_pred_t, joint_names=joint_names)

                if config['test']['plot']:
                    pred_image_path = os.path.join(result_path, 'pred')
                    pathlib.Path(pred_image_path).mkdir(parents=True, exist_ok=True)
                    plot_pose(start_pose, in_between_pose, target_pose, t, skeleton, pred_image_path, prefix='pred_')
                    gt_image_path = os.path.join(result_path, 'gt')
                    pathlib.Path(gt_image_path).mkdir(parents=True, exist_ok=True)
                    plot_pose(start_pose, in_between_true, target_pose, t, skeleton, gt_image_path, prefix='gt_')

                    pred_img = Image.open('results/'+ time_stamp +'/pred/pred_'+str(t)+'.png', 'r')
                    gt_img = Image.open('results/'+ time_stamp +'/gt/gt_'+str(t)+'.png', 'r')

                    img_pred.append(pred_img)
                    img_gt.append(gt_img)
                    img_integrated.append(np.concatenate([pred_img, gt_img.resize(pred_img.size)], 1))
            
            if config['test']['plot']:
                # if i_batch < 49:
                gif_path = os.path.join(result_gif_path, 'img_%02d.gif' % i_batch)
                imageio.mimsave(gif_path, img_integrated, duration=0.1)


if __name__ == '__main__':
    test()
