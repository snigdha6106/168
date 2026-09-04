import torch
import json
import os
import numpy as np

checkpoint = torch.load("model_weights.pth", map_location=torch.device('cpu'), weights_only=False)
state_dict = checkpoint['model_state_dict']

def tensor_to_list(t):
    return t.detach().cpu().numpy().tolist()

model_params = {
    'X_mean': tensor_to_list(torch.tensor(checkpoint['X_mean'])),
    'X_std': tensor_to_list(torch.tensor(checkpoint['X_std'])),
    'y_mean': float(checkpoint['y_mean']),
    'y_std': float(checkpoint['y_std']),
    'lstm_weight_ih_l0': tensor_to_list(state_dict['lstm.weight_ih_l0']),
    'lstm_weight_hh_l0': tensor_to_list(state_dict['lstm.weight_hh_l0']),
    'lstm_bias_ih_l0': tensor_to_list(state_dict['lstm.bias_ih_l0']),
    'lstm_bias_hh_l0': tensor_to_list(state_dict['lstm.bias_hh_l0']),
    
    'lstm_weight_ih_l1': tensor_to_list(state_dict['lstm.weight_ih_l1']),
    'lstm_weight_hh_l1': tensor_to_list(state_dict['lstm.weight_hh_l1']),
    'lstm_bias_ih_l1': tensor_to_list(state_dict['lstm.bias_ih_l1']),
    'lstm_bias_hh_l1': tensor_to_list(state_dict['lstm.bias_hh_l1']),
    
    'fc_weight': tensor_to_list(state_dict['fc.weight']),
    'fc_bias': tensor_to_list(state_dict['fc.bias'])
}

os.makedirs("../frontend/assets", exist_ok=True)
with open("../frontend/assets/model_parameters.json", "w") as f:
    json.dump(model_params, f)

print("Exported model parameters to frontend/assets/model_parameters.json")
