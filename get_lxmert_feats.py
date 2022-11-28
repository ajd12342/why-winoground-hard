from transformers import LxmertTokenizer, LxmertModel, LxmertConfig, LxmertForPreTraining
import torch as th
import numpy as np
import random
import pickle as pkl
import json

winoground_path = '/saltpool0/data/layneberry/WinoGround/'

tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')
model = LxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased')

test_feats = np.load(winoground_path+'winoground_features.npy')
test_boxes = np.load(winoground_path+'winoground_boxes.npy')

WG = [json.loads(line) for line in open(winoground_path+'examples_augmented.jsonl').readlines()]

winoground_sets = []

assert(len(WG) == 400)

for i in range(len(WG)):
    winoground_set = {}
    winoground_set['image0'] = (test_feats[2*i], test_boxes[2*i])
    winoground_set['image1'] = (test_feats[2*i+1], test_boxes[2*i+1])
   
    c0_inputs = tokenizer([WG[i]['caption_0'][j]['variant'] for j in range(len(WG[i]['caption_0']))],padding=True, return_tensors='pt')
   
    # Image 0, Caption 0
    i0c0_outputs = model(visual_feats=th.Tensor(winoground_set['image0'][0]).unsqueeze(0), visual_pos=th.Tensor(winoground_set['image0'][1]).unsqueeze(0), **c0_inputs,output_hidden_states=True)
    winoground_set['output_img_0_cap_0'] = i0c0_outputs
 
    # Image 1, Caption 0
    i1c0_outputs = model(visual_feats=th.Tensor(winoground_set['image1'][0]).unsqueeze(0), visual_pos=th.Tensor(winoground_set['image1'][1]).unsqueeze(0), **c0_inputs,output_hidden_states=True)
    winoground_set['output_img_1_cap_0'] = i1c0_outputs 
   
    c1_inputs = tokenizer([WG[i]['caption_1'][j]['variant'] for j in range(len(WG[i]['caption_1']))],padding=True, return_tensors='pt')
   
    # Image 0, Caption 1
    i0c1_outputs = model(visual_feats=th.Tensor(winoground_set['image0'][0]).unsqueeze(0), visual_pos=th.Tensor(winoground_set['image0'][1]).unsqueeze(0), **c1_inputs,output_hidden_states=True)
    winoground_set['output_img_0_cap_1'] = i0c1_outputs
 
    # Image 1, Caption 1
    i1c1_outputs = model(visual_feats=th.Tensor(winoground_set['image1'][0]).unsqueeze(0), visual_pos=th.Tensor(winoground_set['image1'][1]).unsqueeze(0), **c1_inputs,output_hidden_states=True)
    winoground_set['output_img_1_cap_1'] = i1c1_outputs

    winoground_sets.append(winoground_set)

pkl.dump(winoground_sets, open(winoground_path+'LXMERT_outputs.pkl', 'wb'))
