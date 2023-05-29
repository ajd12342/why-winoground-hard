from transformers import LxmertTokenizer, LxmertModel, LxmertConfig, LxmertForPreTraining
import torch as th
import numpy as np
import random
import pickle as pkl

tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')
model = LxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased')

test_feats = np.load('/data3/scratch/layne/winoground_features.npy')
test_boxes = np.load('/data3/scratch/layne/winogrond_boxes.npy')

test_texts = # TODO

tokenized_input = tokenizer(test_texts, return_tensors='pt', padding='longest')

tokenized_input['visual_feats'] = th.Tensor(test_feats)
tokenized_input['visual_pos'] = th.Tensor(test_boxes)

out = model(**tokenized_input, output_hidden_states=True)

clout = out['language_output']
cvout = out['vision_output']
ulout = out['language_hidden_states']
uvout = out['vision_hidden_states']

print('clout', len(clout))
print('cvout', len(cvout))
print('ulout', len(ulout))
print('uvout', len(uvout))

print(clout.shape)
print(cvout.shape)
print(ulout[0].shape)
print(uvout[0].shape)

pkl.dump(clout, open('/data3/scratch/layne/LXMERT_XL_last_layer.pkl', 'wb'))
pkl.dump(cvout, open('/data3/scratch/layne/LXMERT_XV_last_layer.pkl', 'wb'))
pkl.dump(ulout, open('/data3/scratch/layne/LXMERT_L_hidden_states.pkl', 'wb'))
pkl.dump(uvout, open('/data3/scratch/layne/LXMERT_V_hidden_states.pkl', 'wb'))
