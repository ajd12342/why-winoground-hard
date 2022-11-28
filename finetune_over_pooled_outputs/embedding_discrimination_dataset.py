import torch as th
import numpy as np
import pickle as pkl
import random

from torch.utils.data import Dataset

class EmbeddingDiscriminationDataset(Dataset):
    def __init__(self, data_path, embd_type, img_layer, cap_layer, score_type='text', control_flip_pairs=[]):
        self.score_type = score_type
        self.control_flip_pairs = control_flip_pairs
        self.data = pkl.load(open(data_path, 'rb'))
        self.embd_type = embd_type
        self.img_layer = img_layer
        self.cap_layer = cap_layer

        if 'LXMERT' in data_path:
            self.model = 'LXMERT'
        elif 'UNITER' in data_path:
            self.model = 'UNITER'
        else:
            print('Data path doesn\'t correspond to any recognized model!')

    def __len__(self):
        return 4*len(self.data)

    def embd_size(self):
        if self.model == 'LXMERT':
            return self.data[0]['output_img_0_cap_0']['pooled_output'].shape[1]
        elif self.model == 'UNITER':
            return self.data[0]['output_img_0_cap_0']['pooled_output'].shape[0]

    def __getitem__(self, idx):
        
        set_idx = idx // 4
        pair_idx = idx % 4

        if pair_idx == 0:
            outputs = self.data[set_idx]['output_img_0_cap_0']
            if self.score_type == 'text':
                distractor = self.data[set_idx]['output_img_0_cap_1']
            else:
                distractor = self.data[set_idx]['output_img_1_cap_0']
            if pair_idx in self.control_flip_pairs:
                label = 1
            else:
                label = 0
        elif pair_idx == 1:
            outputs = self.data[set_idx]['output_img_0_cap_1']
            if self.score_type == 'text':
                distractor = self.data[set_idx]['output_img_0_cap_0']
            else:
                distractor = self.data[set_idx]['output_img_1_cap_1']
            if pair_idx in self.control_flip_pairs:
                label = 0
            else:
                label = 1
        elif pair_idx == 2:
            outputs = self.data[set_idx]['output_img_1_cap_0']
            if self.score_type == 'text':
                distractor = self.data[set_idx]['output_img_1_cap_1']
            else:
                distractor = self.data[set_idx]['output_img_0_cap_0']
            if pair_idx in self.control_flip_pairs:
                label = 0
            else:
                label = 1
        else:
            outputs = self.data[set_idx]['output_img_1_cap_1']
            if self.score_type == 'text':
                distractor = self.data[set_idx]['output_img_1_cap_0']
            else:
                distractor = self.data[set_idx]['output_img_0_cap_1']
            if pair_idx in self.control_flip_pairs:
                label = 1
            else:
                label = 0
        
        if self.embd_type == 'pooled':
            return {'opt1':outputs['pooled_output'].squeeze(), 
                    'opt2':distractor['pooled_output'].squeeze(), 
                    'label':label}
        if self.embd_type == 'CLS':
            if self.model == 'LXMERT':
                return {'opt1':outputs['language_hidden_states'][self.cap_layer][:,0].squeeze(),
                        'opt2':distractor['language_hidden_states'][self.cap_layer][:,0].squeeze(),
                        'label':label}
            elif self.model == 'UNITER':
                return {'opt1':outputs['language_hidden_states'][self.cap_layer][0].squeeze(),
                        'opt2':distractor['language_hidden_states'][self.cap_layer][0].squeeze(),
                        'label':label}
        if self.embd_type == 'mean':
            if self.model == 'LXMERT':
                language1 = th.mean(outputs['language_hidden_states'][self.cap_layer],dim=1).squeeze()
                visual1 = th.mean(outputs['vision_hidden_states'][self.img_layer],dim=1).squeeze()
                language2 = th.mean(distractor['language_hidden_states'][self.cap_layer],dim=1).squeeze()
                visual2 = th.mean(distractor['vision_hidden_states'][self.img_layer],dim=1).squeeze()
                return {'opt1':th.cat((language1,visual1)),
                        'opt2':th.cat((language2,visual2)),
                        'label':label}
            elif self.model == 'UNITER':
                language1 = th.mean(outputs['language_hidden_states'][self.cap_layer],dim=0).squeeze()
                visual1 = th.mean(outputs['vision_hidden_states'][self.img_layer],dim=0).squeeze()
                language2 = th.mean(distractor['language_hidden_states'][self.cap_layer],dim=0).squeeze()
                visual2 = th.mean(distractor['vision_hidden_states'][self.img_layer],dim=0).squeeze()
                return {'opt1':th.cat((language1,visual1)),
                        'opt2':th.cat((language2,visual2)),
                        'label':label}
        if self.embd_type == 'max':
            if self.model == 'LXMERT':
                language1 = th.max(outputs['language_hidden_states'][self.cap_layer],dim=1)[0].squeeze()
                visual1 = th.max(outputs['vision_hidden_states'][self.img_layer],dim=1)[0].squeeze()
                language2 = th.max(distractor['language_hidden_states'][self.cap_layer],dim=1)[0].squeeze()
                visual2 = th.max(distractor['vision_hidden_states'][self.img_layer],dim=1)[0].squeeze()
                return {'opt1':th.cat((language1,visual1)),
                        'opt2':th.cat((language2,visual2)),
                        'label':label}
            elif self.model == 'UNITER':
                language1 = th.max(outputs['language_hidden_states'][self.cap_layer],dim=0)[0].squeeze()
                visual1 = th.max(outputs['vision_hidden_states'][self.img_layer],dim=0)[0].squeeze()
                language2 = th.max(distractor['language_hidden_states'][self.cap_layer],dim=0)[0].squeeze()
                visual2 = th.max(distractor['vision_hidden_states'][self.img_layer],dim=0)[0].squeeze()
                return {'opt1':th.cat((language1,visual1)),
                        'opt2':th.cat((language2,visual2)),
                        'label':label}

