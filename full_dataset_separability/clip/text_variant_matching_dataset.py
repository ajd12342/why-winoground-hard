import torch as th
import numpy as np
import pickle as pkl
import random

from torch.utils.data import Dataset

class TextVariantMatchingDataset(Dataset):
    def __init__(self, partition_file, control_flip_pairs=[], layer_idx=0):
        """
        partition_file is 'language_hidden_states_only_feats_[0-3].pkl'
        control_flip_pairs is empty for target task, control split for control task
        layer_idx is which layer's CLS token to probe over
        """
        self.control_flip_pairs = control_flip_pairs
        self.all_texts = pkl.load(open(partition_file,'rb'))
        self.layer_idx = layer_idx

    def __len__(self):
        return 4 * len(self.all_texts)

    def __getitem__(self, idx):
        set_idx = idx // 4
        pair_idx = idx % 4
        """
        Set_idx, out of 100, is which WG set to use
        If pair_idx is 0,
            Reference: Random variant of T0
            Candidate A: Different random variant of T0
            Candidate B: Random variant of T1
            Answer: 0
        If pair_idx is 1,
            Reference: Random variant of T0
            Candidate A: Random variant of T1
            Candidate B: Different random variant of T0
            Answer: 1
        If pair_idx is 2,
            Reference: Random variant of T1
            Candidate A: Random variant of T0
            Candidate B: Different random variant of T1
            Answer: 1
        If pair_idx is 3,
            Reference: Random variant of T1
            Candidate A: Different random variant of T1
            Candidate B: Random variant of T0
            Answer: 0
        """

        caption_0_variants = self.all_texts[set_idx]['cap0'][self.layer_idx]
        caption_1_variants = self.all_texts[set_idx]['cap1'][self.layer_idx]

        cap0_inda = random.randint(0,caption_0_variants.shape[0]-1)
        cap0_indb = random.randint(0,caption_0_variants.shape[0]-1)
        while cap0_inda == cap0_indb:
            cap0_indb = random.randint(0,caption_0_variants.shape[0]-1)

        cap1_inda = random.randint(0,caption_1_variants.shape[0]-1)
        cap1_indb = random.randint(0,caption_1_variants.shape[0]-1)
        while cap1_inda == cap1_indb:
            cap1_indb = random.randint(0,caption_1_variants.shape[0]-1)
        
        if set_idx in self.control_flip_pairs:
            if pair_idx == 0:
                return {'ref':caption_0_variants[cap0_inda],
                        'opt1':caption_0_variants[cap0_indb],
                        'opt2':caption_1_variants[cap1_inda],
                        'label':1}
            elif pair_idx == 1:
                return {'ref':caption_0_variants[cap0_inda],
                        'opt1':caption_1_variants[cap1_inda],
                        'opt2':caption_0_variants[cap0_indb],
                        'label':0}
            elif pair_idx == 2:
                return {'ref':caption_1_variants[cap1_inda],
                        'opt1':caption_0_variants[cap0_inda],
                        'opt2':caption_1_variants[cap1_indb],
                        'label':0}
            else:
                return {'ref':caption_1_variants[cap1_inda],
                        'opt1':caption_1_variants[cap1_indb],
                        'opt2':caption_0_variants[cap0_inda],
                        'label':1}
        else:
            if pair_idx == 0:
                return {'ref':caption_0_variants[cap0_inda],
                        'opt1':caption_0_variants[cap0_indb],
                        'opt2':caption_1_variants[cap1_inda],
                        'label':0}
            elif pair_idx == 1:
                return {'ref':caption_0_variants[cap0_inda],
                        'opt1':caption_1_variants[cap1_inda],
                        'opt2':caption_0_variants[cap0_indb],
                        'label':1}
            elif pair_idx == 2:
                return {'ref':caption_1_variants[cap1_inda],
                        'opt1':caption_0_variants[cap0_inda],
                        'opt2':caption_1_variants[cap1_indb],
                        'label':1}
            else:
                return {'ref':caption_1_variants[cap1_inda],
                        'opt1':caption_1_variants[cap1_indb],
                        'opt2':caption_0_variants[cap0_inda],
                        'label':0}
