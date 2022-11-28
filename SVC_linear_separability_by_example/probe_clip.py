import pickle as pkl
import numpy as np
from sklearn.svm import SVC

winoground_path = '/saltpool0/data/layneberry/WinoGround/'

table = open(winoground_path+'CLS_separability_probe_clip.csv', 'w')
print('Winoground Set,Control?,Layer,Score,Margin',file=table)

feats = pkl.load(open(winoground_path+'clip_variants_feats.pkl','rb'))

import torch
with torch.no_grad():
    for wgset in range(len(feats)):
        for control in [False,True]:
            set1 = feats[wgset]

            # no fusion is relevant
            cap0_vars = set1['cap0']
            cap1_vars = set1['cap1']

            for i in range(len(cap0_vars)):
                label0 = cap0_vars[i].cpu().float().numpy()
                label1 = cap1_vars[i].cpu().float().numpy()
                all_samples = np.concatenate((label0,label1),axis=0) / 100.
                all_labels = np.concatenate((np.zeros(len(label0)),np.ones(len(label1))),axis=0)
                if control:
                    np.random.shuffle(all_labels)
                svc = SVC(kernel='linear',max_iter=100000,C=100)
                svc.fit(all_samples, all_labels)
                score = svc.score(all_samples, all_labels)
                if np.linalg.norm(svc.coef_) == 0:
                    print('Norm is 0 on', wgset)
                    margin = 1000
                else:
                    margin = 2 / np.linalg.norm(svc.coef_) 
                print(str(wgset)+','+str(control)+','+str(i)+','+str(score)+','+str(margin),file=table)

