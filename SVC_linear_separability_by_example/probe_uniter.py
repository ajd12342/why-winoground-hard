import pickle as pkl
import numpy as np
from sklearn.svm import SVC

table = open('CLS_separability_probe_UNITER.csv', 'w')
print('Winoground Set,Control?,Layer,Image,Score,Margin',file=table)

f0 = pkl.load(open('UNITER_variants_feats.pkl','rb'))

for wgset in range(len(f0)): # iterates over sets
    for control in [False,True]: # target or control
        set1 = f0[wgset]

        # All layers, conditioned on image 0
        cap0_vars = set1['img0cap0lhs']
        cap1_vars = set1['img0cap1lhs']

        for i in range(len(cap0_vars)): # iterates over layers
            label0 = cap0_vars[i].cpu().numpy()
            label1 = cap1_vars[i].cpu().numpy()
            all_samples = np.concatenate((label0,label1),axis=0)
            all_labels = np.concatenate((np.zeros(len(label0)),np.ones(len(label1))),axis=0)
            if control:
                np.random.shuffle(all_labels)
            svc = SVC(kernel='linear',max_iter=100000,C=100)
            svc.fit(all_samples, all_labels)
            score = svc.score(all_samples, all_labels)
            margin = 2 / np.linalg.norm(svc.coef_)
            print(str(wgset)+','+str(control)+','+str(i)+',0,'+str(score)+','+str(margin),file=table)

        # All layers, conditioned on image 1
        cap0_vars = set1['img1cap0lhs']
        cap1_vars = set1['img1cap1lhs']

        for i in range(len(cap0_vars)): # iterates over layers
            label0 = cap0_vars[i].cpu().numpy()
            label1 = cap1_vars[i].cpu().numpy()
            all_samples = np.concatenate((label0,label1),axis=0)
            all_labels = np.concatenate((np.zeros(len(label0)),np.ones(len(label1))),axis=0)
            if control:
                np.random.shuffle(all_labels)
            svc = SVC(kernel='linear',max_iter=100000,C=100)
            svc.fit(all_samples, all_labels)
            score = svc.score(all_samples, all_labels)
            margin = 2 / np.linalg.norm(svc.coef_)
            print(str(wgset)+','+str(control)+','+str(i)+',1,'+str(score)+','+str(margin),file=table)

