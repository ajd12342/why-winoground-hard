import pickle as pkl
import numpy as np
from sklearn.svm import SVC

table = open('CLS_separability_probe_LXMERT.csv', 'w')
print('Winoground Set,Control?,Layer,Image,Score,Margin',file=table)

feats = pkl.load(open('LXMERT_variants_feats.pkl','rb'))

for wgset in range(len(feats)):
    for control in [False,True]:
        set1 = feats[wgset]

        # only first 9 are pre-fusion
        cap0_vars = set1['img0cap0lhs'][:9]
        cap1_vars = set1['img0cap1lhs'][:9]

        for i in range(9):
            label0 = cap0_vars[i][:,0].cpu().numpy()
            label1 = cap1_vars[i][:,0].cpu().numpy()
            all_samples = np.concatenate((label0,label1),axis=0)
            all_labels = np.concatenate((np.zeros(len(label0)),np.ones(len(label1))),axis=0)
            if control:
                np.random.shuffle(all_labels)
            svc = SVC(kernel='linear',max_iter=100000,C=100)
            svc.fit(all_samples, all_labels)
            score = svc.score(all_samples, all_labels)
            margin = 2 / np.linalg.norm(svc.coef_)
            print(str(wgset)+','+str(control)+','+str(i)+',N/A,'+str(score)+','+str(margin),file=table)

        # Remaining 5, conditioned on image 0
        cap0_vars = set1['img0cap0lhs'][9:]
        cap1_vars = set1['img0cap1lhs'][9:]

        for i in range(len(cap0_vars)):
            label0 = cap0_vars[i][:,0].cpu().numpy()
            label1 = cap1_vars[i][:,0].cpu().numpy()
            all_samples = np.concatenate((label0,label1),axis=0)
            all_labels = np.concatenate((np.zeros(len(label0)),np.ones(len(label1))),axis=0)
            if control:
                np.random.shuffle(all_labels)
            svc = SVC(kernel='linear',max_iter=100000,C=100)
            svc.fit(all_samples, all_labels)
            score = svc.score(all_samples, all_labels)
            margin = 2 / np.linalg.norm(svc.coef_)
            print(str(wgset)+','+str(control)+','+str(i)+',0,'+str(score)+','+str(margin),file=table)

        # Remaining 5, conditioned on image 1
        cap0_vars = set1['img1cap0lhs'][9:]
        cap1_vars = set1['img1cap1lhs'][9:]

        for i in range(len(cap0_vars)):
            label0 = cap0_vars[i][:,0].cpu().numpy()
            label1 = cap1_vars[i][:,0].cpu().numpy()
            all_samples = np.concatenate((label0,label1),axis=0)
            all_labels = np.concatenate((np.zeros(len(label0)),np.ones(len(label1))),axis=0)
            if control:
                np.random.shuffle(all_labels)
            svc = SVC(kernel='linear',max_iter=100000,C=100)
            svc.fit(all_samples, all_labels)
            score = svc.score(all_samples, all_labels)
            margin = 2 / np.linalg.norm(svc.coef_)
            print(str(wgset)+','+str(control)+','+str(i)+',1,'+str(score)+','+str(margin),file=table)
