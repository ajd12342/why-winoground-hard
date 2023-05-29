from transformers import LxmertTokenizer, LxmertModel, LxmertConfig, LxmertForPreTraining
import torch as th
import numpy as np
import random
import pickle as pkl
from tqdm import tqdm

winoground_path = '/saltpool0/data/layneberry/WinoGround/'

tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')
model = LxmertForPreTraining.from_pretrained('unc-nlp/lxmert-base-uncased').cuda()
s = th.nn.Softmax(dim=1).cuda()

test_feats = np.load(winoground_path+'winoground_features.npy')
test_boxes = np.load(winoground_path+'winoground_boxes.npy')

raw = [json.loads(line) for line in open(winoground_path+'examples.jsonl')]
test_texts = []
for e in raw:
    test_texts.append(e['caption_0'])
    test_texts.append(e['caption_1'])

with th.no_grad():
    full_scores_matrix = th.empty((0,800)).cuda()

    for cap in tqdm(test_texts,total=len(test_texts)):
        cap_repeated = [cap for _ in range(len(test_feats))]
        tokenized_input = tokenizer(cap_repeated, return_tensors='pt', padding='longest')
        for k in tokenized_input.keys():
            tokenized_input[k] = tokenized_input[k].cuda()
        tokenized_input['visual_feats'] = th.Tensor(test_feats).cuda()
        tokenized_input['visual_pos'] = th.Tensor(test_boxes).cuda()
        out = model(**tokenized_input, output_hidden_states=True)
        scores = s(out.cross_relationship_score)[:,1]
        full_scores_matrix = th.cat((full_scores_matrix,scores.unsqueeze(0)),dim=0)

pkl.dump(full_scores_matrix,open(winoground_path+'lxmert_scores.pkl','wb'))
