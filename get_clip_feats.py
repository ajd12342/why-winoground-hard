import json
import torch as th
import pickle as pkl
from transformers import CLIPTokenizer, CLIPTextModel

winoground_path = '/saltpool0/data/layneberry/WinoGround/'

model = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32')
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')

WG = [json.loads(line) for line in open(winoground_path+'examples_augmented.jsonl').readlines()]

representations = []

for s in WG:
    c0_inputs = tokenizer([s['caption_0'][i]['variant'] for i in range(len(s['caption_0']))],padding=True, return_tensors='pt')
    c0_outputs = model(**c0_inputs,output_hidden_states=True)
    c0_eos_ids = th.sum(c0_inputs['attention_mask'],dim=1)-1
    c0_cls_by_layer = [c0_outputs['hidden_states'][i][th.arange(len(c0_outputs['hidden_states'][i])),c0_eos_ids] for i in range(len(c0_outputs['hidden_states']))]
    
    c1_inputs = tokenizer([s['caption_1'][i]['variant'] for i in range(len(s['caption_1']))],padding=True, return_tensors='pt')
    c1_outputs = model(**c1_inputs,output_hidden_states=True)
    c1_eos_ids = th.sum(c1_inputs['attention_mask'],dim=1)-1
    c1_cls_by_layer = [c1_outputs['hidden_states'][i][th.arange(len(c1_outputs['hidden_states'][i])),c1_eos_ids] for i in range(len(c1_outputs['hidden_states']))]

    representations.append({'cap0':c0_cls_by_layer,'cap1':c1_cls_by_layer})

pkl.dump(representations, open(winoground_path+'clip_variants_feats.pkl','wb'))
