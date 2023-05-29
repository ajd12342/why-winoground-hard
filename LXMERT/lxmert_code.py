from transformers import LxmertTokenizer, LxmertForPreTraining, LxmertModel
import torch
import numpy as np

def setup_model(device):
    tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')
    model = LxmertForPreTraining.from_pretrained('unc-nlp/lxmert-base-uncased')
    model.to(device)

    return model, tokenizer

def setup_model_inference(device):
    tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')
    model = LxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased')
    model.to(device)

    return model, tokenizer

def get_matching_score(image, caption, model, tokenizer, device):
    tokenized = tokenizer(caption)
    
    model_input = {}
    # Form input to model
    model_input['input_ids'] = torch.tensor(tokenized['input_ids'], dtype=torch.long)[None,:].to(device)
    model_input['attention_mask'] = torch.tensor(tokenized['attention_mask'])[None,:].to(device)
    model_input['token_type_ids'] = torch.tensor(tokenized['token_type_ids'], dtype=torch.long)[None,:].to(device)
    
    model_input['visual_feats'] = image['features'][None,:]
    model_input['visual_pos'] = image['boxes'][None,:]

    output = model(**model_input)
    itm_scores = output.cross_relationship_score[0] # Batch size 1 so [0]
    itm_scores_prob = torch.nn.functional.softmax(itm_scores, dim=0)
    return itm_scores_prob[1].item()

def get_matching_scores_batched(image, captions, model, tokenizer, device):
    tokenized = tokenizer(captions, padding=True)
    
    model_input = {}
    # Form input to model
    model_input['input_ids'] = torch.tensor(tokenized['input_ids'], dtype=torch.long).to(device)
    model_input['attention_mask'] = torch.tensor(tokenized['attention_mask']).to(device)
    model_input['token_type_ids'] = torch.tensor(tokenized['token_type_ids'], dtype=torch.long).to(device)
    
    model_input['visual_feats'] = image['features'].unsqueeze(0).repeat(model_input['input_ids'].size(0), 1, 1)
    model_input['visual_pos'] = image['boxes'].unsqueeze(0).repeat(model_input['input_ids'].size(0), 1, 1)

    output = model(**model_input)
    itm_scores = output.cross_relationship_score
    itm_scores_prob = torch.nn.functional.softmax(itm_scores, dim=1)
    return itm_scores_prob[:,1]

def run_inference_batched(image, captions, model, tokenizer, device):
    tokenized = tokenizer(captions, padding=True)
    
    model_input = {}
    # Form input to model
    model_input['input_ids'] = torch.tensor(tokenized['input_ids'], dtype=torch.long).to(device)
    model_input['attention_mask'] = torch.tensor(tokenized['attention_mask']).to(device)
    model_input['token_type_ids'] = torch.tensor(tokenized['token_type_ids'], dtype=torch.long).to(device)
     
    model_input['visual_feats'] = image['features'].unsqueeze(0).repeat(model_input['input_ids'].size(0), 1, 1)
    model_input['visual_pos'] = image['boxes'].unsqueeze(0).repeat(model_input['input_ids'].size(0), 1, 1)

    output = model(**model_input, output_hidden_states=True)
    
    return {"caption": tokenized,
            "model_output": output}