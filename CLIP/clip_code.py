import clip
import torch

def setup_model(device):
    model, image_processor = clip.load("ViT-B/32", device=device)
    model.to(device)

    return model, image_processor

def get_matching_score(image, caption, model, image_processor, device):
    processed_image = image_processor(image).unsqueeze(0).to(device)
    processed_caption = clip.tokenize(caption).to(device)

    with torch.no_grad():
        # image_features = model.encode_image(processed_image)
        # text_features = model.encode_text(processed_caption)
        logits_per_image, logits_per_text = model(processed_image, processed_caption)
        print(logits_per_image.shape)
        probs = logits_per_image.softmax(dim=-1)[0]
    
    return probs

# def get_matching_score(image, caption, model, tokenizer, device):
#     tokenized = tokenizer(caption)
    
#     model_input = {}
#     # Form input to model
#     model_input['input_ids'] = torch.tensor(tokenized['input_ids'], dtype=torch.long)[None,:].to(device)
#     model_input['attention_mask'] = torch.tensor(tokenized['attention_mask'])[None,:].to(device)
#     model_input['token_type_ids'] = torch.tensor(tokenized['token_type_ids'], dtype=torch.long)[None,:].to(device)
    
#     model_input['visual_feats'] = image['features'][None,:]
#     model_input['visual_pos'] = image['boxes'][None,:]

#     output = model(**model_input)
#     itm_scores = output.cross_relationship_score[0] # Batch size 1 so [0]
#     itm_scores_prob = torch.nn.functional.softmax(itm_scores, dim=0)
#     return itm_scores_prob[1].item()

def get_matching_scores_batched(image, captions, model, image_processor, device):
    processed_image = image_processor(image).unsqueeze(0).to(device)
    processed_captions = clip.tokenize(captions).to(device)

    with torch.no_grad():
        # image_features = model.encode_image(processed_image)
        # text_features = model.encode_text(processed_captions)
        logits_per_image, logits_per_text = model(processed_image, processed_captions)
        probs = (logits_per_image/100)[0]
    
    return probs

# def run_inference_batched(image, captions, model, tokenizer, device):
#     tokenized = tokenizer(captions, padding=True)
    
#     model_input = {}
#     # Form input to model
#     model_input['input_ids'] = torch.tensor(tokenized['input_ids'], dtype=torch.long).to(device)
#     model_input['attention_mask'] = torch.tensor(tokenized['attention_mask']).to(device)
#     model_input['token_type_ids'] = torch.tensor(tokenized['token_type_ids'], dtype=torch.long).to(device)
     
#     model_input['visual_feats'] = image['features'].unsqueeze(0).repeat(model_input['input_ids'].size(0), 1, 1)
#     model_input['visual_pos'] = image['boxes'].unsqueeze(0).repeat(model_input['input_ids'].size(0), 1, 1)

#     output = model(**model_input, output_hidden_states=True)
    
#     return {"caption": tokenized,
#             "model_output": output}