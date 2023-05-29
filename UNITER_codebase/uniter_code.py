import argparse

import torch
from UNITER.model.pretrain import UniterForPretraining
from UNITER.utils.const import IMG_DIM, IMG_LABEL_DIM

from transformers import BertTokenizer
from UNITER.data.data import get_gather_index

def setup_model(device, checkpoint = "/slash/data3/scratch/anujd/uniter_download/pretrained/uniter-base.pt",
                model_config = "UNITER/config/uniter-base.json"):
    # Prepare model
    checkpoint = torch.load(checkpoint)
    model = UniterForPretraining.from_pretrained(model_config, checkpoint, img_dim=IMG_DIM, img_label_dim = IMG_LABEL_DIM)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    # if 'rank_output' not in checkpoint:
    #     model.init_output()  # zero shot setting

    model.to(device)

    return model, tokenizer

def get_matching_score(image, caption, model, tokenizer, device):

    # Form input to model
    tokenized = tokenizer(caption)
    input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)[None,:].to(device)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0).to(device)

    img_feat = image["features"][None,:]
    img_pos_feat = image["boxes"][None,:]
    attention_mask = torch.ones(input_ids.size(0), input_ids.size(1) + img_feat.size(1), dtype=torch.long).to(device)

    # Make gather_index. Taken from data/mlm.py function mlm_collate
    txt_lens = [i.size(0) for i in input_ids]
    num_bbs = [f.size(0) for f in img_feat]
    bs, max_tl = input_ids.size()
    out_size = attention_mask.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size).to(device)
    # print(txt_lens,num_bbs)

    # TODO: 0's padded so that (L + num_bb) % 8 == 0. See mlm.py MlmDataset __getitem__. Not implementing for now
    sequence_output = model.uniter(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    output_all_encoded_layers=False)
    pooled_output = model.uniter.pooler(sequence_output)
    itm_scores = model.itm_output(pooled_output)[0] # Just first elem of batch since batch size is 1
    itm_scores_prob = torch.nn.functional.softmax(itm_scores, dim=0)
    return itm_scores_prob[1].item()


def get_matching_scores_batched(image, captions, model, tokenizer, device):

    # Form input to model
    tokenized = tokenizer(captions, padding=True)
    input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long).to(device)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0).repeat(input_ids.size(0),1).to(device)

    img_feat = image["features"].unsqueeze(0).repeat(input_ids.size(0),1,1)
    img_pos_feat = image["boxes"].unsqueeze(0).repeat(input_ids.size(0),1,1)

    attention_mask_text = torch.tensor(tokenized["attention_mask"], dtype=torch.long).to(device)
    attention_mask_img = torch.ones(input_ids.size(0), img_feat.size(1), dtype=torch.long).to(device)

    attention_mask = torch.cat([attention_mask_text, attention_mask_img], dim=1)

    # Make gather_index. Taken from data/mlm.py function mlm_collate
    txt_lens = [i.size(0) for i in input_ids] # Not sure if I should change this to only have non-padded lengths
    num_bbs = [f.size(0) for f in img_feat]
    bs, max_tl = input_ids.size()
    out_size = attention_mask.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size).to(device)
    # print(txt_lens,num_bbs)

    # TODO: 0's padded so that (L + num_bb) % 8 == 0. See mlm.py MlmDataset __getitem__. Not implementing for now
    sequence_output = model.uniter(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    output_all_encoded_layers=False)
    pooled_output = model.uniter.pooler(sequence_output)
    itm_scores = model.itm_output(pooled_output)
    itm_scores_prob = torch.nn.functional.softmax(itm_scores, dim=1)
    return itm_scores_prob[:,1]

def run_inference(image, caption, model, tokenizer, device):
    # Form input to model
    tokenized = tokenizer(caption)
    input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)[None,:].to(device)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0).to(device)

    img_feat = image["features"][None,:]
    img_pos_feat = image["boxes"][None,:]
    attention_mask = torch.ones(input_ids.size(0), input_ids.size(1) + img_feat.size(1), dtype=torch.long).to(device)

    # Make gather_index. Taken from data/mlm.py function mlm_collate
    txt_lens = [i.size(0) for i in input_ids]
    num_bbs = [f.size(0) for f in img_feat]
    bs, max_tl = input_ids.size()
    out_size = attention_mask.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size).to(device)
    # print(txt_lens,num_bbs)

    # TODO: 0's padded so that (L + num_bb) % 8 == 0. See mlm.py MlmDataset __getitem__. Not implementing for now
    sequence_output = model.uniter(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    output_all_encoded_layers=True)
    pooled_output = model.uniter.pooler(sequence_output[-1])[0]

    language_hidden_states = tuple(layer_output[0][:txt_lens[0]] for layer_output in sequence_output)
    vision_hidden_states = tuple(layer_output[0][txt_lens[0]:] for layer_output in sequence_output)

    return {"caption": tokenized,
            "model_output": {"vision_hidden_states": vision_hidden_states, 
                            "language_hidden_states": language_hidden_states,
                            "pooled_output": pooled_output}
            }

def run_inference_batched(image, captions, model, tokenizer, device):
    tokenized = tokenizer(captions, padding=True)
    # print(tokenized)
    # print(len(captions))
    input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long).to(device)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0).repeat(len(captions), 1).to(device)
    img_feat = image["features"].unsqueeze(0).repeat(len(captions),1,1)
    img_pos_feat = image["boxes"].unsqueeze(0).repeat(len(captions),1,1)

    attention_mask_text = torch.tensor(tokenized["attention_mask"], dtype=torch.long).to(device)
    attention_mask_image = torch.ones(input_ids.size(0), img_feat.size(1), dtype=torch.long).to(device)
    attention_mask = torch.cat([attention_mask_text, attention_mask_image], dim=1)

    # Make gather_index. Taken from data/mlm.py function mlm_collate
    txt_lens = [i.size(0) for i in input_ids]
    num_bbs = [f.size(0) for f in img_feat]
    bs, max_tl = input_ids.size()
    out_size = attention_mask.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size).to(device)
    # print(txt_lens,num_bbs)

    # TODO: 0's padded so that (L + num_bb) % 8 == 0. See mlm.py MlmDataset __getitem__. Not implementing for now
    sequence_output = model.uniter(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    output_all_encoded_layers=True)
    pooled_output = model.uniter.pooler(sequence_output[-1])

    # language_hidden_states = tuple(layer_output[:, :txt_lens[0]] for layer_output in sequence_output)
    cls_hidden_states = tuple(layer_output[:, 0] for layer_output in sequence_output)
    # vision_hidden_states = tuple(layer_output[:, txt_lens[0]:] for layer_output in sequence_output)

    # return {"caption": tokenized,
    #         "model_output": {"vision_hidden_states": vision_hidden_states, 
    #                         "language_hidden_states": language_hidden_states,
    #                         "pooled_output": pooled_output}
    #         }
    return {"caption": tokenized,
            "model_output": {"cls_hidden_states": cls_hidden_states,
                            "pooled_output": pooled_output}
            }

# def main(opts):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Prepare model
#     checkpoint = torch.load(opts.checkpoint)
#     model = UniterForPretraining.from_pretrained(
#         opts.model_config, checkpoint, img_dim=IMG_DIM, img_label_dim = IMG_LABEL_DIM)
    
#     tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
#     # if 'rank_output' not in checkpoint:
#     #     model.init_output()  # zero shot setting

#     model.to(device)

#     # Run model on a given (text,image) pair
#     txt = "A blue cap"
#     img_path = "dataset/images/ex_0_img_0.png"
#     # Use img_id for now, replace using mapping from id to img_path later
#     img_id = 0



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     # Required parameters
#     parser.add_argument("--checkpoint", default="/slash/data3/scratch/anujd/uniter_download/pretrained/uniter-base.pt", type=str,
#                         help="model checkpoint binary")
#     parser.add_argument("--model_config", default="UNITER/config/uniter-base.json", type=str,
#                         help="model config json")
#     parser.add_argument("--dataset", default="dataset/", type=str)

#     args = parser.parse_args()

#     main(args)
