"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

run inference for Image Text Retrieval
"""
import argparse
import sys

import torch
from dataset.dataset import WinogroundDataset, WinogroundDatasetRawImages
from pathlib import Path

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(args.dataset)

    add_wha = True if args.model == 'uniter' else False
    # Load winoground dataset
    if args.model == 'uniter' or args.model == 'lxmert':
        winoground_dataset = WinogroundDataset(dataset_path/"winoground_features.npy",
                                                dataset_path/"winoground_boxes.npy",
                                                dataset_path/"examples.jsonl",
                                                add_wha, device)
    else:
        winoground_dataset = WinogroundDatasetRawImages(dataset_path/"winoground_image_paths.txt",
                                                        dataset_path/"examples.jsonl",
                                                        device)
    # winoground_dataloader = DataLoader(winoground_dataset, batch_size=None, shuffle=False)


    text_scores = torch.zeros(len(winoground_dataset))
    image_scores = torch.zeros(len(winoground_dataset))
    group_scores = torch.zeros(len(winoground_dataset))

    matching_scores = torch.zeros(len(winoground_dataset))
    accuracies = torch.zeros(len(winoground_dataset)*2)

    if args.model == 'uniter':
        import UNITER_codebase.uniter_code as Uniter
        model, tokenizer = Uniter.setup_model(device)
        get_matching_score = lambda image, caption: Uniter.get_matching_score(image, caption, model, tokenizer, device)
    elif args.model == 'lxmert':
        import LXMERT.lxmert_code as Lxmert
        model, tokenizer = Lxmert.setup_model(device)
        get_matching_score = lambda image, caption: Lxmert.get_matching_score(image, caption, model, tokenizer, device)
    elif args.model == 'clip':
        import CLIP.clip_code as Clip
        model, image_processor = Clip.setup_model(device)
        get_matching_score = lambda image, caption: Clip.get_matching_score(image, caption, model, image_processor, device)
    else:
        raise ValueError(f"Shouldn't reach here; Model {args.model} not supported")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        for i in range(len(winoground_dataset)):
            print(i, file=sys.stderr)
            image_0, caption_0, image_1, caption_1 = winoground_dataset[i]

            img0_text0_score = get_matching_score(image_0, caption_0)
            img0_text1_score = get_matching_score(image_0, caption_1)
            img1_text0_score = get_matching_score(image_1, caption_0)
            img1_text1_score = get_matching_score(image_1, caption_1)

            text_score = img0_text0_score > img0_text1_score and img1_text1_score > img1_text0_score
            image_score = img0_text0_score > img1_text0_score and img1_text1_score > img0_text1_score
            group_score = text_score and image_score

            matching_score = img0_text0_score*img1_text1_score > img0_text1_score*img1_text0_score
            
            text_scores[i] = int(text_score)
            image_scores[i] = int(image_score)
            group_scores[i] = int(group_score)

            matching_scores[i] = int(matching_score)

            accuracy_0 = img0_text0_score > 0.5
            accuracy_1 = img0_text1_score > 0.5
            accuracies[i*2] = int(accuracy_0)
            accuracies[i*2+1] = int(accuracy_1)

    # print(f"Text score: {text_scores.mean():.4f}")
    # print(f"Image score: {image_scores.mean():.4f}")
    # print(f"Group score: {group_scores.mean():.4f}")
    print(f"Matching score: {matching_scores.mean():.4f}")
    # print(f"Accuracy: {accuracies.mean():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model", type=str, required=True, choices=["uniter", "lxmert", "clip"])
    parser.add_argument("--dataset", default="dataset/", type=str)

    args = parser.parse_args()

    main(args)