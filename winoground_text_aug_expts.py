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
import string

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(args.dataset)

    add_wha = True if args.model == 'uniter' else False
    # Load winoground dataset
    if args.model == 'uniter' or args.model == 'lxmert':
        winoground_dataset = WinogroundDataset(dataset_path/"winoground_features.npy",
                                            dataset_path/"winoground_boxes.npy",
                                            dataset_path/"examples_augmented.jsonl",
                                            add_wha, device)
    else:
        winoground_dataset = WinogroundDatasetRawImages(dataset_path/"winoground_image_paths.txt",
                                                        dataset_path/"examples_augmented.jsonl",
                                                        device)
    # winoground_dataloader = DataLoader(winoground_dataset, batch_size=None, shuffle=False)


    text_scores = torch.zeros(len(winoground_dataset))
    image_scores = torch.zeros(len(winoground_dataset))
    group_scores = torch.zeros(len(winoground_dataset))

    accuracies = torch.zeros(len(winoground_dataset)*2)

    if args.model == 'uniter':
        import UNITER_codebase.uniter_code as Uniter
        model, tokenizer = Uniter.setup_model(device)
        get_matching_scores_batched = lambda image, captions: Uniter.get_matching_scores_batched(image, captions, model, tokenizer, device)
    elif args.model == 'lxmert':
        import LXMERT.lxmert_code as Lxmert
        model, tokenizer = Lxmert.setup_model(device)
        get_matching_scores_batched = lambda image, captions: Lxmert.get_matching_scores_batched(image, captions, model, tokenizer, device)
    elif args.model == 'clip':
        import CLIP.clip_code as Clip
        model, image_processor = Clip.setup_model(device)
        get_matching_scores_batched = lambda image, captions: Clip.get_matching_scores_batched(image, captions, model, image_processor, device)
    else:
        raise ValueError(f"Shouldn't reach here; Model {args.model} not supported")
    
    aggregate_function = torch.mean if args.aggregate_function == 'mean' else torch.max

    def aggregate_interpolated(scores):
         # Orig score
        orig_score = scores[0]
        # Aggregate score
        agg_score = aggregate_function(scores)
        # Interpolated score
        interpolated_score = (1.-args.interpolate_ratio)*orig_score + args.interpolate_ratio*agg_score
        return interpolated_score

    # Evaluate
    model.eval()
    with torch.no_grad():
        for i in range(len(winoground_dataset)):
            print(i, file=sys.stderr)
            image_0, caption_0_list, image_1, caption_1_list = winoground_dataset[i]

            captions_0 = [_["variant"] for _ in caption_0_list]
            captions_1 = [_["variant"] for _ in caption_1_list]

            img0_text0_scores = get_matching_scores_batched(image_0, captions_0)
            img0_text1_scores = get_matching_scores_batched(image_0, captions_1)
            img1_text0_scores = get_matching_scores_batched(image_1, captions_0)
            img1_text1_scores = get_matching_scores_batched(image_1, captions_1)

            # print(img0_text0_scores, file=sys.stderr)
            # print(img0_text1_scores, file=sys.stderr)
            # print(img1_text0_scores, file=sys.stderr)
            # print(img1_text1_scores, file=sys.stderr)
            # input()


            img0_text0_score = aggregate_interpolated(img0_text0_scores).item()
            img0_text1_score = aggregate_interpolated(img0_text1_scores).item()
            img1_text0_score = aggregate_interpolated(img1_text0_scores).item()
            img1_text1_score = aggregate_interpolated(img1_text1_scores).item()

            text_score = img0_text0_score > img0_text1_score and img1_text1_score > img1_text0_score
            image_score = img0_text0_score > img1_text0_score and img1_text1_score > img0_text1_score
            group_score = text_score and image_score
            
            text_scores[i] = int(text_score)
            image_scores[i] = int(image_score)
            group_scores[i] = int(group_score)

            accuracy_0 = img0_text0_score > 0.5
            accuracy_1 = img0_text1_score > 0.5
            accuracies[i*2] = int(accuracy_0)
            accuracies[i*2+1] = int(accuracy_1)

    print(f"Text score: {text_scores.mean():.4f}")
    print(f"Image score: {image_scores.mean():.4f}")
    print(f"Group score: {group_scores.mean():.4f}")
    print(f"Accuracy: {accuracies.mean():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model", type=str, required=True, choices=["uniter", "lxmert", "clip"])
    parser.add_argument("--dataset", default="dataset/", type=str)
    parser.add_argument("--aggregate_function", default="max", type=str, choices=["mean", "max"])
    parser.add_argument("--interpolate_ratio", default=1.0, type=float)

    args = parser.parse_args()

    main(args)