import torch
from torch.utils.data import Dataset
import numpy as np
import json
from PIL import Image

class WinogroundDataset(Dataset):
    def __init__(self, winoground_rcnn_features_path, winoground_rcnn_boxes_path, examples_json_path, add_wha, device):
        # Load the data into GPU since it's super small
        self.winoground_rcnn_features = torch.tensor(np.load(winoground_rcnn_features_path), dtype=torch.float).to(device)

        self.winoground_rcnn_boxes = torch.tensor(np.load(winoground_rcnn_boxes_path), dtype=torch.float).to(device)
        
        if add_wha:
            # Add width, height and area to winoground_boxes
            self.winoground_rcnn_boxes = self._add_wha(self.winoground_rcnn_boxes)

        self.winoground_examples = self._load_jsonl_as_list(examples_json_path)

    def _add_wha(self, boxes):
        # Add width, height and area to Winoground boxes
        new_box_features = torch.zeros((boxes.shape[0], boxes.shape[1], 7), dtype=torch.float).to(boxes.device)
        new_box_features[:, :, :4] = boxes
        # width
        new_box_features[:, :, 4] = new_box_features[:, :, 2] - new_box_features[:, :, 0]
        # height
        new_box_features[:, :, 5] = new_box_features[:, :, 3] - new_box_features[:, :, 1]
        # area
        new_box_features[:, :, 6] = new_box_features[:, :, 4] * new_box_features[:, :, 5]
        return new_box_features
    
    def _load_jsonl_as_list(self, path):
        with open(path, 'r') as f:
            return [json.loads(line.strip()) for line in f]
    
    def __len__(self):
        return len(self.winoground_examples)

    def __getitem__(self, idx):
        # idx is the index that corresponds to each (I_0, T_0, I_1, T_1) tuple
        image_0_idx = idx*2
        image_1_idx = idx*2 + 1

        image_0 = {"features": self.winoground_rcnn_features[image_0_idx],
                    "boxes": self.winoground_rcnn_boxes[image_0_idx]}
        
        image_1 = {"features": self.winoground_rcnn_features[image_1_idx],
                    "boxes": self.winoground_rcnn_boxes[image_1_idx]}

        example = self.winoground_examples[idx]
        caption_0 = example['caption_0']
        caption_1 = example['caption_1']

        return image_0, caption_0, image_1, caption_1

class WinogroundDatasetRawImages(Dataset):
    def __init__(self, winoground_image_paths, examples_json_path, device):
        # Load the data
        with open(winoground_image_paths, 'r') as f:
            self.winoground_images = [Image.open(path.strip()) for path in f]

        self.winoground_examples = self._load_jsonl_as_list(examples_json_path)
    
    def _load_jsonl_as_list(self, path):
        with open(path, 'r') as f:
            return [json.loads(line.strip()) for line in f]
    
    def __len__(self):
        return len(self.winoground_examples)

    def __getitem__(self, idx):
        # idx is the index that corresponds to each (I_0, T_0, I_1, T_1) tuple
        image_0_idx = idx*2
        image_1_idx = idx*2 + 1

        image_0 = self.winoground_images[image_0_idx]
        
        image_1 = self.winoground_images[image_1_idx]

        example = self.winoground_examples[idx]
        caption_0 = example['caption_0']
        caption_1 = example['caption_1']

        return image_0, caption_0, image_1, caption_1