import torch as th
import numpy as np
from torch.utils.data import DataLoader
from text_variant_matching_dataset import TextVariantMatchingDataset
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Configure Probe')
parser.add_argument('--epochs', type=int, default=7500, help='Number of epochs to train for')
parser.add_argument('--layers', type=int, default=3, help='Number of nn.Linear layers in the model')
parser.add_argument('--width', type=int, default=1024, help='Model hidden layer width')
parser.add_argument('--control', type=int, default=0, help='Whether to run the control task variant of this probe')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate to use while training')
parser.add_argument('--batch', type=int, default=128, help='Batch size to use while training')
parser.add_argument('--layer_idx', type=int, default=8, help='Which layer to probe over')
args = parser.parse_args()

if args.control:
    # Relative Indexes
    train_ctrl = [2, 3, 5, 9, 11, 13, 15, 18, 20, 21, 22, 23, 25, 26, 27, 28, 32, 34, 35, 36, 37, 40, 41, 42, 43, 45, 47, 50, 51, 52, 54, 58, 63, 65, 66, 67, 68, 69, 71, 73, 77, 78, 80, 82, 83, 86, 89, 93, 94, 96, 98, 100, 103, 104, 105, 107, 108, 110, 113, 114, 115, 116, 118, 120, 121, 127, 130, 131, 132, 133, 134, 135, 138, 139, 143, 146, 148, 149, 150, 151, 152, 160, 161, 165, 167, 168, 170, 172, 176, 178, 180, 181, 185, 189, 190, 195, 196, 197, 198, 199, 202, 207, 211, 212, 213, 215, 216, 219, 221, 223, 224, 226, 228, 229, 230, 232, 233, 234, 236, 238, 244, 245, 246, 248, 250, 252, 253, 255, 257, 260, 264, 266, 267, 268, 269, 270, 272, 276, 279, 280, 281, 283, 285, 286, 287, 289, 290, 291, 293, 294]
    test_ctrl = [2, 4, 6, 12, 17, 18, 21, 24, 25, 26, 27, 31, 32, 34, 35, 36, 38, 41, 43, 46, 47, 48, 49, 53, 54, 57, 58, 62, 64, 65, 66, 67, 72, 73, 75, 76, 77, 78, 79, 81, 82, 84, 87, 88, 89, 90, 93, 94, 95, 98]
    """
    Non-relative
    train_ctrl = [2, 3, 5, 9, 11, 14, 16, 20, 22, 23, 24, 25, 27, 28, 29, 30, 34, 39, 41, 43, 44, 47, 48, 49, 50, 52, 54, 59, 60, 61, 63, 69, 74, 76, 77, 78, 79, 80, 82, 84, 90, 93, 98, 104, 105, 108, 112, 118, 119, 121, 125, 127, 130, 131, 132, 135, 137, 139, 143, 145, 146, 148, 151, 155, 156, 163, 166, 167, 170, 173, 174, 175, 179, 180, 185, 191, 193, 195, 196, 197, 199, 208, 209, 217, 222, 224, 226, 228, 233, 235, 237, 239, 244, 248, 249, 255, 256, 258, 259, 260, 264, 270, 274, 276, 277, 279, 280, 285, 288, 290, 291, 294, 296, 297, 298, 301, 302, 304, 306, 308, 314, 315, 316, 320, 324, 326, 327, 329, 331, 334, 339, 341, 342, 343, 344, 346, 353, 358, 362, 363, 366, 369, 371, 374, 376, 382, 384, 385, 390, 391] 
    test_ctrl = [36, 38, 42, 89, 97, 99, 103, 115, 123, 124, 133, 147, 150, 153, 162, 168, 171, 181, 189, 198, 202, 211, 214, 219, 220, 238, 241, 265, 283, 284, 287, 293, 322, 323, 345, 347, 348, 349, 350, 355, 359, 365, 373, 375, 377, 379, 383, 387, 388, 393]
    """
else:
    train_ctrl = []
    test_ctrl = []

train_loader = DataLoader(TextVariantMatchingDataset('language_hidden_states_CLS_train.pkl', control_flip_pairs=train_ctrl, layer_idx=args.layer_idx), batch_size=args.batch, shuffle=True)
test_loader = DataLoader(TextVariantMatchingDataset('language_hidden_states_CLS_test.pkl', control_flip_pairs=test_ctrl, layer_idx=args.layer_idx), batch_size=args.batch, shuffle=True)

class SimpleMLP(th.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers):
        super(SimpleMLP, self).__init__()
        self.num_layers = num_layers
        if num_layers == 1:
            self.projection = th.nn.Linear(embedding_dim*3, 1)
        else:
            self.first_layer = th.nn.Linear(embedding_dim*3,hidden_dim)
            self.first_layer_activation = th.nn.ReLU()
            self.last_layer = th.nn.Linear(hidden_dim,1)
            self.middle_layers = []
            for i in range(num_layers-2):
                self.middle_layers.append(th.nn.Linear(hidden_dim, hidden_dim))
                self.middle_layers.append(th.nn.ReLU())
            self.middle_layers = th.nn.ModuleList(self.middle_layers)

    def forward(self, x):
        if self.num_layers == 1:
            return self.projection(x)
        else:
            x = self.first_layer(x)
            x = self.first_layer_activation(x)
            for layer in self.middle_layers:
                x = layer(x)
            return self.last_layer(x)

probe = SimpleMLP(768, args.width, args.layers)
probe.cuda()
loss = th.nn.BCEWithLogitsLoss()
opt = th.optim.Adam(probe.parameters(), lr=args.lr)

probe.train()
for i in range(args.epochs):
    epoch_loss = 0
    epoch_acc = 0
    c = 0
    for sample in train_loader:
        out = probe(th.cat((sample['ref'].cuda(), sample['opt1'].cuda(), sample['opt2'].cuda()),dim=1).float())
        l = loss(out.squeeze(), sample['label'].float().cuda())
        c += len(sample['label'])
        epoch_acc += th.sum((out.squeeze()>.5)==sample['label'].cuda())
        # print('Got loss', l)
        #print('From predicting', sample['label'])
        epoch_loss += l.item()
        opt.zero_grad()
        l.backward()
        opt.step()
    if i % 100 == -1:
        print('At epoch', i, 'training loss is', round(epoch_loss/c,4), 'and accuracy on the training set is', str(round((epoch_acc.item()/c)*100,2))+'%')
    if i % 1000 == -1:
        probe.eval()
        total_correct = 0
        total_seen = 0
        for sample in test_loader:
            out = probe(th.cat((sample['ref'].cuda(), sample['opt1'].cuda(), sample['opt2'].cuda()),dim=1).float())
            total_correct += th.sum((out.squeeze()>.5)==sample['label'].cuda())
            total_seen += len(sample['label'])
        print('Test Accuracy Check-In:', str(round((total_correct.item()/total_seen)*100,2))+'%')
        probe.train()

print('Final training set accuracy is', str(round((epoch_acc.item()/c)*100,2))+'%')

probe.eval()
total_correct = 0
total_seen = 0
for _ in range(100):
    for sample in test_loader:
        out = probe(th.cat((sample['ref'].cuda(), sample['opt1'].cuda(), sample['opt2'].cuda()),dim=1).float())
        total_correct += th.sum((out.squeeze()>.5)==sample['label'].cuda())
        total_seen += len(sample['label'])
print('got', total_correct.item(), 'right out of', total_seen, 'for a testing set accuracy of', str(round((total_correct.item()/total_seen)*100,2))+'%')
