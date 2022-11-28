import torch as th
import numpy as np
from torch.utils.data import DataLoader
from embedding_discrimination_dataset import EmbeddingDiscriminationDataset
import argparse

parser = argparse.ArgumentParser(description='Configure Probe')
parser.add_argument('--train_path', type=str, default='../dataset/LXMERT_train.pkl', help='Path to the stored training set model outputs to probe over')
parser.add_argument('--test_path', type=str, default='../dataset/LXMERT_test.pkl', help='Path to the stored testing set model outputs to probe over')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train for')
parser.add_argument('--layers', type=int, default=4, help='Number of nn.Linear layers in the model')
parser.add_argument('--width', type=int, default=1024, help='Model hidden layer width')
parser.add_argument('--control', type=int, default=1, help='Whether to run the control task variant of this probe')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate to use while training')
parser.add_argument('--batch', type=int, default=200, help='Batch size to use while training')
parser.add_argument('--score', type=str, default='image', help='Whether to probe analog of text score or image score')
parser.add_argument('--embd_type', type=str, default='pooled', help='Which embedding to use as the representation to probe over')
parser.add_argument('--img_layer', type=int, default=9, help='Which layer of the image encoder to get the embedding from')
parser.add_argument('--cap_layer', type=int, default=13, help='Which layer of the caption encoder to get the embedding from')
args = parser.parse_args()

if args.control:
    # Indexes of control pairs relative to each set
    train_ctrl = [2, 3, 5, 9, 11, 13, 15, 18, 20, 21, 22, 23, 25, 26, 27, 28, 32, 34, 35, 36, 37, 40, 41, 42, 43, 45, 47, 50, 51, 52, 54, 58, 63, 65, 66, 67, 68, 69, 71, 73, 77, 78, 80, 82, 83, 86, 89, 93, 94, 96, 98, 100, 103, 104, 105, 107, 108, 110, 113, 114, 115, 116, 118, 120, 121, 127, 130, 131, 132, 133, 134, 135, 138, 139, 143, 146, 148, 149, 150, 151, 152, 160, 161, 165, 167, 168, 170, 172, 176, 178, 180, 181, 185, 189, 190, 195, 196, 197, 198, 199, 202, 207, 211, 212, 213, 215, 216, 219, 221, 223, 224, 226, 228, 229, 230, 232, 233, 234, 236, 238, 244, 245, 246, 248, 250, 252, 253, 255, 257, 260, 264, 266, 267, 268, 269, 270, 272, 276, 279, 280, 281, 283, 285, 286, 287, 289, 290, 291, 293, 294]
    test_ctrl = [2, 4, 6, 12, 17, 18, 21, 24, 25, 26, 27, 31, 32, 34, 35, 36, 38, 41, 43, 46, 47, 48, 49, 53, 54, 57, 58, 62, 64, 65, 66, 67, 72, 73, 75, 76, 77, 78, 79, 81, 82, 84, 87, 88, 89, 90, 93, 94, 95, 98]
else:
    train_ctrl = []
    test_ctrl = []

train_loader = DataLoader(EmbeddingDiscriminationDataset(args.train_path, args.embd_type, img_layer=args.img_layer, cap_layer=args.cap_layer, score_type=args.score, control_flip_pairs=train_ctrl), batch_size=args.batch, shuffle=True)
test_loader = DataLoader(EmbeddingDiscriminationDataset(args.test_path, args.embd_type, img_layer=args.img_layer, cap_layer=args.cap_layer, score_type=args.score, control_flip_pairs=test_ctrl), batch_size=args.batch, shuffle=True)

class SimpleMLP(th.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers):
        super(SimpleMLP, self).__init__()
        self.num_layers = num_layers
        if num_layers == 1:
            self.projection = th.nn.Linear(embedding_dim*2, 1)
        else:
            self.first_layer = th.nn.Linear(embedding_dim*2,hidden_dim)
            self.first_layer_activation = th.nn.ReLU()
            self.last_layer = th.nn.Linear(hidden_dim,1)
            self.middle_layers = []
            for i in range(num_layers-2):
                self.middle_layers.append(th.nn.Linear(hidden_dim, hidden_dim).cuda())
                self.middle_layers.append(th.nn.ReLU().cuda())
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

if args.embd_type == 'pooled' or args.embd_type == 'CLS':
    probe = SimpleMLP(train_loader.dataset.embd_size(), args.width, args.layers).cuda()
elif args.embd_type == 'mean' or args.embd_type == 'max':
    probe = SimpleMLP(train_loader.dataset.embd_size()*2, args.width, args.layers).cuda()
loss = th.nn.BCEWithLogitsLoss()
opt = th.optim.Adam(probe.parameters(), lr=args.lr)

probe.train()
for i in range(args.epochs):
    epoch_loss = 0
    epoch_acc = 0
    c = 0
    for sample in train_loader:
        out = probe(th.cat((sample['opt1'].cuda(), sample['opt2'].cuda()),dim=1).float())
        l = loss(out.squeeze(), sample['label'].cuda().float())
        c += len(sample['label'])
        epoch_acc += th.sum((out.squeeze()>.5)==sample['label'].cuda())
        epoch_loss += l.item()
        opt.zero_grad()
        l.backward()
        opt.step()

print('Final training set accuracy is', str(round((epoch_acc.item()/c)*100,2))+'%')

probe.eval()
total_correct = 0
total_seen = 0
for sample in test_loader:
    out = th.sigmoid(probe(th.cat((sample['opt1'].cuda(), sample['opt2'].cuda()),dim=1).float()))
    total_correct += th.sum((out.squeeze()>.5)==sample['label'].cuda())
    total_seen += len(sample['label'])
print('got', total_correct.item(), 'right out of', total_seen, 'for a testing set accuracy of', str(round((total_correct.item()/total_seen)*100,2))+'%')
