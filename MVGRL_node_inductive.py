from sklearn.metrics import adjusted_rand_score
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch_geometric.transforms as T

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn.inits import uniform
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets import Actor
from torch_geometric.data import NeighborSampler


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(SAGEConv(input_dim, hidden_dim))
            else:
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            self.activations.append(nn.PReLU(hidden_dim))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.layers[i]((x, x_target), edge_index)
            x = self.activations[i](x)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, encoder1, encoder2, augmentor, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.augmentor = augmentor
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, _ = aug1(x, edge_index, edge_weight)
        x2, edge_index2, _ = aug2(x, edge_index, edge_weight)
        # x2, edge_index2 = x, edge_index
        z1 = self.encoder1(x1, edge_index1)
        z2 = self.encoder2(x2, edge_index2)
        g1 = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)))
        g2 = self.project(torch.sigmoid(z2.mean(dim=0, keepdim=True)))
        z1n = self.encoder1(*self.corruption(x1, edge_index1))
        z2n = self.encoder2(*self.corruption(x2, edge_index2))
        return z1, z2, g1, g2, z1n, z2n


def train(encoder_model, contrast_model, data, dataloader, optimizer):
    encoder_model.train()
    total_loss = total_examples = 0
    for batch_size, node_id, adjs in dataloader:
        adjs = [adj.to('cuda') for adj in adjs]
        optimizer.zero_grad()
        z1, z2, g1, g2, z1n, z2n = encoder_model(data.x[node_id], adjs)
        # loss = contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, h1n=z1n, h2n=z2n)
        loss = contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, h3=z1n, h4=z2n)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * z1.shape[0]
        total_examples += z1.shape[0]
    return total_loss / total_examples


def test(encoder_model, data, dataloader):
    encoder_model.eval()
    zs = []
    for i, (batch_size, node_id, adjs) in enumerate(dataloader):
        adjs = [adj.to('cuda') for adj in adjs]
        z1, z2, _, _, _, _ = encoder_model(data.x[node_id], adjs)
        z = z1 + z2
        zs.append(z)
        # zs.append(data.x[node_id][:batch_size])
    x = torch.cat(zs, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(x, data.y, split)
    return result


def main():
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    device = torch.device('cuda')
    # path = osp.join(osp.expanduser('~'), 'dataset/graph/Planetoid/')
    # dataset = Planetoid(path, name='Cora', transform=T.NormalizeFeatures())
    dataset = Actor(root='data/film', transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    train_loader = NeighborSampler(data.edge_index, node_idx=None,
                                   sizes=[10, 10, 25], batch_size=128,
                                   shuffle=True, num_workers=0)
    test_loader = NeighborSampler(data.edge_index, node_idx=None,
                                  sizes=[10, 10, 25], batch_size=128,
                                  shuffle=False, num_workers=0)

    aug1 = A.Identity()
    aug2 = A.PPRDiffusion(alpha=0.2)
    gconv1 = GConv(input_dim=dataset.num_features, hidden_dim=512, num_layers=3).to(device)
    gconv2 = GConv(input_dim=dataset.num_features, hidden_dim=512, num_layers=3).to(device)
    encoder_model = Encoder(encoder1=gconv1, encoder2=gconv2, augmentor=(aug1, aug2), hidden_dim=512).to(device)
    contrast_model = DualBranchContrast(loss=L.JSD(), mode='G2L').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.001)

    with tqdm(total=200, desc='(T)') as pbar:
        for epoch in range(1, 201):
            loss = train(encoder_model, contrast_model, data, train_loader, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, data, test_loader)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()
