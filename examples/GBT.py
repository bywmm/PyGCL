import torch
import os.path as osp
import GCL.loss as L
import GCL.augmentor as A
import torch_geometric.transforms as T

from tqdm import tqdm
from functools import partial
from torch.optim import Adam
from GCL.eval import from_PyG_split, LRTrainableEvaluator
from GCL.model.contrast_model import WithinEmbedContrast
from sklearn.metrics import f1_score
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import WikiCS
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GConv, self).__init__()
        self.act = torch.nn.PReLU()
        self.bn = torch.nn.BatchNorm1d(2 * hidden_dim, momentum=0.01)
        self.conv1 = GCNConv(input_dim, 2 * hidden_dim, cached=False)
        self.conv2 = GCNConv(2 * hidden_dim, hidden_dim, cached=False)

    def forward(self, x, edge_index, edge_weight=None):
        z = self.conv1(x, edge_index, edge_weight)
        z = self.bn(z)
        z = self.act(z)
        z = self.conv2(z, edge_index, edge_weight)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    _, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)
    loss = contrast_model(z1, z2)
    loss.backward()
    optimizer.step()
    return loss.item()


def eval(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)
    split = from_PyG_split(data)
    evaluator = LRTrainableEvaluator(
        input_dim=z.size(1), num_classes=data.y.max().item() + 1,
        metrics={'micro_f1': partial(f1_score, average='micro'), 'macro_f1': partial(f1_score, average='macro')},
        split=split, device=data.x.device, test_metric='micro_f1')
    return evaluator(z, data.y)


def main():
    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets', 'WikiCS')
    dataset = WikiCS(path, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])

    gconv = GConv(input_dim=dataset.num_features, hidden_dim=256).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = WithinEmbedContrast(loss=L.BarlowTwins()).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=5e-4)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=400,
        max_epochs=4000)

    with tqdm(total=4000, desc='(T)') as pbar:
        for epoch in range(1, 4001):
            loss = train(encoder_model, contrast_model, data, optimizer)
            scheduler.step()
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = eval(encoder_model, data)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]["mean"]:.4f}±{test_result["micro_f1"]["std"]:.4f},'
          f' F1Ma={test_result["macro_f1"]["mean"]:.4f}±{test_result["macro_f1"]["std"]:.4f}')


if __name__ == '__main__':
    main()
