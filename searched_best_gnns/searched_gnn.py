import torch
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork
import torch_geometric.transforms as T
from model_manager import evaluate
import torch.nn.functional as F


def main():
    model = torch.load(f'{data_name}.pth')
    if data_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root='../dataset/', name=data_name, transform=T.NormalizeFeatures())
        data = dataset[0]
    elif data_name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB('../dataset/', data_name)
        data = dataset[0]
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]
    elif data_name in ['Chameleon']:
        dataset = WikipediaNetwork('../dataset/', data_name)
        data = dataset[0]
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]
    else:
        raise Exception("dataset cannot found")
    model.cuda()
    data.cuda()
    model.eval()
    logits = model(data.x, data.edge_index)
    logits = F.log_softmax(logits, 1)
    test_acc = evaluate(logits, data.y, data.test_mask)
    print(f'dataset: {data_name}, test score: {test_acc}')


if __name__ == '__main__':
    data_name = 'Cora'  # Cora, CiteSeer, PubMed, Cornell, Texas, Wisconsin
    main()
