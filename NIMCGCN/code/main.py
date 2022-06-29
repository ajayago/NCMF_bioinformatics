from torch import nn, optim
from prepareData import prepare_data
from model import Model
from trainData import Dataset
import sys
import numpy as np

class Config(object):
    def __init__(self, sample_id):
        self.data_path = f'../data/PolyP3/{sample_id}'
        self.validation = 5
        self.save_path = f'../data/PolyP3/{sample_id}'
        self.epoch = 300
        self.alpha = 0.2


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, one_index, zero_index, target, input):
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input.transpose(0, 1), target)
        return (1-opt.alpha)*loss_sum[one_index].sum()+opt.alpha*loss_sum[zero_index].sum()


class Sizes(object):
    def __init__(self, dataset):
        self.m = dataset['mm']['data'].size(0)
        self.d = dataset['dd']['data'].size(0)
        self.fg = 256
        self.fd = 256
        self.k = 32


def train(model, train_data, optimizer, opt):
    model.train()
    regression_crit = Myloss()
    one_index = train_data[2][0].cuda().t().tolist()
    zero_index = train_data[2][1].cuda().t().tolist()

    def train_epoch():
        model.zero_grad()
        score = model(train_data)
        loss = regression_crit(one_index, zero_index, train_data[4].cuda(), score)
        loss.backward()
        optimizer.step()
        return loss
    for epoch in range(1, opt.epoch+1):
        train_reg_loss = train_epoch()
        print(train_reg_loss.item()/(len(one_index[0])+len(zero_index[0])))


opt = Config(sample_id=sys.argv[1])


def main():
    dataset = prepare_data(opt)
    sizes = Sizes(dataset)
    train_data = Dataset(opt, dataset)
    for i in range(opt.validation):
        print('-'*50)
        model = Model(sizes)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train(model, train_data[i], optimizer, opt)
    protein_repr, drug_repr = model.get_repr()
    print(drug_repr.size())
    print(protein_repr.size())
    np.save(f"../data/PolyP3/{sys.argv[1]}/protein_repr.npy", protein_repr.cpu().detach().numpy())
    np.save(f"../data/PolyP3/{sys.argv[1]}/drug_repr.npy", drug_repr.cpu().detach().numpy())

if __name__ == "__main__":
    main()
