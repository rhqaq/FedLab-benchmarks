from json import load
import os
import argparse
import random
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms
from torch import nn
import numpy as np
import sys
import torch
from getDataset import GetWearDataSet

sys.path.append("../../")
torch.manual_seed(0)

from fedlab.core.client.serial_trainer import SubsetSerialTrainer
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate
from fedlab.utils.functional import get_best_gpu, load_dict

# configuration
seed_value = 2022  # 设定随机数种子

np.random.seed(seed_value)
random.seed(seed_value)

parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--total_client", type=int, default=100)
parser.add_argument("--com_round", type=int, default=1000)

parser.add_argument("--sample_ratio", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.05)
parser.add_argument("--cuda", type=bool, default=True)

args = parser.parse_args()


class SelfAttentionEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # (N, L, D)
        a = self.attn(x)  # (N, L, 1)
        x = (x * a).sum(dim=1)  # (N, D)
        return x


class LSTM(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=100,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.attn = SelfAttentionEncoder(100)
        self.fc = nn.Linear(100, num_classes)

    def forward(self, inputs):
        # (N,L,D) batch,时序长度,特征数量
        tensor = self.LSTM(inputs)[0]  # (N,L,D)
        tensor = self.attn(tensor)  # (N,D)
        tensor = self.fc(tensor)
        return tensor
# torch model
class MLP(nn.Module):

    def __init__(self, input_size=784, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

data_indices = load_dict("mnist_partition.pkl")
print(data_indices)
# get mnist dataset
# root = "../../tests/data/mnist/"
# trainset = torchvision.datasets.MNIST(root=root,
#                                       train=True,
#                                       download=True,
#                                       transform=transforms.ToTensor())
# # print(trainset.size())
# testset = torchvision.datasets.MNIST(root=root,
#                                      train=False,
#                                      download=True,
#                                      transform=transforms.ToTensor())

DSADataset = GetWearDataSet(15,100,3,5)
trainset = DSADataset.trainset
# print(trainset.size())
testset = DSADataset.testset

test_loader = torch.utils.data.DataLoader(testset,
                                          batch_size=len(testset),
                                          drop_last=False,
                                          shuffle=False)

# setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# data_indices = load_dict("mnist_partition.pkl")
data_indices = load_dict("dsa15_partition.pkl")
print(data_indices)
if args.cuda:
    gpu = get_best_gpu()
    model = LSTM(45,15).cuda(gpu)
else:
    model = LSTM(45,15)

# FL settings
num_per_round = int(args.total_client * args.sample_ratio)
aggregator = Aggregators.fedavg_aggregate
total_client_num = args.total_client  # client总数


# fedlab setup
trainer = SubsetSerialTrainer(model=model,
                              dataset=trainset,
                              data_slices=data_indices,
                              args={
                                  "batch_size": args.batch_size,
                                  "epochs": args.epochs,
                                  "lr": args.lr
                              })

# train procedure
to_select = [i for i in range(total_client_num)]
acc_list = []
for round in range(args.com_round):
    model_parameters = SerializationTool.serialize_model(model)
    selection = random.sample(to_select, num_per_round)
    parameters_list = trainer.local_process(payload=[model_parameters],
                                            id_list=selection)

    SerializationTool.deserialize_model(model, aggregator(parameters_list))

    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, criterion, test_loader)
    acc_list.append(acc)
    print("loss: {:.4f}, acc: {:.2f}".format(loss, acc))

np.save('an15_E{}_B{}_lr{}_cf{}.npy'.format(args.epochs,args.batch_size,args.lr,args.sample_ratio),acc_list)