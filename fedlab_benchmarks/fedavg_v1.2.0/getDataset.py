import numpy as np
import os
import torch
from torch.utils.data import TensorDataset
from fedlab.utils.functional import save_dict

class GetWearDataSet(object):
    def __init__(self, action_num, client_num, shard_num, divide_num):
        # action_num:选取的动作数量
        # client_num:客户机数量
        # shard_num:每个客户机分到的数据label数量
        # divide_num:txt文件中125行数据(5s)要分成几份
        self.action_num = action_num
        self.client_num = client_num
        self.shard_num = shard_num
        self.divide_num = divide_num
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None
        self.train_data_combine, self.valid_data_combine = [0]*5, [0]*5
        self.train_label_combine, self.valid_label_combine = [0]*5, [0]*5
        self.trainset = None
        self.testset = None
        self.DataSetConstruct()

    def DataSetConstruct(self):
        data_dir = r'../data'
        # action_index = [ i+1 for i in np.random.permutation(19)][:self.action_num]
        action_index = [i + 1 for i in range(self.action_num)]
        file_paths = []
        for file in os.listdir(data_dir):
            if int(file[1:]) in action_index:
                file_paths.append(os.path.join(data_dir, file))
        # print(file_paths)

        # 初始化数据

        all_data = [0] * self.action_num
        for index in range(self.action_num):
            for root, dirs, files in os.walk(file_paths[index]):
                for i, file in enumerate(files):
                    path = os.path.join(root, file)
                    a = np.loadtxt(path, delimiter=',')
                    a = torch.FloatTensor(a)
                    a = a.reshape(self.divide_num, -1, 45)
                    if not torch.is_tensor(all_data[index]):
                        all_data[index] = a
                    else:
                        all_data[index] = torch.cat((all_data[index], a), 0)
            all_data[index] = all_data[index].reshape(self.client_num * self.shard_num//self.action_num, -1, 125//self.divide_num, 45)
            # print(all_data[index].size())

        conbine_data = 0
        conbine_label = 0
        for i in range(self.action_num // self.shard_num):
            for j in range(all_data[0].shape[0]):
                # 遍历一个动作里待分配的数据切片，每个切片分给一个client，这里做的就是把分给一个client的切片都合并成b，就是一份数据
                for k in range(self.shard_num):
                    # 将分给一个client的shard数据合并出来
                    if k == 0:
                        label = torch.zeros(all_data[0].shape[1], self.action_num).index_fill(1, torch.tensor(
                            [i * self.shard_num + k]), 1)
                        # one-hot 赋予label
                        # print(label)
                        b = all_data[i * self.shard_num + k][j]
                    else:
                        b = torch.cat((b, all_data[i * self.shard_num + k][j]), 0)
                        # label = torch.cat((label,torch.ones(120,1)*(i*self.shard_num+k)),0)
                        label = torch.cat(
                            (label, torch.zeros(all_data[0].shape[1], self.action_num).index_fill(1, torch.tensor(
                                [i * self.shard_num + k]), 1)), 0)
                b = b.squeeze().unsqueeze(0)
                label = label.unsqueeze(0)

                # print(label.size)
                # print(b.size())
                if not torch.is_tensor(conbine_data):
                    conbine_data = b
                    conbine_label = label
                else:
                    conbine_data = torch.cat((conbine_data, b), 0)
                    conbine_label = torch.cat((conbine_label, label), 0)


        train1, test1, train2, test2, train3, test3 = conbine_data.split([20*self.divide_num, 4*self.divide_num, 20*self.divide_num, 4*self.divide_num, 20*self.divide_num, 4*self.divide_num], dim=1)
        train_l1, test_l1, train_l2, test_l2, train_l3, test_l3 = conbine_label.split([20*self.divide_num, 4*self.divide_num, 20*self.divide_num, 4*self.divide_num, 20*self.divide_num, 4*self.divide_num], dim=1)
        # 由于3种label的数据合并在一起，需要划分训练集和测试集，于是进行了6段split

        self.train_data, self.test_data = torch.cat((train1, train2, train3), 1), torch.cat((test1, test2, test3), 1)
        self.train_label, self.test_label = torch.cat((train_l1, train_l2, train_l3), 1), torch.cat((test_l1, test_l2, test_l3), 1)
        self.train_label = torch.argmax(self.train_label,dim=2)
        self.test_label = torch.argmax(self.test_label, dim=2)

        # #
        self.train_data_size = self.train_data.shape[1]
        self.test_data_size = self.test_data.shape[1]

        client_dict = {}
        conbine_data = 0
        conbine_label = 0

        for i in range(100):
            client_dict[i] = [j for j in range(i*self.train_data_size,(i+1)*self.train_data_size)]
            if not torch.is_tensor(conbine_data):
                conbine_data = self.train_data[i]
                conbine_label = self.train_label[i]
            else:
                conbine_data = torch.cat((conbine_data, self.train_data[i]), 0)
                conbine_label = torch.cat((conbine_label, self.train_label[i]), 0)
        self.trainset = TensorDataset(conbine_data,conbine_label)
        self.testset = TensorDataset(self.test_data.reshape(-1,25,45),self.test_label.reshape(-1))
        save_dict(client_dict,'dsa15_partition.pkl')
        # 为fedlab的使用生成每个client分配数据的序号

if __name__ == "__main__":
    wearabel_data = GetWearDataSet(3, 20, 3, 5)
    print(wearabel_data.train_data.size())
    # print(wearabel_data.train_data.reshape(5,20,-1,25, 45)[0,:,:,:,18:27])
    print(wearabel_data.train_label.size())
    print(wearabel_data.test_label.size())
    print(wearabel_data.test_label[0][0])
    print(wearabel_data.test_label[0][-1])
    print(wearabel_data.test_label[-1][0])
    print(wearabel_data.test_label[-1][-1])