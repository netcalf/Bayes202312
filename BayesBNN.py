import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

import pyro
from pyro.distributions import Normal
from pyro.distributions import Categorical
from pyro.optim import Adam
from pyro.infer import SVI
from pyro.infer import Trace_ELBO

import matplotlib.pyplot as plt
import seaborn as sns

import torch.cuda

from torchvision.datasets import ImageFolder
from torchvision.transforms import Grayscale

#定义神经网络模型
class BNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.out(output)
        return output

# 基于正态分布，初始化weights和bias
def model(x_data, y_data):
    # define prior destributions
    fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight),scale=torch.ones_like(net.fc1.weight))
    fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias))
    outw_prior = Normal(loc=torch.zeros_like(net.out.weight),scale=torch.ones_like(net.out.weight))
    outb_prior = Normal(loc=torch.zeros_like(net.out.bias),scale=torch.ones_like(net.out.bias))

    priors = {
        'fc1.weight': fc1w_prior,
        'fc1.bias': fc1b_prior,
        'out.weight': outw_prior,
        'out.bias': outb_prior}
    
    lifted_module = pyro.random_module("module", net, priors)
    lifted_reg_model = lifted_module()

    lhat = F.log_softmax(lifted_reg_model(x_data))
    pyro.sample("obs", Categorical(logits=lhat), obs=y_data)

# 为了近似后验概率分布，先定义一个函数
def guide(x_data, y_data):
    fc1w_mu = torch.randn_like(net.fc1.weight)
    fc1w_sigma = torch.randn_like(net.fc1.weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = F.softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)

    fc1b_mu = torch.randn_like(net.fc1.bias)
    fc1b_sigma = torch.randn_like(net.fc1.bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = F.softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)

    outw_mu = torch.randn_like(net.out.weight)
    outw_sigma = torch.randn_like(net.out.weight)
    outw_mu_param = pyro.param("outw_mu", outw_mu)
    outw_sigma_param = F.softplus(pyro.param("outw_sigma", outw_sigma))
    outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param).independent(1)

    outb_mu = torch.randn_like(net.out.bias)
    outb_sigma = torch.randn_like(net.out.bias)
    outb_mu_param = pyro.param("outb_mu", outb_mu)
    outb_sigma_param = F.softplus(pyro.param("outb_sigma", outb_sigma))
    outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)

    priors = {
        'fc1.weight': fc1w_prior,
        'fc1.bias': fc1b_prior,
        'out.weight': outw_prior,
        'out.bias': outb_prior}

    lifted_module = pyro.random_module("module", net, priors)

    return lifted_module()
def predict(x):
    sampled_models = [guide(None, None) for _ in range(n_samples)]
    yhats = [model(x).data.to(device) for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return mean

def predict_prob(x):
    sampled_models = [guide(None, None) for _ in range(n_samples)]
    yhats = [model(x.to(device)).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return mean.to('cpu')  # Move the result back to CPU for visualization
#正则化
def normalize(x):
    return (x - x.min()) / (x.max() - x.min())
#可视化
def plot(x, yhats):
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(8, 4))

    softmax_values = F.softmax(torch.Tensor(normalize(yhats.cpu().numpy())), dim=1)
    axL.bar(x=[i for i in range(7)], height=F.softmax(torch.Tensor(normalize(yhats.cpu().numpy()))[0]))
    axL.set_xticks([i for i in range(10)], [i for i in range(10)])
    axR.imshow(x.cpu().numpy()[0])
    plt.show()

if __name__ == '__main__':

    sns.set()

    # 数据集根目录
    train_dataset_root = "./wood6-05/train"
    valid_dataset_root = "./wood6-05/valid"
    # 获取数据集的类别数
    num_classes = len(os.listdir(train_dataset_root))
    print("数据集类别数：", num_classes)
  
    # 定义 transforms，包括将RGB图像转换为单通道灰度图像
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # 调整图像大小
        Grayscale(num_output_channels=1),  # 转换为单通道灰度图像
        transforms.ToTensor(),  # 转换为 Tensor
    ])

    # 加载训练数据集和验证数据集
    train_dataset = ImageFolder(root=train_dataset_root, transform=transform)
    valid_dataset = ImageFolder(root=valid_dataset_root, transform=transform)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False)
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    # net = BNN(28 * 28, 1024, 10)
    net = BNN(28 * 28, 1024, 7)
    net = net.to(device)
    # 定义Optimizer
    optim = Adam({"lr": 0.01})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())

 #获取数据集中每个类别的图像数量，并计算每个类别的样本数量占比
    train_dataset = datasets.ImageFolder(root=train_dataset_root)
    valid_dataset = datasets.ImageFolder(root=valid_dataset_root)
    train_class_num = [0 for i in range(num_classes)]
    valid_class_num = [0 for i in range(num_classes)]
    for i in range(len(train_dataset)):
        train_class_num[train_dataset[i][1]] += 1
    for i in range(len(valid_dataset)):
        valid_class_num[valid_dataset[i][1]] += 1
    print("训练集中每个类别的样本数量：", train_class_num)
    print("验证集中每个类别的样本数量：", valid_class_num)
    train_class_num = np.array(train_class_num)
    valid_class_num = np.array(valid_class_num)
    print("训练集中每个类别的样本数量占比：", train_class_num / train_class_num.sum())
    print("验证集中每个类别的样本数量占比：", valid_class_num / valid_class_num.sum())

    #根据每个类别的样本数量，设置每个分类的先验概率
    class_probabilities=train_class_num/train_class_num.sum()
    # 设置每个分类的先验概率
    priors = {
        'fc1.weight': Normal(loc=torch.zeros_like(net.fc1.weight), scale=torch.ones_like(net.fc1.weight)),
        'fc1.bias': Normal(loc=torch.zeros_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias)),
        'out.weight': Normal(loc=torch.zeros_like(net.out.weight), scale=torch.ones_like(net.out.weight)),
        'out.bias': Normal(loc=torch.zeros_like(net.out.bias), scale=torch.ones_like(net.out.bias))
    }
    # 根据每个分类的样本数量占比调整先验概率
    for i, key in enumerate(priors.keys()):
        priors[key] = Normal(loc=torch.zeros_like(net.state_dict()[key]), scale=torch.ones_like(net.state_dict()[key])) * class_probabilities[i]

    # 训练Bayesian Neural Network
    n_iterations = 10
    for j in range(n_iterations):
        loss = 0
        for batch_id, data in enumerate(train_loader):
            # loss += svi.step(data[0].view(-1, 28 * 28), data[1])
            inputs, labels = data[0].view(-1, 28 * 28).to(device), data[1].to(device)
            loss += svi.step(inputs, labels)
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = loss / normalizer_train
        print("Epoch ", j, " Loss ", total_epoch_loss_train)

    # 使用训练好的模型，预测结果
    print('Prediction when network is forced to predict')
    correct = 0
    total = 0
    n_samples=10
    for j, data in enumerate(valid_loader):
        images, labels = data
        predicted = predict(images.view(-1,28*28).to(device)).cpu().numpy()
        total += labels.size(0)
        correct += (np.argmax(predicted, axis=1) == labels.numpy()).sum()
    print("accuracy: %d %%" % (100 * correct / total))

    # 可视化
    x, y = valid_loader.dataset[0]
    yhats = predict_prob(x.view(-1, 28*28).to(device))
    print("ground truth: ", y)
    print("predicted: ", yhats.cpu().numpy())
    plot(x, yhats)
