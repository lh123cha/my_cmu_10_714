import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    modules = nn.Sequential(
        nn.Linear(dim, hidden_dim),# Linear层的参数是 in_feature*out_feature+out_feature
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),#
        norm(dim) #4*dim个参数
    )
    return nn.Sequential(
        nn.Residual(modules),
        # NOTE ReLU after Residual
        nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    layer1 = nn.Linear(dim,hidden_dim)
    relu = nn.ReLU()
    model=[layer1,relu]
    for i in range(num_blocks):
        model.append(ResidualBlock(hidden_dim,hidden_dim//2,norm,drop_prob))
    output_layer = nn.Linear(hidden_dim,num_classes)
    model.append(output_layer)
    return nn.Sequential(*model)
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    """
    Executes one epoch of training or evaluation, iterating over the entire training dataset once (just like nn_epoch from previous homeworks).
    Returns the average error rate (changed from accuracy) (as a float) and the average loss over all samples (as a float).
    Set the model to training mode at the beginning of the function if opt is given; set the model to eval if opt is not given (i.e. None).
    Parameters
    dataloader (needle.data.DataLoader) - dataloader returning samples from the training dataset
    model (needle.nn.Module) - neural network
    opt (needle.optim.Optimizer) - optimizer instance, or None
    1.判断是否包含opt，训练模式或者测试
    2.
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_func = nn.SoftmaxLoss()
    corrects ,total_loss,total_samples,total_batches =0.0,0.0,0,0
    if opt:
        model.train()
    else:
        model.eval()
    for X,y in dataloader:
        if opt:
            opt.reset_grad()
        predict = model(X)
        loss = loss_func(predict,y)
        corrects += (predict.numpy().argmax(axis=1) == y.numpy()).sum()
        if opt:
            loss.backward()
            opt.step()
        total_loss += loss.numpy()
        total_batches += 1
        total_samples += X.shape[0]
    return (1-corrects/total_samples),total_loss/total_batches
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    trian_accuracy,train_loss,test_accuracy,test_loss = 0.0,0.0,0.0,0.0
    train_dataset =ndl.data.MNISTDataset(data_dir+"/train-images-idx3-ubyte.gz",
                                         data_dir+"/train-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
    test_dataset = ndl.data.MNISTDataset(data_dir+"/t10k-images-idx3-ubyte.gz",
                                         data_dir+"/t10k-labels-idx1-ubyte.gz")
    test_dataloader = ndl.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
    model = MLPResNet(dim=28*28,hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(),lr=lr,weight_decay=weight_decay)
    for epoch_num in range(epochs):
        trian_accuracy,train_loss = epoch(train_dataloader,model,opt)
    test_accuracy,test_loss = epoch(test_dataloader,model,None)
    return trian_accuracy,train_loss,test_accuracy,test_loss
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
