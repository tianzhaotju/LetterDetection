from Data_Reader import Dataset
from base.base_net import BaseNet
from Network import Autoencoder
from base.base_dataset import BaseADDataset
from utils.visualization.plot_images_grid import plot_images_grid
import torch.optim as optim
import time
import torch

data_path = './data/letter'
n_epochs = 100

device = 'cuda'
lr = 0.001
weight_decay = 0.0001
batch_size = 32

def main():
    trainset = Dataset(root=data_path+'/train/')
    testset = Dataset(root=data_path + '/test/')
    net = Autoencoder()
    net = train(trainset, net)
    test(testset, net)


def train(dataset: BaseADDataset, ae_net: BaseNet):
    # Set device for network
    ae_net = ae_net.to(device)

    # Get train data loader
    letter, labels = dataset.loaders(batch_size=batch_size, num_workers=0, shuffle_test= False, shuffle_train= False)

    # Set optimizer (Adam optimizer for now)
    optimizer = optim.Adam(ae_net.parameters(), lr=lr, weight_decay=weight_decay)

    # Training
    start_time = time.time()
    ae_net.train()
    for epoch in range(n_epochs):
        loss_epoch = 0.0
        n_batches = 0
        epoch_start_time = time.time()
        for data, label in zip(letter, labels):
            inputs, _ = data
            lab  , _= label
            inputs = inputs.to(device)
            lab = lab.to(device)
            # Zero the network parameter gradients
            optimizer.zero_grad()
            outputs = ae_net(inputs)
            scores = torch.sum((outputs - lab) ** 2, dim=tuple(range(1, outputs.dim())))
            loss = torch.mean(scores)
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()
            n_batches += 1
        epoch_end_time = time.time()
        print('Epoch: '+str(epoch+1)+'/'+str(n_epochs)+' time: '+str(epoch_end_time-epoch_start_time)+' loss: '+str(loss_epoch/n_batches))

    with torch.no_grad():
        plot_images_grid(inputs, export_img='./log/train/input', title='Input ', nrow=4, padding=4)
        plot_images_grid(lab, export_img='./log/train/labbel', title='Label ', nrow=4, padding=4)
        plot_images_grid(outputs, export_img='./log/train/output', title='Output ', nrow=4, padding=4)

    return ae_net

def test(dataset: BaseADDataset, ae_net: BaseNet):
    # Set device for network
    ae_net = ae_net.to(device)

    # Get test data loader

    letter, labels = dataset.loaders(batch_size=batch_size, num_workers=0, shuffle_test= False, shuffle_train= False)

    loss_epoch = 0.0
    n_batches = 0
    start_time = time.time()

    with torch.no_grad():
        i = 0
        for data, label in zip(letter, labels):
            i += 1
            inputs, _ = data
            lab, _ = label
            inputs = inputs.to(device)
            lab = lab.to(device)
            # Zero the network parameter gradients
            outputs = ae_net(inputs)
            plot_images_grid(inputs[0:16], export_img='./log/test/input'+str(i), title='Input ', nrow=4, padding=4)
            plot_images_grid(lab[0:16], export_img='./log/test/label'+str(i), title='Label ', nrow=4, padding=4)
            plot_images_grid(outputs[0:16], export_img='./log/test/output'+str(i), title='Output ', nrow=4, padding=4)



if __name__ == '__main__':
    main()
