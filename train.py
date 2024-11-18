import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from model import WaveNet10

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_step(model, dataloader, loss_fn, optimizer, device=device):
    model.to(device)
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (y_pred.argmax(dim=1) == y).sum().item() / len(y)
    train_batches = len(dataloader)
    return train_loss / train_batches, train_acc / train_batches


def test_step(model, dataloader, loss_fn, device=device):
    model.to(device)
    model.eval()
    test_loss, test_acc = 0.0, 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        with torch.inference_mode():
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
        test_loss += loss.item()
        test_acc += (y_pred.argmax(dim=1) == y).sum().item() / len(y)
    test_batches = len(dataloader)
    return test_loss / test_batches, test_acc / test_batches


def train(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs, device=device):
    results = {'train_loss': [],
               'train_acc': [],
               'test_loss': [],
               'test_acc': []}
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
        print(f'Epoch: {epoch+1} | train_loss: {train_loss:.3f} | train_acc: {train_acc:.3f}')
        print(f'Epoch: {epoch+1} | test_loss: {test_loss:.3f} | test_acc: {test_acc:.3f}')
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

    return results


def plot_results(results):
    x = np.arange(1, len(results['train_loss']) + 1)
    plt.subplot(121)
    plt.plot(x, results['train_loss'], label='train_loss')
    plt.plot(x, results['test_loss'], label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(x, results['train_acc'], label='train_acc')
    plt.plot(x, results['test_acc'], label='test_acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()

    plt.show()


BATCH_SIZE = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_path = Path('D:/PycharmProjects/data/')
model_path = Path('./models/')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar10_train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
cifar10_test_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
cifar10_train_dataloader = DataLoader(dataset=cifar10_train_dataset,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True)
cifar10_test_dataloader = DataLoader(dataset=cifar10_test_dataset,
                                     batch_size=BATCH_SIZE,
                                     shuffle=False)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
model = WaveNet10(scales=[[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2]],
                  reps=[5, 6, 7],
                  dim=8,
                  dropout=0.1
                  )   # Total params: 71,282
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

res = {'train_loss': [],
       'train_acc': [],
       'test_loss': [],
       'test_acc': []}
epochs = [70, 20, 10, 10]
lr = [0.001, 0.0005, 0.0001, 0.00005]
start = timer()
for i in range(4):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr[i], weight_decay=5e-4)
    results = train(model=model,
                    train_dataloader=cifar10_train_dataloader,
                    test_dataloader=cifar10_test_dataloader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    epochs=epochs[i],
                    path=model_path,
                    device=device,
                    )
    for k in list(res.keys()):
        res[k] += results[k]
end = timer()
print(f'Training time: {end - start:.3f} seconds.')
plot_results(res)
