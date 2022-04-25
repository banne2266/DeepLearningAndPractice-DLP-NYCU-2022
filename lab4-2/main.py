import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.models as models

import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from dataloader import RetinopathyLoader
from resnet import Resnet18, Resnet50

device = torch.device("cuda")
EPOCH = 1
SAVE = 0
LOAD = 1
TRAIN = 0
LOAD_PATH = './resnet50best80.9.pth'
SAVE_PATH = './resnet50beg.pth'

def main():
    net = models.resnet50(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, 5)
    net = net.to(device=device)

    #net = Resnet50().to(device=device)

    if LOAD:
        net.load_state_dict(torch.load(LOAD_PATH))
        print("Model weights loaded sucessful")
    
    
    print(net.eval())

    train_dataset = RetinopathyLoader("./data/", "train")
    train_loader = data.DataLoader(train_dataset, batch_size=16, shuffle=False, drop_last=True)

    test_dataset = RetinopathyLoader("./data/", "test")
    test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

    #optimizer = torch.optim.Adam(net.parameters(), lr = 0.001, weight_decay=5e-4)
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.1, momentum = 0.9, weight_decay=5e-4)
    loss = nn.CrossEntropyLoss()

    y_pred_cm = []
    y_true_cm = []

    for epoch in range(1, EPOCH+1):
        
        if TRAIN:
            total_acc = 0
            total_item = 0
            for batch_id, (x, y_true) in enumerate(train_loader):
                x = x.to(device=device, dtype=torch.float)
                y_true = y_true.to(device=device)

                net.zero_grad()

                y_pred = net(x)
                output = loss(y_pred, y_true)
                output.backward()
                optimizer.step()


                _, y_pred = torch.max(y_pred.data, 1)
                total_acc += (y_pred == y_true).sum()
                total_item += y_pred.shape[0]
                print(f"Training epoch: {epoch}/{EPOCH}, batch: {batch_id}/{len(train_loader)}", end='\r')

            accuracy = total_acc / total_item
            print(f"epoch: {epoch}/{EPOCH}, Training loss: {output.item()}, Training accuracy: {accuracy}")


        total_acc = 0
        total_item = 0
        
        with torch.no_grad():
            for batch_id, (x, y_true) in enumerate(test_loader):
                if epoch == EPOCH:
                    y_true_cm.extend(y_true)
                x = x.to(device=device, dtype=torch.float)
                y_true = y_true.to(device=device)

                y_pred = net(x)
                output = loss(y_pred, y_true)

                _, y_pred = torch.max(y_pred.data, 1)
                total_acc += (y_pred == y_true).sum()
                total_item += y_pred.shape[0]
                print(f"Testing: {batch_id}/{len(test_loader)}", end='\r')
                if epoch == EPOCH:
                    y_pred = np.array(y_pred.cpu())
                    y_pred_cm.extend(y_pred)
        accuracy = total_acc / total_item
        print(f"Test loss: {output.item()}, Test accuracy: {accuracy}")
    
    if SAVE:
        torch.save(net.state_dict(), SAVE_PATH)
        print("Model weights saved sucessful")

    cm = confusion_matrix(y_true_cm, y_pred_cm, normalize='all')
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.show()
    

if __name__ == "__main__":
    main()