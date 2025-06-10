from copy import deepcopy

import numpy as np
import torch
from sklearn.metrics import accuracy_score


def compute_accuracy(test_loader, model):
    var_labels, var_preds = [], [],
    with torch.no_grad():
        for batch, (X, labels) in enumerate(test_loader):
            _, pred = model(X.to(model.device))

            var_labels += list(labels.squeeze().to('cpu').numpy()) if len(labels) > 1 \
                else [labels.squeeze().to('cpu').numpy()]
            var_preds += list(pred.squeeze().to('cpu').numpy()) if len(pred) > 1 \
                else [list(pred.squeeze().to('cpu').numpy())]

        accuracy = accuracy_score(var_labels, np.argmax(var_preds, axis=1))

    return accuracy


def compute_accuracy_graph(graph,model,subset):

    device = model.device
    x = graph[0][0].x.to(device).float()
    edge_index = graph[0][0].edge_index.to(device).long()

    subset = torch.tensor(subset, dtype=torch.long)
    labels = graph[0][1][subset].to(device).long()
    model = model.to(device).float()
    
    #subset should be a partition, already instantiated.
    with torch.no_grad():
        pred = model(x,edge_index)[subset]

    labels = labels.detach().cpu().numpy()   
    pred = pred.detach().cpu().numpy() 

    accuracy = accuracy_score(labels, np.argmax(pred, axis=1))

    return accuracy
    

def compute_relearn_time(model, data_loader, max_accuracy=0.8, max_epochs=100):

    # model = deepcopy(model)

    epochs = 0

    curr_accuracy = compute_accuracy(data_loader, model.model)

    while (curr_accuracy < max_accuracy  # reached the target accuracy
    and epochs < max_epochs):  # fine-tune for a maximum of epochs
        losses, preds, labels_list = [], [], []
        model.model.train()
        for batch, (X, labels) in enumerate(data_loader):
            X, labels = X.to(model.device), labels.to(model.device)
            model.optimizer.zero_grad()
            _, pred = model.model(X)
            loss = model.loss_fn(pred, labels)
            loss.backward()
            model.optimizer.step()

            losses.append(loss.to('cpu').detach().numpy())
            if labels.dim() == 0:  # If labels is a scalar
                labels_list += [labels.item()]  # Add scalar as a single element list
            else:
                labels_list += list(labels.squeeze().long().detach().to('cpu').numpy())
            if pred.dim() == 0:  # If pred is a scalar
                preds += [pred.item()]  # Add scalar as a single element list
            else:
                preds += list(pred.squeeze().detach().to('cpu').numpy())


        curr_accuracy = model.accuracy(labels_list, preds)
        model.lr_scheduler.step()

        epochs += 1

    return epochs


def compute_relearn_time_graph(graph, model, subset, device, max_accuracy=0.8, max_epochs=100):

    # model = deepcopy(model)

    epochs = 0

    curr_accuracy = compute_accuracy_graph(graph, model.model, subset)

    x = graph[0][0].x.to(device).float()
    edge_index = graph[0][0].edge_index.to(device).long()
    subset = torch.tensor(subset, dtype=torch.long)
    labels = graph[0][1][subset].to(device).long()

    model.model = model.model.to(device).float()

    while (curr_accuracy < max_accuracy  
    and epochs < max_epochs):  
        losses, preds, labels_list = [], [], []
        model.model.train()
        
        model.optimizer.zero_grad()
        pred = model.model(x, edge_index)[subset]
        loss = model.loss_fn(pred,labels)
        loss.backward()
        model.optimizer.step()

        losses.append(loss.to('cpu').detach().numpy())
        if labels.dim() == 0:  
            labels_list += [labels.item()]  
        else:
            labels_list += list(labels.squeeze().long().detach().to('cpu').numpy())
        if pred.dim() == 0: 
            preds += [pred.item()]  
        else:
            preds += list(pred.squeeze().detach().to('cpu').numpy())


        curr_accuracy = model.accuracy(labels_list, preds)
        model.lr_scheduler.step()

        epochs += 1

    return epochs
