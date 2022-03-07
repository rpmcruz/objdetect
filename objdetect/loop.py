from time import time
import torch

'''
Convenience functions for the training and evaluation loops. Losses must be a
dictionary, specifying which loss to use for each respective model output and
input data.
'''

def train(model, tr, losses, epochs):
    model.train()
    for epoch in range(epochs):
        print(f'* Epoch {epoch+1} / {epochs}')
        tic = time()
        avg_loss = 0
        for data in tr:
            opt.zero_grad()
            X = data['image'].cuda()
            preds = model(X)
            loss = torch.sum([losses[key](preds[key], data[key]) for key in losses])
            loss.backward()
            opt.step()
            avg_loss += float(loss) / len(tr)
        toc = time()
        print(f'- {toc-tic:.1f}s - Loss: {avg_loss}')
    return model

def evaluate(model, ts, inv_grid_transform):
    all_preds = []
    model.eval()
    for d in ts:
        with torch.no_grad():
            preds = model(_X)
        H_pred.append(_H_pred.detach().cpu().numpy())
        B_pred.append(_B_pred.detach().cpu().numpy())
    H_pred = np.concatenate(H_pred)
    B_pred = np.concatenate(B_pred)
