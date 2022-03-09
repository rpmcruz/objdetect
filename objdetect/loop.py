from time import time
from tqdm import tqdm
import torch

'''
Convenience functions for the training and evaluation loops. Losses must be a
dictionary, specifying which loss to use for each respective model output and
input data.
'''

def train(model, tr, opt, losses, epochs):
    model.train()
    for epoch in range(epochs):
        print(f'* Epoch {epoch+1} / {epochs}')
        tic = time()
        avg_loss = 0
        for data in tr:
            opt.zero_grad()
            X = data['image'].permute(0, 3, 1, 2).cuda()
            preds = model(X)
            loss = sum(losses[key](
                # permutate and flatten so that we have a vector of features
                # squeeze removes the feature dimension if there is only one feature
                preds[key].permute(0, 2, 3, 4, 1).flatten(0, 3).squeeze().cuda(),
                data[key].permute(0, 2, 3, 4, 1).flatten(0, 3).squeeze().cuda()
                ) for key in losses)
            loss.backward()
            opt.step()
            avg_loss += float(loss) / len(tr)
        toc = time()
        print(f'- {toc-tic:.1f}s - Loss: {avg_loss}')

def evaluate(model, ts, inv_grid_transform):
    list_inputs = []
    list_preds = []
    model.eval()
    for data in tqdm(ts):
        X = data['image'].permute(0, 3, 1, 2).cuda()
        with torch.no_grad():
            preds = model(X)
        preds = {k: v.detach().cpu().numpy() for k, v in preds.items()}
        preds = inv_grid_transform(preds)
        list_inputs += inv_grid_transform(data)
        list_preds += preds
    return list_inputs, list_preds
