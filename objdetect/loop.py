from time import time
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
                # get rid of anchors because CE loss doesn't work with 5-dims
                preds[key].flatten(0, 1), data[key].flatten(0, 1).cuda())
                for key in losses)
            loss.backward()
            opt.step()
            avg_loss += float(loss) / len(tr)
        toc = time()
        print(f'- {toc-tic:.1f}s - Loss: {avg_loss}')

def evaluate(model, ts, inv_grid_transform):
    all_preds = []
    model.eval()
    for data in ts:
        X = data['image'].permute(0, 3, 1, 2).cuda()
        with torch.no_grad():
            preds = model(X)
        preds = preds.detach().cpu().numpy()
        preds = inv_grid_transform(preds)
        all_preds.append(preds)
    return all_preds
