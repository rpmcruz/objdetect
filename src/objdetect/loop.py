from time import time
from tqdm import tqdm
import torch

'''
Convenience functions for the training and evaluation loops. Losses must be a
dictionary, specifying which loss to use for each respective model output and
input data. At each epoch, calls scheduler.step(avg_loss) and stops if it
returns True.

Please keep in mind that losses may not produce a scalar -- i.e., you must use
reduction='none'.
'''

def train(model, tr, opt, losses, epochs, scheduler=None):
    model.train()
    for epoch in range(epochs):
        print(f'* Epoch {epoch+1} / {epochs}')
        tic = time()
        avg_loss = 0
        for data in tr:
            opt.zero_grad()
            X = data['image'].permute(0, 3, 1, 2).cuda()
            preds = model(X)

            loss = 0
            confs_grid = data['confs_grid'].permute(0, 2, 3, 4, 1).flatten(0, 3).squeeze().cuda()
            for key in losses:
                # we must mupltiply all loss value (except for confs_grid) by
                # data[confs_grid] (which is 0 or 1) so that if there is no
                # object then the model is *not* penalized.
                coef = 1 if key == 'confs_grid' else confs_grid
                # permutate and flatten so that we have a vector of features
                # squeeze removes the feature dimension if there is only one feature
                _preds = preds[key].permute(0, 2, 3, 4, 1).flatten(0, 3).squeeze().cuda()
                _inputs = data[key].permute(0, 2, 3, 4, 1).flatten(0, 3).squeeze().cuda()
                loss_value = losses[key](_preds, _inputs)
                assert loss_value.ndim > 0, f"Loss {key} cannot produce scalars (ensure you use reduction='none')"
                if len(loss_value.shape) > 1:
                    loss_value = loss_value.sum(1)
                loss += (coef * loss_value).mean()

            loss.backward()
            opt.step()
            avg_loss += float(loss) / len(tr)
        toc = time()
        print(f'- {toc-tic:.1f}s - Loss: {avg_loss}')
        if scheduler:
            if scheduler.step(avg_loss):
                print('Stop due to scheduler')
                break

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

class ConvergeStop:
    def __init__(patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.last_loss = 999999
        self.count = 0

    def step(self, loss):
        self.count = self.count+1 if loss+self.min_delta >= self.last_loss else 0
        self.last_loss = loss
        return self.count >= self.patience
