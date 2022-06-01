'''
Convenience functions for the training and evaluation loops.
'''

import torch
from time import time
from tqdm import tqdm

def train(tr, model, opt, weight_loss_fns, loss_fns, epochs, stop_condition=None):
    '''Trains the model. `weight_loss_fns` and `loss_fns` are dictionaries, specifying whether the loss should be applied to that grid location and what loss to apply.'''
    device = next(model.parameters()).device

    # sanity-check: all losses must have reduction='none'
    data = next(iter(tr))
    preds = model(data['image'].to(device))
    for name, f in loss_fns.items():
        loss_value = f(preds[name], data[name].to(device))
        assert len(loss_value.shape) > 0, f"Loss {name} must have reduction='none'"

    model.train()
    for epoch in range(epochs):
        print(f'* Epoch {epoch+1} / {epochs}')
        tic = time()
        avg_loss = 0
        avg_losses = {name: 0 for name in loss_fns}
        for data in tr:
            X = data['image'].to(device)
            preds = model(X)
            data_cuda = {name: data[name].to(device) for name in loss_fns}

            loss = 0
            for name, f in loss_fns.items():
                W = weight_loss_fns[name](data_cuda)
                true = data_cuda[name]
                pred = preds[name]
                loss_value = (W*f(pred, true)).mean()
                loss += loss_value
                avg_losses[name] += float(loss_value) / len(tr)

            opt.zero_grad()
            loss.backward()
            opt.step()
            avg_loss += float(loss) / len(tr)
        toc = time()
        print(f'- {toc-tic:.1f}s - Avg loss: {avg_loss} - ' + ' - '.join(f'{name} loss: {avg}' for name, avg in avg_losses.items()))
        if stop_condition and stop_condition(avg_loss):
            print('Stopping due to criteria')
            break

def eval(ts, model):
    '''Evaluates the model.'''
    device = next(model.parameters()).device
    inputs = []
    outputs = []
    model.eval()
    for data in tqdm(ts):
        X = data['image'].to(device)
        with torch.no_grad():
            preds = model(X)

        n = X.shape[0]
        inputs += [{k: v[i] for k, v in data.items()} for i in range(n)]
        outputs += [{k: v[i].detach().cpu() for k, v in preds.items()} for i in range(n)]
    return inputs, outputs

class StopPatience:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.min_loss = float('inf')

    def __call__(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
