import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Client:
    def __init__(self, cid, dataset, device="cpu"):
        self.cid = cid
        self.dataset = dataset
        self.device = device

    def local_train(self, model, epochs=1, lr=0.05, batch_size=32):
        """Train a copy of the global model locally.
        Returns the model update (delta = local_weights - global_weights)."""
        local = copy.deepcopy(model).to(self.device)
        global_state = {k: v.clone() for k, v in model.state_dict().items()}
        opt = torch.optim.SGD(local.parameters(), lr=lr, momentum=0.9)
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        local.train()
        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                nn.functional.cross_entropy(local(x), y).backward()
                opt.step()
        return {k: (v.cpu() - global_state[k].cpu()) for k, v in local.state_dict().items()}

    def single_sample_gradient(self, model, index=0):
        """Gradient of the loss on ONE private image w.r.t. the model weights.
        This is the target for the first DLG experiment (Task 2).
        Returns (gradient list, image, label)."""
        m = copy.deepcopy(model).to(self.device).eval()
        x, y = self.dataset[index]
        x = x.unsqueeze(0).to(self.device)
        y = torch.tensor([y], device=self.device)
        loss = nn.functional.cross_entropy(m(x), y)
        grads = torch.autograd.grad(loss, list(m.parameters()))
        return [g.detach().cpu() for g in grads], x.cpu(), y.cpu()
