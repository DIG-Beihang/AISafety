import numpy as np
import torch
from torch.autograd import Variable

from EvalBox.Attack.AdvAttack.attack import Attack


class SignHunter(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, **kwargs):
        """
        @description: Random FGSM
        @param {
            model:
            device:
            kwargs:
        } 
        @return: None
        """
        super(SignHunter, self).__init__(model, device, IsTargeted)

        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.debug = False
        self._parse_params(**kwargs)

    def _parse_params(self, **kwargs):
        """
        @description: 
        @param {
            epsilon:
            alpha:
        } 
        @return: None
        """
        self.eps = float(kwargs.get("epsilon", 0.1))
        self.max_loss_queries = int(kwargs.get("max_queries", 1000))
        self.max_crit_queries = self.max_loss_queries

    def early_stop_crit(self, xs, ys):
        var_xs = torch.tensor(
            xs, dtype=torch.float, device=self.device, requires_grad=False
        )
        return torch.ne(torch.argmax(self.model(var_xs), 1).to(ys.device).data, ys)

    def loss(self, xs, ys):
        var_xs = torch.tensor(
            xs, dtype=torch.float, device=self.device, requires_grad=False
        )
        logits = self.model(var_xs).to(xs.device)
        return self.criterion(logits, ys).data

    def generate(self, xs=None, ys=None):
        """
        @description: 
        @param {
            xs:
            ys:
        } 
        @return: adv_xs
        """
        device = self.device
        batch_size = xs.shape[0]
        num_loss_queries = torch.zeros(batch_size)
        num_crit_queries = torch.zeros(batch_size)
        done_mask = self.early_stop_crit(xs, ys)
        if torch.all(done_mask):
            return xs
        # clamp
        losses = torch.zeros(batch_size)
        adv_xs = xs.clone()
        xo = adv_xs.clone()
        h = 0
        i = 0
        dim = np.prod(xs.shape[1:])
        sgn = torch.ones((batch_size, dim))
        fxs = xs + self.eps * sgn.view(xs.shape)
        bxs = xo
        est_deriv = (self.loss(fxs, ys) - self.loss(bxs, ys))/self.eps
        best_est_deriv = est_deriv
        num_loss_queries += 3
        while True:
            if torch.any(num_crit_queries >= self.max_crit_queries):
                break
            if torch.all(done_mask):
                break

            chunk_len = np.ceil(dim / 2**h).astype(int)
            istart = i*chunk_len
            iend = min(dim, (i+1)*chunk_len)
            sgn[:, istart:iend] *= -1
            fxs = xo + self.eps * sgn.view(xs.shape)
            bxs = xo
            est_deriv = (self.loss(fxs, ys) - self.loss(bxs, ys))/self.eps
            sgn[[i for i, val in enumerate(est_deriv < best_est_deriv) if val], istart:iend] *= -1
            best_est_deriv = (est_deriv >= best_est_deriv) * est_deriv + (est_deriv < best_est_deriv) * best_est_deriv
            new_xs = xo + self.eps * sgn.view(xs.shape)
            new_xs = torch.clamp(new_xs, 0., 1.)
            i += 1
            if i == 2**h or iend == dim:
                h+=1
                i=0
                if h == np.ceil(np.log2(dim)).astype(int) + 1:
                    xo = adv_xs.clone()
                    h = 0
            undone_mask = torch.logical_not(done_mask)
            adv_xs[undone_mask] = new_xs[undone_mask]
            num_loss_queries += 1
            num_crit_queries += undone_mask
            losses[undone_mask] = self.loss(adv_xs, ys)[undone_mask]
            
            done_mask = torch.logical_or(done_mask, self.early_stop_crit(adv_xs, ys))
        if self.debug:
            print(num_crit_queries)
            print(torch.max(torch.abs(adv_xs-xs)))
        return adv_xs