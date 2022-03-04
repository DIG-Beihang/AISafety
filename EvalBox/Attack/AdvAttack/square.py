import numpy as np
import torch
from torch.autograd import Variable

from EvalBox.Attack.AdvAttack.attack import Attack

class Square(Attack):
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
        super(Square, self).__init__(model, device, IsTargeted)

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
        self.n_iters = int(kwargs.get("n_iters", 1000))
        self.p_init = float(kwargs.get("p_init", 0.05))

    def logits(self, xs):
        var_xs = torch.tensor(
            xs, dtype=torch.float, device=self.device, requires_grad=False
        )
        return self.model(var_xs).to("cpu")

    def loss(self, ys, logits, targeted=False, loss_type='margin_loss'):
        y = torch.tensor(ys)
        if loss_type == 'margin_loss':
            mask = torch.nn.functional.one_hot(y, logits.shape[1])
            diff = torch.max(logits)-torch.min(logits)+1
            excluded = logits - diff * mask
            margin = torch.max(logits - diff * (1 - mask), 1)[0] - torch.max(excluded, 1)[0]
            loss = margin * -1 if targeted else margin
        elif loss_type == 'cross_entropy':
            loss = self.criterion(logits, y)
            loss = loss * -1 if not targeted else loss
        else:
            raise ValueError('Wrong loss.')
        return loss.data.numpy()

    def p_selection(self, p_init, it, n_iters):
        """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
        it = int(it / n_iters * 10000)

        if 10 < it <= 50:
            p = p_init / 2
        elif 50 < it <= 200:
            p = p_init / 4
        elif 200 < it <= 500:
            p = p_init / 8
        elif 500 < it <= 1000:
            p = p_init / 16
        elif 1000 < it <= 2000:
            p = p_init / 32
        elif 2000 < it <= 4000:
            p = p_init / 64
        elif 4000 < it <= 6000:
            p = p_init / 128
        elif 6000 < it <= 8000:
            p = p_init / 256
        elif 8000 < it <= 10000:
            p = p_init / 512
        else:
            p = p_init

        return p

    def generate(self, xs, ys):
        """ The Linf square attack """
        x=xs.numpy()
        y=ys.numpy()
        np.random.seed(0)  # important to leave it here as well
        min_val, max_val = 0, 1 if x.max() <= 1 else 255
        c, h, w = x.shape[1:]
        n_features = c*h*w
        n_ex_total = x.shape[0]

        # [c, 1, w], i.e. vertical stripes work best for untargeted attacks
        init_delta = np.random.choice([-self.eps, self.eps], size=[x.shape[0], c, 1, w])
        x_best = np.clip(x + init_delta, min_val, max_val)

        logits = self.logits(x_best)
        loss_min = self.loss(y, logits, self.IsTargeted, loss_type='cross_entropy')
        margin_min = self.loss(y, logits, self.IsTargeted, loss_type='margin_loss')
        n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query

        for i_iter in range(self.n_iters - 1):
            idx_to_fool = margin_min > 0
            x_curr, x_best_curr, y_curr = x[idx_to_fool], x_best[idx_to_fool], y[idx_to_fool]
            if x_curr.shape[0] == 0:
                break
            loss_min_curr, margin_min_curr = loss_min[idx_to_fool], margin_min[idx_to_fool]
            deltas = x_best_curr - x_curr

            p = self.p_selection(self.p_init, i_iter, self.n_iters)
            for i_img in range(x_best_curr.shape[0]):
                s = int(round(np.sqrt(p * n_features / c)))
                s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
                center_h = np.random.randint(0, h - s)
                center_w = np.random.randint(0, w - s)

                x_curr_window = x_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
                x_best_curr_window = x_best_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
                # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
                while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], min_val, max_val) - x_best_curr_window) < 10**-7) == c*s*s:
                    deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = np.random.choice([-self.eps, self.eps], size=[c, 1, 1])

            x_new = np.clip(x_curr + deltas, min_val, max_val)

            logits = self.logits(x_new)
            loss = self.loss(y_curr, logits, self.IsTargeted, loss_type='cross_entropy')
            margin = self.loss(y_curr, logits, self.IsTargeted, loss_type='margin_loss')

            idx_improved = loss < loss_min_curr
            loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
            margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr
            idx_improved = np.reshape(idx_improved, [-1, *[1]*len(x.shape[:-1])])
            x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
            n_queries[idx_to_fool] += 1

            acc = (margin_min > 0.0).sum() / n_ex_total
            acc_corr = (margin_min > 0.0).mean()
            mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min <= 0]), np.median(n_queries[margin_min <= 0])
            avg_margin_min = np.mean(margin_min)
            # print('{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.2f} med#q={:.1f}, avg_margin={:.2f} (n_ex={}, eps={:.3f})'.
            #     format(i_iter+1, acc, acc_corr, mean_nq_ae, median_nq_ae, avg_margin_min, x.shape[0], self.eps))

            if acc == 0:
                break

        return torch.tensor(x_best.astype("float32"))