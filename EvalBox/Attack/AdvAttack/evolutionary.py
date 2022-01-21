
import numpy as np
import torch
from torch.autograd import Variable
import collections

from EvalBox.Attack.AdvAttack.attack import Attack

def fn_mean_square_distance(x1, x2):
        return np.mean((x1 - x2) ** 2) #/ ((x_max - x_min) ** 2)
        
class Evolutionary(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, **kwargs):
        """
        @description: Untargeted Momentum Iterative Method
        @param {
            model:
            device:
            kwargs:
        } 
        @return: None
        """
        super(Evolutionary, self).__init__(model, device, IsTargeted)

        self.criterion = torch.nn.CrossEntropyLoss()

        self._parse_params(**kwargs)

    def _parse_params(self, **kwargs):
        """
        @description: 
        @param {
            epsilon:
            eps_iter:
            num_steps:
            decay_factor:
        } 
        @return: None
        """
        
        self.num_queries = int(kwargs.get("num_queries", 200))
        self.decay_factor = float(kwargs.get("decay_factor", 0.99))
        self.mu = float(kwargs.get("mu", 1e-2))
        self.c = float(kwargs.get("c", 1e-3))
        self.sigma = float(kwargs.get("sigma", 3e-2))

    def generate(self, xs=None, ys=None):
        """
        @description: 
        @param {
            xs:
            ys:
        } 
        @return: adv_xs{numpy.ndarray}
        """
        xs_adv = torch.clone(xs)
        batch_size = xs.shape[0]
        for i in range(batch_size):
            x_adv, dist = self.attack_single(xs[i].numpy(), ys[i])
            print(dist)
            xs_adv[i] = torch.tensor(x_adv)
        return xs_adv
    
    def predict(self, x):
        var_xs = Variable(
                torch.from_numpy(np.expand_dims(x, 0)).float().to(self.device), requires_grad=True
            )
        outputs = self.model(var_xs).cpu().detach().numpy()
        return np.argmax(outputs, 1)[0]

    def attack_single(self, x, y):
        ret = x
        model = self.model
        x_dtype = x.dtype
        x_min = 0.0
        x_max = 1.0
        mu = self.mu
        c = self.c
        sigma = self.sigma
        decay_factor = self.decay_factor
        
        x_label = self.predict(x)
        if x_label != y:
            return x, 0
        while True:
            x_adv = np.random.uniform(x_min, x_max, size=x.shape)
            if self.predict(x_adv)!= y:
                break
        dist = fn_mean_square_distance(x, x_adv)
        stats_adversarial = collections.deque(maxlen=30)

        pert_shape = x.shape

        N = np.prod(pert_shape)
        K = int(N / 20)

        evolution_path = np.zeros(pert_shape, dtype=x_dtype)
        diagonal_covariance = np.ones(pert_shape, dtype=x_dtype)

        # x_adv_label = self.predict(x_adv)

        step = 0 
        while step <= self.num_queries:
            step += 1
            unnormalized_source_direction = x - x_adv
            source_norm = np.linalg.norm(unnormalized_source_direction)

            selection_probability = diagonal_covariance.reshape(-1) / np.sum(diagonal_covariance)
            selected_indices = np.random.choice(N, K, replace=False, p=selection_probability)

            perturbation = np.random.normal(0.0, 1.0, pert_shape).astype(x.dtype)
            factor = np.zeros([N], dtype=x.dtype)
            factor[selected_indices] = 1
            perturbation *= factor.reshape(pert_shape) * np.sqrt(diagonal_covariance)

            perturbation_large = perturbation

            biased = x_adv + mu * unnormalized_source_direction
            candidate = biased + sigma * source_norm * perturbation_large / np.linalg.norm(perturbation_large)
            candidate = x - (x - candidate) / np.linalg.norm(x - candidate) * np.linalg.norm(x - biased)
            candidate = np.clip(candidate, x_min, x_max)

            candidate_label = self.predict(candidate)

            is_adversarial = candidate_label != y

            stats_adversarial.appendleft(is_adversarial)

            if is_adversarial:
                ret = candidate
                new_x_adv = candidate
                new_dist = fn_mean_square_distance(new_x_adv, x)
                evolution_path = decay_factor * evolution_path + np.sqrt(1 - decay_factor ** 2) * perturbation
                diagonal_covariance = (1 - c) * diagonal_covariance + c * (evolution_path ** 2)
            else:
                new_x_adv = None

            if new_x_adv is not None:
                abs_improvement = dist - new_dist
                rel_improvement = abs_improvement / dist
                x_adv, dist = new_x_adv, new_dist
                x_adv_label = candidate_label

            if len(stats_adversarial) == stats_adversarial.maxlen:
                p_step = np.mean(stats_adversarial)
                mu *= np.exp(p_step - 0.2)
                stats_adversarial.clear()
        
        return ret, new_dist