
import numpy as np
import torch
from torch.autograd import Variable
from numpy import linalg as LA

from EvalBox.Attack.AdvAttack.attack import Attack


class SIGNOPT(Attack):
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
        super(SIGNOPT, self).__init__(model, device, IsTargeted)

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
        self.eps = float(kwargs.get("epsilon", 0.1))
        self.eps_iter = float(kwargs.get("eps_iter", 0.01))
        self.num_steps = int(kwargs.get("num_steps", 15))
        self.decay_factor = float(kwargs.get("decay_factor", 1.0))
        self.k = 200

    def generate(self, xs=None, ys=None):
        """
        @description: 
        @param {
            xs:
            ys:
        } 
        @return: adv_xs{numpy.ndarray}
        """
        copy_xs = np.copy(xs.numpy())
        xs_min, xs_max = copy_xs - self.eps, copy_xs + self.eps
        momentum = 0
        targeted = self.IsTargeted
        adv_xs = np.zeros_like(copy_xs)
        for i, image in enumerate(copy_xs):
            adv_xs[i] = self.attack_untargeted(image, ys[i], distortion=self.eps, query_limit=1000, iterations=100)[0][0]
        adv_xs = np.clip(adv_xs, xs_min, xs_max)
        adv_xs = np.clip(adv_xs, 0, 1)
        return torch.tensor(adv_xs)

    def predict_label(self, x):
        var_xs = Variable(
                torch.from_numpy(x).float().to(self.device), requires_grad=True
            )
        outputs = self.model(var_xs).cpu().detach().numpy()
        return np.argmax(outputs, 1)[0]

    def attack_untargeted(self, x0, y0, alpha = 0.2, beta = 0.001, iterations = 1000, query_limit=80000,
                          distortion=None, stopping=1e-8):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """
        x0=np.expand_dims(x0,0)
        model = self.model
        query_count = 0
        ls_total = 0
        
        if (self.predict_label(x0) != y0):
            return x0, 0, True, 0, None

        #### init theta by Gaussian: Calculate a good starting point.
        num_directions = 100
        best_theta, g_theta = None, float('inf')
        init_thetas, init_g_thetas = [], []

        for i in range(num_directions):
            query_count += 1
            theta = np.random.randn(*x0.shape)

            if self.predict_label(x0+theta) != y0:
                initial_lbd = LA.norm(theta.flatten(), np.inf)
                theta /= initial_lbd

                lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
                query_count += count
                
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    init_thetas.append(best_theta)
                    init_g_thetas.append(g_theta)


        ## fail if cannot find a adv direction within 200 Gaussian
        if g_theta == float('inf'):
            # logging.info("Couldn't find valid initial, failed")
            return x0, 0, False, query_count, None

        for g_theta, best_theta in zip(init_g_thetas, init_thetas): # successful attack upon initialization
            if distortion is None or g_theta < distortion:
                x_adv = x0 + g_theta*best_theta
                target = self.predict_label(x_adv)
                # logging.info("\nSucceed: distortion {:.4f} target"
                #     " {:d} queries {:d} LS queries {:d}".format(g_theta, target, query_count, 0))
                return x_adv, g_theta, True, query_count, best_theta


        #### begin attack
        init_thetas = list(reversed(init_thetas))
        init_g_thetas = list(reversed(init_g_thetas))
        query_count_init = query_count
        ls_total_init = ls_total
        alpha_init = alpha
        beta_init = beta
        for init_id in range(1):
            best_theta = init_thetas[init_id]
            g_theta = init_g_thetas[init_id]
            ls_total = ls_total_init
            query_count = query_count_init
            alpha = alpha_init
            beta  = beta_init

            #### Begin Gradient Descent.
            xg, gg = best_theta, g_theta
            distortions = [gg]
            for i in range(iterations):
                sign_gradient, grad_queries = self.sign_grad_v1(x0, y0, xg, initial_lbd=gg, h=beta)

                ## Line search of the step size of gradient descent)
                min_theta, min_g2, alpha, ls_count = self.line_search(model, x0, y0, gg, xg, alpha, sign_gradient, beta)

                if alpha < 1e-6:
                    alpha = 1.0
                    # logging.info("Warning: not moving, beta is {0}".format(beta))
                    beta = beta * 0.1
                    if (beta < 1e-8):
                        break

                xg, g2 = min_theta, min_g2
                gg = g2

                query_count += (grad_queries + ls_count)
                ls_total += ls_count
                distortions.append(gg)

                if query_count > query_limit:
                    break
                
                # if i % 5 == 0:
                    # logging.info("Iteration {:3d} distortion {:.6f} num_queries {:d}".format(i+1, gg, query_count))

                if distortion is not None and gg < distortion:
                    # logging.info("Success: required distortion reached")
                    break

            ## check if successful
            if distortion is None or gg < distortion:
                target = self.predict_label(x0 + gg*xg)
                # logging.info('Succeed at init {}'.format(init_id))
                # logging.info("Succeed distortion {:.4f} target"
                #              " {:d} queries {:d} LS queries {:d}\n".format(gg, target, query_count, ls_total))
                return x0 + gg*xg, gg, True, query_count, xg
            
        return x0, 0, False, query_count, xg


    def line_search(self, model, x0, y0, gg, xg, alpha, sign_gradient, beta):
        ls_count = 0
        min_theta = xg
        min_g2 = gg
        for _ in range(15):
            new_theta = xg - alpha * sign_gradient
            new_theta /= LA.norm(new_theta.flatten(), np.inf)
            new_g2, count = self.fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
            ls_count += count
            alpha = alpha * 2
            if new_g2 < min_g2:
                min_theta = new_theta
                min_g2 = new_g2
            else:
                break

        if min_g2 >= gg:
            for _ in range(15):
                alpha = alpha * 0.25
                new_theta = xg - alpha * sign_gradient
                new_theta /= LA.norm(new_theta.flatten(), np.inf)
                new_g2, count = self.fine_grained_binary_search_local(
                    model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                ls_count += count
                if new_g2 < gg:
                    min_theta = new_theta
                    min_g2 = new_g2
                    break
        return min_theta, min_g2, alpha, ls_count


    def sign_grad_v1(self, x0, y0, theta, initial_lbd, h=0.001, lr=5.0, D=4, target=None,
                     sample_type='gaussian'):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        K = self.k
        sign_grad = np.zeros(theta.shape)
        queries = 0

        for iii in range(K):
            if sample_type == 'gaussian':
                u = np.random.randn(*theta.shape)
            # else:
            #     logging.info('ERROR: UNSUPPORTED SAMPLE_TYPE: {}'.format(sample_type)); exit(1)
            u /= LA.norm(u.flatten(), np.inf)

            new_theta = theta + h*u;
            new_theta /= LA.norm(new_theta.flatten(), np.inf)
            sign = 1

            # Targeted case.
            if (target is not None and 
                self.predict_label(x0+initial_lbd*new_theta) == target):
                sign = -1

            # Untargeted case
            if (target is None and
                self.predict_label(x0+initial_lbd*new_theta) != y0): # success
                sign = -1
            queries += 1
            sign_grad += u*sign
        sign_grad /= K
        
        return sign_grad, queries

     
    ##########################################################################################

    def fine_grained_binary_search_local(self, model, x0, y0, theta, initial_lbd = 1.0, lin_search_ratio=0.01, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
        tol = max(tol, 2e-9)
        if self.predict_label(x0+lbd*theta) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*(1 + lin_search_ratio)
            nquery += 1
            while self.predict_label(x0+lbd_hi*theta) == y0:
                lbd_hi = lbd_hi*(1 + lin_search_ratio)
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*(1 - lin_search_ratio)
            nquery += 1
            while self.predict_label(x0+lbd_lo*theta) != y0 :
                lbd_lo = lbd_lo*(1 - lin_search_ratio)
                nquery += 1
                
        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.predict_label(x0 + lbd_mid*theta) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery


    def fine_grained_binary_search_local(self, model, x0, y0, theta, initial_lbd = 1.0, lin_search_ratio=0.01, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
        tol = max(tol, 2e-9)
        if self.predict_label(x0+lbd*theta) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*(1 + lin_search_ratio)
            nquery += 1
            while self.predict_label(x0+lbd_hi*theta) == y0:
                lbd_hi = lbd_hi*(1 + lin_search_ratio)
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*(1 - lin_search_ratio)
            nquery += 1
            while self.predict_label(x0+lbd_lo*theta) != y0 :
                lbd_lo = lbd_lo*(1 - lin_search_ratio)
                nquery += 1
        
        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.predict_label(x0 + lbd_mid*theta) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery


    def fine_grained_binary_search(self, model, x0, y0, theta, initial_lbd, current_best, tol=1e-5):
        nquery = 0
        if initial_lbd > current_best: 
            if self.predict_label(x0+current_best*theta) == y0:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd
        
        lbd_hi = lbd
        lbd_lo = 0.0
        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if self.predict_label(x0 + lbd_mid*theta) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery
