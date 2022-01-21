import numpy as np
import torch
from torch.autograd import Variable
import torch.autograd as autograd

from EvalBox.Attack.AdvAttack.attack import Attack

import cv2

class NA(Attack):
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
        super(NA, self).__init__(model, device, IsTargeted)

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
        self.n_samples = int(kwargs.get("n_samples", 300))
        self.sigma = float(kwargs.get("sigma", 0.1))
        self.learning_rate = float(kwargs.get("lr", 0.02))
        self.eps = float(kwargs.get("eps", 0.1))
        self.n_iter = int(kwargs.get("n_iter", 500))
        

    def softmax(self, x):
        return np.divide(np.exp(x),np.sum(np.exp(x),-1,keepdims=True))
    
    def torch_arctanh(self, x, eps=1e-6):
        x *= (1. - eps)
        return (np.log((1 + x) / (1 - x))) * 0.5

    def scale_input(self, xs, target_size):
        temp = []
        for x in xs:
            temp.append(cv2.resize(x.transpose(1,2,0), dsize=target_size, interpolation=cv2.INTER_LINEAR).transpose(2,0,1))
        return np.array(temp)

    def generate(self, xs, ys):
        input_size = xs.shape[2:]
        need_scale = input_size[0] != 32 or input_size[1] != 32
        npop = self.n_samples     # population size
        sigma = self.sigma   # noise standard deviation
        alpha = self.learning_rate  # learning rate
        boxplus = 0.5
        boxmul = 0.5
        epsi = self.eps
        epsilon = 1e-30
        test_loss = 0
        correct = 0
        total = 0
        totalImages = 0
        succImages = 0
        faillist = []
        adv_xs = xs.clone()


        model = self.model
        model.eval()


        start = 0
        end = xs.shape[0]
        total = 0
        successlist = []
        printlist = []


    
        for i in range(start, end):

            success = False
            #print('evaluating %d of [%d, %d)' % (i, start, end))
            inputs, targets = xs[i].detach().numpy(), ys[i].detach().numpy()


            input_var = autograd.Variable(torch.unsqueeze(torch.tensor(inputs.astype('float32'), device=self.device), 0), volatile=True)

            modify = np.random.randn(1,3,32,32) * 0.001


            logits = model(input_var).data.cpu().numpy()

            probs = self.softmax(logits)

            if np.argmax(probs[0]) != targets:
                print('skip the wrong example ', i)
                continue

            totalImages += 1

            for runstep in range(self.n_iter):
                Nsample = np.random.randn(npop, 3,32,32)# np.random.randn(npop, 3,32,32)

                modify_try = modify.repeat(npop,0) + sigma*Nsample
                if need_scale:
                    modify_try = self.scale_input(modify_try, input_size)
                newimg = self.torch_arctanh((inputs-boxplus) / boxmul)
                #print('newimg', newimg,flush=True)
                inputimg = np.tanh(newimg+modify_try) * boxmul + boxplus

                
                if runstep % 10 == 0:
                    if need_scale:
                        realmodify = self.scale_input(modify, input_size)
                    else:
                        realmodify = modify
                    realinputimg = np.tanh(newimg+realmodify) * boxmul + boxplus
                    realdist = realinputimg - (np.tanh(newimg) * boxmul + boxplus)
                    realclipdist = np.clip(realdist, -epsi, epsi)
                    #print('realclipdist :', realclipdist, flush=True)
                    realclipinput = realclipdist + (np.tanh(newimg) * boxmul + boxplus)
                    l2real =  np.sum((realclipinput - (np.tanh(newimg) * boxmul + boxplus))**2)**0.5

                    

                    input_var = autograd.Variable(torch.tensor(realclipinput.astype('float32'), device=self.device), volatile=True)
                    
                    outputsreal = model(input_var).data.cpu().numpy()[0]
                    outputsreal = self.softmax(outputsreal)
                    # print('probs ', np.sort(outputsreal)[-1:-6:-1])
                    # print('target label ', np.argsort(outputsreal)[-1:-6:-1])
                    # print('negative_probs ', np.sort(outputsreal)[0:3:1])

                    if (np.argmax(outputsreal) != targets) and (np.abs(realclipdist).max() <= epsi):
                        succImages += 1
                        success = True
                        # print('clipimage succImages: '+str(succImages)+'  totalImages: '+str(totalImages))
                        # print('lirealsucc: '+str(realclipdist.max()))
                        successlist.append(i)
                        printlist.append(runstep)

                        break
                dist = inputimg - (np.tanh(newimg) * boxmul + boxplus)
                clipdist = np.clip(dist, -epsi, epsi)
                clipinput = (clipdist + (np.tanh(newimg) * boxmul + boxplus)).reshape(npop,3,32,32)
                target_onehot =  np.zeros((1,10))

                target_onehot[0][targets]=1.
                clipinput = np.squeeze(clipinput)
                clipinput = np.asarray(clipinput, dtype='float32')
                input_var = autograd.Variable(torch.tensor(clipinput, device=self.device), volatile=True)
                outputs = model(input_var).data.cpu().numpy()
                outputs = self.softmax(outputs)

                target_onehot = target_onehot.repeat(npop,0)



                real = np.log((target_onehot * outputs).sum(1)+epsilon)
                other = np.log(((1. - target_onehot) * outputs - target_onehot * 10000.).max(1)[0]+epsilon)

                loss1 = np.clip(real - other, 0.,1000)

                Reward = 0.5 * loss1

                Reward = -Reward

                A = (Reward - np.mean(Reward)) / (np.std(Reward)+1e-7)


                modify = modify + (alpha/(npop*sigma)) * ((np.dot(Nsample.reshape(npop,-1).T, A)).reshape(3,32,32))
            # if not success:
            #     faillist.append(i)
            #     print('failed:', faillist)
            # else:
            #     print('successed:',successlist)
            adv_xs[i]=torch.tensor(realclipinput)
        #print(faillist)
        success_rate = succImages/float(totalImages)
        #print('run steps: ',printlist)
        #print('succ rate', success_rate)
        return adv_xs