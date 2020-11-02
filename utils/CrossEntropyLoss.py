import numpy as np


class CrossEntropyLoss:
    def __init__(self, weight=None, size_average=True):
        """
        初始化参数，因为要实现 torch.nn.CrossEntropyLoss 的两个比较重要的参数

        :param weight: 给予每个类别不同的权重
        :param size_average: 是否要对 loss 求平均
        """

        self.weight = weight
        self.size_average = size_average

    def __call__(self, input, target):
        """
        计算损失
        这个方法让类的实例表现的像函数一样，像函数一样可以调用

        :param input: (batch_size, C)，C是类别的总数
        :param target: (batch_size, 1)
        :return: 损失
        """

        batch_loss = 0.0
        input = input.detach().numpy()
        target = target.detach().numpy()
        print(input.shape[0], "input.shape[0]")
        for i in range(input.shape[0]):

            numerator = np.exp(input[i, target[i]])  # 分子
            denominator = np.sum(np.exp(input[i, :]))  # 分母

            # 计算单个损失
            loss = -np.log(numerator / denominator)
            if self.weight:
                loss = self.weight[target[i]] * loss
            print("单个损失： ", loss)

            # 损失累加
            batch_loss += loss

        # 整个 batch 的总损失是否要求平均
        if self.size_average == True:
            batch_loss /= input.shape[0]

        return batch_loss

    def forward(self, input, target):
        return CrossEntropyLoss(input, target)
