## 完整说明文档

完整说明文档，请参阅 https://aisafety.readthedocs.io/zh_CN/latest/



## Install

1. Install pytorch

   The code is tested on python 3.6.5 and torch 1.6.0

2. Install requirement

   `` pip install -r requirement.txt``

3. Clone the repository

   `` git clone git@github.com:DIG-Beihang/AISafety.git``



## Usage

#### Test

1. change root to ` test`

   `` cd test``

2. run `testimport.py`

   `` python testimport.py``

- --attack_method: 用于生成对抗样本的攻击算法
- --evaluation_method：用于执行评测的评测算法
- --Data_path：传入数据集路径
- --...：更多参数说明请参看完整说明文档中介绍