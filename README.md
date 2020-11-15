- # 重明

  本项目为重明开源项目的算法内核，重明开源项目是一个用于对抗攻击全流程评测算法学习研究的 __Python库__，其主要研究内容为集成对抗攻击和噪声攻击相关的攻击算法、评测算法、加固防御算法。可灵活测试数据集质量、算法训练、评估和部署等算法全生命周期各项指标。

  - 集成大量攻击、评测、防御加固算法
  - 提供多种可解释性分析工具
  - 提供完备的扩展和使用接口

  

  ### 文档

  **教程**：如果您正在寻找教程，请查看`AISafety/test/`路径下相关示例文件

  **文档**：我们提供了完备的API说明文档以及教程：[说明文档链接](https://aisafety.readthedocs.io/zh_CN/latest/)

  

  ### 下载及使用

  #### STEP 1. 获取项目

  Python环境要求

  ```
  Python 3.6.5及以上
  ```

  克隆本项目并安装依赖：

  ```
  git clone http://git.openi.org.cn/OpenI/AISafety.git
  cd AISafety/
  pip install requirements.txt
  ```

  #### STEP 2. 数据准备

  重明开源项目的`Datasets/`中提供了Cifar10和ILSVR2012-ImageNet数据集。用户可使用上述数据集，或按照完整API文档，进行数据集扩展。

  用户需要使用所选数据集的训练集，执行模型训练过程。由于空间所限，本项目中统一不提供数据集对应的训练集，仅给出测试集以供测试。

  #### STEP 3. 快速开始

  使用`cd test`进入test目录。重明开源项目提供了几个示例的算法文件。如测试在FGSM攻击算法下，Resnet20模型的鲁棒性结果：

  ```python
  # 使用接口文件默认参数
  python testimport.py
  
  # 自定义参数调用
  python testimport.py --attack_method "FGSM" --evaluation_method "ALDp" --model_dir ""
  ```

  上述命令均将测试ResNet20模型，在FGSM算法攻击下，ALDp指标的评测结果变化。并将结果存在`AISafety/test/temp`。

  有关更多示例和用法（例如，如何扩展模型或算法，如何传入参数），请浏览[完备API接口文档](https://aisafety.readthedocs.io/zh_CN/latest/)。

  

  ### 仍有疑问？

  如果你有任何疑问或需要帮助，请随时联系我们。

  

  ### 协议

  AISafety基于MIT协议, 关于协议的更多信息，请参看 [LICENSE](https://git.openi.org.cn/OpenI/AISafety/src/branch/main/LICENSE) 文件。

  