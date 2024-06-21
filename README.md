- classify.py调用说明

1. 如果已经将待测图片处理为tensor了,那么直接调用类ViolenceClass即可

```python
# 假设imgs : torch.Tensor已经完成处理
from classify import ViolenceClass
my_class = ViolenceClass(0) 
my_class.classify(imgs)
```

​    2.如果不想预处理图片,也可以使用命令行格式调用classify.py,需要给出图片保存的文件夹路径,路径可以多个

```
python classify.py --version 0 --dir your_directory1 your_directory2 
```

- ckpt说明

本实验中训练了resnet18和resnet34 2个模型,并重写model类,从而可以以指定**version**的方式自主选择模型结构,便捷train和test过程

| version | 说明          |
| ------- | ------------- |
| 0       | res18最佳ckpt |
| 1       | res34最佳ckpt |

- 数据集

| 名称  | 说明                              |
| ----- | --------------------------------- |
| test1 | 从原train dataset中随机分出1000张 |
| test2 | 由test1经过对抗加噪和高斯加噪得到 |
| test3 | aigc dataset                      |
| train | 剔除test1的原train dataset        |
| val   | 原val dataset                     |

- 代码结构

| 名称                | 说明                                      |
| ------------------- | ----------------------------------------- |
| classify.py         | 应用类ViolenceClass                       |
| dataset_for_fgsm.py | 为对抗加噪设计的dataset类                 |
| dataset.py          | 整合train,val,test1,2,3 datast            |
| fgsm_test.py        | 给test1对抗加噪                           |
| get_test2_noise.py  | 给test2_fgsm高斯加噪                      |
| model.py            | 模型结构                                  |
| test.py             | 测试                                      |
| train.py            | 训练                                      |
| utils.py            | tools,具体是由version来指定ckpt的保存地址 |

