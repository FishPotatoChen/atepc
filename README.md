# ATEPC

## 介绍
毕业设计，方面级情感分析模型。


## 安装教程

分别运行`APC_train.py`和`ATE_train.py`便可训练两个模型。

在运行上面两个文件后，会在`.\logs`文件夹中看到训练结果。

将评论按行写入`comment.txt`文件，运行`ATEPC.py`，便可得到评论的方面词提取结果与情感极性分类结果。

## 使用说明

运行`APC_train_SDR.py`可以看到语义相对距离对模型的影响，结果在`.\logs`文件夹中。

修改`json`文件可以更改模型参数。

`.\atepc_datasets`文件夹中是训练数据和测试数据。

`.\output`保存训练好的模型。

只有运行之后才会出现`.\output`和`.\logs`文件夹。