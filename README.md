# Chinese_NER
或许这是最简单的NER代码了，基于bert+bilstm+crf(Python3.6+Kashgari)

推荐Kashgari这个API，封装了很多方法，只需要一个命令即可调用机器学习方法，训练代码总共20行左右，评估和预测代码更是10行左右

# 环境
我用的是云服务器8核16G的云服务器，看官方教程2核心4G的服务器也能跑

python=3.6

Kashgari=1.1.5

# 数据源
从github上下载的人们日报数据

# 训练
需要下载Bert-Base，chinese并解压到项目根目录下

运行python train.py

# 评估
运行python evaluate.py

# 预测
运行python predict.py

# 参考教程
https://www.jianshu.com/p/1d6689851622

https://eliyar.biz/nlp_chinese_bert_ner/

https://github.com/BrikerMan/Kashgari/blob/v2-trunk/docs/tutorial/text-labeling.md
