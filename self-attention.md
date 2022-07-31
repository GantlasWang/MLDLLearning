self-attention

输入为一组向量，且向量个数可变时，如文字、音频。

考虑整个sequence，输入多少向量就输出多少向量，输出的向量即为考虑整个sequence后的。

步骤：

1、对于输入a1，寻找与a1相关的向量，关联程度attention score设置一个参数α，有两种方式。dp方法比较常用。![image-20220731183552554](C:\Users\Gantlas\AppData\Roaming\Typora\typora-user-images\image-20220731183552554.png)

一般实操中也与自己做关联性，之后对所有α做softmax。

2、对于a2，得到各α后，各输入向量再乘一个矩阵得到各v，各α与各v相乘后相加得到b2。如下

![image-20220731184842117](C:\Users\Gantlas\AppData\Roaming\Typora\typora-user-images\image-20220731184842117.png)



可以从矩阵角度解释。



multi-head self-attention

![image-20220731190233333](C:\Users\Gantlas\AppData\Roaming\Typora\typora-user-images\image-20220731190233333.png)

![image-20220731190244365](C:\Users\Gantlas\AppData\Roaming\Typora\typora-user-images\image-20220731190244365.png)



位置的信息很重要时，还要为位置编码，每个位置有一个自己的向量ei，计算时加到ai上去。



self-attention与CNN的比较

CNN可以认为是简化版的s-a，关联区域为filed，而s-a则为整张图片。

论文：On the Relationship between Self-Attention and Convolutional Layers       2019.11

s-a模型复杂，弹性大，需要更大的数据量，数据量小会过拟合。

CNN模型相对简单，弹性小，不需要那么大的数据量，数据量增大到一定程度收益越来越少。