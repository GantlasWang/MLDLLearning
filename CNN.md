CNN

把所有像素点都作为输入的话，weight数量太大了，会过拟合。根据图像识别的特点，采用选取receptive field的方法，每个field作为每个neuron的输入。

简化方法：

1、设置field检测特征

field之间可以重叠。同一field可以作为不同neuron的输入。field大小随意。通道随意（有这种做法）。形状随意（矩形）。

经典方式：

全通道、kernel size 3*3、一组neuron守备一个field、移动步长：stride、超出范围补零或任意值（padding）

2、neuron共享参数

可以使守备不同filed的neuron共享参数，因为输入不同，所以输出也不同。

经典方式：

不同field的一组neuron中，从第一个开始一一对应，参数相同，每两两一组的neuron的参数为一个filter。

![](C:\Users\Gantlas\AppData\Roaming\Typora\typora-user-images\image-20220731104254641.png)

由外及内，网络的弹性越来越小。



CNN另一种解释

有许多的filter（3*3），各个filter要在图片中找特定的pattern。filter内的数值是未知的（因为是model的参数）。

具体检测方式为：

各filter从左上角开始，与filed做矩阵乘法，filed * filter，每次步长为stride，最后算出各个新矩阵，矩阵中最大值一般为filter所识别模式的位置，这些矩阵叫feature map。可以认为，经过一次卷积后，生成了一张新“图片”，通道数为filter个数。

每次卷积filter之间的关系：（以rgb图片为例）

第一次filter格式为：3 * 3 * 3（3 * 3的大小，三个通道），假设第一次共64个filter。

第二次filter格式为：3 * 3 * 64（3 * 3的大小，64个通道）

即：3 * 3 * channel



共享参数本质上就是一个filter扫过一张图片，即一个filter的卷积过程。

![image-20220731111313916](C:\Users\Gantlas\AppData\Roaming\Typora\typora-user-images\image-20220731111313916.png)

3、pooling

不是一个layer，可以认为是一个类似relu的function，作用是在channel不变的情况下，把图片变小。以2*2的max pooling为例，经过一次卷积后，在2 * 2的范围内找最大值作为代表。



实操中，一般是卷积和pooling交替进行，最后进行做一个flatten（拉直最终的图片）但是有时候不适合用pooling，比如alpha go就没有使用，因为pooling后棋局就变了。

![image-20220731111945352](C:\Users\Gantlas\AppData\Roaming\Typora\typora-user-images\image-20220731111945352.png)



对于图片放大缩小问题，用spatial transformer layer方法。可以认为是单独的一层，放在整个CNN之前，参与到整个训练过程，也可以在CNN内部生成feature map后用。

![image-20220731162957836](C:\Users\Gantlas\AppData\Roaming\Typora\typora-user-images\image-20220731162957836.png)

通过矩阵变换达成平移、旋转、缩放。

会因为矩阵只能是整数，结果是离散值。而微小移动时，由于四舍五入，梯度为0，不能用梯度下降。

![image-20220731164012200](C:\Users\Gantlas\AppData\Roaming\Typora\typora-user-images\image-20220731164012200.png)

解决方法是用插值（interpolation），四个方向上各像素的插值，如下。

![image-20220731164157808](C:\Users\Gantlas\AppData\Roaming\Typora\typora-user-images\image-20220731164157808.png)



![image-20220731153909878](C:\Users\Gantlas\AppData\Roaming\Typora\typora-user-images\image-20220731153909878.png)

Q：如何在loss尽可能低的同时，使现实与理想的差距小？

A：采用deep learning

deep比shallow更有效率，同样的效果，deep参数量小，需要的训练资料少，不容易过拟合。