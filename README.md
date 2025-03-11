<div style="text-align: center;">

# 郭子新

</div>

## 教育经历

**大庆师范学院 电子信息工程**

主修课程：c语言程序设计，数据结构，微机原理，嵌入式Linux，数据库系统概论等。

荣誉奖项：二等奖学金

## 个人自述
对AI有所了解，熟悉常见的机器学习算法，例如线性回归，逻辑回归，Transform。

<br><br>

# 项目1:基于百度AI的文本智能识别系统

## 主要代码
[项目代码](https://gitee.com/mndsdrz/fushi20240230/tree/master/%E9%A1%B9%E7%9B%AE1%20%E5%9F%BA%E4%BA%8E%E7%99%BE%E5%BA%A6AI%E7%9A%84%E6%96%87%E6%9C%AC%E6%99%BA%E8%83%BD%E8%AF%86%E5%88%AB%E7%B3%BB%E7%BB%9F)

## 项目背景

在日常生活中，有些照片上的文字，或者某些网页，软件上的文字无法复制，想要获取文字，比较简单的方法就是拍照或者截屏，然后再由人逐个打字，过程麻烦，并且可能由于打字过多中间会出现一些错误，比如错字，语序颠倒。此系统主要基于百度AI的算法能力主要实现文字识别，文字纠错，图片文字提取，图片信息保存功能（其实是这些功能我用得不频繁，开会员感觉有点浪费，调用百度AI的api按量付费，使用这些功能会比较便宜，而且只要能用网页上网的设备都可以用这些功能）。

## 项目主要技术栈

Java，SpringBoot，Vue+Element，MySql，mybatis，百度AI

## 主要功能

集成百度AI能力，实现图片文字识别，图片物品识别，文字纠错，历史使用记录。

<br><br>
### 模块1：NLP纠错

***输入需要纠错的文字***

![输入需要纠错的文字](./public/Img/输入需要纠错的文字.png)

<br>

***可以获得纠错后正确的句子***

![以获得纠错后正确的句子](./public/Img/可以获得纠错后正确的句子.png)

<br><br>
### 模块2：图像识别

***上传照片***

![上传照片](./public/Img/上传照片上传照片.png)

<br>

***这里我选择上传这张照片***

![选择上传这张照片](./public/Img/选择上传这张照片.png)

<br>

***图片上传好后就自动识别图片内物品种类***

![自动识别图片内物品种类](./public/Img/自动识别图片内物品种类.png)

<br><br>
### 模块3：文字识别

***上传需要识别文字的图片***

![需要识别文字的图片](./public/Img/需要识别文字的图片.png)

<br>

***我上传下面这张图***

![我上传下面这张图](./public/Img/我上传下面这张图.png)

<br>

***图片上传好后就自动识别图片内的文字***

![自动识别图片内的文字](./public/Img/自动识别图片内的文字.png)

<br><br>
### 模块4：历史记录

***由于本项目使用了MySql来记录使用数据，使用上述的三个模块后，获得的三条历史记录***

![获得的三条历史记录](./public/Img/获得的三条历史记录.png)

<br>

***同时，针对历史记录可以任意选择删除某一条记录，我点击箭头所指的这条记录。***

![自主选择删除某一条记录](./public/Img/自主选择删除某一条记录-1.png)

<br>

***点击删除后，mybatis就会将这条历史记录便从数据库中删除，网页上也就不会再显示这条记录了***

![这条历史记录便从数据库中删除](./public/Img/这条历史记录便从数据库中删除.png)

&nbsp;

# 项目2:使用c语言完成一元多项式回归预测纸张数(二次回归)
[项目代码](https://gitee.com/mndsdrz/fushi20240230/tree/master/%E9%A1%B9%E7%9B%AE2%EF%BC%9A%E4%BD%BF%E7%94%A8c%E8%AF%AD%E8%A8%80%E5%AE%8C%E6%88%90%E4%B8%80%E5%85%83%E5%A4%9A%E9%A1%B9%E5%BC%8F%E5%9B%9E%E5%BD%92%E9%A2%84%E6%B5%8B%E7%BA%B8%E5%BC%A0%E6%95%B0(%E4%BA%8C%E6%AC%A1%E5%9B%9E%E5%BD%92))
## 项目背景

![项目背景](./public/Img/项目背景.png)

在一次比赛中，题目要求如上图，设计一个纸张计数测量显示装置，我们组使用的芯片是stm32，由于此芯片性能不是很高，但对于计算结果实时性要求较高，所以选择使用较为快速的c语言为此装置编程。基本原理就是两极板类似一个"电容"，当两极板放入不同张数的A4纸时，两极板之间距离发生改变，这个电容的电容值会随着距离的增大同时增大。由于这种装置不是很稳定，外界条件影响很大，两极板相同距离之间的电容，在不同温度，不同环境下电容值都会有较大的变化。所以在比赛前就提前就测量好不同纸张数对应的电容值误差会比较大。

## 负责模块

为了让这个装置能够适应不同环境，我负责编写一个函数模块，让这个装置能够"自学习"，在比赛现场，通过两极板依次放入不同张数的A4纸，就能预测更大范围内的纸张张数，无需重新测量电容值，修改代码，烧录程序。，

## 项目内容

函数参数传入一个数组，这个数组内会传进来多个电容值（装置一秒会读取36次电容值，一次读取一秒，大约测量10次），使用c语言完成使用梯度下降法的一元多项式回归 (二次回归，因为电容值跳变不是线性的，而是接近一元二次函数的)，通过此方法训练过后的模型，可以通过10次左右的测量，就可以预测大约一百张左右的A4纸。

## 主要代码
```c

// 定义了一个数据点结构体 DataPoint，并且使用了一个简单的梯度下降函数gradientDescent来更新参数 a、b 和 c。
// 这个函数会根据数据点和当前参数计算梯度，并更新参数以最小化误差。

#include <stdio.h>
#include <math.h>

//测量次数
#define MEASURE 10

//每次测量的电容值个数
#define Capacitance 36

// 数据点数量
#define N 100

// 学习率
#define LEARNING_RATE 0.01

// 最大迭代次数
#define MAX_ITER 1000

// 数据点结构体
typedef struct {
    double x;//纸张数
    double y;//电容值
} DataPoint;

// 梯度下降算法
void gradientDescent(DataPoint data[MEASURE], double* a, double* b, double* c) {
    double da, db, dc;
    for (int iter = 0; iter < MAX_ITER; iter++) {
        da = db = dc = 0;
        for (int i = 0; i < N; i++) {
            // 计算预测值和实际值之间的差异
            double y_pred = (*a) * data[i].x * data[i].x + (*b) * data[i].x + *c;
            double error = y_pred - data[i].y;

            // 计算梯度
            da += error * data[i].x * data[i].x;
            db += error * data[i].x;
            dc += error;
        }

        // 更新参数
        *a -= LEARNING_RATE * da / N;
        *b -= LEARNING_RATE * db / N;
        *c -= LEARNING_RATE * dc / N;
    }
}

//每次测量MEASURE次，每次Capacitance个电容值
//paper是一个数组，里面记录着与data储存的电容值对应的纸张数
void trainModel(double data[MEASURE][Capacitance], int paper[MEASURE], double* a, double* b, double* c) {
    //为了防止抖动，单次测量的最高和最低10个数去掉，剩下的16个数求平均值作为单次测量的真实值
    int i, j, k;
    int n = Capacitance;
    double temp;
    for (k = 0; k < MEASURE; k++) {
        for (i = 0; i < n; i++) {
            // 最后i个元素已经在正确位置，不需要再比较
            for (j = 0; j < n - i - 1; j++) {
                if (data[k][j] > data[k][j + 1]) {
                    temp = data[k][j];
                    data[k][j] = data[k][j + 1];
                    data[k][j + 1] = temp;
                }
            }
        }
    }

    int index;
    double sum = 0;

    // 初始化数据点
    DataPoint value[MEASURE];

    for (index = 0; index < MEASURE; index++, sum = 0) {
        value[index].x = paper[index];
        for (int indexValue = 10; indexValue < n - 10; indexValue++)
            sum += data[index][indexValue];
        value[index].y = sum / (n-20);
    }
    
    gradientDescent(value, a, b, c);
}

int main() {
    // 参数初始化
    double a = 1, b = 0.1, c = 0.1;

    double dataPf[MEASURE][Capacitance];
    int paper[MEASURE];
    // 训练模型
    trainModel(dataPf,paper, &a, &b, &c);

    // 输出结果
    printf("多项式回归模型: y = %lfx^2 + %lfx + %lf\n", a, b, c);

    return 0;
}
```

&nbsp;

# 项目3:美国金价预测
[项目代码](https://gitee.com/mndsdrz/fushi20240230/tree/master/%E9%A1%B9%E7%9B%AE3%EF%BC%9A%E9%87%91%E4%BB%B7%E9%A2%84%E6%B5%8B%E6%A8%A1%E5%9E%8B)
## 项目背景

在完成项目使用c语言完成一元多项式回归预测纸张后，觉得机器学习是一件奇妙的事情，能够通过某些数学方法预测一些事情，同时使用c语言完成某些机器学习算法比较繁琐，我尝试使用主流的python实现这些算法。偶尔看到金价大涨的消息，我想尝试能否预测金价的价格，于是在网络上寻找到有关贵金属价格的数据集，尝试能否预测金价。

## 项目内容

数据集本身并不是很理想，不同价格对应日期不同。

![数据集](./public/Img/数据集.png)

于是我对数据集清洗后，通过单变量与多变量线性回归，利用pandas进行数据处理，利用numpy进行数据的维度处理，利用sk-learn进行线性回归模型的训练，使得单变量线性回归mse值为73671.35，R方值为0.75，多变量线性回归mse值为1842.97，R方值为0.99。

y 与y'的均方误差(MSE):

![均方误差(MSE)](./public/Img/均方误差(MSE).png)

R方值(R2):

![R方值](./public/Img/R方值.png)

## 数据清洗

基本思路就是首先将不同价格分为四个部分，GBPGold.csv，USDGold.csv，USDPlatinum.csv，USDSilver.csv。

![分文件](./public/Img/分文件.png)

使得每个文件都类似上图，只有日期与价格，然后使用pandas，读取四个文件，通过日期为关键词，筛选有相同日期的留下，没有的就删除。

[数据清洗代码](https://gitee.com/mndsdrz/fushi20240230/blob/master/%E9%A1%B9%E7%9B%AE3%EF%BC%9A%E9%87%91%E4%BB%B7%E9%A2%84%E6%B5%8B%E6%A8%A1%E5%9E%8B/%E9%A2%84%E6%B5%8B%E9%87%91%E4%BB%B7%E6%95%B0%E6%8D%AE%E9%9B%86/gold-predict.py)

```python
import pandas as pd

dfGBPGold = pd.read_csv('GBPGold.csv', skiprows=0)
dfUSDGold = pd.read_csv('USDGold.csv', skiprows=0)
dfUSDPlatinum = pd.read_csv('USDPlatinum.csv', skiprows=0)
dfUSDSilver = pd.read_csv('USDSilver.csv', skiprows=0)

dateIndexGBPGold={}
dateIndexUSDGold={}
dateIndexUSDPlatinum={}
dateIndexUSDSilver={}

GBPGoldIndex=[]
USDGoldIndex=[]
USDPlatinumIndex=[]
USDSilverIndex=[]

date=[]

for indexGBPGold, rowGBPGold in dfGBPGold.iterrows():
    dateIndexGBPGold[rowGBPGold['GBPGold']]=  indexGBPGold
    date.append(rowGBPGold['GBPGold'])


for indexUSDGold, rowUSDGold in dfUSDGold.iterrows():
    dateIndexUSDGold[rowUSDGold['USDGold']]=  indexUSDGold

for indexUSDPlatinum, rowUSDPlatinum in dfUSDPlatinum.iterrows():
    dateIndexUSDPlatinum[rowUSDPlatinum['USDPlatinum']]=  indexUSDPlatinum

for indexUSDSilver, rowUSDSilver in dfUSDSilver.iterrows():
    dateIndexUSDSilver[rowUSDSilver['USDSilver']]=  indexUSDSilver

for a in date:
    if(dateIndexGBPGold.get(a)!=None and dateIndexUSDGold.get(a)!=None and dateIndexUSDPlatinum.get(a)!=None and dateIndexUSDSilver.get(a)!=None):
        GBPGoldIndex.append(dateIndexGBPGold['{}'.format(a)])
        USDGoldIndex.append(dateIndexUSDGold['{}'.format(a)])
        USDPlatinumIndex.append(dateIndexUSDPlatinum['{}'.format(a)])
        USDSilverIndex.append(dateIndexUSDSilver['{}'.format(a)])

newdfGBPGold=dfGBPGold.loc[GBPGoldIndex]
newdfUSDGold=dfUSDGold.loc[USDGoldIndex]
newdfUSDPlatinum=dfUSDPlatinum.loc[USDPlatinumIndex]
newdfUSDSilver=dfUSDSilver.loc[USDSilverIndex]

newdfGBPGold.to_csv('newdfGBPGold.csv', index=False)
newdfUSDGold.to_csv('newdfUSDGold.csv', index=False)
newdfUSDPlatinum.to_csv('newdfUSDPlatinum.csv', index=False)
newdfUSDSilver.to_csv('newdfUSDSilver.csv', index=False)
```


## 线性回归训练

使用anaconda配置环境，使用Jupyter进行分步骤训练，同时可以显示图像，方便观察数据之间的关系。通过训练，可知通过仅使用线性回归方法，单因子预测金价误差较大，而通过多因子预测准确率会大大提高。

[线性回归训练代码](https://gitee.com/mndsdrz/fushi20240230/blob/master/%E9%A1%B9%E7%9B%AE3%EF%BC%9A%E9%87%91%E4%BB%B7%E9%A2%84%E6%B5%8B%E6%A8%A1%E5%9E%8B/GoldPrice.ipynb)

```python
#数据集来源https://auronum.co.uk/
#数据集内有10004条数据
#单位是1盎司价格
#本项目预测尝试使用线性回归的方式预测美国金价(USDGoldPrice)
#同时使用英国金价(GBPGoldPrice)美国铂价/白金价(USDPlatinumPrice)美国银价(USDSilverPrice)作为多因子，增加预测准确性
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from datetime import datetime
data = pd.read_csv('GoldPrice.csv')
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>USDGoldPrice</th>
      <th>GBPGoldPrice</th>
      <th>USDPlatinumPrice</th>
      <th>USDSilverPrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985/1/29</td>
      <td>303.2</td>
      <td>272.22</td>
      <td>271.5</td>
      <td>6.23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985/1/30</td>
      <td>302.9</td>
      <td>269.01</td>
      <td>272.5</td>
      <td>6.21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985/1/31</td>
      <td>306.6</td>
      <td>271.21</td>
      <td>283.0</td>
      <td>6.37</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985/2/1</td>
      <td>303.6</td>
      <td>270.11</td>
      <td>277.0</td>
      <td>6.23</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985/2/4</td>
      <td>301.2</td>
      <td>270.38</td>
      <td>271.0</td>
      <td>6.16</td>
    </tr>
  </tbody>
</table>
</div>




```python
#数据散点图展示
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter

# 将日期列转换为datetime对象
data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d')

# 设置图表的格式
plt.figure(figsize=(10, 5))
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y/%m/%d'))

# 绘制USDGoldPrice的散点图
plt.scatter(data['date'], data['USDGoldPrice'], color='blue', label='USDGoldPrice',s=1)

# 绘制GBPGoldPrice的散点图
plt.scatter(data['date'], data['GBPGoldPrice'], color='green', label='GBPGoldPrice',s=1)

# 绘制USDPlatinumPrice的散点图
plt.scatter(data['date'], data['USDPlatinumPrice'], color='red', label='USDPlatinumPrice',s=1)

# 绘制USDSilverPrice的散点图
data['USDSilverPrice']*= 50 
#扩大50倍，方便观察数据之间关系
plt.scatter(data['date'], data['USDSilverPrice'], color='orange', label='USDSilverPrice',s=1)

# 添加图例
plt.legend()

# 美化图表
plt.gcf().autofmt_xdate() # 自动调整日期标签
plt.title('Price vs date')
plt.xlabel('date')
plt.ylabel('Price')

# 显示图表
plt.show()

```


    
![png](./public/GoldPrice_files/GoldPrice_1_0.png)
    



```python
#单因子预测
```


```python
# 将日期转换为距离某个固定日期的天数，这里以最小日期为基准
data['day'] = (data['date'] - data['date'].min()).dt.days

# 准备X和y数据
X = data['day'].values.reshape(-1, 1)  # 特征变量（日期数值）
y = data['USDGoldPrice'].values  # 目标变量（价格）
```


```python
#线性回归模型
LR1 = LinearRegression()
#训练模型
LR1.fit(X,y)
```




<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>




```python
#多因子预测
```


```python
#删除掉USDGoldPrice列
X_multi = data.drop(['USDGoldPrice'],axis=1).drop(['date'],axis=1)

#第二个线性模型
LR_multi = LinearRegression()
#train the model
LR_multi.fit(X_multi,y)
```




<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" checked><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>




```python
#两种预测方式结果比较
```


```python
#两种预测方式结果比较
#单因子预测
y_predict_1 = LR1.predict(X)
#多因子预测
y_predict_multi = LR_multi.predict(X_multi)

# 绘制原始数据散点图
plt.scatter(data['date'], data['USDGoldPrice'], label='RowData', color='black',s=1)

# 单因子回归线回归线
plt.plot(data['date'], y_predict_1, color='blue',label='SinglePredict', linewidth=1)
# 多因子回归线
plt.plot(data['date'], y_predict_multi, color='red',label='MultiPredict', linewidth=1)

# 添加图例
plt.legend()

# 设置图表格式
plt.gcf().autofmt_xdate() # 自动调整日期标签
plt.title('USDGoldPrice vs date')
plt.xlabel('date')
plt.ylabel('Price')
plt.show()


```


    
![png](./public/GoldPrice_files/GoldPrice_8_0.png)
    



```python
#模型评估

print("SinglePredict")
# 输出单因子模型参数
print('斜率:', LR1.coef_)
print('截距:', LR1.intercept_)
# 输出单因子模型评估
from sklearn.metrics import mean_squared_error,r2_score
mean_squared_error_1 = mean_squared_error(y,y_predict_1)
r2_score_1 = r2_score(y,y_predict_1)
print("MES:{}".format(mean_squared_error_1))
print("R2:{}".format(r2_score_1))

#输出多因子模型评估
mean_squared_error_multi = mean_squared_error(y,y_predict_multi)
r2_score_multi = r2_score(y,y_predict_multi)
print("\nMultiPredict")
print("MES:{}".format(mean_squared_error_multi))
print("R2:{}".format(r2_score_multi))
```

    SinglePredict
    斜率: [0.1165265]
    截距: -24.309574304240527
    MES:73671.35883842765
    R2:0.7525502113843792
    
    MultiPredict
    MES:1842.976220972269
    R2:0.9938097507159686
    

## 类似项目训练

### 多分类算法的性别分类

使⽤逻辑回归、K近邻算法、决策树实现身⾼体重性别分类。逻辑回归模型， K近邻算法都预测是男性，而决策树分类器预测为女性，所以在此程序中，决策树分类器并不适合通过身高体重判断性别。

[多分类算法的性别分类](https://gitee.com/mndsdrz/fushi20240230/blob/master/%E9%A1%B9%E7%9B%AE3%EF%BC%9A%E9%87%91%E4%BB%B7%E9%A2%84%E6%B5%8B%E6%A8%A1%E5%9E%8B/%E7%B1%BB%E4%BC%BC%E9%A1%B9%E7%9B%AE%E8%AE%AD%E7%BB%83/%E5%A4%9A%E5%88%86%E7%B1%BB%E7%AE%97%E6%B3%95%E7%9A%84%E6%80%A7%E5%88%AB%E5%88%86%E7%B1%BB.ipynb)

&nbsp;

# 项目4:基于预训练模型的搜索排序系统实现
[项目代码](https://gitee.com/mndsdrz/fushi20240230/tree/master/%E9%A1%B9%E7%9B%AE4%EF%BC%9A%E5%9F%BA%E4%BA%8E%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%90%9C%E7%B4%A2%E6%8E%92%E5%BA%8F%E7%B3%BB%E7%BB%9F%E5%AE%9E%E7%8E%B0)
## 项目背景

搜索引擎的排序算法对用户体验至关重要。通常，与查询最相关的文档会被排在搜索结果的前列，以增加用户点击的可能性。一个高效的排序算法能够准确评估文档与查询的相关性，并据此排序，以更好地满足用户的搜索需求。

传统的排序方法主要基于文本匹配来检索信息，但由于搜索时通常使用自然语言，这可能导致准确性不足，使得不同的关键词可能指向相同的查询意图，从而增加了不相关文档在搜索结果中的比例。

ERNIE通过大规模数据预训练，能够有效理解语言的深层含义，因此在处理各种自然语言处理问题时表现出色。将ERNIE应用于搜索排序，能够为用户提供更精确、更符合搜索需求的结果，从而提升搜索引擎的整体体验。

## 系统的功能和工作流程

这个搜索引擎利用了百度的ERNIE大模型来构建一个专门针对文献的搜索系统。在这个系统中，用户的查询会经历召回和排序两个阶段，以确保最终展示的文献信息是按照相关性高低进行排序的。

具体工作流程如下：首先，使用召回模型将整个语料库的文献转换为向量形式，并创建一个基于近似最近邻（ANN）的索引库。当用户提出查询时，系统会通过召回模型从语料库中找到50篇与查询最相关的文献。接着，排序模型会对这些召回的文献进行更精确的相似度评估，并按照这个评估结果重新排序，最终将这个排序结果呈现给用户作为搜索输出。

### 召回模型的结构及每层的作用

召回模型由三个关键部分组成：首先是ERNIE预训练模型，它负责将输入数据转化为语义向量；其次是线性层，它将ERNIE产生的768维向量降维到用户设定的尺寸，这不仅减少了计算负担，也增加了模型的灵活性，使其可以适应不同的应用需求；最后是dropout层，这一层的作用是帮助模型避免过度拟合数据。

### 排序模型的结构及每层的作用

排序模型中包括三层，第一层是预训练模型ERNIE，用于把输入的数据转换成语义向量；第二层是dropout层，用于在训练过程中随机"丢弃"一部分神经元，防止模型过拟合，提高模型的泛化能力；第三层是相似度计算层，用于将预训练模型生成的向量转换为一个相似度得分，这个层直接关联到最终的排序任务。在模型训练时，通过学习句子对之间的相对相似度，优化模型参数以提高排序的准确性。

## 主要代码文件

### **召回模块**[(recall)](https://gitee.com/mndsdrz/fushi20240230/tree/master/%E9%A1%B9%E7%9B%AE4%EF%BC%9A%E5%9F%BA%E4%BA%8E%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%90%9C%E7%B4%A2%E6%8E%92%E5%BA%8F%E7%B3%BB%E7%BB%9F%E5%AE%9E%E7%8E%B0/recall)

### 1.[finetune.ipynb](https://gitee.com/mndsdrz/fushi20240230/blob/master/%E9%A1%B9%E7%9B%AE4%EF%BC%9A%E5%9F%BA%E4%BA%8E%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%90%9C%E7%B4%A2%E6%8E%92%E5%BA%8F%E7%B3%BB%E7%BB%9F%E5%AE%9E%E7%8E%B0/recall/finetune.ipynb) 利用literature\_search\_data数据集进行微调

```python
# 这个代码文件是把一个预训练模型ERNIE1.0构成的神经网络 利用in-batch-negative方法进行训练
```


```python
import abc
import sys
import argparse
import os
import random
import time
import numpy as np
from scipy import stats
import pandas as pd
from tqdm import tqdm 
from scipy.special import softmax
from scipy.special import expit
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import inference
import paddlenlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.downloader import get_path_from_url
from visualdl import LogWriter
from data import convert_pairwise_example
```


```python
paddle.set_device("cpu")
```


```python
def read_text_pair(data_path): 
  with open(data_path, 'r', encoding='utf-8') as f: 
        for line in f: 
            data = line.rstrip().split("\t") 
            if len(data) != 2: 
                continue
            yield {'text_a': data[0], 'text_b': data[1]} 
                                                         

```


```python

train_set_path='recall_dataset/train.csv' 
train_ds = load_dataset(read_text_pair, data_path=train_set_path, lazy=False) 
```


```python
print(type(train_ds))
```


```python

for i in range(3):
    print(train_ds[i])
```


```python
# 下面开始构造训练数据的加载器
```


```python
MODEL_NAME="ernie-1.0"

tokenizer = paddlenlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)
```


```python
def convert_example(example, tokenizer, max_seq_length=512): 
    result = []
    
    for key, text in example.items(): 
        encoded_inputs = tokenizer(text=text, max_seq_len=max_seq_length)
        input_ids = encoded_inputs["input_ids"] 
        token_type_ids = encoded_inputs["token_type_ids"] 
        result += [input_ids, token_type_ids]

    return result 
```


```python
#演示

a=[]
for key, text in train_ds[0].items():
    print(text)
    encoded_inputs = tokenizer(text=text, max_seq_len=512)
    print(encoded_inputs)
    print(encoded_inputs["input_ids"])
    print(encoded_inputs["token_type_ids"])
    a += [encoded_inputs["input_ids"], encoded_inputs["token_type_ids"]]
    print('------------------------------------')
    
print(a)
```


```python
from functools import partial

trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=64) 
```


```python
def batchify_fn(samples):
    fn = Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),   
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'),  
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'),  
    )

    processed_samples = fn(samples) 

    result = []
    for data in processed_samples:
        result.append(data) 

    return result
```


```python
batch_sampler = paddle.io.DistributedBatchSampler(train_ds, batch_size=64, shuffle=True)
train_data_loader = paddle.io.DataLoader(dataset=train_ds.map(trans_func), batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)
```


```python
# 下面开始搭建召回模型
```


```python
pretrained_model = paddlenlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)
```


```python
from base_model import SemanticIndexBase

class SemanticIndexBatchNeg(SemanticIndexBase): 
    def __init__(self, pretrained_model, dropout=None, margin=0.3, scale=30, output_emb_size=None):
        super().__init__(pretrained_model, dropout, output_emb_size)

        self.margin = margin
        self.scale = scale

    def forward(self, query_input_ids,    title_input_ids,    query_token_type_ids=None, query_position_ids=None, query_attention_mask=None,    title_token_type_ids=None, title_position_ids=None, title_attention_mask=None):
        query_cls_embedding = self.get_pooled_embedding(query_input_ids, query_token_type_ids, query_position_ids, query_attention_mask) 

        title_cls_embedding = self.get_pooled_embedding(title_input_ids, title_token_type_ids, title_position_ids, title_attention_mask)    
        
        cosine_sim = paddle.matmul(query_cls_embedding, title_cls_embedding, transpose_y=True)  
        
        margin_diag = paddle.full(shape=[query_cls_embedding.shape[0]], fill_value=self.margin, dtype="float32") 

        cosine_sim = cosine_sim - paddle.diag(margin_diag)

        cosine_sim = cosine_sim * self.scale

        labels = paddle.arange(0, query_cls_embedding.shape[0], dtype='int64') 
        labels = paddle.reshape(labels, shape=[-1, 1]) 

        loss = F.cross_entropy(input=cosine_sim, label=labels)

        return loss
```


```python
model = SemanticIndexBatchNeg(pretrained_model, margin=0.1, scale=20, output_emb_size=256)
```


```python
# 下面开始定义模型训练用到的各种参数，并进行模型训练
```


```python
epochs=3 
num_training_steps = len(train_data_loader) * epochs 
```


```python
lr_scheduler = LinearDecayWithWarmup(5E-5, num_training_steps, 0.0) 
```


```python
decay_params = [
        p.name for n, p in model.named_parameters() 
        if not any(nd in n for nd in ["bias", "norm"]) 
    ] 
```


```python
optimizer = paddle.optimizer.AdamW( 
    learning_rate=lr_scheduler, 
    parameters=model.parameters(), 
    weight_decay=0.0, 
    apply_decay_param_fun=lambda x: x in decay_params)
```


```python
save_dir='model_param'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

global_step = 0 
tic_train = time.time()

for epoch in range(1, epochs + 1): 
    for step, batch in enumerate(train_data_loader, start=1): 
        query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids = batch 

        loss = model(query_input_ids=query_input_ids, title_input_ids=title_input_ids, query_token_type_ids=query_token_type_ids, title_token_type_ids=title_token_type_ids)

        global_step += 1 
        if global_step % 10 == 0: 
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, 10 / (time.time() - tic_train))) 
            tic_train = time.time() 

        loss.backward() 
        optimizer.step() 
        lr_scheduler.step() 
        optimizer.clear_grad() 

        if global_step % 10 == 0: 
            save_path = os.path.join(save_dir, "model_%d" % global_step)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_param_path = os.path.join(save_path, 'model_state.pdparams') 
            paddle.save(model.state_dict(), save_param_path) 
            tokenizer.save_pretrained(save_path) 
```


### 2.[recall.ipynb](https://gitee.com/mndsdrz/fushi20240230/blob/master/%E9%A1%B9%E7%9B%AE4%EF%BC%9A%E5%9F%BA%E4%BA%8E%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%90%9C%E7%B4%A2%E6%8E%92%E5%BA%8F%E7%B3%BB%E7%BB%9F%E5%AE%9E%E7%8E%B0/recall/recall.ipynb) 对测试集query进行召回

```python
from functools import partial
import argparse
import os
import sys
import random
import time

import numpy as np
import hnswlib
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.utils.log import logger
import paddlenlp

from base_model import SemanticIndexBase
from data2 import convert_example, create_dataloader
from data2 import gen_id2corpus
from ann_util import build_index
```


```python
paddle.set_device("cpu") 
```


```python
# 下面开始构建模型并载入模型参数
```


```python
model_name = "ernie-1.0"

pretrained_model = paddlenlp.transformers.AutoModel.from_pretrained(model_name)
```


```python
model = SemanticIndexBase(pretrained_model, output_emb_size=256)
```


```python
params_path = "model_param/model_180/model_state.pdparams" 

if params_path and os.path.isfile(params_path): 
    state_dict = paddle.load(params_path) 
    model.set_dict(state_dict) 
    print("Loaded parameters from %s" % params_path) 
else:
    raise ValueError("Please set params_path with correct pretrained model file")
```


```python
# 下面加载语料库文件，并利用语料库中的数据来构造ANN索引库
```


```python
tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained(model_name)
```


```python
trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=60)
```


```python
def batchify_fn(samples):
    fn = Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  
    )

    processed_samples = fn(samples) 

    result = []
    for data in processed_samples:
        result.append(data) 

    return result
```


```python
corpus_file = "recall_dataset/corpus.csv" 

id2corpus = gen_id2corpus(corpus_file) 
```


```python
print(type(id2corpus))
```


```python
for i in range(10):
    print(id2corpus[i]) 
```


```python
corpus_list = []
for idx, text in id2corpus.items():
    corpus_list.append({idx: text}) 
```


```python
for i in range(10):
    print(corpus_list[i])
```


```python
corpus_ds = MapDataset(corpus_list) 
```


```python
print(type(corpus_ds))
```


```python
for i in range(10):
    print(corpus_ds[i])
```


```python
batch_sampler = paddle.io.BatchSampler(corpus_ds, batch_size=64, shuffle=False)

corpus_data_loader = paddle.io.DataLoader(dataset=corpus_ds.map(trans_func), batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)
```


```python
#如果需要从头构建索引，就运行下面这段代码。

output_emb_size = 256
hnsw_max_elements = 1000000 
hnsw_ef = 100 
hnsw_m = 100 

final_index = build_index(output_emb_size, hnsw_max_elements, hnsw_ef, hnsw_m, corpus_data_loader, model)

save_index_dir = "index_file" 
if not os.path.exists(save_index_dir):
    os.makedirs(save_index_dir)

save_index_path = os.path.join(save_index_dir, "final_index.bin") 
final_index.save_index(save_index_path)
```


```python
'''
#如果有现成的索引文件final_index.bin,就运行这段代码
save_index_path = "index_file/final_index.bin"
output_emb_size = 256
final_index = hnswlib.Index(space="ip", dim=output_emb_size) 
final_index.load_index(save_index_path) 
'''
```


```python
# 下面获取验证数据集中的所有query
```


```python
def get_query_text(similar_text_pair_file): 
    querys = []
    with open(similar_text_pair_file, "r", encoding="utf-8") as f:
        for line in f:
            splited_line = line.rstrip().split("\t") 
            if len(splited_line) != 2: 
                continue

            if not splited_line[0] or not splited_line[1]: 
                continue

            querys.append({"text": splited_line[0]}) 

    return querys
```


```python
similar_text_pair_file = "recall_dataset/dev.csv" 

query_list = get_query_text(similar_text_pair_file) 
```


```python
print(type(query_list))
```


```python
print(query_list)
```


```python
query_ds = MapDataset(query_list) 
```


```python
batch_sampler = paddle.io.BatchSampler(query_ds, batch_size=64, shuffle=False)

query_data_loader = paddle.io.DataLoader(dataset=query_ds.map(trans_func), batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)
```


```python
query_embedding = model.get_semantic_embedding(query_data_loader) 
```


```python
# 下面针对验证集中的query进行召回，生成召回结果文件
```


```python
recall_result_dir = "recall_result_file" 
 
if not os.path.exists(recall_result_dir): 
    os.mkdir(recall_result_dir)

recall_result_file = "recall_result.txt" 

recall_result_file = os.path.join(recall_result_dir, recall_result_file) 
```


```python
# 下面正式开始召回

with open(recall_result_file, "w", encoding="utf-8") as f: 
    for batch_index, batch_query_embedding in enumerate(query_embedding): 
        recalled_idx, cosine_sims = final_index.knn_query(batch_query_embedding.numpy(), 50) 

        batch_size = len(cosine_sims)

        for row_index in range(batch_size):
            text_index = 64 * batch_index + row_index 。
            for idx, doc_idx in enumerate(recalled_idx[row_index]):
                f.write( "{}\t{}\t{}\n".format(query_list[text_index]["text"], id2corpus[doc_idx], 1.0 - cosine_sims[row_index][idx] ) )
```

### 3.[evaluate.ipynb](https://gitee.com/mndsdrz/fushi20240230/blob/master/%E9%A1%B9%E7%9B%AE4%EF%BC%9A%E5%9F%BA%E4%BA%8E%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%90%9C%E7%B4%A2%E6%8E%92%E5%BA%8F%E7%B3%BB%E7%BB%9F%E5%AE%9E%E7%8E%B0/recall/evaluate.ipynb) 对召回的数据计算recall@N

```python
import argparse
import numpy as np
import time
```


```python
def recall(rs, N=10): 
    """
    例子：
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> recall(rs, N=1)
    0.333333
    >>> recall(rs, N=2)
    >>> 0.6666667
    >>> recall(rs, N=3)
    >>> 1.0
    """
    
    recall_flags = [np.sum(r[0:N]) for r in rs] 
                                                
    return np.mean(recall_flags) 
                                 
```


```python
text2similar = {}

similar_text_pair = "recall_dataset/dev.csv"

with open(similar_text_pair, "r", encoding="utf-8") as f: 
    for line in f:
        text, similar_text = line.rstrip().split("\t")
        text2similar[text] = similar_text 
```


```python
rs = [] 

recall_result_file = "recall_result_file/recall_result.txt"
recall_num = 50 

with open(recall_result_file, "r", encoding="utf-8") as f: 
    relevance_labels = []
    for index, line in enumerate(f): 
        if index % recall_num == 0 and index != 0: 
            rs.append(relevance_labels)
            relevance_labels = [] 

        query, recalled_text, cosine_sim = line.rstrip().split("\t") 

        if text2similar[query] == recalled_text: 
            relevance_labels.append(1) 
        else:
            relevance_labels.append(0)
        
```


```python
recalls = [1, 5, 10, 20, 50] 
recall_N = [] 

for topN in recalls:
    R = round(100 * recall(rs, N=topN), 3) 
    recall_N.append(R) 
```


```python
result_tsv_file = "recall_result_file/result.tsv"

with open(result_tsv_file, "w", encoding="utf-8") as f: 
    res = []

    for i in range(len(recalls)): 
        N = recalls[i] 
        recall_val = recall_N[i] 
        print("recall@{}={}".format(N, recall_val)) 
        res.append(str(recall_val)) 
    
    f.write("\t".join(res) + "\n") 
```

在执行evaluate.ipynb的过程中生成的result.tsv 记录recall@1, 5, 10, 20, 50的值

### **排序模块**[（rank）](https://gitee.com/mndsdrz/fushi20240230/tree/master/%E9%A1%B9%E7%9B%AE4%EF%BC%9A%E5%9F%BA%E4%BA%8E%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%90%9C%E7%B4%A2%E6%8E%92%E5%BA%8F%E7%B3%BB%E7%BB%9F%E5%AE%9E%E7%8E%B0/rank)

### 1.[train\_pairwise](https://gitee.com/mndsdrz/fushi20240230/blob/master/%E9%A1%B9%E7%9B%AE4%EF%BC%9A%E5%9F%BA%E4%BA%8E%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%90%9C%E7%B4%A2%E6%8E%92%E5%BA%8F%E7%B3%BB%E7%BB%9F%E5%AE%9E%E7%8E%B0/rank/train_pairwise.ipynb): 训练一个排序模型，并在最后进行评估

```python
import argparse
import os
import random
import time
from functools import partial

import numpy as np
import paddle
import pandas as pd
from data import convert_pairwise_example
from model import PairwiseMatching
from tqdm import tqdm

from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModel, AutoTokenizer, LinearDecayWithWarmup
```


```python
# 下面加载训练集和验证集
```


```python
def read_train(data_path): 
    with open(data_path, 'r', encoding='utf-8') as f: 
        flag = 0
        for line in f: 
            if flag != 0: 
                data = line.rstrip().split("\t") 
                if len(data) != 3: 
                    continue
                yield {'query': data[0], 'title': data[1], 'neg_title': data[2]} 
                                                                                   
            flag = 1
```


```python
def read_dev(data_path): 
    with open(data_path, 'r', encoding='utf-8') as f: 
        flag = 0
        for line in f: 
            if flag != 0: 
                data = line.rstrip().split("\t") 
                if len(data) != 3: 
                    continue
                yield {'query': data[0], 'title': data[1], 'label': data[2]} 
                                                                               
            flag = 1
```


```python
train_file = "rank_dataset/train_pairwise.csv"
train_ds = load_dataset(read_train, data_path=train_file, lazy=False) 

dev_file = "rank_dataset/dev_pairwise.csv"
dev_ds = load_dataset(read_dev, data_path=dev_file, lazy=False) 
```


```python
# 输出训练数据集的前3条数据看一下
for i in range(3):
    print(train_ds[i])
```


```python
# 输出验证数据集的前3条数据看一下
for i in range(3):
    print(dev_ds[i])
```


```python
model_name = "ernie-3.0-medium-zh"

tokenizer = AutoTokenizer.from_pretrained(model_name) 
```


```python
trans_func_train = partial(convert_pairwise_example, tokenizer=tokenizer, max_seq_length=128, phase="train")

trans_func_eval = partial(convert_pairwise_example, tokenizer=tokenizer, max_seq_length=128, phase="eval")
```


```python
#演示
example = train_ds[0]
print(example)

query, pos_title, neg_title = example["query"], example["title"], example["neg_title"]
print(query)
print(pos_title)
print(neg_title)
print('---------------------------------------------')

pos_inputs = tokenizer(text=query, text_pair=pos_title, max_seq_len=128) 
neg_inputs = tokenizer(text=query, text_pair=neg_title, max_seq_len=128) 

print(pos_inputs)
print(neg_inputs)
print('---------------------------------------------')

pos_input_ids = pos_inputs["input_ids"] 
pos_token_type_ids = pos_inputs["token_type_ids"] 

print(pos_input_ids)
print(pos_token_type_ids)
print('---------------------------------------------')

neg_input_ids = neg_inputs["input_ids"] 
neg_token_type_ids = neg_inputs["token_type_ids"] 
print(neg_input_ids)
print(neg_token_type_ids)
print('---------------------------------------------')

result = [pos_input_ids, pos_token_type_ids, neg_input_ids, neg_token_type_ids]
print(result)
```


```python
def batchify_fn_train(samples): 
    fn = Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  
    )

    processed_samples = fn(samples) 

    result = []
    for data in processed_samples:
        result.append(data) 

    return result
```


```python
def batchify_fn_eval(samples): 
    fn = Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  
        Stack(dtype="int64"), 
    )

    processed_samples = fn(samples) 

    result = []
    for data in processed_samples:
        result.append(data) 

    return result
```


```python
batch_sampler_train = paddle.io.DistributedBatchSampler(train_ds, batch_size=32, shuffle=True)

train_data_loader = paddle.io.DataLoader(dataset=train_ds.map(trans_func_train), batch_sampler=batch_sampler_train, collate_fn=batchify_fn_train, return_list=True)
```


```python
batch_sampler_dev = paddle.io.BatchSampler(dev_ds, batch_size=32, shuffle=False)

dev_data_loader = paddle.io.DataLoader(dataset=dev_ds.map(trans_func_eval), batch_sampler=batch_sampler_dev, collate_fn=batchify_fn_eval, return_list=True)
```


```python
# 下面搭建模型，并开始训练
```


```python
pretrained_model = AutoModel.from_pretrained(model_name) 
```


```python
model = PairwiseMatching(pretrained_model, margin=0.1) 
```


```python
epochs = 3 

num_training_steps = len(train_data_loader) * epochs 

lr_scheduler = LinearDecayWithWarmup(2E-5, num_training_steps, 0.0)
```


```python
decay_params = [
    p.name for n, p in model.named_parameters() 
    if not any(nd in n for nd in ["bias", "norm"])
]

```


```python
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=0.0,
    apply_decay_param_fun=lambda x: x in decay_params)
```


```python
metric = paddle.metric.Auc() 
```


```python
@paddle.no_grad() 
def evaluate(model, metric, data_loader): 
    model.eval() 
    metric.reset()
 
    for idx, batch in enumerate(data_loader): 
        input_ids, token_type_ids, labels = batch
        pos_probs = model.predict(input_ids=input_ids, token_type_ids=token_type_ids) 
        neg_probs = 1.0 - pos_probs 
        preds = np.concatenate((neg_probs, pos_probs), axis=1) 

        metric.update(preds=preds, labels=labels) 
        auc = metric.accumulate() 

    print("phase: dev, auc: {:.3}".format(auc)) 
    metric.reset()
    model.train()
```


```python
# 下面正式开始训练模型

save_dir="model_param"
global_step = 0
tic_train = time.time()

for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1): 
        pos_input_ids, pos_token_type_ids, neg_input_ids, neg_token_type_ids = batch

        loss = model(
            pos_input_ids=pos_input_ids,
            neg_input_ids=neg_input_ids,
            pos_token_type_ids=pos_token_type_ids,
            neg_token_type_ids=neg_token_type_ids,
        ) 

        global_step += 1

        if global_step % 10 == 0: 
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, 10 / (time.time() - tic_train)))
            tic_train = time.time()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()

        if global_step % 100 == 0:
            evaluate(model, metric, dev_data_loader)
            
            save_path = os.path.join(save_dir, "model_%d" % global_step) 
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_param_path = os.path.join(save_path, "model_state.pdparams") 
            paddle.save(model.state_dict(), save_param_path)
            tokenizer.save_pretrained(save_path) 
```


```python

```


```python
# 下面是把保存好的模型参数文件加载到一个新的模型中，并进行性能测试的过程

model_2 = PairwiseMatching(pretrained_model, margin=0.1)

params_path = "model_param/model_400/model_state.pdparams"

state_dict = paddle.load(params_path)
model_2.set_dict(state_dict)

evaluate(model_2, metric, dev_data_loader)
```

###  2.[predict\_pairwise](https://gitee.com/mndsdrz/fushi20240230/blob/master/%E9%A1%B9%E7%9B%AE4%EF%BC%9A%E5%9F%BA%E4%BA%8E%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%90%9C%E7%B4%A2%E6%8E%92%E5%BA%8F%E7%B3%BB%E7%BB%9F%E5%AE%9E%E7%8E%B0/rank/predict_pairwise.ipynb): 利用训练好的模型，对测试集进行排序

```python
import argparse
import os
from functools import partial

import numpy as np
import paddle
from data import convert_pairwise_example
from data import read_text_pair
from model import PairwiseMatching
import copy

from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModel, AutoTokenizer
```


```python
paddle.set_device("cpu")
```


```python
@paddle.no_grad()
def predict(model, data_loader): 
    model.eval()

    batch_probs = [] 
    for batch_data in data_loader:
        input_ids, token_type_ids = batch_data
        batch_prob = model.predict(input_ids=input_ids, token_type_ids=token_type_ids).numpy() 
        batch_probs.append(batch_prob)

    conca_batch_probs = np.concatenate(batch_probs, axis=0) 

    return conca_batch_probs
```


```python
# 下面开始搭建模型，并加载参数
```


```python
model_name = "ernie-3.0-medium-zh"

pretrained_model = AutoModel.from_pretrained(model_name)
```


```python
model = PairwiseMatching(pretrained_model)

params_path = "model_param/model_400/model_state.pdparams" 

if params_path and os.path.isfile(params_path): 
    state_dict = paddle.load(params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)
else:
    raise ValueError("Please set --params_path with correct pretrained model file")
```


```python
# 下面开始构造测试数据的加载器
```


```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
```


```python
trans_func = partial(convert_pairwise_example, tokenizer=tokenizer, max_seq_length=64, phase="predict")
```


```python
def batchify_fn(samples):
    fn = Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  
    )

    processed_samples = fn(samples) 

    result = []
    for data in processed_samples:
        result.append(data) 

    return result
```


```python
test_data_path = "rank_dataset/test_pairwise.csv" 

test_ds = load_dataset(read_text_pair, data_path=test_data_path, lazy=False)
```


```python
print(test_ds[0])
```


```python
test_ds_copy = copy.deepcopy(test_ds) 

batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=128, shuffle=False)

test_data_loader = paddle.io.DataLoader(dataset=test_ds.map(trans_func), batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)
```


```python
print(test_ds[0])
print(test_ds_copy[0])
```


```python
# 下面开始计算相似度
```


```python
y_probs = predict(model, test_data_loader)
```


```python
rank_result = [] 

for idx, prob in enumerate(y_probs): 
    text_pair = test_ds_copy[idx] 
    text_pair["pred_prob"] = prob[0] 
    rank_result.append(text_pair) 
    print(text_pair) 
```


```python
rank_result.sort(key=lambda x: x['pred_prob'], reverse=True) # 把rank_result中的各元素，按照'pred_prob'键对应的值从大到小排序

for i in rank_result:
    print(i)
```

### **总程序-两模块合并**[(search_system)](https://gitee.com/mndsdrz/fushi20240230/blob/master/%E9%A1%B9%E7%9B%AE4%EF%BC%9A%E5%9F%BA%E4%BA%8E%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%90%9C%E7%B4%A2%E6%8E%92%E5%BA%8F%E7%B3%BB%E7%BB%9F%E5%AE%9E%E7%8E%B0/search_system/search.ipynb)

```python
from functools import partial
import argparse
import os
import sys
import random
import time

import numpy as np
import hnswlib
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.utils.log import logger
import paddlenlp

from model import PairwiseMatching
from base_model import SemanticIndexBase
from data2 import convert_example, create_dataloader
from data import convert_pairwise_example
from data2 import gen_id2corpus
from ann_util import build_index
import copy
```


```python
# 下面是准备工作
```


```python
def batchify_fn(samples):
    fn = Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  
    )

    processed_samples = fn(samples) 

    result = []
    for data in processed_samples:
        result.append(data) 

    return result
```


```python
# 下面是召回的准备工作
```


```python
corpus_file = "corpus.csv" 

id2corpus = gen_id2corpus(corpus_file) 
```


```python
save_index_path = "final_index.bin" 
output_emb_size = 256
final_index = hnswlib.Index(space="ip", dim=output_emb_size) 
final_index.load_index(save_index_path) 
```


```python
model_name = "ernie-1.0"

tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained(model_name)
```


```python
trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=60)
```


```python
pretrained_model = paddlenlp.transformers.AutoModel.from_pretrained(model_name)

model = SemanticIndexBase(pretrained_model, output_emb_size=256)

params_path = "recall_model_state.pdparams" 

if params_path and os.path.isfile(params_path): 
    state_dict = paddle.load(params_path) 
    model.set_dict(state_dict) 
    print("Loaded parameters from %s" % params_path) 
else:
    raise ValueError("Please set params_path with correct pretrained model file")
```


```python
#下面是排序的准备工作
```


```python
@paddle.no_grad()
def predict(model, data_loader):
    model.eval()

    batch_probs = []
    for batch_data in data_loader:
        input_ids, token_type_ids = batch_data
        batch_prob = model.predict(input_ids=input_ids, token_type_ids=token_type_ids).numpy()
        batch_probs.append(batch_prob)

    conca_batch_probs = np.concatenate(batch_probs, axis=0) 

    return conca_batch_probs
```


```python
rank_model_name = "ernie-3.0-medium-zh"

rank_pretrained_model = paddlenlp.transformers.AutoModel.from_pretrained(rank_model_name)

rank_model = PairwiseMatching(rank_pretrained_model)

rank_params_path = "rank_model_state.pdparams" 

if rank_params_path and os.path.isfile(rank_params_path): 
    state_dict = paddle.load(rank_params_path)
    rank_model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)
else:
    raise ValueError("Please set --params_path with correct pretrained model file")
```


```python
rank_tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained(model_name)
```


```python
rank_trans_func = partial(convert_pairwise_example, tokenizer=rank_tokenizer, max_seq_length=64, phase="predict")
```


```python
#下面开始进行搜索
```


```python
input_query = input("请输入需要查询的文献")
```


```python
querys = []
querys.append({"text": input_query})
```


```python
print(querys)
```


```python
query_ds = MapDataset(querys)

query_batch_sampler = paddle.io.BatchSampler(query_ds, batch_size=1, shuffle=False)

query_data_loader = paddle.io.DataLoader(dataset=query_ds.map(trans_func), batch_sampler=query_batch_sampler, collate_fn=batchify_fn, return_list=True)
```


```python
query_embedding = model.get_semantic_embedding(query_data_loader) 
```


```python
recall_data = []
```


```python
# 下面正式开始召回

for batch_index, batch_query_embedding in enumerate(query_embedding): 
    recalled_idx, cosine_sims = final_index.knn_query(batch_query_embedding.numpy(), 50)                                                                           

    for idx, doc_idx in enumerate(recalled_idx[0]): 
        recall_data.append({"query": input_query, "title": id2corpus[doc_idx]})
        print( "{}\t{}\n".format(id2corpus[doc_idx], 1.0 - cosine_sims[0][idx] )                                                                         
        )
```


```python
for i in recall_data:
    print(i)
```


```python
rank_ds = MapDataset(recall_data)
```


```python
rank_ds_copy = copy.deepcopy(rank_ds)

rank_batch_sampler = paddle.io.BatchSampler(rank_ds, batch_size=16, shuffle=False)

rank_data_loader = paddle.io.DataLoader(dataset=rank_ds.map(rank_trans_func), batch_sampler=rank_batch_sampler, collate_fn=batchify_fn, return_list=True)
```


```python
y_probs = predict(rank_model, rank_data_loader)
```


```python
rank_result = [] 

for idx, prob in enumerate(y_probs): 
    text_pair = rank_ds_copy[idx] 
    text_pair["pred_prob"] = prob[0] 
    rank_result.append(text_pair) 
    print(text_pair) 

```


```python
rank_result.sort(key=lambda x: x['pred_prob'], reverse=True)

for i in rank_result:
    print("{}\t{}".format(i['title'], i['pred_prob']))
```


```python

```


```python
# 下面是可以整体运行的代码

input_query = input("请输入需要查询的文献")

querys = []
querys.append({"text": input_query})

query_ds = MapDataset(querys)

query_data_loader = create_dataloader(
    query_ds, mode="predict", batch_size=1, batchify_fn=batchify_fn, trans_fn=trans_func
)

query_embedding = model.get_semantic_embedding(query_data_loader) 

recall_data = []

# 下面正式开始召回

for batch_index, batch_query_embedding in enumerate(query_embedding): 
    recalled_idx, cosine_sims = final_index.knn_query(batch_query_embedding.numpy(), 50) 
    for idx, doc_idx in enumerate(recalled_idx[0]): 
        recall_data.append({"query": input_query, "title": id2corpus[doc_idx], "sim": 1.0 - cosine_sims[0][idx]})
        print( "{}\t{}\n".format(id2corpus[doc_idx], 1.0 - cosine_sims[0][idx] ) 
        )

rank_ds = MapDataset(recall_data)


rank_ds_copy = copy.deepcopy(rank_ds)

rank_data_loader = create_dataloader(
    rank_ds_copy, mode="predict", batch_size=16, batchify_fn=batchify_fn, trans_fn=rank_trans_func
)

y_probs = predict(rank_model, rank_data_loader)

rank_result = []

for idx, prob in enumerate(y_probs):
    text_pair = rank_ds[idx]
    text_pair["pred_prob"] = prob[0]
    rank_result.append(text_pair)

rank_result.sort(key=lambda x: x['pred_prob'], reverse=True)

for i in rank_result:
    print("{}\t{}".format(i['title'], i['pred_prob']))
```

## 相关论文研读

[Transformer](https://gitee.com/mndsdrz/fushi20240230/raw/master/%E9%A1%B9%E7%9B%AE4%EF%BC%9A%E5%9F%BA%E4%BA%8E%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%90%9C%E7%B4%A2%E6%8E%92%E5%BA%8F%E7%B3%BB%E7%BB%9F%E5%AE%9E%E7%8E%B0/Transformer.pdf)


# 大模型应用

## 1.使用python调用大模型API实现带有GUI界面的大模型对话软件
### 背景
对话平台是一个基于大模型技术的智能聊天机器人，使用python实现了GUI界面，支持用户与多种大模型（如Deepseek、通义千问等）进行实时对话，并可根据需求动态切换模型。平台旨在为用户提供高效、灵活、智能的对话体验，适用于多种场景，以满足对大模型日常需求。

### 主要功能
输入框：用户可以在输入框中输入对话内容。
![input](./public/input.png)
输出框：将用户输入的内容传递给大模型，并显示模型的回复。
![output](./public/output.png)

### 主要代码
```python
import tkinter as tk
from tkinter import scrolledtext, font
from openai import OpenAI

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("大模型聊天")
        self.root.geometry("600x500")
        self.root.configure(bg="#f0f0f0")

        # 聊天区域
        self.chat_area = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, state="disabled", bg="#ffffff", bd=0, highlightthickness=0
        )
        self.chat_area.place(x=10, y=10, width=580, height=400)
        self.chat_area.tag_configure("user", justify="right", background="#e3f2fd", relief="flat", borderwidth=5)
        self.chat_area.tag_configure("bot", justify="left", background="#f5f5f5", relief="flat", borderwidth=5)

        # 输入框
        self.input_frame = tk.Frame(root, bg="#ffffff", bd=0)
        self.input_frame.place(x=10, y=420, width=580, height=70)

        self.input_field = tk.Text(
            self.input_frame, wrap=tk.WORD, bg="#ffffff", bd=0, highlightthickness=0, font=("Arial", 12)
        )
        self.input_field.place(x=5, y=5, width=470, height=60)
        self.input_field.bind("<Return>", self.send_message)

        # 发送按钮
        self.send_button = tk.Button(
            self.input_frame, text="发送", bg="#007bff", fg="black", font=("Arial", 12), bd=0, command=self.send_message
        )
        self.send_button.place(x=480, y=5, width=90, height=60)

    def send_message(self, event=None):
        user_message = self.input_field.get("1.0", tk.END).strip()
        if user_message:
            self.display_message(user_message, "user")
            self.input_field.delete("1.0", tk.END)

            # 调用Deepseek API
            bot_response = self.get_bot_response(user_message)
            self.display_message(bot_response, "bot")

    def display_message(self, message, sender):
        self.chat_area.config(state="normal")
        if sender == "user":
            self.chat_area.insert(tk.END, f"{message}\n", "user")
        else:
            self.chat_area.insert(tk.END, f"{message}\n", "bot")
        self.chat_area.config(state="disabled")
        self.chat_area.yview(tk.END)

    def get_bot_response(self, user_message):
        client = OpenAI(
            api_key="", # 输入你的api_key
            base_url=""# 输入你的base_url
        )

        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": user_message},
            ],
            stream=False
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content
    
root = tk.Tk()
app = ChatApp(root)
root.mainloop()
```
