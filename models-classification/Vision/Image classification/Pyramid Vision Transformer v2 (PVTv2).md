# PVT v2: Improved Baselines with Pyramid Vision Transformer

Wenhai Wang<sup>1,2</sup>, Enze Xie<sup>3</sup>, Xiang Li<sup>4</sup>, Deng-Ping Fan<sup>5</sup>,  
Kaitao Song<sup>4</sup>, Ding Liang<sup>6</sup>, Tong Lu<sup>2</sup>, Ping Luo<sup>3</sup>, Ling Shao<sup>5</sup>

<sup>1</sup>Shanghai AI Laboratory <sup>2</sup>Nanjing University <sup>3</sup>The University of Hong Kong

<sup>4</sup>Nanjing University of Science and Technology <sup>5</sup>IIAI <sup>6</sup>SenseTime Research

wangwenhai@pjlab.org.cn,

## Abstract

*Transformer recently has presented encouraging progress in computer vision. In this work, we present new baselines by improving the original Pyramid Vision Transformer (PVT v1) by adding three designs, including (1) linear complexity attention layer, (2) overlapping patch embedding, and (3) convolutional feed-forward network. With these modifications, PVT v2 reduces the computational complexity of PVT v1 to linear and achieves significant improvements on fundamental vision tasks such as classification, detection, and segmentation. Notably, the proposed PVT v2 achieves comparable or better performances than recent works such as Swin Transformer. We hope this work will facilitate state-of-the-art Transformer researches in computer vision. Code is available at <https://github.com/whai362/PVT>.*

## 1. Introduction

Recent studies on vision Transformer are converging on the backbone network [8, 31, 33, 34, 23, 36, 10, 5] designed for downstream vision tasks, such as image classification, object detection, instance and semantic segmentation. To date, there have been some promising results. For example, Vision Transformer (ViT) [8] first proves that a pure Transformer can archive state-of-the-art performance in image classification. Pyramid Vision Transformer (PVT v1) [33] shows that a pure Transformer backbone can also surpass CNN counterparts in dense prediction tasks such as detection and segmentation tasks [22, 41]. After that, Swin Transformer [23], CoaT [36], LeViT [10], and Twins [5] further improve the classification, detection, and segmentation performance with Transformer backbones.

This work aims to establish stronger and more feasible baselines built on the PVT v1 framework. We report that three design improvements, namely (1) linear complexity attention layer, (2) overlapping patch embedding, and (3) convolutional feed-forward network are orthogonal to the

PVT v1 framework, and when used with PVT v1, they can bring better image classification, object detection, instance and semantic segmentation performance. The improved framework is termed as PVT v2. Specifically, PVT v2-B5<sup>1</sup> yields 83.8% top-1 error on ImageNet, which is better than Swin-B [23] and Twins-SVT-L [5], while our model has fewer parameters and GFLOPs. Moreover, GFL [19] with PVT-B2 archives 50.2 AP on COCO val2017, 2.6 AP higher than the one with Swin-T [23], 5.7 AP higher than the one with ResNet50 [13]. We hope these improved baselines will provide a reference for future research in vision Transformer.

## 2. Related Work

We mainly discuss transformer backbones related to this work. ViT [8] treats each image as a sequence of tokens (patches) with a fixed length, and then feeds them to multiple Transformer layers to perform classification. It is the first work to prove that a pure Transformer can also archive state-of-the-art performance in image classification when training data is sufficient (e.g., ImageNet-22k [7], JFT-300M). DeiT [31] further explores a data-efficient training strategy and a distillation approach for ViT.

To improve image classification performance, recent methods make tailored changes to ViT. T2T ViT [37] concatenates tokens within an overlapping sliding window into one token progressively. TNT [11] utilizes inner and outer Transformer blocks to generate pixel and patch embeddings respectively. CPVT [6] replaces the fixed size position embedding in ViT with conditional position encodings, making it easier to process images of arbitrary resolution. CrossViT [2] processes image patches of different sizes via a dual-branch Transformer. LocalViT [20] incorporates depth-wise convolution into vision Transformers to improve the local continuity of features.

To adapt to dense prediction tasks such as object de-

<sup>1</sup>PVT v2 has 6 different size variants, from B0 to B5 according to the parameter number.Figure 1: Comparison of SRA in PVT v1 and linear SRA in PVT v2.

tection, instance and semantic segmentation, there are also some methods [33, 23, 34, 36, 10, 5] to introduce the pyramid structure in CNNs to the design of Transformer backbones. PVT v1 is the first pyramid structure Transformer, which presents a hierarchical Transformer with four stages, showing that a pure Transformer backbone can be as versatile as CNN counterparts and performs better in detection and segmentation tasks. After that, some improvements [23, 34, 36, 10, 5] are made to enhance the local continuity of features and to remove fixed size position embedding. For example, Swin Transformer [23] replaces fixed size position embedding with relative position biases, and restricts self-attention within shifted windows. CvT [34], CoaT [36], and LeViT [10] introduce convolution-like operations into vision Transformers. Twins [5] combines local attention and global attention mechanisms to obtain stronger feature representation.

### 3. Methodology

#### 3.1. Limitations in PVT v1

There are three main limitations in PVT v1 [33] as follows: (1) Similar to ViT [8], when processing high-resolution input (e.g., shorter side being 800 pixels), the computational complexity of PVT v1 is relatively large. (2) PVT v1 [33] treats an image as a sequence of non-overlapping patches, which loses the local continuity of the image to a certain extent; (3) The position encoding in PVT v1 is fixed-size, which is inflexible for process images of arbitrary size. These problems limit the performance of PVT v1 on vision tasks.

To address these issues, we propose PVT v2, which improves PVT v1 through three designs, which are listed in Sec 3.2, 3.3, and 3.4.

#### 3.2. Linear Spatial Reduction Attention

First, to reduce the high computational cost caused by attention operations, we propose linear spatial reduction at-

Figure 2: Two improvements in PVT v2. (1) Overlapping Patch Embedding. (2) Convolutional Feed-Forward Network.

tention (SRA) layer as illustrated in Fig. 1. Different from SRA [33] which uses convolutions for spatial reduction, linear SRA uses average pooling to reduce the spatial dimension (i.e.,  $h \times w$ ) to a fixed size (i.e.,  $P \times P$ ) before the attention operation. So linear SRA enjoys linear computational and memory costs like a convolutional layer. Specifically, given an input of size  $h \times w \times c$ , the complexity of SRA and linear SRA are:

$$\Omega(\text{SRA}) = \frac{2h^2w^2c}{R^2} + hwc^2R^2, \quad (1)$$

$$\Omega(\text{Linear SRA}) = 2hwP^2c, \quad (2)$$

where  $R$  is the spatial reduction ratio of SRA [33].  $P$  is the pooling size of linear SRA, which is set to 7.

#### 3.3. Overlapping Patch Embedding

Second, to model the local continuity information, we utilize overlapping patch embedding to tokenize images. As shown in Fig. 2(a), we enlarge the patch window, making adjacent windows overlap by half of the area, and pad the feature map with zeros to keep the resolution. In this work, we use convolution with zero paddings to implement overlapping patch embedding. Specifically, given an input of size  $h \times w \times c$ , we feed it to a convolution with the stride of  $S$ , the kernel size of  $2S - 1$ , the padding size of  $S - 1$ , and the kernel number of  $c'$ . The output size is  $\frac{h}{S} \times \frac{w}{S} \times C'$ .<table border="1">
<thead>
<tr>
<th colspan="9">Pyramid Vision Transformer v2</th>
</tr>
<tr>
<th></th>
<th>Output Size</th>
<th>Layer Name</th>
<th>B0</th>
<th>B1</th>
<th>B2</th>
<th>B2-Li</th>
<th>B3</th>
<th>B4</th>
<th>B5</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">Stage 1</td>
<td rowspan="4"><math>\frac{H}{4} \times \frac{W}{4}</math></td>
<td>Overlapping Patch Embedding</td>
<td colspan="7"><math>S_1 = 4</math></td>
</tr>
<tr>
<td></td>
<td><math>C_1 = 32</math></td>
<td colspan="7"><math>C_1 = 64</math></td>
</tr>
<tr>
<td rowspan="2">Transformer Encoder</td>
<td><math>R_1 = 8</math></td>
<td><math>R_1 = 8</math></td>
<td><math>R_1 = 8</math></td>
<td><math>P_1 = 7</math></td>
<td><math>R_1 = 8</math></td>
<td><math>R_1 = 8</math></td>
<td><math>R_1 = 8</math></td>
</tr>
<tr>
<td><math>N_1 = 1</math></td>
<td><math>N_1 = 1</math></td>
<td><math>N_1 = 1</math></td>
<td><math>N_1 = 1</math></td>
<td><math>N_1 = 1</math></td>
<td><math>N_1 = 1</math></td>
<td><math>N_1 = 1</math></td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td><math>E_1 = 8</math></td>
<td><math>E_1 = 8</math></td>
<td><math>E_1 = 8</math></td>
<td><math>E_1 = 8</math></td>
<td><math>E_1 = 8</math></td>
<td><math>E_1 = 8</math></td>
<td><math>E_1 = 4</math></td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td><math>L_1 = 2</math></td>
<td><math>L_1 = 2</math></td>
<td><math>L_1 = 3</math></td>
<td><math>L_1 = 3</math></td>
<td><math>L_1 = 3</math></td>
<td><math>L_1 = 3</math></td>
<td><math>L_1 = 3</math></td>
</tr>
<tr>
<td rowspan="4">Stage 2</td>
<td rowspan="4"><math>\frac{H}{8} \times \frac{W}{8}</math></td>
<td>Overlapping Patch Embedding</td>
<td colspan="7"><math>S_2 = 2</math></td>
</tr>
<tr>
<td></td>
<td><math>C_2 = 64</math></td>
<td colspan="7"><math>C_2 = 128</math></td>
</tr>
<tr>
<td rowspan="2">Transformer Encoder</td>
<td><math>R_2 = 4</math></td>
<td><math>R_2 = 4</math></td>
<td><math>R_2 = 4</math></td>
<td><math>P_2 = 7</math></td>
<td><math>R_2 = 4</math></td>
<td><math>R_2 = 4</math></td>
<td><math>R_2 = 4</math></td>
</tr>
<tr>
<td><math>N_2 = 2</math></td>
<td><math>N_2 = 2</math></td>
<td><math>N_2 = 2</math></td>
<td><math>N_2 = 2</math></td>
<td><math>N_2 = 2</math></td>
<td><math>N_2 = 2</math></td>
<td><math>N_2 = 2</math></td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td><math>E_2 = 8</math></td>
<td><math>E_2 = 8</math></td>
<td><math>E_2 = 8</math></td>
<td><math>E_2 = 8</math></td>
<td><math>E_2 = 8</math></td>
<td><math>E_2 = 8</math></td>
<td><math>E_2 = 4</math></td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td><math>L_2 = 2</math></td>
<td><math>L_2 = 2</math></td>
<td><math>L_2 = 3</math></td>
<td><math>L_2 = 3</math></td>
<td><math>L_2 = 3</math></td>
<td><math>L_2 = 8</math></td>
<td><math>L_2 = 6</math></td>
</tr>
<tr>
<td rowspan="4">Stage 3</td>
<td rowspan="4"><math>\frac{H}{16} \times \frac{W}{16}</math></td>
<td>Overlapping Patch Embedding</td>
<td colspan="7"><math>S_3 = 2</math></td>
</tr>
<tr>
<td></td>
<td><math>C_3 = 160</math></td>
<td colspan="7"><math>C_3 = 320</math></td>
</tr>
<tr>
<td rowspan="2">Transformer Encoder</td>
<td><math>R_3 = 2</math></td>
<td><math>R_3 = 2</math></td>
<td><math>R_3 = 2</math></td>
<td><math>P_3 = 7</math></td>
<td><math>R_3 = 2</math></td>
<td><math>R_3 = 2</math></td>
<td><math>R_3 = 2</math></td>
</tr>
<tr>
<td><math>N_3 = 5</math></td>
<td><math>N_3 = 5</math></td>
<td><math>N_3 = 5</math></td>
<td><math>N_3 = 5</math></td>
<td><math>N_3 = 5</math></td>
<td><math>N_3 = 5</math></td>
<td><math>N_3 = 5</math></td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td><math>E_3 = 4</math></td>
<td><math>E_3 = 4</math></td>
<td><math>E_3 = 4</math></td>
<td><math>E_3 = 4</math></td>
<td><math>E_3 = 4</math></td>
<td><math>E_3 = 4</math></td>
<td><math>E_3 = 4</math></td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td><math>L_3 = 2</math></td>
<td><math>L_3 = 2</math></td>
<td><math>L_3 = 6</math></td>
<td><math>L_3 = 6</math></td>
<td><math>L_3 = 18</math></td>
<td><math>L_3 = 27</math></td>
<td><math>L_3 = 40</math></td>
</tr>
<tr>
<td rowspan="4">Stage 4</td>
<td rowspan="4"><math>\frac{H}{32} \times \frac{W}{32}</math></td>
<td>Overlapping Patch Embedding</td>
<td colspan="7"><math>S_4 = 2</math></td>
</tr>
<tr>
<td></td>
<td><math>C_4 = 256</math></td>
<td colspan="7"><math>C_4 = 512</math></td>
</tr>
<tr>
<td rowspan="2">Transformer Encoder</td>
<td><math>R_4 = 1</math></td>
<td><math>R_4 = 1</math></td>
<td><math>R_4 = 1</math></td>
<td><math>P_4 = 7</math></td>
<td><math>R_4 = 1</math></td>
<td><math>R_4 = 1</math></td>
<td><math>R_4 = 1</math></td>
</tr>
<tr>
<td><math>N_4 = 8</math></td>
<td><math>N_4 = 8</math></td>
<td><math>N_4 = 8</math></td>
<td><math>N_4 = 8</math></td>
<td><math>N_4 = 8</math></td>
<td><math>N_4 = 8</math></td>
<td><math>N_4 = 8</math></td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td><math>E_4 = 4</math></td>
<td><math>E_4 = 4</math></td>
<td><math>E_4 = 4</math></td>
<td><math>E_4 = 4</math></td>
<td><math>E_4 = 4</math></td>
<td><math>E_4 = 4</math></td>
<td><math>E_4 = 4</math></td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td><math>L_4 = 2</math></td>
<td><math>L_4 = 2</math></td>
<td><math>L_4 = 3</math></td>
<td><math>L_4 = 3</math></td>
<td><math>L_4 = 3</math></td>
<td><math>L_4 = 3</math></td>
<td><math>L_4 = 3</math></td>
</tr>
</tbody>
</table>

Table 1: Detailed settings of PVT v2 series. “-Li” denotes PVT v2 with linear SRA.

### 3.4. Convolutional Feed-Forward

Third, inspired by [17, 6, 20], we remove the fixed-size position encoding [8], and introduce zero padding position encoding into PVT. As shown in Fig. 2(b), we add a  $3 \times 3$  depth-wise convolution [16] with the padding size of 1 between the first fully-connected (FC) layer and GELU [15] in feed-forward networks.

### 3.5. Details of PVT v2 Series

We scale up PVT v2 from B0 to B5 By changing the hyper-parameters. which are listed as follows:

- •  $S_i$ : the stride of the overlapping patch embedding in Stage  $i$ ;
- •  $C_i$ : the channel number of the output of Stage  $i$ ;
- •  $L_i$ : the number of encoder layers in Stage  $i$ ;
- •  $R_i$ : the reduction ratio of the SRA in Stage  $i$ ;
- •  $P_i$ : the adaptive average pooling size of the linear SRA in Stage  $i$ ;
- •  $N_i$ : the head number of the Efficient Self-Attention in Stage  $i$ ;
- •  $E_i$ : the expansion ratio of the feed-forward layer [32] in Stage  $i$ ;

Tab. 1 shows the detailed information of PVT v2 series. Our design follows the principles of ResNet [14]. (1) the channel dimension increase while the spatial resolution shrink with the layer goes deeper. (2) Stage 3 is assigned to most of the computation cost.

### 3.6. Advantages of PVT v2

Combining these improvements, PVT v2 can (1) obtain more local continuity of images and feature maps; (2) pro-

cess variable-resolution input more flexibly; (3) enjoy the same linear complexity as CNN.

## 4. Experiment

### 4.1. Image Classification

**Settings.** Image classification experiments are performed on the ImageNet-1K dataset [27], which comprises 1.28 million training images and 50K validation images from 1,000 categories. All models are trained on the training set for fair comparison and report the top-1 error on the validation set. We follow DeiT [31] and apply random cropping, random horizontal flipping [29], label-smoothing regularization [30], mixup [38], and random erasing [40] as data augmentations. During training, we employ AdamW [25] with a momentum of 0.9, a mini-batch size of 128, and a weight decay of  $5 \times 10^{-2}$  to optimize models. The initial learning rate is set to  $1 \times 10^{-3}$  and decreases following the cosine schedule [24]. All models are trained for 300 epochs from scratch on 8 V100 GPUs. We apply a center crop on the validation set to benchmark, where a  $224 \times 224$  patch is cropped to evaluate the classification accuracy.

**Results.** In Tab. 2, we see that PVT v2 is the state-of-the-art method on ImageNet-1K classification. Compared to PVT, PVT v2 has similar flops and parameters, but the image classification accuracy is greatly improved. For example, PVT v2-B1 is 3.6% higher than PVT v1-Tiny, and PVT v2-B4 is 1.9% higher than PVT-Large.

Compared to other recent counterparts, PVT v2 series also has large advantages in terms of accuracy and model size. For example, PVT v2-B5 achieves 83.8% ImageNet top-1 accuracy, which is 0.5% higher than Swin<table border="1">
<thead>
<tr>
<th>Method</th>
<th>#Param (M)</th>
<th>GFLOPs</th>
<th>Top-1 Acc (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>PVTv2-B0 (ours)</td>
<td><b>3.4</b></td>
<td><b>0.6</b></td>
<td><b>70.5</b></td>
</tr>
<tr>
<td>ResNet18 [14]</td>
<td>11.7</td>
<td>1.8</td>
<td>69.8</td>
</tr>
<tr>
<td>DeiT-Tiny/16 [31]</td>
<td><b>5.7</b></td>
<td><b>1.3</b></td>
<td>72.2</td>
</tr>
<tr>
<td>PVTv1-Tiny [33]</td>
<td>13.2</td>
<td>1.9</td>
<td>75.1</td>
</tr>
<tr>
<td>PVTv2-B1 (ours)</td>
<td>13.1</td>
<td>2.1</td>
<td><b>78.7</b></td>
</tr>
<tr>
<td>ResNet50 [14]</td>
<td>25.6</td>
<td>4.1</td>
<td>76.1</td>
</tr>
<tr>
<td>ResNeXt50-32x4d [35]</td>
<td>25.0</td>
<td>4.3</td>
<td>77.6</td>
</tr>
<tr>
<td>RegNetY-4G [26]</td>
<td>21.0</td>
<td>4.0</td>
<td>80.0</td>
</tr>
<tr>
<td>DeiT-Small/16 [31]</td>
<td>22.1</td>
<td>4.6</td>
<td>79.9</td>
</tr>
<tr>
<td>T2T-ViT<sub>t</sub>-14 [37]</td>
<td>22.0</td>
<td>6.1</td>
<td>80.7</td>
</tr>
<tr>
<td>PVTv1-Small [33]</td>
<td>24.5</td>
<td>3.8</td>
<td>79.8</td>
</tr>
<tr>
<td>TNT-S [11]</td>
<td>23.8</td>
<td>5.2</td>
<td>81.3</td>
</tr>
<tr>
<td>Swin-T [23]</td>
<td>29.0</td>
<td>4.5</td>
<td>81.3</td>
</tr>
<tr>
<td>CvT-13 [34]</td>
<td><b>20.0</b></td>
<td>4.5</td>
<td>81.6</td>
</tr>
<tr>
<td>CoaT-Lite Small [36]</td>
<td><b>20.0</b></td>
<td>4.0</td>
<td>81.9</td>
</tr>
<tr>
<td>Twins-SVT-S [5]</td>
<td>24.0</td>
<td><b>2.8</b></td>
<td>81.7</td>
</tr>
<tr>
<td>PVTv2-B2-Li (ours)</td>
<td>22.6</td>
<td>3.9</td>
<td><b>82.1</b></td>
</tr>
<tr>
<td>PVTv2-B2 (ours)</td>
<td>25.4</td>
<td>4.0</td>
<td>82.0</td>
</tr>
<tr>
<td>ResNet101 [14]</td>
<td>44.7</td>
<td>7.9</td>
<td>77.4</td>
</tr>
<tr>
<td>ResNeXt101-32x4d [35]</td>
<td>44.2</td>
<td>8.0</td>
<td>78.8</td>
</tr>
<tr>
<td>RegNetY-8G [26]</td>
<td>39.0</td>
<td>8.0</td>
<td>81.7</td>
</tr>
<tr>
<td>T2T-ViT<sub>t</sub>-19 [37]</td>
<td>39.0</td>
<td>9.8</td>
<td>81.4</td>
</tr>
<tr>
<td>PVTv1-Medium [33]</td>
<td>44.2</td>
<td>6.7</td>
<td>81.2</td>
</tr>
<tr>
<td>CvT-21 [34]</td>
<td><b>32.0</b></td>
<td>7.1</td>
<td>82.5</td>
</tr>
<tr>
<td>PVTv2-B3 (ours)</td>
<td>45.2</td>
<td><b>6.9</b></td>
<td><b>83.2</b></td>
</tr>
<tr>
<td>ResNet152 [14]</td>
<td>60.2</td>
<td>11.6</td>
<td>78.3</td>
</tr>
<tr>
<td>T2T-ViT<sub>t</sub>-24 [37]</td>
<td>64.0</td>
<td>15.0</td>
<td>82.2</td>
</tr>
<tr>
<td>PVTv1-Large [33]</td>
<td>61.4</td>
<td>9.8</td>
<td>81.7</td>
</tr>
<tr>
<td>TNT-B [11]</td>
<td>66.0</td>
<td>14.1</td>
<td>82.8</td>
</tr>
<tr>
<td>Swin-S [23]</td>
<td><b>50.0</b></td>
<td>8.7</td>
<td>83.0</td>
</tr>
<tr>
<td>Twins-SVT-B [5]</td>
<td>56.0</td>
<td><b>8.3</b></td>
<td>83.2</td>
</tr>
<tr>
<td>PVTv2-B4 (ours)</td>
<td>62.6</td>
<td>10.1</td>
<td><b>83.6</b></td>
</tr>
<tr>
<td>ResNeXt101-64x4d [35]</td>
<td>83.5</td>
<td>15.6</td>
<td>79.6</td>
</tr>
<tr>
<td>RegNetY-16G [26]</td>
<td>84.0</td>
<td>16.0</td>
<td>82.9</td>
</tr>
<tr>
<td>ViT-Base/16 [8]</td>
<td>86.6</td>
<td>17.6</td>
<td>81.8</td>
</tr>
<tr>
<td>DeiT-Base/16 [31]</td>
<td>86.6</td>
<td>17.6</td>
<td>81.8</td>
</tr>
<tr>
<td>Swin-B [23]</td>
<td>88.0</td>
<td>15.4</td>
<td>83.3</td>
</tr>
<tr>
<td>Twins-SVT-L [5]</td>
<td>99.2</td>
<td>14.8</td>
<td>83.7</td>
</tr>
<tr>
<td>PVTv2-B5 (ours)</td>
<td><b>82.0</b></td>
<td><b>11.8</b></td>
<td><b>83.8</b></td>
</tr>
</tbody>
</table>

Table 2: **Image classification performance on the ImageNet validation set.** “#Param” refers to the number of parameters. “GFLOPs” is calculated under the input scale of  $224 \times 224$ . “\*” indicates the performance of the method trained under the strategy of its original paper. “-Li” denotes PVT v2 with linear SRA.

Transformer [23] and Twins [5], while our parameters and FLOPs are fewer.

## 4.2. Object Detection

**Settings.** Object detection experiments are conducted on the challenging COCO benchmark [22]. All models are trained on COCO train2017 (118k images) and evaluated on val2017 (5k images). We verify the effectiveness of PVT v2 backbones on top of mainstream detectors, including RetinaNet [21], Mask R-CNN [12], Cascade Mask R-CNN [1], ATSS [39], GFL [19], and Sparse R-CNN [28]. Before training, we use the weights pre-trained on ImageNet to initialize the backbone and Xavier [9] to initialize

the newly added layers. We train all the models with batch size 16 on 8 V100 GPUs, and adopt AdamW [25] with an initial learning rate of  $1 \times 10^{-4}$  as optimizer. Following common practices [21, 12, 3], we adopt  $1 \times$  or  $3 \times$  training schedule (*i.e.*, 12 or 36 epochs) to train all detection models. The training image is resized to have a shorter side of 800 pixels, while the longer side does not exceed 1,333 pixels. When using the  $3 \times$  training schedule, we randomly resize the shorter side of the input image within the range of [640, 800]. In the testing phase, the shorter side of the input image is fixed to 800 pixels.

**Results.** As reported in Tab. 3, PVT v2 significantly outperforms PVT v1 on both one-stage and two-stage object detectors with similar model size. For example, PVT v2-B4 archive 46.1 AP on top of RetinaNet [21], and 47.5 AP<sup>b</sup> on top of Mask R-CNN [12], surpassing the models with PVT v1 by 3.5 AP and 4.6 AP<sup>b</sup>, respectively. We present some qualitative object detection and instance segmentation results on COCO val2017 [22] in Fig. 3, which also shows the good performance of our models.

For a fair comparison between PVT v2 and Swin Transformer [23], we keep all settings the same, including ImageNet-1K pre-training and COCO fine-tuning strategies. We evaluate Swin Transformer and PVT v2 on four state-of-the-arts detectors, including Cascade R-CNN [1], ATSS [39], GFL [19], and Sparse R-CNN [28]. We see PVT v2 obtain much better AP than Swin Transformer among all the detectors, showing its better feature representation ability. For example, on ATSS, PVT v2 has similar parameters and flops compared to Swin-T, but PVT v2 achieves 49.9 AP, which is 2.7 higher than Swin-T. Our PVT v2-Li can largely reduce the computation from 258 to 194 GFLOPs, while only sacrificing a little performance.

## 4.3. Semantic Segmentation

**Settings.** Following PVT v1 [33], we choose ADE20K [41] to benchmark the performance of semantic segmentation. For a fair comparison, we test the performance of PVT v2 backbones by applying it to Semantic FPN [18]. In the training phase, the backbone is initialized with the weights pre-trained on ImageNet [7], and the newly added layers are initialized with Xavier [9]. We optimize our models using AdamW [25] with an initial learning rate of  $1e-4$ . Following common practices [18, 4], we train our models for 40k iterations with a batch size of 16 on 4 V100 GPUs. The learning rate is decayed following the polynomial decay schedule with a power of 0.9. We randomly resize and crop the image to  $512 \times 512$  for training, and rescale to have a shorter side of 512 pixels during testing.

**Results.** As shown in Tab. 5, when using Semantic FPN [18] for semantic segmentation, PVT v2 consistently outperforms PVT v1 [33] and other counterparts. For example, with almost the same number of parameters and<table border="1">
<thead>
<tr>
<th rowspan="2">Backbone</th>
<th colspan="7">RetinaNet 1×</th>
<th colspan="6">Mask R-CNN 1×</th>
</tr>
<tr>
<th>#P (M)</th>
<th>AP</th>
<th>AP<sub>50</sub></th>
<th>AP<sub>75</sub></th>
<th>AP<sub>S</sub></th>
<th>AP<sub>M</sub></th>
<th>AP<sub>L</sub></th>
<th>#P (M)</th>
<th>AP<sup>b</sup></th>
<th>AP<sub>50</sub><sup>b</sup></th>
<th>AP<sub>75</sub><sup>b</sup></th>
<th>AP<sup>m</sup></th>
<th>AP<sub>50</sub><sup>m</sup></th>
<th>AP<sub>75</sub><sup>m</sup></th>
</tr>
</thead>
<tbody>
<tr>
<td>PVTv2-B0</td>
<td><b>13.0</b></td>
<td><b>37.2</b></td>
<td><b>57.2</b></td>
<td><b>39.5</b></td>
<td><b>23.1</b></td>
<td><b>40.4</b></td>
<td><b>49.7</b></td>
<td><b>23.5</b></td>
<td><b>38.2</b></td>
<td><b>60.5</b></td>
<td><b>40.7</b></td>
<td><b>36.2</b></td>
<td><b>57.8</b></td>
<td><b>38.6</b></td>
</tr>
<tr>
<td>ResNet18 [14]</td>
<td><b>21.3</b></td>
<td>31.8</td>
<td>49.6</td>
<td>33.6</td>
<td>16.3</td>
<td>34.3</td>
<td>43.2</td>
<td><b>31.2</b></td>
<td>34.0</td>
<td>54.0</td>
<td>36.7</td>
<td>31.2</td>
<td>51.0</td>
<td>32.7</td>
</tr>
<tr>
<td>PVTv1-Tiny [33]</td>
<td>23.0</td>
<td>36.7</td>
<td>56.9</td>
<td>38.9</td>
<td>22.6</td>
<td>38.8</td>
<td>50.0</td>
<td>32.9</td>
<td>36.7</td>
<td>59.2</td>
<td>39.3</td>
<td>35.1</td>
<td>56.7</td>
<td>37.3</td>
</tr>
<tr>
<td>PVTv2-B1 (ours)</td>
<td>23.8</td>
<td><b>41.2</b></td>
<td><b>61.9</b></td>
<td><b>43.9</b></td>
<td><b>25.4</b></td>
<td><b>44.5</b></td>
<td><b>54.3</b></td>
<td>33.7</td>
<td><b>41.8</b></td>
<td><b>64.3</b></td>
<td><b>45.9</b></td>
<td><b>38.8</b></td>
<td><b>61.2</b></td>
<td><b>41.6</b></td>
</tr>
<tr>
<td>ResNet50 [14]</td>
<td>37.7</td>
<td>36.3</td>
<td>55.3</td>
<td>38.6</td>
<td>19.3</td>
<td>40.0</td>
<td>48.8</td>
<td>44.2</td>
<td>38.0</td>
<td>58.6</td>
<td>41.4</td>
<td>34.4</td>
<td>55.1</td>
<td>36.7</td>
</tr>
<tr>
<td>PVTv1-Small [33]</td>
<td>34.2</td>
<td>40.4</td>
<td>61.3</td>
<td>43.0</td>
<td>25.0</td>
<td>42.9</td>
<td>55.7</td>
<td>44.1</td>
<td>40.4</td>
<td>62.9</td>
<td>43.8</td>
<td>37.8</td>
<td>60.1</td>
<td>40.3</td>
</tr>
<tr>
<td>PVTv2-B2-Li (ours)</td>
<td><b>32.3</b></td>
<td>43.6</td>
<td>64.7</td>
<td>46.8</td>
<td>28.3</td>
<td>47.6</td>
<td>57.4</td>
<td><b>42.2</b></td>
<td>44.1</td>
<td>66.3</td>
<td>48.4</td>
<td>40.5</td>
<td>63.2</td>
<td>43.6</td>
</tr>
<tr>
<td>PVTv2-B2 (ours)</td>
<td>35.1</td>
<td><b>44.6</b></td>
<td><b>65.6</b></td>
<td><b>47.6</b></td>
<td><b>27.4</b></td>
<td><b>48.8</b></td>
<td><b>58.6</b></td>
<td>45.0</td>
<td><b>45.3</b></td>
<td><b>67.1</b></td>
<td><b>49.6</b></td>
<td><b>41.2</b></td>
<td><b>64.2</b></td>
<td><b>44.4</b></td>
</tr>
<tr>
<td>ResNet101 [14]</td>
<td>56.7</td>
<td>38.5</td>
<td>57.8</td>
<td>41.2</td>
<td>21.4</td>
<td>42.6</td>
<td>51.1</td>
<td>63.2</td>
<td>40.4</td>
<td>61.1</td>
<td>44.2</td>
<td>36.4</td>
<td>57.7</td>
<td>38.8</td>
</tr>
<tr>
<td>ResNeXt101-32x4d [35]</td>
<td>56.4</td>
<td>39.9</td>
<td>59.6</td>
<td>42.7</td>
<td>22.3</td>
<td>44.2</td>
<td>52.5</td>
<td><b>62.8</b></td>
<td>41.9</td>
<td>62.5</td>
<td>45.9</td>
<td>37.5</td>
<td>59.4</td>
<td>40.2</td>
</tr>
<tr>
<td>PVTv1-Medium [33]</td>
<td><b>53.9</b></td>
<td>41.9</td>
<td>63.1</td>
<td>44.3</td>
<td>25.0</td>
<td>44.9</td>
<td>57.6</td>
<td>63.9</td>
<td>42.0</td>
<td>64.4</td>
<td>45.6</td>
<td>39.0</td>
<td>61.6</td>
<td>42.1</td>
</tr>
<tr>
<td>PVTv2-B3 (ours)</td>
<td>55.0</td>
<td><b>45.9</b></td>
<td><b>66.8</b></td>
<td><b>49.3</b></td>
<td><b>28.6</b></td>
<td><b>49.8</b></td>
<td><b>61.4</b></td>
<td>64.9</td>
<td><b>47.0</b></td>
<td><b>68.1</b></td>
<td><b>51.7</b></td>
<td><b>42.5</b></td>
<td><b>65.7</b></td>
<td><b>45.7</b></td>
</tr>
<tr>
<td>PVTv1-Large [33]</td>
<td><b>71.1</b></td>
<td>42.6</td>
<td>63.7</td>
<td>45.4</td>
<td>25.8</td>
<td>46.0</td>
<td>58.4</td>
<td><b>81.0</b></td>
<td>42.9</td>
<td>65.0</td>
<td>46.6</td>
<td>39.5</td>
<td>61.9</td>
<td>42.5</td>
</tr>
<tr>
<td>PVTv2-B4 (ours)</td>
<td>72.3</td>
<td><b>46.1</b></td>
<td><b>66.9</b></td>
<td><b>49.2</b></td>
<td><b>28.4</b></td>
<td><b>50.0</b></td>
<td><b>62.2</b></td>
<td>82.2</td>
<td><b>47.5</b></td>
<td><b>68.7</b></td>
<td><b>52.0</b></td>
<td><b>42.7</b></td>
<td><b>66.1</b></td>
<td><b>46.1</b></td>
</tr>
<tr>
<td>ResNeXt101-64x4d [35]</td>
<td>95.5</td>
<td>41.0</td>
<td>60.9</td>
<td>44.0</td>
<td>23.9</td>
<td>45.2</td>
<td>54.0</td>
<td>101.9</td>
<td>42.8</td>
<td>63.8</td>
<td>47.3</td>
<td>38.4</td>
<td>60.6</td>
<td>41.3</td>
</tr>
<tr>
<td>PVTv2-B5 (ours)</td>
<td><b>91.7</b></td>
<td><b>46.2</b></td>
<td><b>67.1</b></td>
<td><b>49.5</b></td>
<td><b>28.5</b></td>
<td><b>50.0</b></td>
<td><b>62.5</b></td>
<td><b>101.6</b></td>
<td><b>47.4</b></td>
<td><b>68.6</b></td>
<td><b>51.9</b></td>
<td><b>42.5</b></td>
<td><b>65.7</b></td>
<td><b>46.0</b></td>
</tr>
</tbody>
</table>

Table 3: **Object detection and instance segmentation on COCO val2017.** “#P” refers to parameter number. AP<sup>b</sup> and AP<sup>m</sup> denote bounding box AP and mask AP, respectively. “-Li” denotes PVT v2 with linear SRA.

<table border="1">
<thead>
<tr>
<th>Backbone</th>
<th>Method</th>
<th>AP<sup>b</sup></th>
<th>AP<sub>50</sub><sup>b</sup></th>
<th>AP<sub>75</sub><sup>b</sup></th>
<th>#P (M)</th>
<th>GFLOPs</th>
</tr>
</thead>
<tbody>
<tr>
<td>ResNet50 [14]</td>
<td rowspan="3">Cascade<br/>Mask<br/>R-CNN [1]</td>
<td>46.3</td>
<td>64.3</td>
<td>50.5</td>
<td>82</td>
<td>739</td>
</tr>
<tr>
<td>Swin-T [23]</td>
<td>50.5</td>
<td>69.3</td>
<td>54.9</td>
<td>86</td>
<td>745</td>
</tr>
<tr>
<td>PVTv2-B2-Li (ours)</td>
<td>50.9</td>
<td>69.5</td>
<td>55.2</td>
<td><b>80</b></td>
<td><b>725</b></td>
</tr>
<tr>
<td>PVTv2-B2 (ours)</td>
<td></td>
<td><b>51.1</b></td>
<td><b>69.8</b></td>
<td><b>55.3</b></td>
<td>83</td>
<td>788</td>
</tr>
<tr>
<td>ResNet50 [14]</td>
<td rowspan="4">ATSS [39]</td>
<td>43.5</td>
<td>61.9</td>
<td>47.0</td>
<td>32</td>
<td>205</td>
</tr>
<tr>
<td>Swin-T [23]</td>
<td>47.2</td>
<td>66.5</td>
<td>51.3</td>
<td>36</td>
<td>215</td>
</tr>
<tr>
<td>PVTv2-B2-Li (ours)</td>
<td>48.9</td>
<td>68.1</td>
<td>53.4</td>
<td><b>30</b></td>
<td><b>194</b></td>
</tr>
<tr>
<td>PVTv2-B2 (ours)</td>
<td><b>49.9</b></td>
<td><b>69.1</b></td>
<td><b>54.1</b></td>
<td>33</td>
<td>258</td>
</tr>
<tr>
<td>ResNet50 [14]</td>
<td rowspan="4">GFL [19]</td>
<td>44.5</td>
<td>63.0</td>
<td>48.3</td>
<td>32</td>
<td>208</td>
</tr>
<tr>
<td>Swin-T [23]</td>
<td>47.6</td>
<td>66.8</td>
<td>51.7</td>
<td>36</td>
<td>215</td>
</tr>
<tr>
<td>PVTv2-B2-Li (ours)</td>
<td>49.2</td>
<td>68.2</td>
<td>53.7</td>
<td><b>30</b></td>
<td><b>197</b></td>
</tr>
<tr>
<td>PVTv2-B2 (ours)</td>
<td><b>50.2</b></td>
<td><b>69.4</b></td>
<td><b>54.7</b></td>
<td>33</td>
<td>261</td>
</tr>
<tr>
<td>ResNet50 [14]</td>
<td rowspan="4">Sparse<br/>R-CNN [28]</td>
<td>44.5</td>
<td>63.4</td>
<td>48.2</td>
<td>106</td>
<td>166</td>
</tr>
<tr>
<td>Swin-T [23]</td>
<td>47.9</td>
<td>67.3</td>
<td>52.3</td>
<td>110</td>
<td>172</td>
</tr>
<tr>
<td>PVTv2-B2-Li (ours)</td>
<td>48.9</td>
<td>68.3</td>
<td>53.4</td>
<td><b>104</b></td>
<td><b>151</b></td>
</tr>
<tr>
<td>PVTv2-B2 (ours)</td>
<td><b>50.1</b></td>
<td><b>69.5</b></td>
<td><b>54.9</b></td>
<td>107</td>
<td>215</td>
</tr>
</tbody>
</table>

Table 4: **Compare with Swin Transformer on object detection.** “AP<sup>b</sup>” denotes bounding box AP. “#P” refers to parameter number. “GFLOPs” is calculated under the input scale of 1280 × 800. “-Li” denotes PVT v2 with linear SRA.

GFLOPs, PVT v2-B1/B2/B3/B4 are at least 5.3% higher than PVT v1-Tiny/Small/Medium/Large. Moreover, although the GFLOPs of PVT-Large are 12% lower than those of ResNeXt101-64x4d, the mIoU is still 8.5 points higher (48.7 vs 40.2). In Fig. 3, we also visualize some qualitative semantic segmentation results on ADE20K [41]. These results demonstrate that PVT v2 backbones can extract powerful features for semantic segmentation, benefiting from the improved designs.

<table border="1">
<thead>
<tr>
<th rowspan="2">Backbone</th>
<th colspan="3">Semantic FPN</th>
</tr>
<tr>
<th>#Param (M)</th>
<th>GFLOPs</th>
<th>mIoU (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>PVTv2-B0 (ours)</td>
<td><b>7.6</b></td>
<td><b>25.0</b></td>
<td><b>37.2</b></td>
</tr>
<tr>
<td>ResNet18 [14]</td>
<td><b>15.5</b></td>
<td><b>32.2</b></td>
<td>32.9</td>
</tr>
<tr>
<td>PVTv1-Tiny [33]</td>
<td>17.0</td>
<td>33.2</td>
<td>35.7</td>
</tr>
<tr>
<td>PVTv2-B1 (ours)</td>
<td>17.8</td>
<td>34.2</td>
<td><b>42.5</b></td>
</tr>
<tr>
<td>ResNet50 [14]</td>
<td>28.5</td>
<td>45.6</td>
<td>36.7</td>
</tr>
<tr>
<td>PVTv1-Small [33]</td>
<td>28.2</td>
<td>44.5</td>
<td>39.8</td>
</tr>
<tr>
<td>PVTv2-B2-Li (ours)</td>
<td><b>26.3</b></td>
<td><b>41.0</b></td>
<td>45.1</td>
</tr>
<tr>
<td>PVTv2-B2 (ours)</td>
<td>29.1</td>
<td>45.8</td>
<td><b>45.2</b></td>
</tr>
<tr>
<td>ResNet101 [14]</td>
<td>47.5</td>
<td>65.1</td>
<td>38.8</td>
</tr>
<tr>
<td>ResNeXt101-32x4d [35]</td>
<td><b>47.1</b></td>
<td>64.7</td>
<td>39.7</td>
</tr>
<tr>
<td>PVTv1-Medium [33]</td>
<td>48.0</td>
<td><b>61.0</b></td>
<td>41.6</td>
</tr>
<tr>
<td>PVTv2-B3 (ours)</td>
<td>49.0</td>
<td>62.4</td>
<td><b>47.3</b></td>
</tr>
<tr>
<td>PVTv1-Large [33]</td>
<td><b>65.1</b></td>
<td><b>79.6</b></td>
<td>42.1</td>
</tr>
<tr>
<td>PVTv2-B4 (ours)</td>
<td>66.3</td>
<td>81.3</td>
<td><b>47.9</b></td>
</tr>
<tr>
<td>ResNeXt101-64x4d [35]</td>
<td>86.4</td>
<td>103.9</td>
<td>40.2</td>
</tr>
<tr>
<td>PVTv2-B5 (ours)</td>
<td><b>85.7</b></td>
<td><b>91.1</b></td>
<td><b>48.7</b></td>
</tr>
</tbody>
</table>

Table 5: **Semantic segmentation performance of different backbones on the ADE20K validation set.** “GFLOPs” is calculated under the input scale of 512 × 512. “-Li” denotes PVT v2 with linear SRA.

## 4.4. Ablation Study

### 4.4.1 Model Analysis

Ablation experiments of PVT v2 is reported in Tab. 6. We see that all three designs can improve the model in terms of performance, parameter number, or computation overhead.

**Overlapping patch embedding (OPE) is important.** Comparing #1 and #2 in Tab. 6, the model with OPE obtains better top-1 accuracy (81.1% vs. 79.8%) on ImageNet and better AP (42.2% vs. 40.4%) on COCO than the one with original patch embedding (PE) [8]. OPE is effective because it can model the local continuity of images and feature map via the overlapping sliding window.Figure 3: Qualitative results of object detection and instance segmentation on COCO val2017 [22], and semantic segmentation on ADE20K [41]. The results (from left to right) are generated by PVT v2-B2-based RetinaNet [21], Mask R-CNN [12], and Semantic FPN [18], respectively.

**Convolutional feed-forward network (CFFN) matters.** Compared to original feed-forward network (FFN) [8], our CFFN contains a zero-padding convolutional layer, which can capture the local continuity of the input tensor. In addition, due to the positional information introduced by zero-padding in OPE and CFFN, we can remove the fixed-size positional embeddings used in PVT v1, making the model flexible to handle variable resolution inputs. As reported in #2 and #3 in Tab. 6, CFFN brings 0.9 points improvement on ImageNet (82.0% vs. 81.1%) and 2.4 points improvement on COCO, which demonstrates its effectiveness.

**Linear SRA (LSRA) contributes to a better model.** As reported in #3 and #4 in Tab. 6, compared to SRA [33], our LSRA significantly reduces the computation overhead (GFLOPs) of the model by 22%, while keeping a comparable top-1 accuracy on ImageNet (82.1% vs. 82.0%), and

<table border="1">
<thead>
<tr>
<th rowspan="2">#</th>
<th rowspan="2">Setting</th>
<th rowspan="2">Top-1 Acc (%)</th>
<th colspan="3">RetinaNet 1x</th>
</tr>
<tr>
<th>#P (M)</th>
<th>GFLOPs</th>
<th>AP</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>PVTv1-Small [33]</td>
<td>79.8</td>
<td>34.2</td>
<td>285.8</td>
<td>40.4</td>
</tr>
<tr>
<td>2</td>
<td>+ OPE</td>
<td>81.1</td>
<td>34.9</td>
<td>288.6</td>
<td>42.2</td>
</tr>
<tr>
<td>3</td>
<td>++ CFFN (PVTv2-B2)</td>
<td>82.0</td>
<td>35.1</td>
<td>290.7</td>
<td><b>44.6</b></td>
</tr>
<tr>
<td>4</td>
<td>+++ LSRA (PVTv2-B2-Li)</td>
<td><b>82.1</b></td>
<td><b>32.3</b></td>
<td><b>227.4</b></td>
<td>43.6</td>
</tr>
</tbody>
</table>

Table 6: Ablation experiments of PVT v2. “OPE”, “CFFN”, and “LSRA” represent overlapping patch embedding, convolutional feed-forward network, and linear SRA, respectively.

only 1 point lower AP on COCO (43.6 vs. 44.6). These results show the low computational cost and good effect of LSRA.Figure 4: **Models' GFLOPs under different input scales.** The growth rate of GFLOPs: ViT-Small/16 [8]>ViT-Small/32 [8]>PVT v1-Small [33]>ResNet50 [14]>PVT v2-B2-Li (ours).

#### 4.4.2 Computation Overhead Analysis

As shown in Figure 4, with increasing input scale, the GFLOPs growth rate of the proposed PVT v2-B2-Li is much lower than that of PVT v1-Small [33], and is similar to that of ResNet-50 [13]. This result proves that our PVT v2-Li successfully addresses the high computational overhead problem caused by the attention layer.

## 5. Conclusion

We study the limitations of Pyramid Vision Transformer (PVT v1) and improve it with three designs, which are overlapping patch embedding, convolutional feed-forward network, and linear spatial reduction attention layer. Extensive experiments on different tasks, such as image classification, object detection, and semantic segmentation demonstrate that the proposed PVT v2 is stronger than its predecessor PVT v1 and other state-of-the-art transformer-based backbones, under comparable numbers of parameters. We hope these improved baselines will provide a reference for future research in vision Transformer.

## References

- [1] Zhaowei Cai and Nuno Vasconcelos. Cascade r-cnn: Delving into high quality object detection. In *Proc. IEEE Conf. Comp. Vis. Patt. Recogn.*, 2018. 4, 5
- [2] Chun-Fu Chen, Quanfu Fan, and Rameswar Panda. Crossvit: Cross-attention multi-scale vision transformer for image classification. *arXiv preprint arXiv:2103.14899*, 2021. 1
- [3] Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, et al. Mmdetection: Open mmlab detection toolbox and benchmark. *arXiv preprint arXiv:1906.07155*, 2019. 4

- [4] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. *IEEE Trans. Pattern Anal. Mach. Intell.*, 2017. 4
- [5] Xiangxiang Chu, Zhi Tian, Yuqing Wang, Bo Zhang, Haibing Ren, Xiaolin Wei, Huaxia Xia, and Chunhua Shen. Twins: Revisiting the design of spatial attention in vision transformers. *arXiv preprint arXiv:2104.13840*, 2021. 1, 2, 4
- [6] Xiangxiang Chu, Zhi Tian, Bo Zhang, Xinlong Wang, Xiaolin Wei, Huaxia Xia, and Chunhua Shen. Conditional positional encodings for vision transformers. *arXiv preprint arXiv:2102.10882*, 2021. 1, 3
- [7] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *Proc. IEEE Conf. Comp. Vis. Patt. Recogn.*, 2009. 1, 4
- [8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. *Proc. Int. Conf. Learn. Representations*, 2021. 1, 2, 3, 4, 5, 6, 7
- [9] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In *Proc. Int. Conf. Artificial Intell. & Stat.*, 2010. 4
- [10] Ben Graham, Alaaeldin El-Noubi, Hugo Touvron, Pierre Stock, Armand Joulin, Hervé Jégou, and Matthijs Douze. Levit: a vision transformer in convnet’s clothing for faster inference. In *Proc. IEEE Int. Conf. Comp. Vis.*, 2021. 1, 2
- [11] Kai Han, An Xiao, Enhua Wu, Jianyuan Guo, Chunjing Xu, and Yunhe Wang. Transformer in transformer. *arXiv preprint arXiv:2103.00112*, 2021. 1, 4
- [12] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask r-cnn. In *Proc. IEEE Int. Conf. Comp. Vis.*, 2017. 4, 6
- [13] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In *Proc. IEEE Int. Conf. Comp. Vis.*, 2015. 1, 7
- [14] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proc. IEEE Conf. Comp. Vis. Patt. Recogn.*, 2016. 3, 4, 5, 7
- [15] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). *arXiv preprint arXiv:1606.08415*, 2016. 3
- [16] Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. *arXiv preprint arXiv:1704.04861*, 2017. 3
- [17] Md. Amirul Islam, Sen Jia, and Neil D. B. Bruce. How much position information do convolutional neural networks encode? In *Proc. Int. Conf. Learn. Representations*, 2020. 3
- [18] Alexander Kirillov, Ross Girshick, Kaiming He, and Piotr Dollár. Panoptic feature pyramid networks. In *Proc. IEEE Conf. Comp. Vis. Patt. Recogn.*, 2019. 4, 6- [19] Xiang Li, Wenhai Wang, Lijun Wu, Shuo Chen, Xiaolin Hu, Jun Li, Jinhui Tang, and Jian Yang. Generalized focal loss: Learning qualified and distributed bounding boxes for dense object detection. In *Proc. Advances in Neural Inf. Process. Syst.*, 2020. [1](#), [4](#), [5](#)
- [20] Yawei Li, Kai Zhang, Jiezhong Cao, Radu Timofte, and Luc Van Gool. Localvit: Bringing locality to vision transformers. *arXiv preprint arXiv:2104.05707*, 2021. [1](#), [3](#)
- [21] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In *Proc. IEEE Int. Conf. Comp. Vis.*, 2017. [4](#), [6](#)
- [22] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In *Proc. Eur. Conf. Comp. Vis.*, 2014. [1](#), [4](#), [6](#)
- [23] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. *arXiv preprint arXiv:2103.14030*, 2021. [1](#), [2](#), [4](#), [5](#)
- [24] Ilya Loshchilov and Frank Hutter. SGDR: stochastic gradient descent with warm restarts. In *Proc. Int. Conf. Learn. Representations*, 2017. [3](#)
- [25] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In *Proc. Int. Conf. Learn. Representations*, 2019. [3](#), [4](#)
- [26] Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr Dollár. Designing network design spaces. In *Proc. IEEE Conf. Comp. Vis. Patt. Recogn.*, 2020. [4](#)
- [27] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. *Int. J. Comput. Vision*, 2015. [3](#)
- [28] Peize Sun, Rufeng Zhang, Yi Jiang, Tao Kong, Chenfeng Xu, Wei Zhan, Masayoshi Tomizuka, Lei Li, Zehuan Yuan, Changhu Wang, et al. Sparse r-cnn: End-to-end object detection with learnable proposals. In *Proc. IEEE Conf. Comp. Vis. Patt. Recogn.*, 2021. [4](#), [5](#)
- [29] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In *Proc. IEEE Conf. Comp. Vis. Patt. Recogn.*, 2015. [3](#)
- [30] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In *Proc. IEEE Conf. Comp. Vis. Patt. Recogn.*, 2016. [3](#)
- [31] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In *Proc. Int. Conf. Mach. Learn.*, 2021. [1](#), [3](#), [4](#)
- [32] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Proc. Advances in Neural Inf. Process. Syst.*, 2017. [3](#)
- [33] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. *arXiv preprint arXiv:2102.12122*, 2021. [1](#), [2](#), [4](#), [5](#), [6](#), [7](#)
- [34] Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, and Lei Zhang. Cvt: Introducing convolutions to vision transformers. *arXiv preprint arXiv:2103.15808*, 2021. [1](#), [2](#), [4](#)
- [35] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. In *Proc. IEEE Conf. Comp. Vis. Patt. Recogn.*, 2017. [4](#), [5](#)
- [36] Weijian Xu, Yifan Xu, Tyler Chang, and Zhuowen Tu. Co-scale conv-attentional image transformers. *arXiv preprint arXiv:2104.06399*, 2021. [1](#), [2](#), [4](#)
- [37] Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zihang Jiang, Francis EH Tay, Jiashi Feng, and Shuicheng Yan. Tokens-to-token vit: Training vision transformers from scratch on imagenet. *arXiv preprint arXiv:2101.11986*, 2021. [1](#), [4](#)
- [38] Hongyi Zhang, Moustapha Cissé, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In *Proc. Int. Conf. Learn. Representations*, 2018. [3](#)
- [39] Shifeng Zhang, Cheng Chi, Yongqiang Yao, Zhen Lei, and Stan Z Li. Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection. In *Proc. IEEE Conf. Comp. Vis. Patt. Recogn.*, 2020. [4](#), [5](#)
- [40] Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, and Yi Yang. Random erasing data augmentation. In *Proc. AAAI Conf. Artificial Intell.*, 2020. [3](#)
- [41] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In *Proc. IEEE Conf. Comp. Vis. Patt. Recogn.*, 2017. [1](#), [4](#), [5](#), [6](#)

