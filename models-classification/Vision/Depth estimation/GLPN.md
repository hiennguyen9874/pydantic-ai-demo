# Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth

Doyeon Kim<sup>1</sup>, Woonghyun Ka<sup>2</sup>, Pyunghwan Ahn<sup>1</sup>, Donggyu Joo<sup>1</sup>,  
Sewhan Chun<sup>2</sup> and Junmo Kim<sup>1,2</sup>

School of Electrical Engineering, KAIST, South Korea<sup>1</sup>

Division of Future Vehicle, KAIST, South Korea<sup>2</sup>

{doyeon\_kim, kwh950724, p.ahn, jdg105, alskdjfhgk, junmo.kim}@kaist.ac.kr

## Abstract

Depth estimation from a single image is an important task that can be applied to various fields in computer vision, and has grown rapidly with the development of convolutional neural networks. In this paper, we propose a novel structure and training strategy for monocular depth estimation to further improve the prediction accuracy of the network. We deploy a hierarchical transformer encoder to capture and convey the global context, and design a lightweight yet powerful decoder to generate an estimated depth map while considering local connectivity. By constructing connected paths between multi-scale local features and the global decoding stream with our proposed selective feature fusion module, the network can integrate both representations and recover fine details. In addition, the proposed decoder shows better performance than the previously proposed decoders, with considerably less computational complexity. Furthermore, we improve the depth-specific augmentation method by utilizing an important observation in depth estimation to enhance the model. Our network achieves state-of-the-art performance over the challenging depth dataset NYU Depth V2. Extensive experiments have been conducted to validate and show the effectiveness of the proposed approach. Finally, our model shows better generalization ability and robustness than other comparative models. The code will be available soon.

As many previous papers have claimed [Chen *et al.*, 2019; Kim *et al.*, 2020], understanding both global and local contexts is crucial for successful depth estimation. There are many cues in monocular depth estimation that require understanding the scene on a global scale, such as the location of objects or the vanishing point. In addition, local connectivity of features is important because adjacent pixels tend to have similar values owing to their coplanar surfaces. Therefore, we propose a new global-local path network to fully extract meaningful features on diverse scales and effectively deliver them throughout the network. First, we adopt a hierarchical transformer as the encoder to model long-range dependencies and capture multi-scale context features. In prior studies, it is observed that the transformer enables the network to enlarge the size of the receptive field [Xie *et al.*, 2021]. Motivated by this knowledge, we leverage the global relationships explicitly by building the global path with multiple transformer blocks. Second, we design a highly utilized decoder with an effective fusion module to enable local features to produce a fine depth map while preserving structural details. Contrary to the transformer, skip connections tend to create smaller receptive fields and help to focus on short-distance information [Luo *et al.*, 2016]. Thus, the proposed architecture is intended to take the complementary advantages of both transformer and skip connections. This is enabled by aggregating the encoded and the decoded features using an input-dependent fusion module, called selective feature fusion (SFF). The SFF module aids the model to selectively focus on the salient regions by estimating the attention map for both features with a very low computational burden. Compared to other decoders, our decoder achieves superior performance with much lower complexity.

## 1 Introduction

Depth estimation is a challenging area that has been actively researched for many years. In particular, monocular depth estimation, which uses a single image to predict depth, is an ill-posed problem due to its inherent ambiguity. With the advent of convolutional neural networks (CNNs), many CNN-based approaches have been proposed for depth estimation and have yielded promising results [Bhat *et al.*, 2021; Lee *et al.*, 2019; Fu *et al.*, 2018]. This paper proposes a new architecture and training strategy to further improve the performance by focusing on the essential properties of monocular depth estimation.

Furthermore, we train the network with an additional task specific data augmentation technique to boost the model capability. Data augmentation plays an important role in optimizing the network and can accelerate the model performance without additional computational costs. Nevertheless, data augmentation for depth estimation has been rarely adopted unlike for other tasks. To the best of our knowledge, *CutDepth* [Ishii and Yamashita, 2021] is the first attempted data augmentation method specifically for depth estimation. We revisit CutDepth with the discovery that the vertical position of an object plays an essential role in monocular depth estimation [Dijk and Croon, 2019]. To this end, we propose aFigure 1: Overall architecture of the proposed network. The main components of the architecture are the encoder, decoder, and skip connections with feature fusion modules.

variant of CutDepth, in which the crop is only applied to the horizontal axis, so that the model adaptively learns to capture vertical long-range information from the training sample.

The proposed network architecture and training strategy are experimented over the popular depth estimation dataset NYU Depth V2 [Silberman *et al.*, 2012] and exhibit the state-of-the-art performance. We validate the model through extensive quantitative and qualitative experiments, and the suggested architecture and data augmentation method demonstrate their effectiveness. In addition, we observe that our network can generalize well under cross-dataset validation and shows robust performance against image corruption.

To summarize, our contributions are as follows:

- • We propose a novel global-local path architecture for monocular depth estimation.
- • We suggest an improved depth-specific data augmentation method to boost the performance of the model.
- • Our network achieves state-of-the-art performance on the most popular dataset NYU Depth V2 and shows higher generalization ability and robustness than previously developed networks.

## 2 Related Work

**Monocular depth estimation** is a computer vision task that predicts corresponding depth maps with given input images. Learning-based monocular depth estimation has been studied following the seminal work of [Saxena *et al.*, 2008] which used monocular cues to predict depth based on a Markov random field. Later, with the development of CNNs, depth estimation networks have utilized the encoded features of deep CNNs that generalize well to various tasks and achieve drastic performance improvement [Eigen *et al.*, 2014; Huynh *et al.*, 2020; Yin *et al.*, 2019]. Recently, BTS [Lee *et al.*, 2019] has suggested a local planar guidance layer that outputs plane coefficients, and then used them in the full resolution depth estimation. AdaBins [Bhat *et al.*, 2021] refor-

mulates the depth estimation problem as a classification task by dividing depth values into bins and shows state-of-the-art performance.

**Transformer** [Vaswani *et al.*, 2017] adopts a self-attention mechanism with multi-layer perceptron (MLP) to overcome the limitation of previous RNN for natural language processing. Since the emergence of the transformer, it has gained considerable attention in various fields. In the field of computer vision, a vision transformer (ViT) [Dosovitskiy *et al.*, 2020] first uses a transformer to solve image classification tasks. The success of ViT in the image classification task accelerates the introduction of the transformer into other tasks. SETR [Zheng *et al.*, 2021] first employs ViT as a backbone and demonstrates the potential of the transformer in dense prediction tasks by achieving new state-of-the-art performance. [Xie *et al.*, 2021] proposed SegFormer, which is a transformer-based segmentation framework, with a simple lightweight MLP decoder.

However, very few attempts have been made to employ a transformer for monocular depth estimation. Adabins [Bhat *et al.*, 2021] uses a minimized version of a vision transformer (mini-ViT) to calculate bin width in an adaptive manner. DPT [Ranftl *et al.*, 2021] employs ViT as an encoder to obtain a global receptive field at different stages and attaches a convolutional decoder to make a dense prediction. However, both Adabins and DPT use CNN-based encoders and transformers simultaneously which increases the computational complexity. In addition, DPT is trained with an extra large-scale dataset. In contrast to these studies, our method use only one encoder and does not require an additional dataset to accomplish state-of-the-art performance.

**Data augmentation** plays an important role in preventing overfitting by increasing the effective amount of training data. Therefore, common methods, such as flipping, color space transformation, cropping, and rotation, are used in several tasks to improve the network performance. However, although various methods, such as CutMix [Yun *et al.*,2019], Copy-Paste [Ghiasi *et al.*, 2021] and CutBlur [Yoo *et al.*, 2020], have been actively proposed in diverse tasks, the depth-specific data augmentation method has rarely been studied. To the best of our knowledge, CutDepth [Ishii and Yamashita, 2021] is the first approach that attempts to augment the data in depth estimation. We accelerate the performance of this depth-specific data augmentation method by emphasizing the vertical location in the image.

### 3 Methods

#### 3.1 Global-Local Path Networks

Our depth estimation framework aims to predict the depth map  $\hat{Y} \in \mathbb{R}^{H \times W \times 1}$  with a given RGB image  $I \in \mathbb{R}^{H \times W \times 3}$ . Thus, we suggest a new architecture with global and local feature paths through the entire network to generate  $\hat{Y}$ . The overall structure of our framework is depicted in Figure 1. Our transformer encoder [Xie *et al.*, 2021] enables the model to learn global dependencies, and the proposed decoder successfully recovers the extracted feature into the target depth map by constructing the local path through skip connection and the feature fusion module. We detail the proposed architecture in the following subsections.

#### 3.2 Encoder

In the encoding phase, we aim to leverage rich global information from an RGB image. To achieve this, we adopt a hierarchical transformer as the encoder. First, the input image  $I$  is embedded as a sequence of patches with a  $3 \times 3$  convolution operation. Then, the embedded patches are used as an input of the transformer block, which comprises of multiple sets of self-attention and the MLP-Conv-MLP layer with a residual skip. To reduce the computational cost in the self-attention layer, the dimension of each attention head is reduced with ratio  $R_i$  for each  $i$ th block. With a given output, we perform patch merging with overlapped convolution. This process allows us to generate multi-scale features during the encoding phase and can be utilized in the decoding phase. We use four transformer blocks and each block generates  $\frac{1}{4}$ ,  $\frac{1}{8}$ ,  $\frac{1}{16}$ ,  $\frac{1}{32}$  scale feature with  $[C_1, C_2, C_3, C_4]$  dimensions.

#### 3.3 Lightweight Decoder

The encoder transforms the input image  $I$  into the bottleneck feature  $F_E^4$  with the size of  $\frac{1}{32}H \times \frac{1}{32}W \times C_4$ . To obtain the estimated depth map, we construct a lightweight and effective decoder to restore the bottleneck feature into the size of  $H \times W \times 1$ . Most of the previous studies have conventionally stacked multiple bilinear upsampling with convolution or deconvolution layers to recover the original size. However, we empirically observe that the model can achieve better performance with much fewer convolution and bilinear upsampling layers of the decoder if we design our restoring path effectively. First, we reduce the channel dimension of the bottleneck feature into  $N_C$  with  $1 \times 1$  convolution to avoid computational complexity. Then we use consecutive bilinear upsampling to enlarge the feature into size of  $H \times W \times N_C$ . Finally, the output is passed through two convolution layers and a sigmoid function to predict depth map  $H \times W \times 1$ .

Figure 2: Detailed description of the SFF module.

And depth map is multiplied with the maximum depth value to scale in meter. This simple decoder can generate as precise a depth map as other baseline structures. However, to further exploit the local structures with fine details, we add skip connection with the proposed fusion module.

#### 3.4 Selective Feature Fusion

We propose a Selective Feature Fusion (SFF) module to adaptively select and integrate local and global features by attaining an attention map for each feature. The detailed structure of SFF is illustrated in Figure 2. To match the dimensions of the decoded features  $F_D$  and  $F_E$ , we first reduce the dimensions of multi-scale local context features to  $N_C$  with the convolution layer. Then, these features are concatenated along the channel dimension and passed through two  $3 \times 3$  Conv-batch normalization-ReLU layers. The final convolution and sigmoid layers produce a two-channel attention map, where each local and global feature is multiplied with each channel to focus on the significant location. Then these multiplied features are added element-wise to construct a hybrid feature  $H_D$ . To strengthen the local continuity we do not reduce the dimension on the  $\frac{1}{4}$  scale feature. We will verify the effectiveness of the proposed decoder in section 4.4.

#### 3.5 Vertical CutDepth

Recently, a depth-specific data augmentation method named *CutDepth* has been proposed [Ishii and Yamashita, 2021], which replaces a part of the input image with the ground-truth depth map to provide diversity to the input image and enable the network to focus on the high-frequency area. In CutDepth, the coordinates  $(l, u)$  and size  $(w, h)$  of the cut region are randomly chosen. However, we believe that the vertical and horizontal directions should not be regarded equally for depth estimation based on the following discovery. A previous study [Dijk and Croon, 2019] suggested that the depth estimation networks mainly use vertical position in the image rather than apparent size or texture to predict the depth of arbitrary obstacles. This motivates us to propose *vertical CutDepth*, which enhances the original CutDepth by preserving the vertical geometric information. In vertical CutDepth, the ground-truth depth map replaces an area on  $I$  with the same location of  $Y$ , but the crop is not applied along the vertical direction. Therefore, the coordinate of the replacement region<table border="1">
<thead>
<tr>
<th>Method</th>
<th>Params (M)</th>
<th><math>\delta_1 \uparrow</math></th>
<th><math>\delta_2 \uparrow</math></th>
<th><math>\delta_3 \uparrow</math></th>
<th>AbsRel <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th><math>\log_{10} \downarrow</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>Eigen [Eigen <i>et al.</i>, 2014]</td>
<td>141</td>
<td>0.769</td>
<td>0.950</td>
<td>0.988</td>
<td>0.158</td>
<td>0.641</td>
<td>-</td>
</tr>
<tr>
<td>Fu [Fu <i>et al.</i>, 2018]</td>
<td>110</td>
<td>0.828</td>
<td>0.965</td>
<td>0.992</td>
<td>0.115</td>
<td>0.509</td>
<td>0.051</td>
</tr>
<tr>
<td>Yin [Yin <i>et al.</i>, 2019]</td>
<td>114</td>
<td>0.875</td>
<td>0.976</td>
<td>0.994</td>
<td>0.108</td>
<td>0.416</td>
<td>0.048</td>
</tr>
<tr>
<td>DAV [Huynh <i>et al.</i>, 2020]</td>
<td>25</td>
<td>0.882</td>
<td>0.980</td>
<td>0.996</td>
<td>0.108</td>
<td>0.412</td>
<td>-</td>
</tr>
<tr>
<td>BTS [Lee <i>et al.</i>, 2019]</td>
<td>47</td>
<td>0.885</td>
<td>0.978</td>
<td>0.994</td>
<td>0.110</td>
<td>0.392</td>
<td>0.047</td>
</tr>
<tr>
<td>Adabins[Bhat <i>et al.</i>, 2021]</td>
<td>78</td>
<td>0.903</td>
<td>0.984</td>
<td>0.997</td>
<td><u>0.103</u></td>
<td>0.364</td>
<td><u>0.044</u></td>
</tr>
<tr>
<td>DPT* [Ranftl <i>et al.</i>, 2021]</td>
<td>123</td>
<td><u>0.904</u></td>
<td><b>0.988</b></td>
<td><b>0.998</b></td>
<td>0.110</td>
<td><u>0.357</u></td>
<td>0.045</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td>62</td>
<td><b>0.915</b></td>
<td><b>0.988</b></td>
<td><u>0.997</u></td>
<td><b>0.098</b></td>
<td><b>0.344</b></td>
<td><b>0.042</b></td>
</tr>
</tbody>
</table>

Table 1: Performance on the NYU Depth V2 dataset. DPT\* is trained with an extra dataset.

$(l, u)$  and size  $(w, h)$  are calculated as follows:

$$\begin{aligned} (l, u) &= (\alpha \times W, 0) \\ (w, h) &= (\max((W - \alpha \times W) \times \beta \times p, 1), H) \end{aligned} \quad (1)$$

where  $\alpha$  and  $\beta$  are  $\mathcal{U}(0, 1)$ .  $p$  is a hyperparameter that is set at a value of  $(0, 1]$ . By maintaining the vertical range of the input RGB image, the network can capture the long-range vertical direction for better prediction, as shown in the results. We set the value of  $p$  to 0.75 by performing various settings of  $p$  (Section 4.4).

### 3.6 Training Loss

In order to calculate the distance between predicted output  $\hat{Y}$  and ground truth depth map  $Y$ , we use scale-invariant log scale loss [Eigen *et al.*, 2014] to train the model.  $y_i^*$  and  $y_i$  indicates  $i$ th pixel in  $\hat{Y}$  and  $Y$ . The equation of training loss is as follows:

$$L = \frac{1}{n} \sum_i d_i^2 - \frac{1}{2n^2} \left( \sum_i d_i^2 \right)^2 \quad (2)$$

where  $d_i = \log y_i - \log y_i^*$ .

## 4 Experiments

To validate our approach, we perform several experiments on the NYU Depth V2 and SUN RGB-D datasets. We compare our model with existing methods through quantitative and qualitative evaluation, and an ablation study is conducted to show the effectiveness of each contribution. Additionally, we provide other results on additional dataset in supplementary material.

### 4.1 Dataset

**NYU Depth V2** [Silberman *et al.*, 2012] contains  $640 \times 480$  images and corresponding depth maps of various indoor scenes acquired using a Microsoft Kinect camera. We train our network using approximately 24K images on a random crop of  $576 \times 448$  and test on 654 images. To facilitate a fair comparison, we perform the evaluation on a pre-defined center cropping by Eigen [Eigen *et al.*, 2014] with a maximum range of 10 m.

**SUN RGB-D** [Song *et al.*, 2015] contains approximately 10K RGB-D images of various indoor scenes captured by four different sensors, along with the corresponding depth

<table border="1">
<thead>
<tr>
<th>Method</th>
<th><math>\delta_1 \uparrow</math></th>
<th><math>\delta_2 \uparrow</math></th>
<th><math>\delta_3 \uparrow</math></th>
<th>AbsRel <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th><math>\log_{10} \downarrow</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>Yin [Yin <i>et al.</i>, 2019]</td>
<td>0.696</td>
<td>0.912</td>
<td>0.973</td>
<td>0.183</td>
<td>0.541</td>
<td>0.082</td>
</tr>
<tr>
<td>BTS [Lee <i>et al.</i>, 2019]</td>
<td>0.740</td>
<td>0.933</td>
<td>0.980</td>
<td>0.172</td>
<td>0.515</td>
<td>0.075</td>
</tr>
<tr>
<td>Adabins [Bhat <i>et al.</i>, 2021]</td>
<td><u>0.771</u></td>
<td><u>0.944</u></td>
<td><u>0.983</u></td>
<td><u>0.159</u></td>
<td><u>0.476</u></td>
<td><u>0.068</u></td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.814</b></td>
<td><b>0.964</b></td>
<td><b>0.991</b></td>
<td><b>0.144</b></td>
<td><b>0.418</b></td>
<td><b>0.061</b></td>
</tr>
</tbody>
</table>

Table 2: Performance on the SUN RGB-D dataset with the NYU Depth V2 trained model. We test the model without any fine-tuning.

and segmentation maps. We use this dataset for evaluating pre-trained models; thus, only the official test set of 5050 images is used. Image sizes are not constant throughout this dataset, and thus we resize the image to the largest multiple of 32 below the image size, and then pass the resized image to predict the depth map, which is then resized to the original image.

### 4.2 Implementation Details

We implement the proposed network using the PyTorch framework. For training, we use the one-cycle learning rate strategy with an Adam optimizer. The learning rate increases from  $3e-5$  to  $1e-4$  following a poly LR schedule with a factor of 0.9 in the first half of the total iteration, and then decreases from  $1e-4$  to  $3e-5$  in the last half. The total number of epochs is set to 25 with a batch size of 12. We use pre-trained weights from the MiT-b4 [Xie *et al.*, 2021] network and initialize our encoder. The values of  $N_C$ ,  $R_i$  and  $C_i$  are 64, [8, 4, 2, 1] and [64, 128, 320, 512], respectively.

In the case of data augmentation, the following strategies are used with the proposed vertical CutDepth with 50% probability: horizontal flips, random brightness( $\pm 0.2$ ), contrast( $\pm 0.2$ ), gamma( $\pm 20$ ), hue( $\pm 20$ ), saturation( $\pm 30$ ), and value( $\pm 20$ ). We apply  $p = 0.75$  for vertical CutDepth with 25% possibility.

### 4.3 Comparison with State-of-the-Arts

**NYU Depth V2.** Table 1 presents the performance comparison of the NYU Depth V2 dataset. DPT [Ranftl *et al.*, 2021] uses a much larger dataset of 1.4M images for training the model. As listed in the table, the proposed model shows state-of-the-art performance in most of the evaluation metrics which we attribute to the proposed architecture and enhanced depth-specific data augmentation method. Furthermore, our model achieves higher performance than the recently developed state-of-the-art models (Adabins, DPT) with lesser parameters. This suggests that the combination of the trans-Figure 3: Qualitative comparison with previous works on the NYU Depth V2 dataset.

Figure 4: Examples of estimated depth maps on the SUN RGB-D dataset.

former encoder and the proposed compact decoder clearly makes an important contribution to estimate accurate depth maps in an efficient manner. The visualized results are shown in Figure 3. In the figure, our model shows an accurate estimation of depth values for the provided example images and is more robust to various illumination conditions as compared to other methods.

**SUN RGB-D.** We test our network on an additional indoor dataset SUN RGB-D to show the generalization performance.

<table border="1">
<thead>
<tr>
<th>Method</th>
<th>Params (M)</th>
<th><math>\delta_1 \uparrow</math></th>
<th><math>\delta_2 \uparrow</math></th>
<th><math>\delta_3 \uparrow</math></th>
<th>AbsRel <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>Baseline - Dconv</td>
<td>4.03</td>
<td>0.898</td>
<td>0.986</td>
<td><b>0.997</b></td>
<td>0.110</td>
<td>0.359</td>
</tr>
<tr>
<td>Baseline - UNet</td>
<td>4.95</td>
<td>0.901</td>
<td><b>0.987</b></td>
<td><b>0.997</b></td>
<td>0.109</td>
<td>0.363</td>
</tr>
<tr>
<td>Ours (w/o SFF)</td>
<td><b>0.38</b></td>
<td>0.905</td>
<td>0.986</td>
<td><b>0.997</b></td>
<td>0.104</td>
<td>0.357</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.66</b></td>
<td><b>0.908</b></td>
<td><b>0.987</b></td>
<td><b>0.997</b></td>
<td><b>0.101</b></td>
<td><b>0.351</b></td>
</tr>
<tr>
<td>[Xie <i>et al.</i>, 2021]</td>
<td>3.19</td>
<td>0.893</td>
<td>0.983</td>
<td>0.995</td>
<td>0.112</td>
<td>0.379</td>
</tr>
<tr>
<td>[Lee <i>et al.</i>, 2019]</td>
<td>5.79</td>
<td>0.906</td>
<td>0.985</td>
<td><b>0.997</b></td>
<td>0.102</td>
<td>0.356</td>
</tr>
<tr>
<td>[Ranfli <i>et al.</i>, 2021]</td>
<td>14.15</td>
<td>0.907</td>
<td><b>0.987</b></td>
<td><b>0.997</b></td>
<td>0.103</td>
<td>0.354</td>
</tr>
</tbody>
</table>

Table 3: Comparison with multiple decoders. All results in this table are obtained from the same encoder.

The network is trained on the NYU Depth V2 dataset and evaluated with a test set of SUN RGB-D without any fine-tuning process. Table 2 compares the results with those obtained by comparative studies. The proposed approach outperforms competing methods in all metrics. As shown in Figure 4, reasonable result depth maps are generated through our model without additional training.

#### 4.4 Ablation Study

In this subsection, we validate the effectiveness of our approach through several experiments conducted on the NYU Depth V2 dataset.

**Comparison with different decoder designs.** Table 3 demonstrates the comparison results with different decoder design. In this experiment, vertical CutDepth is omitted to solely compare the effectiveness of the decoder. As our<table border="1">
<thead>
<tr>
<th>Corruption Type</th>
<th>Method</th>
<th><math>\delta_1 \uparrow</math></th>
<th><math>\delta_2 \uparrow</math></th>
<th><math>\delta_3 \uparrow</math></th>
<th>AbsRel <math>\downarrow</math></th>
<th>SqRel <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th>RMSElog <math>\downarrow</math></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">Clean</td>
<td>BTS</td>
<td>0.885</td>
<td>0.978</td>
<td>0.994</td>
<td>0.110</td>
<td>0.066</td>
<td>0.392</td>
<td>0.142</td>
</tr>
<tr>
<td>Adabins</td>
<td>0.903</td>
<td>0.984</td>
<td><b>0.997</b></td>
<td>0.103</td>
<td>0.057</td>
<td>0.364</td>
<td>0.130</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.915</b></td>
<td><b>0.988</b></td>
<td><b>0.997</b></td>
<td><b>0.098</b></td>
<td><b>0.049</b></td>
<td><b>0.344</b></td>
<td><b>0.124</b></td>
</tr>
<tr>
<td rowspan="3">Gaussian Noise</td>
<td>BTS</td>
<td>0.223</td>
<td>0.384</td>
<td>0.543</td>
<td>0.435</td>
<td>0.824</td>
<td>1.589</td>
<td>0.743</td>
</tr>
<tr>
<td>Adabins</td>
<td>0.347</td>
<td>0.553</td>
<td>0.708</td>
<td>0.343</td>
<td>0.578</td>
<td>1.299</td>
<td>0.544</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.775</b></td>
<td><b>0.940</b></td>
<td><b>0.983</b></td>
<td><b>0.161</b></td>
<td><b>0.126</b></td>
<td><b>0.541</b></td>
<td><b>0.198</b></td>
</tr>
<tr>
<td rowspan="3">Motion Blur</td>
<td>BTS</td>
<td>0.677</td>
<td>0.850</td>
<td>0.922</td>
<td>0.189</td>
<td>0.207</td>
<td>0.701</td>
<td>0.279</td>
</tr>
<tr>
<td>Adabins</td>
<td>0.697</td>
<td>0.859</td>
<td>0.927</td>
<td>0.180</td>
<td>0.182</td>
<td>0.643</td>
<td>0.262</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.807</b></td>
<td><b>0.946</b></td>
<td><b>0.981</b></td>
<td><b>0.139</b></td>
<td><b>0.103</b></td>
<td><b>0.494</b></td>
<td><b>0.183</b></td>
</tr>
<tr>
<td rowspan="3">Contrast</td>
<td>BTS</td>
<td>0.697</td>
<td>0.864</td>
<td>0.932</td>
<td>0.181</td>
<td>0.198</td>
<td>0.689</td>
<td>0.263</td>
</tr>
<tr>
<td>Adabins</td>
<td>0.654</td>
<td>0.836</td>
<td>0.917</td>
<td>0.198</td>
<td>0.234</td>
<td>0.752</td>
<td>0.283</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.860</b></td>
<td><b>0.971</b></td>
<td><b>0.992</b></td>
<td><b>0.117</b></td>
<td><b>0.074</b></td>
<td><b>0.427</b></td>
<td><b>0.152</b></td>
</tr>
<tr>
<td rowspan="3">Snow</td>
<td>BTS</td>
<td>0.410</td>
<td>0.649</td>
<td>0.803</td>
<td>0.298</td>
<td>0.423</td>
<td>1.114</td>
<td>0.458</td>
</tr>
<tr>
<td>Adabins</td>
<td>0.410</td>
<td>0.656</td>
<td>0.817</td>
<td>0.292</td>
<td>0.410</td>
<td>1.094</td>
<td>0.440</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.723</b></td>
<td><b>0.926</b></td>
<td><b>0.981</b></td>
<td><b>0.170</b></td>
<td><b>0.138</b></td>
<td><b>0.598</b></td>
<td><b>0.217</b></td>
</tr>
</tbody>
</table>

Table 4: Robustness experiment results on corrupted images of NYU Depth V2 datasets. The results of BTS and Adabins are obtained from distributed pre-trained weights.

<table border="1">
<thead>
<tr>
<th>Method</th>
<th><math>\delta_1 \uparrow</math></th>
<th><math>\delta_2 \uparrow</math></th>
<th><math>\delta_3 \uparrow</math></th>
<th>AbsRel <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th><math>\log_{10} \downarrow</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>Baseline</td>
<td>0.908</td>
<td>0.987</td>
<td>0.997</td>
<td>0.101</td>
<td>0.351</td>
<td>0.043</td>
</tr>
<tr>
<td>+ CutDepth</td>
<td>0.909</td>
<td>0.986</td>
<td>0.997</td>
<td>0.102</td>
<td>0.348</td>
<td><b>0.042</b></td>
</tr>
<tr>
<td>+ Ours (<math>p=0.25</math>)</td>
<td>0.911</td>
<td><b>0.988</b></td>
<td>0.997</td>
<td>0.102</td>
<td>0.354</td>
<td>0.043</td>
</tr>
<tr>
<td>+ Ours (<math>p=0.50</math>)</td>
<td>0.911</td>
<td><b>0.988</b></td>
<td>0.997</td>
<td>0.100</td>
<td>0.348</td>
<td><b>0.042</b></td>
</tr>
<tr>
<td>+ Ours (<math>p=0.75</math>)</td>
<td><b>0.915</b></td>
<td><b>0.988</b></td>
<td><b>0.998</b></td>
<td><b>0.098</b></td>
<td><b>0.343</b></td>
<td><b>0.042</b></td>
</tr>
<tr>
<td>+ Ours (<math>p=1.00</math>)</td>
<td>0.910</td>
<td><b>0.988</b></td>
<td>0.997</td>
<td>0.101</td>
<td>0.351</td>
<td>0.043</td>
</tr>
</tbody>
</table>

Table 5: Experimental results with data augmentation.

study aims to avoid computationally demanding decoders, we construct simple baselines and compare them with ours. Baseline-Dconv consists of consecutive deconvolution-batch normalization-ReLU blocks to obtain the desired depth map. In addition, Baseline-UNet is an improved structure from Baseline-Dconv that has skip connections between the encoder and decoder. As detailed in the table, our decoder achieves better performance than the baselines. Even without an SFF, it already shows better performance than other decoders while having fewer parameters. The powerful encoding ability of our encoder and the effectively designed decoder enables the network to produce a finely detailed depth map. In addition, our proposed SFF leverages additional performance of our model.

We additionally provide comparison with existing decoder architectures which integrates multi-scale features, in the bottom part of Table 3. Despite the compactness of the proposed decoder, our network outperforms other networks. Our decoder has only 0.66M parameters while the MLP-decoder [Xie *et al.*, 2021], BTS [Lee *et al.*, 2019] and DPT [Ranftl *et al.*, 2021] have 3.19M, 5.79M and 14.15M parameters, respectively, and thus, are highly heavier than ours. This indicates that we have effectively designed the restoring path for our encoder, which enables the proposed model to record fine performance with very few parameters.

**Effectiveness of the vertical CutDepth.** We perform an ablation study on the data augmentation method used to train

the network. The results are shown in Table 5. The first row of the table represents the baseline, which is only trained with traditional data augmentation except for CutDepth, and the second row shows the result obtained from adopting the basic CutDepth method. Then, we apply the proposed vertical CutDepth with different choices of hyperparameter  $p$ . As detailed in the table, CutDepth helps the model to achieve slightly better performance than the baseline. However, by applying vertical CutDepth, the network shows further improvement. This proves that utilizing vertical features enhances accurate depth estimation as compared to the case of simply cropping the random area. In addition, the model achieves the best performance with a setting of  $p = 0.75$ .

## 4.5 Robustness of the model

In this subsection, we demonstrate the robustness of the proposed method against natural image corruptions. Model robustness for depth estimation is essential because real world images always have a high possibility of being corrupted to a certain degree. Under these circumstances, it is beneficial to design a robust model so that it can perform the given task without being critically corrupted. Following the previous study on the robustness of CNNs [Hendrycks and Dietterich, 2018], we test our model on images that are corrupted by 16 different methods. Each corruption is applied with five different intensities, and the performance is averaged over all test images and all five intensities.

Table 4 presents the depth estimation results for the corrupted images of the NYU Depth V2 test set. Due to space constraints, we provide the complete table in the supplementary material and present results on a few corruption types in Table 4. The results show that our model is clearly more robust to all types of corruption than the compared models. The experimental results indicate that our model shows stronger robustness and thus is more appropriate for safety-critical applications.## 5 Conclusion

This paper proposes a new architecture for monocular depth estimation to deliver meaningful global and local features and generate a precisely estimated depth map. We further exploit the depth-specific data augmentation technique to improve the performance of the model by considering the knowledge that the use of vertical position is a crucial property of depth estimation. The proposed method shows improvement over state-of-the-art performance for the NYU Depth V2 dataset. Moreover, extensive experimental results demonstrate the effectiveness and generalization ability of our network.

## References

[Bhat *et al.*, 2021] Shariq Farooq Bhat, Ibraheem Alhashim, and Peter Wonka. Adabins: Depth estimation using adaptive bins. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 4009–4018, 2021.

[Chen *et al.*, 2019] Xiaotian Chen, Xuejin Chen, and Zheng-Jun Zha. Structure-aware residual pyramid network for monocular depth estimation. *arXiv preprint arXiv:1907.06023*, 2019.

[Dijk and Croon, 2019] Tom van Dijk and Guido de Croon. How do neural networks see depth in single images? In *Proceedings of the IEEE International Conference on Computer Vision*, pages 2183–2191, 2019.

[Dosovitskiy *et al.*, 2020] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*, 2020.

[Eigen *et al.*, 2014] David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. In *Advances in neural information processing systems (NIPS)*, pages 2366–2374, 2014.

[Fu *et al.*, 2018] Huan Fu, Mingming Gong, Chaohui Wang, Kayhan Batmanghelich, and Dacheng Tao. Deep ordinal regression network for monocular depth estimation. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 2002–2011, 2018.

[Garg *et al.*, 2016] Ravi Garg, Vijay Kumar BG, Gustavo Carneiro, and Ian Reid. Unsupervised cnn for single view depth estimation: Geometry to the rescue. In *Proceedings of the European Conference on Computer Vision (ECCV)*, pages 740–756. Springer, 2016.

[Geiger *et al.*, 2013] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. *The International Journal of Robotics Research*, 32(11):1231–1237, 2013.

[Ghiassi *et al.*, 2021] Golnaz Ghiassi, Yin Cui, Aravind Srinivas, Rui Qian, Tsung-Yi Lin, Ekin D Cubuk, Quoc V Le, and Barret Zoph. Simple copy-paste is a strong data augmentation method for instance segmentation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 2918–2928, 2021.

[Hendrycks and Dietterich, 2018] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In *International Conference on Learning Representations*, 2018.

[Huynh *et al.*, 2020] Lam Huynh, Phong Nguyen-Ha, Jiri Matas, Esa Rahtu, and Janne Heikkilä. Guiding monocular depth estimation using depth-attention volume. In *European Conference on Computer Vision*, pages 581–597. Springer, 2020.

[Ishii and Yamashita, 2021] Yasunori Ishii and Takayoshi Yamashita. Cutdepth: Edge-aware data augmentation in depth estimation. *arXiv preprint arXiv:2107.07684*, 2021.

[Kim *et al.*, 2020] Doyeon Kim, Sihaeng Lee, Janghyeon Lee, and Junmo Kim. Leveraging contextual information for monocular depth estimation. *IEEE Access*, 8:147808–147817, 2020.

[Koch *et al.*, 2018] Tobias Koch, Lukas Liebel, Friedrich Fraundorfer, and Marco Korner. Evaluation of cnn-based single-image depth estimation methods. In *Proceedings of the European Conference on Computer Vision (ECCV) Workshops*, pages 0–0, 2018.

[Lee *et al.*, 2019] Jin Han Lee, Myung-Kyu Han, Dong Wook Ko, and Il Hong Suh. From big to small: Multi-scale local planar guidance for monocular depth estimation. *arXiv preprint arXiv:1907.10326*, 2019.

[Luo *et al.*, 2016] Wenjie Luo, Yujia Li, Raquel Urtasun, and Richard Zemel. Understanding the effective receptive field in deep convolutional neural networks. In *Proceedings of the 30th International Conference on Neural Information Processing Systems*, pages 4905–4913, 2016.

[Ramamonjisoa and Lepetit, 2019] Michael Ramamonjisoa and Vincent Lepetit. Sharpnet: Fast and accurate recovery of occluding contours in monocular depth estimation. In *Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops*, pages 0–0, 2019.

[Ranftl *et al.*, 2021] René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. *arXiv preprint arXiv:2103.13413*, 2021.

[Saxena *et al.*, 2008] Ashutosh Saxena, Min Sun, and Andrew Y Ng. Make3d: Learning 3d scene structure from a single still image. *IEEE transactions on pattern analysis and machine intelligence*, 31(5):824–840, 2008.

[Silberman *et al.*, 2012] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgb-d images. In *European conference on computer vision*, pages 746–760. Springer, 2012.

[Song *et al.*, 2015] Shuran Song, Samuel P. Lichtenberg, and Jianxiong Xiao. Sun rgb-d: A rgb-d scene understanding benchmark suite. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, June 2015.[Swami *et al.*, 2020] Kunal Swami, Prasanna Vishnu Bondada, and Pankaj Kumar Bajpai. Aced: Accurate and edge-consistent monocular depth estimation. In *2020 IEEE International Conference on Image Processing (ICIP)*, pages 1376–1380. IEEE, 2020.

[Vaswani *et al.*, 2017] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Advances in neural information processing systems*, pages 5998–6008, 2017.

[Xie *et al.*, 2021] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. *arXiv preprint arXiv:2105.15203*, 2021.

[Yin *et al.*, 2019] Wei Yin, Yifan Liu, Chunhua Shen, and Youliang Yan. Enforcing geometric constraints of virtual normal for depth prediction. In *Proceedings of the IEEE International Conference on Computer Vision*, pages 5684–5693, 2019.

[Yoo *et al.*, 2020] Jaejun Yoo, Namhyuk Ahn, and Kyung-Ah Sohn. Rethinking data augmentation for image super-resolution: A comprehensive analysis and a new strategy. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 8375–8384, 2020.

[Yun *et al.*, 2019] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. *CoRR*, abs/1905.04899, 2019.

[Zheng *et al.*, 2021] Sixiao Zheng, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu, Jianfeng Feng, Tao Xiang, Philip HS Torr, et al. Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6881–6890, 2021.## 6 Appendix: Additional dataset results

In this section, we provide additional results on KITTI [Geiger *et al.*, 2013] and iBims-1 [Koch *et al.*, 2018] datasets. KITTI is an outdoor depth estimation dataset and iBims-1 is an indoor dataset.

### 6.1 KITTI

KITTI [Geiger *et al.*, 2013] contains stereo camera images and corresponding 3D LiDAR scans of various driving scenes acquired by car mounted sensors. The size of RGB images is around  $1224 \times 368$ . We train our network using approximately 23K images on a random crop of  $704 \times 352$  and test on 697 images. To compare our performance with previous works, we use the crop as defined by Garg [Garg *et al.*, 2016] and a maximum value of 80m for evaluation. The results on the KITTI dataset are shown in Table 6. As shown in the table, our model outperforms other previous studies.

<table border="1">
<thead>
<tr>
<th>Method</th>
<th>Params (M)</th>
<th><math>\delta_1 \uparrow</math></th>
<th><math>\delta_2 \uparrow</math></th>
<th><math>\delta_3 \uparrow</math></th>
<th>AbsRel <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th>RMSE log <math>\downarrow</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>Fu [Fu <i>et al.</i>, 2018]</td>
<td>110</td>
<td>0.932</td>
<td>0.984</td>
<td>0.994</td>
<td>0.072</td>
<td>2.727</td>
<td>0.120</td>
</tr>
<tr>
<td>Yin [Yin <i>et al.</i>, 2019]</td>
<td>114</td>
<td>0.938</td>
<td>0.984</td>
<td><u>0.998</u></td>
<td>0.072</td>
<td>3.258</td>
<td>0.117</td>
</tr>
<tr>
<td>BTS [Lee <i>et al.</i>, 2019]</td>
<td>113</td>
<td>0.956</td>
<td>0.993</td>
<td><u>0.998</u></td>
<td>0.059</td>
<td>2.756</td>
<td>0.088</td>
</tr>
<tr>
<td>DPT* [Ranftl <i>et al.</i>, 2021]</td>
<td>123</td>
<td>0.959</td>
<td>0.995</td>
<td><b>0.999</b></td>
<td>0.062</td>
<td>2.573</td>
<td>0.092</td>
</tr>
<tr>
<td>Adabins [Bhat <i>et al.</i>, 2021]</td>
<td>78</td>
<td><u>0.964</u></td>
<td><u>0.995</u></td>
<td><b>0.999</b></td>
<td><u>0.058</u></td>
<td><u>2.360</u></td>
<td><u>0.088</u></td>
</tr>
<tr>
<td><b>Ours</b></td>
<td>62</td>
<td><b>0.967</b></td>
<td><b>0.996</b></td>
<td><b>0.999</b></td>
<td><b>0.057</b></td>
<td><b>2.297</b></td>
<td><b>0.086</b></td>
</tr>
</tbody>
</table>

Table 6: Performance on the KITTI dataset. DPT\* is trained with extra dataset.

### 6.2 iBims-1

iBims-1 [Koch *et al.*, 2018] (independent Benchmark images and matched scans version 1) is a high quality RGB-D dataset acquired using a digital single-lens reflex (DSLR) camera and high-precision laser scanner. iBims-1 can be characterized by accurate edges and planar regions, consistent depth values, and accurate absolute distances. We evaluate with our NYU Depth V2 trained model without any fine-tuning. Results on iBims-1 dataset are listed in Table 7.

<table border="1">
<thead>
<tr>
<th>Method</th>
<th><math>\delta_1 \uparrow</math></th>
<th><math>\delta_2 \uparrow</math></th>
<th><math>\delta_3 \uparrow</math></th>
<th>AbsRel <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th>log10 <math>\downarrow</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>VNL [Yin <i>et al.</i>, 2019]</td>
<td>0.54</td>
<td>0.84</td>
<td>0.93</td>
<td>0.24</td>
<td>1.06</td>
<td><u>0.11</u></td>
</tr>
<tr>
<td>BTS [Lee <i>et al.</i>, 2019]</td>
<td>0.53</td>
<td>0.84</td>
<td>0.94</td>
<td>0.24</td>
<td>1.10</td>
<td>0.12</td>
</tr>
<tr>
<td>DORN [Fu <i>et al.</i>, 2018]</td>
<td>0.55</td>
<td>0.81</td>
<td>0.92</td>
<td>0.24</td>
<td>1.13</td>
<td>0.12</td>
</tr>
<tr>
<td>AdaBins [Bhat <i>et al.</i>, 2021]</td>
<td>0.55</td>
<td>0.86</td>
<td><u>0.95</u></td>
<td><u>0.22</u></td>
<td>1.07</td>
<td><u>0.11</u></td>
</tr>
<tr>
<td>SharpNet [Ramamonjisoa and Lepetit, 2019]</td>
<td>0.59</td>
<td>0.84</td>
<td>0.94</td>
<td>0.26</td>
<td>1.07</td>
<td><u>0.11</u></td>
</tr>
<tr>
<td>ACED [Swami <i>et al.</i>, 2020]</td>
<td><u>0.60</u></td>
<td><u>0.87</u></td>
<td><u>0.95</u></td>
<td><b>0.20</b></td>
<td><u>1.03</u></td>
<td><b>0.10</b></td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.61</b></td>
<td><b>0.89</b></td>
<td><b>0.96</b></td>
<td><b>0.20</b></td>
<td><b>1.01</b></td>
<td><b>0.10</b></td>
</tr>
</tbody>
</table>

Table 7: Performance on the iBims-1 dataset.## 7 Appendix: Robustness of the Model

In Table 8, we report a full table of the results on the corrupted NYU Depth V2 dataset. (Section 4.5 of the main paper)

<table border="1">
<thead>
<tr>
<th colspan="2">Corruption Type</th>
<th>Method</th>
<th><math>\delta_1 \uparrow</math></th>
<th><math>\delta_2 \uparrow</math></th>
<th><math>\delta_3 \uparrow</math></th>
<th>AbsRel <math>\downarrow</math></th>
<th>SqRel <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th>RMSElog <math>\downarrow</math></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="2" rowspan="3">Clean</td>
<td>BTS</td>
<td>0.885</td>
<td>0.978</td>
<td>0.994</td>
<td>0.110</td>
<td>0.066</td>
<td>0.392</td>
<td>0.142</td>
</tr>
<tr>
<td>Adabins</td>
<td>0.903</td>
<td>0.984</td>
<td><b>0.997</b></td>
<td>0.103</td>
<td>0.057</td>
<td>0.364</td>
<td>0.130</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.915</b></td>
<td><b>0.988</b></td>
<td><b>0.997</b></td>
<td><b>0.098</b></td>
<td><b>0.049</b></td>
<td><b>0.344</b></td>
<td><b>0.124</b></td>
</tr>
<tr>
<td rowspan="8">Noise</td>
<td rowspan="3">Gaussian Noise</td>
<td>BTS</td>
<td>0.223</td>
<td>0.384</td>
<td>0.543</td>
<td>0.435</td>
<td>0.824</td>
<td>1.589</td>
<td>0.743</td>
</tr>
<tr>
<td>Adabins</td>
<td>0.347</td>
<td>0.553</td>
<td>0.708</td>
<td>0.343</td>
<td>0.578</td>
<td>1.299</td>
<td>0.544</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.775</b></td>
<td><b>0.940</b></td>
<td><b>0.983</b></td>
<td><b>0.161</b></td>
<td><b>0.126</b></td>
<td><b>0.541</b></td>
<td><b>0.198</b></td>
</tr>
<tr>
<td rowspan="2">Shot</td>
<td>BTS</td>
<td>0.280</td>
<td>0.448</td>
<td>0.600</td>
<td>0.399</td>
<td>0.736</td>
<td>1.482</td>
<td>0.669</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.436</b></td>
<td><b>0.653</b></td>
<td><b>0.791</b></td>
<td><b>0.293</b></td>
<td><b>0.460</b></td>
<td><b>1.141</b></td>
<td><b>0.454</b></td>
</tr>
<tr>
<td rowspan="3">Impulse</td>
<td>BTS</td>
<td>0.116</td>
<td>0.249</td>
<td>0.420</td>
<td>0.504</td>
<td>1.006</td>
<td>1.818</td>
<td>0.875</td>
</tr>
<tr>
<td>Adabins</td>
<td>0.377</td>
<td>0.589</td>
<td>0.736</td>
<td>0.327</td>
<td>0.541</td>
<td>1.246</td>
<td>0.518</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.760</b></td>
<td><b>0.938</b></td>
<td><b>0.984</b></td>
<td><b>0.167</b></td>
<td><b>0.131</b></td>
<td><b>0.556</b></td>
<td><b>0.204</b></td>
</tr>
<tr>
<td rowspan="4">Speckle</td>
<td>BTS</td>
<td>0.456</td>
<td>0.633</td>
<td>0.756</td>
<td>0.302</td>
<td>0.500</td>
<td>1.159</td>
<td>0.492</td>
</tr>
<tr>
<td>Adabins</td>
<td>0.639</td>
<td>0.834</td>
<td>0.918</td>
<td>0.200</td>
<td>0.244</td>
<td>0.805</td>
<td>0.294</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.830</b></td>
<td><b>0.965</b></td>
<td><b>0.991</b></td>
<td><b>0.136</b></td>
<td><b>0.091</b></td>
<td><b>0.467</b></td>
<td><b>0.168</b></td>
</tr>
<tr>
<td>Motion</td>
<td>BTS</td>
<td>0.677</td>
<td>0.850</td>
<td>0.922</td>
<td>0.189</td>
<td>0.207</td>
<td>0.701</td>
<td>0.279</td>
</tr>
<tr>
<td rowspan="4">Blur</td>
<td>Adabins</td>
<td>0.697</td>
<td>0.859</td>
<td>0.927</td>
<td>0.180</td>
<td>0.182</td>
<td>0.643</td>
<td>0.262</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.807</b></td>
<td><b>0.946</b></td>
<td><b>0.981</b></td>
<td><b>0.139</b></td>
<td><b>0.103</b></td>
<td><b>0.494</b></td>
<td><b>0.183</b></td>
</tr>
<tr>
<td rowspan="2">Defocus</td>
<td>BTS</td>
<td>0.511</td>
<td>0.684</td>
<td>0.786</td>
<td>0.276</td>
<td>0.415</td>
<td>1.002</td>
<td>0.436</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.599</b></td>
<td><b>0.769</b></td>
<td><b>0.859</b></td>
<td><b>0.227</b></td>
<td><b>0.277</b></td>
<td><b>0.793</b></td>
<td><b>0.341</b></td>
</tr>
<tr>
<td rowspan="4">Glass</td>
<td>BTS</td>
<td>0.671</td>
<td>0.855</td>
<td>0.927</td>
<td>0.193</td>
<td>0.224</td>
<td>0.747</td>
<td>0.285</td>
</tr>
<tr>
<td>Adabins</td>
<td>0.743</td>
<td><b>0.914</b></td>
<td>0.967</td>
<td>0.165</td>
<td>0.149</td>
<td>0.619</td>
<td>0.223</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.770</b></td>
<td><b>0.914</b></td>
<td><b>0.978</b></td>
<td><b>0.155</b></td>
<td><b>0.132</b></td>
<td><b>0.573</b></td>
<td><b>0.202</b></td>
</tr>
<tr>
<td rowspan="2">Gaussian</td>
<td>BTS</td>
<td>0.530</td>
<td>0.688</td>
<td>0.779</td>
<td>0.274</td>
<td>0.422</td>
<td>0.989</td>
<td>0.437</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.595</b></td>
<td><b>0.738</b></td>
<td><b>0.814</b></td>
<td><b>0.244</b></td>
<td><b>0.341</b></td>
<td><b>0.847</b></td>
<td><b>0.379</b></td>
</tr>
<tr>
<td rowspan="8">Digital</td>
<td rowspan="2">Brightness</td>
<td>BTS</td>
<td>0.842</td>
<td>0.965</td>
<td>0.990</td>
<td>0.124</td>
<td>0.084</td>
<td>0.457</td>
<td>0.166</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.862</b></td>
<td><b>0.972</b></td>
<td><b>0.994</b></td>
<td><b>0.117</b></td>
<td><b>0.073</b></td>
<td><b>0.427</b></td>
<td><b>0.152</b></td>
</tr>
<tr>
<td rowspan="2">Contrast</td>
<td>BTS</td>
<td>0.697</td>
<td>0.864</td>
<td>0.932</td>
<td>0.181</td>
<td>0.198</td>
<td>0.689</td>
<td>0.263</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.654</b></td>
<td><b>0.836</b></td>
<td><b>0.917</b></td>
<td><b>0.198</b></td>
<td><b>0.234</b></td>
<td><b>0.752</b></td>
<td><b>0.283</b></td>
</tr>
<tr>
<td rowspan="2">Saturation</td>
<td>BTS</td>
<td>0.814</td>
<td>0.950</td>
<td>0.983</td>
<td>0.135</td>
<td>0.103</td>
<td>0.505</td>
<td>0.182</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.839</b></td>
<td><b>0.965</b></td>
<td><b>0.991</b></td>
<td><b>0.125</b></td>
<td><b>0.086</b></td>
<td><b>0.465</b></td>
<td><b>0.162</b></td>
</tr>
<tr>
<td rowspan="2">JPEG Compression</td>
<td>BTS</td>
<td>0.786</td>
<td>0.942</td>
<td>0.983</td>
<td>0.154</td>
<td>0.124</td>
<td>0.532</td>
<td>0.195</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.804</b></td>
<td><b>0.954</b></td>
<td><b>0.988</b></td>
<td><b>0.153</b></td>
<td><b>0.115</b></td>
<td><b>0.493</b></td>
<td><b>0.182</b></td>
</tr>
<tr>
<td rowspan="8">Weather</td>
<td rowspan="2">Snow</td>
<td>BTS</td>
<td>0.410</td>
<td>0.649</td>
<td>0.803</td>
<td>0.298</td>
<td>0.423</td>
<td>1.114</td>
<td>0.458</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.410</b></td>
<td><b>0.656</b></td>
<td><b>0.817</b></td>
<td><b>0.292</b></td>
<td><b>0.410</b></td>
<td><b>1.094</b></td>
<td><b>0.440</b></td>
</tr>
<tr>
<td rowspan="2">Spatter</td>
<td>BTS</td>
<td>0.705</td>
<td>0.878</td>
<td>0.945</td>
<td>0.176</td>
<td>0.168</td>
<td>0.642</td>
<td>0.250</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.699</b></td>
<td><b>0.890</b></td>
<td><b>0.964</b></td>
<td><b>0.173</b></td>
<td><b>0.155</b></td>
<td><b>0.625</b></td>
<td><b>0.234</b></td>
</tr>
<tr>
<td rowspan="2">Fog</td>
<td>BTS</td>
<td>0.588</td>
<td>0.798</td>
<td>0.893</td>
<td>0.227</td>
<td>0.273</td>
<td>0.835</td>
<td>0.332</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.523</b></td>
<td><b>0.748</b></td>
<td><b>0.873</b></td>
<td><b>0.252</b></td>
<td><b>0.308</b></td>
<td><b>0.898</b></td>
<td><b>0.357</b></td>
</tr>
<tr>
<td rowspan="2">Frost</td>
<td>BTS</td>
<td>0.515</td>
<td>0.734</td>
<td>0.850</td>
<td>0.261</td>
<td>0.359</td>
<td>0.996</td>
<td>0.400</td>
</tr>
<tr>
<td><b>Ours</b></td>
<td><b>0.439</b></td>
<td><b>0.691</b></td>
<td><b>0.842</b></td>
<td><b>0.280</b></td>
<td><b>0.398</b></td>
<td><b>1.074</b></td>
<td><b>0.413</b></td>
</tr>
</tbody>
</table>

Table 8: Robustness experiment results on corrupted images of NYU Depth V2 datasets.## 8 Appendix: Detailed structure of baseline decoder

We illustrate the detailed structure of Baseline-DConv and Baseline-UNet in Figure 5. We use transposed convolution with  $K = 3, S = 2, P = 1$  parameters to upscale the given feature into 2x size. For Baseline-UNet, features from encoder  $F_E^3, F_E^2, F_E^1$  are concatenated in channel dimension.

(a) Baseline-DConv: The decoder structure starts with an input feature  $F_E^4$  on the right. It passes through a series of blocks:  $1/16 \times 1/16 \times 512$ ,  $1/8 \times 1/8 \times 256$ ,  $1/4 \times 1/4 \times 128$ ,  $1/2 \times 1/2 \times 64$ , and  $1 \times 1 \times 32$ . Each block is followed by a **ConvTranspose**, **Batch Norm**, and **ReLU** layer. The final block is followed by a **Conv-ReLU-Conv** layer and a **Sigmoid** layer, resulting in the output  $\hat{y}$ .

(b) Baseline-UNet: The decoder structure starts with an input feature  $F_E^4$  on the right. It passes through a series of blocks:  $1/16 \times 1/16 \times 1024$ ,  $1/8 \times 1/8 \times 576$ ,  $1/4 \times 1/4 \times 256$ ,  $1/2 \times 1/2 \times 64$ , and  $1 \times 1 \times 32$ . Each block is followed by a **ConvTranspose**, **Batch Norm**, and **ReLU** layer. The final block is followed by a **Conv-ReLU-Conv** layer and a **Sigmoid** layer, resulting in the output  $\hat{y}$ . Additionally, features from the encoder are concatenated:  $F_E^3$  is concatenated with the output of the  $1/16 \times 1/16 \times 1024$  block,  $F_E^2$  is concatenated with the output of the  $1/8 \times 1/8 \times 576$  block, and  $F_E^1$  is concatenated with the output of the  $1/4 \times 1/4 \times 256$  block.

Figure 5: The detailed structure of (a) Baseline-DConv (b) Baseline-UNet.

