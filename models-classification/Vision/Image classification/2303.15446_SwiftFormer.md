# SwiftFormer: Efficient Additive Attention for Transformer-based Real-time Mobile Vision Applications

Abdelrahman Shaker<sup>1</sup>✉ Muhammad Maaz<sup>1</sup> Hanoona Rasheed<sup>1</sup> Salman Khan<sup>1</sup>  
Ming-Hsuan Yang<sup>2,3,4</sup> Fahad Shahbaz Khan<sup>1,5</sup>

<sup>1</sup>Mohamed bin Zayed University of AI <sup>2</sup>University of California, Merced

<sup>3</sup>Yonsei University <sup>4</sup>Google Research <sup>5</sup>Linköping University

## Abstract

*Self-attention has become a defacto choice for capturing global context in various vision applications. However, its quadratic computational complexity with respect to image resolution limits its use in real-time applications, especially for deployment on resource-constrained mobile devices. Although hybrid approaches have been proposed to combine the advantages of convolutions and self-attention for a better speed-accuracy trade-off, the expensive matrix multiplication operations in self-attention remain a bottleneck. In this work, we introduce a novel efficient additive attention mechanism that effectively replaces the quadratic matrix multiplication operations with linear element-wise multiplications. Our design shows that the key-value interaction can be replaced with a linear layer without sacrificing any accuracy. Unlike previous state-of-the-art methods, our efficient formulation of self-attention enables its usage at all stages of the network. Using our proposed efficient additive attention, we build a series of models called “SwiftFormer” which achieves state-of-the-art performance in terms of both accuracy and mobile inference speed. Our small variant achieves 78.5% top-1 ImageNet-1K accuracy with only 0.8 ms latency on iPhone 14, which is more accurate and 2× faster compared to MobileViT-v2. Code: <https://tinyurl.com/5ft8v46w>*

## 1. Introduction

In recent years, transformer models have shown remarkable success in various vision applications such as classification [9, 10, 23, 24, 44], detection [2, 28, 33, 58, 61], and segmentation [4, 40]. However, deploying these models on resource-constrained mobile devices for real-time applications remains challenging due to their inherently complex

Figure 1. **Latency vs Accuracy Comparison.** Compared to the recent EfficientFormer-L1 [20], our SwiftFormer-L1 achieves an absolute gain of 1.7% in terms of top-1 accuracy with the same latency and without requiring any neural architecture search.

nature [20, 29]. Specifically, vision transformers (ViTs) rely on global self-attention, which has a quadratic complexity with respect to the input image resolution, making it impractical for deployment on low-powered mobile devices [31]. As a result, convolutional neural networks (CNNs) are still the preferred choice for real-time deployment on mobile devices, primarily because the convolution operation is computationally efficient [15, 39]. However, a major limitation of CNNs is their reliance on local connections and stationary weights, which can limit their ability to adapt to variable input resolutions and capture long-range dependencies in the data. Therefore, developing more efficient and flexible models that combine the strengths of both CNNs and transformers is critical, particularly for mobile devices with limited computational resources.

To achieve this goal, several hybrid approaches have

✉ abdelrahman.youssief@mbzuai.ac.aebeen proposed that use lightweight CNN modules in the high-resolution early stages and self-attention in the low-resolution later stages [20, 29, 57]. This approach effectively increases the receptive field of the network and strives to achieve a trade-off between speed and accuracy. Furthermore, different efficient variants of computing self-attention have been proposed to reduce the model complexity. These include computing attention across feature dimensions to implicitly model the global context [29], computing attention within local windows [24], pooling spatial features before applying self-attention [10], and sparsely attending to a fixed number of tokens [34], to name a few.

Although these approaches effectively reduce network complexity, they still involve inefficient matrix multiplication operations that significantly impact latency on mobile devices. To address this issue, Mehta et al. [31] propose a separable self-attention mechanism that replaces matrix multiplication operations to element-wise multiplications. This is achieved by projecting queries to context scores, followed by element-wise multiplication with keys to calculate context vectors for encoding global context.

In this work, we propose efficient additive attention, which eliminates the need for expensive matrix multiplication operations in computing self-attention. Additionally, we propose to compute the global context using only the query-key interactions followed by a linear transformation, without requiring explicit key-value interactions. This significantly reduces the computational complexity and enables us to use the proposed attention block in all stages of the network. Our contributions are as follows:

- • We introduce *efficient additive attention*, a new approach for computing self-attention in vision backbones that eliminates the need for expensive matrix multiplication operations, significantly reducing the computational complexity of the model.
- • Unlike previous methods, our proposed efficient attention design can be used at all stages of the network, enabling more effective contextual information capture and achieving superior speed-accuracy trade-off.
- • We build a series of efficient generic classification models called “SwiftFormer”, which utilize our proposed *efficient additive attention*. Our *small* model achieves 78.5% top-1 ImageNet-1K [8] accuracy while running at only 0.8 ms latency on iPhone 14. Moreover, our large model achieves 83.0% accuracy with a latency of only 1.9 ms. Our model achieves state-of-the-art performance, outperforming recent MobileViT-v2 [31] and EfficientFormer [20] by obtaining a better trade-off between accuracy and latency (see Fig. 1).

## 2. Related Work

**Efficient CNNs:** Designing efficient CNNs for mobile vision applications has received much attention in recent years. MobileNet architectures [15, 16, 39] propose depth-wise separable convolutions as well as efficient inverted residual blocks for improved performance on various vision tasks. Other methods aim to improve the efficiency by leveraging depth-wise dilated convolutions [32], channel shuffling and pointwise group convolutions [27, 59], network pruning [12, 50], low bit-width [1, 17], and neural architecture search [15, 41]. CNN-based methods are well-performing, efficient, and fast to train and run on edge devices, resulting in widespread usage in the industry. However, they are spatially local and lack global interaction between the features, which deeply affects their performance.

**Efficient Transformers:** ViTs [9] have been widely used in numerous vision tasks, and significant advances have been made in terms of data efficiency [22, 44], transformer architecture [3, 20, 30, 56], and token mechanisms [43, 52]. Reducing the number of visual tokens is a major modification in the transformer architecture for efficient deployment. Instead of using a fixed feature representation through the whole architecture, some methods employ a hierarchical design where the resolution is gradually decreased through the stages, including down-sampling techniques [10, 14, 38, 40] and pyramidal structures [47, 49]. Recently, a few methods [11, 34, 37] propose token sparsification techniques to encode only a subset of the most informative tokens.

Numerous approaches have recently been proposed to reduce the quadratic complexity of self-attention, the computational bottleneck in transformer-based architectures, by computing its approximated variants [5, 19, 29, 31, 34, 45, 46]. EdgeViT [34] uses a global sparse attention module attending only to a few tokens to improve the efficiency, while [48] down-samples the key and value vectors that lead to a better efficiency-accuracy trade-off. EdgeNeXt [29] adopts transposed self-attention operation to compute the attention maps across the channel dimension instead of the spatial dimension, followed by token mixing, to have a linear complexity with respect to the number of tokens. Reformer [19] replaces the dot-product attention with a locality-sensitive hashing to group the tokens and reduced the complexity from  $O(n^2)$  to  $O(n \log n)$ . However, this design is only efficient on longer sequences, which is typically not the case for ViTs. LinFormer [46] is a low-rank matrix factorization method that approximates the self-attention matrix with a low-rank matrix, reducing the complexity from  $O(n^2)$  to  $O(n)$ . Although matrix factorization methods theoretically reduce the complexity of self-attention, they use expensive projections for computing attention, which may not reflect the reduction in FLOPs and parameters into actual speed on mobile platforms.

Although these methods show promise and have reducedthe complexity of self-attention theoretically, they are inadequate for reducing the inference speed for mobile deployment. Since the complexity of the multi-headed self-attention (MHSA) is higher in the earlier stage compared to the last stages, EfficientFormer [20] incorporates MHSA in the last stage only to learn contextual information from the high-level features without increasing the inference speed significantly. Recently, MobileViT-v2 [31] proposes separable self-attention that uses element-wise operations instead of the dot-product to compute the attention maps with linear complexity. Different from the existing approaches, we propose a consistent hybrid design with an *efficient additive attention* mechanism to model the contextual information with linear complexity. Instead of capturing the pairwise interactions between keys, queries, and values using the dot-product, we use element-wise operations with learnable attention weights to model the interactions between query and keys only, leading to better inference speed.

### 3. Method

**Motivation:** To motivate our method, we first distinguish three desirable characteristics to be considered when designing an efficient yet accurate approach for resource constrained mobile devices.

**Efficient Global Context Modeling:** As discussed earlier, most existing approaches either employ the standard MHSA or an approximated variant to learn the global context. However, they struggle to operate as fast as MobileNets on resource-constrained devices. This is likely due to the computation-intensive multiplicative operations during attention computation or reliance on advanced reshaping and indexing operations in these approaches. For instance, the recent MobileViT-v2 [31] is  $2\times$  slower than MobileNet-v2 [39]. Instead of using matrix multiplications, we argue that encoding the global context using an efficient additive attention design can reduce the operations with respect to the number of tokens. This is expected to help operate at comparable speed and model size, while achieving superior accuracy compared to MobileNets.

**Rethinking key-value interactions:** Other than multiplicative operations during attention computation, additive attention has been recently explored in the NLP domain [51]. However, in the standard form, it performs three-step processing to model query, key, and value interactions. Each step feeds into the subsequent one, thereby requiring sequential processing. Here, we rethink the additive attention for vision tasks by alleviating the need to compute explicit interactions between key-value. We argue that eliminating key-value interactions and replacing them with a simple linear transformation empirically encodes better contextual representation. Our design encodes only global queries and key interactions to learn the global contextual information, followed by a linear transformation to compute global

context-aware attention weights.

**Consistent Hybrid Design:** Most existing works employ MHSA or the approximated variant in the last stages, while avoiding its usage in the earlier stages. This is because the computational complexity of MHSA grows quadratically with the length of the tokens, making it impractical to incorporate during the initial stages. This constraint adds to the design complexity and requires a careful selection of stages, where MHSA can be applied. In contrast, our proposed SwiftFormer module has linear complexity with respect to the token length and can be incorporated in all stages to learn consistent global context at each scale. This consistency improves the model performance and makes it more generalizable and scalable for high-resolution images.

### 3.1. Overview of Attention Modules

Vision transformer models are built upon the self-attention (see Fig. 2 (a)), which can effectively model the interactions between the input tokens. Specifically, the self-attention has  $\mathbf{x}$  as an input, where  $\mathbf{x} \in \mathbb{R}^{n \times d}$ , comprising  $n$  tokens with  $d$ -dimensional embedding vector. The input  $\mathbf{x}$  is projected to query ( $\mathbf{Q}$ ), key ( $\mathbf{K}$ ), and value ( $\mathbf{V}$ ) using three matrices,  $\mathbf{W}_{\mathbf{Q}}$ ,  $\mathbf{W}_{\mathbf{K}}$ , and  $\mathbf{W}_{\mathbf{V}}$ . Each self-attention layer comprises  $h$  heads, which allows the model to attend to different views of the input. The self-attention can be described as:

$$\hat{\mathbf{x}} = \text{Softmax}\left(\frac{\mathbf{Q} \cdot \mathbf{K}^{\top}}{\sqrt{d}}\right) \cdot \mathbf{V}. \quad (1)$$

The attention scores between each pair of tokens in  $\mathbf{Q}$  and  $\mathbf{K}$  are computed using the dot-product operation. Next, these scores are normalized followed by Softmax to weigh the interactions between the tokens. Finally, the weighted interactions are multiplied by  $\mathbf{V}$  using the dot-product operation to produce the final weighted output. Overall, the complexity of the self-attention is  $O(n^2 \cdot d)$ , where  $n$  is the number of tokens and  $d$  is the hidden dimension. The computational and memory demands of  $\mathbf{Q} \cdot \mathbf{K}^{\top}$  increase quadratically as the number of tokens grows, leading to slow inference speed and high memory usage, making it impractical to run in real-time for long sequences.

To alleviate this issue, [29] proposes the transpose self-attention (see Fig. 2 (b)) to reduce the complexity from quadratic to linear with respect to the number of tokens. Here, the dot-product operation is applied across the channel dimension instead of the spatial dimension. This allows the model to learn feature maps with implicit contextual representation. The attention can be described as:

$$\hat{\mathbf{x}} = \mathbf{V} \cdot \text{Softmax}\left(\frac{\mathbf{Q}^{\top} \cdot \mathbf{K}}{\sqrt{d}}\right). \quad (2)$$

The transpose self-attention has a computational complexity of  $O(n \cdot d^2)$ . While this complexity scales linearly withFigure 2 illustrates four self-attention modules: (a) Self-attention, (b) Transpose Self-attention, (c) Separable Self-attention, and (d) Efficient Additive Attention. The diagram shows the flow of data from input  $x$  through various linear projections and operations like Softmax, element-wise summation, dot product, and broadcasted element-wise multiplication to produce the final output  $\hat{x}$ .

Figure 2. **Comparison with different self-attention modules.** (a) is a typical self-attention used in ViTs [9]. (b) is the transpose self-attention used in EdgeNeXt [29], where the self-attention operation is applied across channel feature dimensions ( $d \times d$ ) instead of the spatial dimension ( $n \times n$ ). (c) is the separable self-attention of MobileViT-v2 [31], it uses element-wise operations to compute the context vector from the interactions of  $\mathbf{Q}$  and  $\mathbf{K}$  matrices. Then, the context vector is multiplied by  $\mathbf{V}$  matrix to produce the final output. (d) Our proposed efficient additive self-attention. Here, the query matrix is multiplied by learnable weights and pooled to produce global queries. Then, the matrix  $\mathbf{K}$  is element-wise multiplied by the broadcasted global queries, resulting the global context representation.

the number of tokens  $n$ , it remains quadratic with respect to the feature dimension  $d$ . Further, the dot-product operation is still utilized between the query and key matrices.

The separable self-attention mechanism (see Fig. 2 (c)) aims to address the bottleneck of the standard self-attention. Here, the interactions between the queries ( $\mathbf{Q}$ ), keys ( $\mathbf{K}$ ), and values ( $\mathbf{V}$ ) are encoded using element-wise operations. First, the query matrix  $\mathbf{Q}$  is projected to produce a vector  $\mathbf{q}$  of dimensions  $n \times 1$ , and then fed into Softmax to generate the context scores, which captures the importance of each query element. Then, the context scores are multiplied by the key matrix  $\mathbf{K}$  and pooled to compute a context vector, which encodes the contextual information. Finally, the context vector is multiplied element-wise with the value matrix  $\mathbf{V}$  to propagate the contextual information and produce the final output  $\hat{x}$ . It can be summarized as:

$$\hat{x} = \mathbf{V} * \sum \mathbf{K} * \text{Softmax}(\mathbf{q}). \quad (3)$$

Here,  $*$  denotes the element-wise multiplication operation.

### 3.2. Efficient Additive Attention

The typical additive attention mechanism in NLP captures the global context by utilizing pairwise interactions between the tokens via element-wise multiplications instead of using dot-product operation. It encodes the relevance scores for the contextual information of the input sequence based on the interactions of the three attention components ( $\mathbf{Q}$ ,  $\mathbf{K}$ ,  $\mathbf{V}$ ). In contrast, we show that key-value interactions can be removed without sacrificing the performance and only

focusing on effectively encoding query-key interactions by incorporating a linear projection layer is sufficient to learn the relationship between the tokens (see Fig. 2 (d)). This approach, named efficient additive attention, has a faster inference speed and produces more robust contextual representations as demonstrated by our performance on image classification, object detection, and segmentation tasks (Sec. 4). Specifically, the input embedding matrix  $\mathbf{x}$  is transformed into query ( $\mathbf{Q}$ ) and key ( $\mathbf{K}$ ) using two matrices  $\mathbf{W}_q$ ,  $\mathbf{W}_k$ , where  $\mathbf{Q}, \mathbf{K} \in \mathbb{R}^{n \times d}$ ,  $\mathbf{W}_q, \mathbf{W}_k \in \mathbb{R}^{d \times d}$ ,  $n$  is the token length and  $d$  is the dimensions of the embedding vector. Next, the query matrix  $\mathbf{Q}$  is multiplied by learnable parameter vector  $\mathbf{w}_a \in \mathbb{R}^d$  to learn the attention weights of the query, producing global attention query vector  $\alpha \in \mathbb{R}^n$  as:

$$\alpha = \mathbf{Q} \cdot \mathbf{w}_a / \sqrt{d} \quad (4)$$

Then, the query matrix is pooled based on the learned attention weights, resulting in a single global query vector  $\mathbf{q} \in \mathbb{R}^d$  as follows:

$$\mathbf{q} = \sum_{i=1}^n \alpha_i * \mathbf{Q}_i. \quad (5)$$

Next, the interactions between the global query vector  $\mathbf{q} \in \mathbb{R}^d$  and key matrix  $\mathbf{K} \in \mathbb{R}^{n \times d}$  are encoded using the element-wise product to form global context ( $\mathbb{R}^{n \times d}$ ). This matrix shares a similarity with the attention matrix in MHSA and captures information from every token and has the flexibility to learn the correlation in the input sequence.The diagram illustrates the architecture of the proposed model. The top row shows a sequence of four stages (Stage 1 to Stage 4). Each stage consists of a Conv. Encoder followed by a SwiftFormer Encoder, with a downsampling layer between stages. The input image is fed into the Patch Embed layer, followed by hierarchical stages at four different scales  $\{\frac{1}{4}, \frac{1}{8}, \frac{1}{16}, \frac{1}{32}\}$ . The bottom row details the internal structure of the Conv. Encoder and the SwiftFormer Encoder. The Conv. Encoder consists of a  $3 \times 3$  DWConv, Norm,  $1 \times 1$  Conv, GeLU, and  $1 \times 1$  Conv, with a skip connection. The SwiftFormer Encoder consists of a Local Representation (DWConv  $3 \times 3$ , Conv  $1 \times 1$ ), Efficient Additive Attention, and Linear layers.

Figure 3. **Top Row:** Overview of our proposed architecture. The input image is fed into the patch embedding layer, followed by hierarchical stages at four different scales  $\{\frac{1}{4}, \frac{1}{8}, \frac{1}{16}, \frac{1}{32}\}$ . Each stage is consistent and compose of Conv. Encoder blocks followed by SwiftFormer Encoder. Between two consecutive stages, we incorporate downsampling layer to reduce the spatial size by a factor of two and increase the feature dimensions. **Bottom Row:** We show the design of the Conv. Encoder (left) and the SwiftFormer Encoder (right). The Conv. Encoder is designed to learn effective local representations and consists of  $3 \times 3$  depth-wise convolutions followed by two point-wise convolutions for channel mixing. The SwiftFormer Encoder aims to learn enriched local-global representations. It begins with local convolutional layers to extract local features, followed by the efficient additive attention module (see Fig. 2 (d)) and linear layers.

However, it is comparatively inexpensive to compute compared to MHSA and has linear complexity with the token length. Inspired by the transformer architecture, we employ a linear transformation layer to query-key interactions to learn the hidden representation of the tokens. The output of the efficient additive attention  $\hat{x}$  can be described as:

$$\hat{x} = \hat{Q} + \mathbf{T}(\mathbf{K} * \mathbf{q}). \quad (6)$$

where  $\hat{Q}$  denotes to the normalized query matrix,  $\mathbf{T}$  denotes to the linear transformation.

### 3.3. SwiftFormer Architecture

Our SwiftFormer is based on the recently introduced EfficientFormer [20]. The main idea of EfficientFormer is to introduce 4D MetaBlocks based on PoolFormer [53] to learn local representations efficiently, while using 3D MetaBlocks based on self-attention to encode global context. However, the performance of EfficientFormer is limited by two design choices. Firstly, it uses ineffective token mixing, and secondly, it only employs 3D MetaBlocks in the last stage due to quadratic complexity of MHSA. This likely leads to inconsistent and insufficient contextual representation. To address these limitations, our SwiftFormer improves the token mixing by using a simple yet effective Conv. Encoder. Further, we introduce efficient additive attention module that can be incorporated in all stages (Sec.

3.2). This leads to more consistent learning of local-global representations. It is worth mentioning that EfficientFormer employs a latency-driven slimming method to obtain optimal configurations for its model variants, which leads to maximizing the speed. In contrast, our SwiftFormer models are built without using any neural architecture search.

Fig. 3 shows an overview of our proposed architecture. The main components are: (i) Effective Conv. Encoder, and (ii) SwiftFormer Encoder. In contrast to other hybrid designs, the proposed architecture is consistent and has Conv. Encoders followed by SwiftFormer Encoder in all stages. Our architecture extracts hierarchical features at four different scales across four stages. At the beginning of the network, the input image of size  $H \times W \times 3$  is fed through Patch Embedding layer, implemented with two  $3 \times 3$  convolutions with a stride of 2, resulting  $\frac{H}{4} \times \frac{W}{4} \times C_1$  feature maps. Then, the output feature maps are fed into the first stage, which begins with Conv. Encoder to extract spatial features, followed by SwiftFormer to learn the local-global information. Between two consecutive stages, there is a downsampling layer to increase the channel dimension and reduce the token length. Next, the resulting feature maps are subsequently fed into the second, third, and fourth stages of the architecture, producing  $\frac{H}{8} \times \frac{W}{8} \times C_2$ ,  $\frac{H}{16} \times \frac{W}{16} \times C_3$ , and  $\frac{H}{32} \times \frac{W}{32} \times C_4$  dimensional feature maps, respectively. Hence, each stage learns local-global features at different scales of the input image, which allows the network to have enriched representation.

**Effective Conv. Encoder:** The baseline EfficientFormer [20] employs  $3 \times 3$  average pooling layers as a local token mixer, similar to PoolFormer [53]. Although PoolFormer layers are known for their fast inference speed, replacing them with depth-wise convolutions does not increase the latency. Further, it improves the performance without increasing the parameters and latency. Specifically, the features maps  $\mathcal{X}_i$  are fed into  $3 \times 3$  depth-wise convolution (DWConv) followed by Batch Normalization (BN). Then, the resulting features are fed into two point-wise convolutions (Conv<sub>1</sub>) alongside GeLU activation. Finally, we incorporate a skip connection to enable information to flow across the network. The Conv. Encoder is defined as:

$$\hat{\mathcal{X}}_i = \text{Conv}_1(\text{Conv}_{1,G}(\text{DWConv}_{BN}(\mathcal{X}_i))) + \mathcal{X}_i. \quad (7)$$

where  $\mathcal{X}_i$  refers to the input features,  $\text{Conv}_{1,G}$  refers to point-wise convolution followed by GeLU,  $\text{DWConv}_{BN}$  refers to depth-wise convolution followed by Batch Normalization, and  $\hat{\mathcal{X}}_i$  refers to the output feature maps.

**SwiftFormer Encoder:** This module is carefully designed to efficiently encode enriched local-global representation in each stage. As shown in Fig. 3, the initial block of the SwiftFormer Encoder is composed of  $3 \times 3$  depth-wise convolution followed by point-wise convolution, which enables the module to learn spatial information and encode local repre-sentation. Then, the resulting feature maps are fed into the efficient additive attention block, which aims to learn contextual information at each scale of the input size. Finally, the output feature maps are fed into a Linear block, which composes of two  $1 \times 1$  point-wise convolution layers, Batch Normalization, and GeLU activation to generate non-linear features. The SwiftFormer Encoder is described as:

$$\begin{aligned}\hat{\mathcal{X}}_i &= \text{Conv}_1(\text{DWConv}_{BN}(\hat{\mathcal{X}}_i)), \\ \hat{\mathcal{X}}_i &= \text{QK}(\hat{\mathcal{X}}_i) + \hat{\mathcal{X}}_i, \\ \hat{\mathcal{X}}_{i+1} &= \text{Conv}_1(\text{Conv}_{BN,1,G}(\hat{\mathcal{X}}_i)) + \hat{\mathcal{X}}_i.\end{aligned}\quad (8)$$

where  $\text{Conv}_{BN,1,G}$  denotes to Batch Normalization, followed by,  $1 \times 1$  Conv layer, followed by GeLU, and QK denotes the efficient additive attention (explained in Sec. 6).

## 4. Experiments

We evaluate our SwiftFormer models across four downstream tasks: classification on ImageNet-1K [8], object detection and instance segmentation on MS-COCO 2017 [21], and semantic segmentation on ADE20K [60].

### 4.1. Implementation Details

**ImageNet-1K [8]:** All of our models are trained from scratch on ImageNet-1K dataset for 300 epochs with AdamW optimizer [26] and cosine learning rate scheduler with an initial learning rate of  $1e^{-3}$ . We use a linear warm-up for 5 epochs. We use an image resolution of  $224 \times 224$  for both training and testing. Following the training recipe of [20], we use the same teacher model for distillation [36]. The experiments are conducted with PyTorch 1.12 [35] using 8 NVIDIA A100 GPUs. The latency is measured using iPhone 14 (iOS 16), and the throughput is measured using A100 40 GB GPU. For latency measurements, we compile the models using CoreML library [6] and perform inference with a batch size of 1. For the throughput on A100, the inference is performed using a batch size of 128.

**MS-COCO 2017 [21]:** We use our ImageNet pre-trained models as the backbones in Mask-RCNN framework for object detection and instance segmentation on MS-COCO 2017 dataset. The dataset contains 118K training and 5K validation images. Following [20], we finetune our models for 12 epochs with an image size of  $1333 \times 800$  and batch size of 32 using AdamW optimizer. We use learning rate of  $2e^{-4}$  and report the performance for detection and instance segmentation in terms of mean average precision (mAP).

**ADE20K [60]:** The dataset comprises 20K training and 2K validation images and contains 150 class categories for scene parsing. Similar to [20] we use our ImageNet pre-trained models to extract image features and semantic FPN [18] as a decoder for segmentation. The model is trained with an image size of  $512 \times 512$  for 40K iterations with a batch size of 32 using AdamW optimizer. We use

poly learning rate scheduling with an initial learning rate of  $2e^{-4}$ . We report the semantic segmentation performance in terms of mean intersection over union (mIoU).

### 4.2. Baseline Comparison

Table 1 illustrates the impact of integrating our proposed contributions into the baseline EfficientFormer-L1 [20] model in terms of ImageNet-1K top-1 accuracy and inference speed. The first row shows the results of the baseline model, which only includes the self-attention based transformer block in the final stage of the network and achieves a top-1 accuracy of 79.2% with a latency of 1.1 ms on an iPhone 14 mobile device. The second row replaces the pool mixers in the baseline model with our proposed Conv. Encoder, resulting in an improvement in performance to 79.9% while maintaining the same latency. In the third row, we replace the self-attention of the transformer block in the baseline by our proposed efficient additive attention module. Although the performance drops by 0.2%, the inference speed improves by 0.1 ms, and the model has linear complexity with the number of tokens. This enables us to integrate the SwiftFormer Encoder that built on the efficient additive attention into all stages and achieve better performance while maintaining the same inference speed as of baseline (first versus last row).

<table border="1">
<thead>
<tr>
<th>Method</th>
<th>Latency (ms)</th>
<th>Top-1 (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>EfficientFormer-L1 (Baseline)</td>
<td>1.1</td>
<td>79.2</td>
</tr>
<tr>
<td>+ Replace Pool Mixer by effective Conv. Encoder</td>
<td>1.1</td>
<td>79.9</td>
</tr>
<tr>
<td>+ Replace Self-attention by efficient additive attention</td>
<td>1.0</td>
<td>79.7</td>
</tr>
<tr>
<td>+ Incorporate SwiftFormer block across all stages</td>
<td><b>1.1</b></td>
<td><b>80.9</b></td>
</tr>
</tbody>
</table>

Table 1. Baseline comparison between our SwiftFormer-L1 and EfficientFormer-L1 [20] on the ImageNet-1K dataset. The latency is measured on iPhone14 Neural Engine.

### 4.3. Image Classification

Table 2 presents a comparison of our proposed SwiftFormer models (XS, S, L1, and L3) with previous state-of-the-art ConvNets, transformer-based, and hybrid models. We show that our models set new state-of-the-art results, and outperform the recently introduced EfficientFormer [20] and MobileViT-v2 [31] in all model variants. This comprehensive evaluation shows the advantage of our proposed models in terms of both accuracy and latency on mobile devices.

**Comparison with ConvNets:** Our SwiftFormer models surpass the widely used lightweight CNNs counterparts significantly in terms of top-1 accuracy, while running faster than the highly optimized MobileNet-v2 and MobileNet-v3 on an iPhone 14 mobile device. Specifically, our SwiftFormer-XS runs 0.1 ms faster than MobileNet-v2  $\times 1.0$  and MobileNet-v3-Large  $\times 0.75$  and achieve better top-1 accuracy with a margin of 3.9% and 2.4% respectively.<table border="1">
<thead>
<tr>
<th>Model</th>
<th>Type</th>
<th>Latency (ms) ↓</th>
<th>Throughput (A100) ↑</th>
<th>Params(M) ↓</th>
<th>GMACs ↓</th>
<th>Neural Search</th>
<th>Top-1(%) ↑</th>
</tr>
</thead>
<tbody>
<tr>
<td>MobileNet-v2×1.0 [39]</td>
<td>ConvNet</td>
<td>0.8</td>
<td>9889</td>
<td>3.5</td>
<td>0.3</td>
<td>✗</td>
<td>71.8</td>
</tr>
<tr>
<td>MobileNet-v3-Large×0.75 [15]</td>
<td>ConvNet</td>
<td>0.8</td>
<td>10934</td>
<td>4.0</td>
<td>0.2</td>
<td>✓</td>
<td>73.3</td>
</tr>
<tr>
<td>EdgeViT-XXS [34]</td>
<td>Hybrid</td>
<td>1.7</td>
<td>5965</td>
<td>4.1</td>
<td>0.6</td>
<td>✗</td>
<td>74.4</td>
</tr>
<tr>
<td>MobileViT-XS [30]</td>
<td>Hybrid</td>
<td>1.5</td>
<td>3707</td>
<td>2.3</td>
<td>0.7</td>
<td>✗</td>
<td>74.8</td>
</tr>
<tr>
<td><b>SwiftFormer-XS</b></td>
<td><b>Hybrid</b></td>
<td><b>0.7</b></td>
<td><b>6034</b></td>
<td><b>3.5</b></td>
<td><b>0.6</b></td>
<td>✗</td>
<td><b>75.7</b></td>
</tr>
<tr>
<td>MobileNet-v2×1.4 [39]</td>
<td>ConvNet</td>
<td>0.9</td>
<td>7447</td>
<td>6.1</td>
<td>0.6</td>
<td>✗</td>
<td>74.7</td>
</tr>
<tr>
<td>MobileNet-v3-Large [15]</td>
<td>ConvNet</td>
<td>0.9</td>
<td>10351</td>
<td>5.4</td>
<td>0.3</td>
<td>✓</td>
<td>75.1</td>
</tr>
<tr>
<td>EfficientNet-b0 [42]</td>
<td>ConvNet</td>
<td>1.3</td>
<td>8537</td>
<td>5.3</td>
<td>0.4</td>
<td>✓</td>
<td>77.1</td>
</tr>
<tr>
<td>DeiT-T [44]</td>
<td>Transformer</td>
<td>3.8</td>
<td>5860</td>
<td>5.7</td>
<td>3.8</td>
<td>✗</td>
<td>72.2</td>
</tr>
<tr>
<td>EdgeViT-XS [34]</td>
<td>Hybrid</td>
<td>2.7</td>
<td>4812</td>
<td>6.7</td>
<td>1.1</td>
<td>✗</td>
<td>77.5</td>
</tr>
<tr>
<td>MobileViT-v2×1.0 [31]</td>
<td>Hybrid</td>
<td>1.7</td>
<td>3201</td>
<td>4.9</td>
<td>1.8</td>
<td>✗</td>
<td>78.1</td>
</tr>
<tr>
<td><b>SwiftFormer-S</b></td>
<td><b>Hybrid</b></td>
<td><b>0.8</b></td>
<td><b>5051</b></td>
<td><b>6.1</b></td>
<td><b>1.0</b></td>
<td>✗</td>
<td><b>78.5</b></td>
</tr>
<tr>
<td>MobileFormer-508M [3]</td>
<td>Hybrid</td>
<td>3.0</td>
<td>4443</td>
<td>14.0</td>
<td>0.5</td>
<td>✗</td>
<td>79.3</td>
</tr>
<tr>
<td>PoolFormer-S12 [53]</td>
<td>Pool</td>
<td>1.2</td>
<td>3227</td>
<td>12.0</td>
<td>1.8</td>
<td>✗</td>
<td>77.2</td>
</tr>
<tr>
<td>EfficientFormer-L1 [20]</td>
<td>Hybrid</td>
<td>1.1</td>
<td>5046</td>
<td>12.3</td>
<td>1.3</td>
<td>✓</td>
<td>79.2</td>
</tr>
<tr>
<td>MobileViT-v2×1.5 [31]</td>
<td>Hybrid</td>
<td>3.4</td>
<td>2356</td>
<td>10.6</td>
<td>4.0</td>
<td>✗</td>
<td>80.4</td>
</tr>
<tr>
<td><b>SwiftFormer-L1</b></td>
<td><b>Hybrid</b></td>
<td><b>1.1</b></td>
<td><b>4469</b></td>
<td><b>12.1</b></td>
<td><b>1.6</b></td>
<td>✗</td>
<td><b>80.9</b></td>
</tr>
<tr>
<td>ResNet-50 [13]</td>
<td>ConvNet</td>
<td>1.9</td>
<td>4835</td>
<td>25.5</td>
<td>4.1</td>
<td>✗</td>
<td>78.5</td>
</tr>
<tr>
<td>PoolFormer-S36 [53]</td>
<td>Pool</td>
<td>2.8</td>
<td>1114</td>
<td>31.0</td>
<td>5.0</td>
<td>✗</td>
<td>81.4</td>
</tr>
<tr>
<td>ConvNeXt-T [25]</td>
<td>ConvNet</td>
<td>2.5</td>
<td>3235</td>
<td>28.6</td>
<td>4.5</td>
<td>✗</td>
<td>82.1</td>
</tr>
<tr>
<td>DeiT-S [44]</td>
<td>Transformer</td>
<td>9.9</td>
<td>2990</td>
<td>22.5</td>
<td>4.5</td>
<td>✗</td>
<td>81.8</td>
</tr>
<tr>
<td>Swin-T [25]</td>
<td>Transformer</td>
<td>NA</td>
<td>2635</td>
<td>28.3</td>
<td>4.5</td>
<td>✗</td>
<td>81.3</td>
</tr>
<tr>
<td>MobileViT-v2×2.0 [31]</td>
<td>Hybrid</td>
<td>5.0</td>
<td>1906</td>
<td>18.5</td>
<td>7.5</td>
<td>✗</td>
<td>81.2</td>
</tr>
<tr>
<td>EfficientFormer-L3 [20]</td>
<td>Hybrid</td>
<td>2.0</td>
<td>2691</td>
<td>31.3</td>
<td>3.9</td>
<td>✓</td>
<td>82.4</td>
</tr>
<tr>
<td><b>SwiftFormer-L3</b></td>
<td><b>Hybrid</b></td>
<td><b>1.9</b></td>
<td><b>2890</b></td>
<td><b>28.5</b></td>
<td><b>4.0</b></td>
<td>✗</td>
<td><b>83.0</b></td>
</tr>
</tbody>
</table>

Table 2. **Comparison of our proposed SwiftFormer with the state-of-the-art counterpart models on ImgeNet-1K.** The latency is measured on iPhone 14 Neural Engine (iOS 16) and the throughput is measured on Nvidia A100 GPU. Our models run faster than MobileNets, Hybrid, and Transformer models, with a better trade-off between accuracy and model complexity. The error for the latency measurement is less than  $\pm 0.1$  ms. Our results are shown in bold for all model variants.

Our SwiftFormer-S runs faster than EfficientNet-b0 [42] by  $1.6\times$  and achieves 1.4% higher top-1 accuracy. Further, our SwiftFormer-L3 achieves 4.5% and 0.9% gain in top-1 accuracy over ResNet-50 and ConvNeXt-T, respectively, while running at the same latency as ResNet-50 and  $1.3\times$  faster than ConvNeXt-T. This demonstrates that our SwiftFormer models, powered by our proposed efficient additive attention, run faster than the lightweight CNN models on mobile devices and achieve superior performance. Recent device-level optimizations for CNN-based models, such as dedicated hardware implementations for convolutions with batch normalization and non-linearity, likely contribute to the high throughput of fully CNN-based models on A100.

**Comparison with transformer models:** Although transformer models usually outperform CNN-based models in terms of accuracy, they tend to suffer from high latency when running on resource-constrained mobile devices. For instance, DeiT-S, which has a similar model size to ResNet-50 and achieves higher top-1 accuracy by 3.3%, but ResNet-50 runs approximately  $5.2\times$  faster on an iPhone 14 mobile device. In contrast, our SwiftFormer-L3 model achieves 1.2% higher accuracy than DeiT-S, while running at the same speed as ResNet-50. Further our SwiftFormer-S model runs approximately  $4.7\times$  faster than DeiT-T on an iPhone 14 mobile device and has 6.3% better accuracy.

**Comparison with hybrid models:** Although most exist-

ing hybrid approaches achieve higher accuracy compared to their lightweight CNN counterparts, they still underperform the fully CNN-based models in terms of latency due to the quadratic complexity of multi-head self-attention. For example, EdgeViT-XXS runs at approximately  $2\times$  slower compared to MobileNet-v3-Large×0.75. On the other hand, our SwiftFormer-XS has better latency as compared to lightweight CNNs and approximately is  $2\times$  faster than EdgeViT-XXS and MobileViT-XS, with an overall 1.3% and 0.9% higher top-1 accuracy respectively. Further, our SwiftFormer-L1 model is  $3\times$  faster than the state-of-the-art MobileViT-v2×1.5 with 0.5% better top-1 accuracy. Our SwiftFormer-L3 model achieves 83.0% top-1 accuracy and runs at 1.9 ms, which is  $2.6\times$  faster than MobileViT-v2×2.0 with an absolute 1.8% accuracy gain.

#### 4.4. Object Detection and Instance Segmentation

Table 3 compares the object detection and instance segmentation results of Mask-RCNN with different lightweight backbones. Our SwiftFormer-L1 backbone achieves 41.2 AP box, surpassing the lightweight ResNet18 and PoolFormer-S12 backbones by 7.2 and 3.9 points respectively. Further, it performs better than the previous state-of-the-art EfficientFormer-L1 backbone by 3.3 AP box. For instance segmentation, our method achieves 38.1 AP mask score which is 2.7 points better than the previous state-<table border="1">
<thead>
<tr>
<th rowspan="2">Backbone</th>
<th colspan="6">Detection &amp; Instance Segmentation</th>
<th rowspan="2">Semantic mIoU(%)</th>
</tr>
<tr>
<th><math>AP^{box}</math></th>
<th><math>AP_{50}^{box}</math></th>
<th><math>AP_{75}^{box}</math></th>
<th><math>AP^{mask}</math></th>
<th><math>AP_{50}^{mask}</math></th>
<th><math>AP_{75}^{mask}</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>ResNet18 [13]</td>
<td>34.0</td>
<td>54.0</td>
<td>36.7</td>
<td>31.2</td>
<td>51.0</td>
<td>32.7</td>
<td>32.9</td>
</tr>
<tr>
<td>PoolFormer-S12 [53]</td>
<td>37.3</td>
<td>59.0</td>
<td>40.1</td>
<td>34.6</td>
<td>55.8</td>
<td>36.9</td>
<td>37.2</td>
</tr>
<tr>
<td>EfficientFormer-L1 [20]</td>
<td>37.9</td>
<td>60.3</td>
<td>41.0</td>
<td>35.4</td>
<td>57.3</td>
<td>37.3</td>
<td>38.9</td>
</tr>
<tr>
<td><b>SwiftFormer-L1</b></td>
<td><b>41.2</b></td>
<td><b>63.2</b></td>
<td><b>44.8</b></td>
<td><b>38.1</b></td>
<td><b>60.2</b></td>
<td><b>40.7</b></td>
<td><b>41.4</b></td>
</tr>
<tr>
<td>ResNet50 [13]</td>
<td>38.0</td>
<td>58.6</td>
<td>41.4</td>
<td>34.4</td>
<td>55.1</td>
<td>36.7</td>
<td>36.7</td>
</tr>
<tr>
<td>PoolFormer-S24 [53]</td>
<td>40.1</td>
<td>62.2</td>
<td>43.4</td>
<td>37.0</td>
<td>59.1</td>
<td>39.6</td>
<td>40.3</td>
</tr>
<tr>
<td>EfficientFormer-L3 [20]</td>
<td>41.4</td>
<td>63.9</td>
<td>44.7</td>
<td>38.1</td>
<td>61.0</td>
<td>40.4</td>
<td>43.5</td>
</tr>
<tr>
<td><b>SwiftFormer-L3</b></td>
<td><b>42.7</b></td>
<td><b>64.4</b></td>
<td><b>46.7</b></td>
<td><b>39.1</b></td>
<td><b>61.7</b></td>
<td><b>41.8</b></td>
<td><b>43.9</b></td>
</tr>
</tbody>
</table>

Table 3. **Results using SwiftFormer as a backbone on dense prediction tasks:** Object detection and instance segmentation on COCO, whereas semantic segmentation on ADE20K. Our approach outperforms the recent EfficientFormer on all three tasks.

Figure 4. **Qualitative results on COCO.** The qualitative examples for object detection and instance segmentation on the COCO 2017 validation set. The visualizations show that our SwiftFormer-L1 based model can accurately detect and segment the instances in images.

Figure 5. **Qualitative results on ADE20K.** The qualitative examples for semantic segmentation on the ADE20K validation set. **Top:** Ground truth masks. **Bottom:** The semantic segmentation results. Our model can accurately segment various indoor and outdoor scenes.

of-the-art. Similar trend is observed for SwiftFormer-L3 backbone, which surpasses the previous state-of-the-art EfficientFormer-L3 backbone by 1.3 points and 1.0 points in AP box and mask respectively. The improvement in the downstream detection and instance segmentation tasks illustrates the effectiveness of our SwiftFormer backbone models for the dense prediction tasks.

#### 4.5. Semantic Segmentation

Table 3 shows the semantic segmentation results of SwiftFormer backbone-based models as compared to previously proposed backbones. We achieve 41.4% mean intersection over union score using SwiftFormer-L1, surpassing ResNet18 by 8.5%, PoolFormer-S12 by 4.2%, and the state-of-the-art EfficientFormer-L1 by 2.5%. Similarly,

our SwiftFormer-L3 backbone-based segmentation model achieves 43.9 mIoU, surpassing all previous methods.

## 5. Conclusion

Transformers have gained popularity in vision applications due to their effective use of self-attention computation. However, their use in mobile vision applications is challenging due to the quadratic nature of the self-attention, which is computationally expensive on mobile devices. To address this issue, many hybrid approaches and efficient variants of self-attention have been proposed. In this work, we propose a novel efficient additive attention that replaces the expensive matrix multiplication operations with element-wise multiplications, and eliminates explicit keys-Figure 6. **Qualitative comparison with baseline.** Qualitative comparison between the recent EfficientFormer-L1 (**top**) and our SwiftFormer-L1 (**bottom**) on example images from COCO validation set for dense prediction tasks: detection and instance segmentation. Here, EfficientFormer-L1 misclassifies a bird as a sheep in the first example and as a bear in the second example, while misclassifying kites as birds in the third example. Our SwiftFormer-L1 accurately detects and segments objects in these examples.

values interaction. Our proposed attention is linear with respect to the input tokens and can be used in all stages of the network. We show state-of-the-art results on image classification, object detection, and segmentation benchmarks.

## References

- [1] Adrian Bulat and Georgios Tzimiropoulos. Bit-mixer: Mixed-precision networks with runtime bit-width selection. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2021. 2
- [2] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In *European Conference on Computer Vision*, 2020. 1
- [3] Yinpeng Chen, Xiyang Dai, Dongdong Chen, Mengchen Liu, Xiaoyi Dong, Lu Yuan, and Zicheng Liu. Mobileformer: Bridging mobilenet and transformer. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2022. 2, 7
- [4] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2022. 1
- [5] Xiangxiang Chu, Zhi Tian, Yuqing Wang, Bo Zhang, Haibing Ren, Xiaolin Wei, Huaxia Xia, and Chunhua Shen. Twins: Revisiting the design of spatial attention in vision transformers. In *Advances in Neural Information Processing Systems*, 2021. 2
- [6] CoreMLTools. Use coremltools to convert models from third-party libraries to core ml., 2021. 6
- [7] Ekin Dogus Cubuk, Barret Zoph, Jon Shlens, and Quoc Le. Randaugment: Practical automated data augmentation with a reduced search space. In *Advances in Neural Information Processing Systems*, 2020. 12
- [8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image

- database. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2009. 2, 6
- [9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*, 2020. 1, 2, 4
- [10] Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, and Christoph Feichtenhofer. Multiscale vision transformers. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2021. 1, 2
- [11] Mohsen Fayyaz, Soroush Abbasi Kouhpayegani, Farnoush Rezaei Jafari, Eric Sommerlade, Hamid Reza Vaezi Joze, Hamed Pirsivash, and Juergen Gall. Adaptive token sampling for efficient vision transformers. 2022. 2
- [12] TSong Han, Huizi Mao, and William J. Dally. Deep compression: Compressing deep neural network with pruning, trained quantization and huffman coding. In *International Conference on Learning Representations*, 2016. 2
- [13] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2016. 7, 8
- [14] Byeongho Heo, Sangdoo Yun, Dongyoon Han, Sanghyuk Chun, Junsuk Choe, and Seong Joon Oh. Rethinking spatial dimensions of vision transformers. In *ICCV*, 2021. 2
- [15] Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, et al. Searching for mobilenet-v3. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2019. 1, 2, 7
- [16] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. *CoRR*, abs/1704.04861, 2017. 2
- [17] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. Quantization and training of neural networks for efficient integer-arithmetic-only inference. In *CVPR*, 2018. 2
- [18] A. Kirillov, R. Girshick, K. He, and P. Dollar. Panoptic feature pyramid networks. In *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019. 6
- [19] Nikita Kitaev, Lukasz Kaiser, and Anselm Levkaya. Reformer: The efficient transformer. In *International Conference on Learning Representations*, 2020. 2
- [20] Yanyu Li, Geng Yuan, Yang Wen, Ju Hu, Georgios Evangelidis, Sergey Tulyakov, Yanzhi Wang, and Jian Ren. Efficientformer: Vision transformers at mobilenet speed. 2022. 1, 2, 3, 5, 6, 7, 8, 12
- [21] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In *European Conference on Computer Vision*, 2014. 6, 12- [22] Yahui Liu, Enver Sangineto, Wei Bi, Nicu Sebe, Bruno Lepri, and Marco De Nadai. Efficient training of visual transformers with small datasets. In *Advances in Neural Information Processing Systems*, 2021. 2
- [23] Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, et al. Swin transformer v2: Scaling up capacity and resolution. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2022. 1
- [24] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2021. 1, 2
- [25] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2022. 7
- [26] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In *International Conference on Learning Representations*, 2019. 6
- [27] Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, and Jian Sun. Shufflenet v2: Practical guidelines for efficient cnn architecture design. In *European Conference on Computer Vision*, 2018. 2
- [28] Muhammad Maaz, Hanoona Rasheed, Salman Khan, Fahad Shahbaz Khan, Rao Muhammad Anwer, and Ming-Hsuan Yang. Class-agnostic object detection with multi-modal transformer. In *European Conference on Computer Vision*, 2022. 1
- [29] Muhammad Maaz, Abdelrahman Shaker, Hisham Cholakkal, Salman Khan, Syed Waqas Zamir, Rao Muhammad Anwer, and Fahad Shahbaz Khan. Edgenext: Efficiently amalgamated cnn-transformer architecture for mobile vision applications. In *CADL2022*, 2022. 1, 2, 3, 4
- [30] Sachin Mehta and Mohammad Rastegari. Mobilevit: light-weight, general-purpose, and mobile-friendly vision transformer. In *International Conference on Learning Representations*, 2022. 2, 7
- [31] Sachin Mehta and Mohammad Rastegari. Separable self-attention for mobile vision transformers. *Transactions on Machine Learning Research*, 2023. 1, 2, 3, 4, 6, 7
- [32] Sachin Mehta, Mohammad Rastegari, Linda Shapiro, and Hannaneh Hajishirzi. Espnetv2: A light-weight, power efficient, and general purpose convolutional neural network. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2019. 2
- [33] Depu Meng, Xiaokang Chen, Zejia Fan, Gang Zeng, Houqiang Li, Yuhui Yuan, Lei Sun, and Jingdong Wang. Conditional detr for fast training convergence. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2021. 1
- [34] Junting Pan, Adrian Bulat, Fuwen Tan, Xiatian Zhu, Lukasz Dudziak, Hongsheng Li, Georgios Tzimiropoulos, and Brais Martinez. Edgevits: Competing light-weight cnns on mobile devices with vision transformers. In *European Conference on Computer Vision*, 2022. 2, 7
- [35] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raion, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems*, 2019. 6
- [36] Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr Dollár. Designing network design spaces. In *CVPR*, 2020. 6, 12
- [37] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit: Efficient vision transformers with dynamic token sparsification. In *Advances in Neural Information Processing Systems*, 2021. 2
- [38] Michael Ryoo, AJ Piergiovanni, Anurag Arnab, Mostafa Dehghani, and Anelia Angelova. Tokenlearner: Adaptive space-time tokenization for videos. In *Advances in Neural Information Processing Systems*, 2021. 2
- [39] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenet-v2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2018. 1, 2, 3, 7
- [40] Abdelrahman Shaker, Muhammad Maaz, Hanoona Rasheed, Salman Khan, Ming-Hsuan Yang, and Fahad Shahbaz Khan. Unetr++: Delving into efficient and accurate 3d medical image segmentation. *arXiv:2212.04497*, 2022. 1, 2
- [41] Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, and Quoc V Le. Mnasnet: Platform-aware neural architecture search for mobile. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2019. 2
- [42] Mingxing Tan and Quoc Le. EfficientNet: Rethinking model scaling for convolutional neural networks. In *Proceedings of the 36th International Conference on Machine Learning*, 2019. 7
- [43] Ilya Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, et al. Mlp-mixer: An all-mlp architecture for vision. In *Advances in Neural Information Processing Systems*, 2021. 2
- [44] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In *International Conference on Machine Learning*, 2021. 1, 2, 7, 12
- [45] Zhengzhong Tu, Hossein Talebi, Han Zhang, Feng Yang, Peyman Milanfar, Alan Bovik, and Yinxiao Li. Maxvit: Multi-axis vision transformer. In *ECCV*, 2022. 2
- [46] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. *arXiv preprint arXiv:2006.04768*, 2020. 2
- [47] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In *CVPR*, 2021. 2- [48] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 568–578, 2021. [2](#)
- [49] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Ptv2: Improved baselines with pyramid vision transformer. *Computational Visual Media*, 2022. [2](#)
- [50] Wei Wen, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li. Learning structured sparsity in deep neural networks. In *Advances in Neural Information Processing Systems*, 2016. [2](#)
- [51] Chuhan Wu, Fangzhao Wu, Tao Qi, and Yongfeng Huang. Fastformer: Additive attention can be all you need. *arXiv preprint arXiv:2108.09084*, 2021. [3](#)
- [52] Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen Zhou, Xinchao Wang, Jiashi Feng, and Shuicheng Yan. Metaformer is actually what you need for vision. In *CVPR*, 2022. [2](#)
- [53] Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen Zhou, Xinchao Wang, Jiashi Feng, and Shuicheng Yan. Metaformer is actually what you need for vision. In *CVPR*, 2022. [5](#), [7](#), [8](#)
- [54] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2019. [12](#)
- [55] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In *International Conference on Learning Representations*, 2018. [12](#)
- [56] Haokui Zhang, Wenzhe Hu, and Xiaoyu Wang. Edgeformer: Improving light-weight convnets by learning from vision transformers. *arXiv preprint arXiv:2203.03952*, 2022. [2](#)
- [57] Haokui Zhang, Wenzhe Hu, and Xiaoyu Wang. Parc-net: Position aware circular convolution with merits from convnets and transformer. In *European Conference on Computer Vision*, 2022. [2](#)
- [58] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel Ni, and Harry Shum. Dino: Detr with improved denoising anchor boxes for end-to-end object detection. In *International Conference on Learning Representations*, 2022. [1](#)
- [59] Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, and Jian Sun. Shufflenet: An extremely efficient convolutional neural network for mobile devices. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2018. [2](#)
- [60] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2017. [6](#)
- [61] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. In *International Conference on Learning Representations*, 2021. [1](#)## Supplemental Material

In this section, we provide additional details regarding:

- • Architecture Details of SwiftFormer (Appendix A)
- • Implementation Details (Appendix B)
- • Ablations (Appendix C)
- • Error Analysis on COCO Dataset (Appendix D)
- • Qualitative Results (Appendix E)
- • Discussion (Appendix F)

### A. Architecture Details of SwiftFormer

The detailed network architectures for SwiftFormer-XS, SwiftFormer-S, SwiftFormer-L1, and SwiftFormer-L3 are provided in Table 4. We report the resolution, the number of channels ( $C$ ) and the number of repeated blocks ( $N$ ) of each stage for all the model variants. For all variants, we use an expansion ratio of 4 in the Conv. Encoder. Since our architectures are not built using any neural architecture search, the number of channels and blocks for our models are selected to have a similar model size and GMACs with previous state-of-the-art methods in each variant.

### B. Additional Implementation Details

We train and report the accuracy of our SwiftFormer models at  $224 \times 224$  resolution for a fair comparison with the baseline and previous methods. We use a batch size of 2048 during training. The experiments for the SwiftFormer models were conducted on eight A100 GPUs, with an average training time of 36 hours for the classification. To enhance the robustness of the models, we apply several data augmentations during training. Specifically, we employed color jitter with a ratio of 0.4, RandAugment [7] with a magnitude of 9 and standard deviation of 0.5, gradient clipping of 0.01, Mixup [55] and Cutmix [54] with percentages of 1 and 0.8, respectively, label smoothing with a value of 0.1, and random erase with a probability of 0.25. Similar to DeiT [44] and EfficientFormer [20], we employ RegNetY-16GF [36] with 82.9% top-1 accuracy as our teacher model for hard distillation.

### C. Additional Ablations

We investigate the effect of QKV interactions and observe that eliminating key-value interactions and replacing them with a simple linear transformation results in 10% reduction in latency. In addition to latency reduction, the top-1 accuracy is improved by 0.4%. Overall, our results demonstrate the effectiveness of the proposed SwiftFormer encoder and highlight the potential benefits of simplifying the QKV interactions in the efficient additive attention mechanism.

Figure 7. **Error analysis for the performance on COCO.** The baseline EfficientFormer-L1 (left) and our SwiftFormer-L1 (right) across all categories, on the all-objects (top) and large-sized objects (bottom). The plot in each image indicates a series of precision-recall curves using different evaluation configurations [21], with the legend indicating the area under each curve in brackets. Our SwiftFormer-L1 provides consistent improvements over the baseline EfficientFormer-L1.

### D. Error Analysis on COCO Dataset

Fig. 7 shows the error analysis plot of the baseline EfficientFormer-L1 (left) and our SwiftFormer-L1 (right) for all-objects and the large-sized objects. We show the area under each curve in brackets in the legend. It is noted that our results are better compared to the baseline, especially for large-sized objects. For instance, the overall AP of large-sized objects of EfficientFormer-L1 at IoU=0.75 is 0.558 and perfect localization increases the AP to 0.793. Excluding the background false positives likely increase the performance to 0.928 AP. In the case of SwiftFormer, the overall AP at IoU=0.75 is 0.610 and perfect localization increases the AP to 0.837. Further, excluding the background false positives likely increase the performance to 0.946 AP. When analyzing the performance on small and medium-sized objects, we still have an improvement achieved by our SwiftFormer model.

### E. Qualitative Results

Fig. 8 and 9 shows additional qualitative results of our SwiftFormer model for instance segmentation/detection and semantic segmentation respectively. Our model accurately localizes and segments the objects in diverse scenes. It also provides high-quality segmentation masks on ADE20K validation dataset.<table border="1">
<thead>
<tr>
<th rowspan="2">Stage</th>
<th rowspan="2">Output Resolution</th>
<th rowspan="2">Type</th>
<th rowspan="2">Config</th>
<th colspan="4">SwiftFormer</th>
</tr>
<tr>
<th>XS</th>
<th>S</th>
<th>L1</th>
<th>L3</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">stem</td>
<td rowspan="2"><math>\frac{H}{2} \times \frac{W}{2}</math></td>
<td rowspan="2">Patch Embed.</td>
<td>Patch Size</td>
<td colspan="4"><math>k = 3 \times 3, s = 2</math></td>
</tr>
<tr>
<td>Embed. Dim.</td>
<td>24</td>
<td>24</td>
<td>24</td>
<td>32</td>
</tr>
<tr>
<td rowspan="2"><math>\frac{H}{4} \times \frac{W}{4}</math></td>
<td rowspan="2">Patch Embed.</td>
<td>Patch Size,</td>
<td colspan="4"><math>k = 3 \times 3, s = 2</math></td>
</tr>
<tr>
<td>Embed. Dim.</td>
<td>48</td>
<td>48</td>
<td>48</td>
<td>64</td>
</tr>
<tr>
<td rowspan="2">1</td>
<td rowspan="2"><math>\frac{H}{4} \times \frac{W}{4}</math></td>
<td rowspan="2">Hybrid</td>
<td>Conv. Encoder <math>[C, N]</math></td>
<td>48, 2</td>
<td>48, 2</td>
<td>48, 3</td>
<td>64, 3</td>
</tr>
<tr>
<td>SwiftFormer Encoder <math>[C, N]</math></td>
<td>48, 1</td>
<td>48, 1</td>
<td>48, 1</td>
<td>64, 1</td>
</tr>
<tr>
<td rowspan="4">2</td>
<td rowspan="2"><math>\frac{H}{8} \times \frac{W}{8}</math></td>
<td rowspan="2">Down-sampling</td>
<td>Patch Size</td>
<td colspan="4"><math>k = 3 \times 3, s = 2</math></td>
</tr>
<tr>
<td>Embed. Dim.</td>
<td>56</td>
<td>64</td>
<td>96</td>
<td>128</td>
</tr>
<tr>
<td rowspan="2"><math>\frac{H}{8} \times \frac{W}{8}</math></td>
<td rowspan="2">Hybrid</td>
<td>Conv. Encoder <math>[C, N]</math></td>
<td>56, 2</td>
<td>64, 2</td>
<td>96, 2</td>
<td>128, 3</td>
</tr>
<tr>
<td>SwiftFormer Encoder <math>[C, N]</math></td>
<td>56, 1</td>
<td>64, 1</td>
<td>96, 1</td>
<td>128, 1</td>
</tr>
<tr>
<td rowspan="4">3</td>
<td rowspan="2"><math>\frac{H}{16} \times \frac{W}{16}</math></td>
<td rowspan="2">Down-sampling</td>
<td>Patch Size</td>
<td colspan="4"><math>k = 3 \times 3, s = 2</math></td>
</tr>
<tr>
<td>Embed. Dim.</td>
<td>112</td>
<td>168</td>
<td>192</td>
<td>320</td>
</tr>
<tr>
<td rowspan="2"><math>\frac{H}{16} \times \frac{W}{16}</math></td>
<td rowspan="2">Hybrid</td>
<td>Conv. Encoder <math>[C, N]</math></td>
<td>112, 5</td>
<td>168, 8</td>
<td>192, 9</td>
<td>320, 11</td>
</tr>
<tr>
<td>SwiftFormer Encoder <math>[C, N]</math></td>
<td>112, 1</td>
<td>168, 1</td>
<td>192, 1</td>
<td>320, 1</td>
</tr>
<tr>
<td rowspan="4">4</td>
<td rowspan="2"><math>\frac{H}{32} \times \frac{W}{32}</math></td>
<td rowspan="2">Down-sampling</td>
<td>Patch Size</td>
<td colspan="4"><math>k = 3 \times 3, s = 2</math></td>
</tr>
<tr>
<td>Embed. Dim.</td>
<td>220</td>
<td>224</td>
<td>384</td>
<td>512</td>
</tr>
<tr>
<td rowspan="2"><math>\frac{H}{32} \times \frac{W}{32}</math></td>
<td rowspan="2">Hybrid</td>
<td>Conv. Encoder <math>[C, N]</math></td>
<td>220, 3</td>
<td>224, 5</td>
<td>384, 4</td>
<td>512, 5</td>
</tr>
<tr>
<td>SwiftFormer Encoder <math>[C, N]</math></td>
<td>220, 1</td>
<td>224, 1</td>
<td>384, 1</td>
<td>512, 1</td>
</tr>
<tr>
<td rowspan="2">GMACs</td>
<td rowspan="2"></td>
<td rowspan="2"></td>
<td rowspan="2"></td>
<td>0.6G</td>
<td>1.0G</td>
<td>1.6G</td>
<td>4.0G</td>
</tr>
<tr>
<td>Parameters</td>
<td>3.5M</td>
<td>6.1M</td>
<td>12.1M</td>
<td>28.5M</td>
</tr>
</tbody>
</table>

Table 4. **SwiftFormer Architectures.** Description of the configurations of the model variants with respect to the output resolution, the output channels  $C$ , the number of blocks  $N$ , and the model’s GMACs and parameters. Between two consecutive stages, we incorporate a downsampling layer to increase the number of channels and reduce the resolution by two.

Figure 8. **Additional qualitative results on COCO.** Detection and instance segmentation results of our model.

## F. Discussion

The positional encoding and attention biases in vision transformers both play a crucial role in providing spatial information about the input sequence, particularly in dense prediction tasks. However, the attention bias is sensitive to input resolution and can make the model fragile when incorporated into these tasks. Meanwhile, typical positional encoding can slow down the inference of the model on resource-constrained devices. To overcome these chal-

lenges, we introduce an efficient additive attention mechanism that does not include positional encoding or attention biases, allowing for fast inference speed. In addition, our SwiftFormer models have shown promising results in downstream tasks. To the best of our knowledge, SwiftFormer is currently the most efficient hybrid architecture for real-time mobile vision applications.Figure 9. **Additional qualitative results on ADE20K.** Top row shows the ground truth masks and the bottom row shows the predictions of our model.

