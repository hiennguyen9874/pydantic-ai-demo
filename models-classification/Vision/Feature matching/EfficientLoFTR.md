Title: Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed

URL Source: https://arxiv.org/html/2403.04765

Published Time: Wed, 13 Mar 2024 00:14:43 GMT

Markdown Content:
Yifan Wang*{}^{*}start_FLOATSUPERSCRIPT * end_FLOATSUPERSCRIPT Xingyi He*{}^{*}start_FLOATSUPERSCRIPT * end_FLOATSUPERSCRIPT Sida Peng Dongli Tan Xiaowei Zhou††{}^{\dagger}start_FLOATSUPERSCRIPT † end_FLOATSUPERSCRIPT

Zhejiang University

###### Abstract

We present a novel method for efficiently producing semi-dense matches across images. Previous detector-free matcher LoFTR has shown remarkable matching capability in handling large-viewpoint change and texture-poor scenarios but suffers from low efficiency. We revisit its design choices and derive multiple improvements for both efficiency and accuracy. One key observation is that performing the transformer over the entire feature map is redundant due to shared local information, therefore we propose an aggregated attention mechanism with adaptive token selection for efficiency. Furthermore, we find spatial variance exists in LoFTR’s fine correlation module, which is adverse to matching accuracy. A novel two-stage correlation layer is proposed to achieve accurate subpixel correspondences for accuracy improvement. Our efficiency optimized model is ∼2.5×\sim 2.5\times∼ 2.5 × faster than LoFTR which can even surpass state-of-the-art efficient sparse matching pipeline SuperPoint + LightGlue. Moreover, extensive experiments show that our method can achieve higher accuracy compared with competitive semi-dense matchers, with considerable efficiency benefits. This opens up exciting prospects for large-scale or latency-sensitive applications such as image retrieval and 3D reconstruction. Project page: [https://zju3dv.github.io/efficientloftr/](https://zju3dv.github.io/efficientloftr/).

††*{}^{*}start_FLOATSUPERSCRIPT * end_FLOATSUPERSCRIPT Equal contribution. The authors from Zhejiang University are affiliated with the State Key Lab of CAD&CG. ††{}^{\dagger}start_FLOATSUPERSCRIPT † end_FLOATSUPERSCRIPT Corresponding author: Xiaowei Zhou.
1 Introduction
--------------

Image matching is the cornerstone of many 3D computer vision tasks, which aim to find a set of highly accurate correspondences given an image pair. The established matches between images have broad usages such as reconstructing the 3D world by structure from motion(SfM)[[1](https://arxiv.org/html/2403.04765v2#bib.bib1), [47](https://arxiv.org/html/2403.04765v2#bib.bib47), [29](https://arxiv.org/html/2403.04765v2#bib.bib29), [21](https://arxiv.org/html/2403.04765v2#bib.bib21)] or SLAM system[[35](https://arxiv.org/html/2403.04765v2#bib.bib35), [34](https://arxiv.org/html/2403.04765v2#bib.bib34)], and visual localization[[42](https://arxiv.org/html/2403.04765v2#bib.bib42), [44](https://arxiv.org/html/2403.04765v2#bib.bib44)], etc. Previous methods typically follow a two-stage pipeline: they first detect[[41](https://arxiv.org/html/2403.04765v2#bib.bib41)] and describe[[53](https://arxiv.org/html/2403.04765v2#bib.bib53)] a set of keypoints on each image, and then establish keypoint correspondences by handcrafted[[31](https://arxiv.org/html/2403.04765v2#bib.bib31)] or learning-based matchers[[43](https://arxiv.org/html/2403.04765v2#bib.bib43), [30](https://arxiv.org/html/2403.04765v2#bib.bib30)]. These detector-based methods are efficient but suffer from robustly detecting repeatable keypoints across challenging pairs, such as extreme viewpoint changes and texture-poor regions.

![Image 1: Refer to caption](https://arxiv.org/html/2403.04765v2/x1.png)

Figure 1: Matching Accuracy and Efficiency Comparisons. Our method achieves competitive accuracy compared with semi-dense matchers(\scalerel*![Image 2: Refer to caption](https://arxiv.org/html/2403.04765v2/x6.png)B) at a significantly higher speed. Compared with dense matcher ROMA(\scalerel*![Image 3: Refer to caption](https://arxiv.org/html/2403.04765v2/x7.png)B), our method is ∼7.5×\sim 7.5\times∼ 7.5 × faster. Moreover, our efficiency optimized model(\scalerel*![Image 4: Refer to caption](https://arxiv.org/html/2403.04765v2/x4.png)B) can surpass the robust sparse matching pipeline(\scalerel*![Image 5: Refer to caption](https://arxiv.org/html/2403.04765v2/x8.png)B) SuperPoint(SP) + LightGlue(LG) on efficiency with considerably better accuracy. 

Recently, LoFTR[[50](https://arxiv.org/html/2403.04765v2#bib.bib50)] introduces a detector-free matching paradigm with transformer to directly establish semi-dense correspondences between two images without detecting keypoints. With the help of the transformer mechanism to capture the global image context and the detector-free design, LoFTR shows a strong capability of matching challenging pairs, especially in texture-poor scenarios. To reduce the computation burden, LoFTR adopts a coarse-to-fine pipeline by first performing dense matching on downsampled coarse features maps, where transformer is applied. Then, the feature locations of coarse matches on one image are fixed, while their subpixel correspondences are searched on the other image by cropping feature patches based on coarse match, performing the feature correlation, and calculating expectation over the correlation patch.

Despite its impressive matching performance, LoFTR suffers from limited efficiency due to the large token size of performing transformer on the entire coarse feature map, which significantly barricades practical large-scale usages such as image retrieval[[19](https://arxiv.org/html/2403.04765v2#bib.bib19)] and SfM[[47](https://arxiv.org/html/2403.04765v2#bib.bib47)]. A large bunch of LoFTR’s follow-up works[[59](https://arxiv.org/html/2403.04765v2#bib.bib59), [7](https://arxiv.org/html/2403.04765v2#bib.bib7), [52](https://arxiv.org/html/2403.04765v2#bib.bib52), [18](https://arxiv.org/html/2403.04765v2#bib.bib18), [36](https://arxiv.org/html/2403.04765v2#bib.bib36)] have attempted to improve its matching accuracy. However, there are rare methods that focus on matching efficiency of detector-free matching. QuadTree Attention[[52](https://arxiv.org/html/2403.04765v2#bib.bib52)] incorporates multi-scale transformation with a gradually narrowed attention span to avoid performing attention on large feature maps. This strategy can reduce the computation cost, but it also divides a single coarse attention process into multiple steps, leading to increased latency.

In this paper, we revisit the design decisions of the detector-free matcher LoFTR, and propose a new matching algorithm that squeezes out redundant computations for significantly better efficiency while further improving the accuracy. As shown in Fig.[1](https://arxiv.org/html/2403.04765v2#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed"), our approach achieves the best inference speed compared with recent image matching methods while being competitive in terms of accuracy. Our key innovations lie in introducing a token aggregation mechanism for efficient feature transformation and a two-stage correlation layer for correspondence refinement. Specifically, we find that densely performing global attention over the entire coarse feature map as in LoFTR is unnecessary, as the attention information is similar and shared in the local region. Therefore, we devise an aggregated attention mechanism to perform feature transformation on adaptively selected tokens, which is significantly compact and effectively reduces the cost of local feature transformation.

In addition, we observe that there can be spatial variance in the matching refinement phase of LoFTR, which is caused by the expectation over the entire correlation patch when noisy feature correlation exists. To solve this issue, our approach designs a two-stage correlation layer that first locates pixel-level matches with the accurate mutual-nearest-neighbor matching on fine feature patches, and then further refines matches for subpixel-level by performing the correlation and expectation locally within tiny patches.

Extensive experiments are conducted on multiple tasks, including homography estimation, relative pose recovery, as well as visual localization, to show the efficacy of our method. Our pipeline pushes detector-free matching to unprecedented efficiency, which is ∼2.5 similar-to absent 2.5\sim 2.5∼ 2.5 times faster than LoFTR and can even surpass the current state-of-the-art efficient sparse matcher LightGlue[[30](https://arxiv.org/html/2403.04765v2#bib.bib30)]. Moreover, our framework can achieve comparable or even better matching accuracy compared with competitive detector-free baselines[[7](https://arxiv.org/html/2403.04765v2#bib.bib7), [14](https://arxiv.org/html/2403.04765v2#bib.bib14), [15](https://arxiv.org/html/2403.04765v2#bib.bib15)] with considerably higher efficiency.

In summary, this paper has the following contributions:

*   •A new detector-free matching pipeline with multiple improvements based on the comprehensive revisiting of LoFTR, which is significantly more efficient and with better accuracy. 
*   •A novel aggregated attention network for efficient local feature transformation. 
*   •A novel two-stage correlation refinement layer for accurate and subpixel-level refined correspondences. 

2 Related Work
--------------

#### Detector-Based Image Matching.

Classical image matching methods[[31](https://arxiv.org/html/2403.04765v2#bib.bib31), [41](https://arxiv.org/html/2403.04765v2#bib.bib41), [4](https://arxiv.org/html/2403.04765v2#bib.bib4)] adopt handcrafted critics for detecting keypoints, describing and then matching them. Recent methods draw benefits from deep neural networks for both detection[[41](https://arxiv.org/html/2403.04765v2#bib.bib41), [46](https://arxiv.org/html/2403.04765v2#bib.bib46), [24](https://arxiv.org/html/2403.04765v2#bib.bib24)] and description[[53](https://arxiv.org/html/2403.04765v2#bib.bib53), [33](https://arxiv.org/html/2403.04765v2#bib.bib33), [54](https://arxiv.org/html/2403.04765v2#bib.bib54), [13](https://arxiv.org/html/2403.04765v2#bib.bib13)], where the robustness and discriminativeness of local descriptors are significantly improved. Besides, some methods [[12](https://arxiv.org/html/2403.04765v2#bib.bib12), [10](https://arxiv.org/html/2403.04765v2#bib.bib10), [38](https://arxiv.org/html/2403.04765v2#bib.bib38), [32](https://arxiv.org/html/2403.04765v2#bib.bib32), [55](https://arxiv.org/html/2403.04765v2#bib.bib55)] managed to learn the detector and descriptor together. SuperGlue[[43](https://arxiv.org/html/2403.04765v2#bib.bib43)] is a pioneering method that first introduces the transformer mechanism into matching, which has shown notable improvements over classical handcrafted matchers. As a side effect, it also costs more time, especially with many keypoints to match. To improve the efficiency, some subsequent works, such as [[6](https://arxiv.org/html/2403.04765v2#bib.bib6), [48](https://arxiv.org/html/2403.04765v2#bib.bib48)], endeavor to reduce the size of the attention mechanism, albeit at the cost of sacrificing performance. LightGlue[[30](https://arxiv.org/html/2403.04765v2#bib.bib30)] introduces a new scheme for efficient sparse matching that is adaptive to the matching difficulty, where the attention process can be stopped earlier for easy pairs. It is faster than SuperGlue and can achieve competitive performance. However, robustly detecting keypoints across images is still challenging, especially for texture-poor regions. Unlike them, our method focuses on the efficiency of the detector-free method, which eliminates the restriction of keypoint detection and shows superior performance for challenging pairs.

Detector-Free Image Matching. Detector-free methods directly match images instead of relying on a set of detected keypoints, producing semi-dense or dense matches. NC-Net[[39](https://arxiv.org/html/2403.04765v2#bib.bib39)] represents all features and possible matches as a 4D correlation volume. Sparse NC-Net[[40](https://arxiv.org/html/2403.04765v2#bib.bib40)] utilizes sparse correlation layers to ease resolution limitations. Subsequently, DRC-Net[[27](https://arxiv.org/html/2403.04765v2#bib.bib27)] improves efficiency and further improves performance in a coarse-to-fine manner.

LoFTR[[50](https://arxiv.org/html/2403.04765v2#bib.bib50)] first employs the Transformer in detector-free matching to model the long-range dependencies. It shows remarkable matching capabilities, however, suffers from low efficiency due to the huge computation of densely transforming entire coarse feature maps. Many follow-up works further improve the matching accuracy. Matchformer[[59](https://arxiv.org/html/2403.04765v2#bib.bib59)] and AspanFormer[[7](https://arxiv.org/html/2403.04765v2#bib.bib7)] perform attention on multi-scale features, where local attention regions of [[7](https://arxiv.org/html/2403.04765v2#bib.bib7)] are found with the help of estimated flow. QuadTree[[52](https://arxiv.org/html/2403.04765v2#bib.bib52)] gradually restricts the attention span during hierarchical attention to relevant areas, which can reduce overall computation. However, these designs contribute marginally or even decrease efficiency, since the hierarchical nature of multi-scale attention will further introduce latencies. TopicFM[[18](https://arxiv.org/html/2403.04765v2#bib.bib18)] first assigns features with similar semantic meanings to the same topic, where attention is conducted within each topic for efficiency. Since it needs to sequentially process each token’s features for transformation, the efficiency improvement is limited. Moreover, performing local attention within topics can potentially restrict the capability of modeling long-range dependencies. Compared with them, the proposed aggregated attention module in our method significantly improves efficiency while achieving better accuracy.

Dense matching methods[[56](https://arxiv.org/html/2403.04765v2#bib.bib56), [14](https://arxiv.org/html/2403.04765v2#bib.bib14), [15](https://arxiv.org/html/2403.04765v2#bib.bib15)] are designed to estimate all possible correspondences between two images, which show strong robustness. However, they are generally much slower compared with sparse and semi-dense methods. Unlike them, our method produces semi-dense matches with competitive performance and considerably better efficiency.

Transformer has been broadly used in multiple vision tasks, including feature matching. The efficiency and memory footprint of handling large token sizes are the main limitations of transformer[[58](https://arxiv.org/html/2403.04765v2#bib.bib58)], where some methods[[60](https://arxiv.org/html/2403.04765v2#bib.bib60), [22](https://arxiv.org/html/2403.04765v2#bib.bib22), [23](https://arxiv.org/html/2403.04765v2#bib.bib23)] attempt to reduce the complexity to a linear scale to alleviate these problems. Some methods[[26](https://arxiv.org/html/2403.04765v2#bib.bib26), [9](https://arxiv.org/html/2403.04765v2#bib.bib9)] propose optimizing transformer models for specific hardware architectures for memory and running-time efficiency. They are orthogonal to our method and can be naturally adapted into the pipeline for further efficiency improvement.

![Image 6: Refer to caption](https://arxiv.org/html/2403.04765v2/x9.png)

Figure 2: Pipeline Overview.(1) Given an image pair, a CNN network extracts coarse feature maps 𝐅~A subscript normal-~𝐅 𝐴\tilde{\textbf{F}}_{A}over~ start_ARG F end_ARG start_POSTSUBSCRIPT italic_A end_POSTSUBSCRIPT and 𝐅~B subscript normal-~𝐅 𝐵\tilde{\textbf{F}}_{B}over~ start_ARG F end_ARG start_POSTSUBSCRIPT italic_B end_POSTSUBSCRIPT, as well as fine features. (2) Then, we transform coarse features for more discriminative feature maps by interleaving our aggregated self- and cross-attention N 𝑁 N italic_N times, where adaptively feature aggregation is performed to reduce token size before each attention for efficiency. (3) Transformed coarse features are correlated for the score matrix 𝒮 𝒮\mathcal{S}caligraphic_S. Mutual-nearest-neighbor(MNN) searching is followed to establish coarse matches {ℳ c}subscript ℳ 𝑐\{\mathcal{M}_{c}\}{ caligraphic_M start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT }. (4) To refine coarse matches, discriminative fine features 𝐅^A t superscript subscript normal-^𝐅 𝐴 𝑡\hat{\textbf{F}}_{A}^{t}over^ start_ARG F end_ARG start_POSTSUBSCRIPT italic_A end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT, 𝐅^B t superscript subscript normal-^𝐅 𝐵 𝑡\hat{\textbf{F}}_{B}^{t}over^ start_ARG F end_ARG start_POSTSUBSCRIPT italic_B end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT in full resolution are obtained by fusing transformed coarse features 𝐅~A t superscript subscript normal-~𝐅 𝐴 𝑡\tilde{\textbf{F}}_{A}^{t}over~ start_ARG F end_ARG start_POSTSUBSCRIPT italic_A end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT, 𝐅~B t superscript subscript normal-~𝐅 𝐵 𝑡\tilde{\textbf{F}}_{B}^{t}over~ start_ARG F end_ARG start_POSTSUBSCRIPT italic_B end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT with backbone features. Feature patches are then cropped centered at each coarse match ℳ c subscript ℳ 𝑐\mathcal{M}_{c}caligraphic_M start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT. A two-stage refinement is followed to obtain sub-pixel correspondence ℳ f subscript ℳ 𝑓\mathcal{M}_{f}caligraphic_M start_POSTSUBSCRIPT italic_f end_POSTSUBSCRIPT. 

3 Method
--------

Given a pair of images 𝐈 A,𝐈 B subscript 𝐈 𝐴 subscript 𝐈 𝐵\bm{\mathrm{I}}_{A},\bm{\mathrm{I}}_{B}bold_I start_POSTSUBSCRIPT italic_A end_POSTSUBSCRIPT , bold_I start_POSTSUBSCRIPT italic_B end_POSTSUBSCRIPT, our objective is to establish a set of reliable correspondences between them. We achieve this by a coarse-to-fine matching pipeline, which first establishes coarse matches on downsampled feature maps and then refines them for high accuracy. An overview of our pipeline is shown in Fig.[2](https://arxiv.org/html/2403.04765v2#S2.F2 "Figure 2 ‣ Detector-Based Image Matching. ‣ 2 Related Work ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed").

### 3.1 Local Feature Extraction

Image feature maps are first extracted by a lightweight backbone for later transformation and matching. Unlike LoFTR and many other detector-free matchers that use a heavy multi-branch ResNet[[20](https://arxiv.org/html/2403.04765v2#bib.bib20)] network for feature extraction, we alternate to a lightweight single-branch network with reparameterization[[11](https://arxiv.org/html/2403.04765v2#bib.bib11)] to achieve better inference efficiency while preserving the model performance.

In particular, a multi-branch CNN network with residual connections is applied during training for maximum representational power. At inference time, we losslessly convert the feature backbone into an efficient single-branch network by adopting the reparameterization technique[[11](https://arxiv.org/html/2403.04765v2#bib.bib11)], which is achieved by fusing parallel convolution kernels into a single one. Then, the intermediate 1/8 1 8\nicefrac{{1}}{{8}}/ start_ARG 1 end_ARG start_ARG 8 end_ARG down-sampled coarse features 𝐅~A subscript~𝐅 𝐴\tilde{\textbf{F}}_{A}over~ start_ARG F end_ARG start_POSTSUBSCRIPT italic_A end_POSTSUBSCRIPT, 𝐅~B subscript~𝐅 𝐵\tilde{\textbf{F}}_{B}over~ start_ARG F end_ARG start_POSTSUBSCRIPT italic_B end_POSTSUBSCRIPT and fine features in 1/4 1 4\nicefrac{{1}}{{4}}/ start_ARG 1 end_ARG start_ARG 4 end_ARG and 1/2 1 2\nicefrac{{1}}{{2}}/ start_ARG 1 end_ARG start_ARG 2 end_ARG resolutions are extracted efficiently for later coarse-to-fine matching.

### 3.2 Efficient Local Feature Transformation

After the feature extraction, the coarse-level feature maps 𝐅~A subscript~𝐅 𝐴\tilde{\textbf{F}}_{A}over~ start_ARG F end_ARG start_POSTSUBSCRIPT italic_A end_POSTSUBSCRIPT and 𝐅~B subscript~𝐅 𝐵\tilde{\textbf{F}}_{B}over~ start_ARG F end_ARG start_POSTSUBSCRIPT italic_B end_POSTSUBSCRIPT are transformed by interleaving self- and cross-attention 1 1 1 We feed feature of one image as query and feature of the other image as key and value into cross-attention, similar to SG[[43](https://arxiv.org/html/2403.04765v2#bib.bib43)] and LoFTR[[50](https://arxiv.org/html/2403.04765v2#bib.bib50)].n 𝑛 n italic_n times to improve discriminativeness. The transformed features are denoted as 𝐅~A t superscript subscript~𝐅 𝐴 𝑡\tilde{\textbf{F}}_{A}^{t}over~ start_ARG F end_ARG start_POSTSUBSCRIPT italic_A end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT, 𝐅~B t superscript subscript~𝐅 𝐵 𝑡\tilde{\textbf{F}}_{B}^{t}over~ start_ARG F end_ARG start_POSTSUBSCRIPT italic_B end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT.

Previous methods often perform attention on the entire coarse-level feature maps, where linear attention instead of vanilla attention is applied to ensure a manageable computation cost. However, the efficiency is still limited due to the large token size of coarse features. Moreover, the usage of linear attention leads to sub-optimal model capability. Unlike them, we propose efficient aggregated attention for both efficiency and performance.

#### Preliminaries.

First, we provide a brief overview of the commonly used vanilla attention and linear attention. Vanilla attention is a core mechanism in transformer encoder layer, relying on three inputs: query Q, key K, and value V. The resultant output is a weighted sum of the value, where the weighted matrix is determined by the query and its corresponding key. Formally, the attention function is defined as follows:

VanillaAttention⁡(Q,K,V)=softmax⁡(Q⁢K T)⁢V.VanillaAttention 𝑄 𝐾 𝑉 softmax 𝑄 superscript 𝐾 𝑇 𝑉\operatorname{VanillaAttention}(Q,K,V)=\operatorname{softmax}({QK^{T}})V\enspace.roman_VanillaAttention ( italic_Q , italic_K , italic_V ) = roman_softmax ( italic_Q italic_K start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT ) italic_V .(1)

However, applying the vanilla attention directly to dense local features is impractical due to the significant token size. To address this issue, previous methods use linear attention to reduce the computational complexity from quadratic to linear:

LinearAttention⁡(Q,K,V)=ϕ⁢(Q)⁢(ϕ⁢(K)T⁢ϕ⁢(V)).LinearAttention 𝑄 𝐾 𝑉 italic-ϕ 𝑄 italic-ϕ superscript 𝐾 𝑇 italic-ϕ 𝑉\operatorname{LinearAttention}(Q,K,V)=\phi(Q)(\phi(K)^{T}\phi(V))\enspace.roman_LinearAttention ( italic_Q , italic_K , italic_V ) = italic_ϕ ( italic_Q ) ( italic_ϕ ( italic_K ) start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT italic_ϕ ( italic_V ) ) .(2)

where ϕ⁢(⋅)=e⁢l⁢u⁢(⋅)+1 italic-ϕ⋅𝑒 𝑙 𝑢⋅1\phi(\cdot)=elu(\cdot)+1 italic_ϕ ( ⋅ ) = italic_e italic_l italic_u ( ⋅ ) + 1. However, it comes at the cost of reduced representational power, which is also observed by[[5](https://arxiv.org/html/2403.04765v2#bib.bib5)].

![Image 7: Refer to caption](https://arxiv.org/html/2403.04765v2/x10.png)

Figure 3: Detailed Transformer Module Comparison. Unlike LoFTR which uses all tokens of feature maps to compute attention and resort to linear attention to reduce the computational cost, the proposed attention module first aggregates features for salient tokens, which is significantly more efficient for attention. Then the vanilla attention is utilized to transform aggregated features, where relative positional encoding is inserted to capture the spatial information. Transformed features are upsampled and fused with the original features to form the final features. 

#### Aggragated Attention Module.

After comprehensively investigating the mechanism of the transformer on coarse feature maps, we have two observations that motivate us to devise a new efficient aggregated attention. First, the attention regions of neighboring query tokens are similar, thus we can aggregate the neighboring tokens of f i subscript 𝑓 𝑖 f_{i}italic_f start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT to prevent the redundant computation. Second, most of the attention weights of each query token are concentrated on a small number of key tokens, hence we can select the salient tokens of f j subscript 𝑓 𝑗 f_{j}italic_f start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT before attention to reduce the computation.

Therefore, we propose to first aggregate the f i subscript 𝑓 𝑖 f_{i}italic_f start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT token utilizing a depth-wise convolution network, and f j subscript 𝑓 𝑗 f_{j}italic_f start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT is aggregated by a max pooling layer to get reduced salient tokens:

f i′=Conv2D⁡(f i),f j′=MaxPool⁡(f j),formulae-sequence subscript superscript 𝑓′𝑖 Conv2D subscript 𝑓 𝑖 subscript superscript 𝑓′𝑗 MaxPool subscript 𝑓 𝑗 f^{\prime}_{i}=\operatorname{Conv2D}(f_{i}),\enspace f^{\prime}_{j}=% \operatorname{MaxPool}(f_{j})\enspace,italic_f start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = Conv2D ( italic_f start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) , italic_f start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT = roman_MaxPool ( italic_f start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) ,(3)

where Conv2D Conv2D\operatorname{Conv2D}Conv2D is implemented by a strided depthwise convolution with a kernel size of s×s 𝑠 𝑠 s\times s italic_s × italic_s, identical to that of the max-pooling layer. Then positional encoding and vanilla attention are followed to process reduced tokens. Positional encoding(PE) can help to model the spatial location contexts, where RoPE[[49](https://arxiv.org/html/2403.04765v2#bib.bib49)] is adopted in practice to account for more robust relative positions, inspired by[[30](https://arxiv.org/html/2403.04765v2#bib.bib30)]. Note that the PE layer is enabled exclusively for self-attention and skipped during cross-attention. The transformed feature map is then upsampled and fused with f i subscript 𝑓 𝑖 f_{i}italic_f start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT for the final feature map. Due to the aggregation and selection, the number of tokens in f i′subscript superscript 𝑓′𝑖 f^{\prime}_{i}italic_f start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT and f j′subscript superscript 𝑓′𝑗 f^{\prime}_{j}italic_f start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT is reduced by s 2 superscript 𝑠 2 s^{2}italic_s start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT, which contributes to the efficiency of the attention phase.

### 3.3 Coarse-level Matching Module

We establish coarse-level matches based on the previously transformed coarse feature maps 𝐅~A t superscript subscript~𝐅 𝐴 𝑡\tilde{\textbf{F}}_{A}^{t}over~ start_ARG F end_ARG start_POSTSUBSCRIPT italic_A end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT, 𝐅~B t superscript subscript~𝐅 𝐵 𝑡\tilde{\textbf{F}}_{B}^{t}over~ start_ARG F end_ARG start_POSTSUBSCRIPT italic_B end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT. Coarse correspondences indicate rough match regions for later subpixel-level matching in the refinement phase. To achieve this, 𝐅~A t superscript subscript~𝐅 𝐴 𝑡\tilde{\textbf{F}}_{A}^{t}over~ start_ARG F end_ARG start_POSTSUBSCRIPT italic_A end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT and 𝐅~B t superscript subscript~𝐅 𝐵 𝑡\tilde{\textbf{F}}_{B}^{t}over~ start_ARG F end_ARG start_POSTSUBSCRIPT italic_B end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT are densely correlated to build a score matrix 𝒮 𝒮\mathcal{S}caligraphic_S. The softmax operator on both 𝒮 𝒮\mathcal{S}caligraphic_S dimensions (referred to as dual-softmax) is then applied to obtain the probability of mutual nearest matching, which is commonly used in [[39](https://arxiv.org/html/2403.04765v2#bib.bib39), [57](https://arxiv.org/html/2403.04765v2#bib.bib57), [50](https://arxiv.org/html/2403.04765v2#bib.bib50)]. The coarse correspondences {ℳ c}subscript ℳ 𝑐\{\mathcal{M}_{c}\}{ caligraphic_M start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT } are established by selecting matches above the score threshold τ 𝜏\tau italic_τ while satisfying the mutual-nearest-neighbor (MNN) constraint.

#### Efficient Inference Strategy.

We observe that the dual-softmax operator in the coarse matching can significantly restrict the efficiency in inference due to the large token size, especially for high-resolution images. Moreover, we find that the dual-softmax operator is crucial for training, dropping it at inference time while directly using the score matrix 𝒮 𝒮\mathcal{S}caligraphic_S for MNN matching can also work well with better efficiency.

The reason for using the dual-softmax operator in training is that it can help to train discriminative features. Intuitively, with the softmax operation, the matching score between two pixels can also conditioned on other pixels. This mechanism forces the network to improve feature similarity of true correspondences while suppressing similarity with irrelevant points. With trained discriminative features, the softmax operation can be potentially eliminated during inference.

We denote the model skipping dual-softmax layer in inference as _efficiency optimized model_. Results in Tab.[1](https://arxiv.org/html/2403.04765v2#S4.T1 "Table 1 ‣ 4 Experiments ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed") demonstrate the effectiveness of this design.

### 3.4 Subpixel-Level Refinement Module

As overviewed in Fig.[2](https://arxiv.org/html/2403.04765v2#S2.F2 "Figure 2 ‣ Detector-Based Image Matching. ‣ 2 Related Work ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed")(4), with established coarse matches {ℳ c}subscript ℳ 𝑐{\{\mathcal{M}_{c}\}}{ caligraphic_M start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT }, we refine them for sub-pixel accuracy with our refinement module. It is composed of an efficient feature patch extractor for discriminative fine features, followed by a two-stage feature correlation layer for final matches {ℳ f}subscript ℳ 𝑓{\{\mathcal{M}_{f}\}}{ caligraphic_M start_POSTSUBSCRIPT italic_f end_POSTSUBSCRIPT }.

#### Efficient Fine Feature Extraction.

We first extract discriminative fine feature patches centered at each coarse match ℳ c subscript ℳ 𝑐\mathcal{M}_{c}caligraphic_M start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT by an efficient fusion network for later match refinement. For efficiency, our key idea here is to re-leverage the previously transformed coarse features 𝐅~A t superscript subscript~𝐅 𝐴 𝑡\tilde{\textbf{F}}_{A}^{t}over~ start_ARG F end_ARG start_POSTSUBSCRIPT italic_A end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT, 𝐅~B t superscript subscript~𝐅 𝐵 𝑡\tilde{\textbf{F}}_{B}^{t}over~ start_ARG F end_ARG start_POSTSUBSCRIPT italic_B end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT to obtain cross-view attended discriminative fine features, instead of introducing additional feature transform networks as in LoFTR[[50](https://arxiv.org/html/2403.04765v2#bib.bib50)].

To be specific, 𝐅~A t superscript subscript~𝐅 𝐴 𝑡\tilde{\textbf{F}}_{A}^{t}over~ start_ARG F end_ARG start_POSTSUBSCRIPT italic_A end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT and 𝐅~B t superscript subscript~𝐅 𝐵 𝑡\tilde{\textbf{F}}_{B}^{t}over~ start_ARG F end_ARG start_POSTSUBSCRIPT italic_B end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT are adaptively fused with 1/4 1 4\nicefrac{{1}}{{4}}/ start_ARG 1 end_ARG start_ARG 4 end_ARG and 1/2 1 2\nicefrac{{1}}{{2}}/ start_ARG 1 end_ARG start_ARG 2 end_ARG resolution backbone features by convolution and upsampling to obtain fine feature maps 𝐅^A t superscript subscript^𝐅 𝐴 𝑡\hat{\textbf{F}}_{A}^{t}over^ start_ARG F end_ARG start_POSTSUBSCRIPT italic_A end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT, 𝐅^B t superscript subscript^𝐅 𝐵 𝑡\hat{\textbf{F}}_{B}^{t}over^ start_ARG F end_ARG start_POSTSUBSCRIPT italic_B end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT in the original image resolution. Then local feature patches are cropped on fine feature maps centered at each coarse match. Since only shallow feed-forward networks are included, our fine feature fusion network is remarkably efficient.

#### Two-Stage Correlation for Refinement.

Based on the extracted fine local feature patches of coarse matches, we search for high-accurate sub-pixel matches. To refine a coarse match, a commonly used strategy[[50](https://arxiv.org/html/2403.04765v2#bib.bib50), [7](https://arxiv.org/html/2403.04765v2#bib.bib7), [18](https://arxiv.org/html/2403.04765v2#bib.bib18)] is to select the center-patch feature of 𝐈 A subscript 𝐈 𝐴\bm{\mathrm{I}}_{A}bold_I start_POSTSUBSCRIPT italic_A end_POSTSUBSCRIPT as a fixed reference point, and perform feature correlation and expectation on the entire corresponding feature patch for its fine match. However, this refinement-by-expectations will introduce location variance to the final match, because irrelevant regions also have weights and can affect results. Therefore, we propose a novel two-stage correlation module to solve this problem.

Our idea is to utilize a mutual-nearest-neighbor(MNN) matching to get intermediate pixel-level refined matches in the first stage, and then refine them for subpixel accuracy by correlation and expectation. Motivations are that MNN matching don’t have spatial variance since matches are selected by directly indexing pixels with maximum scores, but cannot achieve sub-pixel accuracy. Conversely, refinement-by-expectation can achieve sub-pixel accuracy but variance exists. The proposed two-stage refinement can draw benefits by combining the best of both worlds.

Specifically, to refine a coarse-level correspondence ℳ c subscript ℳ 𝑐\mathcal{M}_{c}caligraphic_M start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT, the first-stage refinement phase densely correlates their fine feature patches to obtain the local patch score matrix 𝒮 l subscript 𝒮 𝑙\mathcal{S}_{l}caligraphic_S start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT. MNN searching is then applied on 𝒮 l subscript 𝒮 𝑙\mathcal{S}_{l}caligraphic_S start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT to get intermediate pixel-level fine matches. To limit the overall match number, we select the top-1 1 1 1 fine match for one coarse match by sorting the correlation scores.

Then, we further refine these pixel-level matches for subpixel accuracy by our second-stage refinement. Since the matching accuracy has already significantly improved in first-stage refinement, now we can use a tiny local window for correlation and expectation with a maximum suppression of location variance. In practice, we correlate the feature of each point in 𝐈 A subscript 𝐈 𝐴\textbf{I}_{A}I start_POSTSUBSCRIPT italic_A end_POSTSUBSCRIPT with a 3×3 3 3 3\times 3 3 × 3 feature patch centered at its fine match in 𝐈 B subscript 𝐈 𝐵\textbf{I}_{B}I start_POSTSUBSCRIPT italic_B end_POSTSUBSCRIPT. The softmax operator is then applied to get a match distribution matrix and the final refined match is obtained by calculating expectations.

### 3.5 Supervision

The entire pipeline is trained end-to-end by supervising the coarse and refinement matching modules separately.

#### Coarse-Level Matching Supervision.

The coarse ground truth matches {ℳ c}g⁢t subscript subscript ℳ 𝑐 𝑔 𝑡\{\mathcal{M}_{c}\}_{gt}{ caligraphic_M start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_g italic_t end_POSTSUBSCRIPT with a total number of N 𝑁 N italic_N are built by warping grid-level points from 𝐈 A subscript 𝐈 𝐴\textbf{I}_{A}I start_POSTSUBSCRIPT italic_A end_POSTSUBSCRIPT to 𝐈 B subscript 𝐈 𝐵\textbf{I}_{B}I start_POSTSUBSCRIPT italic_B end_POSTSUBSCRIPT via depth maps and image poses following previous methods[[43](https://arxiv.org/html/2403.04765v2#bib.bib43), [50](https://arxiv.org/html/2403.04765v2#bib.bib50)]. The produced correlation score matrix 𝒮 𝒮\mathcal{S}caligraphic_S in coarse matching is supervised by minimizing the log-likelihood loss over locations of {ℳ c}g⁢t subscript subscript ℳ 𝑐 𝑔 𝑡\{\mathcal{M}_{c}\}_{gt}{ caligraphic_M start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_g italic_t end_POSTSUBSCRIPT:

ℒ c=−1 N⁢∑(i~,j~)∈{ℳ c}g⁢t log⁡𝒮⁢(i~,j~).subscript ℒ 𝑐 1 𝑁 subscript~𝑖~𝑗 subscript subscript ℳ 𝑐 𝑔 𝑡 𝒮~𝑖~𝑗\mathcal{L}_{c}=-\frac{1}{N}\sum_{(\tilde{i},\tilde{j})\in\{\mathcal{M}_{c}\}_% {gt}}\log\mathcal{S}\left(\tilde{i},\tilde{j}\right)\enspace.caligraphic_L start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT = - divide start_ARG 1 end_ARG start_ARG italic_N end_ARG ∑ start_POSTSUBSCRIPT ( over~ start_ARG italic_i end_ARG , over~ start_ARG italic_j end_ARG ) ∈ { caligraphic_M start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_g italic_t end_POSTSUBSCRIPT end_POSTSUBSCRIPT roman_log caligraphic_S ( over~ start_ARG italic_i end_ARG , over~ start_ARG italic_j end_ARG ) .(4)

#### Fine-Level Matching Supervision.

We train the proposed two-stage fine-level matching module by separately supervising the two phases. The first stage fine loss ℒ f⁢1 subscript ℒ 𝑓 1\mathcal{L}_{f1}caligraphic_L start_POSTSUBSCRIPT italic_f 1 end_POSTSUBSCRIPT is to minimize the log-likelihood loss of each fine local score matrix 𝒮 l subscript 𝒮 𝑙\mathcal{S}_{l}caligraphic_S start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT based on the pixel-level ground truth fine matches, similar to coarse loss. The second stage is trained by ℒ f⁢2 subscript ℒ 𝑓 2\mathcal{L}_{f2}caligraphic_L start_POSTSUBSCRIPT italic_f 2 end_POSTSUBSCRIPT that calculates the ℓ 2 subscript ℓ 2\ell_{2}roman_ℓ start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT loss between the final subpixel matches {ℳ f}subscript ℳ 𝑓\{\mathcal{M}_{f}\}{ caligraphic_M start_POSTSUBSCRIPT italic_f end_POSTSUBSCRIPT } and ground truth fine matches {ℳ f}g⁢t subscript subscript ℳ 𝑓 𝑔 𝑡\{\mathcal{M}_{f}\}_{gt}{ caligraphic_M start_POSTSUBSCRIPT italic_f end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_g italic_t end_POSTSUBSCRIPT.

The total loss is the weighted sum of all supervisions: ℒ=ℒ c+α⁢ℒ f⁢1+β⁢ℒ f⁢2 ℒ subscript ℒ 𝑐 𝛼 subscript ℒ 𝑓 1 𝛽 subscript ℒ 𝑓 2\mathcal{L}=\mathcal{L}_{c}+\alpha\mathcal{L}_{f1}+\beta\mathcal{L}_{f2}caligraphic_L = caligraphic_L start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT + italic_α caligraphic_L start_POSTSUBSCRIPT italic_f 1 end_POSTSUBSCRIPT + italic_β caligraphic_L start_POSTSUBSCRIPT italic_f 2 end_POSTSUBSCRIPT.

4 Experiments
-------------

In this section, we evaluate the performance of our method on several downstream tasks, including homography estimation, pairwise pose estimation and visual localization. Furthermore, we evaluate the effectiveness of our design by conducting detailed ablation studies.

Category Method MegaDepth Dataset ScanNet Dataset Time(ms)
AUC@5°°\degree°AUC@10°°\degree°AUC@20°°\degree°AUC@5°°\degree°AUC@10°°\degree°AUC@20°°\degree°
Sparse SP + NN 31.7 46.8 60.1 7.5 18.6 32.1 10.8
SP + SG 49.7 67.1 80.6 16.2 32.8 49.7 48.3
SP + LG 49.9 67.0 80.1 14.8 30.8 47.5 31.9/30.7
Semi-Dense DRC-Net 27.0 42.9 58.3 7.7 17.9 30.5 328.0
LoFTR 52.8 69.2 81.2 16.9 33.6 50.6 66.2
QuadTree 54.6 70.5 82.2 19.0 37.3 53.5 100.7
MatchFormer 53.3 69.7 81.8 15.8 32.0 48.0 128.9
TopicFM 54.1 70.1 81.6 17.3 35.5 50.9 66.4
AspanFormer 55.3 71.5 83.1 19.6 37.7 54.4 81.6
Ours 56.4 72.2 83.5 19.2 37.0 53.6 40.1/34.4
Ours(Optimized)55.4 71.4 82.9 17.4 34.4 51.2 35.6/27.0
Dense DKM 60.4 74.9 85.1 26.64 47.07 64.17 210.8
ROMA 62.6 76.7 86.3 28.9 50.4 68.3 302.7

Table 1: Results of Relative Pose Estimation on MegaDepth Dataset and ScanNet Dataset. We use the models trained on the MegaDepth dataset to evaluate all methods on both datasets, which can show the intra- and inter-dataset generalization abilities. The AUC of pose error at different thresholds, along with the processing time for matching image pair at a resolution of 640×480 640 480 640\times 480 640 × 480, is presented. For SP + LG, Ours, and Ours(Optimized), the running times of the model using FP32/Mixed-Precision numerical precisions are shown. 

### 4.1 Implementation Details

We adopt RepVGG[[11](https://arxiv.org/html/2403.04765v2#bib.bib11)] as our feature backbone, and self- and cross-attention are interleaved for N=4 𝑁 4 N=4 italic_N = 4 times to transform coarse features. For each attention, we aggregate features by a depth-wise convolution layer and a max-pooling layer, both with a kernel size of 4×4 4 4 4\times 4 4 × 4. Our model is trained on the MegaDepth dataset[[28](https://arxiv.org/html/2403.04765v2#bib.bib28)], which is a large-scale outdoor dataset. The test scenes are separated from training data following[[50](https://arxiv.org/html/2403.04765v2#bib.bib50)]. The loss function’s weights α 𝛼\alpha italic_α and β 𝛽\beta italic_β are set to 1.0 1.0 1.0 1.0 and 0.25 0.25 0.25 0.25, respectively. We use the AdamW optimizer with an initial learning rate of 4×10−3 4 superscript 10 3 4\times 10^{-3}4 × 10 start_POSTSUPERSCRIPT - 3 end_POSTSUPERSCRIPT. The network training takes about 15 hours with a batch size of 16 on 8 8 8 8 NVIDIA V100 GPUs. And the coarse and fine stages are trained together from scratch. The trained model on MegaDepth is used to evaluate all datasets and tasks in our experiments to demonstrate the generalization ability.

### 4.2 Relative Pose Estimation

Datasets. We use the outdoor MegaDepth[[28](https://arxiv.org/html/2403.04765v2#bib.bib28)] dataset and indoor ScanNet[[8](https://arxiv.org/html/2403.04765v2#bib.bib8)] dataset for the evaluation of relative pose estimation to demonstrate the efficacy of our method.

MegaDepth dataset is a large-scale dataset containing sparse 3D reconstructions from 196 scenes. The key challenges on this dataset are large viewpoints and illumination changes, as well as repetitive patterns. We follow the test split of the previous method[[50](https://arxiv.org/html/2403.04765v2#bib.bib50)] that uses 1500 sampled pairs from scenes “Sacre Coeur” and “St. Peter’s Square” for evaluation. Images are resized so that the longest edge equals 1200 for all semi-dense and dense methods. Following[[30](https://arxiv.org/html/2403.04765v2#bib.bib30)], sparse methods are provided resized images with longest edge equals 1600.

ScanNet dataset contains 1613 sequences with ground-truth depth maps and camera poses. They depict indoor scenes with viewpoint changes and texture-less regions. We use the sampled test pairs from[[43](https://arxiv.org/html/2403.04765v2#bib.bib43)] for the evaluation, where images are resized to 640×480 640 480 640\times 480 640 × 480 for all methods.

Baselines. We compare the proposed method with three categories of methods: 1) sparse keypoint detection and matching methods, including SuperPoint[[10](https://arxiv.org/html/2403.04765v2#bib.bib10)] with Nearest-Neighbor(NN), SuperGlue(SG)[[43](https://arxiv.org/html/2403.04765v2#bib.bib43)], LightGlue(LG)[[30](https://arxiv.org/html/2403.04765v2#bib.bib30)] matchers, 2) semi-dense matchers, including DRC-Net[[27](https://arxiv.org/html/2403.04765v2#bib.bib27)], LoFTR[[50](https://arxiv.org/html/2403.04765v2#bib.bib50)], QuadTree Attention[[52](https://arxiv.org/html/2403.04765v2#bib.bib52)], MatchFormer[[59](https://arxiv.org/html/2403.04765v2#bib.bib59)], AspanFormer[[7](https://arxiv.org/html/2403.04765v2#bib.bib7)], TopicFM[[18](https://arxiv.org/html/2403.04765v2#bib.bib18)], and 3) state-of-the-art dense matcher ROMA[[15](https://arxiv.org/html/2403.04765v2#bib.bib15)] that predict matches for each pixel.

Evaluation protocol. Following previous methods, the recovered relative poses by matches are evaluated for reflecting matching accuracy. The pose error is defined as the maximum of angular errors in rotation and translation. We report the AUC of the pose error at thresholds(5°, 10°, and 20°). Moreover, the running time of matching each image pair in the ScanNet dataset is reported for comprehensively understanding the matching accuracy and efficiency balance. We use a single NVIDIA 3090 to evaluate the running time of all methods.

Results. As shown in Tab.[1](https://arxiv.org/html/2403.04765v2#S4.T1 "Table 1 ‣ 4 Experiments ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed"), the proposed method achieves competitive performances compared with sparse and semi-dense methods on both datasets. Qualitative comparisons are shown in Fig.[4](https://arxiv.org/html/2403.04765v2#S4.F4 "Figure 4 ‣ 4.4 Visual Localization ‣ 4 Experiments ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed"). Specifically, our method outperforms the best semi-dense baseline AspanFormer on all metrics of the MegaDepth dataset and has lower but comparable performance on the ScanNet dataset, with ∼2 similar-to absent 2\sim 2∼ 2 times faster. Our optimized model that eliminates the dual-softmax operator in coarse-level matching further brings efficiency improvements, with slight performance decreases. Using this strategy, our method can outperform the efficient and robust sparse method SP + LG in efficiency with significantly higher accuracy. Dense matcher ROMA shows remarkable matching capability but is slow for applications in practice. Moreover, since ROMA utilizes the pre-trained DINOv2[[37](https://arxiv.org/html/2403.04765v2#bib.bib37)] backbone, its strong generalizability on ScanNet may be attributed to the similar indoor training data in DINOv2, where other methods are trained on outdoor MegaDepth only. Compared with it, our method is ∼7.5 similar-to absent 7.5\sim 7.5∼ 7.5 times faster, which has a good balance between accuracy and efficiency.

Category Method Homography est. AUC
@3px@5px@10px
Sparse D2Net + NN 23.2 35.9 53.6
R2D2 + NN 50.6 63.9 76.8
DISK + NN 52.3 64.9 78.9
SP + SG 53.9 68.3 81.7
Semi-Dense Sparse-NCNet 48.9 54.2 67.1
DRC-Net 50.6 56.2 68.3
LoFTR 65.9 75.6 84.6
Ours 66.5 76.4 85.5

Table 2: Results of Homography Estimation on HPatches Dataset. Our method is compared with sparse and semi-dense methods. The AUC of reprojection error of corner points at different thresholds is reported. 

### 4.3 Homography Estimation

Dataset. We evaluate our method on HPatches dataset[[3](https://arxiv.org/html/2403.04765v2#bib.bib3)]. HPatches dataset depicts planar scenes divided into sequences. Images are taken under different viewpoints or illumination changes.

Baselines. We compare our method with sparse methods including D2Net[[12](https://arxiv.org/html/2403.04765v2#bib.bib12)], R2D2[[38](https://arxiv.org/html/2403.04765v2#bib.bib38)], DISK[[57](https://arxiv.org/html/2403.04765v2#bib.bib57)] detectors with NN matcher, and SuperPoint[[10](https://arxiv.org/html/2403.04765v2#bib.bib10)] + SuperGlue[[43](https://arxiv.org/html/2403.04765v2#bib.bib43)]. As for semi-dense methods, we compare with Sparse-NCNet[[40](https://arxiv.org/html/2403.04765v2#bib.bib40)], DRC-Net[[27](https://arxiv.org/html/2403.04765v2#bib.bib27)], and LoFTR[[50](https://arxiv.org/html/2403.04765v2#bib.bib50)]. For SuperGlue and all semi-dense methods, we use their models trained on MegaDepth dataset for evaluation.

Evaluation Protocol. Following SuperGlue[[43](https://arxiv.org/html/2403.04765v2#bib.bib43)] and LoFTR[[50](https://arxiv.org/html/2403.04765v2#bib.bib50)], we resize all images for matching so that their smallest edge equals 480 pixels. We collect the mean reprojection error of corner points, and report the area under the cumulative curve (AUC) under 3 different thresholds, including 3 3 3 3 px, 5 5 5 5 px, and 10 10 10 10 px. For all baselines, we employ the same RANSAC method as a robust homography estimator for a fair comparison. Following LoFTR, we select only the top 1000 predicted matches of semi-dense methods for the sake of fairness.

Results. As shown in Tab.[2](https://arxiv.org/html/2403.04765v2#S4.T2 "Table 2 ‣ 4.2 Relative Pose Estimation ‣ 4 Experiments ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed"), even though the number of matches is restricted, our method can also work remarkably well and outperform sparse methods significantly. Compared with semi-dense, our method can surpass them with significantly higher efficiency. We attribute this to the effectiveness of two-stage refinement for accuracy improvement and proposed aggregation module for efficiency.

Method DUC1 DUC2
(0.25m,2°°\degree°)/(0.5m,5°°\degree°)/(1.0m,10°°\degree°)
SP+SG 49.0 / 68.7 / 80.8 53.4 / 77.1 / 82.4
LoFTR 47.5 / 72.2 / 84.8 54.2 / 74.8 / 85.5
TopicFM 52.0 / 74.7 / 87.4 53.4 / 74.8 / 83.2
PATS 55.6 / 71.2 / 81.0 58.8 / 80.9 / 85.5
AspanFormer 51.5 / 73.7 / 86.0 55.0 / 74.0 / 81.7
Ours 52.0 / 74.7 / 86.9 58.0 / 80.9 / 89.3

Table 3: Results of Visual Localization on InLoc Dataset.

Method Day Night
(0.25m,2°°\degree°)/(0.5m,5°°\degree°)/(1.0m,10°°\degree°)
SP+SG 89.8 / 96.1 / 99.4 77.0 / 90.6 / 100.0
LoFTR 88.7 / 95.6 / 99.0 78.5 / 90.6 / 99.0
TopicFM 90.2 / 95.9 / 98.9 77.5 / 91.1 / 99.5
PATS 89.6 / 95.8 / 99.3 73.8 / 92.1 / 99.5
AspanFormer 89.4 / 95.6 / 99.0 77.5 / 91.6 / 99.5
Ours 89.6 / 96.2 / 99.0 77.0 / 91.1 / 99.5

Table 4: Results of Visual Localization on Aachen v1.1 Dataset.

### 4.4 Visual Localization

Datasets and Evaluation Protocols. Visual localization is an important downstream task of image matching, which aims to estimate the 6-DoF poses of query images based on the 3D scene model. We conduct experiments on two commonly used benchmarks, including InLoc[[51](https://arxiv.org/html/2403.04765v2#bib.bib51)] dataset and Aachen v1.1[[45](https://arxiv.org/html/2403.04765v2#bib.bib45)] dataset, for evaluation to demonstrate the superiority of our method. InLoc dataset is captured on indoor scenes with plenty of repetitive structures and texture-less regions, where each database image has a corresponding depth map. Aachen v1.1 is a challenging large-scale outdoor dataset for localization with large-viewpoint and day-and-night illumination changes, which particularly relies on the robustness of matching methods. We adopt its full localization track for benchmarking.

Following[[50](https://arxiv.org/html/2403.04765v2#bib.bib50), [7](https://arxiv.org/html/2403.04765v2#bib.bib7)], the open-sourced localization framework HLoc[[42](https://arxiv.org/html/2403.04765v2#bib.bib42)] is utilized. For both datasets, the percentage of pose errors satisfying both angular and distance thresholds is reported following the benchmarks, where different thresholds are used. For the InLoc dataset, the metrics of two test scenes including DUC1 and DUC2 are separately reported. As for the Aachen v1.1 dataset, the metrics corresponding to the daytime and nighttime divisions are reported.

Baselines. We compare the proposed method with both detector-based method SuperPoint[[10](https://arxiv.org/html/2403.04765v2#bib.bib10)]+SuperGlue[[43](https://arxiv.org/html/2403.04765v2#bib.bib43)] and detector-free methods including LoFTR[[50](https://arxiv.org/html/2403.04765v2#bib.bib50)], TopicFM[[18](https://arxiv.org/html/2403.04765v2#bib.bib18)], PATS[[36](https://arxiv.org/html/2403.04765v2#bib.bib36)] and Aspanformer[[7](https://arxiv.org/html/2403.04765v2#bib.bib7)].

Results. We adhere to the pipeline and evaluation settings of the online visual localization benchmark 2 2 2 https://www.visuallocalization.net/benchmark to ensure fairness. As presented in Tab.[3](https://arxiv.org/html/2403.04765v2#S4.T3 "Table 3 ‣ 4.3 Homography Estimation ‣ 4 Experiments ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed"), the proposed method achieves competitive results, taking both detector-based and detector-free methods into account. Being a method primarily geared towards efficiency, our approach can deliver results comparable to those of many accuracy-oriented methods. As depicted in Tab.[4](https://arxiv.org/html/2403.04765v2#S4.T4 "Table 4 ‣ 4.3 Homography Estimation ‣ 4 Experiments ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed"), our method also demonstrates performance on par with the best-performing approaches.

![Image 8: Refer to caption](https://arxiv.org/html/2403.04765v2/x11.png)

Figure 4: Qualitative Results. Our method is compared with the sparse matching pipeline SuperPoint[[10](https://arxiv.org/html/2403.04765v2#bib.bib10)]+LightGlue[[30](https://arxiv.org/html/2403.04765v2#bib.bib30)], semi-dense matcher AspanFormer[[7](https://arxiv.org/html/2403.04765v2#bib.bib7)]. Image pairs with texture-poor regions and large-viewpoint changes can be robustly matched by our method. The red color indicates epipolar error beyond 5×10−4 5 superscript 10 4 5\times 10^{-4}5 × 10 start_POSTSUPERSCRIPT - 4 end_POSTSUPERSCRIPT (in the normalized image coordinates). 

Method Pose Estimation AUC Time(ms)
@5°@10°@20°
Ours Full 56.4 72.2 83.5 139.2
1) Ours Optimal(w/o dual-softmax)55.4 71.4 82.9 102.0
2) Replace Agg. Attention to LoFTR’s Trans.54.7 70.5 82.2 171.4
3) Replace two-stage refine. to LoFTR’s refine.54.7 70.9 82.7 135.3
4) No second-stage refinement 55.8 71.8 83.3 138.1
5) Replace RepVGG with ResNet 55.4 71.4 82.9 156.2

Table 5: Ablation Studies. The components of our method are ablated on the MegaDepth dataset for a comprehensive understanding of our method, where averaged running times for an image pair with high-resolution 1200×1200 1200 1200 1200\times 1200 1200 × 1200 are reported. 

### 4.5 Ablation Studies

In this part, we conduct detailed ablation studies to analyze the effectiveness of our proposed modules with results shown in Tab.[5](https://arxiv.org/html/2403.04765v2#S4.T5 "Table 5 ‣ 4.4 Visual Localization ‣ 4 Experiments ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed"). 1) Without dual-softmax, our optimal model can bring huge efficiency improvement in high-resolution images. 2) In coarse feature transformation, replacing the proposed aggregated attention module with LoFTR’s transformer can bring significant efficiency dropping, as well as accuracy decrease. Note that the replaced transformer is also equipped with RoPE same as ours for fair comparison. This demonstrates the efficacy of the proposed module that performing vanilla attention on aggregated features can achieve higher efficiency with even better matching accuracy. 3) Compared with using LoFTR’s refinement that performs expectation on the entire correlation patch, the proposed two-stage refinement layer can bring accuracy improvement with neglectable latency. We attribute this to the two-stage refinement’s property that can maximize the suppression of location variance in correlation refinement. 4) Dropping the second refinement stage will lead to degraded pose accuracy with minor efficiency changes, especially on the strict AUC@5⁢°5°5\degree 5 ° metric. 5) Changing the backbone from reparameterized VGG[[11](https://arxiv.org/html/2403.04765v2#bib.bib11)] back to multi-branch ResNet[[20](https://arxiv.org/html/2403.04765v2#bib.bib20)] leads to decreased efficiency with similar accuracy, which demonstrates the effectiveness of our design choice for efficiency.

5 Conclusions
-------------

This paper introduces a new semi-dense local feature matcher based on the success of LoFTR. We revisit its designs and propose several improvements for both efficiency and matching accuracy. A key observation is that performing the Transformer on the entire coarse feature map is redundant due to the similar local information, where an aggregated attention module is proposed to perform transformer on reduced tokens with significantly better efficiency and competitive performance. Moreover, a two-stage correlation layer is devised to solve the location variance problem in LoFTR’s refinement design, which further brings accuracy improvements. As a result, our method can achieve ∼2.5 similar-to absent 2.5\sim 2.5∼ 2.5 times faster compared with LoFTR with better matching accuracy. Moreover, as a semi-dense matching method, the proposed method can achieve comparable efficiency with the recent robust sparse feature matcher LightGlue[[30](https://arxiv.org/html/2403.04765v2#bib.bib30)]. We believe this opens up the applications of our method in large-scale or latency-sensitive downstream tasks, such as image retrieval and 3D reconstruction. Please refer to the supplementary material for discussions about limitations and future works.

References
----------

*   Agarwal et al. [2009] Sameer Agarwal, Yasutaka Furukawa, Noah Snavely, Ian Simon, Brian Curless, Steven M. Seitz, and Richard Szeliski. Building rome in a day. _ICCV_, 2009. 
*   Arandjelović et al. [2016] Relja Arandjelović, Petr Gronát, Akihiko Torii, Tomás Pajdla, and Josef Sivic. Netvlad: Cnn architecture for weakly supervised place recognition. _CVPR_, pages 5297–5307, 2016. 
*   Balntas et al. [2017] Vassileios Balntas, Karel Lenc, Andrea Vedaldi, and Krystian Mikolajczyk. Hpatches: A benchmark and evaluation of handcrafted and learned local descriptors. In _CVPR_, pages 5173–5182, 2017. 
*   Bay et al. [2008] Herbert Bay, Andreas Ess, Tinne Tuytelaars, and Luc Van Gool. Speeded-up robust features (surf). _CVIU_, 110(3):346–359, 2008. 
*   Cai et al. [2022] Han Cai, Chuang Gan, and Song Han. Efficientvit: Enhanced linear attention for high-resolution low-computation visual recognition. _arXiv preprint arXiv:2205.14756_, 2022. 
*   Chen et al. [2021] Hongkai Chen, Zixin Luo, Jiahui Zhang, Lei Zhou, Xuyang Bai, Zeyu Hu, Chiew-Lan Tai, and Long Quan. Learning to match features with seeded graph matching network. In _ICCV_, pages 6301–6310, 2021. 
*   Chen et al. [2022] Hongkai Chen, Zixin Luo, Lei Zhou, Yurun Tian, Mingmin Zhen, Tian Fang, David Mckinnon, Yanghai Tsin, and Long Quan. Aspanformer: Detector-free image matching with adaptive span transformer. In _ECCV_, pages 20–36. Springer, 2022. 
*   Dai et al. [2017] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In _CVPR_, pages 5828–5839, 2017. 
*   Dao et al. [2022] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In _NeurIPS_, 2022. 
*   DeTone et al. [2018] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superpoint: Self-supervised interest point detection and description. _CVPRW_, 2018. 
*   Ding et al. [2021] Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, and Jian Sun. Repvgg: Making vgg-style convnets great again. In _CVPR_, pages 13733–13742, 2021. 
*   Dusmanu et al. [2019] Mihai Dusmanu, Ignacio Rocco, Tomas Pajdla, Marc Pollefeys, Josef Sivic, Akihiko Torii, and Torsten Sattler. D2-net: A trainable cnn for joint description and detection of local features. In _CVPR_, pages 8092–8101, 2019. 
*   Ebel et al. [2019] Patrick Ebel, Anastasiia Mishchuk, Kwang Moo Yi, Pascal Fua, and Eduard Trulls. Beyond cartesian representations for local descriptors. In _ICCV_, pages 253–262, 2019. 
*   Edstedt et al. [2023a] Johan Edstedt, Ioannis Athanasiadis, Mårten Wadenbäck, and Michael Felsberg. Dkm: Dense kernelized feature matching for geometry estimation. In _CVPR_, pages 17765–17775, 2023a. 
*   Edstedt et al. [2023b] Johan Edstedt, Qiyu Sun, Georg Bökman, Mårten Wadenbäck, and Michael Felsberg. Roma: Revisiting robust losses for dense feature matching. _arXiv preprint arXiv:2305.15404_, 2023b. 
*   Fan et al. [2021] Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, and Christoph Feichtenhofer. Multiscale vision transformers. _ICCV_, pages 6804–6815, 2021. 
*   Fischler and Bolles [1981] Martin A. Fischler and Robert C. Bolles. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. _Commun. ACM_, 24:381–395, 1981. 
*   Giang et al. [2023] Khang Truong Giang, Soohwan Song, and Sung-Guk Jo. Topicfm: Robust and interpretable topic-assisted feature matching. In _AAAI_, 2023. 
*   Hausler et al. [2021] Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford, and Tobias Fischer. Patch-netvlad: Multi-scale fusion of locally-global descriptors for place recognition. _CVPR_, pages 14136–14147, 2021. 
*   He et al. [2015] Kaiming He, X. Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. _CVPR_, pages 770–778, 2015. 
*   He et al. [2024] Xingyi He, Jiaming Sun, Yifan Wang, Sida Peng, Qixing Huang, Hujun Bao, and Xiaowei Zhou. Detector-free structure from motion. In _CVPR_, 2024. 
*   Katharopoulos et al. [2020] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franccois Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In _ICML_, 2020. 
*   Kitaev et al. [2020] Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. _ArXiv_, abs/2001.04451, 2020. 
*   Laguna et al. [2019] Axel Barroso Laguna, Edgar Riba, Daniel Ponsa, and Krystian Mikolajczyk. Key.net: Keypoint detection by handcrafted and learned cnn filters. _ICCV_, pages 5835–5843, 2019. 
*   Larsson and contributors [2020] Viktor Larsson and contributors. PoseLib - Minimal Solvers for Camera Pose Estimation, 2020. 
*   Lefaudeux et al. [2022] Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick Labatut, and Daniel Haziza. xformers: A modular and hackable transformer modelling library. [https://github.com/facebookresearch/xformers](https://github.com/facebookresearch/xformers), 2022. 
*   Li et al. [2020] Xinghui Li, Kai Han, Shuda Li, and Victor Prisacariu. Dual-resolution correspondence networks. In _NeurIPS_, 2020. 
*   Li and Snavely [2018] Zhengqi Li and Noah Snavely. Megadepth: Learning single-view depth prediction from internet photos. In _CVPR_, pages 2041–2050, 2018. 
*   Lindenberger et al. [2021] Philipp Lindenberger, Paul-Edouard Sarlin, Viktor Larsson, and Marc Pollefeys. Pixel-perfect structure-from-motion with featuremetric refinement. _ICCV_, pages 5967–5977, 2021. 
*   Lindenberger et al. [2023] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Pollefeys. LightGlue: Local Feature Matching at Light Speed. In _ICCV_, 2023. 
*   LoweDavid [2004] G LoweDavid. Distinctive image features from scale-invariant keypoints. _IJCV_, 2004. 
*   Luo et al. [2020] Zixin Luo, Lei Zhou, Xuyang Bai, Hongkai Chen, Jiahui Zhang, Yao Yao, Shiwei Li, Tian Fang, and Long Quan. Aslfeat: Learning local features of accurate shape and localization. In _CVPR_, pages 6589–6598, 2020. 
*   Mishchuk et al. [2017] Anastasiia Mishchuk, Dmytro Mishkin, Filip Radenovic, and Jiri Matas. Working hard to know your neighbor’s margins: Local descriptor learning loss. _NeurIPS_, 30, 2017. 
*   Mur-Artal and Tardós [2016] Raul Mur-Artal and Juan D. Tardós. Orb-slam2: An open-source slam system for monocular, stereo, and rgb-d cameras. _TR_, 33:1255–1262, 2016. 
*   Mur-Artal et al. [2015] Raul Mur-Artal, José M.M. Montiel, and Juan D. Tardós. Orb-slam: A versatile and accurate monocular slam system. _TR_, 31:1147–1163, 2015. 
*   Ni et al. [2023] Junjie Ni, Yijin Li, Zhaoyang Huang, Hongsheng Li, Hujun Bao, Zhaopeng Cui, and Guofeng Zhang. Pats: Patch area transportation with subdivision for local feature matching. _CVPR_, pages 17776–17786, 2023. 
*   Oquab et al. [2023] Maxime Oquab, Timothée Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2023. 
*   Revaud et al. [2019] Jérôme Revaud, César Roberto de Souza, M. Humenberger, and Philippe Weinzaepfel. R2d2: Reliable and repeatable detector and descriptor. In _NeurIPS_, 2019. 
*   Rocco et al. [2018] Ignacio Rocco, Mircea Cimpoi, Relja Arandjelović, Akihiko Torii, Tomas Pajdla, and Josef Sivic. Neighbourhood consensus networks. _NeurIPS_, 31, 2018. 
*   Rocco et al. [2020] Ignacio Rocco, Relja Arandjelović, and Josef Sivic. Efficient neighbourhood consensus networks via submanifold sparse convolutions. In _ECCV_, pages 605–621. Springer, 2020. 
*   Rosten and Drummond [2006] Edward Rosten and Tom Drummond. Machine learning for high-speed corner detection. In _ECCV_, pages 430–443. Springer, 2006. 
*   Sarlin et al. [2019] Paul-Edouard Sarlin, César Cadena, Roland Y. Siegwart, and Marcin Dymczyk. From coarse to fine: Robust hierarchical localization at large scale. _CVPR_, pages 12708–12717, 2019. 
*   Sarlin et al. [2020] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. SuperGlue: Learning feature matching with graph neural networks. In _CVPR_, 2020. 
*   Sarlin et al. [2021] Paul-Edouard Sarlin, Ajaykumar Unagar, Måns Larsson, Hugo Germain, Carl Toft, Victor Larsson, Marc Pollefeys, Vincent Lepetit, Lars Hammarstrand, Fredrik Kahl, and Torsten Sattler. Back to the Feature: Learning Robust Camera Localization from Pixels to Pose. In _CVPR_, 2021. 
*   Sattler et al. [2018] Torsten Sattler, Will Maddern, Carl Toft, Akihiko Torii, Lars Hammarstrand, Erik Stenborg, Daniel Safari, Masatoshi Okutomi, Marc Pollefeys, Josef Sivic, et al. Benchmarking 6dof outdoor visual localization in changing conditions. In _CVPR_, pages 8601–8610, 2018. 
*   Savinov et al. [2017] Nikolay Savinov, Akihito Seki, Lubor Ladicky, Torsten Sattler, and Marc Pollefeys. Quad-networks: unsupervised learning to rank for interest point detection. In _CVPR_, pages 1822–1830, 2017. 
*   Schönberger and Frahm [2016] Johannes L. Schönberger and Jan-Michael Frahm. Structure-from-motion revisited. In _CVPR_, 2016. 
*   Shi et al. [2022] Yan Shi, Jun-Xiong Cai, Yoli Shavit, Tai-Jiang Mu, Wensen Feng, and Kai Zhang. Clustergnn: Cluster-based coarse-to-fine graph neural network for efficient feature matching. In _CVPR_, pages 12517–12526, 2022. 
*   Su et al. [2021] Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. _ArXiv_, abs/2104.09864, 2021. 
*   Sun et al. [2021] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou. Loftr: Detector-free local feature matching with transformers. In _CVPR_, pages 8922–8931, 2021. 
*   Taira et al. [2018] Hajime Taira, Masatoshi Okutomi, Torsten Sattler, Mircea Cimpoi, Marc Pollefeys, Josef Sivic, Tomas Pajdla, and Akihiko Torii. Inloc: Indoor visual localization with dense matching and view synthesis. In _CVPR_, pages 7199–7209, 2018. 
*   Tang et al. [2022] Shitao Tang, Jiahui Zhang, Siyu Zhu, and Ping Tan. Quadtree attention for vision transformers. _ICLR_, 2022. 
*   Tian et al. [2017] Yurun Tian, Bin Fan, and Fuchao Wu. L2-net: Deep learning of discriminative patch descriptor in euclidean space. In _CVPR_, pages 661–669, 2017. 
*   Tian et al. [2019] Yurun Tian, Xin Yu, Bin Fan, Fuchao Wu, Huub Heijnen, and Vassileios Balntas. Sosnet: Second order similarity regularization for local descriptor learning. In _CVPR_, pages 11016–11025, 2019. 
*   Tian et al. [2020] Yurun Tian, Vassileios Balntas, Tony Ng, Axel Barroso-Laguna, Yiannis Demiris, and Krystian Mikolajczyk. D2d: Keypoint extraction with describe to detect approach. In _ACCV_, 2020. 
*   Truong et al. [2021] Prune Truong, Martin Danelljan, Luc Van Gool, and Radu Timofte. Learning accurate dense correspondences and when to trust them. In _CVPR_, pages 5714–5724, 2021. 
*   Tyszkiewicz et al. [2020] Michał Tyszkiewicz, Pascal Fua, and Eduard Trulls. Disk: Learning local features with policy gradient. _NeurIPS_, 2020. 
*   Vaswani et al. [2017] Ashish Vaswani, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _NeurIPS_, 2017. 
*   Wang et al. [2022] Qing Wang, Jiaming Zhang, Kailun Yang, Kunyu Peng, and Rainer Stiefelhagen. Matchformer: Interleaving attention in transformers for feature matching. In _ACCV_, 2022. 
*   Wang et al. [2020] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. _ArXiv_, abs/2006.04768, 2020. 
*   Yu et al. [2022] Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen Zhou, Xinchao Wang, Jiashi Feng, and Shuicheng Yan. Metaformer is actually what you need for vision. _CVPR_, pages 10809–10819, 2022. 

Supplementary Material
----------------------

Appendix A Insight and Discussion about Aggregated Attention Module
-------------------------------------------------------------------

Some previous works explored using pooling in ViT but are with different design choices from our method due to different tasks. PoolFormer[[61](https://arxiv.org/html/2403.04765v2#bib.bib61)] replaces the multi-head attention with pooling, which cannot be used for cross-attention in matching that two images are not pixel-aligned. MVit[[16](https://arxiv.org/html/2403.04765v2#bib.bib16)] uses pooling to reduce tokens like ours, but they cannot get high-res features that are required for matching.

Differently, we propose to first conduct attention on aggregated features and then _upsample_ before feed-forward network(FFN) for later fusion with input feature, as shown in Fig.3. This aggregate-and-upsample block can minimize information loss in aggregation and efficiently get high-res informative features, where conducting upsampling before fusion is crucial to fuse smoothly interpolated messages with a detailed feature map. Ablation is in Tab.[10](https://arxiv.org/html/2403.04765v2#A3.T10 "Table 10 ‣ Additional ablation studies on the ScanNet dataset ‣ C.1 More Ablation Studies ‣ Appendix C More Experiments Results ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed")(8).

Moreover, We perform Conv on Q value instead of pooling because salient tokens should _not_ represent neighbors to query attention. The transformer is crucial for enhancing non-salient features for matching. Pooling on Q causes the attention of texture-less areas dominated by neighboring salient tokens, reducing the performance as ablated in Tab.[10](https://arxiv.org/html/2403.04765v2#A3.T10 "Table 10 ‣ Additional ablation studies on the ScanNet dataset ‣ C.1 More Ablation Studies ‣ Appendix C More Experiments Results ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed")(6,7).

Appendix B Implementation Details
---------------------------------

### B.1 Local Feature Extraction

RepVGG[[11](https://arxiv.org/html/2403.04765v2#bib.bib11)] blocks are used to build a four-stage feature backbone. We use a width of 64 and a stride of 1 for the first stage and widths of [64, 128, 256] and strides of 2 for the subsequent three stages. Each stage is composed of [1, 2, 4, 14] RepVGG blocks and ReLU activations, respectively. The output of the last stage in 1/8 1 8\nicefrac{{1}}{{8}}/ start_ARG 1 end_ARG start_ARG 8 end_ARG image resolution is used for efficient local feature transformer modules to get attended coarse feature maps. The second and third stages’ feature maps are in 1/2 1 2\nicefrac{{1}}{{2}}/ start_ARG 1 end_ARG start_ARG 2 end_ARG and 1/4 1 4\nicefrac{{1}}{{4}}/ start_ARG 1 end_ARG start_ARG 4 end_ARG image resolutions, respectively, which are used for fusing with transformed coarse features for fine features.

### B.2 Position Encoding

We use the 2D extension of Rotary position encoding[[49](https://arxiv.org/html/2403.04765v2#bib.bib49)] to encode the relative position between coarse features in self-attention modules. Given the projected features q 𝑞 q italic_q and k 𝑘 k italic_k, the attention score between two features q i subscript 𝑞 𝑖 q_{i}italic_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT and k j subscript 𝑘 𝑗 k_{j}italic_k start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT is computed as:

a i⁢j=q i T⁢R⁢(x j−x i,y j−y i)⁢k j,subscript 𝑎 𝑖 𝑗 superscript subscript 𝑞 𝑖 𝑇 𝑅 subscript 𝑥 𝑗 subscript 𝑥 𝑖 subscript 𝑦 𝑗 subscript 𝑦 𝑖 subscript 𝑘 𝑗 a_{ij}=q_{i}^{T}R(x_{j}-x_{i},y_{j}-y_{i})k_{j}\enspace,italic_a start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT = italic_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT italic_R ( italic_x start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT - italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_y start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT - italic_y start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) italic_k start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ,(5)

where x i,y i,x j,y j subscript 𝑥 𝑖 subscript 𝑦 𝑖 subscript 𝑥 𝑗 subscript 𝑦 𝑗 x_{i},y_{i},x_{j},y_{j}italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_y start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_x start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT , italic_y start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT are the coordinates of q i subscript 𝑞 𝑖 q_{i}italic_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT and k j subscript 𝑘 𝑗 k_{j}italic_k start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT, R 𝑅 R italic_R is a block diagonal matrix:

R⁢(Δ⁢x,Δ⁢y)=(R 1⁢(Δ⁢x,Δ⁢y)R 2⁢(Δ⁢x,Δ⁢y)⋱R d/4⁢(Δ⁢x,Δ⁢y)),𝑅 Δ 𝑥 Δ 𝑦 matrix subscript 𝑅 1 Δ 𝑥 Δ 𝑦 missing-subexpression missing-subexpression missing-subexpression missing-subexpression subscript 𝑅 2 Δ 𝑥 Δ 𝑦 missing-subexpression missing-subexpression missing-subexpression missing-subexpression⋱missing-subexpression missing-subexpression missing-subexpression missing-subexpression subscript 𝑅 𝑑 4 Δ 𝑥 Δ 𝑦 R(\Delta{x},\Delta{y})=\begin{pmatrix}R_{1}(\Delta{x},\Delta{y})&&&\\ &R_{2}(\Delta{x},\Delta{y})&&\\ &&\ddots&\\ &&&R_{d/4}(\Delta{x},\Delta{y})\end{pmatrix}\enspace,italic_R ( roman_Δ italic_x , roman_Δ italic_y ) = ( start_ARG start_ROW start_CELL italic_R start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ( roman_Δ italic_x , roman_Δ italic_y ) end_CELL start_CELL end_CELL start_CELL end_CELL start_CELL end_CELL end_ROW start_ROW start_CELL end_CELL start_CELL italic_R start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT ( roman_Δ italic_x , roman_Δ italic_y ) end_CELL start_CELL end_CELL start_CELL end_CELL end_ROW start_ROW start_CELL end_CELL start_CELL end_CELL start_CELL ⋱ end_CELL start_CELL end_CELL end_ROW start_ROW start_CELL end_CELL start_CELL end_CELL start_CELL end_CELL start_CELL italic_R start_POSTSUBSCRIPT italic_d / 4 end_POSTSUBSCRIPT ( roman_Δ italic_x , roman_Δ italic_y ) end_CELL end_ROW end_ARG ) ,(6)

R k⁢(Δ⁢x,Δ⁢y)=(cos⁡(θ k⁢Δ⁢x)−sin⁡(θ k⁢Δ⁢x)0 0 sin⁡(θ k⁢Δ⁢x)cos⁡(θ k⁢Δ⁢x)0 0 0 0 cos⁡(θ k⁢Δ⁢y)−sin⁡(θ k⁢Δ⁢y)0 0 sin⁡(θ k⁢Δ⁢y)cos⁡(θ k⁢Δ⁢y)),subscript 𝑅 𝑘 Δ 𝑥 Δ 𝑦 matrix subscript 𝜃 𝑘 Δ 𝑥 subscript 𝜃 𝑘 Δ 𝑥 0 0 subscript 𝜃 𝑘 Δ 𝑥 subscript 𝜃 𝑘 Δ 𝑥 0 0 0 0 subscript 𝜃 𝑘 Δ 𝑦 subscript 𝜃 𝑘 Δ 𝑦 0 0 subscript 𝜃 𝑘 Δ 𝑦 subscript 𝜃 𝑘 Δ 𝑦 R_{k}(\Delta{x},\Delta{y})=\begin{pmatrix}\cos(\theta_{k}\Delta{x})&-\sin(% \theta_{k}\Delta{x})&0&0\\ \sin(\theta_{k}\Delta{x})&\cos(\theta_{k}\Delta{x})&0&0\\ 0&0&\cos(\theta_{k}\Delta{y})&-\sin(\theta_{k}\Delta{y})\\ 0&0&\sin(\theta_{k}\Delta{y})&\cos(\theta_{k}\Delta{y})\end{pmatrix}\enspace,italic_R start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ( roman_Δ italic_x , roman_Δ italic_y ) = ( start_ARG start_ROW start_CELL roman_cos ( italic_θ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT roman_Δ italic_x ) end_CELL start_CELL - roman_sin ( italic_θ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT roman_Δ italic_x ) end_CELL start_CELL 0 end_CELL start_CELL 0 end_CELL end_ROW start_ROW start_CELL roman_sin ( italic_θ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT roman_Δ italic_x ) end_CELL start_CELL roman_cos ( italic_θ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT roman_Δ italic_x ) end_CELL start_CELL 0 end_CELL start_CELL 0 end_CELL end_ROW start_ROW start_CELL 0 end_CELL start_CELL 0 end_CELL start_CELL roman_cos ( italic_θ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT roman_Δ italic_y ) end_CELL start_CELL - roman_sin ( italic_θ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT roman_Δ italic_y ) end_CELL end_ROW start_ROW start_CELL 0 end_CELL start_CELL 0 end_CELL start_CELL roman_sin ( italic_θ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT roman_Δ italic_y ) end_CELL start_CELL roman_cos ( italic_θ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT roman_Δ italic_y ) end_CELL end_ROW end_ARG ) ,(7)

where θ k=1 10000 4⁢k/d,k∈[1,2,…,d/4]formulae-sequence subscript 𝜃 𝑘 1 superscript 10000 4 𝑘 𝑑 𝑘 1 2…𝑑 4\theta_{k}=\frac{1}{10000^{4k/d}},\enspace k\in[1,2,...,d/4]italic_θ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT = divide start_ARG 1 end_ARG start_ARG 10000 start_POSTSUPERSCRIPT 4 italic_k / italic_d end_POSTSUPERSCRIPT end_ARG , italic_k ∈ [ 1 , 2 , … , italic_d / 4 ] encode the index of feature channels.

Compared to the absolute position encoding used in previous methods[[50](https://arxiv.org/html/2403.04765v2#bib.bib50), [59](https://arxiv.org/html/2403.04765v2#bib.bib59), [7](https://arxiv.org/html/2403.04765v2#bib.bib7), [52](https://arxiv.org/html/2403.04765v2#bib.bib52), [18](https://arxiv.org/html/2403.04765v2#bib.bib18), [36](https://arxiv.org/html/2403.04765v2#bib.bib36)], we utilize 2D RoPE to allow the model to focus more on the interaction between features rather than their specific locations, which benefits capturing the context of local features. Moreover, relative position encoding is more robust to transformations like rotation, translation, and scaling, which is important for matching local features in different views.

Appendix C More Experiments Results
-----------------------------------

### C.1 More Ablation Studies

In this part, we conduct more ablation studies on the MegaDepth and ScanNet dataset to validate the design choices of our proposed modules.

#### Position Encoding

We compare the performance of our 2D RoPE with the sinusoidal position encoding[[58](https://arxiv.org/html/2403.04765v2#bib.bib58)] in Tab.[6](https://arxiv.org/html/2403.04765v2#A3.T6 "Table 6 ‣ Position Encoding ‣ C.1 More Ablation Studies ‣ Appendix C More Experiments Results ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed"). The results show that using 2D RoPE can achieve better performance than sinusoidal position encoding.

![Image 9: Refer to caption](https://arxiv.org/html/2403.04765v2/x12.png)

Figure 5: Qualitative Results. Our method is compared with the sparse matching pipeline SuperPoint[[10](https://arxiv.org/html/2403.04765v2#bib.bib10)]+LightGlue[[30](https://arxiv.org/html/2403.04765v2#bib.bib30)], semi-dense matcher AspanFormer[[7](https://arxiv.org/html/2403.04765v2#bib.bib7)]. The red color indicates epipolar error beyond 5×10−4 5 superscript 10 4 5\times 10^{-4}5 × 10 start_POSTSUPERSCRIPT - 4 end_POSTSUPERSCRIPT on ScanNet and 1×10−4 1 superscript 10 4 1\times 10^{-4}1 × 10 start_POSTSUPERSCRIPT - 4 end_POSTSUPERSCRIPT on MegaDepth (in the normalized image coordinates). Since no ground-truth pose is available on InLoc dataset, we color the match with predicted confidence. Red indicates higher confidence and blue for the opposite. 

Position Encoding Pose Estimation AUC Time(ms)
@5°@10°@20°
RoPE 56.4 72.2 83.5 139.2
sinusoidal 55.5 71.5 83.1 137.5

Table 6:  Impact of position encoding on the MegaDepth dataset, where averaged running times for an image pair with high-resolution 1200×1200 1200 1200 1200\times 1200 1200 × 1200 are reported. 

#### Aggregation Range

We show the performance of our method with different aggregation range s 𝑠 s italic_s in Tab.[7](https://arxiv.org/html/2403.04765v2#A3.T7 "Table 7 ‣ Aggregation Range ‣ C.1 More Ablation Studies ‣ Appendix C More Experiments Results ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed"). In our aggregated attention module, we use a 4×4 4 4 4\times 4 4 × 4 aggregation range to reduce token size before performing attention. Using a smaller aggregation range leads to more tokens, with slight performance changes but significantly slower matching speed. This validates the effectiveness of our parameter choice in the aggregation attention module.

aggregation range Pose Estimation AUC Time(ms)
@5°@10°@20°
s=4 𝑠 4 s=4 italic_s = 4 56.4 72.2 83.5 139.2
s=2 𝑠 2 s=2 italic_s = 2 56.2 72.2 83.6 271.1

Table 7:  Impact of aggregation range on the MegaDepth dataset, where averaged running times for an image pair with high-resolution 1200×1200 1200 1200 1200\times 1200 1200 × 1200 are reported. 

#### Image Resolution

We test the performance of our method with different image resolutions to show the performance and efficiency changes. Results are shown in Tab.[8](https://arxiv.org/html/2403.04765v2#A3.T8 "Table 8 ‣ Image Resolution ‣ C.1 More Ablation Studies ‣ Appendix C More Experiments Results ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed"). Compared with the default resolution 1184×1184 1184 1184 1184\times 1184 1184 × 1184 used in the MegaDepth evaluation, using a larger image size leads to noticeable accuracy improvement with a slower matching speed. Our method can still achieve competitive performance using low-resolution 640×640 640 640 640\times 640 640 × 640 images with the fastest speed. Therefore, our method is pretty robust in image resolution choices for flexible real-world applications.

Resolution Pose Estimation AUC Time(ms)
@5°@10°@20°
640×640 640 640 640\times 640 640 × 640 51.0 67.4 79.8 41.7
800×800 800 800 800\times 800 800 × 800 53.4 70.0 81.9 58.2
960×960 960 960 960\times 960 960 × 960 54.7 70.7 82.4 81.8
1184×1184 1184 1184 1184\times 1184 1184 × 1184 56.4 72.2 83.5 139.2
1408×1408 1408 1408 1408\times 1408 1408 × 1408 56.2 73.1 83.4 223.9

Table 8:  Impact of test image resolution on the MegaDepth dataset. 

#### Linear Attention After Aggregation

Using linear attention in our aggregated attention module introduces minor efficiency gain on high-resolution but with accuracy dropping as shown in Tab.[9](https://arxiv.org/html/2403.04765v2#A3.T9 "Table 9 ‣ Linear Attention After Aggregation ‣ C.1 More Ablation Studies ‣ Appendix C More Experiments Results ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed").

Method MegaDepth Dataset Time(ms)ScanNet Dataset Time(ms)
AUC@5°°\degree°AUC@10°°\degree°AUC@20°°\degree°AUC@5°°\degree°AUC@10°°\degree°AUC@20°°\degree°
Full 56.4 72.2 83.5 139.1 19.2 37.0 53.6 34.4
Linear 54.1 70.3 82.1 132.7 16.8 33.2 49.0 36.9

Table 9:  Impact of linear attention after aggregation on the MegaDepth and ScanNet dataset, where the resolution are 1200×1200 1200 1200 1200\times 1200 1200 × 1200 and 640×480 640 480 640\times 480 640 × 480, respectively. 

#### Additional ablation studies on the ScanNet dataset

We further repeat the ablation studies in the main paper and conduct additional ablation studies on the ScanNet dataset to validate the design choices of our proposed modules. Results are shown in Tab.[10](https://arxiv.org/html/2403.04765v2#A3.T10 "Table 10 ‣ Additional ablation studies on the ScanNet dataset ‣ C.1 More Ablation Studies ‣ Appendix C More Experiments Results ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed").

Method Pose Estimation AUC Time(ms)
@5°@10°@20°
Ours Full 19.2 37.0 53.6 34.4
1) Ours Optimal(w/o dual-softmax)17.4 34.4 51.2 27.0
2) Replace Agg. Attention to LoFTR’s Trans.17.1 33.2 49.4 41.3
3) Replace two-stage refine. to LoFTR’s refine.18.1 35.8 52.4 31.8
4) No second-stage refinement 18.8 36.7 53.4 32.2
5) Replace RepVGG with ResNet 18.6 36.3 52.8 38.1
6) Both Conv in Agg. Attention 18.6 35.8 52.5 34.4
7) Both Pool in Agg. Attention 18.3 35.2 51.7 34.1
8) Upsample after FFN 17.3 34.6 51.4 32.6

Table 10:  The components of our method are ablated on the ScanNet dataset again for a comprehensive understanding of our method, where averaged running times for an image pair with resolution 640×480 640 480 640\times 480 640 × 480 are reported. 

### C.2 More Qualitative Results

More qualitative results on the ScanNet dataset, InLoc dataset, and MegaDepth dataset are shown in Fig.[5](https://arxiv.org/html/2403.04765v2#A3.F5 "Figure 5 ‣ Position Encoding ‣ C.1 More Ablation Studies ‣ Appendix C More Experiments Results ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed").

### C.3 Additional Results on other RANSAC setting

LightGlue uses a RANSAC setting different from other baseline papers[[43](https://arxiv.org/html/2403.04765v2#bib.bib43), [50](https://arxiv.org/html/2403.04765v2#bib.bib50), [52](https://arxiv.org/html/2403.04765v2#bib.bib52), [59](https://arxiv.org/html/2403.04765v2#bib.bib59), [18](https://arxiv.org/html/2403.04765v2#bib.bib18), [7](https://arxiv.org/html/2403.04765v2#bib.bib7), [14](https://arxiv.org/html/2403.04765v2#bib.bib14), [15](https://arxiv.org/html/2403.04765v2#bib.bib15)] in relative pose estimation evaluations on MegaDepth. We further conduct experiments following LightGlue’s setting (OpenCV RANSAC[[17](https://arxiv.org/html/2403.04765v2#bib.bib17)] and LO-RANSAC[[25](https://arxiv.org/html/2403.04765v2#bib.bib25)] with carefully tuned RANSAC inlier thresholds), as shown in Tab.[11](https://arxiv.org/html/2403.04765v2#A3.T11 "Table 11 ‣ C.3 Additional Results on other RANSAC setting ‣ Appendix C More Experiments Results ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed"). Using the naive RANSAC method, the performance gap between ours and LightGlue becomes larger after RANSAC threshold tuning(compared with our untuned results in Tab.[1](https://arxiv.org/html/2403.04765v2#S4.T1 "Table 1 ‣ 4 Experiments ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed")). This demonstrates that our method can achieve significantly better accuracy without depending on the sophisticated modern RANSAC method, thereby revealing its superior match quality. Using the stronger outlier filter LO-RANSAC, the accuracy of all methods is boosted and our method consistently achieves better performance than LightGlue, especially on the AUC 5⁢°5°5\degree 5 ° metric.

Method RANSAC LO-RANSAC
AUC@5°°\degree°AUC@10°°\degree°AUC@20°°\degree°AUC@5°°\degree°AUC@10°°\degree°AUC@20°°\degree°
LightGlue 49.9 67.0 80.1 66.8 79.3 87.9
AspanFormer 58.3 73.3 84.2 69.4 81.1 88.9
Ours 58.4 73.4 84.2 69.5 80.9 88.8

Table 11: Results of Relative Pose Estimation on MegaDepth Dataset following LightGlue’s setting. The AUC of pose error at different thresholds is presented. 

Appendix D Details About Timing
-------------------------------

The running times evaluated in the paper are averaged over all pairs in the test dataset with a warm-up of 50 pairs for accurate measurement. All the methods are tested on a single NVIDIA RTX 3090 GPU with 14 cores of Intel Xeon Gold 6330 CPU.

We further report each part running time of our method in Tab.[12](https://arxiv.org/html/2403.04765v2#A4.T12 "Table 12 ‣ Appendix D Details About Timing ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed"), where both full and optimized models are shown. We noticed that skipping the dual-softmax of Coarse Matching in the optimized model can significantly reduce the running time. What’s more, with the benefit of Mixed-Precision, the running time of feature extraction can be further reduced.

Process Time(ms)
Full Optimized
Total 40.1 27.0
Feature Backbone 9.1 5.8
Coarse Feature Transformation 11.7 12.9
Coarse Matching 8.3 1.7
Fine Feature Fusion 8.0 4.8
Two-Stage Refinement 3.0 2.0

Table 12:  Time cost for an image pair of 640×480 640 480 640\times 480 640 × 480 on the ScanNet dataset. The optimized model uses Mixed-Precision numerical accuracy and drops the dual-softmax operator in the coarse matching phase. 

Method Time(ms)#Matches
Matching RANSAC
SP + SG 43.6 0.53 487
DRC-Net 143.9 2.78 1019
LoFTR 76.2 1.59 995
Ours 45.9/38.6 1.39 997

Table 13:  Running times of different methods on HPatches dataset. All images are resized so that their short edge equals 480 pixels following SuperGlue[[43](https://arxiv.org/html/2403.04765v2#bib.bib43)] and LoFTR[[50](https://arxiv.org/html/2403.04765v2#bib.bib50)]. For Ours, the running times of the model using FP32/Mixed-Precision numerical precisions are shown. 

Method Time(ms)
Aachen InLoc
SP+SG 55.9 83.3
LoFTR 83.2 147.6
TopicFM 66.0 119.6
PATS 315.8 1148.0
AspanFormer 95.4 164.5
Ours 40.6/25.5 82.0/46.3

Table 14:  Running times of different methods on Aachen and InLoc dataset. To measure the running time, we sample 818 and 356 pairs of images from the NetVLAD[[2](https://arxiv.org/html/2403.04765v2#bib.bib2)]’s retrieval results for Aachen and InLoc, respectively. All images are resized so that their long edge equals 1024 pixels following HLoc[[42](https://arxiv.org/html/2403.04765v2#bib.bib42)]. For Ours, the running times of the model using FP32/Mixed-Precision numerical precisions are shown. 

The latency on 3 3 3 3 datasets included in [Tabs.2](https://arxiv.org/html/2403.04765v2#S4.T2 "Table 2 ‣ 4.2 Relative Pose Estimation ‣ 4 Experiments ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed"), [3](https://arxiv.org/html/2403.04765v2#S4.T3 "Table 3 ‣ 4.3 Homography Estimation ‣ 4 Experiments ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed") and[4](https://arxiv.org/html/2403.04765v2#S4.T4 "Table 4 ‣ 4.3 Homography Estimation ‣ 4 Experiments ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed") are shown in Tab.[13](https://arxiv.org/html/2403.04765v2#A4.T13 "Table 13 ‣ Appendix D Details About Timing ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed") and Tab.[14](https://arxiv.org/html/2403.04765v2#A4.T14 "Table 14 ‣ Appendix D Details About Timing ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed"), where conclusion and speed rankings are the same as indicated in Tab.[1](https://arxiv.org/html/2403.04765v2#S4.T1 "Table 1 ‣ 4 Experiments ‣ Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed"). We further show RANSAC time on HPatches dataset. Both matching and RANSAC latency of our method are smaller than LoFTR with a similar number of matches.

Appendix E Limitations and Future Works
---------------------------------------

We find that our method may fail when strong repetitive structures exist, such as matching an image pair that depicts different scenes containing the same chair. We think this may be due to the current model focusing more on local features for accurate matching, where global semantic context is lacking. Therefore, the mechanism of high-level contexts can be added to the model for performance improvement on ambiguous scenes. Moreover, we believe the efficiency of our method can be further improved by adopting the early stop strategy of LightGlue[[30](https://arxiv.org/html/2403.04765v2#bib.bib30)] since the contribution of the proposed efficient aggregation module is orthogonal to it.

