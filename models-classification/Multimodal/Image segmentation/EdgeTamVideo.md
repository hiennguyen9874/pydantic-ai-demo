Title: EdgeTAM: On-Device Track Anything Model

URL Source: https://arxiv.org/html/2501.07256

Published Time: Tue, 14 Jan 2025 02:15:02 GMT

Markdown Content:
Chong Zhou 1,2, Chenchen Zhu 1, Yunyang Xiong 1, Saksham Suri 1, Fanyi Xiao 1, Lemeng Wu 1

Raghuraman Krishnamoorthi 1, Bo Dai 3, Chen Change Loy 2, Vikas Chandra 1, Bilge Soran 1

1 Meta Reality Labs, 2 Nanyang Technological University, 3 Shanghai AI Laboratory

###### Abstract

On top of Segment Anything Model (SAM), SAM 2 further extends its capability from image to video inputs through a memory bank mechanism and obtains a remarkable performance compared with previous methods, making it a foundation model for video segmentation task. In this paper, we aim at making SAM 2 much more efficient so that it even runs on mobile devices while maintaining a comparable performance. Despite several works optimizing SAM for better efficiency, we find they are not sufficient for SAM 2 because they all focus on compressing the image encoder, while our benchmark shows that the newly introduced memory attention blocks are also the latency bottleneck. Given this observation, we propose EdgeTAM, which leverages a novel 2D Spatial Perceiver to reduce the computational cost. In particular, the proposed 2D Spatial Perceiver encodes the densely stored frame-level memories with a lightweight Transformer that contains a fixed set of learnable queries. Given that video segmentation is a dense prediction task, we find preserving the spatial structure of the memories is essential so that the queries are split into global-level and patch-level groups. We also propose a distillation pipeline that further improves the performance without inference overhead. As a result, EdgeTAM achieves 87.7, 70.0, 72.3, and 71.7 𝒥 𝒥\mathcal{J}caligraphic_J&ℱ ℱ\mathcal{F}caligraphic_F on DAVIS 2017, MOSE, SA-V val, and SA-V test, while running at 16 FPS on iPhone 15 Pro Max.

1 Introduction
--------------

![Image 1: Refer to caption](https://arxiv.org/html/2501.07256v1/extracted/6127871/figure/graphics/teaser-v4.png)

Figure 1: Speed-performance trade-offs on iPhone 15 Pro Max and NVIDIA A100. EdgeTAM is significantly faster than SAM 2 on edge devices and compare to other VOS methods, it is also more accurate on the challenging SA-V val dataset. Note that, EdgeTAM can run at 16 FPS on iPhone 15 Pro Max.

Segment Anything Model (SAM) [[31](https://arxiv.org/html/2501.07256v1#bib.bib31)] is the first foundation model for promptable image segmentation. Various studies show its magnificent capabilities on zero-shot generalization and transfer learning [[39](https://arxiv.org/html/2501.07256v1#bib.bib39), [8](https://arxiv.org/html/2501.07256v1#bib.bib8), [70](https://arxiv.org/html/2501.07256v1#bib.bib70), [55](https://arxiv.org/html/2501.07256v1#bib.bib55)]. On top of SAM, recently, SAM 2 [[48](https://arxiv.org/html/2501.07256v1#bib.bib48)] extends the original SAM to handle both image and video inputs, with a memory bank mechanism, and is trained with a new large-scale multi-grained video tracking dataset (SA-V).

![Image 2: Refer to caption](https://arxiv.org/html/2501.07256v1/extracted/6127871/figure/graphics/benchmark-v1.png)

Figure 2: Single frame latency (ms) on iPhone. In (a), we show that only replacing image encoder with more compact backbones is not enough for further speed-up since decoder is also a bottleneck. In (b), through reducing the number of memory attention blocks and removing certain modules, we find that the cross attention (CA) is the root cause.

Despite achieving an astonishing performance compared to previous video object segmentation (VOS) models and allowing more diverse user prompts, SAM 2, as a server-side foundation model, is not efficient for on-device inference. For instance, the smallest SAM 2 variant runs at only around 1 FPS on an iPhone 15 Pro Max 1 1 1 We convert to CoreML model with coremltools [[1](https://arxiv.org/html/2501.07256v1#bib.bib1)] and benchmark with CPU and NPU. Throughout the paper, we interchangeably use iPhone and iPhone 15 Pro Max for simplicity.. Furthermore, existing methods [[83](https://arxiv.org/html/2501.07256v1#bib.bib83), [71](https://arxiv.org/html/2501.07256v1#bib.bib71), [86](https://arxiv.org/html/2501.07256v1#bib.bib86)] that optimize SAM for better efficiency only consider squeezing its image encoder since the mask decoder is extremely lightweight. But as shown in Fig.[2](https://arxiv.org/html/2501.07256v1#S1.F2 "Figure 2 ‣ 1 Introduction ‣ EdgeTAM: On-Device Track Anything Model"), this is not sufficient for SAM 2 because even when the image encoder is replaced with much more compact visual backbones, such as ViT-Tiny [[58](https://arxiv.org/html/2501.07256v1#bib.bib58)] and RepViT [[64](https://arxiv.org/html/2501.07256v1#bib.bib64)], the latency does not improve by much due to the computationally demanding memory attention blocks that are newly introduced in SAM 2. Specifically, SAM 2 encodes past frames with a memory encoder, and these frame-level memories together with object-level pointers (obtained from the mask decoder) serve as the memory bank. These are then fused with the features of current frame via memory attention blocks. As these memories are densely encoded, this leads to a huge matrix multiplication during the cross-attention between current frame features and memory features. Therefore, despite containing relatively fewer parameters than the image encoder, the computational complexity of the memory attention is not affordable for on-device inference. The hypothesis is further proved by Fig.[2](https://arxiv.org/html/2501.07256v1#S1.F2 "Figure 2 ‣ 1 Introduction ‣ EdgeTAM: On-Device Track Anything Model"), where reducing the number of memory attention blocks almost linearly cuts down the overall decoding latency and within each memory attention block, removing the cross attention gives the most significant speed-up.

To make such a video-based tracking model run on device, in EdgeTAM, we look at exploiting the redundancy in videos. To do this in practice, we propose to compress the raw frame-level memories before performing memory attention. We start with naïve spatial pooling and observe a significant performance degradation, especially when using low-capacity backbones. To mitigate this issue, we turn to learning-based compressors such as Perceiver[[29](https://arxiv.org/html/2501.07256v1#bib.bib29), [30](https://arxiv.org/html/2501.07256v1#bib.bib30)], which summarizes the dense feature map with a small fixed set of learned queries. However, naïvely incorporating a Perceiver also leads to a severe drop in performance. We hypothesize that as a dense prediction task, the video segmentation requires preserving the spatial structure of the memory bank, which a naïve Perceiver discards.

Given these observations, we propose a novel lightweight module that compresses frame-level memory feature maps while preserving the 2D spatial structure, named 2D Spatial Perceiver. Specifically, we split the learnable queries into two groups, where one group functions similarly to the original Perceiver, where each query performs global attention on the input features and outputs a single vector as the frame-level summarization. In the other group, the queries have 2D priors, _i.e._, each query is only responsible for compressing a non-overlapping local patch, thus the output maintains the spatial structure while reducing the total number of tokens. As a plug-in module, 2D Spatial Perceiver can be integrated with any variants of SAM 2 and speed up the memory attention by 8×8\times 8 × with comparable performance. For instance, when using RepViT-M1 [[64](https://arxiv.org/html/2501.07256v1#bib.bib64)] as the backbone and two memory attention blocks, leveraging the 2D Spatial Perceiver yields 16 FPS on iPhone, which is 6.4×6.4\times 6.4 × faster than the baseline and even surpasses it on the challenging SA-V val set [[48](https://arxiv.org/html/2501.07256v1#bib.bib48)] by 0.9 𝒥 𝒥\mathcal{J}caligraphic_J&ℱ ℱ\mathcal{F}caligraphic_F.

In addition to the architecture improvement, we further propose a distillation pipeline that transfers the knowledge of the powerful teacher SAM 2 to our student model, which improves the accuracy at no cost of inference overhead. Specifically, the training procedure of SAM 2 has two stages, where firstly the model is trained with the promptable image segmentation task on SA-1B[[31](https://arxiv.org/html/2501.07256v1#bib.bib31)] with memory-related module detached, then in the second stage, it is trained with all modules included for the promptable video segmentation task on both SA-1B and SA-V [[48](https://arxiv.org/html/2501.07256v1#bib.bib48)] datasets. We find that in both stages, aligning the features from image encoders of the original SAM 2 and our efficient variant benefits the performance. Besides, we further align the feature output from the memory attention between the teacher SAM 2 and our student model in the second stage so that in addition to the image encoder, memory-related modules can also receive supervision signals from the SAM 2 teacher. As a result, with the proposed distillation pipeline, we improve the 𝒥 𝒥\mathcal{J}caligraphic_J&ℱ ℱ\mathcal{F}caligraphic_F on SA-V val and test by 1.3 and 3.3, respectively.

Putting together, we propose EdgeTAM (Track Anything Model for Edge devices), that adopts a 2D Spatial Perceiver for efficiency and knowledge distillation for accuracy. Our contributions can be summarized in the following:

*   •Through comprehensive benchmark, we reveal that the latency bottleneck lies in the memory attention module. 
*   •Given the latency analysis, we propose a 2D Spatial Perceiver that significantly cuts down the memory attention computational cost with comparable performance, which can be integrated with any SAM 2 variants. 
*   •We experiment with a distillation pipeline that performs feature-wise alignment with the original SAM 2 in both the image and video segmentation stages and observe performance improvements without any additional cost during inference. 
*   •The resulting EdgeTAM can run at 16 FPS on an iPhone, which is notably faster than existing video object segmentation models and surpasses or is on par with the previous state-of-the-art methods. To our knowledge, it is the first model running on device for the task of unified segmentation and tracking. 

2 Related Work
--------------

Video Object Segmentation (VOS).  The objective of the VOS task is, given the ground-truth (GT) object segmentation mask on the first frame, tracking and predicting the object mask throughout the following frames in the video. Online learning approaches [[7](https://arxiv.org/html/2501.07256v1#bib.bib7), [40](https://arxiv.org/html/2501.07256v1#bib.bib40), [61](https://arxiv.org/html/2501.07256v1#bib.bib61), [69](https://arxiv.org/html/2501.07256v1#bib.bib69), [46](https://arxiv.org/html/2501.07256v1#bib.bib46), [38](https://arxiv.org/html/2501.07256v1#bib.bib38), [4](https://arxiv.org/html/2501.07256v1#bib.bib4), [41](https://arxiv.org/html/2501.07256v1#bib.bib41), [45](https://arxiv.org/html/2501.07256v1#bib.bib45), [49](https://arxiv.org/html/2501.07256v1#bib.bib49), [52](https://arxiv.org/html/2501.07256v1#bib.bib52), [26](https://arxiv.org/html/2501.07256v1#bib.bib26)] formulate the task as a semi-supervised learning problem, where during test time, the model is fine-tuned with the GT mask on the first frame. However, this line of work usually suffers from inference inefficiency, being input sensitive and hard to scale up with large amounts of training data. To avoid test-time training, offline-trained models propose to leverage template matching [[27](https://arxiv.org/html/2501.07256v1#bib.bib27), [10](https://arxiv.org/html/2501.07256v1#bib.bib10), [43](https://arxiv.org/html/2501.07256v1#bib.bib43), [75](https://arxiv.org/html/2501.07256v1#bib.bib75), [77](https://arxiv.org/html/2501.07256v1#bib.bib77), [62](https://arxiv.org/html/2501.07256v1#bib.bib62), [79](https://arxiv.org/html/2501.07256v1#bib.bib79), [74](https://arxiv.org/html/2501.07256v1#bib.bib74)], or memory bank [[34](https://arxiv.org/html/2501.07256v1#bib.bib34), [44](https://arxiv.org/html/2501.07256v1#bib.bib44)] to keep track of the identity information in the annotated and predicted frames. In terms of the network architecture, some works adopt recurrent networks for spatial-temporal encoding [[72](https://arxiv.org/html/2501.07256v1#bib.bib72), [33](https://arxiv.org/html/2501.07256v1#bib.bib33), [60](https://arxiv.org/html/2501.07256v1#bib.bib60), [32](https://arxiv.org/html/2501.07256v1#bib.bib32)], while recently, Transformer-based models [[32](https://arxiv.org/html/2501.07256v1#bib.bib32), [12](https://arxiv.org/html/2501.07256v1#bib.bib12), [78](https://arxiv.org/html/2501.07256v1#bib.bib78), [76](https://arxiv.org/html/2501.07256v1#bib.bib76), [66](https://arxiv.org/html/2501.07256v1#bib.bib66), [84](https://arxiv.org/html/2501.07256v1#bib.bib84), [68](https://arxiv.org/html/2501.07256v1#bib.bib68), [80](https://arxiv.org/html/2501.07256v1#bib.bib80), [19](https://arxiv.org/html/2501.07256v1#bib.bib19), [51](https://arxiv.org/html/2501.07256v1#bib.bib51), [11](https://arxiv.org/html/2501.07256v1#bib.bib11), [3](https://arxiv.org/html/2501.07256v1#bib.bib3), [14](https://arxiv.org/html/2501.07256v1#bib.bib14)] demonstrate better performance.

Segment Anything Model (SAM).  SAM [[31](https://arxiv.org/html/2501.07256v1#bib.bib31)] defines a new prompt-based segmentation task where the user prompts can be points, boxes, and masks. SAM 2 [[48](https://arxiv.org/html/2501.07256v1#bib.bib48)] further extends the task to the video inputs, namely promptable video segmentation (PVS). Different from VOS, users can provide annotations at any frame and at multiple time steps with any combination of SAM prompts, making VOS a special case of PVS. Both SAM and SAM 2 follow the same meta architecture of image encoder and prompt-based mask decoder, but to capture temporal information, SAM 2 supplements a memory banking mechanism. Thanks to training on diverse and large-scale datasets, SA-1B [[31](https://arxiv.org/html/2501.07256v1#bib.bib31)] and SA-V [[48](https://arxiv.org/html/2501.07256v1#bib.bib48)], SAM excels in both general perception and downstream tasks [[39](https://arxiv.org/html/2501.07256v1#bib.bib39), [8](https://arxiv.org/html/2501.07256v1#bib.bib8), [70](https://arxiv.org/html/2501.07256v1#bib.bib70), [55](https://arxiv.org/html/2501.07256v1#bib.bib55), [81](https://arxiv.org/html/2501.07256v1#bib.bib81), [9](https://arxiv.org/html/2501.07256v1#bib.bib9)]. To make SAM more efficient and more friendly to low-capacity devices, several works [[83](https://arxiv.org/html/2501.07256v1#bib.bib83), [71](https://arxiv.org/html/2501.07256v1#bib.bib71), [86](https://arxiv.org/html/2501.07256v1#bib.bib86), [85](https://arxiv.org/html/2501.07256v1#bib.bib85), [63](https://arxiv.org/html/2501.07256v1#bib.bib63)] propose to squeeze its image encoder to more compact visual backbones with knowledge distillation and/or masked image pre-training. However, through our benchmark, we find that apart from the image encoder, the newly introduced memory-related modules in SAM 2 are also the speed bottleneck; thus, replacing the image encoder is no longer sufficient. Therefore, we propose a novel plug-in module to accelerate memory fusion to address the problem, together with a distillation pipeline adapted for video inputs.

3 Methodology
-------------

In this section, we first briefly introduce the Segment Anything Model 2 (SAM 2), which our model is based on. Then, we propose our architecture-level improvements and knowledge distillation pipeline, respectively.

### 3.1 Preliminary: SAM 2

Overall, SAM 2 consists of four components, namely image encoder E img subscript 𝐸 img E_{\text{img}}italic_E start_POSTSUBSCRIPT img end_POSTSUBSCRIPT, mask decoder D 𝐷 D italic_D, memory encoder E mem subscript 𝐸 mem E_{\text{mem}}italic_E start_POSTSUBSCRIPT mem end_POSTSUBSCRIPT, and memory attention A 𝐴 A italic_A, with the former two almost identical to the original SAM except for the skip connection between the two. In particular, E img subscript 𝐸 img E_{\text{img}}italic_E start_POSTSUBSCRIPT img end_POSTSUBSCRIPT is a hierarchical backbone called Hiera [[50](https://arxiv.org/html/2501.07256v1#bib.bib50)], which outputs feature maps in three different strides, 4, 8, and 16 denoted by F 4 subscript 𝐹 4 F_{4}italic_F start_POSTSUBSCRIPT 4 end_POSTSUBSCRIPT, F 8 subscript 𝐹 8 F_{8}italic_F start_POSTSUBSCRIPT 8 end_POSTSUBSCRIPT, F 16 subscript 𝐹 16 F_{16}italic_F start_POSTSUBSCRIPT 16 end_POSTSUBSCRIPT, respectively:

{F 4,F 8,F 16}=E img⁢(I),subscript 𝐹 4 subscript 𝐹 8 subscript 𝐹 16 subscript 𝐸 img 𝐼\{F_{4},F_{8},F_{16}\}=E_{\text{img}}(I),{ italic_F start_POSTSUBSCRIPT 4 end_POSTSUBSCRIPT , italic_F start_POSTSUBSCRIPT 8 end_POSTSUBSCRIPT , italic_F start_POSTSUBSCRIPT 16 end_POSTSUBSCRIPT } = italic_E start_POSTSUBSCRIPT img end_POSTSUBSCRIPT ( italic_I ) ,(1)

where I 𝐼 I italic_I is the current frame input. Then, F 16 subscript 𝐹 16 F_{16}italic_F start_POSTSUBSCRIPT 16 end_POSTSUBSCRIPT is fused with memory features {M 1,M 2,…,M T}subscript 𝑀 1 subscript 𝑀 2…subscript 𝑀 𝑇\{M_{1},M_{2},\dots,M_{T}\}{ italic_M start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_M start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_M start_POSTSUBSCRIPT italic_T end_POSTSUBSCRIPT }2 2 2 For simplicity, M 𝑀 M italic_M denotes the frame-level memory feature map and we omit the object pointers (vectors from the mask decoder), which add negligible computational cost. from previous T 𝑇 T italic_T frames with the memory attention A 𝐴 A italic_A. The memory attention is essentially a stack of Transformer [[59](https://arxiv.org/html/2501.07256v1#bib.bib59)] blocks. In this setup, F 16 subscript 𝐹 16 F_{16}italic_F start_POSTSUBSCRIPT 16 end_POSTSUBSCRIPT serves as the queries, while memory features, concatenated along the temporal dimension, provide the keys and values:

F M=A⁢(F 16,M 1,M 2,…,M T),subscript 𝐹 𝑀 𝐴 subscript 𝐹 16 subscript 𝑀 1 subscript 𝑀 2…subscript 𝑀 𝑇 F_{M}=A(F_{16},M_{1},M_{2},\dots,M_{T}),italic_F start_POSTSUBSCRIPT italic_M end_POSTSUBSCRIPT = italic_A ( italic_F start_POSTSUBSCRIPT 16 end_POSTSUBSCRIPT , italic_M start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_M start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_M start_POSTSUBSCRIPT italic_T end_POSTSUBSCRIPT ) ,(2)

where F M subscript 𝐹 𝑀 F_{M}italic_F start_POSTSUBSCRIPT italic_M end_POSTSUBSCRIPT is the image feature conditioned on memories. Next, mask decoder D 𝐷 D italic_D encodes the user prompt and decodes the mask prediction O 𝑂 O italic_O given the prompt embedding P 𝑃 P italic_P and image features F M subscript 𝐹 𝑀 F_{M}italic_F start_POSTSUBSCRIPT italic_M end_POSTSUBSCRIPT, F 4 subscript 𝐹 4 F_{4}italic_F start_POSTSUBSCRIPT 4 end_POSTSUBSCRIPT, F 8 subscript 𝐹 8 F_{8}italic_F start_POSTSUBSCRIPT 8 end_POSTSUBSCRIPT:

O=D⁢(F M,F 4,F 8,P).𝑂 𝐷 subscript 𝐹 𝑀 subscript 𝐹 4 subscript 𝐹 8 𝑃 O=D(F_{M},F_{4},F_{8},P).italic_O = italic_D ( italic_F start_POSTSUBSCRIPT italic_M end_POSTSUBSCRIPT , italic_F start_POSTSUBSCRIPT 4 end_POSTSUBSCRIPT , italic_F start_POSTSUBSCRIPT 8 end_POSTSUBSCRIPT , italic_P ) .(3)

Finally, F 16 subscript 𝐹 16 F_{16}italic_F start_POSTSUBSCRIPT 16 end_POSTSUBSCRIPT and O 𝑂 O italic_O are fused and encoded with the memory encoder E mem subscript 𝐸 mem E_{\text{mem}}italic_E start_POSTSUBSCRIPT mem end_POSTSUBSCRIPT and enqueued the memory bank in a first-in-first-out manner:

M T+1=E mem⁢(F 16,O).subscript 𝑀 𝑇 1 subscript 𝐸 mem subscript 𝐹 16 𝑂 M_{T+1}=E_{\text{mem}}(F_{16},O).italic_M start_POSTSUBSCRIPT italic_T + 1 end_POSTSUBSCRIPT = italic_E start_POSTSUBSCRIPT mem end_POSTSUBSCRIPT ( italic_F start_POSTSUBSCRIPT 16 end_POSTSUBSCRIPT , italic_O ) .(4)

![Image 3: Refer to caption](https://arxiv.org/html/2501.07256v1/extracted/6127871/figure/graphics/arch-v1.png)

Figure 3: Overall architecture of EdgeTAM. The meta architecture of EdgeTAM follow SAM 2 and the main difference is the proposed plug-in module, 2D Spatial Perceiver, which is marked with orange dotted box.

### 3.2 EdgeTAM

Naïve Adaptations.  As shown in Fig.[3](https://arxiv.org/html/2501.07256v1#S3.F3 "Figure 3 ‣ 3.1 Preliminary: SAM 2 ‣ 3 Methodology ‣ EdgeTAM: On-Device Track Anything Model"), the meta architecture of SAM 2 follows closely with SAM, whose image encoder is the heaviest component in terms of parameters and computation. While the newly introduced memory-related module takes up only a small proportion of the total parameters, our benchmark (Fig. [2](https://arxiv.org/html/2501.07256v1#S1.F2 "Figure 2 ‣ 1 Introduction ‣ EdgeTAM: On-Device Track Anything Model")) shows that memory attention is also a latency bottleneck. Therefore, a naïve technique to push for improved efficiency is to substitute the image encoder with compact backbones and to reduce the number of memory attention blocks. To this end, following EdgeSAM [[86](https://arxiv.org/html/2501.07256v1#bib.bib86)], we opt for RepViT-M1[[64](https://arxiv.org/html/2501.07256v1#bib.bib64)] as the backbone and decrease the memory attention from 4 to 2 blocks. However, the inference throughput is still far from being satisfactory when deployed on mobile devices (merely 2.5 FPS on iPhone 15 Pro Max).

Taking a closer look, we observe that each memory feature M t subscript 𝑀 𝑡 M_{t}italic_M start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT has the same size as the image feature F M∈ℛ C×H×W subscript 𝐹 𝑀 superscript ℛ 𝐶 𝐻 𝑊 F_{M}\in\mathcal{R}^{C\times H\times W}italic_F start_POSTSUBSCRIPT italic_M end_POSTSUBSCRIPT ∈ caligraphic_R start_POSTSUPERSCRIPT italic_C × italic_H × italic_W end_POSTSUPERSCRIPT, where C⁢=⁢64 𝐶=64 C\text{ = }64 italic_C = 64, H⁢=⁢W⁢=⁢64 𝐻=𝑊=64 H\text{ = }W\text{ = }64 italic_H = italic_W = 64 denote channels, height and width respectively. With T 𝑇 T italic_T frames in the memory bank, the computational complexity of memory attention becomes 𝒪⁢(T⁢C⁢H 2⁢W 2)𝒪 𝑇 𝐶 superscript 𝐻 2 superscript 𝑊 2\mathcal{O}(TCH^{2}W^{2})caligraphic_O ( italic_T italic_C italic_H start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT italic_W start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ), which translates to a huge matrix multiplication that mobile devices with limited scale of parallelism perform inefficiently. While T 𝑇 T italic_T is already relatively small compared to other VOS methods, reducing it will lead to the degradation of temporal consistency and occlusion handling. On the other hand, videos are known to be information redundant. Thus, we propose to summarize the memory spatially before performing memory attention.

Global Perceiver.  Inspired by Perceiver [[29](https://arxiv.org/html/2501.07256v1#bib.bib29), [30](https://arxiv.org/html/2501.07256v1#bib.bib30)], we encode each memory feature M t subscript 𝑀 𝑡 M_{t}italic_M start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT with a stack of attention modules to compress the densely stored memories M t∈ℛ C×H×W subscript 𝑀 𝑡 superscript ℛ 𝐶 𝐻 𝑊 M_{t}\in\mathcal{R}^{C\times H\times W}italic_M start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ∈ caligraphic_R start_POSTSUPERSCRIPT italic_C × italic_H × italic_W end_POSTSUPERSCRIPT into a small set of vectors G t∈ℛ C×N g subscript 𝐺 𝑡 superscript ℛ 𝐶 subscript 𝑁 𝑔 G_{t}\in\mathcal{R}^{C\times N_{g}}italic_G start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ∈ caligraphic_R start_POSTSUPERSCRIPT italic_C × italic_N start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT end_POSTSUPERSCRIPT, where N g subscript 𝑁 𝑔 N_{g}italic_N start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT is the number of learnable latents and N g≪H×W much-less-than subscript 𝑁 𝑔 𝐻 𝑊 N_{g}\ll H\times W italic_N start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT ≪ italic_H × italic_W. Specifically, we denote the latents as Z g∈ℛ C×N g subscript 𝑍 𝑔 superscript ℛ 𝐶 subscript 𝑁 𝑔 Z_{g}\in\mathcal{R}^{C\times N_{g}}italic_Z start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT ∈ caligraphic_R start_POSTSUPERSCRIPT italic_C × italic_N start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT end_POSTSUPERSCRIPT and perform single-head cross attention (CA) between Z g subscript 𝑍 𝑔 Z_{g}italic_Z start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT and M t subscript 𝑀 𝑡 M_{t}italic_M start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT, followed by self attention (SA) as follows:

Z g′superscript subscript 𝑍 𝑔′\displaystyle Z_{g}^{\prime}italic_Z start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT=CA⁢(Q⁢(Z g),K⁢(M t+p),V⁢(M t+p)),absent CA 𝑄 subscript 𝑍 𝑔 𝐾 subscript 𝑀 𝑡 𝑝 𝑉 subscript 𝑀 𝑡 𝑝\displaystyle=\text{CA}(Q(Z_{g}),K(M_{t}+p),V(M_{t}+p)),= CA ( italic_Q ( italic_Z start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT ) , italic_K ( italic_M start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT + italic_p ) , italic_V ( italic_M start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT + italic_p ) ) ,(5)
G t subscript 𝐺 𝑡\displaystyle G_{t}italic_G start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT=SA⁢(Z g′),absent SA superscript subscript 𝑍 𝑔′\displaystyle=\text{SA}(Z_{g}^{\prime}),= SA ( italic_Z start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ) ,

where Q 𝑄 Q italic_Q, K 𝐾 K italic_K, and V 𝑉 V italic_V represent the projections for query, key, and value in CA, respectively. Z g′superscript subscript 𝑍 𝑔′Z_{g}^{\prime}italic_Z start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT is the intermediate feature and p 𝑝 p italic_p denotes the positional embeddings [[53](https://arxiv.org/html/2501.07256v1#bib.bib53)]. Here, each latent can attend globally to the memory feature and summarize it into a single vector. While the Global Perceiver introduces negligible inference cost, it cuts down the complexity of the memory attention to 𝒪⁢(T⁢C⁢H⁢W⁢N g)𝒪 𝑇 𝐶 𝐻 𝑊 subscript 𝑁 𝑔\mathcal{O}(TCHWN_{g})caligraphic_O ( italic_T italic_C italic_H italic_W italic_N start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT ). However, despite adding positional embeddings to the input of Global Perceiver, the resulting compressed memories contain only implicit positional information as the output does not maintain its spatial structure. Meanwhile, as a dense prediction task, video object segmentation requires more explicit positional information [[48](https://arxiv.org/html/2501.07256v1#bib.bib48)] and local features [[51](https://arxiv.org/html/2501.07256v1#bib.bib51)]. We thus further propose a 2D Spatial Perceiver for this purpose.

![Image 4: Refer to caption](https://arxiv.org/html/2501.07256v1/extracted/6127871/figure/graphics/distill-v2.png)

Figure 4: The distillation pipeline in EdgeTAM. In the image pre-training stage, we align the features from teacher’s and student’s image encoder. And in the video training stage, we additionally align the features output from memory attention between teacher and student. For both stages, task-specific losses are used.

2D Spatial Perceiver.  Similar to the Global Perceiver, 2D Spatial Perceiver shares the same network architecture and parameters. However, we assign spatial prior to the learnable latents Z l∈ℛ C×N l subscript 𝑍 𝑙 superscript ℛ 𝐶 subscript 𝑁 𝑙 Z_{l}\in\mathcal{R}^{C\times N_{l}}italic_Z start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ∈ caligraphic_R start_POSTSUPERSCRIPT italic_C × italic_N start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_POSTSUPERSCRIPT and restrict each latent to only attend to a local window. Specifically, we perform the window partition [[36](https://arxiv.org/html/2501.07256v1#bib.bib36)] to split the memory feature map into N l subscript 𝑁 𝑙 N_{l}italic_N start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT non-overlapping patches, and move the positional embedding p′superscript 𝑝′p^{\prime}italic_p start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT from input to output L t subscript 𝐿 𝑡 L_{t}italic_L start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT:

M t′superscript subscript 𝑀 𝑡′\displaystyle M_{t}^{\prime}italic_M start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT=window_partition⁢(M t),absent window_partition subscript 𝑀 𝑡\displaystyle=\text{window\_partition}(M_{t}),= window_partition ( italic_M start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) ,(6)
Z l′superscript subscript 𝑍 𝑙′\displaystyle Z_{l}^{\prime}italic_Z start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT=CA⁢(Q⁢(Z l),K⁢(M t′),V⁢(M t′)),absent CA 𝑄 subscript 𝑍 𝑙 𝐾 superscript subscript 𝑀 𝑡′𝑉 superscript subscript 𝑀 𝑡′\displaystyle=\text{CA}(Q(Z_{l}),K(M_{t}^{\prime}),V(M_{t}^{\prime})),= CA ( italic_Q ( italic_Z start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ) , italic_K ( italic_M start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ) , italic_V ( italic_M start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ) ) ,
L t′superscript subscript 𝐿 𝑡′\displaystyle L_{t}^{\prime}italic_L start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT=SA⁢(Z l′),absent SA superscript subscript 𝑍 𝑙′\displaystyle=\text{SA}(Z_{l}^{\prime}),= SA ( italic_Z start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ) ,
L t subscript 𝐿 𝑡\displaystyle L_{t}italic_L start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT=window_unpartition⁢(L t)+p′.absent window_unpartition subscript 𝐿 𝑡 superscript 𝑝′\displaystyle=\text{window\_unpartition}(L_{t})+p^{\prime}.= window_unpartition ( italic_L start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) + italic_p start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT .

The different designs of Global and 2D Spatial Perceiver encourage different behaviors, where global latents Z g subscript 𝑍 𝑔 Z_{g}italic_Z start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT have certain redundancy (multiple latents attend to the same input) and can dynamically distribute all over the image whereas 2D latents Z l subscript 𝑍 𝑙 Z_{l}italic_Z start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT are forced to deal with local patches. And both possess desirable merits for feature summarization. Therefore, we combine them by flattening along the spatial dimension and concatenating along the flattened dimension. Note that, our implementation stacks the blocks in Eq.[5](https://arxiv.org/html/2501.07256v1#S3.E5 "Equation 5 ‣ 3.2 EdgeTAM ‣ 3 Methodology ‣ EdgeTAM: On-Device Track Anything Model") and Eq.[6](https://arxiv.org/html/2501.07256v1#S3.E6 "Equation 6 ‣ 3.2 EdgeTAM ‣ 3 Methodology ‣ EdgeTAM: On-Device Track Anything Model") twice. Overall, when applying the proposed modules, the complexity of memory attention decreases from 𝒪⁢(T⁢C⁢H 2⁢W 2)𝒪 𝑇 𝐶 superscript 𝐻 2 superscript 𝑊 2\mathcal{O}(TCH^{2}W^{2})caligraphic_O ( italic_T italic_C italic_H start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT italic_W start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ) to 𝒪⁢(T⁢C⁢H⁢W⁢(N g+N l))𝒪 𝑇 𝐶 𝐻 𝑊 subscript 𝑁 𝑔 subscript 𝑁 𝑙\mathcal{O}(TCHW(N_{g}+N_{l}))caligraphic_O ( italic_T italic_C italic_H italic_W ( italic_N start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT + italic_N start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ) ). In practice, we control the speed-up ratio to around T 𝑇 T italic_T times, _i.e.,_(H⁢W)/(N g+N l)≈T 𝐻 𝑊 subscript 𝑁 𝑔 subscript 𝑁 𝑙 𝑇(HW)/(N_{g}+N_{l})\approx T( italic_H italic_W ) / ( italic_N start_POSTSUBSCRIPT italic_g end_POSTSUBSCRIPT + italic_N start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ) ≈ italic_T, so that the self and cross attention blocks in memory attention have similar complexity.

### 3.3 Distillation Pipeline

As shown in Fig.[4](https://arxiv.org/html/2501.07256v1#S3.F4 "Figure 4 ‣ 3.2 EdgeTAM ‣ 3 Methodology ‣ EdgeTAM: On-Device Track Anything Model"), the training pipeline of SAM 2 can be divided into image segmentation pre-training S i⁢m⁢g subscript 𝑆 𝑖 𝑚 𝑔 S_{img}italic_S start_POSTSUBSCRIPT italic_i italic_m italic_g end_POSTSUBSCRIPT and video segmentation training S v⁢i⁢d subscript 𝑆 𝑣 𝑖 𝑑 S_{vid}italic_S start_POSTSUBSCRIPT italic_v italic_i italic_d end_POSTSUBSCRIPT stages. Previous methods [[83](https://arxiv.org/html/2501.07256v1#bib.bib83), [71](https://arxiv.org/html/2501.07256v1#bib.bib71), [86](https://arxiv.org/html/2501.07256v1#bib.bib86)] demonstrate that knowledge distillation on S i⁢m⁢g subscript 𝑆 𝑖 𝑚 𝑔 S_{img}italic_S start_POSTSUBSCRIPT italic_i italic_m italic_g end_POSTSUBSCRIPT helps improve performance on images. Here, we extend this idea to the video domain and treat the distillation loss as an auxiliary loss, meaning task-specific losses are also implemented during training.

Particularly, during S i⁢m⁢g subscript 𝑆 𝑖 𝑚 𝑔 S_{img}italic_S start_POSTSUBSCRIPT italic_i italic_m italic_g end_POSTSUBSCRIPT, we adopt the same task-specific losses ℒ task subscript ℒ task\mathcal{L}_{\text{task}}caligraphic_L start_POSTSUBSCRIPT task end_POSTSUBSCRIPT as SAM (dice loss [[54](https://arxiv.org/html/2501.07256v1#bib.bib54)] and focal loss [[35](https://arxiv.org/html/2501.07256v1#bib.bib35)] for mask prediction and L1 loss for mask confidence prediction) and meanwhile, align the image encoder feature map (F 16 subscript 𝐹 16 F_{16}italic_F start_POSTSUBSCRIPT 16 end_POSTSUBSCRIPT in Eq.[1](https://arxiv.org/html/2501.07256v1#S3.E1 "Equation 1 ‣ 3.1 Preliminary: SAM 2 ‣ 3 Methodology ‣ EdgeTAM: On-Device Track Anything Model")) between the teacher and student models with MSE loss ℒ img subscript ℒ img\mathcal{L}_{\text{img}}caligraphic_L start_POSTSUBSCRIPT img end_POSTSUBSCRIPT. The pre-training loss ℒ sam subscript ℒ sam\mathcal{L}_{\text{sam}}caligraphic_L start_POSTSUBSCRIPT sam end_POSTSUBSCRIPT can be formulated with:

ℒ sam=ℒ task⁢(O,GT)+γ⋅ℒ img⁢(F 16 t,F 16 s),subscript ℒ sam subscript ℒ task 𝑂 GT⋅𝛾 subscript ℒ img superscript subscript 𝐹 16 𝑡 superscript subscript 𝐹 16 𝑠\mathcal{L}_{\text{sam}}=\mathcal{L}_{\text{task}}(O,\text{GT})+\gamma\cdot% \mathcal{L}_{\text{img}}(F_{16}^{t},F_{16}^{s}),caligraphic_L start_POSTSUBSCRIPT sam end_POSTSUBSCRIPT = caligraphic_L start_POSTSUBSCRIPT task end_POSTSUBSCRIPT ( italic_O , GT ) + italic_γ ⋅ caligraphic_L start_POSTSUBSCRIPT img end_POSTSUBSCRIPT ( italic_F start_POSTSUBSCRIPT 16 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT , italic_F start_POSTSUBSCRIPT 16 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT ) ,(7)

where O 𝑂 O italic_O is the mask prediction obtained from Eq.[1](https://arxiv.org/html/2501.07256v1#S3.E1 "Equation 1 ‣ 3.1 Preliminary: SAM 2 ‣ 3 Methodology ‣ EdgeTAM: On-Device Track Anything Model") and Eq.[3](https://arxiv.org/html/2501.07256v1#S3.E3 "Equation 3 ‣ 3.1 Preliminary: SAM 2 ‣ 3 Methodology ‣ EdgeTAM: On-Device Track Anything Model"). Here Eq.[2](https://arxiv.org/html/2501.07256v1#S3.E2 "Equation 2 ‣ 3.1 Preliminary: SAM 2 ‣ 3 Methodology ‣ EdgeTAM: On-Device Track Anything Model") is skipped due to the lack of memory bank and F M=I subscript 𝐹 𝑀 𝐼 F_{M}=I italic_F start_POSTSUBSCRIPT italic_M end_POSTSUBSCRIPT = italic_I. Here, GT, γ 𝛾\gamma italic_γ, F 16 t superscript subscript 𝐹 16 𝑡 F_{16}^{t}italic_F start_POSTSUBSCRIPT 16 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT and F 16 s superscript subscript 𝐹 16 𝑠 F_{16}^{s}italic_F start_POSTSUBSCRIPT 16 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT denote the ground-truth labels, loss weight, teacher and student image encoder features respectively.

Finally, in stage S v⁢i⁢d subscript 𝑆 𝑣 𝑖 𝑑 S_{vid}italic_S start_POSTSUBSCRIPT italic_v italic_i italic_d end_POSTSUBSCRIPT, the task-specific losses include an additional BCE loss for occlusion prediction. Besides, in order to let student’s memory-related modules receive supervision from the teacher, apart from ℒ img subscript ℒ img\mathcal{L}_{\text{img}}caligraphic_L start_POSTSUBSCRIPT img end_POSTSUBSCRIPT, we add another MSE loss ℒ mem subscript ℒ mem\mathcal{L}_{\text{mem}}caligraphic_L start_POSTSUBSCRIPT mem end_POSTSUBSCRIPT to align the F M t superscript subscript 𝐹 𝑀 𝑡 F_{M}^{t}italic_F start_POSTSUBSCRIPT italic_M end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT and F M s superscript subscript 𝐹 𝑀 𝑠 F_{M}^{s}italic_F start_POSTSUBSCRIPT italic_M end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT from teacher and student (Eq.[2](https://arxiv.org/html/2501.07256v1#S3.E2 "Equation 2 ‣ 3.1 Preliminary: SAM 2 ‣ 3 Methodology ‣ EdgeTAM: On-Device Track Anything Model")). The resulting total loss becomes:

ℒ sam2=ℒ task⁢(O,GT)+α⋅ℒ img⁢(F 16 t,F 16 s)+β⋅ℒ mem⁢(F M t,F M s),subscript ℒ sam2 subscript ℒ task 𝑂 GT⋅𝛼 subscript ℒ img superscript subscript 𝐹 16 𝑡 superscript subscript 𝐹 16 𝑠⋅𝛽 subscript ℒ mem superscript subscript 𝐹 𝑀 𝑡 superscript subscript 𝐹 𝑀 𝑠\mathcal{L}_{\text{sam2}}=\mathcal{L}_{\text{task}}(O,\text{GT})+\alpha\cdot% \mathcal{L}_{\text{img}}(F_{16}^{t},F_{16}^{s})+\beta\cdot\mathcal{L}_{\text{% mem}}(F_{M}^{t},F_{M}^{s}),caligraphic_L start_POSTSUBSCRIPT sam2 end_POSTSUBSCRIPT = caligraphic_L start_POSTSUBSCRIPT task end_POSTSUBSCRIPT ( italic_O , GT ) + italic_α ⋅ caligraphic_L start_POSTSUBSCRIPT img end_POSTSUBSCRIPT ( italic_F start_POSTSUBSCRIPT 16 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT , italic_F start_POSTSUBSCRIPT 16 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT ) + italic_β ⋅ caligraphic_L start_POSTSUBSCRIPT mem end_POSTSUBSCRIPT ( italic_F start_POSTSUBSCRIPT italic_M end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_t end_POSTSUPERSCRIPT , italic_F start_POSTSUBSCRIPT italic_M end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT ) ,(8)

with α 𝛼\alpha italic_α and β 𝛽\beta italic_β serving as the loss weights.

4 Experiments
-------------

### 4.1 Implementation Details

Training.  In general, the training procedure of EdgeTAM follows SAM 2. We set the input resolution to 1024×1024 1024 1024 1024\times 1024 1024 × 1024. During the image segmentation pre-training stage, we train on the SA-1B dataset for 2 epochs with a batch size of 128. We use AdamW [[37](https://arxiv.org/html/2501.07256v1#bib.bib37)] as the optimizer (β 1,β 2⁢=⁢0.9,0.999 subscript 𝛽 1 subscript 𝛽 2=0.9 0.999\beta_{1},\beta_{2}\text{=}0.9,0.999 italic_β start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_β start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT = 0.9 , 0.999) and set the learning rate to 4⁢e−4 4 superscript 𝑒 4 4e^{-4}4 italic_e start_POSTSUPERSCRIPT - 4 end_POSTSUPERSCRIPT with a reciprocal square-root scheduler [[82](https://arxiv.org/html/2501.07256v1#bib.bib82)]. We perform L 2 subscript L 2\text{L}_{2}L start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT gradient clipping at 0.1 and set weight decay to 0.1. The loss weights for dice, focal, IoU, and ℒ img subscript ℒ img\mathcal{L}_{\text{img}}caligraphic_L start_POSTSUBSCRIPT img end_POSTSUBSCRIPT are 20, 1, 1, and 1, respectively. For each training sample, we allow a maximum of 64 objects and add 7 correction points iteratively. Random horizontal flip is the only data augmentation in this stage. For video segmentation training, we train on SA-V, a 10% randomly sampled subset of SA-1B, DAVIS, MOSE, and YTVOS for 130K iterations with a 256 batch size. Most configurations follow the previous stage, except that the learning rate equals 6⁢e−5 6 superscript 𝑒 5 6e^{-5}6 italic_e start_POSTSUPERSCRIPT - 5 end_POSTSUPERSCRIPT for the image encoder and 3⁢e−4 3 superscript 𝑒 4 3e^{-4}3 italic_e start_POSTSUPERSCRIPT - 4 end_POSTSUPERSCRIPT for others with a cosine scheduler. The loss balancing factor for dice is 20 and 1 for focal, IoU, occlusion, ℒ img subscript ℒ img\mathcal{L}_{\text{img}}caligraphic_L start_POSTSUBSCRIPT img end_POSTSUBSCRIPT, and ℒ mem subscript ℒ mem\mathcal{L}_{\text{mem}}caligraphic_L start_POSTSUBSCRIPT mem end_POSTSUBSCRIPT. Each video sample contains 8 frames with almost 3 objects and is augmented with horizontal flip, color jitter, affine, and grayscale transformations.

Progressive fine-tuning with longer training samples.  Following SAM 2.1, we fine-tune the trained EdgeTAM model on 16-frame sequences. During the fine-tuning, we freeze the image encoder and do not apply distillation. The training set is the same as the video segmentation training stage but the total iterations are reduced to 1/3 of the original schedule. Furthermore, given that EdgeTAM consumes much less VRAM than SAM 2, we are able to further fine-tune the 16-frame model with 32-frame training samples with the same schedule. Note that the memory bank size stays the same and only the training samples become longer, so the inference cost remains the same.

![Image 5: Refer to caption](https://arxiv.org/html/2501.07256v1/extracted/6127871/figure/graphics/all-pvs-v2.png)

Figure 5: Zero-shot PVS accuracy across 9 datasets in offline and online settings.

Table 1: Zero-shot accuracy on the SA task across 23 datasets. We report 1 (5) click mIoU results. FPS is measured on iPhone. Our mix does not contain the internal datasets that SAM 2 uses.

Model Data SA-23 All SA-23 Image SA-23 Video FPS
SAM SA-1B 58.1 (81.3)60.8 (82.1)54.5 (80.3)-
SAM 2 SA-1B 58.9 (81.7)60.8 (82.1)56.4 (81.2)1.3
SAM 2 SAM2’s mix 61.4 (83.7)63.1 (83.9)59.1 (83.3)1.3
SAM 2.1 SAM2’s mix 61.9 (83.5)63.3 (83.8)60.1 (83.2)1.3
EdgeTAM Our mix 55.5 (81.7)56.0 (81.9)54.8 (81.5)40.4

Model.  By default, we use RepViT-M1 [[64](https://arxiv.org/html/2501.07256v1#bib.bib64)] pre-trained on ImageNet [[17](https://arxiv.org/html/2501.07256v1#bib.bib17)] classification as the image encoder. We also experiment with ViT-Tiny [[58](https://arxiv.org/html/2501.07256v1#bib.bib58)] pre-trained with MAE [[24](https://arxiv.org/html/2501.07256v1#bib.bib24)] on ImageNet. The number of memory attention blocks is 2 and we allocate 256 learnable latents for both Global Perceiver and 2D Spatial Perceiver. The memory bank sizes for frame-level memories and object pointers are 7 and 16 following SAM 2. The positional embeddings of Global Perceiver and 2D Spatial Perceiver are sinusoidal, and 2D-RoPE [[53](https://arxiv.org/html/2501.07256v1#bib.bib53)], respectively. We use the SAM2-HieraB+ as the teacher with the publicly available checkpoint 3 3 3 https://github.com/facebookresearch/sam2.

### 4.2 Datasets

Training.  We train on SA-1B [[31](https://arxiv.org/html/2501.07256v1#bib.bib31)], SA-V [[48](https://arxiv.org/html/2501.07256v1#bib.bib48)], DAVIS [[47](https://arxiv.org/html/2501.07256v1#bib.bib47)], MOSE [[18](https://arxiv.org/html/2501.07256v1#bib.bib18)], and YTVOS [[73](https://arxiv.org/html/2501.07256v1#bib.bib73)] datasets. SA-1B contains 11M images with 1.1B mask annotations in diverse granularities (in both part-level and object-level). The average resolution of images in SA-1B is 3300×4950 3300 4950 3300\times 4950 3300 × 4950 pixels. So far, it is the largest dataset available for image segmentation tasks. SA-V follows the criteria of SA-1B and collects 190.9K masklet annotations across 50.9K videos, which have an average duration of 14 seconds with 54%/46% indoor/outdoor scenes and are resampled to 24 FPS. Note that, the annotation frame rate is 6 FPS. Besides, 293/278 masklets from 155/150 videos are reserved as the SA-V val/test splits, which are manually picked to focus on challenging cases with fast-moving, complex occlusions, and disappearance.

Evaluation.  Our evaluation can be split into three settings: (1) Promptable Video Segmentation (PVS), where the user can click on any frames in the video to indicate an object of interest; (2) Segment Anything (SA), which is same as PVS but works with images; (3) Semi-supervised Video Object Segmentation (VOS), where ground-truth masks on the first frame are available during inference. For the video task, we report 𝒥 𝒥\mathcal{J}caligraphic_J&ℱ ℱ\mathcal{F}caligraphic_F[[47](https://arxiv.org/html/2501.07256v1#bib.bib47)] and 𝒢 𝒢\mathcal{G}caligraphic_G[[73](https://arxiv.org/html/2501.07256v1#bib.bib73)] as the metric and for images, we use mIoU.

For PVS, we evaluate with the zero-shot protocol across 9 datasets with both online and offline modes. For SA, we evaluate on SA-23 [[31](https://arxiv.org/html/2501.07256v1#bib.bib31)], which consists of 23 open-source datasets in both video (each frame is considered as an image) and image domains. Finally, for VOS, we provide performance on the popular DAVIS 2017 [[47](https://arxiv.org/html/2501.07256v1#bib.bib47)], MOSE [[18](https://arxiv.org/html/2501.07256v1#bib.bib18)], and YouTubeVOS [[73](https://arxiv.org/html/2501.07256v1#bib.bib73)] val sets and the challenging SA-V val/test set [[48](https://arxiv.org/html/2501.07256v1#bib.bib48)].

Table 2: Performance on the VOS task. We report the 𝒢 𝒢\mathcal{G}caligraphic_G for YTVOS and 𝒥 𝒥\mathcal{J}caligraphic_J&ℱ ℱ\mathcal{F}caligraphic_F for other datasets. The FPS on A100 is obtained with torch compile. Nota that, for SAM 2, SAM 2.1, and EdgeTAM, we evaluate all the datasets with the same model.

Method MOSE val DAVIS 2017 val SA-V val SA-V test YTVOS 2019 val A100 V100 iPhone
STCN [[12](https://arxiv.org/html/2501.07256v1#bib.bib12)]52.5 85.4 61.0 62.5 82.7 62.8 13.2-
SwinB-AOT [[78](https://arxiv.org/html/2501.07256v1#bib.bib78)]59.4 85.4 51.1 50.3 84.5---
SwinB-DeAOT [[76](https://arxiv.org/html/2501.07256v1#bib.bib76)]59.9 86.2 61.4 61.8 86.1---
RDE [[32](https://arxiv.org/html/2501.07256v1#bib.bib32)]46.8 84.2 51.8 53.9 81.9 88.8 24.4-
XMem [[11](https://arxiv.org/html/2501.07256v1#bib.bib11)]59.6 86.0 60.1 62.3 85.6 61.2 22.6-
SimVOS-B [[68](https://arxiv.org/html/2501.07256v1#bib.bib68)]-88.0 44.2 44.1 84.2-3.3-
JointFormer [[84](https://arxiv.org/html/2501.07256v1#bib.bib84)]-90.1--87.4-3.0-
ISVOS [[66](https://arxiv.org/html/2501.07256v1#bib.bib66)]-88.2--86.3-5.8-
DEVA [[13](https://arxiv.org/html/2501.07256v1#bib.bib13)]66.0 87.0 55.4 56.2 85.4 65.2 25.3-
Cutie-base [[14](https://arxiv.org/html/2501.07256v1#bib.bib14)]69.9 87.9 60.7 62.7 87.0 65.0 36.4-
Cutie-base+ [[14](https://arxiv.org/html/2501.07256v1#bib.bib14)]71.7 88.1 61.3 62.8 87.5 57.2 17.9-
SAM 2-B+ [[48](https://arxiv.org/html/2501.07256v1#bib.bib48)]75.8 90.9 73.6 74.1 88.4 64.8-0.7
SAM 2.1-B+ [[48](https://arxiv.org/html/2501.07256v1#bib.bib48)]76.6 90.2 76.8 77.0 88.6 64.1-0.7
EdgeTAM 70.0 87.7 72.3 71.7 86.2 150.9-15.7

Table 3: Ablation Studies.

(a)Effectiveness of each proposed component.

Memory Efficiency Distill SA-V val SA-V test FPS
-63.5 62.1 2.5
Average Pooling 61.8 59.8 15.7
2D Perceiver 64.4 62.5 15.7
2D Perceiver✓65.7 65.8 15.7

(b)Latents allocation for 2D Perceiver.

Global Latents 2D Latents SA-V val SA-V test FPS
0 0 63.5 62.1 2.5
256 0 62.0 60.6 15.7
0 256 63.1 62.4 15.7
256 256 64.4 62.5 15.7

(c)EdgeTAM with different backbones and # of memory attention blocks.

Image Encoder Mem. Attn.Blocks SA-V val SA-V test FPS
ViT-Tiny 1 65.1 64.1 8.5
ViT-Tiny 2 67.9 66.0 7.4
RepViT-M1 1 64.3 61.6 22.2
RepViT-M1 2 65.7 65.8 15.7
RepViT-M1 4 65.0 65.6 10.0

(d)Ablation on using self attention in 2D Perceiver.

Self-Attn in Perceiver SA-V val SA-V test FPS
No 62.6 62.7 15.7
Yes 64.4 62.5 15.7

### 4.3 Promptable Video Segmentation (PVS)

One of the key features of EdgeTAM is that it follows the same meta architecture of SAM 2, which enables it to perform promptable video segmentation with various user inputs on any frames. As shown in Fig.[5](https://arxiv.org/html/2501.07256v1#S4.F5 "Figure 5 ‣ 4.1 Implementation Details ‣ 4 Experiments ‣ EdgeTAM: On-Device Track Anything Model"), we follow the same online and offline PVS settings as SAM 2, which simulate user interaction in the real world. The offline mode allows multiple times of playbacks to only add correction points on the frames with large errors, while the online mode only annotates the frames in a single forward pass. Compared to SAM + XMem++ and SAM + Cuite, EdgeTAM outperforms both across all settings with considerable margins. Besides, thanks to being trained in an end-to-end manner and distilled with the SAM 2 teacher the gap becomes larger as the number of annotated frames increases. Besides, even compared with the original SAM 2, EdgeTAM achieves comparable results despite being significantly smaller and faster.

![Image 6: Refer to caption](https://arxiv.org/html/2501.07256v1/extracted/6127871/figure/graphics/vis-v1.png)

Figure 6: Qualitative results of EdgeTAM compared with SAM 2. In the upper example, we show tracking multiple instances from the same class, which also stay closely to each other. Our EdgeTAM delivers similar mask quality as SAM 2. In the lower example, we demonstrate a fast moving object with large distortion. While in general, EgdeTAM yields results that the boundary well, it outputs different granularities as SAM 2, not tracking the bird feet.

### 4.4 Segment Anything (SA)

Both SAM 2 and EdgeTAM can function as image segmentation models with the memory module detached. As shown in Tab.[1](https://arxiv.org/html/2501.07256v1#S4.T1 "Table 1 ‣ 4.1 Implementation Details ‣ 4 Experiments ‣ EdgeTAM: On-Device Track Anything Model"), EdgeTAM achieves comparable mIoU performance with SAM and SAM 2, especially with more input points. For example, with five input points, on average, EdgeTAM even surpasses SAM-H (81.7 _v.s._ 81.3), which is dedicated to image segmentation. Note that, our EdgeTAM is not trained with the internal datasets that both SAM 2 and SAM 2.1 use. Given its real-time speed, EdgeTAM can be used as a unified on-device segmentation model for both images and videos.

### 4.5 Video Object Segmentation (VOS)

While EdgeTAM is trained only with the SA-V and SA-1B dataset, as shown in Tab.[2](https://arxiv.org/html/2501.07256v1#S4.T2 "Table 2 ‣ 4.2 Datasets ‣ 4 Experiments ‣ EdgeTAM: On-Device Track Anything Model"), on MOSE, DAVIS, and YTVOS, it is on par or surpasses previous state-of-the-art VOS models that are trained on these datasets. This demonstrates the robustness of EdgeTAM under the zero-shot setting. More importantly, it is impractical to deploy multiple models on device with one for certain types of data. Also, thanks to training on SA-V, EdgeTAM surpasses all its counterparts except for SAM 2 and SAM 2.1 on SA-V val and test. Note that, the masks in SA-V val/test have different granularities, while those of other datasets are at object-level. This shows the flexibility of EdgeTAM. In addition, for speed benchmarking, our main goal is inference on edge devices and we observe even with torch compile, the streaming multiprocessor utilization of EdgeTAM is still relatively low. Through the Torch profile, we find that on high-end GPU, the CPU (CUDA kernel launching) becomes the bottleneck for EdgeTAM. Thus, we encourage focusing on edge device latency, which EdgeTAM is designed for.

### 4.6 Ablations

For all the ablation studies, we train with one-third of the original training schedule (43k steps). As shown in Tab.[3(d)](https://arxiv.org/html/2501.07256v1#S4.T3.st4 "Table 3(d) ‣ Table 3 ‣ 4.2 Datasets ‣ 4 Experiments ‣ EdgeTAM: On-Device Track Anything Model")(a), we first ablate the effectiveness of each proposed component. In the table, we set the baseline as RepViT-M1 with two memory attention blocks and we also compare with simply downsampling the spatial memories instead of using the 2D Perceiver. Experiments show that 2D Spatial Perceiver is both faster and more accurate than the baseline and 4×\times×4 average pooling (0.4 to 2.7 better). Besides, the proposed distillation pipeline further improves the 𝒥 𝒥\mathcal{J}caligraphic_J&ℱ ℱ\mathcal{F}caligraphic_F on SA-V val and test by 1.3 and 3.3. Then, in Tab.[3(d)](https://arxiv.org/html/2501.07256v1#S4.T3.st4 "Table 3(d) ‣ Table 3 ‣ 4.2 Datasets ‣ 4 Experiments ‣ EdgeTAM: On-Device Track Anything Model")(b), we vary the number of global and 2D latents and find that using both yields the best performance and speed-up. Note that, using 2D latents speed up the baseline by 6.3 ×\times× with better performance. Tab.[3(d)](https://arxiv.org/html/2501.07256v1#S4.T3.st4 "Table 3(d) ‣ Table 3 ‣ 4.2 Datasets ‣ 4 Experiments ‣ EdgeTAM: On-Device Track Anything Model")(c) shows using 2D Perceiver on different combinations of image encoders and the number of memory attention blocks. And we opt for RepViT-M1 with two memory attentions for the best trade-off. Finally, in Tab.[3(d)](https://arxiv.org/html/2501.07256v1#S4.T3.st4 "Table 3(d) ‣ Table 3 ‣ 4.2 Datasets ‣ 4 Experiments ‣ EdgeTAM: On-Device Track Anything Model")(d), we study whether to use self attention in the 2D Perceiver network. The motivation here is that as each 2D latent attends to a local patch that has no overlap with each other, incorporating self attention blocks will encourage the communication between 2D latents to yield better features. Our results verify this hypothesis.

Table 4: Zero-shot accuracy across 17 video datasets under semi-supervised VOS evaluation using different prompts. For all prompt types, the annotation is only provided on the first frame. †: When the ground-truth mask is available, SAM is not used for XMem++ and Cuite.

Method 1-click 3-click 5-click bounding box ground-truth mask†
SAM + XMem++ [[3](https://arxiv.org/html/2501.07256v1#bib.bib3)]56.9 68.4 70.6 67.6 72.7
SAM + Cutie [[14](https://arxiv.org/html/2501.07256v1#bib.bib14)]56.7 70.1 72.2 69.4 74.1
SAM 2 [[48](https://arxiv.org/html/2501.07256v1#bib.bib48)]64.3 73.2 75.4 72.9 77.6
SAM 2.1 [[48](https://arxiv.org/html/2501.07256v1#bib.bib48)]64.7 75.3 77.6 74.4 79.3
EdgeTAM 54.4 72.7 75.5 71.3 77.0

### 4.7 Qualitative Results

In Fig.[6](https://arxiv.org/html/2501.07256v1#S4.F6 "Figure 6 ‣ 4.3 Promptable Video Segmentation (PVS) ‣ 4 Experiments ‣ EdgeTAM: On-Device Track Anything Model"), we compare the visualization results of EdgeTAM and SAM 2 on the YouTubeVOS val dataset. We pick two representative examples, one with multiple instances from the same class gathering together, and the other with a fast-moving object with a large distortion. For the first example, EdgeTAM yields similar results as SAM 2 and keeps the identity of each instance throughout the clip. However, in the second example, we observe that EdgeTAM falls into a typical failure case that the tracking granularity might always follow SAM 2. In the example, EdgeTAM does not include the bird feet in the mask predictions given that in previous frames, the feet are not visible.

5 Conclusion
------------

In this paper, we identify that the latency bottleneck of SAM 2 lies in the memory attention module and propose EdgeTAM to reduce the heavy overhead of cross attention with minimal performance degradation. Specifically, we propose 2D Spatial Perceiver to encode the densely stored frame-level memories into much smaller token sets while preserving their 2D spatial structure, which is essential for dense prediction tasks. As a plug-in module, 2D Spatial Perceiver can be applied to any SAM 2 variants. Besides, we also extend the knowledge distillation pipeline used in SAM for image segmentation to the video domain, which further improves the performance of EdgeTAM without inference-time cost. Our experiments show EdgeTAM nicely preserves the capability of SAM 2 across PVS, VOS, and SA tasks. More importantly, it runs 22×\times× faster than SAM 2 and achieves 16 FPS on iPhone 15 Pro Max.

A Video Object Segmentation (VOS)
---------------------------------

In our main submission, we follow the standard semi-supervised video object segmentation protocol, where the ground-truth masks on the first frame are available during inference. In Tab.[4](https://arxiv.org/html/2501.07256v1#S4.T4 "Table 4 ‣ 4.6 Ablations ‣ 4 Experiments ‣ EdgeTAM: On-Device Track Anything Model"), we follow SAM 2 [[48](https://arxiv.org/html/2501.07256v1#bib.bib48)] and instead of making the masks on the first frame available, we prompt the object of interest with clicks or boxes on the first frame. Given that XMem++ and Cutie do not support these prompts, we convert the prompt to masks with SAM [[31](https://arxiv.org/html/2501.07256v1#bib.bib31)]. We evaluate on 17 zero-shot datasets including EndoVis 2018 [[2](https://arxiv.org/html/2501.07256v1#bib.bib2)], ESD [[28](https://arxiv.org/html/2501.07256v1#bib.bib28)], LVOSv2 [[25](https://arxiv.org/html/2501.07256v1#bib.bib25)], LV-VIS [[65](https://arxiv.org/html/2501.07256v1#bib.bib65)], UVO [[67](https://arxiv.org/html/2501.07256v1#bib.bib67)], VOST [[56](https://arxiv.org/html/2501.07256v1#bib.bib56)], PUMaVOS [[3](https://arxiv.org/html/2501.07256v1#bib.bib3)], Virtual KITTI 2 [[6](https://arxiv.org/html/2501.07256v1#bib.bib6)], VIPSeg [[42](https://arxiv.org/html/2501.07256v1#bib.bib42)], Wildfires [[57](https://arxiv.org/html/2501.07256v1#bib.bib57)], VISOR [[16](https://arxiv.org/html/2501.07256v1#bib.bib16)], FBMS [[5](https://arxiv.org/html/2501.07256v1#bib.bib5)], Ego-Exo4D [[22](https://arxiv.org/html/2501.07256v1#bib.bib22)], Cityscapes [[15](https://arxiv.org/html/2501.07256v1#bib.bib15)], Lindenthal Camera [[23](https://arxiv.org/html/2501.07256v1#bib.bib23)], HT1080WT Cells [[21](https://arxiv.org/html/2501.07256v1#bib.bib21)], and Drosophila Heart [[20](https://arxiv.org/html/2501.07256v1#bib.bib20)].

In this evaluation suite, except for the 1-click setting, EdgeTAM surpasses the strong baselines, SAM + XMem++ and SAM + Cutie, by 2 to 5 percent. Compared to SAM 2 and SAM 2.1, EdgeTAM still preserves comparable performance especially with more accurate prompts, such as 5-click and ground-truth mask.

Table 5: Hyperparameters and details of EdgeTAM image segmentation pre-training and video segmentation training.

(a)Image segmentation pre-training.

Config Value
data SA-1B
steps∼similar-to\sim∼175K
resolution 1024
precision bfloat16
optimizer AdamW
optimizer momentum β 1,β 2=0.9,0.999 formulae-sequence subscript 𝛽 1 subscript 𝛽 2 0.9 0.999\beta_{1},\beta_{2}=0.9,0.999 italic_β start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_β start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT = 0.9 , 0.999
gradient clipping type: ℓ 2 subscript ℓ 2\ell_{2}roman_ℓ start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT, max: 0.1
weight decay 0.1
learning rate (lr)4⁢e−4 4 superscript 𝑒 4 4e^{-4}4 italic_e start_POSTSUPERSCRIPT - 4 end_POSTSUPERSCRIPT
lr schedule reciprocal sqrt
timescale=1000
warmup linear, 1K iters
cooldown linear, 5K iters
augmentation hflip
batch size 128
mask losses (weight)focal (20), dice (1)
IoU loss (weight)ℓ 1 subscript ℓ 1\ell_{1}roman_ℓ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT (1)
distill loss (weight)MSE (1)
max. masks per img.64
# correction points 7

(b)Video segmentation training.

Config Value
data SA-1B, SA-V, DAVIS, MOSE, YTVOS
steps∼similar-to\sim∼130K
resolution 1024
precision bfloat16
optimizer AdamW
optimizer momentum β 1,β 2=0.9,0.999 formulae-sequence subscript 𝛽 1 subscript 𝛽 2 0.9 0.999\beta_{1},\beta_{2}=0.9,0.999 italic_β start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_β start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT = 0.9 , 0.999
gradient clipping type: ℓ 2 subscript ℓ 2\ell_{2}roman_ℓ start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT, max: 0.1
weight decay 0.1
learning rate (lr)backbone: 6⁢e−5 6 superscript 𝑒 5 6e^{-5}6 italic_e start_POSTSUPERSCRIPT - 5 end_POSTSUPERSCRIPT, other: 3⁢e−4 3 superscript 𝑒 4 3e^{-4}3 italic_e start_POSTSUPERSCRIPT - 4 end_POSTSUPERSCRIPT
lr schedule cosine
warmup linear, 15K iters
img. augmentation hflip
vid. augmentation hflip,
affine (deg: 25, shear: 20),
colorjitter (0.1),
grayscale (0.05),
per frame colorjitter (0.1)
batch size 256
mask losses (weight)focal (20), dice (1)
IoU loss (weight)ℓ 1 subscript ℓ 1\ell_{1}roman_ℓ start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT (1)
occlusion loss (weight)cross-entropy (1)
distill loss (weight)MSE (1) for both ℒ img subscript ℒ img\mathcal{L}_{\text{img}}caligraphic_L start_POSTSUBSCRIPT img end_POSTSUBSCRIPT and ℒ mem subscript ℒ mem\mathcal{L}_{\text{mem}}caligraphic_L start_POSTSUBSCRIPT mem end_POSTSUBSCRIPT
max. masks per frame image: 32, video: 3
# correction points 7

B Implementation Details
------------------------

We generally follow the original SAM 2 training hyper-parameters for image segmentation pre-training [[31](https://arxiv.org/html/2501.07256v1#bib.bib31)] and video segmentation training [[48](https://arxiv.org/html/2501.07256v1#bib.bib48)]. Here, we highlight only the differences, and the full training details are shown in Tab.[5(b)](https://arxiv.org/html/2501.07256v1#S1.T5.st2 "Table 5(b) ‣ Table 5 ‣ A Video Object Segmentation (VOS) ‣ EdgeTAM: On-Device Track Anything Model"). First, we do not apply drop path or layer-wise decay in the image encoder. Second, our image pre-training stage adopts a 128 batch size and a total of 175K training steps. In the video training stage, we reduce the maximum number of masks per image from 64 to 32. More importantly, we do not train on the SAM 2 Internal dataset so the total training steps are reduced from 300K to 130K. Finally, our training involves distillation losses in both stages.

C Speed Benchmark
-----------------

In Tab.[2](https://arxiv.org/html/2501.07256v1#S4.T2 "Table 2 ‣ 4.2 Datasets ‣ 4 Experiments ‣ EdgeTAM: On-Device Track Anything Model"), we provide the throughput FPS on both server GPUs (NVIDIA A100 and V100) and mobile NPU (iPhone 15 Pro Max). The V100 benchmarks are collected from each individual paper and we benchmark with the other two hardware by ourselves. In particular, to optimize the throughput, on A100, we torch compile all the models. For mobile NPU, we convert the model to CoreML format with coremltools [[1](https://arxiv.org/html/2501.07256v1#bib.bib1)] and benchmark with the performance report tool of XCode with iOS 18.1 on an iPhone 15 Pro Max. Note that, the speed-up ratios of EdgeTAM _v.s._ SAM 2 are less pronounced on A100 than on iPhone. To understand the root cause, we monitor the streaming multiprocessor (SM) utilization of both models on A100 and find that even with torch compile, the SM usage of EdgeTAM is less than 50% and the inference is bottlenecked on CPU and IO. We think it is because high-end server GPUs, such as A100, have an enormous amount of parallel executable units (EU) and given the tiny size of EdgeTAM, it cannot occupy all the EUs at the same time. However, the design objective of EdgeTAM is edge devices, such as mobile phones, where we see 22×22\times 22 × speed-up compared with SAM 2.

References
----------

*   cor [2021] Core ml tools. [https://github.com/apple/coremltools](https://github.com/apple/coremltools), 2021. 
*   Allan et al. [2020] Max Allan, Satoshi Kondo, Sebastian Bodenstedt, Stefan Leger, Rahim Kadkhodamohammadi, Imanol Luengo, Felix Fuentes, Evangello Flouty, Ahmed Mohammed, Marius Pedersen, et al. 2018 robotic scene segmentation challenge. _arXiv preprint arXiv:2001.11190_, 2020. 
*   Bekuzarov et al. [2023] Maksym Bekuzarov, Ariana Bermudez, Joon-Young Lee, and Hao Li. Xmem++: Production-level video segmentation from few annotated frames. In _ICCV_, 2023. 
*   Bhat et al. [2020] Goutam Bhat, Felix Järemo Lawin, Martin Danelljan, Andreas Robinson, Michael Felsberg, Luc Van Gool, and Radu Timofte. Learning what to learn for video object segmentation. In _ECCV_, 2020. 
*   Brox et al. [2010] T Brox, J Malik, and P Ochs. Freiburg-berkeley motion segmentation dataset (fbms-59). In _ECCV_, 2010. 
*   Cabon et al. [2020] Yohann Cabon, Naila Murray, and Martin Humenberger. Virtual kitti 2. _arXiv preprint arXiv:2001.10773_, 2020. 
*   Caelles et al. [2017] Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixé, Daniel Cremers, and Luc Van Gool. One-shot video object segmentation. In _CVPR_, 2017. 
*   Chen et al. [2024a] Keyan Chen, Chenyang Liu, Hao Chen, Haotian Zhang, Wenyuan Li, Zhengxia Zou, and Zhenwei Shi. Rsprompter: Learning to prompt for remote sensing instance segmentation based on visual foundation model. _IEEE Transactions on Geoscience and Remote Sensing_, 2024a. 
*   Chen et al. [2024b] Tianrun Chen, Ankang Lu, Lanyun Zhu, Chaotao Ding, Chunan Yu, Deyi Ji, Zejian Li, Lingyun Sun, Papa Mao, and Ying Zang. Sam2-adapter: Evaluating & adapting segment anything 2 in downstream tasks: Camouflage, shadow, medical image segmentation, and more. _arXiv preprint arXiv:2408.04579_, 2024b. 
*   Chen et al. [2018] Yuhua Chen, Jordi Pont-Tuset, Alberto Montes, and Luc Van Gool. Blazingly fast video object segmentation with pixel-wise metric learning. In _CVPR_, 2018. 
*   Cheng and Schwing [2022] Ho Kei Cheng and Alexander G Schwing. Xmem: Long-term video object segmentation with an atkinson-shiffrin memory model. In _ECCV_, 2022. 
*   Cheng et al. [2021] Ho Kei Cheng, Yu-Wing Tai, and Chi-Keung Tang. Rethinking space-time networks with improved memory coverage for efficient video object segmentation. In _NeurIPS_, 2021. 
*   Cheng et al. [2023] Ho Kei Cheng, Seoung Wug Oh, Brian Price, Alexander Schwing, and Joon-Young Lee. Tracking anything with decoupled video segmentation. In _ICCV_, 2023. 
*   Cheng et al. [2024] Ho Kei Cheng, Seoung Wug Oh, Brian Price, Joon-Young Lee, and Alexander Schwing. Putting the object back into video object segmentation. In _CVPR_, 2024. 
*   Cordts et al. [2016] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In _CVPR_, 2016. 
*   Darkhalil et al. [2022] Ahmad Darkhalil, Dandan Shan, Bin Zhu, Jian Ma, Amlan Kar, Richard Higgins, Sanja Fidler, David Fouhey, and Dima Damen. Epic-kitchens visor benchmark: Video segmentations and object relations. In _NeurIPS_, 2022. 
*   Deng et al. [2009] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In _CVPR_, 2009. 
*   Ding et al. [2023] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, Philip HS Torr, and Song Bai. Mose: A new dataset for video object segmentation in complex scenes. In _ICCV_, 2023. 
*   Duke et al. [2021] Brendan Duke, Abdalla Ahmed, Christian Wolf, Parham Aarabi, and Graham W Taylor. Sstvos: Sparse spatiotemporal transformers for video object segmentation. In _CVPR_, 2021. 
*   Fishman et al. [2023] Matthew Fishman, Abigail Matt, Fei Wang, Elena Gracheva, Jiantao Zhu, Xiangping Ouyang, Andrey Komarov, Yuxuan Wang, Hongwu Liang, and Chao Zhou. A drosophila heart optical coherence microscopy dataset for automatic video segmentation. _Scientific data_, 2023. 
*   Gómez-de Mariscal et al. [2021] Estibaliz Gómez-de Mariscal, Hasini Jayatilaka, Özgün Çiçek, Thomas Brox, Denis Wirtz, and Arrate Muñoz-Barrutia. Search for temporal cell segmentation robustness in phase-contrast microscopy videos. _arXiv preprint arXiv:2112.08817_, 2021. 
*   Grauman et al. [2024] Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Triantafyllos Afouras, Kumar Ashutosh, Vijay Baiyya, Siddhant Bansal, Bikram Boote, et al. Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives. In _CVPR_, 2024. 
*   Haucke and Steinhage [2021] Timm Haucke and Volker Steinhage. Exploiting depth information for wildlife monitoring. _arXiv preprint arXiv:2102.05607_, 2021. 
*   He et al. [2022] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In _CVPR_, 2022. 
*   Hong et al. [2024] Lingyi Hong, Zhongying Liu, Wenchao Chen, Chenzhi Tan, Yuang Feng, Xinyu Zhou, Pinxue Guo, Jinglun Li, Zhaoyu Chen, Shuyong Gao, et al. Lvos: A benchmark for large-scale long-term video object segmentation. _arXiv preprint arXiv:2404.19326_, 2024. 
*   Hu et al. [2017] Yuan-Ting Hu, Jia-Bin Huang, and Alexander Schwing. Maskrnn: Instance level video object segmentation. In _NeurIPS_, 2017. 
*   Hu et al. [2018] Yuan-Ting Hu, Jia-Bin Huang, and Alexander G Schwing. Videomatch: Matching based video object segmentation. In _ECCV_, 2018. 
*   Huang et al. [2023] Xiaoqian Huang, Kachole Sanket, Abdulla Ayyad, Fariborz Baghaei Naeini, Dimitrios Makris, and Yahya Zweiri. A neuromorphic dataset for object segmentation in indoor cluttered environment. _arXiv preprint arXiv:2302.06301_, 2023. 
*   Jaegle et al. [2021] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In _ICML_, 2021. 
*   Jaegle et al. [2022] Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, et al. Perceiver io: A general architecture for structured inputs & outputs. In _ICLR_, 2022. 
*   Kirillov et al. [2023] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In _CVPR_, 2023. 
*   Li et al. [2022] Mingxing Li, Li Hu, Zhiwei Xiong, Bang Zhang, Pan Pan, and Dong Liu. Recurrent dynamic embedding for video object segmentation. In _CVPR_, 2022. 
*   Li et al. [2020] Yu Li, Zhuoran Shen, and Ying Shan. Fast video object segmentation using the global context module. In _ECCV_, 2020. 
*   Liang et al. [2020] Yongqing Liang, Xin Li, Navid Jafari, and Jim Chen. Video object segmentation with adaptive feature bank and uncertain-region refinement. In _NeurIPS_, 2020. 
*   Lin et al. [2017] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In _ICCV_, 2017. 
*   Liu et al. [2021] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In _ICCV_, 2021. 
*   Loshchilov and Hutter [2019] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _ICLR_, 2019. 
*   Luiten et al. [2018] Jonathon Luiten, Paul Voigtlaender, and Bastian Leibe. Premvos: Proposal-generation, refinement and merging for video object segmentation. In _ACCV_, 2018. 
*   Ma et al. [2024] Jun Ma, Yuting He, Feifei Li, Lin Han, Chenyu You, and Bo Wang. Segment anything in medical images. _Nature Communications_, 2024. 
*   Maninis et al. [2018] K-K Maninis, Sergi Caelles, Yuhua Chen, Jordi Pont-Tuset, Laura Leal-Taixé, Daniel Cremers, and Luc Van Gool. Video object segmentation without temporal information. _IEEE TPAMI_, 2018. 
*   Meinhardt and Leal-Taixé [2020] Tim Meinhardt and Laura Leal-Taixé. Make one-shot video object segmentation efficient again. In _NeurIPS_, 2020. 
*   Miao et al. [2022] Jiaxu Miao, Xiaohan Wang, Yu Wu, Wei Li, Xu Zhang, Yunchao Wei, and Yi Yang. Large-scale video panoptic segmentation in the wild: A benchmark. In _CVPR_, 2022. 
*   Oh et al. [2018] Seoung Wug Oh, Joon-Young Lee, Kalyan Sunkavalli, and Seon Joo Kim. Fast video object segmentation by reference-guided mask propagation. In _CVPR_, 2018. 
*   Oh et al. [2019] Seoung Wug Oh, Joon-Young Lee, Ning Xu, and Seon Joo Kim. Video object segmentation using space-time memory networks. In _ICCV_, 2019. 
*   Park et al. [2021] Hyojin Park, Jayeon Yoo, Seohyeong Jeong, Ganesh Venkatesh, and Nojun Kwak. Learning dynamic network using a reuse gate function in semi-supervised video object segmentation. In _CVPR_, 2021. 
*   Perazzi et al. [2017] Federico Perazzi, Anna Khoreva, Rodrigo Benenson, Bernt Schiele, and Alexander Sorkine-Hornung. Learning video object segmentation from static images. In _CVPR_, 2017. 
*   Pont-Tuset et al. [2017] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbeláez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. _arXiv preprint arXiv:1704.00675_, 2017. 
*   Ravi et al. [2024] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. _arXiv preprint arXiv:2408.00714_, 2024. 
*   Robinson et al. [2020] Andreas Robinson, Felix Jaremo Lawin, Martin Danelljan, Fahad Shahbaz Khan, and Michael Felsberg. Learning fast and robust target models for video object segmentation. In _CVPR_, 2020. 
*   Ryali et al. [2023] Chaitanya Ryali, Yuan-Ting Hu, Daniel Bolya, Chen Wei, Haoqi Fan, Po-Yao Huang, Vaibhav Aggarwal, Arkabandhu Chowdhury, Omid Poursaeed, Judy Hoffman, et al. Hiera: A hierarchical vision transformer without the bells-and-whistles. In _ICML_, 2023. 
*   Shaker et al. [2024] Abdelrahman Shaker, Syed Talal Wasim, Martin Danelljan, Salman Khan, Ming-Hsuan Yang, and Fahad Shahbaz Khan. Efficient video object segmentation via modulated cross-attention memory. _arXiv preprint arXiv:2403.17937_, 2024. 
*   Shin Yoon et al. [2017] Jae Shin Yoon, Francois Rameau, Junsik Kim, Seokju Lee, Seunghak Shin, and In So Kweon. Pixel-level matching for video object segmentation using convolutional neural networks. In _ICCV_, 2017. 
*   Su et al. [2024] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. _Neurocomputing_, 2024. 
*   Sudre et al. [2017] Carole H Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, and M Jorge Cardoso. Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations. In _MICCAIW_, 2017. 
*   Tang et al. [2023] Lv Tang, Haoke Xiao, and Bo Li. Can sam segment anything? when sam meets camouflaged object detection. _arXiv preprint arXiv:2304.04709_, 2023. 
*   Tokmakov et al. [2023] Pavel Tokmakov, Jie Li, and Adrien Gaidon. Breaking the” object” in video object segmentation. In _CVPR_, 2023. 
*   Toulouse et al. [2017] Tom Toulouse, Lucile Rossi, Antoine Campana, Turgay Celik, and Moulay A Akhloufi. Computer vision for wildfire research: An evolving image dataset for processing and analysis. _Fire Safety Journal_, 2017. 
*   Touvron et al. [2021] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In _ICML_, 2021. 
*   Vaswani [2017] A Vaswani. Attention is all you need. _Advances in Neural Information Processing Systems_, 2017. 
*   Ventura et al. [2019] Carles Ventura, Miriam Bellver, Andreu Girbau, Amaia Salvador, Ferran Marques, and Xavier Giro-i Nieto. Rvos: End-to-end recurrent network for video object segmentation. In _CVPR_, 2019. 
*   Voigtlaender and Leibe [2017] Paul Voigtlaender and Bastian Leibe. Online adaptation of convolutional neural networks for video object segmentation. In _BMVC_, 2017. 
*   Voigtlaender et al. [2019] Paul Voigtlaender, Yuning Chai, Florian Schroff, Hartwig Adam, Bastian Leibe, and Liang-Chieh Chen. Feelvos: Fast end-to-end embedding learning for video object segmentation. In _CVPR_, 2019. 
*   Wang et al. [2023a] Ao Wang, Hui Chen, Zijia Lin, Jungong Han, and Guiguang Ding. Repvit-sam: Towards real-time segmenting anything. _arXiv preprint arXiv:2312.05760_, 2023a. 
*   Wang et al. [2024] Ao Wang, Hui Chen, Zijia Lin, Jungong Han, and Guiguang Ding. Repvit: Revisiting mobile cnn from vit perspective. In _CVPR_, 2024. 
*   Wang et al. [2023b] Haochen Wang, Cilin Yan, Shuai Wang, Xiaolong Jiang, Xu Tang, Yao Hu, Weidi Xie, and Efstratios Gavves. Towards open-vocabulary video instance segmentation. In _ICCV_, 2023b. 
*   Wang et al. [2023c] Junke Wang, Dongdong Chen, Zuxuan Wu, Chong Luo, Chuanxin Tang, Xiyang Dai, Yucheng Zhao, Yujia Xie, Lu Yuan, and Yu-Gang Jiang. Look before you match: Instance understanding matters in video object segmentation. In _CVPR_, 2023c. 
*   Wang et al. [2021] Weiyao Wang, Matt Feiszli, Heng Wang, and Du Tran. Unidentified video objects: A benchmark for dense, open-world segmentation. In _ICCV_, 2021. 
*   Wu et al. [2023] Qiangqiang Wu, Tianyu Yang, Wei Wu, and Antoni B Chan. Scalable video object segmentation with simplified framework. In _ICCV_, 2023. 
*   Xiao et al. [2018] Huaxin Xiao, Jiashi Feng, Guosheng Lin, Yu Liu, and Maojun Zhang. Monet: Deep motion exploitation for video object segmentation. In _CVPR_, 2018. 
*   Xie et al. [2024] Junyu Xie, Charig Yang, Weidi Xie, and Andrew Zisserman. Moving object segmentation: All you need is sam (and flow). _arXiv preprint arXiv:2404.12389_, 2024. 
*   Xiong et al. [2024] Yunyang Xiong, Bala Varadarajan, Lemeng Wu, Xiaoyu Xiang, Fanyi Xiao, Chenchen Zhu, Xiaoliang Dai, Dilin Wang, Fei Sun, Forrest Iandola, et al. Efficientsam: Leveraged masked image pretraining for efficient segment anything. In _CVPR_, 2024. 
*   Xu et al. [2018a] Ning Xu, Linjie Yang, Yuchen Fan, Jianchao Yang, Dingcheng Yue, Yuchen Liang, Brian Price, Scott Cohen, and Thomas Huang. Youtube-vos: Sequence-to-sequence video object segmentation. In _ECCV_, 2018a. 
*   Xu et al. [2018b] Ning Xu, Linjie Yang, Yuchen Fan, Dingcheng Yue, Yuchen Liang, Jianchao Yang, and Thomas Huang. Youtube-vos: A large-scale video object segmentation benchmark. _arXiv preprint arXiv:1809.03327_, 2018b. 
*   Xu et al. [2022] Xiaohao Xu, Jinglu Wang, Xiao Li, and Yan Lu. Reliable propagation-correction modulation for video object segmentation. In _AAAI_, 2022. 
*   Yang et al. [2018] Linjie Yang, Yanran Wang, Xuehan Xiong, Jianchao Yang, and Aggelos K Katsaggelos. Efficient video object segmentation via network modulation. In _CVPR_, 2018. 
*   Yang and Yang [2022] Zongxin Yang and Yi Yang. Decoupling features in hierarchical propagation for video object segmentation. In _NeurIPS_, 2022. 
*   Yang et al. [2020] Zongxin Yang, Yunchao Wei, and Yi Yang. Collaborative video object segmentation by foreground-background integration. In _ECCV_, 2020. 
*   Yang et al. [2021a] Zongxin Yang, Yunchao Wei, and Yi Yang. Associating objects with transformers for video object segmentation. In _NeurIPS_, 2021a. 
*   Yang et al. [2021b] Zongxin Yang, Yunchao Wei, and Yi Yang. Collaborative video object segmentation by multi-scale foreground-background integration. _IEEE TPAMI_, 2021b. 
*   Yang et al. [2024] Zongxin Yang, Jiaxu Miao, Yunchao Wei, Wenguan Wang, Xiaohan Wang, and Yi Yang. Scalable video object segmentation with identification mechanism. _IEEE TPAMI_, 2024. 
*   Yu et al. [2024] Jieming Yu, An Wang, Wenzhen Dong, Mengya Xu, Mobarakol Islam, Jie Wang, Long Bai, and Hongliang Ren. Sam 2 in robotic surgery: An empirical evaluation for robustness and generalization in surgical video segmentation. _arXiv preprint arXiv:2408.04593_, 2024. 
*   Zhai et al. [2022] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. In _CVPR_, 2022. 
*   Zhang et al. [2023a] Chaoning Zhang, Dongshen Han, Yu Qiao, Jung Uk Kim, Sung-Ho Bae, Seungkyu Lee, and Choong Seon Hong. Faster segment anything: Towards lightweight sam for mobile applications. _arXiv preprint arXiv:2306.14289_, 2023a. 
*   Zhang et al. [2023b] Jiaming Zhang, Yutao Cui, Gangshan Wu, and Limin Wang. Joint modeling of feature, correspondence, and a compressed memory for video object segmentation. _arXiv preprint arXiv:2308.13505_, 2023b. 
*   Zhao et al. [2023] Xu Zhao, Wenchao Ding, Yongqi An, Yinglong Du, Tao Yu, Min Li, Ming Tang, and Jinqiao Wang. Fast segment anything. _arXiv preprint arXiv:2306.12156_, 2023. 
*   Zhou et al. [2023] Chong Zhou, Xiangtai Li, Chen Change Loy, and Bo Dai. Edgesam: Prompt-in-the-loop distillation for on-device deployment of sam. _arXiv preprint arXiv:2312.06660_, 2023.

