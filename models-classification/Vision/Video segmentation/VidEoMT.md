Title: VidEoMT: Your ViT is Secretly Also a Video Segmentation Model

URL Source: https://arxiv.org/html/2602.17807

Markdown Content:
Narges Norouzi 1 Idil Esen Zulfikar 2,† Niccolò Cavagnero 1,† Tommie Kerssies 1

Bastian Leibe 2 Gijs Dubbelman 1 Daan de Geus 1

1 Eindhoven University of Technology 2 RWTH Aachen University

###### Abstract

Existing online video segmentation models typically combine a per-frame segmenter with complex specialized tracking modules. While effective, these modules introduce significant architectural complexity and computational overhead. Recent studies suggest that plain Vision Transformer (ViT) encoders, when scaled with sufficient capacity and large-scale pre-training, can conduct accurate image segmentation without requiring specialized modules. Motivated by this observation, we propose the Video Encoder-only Mask Transformer (VidEoMT), a simple encoder-only video segmentation model that eliminates the need for dedicated tracking modules. To enable temporal modeling in an encoder-only ViT, VidEoMT introduces a lightweight query propagation mechanism that carries information across frames by reusing queries from the previous frame. To balance this with adaptability to new content, it employs a query fusion strategy that combines the propagated queries with a set of temporally-agnostic learned queries. As a result, VidEoMT attains the benefits of a tracker without added complexity, achieving competitive accuracy while being 5×\times–10×\times faster, running at up to 160 FPS with a ViT-L backbone. Code: [https://www.tue-mps.org/videomt/](https://www.tue-mps.org/videomt/).

†††Equal contribution.
1 Introduction
--------------

The video segmentation task involves segmenting and classifying objects in each frame, while also matching those objects across frames. As such, a video segmentation model should have the capability to localize objects, classify them, and track them across different frames. For this reason, great progress has been made through the introduction of specialized neural network components that are designed to improve one or more of these capabilities. Current methods obtain state-of-the-art performance by combining many such specialized components within increasingly complex models[[21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation"), [39](https://arxiv.org/html/2602.17807v1#bib.bib7 "DVIS: Decoupled Video Instance Segmentation Framework"), [40](https://arxiv.org/html/2602.17807v1#bib.bib8 "DVIS++: Improved Decoupled Framework for Universal Video Segmentation"), [41](https://arxiv.org/html/2602.17807v1#bib.bib9 "Improving Video Segmentation via Dynamic Anchor Queries")], building upon years of prior work. This trend of increasing complexity motivates us to explore whether such complexity is necessary, or if this task can be solved with similar accuracy using a simpler approach.

Figure 1: CAVIS vs. VidEoMT (Ours). VidEoMT is much faster than both CAVIS[[21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation")] and a combination of EoMT[[19](https://arxiv.org/html/2602.17807v1#bib.bib1 "Your ViT is Secretly an Image Segmentation Model")] and CAVIS, while maintaining competitive AP across different sizes of DINOv2[[30](https://arxiv.org/html/2602.17807v1#bib.bib33 "DINOv2: Learning Robust Visual Features without Supervision")]. Evaluated on YouTube-VIS 2019 val[[38](https://arxiv.org/html/2602.17807v1#bib.bib55 "Video Instance Segmentation")].

![Image 1: Refer to caption](https://arxiv.org/html/2602.17807v1/x1.png)

Figure 2: Current State-of-the-Art Video Segmentation Methods vs. VidEoMT (Ours). We compare the architectures of current state-of-the-art video segmentation methods – using CAVIS[[21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation")] as a representative example – and our encoder-only VidEoMT method. VidEoMT streamlines the video segmentation framework, relying on the power of large-scale pre-training with vision foundation models rather than handcrafted task-specific components. TF means Transformer and CA means context-aware.

In this paper, we hypothesize that a simpler approach can match the accuracy of more complex models by making use of powerful vision foundation models (VFMs). These VFMs[[33](https://arxiv.org/html/2602.17807v1#bib.bib66 "DINOv3"), [30](https://arxiv.org/html/2602.17807v1#bib.bib33 "DINOv2: Learning Robust Visual Features without Supervision"), [14](https://arxiv.org/html/2602.17807v1#bib.bib67 "EVA-02: A Visual Representation for Neon Genesis")], which typically adopt the Vision Transformer (ViT) architecture[[13](https://arxiv.org/html/2602.17807v1#bib.bib30 "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale")], are pre-trained on large amounts of data, and have proven to be solid foundations for subsequent finetuning for downstream tasks. For video segmentation, VFMs like DINOv2[[30](https://arxiv.org/html/2602.17807v1#bib.bib33 "DINOv2: Learning Robust Visual Features without Supervision")] are incorporated in most recent models[[40](https://arxiv.org/html/2602.17807v1#bib.bib8 "DVIS++: Improved Decoupled Framework for Universal Video Segmentation"), [21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation"), [41](https://arxiv.org/html/2602.17807v1#bib.bib9 "Improving Video Segmentation via Dynamic Anchor Queries")] by being extended with many specialized components. However, we believe that these strong pre-trained ViTs can learn to take over many of the functionalities of the specialized components that are typically added on top, making these components _redundant_. This is inspired by Kerssies _et al_.[[19](https://arxiv.org/html/2602.17807v1#bib.bib1 "Your ViT is Secretly an Image Segmentation Model")], who showed that state-of-the-art image segmentation can be achieved by simply adding a few learnable queries to a large pre-trained ViT with a model called EoMT, without needing specialized components. EoMT demonstrates that a large, pre-trained ViT encoder can learn to effectively localize and classify objects, which is also required for video segmentation. However, video segmentation has an additional requirement: temporally tracking objects across frames.

We expect that the pre-trained ViT encoders from VFMs will also be able to learn to track objects because of the training objectives of these VFMs. For instance, DINO-style models[[2](https://arxiv.org/html/2602.17807v1#bib.bib32 "Emerging Properties in Self-Supervised Vision Transformers"), [30](https://arxiv.org/html/2602.17807v1#bib.bib33 "DINOv2: Learning Robust Visual Features without Supervision"), [33](https://arxiv.org/html/2602.17807v1#bib.bib66 "DINOv3")] employ training objectives that promote consistent feature representations for a given object across different views. Cross-view consistent features are crucial for tracking, as they allow for the identification of the same object in different frames. This makes pre-trained ViTs from VFMs highly suited for video segmentation.

We verify our hypothesis about the redundancy of complex components in video segmentation models by taking state-of-the-art video segmentation models[[40](https://arxiv.org/html/2602.17807v1#bib.bib8 "DVIS++: Improved Decoupled Framework for Universal Video Segmentation"), [21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation"), [41](https://arxiv.org/html/2602.17807v1#bib.bib9 "Improving Video Segmentation via Dynamic Anchor Queries")] and evaluating the effect of removing their specialized modules. These existing models all follow roughly the same paradigm: they first employ a segmenter, which predicts frame-level segmentation masks and class labels and outputs object-level feature queries, and then apply a tracker to match object-level feature queries across different video frames. As shown in [Fig.2](https://arxiv.org/html/2602.17807v1#S1.F2 "In 1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model") (left) for the example of CAVIS [[21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation")], both the segmenter and the tracker consist of many specialized components. We first replace the complex segmenter with EoMT[[15](https://arxiv.org/html/2602.17807v1#bib.bib40 "Gaussian Error Linear Units (GELUs)")], followed by a step-by-step removal of specialized tracking modules.

Next, we move away from the conventional decoupling of segmenter and tracker, and instead explore whether temporal modeling can be conducted within a ViT encoder. To this end, we introduce a lightweight approach based on (1) query propagation, where object-level queries are carried across frames to enable temporal modeling in an encoder-only framework, and (2) query fusion, which combines propagated queries with learned queries to allow the identification of newly appearing objects. This leads to the design of the Video Encoder-only Mask Transformer (VidEoMT), which unifies segmentation and temporal association within the ViT, as illustrated in [Fig.2](https://arxiv.org/html/2602.17807v1#S1.F2 "In 1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model") (right).

By no longer requiring complex specialized components and performing all computations within a single ViT-style model, VidEoMT is remarkably efficient. Through experiments, we find that VidEoMT with a ViT-Large backbone is over 10×10\times faster than existing state-of-the-art methods on the YouTube-VIS[[38](https://arxiv.org/html/2602.17807v1#bib.bib55 "Video Instance Segmentation")] benchmarks, achieving processing speeds of up to 160 FPS, as shown in [Fig.1](https://arxiv.org/html/2602.17807v1#S1.F1 "In 1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). Importantly, this speed is obtained while maintaining a comparable accuracy. These findings are further validated on the VIPSeg[[27](https://arxiv.org/html/2602.17807v1#bib.bib57 "Large-Scale Video Panoptic Segmentation in the Wild: A Benchmark")] and VSPW[[28](https://arxiv.org/html/2602.17807v1#bib.bib56 "VSPW: A Large-scale Dataset for Video Scene Parsing in the Wild")] benchmarks, where VidEoMT consistently achieves speedups of 5×\times–10×\times with negligible impact on accuracy. Such speed-up factors can be a veritable game changer for applications, enabling online video processing across a wide range of use cases. These results also validate our hypothesis that a large, extensively pre-trained ViT can take over the functionalities of specialized components to conduct accurate video segmentation, without requiring additional complex components.

In summary, we make the following contributions:

*   •
We propose VidEoMT, a simple and highly efficient architecture for video segmentation, that unifies segmentation and temporal association within a single ViT encoder.

*   •
Using VidEoMT, we demonstrate that a sufficiently large, pre-trained ViT can learn to take over the functionality of specialized components for video segmentation.

*   •
We show that VidEoMT, with its simple encoder-only architecture, can achieve accuracies comparable to the state of the art while being up to 10×10\times faster.

2 Related Work
--------------

Image Segmentation. Image segmentation requires that objects in an image are segmented and classified. Early image segmentation models treated this task as a per-pixel classification problem, predicting a class label for each pixel[[4](https://arxiv.org/html/2602.17807v1#bib.bib63 "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs"), [5](https://arxiv.org/html/2602.17807v1#bib.bib61 "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"), [23](https://arxiv.org/html/2602.17807v1#bib.bib62 "Fully Convolutional Networks for Semantic Segmentation")]. Later works propose an alternative mask classification approach, where a model predicts a segment – consisting of a segmentation mask and class label – for each object in the image[[9](https://arxiv.org/html/2602.17807v1#bib.bib26 "Per-Pixel Classification is Not All You Need for Semantic Segmentation")]. These mask classification methods typically make use of Mask Transformers, which use image features from a backbone and learnable queries to predict a segmentation mask and class label for each query with a Transformer decoder[[35](https://arxiv.org/html/2602.17807v1#bib.bib64 "MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers"), [8](https://arxiv.org/html/2602.17807v1#bib.bib27 "Masked-attention Mask Transformer for Universal Image Segmentation"), [18](https://arxiv.org/html/2602.17807v1#bib.bib65 "OneFormer: One Transformer To Rule Universal Image Segmentation"), [3](https://arxiv.org/html/2602.17807v1#bib.bib28 "PEM: Prototype-based Efficient MaskFormer for Image Segmentation")]. Recently, EoMT[[19](https://arxiv.org/html/2602.17807v1#bib.bib1 "Your ViT is Secretly an Image Segmentation Model")] has demonstrated that it is possible to conduct accurate image segmentation without a decoder or other task-specific components, by simply feeding the learnable queries directly into a large, pre-trained ViT. In this work, inspired by EoMT, we investigate whether video segmentation models can be simplified in a similar manner, with the goal of improving efficiency while preserving high accuracy.

Video Segmentation. Video segmentation is a well-established computer vision task, encompassing video instance segmentation (VIS)[[38](https://arxiv.org/html/2602.17807v1#bib.bib55 "Video Instance Segmentation")], video panoptic segmentation (VPS)[[20](https://arxiv.org/html/2602.17807v1#bib.bib43 "Video Panoptic Segmentation")], and video semantic segmentation (VSS)[[29](https://arxiv.org/html/2602.17807v1#bib.bib45 "Semantic Video Segmentation by Gated Recurrent Flow Propagation")], where the primary objective is to segment, classify, and track all objects of interest in a video. Current VIS, VPS, and VSS methods typically use Mask Transformer-based architectures[[16](https://arxiv.org/html/2602.17807v1#bib.bib14 "VITA: Video Instance Segmentation via Object Token Association"), [17](https://arxiv.org/html/2602.17807v1#bib.bib16 "MinVIS: A Minimal Video Instance Segmentation Framework without Video-based Training"), [39](https://arxiv.org/html/2602.17807v1#bib.bib7 "DVIS: Decoupled Video Instance Segmentation Framework"), [40](https://arxiv.org/html/2602.17807v1#bib.bib8 "DVIS++: Improved Decoupled Framework for Universal Video Segmentation"), [21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation"), [41](https://arxiv.org/html/2602.17807v1#bib.bib9 "Improving Video Segmentation via Dynamic Anchor Queries"), [37](https://arxiv.org/html/2602.17807v1#bib.bib20 "Mask Propagation for Efficient Video Semantic Segmentation"), [32](https://arxiv.org/html/2602.17807v1#bib.bib22 "Video-kMaX: A Simple Unified Approach for Online and Near-Online Video Panoptic Segmentation"), [7](https://arxiv.org/html/2602.17807v1#bib.bib13 "Mask2Former for Video Instance Segmentation")]. They extend Mask Transformers for image segmentation[[8](https://arxiv.org/html/2602.17807v1#bib.bib27 "Masked-attention Mask Transformer for Universal Image Segmentation")] into the video domain by incorporating specialized tracking components or enhancing temporal representations. The most recent methods[[17](https://arxiv.org/html/2602.17807v1#bib.bib16 "MinVIS: A Minimal Video Instance Segmentation Framework without Video-based Training"), [39](https://arxiv.org/html/2602.17807v1#bib.bib7 "DVIS: Decoupled Video Instance Segmentation Framework"), [40](https://arxiv.org/html/2602.17807v1#bib.bib8 "DVIS++: Improved Decoupled Framework for Universal Video Segmentation"), [21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation"), [41](https://arxiv.org/html/2602.17807v1#bib.bib9 "Improving Video Segmentation via Dynamic Anchor Queries")] are _universal_ models, which can handle VIS, VPS, and VSS within a single framework. These models follow a decoupled paradigm, where the segmentation and tracking sub-tasks are separated. First, a segmenter conducts image segmentation for each frame, and then a tracker associates these segmented objects over time. Generally, both the segmenter and the tracker contain various specialized components, which increase accuracy but reduce efficiency. In this work, we analyze these universal video segmentation models and demonstrate that they can be simplified to an encoder-only design, significantly improving efficiency while achieving competitive accuracy.

3 Method
--------

### 3.1 Task Definition

We consider the task of _online video segmentation_, where the goal is to assign a class label and binary mask to each object in every frame, while also associating predictions of the same object across time steps to ensure temporal consistency. Here, we use the term “object” broadly to refer to either object instances (as in VIS), semantic classes (as in VSS), or both (as in VPS).

Formally, a video is a sequence of T T frames 𝒱={𝐈 1,𝐈 2,…,𝐈 T}\mathcal{V}=\{\mathbf{I}_{1},\mathbf{I}_{2},\dots,\mathbf{I}_{T}\}. For each frame 𝐈 t∈ℝ 3×H×W\mathbf{I}_{t}\in\mathbb{R}^{3\times H\times W} with spatial resolution (H,W)(H,W), a model should yield a set of K t K_{t} predictions 𝒴 t={(𝐦 t,i,c t,i)}i=1 K t\mathcal{Y}_{t}=\{(\mathbf{m}_{t,i},c_{t,i})\}_{i=1}^{K_{t}}, where 𝐦 t,i∈{0,1}H×W\mathbf{m}_{t,i}\in\{0,1\}^{H\times W} is a binary segmentation mask, and c t,i∈{1,…,C}c_{t,i}\in\{1,\dots,C\} is a semantic category label from C C classes. Additionally, these per-frame predictions must be temporally associated across frames to maintain identity consistency. That is, each prediction (𝐦 t,i,c t,i)(\mathbf{m}_{t,i},c_{t,i}) at time t t should be matched to a corresponding prediction in a previous frame at t−1 t-1 if they refer to the same object. The task must be solved in an _online_ manner: at timestep t t, predictions 𝒴 t\mathcal{Y}_{t} may only depend on the current frame 𝐈 t\mathbf{I}_{t} and earlier frames {𝐈 1,…,𝐈 t−1}\{\mathbf{I}_{1},\dots,\mathbf{I}_{t-1}\}.

### 3.2 Preliminaries

Current state-of-the-art online video segmentation models[[39](https://arxiv.org/html/2602.17807v1#bib.bib7 "DVIS: Decoupled Video Instance Segmentation Framework"), [40](https://arxiv.org/html/2602.17807v1#bib.bib8 "DVIS++: Improved Decoupled Framework for Universal Video Segmentation"), [21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation"), [41](https://arxiv.org/html/2602.17807v1#bib.bib9 "Improving Video Segmentation via Dynamic Anchor Queries")] typically decompose the video segmentation pipeline into two distinct components: a segmenter, which is responsible for generating segmentation masks and class labels for each frame, and a tracker that ensures temporal association by linking the segmenter’s predictions across frames, associating object instances over time (see [Fig.2](https://arxiv.org/html/2602.17807v1#S1.F2 "In 1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model")).

Segmenter. The segmenter 𝒮\mathcal{S} generates frame-level image segmentation predictions, yielding a class label and a binary mask for each object. To obtain these predictions, state-of-the-art methods combine a pre-trained ViT[[13](https://arxiv.org/html/2602.17807v1#bib.bib30 "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"), [30](https://arxiv.org/html/2602.17807v1#bib.bib33 "DINOv2: Learning Robust Visual Features without Supervision")], a ViT-Adapter[[6](https://arxiv.org/html/2602.17807v1#bib.bib34 "Vision Transformer Adapter for Dense Predictions")], and a Mask Transformer segmentation decoder[[8](https://arxiv.org/html/2602.17807v1#bib.bib27 "Masked-attention Mask Transformer for Universal Image Segmentation")]. The ViT encoder embeds an input image 𝐈 t\mathbf{I}_{t} into non-overlapping patch tokens and processes them with L L Transformer blocks. A CNN-based ViT-Adapter augments the encoder with multi-scale features, which are fused and refined in the Mask2Former[[8](https://arxiv.org/html/2602.17807v1#bib.bib27 "Masked-attention Mask Transformer for Universal Image Segmentation")] head by a pixel decoder, producing a set of enriched features {𝐅 4,𝐅 8,𝐅 16,𝐅 32}\{\mathbf{F}_{4},\mathbf{F}_{8},\mathbf{F}_{16},\mathbf{F}_{32}\}, with 𝐅 i∈ℝ D×(H/i)×(W/i)\mathbf{F}_{i}\in\mathbb{R}^{D\times(H/i)\times(W/i)}. A Transformer decoder then updates N N learnable queries 𝐐 lrn={𝐪 i lrn∈ℝ D}i=1 N\mathbf{Q}^{\textrm{lrn}}=\{\mathbf{q}^{\textrm{lrn}}_{i}\in\mathbb{R}^{D}\}_{i=1}^{N} through cross- and self-attention, yielding refined queries 𝐐 𝒮={𝐪 i 𝒮∈ℝ D}i=1 N\mathbf{Q}^{\mathcal{S}}=\{\mathbf{q}^{\mathcal{S}}_{i}\in\mathbb{R}^{D}\}_{i=1}^{N}. These refined queries, in combination with the feature maps 𝐅 4\mathbf{F}_{4}, are used to generate the final segmentation outputs: class labels are predicted via a linear layer applied to each query, while binary masks are produced by passing the queries through a three-layer MLP followed by a dot product with the pixel-level feature maps.

Tracker.The objective of the tracker 𝒯\mathcal{T} is to associate the segmenter’s predictions across frames to maintain consistent object identities over time. Rather than relying on the predicted masks and class labels from the segmenter, the tracker performs association based on the per-object query embeddings 𝐐 𝒮\mathbf{Q}^{\mathcal{S}}. Formally, the tracker 𝒯\mathcal{T} aligns the query embeddings from the current frame, 𝐐 t 𝒮\mathbf{Q}^{\mathcal{S}}_{t}, with the temporally updated queries from the previous frame, 𝐐 t−1 𝒯\mathbf{Q}^{\mathcal{T}}_{t-1}, to achieve correspondence between object instances over time, producing temporally updated queries 𝐐 t 𝒯\mathbf{Q}^{\mathcal{T}}_{t}:

𝐐 t 𝒯=𝒯​(𝐐 t 𝒮,𝐐 t−1 𝒯).\mathbf{Q}^{\mathcal{T}}_{t}=\mathcal{T}\!\left(\mathbf{Q}^{\mathcal{S}}_{t},\mathbf{Q}^{\mathcal{T}}_{t-1}\right).(1)

In practice, 𝒯\mathcal{T} consists of L L Transformer blocks with cross-attention, self-attention, and feed-forward layers. During cross-attention, queries 𝐐 t−1 𝒯\mathbf{Q}^{\mathcal{T}}_{t-1} serve as queries and 𝐐 t 𝒮\mathbf{Q}^{\mathcal{S}}_{t} as keys and values, allowing the tracker to align and update the representations of identical objects across consecutive frames. The output of the cross-attention layer is then refined by a self-attention layer, which further enhances temporal coherence among the updated queries. These operations ensure that a consistent query ordering is obtained, meaning that 𝐐 t 𝒯\mathbf{Q}^{\mathcal{T}}_{t} represents the same objects as 𝐐 t−1 𝒯\mathbf{Q}^{\mathcal{T}}_{t-1}, in the same order. The refined queries 𝐐 t 𝒯\mathbf{Q}^{\mathcal{T}}_{t} can then be used to predict temporally consistent masks and class labels, following the same procedure as in the segmenter.

Context-Aware Features. To enrich query embeddings 𝐐 t 𝒮\mathbf{Q}^{\mathcal{S}}_{t} with information from the local neighborhood of each object, the state-of-the-art method CAVIS[[21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation")] introduces _context-aware features_. Concretely, given predicted masks 𝐌 t={𝐦 t,i}i=1 K\mathbf{M}_{t}=\{\mathbf{m}_{t,i}\}_{i=1}^{K} and features 𝐅 4,t\mathbf{F}_{4,t} at timestep t t, binary boundary maps 𝐁 t,i∈{0,1}(H/4)×(W/4)\mathbf{B}_{t,i}\in\{0,1\}^{(H/4)\times(W/4)} are extracted using a Laplacian filter. Next, the features 𝐅 4,t\mathbf{F}_{4,t} are smoothed with an average filter, yielding 𝐅 4,t 𝒜\mathbf{F}^{\mathcal{A}}_{4,t}. Finally, the context-aware features 𝐐 t 𝒜={𝐪 t,i 𝒜∈ℝ D}i=1 N\mathbf{Q}^{\mathcal{A}}_{t}=\{\mathbf{q}^{\mathcal{A}}_{t,i}\in\mathbb{R}^{D}\}_{i=1}^{N}, extracted by pooling the smoothed features at boundary pixels, are concatenated with the per-frame query embeddings 𝐐 t 𝒮\mathbf{Q}^{\mathcal{S}}_{t}. This process produces an enriched set of queries 𝐐 t 𝒞={𝐪 t,i 𝒞∈ℝ 2​D}i=1 N\mathbf{Q}^{\mathcal{C}}_{t}=\{\mathbf{q}^{\mathcal{C}}_{t,i}\in\mathbb{R}^{2D}\}_{i=1}^{N}, which are then fed into the tracker’s Transformer blocks in place of the segmenter queries 𝐐 t 𝒮\mathbf{Q}^{\mathcal{S}}_{t}.

Re-identification Layers. To further improve robustness, recent methods[[40](https://arxiv.org/html/2602.17807v1#bib.bib8 "DVIS++: Improved Decoupled Framework for Universal Video Segmentation"), [41](https://arxiv.org/html/2602.17807v1#bib.bib9 "Improving Video Segmentation via Dynamic Anchor Queries"), [21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation")] employ _re-identification layers_. These layers are paired with contrastive training objectives to enforce similarity between embeddings of the same instance while separating those of different instances. In practice, query embeddings 𝐐 t 𝒮\mathbf{Q}^{\mathcal{S}}_{t} from the segmenter are usually fed to the re-identification layers, implemented as a 3-layer MLP. In CAVIS, this MLP is instead applied to the context-aware queries 𝐐 t 𝒞\mathbf{Q}^{\mathcal{C}}_{t}:

𝐐 t ℛ=MLP​(𝐐 t 𝒞).\mathbf{Q}^{\mathcal{R}}_{t}=\texttt{MLP}\!\left(\mathbf{Q}^{\mathcal{C}}_{t}\right).(2)

This yields enhanced queries 𝐐 t ℛ\mathbf{Q}^{\mathcal{R}}_{t} which are subjected to contrastive learning and are fed into the tracker’s Transformer blocks, in place of the context-aware queries 𝐐 t 𝒞\mathbf{Q}^{\mathcal{C}}_{t}.

![Image 2: Refer to caption](https://arxiv.org/html/2602.17807v1/x2.png)

Figure 3: VidEoMT architecture. For the initial video frame at t=0 t=0, learnable queries are concatenated to the patch tokens after the first L 1 L_{1} ViT blocks. Both sets of tokens are then jointly processed in the last L 2 L_{2} blocks, outputting predictions and track queries. For successive frames, learnable queries and previous-frame track queries are fed to the query fusion module before being processed by the ViT blocks.

### 3.3 Removing Task-specific Components

Recently, EoMT[[19](https://arxiv.org/html/2602.17807v1#bib.bib1 "Your ViT is Secretly an Image Segmentation Model")] has challenged the dominant paradigm in image segmentation that relies on many specialized components, showing that this task can be performed in an encoder-only fashion, given a sufficiently large ViT model and strong pre-training. This is also relevant in the video domain, as the segmenter modules of state-of-the-art video segmentation models also use the same specialized components. In EoMT[[19](https://arxiv.org/html/2602.17807v1#bib.bib1 "Your ViT is Secretly an Image Segmentation Model")], learned queries 𝐐 lrn\mathbf{Q}^{\textrm{lrn}} are injected into the last L 2 L_{2} (usually L 2=4 L_{2}=4) layers of a ViT encoder and processed jointly with patch tokens, yielding updated queries 𝐐 𝒮\mathbf{Q}^{\mathcal{S}} which can be used to produce segmentation predictions {(𝐜 i,𝐦 i)}i=1 K\{(\mathbf{c}_{i},\mathbf{m}_{i})\}_{i=1}^{K}. Despite its simplicity, EoMT performs competitively with complex frameworks while greatly improving efficiency.

Inspired by this result, we explore a similar simplification for video segmentation, where inference speed is even more critical. Our hypothesis is that a strong ViT can handle not only segmentation but also temporal association within a unified _encoder-only_ architecture, removing the need for explicit tracking modules. To verify this, starting from the state-of-the-art CAVIS model, we first replace its heavy segmenter with EoMT, and then we progressively remove video-specific components to evaluate whether the encoder can also learn to conduct temporal association.

Replacing the Segmenter. In the current state-of-the-art video segmentation models, such as CAVIS[[21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation")], the segmenter 𝒮\mathcal{S} is composed of an inefficient ViT-Adapter[[6](https://arxiv.org/html/2602.17807v1#bib.bib34 "Vision Transformer Adapter for Dense Predictions")] and a complex and resource-intensive Mask2Former[[8](https://arxiv.org/html/2602.17807v1#bib.bib27 "Masked-attention Mask Transformer for Universal Image Segmentation")] pixel decoder and Transformer decoder. We replace the entire segmenter with EoMT[[19](https://arxiv.org/html/2602.17807v1#bib.bib1 "Your ViT is Secretly an Image Segmentation Model")], which integrates query tokens directly into the ViT and predicts object representations without specialized components. This greatly simplifies the pipeline and it is expected to consistently improve inference speed, similar to the original findings for EoMT[[19](https://arxiv.org/html/2602.17807v1#bib.bib1 "Your ViT is Secretly an Image Segmentation Model")].

Removing Context‐Aware Features. The context-aware features in CAVIS[[21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation")] explicitly encode information from the spatial neighborhood of each instance to stabilize predictions under appearance changes or occlusion. Extracting these features requires convolutional filtering over high-resolution features, repeated for every query in all frames of a video, making it inefficient. We hypothesize that the auxiliary context added by these features is not strictly necessary when leveraging a strong pre-trained ViT, as its features are already fine-grained enough to be easily fine-tuned to capture specific object identity and to maintain stability under appearance changes or occlusion.

Removing Re‐identification Layers. While effective, re-identification layers add complexity at both inference and training time, where the associated contrastive losses are memory-intensive and slow to optimize. We hypothesize that with large-scale pre-training, the features of the ViT encoder already contain rich instance-level information. Since the segmentation queries explicitly cross-attend to these features, they effectively inherit this instance-discriminative knowledge and preserve it across frames. Therefore, we can eliminate these layers, to not only simplify the whole pipeline but also make training more affordable.

### 3.4 VidEoMT

After the previously described simplifications, the model consists of EoMT[[19](https://arxiv.org/html/2602.17807v1#bib.bib1 "Your ViT is Secretly an Image Segmentation Model")] as the segmenter combined with a simplified tracker 𝒯\mathcal{T}. The tracker ensures that a given object is represented by the same query index across frames, preserving temporal consistency. However, this comes at the cost of considerable architectural complexity and significant computational overhead.

We hypothesize that strong pre-training, _e.g_., with DINOv2[[30](https://arxiv.org/html/2602.17807v1#bib.bib33 "DINOv2: Learning Robust Visual Features without Supervision")], already equips the ViT encoder with representations strong enough to enable temporal association within the encoder itself with only minimal changes, without the need for specialized tracking components. Hence, we move away from the conventional decoupling of segmenter and tracker and adopt a unified encoder-only design.

Enabling temporal modeling within an encoder-only framework presents two key challenges: (i) effectively integrating information from the previous frame to maintain temporal continuity, and (ii) preserving the model’s ability to detect and recognize newly appearing objects. To address the first challenge, we introduce a query propagation mechanism that carries object-level information across frames, enabling temporal continuity within the encoder. To handle the second challenge, we propose a lightweight query fusion strategy that combines propagated queries with learnable ones, allowing the model to better detect newly appearing objects. The resulting model, which we name Video Encoder-only Mask Transformer (VidEoMT), performs temporal association without specialized tracking components, as visualized in [Fig.3](https://arxiv.org/html/2602.17807v1#S3.F3 "In 3.2 Preliminaries ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model").

Query Propagation. When the tracker is entirely removed, the model reduces to a purely image-level EoMT architecture that processes each frame independently. In this case, queries 𝐐 𝒮\mathbf{Q}^{\mathcal{S}} are the model’s output, and there are no queries 𝐐 𝒯\mathbf{Q}^{\mathcal{T}}, as there is no longer a tracker.

We reintroduce temporal modeling through query propagation. At timestep t=0 t=0, we follow the standard EoMT setup and feed learnable queries 𝐐 lrn\mathbf{Q}^{\textrm{lrn}} into the last L 2 L_{2} layers of the ViT to produce object query embeddings 𝐐 0 𝒮\mathbf{Q}^{\mathcal{S}}_{0} and the corresponding segmentation predictions. At subsequent timesteps t>0 t>0, instead of reusing the learnable queries, we use the track queries, _i.e_., the propagated queries from the previous frame 𝐐 t−1 𝒮\mathbf{Q}^{\mathcal{S}}_{t-1}, as input to the last L 2 L_{2} layers of the ViT. During these timesteps, the segmentation procedure remains identical to that of EoMT, the only difference is that the propagated queries replace the learnable ones.

This strategy enables information to flow across time without additional computational cost per frame, allowing for temporal consistency across frames. However, since the ViT only receives information from the previous frame, the influence of the learnable queries 𝐐 lrn\mathbf{Q}^{\textrm{lrn}} gradually diminishes, causing the model to lose the ability to recognize newly appearing objects in the video.

Query Fusion. To address this limitation, we introduce query fusion, illustrated in Figure[3](https://arxiv.org/html/2602.17807v1#S3.F3 "Figure 3 ‣ 3.2 Preliminaries ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). In this design, queries from the previous frame 𝐐 t−1 𝒮\mathbf{Q}^{\mathcal{S}}_{t-1} are first transformed by a lightweight linear layer and then combined with the original learned queries 𝐐 lrn\mathbf{Q}^{\textrm{lrn}} through element-wise addition:

𝐐 t ℱ=Linear​(𝐐 t−1 𝒮)+𝐐 lrn.\mathbf{Q}_{t}^{\mathcal{F}}=\texttt{Linear}\!\left(\mathbf{Q}^{\mathcal{S}}_{t-1}\right)+\mathbf{Q}^{\textrm{lrn}}.(3)

The element-wise addition is possible because the supervision strategy guarantees that the query order remains the same across frames. This fusion ensures that the model has access to temporal context from the past through 𝐐 t−1 𝒮\mathbf{Q}^{\mathcal{S}}_{t-1}, as well as learnable queries 𝐐 lrn\mathbf{Q}^{\textrm{lrn}} to enable adaptability to new objects. By balancing temporal context propagation and adaptability to new objects, query fusion allows our encoder-only framework to conduct accurate object tracking with negligible additional architectural complexity.

Training. VidEoMT is trained using the same objective function as Mask2Former[[8](https://arxiv.org/html/2602.17807v1#bib.bib27 "Masked-attention Mask Transformer for Universal Image Segmentation")]. We use the cross-entropy loss for classification and the binary cross-entropy and Dice losses for segmentation predictions. To ensure temporally consistent supervision, we follow the ground-truth matching strategy of DVIS++[[40](https://arxiv.org/html/2602.17807v1#bib.bib8 "DVIS++: Improved Decoupled Framework for Universal Video Segmentation")]. Here, a ground-truth object is only matched to a query in the frame where the object first appears. In the remaining frames, the ground-truth object stays matched to this query, ensuring temporal consistency.

4 Experiments
-------------

Datasets and Evaluation Metrics. We evaluate VidEoMT on six major benchmarks for video segmentation: OVIS[[31](https://arxiv.org/html/2602.17807v1#bib.bib54 "Occluded Video Instance Segmentation: A Benchmark")] and YouTube-VIS 2019, 2021, and 2022[[38](https://arxiv.org/html/2602.17807v1#bib.bib55 "Video Instance Segmentation")] for Video Instance Segmentation (VIS), VIPSeg[[27](https://arxiv.org/html/2602.17807v1#bib.bib57 "Large-Scale Video Panoptic Segmentation in the Wild: A Benchmark")] for Video Panoptic Segmentation (VPS), and VSPW[[28](https://arxiv.org/html/2602.17807v1#bib.bib56 "VSPW: A Large-scale Dataset for Video Scene Parsing in the Wild")] for Video Semantic Segmentation (VSS). We adopt Average Precision (AP) and Average Recall (AR) metrics[[38](https://arxiv.org/html/2602.17807v1#bib.bib55 "Video Instance Segmentation")] for VIS, Video Panoptic Quality (VPQ)[[20](https://arxiv.org/html/2602.17807v1#bib.bib43 "Video Panoptic Segmentation")] and Segmentation and Tracking Quality (STQ)[[36](https://arxiv.org/html/2602.17807v1#bib.bib58 "STEP: Segmenting and Tracking Every Pixel")] for VPS, and mean IoU (mIoU) and Video Consistency (mVC)[[28](https://arxiv.org/html/2602.17807v1#bib.bib56 "VSPW: A Large-scale Dataset for Video Scene Parsing in the Wild")] for VSS.

Implementation Details. Similar to the state-of-the-art models CAVIS[[21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation")] and DVIS-DAQ[[41](https://arxiv.org/html/2602.17807v1#bib.bib9 "Improving Video Segmentation via Dynamic Anchor Queries")], we use a DINOv2-pretrained ViT[[30](https://arxiv.org/html/2602.17807v1#bib.bib33 "DINOv2: Learning Robust Visual Features without Supervision")] as the default backbone of VidEoMT. We adopt a batch size of 8 with 5 frames as a temporal window, using mixed precision and the AdamW optimizer[[24](https://arxiv.org/html/2602.17807v1#bib.bib46 "Decoupled Weight Decay Regularization")] with a learning rate of 10−4 10^{-4}. Following EoMT[[19](https://arxiv.org/html/2602.17807v1#bib.bib1 "Your ViT is Secretly an Image Segmentation Model")], we apply layer-wise learning rate decay (LLRD)[[12](https://arxiv.org/html/2602.17807v1#bib.bib47 "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding")] with a factor of 0.6 and a polynomial learning rate decay with a power of 0.9. The number of iterations and training video resolutions follow the settings of CAVIS[[21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation")] for fair comparison. We refer to the supplementary material for additional implementation details.

To assess the computational efficiency, we measure both FPS and FLOPs. FPS is reported as the average number of video frames processed per second on the validation set with a batch size of 1, evaluated on an NVIDIA H100 GPU with FlashAttention v2[[10](https://arxiv.org/html/2602.17807v1#bib.bib51 "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning")] and torch.compile[[1](https://arxiv.org/html/2602.17807v1#bib.bib49 "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation")] (default settings) enabled. FLOPs are calculated using fvcore[[26](https://arxiv.org/html/2602.17807v1#bib.bib48 "fvcore")], averaging over all images in the validation set. We report the results in GFLOPs, _i.e_., FLOPs ×10 9\times 10^{9}.

5 Results
---------

### 5.1 Main Results

From CAVIS to VidEoMT.In[Tab.1](https://arxiv.org/html/2602.17807v1#S5.T1 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), we report a stepwise transformation from state-of-the-art video segmentation method CAVIS[[21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation")] to our proposed VidEoMT. We gradually remove specialized tracking components to obtain the lightweight EoMT baseline, and we then introduce modifications to EoMT to support tracking. For details of the architectures at different steps, see the supplementary.

In step (1), we find that replacing the segmenter with EoMT[[19](https://arxiv.org/html/2602.17807v1#bib.bib1 "Your ViT is Secretly an Image Segmentation Model")] improves FPS by almost 3×\times, while AP drops by only 0.8. In steps (2)–(3), we observe that removing context-aware features and the re-identification layers further increases speed by 1.8×\times to 74 FPS, with almost no impact on accuracy. These results demonstrate that the DINOv2 ViT encoder can take over the functionality of these components without degrading performance. In step (4), we note that the elimination of the tracker, which results in the naive, per-frame application of EoMT, yields a speedup of more than 10×\times to 162 FPS compared to CAVIS’s 15 FPS, but also leads to a substantial 7.6 AP drop. Interestingly, though, even without any tracking modules and just relying on the queries, the model still retains reasonable accuracy. This shows that EoMT can learn to output objects in a somewhat consistent order across frames, despite processing them independently without temporal interaction.

Step Method AP Params GFLOPs FPS
(0)CAVIS[[21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation")]68.9 358M 838 15
(1)w/ EoMT as Segmenter 68.1 328M 699 42
(2)w/o Context-aware Features 68.4 327M 581 72
(3)w/o Re-identification Layers 68.0 326M 580 74
(4)w/o Tracker = EoMT 61.3 316M 565 162
–EoMT[[19](https://arxiv.org/html/2602.17807v1#bib.bib1 "Your ViT is Secretly an Image Segmentation Model")]61.3 316M 565 162
(5)w/ Query Propagation 63.9 316M 565 162
(6)w/ Query Fusion = VidEoMT 68.6 318M 566 160

Table 1: From CAVIS to VidEoMT. Stepwise removal of CAVIS modules toward EoMT, and modifications extending it to our VidEoMT. Evaluated on YouTube-VIS 2019 val[[38](https://arxiv.org/html/2602.17807v1#bib.bib55 "Video Instance Segmentation")].

Table 2: Online VIS on YouTube-VIS 2019 and 2021.

Table 3: Online VIS on YouTube-VIS 2022 and OVIS.†Input resolution of 544 (shortest image side) for OVIS, default for others.

Applying query propagation in step(5) is necessary to introduce temporal modeling in EoMT, improving the AP by +2.6 without increasing the computational cost. However, we find that the model struggles with identifying newly appearing objects over time. In the final step(6), we show that query fusion allows VidEoMT to recover nearly all of the original accuracy, while achieving a speedup of more than 10×\times compared to CAVIS. Notably, the gain in inference speed is much larger than in FLOPs. This is the case because VidEoMT almost purely consists of a plain ViT. As such, it can better leverage dedicated hardware and software optimizations for the Transformer architecture without being bottlenecked by inefficient specialized modules[[19](https://arxiv.org/html/2602.17807v1#bib.bib1 "Your ViT is Secretly an Image Segmentation Model")].

Overall, these results show that VidEoMT achieves an excellent balance between accuracy and efficiency, as heavy modules in CAVIS can be safely removed, while our encoder-only framework effectively restores performance with negligible computational cost. Moreover, these results confirm our hypothesis that a VFM-pretrained ViT can be trained to conduct both segmentation and tracking within the same encoder, without complex tracking modules.

### 5.2 Comparison with State-of-the-Art Models

Video Instance Segmentation (VIS). We first compare VidEoMT with state-of-the-art VIS models across four datasets. The results, reported in [Tabs.2](https://arxiv.org/html/2602.17807v1#S5.T2 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model") and[3](https://arxiv.org/html/2602.17807v1#S5.T3 "Table 3 ‣ 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), demonstrate that VidEoMT consistently outperforms DVIS[[39](https://arxiv.org/html/2602.17807v1#bib.bib7 "DVIS: Decoupled Video Instance Segmentation Framework")] and DVIS++[[40](https://arxiv.org/html/2602.17807v1#bib.bib8 "DVIS++: Improved Decoupled Framework for Universal Video Segmentation")], while being 5×\times–8×\times faster. Compared to DVIS-DAQ[[41](https://arxiv.org/html/2602.17807v1#bib.bib9 "Improving Video Segmentation via Dynamic Anchor Queries")], VidEoMT is over 14×\times faster while achieving higher accuracy on all benchmarks except OVIS, where the gap is within 2 AP points. Similarly, VidEoMT surpasses CAVIS on YouTube-VIS 2022, and achieves comparable accuracy on YouTube-VIS 2019 and OVIS, and remains within 2 AP on YouTube-VIS 2021, while being 7×\times to more than 10×\times faster. Finally, we note that VidEoMT is also both faster and more accurate than MinVIS[[17](https://arxiv.org/html/2602.17807v1#bib.bib16 "MinVIS: A Minimal Video Instance Segmentation Framework without Video-based Training")], which was specifically designed for efficiency and simplicity. Overall, VidEoMT demonstrates a significantly superior trade-off between accuracy and efficiency compared to existing approaches.

Table 4: Online VPS on VIPSeg.

Table 5: Online VSS on VSPW.

Video Panoptic Segmentation (VPS).[Tab.4](https://arxiv.org/html/2602.17807v1#S5.T4 "In 5.2 Comparison with State-of-the-Art Models ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model") compares VidEoMT with state-of-the-art methods for the VPS task on the VIPSeg benchmark. VidEoMT incurs only a minor VPQ drop compared to DVIS++ and CAVIS, while running 5×\times–7×\times faster. Compared to DVIS-DAQ, which obtains the highest VPQ of 57.4 but runs at the lowest FPS of 4, VidEoMT sacrifices just 2.2 VPQ while delivering nearly 19×\times higher speed. These results confirm that VidEoMT also provides a significantly better accuracy and efficiency balance for video panoptic segmentation.

Video Semantic Segmentation (VSS).[Tab.5](https://arxiv.org/html/2602.17807v1#S5.T5 "In 5.2 Comparison with State-of-the-Art Models ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model") compares VidEoMT with state-of-the-art VSS methods on the VSPW benchmark. VidEoMT outperforms existing methods, improving the mIoU by +2.1 compared to DVIS++ while also achieving a higher temporal consistency with +0.8 mVC 16 and being more than 5×\times faster. These results confirm the general applicability and strength of VidEoMT on yet another video segmentation task.

### 5.3 Further Analyses

EoMT as a Segmenter. In this work, we propose a unified architecture that performs video segmentation in an encoder-only fashion. Alternatively, one could consider augmenting EoMT with recent trackers. In [Tab.6](https://arxiv.org/html/2602.17807v1#S5.T6 "In 5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), we compare VidEoMT to alternative approaches where EoMT is used as a segmenter and existing trackers are applied on top. Compared to the best alternative approach, EoMT + CAVIS, VidEoMT achieves slightly better AP while being nearly 4×4\times faster. These results show that our VidEoMT is not only more streamlined but also considerably faster and even more accurate than alternative strategies.

Query Propagation. In VidEoMT, we directly propagate object queries into the ViT encoder. To verify that this is just as effective as propagating them into a separate decoder, we take a DINOv2 + ViT-Adapter encoder and Mask2Former decoder[[8](https://arxiv.org/html/2602.17807v1#bib.bib27 "Masked-attention Mask Transformer for Universal Image Segmentation")], and apply query propagation into the decoder. We evaluate two propagation variants: TrackFormer[[25](https://arxiv.org/html/2602.17807v1#bib.bib36 "TrackFormer: Multi-Object Tracking with Transformers")], which was originally introduced for object tracking rather than video segmentation, and our query fusion approach that combines propagated queries with learned queries. The results in [Tab.7](https://arxiv.org/html/2602.17807v1#S5.T7 "In 5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model") show that our encoder-only approach achieves an AP score comparable to the encoder–decoder design, validating the effectiveness of the proposed encoder-only architecture. Furthermore, our query fusion strategy slightly improves AP over TrackFormer[[25](https://arxiv.org/html/2602.17807v1#bib.bib36 "TrackFormer: Multi-Object Tracking with Transformers")] while also being significantly faster. Overall, VidEoMT is just as effective but considerably faster than both alternative approaches. See the supplementary material for additional experiments on temporal propagation and more details on these alternative methods.

Impact of Pre-training. In this work, we hypothesize that large-scale pre-training with VFMs enables the ViT encoder in VidEoMT to take over the functionalities of specialized components. To verify this hypothesis, in [Tab.8](https://arxiv.org/html/2602.17807v1#S5.T8 "In 5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), we compare the performance of VidEoMT and CAVIS using large-scale pre-training with DINOv3[[33](https://arxiv.org/html/2602.17807v1#bib.bib66 "DINOv3")], DINOv2[[30](https://arxiv.org/html/2602.17807v1#bib.bib33 "DINOv2: Learning Robust Visual Features without Supervision")], and EVA-02[[14](https://arxiv.org/html/2602.17807v1#bib.bib67 "EVA-02: A Visual Representation for Neon Genesis")], as well as medium-scale ImageNet-21K and small-scale ImageNet-1K pre-training[[34](https://arxiv.org/html/2602.17807v1#bib.bib39 "DeiT III: Revenge of the ViT"), [11](https://arxiv.org/html/2602.17807v1#bib.bib38 "ImageNet: A Large-Scale Hierarchical Image Database")]. We observe that, with strong pre-training (DINOv2, DINOv3, EVA-02), VidEoMT attains performance comparable to CAVIS, while the performance gap increases in favor of CAVIS as the pre-training scale decreases. Notably, DINOv3 offers only marginal improvements over DINOv2, which we attribute to DINOv3 being designed to be kept frozen rather than fine-tuned. Nevertheless, VidEoMT remains highly effective under DINOv3 pre-training. Overall, these results support our hypothesis that large-scale pre-training is necessary to unleash the potential of VidEoMT. While EoMT[[19](https://arxiv.org/html/2602.17807v1#bib.bib1 "Your ViT is Secretly an Image Segmentation Model")] showed this effect for image segmentation, our results demonstrate that large-scale pre-training also enables the ViT encoder to take over the functionalities of specialized video segmentation components.

Table 6: Alternative approaches: EoMT as a segmenter. Comparison of EoMT equipped with state-of-the-art trackers and our proposed VidEoMT. Evaluated on YouTube-VIS 2019 val.

Table 7: Alternative approaches: Query propagation in the decoder. Comparison of ViT-Adapter + Mask2Former (M2F) equipped with TrackFormer or our query fusion strategy and the proposed VidEoMT. All methods use a ViT-L backbone with DINOv2 pre-training. Evaluated on YouTube-VIS 2019 val.

Table 8: Impact of pre-training. VidEoMT performs better with larger-scale pre-training. Evaluated on YouTube-VIS 2019 val.

Table 9: Impact of model size. VidEoMT performs better as ViT[[13](https://arxiv.org/html/2602.17807v1#bib.bib30 "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale")] size increases. Evaluated on YouTube-VIS 2019 val.

Impact of Model Size. Similarly, we hypothesize that increased model size positively impacts the ViT’s ability to conduct segmentation and tracking. In [Tab.9](https://arxiv.org/html/2602.17807v1#S5.T9 "In 5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), we assess this by evaluating CAVIS and VidEoMT for ViT model sizes L, B, and S. The results show that the gap between the CAVIS baseline and VidEoMT decreases as model size increases, confirming our hypothesis. Additionally, while there is a moderate performance gap between CAVIS and VidEoMT for smaller model sizes, VidEoMT with a large ViT-L backbone is still an order of magnitude faster than CAVIS with a small ViT-S backbone, while being much more accurate. This further highlights the effectiveness of VidEoMT.

6 Conclusion
------------

We have introduced VidEoMT, an encoder-only video segmentation architecture that unifies segmentation and temporal association within a single ViT encoder. Through a step-by-step reduction of prior models, we showed that heavy specialized modules are no longer required. We replace them with a lightweight query propagation method, enhanced with an efficient query fusion mechanism. This design achieves an order-of-magnitude speedup while preserving or improving accuracy across multiple video segmentation benchmarks. Overall, our findings suggest that a sufficiently large and well-pretrained ViT can take over much of the functionality that was previously handled by complex downstream components in video segmentation. We hope that this work can serve as a foundation for applications with strict efficiency requirements.

#### Acknowledgments

This work was partly funded by the EU project MODI (grant no. 101076810), the KDT JU project EdgeAI (grant no. 101097300), and the BMFTR project WestAI (grant nos. 01IS22094D and 16IS22094D). The experiments utilized both the Dutch national infrastructure, supported by the SURF Cooperative under grant nos. EINF-14337, EINF-11307, and EINF-15136 and funded by the Dutch Research Council (NWO), and the computing resources provided by the Gauss Centre for Supercomputing e.V. through the John von Neumann Institute for Computing (NIC) on the GCS supercomputer JUWELS at Jülich Supercomputing Centre.

References
----------

*   [1]J. Ansel, E. Yang, H. He, N. Gimelshein, A. Jain, M. Voznesensky, B. Bao, P. Bell, D. Berard, E. Burovski, G. Chauhan, A. Chourdia, W. Constable, A. Desmaison, Z. DeVito, E. Ellison, W. Feng, J. Gong, M. Gschwind, B. Hirsh, S. Huang, K. Kalambarkar, L. Kirsch, M. Lazos, M. Lezcano, Y. Liang, J. Liang, Y. Lu, C. K. Luk, B. Maher, et al. (2024)PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation. In ASPLOS, Cited by: [§A.2](https://arxiv.org/html/2602.17807v1#A1.SS2.p1.1 "A.2 Evaluation ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§4](https://arxiv.org/html/2602.17807v1#S4.p3.1 "4 Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [2] (2021)Emerging Properties in Self-Supervised Vision Transformers. In ICCV, Cited by: [§1](https://arxiv.org/html/2602.17807v1#S1.p3.1 "1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [3]N. Cavagnero, G. Rosi, C. Cuttano, F. Pistilli, M. Ciccone, G. Averta, and F. Cermelli (2024)PEM: Prototype-based Efficient MaskFormer for Image Segmentation. In CVPR, Cited by: [§2](https://arxiv.org/html/2602.17807v1#S2.p1.1 "2 Related Work ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [4]L. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille (2018)DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. IEEE TPAMI 40 (4),  pp.834–848. Cited by: [§2](https://arxiv.org/html/2602.17807v1#S2.p1.1 "2 Related Work ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [5]L. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam (2018)Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. In ECCV, Cited by: [§2](https://arxiv.org/html/2602.17807v1#S2.p1.1 "2 Related Work ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [6]Z. Chen, Y. Duan, W. Wang, J. He, T. Lu, J. Dai, and Y. Qiao (2023)Vision Transformer Adapter for Dense Predictions. In ICLR, Cited by: [§A.3](https://arxiv.org/html/2602.17807v1#A1.SS3.p1.1 "A.3 Visualizations of Model Configurations ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§A.5](https://arxiv.org/html/2602.17807v1#A1.SS5.p1.1 "A.5 Architectures of Alternative Approaches ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§3.2](https://arxiv.org/html/2602.17807v1#S3.SS2.p2.9 "3.2 Preliminaries ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§3.3](https://arxiv.org/html/2602.17807v1#S3.SS3.p3.1 "3.3 Removing Task-specific Components ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [7]B. Cheng, A. Choudhuri, I. Misra, A. Kirillov, R. Girdhar, and A. G. Schwing (2021)Mask2Former for Video Instance Segmentation. arXiv preprint arXiv:2112.10764. Cited by: [§2](https://arxiv.org/html/2602.17807v1#S2.p2.1 "2 Related Work ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [8]B. Cheng, I. Misra, A. G. Schwing, A. Kirillov, and R. Girdhar (2022)Masked-attention Mask Transformer for Universal Image Segmentation. In CVPR, Cited by: [§A.3](https://arxiv.org/html/2602.17807v1#A1.SS3.p1.1 "A.3 Visualizations of Model Configurations ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§A.4](https://arxiv.org/html/2602.17807v1#A1.SS4.p3.3 "A.4 Hyperparameters ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§A.4](https://arxiv.org/html/2602.17807v1#A1.SS4.p3.4 "A.4 Hyperparameters ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§A.5](https://arxiv.org/html/2602.17807v1#A1.SS5.p1.1 "A.5 Architectures of Alternative Approaches ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§2](https://arxiv.org/html/2602.17807v1#S2.p1.1 "2 Related Work ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§2](https://arxiv.org/html/2602.17807v1#S2.p2.1 "2 Related Work ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§3.2](https://arxiv.org/html/2602.17807v1#S3.SS2.p2.9 "3.2 Preliminaries ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§3.3](https://arxiv.org/html/2602.17807v1#S3.SS3.p3.1 "3.3 Removing Task-specific Components ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§3.4](https://arxiv.org/html/2602.17807v1#S3.SS4.p8.1 "3.4 VidEoMT ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§5.3](https://arxiv.org/html/2602.17807v1#S5.SS3.p2.1 "5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 7](https://arxiv.org/html/2602.17807v1#S5.T7.2.2.1.2 "In 5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 7](https://arxiv.org/html/2602.17807v1#S5.T7.2.3.2.2 "In 5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [9]B. Cheng, A. Schwing, and A. Kirillov (2021)Per-Pixel Classification is Not All You Need for Semantic Segmentation. In NeurIPS, Cited by: [§2](https://arxiv.org/html/2602.17807v1#S2.p1.1 "2 Related Work ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [10]T. Dao (2024)FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. In ICLR, Cited by: [§A.2](https://arxiv.org/html/2602.17807v1#A1.SS2.p1.1 "A.2 Evaluation ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§4](https://arxiv.org/html/2602.17807v1#S4.p3.1 "4 Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [11]J. Deng, W. Dong, R. Socher, L. Li, K. Li, and L. Fei-Fei (2009)ImageNet: A Large-Scale Hierarchical Image Database. In CVPR, Cited by: [§5.3](https://arxiv.org/html/2602.17807v1#S5.SS3.p3.1 "5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [12]J. Devlin, M. Chang, K. Lee, and K. Toutanova (2019)BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In NAACL, Cited by: [§A.4](https://arxiv.org/html/2602.17807v1#A1.SS4.p2.1 "A.4 Hyperparameters ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§4](https://arxiv.org/html/2602.17807v1#S4.p2.1 "4 Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [13]A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby (2021)An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In ICLR, Cited by: [§A.1](https://arxiv.org/html/2602.17807v1#A1.SS1.p1.1 "A.1 Training ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table B](https://arxiv.org/html/2602.17807v1#A2.T2 "In B.2 Impact of Model Size ‣ Appendix B Additional Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table B](https://arxiv.org/html/2602.17807v1#A2.T2.12.2.1 "In B.2 Impact of Model Size ‣ Appendix B Additional Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§1](https://arxiv.org/html/2602.17807v1#S1.p2.1 "1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§3.2](https://arxiv.org/html/2602.17807v1#S3.SS2.p2.9 "3.2 Preliminaries ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 9](https://arxiv.org/html/2602.17807v1#S5.T9 "In 5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 9](https://arxiv.org/html/2602.17807v1#S5.T9.23.2.1 "In 5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [14]Y. Fang, Q. Sun, X. Wang, T. Huang, X. Wang, and Y. Cao (2024)EVA-02: A Visual Representation for Neon Genesis. Image and Vision Computing. Cited by: [§B.3](https://arxiv.org/html/2602.17807v1#A2.SS3.p1.1 "B.3 Impact of Pre-training ‣ Appendix B Additional Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§1](https://arxiv.org/html/2602.17807v1#S1.p2.1 "1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§5.3](https://arxiv.org/html/2602.17807v1#S5.SS3.p3.1 "5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [15]D. Hendrycks and K. Gimpel (2016)Gaussian Error Linear Units (GELUs). arXiv preprint arXiv:1606.08415. Cited by: [§1](https://arxiv.org/html/2602.17807v1#S1.p4.1 "1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [16]M. Heo, S. Hwang, S. W. Oh, J. Lee, and S. J. Kim (2022)VITA: Video Instance Segmentation via Object Token Association. In NeurIPS, Cited by: [§2](https://arxiv.org/html/2602.17807v1#S2.p2.1 "2 Related Work ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [17]D. Huang, Z. Yu, and A. Anandkumar (2022)MinVIS: A Minimal Video Instance Segmentation Framework without Video-based Training. In NeurIPS, Cited by: [§2](https://arxiv.org/html/2602.17807v1#S2.p2.1 "2 Related Work ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§5.2](https://arxiv.org/html/2602.17807v1#S5.SS2.p1.5 "5.2 Comparison with State-of-the-Art Models ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 2](https://arxiv.org/html/2602.17807v1#S5.T2.2.3.3.1 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 3](https://arxiv.org/html/2602.17807v1#S5.T3.5.7.1.1 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [18]J. Jain, J. Li, M. T. Chiu, A. Hassani, N. Orlov, and H. Shi (2023)OneFormer: One Transformer To Rule Universal Image Segmentation. In CVPR, Cited by: [§2](https://arxiv.org/html/2602.17807v1#S2.p1.1 "2 Related Work ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [19]T. Kerssies, N. Cavagnero, A. Hermans, N. Norouzi, G. Averta, B. Leibe, G. Dubbelman, and D. de Geus (2025)Your ViT is Secretly an Image Segmentation Model. In CVPR, Cited by: [§A.3](https://arxiv.org/html/2602.17807v1#A1.SS3.p1.1 "A.3 Visualizations of Model Configurations ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure 1](https://arxiv.org/html/2602.17807v1#S1.F1 "In 1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure 1](https://arxiv.org/html/2602.17807v1#S1.F1.5.2.1 "In 1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§1](https://arxiv.org/html/2602.17807v1#S1.p2.1 "1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§2](https://arxiv.org/html/2602.17807v1#S2.p1.1 "2 Related Work ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§3.3](https://arxiv.org/html/2602.17807v1#S3.SS3.p1.5 "3.3 Removing Task-specific Components ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§3.3](https://arxiv.org/html/2602.17807v1#S3.SS3.p3.1 "3.3 Removing Task-specific Components ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§3.4](https://arxiv.org/html/2602.17807v1#S3.SS4.p1.1 "3.4 VidEoMT ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§4](https://arxiv.org/html/2602.17807v1#S4.p2.1 "4 Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§5.1](https://arxiv.org/html/2602.17807v1#S5.SS1.p2.3 "5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§5.1](https://arxiv.org/html/2602.17807v1#S5.SS1.p3.1 "5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§5.3](https://arxiv.org/html/2602.17807v1#S5.SS3.p3.1 "5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 1](https://arxiv.org/html/2602.17807v1#S5.T1.6.9.3.2 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [20]D. Kim, S. Woo, J. Lee, and I. S. Kweon (2020)Video Panoptic Segmentation. In CVPR, Cited by: [§2](https://arxiv.org/html/2602.17807v1#S2.p2.1 "2 Related Work ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§4](https://arxiv.org/html/2602.17807v1#S4.p1.1 "4 Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [21]S. Lee, J. Seo, K. Han, M. Choi, and S. Im (2025)Context-Aware Video Instance Segmentation. In ICCV, Cited by: [Figure A](https://arxiv.org/html/2602.17807v1#A1.F1 "In A.2 Evaluation ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure A](https://arxiv.org/html/2602.17807v1#A1.F1.9.2.1 "In A.2 Evaluation ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§A.1](https://arxiv.org/html/2602.17807v1#A1.SS1.p1.1 "A.1 Training ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§A.3](https://arxiv.org/html/2602.17807v1#A1.SS3.p1.1 "A.3 Visualizations of Model Configurations ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§A.4](https://arxiv.org/html/2602.17807v1#A1.SS4.p1.2 "A.4 Hyperparameters ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure B](https://arxiv.org/html/2602.17807v1#A3.F2 "In Appendix C Qualitative Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure B](https://arxiv.org/html/2602.17807v1#A3.F2.64.2.1 "In Appendix C Qualitative Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure C](https://arxiv.org/html/2602.17807v1#A3.F3 "In Appendix C Qualitative Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure C](https://arxiv.org/html/2602.17807v1#A3.F3.64.2.1 "In Appendix C Qualitative Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure D](https://arxiv.org/html/2602.17807v1#A3.F4 "In Appendix C Qualitative Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure D](https://arxiv.org/html/2602.17807v1#A3.F4.64.2.1 "In Appendix C Qualitative Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Appendix C](https://arxiv.org/html/2602.17807v1#A3.p1.1 "Appendix C Qualitative Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure 1](https://arxiv.org/html/2602.17807v1#S1.F1 "In 1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure 1](https://arxiv.org/html/2602.17807v1#S1.F1.5.2.1 "In 1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure 2](https://arxiv.org/html/2602.17807v1#S1.F2 "In 1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure 2](https://arxiv.org/html/2602.17807v1#S1.F2.8.2.1 "In 1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§1](https://arxiv.org/html/2602.17807v1#S1.p1.1 "1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§1](https://arxiv.org/html/2602.17807v1#S1.p2.1 "1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§1](https://arxiv.org/html/2602.17807v1#S1.p4.1 "1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§2](https://arxiv.org/html/2602.17807v1#S2.p2.1 "2 Related Work ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§3.2](https://arxiv.org/html/2602.17807v1#S3.SS2.p1.1 "3.2 Preliminaries ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§3.2](https://arxiv.org/html/2602.17807v1#S3.SS2.p4.11 "3.2 Preliminaries ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§3.2](https://arxiv.org/html/2602.17807v1#S3.SS2.p5.3 "3.2 Preliminaries ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§3.3](https://arxiv.org/html/2602.17807v1#S3.SS3.p3.1 "3.3 Removing Task-specific Components ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§3.3](https://arxiv.org/html/2602.17807v1#S3.SS3.p4.1 "3.3 Removing Task-specific Components ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§4](https://arxiv.org/html/2602.17807v1#S4.p2.1 "4 Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§5.1](https://arxiv.org/html/2602.17807v1#S5.SS1.p1.1 "5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 1](https://arxiv.org/html/2602.17807v1#S5.T1.6.8.2.2 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 2](https://arxiv.org/html/2602.17807v1#S5.T2.2.8.8.1 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 3](https://arxiv.org/html/2602.17807v1#S5.T3.5.11.5.1 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 4](https://arxiv.org/html/2602.17807v1#S5.T4.2.5.3.1 "In 5.2 Comparison with State-of-the-Art Models ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 6](https://arxiv.org/html/2602.17807v1#S5.T6.2.2.1.2 "In 5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [22]T. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick (2014)Microsoft COCO: Common Objects in Context. In ECCV, Cited by: [§A.1](https://arxiv.org/html/2602.17807v1#A1.SS1.p1.1 "A.1 Training ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [23]J. Long, E. Shelhamer, and T. Darrell (2015)Fully Convolutional Networks for Semantic Segmentation. In CVPR, Cited by: [§2](https://arxiv.org/html/2602.17807v1#S2.p1.1 "2 Related Work ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [24]I. Loshchilov and F. Hutter (2019)Decoupled Weight Decay Regularization. In ICLR, Cited by: [§A.4](https://arxiv.org/html/2602.17807v1#A1.SS4.p2.1 "A.4 Hyperparameters ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§4](https://arxiv.org/html/2602.17807v1#S4.p2.1 "4 Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [25]T. Meinhardt, A. Kirillov, L. Leal-Taixe, and C. Feichtenhofer (2022)TrackFormer: Multi-Object Tracking with Transformers. In CVPR, Cited by: [§A.5](https://arxiv.org/html/2602.17807v1#A1.SS5.p1.1 "A.5 Architectures of Alternative Approaches ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§A.5](https://arxiv.org/html/2602.17807v1#A1.SS5.p2.5 "A.5 Architectures of Alternative Approaches ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§B.1](https://arxiv.org/html/2602.17807v1#A2.SS1.p5.1 "B.1 Query Propagation Methods ‣ Appendix B Additional Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§5.3](https://arxiv.org/html/2602.17807v1#S5.SS3.p2.1 "5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 7](https://arxiv.org/html/2602.17807v1#S5.T7.2.2.1.3 "In 5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [26]Meta Research (2023)fvcore. Cited by: [§A.2](https://arxiv.org/html/2602.17807v1#A1.SS2.p1.1 "A.2 Evaluation ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§4](https://arxiv.org/html/2602.17807v1#S4.p3.1 "4 Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [27]J. Miao, X. Wang, Y. Wu, W. Li, X. Zhang, Y. Wei, and Y. Yang (2022)Large-Scale Video Panoptic Segmentation in the Wild: A Benchmark. In CVPR, Cited by: [§A.4](https://arxiv.org/html/2602.17807v1#A1.SS4.p1.2 "A.4 Hyperparameters ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure D](https://arxiv.org/html/2602.17807v1#A3.F4 "In Appendix C Qualitative Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure D](https://arxiv.org/html/2602.17807v1#A3.F4.64.2.1 "In Appendix C Qualitative Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Appendix C](https://arxiv.org/html/2602.17807v1#A3.p1.1 "Appendix C Qualitative Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§1](https://arxiv.org/html/2602.17807v1#S1.p6.3 "1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§4](https://arxiv.org/html/2602.17807v1#S4.p1.1 "4 Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 4](https://arxiv.org/html/2602.17807v1#S5.T4.2.1.1.4 "In 5.2 Comparison with State-of-the-Art Models ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [28]J. Miao, Y. Wei, Y. Wu, C. Liang, G. Li, and Y. Yang (2021)VSPW: A Large-scale Dataset for Video Scene Parsing in the Wild. In CVPR, Cited by: [§A.4](https://arxiv.org/html/2602.17807v1#A1.SS4.p1.2 "A.4 Hyperparameters ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§1](https://arxiv.org/html/2602.17807v1#S1.p6.3 "1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§4](https://arxiv.org/html/2602.17807v1#S4.p1.1 "4 Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 5](https://arxiv.org/html/2602.17807v1#S5.T5.2.1.1.4 "In 5.2 Comparison with State-of-the-Art Models ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [29]D. Nilsson and C. Sminchisescu (2018)Semantic Video Segmentation by Gated Recurrent Flow Propagation. In CVPR, Cited by: [§2](https://arxiv.org/html/2602.17807v1#S2.p2.1 "2 Related Work ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [30]M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. (2024)DINOv2: Learning Robust Visual Features without Supervision. TMLR. Cited by: [§A.1](https://arxiv.org/html/2602.17807v1#A1.SS1.p1.1 "A.1 Training ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§A.3](https://arxiv.org/html/2602.17807v1#A1.SS3.p1.1 "A.3 Visualizations of Model Configurations ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§A.5](https://arxiv.org/html/2602.17807v1#A1.SS5.p1.1 "A.5 Architectures of Alternative Approaches ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§B.3](https://arxiv.org/html/2602.17807v1#A2.SS3.p1.1 "B.3 Impact of Pre-training ‣ Appendix B Additional Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure 1](https://arxiv.org/html/2602.17807v1#S1.F1 "In 1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure 1](https://arxiv.org/html/2602.17807v1#S1.F1.5.2.1 "In 1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§1](https://arxiv.org/html/2602.17807v1#S1.p2.1 "1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§1](https://arxiv.org/html/2602.17807v1#S1.p3.1 "1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§3.2](https://arxiv.org/html/2602.17807v1#S3.SS2.p2.9 "3.2 Preliminaries ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§3.4](https://arxiv.org/html/2602.17807v1#S3.SS4.p2.1 "3.4 VidEoMT ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§4](https://arxiv.org/html/2602.17807v1#S4.p2.1 "4 Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§5.3](https://arxiv.org/html/2602.17807v1#S5.SS3.p3.1 "5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [31]J. Qi, Y. Gao, Y. Hu, X. Wang, X. Liu, X. Bai, S. Belongie, A. Yuille, P. H. Torr, and S. Bai (2022)Occluded Video Instance Segmentation: A Benchmark. IJCV 130 (8),  pp.2022–2039. Cited by: [§A.4](https://arxiv.org/html/2602.17807v1#A1.SS4.p1.2 "A.4 Hyperparameters ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure C](https://arxiv.org/html/2602.17807v1#A3.F3 "In Appendix C Qualitative Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure C](https://arxiv.org/html/2602.17807v1#A3.F3.64.2.1 "In Appendix C Qualitative Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Appendix C](https://arxiv.org/html/2602.17807v1#A3.p1.1 "Appendix C Qualitative Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§4](https://arxiv.org/html/2602.17807v1#S4.p1.1 "4 Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 3](https://arxiv.org/html/2602.17807v1#S5.T3.5.6.1.5 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [32]I. Shin, D. Kim, Q. Yu, J. Xie, H. Kim, B. Green, I. S. Kweon, K. Yoon, and L. Chen (2024)Video-kMaX: A Simple Unified Approach for Online and Near-Online Video Panoptic Segmentation. In WACV, Cited by: [§2](https://arxiv.org/html/2602.17807v1#S2.p2.1 "2 Related Work ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [33]O. Siméoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, F. Massa, D. Haziza, L. Wehrstedt, J. Wang, T. Darcet, T. Moutakanni, L. Sentana, C. Roberts, A. Vedaldi, J. Tolan, J. Brandt, C. Couprie, J. Mairal, H. Jégou, P. Labatut, and P. Bojanowski (2025)DINOv3. arXiv preprint arXiv:2508.10104. Cited by: [§B.3](https://arxiv.org/html/2602.17807v1#A2.SS3.p1.1 "B.3 Impact of Pre-training ‣ Appendix B Additional Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§1](https://arxiv.org/html/2602.17807v1#S1.p2.1 "1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§1](https://arxiv.org/html/2602.17807v1#S1.p3.1 "1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§5.3](https://arxiv.org/html/2602.17807v1#S5.SS3.p3.1 "5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [34]H. Touvron, M. Cord, and H. Jégou (2022)DeiT III: Revenge of the ViT. In ECCV, Cited by: [§5.3](https://arxiv.org/html/2602.17807v1#S5.SS3.p3.1 "5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [35]H. Wang, Y. Zhu, H. Adam, A. Yuille, and L. Chen (2021)MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers. In CVPR, Cited by: [§2](https://arxiv.org/html/2602.17807v1#S2.p1.1 "2 Related Work ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [36]M. Weber, J. Xie, M. Collins, Y. Zhu, P. Voigtlaender, H. Adam, B. Green, A. Geiger, B. Leibe, D. Cremers, et al. (2021)STEP: Segmenting and Tracking Every Pixel. In NeurIPS, Cited by: [§4](https://arxiv.org/html/2602.17807v1#S4.p1.1 "4 Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [37]Y. Weng, M. Han, H. He, M. Li, L. Yao, X. Chang, and B. Zhuang (2023)Mask Propagation for Efficient Video Semantic Segmentation. In NeurIPS, Cited by: [§2](https://arxiv.org/html/2602.17807v1#S2.p2.1 "2 Related Work ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [38]L. Yang, Y. Fan, and N. Xu (2019)Video Instance Segmentation. In ICCV, Cited by: [§A.4](https://arxiv.org/html/2602.17807v1#A1.SS4.p1.2 "A.4 Hyperparameters ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table A](https://arxiv.org/html/2602.17807v1#A2.T1.1.2.1.2 "In Appendix B Additional Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure B](https://arxiv.org/html/2602.17807v1#A3.F2 "In Appendix C Qualitative Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure B](https://arxiv.org/html/2602.17807v1#A3.F2.64.2.1 "In Appendix C Qualitative Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Appendix C](https://arxiv.org/html/2602.17807v1#A3.p1.1 "Appendix C Qualitative Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure 1](https://arxiv.org/html/2602.17807v1#S1.F1 "In 1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Figure 1](https://arxiv.org/html/2602.17807v1#S1.F1.5.2.1 "In 1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§1](https://arxiv.org/html/2602.17807v1#S1.p6.3 "1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§2](https://arxiv.org/html/2602.17807v1#S2.p2.1 "2 Related Work ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§4](https://arxiv.org/html/2602.17807v1#S4.p1.1 "4 Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 1](https://arxiv.org/html/2602.17807v1#S5.T1 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 1](https://arxiv.org/html/2602.17807v1#S5.T1.17.2.1 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 2](https://arxiv.org/html/2602.17807v1#S5.T2.2.1.1.4 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 2](https://arxiv.org/html/2602.17807v1#S5.T2.2.1.1.5 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 3](https://arxiv.org/html/2602.17807v1#S5.T3.5.6.1.4 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [39]T. Zhang, X. Tian, Y. Wu, S. Ji, X. Wang, Y. Zhang, and P. Wan (2023)DVIS: Decoupled Video Instance Segmentation Framework. In CVPR, Cited by: [§1](https://arxiv.org/html/2602.17807v1#S1.p1.1 "1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§2](https://arxiv.org/html/2602.17807v1#S2.p2.1 "2 Related Work ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§3.2](https://arxiv.org/html/2602.17807v1#S3.SS2.p1.1 "3.2 Preliminaries ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§5.2](https://arxiv.org/html/2602.17807v1#S5.SS2.p1.5 "5.2 Comparison with State-of-the-Art Models ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 2](https://arxiv.org/html/2602.17807v1#S5.T2.2.4.4.1 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 3](https://arxiv.org/html/2602.17807v1#S5.T3.5.8.2.1 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 4](https://arxiv.org/html/2602.17807v1#S5.T4.2.3.1.1 "In 5.2 Comparison with State-of-the-Art Models ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 5](https://arxiv.org/html/2602.17807v1#S5.T5.2.3.1.1 "In 5.2 Comparison with State-of-the-Art Models ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [40]T. Zhang, X. Tian, Y. Zhou, S. Ji, X. Wang, X. Tao, Y. Zhang, P. Wan, Z. Wang, and Y. Wu (2025)DVIS++: Improved Decoupled Framework for Universal Video Segmentation. PAMI. Cited by: [§A.1](https://arxiv.org/html/2602.17807v1#A1.SS1.p1.1 "A.1 Training ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§1](https://arxiv.org/html/2602.17807v1#S1.p1.1 "1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§1](https://arxiv.org/html/2602.17807v1#S1.p2.1 "1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§1](https://arxiv.org/html/2602.17807v1#S1.p4.1 "1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§2](https://arxiv.org/html/2602.17807v1#S2.p2.1 "2 Related Work ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§3.2](https://arxiv.org/html/2602.17807v1#S3.SS2.p1.1 "3.2 Preliminaries ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§3.2](https://arxiv.org/html/2602.17807v1#S3.SS2.p5.3 "3.2 Preliminaries ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§3.4](https://arxiv.org/html/2602.17807v1#S3.SS4.p8.1 "3.4 VidEoMT ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§5.2](https://arxiv.org/html/2602.17807v1#S5.SS2.p1.5 "5.2 Comparison with State-of-the-Art Models ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 2](https://arxiv.org/html/2602.17807v1#S5.T2.2.6.6.1 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 3](https://arxiv.org/html/2602.17807v1#S5.T3.5.10.4.1 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 4](https://arxiv.org/html/2602.17807v1#S5.T4.2.4.2.1 "In 5.2 Comparison with State-of-the-Art Models ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 5](https://arxiv.org/html/2602.17807v1#S5.T5.2.4.2.1 "In 5.2 Comparison with State-of-the-Art Models ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 6](https://arxiv.org/html/2602.17807v1#S5.T6.2.3.2.2 "In 5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 
*   [41]Y. Zhou, T. Zhang, S. Ji, S. Yan, and X. Li (2024)Improving Video Segmentation via Dynamic Anchor Queries. In ECCV, Cited by: [§A.1](https://arxiv.org/html/2602.17807v1#A1.SS1.p1.1 "A.1 Training ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§1](https://arxiv.org/html/2602.17807v1#S1.p1.1 "1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§1](https://arxiv.org/html/2602.17807v1#S1.p2.1 "1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§1](https://arxiv.org/html/2602.17807v1#S1.p4.1 "1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§2](https://arxiv.org/html/2602.17807v1#S2.p2.1 "2 Related Work ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§3.2](https://arxiv.org/html/2602.17807v1#S3.SS2.p1.1 "3.2 Preliminaries ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§3.2](https://arxiv.org/html/2602.17807v1#S3.SS2.p5.3 "3.2 Preliminaries ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§4](https://arxiv.org/html/2602.17807v1#S4.p2.1 "4 Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [§5.2](https://arxiv.org/html/2602.17807v1#S5.SS2.p1.5 "5.2 Comparison with State-of-the-Art Models ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 2](https://arxiv.org/html/2602.17807v1#S5.T2.2.5.5.1 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 2](https://arxiv.org/html/2602.17807v1#S5.T2.2.7.7.1 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 3](https://arxiv.org/html/2602.17807v1#S5.T3.4.4.1 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 3](https://arxiv.org/html/2602.17807v1#S5.T3.5.9.3.1 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 4](https://arxiv.org/html/2602.17807v1#S5.T4.2.6.4.1 "In 5.2 Comparison with State-of-the-Art Models ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [Table 6](https://arxiv.org/html/2602.17807v1#S5.T6.2.4.3.2 "In 5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). 

\thetitle

Supplementary Material

Appendix
--------

##### Table of contents

*   •
§[A](https://arxiv.org/html/2602.17807v1#A1 "Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"): Implementation Details

*   •
§[B](https://arxiv.org/html/2602.17807v1#A2 "Appendix B Additional Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"): Additional Experiments

*   •
§[C](https://arxiv.org/html/2602.17807v1#A3 "Appendix C Qualitative Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"): Qualitative Results

Appendix A Implementation Details
---------------------------------

### A.1 Training

Following state-of-the-art models CAVIS[[21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation")], DVIS-DAQ[[41](https://arxiv.org/html/2602.17807v1#bib.bib9 "Improving Video Segmentation via Dynamic Anchor Queries")] and DVIS++[[40](https://arxiv.org/html/2602.17807v1#bib.bib8 "DVIS++: Improved Decoupled Framework for Universal Video Segmentation")], we adopt a DINOv2-pretrained ViT[[30](https://arxiv.org/html/2602.17807v1#bib.bib33 "DINOv2: Learning Robust Visual Features without Supervision"), [13](https://arxiv.org/html/2602.17807v1#bib.bib30 "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale")] as the backbone of VidEoMT, and we train our model in two stages. In stage one, we train the model for image segmentation only. Concretely, we train on COCO[[22](https://arxiv.org/html/2602.17807v1#bib.bib53 "Microsoft COCO: Common Objects in Context")] instance segmentation and the target video segmentation dataset without applying any temporal supervision. In the second stage, we introduce temporal modeling and fine-tune the model from stage one for video segmentation. Unlike CAVIS, DVIS-DAQ, and DVIS++, which freeze the DINOv2-initialized ViT encoder after stage one, we keep fine-tuning the ViT encoder for VidEoMT. We explore fine-tuning the ViT encoder for the CAVIS and DVIS++ baselines in[Tabs.1](https://arxiv.org/html/2602.17807v1#S5.T1 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model") and[2](https://arxiv.org/html/2602.17807v1#S5.T2 "Table 2 ‣ 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model") as well, but find that the loss diverges or the memory increases beyond the GPUs’ limits. For our VidEoMT, note that fine-tuning the encoder is necessary because our model is encoder-only, meaning that the encoder weights need to be optimized to allow the model to be trained for video segmentation.

### A.2 Evaluation

During evaluation, we process videos in a frame-by-frame fashion, as is required for online video segmentation. We evaluate efficiency in terms of FPS and GFLOPs. All metrics are measured on a single NVIDIA H100 GPU using PyTorch 2.7 and CUDA 12.6. We use a batch size of 1 frame to report mean values computed across all frames in the entire validation set. FPS is measured using FlashAttention v2[[10](https://arxiv.org/html/2602.17807v1#bib.bib51 "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning")] and torch.compile[[1](https://arxiv.org/html/2602.17807v1#bib.bib49 "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation")] with default settings and automatic mixed precision, after 100 warm-up iterations. FLOPs are measured with fvcore[[26](https://arxiv.org/html/2602.17807v1#bib.bib48 "fvcore")], and reported in GFLOPs (FLOPs ×10−9\times 10^{-9}).

![Image 3: Refer to caption](https://arxiv.org/html/2602.17807v1/x3.png)

(a)Step (1): w/ EoMT as the Segmenter

![Image 4: Refer to caption](https://arxiv.org/html/2602.17807v1/x4.png)

(b)Step (2): w/o Context-aware Features

![Image 5: Refer to caption](https://arxiv.org/html/2602.17807v1/x5.png)

(c)Step (3): w/o Re-identification Layers

![Image 6: Refer to caption](https://arxiv.org/html/2602.17807v1/x6.png)

(d)Step (4): w/o Tracker Blocks

Figure A: Removing specialized components. This figure visualizes the step-by-step removal of complex, specialized components from the CAVIS[[21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation")] model, as reported in the results in Tab.6 of the main manuscript.

### A.3 Visualizations of Model Configurations

In[Sec.3.3](https://arxiv.org/html/2602.17807v1#S3.SS3 "3.3 Removing Task-specific Components ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model") and[Tab.1](https://arxiv.org/html/2602.17807v1#S5.T1 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), we gradually remove specialized components from the state-of-the-art video segmentation model CAVIS[[21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation")], which is visualized in[Fig.2](https://arxiv.org/html/2602.17807v1#S1.F2 "In 1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model") (left). To provide more details, we additionally illustrate the architectures at intermediate steps (1) to (4) in[Fig.A](https://arxiv.org/html/2602.17807v1#A1.F1 "In A.2 Evaluation ‣ Appendix A Implementation Details ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). In the first step, we replace CAVIS’s original segmenter – consisting of DINOv2[[30](https://arxiv.org/html/2602.17807v1#bib.bib33 "DINOv2: Learning Robust Visual Features without Supervision")], ViT-Adapter[[6](https://arxiv.org/html/2602.17807v1#bib.bib34 "Vision Transformer Adapter for Dense Predictions")], and Mask2Former’s pixel decoder and Transformer decoder[[8](https://arxiv.org/html/2602.17807v1#bib.bib27 "Masked-attention Mask Transformer for Universal Image Segmentation")] – with EoMT[[19](https://arxiv.org/html/2602.17807v1#bib.bib1 "Your ViT is Secretly an Image Segmentation Model")]. In the second step, we remove the context-aware features module and directly forward the segmenter’s output queries to the re-identification layers. In the third step, we also remove the re-identification layers, sending the segmenter’s output queries directly to the tracker’s Transformer blocks. Subsequently, in the fourth step, we discard the tracker altogether, and naively apply EoMT only on a per-frame basis. In this step, temporal association is then obtained in the simplest possible way: we assign all objects predicted from the same query across frames to the same track, without any additional post-hoc temporal matching.

In step(5), which we do not visualize here, we propagate queries by directly feeding the output segmentation queries from frame t−1 t-1 into the encoder for frame t t. Finally, in step(6), we introduce our query fusion design, where propagated queries are fused with learnable queries. The resulting architecture is visualized in[Fig.2](https://arxiv.org/html/2602.17807v1#S1.F2 "In 1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model") (right).

### A.4 Hyperparameters

For step(0) in[Tab.1](https://arxiv.org/html/2602.17807v1#S5.T1 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), we report results using the official CAVIS[[21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation")] model weights, which we were able to reproduce. For all subsequent steps, we train the models using the same settings as CAVIS with respect to input size, number of iterations, batch size, and number of sampled frames. Specifically, we use a batch size of 8, train on 8 NVIDIA H100 GPUs, and sample 5 frames from a video clip. We train for 160k iterations on YouTube-VIS[[38](https://arxiv.org/html/2602.17807v1#bib.bib55 "Video Instance Segmentation")] (all versions) and OVIS[[31](https://arxiv.org/html/2602.17807v1#bib.bib54 "Occluded Video Instance Segmentation: A Benchmark")], for 40k iterations on VIPSeg[[27](https://arxiv.org/html/2602.17807v1#bib.bib57 "Large-Scale Video Panoptic Segmentation in the Wild: A Benchmark")], and for 20k iterations on VSPW[[28](https://arxiv.org/html/2602.17807v1#bib.bib56 "VSPW: A Large-scale Dataset for Video Scene Parsing in the Wild")]. Additionally, all VidEoMT models use N=200 N=200 learnable queries with a feature dimension of D=1024 D=1024.

For all experiments that use EoMT as the segmenter, as well as for all experiments with VidEoMT, we keep the optimization strategy identical to that of EoMT. Concretely, we use automatic mixed precision and the AdamW optimizer[[24](https://arxiv.org/html/2602.17807v1#bib.bib46 "Decoupled Weight Decay Regularization")] with a learning rate of 10−4 10^{-4}. We apply layer-wise learning rate decay (LLRD)[[12](https://arxiv.org/html/2602.17807v1#bib.bib47 "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding")] with a factor of 0.6 and polynomial learning rate decay with a power of 0.9. A two-stage linear warm-up strategy is used for all models, including the baselines. Specifically, we first warm up the randomly initialized parameters for 500 iterations while keeping the pre-trained parameters frozen. Then, after 500 iterations, we warm up the pre-trained parameters for 1000 iterations. In both stages, the initial learning rate is set to 0.

To supervise our models, we adopt the same loss functions as Mask2Former[[8](https://arxiv.org/html/2602.17807v1#bib.bib27 "Masked-attention Mask Transformer for Universal Image Segmentation")]. Across all tasks and datasets, we use the cross-entropy (CE) loss for the classification predictions, and the binary cross-entropy (BCE) loss together with Dice loss for segmentation predictions. The total loss is a weighted sum of these components:

ℒ tot=λ bce​ℒ bce+λ dice​ℒ dice+λ ce​ℒ ce.\mathcal{L}_{\textrm{tot}}=\lambda_{\textrm{bce}}\mathcal{L}_{\textrm{bce}}+\lambda_{\textrm{dice}}\mathcal{L}_{\textrm{dice}}+\lambda_{\textrm{ce}}\mathcal{L}_{\textrm{ce}}.(4)

where λ bce\lambda_{\textrm{bce}}, λ dice\lambda_{\textrm{dice}}, and λ ce\lambda_{\textrm{ce}} are set to 5.0, 5.0, and 2.0, respectively, following Mask2Former[[8](https://arxiv.org/html/2602.17807v1#bib.bib27 "Masked-attention Mask Transformer for Universal Image Segmentation")].

### A.5 Architectures of Alternative Approaches

In[Tab.7](https://arxiv.org/html/2602.17807v1#S5.T7 "In 5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), we compare VidEoMT with an alternative encoder–decoder architecture that performs temporal modeling in the decoder, with two different temporal modeling approaches: our proposed query fusion and a TrackFormer-based design[[25](https://arxiv.org/html/2602.17807v1#bib.bib36 "TrackFormer: Multi-Object Tracking with Transformers")]. As the encoder, we use DINOv2 + ViT-Adapter[[6](https://arxiv.org/html/2602.17807v1#bib.bib34 "Vision Transformer Adapter for Dense Predictions"), [30](https://arxiv.org/html/2602.17807v1#bib.bib33 "DINOv2: Learning Robust Visual Features without Supervision")], and as the decoder we use a Transformer that follows the architecture of the Mask2Former Transformer decoder for segmentation.[[8](https://arxiv.org/html/2602.17807v1#bib.bib27 "Masked-attention Mask Transformer for Universal Image Segmentation")]. Concretely, we adopt the standard Mask2Former decoder with 9 layers, each composed of cross-attention, self-attention, and feed-forward blocks, operating with a hidden dimension of 256. To introduce temporal modeling, we feed the track queries and learnable queries into the decoder instead of the encoder’s Transformer blocks, which we would do for VidEoMT. At the output of the decoder, the resulting queries are used to predict segmentation masks and classes in the same way as for VidEoMT. For query fusion, we adopt the same approach as described in[Sec.3.4](https://arxiv.org/html/2602.17807v1#S3.SS4 "3.4 VidEoMT ‣ 3 Method ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model").

The described encoder–decoder approach, which is much less efficient than the encoder-only VidEoMT method (see[Tab.7](https://arxiv.org/html/2602.17807v1#S5.T7 "In 5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model")), somewhat resembles TrackFormer[[25](https://arxiv.org/html/2602.17807v1#bib.bib36 "TrackFormer: Multi-Object Tracking with Transformers")], a method for bounding-box multi-object tracking (MOT). TrackFormer also applies temporal modeling by propagating queries into the decoder, but follows a more complex approach to do so. To assess the effectiveness of our query fusion approach compared to TrackFormer’s approach, we therefore additionally implement TrackFormer’s temporal modeling strategy in the encoder–decoder setting, while staying as close as possible to the original implementation. Specifically, we make predictions for the first frame using a set of 400 learnable queries. Using these predictions, only the N N queries with a classification score s>0.8 s>0.8 are kept and converted into track queries. For the next frame, these track queries are concatenated with the 400 original learnable queries, which are then fed to the decoder for that frame. In subsequent frames, the decoder updates the propagated track queries such that they predict the masks for the same objects in the new frames. Again, newly detected queries with scores s>0.8 s>0.8 are added as additional track queries, and non-maximum suppression (NMS) with an IoU threshold of σ NMS=0.9\sigma_{\text{NMS}}=0.9 is applied to remove near-duplicate predictions. Note that this NMS operation is the main reason for the TrackFormer approach’s inefficiency compared to VidEoMT’s query propagation mechanism. Finally, at each frame, track queries are removed if their score remains below s<0.8 s<0.8 for five consecutive frames, indicating that the object they are tracking has disappeared from the scene.

Appendix B Additional Experiments
---------------------------------

Table A: Query propagation methods. Comparison of alternative strategies for temporal propagation.

### B.1 Query Propagation Methods

VidEoMT propagates queries by fusing the learnable queries with the propagated track queries. In [Tab.A](https://arxiv.org/html/2602.17807v1#A2.T1 "In Appendix B Additional Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), we compare this approach with alternative methods to propagate queries.

We start with the no propagation approach, the simplest variant, where the model receives only the learnable queries – similar to EoMT – but is fine-tuned for video segmentation. This setting performs the worst, as it lacks any form of explicit temporal modeling.

Next, in the propagation only variant, we introduce temporal modeling by directly propagating the output queries from the previous frame into the current frame’s encoder. This is step (5) in[Tab.1](https://arxiv.org/html/2602.17807v1#S5.T1 "In 5.1 Main Results ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). However, this approach struggles to detect new objects effectively, as the influence of the learnable queries diminishes over time.

Non-object reset improves over this by replacing a propagated query with a learnable query if it did not predict an object in the previous frame, but this still underperforms the default fusion approach.

Finally, we evaluate the TrackFormer approach[[25](https://arxiv.org/html/2602.17807v1#bib.bib36 "TrackFormer: Multi-Object Tracking with Transformers")] of only propagating queries for detected objects and introducing new learnable queries to detect new objects. This approach performs slightly worse than our fusion approach, but most importantly it is considerably slower because it requires filtering out duplicate detections that should not be propagated. Overall, these results demonstrate that our fusion approach is the most accurate and efficient.

### B.2 Impact of Model Size

In[Tab.9](https://arxiv.org/html/2602.17807v1#S5.T9 "In 5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model") of the main manuscript, we report the impact of model size for both VidEoMT and CAVIS. In this section, in [Tab.B](https://arxiv.org/html/2602.17807v1#A2.T2 "In B.2 Impact of Model Size ‣ Appendix B Additional Experiments ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), we additionally report the results of the EoMT + CAVIS combination, which we also visualize in[Fig.1](https://arxiv.org/html/2602.17807v1#S1.F1 "In 1 Introduction ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"). Compared to the alternative approach of extending EoMT with a CAVIS tracker, VidEoMT consistently performs better in terms of both efficiency and accuracy across all backbones. This highlights the effectiveness of VidEoMT over the more naive approach of extending EoMT with a state-of-the-art tracker.

Table B: Impact of model size. VidEoMT performs better as ViT[[13](https://arxiv.org/html/2602.17807v1#bib.bib30 "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale")] size increases. Evaluated on YouTube-VIS 2019 val.

### B.3 Impact of Pre-training

In[Tab.8](https://arxiv.org/html/2602.17807v1#S5.T8 "In 5.3 Further Analyses ‣ 5 Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model") of the main manuscript, we observe that DINOv3[[33](https://arxiv.org/html/2602.17807v1#bib.bib66 "DINOv3")] and EVA-02[[14](https://arxiv.org/html/2602.17807v1#bib.bib67 "EVA-02: A Visual Representation for Neon Genesis")] are slower than DINOv2[[30](https://arxiv.org/html/2602.17807v1#bib.bib33 "DINOv2: Learning Robust Visual Features without Supervision")], despite having a similar number of GFLOPs. Since both DINOv3 and EVA-02 use rotary positional embeddings (RoPE), we attribute a significant part of this slowdown to RoPE, as it introduces additional element-wise operations in the attention layers. When we disable RoPE in these models, we obtain faster models, confirming that RoPE is one of the main sources of the slowdown. Other implementation details may also play a secondary role.

Appendix C Qualitative Results
------------------------------

In [Figs.B](https://arxiv.org/html/2602.17807v1#A3.F2 "In Appendix C Qualitative Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), [C](https://arxiv.org/html/2602.17807v1#A3.F3 "Figure C ‣ Appendix C Qualitative Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model") and[D](https://arxiv.org/html/2602.17807v1#A3.F4 "Figure D ‣ Appendix C Qualitative Results ‣ VidEoMT: Your ViT is Secretly Also a Video Segmentation Model"), we visualize the predictions of both CAVIS[[21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation")] and VidEoMT for VIS and VPS on the YouTube-VIS 2019[[38](https://arxiv.org/html/2602.17807v1#bib.bib55 "Video Instance Segmentation")], OVIS[[31](https://arxiv.org/html/2602.17807v1#bib.bib54 "Occluded Video Instance Segmentation: A Benchmark")], and VIPSeg[[27](https://arxiv.org/html/2602.17807v1#bib.bib57 "Large-Scale Video Panoptic Segmentation in the Wild: A Benchmark")] datasets.

![Image 7: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/35d5e5149d/00000.jpg)

![Image 8: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/35d5e5149d/00005.jpg)

![Image 9: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/35d5e5149d/00010.jpg)

![Image 10: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/35d5e5149d/00015.jpg)

![Image 11: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/35d5e5149d/00020.jpg)

![Image 12: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/35d5e5149d/00025.jpg)

![Image 13: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/35d5e5149d/00030.jpg)

![Image 14: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/35d5e5149d/00035.jpg)

![Image 15: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/35d5e5149d/00040.jpg)

![Image 16: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/35d5e5149d/00045.jpg)

![Image 17: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/35d5e5149d/00050.jpg)

![Image 18: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/35d5e5149d/00055.jpg)

![Image 19: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/35d5e5149d/00060.jpg)

CAVIS (15 FPS)

![Image 20: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/35d5e5149d/00000.jpg)

![Image 21: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/35d5e5149d/00005.jpg)

![Image 22: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/35d5e5149d/00010.jpg)

![Image 23: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/35d5e5149d/00015.jpg)

![Image 24: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/35d5e5149d/00020.jpg)

![Image 25: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/35d5e5149d/00025.jpg)

![Image 26: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/35d5e5149d/00030.jpg)

![Image 27: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/35d5e5149d/00035.jpg)

![Image 28: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/35d5e5149d/00040.jpg)

![Image 29: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/35d5e5149d/00045.jpg)

![Image 30: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/35d5e5149d/00050.jpg)

![Image 31: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/35d5e5149d/00055.jpg)

![Image 32: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/35d5e5149d/00060.jpg)

VidEoMT (160 FPS)

![Image 33: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/6cb5b08d93/00000.jpg)

![Image 34: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/6cb5b08d93/00005.jpg)

![Image 35: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/6cb5b08d93/00010.jpg)

![Image 36: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/6cb5b08d93/00015.jpg)

![Image 37: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/6cb5b08d93/00020.jpg)

![Image 38: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/6cb5b08d93/00025.jpg)

![Image 39: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/6cb5b08d93/00030.jpg)

![Image 40: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/6cb5b08d93/00035.jpg)

![Image 41: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/6cb5b08d93/00040.jpg)

![Image 42: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/6cb5b08d93/00045.jpg)

![Image 43: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/6cb5b08d93/00050.jpg)

![Image 44: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/6cb5b08d93/00055.jpg)

![Image 45: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/yt-2019/6cb5b08d93/00060.jpg)

CAVIS (15 FPS)

![Image 46: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/6cb5b08d93/00000.jpg)

![Image 47: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/6cb5b08d93/00005.jpg)

![Image 48: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/6cb5b08d93/00010.jpg)

![Image 49: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/6cb5b08d93/00015.jpg)

![Image 50: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/6cb5b08d93/00020.jpg)

![Image 51: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/6cb5b08d93/00025.jpg)

![Image 52: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/6cb5b08d93/00030.jpg)

![Image 53: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/6cb5b08d93/00035.jpg)

![Image 54: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/6cb5b08d93/00040.jpg)

![Image 55: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/6cb5b08d93/00045.jpg)

![Image 56: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/6cb5b08d93/00050.jpg)

![Image 57: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/6cb5b08d93/00055.jpg)

![Image 58: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/yt-2019/6cb5b08d93/00060.jpg)

VidEoMT (160 FPS)

Figure B: Qualitative results for video instance segmentation. We compare CAVIS[[21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation")] to VidEoMT on the YouTube-VIS 2019 dataset[[38](https://arxiv.org/html/2602.17807v1#bib.bib55 "Video Instance Segmentation")].

![Image 59: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/d4f4cf55/img_0000001.jpg)

![Image 60: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/d4f4cf55/img_0000002.jpg)

![Image 61: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/d4f4cf55/img_0000003.jpg)

![Image 62: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/d4f4cf55/img_0000004.jpg)

![Image 63: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/d4f4cf55/img_0000005.jpg)

![Image 64: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/d4f4cf55/img_0000006.jpg)

![Image 65: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/d4f4cf55/img_0000007.jpg)

![Image 66: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/d4f4cf55/img_0000008.jpg)

![Image 67: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/d4f4cf55/img_0000009.jpg)

![Image 68: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/d4f4cf55/img_0000010.jpg)

![Image 69: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/d4f4cf55/img_0000011.jpg)

![Image 70: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/d4f4cf55/img_0000012.jpg)

![Image 71: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/d4f4cf55/img_0000013.jpg)

CAVIS (15 FPS)

![Image 72: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/d4f4cf55/img_0000001.jpg)

![Image 73: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/d4f4cf55/img_0000002.jpg)

![Image 74: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/d4f4cf55/img_0000003.jpg)

![Image 75: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/d4f4cf55/img_0000004.jpg)

![Image 76: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/d4f4cf55/img_0000005.jpg)

![Image 77: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/d4f4cf55/img_0000006.jpg)

![Image 78: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/d4f4cf55/img_0000007.jpg)

![Image 79: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/d4f4cf55/img_0000008.jpg)

![Image 80: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/d4f4cf55/img_0000009.jpg)

![Image 81: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/d4f4cf55/img_0000010.jpg)

![Image 82: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/d4f4cf55/img_0000011.jpg)

![Image 83: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/d4f4cf55/img_0000012.jpg)

![Image 84: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/d4f4cf55/img_0000013.jpg)

VidEoMT (112 FPS)

![Image 85: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/1f17cd7c/img_0000001.jpg)

![Image 86: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/1f17cd7c/img_0000002.jpg)

![Image 87: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/1f17cd7c/img_0000003.jpg)

![Image 88: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/1f17cd7c/img_0000004.jpg)

![Image 89: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/1f17cd7c/img_0000005.jpg)

![Image 90: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/1f17cd7c/img_0000006.jpg)

![Image 91: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/1f17cd7c/img_0000007.jpg)

![Image 92: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/1f17cd7c/img_0000008.jpg)

![Image 93: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/1f17cd7c/img_0000009.jpg)

![Image 94: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/1f17cd7c/img_0000010.jpg)

![Image 95: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/1f17cd7c/img_0000011.jpg)

![Image 96: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/1f17cd7c/img_0000012.jpg)

![Image 97: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/ovis/1f17cd7c/img_0000013.jpg)

CAVIS (15 FPS)

![Image 98: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/1f17cd7c/img_0000001.jpg)

![Image 99: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/1f17cd7c/img_0000002.jpg)

![Image 100: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/1f17cd7c/img_0000003.jpg)

![Image 101: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/1f17cd7c/img_0000004.jpg)

![Image 102: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/1f17cd7c/img_0000005.jpg)

![Image 103: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/1f17cd7c/img_0000006.jpg)

![Image 104: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/1f17cd7c/img_0000007.jpg)

![Image 105: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/1f17cd7c/img_0000008.jpg)

![Image 106: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/1f17cd7c/img_0000009.jpg)

![Image 107: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/1f17cd7c/img_0000010.jpg)

![Image 108: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/1f17cd7c/img_0000011.jpg)

![Image 109: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/1f17cd7c/img_0000012.jpg)

![Image 110: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/ovis/1f17cd7c/img_0000013.jpg)

VidEoMT (112 FPS)

Figure C: Qualitative results for video instance segmentation. We compare CAVIS[[21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation")] to VidEoMT on the OVIS dataset[[31](https://arxiv.org/html/2602.17807v1#bib.bib54 "Occluded Video Instance Segmentation: A Benchmark")].

![Image 111: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/1010_kI0mOZirPGs/00000095.jpg)

![Image 112: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/1010_kI0mOZirPGs/00000098.jpg)

![Image 113: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/1010_kI0mOZirPGs/00000101.jpg)

![Image 114: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/1010_kI0mOZirPGs/00000104.jpg)

![Image 115: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/1010_kI0mOZirPGs/00000107.jpg)

![Image 116: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/1010_kI0mOZirPGs/00000110.jpg)

![Image 117: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/1010_kI0mOZirPGs/00000113.jpg)

![Image 118: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/1010_kI0mOZirPGs/00000116.jpg)

![Image 119: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/1010_kI0mOZirPGs/00000119.jpg)

![Image 120: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/1010_kI0mOZirPGs/00000122.jpg)

![Image 121: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/1010_kI0mOZirPGs/00000125.jpg)

![Image 122: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/1010_kI0mOZirPGs/00000128.jpg)

![Image 123: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/1010_kI0mOZirPGs/00000131.jpg)

CAVIS (10 FPS)

![Image 124: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/1010_kI0mOZirPGs/00000095.jpg)

![Image 125: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/1010_kI0mOZirPGs/00000098.jpg)

![Image 126: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/1010_kI0mOZirPGs/00000101.jpg)

![Image 127: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/1010_kI0mOZirPGs/00000104.jpg)

![Image 128: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/1010_kI0mOZirPGs/00000107.jpg)

![Image 129: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/1010_kI0mOZirPGs/00000110.jpg)

![Image 130: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/1010_kI0mOZirPGs/00000113.jpg)

![Image 131: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/1010_kI0mOZirPGs/00000116.jpg)

![Image 132: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/1010_kI0mOZirPGs/00000119.jpg)

![Image 133: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/1010_kI0mOZirPGs/00000122.jpg)

![Image 134: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/1010_kI0mOZirPGs/00000125.jpg)

![Image 135: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/1010_kI0mOZirPGs/00000128.jpg)

![Image 136: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/1010_kI0mOZirPGs/00000131.jpg)

VidEoMT (75 FPS)

![Image 137: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/605_AymiAkCRAFM/00000287.jpg)

![Image 138: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/605_AymiAkCRAFM/00000290.jpg)

![Image 139: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/605_AymiAkCRAFM/00000293.jpg)

![Image 140: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/605_AymiAkCRAFM/00000296.jpg)

![Image 141: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/605_AymiAkCRAFM/00000299.jpg)

![Image 142: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/605_AymiAkCRAFM/00000302.jpg)

![Image 143: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/605_AymiAkCRAFM/00000305.jpg)

![Image 144: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/605_AymiAkCRAFM/00000308.jpg)

![Image 145: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/605_AymiAkCRAFM/00000311.jpg)

![Image 146: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/605_AymiAkCRAFM/00000314.jpg)

![Image 147: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/605_AymiAkCRAFM/00000317.jpg)

![Image 148: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/605_AymiAkCRAFM/00000320.jpg)

![Image 149: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/cavis/vipseg/605_AymiAkCRAFM/00000323.jpg)

CAVIS (10 FPS)

![Image 150: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/605_AymiAkCRAFM/00000287.jpg)

![Image 151: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/605_AymiAkCRAFM/00000290.jpg)

![Image 152: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/605_AymiAkCRAFM/00000293.jpg)

![Image 153: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/605_AymiAkCRAFM/00000296.jpg)

![Image 154: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/605_AymiAkCRAFM/00000299.jpg)

![Image 155: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/605_AymiAkCRAFM/00000302.jpg)

![Image 156: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/605_AymiAkCRAFM/00000305.jpg)

![Image 157: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/605_AymiAkCRAFM/00000308.jpg)

![Image 158: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/605_AymiAkCRAFM/00000311.jpg)

![Image 159: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/605_AymiAkCRAFM/00000314.jpg)

![Image 160: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/605_AymiAkCRAFM/00000317.jpg)

![Image 161: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/605_AymiAkCRAFM/00000320.jpg)

![Image 162: Refer to caption](https://arxiv.org/html/2602.17807v1/appendix/img/videomt/vipseg/605_AymiAkCRAFM/00000323.jpg)

VidEoMT (75 FPS)

Figure D: Qualitative results for video panoptic segmentation. We compare CAVIS[[21](https://arxiv.org/html/2602.17807v1#bib.bib19 "Context-Aware Video Instance Segmentation")] to VidEoMT on the VIPSeg dataset[[27](https://arxiv.org/html/2602.17807v1#bib.bib57 "Large-Scale Video Panoptic Segmentation in the Wild: A Benchmark")].

