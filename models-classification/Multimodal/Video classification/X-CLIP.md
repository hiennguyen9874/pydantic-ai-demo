# Expanding Language-Image Pretrained Models for General Video Recognition

Bolin Ni<sup>1,2 \*</sup>, Houwen Peng<sup>1 †</sup>, Minghao Chen<sup>1,3 \*</sup>, Songyang Zhang<sup>4</sup>,  
Gaofeng Meng<sup>2</sup>, Jianlong Fu<sup>1</sup>, Shiming Xiang<sup>2</sup>, Haibin Ling<sup>3</sup>

<sup>1</sup> Microsoft Research

<sup>2</sup> Chinese Academy of Sciences

<sup>3</sup> Stony Brook University

<sup>4</sup> University of Rochester

**Abstract.** Contrastive language-image pretraining has shown great success in learning visual-textual joint representation from web-scale data, demonstrating remarkable “zero-shot” generalization ability for various image tasks. However, how to effectively expand such new language-image pretraining methods to video domains is still an open problem. In this work, we present a simple yet effective approach that adapts the pretrained language-image models to video recognition directly, instead of pretraining a new model from scratch. More concretely, to capture the long-range dependencies of frames along the temporal dimension, we propose a cross-frame attention mechanism that explicitly exchanges information across frames. Such module is lightweight and can be plugged into pretrained language-image models seamlessly. Moreover, we propose a video-specific prompting scheme, which leverages video content information for generating discriminative textual prompts. Extensive experiments demonstrate that our approach is effective and can be generalized to different video recognition scenarios. In particular, under fully-supervised settings, our approach achieves a top-1 accuracy of 87.1% on Kinetics-400, while using  $12\times$  fewer FLOPs compared with Swin-L and ViViT-H. In zero-shot experiments, our approach surpasses the current state-of-the-art methods by +7.6% and +14.9% in terms of top-1 accuracy under two popular protocols. In few-shot scenarios, our approach outperforms previous best methods by +32.1% and +23.1% when the labeled data is extremely limited. Code and models are available at [aka.ms/X-CLIP](https://aka.ms/X-CLIP).

**Keywords:** Video Recognition, Contrastive Language-Image Pretraining

## 1 Introduction

Video recognition is one of the most fundamental yet challenging tasks in video understanding. It plays a vital role in numerous vision applications, such as micro-video recommendation [63], sports video analysis [41], autonomous driving [19], and so on. Over the past few years, based upon convolutional neural networks and now transformers, video recognition has achieved remarkable progress [63,22]. Most existing works follow a closed-set learning setting, where all the categories

---

\* Work done during internship at Microsoft Research. † Project Lead.Fig. 1: Comparison with state-of-the-art methods on Kinetics-400 [23] in terms of throughput, the number of views, and FLOPs. Best viewed in color.

are pre-defined. Such method is unrealistic for many real-world applications, such as automatic tagging of web videos, where information regarding new video categories is not available during training. It is thus very challenging for closed-set methods to train a classifier for recognizing unseen or unfamiliar categories.

Fortunately, recent work in large-scale contrastive language-image pretraining, such as CLIP [37], ALIGN [20], and Florence [55], has shown great potentials in addressing this challenge. The core idea is to learn visual or visual-language representation with natural language supervision using web-scale image-text data. After pretraining, natural language is used to reference learned visual concepts (or describe new ones), thus enabling zero/few-shot transfer of the models to downstream tasks. Inspired by these works [37,20,55], we consider to use text as the supervision signals to learn a new video representation for general recognition scenarios, including zero-shot, few-shot, and fully-supervised.

However, directly training a language-video model is unaffordable for many of us, because it requires large-scale video-text pretraining data as well as a massive number of GPU resources (*e.g.*, thousands of GPU days). A feasible solution is to adapt the pretrained language-image models to video domain. Very recently, there are several studies exploring how to transfer the knowledge from the pretrained language-image models to other downstream tasks, *e.g.*, point cloud understanding [59] and dense prediction [38,60]. However, the transfer and adaptation to video recognition is not well explored. When adapting the pretrained cross-modality models from image to video domain, there are two key issues to be solved: 1) how to leverage the temporal information contained in videos, and 2) how to acquire discriminative text representation for a video.

For the first question, we present a new architecture for video temporal modeling. It consists of two key components: a cross-frame communication transformer and a multi-frame integration transformer. Specifically, the cross-frame communication transformer takes raw frames as input and provides a frame-level representation using a pretrained language-image model, while allowing information exchange between frames with message tokens. Each message tokennot only depicts the semantics of the current frame, but also communicates with other frames to model their dependencies. The multi-frame integration transformer then simply transfer the frame-level representations to video-level.

For the second question, we employ the text encoder pretrained in the language-image models and expand it with a video-specific prompting scheme. The key idea is to leverage video content information to enhance text prompting. The intuition behind is that appropriate contextual information can help the recognition. For example, if there is extra video content information about “**in the water**”, the actions “**swimming**” and “**running**” will be much easier to be distinguished. In contrast to prior work manually designing a fixed set of text prompts, this work proposes a learnable prompting mechanism, which integrates both semantic labels and representation of videos for automatic prompt generation.

With the above two issues addressed, we can smoothly adapt existing image-level cross-modality pretrained models to video domains. Without loss of generality, here we choose the CLIP [37] and Florence [55] models and eXpand them for general video recognition, forming new model families called X-CLIP and X-Florence, respectively. Comprehensive experiments demonstrate our expanded models are generally effective. In particular, under the fully-supervised setting, X-CLIP-L/14 achieves competitive performance on Kinetics-400/600 with top-1 accuracies of 87.1%/88.3%, surpassing ViViT-H [3] by 2.3%/2.5% while using 12 $\times$  fewer FLOPs, as shown in Fig. 1. In zero-shot experiments, X-Florence surpasses the state-of-the-art ActionCLIP [49] by +7.6% and +14.9% under two popular protocols. In few-shot experiments, X-CLIP outperforms other prevailing methods by +32.1% and +23.1% when the data is extremely limited.

In summary, our contributions are three-fold:

- – We propose a new cross-frame communication attention for video temporal modeling. This module is light and efficient, and can be seamlessly plugged into existing language-image pretrained models, without undermining their original parameters and performance.
- – We design a video-specific prompting technique to yield instance-level textual representation automatically. It leverages video content information to enhance the textual prompt generation.
- – Our work presents a new way of expanding existing large-scale language-image pretrained models for general video recognition and other potential video tasks. Extensive experiments demonstrate the superiority and good generalization ability of our method under various learning configurations.

## 2 Related Work

**Visual-language Pretraining.** Visual-language pretraining has achieved remarkable progress over the past few years [44,43,32,62]. In particular, contrastive language-image pretraining demonstrates very impressive “zero-shot” transfer and generalization capacities [37,20,55]. One of the most representative works is the recent CLIP [37]. A large amount of follow-up works have been proposed to leverage the pretrained models for downstream tasks. For example, CoOp [61],CLIP-Adapter [16] and Tip-Adapter [58] use the pretrained CLIP for improving the few-shot transfer, while PointCLIP [59] and DenseCLIP [38,60] transfer the knowledge to point cloud understanding and dense prediction, respectively. VideoCLIP [52] extends the image-level pretraining to video by substituting the image-text data with video-text pairs [32]. However, such video-text pretraining is computationally expensive and requires a large amount of curated video-text data which is not easy to acquire. In contrast, our method directly adapts the existing pretrained model to video recognition, largely saving the training cost.

There are two concurrent works mostly related to ours. One is ActionCLIP [49], while the other is [21]. Both of them introduce visual-language pretrained models to video understanding. ActionCLIP proposes a “pretrain, prompt and finetune” framework for action recognition, while [21] proposes to optimize a few random vectors for adapting CLIP to various video understanding tasks. In contrast, our method is more general. It supports adapting various language-image models, such as CLIP and Florence [55], from image to video. Moreover, we propose a lightweight and efficient cross-frame attention module for video temporal modeling, while presenting a new video-specific text prompting scheme.

**Video Recognition.** One key factor to build a robust video recognition model is to exploit the temporal information. Among many methods, 3D convolution is widely used [45,46,36,51], while it suffers from high computational cost. For efficiency purposes, some studies [46,36,51] factorize convolutions across spatial and temporal dimensions, while others insert the specific temporal modules into 2D CNNs [28,26,31]. Nevertheless, the limited receptive field of CNNs gives the rise of transformer-based methods [3,5,30,12,54], which achieve very promising performance recently. However, these transformer-based methods are either computationally intensive or insufficient in exploiting the temporal information. For example, ViViT [3] disregards the temporal information in the early stage. Video Swin [30] utilizes 3D attention while having high computational cost.

The temporal modeling scheme in our method shares a similar spirit with the recent proposed video transformers, *i.e.*, VTN [33], ViViT [3], and AVT [18]. They all use a frame-level encoder followed by a temporal encoder, but our method has two fundamental differences. 1) In [33,3,18], each frame is encoded separately, resulting in no temporal interaction before final aggregation. This late fusion strategy does not fully make use of the temporal cues. By contrast, our method replaces the spatial attention with the proposed cross-frame attention, which allows global spatio-temporal modeling for all frames. 2) Similar to previous works [30,12,13,5], both ViViT [3] and VTN [33] adopt a dense temporal sampling strategy and ensemble the predictions of multiple views at inference, which is time-consuming. On the contrary, we empirically analyze different sampling methods for late fusion, and demonstrate that a sparse sampling is good enough, achieving better performance with fewer FLOPs than the dense strategy, as verified in Sec. 4.5 (Analysis).The diagram illustrates the X-CLIP framework architecture. It starts with a video clip (represented by a sequence of frames) and a text description (e.g., 'Air drum', 'Cry', 'Swim'). The video clip is processed by a 'Patch Embedding' layer, which feeds into a 'Cross-frame Communication Transformer'. The output of this transformer is then processed by a 'Video-specific Prompting' layer. The output of the 'Video-specific Prompting' layer is fed into a 'Multi-frame Integration Transformer'. The final output of the 'Multi-frame Integration Transformer' is a score for 'growth-truth'. The diagram also shows the initialization methods for the model: Pretrained initialization (smiley face), Partially pretrained initialization (half-smiley face), and Random initialization (sad face).

Fig. 2: An overview of our framework. The details are elaborated in Sec. 3.1.

### 3 Approach

In this section, we present our proposed framework in detail. First, we briefly overview our video-text framework in Sec. 3.1. Then, we depict the architecture of the video encoder, especially for the proposed cross-frame attention in Sec. 4.1. Finally, we introduce a video-specific prompting scheme in Sec. 3.3.

#### 3.1 Overview

Most prior works in video recognition learn discriminative feature embeddings supervised by a one-hot label [3,5,13,48]. While in this work, inspired by the recent contrastive language-image pretraining [37,20,55], we propose to use text as the supervision, since the text provides more semantic information. As shown in Fig. 2, our method learns to align the video representation and its corresponding text representation by jointly training a video encoder and a text encoder. Rather than pretraining a new video-text model from scratch, our method is built upon prior language-image models and expands them with video temporal modeling and video-adaptive textual prompts. Such a strategy allows us to fully take advantage of existing large-scale pretrained models while transferring their powerful generalizability from image to video in a seamless fashion.

Formally, given a video clip  $V \in \mathcal{V}$  and a text description  $C \in \mathcal{C}$ , where  $\mathcal{V}$  is a set of videos and  $\mathcal{C}$  is a collection of category names, we feed the video  $V$  into the video encoder  $f_{\theta_v}$  and the text  $C$  into the text encoder  $f_{\theta_c}$  to obtain a video representation  $\mathbf{v}$  and a text representation  $\mathbf{c}$  respectively, where

$$\mathbf{v} = f_{\theta_v}(V), \quad \mathbf{c} = f_{\theta_c}(C). \quad (1)$$

Then, a video-specific prompt generator  $f_{\theta_p}$  is employed to yield instance-level textual representation for each video. It takes the video representation  $\mathbf{v}$  and text representation  $\mathbf{c}$  as inputs, formulated as

$$\hat{\mathbf{c}} = f_{\theta_p}(\mathbf{c}, \mathbf{v}). \quad (2)$$Figure 3 consists of two parts. Part (a) illustrates the Cross-frame Communication Transformer Block. It shows a sequence of frames being processed by a series of blocks. Each block contains an 'Intra-Frame Diffusion Attention' layer, followed by a 'Cross-frame Fusion Attention' layer, and then a 'FFN' (Feed-Forward Network) layer. The 'Cross-frame Fusion Attention' layer facilitates information exchange between frames. Part (b) compares four different space-time attention mechanisms: 1) Joint Space-Time Attention (ST), 2) Divided Space-Time Attention (+T), 3) Window-based Space-Time Attention (ST), and 4) Cross-frame Attention (+T) (Ours). Each mechanism is represented by a 3D grid of blocks, with different blocks highlighted in orange to show the attention pattern.

Fig. 3: (a) Cross-frame Attention. (b) compares different space-time attention mechanisms used in existing video transformer backbones [3,5,30].

Finally, a cosine similarity function  $\text{sim}(\mathbf{v}, \hat{\mathbf{c}})$  is utilized to compute the similarity between the visual and textual representations:

$$\text{sim}(\mathbf{v}, \hat{\mathbf{c}}) = \frac{\langle \mathbf{v}, \hat{\mathbf{c}} \rangle}{\|\mathbf{v}\| \|\hat{\mathbf{c}}\|}. \quad (3)$$

The goal of our method is to maximize the  $\text{sim}(\mathbf{v}, \hat{\mathbf{c}})$  if  $V$  and  $C$  are matched and otherwise minimize it.

### 3.2 Video Encoder

Our proposed video encoder is composed of two cascaded vision transformers: a cross-frame communication transformer and a multi-frame integration transformer. The cross-frame transformer takes raw frames as input and provides a frame-level representation using a pretrained language-image model, while allowing information exchange between frames. The multi-frame integration transformer then simply integrates the frame-level representations and outputs video features.

Specifically, given a video clip  $V \in \mathbb{R}^{T \times H \times W \times 3}$  of  $T$  sampled frames with  $H$  and  $W$  denote the spatial resolution, following ViT [11], the  $t$ -th frame is divided into  $N$  non-overlapping patches  $\{\mathbf{x}_{t,i}\}_{i=1}^N \in \mathbb{R}^{P^2 \times 3}$  with each of size  $P \times P$  pixels, where  $t \in \{1, \dots, T\}$  denotes the temporal index, and  $N = HW/P^2$ . The patches  $\{\mathbf{x}_{t,i}\}_{i=1}^N$  are then embedded into patch embeddings using a linear projection  $\mathbf{E} \in \mathbb{R}^{3P^2 \times D}$ . After that, we prepend a learnable embedding  $\mathbf{x}_{class}$  to the sequence of embedded patches, called [class] token. Its state at the output of the encoder serves as the frame representation. The input of the cross-frame communication transformer at the frame  $t$  is denoted as:

$$\mathbf{z}_t^{(0)} = [\mathbf{x}_{class}, \mathbf{E}\mathbf{x}_{t,1}, \mathbf{E}\mathbf{x}_{t,2}, \dots, \mathbf{E}\mathbf{x}_{t,N}] + \mathbf{e}^{spa}, \quad (4)$$where  $\mathbf{e}^{spa}$  represents the spatial position encoding.

Then we feed the patch embeddings into an  $L_c$ -layer Cross-frame Communication Transformer (CCT) to obtain the frame-level representation  $\mathbf{h}_t$ :

$$\begin{aligned} \mathbf{z}_t^{(l)} &= \text{CCT}^{(l)}(\mathbf{z}_t^{(l-1)}), \quad l = 1, \dots, L_c \\ \mathbf{h}_t &= \mathbf{z}_{t,0}^{(L_c)}, \end{aligned} \quad (5)$$

where  $l$  denotes the block index in CCT,  $\mathbf{z}_{t,0}^{(L_c)}$  represents the final output of the [class] token. CCT is built-up with the proposed cross-frame attention, as will be elaborated later.

At last, the  $L_m$ -layer Multi-frame Integration Transformer (MIT) takes all frame representation  $\mathbf{H} = [\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_T]$  as input and outputs the video-level representation  $\mathbf{v}$  as following:

$$\mathbf{v} = \text{AvgPool}(\text{MIT}(\mathbf{H} + \mathbf{e}^{temp})), \quad (6)$$

where AvgPool and  $\mathbf{e}^{temp}$  denote the average pooling and temporal position encoding, respectively. We use standard learnable absolute position embeddings [47] for  $\mathbf{e}^{spa}$  and  $\mathbf{e}^{temp}$ . The multi-frame integration transformer is constructed by the standard multi-head self-attention and feed-forward networks [47].

**Cross-frame Attention.** To enable a cross-frame information exchange, we propose a new attention module. It consists of two types of attentions, *i.e.*, cross-frame fusion attention (CFA) and intra-frame diffusion attention (IFA), with a feed-forward network (FFN). We introduce a *message token* mechanism for each frame to abstract, send and receive information, thus enabling visual information to exchange across frames, as shown in Fig. 3(a). In detail, the message token  $\mathbf{m}_t^{(l)}$  for the  $t$ -th frame at the  $l$ -th layer is obtained by employing a linear transformation on the [class] token  $\mathbf{z}_{t,0}^{(l-1)}$ . This allows message tokens to abstract the visual information of the current frame.

Then, the cross-frame fusion attention (CFA) involves all message tokens to learn the global spatio-temporal dependencies of the input video. Mathematically, this process at  $l$ -th block can be expressed as:

$$\hat{\mathbf{M}}^{(l)} = \mathbf{M}^{(l)} + \text{CFA}(\text{LN}(\mathbf{M}^{(l)})), \quad (7)$$

where  $\hat{\mathbf{M}}^{(l)} = [\hat{\mathbf{m}}_1^{(l)}, \hat{\mathbf{m}}_2^{(l)}, \dots, \hat{\mathbf{m}}_T^{(l)}]$  and LN indicates layer normalization [4].

Next, the intra-frame diffusion (IFA) takes the frame tokens with the associated message token to learn visual representation, while the involved message token could also diffuse global spatio-temporal dependencies for learning. Mathematically, this process at  $l$ -th block can be formulated as:

$$[\hat{\mathbf{z}}_t^{(l)}, \hat{\mathbf{m}}_t^{(l)}] = [\mathbf{z}_t^{(l-1)}, \hat{\mathbf{m}}_t^{(l)}] + \text{IFA}(\text{LN}([\mathbf{z}_t^{(l-1)}, \hat{\mathbf{m}}_t^{(l)}])), \quad (8)$$

where  $[\cdot, \cdot]$  concatenates the features of frame tokens and message tokens.

Finally, the feed-forward network(FFN) performs on the frame tokens as:

$$\mathbf{z}_t^{(l)} = \hat{\mathbf{z}}_t^{(l)} + \text{FFN}(\text{LN}(\hat{\mathbf{z}}_t^{(l)})). \quad (9)$$Note that the message token is dropped before the FFN layer and does not pass through the next block, since it is generated online and used for frames communication within each block. Alternating the fusion and diffusion attentions through  $L_c$  blocks, the cross-frame communication transformer (CCT) can encode the global spatial and temporal information of video frames. Compared to other space-time attention mechanisms [3,5,30], as presented in Fig. 3(b), our proposed cross-frame attention models the global spatio-temporal information while greatly reducing the computational cost.

*Initialization.* When adapting the pretrained image encoder to the video encoder, there are two key modifications. 1) The intra-frame diffusion attention (IFA) inherits the weights directly from the pretrained models, while the cross-frame fusion attention (CFA) is randomly initialized. 2) The multi-frame integration transformer is appended to the pretrained models with random initialization.

### 3.3 Text Encoder

We employ the pretrained text encoder and expand it with a video-specific prompting scheme. The key idea is to use video content to enhance the text representation. Given a description  $C$  about a video, the text representation  $\mathbf{c}$  is obtained by the text encoder, where  $\mathbf{c} = f_{\theta_c}(C)$ . For video recognition, how to generate a good text description  $C$  for each video is a challenging problem. Previous work, such as CLIP [37], usually defines textual prompts manually, such as “A photo of a {label}”. However, in this work, we empirically show that such manually-designed prompts do not improve the performance for video recognition (as presented in Tab. 9). In contrast, we just use the “{label}” as the text description  $C$  and then propose a learnable text prompting scheme.

**Video-specific prompting.** When understanding an image or a video, human can instinctively seek helps from discriminative visual cues. For example, the extra video semantic information of “in the water” will make it easier to distinguish “swimming” from “running”. However, it is difficult to acquire such visual semantics in video recognition tasks, because 1) the datasets only provide the category names, such as “swimming” and “running”, which are pre-defined and fixed; and 2) the videos in the same class share the identical category name, but their visual context and content are different. To address these issues, we propose a learnable prompting scheme to generate textual representation automatically. Concretely, we design a video-specific prompting module, which takes the video content representation  $\bar{\mathbf{z}}$  and text representation  $\mathbf{c}$  as inputs. Each block in the video-specific prompting module is consisting of a multi-head self-attention (MHSA) [47] followed by a feed-forward network to learn the prompts,

$$\begin{aligned}\bar{\mathbf{c}} &= \mathbf{c} + \text{MHSA}(\mathbf{c}, \bar{\mathbf{z}}), \\ \tilde{\mathbf{c}} &= \bar{\mathbf{c}} + \text{FFN}(\bar{\mathbf{c}}),\end{aligned}\tag{10}$$

where  $\mathbf{c}$  is the text embedding,  $\bar{\mathbf{z}} \in \mathbb{R}^{N \times d}$  is the average of  $\{\mathbf{z}_t^{(L_c)}\}_{t=1}^T$  along the temporal dimension, and  $\tilde{\mathbf{c}}$  is the video-specific prompts. We use text representa-Table 1: Comparison with state-of-the-art on Kinetics-400. We report the FLOPs and throughput per view. Throughput is measured using the GitHub repository of [29] on a V100 GPU. \* indicates pretraining with a video-text collection.

<table border="1">
<thead>
<tr>
<th>Method</th>
<th>Pretrain</th>
<th>Frames</th>
<th>Top-1</th>
<th>Top-5</th>
<th>Views</th>
<th>FLOPs(G)</th>
<th>Throughput</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="8"><i>Methods with random initialization</i></td>
</tr>
<tr>
<td>MViTv1-B, 64×3 [12]</td>
<td>-</td>
<td>64</td>
<td>81.2</td>
<td>95.1</td>
<td>3 × 3</td>
<td>455</td>
<td>7</td>
</tr>
<tr>
<td colspan="8"><i>Methods with ImageNet pretraining</i></td>
</tr>
<tr>
<td>Uniformer-B [25]</td>
<td>IN-1k</td>
<td>32</td>
<td>83.0</td>
<td>95.4</td>
<td>4 × 3</td>
<td>259</td>
<td>-</td>
</tr>
<tr>
<td>TimeSformer-L [5]</td>
<td>IN-21k</td>
<td>96</td>
<td>80.7</td>
<td>94.7</td>
<td>1 × 3</td>
<td>2380</td>
<td>3</td>
</tr>
<tr>
<td>Mformer-HR [34]</td>
<td>IN-21k</td>
<td>16</td>
<td>81.1</td>
<td>95.2</td>
<td>10 × 3</td>
<td>959</td>
<td>-</td>
</tr>
<tr>
<td>Swin-L [30]</td>
<td>IN-21k</td>
<td>32</td>
<td>83.1</td>
<td>95.9</td>
<td>4 × 3</td>
<td>604</td>
<td>6</td>
</tr>
<tr>
<td>Swin-L (384↑) [30]</td>
<td>IN-21k</td>
<td>32</td>
<td>84.9</td>
<td>96.7</td>
<td>10 × 5</td>
<td>2107</td>
<td>-</td>
</tr>
<tr>
<td>MViTv2-L (312↑) [27]</td>
<td>IN-21k</td>
<td>40</td>
<td>86.1</td>
<td>97.0</td>
<td>5 × 3</td>
<td>2828</td>
<td>-</td>
</tr>
<tr>
<td colspan="8"><i>Methods with web-scale image pretraining</i></td>
</tr>
<tr>
<td>ViViT-H/16x2 [3]</td>
<td>JFT-300M</td>
<td>32</td>
<td>84.8</td>
<td>95.8</td>
<td>4 × 3</td>
<td>8316</td>
<td>-</td>
</tr>
<tr>
<td>TokenLearner-L/10 [40]</td>
<td>JFT-300M</td>
<td>-</td>
<td>85.4</td>
<td>96.3</td>
<td>4 × 3</td>
<td>4076</td>
<td>-</td>
</tr>
<tr>
<td>CoVeR [56]</td>
<td>JFT-3B</td>
<td>-</td>
<td><b>87.2</b></td>
<td>-</td>
<td>1 × 3</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td colspan="8"><i>Methods with web-scale language-image pretraining</i></td>
</tr>
<tr>
<td>ActionCLIP-B/16 [49]</td>
<td>CLIP-400M</td>
<td>32</td>
<td>83.8</td>
<td>96.2</td>
<td>10 × 3</td>
<td>563</td>
<td>-</td>
</tr>
<tr>
<td>A6 [21]</td>
<td>CLIP-400M</td>
<td>16</td>
<td>76.9</td>
<td>93.5</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>MTV-H [54]</td>
<td>WTS*</td>
<td>32</td>
<td><b>89.1</b></td>
<td>98.2</td>
<td>4 × 3</td>
<td>3705</td>
<td>-</td>
</tr>
<tr>
<td>X-Florence (384↑)</td>
<td>FLD-900M</td>
<td>8</td>
<td>86.2</td>
<td>96.6</td>
<td>4 × 3</td>
<td>2114</td>
<td>6</td>
</tr>
<tr>
<td>X-Florence</td>
<td>FLD-900M</td>
<td>32</td>
<td>86.5</td>
<td>96.9</td>
<td>4 × 3</td>
<td>2822</td>
<td>2</td>
</tr>
<tr>
<td>X-CLIP-B/16</td>
<td>IN-21k</td>
<td>8</td>
<td>81.1</td>
<td>94.7</td>
<td>4 × 3</td>
<td>145</td>
<td>33</td>
</tr>
<tr>
<td>X-CLIP-B/32</td>
<td rowspan="6">CLIP-400M</td>
<td>8</td>
<td>80.4</td>
<td>95.0</td>
<td>4 × 3</td>
<td>39</td>
<td>136</td>
</tr>
<tr>
<td>X-CLIP-B/32</td>
<td>16</td>
<td>81.1</td>
<td>95.5</td>
<td>4 × 3</td>
<td>75</td>
<td>69</td>
</tr>
<tr>
<td>X-CLIP-B/16</td>
<td>8</td>
<td>83.8</td>
<td>96.7</td>
<td>4 × 3</td>
<td>145</td>
<td>33</td>
</tr>
<tr>
<td>X-CLIP-B/16</td>
<td>16</td>
<td>84.7</td>
<td>96.8</td>
<td>4 × 3</td>
<td>287</td>
<td>17</td>
</tr>
<tr>
<td>X-CLIP-L/14</td>
<td>8</td>
<td>87.1</td>
<td>97.6</td>
<td>4 × 3</td>
<td>658</td>
<td>8</td>
</tr>
<tr>
<td>X-CLIP-L/14 (336↑)</td>
<td>16</td>
<td><b>87.7</b></td>
<td>97.4</td>
<td>4 × 3</td>
<td>3086</td>
<td>2</td>
</tr>
</tbody>
</table>

tion  $\mathbf{c}$  as query and the video content representation  $\bar{\mathbf{z}}$  as key and value. This implementation allow the text representation to extract the related visual context from videos. We then enhance the text embedding  $\mathbf{c}$  with the video-specific prompts  $\tilde{\mathbf{c}}$  as follows,  $\hat{\mathbf{c}} = \mathbf{c} + \alpha\tilde{\mathbf{c}}$ , where  $\alpha$  is a learnable parameter with an initial value of 0.1. The  $\hat{\mathbf{c}}$  is finally used for classification in Eq. (3).

## 4 Experiments

In this section, we conduct experiments on different settings, *i.e.*, fully-supervised, zero-shot and few-shot, followed by the ablation studies of the proposed method.

### 4.1 Experimental Setup

**Architectures and Datasets.** We expand CLIP and Florence to derive four variants: X-CLIP-B/32, X-CLIP-B/16, X-CLIP-L/14 and X-Florence, respectively. In detail, there are three parts in our framework: a cross-frame communication transformer followed by a multi-frame integration transformer and a text encoder. X-CLIP-B/32 adopts ViT-B/32 as parts of the cross-frame communication transformer, X-CLIP-B/16 uses ViT-B/16, while X-CLIP-L/14 employs ViT-L/14. ForTable 2: Comparison with state-of-the-art on Kinetics-600.

<table border="1">
<thead>
<tr>
<th>Method</th>
<th>Pretrain</th>
<th>Frames</th>
<th>Top-1</th>
<th>Top-5</th>
<th>Views</th>
<th>FLOPs</th>
<th>Throughput</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="8"><i>Methods with random initialization</i></td>
</tr>
<tr>
<td>MViT-B-24, 32×3 [12]</td>
<td>-</td>
<td>32</td>
<td>83.8</td>
<td>96.3</td>
<td>5 × 1</td>
<td>236</td>
<td>-</td>
</tr>
<tr>
<td colspan="8"><i>Methods with ImageNet pretraining</i></td>
</tr>
<tr>
<td>Swin-L (384↑) [30]</td>
<td>IN-21k</td>
<td>32</td>
<td>86.1</td>
<td>97.3</td>
<td>10 × 5</td>
<td>2107</td>
<td>-</td>
</tr>
<tr>
<td colspan="8"><i>Methods with web-scale pretraining</i></td>
</tr>
<tr>
<td>ViViT-L/16x2 320 [3]</td>
<td>JFT-300M</td>
<td>32</td>
<td>83.0</td>
<td>95.7</td>
<td>4 × 3</td>
<td>3992</td>
<td>-</td>
</tr>
<tr>
<td>ViViT-H/16x2 [3]</td>
<td>JFT-300M</td>
<td>32</td>
<td>85.8</td>
<td>96.5</td>
<td>4 × 3</td>
<td>8316</td>
<td>-</td>
</tr>
<tr>
<td>TokenLearner-L/10 [40]</td>
<td>JFT-300M</td>
<td>-</td>
<td>86.3</td>
<td>97.0</td>
<td>4 × 3</td>
<td>4076</td>
<td>-</td>
</tr>
<tr>
<td>Florence (384↑) [55]</td>
<td>FLD-900M</td>
<td>-</td>
<td>87.8</td>
<td>97.8</td>
<td>4 × 3</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>CoVeR [56]</td>
<td>JFT-3B</td>
<td>-</td>
<td><b>87.9</b></td>
<td>-</td>
<td>1 × 3</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>MTV-H [54]</td>
<td>WTS*</td>
<td>32</td>
<td><b>89.6</b></td>
<td>98.3</td>
<td>4 × 3</td>
<td>3705</td>
<td>-</td>
</tr>
<tr>
<td>X-CLIP-B/16</td>
<td rowspan="3">CLIP-400M</td>
<td>8</td>
<td>85.3</td>
<td>97.1</td>
<td>4 × 3</td>
<td>145</td>
<td>74</td>
</tr>
<tr>
<td>X-CLIP-B/16</td>
<td>16</td>
<td>85.8</td>
<td>97.3</td>
<td>4 × 3</td>
<td>287</td>
<td>40</td>
</tr>
<tr>
<td>X-CLIP-L/14</td>
<td>8</td>
<td><b>88.3</b></td>
<td>97.7</td>
<td>4 × 3</td>
<td>658</td>
<td>20</td>
</tr>
</tbody>
</table>

all X-CLIP variants, we use a simple 1-layer multi-frame integration transformer. For X-Florence, we stack a 4-layer multi-frame integration transformer. The number of the video-specific prompting blocks is set to 2 for all variants. We evaluate the efficacy of our method on four benchmarks: *Kinetics-400&600* [23,7], *UCF-101* [42] and *HMDB-51* [24]. More details about architectures and datasets are provided in the *supplementary materials*.

## 4.2 Fully-supervised Experiments

**Training and Inference.** We sample 8 or 16 frames with a sparse sampling method in fully-supervised experiments. All the expanded models are trained with 32 NVIDIA 32G V100 GPUs. The detailed hyperparameters are showed in the *supplementary materials*.

**Results.** In Tab. 1, we report the results on Kinetics-400 and compare our method with state-of-the-art under different pretraining, including random initialization, ImageNet-1k/21k [10] pretraining, web-scale image and language-image pretraining. We develop a family of models with different FLOPs by setting the number of sampled frames to 8 or 16.

Compared to the methods pretrained on ImageNet-21k [10], our X-CLIP-B/16<sub>8f</sub> (with 8 sampled frames) surpasses Swin-L [29] by +0.7% with 4× fewer FLOPs and running 5× faster(as presented in Fig. 1). The underlying reason is that the 3D shift-window attention in Swin is inefficient. Also, our X-CLIP-L/14<sub>8f</sub> outperforms MViTv2-L [27] by +1.0% with 5× fewer FLOPs. In addition, when pretraining the video encoder only on IN-21k, our method achieves higher performance than the recent TimeSformer-L [5] with fewer computation cost.

When compared to the methods using web-scale image pretraining, our X-CLIP is also competitive. For example, X-CLIP-L/14<sub>8f</sub> achieves +2.3% higher accuracy than ViViT-H [3] with 12× fewer FLOPs. MTV-H [54] achieves better results than ours, but it uses much more pretraining data. Specifically, MTV-HTable 3: Zero-shot performances on HMDB51 [24] and UCF101 [42].

<table border="1">
<thead>
<tr>
<th>Method</th>
<th>HMDB-51</th>
<th>UCF-101</th>
</tr>
</thead>
<tbody>
<tr>
<td>MTE [53]</td>
<td>19.7 <math>\pm</math> 1.6</td>
<td>15.8 <math>\pm</math> 1.3</td>
</tr>
<tr>
<td>ASR [50]</td>
<td>21.8 <math>\pm</math> 0.9</td>
<td>24.4 <math>\pm</math> 1.0</td>
</tr>
<tr>
<td>ZSECOC [35]</td>
<td>22.6 <math>\pm</math> 1.2</td>
<td>15.1 <math>\pm</math> 1.7</td>
</tr>
<tr>
<td>UR [64]</td>
<td>24.4 <math>\pm</math> 1.6</td>
<td>17.5 <math>\pm</math> 1.6</td>
</tr>
<tr>
<td>TS-GCN [15]</td>
<td>23.2 <math>\pm</math> 3.0</td>
<td>34.2 <math>\pm</math> 3.1</td>
</tr>
<tr>
<td>E2E [6]</td>
<td>32.7</td>
<td>48</td>
</tr>
<tr>
<td>ER-ZSAR [8]</td>
<td>35.3 <math>\pm</math> 4.6</td>
<td>51.8 <math>\pm</math> 2.9</td>
</tr>
<tr>
<td>ActionCLIP [49]</td>
<td>40.8 <math>\pm</math> 5.4</td>
<td>58.3 <math>\pm</math> 3.4</td>
</tr>
<tr>
<td>X-CLIP-B/16</td>
<td><b>44.6 <math>\pm</math> 5.2</b><br/>(+3.8)</td>
<td><b>72.0 <math>\pm</math> 2.3</b><br/>(+13.7)</td>
</tr>
<tr>
<td>X-Florence</td>
<td><b>48.4 <math>\pm</math> 4.9</b><br/>(+7.6)</td>
<td><b>73.2 <math>\pm</math> 4.2</b><br/>(+14.9)</td>
</tr>
</tbody>
</table>

Table 4: Zero-shot performance on Kinetics-600 [7].

<table border="1">
<thead>
<tr>
<th>Method</th>
<th>Top-1 Acc.</th>
<th>Top-5 Acc.</th>
</tr>
</thead>
<tbody>
<tr>
<td>DEVISE [14]</td>
<td>23.8 <math>\pm</math> 0.3</td>
<td>51.0 <math>\pm</math> 0.6</td>
</tr>
<tr>
<td>ALE [1]</td>
<td>23.4 <math>\pm</math> 0.8</td>
<td>50.3 <math>\pm</math> 1.4</td>
</tr>
<tr>
<td>SJE [2]</td>
<td>22.3 <math>\pm</math> 0.6</td>
<td>48.2 <math>\pm</math> 0.4</td>
</tr>
<tr>
<td>ESZSL [39]</td>
<td>22.9 <math>\pm</math> 1.2</td>
<td>48.3 <math>\pm</math> 0.8</td>
</tr>
<tr>
<td>DEM [57]</td>
<td>23.6 <math>\pm</math> 0.7</td>
<td>49.5 <math>\pm</math> 0.4</td>
</tr>
<tr>
<td>GCN [17]</td>
<td>22.3 <math>\pm</math> 0.6</td>
<td>49.7 <math>\pm</math> 0.6</td>
</tr>
<tr>
<td>ER-ZSAR [8]</td>
<td>42.1 <math>\pm</math> 1.4</td>
<td>73.1 <math>\pm</math> 0.3</td>
</tr>
<tr>
<td>X-CLIP-B/16</td>
<td><b>65.2 <math>\pm</math> 0.4</b><br/>(+23.1)</td>
<td><b>86.1 <math>\pm</math> 0.8</b><br/>(+13.0)</td>
</tr>
<tr>
<td>X-Florence</td>
<td><b>68.8 <math>\pm</math> 0.9</b><br/>(+26.7)</td>
<td><b>88.4 <math>\pm</math> 0.6</b><br/>(+15.3)</td>
</tr>
</tbody>
</table>

uses a 70M video-text dataset with about 17B images, which are much larger than the 400M image-text data used in CLIP pretraining.

Moreover, compared to ActionCLIP [49], which also adopts CLIP as the pretrained model, our X-CLIP-L/14<sub>8f</sub> is still superior, getting +3.3% higher accuracy with fewer FLOPs. There are two factors leading to the smaller FLOPs of our method. One is that X-CLIP does not use 3D attention like [30] and has fewer layers. The other factor is that X-CLIP samples fewer frames for each video clip, such as 8 or 16 frames, while ActionCLIP [49] using 32 frames.

In addition, we report the results on Kinetics-600 in Tab. 2. Using only 8 frames, our X-CLIP-B/16<sub>8f</sub> achieves a higher top-1 accuracy compared to ViViT-L, while using 27 $\times$  fewer FLOPs. More importantly, our X-CLIP-L/14<sub>8f</sub> achieves 88.3% top-1 accuracy while using 5 $\times$  fewer FLOPs compared to the current state-of-the-art method MTV-H [54].

From the above fully-supervised experiments, we can observe that, our X-CLIP method achieves very competitive performance compared to prevailing video transformer models [56,55,54,49,21]. This mainly attributes to two factors. 1) The proposed cross-frame attention can effectively model temporal dependencies of video frames. 2) The joint language-image representation is successfully transferred to videos, unveiling its powerful generalization ability for recognition.

### 4.3 Zero-shot Experiments

**Training and Inference.** We pretrain X-CLIP-B/16 on Kinetics-400 with 32 frames. The single view inference is adopted. More details about the evaluation protocols are provided in the *supplementary materials*.

**Results.** Zero-shot video recognition is very challenging, because the categories in the test set are unseen to the model during training. We report the results in Tab. 3 and Tab. 4. On HMDB-51 [24] and UCF-101 [42] benchmarks, our X-CLIP outperforms the previous best results by +3.8% and +13.7% in terms of top-1 accuracy respectively, as reported in Tab. 3. On Kinetics-600 [7] as presented in Tab. 4, our X-CLIP outperforms the state-of-the-art ER-ZSAR [8] byTable 5: Few-shot results. Top-1 accuracy is reported with 32 frames.

<table border="1">
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="4">HMDB-51</th>
<th colspan="4">UCF-101</th>
</tr>
<tr>
<th><math>K=2</math></th>
<th><math>K=4</math></th>
<th><math>K=8</math></th>
<th><math>K=16</math></th>
<th><math>K=2</math></th>
<th><math>K=4</math></th>
<th><math>K=8</math></th>
<th><math>K=16</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>TSM [28]</td>
<td>17.5</td>
<td>20.9</td>
<td>18.4</td>
<td>31.0</td>
<td>25.3</td>
<td>47.0</td>
<td>64.4</td>
<td>61.0</td>
</tr>
<tr>
<td>TimeSformer [5]</td>
<td>19.6</td>
<td>40.6</td>
<td>49.4</td>
<td>55.4</td>
<td>48.5</td>
<td>75.6</td>
<td>83.7</td>
<td>89.4</td>
</tr>
<tr>
<td>Swin-B [30]</td>
<td>20.9</td>
<td>41.3</td>
<td>47.9</td>
<td>56.1</td>
<td>53.3</td>
<td>74.1</td>
<td>85.8</td>
<td>88.7</td>
</tr>
<tr>
<td>X-CLIP-B/16</td>
<td><b>53.0</b></td>
<td><b>57.3</b></td>
<td><b>62.8</b></td>
<td><b>64.0</b></td>
<td><b>76.4</b></td>
<td><b>83.4</b></td>
<td><b>88.3</b></td>
<td><b>91.4</b></td>
</tr>
<tr>
<td></td>
<td>(+32.1)</td>
<td>(+16.0)</td>
<td>(+13.4)</td>
<td>(+7.9)</td>
<td>(+23.1)</td>
<td>(+7.8)</td>
<td>(+2.5)</td>
<td>(+2.0)</td>
</tr>
<tr>
<td>X-Florence</td>
<td><b>51.6</b></td>
<td><b>57.8</b></td>
<td><b>64.1</b></td>
<td><b>64.2</b></td>
<td><b>84.0</b></td>
<td><b>88.5</b></td>
<td><b>92.5</b></td>
<td><b>94.8</b></td>
</tr>
<tr>
<td></td>
<td>(+30.7)</td>
<td>(+16.5)</td>
<td>(+14.7)</td>
<td>(+8.1)</td>
<td>(+30.7)</td>
<td>(+12.9)</td>
<td>(+6.7)</td>
<td>(+5.4)</td>
</tr>
</tbody>
</table>

+23.1%. Such remarkable improvements can be attributed to the proposed video-text learning framework, which leverages the large-scale visual-text pretraining and seamlessly integrates temporal cues and textual prompts.

#### 4.4 Few-shot Experiments

**Training and Inference.** A general  $K$ -shot setting is considered, *i.e.*,  $K$  examples are sampled from each category randomly for training. We compare with some representative methods, *i.e.*, TSM [28], TimeSformer [5] and Swin [30]. More details about the comparison methods and evaluation protocols are provided in the *supplementary materials*.

**Results.** Tab. 5 presents the results of  $K$ -shot learning. For the extreme case where  $K=2$ , we observe that for those single-modality methods, the performance drops significantly, demonstrating that over-fitting occurs due to the serious lack of data. In contrast, X-CLIP shows robustness by surpassing them with large margins. For example, X-CLIP-B/16 outperforms Swin-B by +32.1% and +23.1% in terms of top-1 accuracy on HMDB-51 and UCF-101 with  $K=2$ , respectively. Such large improvements are mainly due to the exploitation of the semantics in text representation. It further verifies the efficacy of transferring the knowledge of the pretrained language-image models to the few-shot models. We also observe that the performance gap between our method and others decreases as the sample size increases. It demonstrates increasing data can mitigate the over-fitting for other methods. Besides, it is noteworthy that the comparison of methods with CLIP pretraining and ImageNet pretraining is not fair enough. Hence, in Sec. 4.5, we provide an additional ablation analysis and verify the performance gains mainly comes from the use of textual information, rather than the CLIP pretraining.

#### 4.5 Ablation and Analysis

Unless stated otherwise, the fully-supervised experiments are performed on Kinetics-400, while the few-shot experiments are conducted on HMDB-51 with  $K=2$ . The zero-shot evaluation is on the first split of the validation set of UCF-101. We use X-CLIP-B/16<sub>8f</sub> with single-view inference in all experiments.

**Ablation.** *The effects of the proposed components.* Tab. 6 shows the performance evolution from the pretrained image CLIP to our expanded video X-CLIP. First,Table 6: Component-wise analysis of our X-CLIP and other techniques.

<table border="1">
<thead>
<tr>
<th>Components</th>
<th>Top-1.(%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Baseline(CLIP-Mean)</td>
<td>80.0</td>
</tr>
<tr>
<td>+ Cross-frame Communication</td>
<td>81.2(+1.2)</td>
</tr>
<tr>
<td>+ Multi-frame Integration</td>
<td>81.7(+1.7)</td>
</tr>
<tr>
<td>+ Video-specific Prompt</td>
<td>82.3(+2.3)</td>
</tr>
<tr>
<td>Techniques</td>
<td></td>
</tr>
<tr>
<td>+ 4×3-views Inference</td>
<td>83.8(+3.8)</td>
</tr>
</tbody>
</table>

Table 8: Ablation study on the effect of the text information.

<table border="1">
<thead>
<tr>
<th>Method</th>
<th>Zero-shot</th>
<th>Few-shot</th>
<th>Fully.</th>
</tr>
</thead>
<tbody>
<tr>
<td>w/o text</td>
<td>/</td>
<td>32.0</td>
<td>81.6</td>
</tr>
<tr>
<td>w/ text</td>
<td><b>70.0</b></td>
<td><b>50.8(+18.8)</b></td>
<td><b>82.3(+0.7)</b></td>
</tr>
</tbody>
</table>

Table 7: Ablation study on which part to finetune. ✓means finetuning. The CUDA memory is calculated on 2 video inputs, each containing 8 frames.

<table border="1">
<thead>
<tr>
<th>Visual Text</th>
<th>Zero.</th>
<th>Few.</th>
<th>Fully.</th>
<th>Mem.(G)</th>
</tr>
</thead>
<tbody>
<tr>
<td>✓</td>
<td>✓</td>
<td><b>72.9</b></td>
<td><b>54.6</b></td>
<td><b>82.4</b></td>
<td>22</td>
</tr>
<tr>
<td>✓</td>
<td>✗</td>
<td><b>70.0</b></td>
<td>50.8</td>
<td><b>82.3</b></td>
<td>6</td>
</tr>
<tr>
<td>✗</td>
<td>✓</td>
<td>66.8</td>
<td><b>53.4</b></td>
<td>79.3</td>
<td>20</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
<td>64.2</td>
<td>47.3</td>
<td>79.1</td>
<td>4</td>
</tr>
</tbody>
</table>

Table 9: Comparison with different prompting methods.

<table border="1">
<thead>
<tr>
<th>Method</th>
<th>Fully.</th>
<th>Few.</th>
<th>Zero.</th>
</tr>
</thead>
<tbody>
<tr>
<td>w/o prompt</td>
<td>81.7</td>
<td>49.6</td>
<td>63.2</td>
</tr>
<tr>
<td>Ensemble. [37]</td>
<td>81.7</td>
<td>49.6</td>
<td>63.9</td>
</tr>
<tr>
<td>Vectors. [61]</td>
<td>82.0</td>
<td>49.9</td>
<td>63.2</td>
</tr>
<tr>
<td>Ours</td>
<td><b>82.3(+0.3)</b></td>
<td><b>50.8(+0.9)</b></td>
<td><b>70.0(+6.1)</b></td>
</tr>
</tbody>
</table>

we design a simple baseline that averages the CLIP features of all video frames for classification, called CLIP-Mean. It uses the text supervision but does not utilize prompting technique. We can observe that equipping the original transformer in CLIP with our proposed cross-frame communication mechanism, *i.e.* Eq. (7-9), can improve the accuracy by +1.2%. Then, appending 1-layer multi-frame integration transformer (MIT) can further improve the accuracy by +0.5%. This illustrates that our X-CLIP framework can effectively leverage temporal cues in video clips. With the proposed video-specific prompting, X-CLIP can surpass the CLIP-Mean baseline by +2.3%. It demonstrates that the video-specific prompting scheme can generate more discriminative textual representation. Meanwhile, additionally using multi-view inference can boost the performance by +1.5%. Overall, with our proposed methods and all the techniques mentioned above, X-CLIP can boost the top-1 accuracy of the CLIP-Mean baseline from 80.0% to 83.8%.

*Which branch to finetune?* In order to demonstrate which branch should be finetuned when transferred to different downstream tasks, we separately freeze the parameters of the pretrained image and text encoder. Note that the randomly initialized parameters are always finetuned. From Tab. 7, we summarize the following observations. 1) For fully-supervised setting, finetuning the image encoder brings +3.0% improvements, while freezing the text encoder reduces the CUDA memory from 22G to 6G with minor performance loss. 2) For few-shot setting, we find the top-2 results are achieved by finetuning the text encoder. We conjecture the reason is that with few samples, the text encoder suffers less from the over-fitting than the over-parameterized image model. 3) For zero-shot setting, finetuning both the image and the text encoder achieves the best results.

*The effects of text.* To evaluate the impact of text, we replace the text encoder with a randomly initialized fully-connected layer as the classification head. From Tab. 8, we can observe that, without the text branch, the model cannot adapt toTable 10: Ablation study on the different pretraining.

<table border="1">
<thead>
<tr>
<th>Pretrain</th>
<th>Top-1.<br/>(%)</th>
<th>Top-5.<br/>(%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>ImageNet-1k</td>
<td>75.9</td>
<td>90.2</td>
</tr>
<tr>
<td>ImageNet-21k</td>
<td>79.8</td>
<td>94.0</td>
</tr>
</tbody>
</table>

Table 11: Comparison of two sampling methods.

<table border="1">
<thead>
<tr>
<th rowspan="2">#F</th>
<th rowspan="2">Train \ Test</th>
<th colspan="2">multi-view <math>\rightarrow</math> single-view</th>
</tr>
<tr>
<th>Dense</th>
<th>Sparse</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">8</td>
<td>Dense</td>
<td>81.9 <math>\rightarrow</math> 77.8(-4.1)</td>
<td>82.4 <math>\rightarrow</math> 81.1(-1.3)</td>
</tr>
<tr>
<td>Sparse</td>
<td>82.2 <math>\rightarrow</math> 77.3(-4.9)</td>
<td><b>83.4 <math>\rightarrow</math> 82.3(-1.1)</b></td>
</tr>
<tr>
<td rowspan="2">32</td>
<td>Dense</td>
<td>82.8 <math>\rightarrow</math> 78.8(-4.0)</td>
<td>83.2 <math>\rightarrow</math> 83.0(-0.2)</td>
</tr>
<tr>
<td>Sparse</td>
<td>83.0 <math>\rightarrow</math> 77.9(-5.1)</td>
<td><b>84.4 <math>\rightarrow</math> 84.2(-0.2)</b></td>
</tr>
</tbody>
</table>

zero-shot setting, because there is no data to initialize the head. For the few-shot and fully-supervised experiments, text information can bring +18.8% and +0.7% gains, respectively. This indicates the semantic information involved in text representation is beneficial to classification, especially for low-shot learning.

*The effects of pretraining.* We further investigate the effects of pretraining when expanding the language-image models to video. We use ViT-B/16 pretrained on ImageNet-1k/21k as the video encoder in our framework. As represented in Tab. 10, though the pretrained image encoder and text encoder are not in a joint embedding space, the model with IN-21k and IN-1k pretraining still achieve 79.8% and 75.9% top-1 accuracy on Kinetics-400, yet much inferior to the original CLIP large-scale pretraining (82.3%).

*Analysis. Comparison with other prompting methods.* We compare with two existing methods in Tab. 9: prompt ensembling [37] with 16 handcraft templates and learnable vectors [61] with length 16. It can be seen that our video-specific prompts outperforms others, especially in zero-shot setting (+6.1%). This demonstrates the efficacy of our method, which generates more adaptive prompts and better textual representation for unseen videos.

*Dense v.s. sparse sampling.* We further explore what is the best sampling strategy for our method in Tab. 11. We find that the dense sampling does not perform well as in previous works [30,13,3]. In contrast, the sparse sampling best matches our method. Regardless of the number of frames and views, using sparse sampling both in training and inference achieves the best performance.

*Single-view v.s. multi-view inference.* Although it can improve performance, multi-view inference takes relatively high computational cost, because the cost grows linearly with the number of views. In Tab. 11, we show that our multimodality models with sparse sampling is robust to the number of views, *i.e.*, single-view can achieve comparable performance to 10 temporal views. The underlying reason is the language-image models provide robust representation.

## 5 Conclusion

In this work, we present a simple approach that adapts the pretrained language-image models to video recognition. To capture the temporal information, we propose a cross-frame attention mechanism that explicitly exchanges information across frames. A video-specific prompting technique is designed to yield instance-level discriminative textual representation. Extensive experiments under three different learning scenarios demonstrate the effectiveness of our method. In future work, we plan to extend our method to different video tasks beyond classification.## References

1. 1. Akata, Z., Perronnin, F., Harchaoui, Z., Schmid, C.: Label-embedding for image classification. *IEEE T-PAMI* pp. 1425–1438 (2015)
2. 2. Akata, Z., Reed, S., Walter, D., Lee, H., Schiele, B.: Evaluation of output embeddings for fine-grained image classification. In: *CVPR*. pp. 2927–2936 (2015)
3. 3. Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lučić, M., Schmid, C.: Vivit: A video vision transformer. In: *ICCV*. pp. 6836–6846 (2021)
4. 4. Ba, J.L., Kiros, J.R., Hinton, G.E.: Layer normalization. *arXiv preprint arXiv:1607.06450* (2016)
5. 5. Bertasius, G., Wang, H., Torresani, L.: Is space-time attention all you need for video understanding? In: *ICML*. pp. 813–824 (2021)
6. 6. Brattoli, B., Tighe, J., Zhdanov, F., Perona, P., Chalupka, K.: Rethinking zero-shot video classification: End-to-end training for realistic applications. In: *CVPR*. pp. 4613–4623 (2020)
7. 7. Carreira, J., Noland, E., Banki-Horvath, A., Hillier, C., Zisserman, A.: A short note about kinetics-600. *arXiv preprint arXiv:1808.01340* (2018)
8. 8. Chen, S., Huang, D.: Elaborative rehearsal for zero-shot action recognition. In: *ICCV*. pp. 13638–13647 (2021)
9. 9. Contributors, M.: Openmmlab’s next generation video understanding toolbox and benchmark. <https://github.com/open-mmlab/mmaction2> (2020)
10. 10. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A large-scale hierarchical image database. In: *CVPR*. pp. 248–255. Ieee (2009)
11. 11. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. In: *ICLR* (2021)
12. 12. Fan, H., Xiong, B., Mangalam, K., Li, Y., Yan, Z., Malik, J., Feichtenhofer, C.: Multiscale vision transformers. In: *ICCV*. pp. 6824–6835 (2021)
13. 13. Feichtenhofer, C., Fan, H., Malik, J., He, K.: Slowfast networks for video recognition. In: *ICCV*. pp. 6202–6211 (2019)
14. 14. Frome, A., Corrado, G.S., Shlens, J., Bengio, S., Dean, J., Ranzato, M., Mikolov, T.: Devise: A deep visual-semantic embedding model. In: *NeurIPS*. pp. 2121–2129 (2013)
15. 15. Gao, J., Zhang, T., Xu, C.: I know the relationships: Zero-shot action recognition via two-stream graph convolutional networks and knowledge graphs. In: *AAAI*. vol. 33, pp. 8303–8311 (2019)
16. 16. Gao, P., Geng, S., Zhang, R., Ma, T., Fang, R., Zhang, Y., Li, H., Qiao, Y.: Clip-adapter: Better vision-language models with feature adapters. *arXiv preprint arXiv:2110.04544* (2021)
17. 17. Ghosh, P., Saini, N., Davis, L.S., Shrivastava, A.: All about knowledge graphs for actions. *arXiv preprint arXiv:2008.12432* (2020)
18. 18. Girdhar, R., Grauman, K.: Anticipative video transformer. In: *ICCV* (2021)
19. 19. Herath, S., Harandi, M., Porikli, F.: Going deeper into action recognition: A survey. *Image and vision computing* (2017)
20. 20. Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q., Sung, Y.H., Li, Z., Duerig, T.: Scaling up visual and vision-language representation learning with noisy text supervision. In: *ICML*. pp. 4904–4916 (2021)
21. 21. Ju, C., Han, T., Zheng, K., Zhang, Y., Xie, W.: Prompting visual-language models for efficient video understanding. In: *CVPR* (2022)1. 22. Karpathy, A., Toderici, G., Shetty, S., Leung, T., Sukthankar, R., Fei-Fei, L.: Large-scale video classification with convolutional neural networks. In: CVPR. pp. 1725–1732 (2014)
2. 23. Kay, W., Carreira, J., Simonyan, K., Zhang, B., Hillier, C., Vijayanarasimhan, S., Viola, F., Green, T., Back, T., Natsev, P., et al.: The kinetics human action video dataset. arXiv preprint arXiv:1705.06950 (2017)
3. 24. Kuehne, H., Jhuang, H., Garrote, E., Poggio, T., Serre, T.: Hmdb: a large video database for human motion recognition. In: ICCV. pp. 2556–2563 (2011)
4. 25. Li, K., Wang, Y., Zhang, J., Gao, P., Song, G., Liu, Y., Li, H., Qiao, Y.: Uniformer: Unifying convolution and self-attention for visual recognition. In: ICLR (2022)
5. 26. Li, Y., Ji, B., Shi, X., Zhang, J., Kang, B., Wang, L.: Tea: Temporal excitation and aggregation for action recognition. In: CVPR. pp. 909–918 (2020)
6. 27. Li, Y., Wu, C.Y., Fan, H., Mangalam, K., Xiong, B., Malik, J., Feichtenhofer, C.: Improved multiscale vision transformers for classification and detection. In: CVPR (2022)
7. 28. Lin, J., Gan, C., Han, S.: Tsm: Temporal shift module for efficient video understanding. In: ICCV. pp. 7083–7093 (2019)
8. 29. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: ICCV (2021)
9. 30. Liu, Z., Ning, J., Cao, Y., Wei, Y., Zhang, Z., Lin, S., Hu, H.: Video swin transformer. In: CVPR (2022)
10. 31. Liu, Z., Wang, L., Wu, W., Qian, C., Lu, T.: Tam: Temporal adaptive module for video recognition. In: ICCV. pp. 13708–13718 (2021)
11. 32. Miech, A., Zhukov, D., Alayrac, J.B., Tapaswi, M., Laptev, I., Sivic, J.: Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In: ICCV. pp. 2630–2640 (2019)
12. 33. Neimark, D., Bar, O., Zohar, M., Asselmann, D.: Video transformer network. arXiv preprint arXiv:2102.00719 (2021)
13. 34. Patrick, M., Campbell, D., Asano, Y., Misra, I., Metze, F., Feichtenhofer, C., Vedaldi, A., Henriques, J.F.: Keeping your eye on the ball: Trajectory attention in video transformers. In: NeurIPS (2021)
14. 35. Qin, J., Liu, L., Shao, L., Shen, F., Ni, B., Chen, J., Wang, Y.: Zero-shot action recognition with error-correcting output codes. In: CVPR. pp. 2833–2842 (2017)
15. 36. Qiu, Z., Yao, T., Mei, T.: Learning spatio-temporal representation with pseudo-3d residual networks. In: ICCV. pp. 5533–5541 (2017)
16. 37. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: ICML (2021)
17. 38. Rao, Y., Zhao, W., Chen, G., Tang, Y., Zhu, Z., Huang, G., Zhou, J., Lu, J.: Denseclip: Language-guided dense prediction with context-aware prompting. In: CVPR (2022)
18. 39. Romera-Paredes, B., Torr, P.: An embarrassingly simple approach to zero-shot learning. In: ICML. pp. 2152–2161 (2015)
19. 40. Ryoo, M., Piergiovanni, A., Arnab, A., Dehghani, M., Angelova, A.: Tokenlearner: Adaptive space-time tokenization for videos. In: NeurIPS (2021)
20. 41. Selva, J., Johansen, A.S., Escalera, S., Nasrollahi, K., Moeslund, T.B., Clapés, A.: Video transformers: A survey. arXiv preprint arXiv:2201.05991 (2022)
21. 42. Soomro, K., Zamir, A.R., Shah, M.: Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402 (2012)1. 43. Sun, C., Baradel, F., Murphy, K., Schmid, C.: Learning video representations using contrastive bidirectional transformer. In: ECCV (2020)
2. 44. Sun, C., Myers, A., Vondrick, C., Murphy, K., Schmid, C.: Videobert: A joint model for video and language representation learning. In: ICCV. pp. 7464–7473 (2019)
3. 45. Tran, D., Bourdev, L., Fergus, R., Torresani, L., Paluri, M.: Learning spatiotemporal features with 3d convolutional networks. In: ICCV. pp. 4489–4497 (2015)
4. 46. Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y., Paluri, M.: A closer look at spatiotemporal convolutions for action recognition. In: CVPR (2018)
5. 47. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. In: NeurIPS (2017)
6. 48. Wang, L., Xiong, Y., Wang, Z., Qiao, Y., Lin, D., Tang, X., Van Gool, L.: Temporal segment networks: Towards good practices for deep action recognition. In: ECCV. pp. 20–36 (2016)
7. 49. Wang, M., Xing, J., Liu, Y.: Actionclip: A new paradigm for video action recognition. arXiv preprint arXiv:2109.08472 (2021)
8. 50. Wang, Q., Chen, K.: Alternative semantic representations for zero-shot human action recognition. In: ECML PKDD. pp. 87–102 (2017)
9. 51. Xie, S., Sun, C., Huang, J., Tu, Z., Murphy, K.: Rethinking spatiotemporal feature learning: Speed-accuracy trade-offs in video classification. In: ECCV. pp. 305–321 (2018)
10. 52. Xu, H., Ghosh, G., Huang, P.Y., Okhonko, D., Aghajanyan, A., Metze, F., Zettlemoyer, L., Feichtenhofer, C.: Videoclip: Contrastive pre-training for zero-shot video-text understanding. In: EMNLP (2021)
11. 53. Xu, X., Hospedales, T.M., Gong, S.: Multi-task zero-shot action recognition with prioritised data augmentation. In: ECCV. pp. 343–359 (2016)
12. 54. Yan, S., Xiong, X., Arnab, A., Lu, Z., Zhang, M., Sun, C., Schmid, C.: Multiview transformers for video recognition. In: CVPR (2022)
13. 55. Yuan, L., Chen, D., Chen, Y.L., Codella, N., Dai, X., Gao, J., Hu, H., Huang, X., Li, B., Li, C., et al.: Florence: A new foundation model for computer vision. arXiv preprint arXiv:2111.11432 (2021)
14. 56. Zhang, B., Yu, J., Fifty, C., Han, W., Dai, A.M., Pang, R., Sha, F.: Co-training transformer with videos and images improves action recognition. arXiv preprint arXiv:2112.07175 (2021)
15. 57. Zhang, L., Xiang, T., Gong, S.: Learning a deep embedding model for zero-shot learning. In: CVPR. pp. 2021–2030 (2017)
16. 58. Zhang, R., Fang, R., Gao, P., Zhang, W., Li, K., Dai, J., Qiao, Y., Li, H.: Tip-adapter: Training-free clip-adapter for better vision-language modeling. arXiv preprint arXiv:2111.03930 (2021)
17. 59. Zhang, R., Guo, Z., Zhang, W., Li, K., Miao, X., Cui, B., Qiao, Y., Gao, P., Li, H.: Pointclip: Point cloud understanding by clip. In: CVPR (2021)
18. 60. Zhou, C., Loy, C.C., Dai, B.: Denseclip: Extract free dense labels from clip. arXiv preprint arXiv:2112.01071 (2021)
19. 61. Zhou, K., Yang, J., Loy, C.C., Liu, Z.: Learning to prompt for vision-language models. arXiv preprint arXiv:2109.01134 (2021)
20. 62. Zhu, L., Yang, Y.: Actbert: Learning global-local video-text representations. In: CVPR. pp. 8746–8755 (2020)
21. 63. Zhu, Y., Li, X., Liu, C., Zolfaghari, M., Xiong, Y., Wu, C., Zhang, Z., Tighe, J., Manmatha, R., Li, M.: A comprehensive study of deep video action recognition. arXiv preprint arXiv:2012.06567 (2020)
22. 64. Zhu, Y., Long, Y., Guan, Y., Newsam, S., Shao, L.: Towards universal representation for unseen action recognition. In: CVPR. pp. 9436–9445 (2018)# Expanding Language-Image Pretrained Models for General Video Recognition

---

## Supplementary Material

This supplementary material contains additional details of the main manuscript, and provides more experiment analysis. In Sec. 1, we present the details of our proposed architectures and the comparison methods. Next, we elaborate the hyperparameters in Sec. 2. Then, we overview the four datasets and provide the evaluation protocols of our experiments in Sec. 3. Finally, we provide more experiment analysis in Sec. 4.

## 1 Architecture Details

In this section, we elaborate the details of the proposed architectures in Sec. 1.1 and the compared architectures in the few-shot experiments in Sec. 1.2.

### 1.1 The proposed architectures

X-CLIP-B/32 adopts ViT-B/32 ( $L_c=12$ ,  $N_h=12$ ,  $d=768$ ,  $p=32$ ) as parts of the cross-frame communication transformer, X-CLIP-B/16 uses ViT-B/16 ( $L_c=12$ ,  $N_h=12$ ,  $d=768$ ,  $p=16$ ), while X-CLIP-L/14 employs ViT-L/14 ( $L_c=24$ ,  $N_h=16$ ,  $d=1,024$ ,  $p=14$ ), where  $L_c$  denotes the layers,  $N_h$  refers to the number of attention heads,  $d$  represents the embedding dimension and  $p$  is the patch size. We use a simple 1-layer multi-frame integration transformer for all three X-CLIP variants ( $L_m=1$ ,  $N_h=8$  for X-CLIP-B while  $N_h=12$  for X-CLIP-L). The text encoder is the same as in CLIP [37]. For Florence, we replace the cross-frame communication transformer with the pretrained CoSwin-H [55] visual encoder. We stack a 4-layer multi-frame integration transformer on top of CoSwin-H. The text encoder is the same as in Florence [55].

### 1.2 Other compared architectures

In few-shot experiments, we implemented the Video Swin [30], TSM [28] and TimeSformer [5] using MMACTION2 [9] library with the default hyperparameters. The TSM-R50 is initialized with ImageNet-1k pretraining, while the Video Swin-B and TimeSformer are initialized with ImageNet-21k pretraining.

## 2 Hyperparameter Details

In this section, we present the elaborated training hyperparameters in Sec. 2.1 and the hand-craft prompt templates in the Tab. 9 of the main manuscript in Sec. 2.2.Table 1: The training hyperparameters for all the experiments.

<table border="1">
<thead>
<tr>
<th></th>
<th>Fully-sup.</th>
<th>Few-shot</th>
<th>Zero-shot</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="4"><i>Optimisation</i></td>
</tr>
<tr>
<td>Optimizer</td>
<td></td>
<td>AdamW</td>
<td></td>
</tr>
<tr>
<td>Optimizer betas</td>
<td></td>
<td>(0.9, 0.98)</td>
<td></td>
</tr>
<tr>
<td>Batch size</td>
<td>256</td>
<td>64</td>
<td>256</td>
</tr>
<tr>
<td>Learning rate schedule</td>
<td></td>
<td>cosine</td>
<td></td>
</tr>
<tr>
<td>Linear warmup epochs</td>
<td></td>
<td>5</td>
<td></td>
</tr>
<tr>
<td>Base learning rate</td>
<td>8e-6</td>
<td>2e-6</td>
<td>8e-6</td>
</tr>
<tr>
<td>Minimal learning rate</td>
<td>8e-8</td>
<td>2e-8</td>
<td>8e-8</td>
</tr>
<tr>
<td>Epochs</td>
<td>30</td>
<td>50</td>
<td>10</td>
</tr>
<tr>
<td colspan="4"><i>Data augmentation</i></td>
</tr>
<tr>
<td>RandomFlip</td>
<td></td>
<td>0.5</td>
<td></td>
</tr>
<tr>
<td>MultiScaleCrop</td>
<td></td>
<td>(1, 0.875, 0.75, 0.66)</td>
<td></td>
</tr>
<tr>
<td>ColorJitter</td>
<td></td>
<td>0.8</td>
<td></td>
</tr>
<tr>
<td>GrayScale</td>
<td></td>
<td>0.2</td>
<td></td>
</tr>
<tr>
<td>Label smoothing</td>
<td></td>
<td>0.1</td>
<td></td>
</tr>
<tr>
<td>Mixup</td>
<td></td>
<td>0.8</td>
<td></td>
</tr>
<tr>
<td>Cutmix</td>
<td></td>
<td>1.0</td>
<td></td>
</tr>
<tr>
<td colspan="4"><i>Other regularisation</i></td>
</tr>
<tr>
<td>Weight decay</td>
<td></td>
<td>0.001</td>
<td></td>
</tr>
</tbody>
</table>

## 2.1 Training Hyperparameters

Tab. 1 presents the hyperparameters for our experiments, corresponding to Section 4.2-4.4 of the main manuscript. It is noteworthy that the learning rate of the randomly initialized parameters is  $10\times$  higher than the base learning rate.

## 2.2 Hand-craft Prompt Templates

In Tab. 9 of the main manuscript, we compare our video-specific prompting scheme with the existing prompt ensemble method [37] and demonstrate the superiority of our method. We construct 16 hand-craft templates [49] totally. We randomly choose one template in each training iteration, and the result in inference is the average result of all templates. The complete list of templates is as follows: a photo of action {label}; a picture of action {label}; Human action of {label}; {label}, an action; {label}, this is an action; {label}, a video of action; Playing action of {label}; {label}; Playing a kind of action, {label}; Doing a kind of action, {label}; Look, the human is {label}; Can you recognize the action of {label}; Video classification of {label}; A video of {label}; The man is {label}; The woman is {label}.

## 3 Datasets and Evaluation Protocols

In this section, we overview the four datasets briefly in Sec. 3.1. Then, we provide the evaluation protocols for different experiment settings, *i.e.*, zero-shot, few-shot and fully-supervised in Sec. 3.2-3.4, respectively.### 3.1 Datasets Overview

- – *Kinetics-400&600*. The Kinetics [23,7] dataset consists of 10-second video clips collected from YouTube. In particular, Kinetics-400 [23] consists of ~240k training videos and ~20k validation videos with 400 classes, while Kinetics-600 [7] consists of ~410k training videos and ~29k validation videos from 600 classes.
- – *UCF-101* [42]. UCF-101 is a video recognition dataset for realistic actions, collected from YouTube, including 13,320 video clips with 101 action categories in total. There are three splits of the training and test data.
- – *HMDB-51* [24]. It has around 7,000 videos with 51 classes, which is relatively small compared to UCF-101 and Kinetics. HMDB-51 has three splits of the training and test data.

### 3.2 Fully-supervised Experiments

We conduct the fully-supervised experiments on Kinetics-400&600. We use the complete training and validation sets for training and inference, respectively. During training, a sparse sampling strategy [48] is used. The number of frames is set to 8 or 16. We spatially scale the shorter side of each frame to 256 and take a 224 center crop. Following [30,3,5], we adopt the multi-view inference with 3 spatial crops and 4 temporal clips.

### 3.3 Few-shot Experiments

We randomly sample 2, 4, 8 and 16 videos from each class on UCF-101 and HMDB-51 for constructing the training set. For evaluation, we use the first split of the test set on UCF-101 and HMDB-51. We report the results with a single view of 32 frames.

### 3.4 Zero-shot Experiments

We train X-CLIP-B/16 with 32 frames on Kinetics-400. The single-view inference is adopted for our method. The same as [8,37], we apply the following two evaluation protocols in our zero-shot experiments. 1) *Evaluation for HMDB-51 and UCF-101*. Following [37], the prediction is conducted on the three splits of the test data, and we report the average top-1 accuracy and standard deviation. 2) *Evaluation for Kinetics-600*. Following [8], the 220 new categories outside Kinetics-400 [23] in Kinetics-600 are used for evaluation. The evaluation is conducted three times. For each iteration, we randomly sampled 160 categories for evaluation from the 220 categories in Kinetics-600.

## 4 Additional Experiments Analysis

In this section, we further compare different methods of adapting an image encoder to a video encoder in Sec. 4.1. Besides, we provide an analysis of aligningTable 2: Comparison with different video encoders. The video encoders are adapted from ViT-B/16 [37]. The fully-supervised experiment is conducted on Kinetics-400 [23]. The few-shot(2-shot) experiment is conducted on HMDB-51, and zero-shot experiment is conducted on UCF-101 [42].

<table border="1">
<thead>
<tr>
<th>Method</th>
<th>Zero-shot</th>
<th>Few-shot</th>
<th>Fully-supervised</th>
<th>FLOPs</th>
</tr>
</thead>
<tbody>
<tr>
<td>CLIP-One</td>
<td>62.5</td>
<td>46.2</td>
<td>69.9</td>
<td>14</td>
</tr>
<tr>
<td>CLIP-Joint</td>
<td>69.3</td>
<td>41.3</td>
<td>82.1</td>
<td>184</td>
</tr>
<tr>
<td><b>X-CLIP</b></td>
<td><b>70.0(+0.7)</b></td>
<td><b>50.8(+4.6)</b></td>
<td><b>82.3(+0.2)</b></td>
<td>145</td>
</tr>
</tbody>
</table>

Table 3: Comparison between the multi-modal framework and single-modal framework under ImageNet pretraining.

<table border="1">
<thead>
<tr>
<th>Method</th>
<th>Zero-shot</th>
<th>Few-shot</th>
<th>Method</th>
<th>Zero-shot</th>
<th>Few-shot</th>
</tr>
</thead>
<tbody>
<tr>
<td>w/o text</td>
<td>/</td>
<td>39.4</td>
<td>w/o text</td>
<td>/</td>
<td>10.8</td>
</tr>
<tr>
<td>w/ text</td>
<td><b>62.8</b></td>
<td><b>50.7(+11.3)</b></td>
<td>w/ text</td>
<td><b>58.0</b></td>
<td><b>46.0(+35.2)</b></td>
</tr>
</tbody>
</table>

(a) ImageNet-21k pretraining.

(b) ImageNet-1k pretraining.

the ImageNet pretrained video encoder and the CLIP pretrained text encoder in Sec. 4.2. Last, we further evaluate our proposed cross-frame communication transformer and multi-frame integration transformer on a simple single-modality classification setting in Sec. 4.3.

#### 4.1 Comparison with other video encoders adapted from images

Researchers have proposed several ways of adapting an image encoder to a video encoder [3,5]. We compare with two existing methods in Tab. 2. The first method is named “CLIP-One”, in which we randomly sample one frame and feed it to the pretrained image encoder. The second method is named “CLIP-Joint”, where we apply the joint space-time attention [3] that simply forwards all spatio-temporal tokens extracted from the video through the image encoder. Although the CLIP-Joint also considers global spatio-temporal information in videos, it takes more computational overhead than our proposed X-CLIP. What is more, our method surpasses the CLIP-Joint by +0.2% and +9.5% in the fully-supervised and few-shot experiments, respectively. We conjecture the reasons are two-fold. 1) CLIP-Joint considers the joint spatio-temporal tokens and therefore breaks the customary input pattern of the pretrained image encoder, which may impede the representation ability. In contrast, our method maintains the input pattern of the pretrained image encoder via modeling frame-level information, thus leveraging the strong representation ability of the pretrained image encoder. 2) The joint space-time attention requires more training data and training time to converge than our method.Table 4: Evaluating the proposed architecture in the single-modality framework.

<table border="1">
<thead>
<tr>
<th>Method</th>
<th>Top-1(%)</th>
<th>Top-5(%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>ViT-B/32-Mean</td>
<td>45.3</td>
<td>68.5</td>
</tr>
<tr>
<td>ViT-B/32 (Ours)</td>
<td><b>47.8(+2.5)</b></td>
<td><b>71.8(+3.3)</b></td>
</tr>
</tbody>
</table>

#### 4.2 Can ImageNet pretrained video encoder align with CLIP pretrained text encoder?

We have demonstrated that the video encoder with ImageNet pretraining still achieves competitive performance on the fully-supervised experiment in Tab. 10 of the main manuscript. However, the two embedding spaces of the ImageNet pretrained visual encoder and CLIP pretrained text encoder are not well aligned, which raises a question: *can we align the two embedding spaces without the web-scale joint pretraining, and then transfer the knowledge to zero-shot experiments?* To answer this question, we build an ImageNet pretrained video encoder and separate the text encoder from the pretrained CLIP. Then, we finetune the video encoder with the text supervision on Kinetics-400 to align the two embedding spaces. As a comparison, we also finetune a same video encoder but supervised by the discrete one-hot labels. Finally, we conduct the few-shot and zero-shot experiments using the two finetuned models to verify the transfer ability. The categories in few-shot and zero-shot experiments are not seen in finetuning. From Tab. 3, we can observe that the aligned model, *i.e.*, the model supervised by text information, achieves superior performance and surpasses the unaligned model by a large margin. It indicates that the ImageNet pretrained video encoder can still align with the CLIP pretrained text encoder by an acquired finetuning process using limited samples. The results also show the generality and flexibility of our proposed framework.

#### 4.3 Evaluation of the proposed architectures in the single-modality framework

We further evaluate the proposed cross-frame communication transformer and multi-frame integration transformer on a simple classification setting, *i.e.*, training from scratch with a single-modality framework on Kinetics-400. We use ViT-B/32<sub>8f</sub> as the backbone and adopt a fully-connected layer as the classification head. ViT-B/32-Mean averages the representation of all frames, while our method uses the cross-frame attention and stacks 1-layer multi-frame integration transformer on the top. We train both models 100 epochs with a learning rate  $1 \times 10^{-4}$ , and all the other hyperparameters are the same as in Tab. 1. From Tab. 4, it can be seen that our method outperforms the baseline +2.5% in terms of top-1 accuracy, which illustrates that our proposed architecture does not rely on pretraining and can help general video classification.

