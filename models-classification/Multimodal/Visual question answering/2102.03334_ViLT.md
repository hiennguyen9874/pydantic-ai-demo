# ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision

Wonjae Kim <sup>\*1†</sup> Bokyung Son <sup>\*1</sup> Ildoo Kim <sup>2</sup>

## Abstract

Vision-and-Language Pre-training (VLP) has improved performance on various joint vision-and-language downstream tasks. Current approaches to VLP heavily rely on image feature extraction processes, most of which involve region supervision (e.g., object detection) and the convolutional architecture (e.g., ResNet). Although disregarded in the literature, we find it problematic in terms of both (1) efficiency/speed, that simply extracting input features requires much more computation than the multimodal interaction steps; and (2) expressive power, as it is upper bounded to the expressive power of the visual embedder and its predefined visual vocabulary. In this paper, we present a minimal VLP model, Vision-and-Language Transformer (ViLT), monolithic in the sense that the processing of visual inputs is drastically simplified to just the same convolution-free manner that we process textual inputs. We show that ViLT is up to tens of times faster than previous VLP models, yet with competitive or better downstream task performance. Our code and pre-trained weights are available at <https://github.com/dandelin/vilt>.

## 1. Introduction

The pre-train-and-fine-tune scheme has been expanded to a joint domain of vision and language, giving birth to the category of *Vision-and-Language Pre-training (VLP)* models (Lu et al., 2019; Chen et al., 2019; Su et al., 2019; Li et al., 2019; Tan & Bansal, 2019; Li et al., 2020a; Lu et al., 2020; Cho et al., 2020; Qi et al., 2020; Zhou et al., 2020; Huang

<sup>\*</sup>Equal contribution <sup>†</sup>Current affiliation: NAVER AI Lab, Seongnam, Gyeonggi, Republic of Korea. <sup>1</sup>Kakao Enterprise, Seongnam, Gyeonggi, Republic of Korea <sup>2</sup>Kakao Brain, Seongnam, Gyeonggi, Republic of Korea. Correspondence to: Wonjae Kim <wonjae.kim@navercorp.com>.

Proceedings of the 38<sup>th</sup> International Conference on Machine Learning, PMLR 139, 2021. Copyright 2021 by the author(s).

Figure 1. Visual comparison of conventional VLP architectures and our proposed ViLT. We have entirely removed convolutional neural networks from the VLP pipeline without hurting performance on downstream tasks. ViLT is the first VLP model of which the modal-specific components require *less* computation than the transformer component for multimodal interactions.

et al., 2020; Li et al., 2020b; Gan et al., 2020; Yu et al., 2020; Zhang et al., 2021). These models are pre-trained with image text matching and masked language modeling objectives<sup>1</sup> on images and their aligned descriptions, and are fine-tuned on vision-and-language downstream tasks where the inputs involve two modalities.

To be fed into VLP models, image pixels need to be initially embedded in a dense form alongside language tokens. Since the seminal work of Krizhevsky et al. (2012), deep convolutional networks have been regarded as essential for this visual embedding step. Most VLP models employ an object detector pre-trained on the Visual Genome dataset (Krishna et al., 2017) annotated with 1,600 object classes and 400 attribute classes as in Anderson et al. (2018). Pixel-

<sup>1</sup>While some works employ additional objectives and data structures, these two objectives apply to almost every VLP model.Figure 2. Four categories of vision-and-language models. The height of each rectangle denotes its relative computational size. VE, TE, and MI are short for visual embedder, textual embedder, and modality interaction, respectively.

BERT (Huang et al., 2020) is one exception of this trend, as it uses ResNet variants (He et al., 2016; Xie et al., 2017) pre-trained on ImageNet classification (Russakovsky et al., 2015) embedding pixels in place of object detection modules.

To this date, most VLP studies have focused on improving performance by increasing the power of visual embedders. The shortcomings of having a heavy visual embedder are often disregarded in academic experiments because region features are commonly cached in advance at training time to ease the burden of feature extraction. However, the limitations are still evident in real-world applications as the queries in the wild have to undergo a slow extraction process.

To this end, we shift our attention to the lightweight and fast embedding of visual inputs. Recent work (Dosovitskiy et al., 2020; Touvron et al., 2020) demonstrated that using a simple linear projection of a patch is effective enough to embed pixels before feeding them into transformers. Whereas being the solid mainstream for text (Devlin et al., 2019), it is only recently that transformers (Vaswani et al., 2017) are used for images as well. We presume that the transformer module—used for modality interaction in VLP models—can also manage to process visual features in place of a convolutional visual embedder, just as it processes textual features.

This paper proposes the Vision-and-Language Transformer (ViLT) that handles two modalities in a single unified manner. It mainly differs from previous VLP models in its shallow, convolution-free embedding of pixel-level inputs. Removing deep embedders solely dedicated to visual inputs significantly cuts down the model size and running time by design. Figure 1 shows that our parameter-efficient model is tens of times faster than VLP models with region features and at least four times faster than those with grid features while exhibiting similar or even better performance on vision-and-language downstream tasks.

Our key contributions can be summarized as follows:

- • ViLT is the *simplest* architecture by far for a vision-and-language model as it commissions the transformer module to extract and process visual features in place of a separate deep visual embedder. This design inherently leads to significant runtime and parameter efficiency.
- • For the first time, we achieve competent performance on vision-and-language tasks without using region features or deep convolutional visual embedders in general.
- • Also, for the first time, we empirically show that whole word masking and image augmentations that were unprecedented in VLP training schemes further drive downstream performance.

## 2. Background

### 2.1. Taxonomy of Vision-and-Language Models

We propose a taxonomy of vision-and-language models based on two points: (1) whether the two modalities have an even level of expressiveness in terms of dedicated parameters and/or computation; and (2) whether the two modalities interact in a deep network. A combination of these points leads to four archetypes in Figure 2.

The *visual semantic embedding* (VSE) models such as VSE++ (Faghri et al., 2017) and SCAN (Lee et al., 2018) belong to Figure 2a. They use separate embedders for image and text, with the former being much heavier. Then, they represent the similarity of the embedded features from the two modalities with simple dot products or shallow attention layers.

CLIP (Radford et al., 2021) belongs to Figure 2b as it uses separate but equally expensive transformer embedders for each modality. Interaction between the pooled image vector and text vector is still shallow (dot product). Despite CLIP’s remarkable zero-shot performance on image-to-textretrieval, we could not observe the same level of performance on other vision-and-language downstream tasks. For instance, fine-tuning the MLP head on NLVR2 (Suhr et al., 2018) with the dot product of pooled visual and textual vectors from CLIP as the multimodal representation gives a low dev accuracy of  $50.99 \pm 0.38$  (ran with three different seeds); as chance level accuracy is 0.5, we conclude that the representations are incapable of learning this task. It also matches the findings of Suhr et al. (2018) that all models with simply fused multimodal representation failed to learn NLVR2.

This result backs up our speculation that simple fusion of outputs even from high-performing unimodal embedders may not be sufficient to learn complex vision-and-language tasks, bolstering the need for a more rigorous inter-modal interaction scheme.

Unlike models with shallow interaction, the more recent VLP models that fall under Figure 2c use a deep transformer to model the interaction of image and text features. Aside from the interaction module, however, convolutional networks are still involved in extracting and embedding image features, which accounts for most of the computation as depicted in Figure 1. Modulation-based vision-and-language models (Perez et al., 2018; Nguyen et al., 2020) also fall under Figure 2c, with their visual CNN stems corresponding to visual embedder, RNNs producing the modulation parameters to textual embedder, and modulated CNNs to modality interaction.

Our proposed ViLT is the first model of type Figure 2d where the embedding layers of raw pixels are shallow and computationally light as of text tokens. This architecture thereby concentrates most of the computation on modeling modality interactions.

## 2.2. Modality Interaction Schema

At the very core of contemporary VLP models lie transformers. They get visual and textual embedding sequences as input, model inter-modal and optionally intra-modal interactions throughout layers, then output a contextualized feature sequence.

Bugliarello et al. (2020) classifies interaction schema into two categories: (1) *single-stream* approaches (e.g., VisualBERT (Li et al., 2019), UNITER (Chen et al., 2019)) where layers collectively operate on a concatenation of image and text inputs; and (2) *dual-stream* approaches (e.g., ViLBERT (Lu et al., 2019), LXMERT (Tan & Bansal, 2019)) where the two modalities are not concatenated at the input level. We follow the single-stream approach for our interaction transformer module because the dual-stream approach introduces additional parameters.

## 2.3. Visual Embedding Schema

Whereas all performant VLP models share the same textual embedder— tokenizer from pre-trained BERT, word and position embeddings resembling those of BERT— they differ on visual embedders. Still, in most (if not all) cases, visual embedding is the bottleneck of existing VLP models. We focus on cutting corners on this step by introducing patch projection instead of using region or grid features for which heavy extraction modules are used.

**Region Feature.** VLP models dominantly utilize region features, also known as bottom-up features (Anderson et al., 2018). They are obtained from an off-the-shelf object detector like Faster R-CNN (Ren et al., 2016).

The general pipeline of generating region features is as follows. First, a region proposal network (RPN) proposes regions of interest (RoI) based on the grid features pooled from the CNN backbone. Non-maximum suppression (NMS) then reduces the number of RoIs to a few thousand. After being pooled by operations such as RoI Align (He et al., 2017), the RoIs go through RoI heads and become region features. NMS is again applied to every class, finally reducing the number of features under a hundred.

The above process involves several factors that affect the performance and runtime: the backbone, the style of NMS, the RoI heads. Previous works were lenient with controlling these factors, making varying choices from each other as listed in Table 7.<sup>2</sup>

- • Backbone: ResNet-101 (Lu et al., 2019; Tan & Bansal, 2019; Su et al., 2019) and ResNext-152 (Li et al., 2019; 2020a; Zhang et al., 2021) are two commonly used backbones.
- • NMS: NMS is typically done in a *per-class* fashion. Applying NMS to each and every class becomes a major runtime bottleneck with a large number of classes, e.g. 1.6K in the VG dataset (Jiang et al., 2020). *Class-agnostic* NMS was recently introduced to tackle this issue (Zhang et al., 2021).
- • RoI head: C4 heads were initially used (Anderson et al., 2018). FPN-MLP heads were introduced later (Jiang et al., 2018). As heads operate for each and every RoI, they pose a substantial runtime burden.

However lightweight, object detectors are less likely to be faster than the backbone or a single-layer convolution. Freezing the visual backbone and caching the region features in advance only helps at training time and not during

<sup>2</sup>Bugliarello et al. (2020) showed that a controlled setup bridges the performance gap of various region-feature-based VLP models.Figure 3. Model overview. Illustration inspired by Dosovitskiy et al. (2020).

inference, not to mention that it could hold performance back.

**Grid Feature.** Besides detector heads, the output feature grid of convolutional neural networks such as ResNets can also be used as visual features for vision-and-language pre-training. Direct use of grid features was first proposed by VQA-specific models (Jiang et al., 2020; Nguyen et al., 2020), mainly to avoid using severely slow region selection operations.

X-LXMERT (Cho et al., 2020) revisited grid features by fixing the region proposals to grids instead of those from the region proposal networks. However, their caching of features excluded further tuning of the backbone.

Pixel-BERT is the only VLP model that replaces the VG-pre-trained object detector with a ResNet variant backbone pre-trained with ImageNet classification. Unlike frozen detectors in region-feature-based VLP models, the backbone of Pixel-BERT is tuned during vision-and-language pre-training. The downstream performance of Pixel-BERT with ResNet-50 falls below region-feature-based VLP models, but it matches that of other competitors with the use of a much heavier ResNeXt-152.

We claim that grid features are not the go-to option, however, since deep CNNs are still expensive that they account for a large portion of the whole computation as in Figure 1.

**Patch Projection.** To minimize overhead, we adopt the simplest visual embedding scheme: *linear projection* that operates on image patches. The patch projection embedding was introduced by ViT (Dosovitskiy et al., 2020) for image classification tasks. Patch projection drastically simplifies the visual embedding step to the level of textual embedding, which also consists of simple projection (lookup) operations.

We use a  $32 \times 32$  patch projection which only requires 2.4M parameters. This is in sharp contrast to complex ResNe(X)t backbones<sup>3</sup> and detection components. Its running time is also ignorable as shown in Figure 1. We make a detailed runtime analysis in Section 4.6.

### 3. Vision-and-Language Transformer

#### 3.1. Model Overview

ViLT has a succinct architecture as a VLP model with a minimal visual embedding pipeline and following the single-stream approach.

We deviate from the literature that we initialize the interaction transformer weights from pre-trained ViT instead of BERT. Such initialization exploits the power of the interaction layers to process visual features while lacking a separate deep visual embedder.<sup>4</sup>

$$\bar{t} = [t_{\text{class}}; t_1 T; \dots; t_L T] + T^{\text{pos}} \quad (1)$$

$$\bar{v} = [v_{\text{class}}; v_1 V; \dots; v_N V] + V^{\text{pos}} \quad (2)$$

$$z^0 = [\bar{t} + t^{\text{type}}; \bar{v} + v^{\text{type}}] \quad (3)$$

$$\hat{z}^d = \text{MSA}(\text{LN}(z^{d-1})) + z^{d-1}, \quad d = 1 \dots D \quad (4)$$

$$z^d = \text{MLP}(\text{LN}(\hat{z}^d)) + \hat{z}^d, \quad d = 1 \dots D \quad (5)$$

$$p = \tanh(z_0^D W_{\text{pool}}) \quad (6)$$

ViT consists of stacked blocks that include a multiheaded self-attention (MSA) layer and an MLP layer. The position of layer normalization (LN) in ViT is the only difference from BERT: LN comes after MSA and MLP in BERT (“post-norm”) and before in ViT (“pre-norm”). The input

<sup>3</sup>Parameters for R50 is 25M, R101 is 44M, and X152 is 60M.

<sup>4</sup>We also experimented with initializing the layers from BERT weights and using the pre-trained patch projection from ViT, but it did not work.text  $t \in \mathbb{R}^{L \times |V|}$  is embedded to  $\bar{t} \in \mathbb{R}^{L \times H}$  with a word embedding matrix  $T \in \mathbb{R}^{|V| \times H}$  and a position embedding matrix  $T^{\text{pos}} \in \mathbb{R}^{(L+1) \times H}$ .

The input image  $I \in \mathbb{R}^{C \times H \times W}$  is sliced into patches and flattened to  $v \in \mathbb{R}^{N \times (P^2 \cdot C)}$  where  $(P, P)$  is the patch resolution and  $N = HW/P^2$ . Followed by linear projection  $V \in \mathbb{R}^{(P^2 \cdot C) \times H}$  and position embedding  $V^{\text{pos}} \in \mathbb{R}^{(N+1) \times H}$ ,  $v$  is embedded into  $\bar{v} \in \mathbb{R}^{N \times H}$ .

The text and image embeddings are summed with their corresponding modal-type embedding vectors  $t^{\text{type}}, v^{\text{type}} \in \mathbb{R}^H$ , then are concatenated into a combined sequence  $z^0$ . The contextualized vector  $z$  is iteratively updated through  $D$ -depth transformer layers up until the final contextualized sequence  $z^D$ .  $p$  is a pooled representation of the whole multimodal input, and is obtained by applying linear projection  $W_{\text{pool}} \in \mathbb{R}^{H \times H}$  and hyperbolic tangent upon the first index of sequence  $z^D$ .

For all experiments, we use weights from ViT-B/32 pre-trained on ImageNet, hence the name ViLT-B/32.<sup>5</sup> Hidden size  $H$  is 768, layer depth  $D$  is 12, patch size  $P$  is 32, MLP size is 3,072, and the number of attention heads is 12.

### 3.2. Pre-training Objectives

We train ViLT with two objectives commonly used to train VLP models: image text matching (ITM) and masked language modeling (MLM).

**Image Text Matching.** We randomly replace the aligned image with a different image with the probability of 0.5. A single linear layer ITM head projects the pooled output feature  $p$  to logits over binary class, and we compute negative log-likelihood loss as our ITM loss.

Plus, inspired by the word region alignment objective in Chen et al. (2019), we design word patch alignment (WPA) that computes the alignment score between two subsets of  $z^D$ :  $z^D|_t$  (textual subset) and  $z^D|_v$  (visual subset), using the inexact proximal point method for optimal transports (IPOT) (Xie et al., 2020). We set the hyperparameters of IPOT following Chen et al. (2019) ( $\beta = 0.5, N = 50$ ), and add the approximate wasserstein distance multiplied by 0.1 to the ITM loss.

**Masked Language Modeling.** This objective is to predict the ground truth labels of masked text tokens  $t_{\text{masked}}$  from its contextualized vector  $z_{\text{masked}}^D|_t$ . Following the heuristics of Devlin et al. (2019), we randomly mask  $t$  with the probability of 0.15.

<sup>5</sup>ViT-B/32 is pre-trained with ImageNet-21K and fine-tuned on ImageNet-1K for image classification. We expect that weights pre-trained on larger datasets (e.g., JFT-300M) would yield better performance.

We use a two-layer MLP MLM head that inputs  $z_{\text{masked}}^D|_t$  and outputs logits over vocabulary, just as the MLM objective of BERT. The MLM loss is then computed as the negative log-likelihood loss for the masked tokens.

### 3.3. Whole Word Masking

Whole word masking is a masking technique that masks all consecutive subword tokens that compose a whole word. It is shown to be effective on downstream tasks when applied to original and Chinese BERT (Cui et al., 2019).

We hypothesize that whole word masking is particularly crucial for VLP in order to make full use of information from the other modality. For example, the word ‘giraffe’ is tokenized into three wordpiece tokens [‘gi’, ‘##raf’, ‘##fe’] with the pre-trained bert-base-uncased tokenizer. If not all tokens are masked, say, [‘gi’, ‘[MASK]’, ‘##fe’], the model may solely rely on the nearby two language tokens [‘gi’, ‘##fe’] to predict the masked ‘##raf’ rather than using the information from the image.

We mask whole words with a mask probability of 0.15 during pre-training. We discuss its impact in Section 4.5.

### 3.4. Image Augmentation

Image augmentation reportedly improves the generalization power of vision models (Shorten & Khoshgoftar, 2019). DeiT (Touvron et al., 2020) that builds on ViT experimented with various augmentation techniques (Zhang et al., 2017; Yun et al., 2019; Berman et al., 2019; Hoffer et al., 2020; Cubuk et al., 2020), and found them beneficial for ViT training. However, the effects of image augmentation have not been explored within VLP models. Caching visual features restrains region-feature-based VLP models from using image augmentation. Notwithstanding its applicability, neither did Pixel-BERT study its effects.

To this end, we apply RandAugment (Cubuk et al., 2020) during fine-tuning. We use all the original policies except two: color inversion, because texts often contain color information as well, and cutout, as it may clear out small but important objects dispersed throughout the whole image. We use  $N = 2, M = 9$  as the hyperparameters. We discuss its impact in Section 4.5 and Section 5.

## 4. Experiments

### 4.1. Overview

We use four datasets for pre-training: Microsoft COCO (MSCOCO) (Lin et al., 2014), Visual Genome (VG) (Krishna et al., 2017), SBU Captions (SBU) (Ordonez et al., 2011), and Google Conceptual Captions (GCC) (SharmaTable 1. Pre-training dataset statistics. Caption length is the length of tokens from pre-trained `bert-base-uncased` tokenizer. † GCC and SBU provide only image urls, so we collect the images from urls which were still accessible.

<table border="1">
<thead>
<tr>
<th>Dataset</th>
<th># Images</th>
<th># Captions</th>
<th>Caption Length</th>
</tr>
</thead>
<tbody>
<tr>
<td>MSCOCO</td>
<td>113K</td>
<td>567K</td>
<td><math>11.81 \pm 2.81</math></td>
</tr>
<tr>
<td>VG</td>
<td>108K</td>
<td>5.41M</td>
<td><math>5.53 \pm 1.76</math></td>
</tr>
<tr>
<td>GCC†</td>
<td>3.01M</td>
<td>3.01M</td>
<td><math>10.66 \pm 4.93</math></td>
</tr>
<tr>
<td>SBU†</td>
<td>867K</td>
<td>867K</td>
<td><math>15.0 \pm 7.74</math></td>
</tr>
</tbody>
</table>

et al., 2018). Table 1 reports the dataset statistics.

We evaluate ViLT on two widely explored types of vision-and-language downstream tasks: for *classification*, we use VQAv2 (Goyal et al., 2017) and NLVR2 (Suhr et al., 2018), and for *retrieval*, we use MSCOCO and Flickr30K (F30K) (Plummer et al., 2015) re-splitted by Karpathy & Fei-Fei (2015). For the classification tasks, we fine-tune three times with different initialization seeds for the head and data ordering and report the mean scores. We report the standard deviation in Table 5 along with ablation studies. For the retrieval tasks, we only fine-tune once.

## 4.2. Implementation Details

For all experiments, we use AdamW optimizer (Loshchilov & Hutter, 2018) with base learning rate of  $10^{-4}$  and weight decay of  $10^{-2}$ . The learning rate was warmed up for 10% of the total training steps and was decayed linearly to zero for the rest of the training. Note that downstream performance may be further improved if we customize the hyperparameters to each task.

We resize the shorter edge of input images to 384 and limit the longer edge to under 640 while preserving the aspect ratio. This resizing scheme is also used during object detection in other VLP models, but with a larger size of the shorter edge (800). Patch projection of ViLT-B/32 yields  $12 \times 20 = 240$  patches for an image with a resolution of  $384 \times 640$ . As this is a rarely reached upper limit, we sample 200 patches at maximum during pre-training. We interpolate  $V^{\text{pos}}$  of ViT-B/32 to fit the size of each image and pad the patches for batch training. Note that the resulting image resolution is four times smaller than  $800 \times 1,333$ , which is the size that all other VLP models use for inputs to their visual embedders.

We use the `bert-base-uncased` tokenizer to tokenize text inputs. Instead of fine-tuning from pre-trained BERT, we learn the textual embedding-related parameters  $t_{\text{class}}$ ,  $T$ , and  $T^{\text{pos}}$  from scratch. Although beneficial *prima facie*, employing a pre-trained text-only BERT does not guarantee performance gain for vision and language downstream tasks. Counterevidence has already been reported by Tan & Bansal

Table 2. Comparison of ViLT-B/32 with other models on downstream classification tasks. We use MCAN (Yu et al., 2019) and MaxEnt (Suhr et al., 2018) for VQAv2 and NLVR2 w/o VLP SOTA results. † additionally used GQA, VQAv2, VG-QA for pre-training. ‡ made additional use of the Open Images (Kuznetsova et al., 2020) dataset. ① indicates RandAugment is applied during fine-tuning. ⊕ indicates model trained for a longer 200K pre-training steps.

<table border="1">
<thead>
<tr>
<th>Visual Embed</th>
<th>Model</th>
<th>Time (ms)</th>
<th>VQAv2 test-dev</th>
<th>NLVR2 dev</th>
<th>NLVR2 test-P</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="7">Region</td>
<td>w/o VLP SOTA</td>
<td>~900</td>
<td>70.63</td>
<td>54.80</td>
<td>53.50</td>
</tr>
<tr>
<td>ViLBERT</td>
<td>~920</td>
<td>70.55</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>VisualBERT</td>
<td>~925</td>
<td>70.80</td>
<td>67.40</td>
<td>67.00</td>
</tr>
<tr>
<td>LXMERT</td>
<td>~900</td>
<td>72.42</td>
<td>74.90</td>
<td>74.50</td>
</tr>
<tr>
<td>UNITER-Base</td>
<td>~900</td>
<td>72.70</td>
<td>75.85</td>
<td>75.80</td>
</tr>
<tr>
<td>OSCAR-Base†</td>
<td>~900</td>
<td>73.16</td>
<td>78.07</td>
<td>78.36</td>
</tr>
<tr>
<td>VinVL-Base‡</td>
<td>~650</td>
<td>75.95</td>
<td>82.05</td>
<td>83.08</td>
</tr>
<tr>
<td rowspan="2">Grid</td>
<td>Pixel-BERT-X152</td>
<td>~160</td>
<td>74.45</td>
<td>76.50</td>
<td>77.20</td>
</tr>
<tr>
<td>Pixel-BERT-R50</td>
<td>~60</td>
<td>71.35</td>
<td>71.70</td>
<td>72.40</td>
</tr>
<tr>
<td rowspan="3">Linear</td>
<td>ViLT-B/32</td>
<td>~15</td>
<td>70.33</td>
<td>74.41</td>
<td>74.57</td>
</tr>
<tr>
<td>ViLT-B/32①</td>
<td>~15</td>
<td>70.85</td>
<td>74.91</td>
<td>75.57</td>
</tr>
<tr>
<td>ViLT-B/32①⊕</td>
<td>~15</td>
<td>71.26</td>
<td>75.70</td>
<td>76.13</td>
</tr>
</tbody>
</table>

(2019), where initializing with pre-trained BERT parameters led to weaker performance than pre-training from scratch.

We pre-train ViLT-B/32 for 100K or 200K steps on 64 NVIDIA V100 GPUs with a batch size of 4,096. For all downstream tasks, we train for ten epochs with a batch size of 256 for VQAv2/retrieval tasks and 128 for NLVR2.

## 4.3. Classification Tasks

We evaluate ViLT-B/32 on two commonly used datasets: VQAv2 and NLVR2. We use a two-layer MLP of hidden size 1,536 as the fine-tuned downstream head.

**Visual Question Answering.** The VQAv2 task asks for answers given pairs of an image and a question in natural language. The annotated answers are originally in free-form natural language, but it is a common practice to convert the task to a classification task with 3,129 answer classes. Following this practice, we fine-tune ViLT-B/32 on the VQAv2 train and validation sets while reserving 1,000 validation images and their related questions for internal validation.

We report the test-dev score results<sup>6</sup> from the submission to the evaluation server. ViLT falls short of VQA score compared to other VLP models with a heavy visual embedder. We suspect a detached object representation generated by the object detector eases the training of VQA since questions in VQA typically ask about objects.

<sup>6</sup>VQA score is calculated by comparing the inferred answer to 10 ground-truth answers: see <https://visualqa.org/evaluation.html> for details.Table 3. Comparison of ViLT-B/32 with other VLP models on downstream zero-shot retrieval tasks. We exclude the models of which zero-shot retrieval performances were not reported in their original papers. † is pre-trained with a 10M proprietary vision-and-language dataset in addition to the 4M dataset of GCC+SBU. ⊕ indicates model trained for a longer 200K pre-training steps.

<table border="1">
<thead>
<tr>
<th rowspan="2">Visual Embed</th>
<th rowspan="2">Model</th>
<th rowspan="2">Time (ms)</th>
<th colspan="6">Zero-Shot Text Retrieval</th>
<th colspan="6">Zero-Shot Image Retrieval</th>
</tr>
<tr>
<th colspan="3">Flickr30k (1K)</th>
<th colspan="3">MSCOCO (5K)</th>
<th colspan="3">Flickr30k (1K)</th>
<th colspan="3">MSCOCO (5K)</th>
</tr>
<tr>
<th></th>
<th></th>
<th></th>
<th>R@1</th>
<th>R@5</th>
<th>R@10</th>
<th>R@1</th>
<th>R@5</th>
<th>R@10</th>
<th>R@1</th>
<th>R@5</th>
<th>R@10</th>
<th>R@1</th>
<th>R@5</th>
<th>R@10</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">Region</td>
<td>ViLBERT</td>
<td>~900</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>31.9</td>
<td>61.1</td>
<td>72.8</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>Unicoder-VL</td>
<td>~925</td>
<td>64.3</td>
<td>85.8</td>
<td>92.3</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>48.4</td>
<td>76.0</td>
<td>85.2</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>UNITER-Base</td>
<td>~900</td>
<td>80.7</td>
<td>95.7</td>
<td>98.0</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>66.2</td>
<td>88.4</td>
<td>92.9</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>ImageBERT†</td>
<td>~925</td>
<td>70.7</td>
<td>90.2</td>
<td>94.0</td>
<td>44.0</td>
<td>71.2</td>
<td>80.4</td>
<td>54.3</td>
<td>79.6</td>
<td>87.5</td>
<td>32.3</td>
<td>59.0</td>
<td>70.2</td>
</tr>
<tr>
<td rowspan="2">Linear</td>
<td>ViLT-B/32</td>
<td>~15</td>
<td>69.7</td>
<td>91.0</td>
<td>96.0</td>
<td>53.4</td>
<td>80.7</td>
<td>88.8</td>
<td>51.3</td>
<td>79.9</td>
<td>87.9</td>
<td>37.3</td>
<td>67.4</td>
<td>79.0</td>
</tr>
<tr>
<td>ViLT-B/32⊕</td>
<td>~15</td>
<td>73.2</td>
<td>93.6</td>
<td>96.5</td>
<td>56.5</td>
<td>82.6</td>
<td>89.6</td>
<td>55.0</td>
<td>82.5</td>
<td>89.8</td>
<td>40.4</td>
<td>70.0</td>
<td>81.1</td>
</tr>
</tbody>
</table>

Table 4. Comparison of ViLT-B/32 with other models on downstream retrieval tasks. We use SCAN for w/o VLP SOTA results. † additionally used GQA, VQAv2, VG-QA for pre-training. ‡ additionally used the Open Images dataset. ⊕ indicates RandAugment is applied during fine-tuning. ⊕ indicates model trained for a longer 200K pre-training steps.

<table border="1">
<thead>
<tr>
<th rowspan="2">Visual Embed</th>
<th rowspan="2">Model</th>
<th rowspan="2">Time (ms)</th>
<th colspan="6">Text Retrieval</th>
<th colspan="6">Image Retrieval</th>
</tr>
<tr>
<th colspan="3">Flickr30k (1K)</th>
<th colspan="3">MSCOCO (5K)</th>
<th colspan="3">Flickr30k (1K)</th>
<th colspan="3">MSCOCO (5K)</th>
</tr>
<tr>
<th></th>
<th></th>
<th></th>
<th>R@1</th>
<th>R@5</th>
<th>R@10</th>
<th>R@1</th>
<th>R@5</th>
<th>R@10</th>
<th>R@1</th>
<th>R@5</th>
<th>R@10</th>
<th>R@1</th>
<th>R@5</th>
<th>R@10</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="6">Region</td>
<td>w/o VLP SOTA</td>
<td>~900</td>
<td>67.4</td>
<td>90.3</td>
<td>95.8</td>
<td>50.4</td>
<td>82.2</td>
<td>90.0</td>
<td>48.6</td>
<td>77.7</td>
<td>85.2</td>
<td>38.6</td>
<td>69.3</td>
<td>80.4</td>
</tr>
<tr>
<td>ViLBERT-Base</td>
<td>~920</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>58.2</td>
<td>84.9</td>
<td>91.5</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>Unicoder-VL</td>
<td>~925</td>
<td>86.2</td>
<td>96.3</td>
<td>99.0</td>
<td>62.3</td>
<td>87.1</td>
<td>92.8</td>
<td>71.5</td>
<td>91.2</td>
<td>95.2</td>
<td>48.4</td>
<td>76.7</td>
<td>85.9</td>
</tr>
<tr>
<td>UNITER-Base</td>
<td>~900</td>
<td>85.9</td>
<td>97.1</td>
<td>98.8</td>
<td>64.4</td>
<td>87.4</td>
<td>93.1</td>
<td>72.5</td>
<td>92.4</td>
<td>96.1</td>
<td>50.3</td>
<td>78.5</td>
<td>87.2</td>
</tr>
<tr>
<td>OSCAR-Base†</td>
<td>~900</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>70.0</td>
<td>91.1</td>
<td>95.5</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>54.0</td>
<td>80.8</td>
<td>88.5</td>
</tr>
<tr>
<td>VinVL-Base†‡</td>
<td>~650</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>74.6</td>
<td>92.6</td>
<td>96.3</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>58.1</td>
<td>83.2</td>
<td>90.1</td>
</tr>
<tr>
<td rowspan="2">Grid</td>
<td>Pixel-BERT-X152</td>
<td>~160</td>
<td>87.0</td>
<td>98.9</td>
<td>99.5</td>
<td>63.6</td>
<td>87.5</td>
<td>93.6</td>
<td>71.5</td>
<td>92.1</td>
<td>95.8</td>
<td>50.1</td>
<td>77.6</td>
<td>86.2</td>
</tr>
<tr>
<td>Pixel-BERT-R50</td>
<td>~60</td>
<td>75.7</td>
<td>94.7</td>
<td>97.1</td>
<td>59.8</td>
<td>85.5</td>
<td>91.6</td>
<td>53.4</td>
<td>80.4</td>
<td>88.5</td>
<td>41.1</td>
<td>69.7</td>
<td>80.5</td>
</tr>
<tr>
<td rowspan="3">Linear</td>
<td>ViLT-B/32</td>
<td>~15</td>
<td>81.4</td>
<td>95.6</td>
<td>97.6</td>
<td>61.8</td>
<td>86.2</td>
<td>92.6</td>
<td>61.9</td>
<td>86.8</td>
<td>92.8</td>
<td>41.3</td>
<td>72.0</td>
<td>82.5</td>
</tr>
<tr>
<td>ViLT-B/32⊕</td>
<td>~15</td>
<td>83.7</td>
<td>97.2</td>
<td>98.1</td>
<td>62.9</td>
<td>87.1</td>
<td>92.7</td>
<td>62.2</td>
<td>87.6</td>
<td>93.2</td>
<td>42.6</td>
<td>72.8</td>
<td>83.4</td>
</tr>
<tr>
<td>ViLT-B/32⊕⊕</td>
<td>~15</td>
<td>83.5</td>
<td>96.7</td>
<td>98.6</td>
<td>61.5</td>
<td>86.3</td>
<td>92.7</td>
<td>64.4</td>
<td>88.7</td>
<td>93.8</td>
<td>42.7</td>
<td>72.9</td>
<td>83.1</td>
</tr>
</tbody>
</table>

**Natural Language for Visual Reasoning.** The NLVR2 task is a binary classification task given triplets of two images and a question in natural language. As there are two input images unlike the pre-training setup, multiple strategies exist<sup>7</sup>. Following OSCAR (Li et al., 2020b) and VinVL (Zhang et al., 2021), we use the *pair* method. Here, the triplet input is reformulated into two pairs (question, image1) and (question, image2), and each pair goes through the ViLT. The head takes the concatenation of two pooled representations ( $p$ ) as input and outputs the binary prediction.

Table 2 shows the results. ViLT-B/32 maintains competitive performance on both datasets considering its remarkable inference speed.

#### 4.4. Retrieval Tasks

We fine-tune ViLT-B/32 on the Karpathy & Fei-Fei (2015) split of MSCOCO and F30K. For image-to-text and text-to-image retrieval, we measure both zero-shot and fine-tuned performance<sup>8</sup>. We initialize the similarity score head from

<sup>7</sup>UNITER proposed three downstream head setups: pair, triplet, and pair-biattn.

<sup>8</sup>R@K corresponds to whether the ground truth is included among top K results from the validation set.

the pre-trained ITM head, particularly the part that computes the true-pair logits. We sample 15 random texts as negative samples and tune the model with cross-entropy loss that maximizes the scores on positive pairs.

We report the zero shot retrieval results in Table 3 and fine-tuned results in Table 4. At zero-shot retrieval, ViLT-B/32 performs better in general than ImageBERT despite ImageBERT’s pre-training on a larger (14M) dataset. At fine-tuned retrieval, recalls for ViLT-B/32 are higher by a large margin than the second fastest model (Pixel-BERT-R50).

#### 4.5. Ablation Study

In Table 5, we perform various ablations. More training steps, whole word masking, and image augmentation come to be beneficial, whereas an additional training objective does not help.

It has been reported that the number of training iterations affects the performance of self-supervised models (Devlin et al., 2019; Chen et al., 2020a;b). As VLP is also a form of self-supervised training, we examine the effects of training durations. As expected, the performance constantly increases as we train the model for longer training steps (rows 1~3). Masking whole words for the MLM objective (rows 3~4) and fine-tuning with augmentation (row 6) alsoTable 5. Ablation study of ViLT-B/32.  $\textcircled{W}$  denotes whether whole word masking is used for pre-training.  $\textcircled{M}$  denotes whether MPP objective is used for pre-training.  $\textcircled{A}$  denotes whether RandAugment is used during fine-tuning.

<table border="1">
<thead>
<tr>
<th rowspan="2">Training Steps</th>
<th colspan="3">Ablation</th>
<th rowspan="2">VQAv2 test-dev</th>
<th colspan="2">NLVR2</th>
<th colspan="2">Flickr30k R@1 (1K)</th>
<th colspan="2">MSCOCO R@1 (5K)</th>
</tr>
<tr>
<th><math>\textcircled{W}</math></th>
<th><math>\textcircled{M}</math></th>
<th><math>\textcircled{A}</math></th>
<th>dev</th>
<th>test-P</th>
<th>TR (ZS)</th>
<th>IR (ZS)</th>
<th>TR (ZS)</th>
<th>IR (ZS)</th>
</tr>
</thead>
<tbody>
<tr>
<td>25K</td>
<td>X</td>
<td>X</td>
<td>X</td>
<td>68.96 <math>\pm</math> 0.07</td>
<td>70.83 <math>\pm</math> 0.19</td>
<td>70.83 <math>\pm</math> 0.23</td>
<td>75.39 (45.12)</td>
<td>52.52 (31.80)</td>
<td>53.72 (31.55)</td>
<td>34.88 (21.58)</td>
</tr>
<tr>
<td>50K</td>
<td>X</td>
<td>X</td>
<td>X</td>
<td>69.80 <math>\pm</math> 0.01</td>
<td>71.93 <math>\pm</math> 0.27</td>
<td>72.92 <math>\pm</math> 0.82</td>
<td>78.13 (55.57)</td>
<td>57.36 (40.94)</td>
<td>57.00 (39.56)</td>
<td>37.47 (27.51)</td>
</tr>
<tr>
<td>100K</td>
<td>X</td>
<td>X</td>
<td>X</td>
<td>70.16 <math>\pm</math> 0.01</td>
<td>73.54 <math>\pm</math> 0.02</td>
<td>74.15 <math>\pm</math> 0.27</td>
<td>79.39 (66.99)</td>
<td>60.50 (47.62)</td>
<td>60.15 (51.25)</td>
<td>40.45 (34.59)</td>
</tr>
<tr>
<td>100K</td>
<td>O</td>
<td>X</td>
<td>X</td>
<td>70.33 <math>\pm</math> 0.01</td>
<td>74.41 <math>\pm</math> 0.21</td>
<td>74.57 <math>\pm</math> 0.09</td>
<td>81.35 (69.73)</td>
<td>61.86 (51.28)</td>
<td>61.79 (53.40)</td>
<td>41.25 (37.26)</td>
</tr>
<tr>
<td>100K</td>
<td>O</td>
<td>O</td>
<td>X</td>
<td>70.21 <math>\pm</math> 0.05</td>
<td>72.76 <math>\pm</math> 0.50</td>
<td>73.54 <math>\pm</math> 0.47</td>
<td>78.91 (63.67)</td>
<td>58.76 (46.96)</td>
<td>59.53 (47.75)</td>
<td>40.08 (32.28)</td>
</tr>
<tr>
<td>100K</td>
<td>O</td>
<td>X</td>
<td>O</td>
<td>70.85 <math>\pm</math> 0.13</td>
<td>74.91 <math>\pm</math> 0.29</td>
<td>75.57 <math>\pm</math> 0.61</td>
<td>83.69 (69.73)</td>
<td>62.22 (51.28)</td>
<td>62.88 (53.40)</td>
<td>42.62 (37.26)</td>
</tr>
<tr>
<td>200K</td>
<td>O</td>
<td>X</td>
<td>O</td>
<td>71.26 <math>\pm</math> 0.06</td>
<td>75.70 <math>\pm</math> 0.32</td>
<td>76.13 <math>\pm</math> 0.39</td>
<td>83.50 (73.24)</td>
<td>64.36 (54.96)</td>
<td>61.49 (56.51)</td>
<td>42.70 (40.42)</td>
</tr>
</tbody>
</table>

Table 6. Comparison of VLP models in terms of parameter size, FLOPs, and inference latency. Since FLOPs are proportional to input size, we denote the number of input tokens (image+text) as superscripts ("?") when text length is unreported; we arbitrarily use length 40). Although not captured in FLOPs count nor parameter size (because it is not a tensor operation), note that *per-class* NMS for 1,600 classes amounts to more than 500 ms in latency. NMS latency varies a lot according to the number of detected classes.

<table border="1">
<thead>
<tr>
<th>Visual Embed</th>
<th>Model</th>
<th>#Params (M)</th>
<th>#FLOPs (G)</th>
<th>Time (ms)</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="8">Region</td>
<td>ViLBERT<sup>36+36</sup></td>
<td>274.3</td>
<td>958.1</td>
<td>~900</td>
</tr>
<tr>
<td>VisualBERT<sup>36+128</sup></td>
<td>170.3</td>
<td>425.0</td>
<td>~925</td>
</tr>
<tr>
<td>LXMERT<sup>36+20</sup></td>
<td>239.8</td>
<td>952.0</td>
<td>~900</td>
</tr>
<tr>
<td>UNITER-Base<sup>36+60</sup></td>
<td>154.7</td>
<td>949.9</td>
<td>~900</td>
</tr>
<tr>
<td>OSCAR-Base<sup>50+35</sup></td>
<td>154.7</td>
<td>956.4</td>
<td>~900</td>
</tr>
<tr>
<td>VinVL-Base<sup>50+35</sup></td>
<td>157.3</td>
<td>1023.3</td>
<td>~650</td>
</tr>
<tr>
<td>Unicoder-VL<sup>100+?</sup></td>
<td>170.3</td>
<td>419.7</td>
<td>~925</td>
</tr>
<tr>
<td>ImageBERT<sup>100+44</sup></td>
<td>170.3</td>
<td>420.6</td>
<td>~925</td>
</tr>
<tr>
<td rowspan="2">Grid</td>
<td>Pixel-BERT-X152<sup>146+?</sup></td>
<td>144.3</td>
<td>185.8</td>
<td>~160</td>
</tr>
<tr>
<td>Pixel-BERT-R50<sup>260+?</sup></td>
<td>94.9</td>
<td>136.8</td>
<td>~60</td>
</tr>
<tr>
<td>Linear</td>
<td>ViLT-B/32<sup>200+40</sup></td>
<td>87.4</td>
<td>55.9</td>
<td>~15</td>
</tr>
</tbody>
</table>

drive performance. Further increase in training iterations to 200K improved performance on VQAv2, NLVR2, and zero-shot retrieval. We stop increasing the number of iterations over 200K as the fine-tuned text retrieval performance decreases afterward.

An additional masked region modeling (MRM) objective has been the key for performance boost in VLP models such as Chen et al. (2019). We experiment with masked patch prediction (MPP) (Dosovitskiy et al., 2020) which mimics the effect of MRM in a form compatible with patch projections. The patch  $v$  is masked with the probability of 0.15, and the model predicts the mean RGB value of the masked patch from its contextualized vector  $z_{\text{masked}}^D|_v$ . However, MPP turns out not to be contributing to downstream performance (rows 4~5). This result is in sharp contrast to the MRM objective on supervision signals from object detection.

Table 7. VLP model components. "PC" is for *per-class* manner NMS and "CA" is for *class-agnostic*. Following Tan & Bansal (2019), one single-modality layer is counted as 0.5 multi-modality layer.

<table border="1">
<thead>
<tr>
<th>Visual Embed</th>
<th>Model</th>
<th>CNN Backbone</th>
<th>RoI Head</th>
<th>NMS</th>
<th>Trans. Layers</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="8">Region</td>
<td>ViLBERT</td>
<td>R101</td>
<td>C4</td>
<td>PC</td>
<td>~15</td>
</tr>
<tr>
<td>VisualBERT</td>
<td>X152</td>
<td>FPN</td>
<td>PC</td>
<td>12</td>
</tr>
<tr>
<td>LXMERT</td>
<td>R101</td>
<td>C4</td>
<td>PC</td>
<td>~12</td>
</tr>
<tr>
<td>UNITER-Base</td>
<td>R101</td>
<td>C4</td>
<td>PC</td>
<td>12</td>
</tr>
<tr>
<td>OSCAR-Base</td>
<td>R101</td>
<td>C4</td>
<td>PC</td>
<td>12</td>
</tr>
<tr>
<td>VinVL-Base</td>
<td>X152</td>
<td>C4</td>
<td>CA</td>
<td>12</td>
</tr>
<tr>
<td>Unicoder-VL</td>
<td>X152</td>
<td>FPN</td>
<td>PC</td>
<td>12</td>
</tr>
<tr>
<td>ImageBERT</td>
<td>X152</td>
<td>FPN</td>
<td>PC</td>
<td>12</td>
</tr>
<tr>
<td rowspan="2">Grid</td>
<td>Pixel-BERT-X152</td>
<td>X152</td>
<td>-</td>
<td>-</td>
<td>12</td>
</tr>
<tr>
<td>Pixel-BERT-R50</td>
<td>R50</td>
<td>-</td>
<td>-</td>
<td>12</td>
</tr>
<tr>
<td>Linear</td>
<td>ViLT-B/32</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>12</td>
</tr>
</tbody>
</table>

#### 4.6. Complexity Analysis of VLP Models

We analyze the complexity of VLP models in various terms. In Table 6, we report the number of parameters, the number of floating-point operations (FLOPs), and the inference latency of the visual embedder and transformer. We exclude the textual embedder because it is shared by all VLP models<sup>9</sup>. The latency is averaged over 10K times on a Xeon E5-2650 CPU and an NVIDIA P40 GPU.

The input size in terms of image resolution and the length of concatenated multimodal input sequence affects the number of FLOPs. We co-note the sequence lengths. The image resolution is  $800 \times 1,333$  for region-based VLP models and Pixel-BERT-R50,  $600 \times 1,000$  for Pixel-BERT-X152, and  $384 \times 640$  for ViLT-B/32.

In Pixel-BERT and ViLT, visual tokens are sampled during pre-training and used in full during fine-tuning. We report the maximum number of visual tokens.

We observe that the runtime of BERT-base-like transformers varies only by  $< 1$  ms for input sequences of length under 300. Since patch projection of ViLT-B/32 generates at most

<sup>9</sup>FLOPs and time are neglectable because the operation is an embedding lookup. The 30K embedding dictionary used by bert-base-uncased has 23.47 M parametersFigure 4. Visualizations of transportation plan of word patch alignment. Best viewed zoomed in.

240 image tokens, our model can still be efficient even though it receives a combination of image and text tokens.

#### 4.7. Visualization

Figure 4 is an example of a cross-modal alignment. The transportation plan of WPA expresses a heatmap for a text token highlighted in pink color. Each square tile represents a patch, and its opacity indicates how much mass is transported from the highlighted word token.

More IPOT iterations— more than over 50 as in the training phase— help the visualization heatmap converge; empirically, 1,000 iterations are sufficient to get a clearly identifiable heatmap. We z-normalize the plan for each token and clamp the values to  $[1.0, 3.0]$ .

## 5. Conclusion and Future Work

In this paper, we present a minimal VLP architecture, Vision-and-Language Transformer (ViLT). ViLT is competent to competitors which are heavily equipped with convolutional visual embedding networks (e.g., Faster R-CNN and ResNets). We ask for future work on VLP to focus more on the modality interactions inside the transformer module rather than engaging in an arms race that merely powers up unimodal embedders.

Although remarkable as it is, ViLT-B/32 is more of a proof of concept that efficient VLP models free of convolution and region supervision can still be competent. We wrap up by pointing out a few factors that may add to the ViLT family.

**Scalability.** As shown in papers on large-scale transformers (Devlin et al., 2019; Dosovitskiy et al., 2020), the per-

formance of pre-trained transformers scale well given an appropriate amount of data. This observation paves the way for even better performing ViLT variants (e.g., ViLT-L (large) and ViLT-H (huge)). We leave training larger models for future work because aligned vision-and-language datasets are yet scarce.

**Masked Modeling for Visual Inputs.** Considering the success of MRM, we speculate that the masked modeling objective for the visual modality helps by preserving the information up until the last layer of the transformer. However, as observed in Table 5, a naive variant of MRM on image patches (MPP) fails.

Cho et al. (2020) proposed to train their grid RoIs on masked object classification (MOC) tasks. However, the visual vocabulary cluster in this work was fixed during the vision and language pre-training together with the visual backbone. For trainable visual embedders, one-time clustering is not a viable option. We believe that alternating clustering (Caron et al., 2018; 2019) or simultaneous clustering (Asano et al., 2019; Caron et al., 2020) methods studied in visual unsupervised learning research could be applied.

We encourage future work that does not use region supervision to devise a more sophisticated masking objective for the visual modality.

**Augmentation Strategies.** Previous work on contrastive visual representation learning (Chen et al., 2020a;b) showed that gaussian blur, not employed by RandAugment, brings noticeable gains to downstream performance compared with a simpler augmentation strategy (He et al., 2020). Exploration of appropriate augmentation strategies for textual and visual inputs would be a valuable addition.## References

Anderson, P., He, X., Buehler, C., Teney, D., Johnson, M., Gould, S., and Zhang, L. Bottom-up and top-down attention for image captioning and visual question answering. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 6077–6086, 2018.

Asano, Y., Rupprecht, C., and Vedaldi, A. Self-labelling via simultaneous clustering and representation learning. In *International Conference on Learning Representations*, 2019.

Berman, M., Jégou, H., Vedaldi, A., Kokkinos, I., and Douze, M. Multigrain: a unified image embedding for classes and instances. *arXiv preprint arXiv:1902.05509*, 2019.

Bugliarello, E., Cotterell, R., Okazaki, N., and Elliott, D. Multimodal pretraining unmasked: Unifying the vision and language berts. *arXiv preprint arXiv:2011.15124*, 2020.

Caron, M., Bojanowski, P., Joulin, A., and Douze, M. Deep clustering for unsupervised learning of visual features. In *Proceedings of the European Conference on Computer Vision (ECCV)*, pp. 132–149, 2018.

Caron, M., Bojanowski, P., Mairal, J., and Joulin, A. Unsupervised pre-training of image features on non-curated data. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 2959–2968, 2019.

Caron, M., Misra, I., Mairal, J., Goyal, P., Bojanowski, P., and Joulin, A. Unsupervised learning of visual features by contrasting cluster assignments. *arXiv preprint arXiv:2006.09882*, 2020.

Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. A simple framework for contrastive learning of visual representations. In *International conference on machine learning*, pp. 1597–1607. PMLR, 2020a.

Chen, X., Fan, H., Girshick, R., and He, K. Improved baselines with momentum contrastive learning. *arXiv preprint arXiv:2003.04297*, 2020b.

Chen, Y.-C., Li, L., Yu, L., Kholy, A. E., Ahmed, F., Gan, Z., Cheng, Y., and Liu, J. Uniter: Learning universal image-text representations. *arXiv preprint arXiv:1909.11740*, 2019.

Cho, J., Lu, J., Schwenk, D., Hajishirzi, H., and Kembhavi, A. X-lxmert: Paint, caption and answer questions with multi-modal transformers. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 8785–8805, 2020.

Cubuk, E. D., Zoph, B., Shlens, J., and Le, Q. V. Randaugment: Practical automated data augmentation with a reduced search space. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops*, pp. 702–703, 2020.

Cui, Y., Che, W., Liu, T., Qin, B., Yang, Z., Wang, S., and Hu, G. Pre-training with whole word masking for chinese bert. *arXiv preprint arXiv:1906.08101*, 2019.

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pp. 4171–4186, 2019.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al. An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*, 2020.

Faghri, F., Fleet, D. J., Kiros, J. R., and Fidler, S. Vse++: Improving visual-semantic embeddings with hard negatives. *arXiv preprint arXiv:1707.05612*, 2017.

Gan, Z., Chen, Y.-C., Li, L., Zhu, C., Cheng, Y., and Liu, J. Large-scale adversarial training for vision-and-language representation learning. *arXiv preprint arXiv:2006.06195*, 2020.

Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., and Parikh, D. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp. 6904–6913, 2017.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016.

He, K., Gkioxari, G., Dollár, P., and Girshick, R. Mask r-cnn. In *Proceedings of the IEEE international conference on computer vision*, pp. 2961–2969, 2017.

He, K., Fan, H., Wu, Y., Xie, S., and Girshick, R. Momentum contrast for unsupervised visual representation learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 9729–9738, 2020.

Hoffer, E., Ben-Nun, T., Hubara, I., Giladi, N., Hoefler, T., and Soudry, D. Augment your batch: Improving generalization through instance repetition. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 8129–8138, 2020.Huang, Z., Zeng, Z., Liu, B., Fu, D., and Fu, J. Pixel-bert: Aligning image pixels with text by deep multi-modal transformers. *arXiv preprint arXiv:2004.00849*, 2020.

Jiang, H., Misra, I., Rohrbach, M., Learned-Miller, E., and Chen, X. In defense of grid features for visual question answering. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 10267–10276, 2020.

Jiang, Y., Natarajan, V., Chen, X., Rohrbach, M., Batra, D., and Parikh, D. Pythia v0. 1: the winning entry to the vqa challenge 2018. *arXiv preprint arXiv:1807.09956*, 2018.

Karpathy, A. and Fei-Fei, L. Deep visual-semantic alignments for generating image descriptions. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 3128–3137, 2015.

Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L.-J., Shamma, D. A., et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. *International journal of computer vision*, 123(1):32–73, 2017.

Krizhevsky, A., Sutskever, I., and Hinton, G. E. Imagenet classification with deep convolutional neural networks. In *NIPS*, 2012.

Kuznetsova, A., Rom, H., Alldrin, N., Uijlings, J., Krasin, I., Pont-Tuset, J., Kamali, S., Popov, S., Malloci, M., Kolesnikov, A., et al. The open images dataset v4. *International Journal of Computer Vision*, pp. 1–26, 2020.

Lee, K.-H., Chen, X., Hua, G., Hu, H., and He, X. Stacked cross attention for image-text matching. In *Proceedings of the European Conference on Computer Vision (ECCV)*, pp. 201–216, 2018.

Li, G., Duan, N., Fang, Y., Gong, M., Jiang, D., and Zhou, M. Unicoder-vl: A universal encoder for vision and language by cross-modal pre-training. In *AAAI*, pp. 11336–11344, 2020a.

Li, L. H., Yatskar, M., Yin, D., Hsieh, C.-J., and Chang, K.-W. Visualbert: A simple and performant baseline for vision and language. *arXiv preprint arXiv:1908.03557*, 2019.

Li, X., Yin, X., Li, C., Zhang, P., Hu, X., Zhang, L., Wang, L., Hu, H., Dong, L., Wei, F., et al. Oscar: Object-semantics aligned pre-training for vision-language tasks. In *European Conference on Computer Vision*, pp. 121–137. Springer, 2020b.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., and Zitnick, C. L. Microsoft coco: Common objects in context. In *European conference on computer vision*, pp. 740–755. Springer, 2014.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In *International Conference on Learning Representations*, 2018.

Lu, J., Batra, D., Parikh, D., and Lee, S. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In *Advances in Neural Information Processing Systems*, pp. 13–23, 2019.

Lu, J., Goswami, V., Rohrbach, M., Parikh, D., and Lee, S. 12-in-1: Multi-task vision and language representation learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 10437–10446, 2020.

Nguyen, D.-K., Goswami, V., and Chen, X. Revisiting modulated convolutions for visual counting and beyond. *arXiv preprint arXiv:2004.11883*, 2020.

Ordonez, V., Kulkarni, G., and Berg, T. Im2text: Describing images using 1 million captioned photographs. *Advances in neural information processing systems*, 24:1143–1151, 2011.

Perez, E., Strub, F., De Vries, H., Dumoulin, V., and Courville, A. Film: Visual reasoning with a general conditioning layer. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 32, 2018.

Plummer, B. A., Wang, L., Cervantes, C. M., Caicedo, J. C., Hockenmaier, J., and Lazebnik, S. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In *Proceedings of the IEEE international conference on computer vision*, pp. 2641–2649, 2015.

Qi, D., Su, L., Song, J., Cui, E., Bharti, T., and Sacheti, A. Imagebert: Cross-modal pre-training with large-scale weak-supervised image-text data. *arXiv preprint arXiv:2001.07966*, 2020.

Radford, A., Sutskever, I., Kim, J., Krueger, G., and Agarwal, S. Learning transferable visual models from natural language supervision, 2021.

Ren, S., He, K., Girshick, R., and Sun, J. Faster r-cnn: Towards real-time object detection with region proposal networks. *IEEE transactions on pattern analysis and machine intelligence*, 39(6):1137–1149, 2016.

Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., et al. Imagenet large scale visual recognition challenge. *International journal of computer vision*, 115(3): 211–252, 2015.Sharma, P., Ding, N., Goodman, S., and Soricut, R. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 2556–2565, 2018.

Shorten, C. and Khoshgoftaar, T. M. A survey on image data augmentation for deep learning. *Journal of Big Data*, 6(1):1–48, 2019.

Su, W., Zhu, X., Cao, Y., Li, B., Lu, L., Wei, F., and Dai, J. Vi-bert: Pre-training of generic visual-linguistic representations. *arXiv preprint arXiv:1908.08530*, 2019.

Suhr, A., Zhou, S., Zhang, A., Zhang, I., Bai, H., and Artzi, Y. A corpus for reasoning about natural language grounded in photographs. *arXiv preprint arXiv:1811.00491*, 2018.

Tan, H. and Bansal, M. Lxmert: Learning cross-modality encoder representations from transformers. *arXiv preprint arXiv:1908.07490*, 2019.

Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., and Jégou, H. Training data-efficient image transformers & distillation through attention. *arXiv preprint arXiv:2012.12877*, 2020.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. *Advances in neural information processing systems*, 30:5998–6008, 2017.

Xie, S., Girshick, R., Dollár, P., Tu, Z., and He, K. Aggregated residual transformations for deep neural networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 1492–1500, 2017.

Xie, Y., Wang, X., Wang, R., and Zha, H. A fast proximal point method for computing exact wasserstein distance. In *Uncertainty in Artificial Intelligence*, pp. 433–453. PMLR, 2020.

Yu, F., Tang, J., Yin, W., Sun, Y., Tian, H., Wu, H., and Wang, H. Ernie-vil: Knowledge enhanced vision-language representations through scene graph. *arXiv preprint arXiv:2006.16934*, 2020.

Yu, Z., Yu, J., Cui, Y., Tao, D., and Tian, Q. Deep modular co-attention networks for visual question answering. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 6281–6290, 2019.

Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., and Yoo, Y. Cutmix: Regularization strategy to train strong classifiers with localizable features. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 6023–6032, 2019.

Zhang, H., Cisse, M., Dauphin, Y. N., and Lopez-Paz, D. mixup: Beyond empirical risk minimization. *arXiv preprint arXiv:1710.09412*, 2017.

Zhang, P., Li, X., Hu, X., Yang, J., Zhang, L., Wang, L., Choi, Y., and Gao, J. Vinvl: Making visual representations matter in vision-language models. *arXiv preprint arXiv:2101.00529*, 2021.

Zhou, L., Palangi, H., Zhang, L., Hu, H., Corso, J. J., and Gao, J. Unified vision-language pre-training for image captioning and vqa. In *AAAI*, pp. 13041–13049, 2020.

