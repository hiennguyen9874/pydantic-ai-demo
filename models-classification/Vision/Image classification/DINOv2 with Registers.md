Title: Vision Transformers Need Registers

URL Source: https://arxiv.org/html/2309.16588

Published Time: Mon, 15 Apr 2024 00:36:40 GMT

Markdown Content:
Timothée Darcet 1,2&Maxime Oquab 1&Julien Mairal 2&Piotr Bojanowski 1&1 FAIR, Meta 

2 Univ. Grenoble Alpes, Inria, CNRS, Grenoble INP, LJK, 38000 Grenoble, France

###### Abstract

Transformers have recently emerged as a powerful tool for learning visual representations. In this paper, we identify and characterize artifacts in feature maps of both supervised and self-supervised ViT networks. The artifacts correspond to high-norm tokens appearing during inference primarily in low-informative background areas of images, that are repurposed for internal computations. We propose a simple yet effective solution based on providing additional tokens to the input sequence of the Vision Transformer to fill that role. We show that this solution fixes that problem entirely for both supervised and self-supervised models, sets a new state of the art for self-supervised visual models on dense visual prediction tasks, enables object discovery methods with larger models, and most importantly leads to smoother feature maps and attention maps for downstream visual processing.

0 0 footnotetext: Correspondence to timdarcet@meta.com

Figure 1:  Register tokens enable interpretable attention maps in all vision transformers, similar to the original DINO method(Caron et al., [2021](https://arxiv.org/html/2309.16588v2#bib.bib5)). Attention maps are calculated in high resolution for better visualisation. More qualitative results are available in appendix [H](https://arxiv.org/html/2309.16588v2#A8 "Appendix H Qualitative Results ‣ Vision Transformers Need Registers"). 

1 Introduction
--------------

Embedding images into generic features that can serve multiple purposes in computer vision has been a long-standing problem. First methods relied on handcrafted principles, such as SIFT(Lowe, [2004](https://arxiv.org/html/2309.16588v2#bib.bib19)), before the scale of data and deep learning techniques allowed for end-to-end training. Pursuing generic feature embeddings is still relevant today, as collecting valuable annotated data for many specific tasks remains difficult. This difficulty arises because of the required expertise (_e.g._, medical data, or remote sensing) or the cost at scale. Today, it is common to pretrain a model for a task for which plenty of data is available and extract a subset of the model to use as a feature extractor. Multiple approaches offer this possibility; supervised methods, building on classification or text-image alignment, allow training strong feature models to unlock downstream tasks. Alternatively, self-supervised methods building on the Transformer architecture have attracted significant attention due to their high prediction performance on downstream tasks and the intriguing ability of some models to provide unsupervised segmentations(Caron et al., [2021](https://arxiv.org/html/2309.16588v2#bib.bib5))

In particular, the DINO algorithm is shown to produce models that contain explicit information about the semantic layout of an image. Indeed, qualitative results show that the last attention layer naturally focuses on semantically consistent parts of images and often produces interpretable attention maps. Exploiting these properties, object discovery algorithms such as LOST (Siméoni et al., [2021](https://arxiv.org/html/2309.16588v2#bib.bib26)) build on top of DINO. Such algorithms can detect objects without supervision by gathering information in attention maps. They are effectively unlocking a new frontier in computer vision.

DINOv2(Oquab et al., [2023](https://arxiv.org/html/2309.16588v2#bib.bib20)), a follow-up to DINO, provides features that allow tackling dense prediction tasks. DINOv2 features lead to successful monocular depth estimation and semantic segmentation with a frozen backbone and linear models. Despite the strong performance on dense tasks, we observed that DINOv2 is surprisingly incompatible with LOST. When used to extract features, it delivers disappointing performance, only on par with supervised alternative backbones in this scenario. This suggests that DINOv2 behaves differently than DINO. The investigation described in this work notably exposes the presence of artefacts in the feature maps of DINOv2 that were not present in the first version of this model. These are observable qualitatively using straightforward methods. Also surprisingly, applying the same observations to supervised vision transformers exposes similar artifacts, as shown in Fig.[2](https://arxiv.org/html/2309.16588v2#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Vision Transformers Need Registers"). This suggests that DINO is, in fact, an exception, while DINOv2 models match the baseline behavior of vision transformers.

Figure 2:  Illustration of artifacts observed in the attention maps of modern vision transformers. We consider ViTs trained with label supervision (DeiT-III), text-supervision (OpenCLIP) or self-supervision (DINO and DINOv2). Interestingly, all models but DINO exhibit peaky outlier values in the attention maps. The goal of this work is to understand and mitigate this phenomenon. 

In this work, we set out to better understand this phenomenon and develop methods to detect these artifacts. We observe that they are tokens with roughly 10x higher norm at the output and correspond to a small fraction of the total sequence (around 2%). We also show that these tokens appear around the middle layers of the vision transformer, and that they only appear after a sufficiently long training of a sufficiently big transformer. In particular, we show that these outlier tokens appear in patches similar to their neighbors, meaning patches that convey little additional information.

As part of our investigation, we evaluate the outlier tokens with simple linear models to understand the information they contain. We observe that, compared to non-outlier tokens, they hold less information about their original position in the image or the original pixels in their patch. This observation suggests that the model discards the local information contained in these patches during inference. On the other hand, learning an image classifier on outlier patches yields significantly stronger accuracy than doing so on the other patches, suggesting that they contain global information about the image. We propose the following interpretation to these elements: the model learns to recognize patches containing little useful information, and recycle the corresponding tokens to aggregate global image information while discarding spatial information.

This interpretation is consistent with an inner mechanism in transformer models that allows performing computations within a restricted set of tokens. In order to test this hypothesis, we append additional tokens - that we call registers - to the token sequence, independent of the input image. We train several models with and without this modification and observe that the outlier tokens disappear from the sequence entirely. As a result, the performance of the models increases in dense prediction tasks, and the resulting feature maps are significantly smoother. These smooth feature maps enable object discovery methods like LOST mentioned above with the updated models.

2 Problem Formulation
---------------------

As shown in Fig.[2](https://arxiv.org/html/2309.16588v2#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Vision Transformers Need Registers"), most modern vision transformers exhibit artifacts in the attention maps. The unsupervised DINO backbone(Caron et al., [2021](https://arxiv.org/html/2309.16588v2#bib.bib5)) has been previously praised for the quality of local features and interpretability of attention maps. Surprisingly, the outputs of the subsequent DINOv2 models have been shown to hold good local information but exhibit undesirable artifacts in attention maps. In this section, we propose to study why and when these artifacts appear. While this work focuses on alleviating artefacts in all vision transformers, we focus our analysis on DINOv2.

### 2.1 Artifacts in the local features of DINOv2

##### Artifacts are high-norm outlier tokens.

We want to find a quantitative way of characterizing artefacts that appear in the local features. We observe that an important difference between “artifact” patches and other patches is the norm of their token embedding at the output of the model. In Fig.[3](https://arxiv.org/html/2309.16588v2#S2.F3 "Figure 3 ‣ Outliers appear during the training of large models. ‣ 2.1 Artifacts in the local features of DINOv2 ‣ 2 Problem Formulation ‣ Vision Transformers Need Registers") (left), we compare the norm of local features for a DINO and DINOv2 model given a reference image. We clearly see that the norm of artifact patches is much higher than the norm of other patches. We also plot the distribution of feature norms over a small dataset of images in Fig.[3](https://arxiv.org/html/2309.16588v2#S2.F3 "Figure 3 ‣ Outliers appear during the training of large models. ‣ 2.1 Artifacts in the local features of DINOv2 ‣ 2 Problem Formulation ‣ Vision Transformers Need Registers") (right), which is clearly bimodal, allowing us to choose a simple criterion for the rest of this section: tokens with norm higher than 150 will be considered as “high-norm” tokens, and we will study their properties relative to regular tokens. This hand-picked cutoff value can vary across models. In the rest of this work, we use “high-norm” and “outlier” interchangeably.

##### Outliers appear during the training of large models.

We make several additional observations about the conditions in which these outlier patches appear during the training of DINOv2. This analysis is illustrated in Fig.[4](https://arxiv.org/html/2309.16588v2#S2.F4 "Figure 4 ‣ High-norm tokens appear where patch information is redundant. ‣ 2.1 Artifacts in the local features of DINOv2 ‣ 2 Problem Formulation ‣ Vision Transformers Need Registers"). First, these high-norm patches seem to differentiate themselves from other patches around layer 15 of this 40-layer ViT (Fig.[3(a)](https://arxiv.org/html/2309.16588v2#S2.F3.sf1 "3(a) ‣ Figure 4 ‣ High-norm tokens appear where patch information is redundant. ‣ 2.1 Artifacts in the local features of DINOv2 ‣ 2 Problem Formulation ‣ Vision Transformers Need Registers")). Second, when looking at the distribution of norms along training of DINOv2, we see that these outliers only appear after one third of training (Fig.[3(b)](https://arxiv.org/html/2309.16588v2#S2.F3.sf2 "3(b) ‣ Figure 4 ‣ High-norm tokens appear where patch information is redundant. ‣ 2.1 Artifacts in the local features of DINOv2 ‣ 2 Problem Formulation ‣ Vision Transformers Need Registers")). Finally, when analyzing more closely models of different size (Tiny, Small, Base, Large, Huge and giant), we see that only the three largest models exhibit outliers (Fig.[3(c)](https://arxiv.org/html/2309.16588v2#S2.F3.sf3 "3(c) ‣ Figure 4 ‣ High-norm tokens appear where patch information is redundant. ‣ 2.1 Artifacts in the local features of DINOv2 ‣ 2 Problem Formulation ‣ Vision Transformers Need Registers")).

![Image 1: Refer to caption](https://arxiv.org/html/2309.16588v2/x1.png)

![Image 2: Refer to caption](https://arxiv.org/html/2309.16588v2/x2.png)

![Image 3: Refer to caption](https://arxiv.org/html/2309.16588v2/x3.png)

![Image 4: Refer to caption](https://arxiv.org/html/2309.16588v2/x4.png)

Figure 3:  Comparison of local feature norms for DINO ViT-B/16 and DINOv2 ViT-g/14. We observe that DINOv2 has a few outlier patches, whereas DINO does not present these artifacts. For DINOv2, although most patch tokens have a norm between 0 and 100, a small proportion of tokens have a very high norm. We measure the proportion of tokens with norm larger than 150 at 2.37%. 

##### High-norm tokens appear where patch information is redundant.

To verify this, we measure the cosine similarity between high-norm tokens and their 4 neighbors right after the patch embedding layer (at the beginning of the vision transformer). We illustrate the density plot in Fig. [4(a)](https://arxiv.org/html/2309.16588v2#S2.F4.sf1 "4(a) ‣ Figure 5 ‣ High-norm tokens hold little local information. ‣ 2.1 Artifacts in the local features of DINOv2 ‣ 2 Problem Formulation ‣ Vision Transformers Need Registers"). We observe that high-norm tokens appear on patches that are very similar to their neighbors. This suggests that these patches contrain redundant information and that the model could discard their information without hurting the quality of the image representation. This matches qualitative observations (see Fig.[2](https://arxiv.org/html/2309.16588v2#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Vision Transformers Need Registers")) that they often appear in uniform, background areas.

![Image 5: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/230802_norm_histogram_by_layer.png)

(a) Norms along layers.

![Image 6: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/norm_distrib_vs_pretrain_iter.png)

(b) Norms along iterations.

![Image 7: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/norm_distrib_vs_model_size.png)

(c) Norms across model size.

Figure 4:  Illustration of several properties of outlier tokens in the 40-layer DINOv2 ViT-g model. (a): Distribution of output token norms along layers. (b): Distribution of norms along training iterations. (c): Distribution of norms for different model sizes. The outliers appear around the middle of the model during training; they appear with models larger than and including ViT-Large. 

##### High-norm tokens hold little local information.

In order to better understand the nature of these tokens, we propose to probe the patch embeddings for different types of information. For that we consider two different tasks: position prediction and pixel reconstruction. For each of these tasks, we train a linear model on top of the patch embeddings, and measure the performance of this model. We compare the performance achieved with high-norm tokens and with other tokens, to see if high-norm tokens contain different information than “normal” tokens.

*   •Position prediction. We train a linear model to predict the position of each patch token in the image, and measure its accuracy. We note that this position information was injected in the tokens before the first ViT layer in the form of absolute position embeddings. We observe that high-norm tokens have much lower accuracy than the other tokens (Fig.[4(b)](https://arxiv.org/html/2309.16588v2#S2.F4.sf2 "4(b) ‣ Figure 5 ‣ High-norm tokens hold little local information. ‣ 2.1 Artifacts in the local features of DINOv2 ‣ 2 Problem Formulation ‣ Vision Transformers Need Registers")), suggesting they contain less information about their position in the image. 
*   •Pixel reconstruction. We train a linear model to predict the pixel values of the image from the patch embeddings, and measure the accuracy of this model. We observe again that high-norm tokens achieve much lower accuracy than other tokens (Fig.[4(b)](https://arxiv.org/html/2309.16588v2#S2.F4.sf2 "4(b) ‣ Figure 5 ‣ High-norm tokens hold little local information. ‣ 2.1 Artifacts in the local features of DINOv2 ‣ 2 Problem Formulation ‣ Vision Transformers Need Registers")). This suggests that high-norm tokens contain less information to reconstruct the image than the others. 

![Image 8: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/230802_kde_cossim_neighbors_2.png)

(a) Cosine similarity to neighbors.

(b) Linear probing for local information.

Figure 5: (a): Distribution of cosine similarity between input patches and their 4 neighbors. We plot separately artifact patches (norm of the _output token_ over 150) and normal patches. (b): Local information probing on normal and outlier patch tokens. We train two models: one for predicting position, and one for reconstructing the input patch. Outlier tokens have much lower scores than the other tokens, suggesting they are storing less local patch information. 

##### Artifacts hold global information.

In order to evaluate how much global information is gathered in the high-norm tokens, we propose to evaluate them on standard image representation learning benchmarks. For each image in a classification dataset, we forward it through DINOv2-g and extract the patch embeddings. From those, we choose a single token at random, either high-norm or normal. This token is then considered as the image representation. We then train a logistic regression classifier to predict the image class from this representation, and measure the accuracy. We observe that the high-norm tokens have a much higher accuracy than the other tokens (Table[1](https://arxiv.org/html/2309.16588v2#S2.T1 "Table 1 ‣ Artifacts hold global information. ‣ 2.1 Artifacts in the local features of DINOv2 ‣ 2 Problem Formulation ‣ Vision Transformers Need Registers")). This suggests that outlier tokens contain more global information than other patch tokens.

Table 1:  Image classification via linear probing on normal and outlier patch tokens. We also report the accuracy of classifiers learnt on the class token. We see that outlier tokens have a much higher accuracy than regular ones, suggesting they are effectively storing global image information. 

### 2.2 Hypothesis and remediation

Having made these observations, we make the following hypothesis: _large_, _sufficiently trained_ models learn to recognize _redundant_ tokens, and to use them as places to _store_, _process_ and _retrieve_ global information. Furthermore, we posit that while this behavior is not bad in itself, the fact that it happens inside the patch tokens is undesirable. Indeed, it leads the model to discard local patch information (Tab.[4(b)](https://arxiv.org/html/2309.16588v2#S2.F4.sf2 "4(b) ‣ Figure 5 ‣ High-norm tokens hold little local information. ‣ 2.1 Artifacts in the local features of DINOv2 ‣ 2 Problem Formulation ‣ Vision Transformers Need Registers")), possibly incurring decreased performance on dense prediction tasks.

We therefore propose a simple fix to this issue: we explicitly add new tokens to the sequence, that the model can learn to use as registers. We add these tokens after the patch embedding layer, with a learnable value, similarly to the [CLS] token. At the end of the vision transformer, these tokens are discarded, and the [CLS] token and patch tokens are used as image representations, as usual. This mechanism was first proposed in Memory Transformers (Burtsev et al., [2020](https://arxiv.org/html/2309.16588v2#bib.bib3)), improving translation tasks in NLP. Interestingly, we show here that this mechanism admits a natural justification for vision transformers, fixing an interpretability and performance issue that was present otherwise.

We note that we have not been able to fully determine which aspects of the training led to the appearance of artifacts in different models. The pretraining paradigm seems to play a role, as OpenCLIP and DeiT-III exhibit outliers both at size B and L (Fig. [2](https://arxiv.org/html/2309.16588v2#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Vision Transformers Need Registers")). However, the model size and training length also play important parts, as observed in Fig. [4](https://arxiv.org/html/2309.16588v2#S2.F4 "Figure 4 ‣ High-norm tokens appear where patch information is redundant. ‣ 2.1 Artifacts in the local features of DINOv2 ‣ 2 Problem Formulation ‣ Vision Transformers Need Registers").

![Image 9: Refer to caption](https://arxiv.org/html/2309.16588v2/x5.png)

Figure 6:  Illustration of the proposed remediation and resulting model. We add N 𝑁 N italic_N additional learnable input tokens (depicted in yellow), that the model can use as _registers_. At the output of the model, only the patch tokens and [CLS] tokens are used, both during training and inference. 

3 Experiments
-------------

In this section, we validate the proposed solution by training vision transformers with additional [reg] register tokens. We evaluate the effectiveness of our approach by a quantitative and qualitative analysis. We then ablate the number of registers used for training, to check that they do not cause a performance regression, evaluate an unsupervised object discovery method atop our features and finally provide a qualitative analysis of the patterns learnt by the registers.

### 3.1 Training algorithms and data

As the proposed solution is a simple architectural change, we can easily apply it to any training procedure. We try it on three different state-of-the-art training methods for supervised, text-supervised, and unsupervised learning, shortly described below.

DEIT-III(Touvron et al., [2022](https://arxiv.org/html/2309.16588v2#bib.bib29)) is a simple and robust supervised training recipe for classification with ViTs on ImageNet-1k and ImageNet-22k. We choose this method as an example of label-supervised training as it is simple, uses the base ViT architecture, achieves strong classification results, and is easy to reproduce and modify with our improvements. We run this method on the ImageNet-22k dataset, using the ViT-B settings, as provided in the official repository 1 1 1[https://github.com/facebookresearch/deit](https://github.com/facebookresearch/deit).

OpenCLIP(Ilharco et al., [2021](https://arxiv.org/html/2309.16588v2#bib.bib13)) is a strong training method for producing text-image aligned models, following the original CLIP work. We chose this method as an example of text-supervised training because it is open-source, uses the base ViT architecture, and is easy to reproduce and modify with our improvements. We run the OpenCLIP method on a text-image-aligned corpus based on Shutterstock that includes only licensed image and text data. We use a ViT-B/16 image encoder, as proposed in the official repository 2 2 2[https://github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip).

DINOv2(Oquab et al., [2023](https://arxiv.org/html/2309.16588v2#bib.bib20)) is a self-supervised method for learning visual features, following the DINO work. We apply our changes to this method as it is the main focus of our study. We run this method on ImageNet-22k with the ViT-L configuration. We use the official repository 3 3 3[https://github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2).

### 3.2 Evaluation of the proposed solution

As shown in Fig.[1](https://arxiv.org/html/2309.16588v2#S0.F1 "Figure 1 ‣ Vision Transformers Need Registers"), we get rid of the artifacts by training models with additional register tokens. In the appendix, we provide additional qualitative results for more images in Fig.[19](https://arxiv.org/html/2309.16588v2#A8.F19 "Figure 19 ‣ Appendix H Qualitative Results ‣ Vision Transformers Need Registers"). In order to quantitatively measure this effect, for each model, we probe the norm of features at the output of the model. We report these norms for all three algorithms with and without registers in Fig.[7](https://arxiv.org/html/2309.16588v2#S3.F7 "Figure 7 ‣ 3.2 Evaluation of the proposed solution ‣ 3 Experiments ‣ Vision Transformers Need Registers"). We see that when training with registers, models do not exhibit large-norm tokens at the output, which confirms the initial qualitative assessment.

![Image 10: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/fig_norm_distrib_before_after_stripplot-2.png)

Figure 7:  Effect of register tokens on the distribution of output norms on DINOv2, OpenCLIP and DeiT-III. Using register tokens effectively removes the norm outliers that were present previously. 

Performance regression. In the previous section, we have shown that the proposed approach removes artifacts from local feature maps. In this experiment, we want to check that the use of register tokens does not affect the representation quality of those features. We run linear probing on ImageNet classification, ADE20k Segmentation, and NYUd monocular depth estimation. We follow the experimental protocol outlined in Oquab et al. ([2023](https://arxiv.org/html/2309.16588v2#bib.bib20)). We summarize the performance of the models described in Sec.[3.1](https://arxiv.org/html/2309.16588v2#S3.SS1 "3.1 Training algorithms and data ‣ 3 Experiments ‣ Vision Transformers Need Registers") with and without register tokens in Table[1(a)](https://arxiv.org/html/2309.16588v2#S3.T1.st1 "1(a) ‣ Table 2 ‣ 3.2 Evaluation of the proposed solution ‣ 3 Experiments ‣ Vision Transformers Need Registers"). We see that when using registers, models do not lose performance and sometimes even work better. For completeness, we also provided the zero-shot classification performance on ImageNet for OpenCLIP (Table[1(b)](https://arxiv.org/html/2309.16588v2#S3.T1.st2 "1(b) ‣ Table 2 ‣ 3.2 Evaluation of the proposed solution ‣ 3 Experiments ‣ Vision Transformers Need Registers")), which remains unchanged. Please note that the absolute performance of our OpenCLIP reproduction is lower due to the data source we used.

(a) Linear evaluation with frozen features.

(b) Zero-shot classification.

Table 2:  Evaluation of downstream performance of the models that we trained, with and without registers. We consider linear probing of frozen features for all three models, and zero-shot evaluation for the OpenCLIP model. We see that using register not only does not degrade performance, but even improves it by a slight margin in some cases. 

Number of register tokens. As described in Sec.[2.2](https://arxiv.org/html/2309.16588v2#S2.SS2 "2.2 Hypothesis and remediation ‣ 2 Problem Formulation ‣ Vision Transformers Need Registers"), we propose alleviating the feature maps’ artifacts by adding register tokens. In this experiment, we study the influence of the number of such tokens on local features and downstream performance. We train DINOv2 ViT-L/14 models with 0, 1, 2, 4, 8 or 16 registers. In Fig.[8](https://arxiv.org/html/2309.16588v2#S3.F8 "Figure 8 ‣ 3.2 Evaluation of the proposed solution ‣ 3 Experiments ‣ Vision Transformers Need Registers"), we report the results of this analysis. In Fig.[8](https://arxiv.org/html/2309.16588v2#S3.F8 "Figure 8 ‣ 3.2 Evaluation of the proposed solution ‣ 3 Experiments ‣ Vision Transformers Need Registers")(top), we qualitatively study the attention maps and observe that the visible artifacts disappear when adding at least one register. We then examine in Fig.[8](https://arxiv.org/html/2309.16588v2#S3.F8 "Figure 8 ‣ 3.2 Evaluation of the proposed solution ‣ 3 Experiments ‣ Vision Transformers Need Registers")(bottom) performance on downstream evaluation benchmarks, following the protocol from Oquab et al. ([2023](https://arxiv.org/html/2309.16588v2#bib.bib20)). There seems to be an optimal number of registers for dense tasks, and adding one brings most of the benefit. This optimum is likely explained by the disappearance of artifacts, leading to better local features. On ImageNet, however, performance improves when using more registers. In all our experiments, we kept 4 4 4 4 register tokens.

Input 0 [reg]1 [reg]2 [reg]4 [reg]8 [reg]16 [reg]
![Image 11: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/230916_attmap_vs_n_reg/pyrrhus_orig.png)![Image 12: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/230916_attmap_vs_n_reg/100cc_pyrrhus_0reg.png)![Image 13: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/230916_attmap_vs_n_reg/100cc_pyrrhus_1reg.png)![Image 14: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/230916_attmap_vs_n_reg/100cc_pyrrhus_2reg.png)![Image 15: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/230916_attmap_vs_n_reg/100cc_pyrrhus_4reg.png)![Image 16: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/230916_attmap_vs_n_reg/100cc_pyrrhus_8reg.png)![Image 17: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/230916_attmap_vs_n_reg/100cc_pyrrhus_16reg.png)

![Image 18: Refer to caption](https://arxiv.org/html/2309.16588v2/x6.png)

Figure 8:  Ablation of the the number of register tokens used with a DINOv2 model. (top): qualitative visualization of artifacts appearing as a function of number of registers. (bottom): performance on three tasks (ImageNet, ADE-20k and NYUd) as a function of number of registers used. While one register is sufficient to remove artefacts, using more leads to improved downstream performance. 

### 3.3 Object discovery

Recent unsupervised object discovery methods rely on the quality and smoothness of local feature maps(Siméoni et al., [2021](https://arxiv.org/html/2309.16588v2#bib.bib26); Wang et al., [2023](https://arxiv.org/html/2309.16588v2#bib.bib30)). By leveraging DINO Caron et al. ([2021](https://arxiv.org/html/2309.16588v2#bib.bib5)), these methods have significantly surpassed the previous state of the art. However, the algorithm leads to poor performance when applied to modern backbones such as DINOv2 Oquab et al. ([2023](https://arxiv.org/html/2309.16588v2#bib.bib20)) or supervised ones Touvron et al. ([2022](https://arxiv.org/html/2309.16588v2#bib.bib29)). We posit that this can be alleviated by the method proposed in this work. We run LOST(Siméoni et al., [2021](https://arxiv.org/html/2309.16588v2#bib.bib26)) on features extracted from backbones trained using the algorithms described in Sec.[3.1](https://arxiv.org/html/2309.16588v2#S3.SS1 "3.1 Training algorithms and data ‣ 3 Experiments ‣ Vision Transformers Need Registers") with and without registers. We run object discovery on PASCAL VOC 2007 and 2012 and COCO 20k. We use values for DeiT and OpenCLIP, and for DINOv2, we use keys. Because the output features may have different conditioning, we manually add a bias to the gram matrix of features. The results of this experiment are presented in Table[3](https://arxiv.org/html/2309.16588v2#S3.T3 "Table 3 ‣ 3.3 Object discovery ‣ 3 Experiments ‣ Vision Transformers Need Registers"). For DINOv2 and DeiT-III, adding registers significantly improves the discovery performance. For OpenCLIP, the performance is slighty worse with registers (see Sec.[C](https://arxiv.org/html/2309.16588v2#A3 "Appendix C Analysis of LOST performance ‣ Vision Transformers Need Registers") for analysis). The performance of DINOv2 on VOC2007 still does not match that of DINO as reported by Siméoni et al. ([2021](https://arxiv.org/html/2309.16588v2#bib.bib26)) (61.9 61.9 61.9 61.9 corloc). However, the model with registers gets an improvement of 20.1 20.1 20.1 20.1 corloc (55.4 55.4 55.4 55.4 versus 35.3 35.3 35.3 35.3).

Table 3:  Unsupervised Object Discovery using LOST(Siméoni et al., [2021](https://arxiv.org/html/2309.16588v2#bib.bib26)) on models with and without registers. We evaluated three types of models trained with various amounts of supervision on VOC 2007, 2012 and COCO. We measure performance using corloc. We observe that adding register tokens makes all models significantly more viable for usage in object discovery. 

### 3.4 Qualitative evaluation of registers

In this final experiment, we qualitatively probe for the behavior of register tokens. We want to verify if they all exhibit similar attention patterns or whether a differentiation automatically emerges. To this end, we plot the attention maps of the class and register tokens to patch tokens. The result of this visualization is shown in Fig.[9](https://arxiv.org/html/2309.16588v2#S3.F9 "Figure 9 ‣ 3.4 Qualitative evaluation of registers ‣ 3 Experiments ‣ Vision Transformers Need Registers"). We see that registers do not have a completely aligned behavior. Some selected registers exhibit interesting attention patterns, attending to the different objects in the scene. While nothing enforced this behavior, their activations had some natural diversity. We leave the study of the regularization of registers for future work.

Figure 9:  Comparison of the attention maps of the [CLS] and register tokens. Register tokens sometimes attend to different parts of the feature map, similarly to slot attention (Locatello et al., [2020](https://arxiv.org/html/2309.16588v2#bib.bib18)). This behaviour was never required from the model, and emerged naturally from training. 

4 Related Work
--------------

Feature extraction with pretrained models. Using pretrained neural network models for extracting visual features has stood the test of time since the AlexNet (Krizhevsky et al., [2012](https://arxiv.org/html/2309.16588v2#bib.bib17)) CNN model pretrained on ImageNet-1k(Russakovsky et al., [2015](https://arxiv.org/html/2309.16588v2#bib.bib23)). More recent models have upgraded the same setup with modern architectures, such as ResNets (used in, _e.g._, DETR,Carion et al., [2020](https://arxiv.org/html/2309.16588v2#bib.bib4)) or even Vision Transformers. As Transformers are easily able to handle different modalities during training, off-the-shelf backbones are now commonly trained on label supervision (_e.g._, DeiT-III on ImageNet-22k,Touvron et al., [2022](https://arxiv.org/html/2309.16588v2#bib.bib29)) or text supervision (e.g., CLIP(Radford et al., [2021](https://arxiv.org/html/2309.16588v2#bib.bib22))), providing strong visual foundation models, scaling well with model sizes, and enabling excellent performance on a variety of tasks including detection (Carion et al., [2020](https://arxiv.org/html/2309.16588v2#bib.bib4)) and segmentation (Zheng et al., [2021](https://arxiv.org/html/2309.16588v2#bib.bib33); Kirillov et al., [2023](https://arxiv.org/html/2309.16588v2#bib.bib16)). In this context, supervision relies on annotations in the form of labels or text alignment; the dataset biases (Torralba & Efros, [2011](https://arxiv.org/html/2309.16588v2#bib.bib28)) are not well characterized, yet they drive learning and shape the learned models. An alternative approach consists of not using supervision and letting the models learn from the data via a pretext task that is designed to require understanding the content of images(Doersch et al., [2015](https://arxiv.org/html/2309.16588v2#bib.bib8)). This self-supervised learning paradigm was explored in multiple methods using Vision Transformers: MAE (He et al., [2022](https://arxiv.org/html/2309.16588v2#bib.bib12)) trains a model at reconstructing pixel values of hidden areas of an image and then applies fine-tuning to address a new task. With a different approach, the self-distillation family of methods (He et al., [2020](https://arxiv.org/html/2309.16588v2#bib.bib11); Caron et al., [2021](https://arxiv.org/html/2309.16588v2#bib.bib5); Zhou et al., [2022](https://arxiv.org/html/2309.16588v2#bib.bib34)) showcase strong performance using frozen backbones, allowing for more robustness to domain shifts for task-specific downstream models. In this work, we focused the analysis on self-supervised learning, and more specifically on the DINOv2 approach (Oquab et al., [2023](https://arxiv.org/html/2309.16588v2#bib.bib20)), which has shown to be particularly effective for learning local features. We showed that despite excellent benchmark scores, DINOv2 features exhibit undesirable artifacts and that correcting these artifacts in the learning process allows for further improvements in the benchmark performances. These phenomenon is even more surprising as DINOv2 builds upon DINO(Caron et al., [2021](https://arxiv.org/html/2309.16588v2#bib.bib5)), which does not show signs of artifacts. We then further showed that the correction techniques hold for supervised paradigms by testing on DeiT-III and OpenCLIP.

Additional tokens in transformers. Extending the transformer sequence with special tokens was popularized in BERT(Devlin et al., [2019](https://arxiv.org/html/2309.16588v2#bib.bib7)). However, most approaches add new tokens either to provide the network with new information as for example [SEP] tokens in BERT, provide opportunity to spend more computation on the input as seen with the tape tokens in AdaTape(Xue et al., [2023](https://arxiv.org/html/2309.16588v2#bib.bib31)), or to gather information in these tokens, and use their output value as an output of the model: for classification, as [CLS] tokens in BERT and ViT (Dosovitskiy et al., [2021](https://arxiv.org/html/2309.16588v2#bib.bib9)); for generative learning, as [MASK] in BERT and BEiT(Bao et al., [2021](https://arxiv.org/html/2309.16588v2#bib.bib1)); for detection, as object queries in DETR(Carion et al., [2020](https://arxiv.org/html/2309.16588v2#bib.bib4)), detection tokens in YOLOS(Fang et al., [2021](https://arxiv.org/html/2309.16588v2#bib.bib10)), and ViDT(Song et al., [2021](https://arxiv.org/html/2309.16588v2#bib.bib27)); or for accumulating information from possibly multiple modalities before decoding, as latent token arrays in Perceivers (Jaegle et al., [2021](https://arxiv.org/html/2309.16588v2#bib.bib14); [2022](https://arxiv.org/html/2309.16588v2#bib.bib15)). Different to these works, the tokens we add to the sequence add no information, and their output value is not used for any purpose. They are simply registers where the model can learn to store and retrieve information during the forward pass. The Memory Transformer(Burtsev et al., [2020](https://arxiv.org/html/2309.16588v2#bib.bib3)), closer to our work, presents a simple approach to improve transformer models using memory tokens added to the token sequence, improving translation performance. In follow-up work, Bulatov et al. ([2022](https://arxiv.org/html/2309.16588v2#bib.bib2)) address complex copy-repeat-reverse tasks. Sandler et al. ([2022](https://arxiv.org/html/2309.16588v2#bib.bib24)) extend this line to the vision domain for fine-tuning but observe that such tokens do not transfer well across tasks. In contrast, we do not perform fine-tuning and employ additional tokens during pretraining to improve the features obtained for all tasks downstream. More importantly, our study contributes the following new insight in Sec.[2](https://arxiv.org/html/2309.16588v2#S2 "2 Problem Formulation ‣ Vision Transformers Need Registers"): the mechanism implemented through memory tokens already appears naturally in Vision Transformers; our study shows that such tokens allow us not to create but to isolate this existing behavior, and thus avoid collateral side-effects.

Attention maps of vision transformers. Visualising the attention map from [CLS] token to patch tokens was popularized in DINO(Caron et al., [2021](https://arxiv.org/html/2309.16588v2#bib.bib5)). It was shown there that the attention maps of DINO were clean of artifacts, as opposed to the attention maps of previous vision transformers. Other works have since reported interesting attention maps using various techniques: by modifying the optimisation procedure(Chen et al., [2022](https://arxiv.org/html/2309.16588v2#bib.bib6)), by steering the attention scores towards useful image parts(Shi et al., [2023](https://arxiv.org/html/2309.16588v2#bib.bib25)), by modifying the architecture of the transformer layers(Yu et al., [2024](https://arxiv.org/html/2309.16588v2#bib.bib32)), or by introducing a learnable pooling to produce the [CLS] token(Psomas et al., [2023](https://arxiv.org/html/2309.16588v2#bib.bib21)).

5 Conclusion
------------

In this work, we exposed artifacts in the feature maps of DINOv2 models, and found this phenomenon to be present in multiple existing popular models. We have described a simple method to detect these artifacts by observing that they correspond to tokens with an outlier norm value at the output of the Transformer model. Studying their location, we have proposed an interpretation that models naturally recycle tokens from low-informative areas and repurpose them into a different role for inference. Following this interpretation, we have proposed a simple fix, consisting of appending additional tokens to the input sequence that are not used as outputs, and have found that this entirely removes the artifacts, improving the performance in dense prediction and object discovery. Moreover, we have shown that the proposed solution also removes the same artifacts present in supervised models such as DeiT-III and OpenCLIP, confirming the generality of our solution.

#### Acknowledgments

We thank Hu Xu, Oriane Siméoni, Mido Assran and Armand Joulin for their insightful discussions and help during the course of this work. We thank Pyrrhus for posing for fig[8](https://arxiv.org/html/2309.16588v2#S3.F8 "Figure 8 ‣ 3.2 Evaluation of the proposed solution ‣ 3 Experiments ‣ Vision Transformers Need Registers"). Julien Mairal was supported by ANR 3IA MIAI@Grenoble Alpes (ANR-19-P3IA-0003) and by ERC grant number 101087696 (APHELEIA project).

References
----------

*   Bao et al. (2021) Hangbo Bao, Li Dong, and Furu Wei. Beit: Bert pre-training of image transformers. In _ICLR_, 2021. 
*   Bulatov et al. (2022) Aydar Bulatov, Yury Kuratov, and Mikhail Burtsev. Recurrent memory transformer. In _NeurIPS_, 2022. 
*   Burtsev et al. (2020) Mikhail S Burtsev, Yuri Kuratov, Anton Peganov, and Grigory V Sapunov. Memory transformer. _arXiv preprint arXiv:2006.11527_, 2020. 
*   Carion et al. (2020) Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In _ECCV_, 2020. 
*   Caron et al. (2021) Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In _ICCV_, 2021. 
*   Chen et al. (2022) Xiangning Chen, Cho-Jui Hsieh, and Boqing Gong. When vision transformers outperform resnets without pre-training or strong data augmentations. In _ICLR_, 2022. 
*   Devlin et al. (2019) Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. _NAACL_, 2019. 
*   Doersch et al. (2015) Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In _ICCV_, 2015. 
*   Dosovitskiy et al. (2021) Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In _ICLR_, 2021. 
*   Fang et al. (2021) Yuxin Fang, Bencheng Liao, Xinggang Wang, Jiemin Fang, Jiyang Qi, Rui Wu, Jianwei Niu, and Wenyu Liu. You only look at one sequence: Rethinking transformer in vision through object detection. In _NeurIPS_, 2021. 
*   He et al. (2020) Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In _CVPR_, 2020. 
*   He et al. (2022) Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In _CVPR_, 2022. 
*   Ilharco et al. (2021) Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip. 2021. 
*   Jaegle et al. (2021) Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In _ICML_, 2021. 
*   Jaegle et al. (2022) Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Andrew Brock, Evan Shelhamer, Olivier J. H’enaff, Matthew M. Botvinick, Andrew Zisserman, Oriol Vinyals, and João Carreira. Perceiver io: A general architecture for structured inputs & outputs. In _ICLR_, 2022. 
*   Kirillov et al. (2023) Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. _arXiv preprint arXiv:2304.02643_, 2023. 
*   Krizhevsky et al. (2012) Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In _NeurIPS_, 2012. 
*   Locatello et al. (2020) Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. Object-centric learning with slot attention. In _NeurIPS_, 2020. 
*   Lowe (2004) David G Lowe. Distinctive image features from scale-invariant keypoints. _IJCV_, 2004. 
*   Oquab et al. (2023) Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. _arXiv preprint arXiv:2304.07193_, 2023. 
*   Psomas et al. (2023) Bill Psomas, Ioannis Kakogeorgiou, Konstantinos Karantzalos, and Yannis Avrithis. Keep it simpool: Who said supervised transformers suffer from attention deficit? In _ICCV_, 2023. 
*   Radford et al. (2021) Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _ICML_, 2021. 
*   Russakovsky et al. (2015) Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. _IJCV_, 2015. 
*   Sandler et al. (2022) Mark Sandler, Andrey Zhmoginov, Max Vladymyrov, and Andrew Jackson. Fine-tuning image transformers using learnable memory. In _CVPR_, 2022. 
*   Shi et al. (2023) Baifeng Shi, Siyu Gai, Trevor Darrell, and Xin Wang. Toast: Transfer learning via attention steering, 2023. 
*   Siméoni et al. (2021) Oriane Siméoni, Gilles Puy, Huy V Vo, Simon Roburin, Spyros Gidaris, Andrei Bursuc, Patrick Pérez, Renaud Marlet, and Jean Ponce. Localizing objects with self-supervised transformers and no labels. In _BMVC_, 2021. 
*   Song et al. (2021) Hwanjun Song, Deqing Sun, Sanghyuk Chun, Varun Jampani, Dongyoon Han, Byeongho Heo, Wonjae Kim, and Ming-Hsuan Yang. Vidt: An efficient and effective fully transformer-based object detector. In _ICLR_, 2021. 
*   Torralba & Efros (2011) Antonio Torralba and Alexei A. Efros. Unbiased look at dataset bias. In _CVPR_, 2011. 
*   Touvron et al. (2022) Hugo Touvron, Matthieu Cord, and Hervé Jégou. Deit iii: Revenge of the vit. In _ECCV_, 2022. 
*   Wang et al. (2023) Xudong Wang, Rohit Girdhar, Stella X Yu, and Ishan Misra. Cut and learn for unsupervised object detection and instance segmentation. In _CVPR_, 2023. 
*   Xue et al. (2023) Fuzhao Xue, Valerii Likhosherstov, Anurag Arnab, Neil Houlsby, Mostafa Dehghani, and Yang You. Adaptive computation with elastic input sequence. In _ICML_, 2023. 
*   Yu et al. (2024) Yaodong Yu, Tianzhe Chu, Shengbang Tong, Ziyang Wu, Druv Pai, Sam Buchanan, and Yi Ma. Emergence of segmentation with minimalistic white-box transformers. In _CPAL_, 2024. 
*   Zheng et al. (2021) Sixiao Zheng, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu, Jianfeng Feng, Tao Xiang, Philip HS Torr, et al. Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers. In _CVPR_, 2021. 
*   Zhou et al. (2022) Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. ibot: Image bert pre-training with online tokenizer. In _ICLR_, 2022. 

Appendix A Interpolation artifacts and outlier position distribution
--------------------------------------------------------------------

![Image 19: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/230918_interp_artifacts/aliased.png)

![Image 20: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/230918_interp_artifacts/antialiased.png)

Figure 10:  Feature norms along locations: proportion of tokens with norm larger than the cutoff value at a given location. Left: official DINOv2 model (no antialiasing), right: our models (with antialiasing). At some positions, more than 20% of tokens have a high norm. 

![Image 21: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/230918_interp_artifacts/interp_artifacts.png)

Figure 11:  Propagating unit gradients through a bicubic interpolation (16×16→7×7→16 16 7 7 16\times 16\rightarrow 7\times 7 16 × 16 → 7 × 7) without antialiasing. We observe a striping pattern similar to the one of Fig. [10](https://arxiv.org/html/2309.16588v2#A1.F10 "Figure 10 ‣ Appendix A Interpolation artifacts and outlier position distribution ‣ Vision Transformers Need Registers") (left). 

We plot in Figure [10](https://arxiv.org/html/2309.16588v2#A1.F10 "Figure 10 ‣ Appendix A Interpolation artifacts and outlier position distribution ‣ Vision Transformers Need Registers") (left) the proportion of outlier tokens, characterized by a norm larger than the cutoff value defined manually, following the distribution of norms shown in Fig. [3](https://arxiv.org/html/2309.16588v2#S2.F3 "Figure 3 ‣ Outliers appear during the training of large models. ‣ 2.1 Artifacts in the local features of DINOv2 ‣ 2 Problem Formulation ‣ Vision Transformers Need Registers") (main text). We make two observations:

First, the distribution has a vertical-striped pattern. We investigate this phenomenon and notice that in the original DINOv2 implementation, during training the position embeddings are interpolated from a 16×16 16 16 16\times 16 16 × 16 map into a 7×7 7 7 7\times 7 7 × 7 map, without antialiasing. Propagating unit gradients through such an interpolation function (bicubic resize) leads to the following gradients, shown in Fig. [11](https://arxiv.org/html/2309.16588v2#A1.F11 "Figure 11 ‣ Appendix A Interpolation artifacts and outlier position distribution ‣ Vision Transformers Need Registers"). In this work, when producing results with DINOv2 (especially for the results in Tables [1(a)](https://arxiv.org/html/2309.16588v2#S3.T1.st1 "1(a) ‣ Table 2 ‣ 3.2 Evaluation of the proposed solution ‣ 3 Experiments ‣ Vision Transformers Need Registers"),[3](https://arxiv.org/html/2309.16588v2#S3.T3 "Table 3 ‣ 3.3 Object discovery ‣ 3 Experiments ‣ Vision Transformers Need Registers")), we always apply antialiasing in the interpolation operator, removing the striping pattern, which gives an updated distribution of outlier positions as shown in Fig. [10](https://arxiv.org/html/2309.16588v2#A1.F10 "Figure 10 ‣ Appendix A Interpolation artifacts and outlier position distribution ‣ Vision Transformers Need Registers") (right).

Second, the outliers tend to appear in areas closer to the border of the feature map rather than in the center. Our interpretation is that the base model tends to recycle tokens in low-informative areas to use as registers: pictures produced by people tend to be object-centric, and in this case the border areas often correspond to background, which contains less information than the center.

Appendix B Complexity analysis
------------------------------

![Image 22: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/23098_param_flop_vs_n_reg.png)

Figure 12:  Increase in model parameter and FLOP count when adding different numbers of registers. Adding registers can increase model FLOP count by up to 6% for 16 registers. However, in the more common case of using 4 registers, that we use in most of our experiments, this increase is below 2%. In all cases, the increase in model parameters is negligible. 

Since our proposed fix introduces new tokens, it also increases the number of learnable parameters and the FLOP count of the model. We show in Fig. [12](https://arxiv.org/html/2309.16588v2#A2.F12 "Figure 12 ‣ Appendix B Complexity analysis ‣ Vision Transformers Need Registers") the relationship between number of registers and increase in model FLOP count and parameter count. We observe that adding registers induces a negligible change in number of parameters, and a slight change in FLOP count. Still, for n=4 𝑛 4 n=4 italic_n = 4 registers, the increase in FLOPs stays below 2%.

Appendix C Analysis of LOST performance
---------------------------------------

The results presented in Sec.[3.3](https://arxiv.org/html/2309.16588v2#S3.SS3 "3.3 Object discovery ‣ 3 Experiments ‣ Vision Transformers Need Registers") show that adding registers allows us to obtain better object discovery performance with DINOv2 models. The conclusions for the two other models studied in this work could be more crisp. In order to understand why this is so, we qualitatively study the impact of removing artifacts on the intermediate computations in the LOST algorithm. We show the intermediate outputs of LOST for all models on a given input image in Fig.[13](https://arxiv.org/html/2309.16588v2#A3.F13 "Figure 13 ‣ Appendix C Analysis of LOST performance ‣ Vision Transformers Need Registers").

Adding registers improves the scores and the resulting seed expansion for DeiT-III and DINOv2. This observation is coherent with the improved numbers reported in Table[3](https://arxiv.org/html/2309.16588v2#S3.T3 "Table 3 ‣ 3.3 Object discovery ‣ 3 Experiments ‣ Vision Transformers Need Registers"). For OpenCLIP, however, the LOST algorithm seems robust to the type of outliers observed in the local features. Adding registers does remove artifacts (as clearly shown in Fig.[20](https://arxiv.org/html/2309.16588v2#A8.F20 "Figure 20 ‣ Appendix H Qualitative Results ‣ Vision Transformers Need Registers")) but does not have much impact on the LOST score. It is also worth noting that OpenCLIP, with or without registers, provides comparable performance to DINOv2 without registers and DeiT-III with registers. The qualitative assessment is coherent with the numbers reported in Table[3](https://arxiv.org/html/2309.16588v2#S3.T3 "Table 3 ‣ 3.3 Object discovery ‣ 3 Experiments ‣ Vision Transformers Need Registers").

![Image 23: Refer to caption](https://arxiv.org/html/2309.16588v2/x7.png)

Figure 13:  Illustration of the intermediate computations in the LOST algorithm for all models. Adding registers drastically improves the look of all intermediate steps for DeiT-III and DINOv2. The difference is less striking for the OpenCLIP model. 

A surprising observation is that despite the existence of high-norm patches in the output of OpenCLIP models without registers (as seen in Fig.[7](https://arxiv.org/html/2309.16588v2#S3.F7 "Figure 7 ‣ 3.2 Evaluation of the proposed solution ‣ 3 Experiments ‣ Vision Transformers Need Registers")), the seed expansion score in Fig.[13](https://arxiv.org/html/2309.16588v2#A3.F13 "Figure 13 ‣ Appendix C Analysis of LOST performance ‣ Vision Transformers Need Registers") looks smooth. In the LOST experiment with OpenCLIP models, we do not use the features directly, but the values from the computation of attention maps. In Fig.[14](https://arxiv.org/html/2309.16588v2#A3.F14 "Figure 14 ‣ Appendix C Analysis of LOST performance ‣ Vision Transformers Need Registers"), we show the seed expansion score for OpenCLIP models with and without registers for keys, queries and values. We see that artifacts are clearly visible as spots in the background for keys and queries, for the model without registers. As soon as registers are used, the LOST score is focusing on the object, with a smoother score for values. We qualitatively observe that for the OpenCLIP model, the value projection filters out the outliers even without registers. This means that the outliers appear to live in the null space of the value projection layer; the investigation for this phenomenon is left for future work.

![Image 24: Refer to caption](https://arxiv.org/html/2309.16588v2/x8.png)

Figure 14:  Illustration of the seed expansion score in LOST for an OpenCLIP model with and without registers for the three types of features considered: keys, queries, and values. The score is qualitatively improved across all features, with fewer artifacts appearing. Interestingly, the seed expansion map computed using values does not exhibit artifacts with nor without registers. 

Appendix D Behavior of models trained with registers
----------------------------------------------------

In order to better understand the phenomenon at hand, we examine the question of to what extent did the register tokens ”replace” the high-norm tokens and took on the same role.

### D.1 Norms

![Image 25: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/231120_fig_norm_distrib_vs_token_type_dinov2_0reg.png)

(a) DINOv2 - no register

![Image 26: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/231120_fig_norm_distrib_vs_token_type_dinov2_4reg.png)

(b) DINOv2 - 4 registers

Figure 15:  Distribution of token norms for a DINOv2 model without (left) and with (right) 4 registers. Introducing registers entirely negates the high-norm outliers among the patch tokens. 

In Fig. [15](https://arxiv.org/html/2309.16588v2#A4.F15 "Figure 15 ‣ D.1 Norms ‣ Appendix D Behavior of models trained with registers ‣ Vision Transformers Need Registers") we compare the distribution of token norms for a model with or without registers. This figure is similar to Fig. [7](https://arxiv.org/html/2309.16588v2#S3.F7 "Figure 7 ‣ 3.2 Evaluation of the proposed solution ‣ 3 Experiments ‣ Vision Transformers Need Registers") but with a finer granularity, as we also plot the norm distribution of individual register tokens and [CLS] tokens. We observe the following: with registers, the norms of patch tokens do not contain outliers anymore, and the high-norm tokens are entirely contained in the set of registers. As a result, we conclude that the behavior leading to high-norm outliers in the model is effectively absorbed in the registers.

An additional interesting observation is that the norms of the registers appear to be quantized, compared to the previous outliers; we leave the investigation of this phenomenon for future work.

### D.2 Information held by tokens

We report on table [4](https://arxiv.org/html/2309.16588v2#A4.T4 "Table 4 ‣ D.2 Information held by tokens ‣ Appendix D Behavior of models trained with registers ‣ Vision Transformers Need Registers") the linear probing performance of models trained with and without registers, when using different tokens as representations. We evaluate on the aircrafts dataset, as it showed clear conclusions in the similar table [1](https://arxiv.org/html/2309.16588v2#S2.T1 "Table 1 ‣ Artifacts hold global information. ‣ 2.1 Artifacts in the local features of DINOv2 ‣ 2 Problem Formulation ‣ Vision Transformers Need Registers"). We observe that adding a register does not significantly modify the scores obtained with the [CLS] or patch tokens. However, the outlier patches are removed, and their behavior is transferred to the newly added register.

Table 4: Linear probing of models with and without registers on the Aircraft dataset, using various tokens as representation. We observe that the behavior of the outlier tokens, aggregating global information, is absorbed into the register.

We further conduct an evaluation of the local information contained in the patch tokens of a model trained with and without registers (table [5](https://arxiv.org/html/2309.16588v2#A4.T5 "Table 5 ‣ D.2 Information held by tokens ‣ Appendix D Behavior of models trained with registers ‣ Vision Transformers Need Registers")). We observe that the non-outliers patches, in both cases, hold similar local information, confirming that the registers only remove the outlier behavior, without significantly modifying the information held by the other patches.

Table 5: Linear probing for local information on the patch tokens of models trained without or with registers. We only consider patches considered ”normal”, i.e. not the high-norm outliers. We observe that adding registers does not significantly modify the scores of these patches.

### D.3 Positional focus

![Image 27: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/231121_average_attn_location_cls.png)

(a) [CLS]

![Image 28: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/231121_average_attn_location_reg0.png)

(b) reg 0 0{}_{0}start_FLOATSUBSCRIPT 0 end_FLOATSUBSCRIPT

![Image 29: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/231121_average_attn_location_reg1.png)

(c) reg 1 1{}_{1}start_FLOATSUBSCRIPT 1 end_FLOATSUBSCRIPT

![Image 30: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/231121_average_attn_location_reg2.png)

(d) reg 2 2{}_{2}start_FLOATSUBSCRIPT 2 end_FLOATSUBSCRIPT

![Image 31: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/231121_average_attn_location_reg3.png)

(e) reg 3 3{}_{3}start_FLOATSUBSCRIPT 3 end_FLOATSUBSCRIPT

![Image 32: Refer to caption](https://arxiv.org/html/2309.16588v2/extracted/5532707/resources/231122_average_attmap_patch_1.png)

(f) patch

Figure 16:  Average attention map of registers and [CLS] token. There is a variability observed, with register 3 of this model focusing more on border areas. We also include the average attention map of a patch for comparison. The patch has a much more focused average attention. 

In Fig. [16](https://arxiv.org/html/2309.16588v2#A4.F16 "Figure 16 ‣ D.3 Positional focus ‣ Appendix D Behavior of models trained with registers ‣ Vision Transformers Need Registers") we display the positional focus for the class token and the 4 registers of a DINOv2+reg model. We produce these plots by running the model on a random subset of ImageNet-22k, and averaging the attention maps for the corresponding tokens at the last layer. We note that ImageNet-22k contains mostly object-centric images rather than scenes, which explains why the average attention maps correspond to centered blobs.

We make several observations. First, the attention maps for registers can be different of each other; for example, register 3 tends to focus on border areas, while the other registers tend to focus on more centered areas. Register 2 tends to focus slightly more on the upper areas of images that others. This is consistent with Fig. [9](https://arxiv.org/html/2309.16588v2#S3.F9 "Figure 9 ‣ 3.4 Qualitative evaluation of registers ‣ 3 Experiments ‣ Vision Transformers Need Registers"), where we show registers focusing on different large areas of the image, suggesting some level of specialization.

Second, by comparing the register maps to the [CLS] token map and to a patch token map, we observe that registers produce maps with a large support area, very similarly to the [CLS] token, and very different of a typical patch token which is more localized. As the [CLS] token is known to carry global information (as proven by the linear probing classification performance): this suggests that registers also carry global information.

Appendix E Masked autoencoders
------------------------------

Masked Autoencoding (He et al., [2022](https://arxiv.org/html/2309.16588v2#bib.bib12)) is another common way of pretraining self-supervised models. We observe in Fig.[17](https://arxiv.org/html/2309.16588v2#A5.F17 "Figure 17 ‣ Appendix E Masked autoencoders ‣ Vision Transformers Need Registers") that there are no artifacts in the maps produced by MAE: our hypothesis is that the absence of artifacts is due to the training procedure using only a local loss on the patch tokens, rather than an objective involving global aggregation of information. However, we also note that the performance of MAE models is very low for self-supervised representation learning (75% linear probing performance on ImageNet classification for ViT-Large), preventing it from being used as is, and making fine-tuning a requirement.

![Image 33: Refer to caption](https://arxiv.org/html/2309.16588v2/x9.png)

Figure 17: First three principal components of the output feature map of a ViT-Large Masked Autoencoder.

Appendix F Behavior per attention head
--------------------------------------

In this section, we investigate whether the artifacts appear only on the attention maps for specific heads of the last vision transformer block, or for all of them. We show in Fig.[18](https://arxiv.org/html/2309.16588v2#A6.F18 "Figure 18 ‣ Appendix F Behavior per attention head ‣ Vision Transformers Need Registers") the input image along with the attention maps for different heads. We observe that the artifacts appear for all attention heads, despite heads focusing on different areas of the object. We still observe that some heads focus more on artifacts than others.

![Image 34: Refer to caption](https://arxiv.org/html/2309.16588v2/x10.png)

Figure 18: Attention maps of the [CLS] token to the patch tokens, shown here separately per attention head. We produce these maps with a DINOv2-L model trained without registers.

Appendix G Variance on token information probing
------------------------------------------------

The results presented in table [1](https://arxiv.org/html/2309.16588v2#S2.T1 "Table 1 ‣ Artifacts hold global information. ‣ 2.1 Artifacts in the local features of DINOv2 ‣ 2 Problem Formulation ‣ Vision Transformers Need Registers") are obtained by taking a random patch token, either normal or outlier. However, the choice of this token adds a significant source of variance in the evaluation. For thoroughness, we report in table [6](https://arxiv.org/html/2309.16588v2#A7.T6 "Table 6 ‣ Appendix G Variance on token information probing ‣ Vision Transformers Need Registers") the standard deviation of the scores obtained relative to this choice.

dataset Airc.CF10 CF100 CUB Cal101 Cars DTD
token
normal 17.1±0.5 97.1±0.1 81.3±0.3 18.6±0.6 73.2±1.3 10.8±0.3 63.1±0.8
outlier 79.1±0.5 99.3±0.0 93.7±0.3 84.9±2.1 97.6±0.7 85.2±0.9 84.9±0.9
[CLS]87.3 99.4 94.5 91.3 96.9 91.5 85.2
dataset Flow.Food IN1k P205 Pets SUN VOC
token
normal 59.5±1.2 74.2±0.3 65.8±0.1 53.1±0.3 47.8±0.5 37.7±0.3 70.8±0.5
outlier 99.6±0.0 93.5±0.2 69.0±0.7 55.1±1.0 94.1±0.2 78.5±0.2 89.7±0.1
[CLS]99.7 94.7 86.0 66.4 96.9 78.6 89.1

Table 6: Image classification via linear probing on normal and outlier patch tokens. As we select the patch tokens randomly among the set of eligible tokens, this adds a source of variability. We report the standard deviation of this variability in grey along with the scores. This table is a detailed view of table [1](https://arxiv.org/html/2309.16588v2#S2.T1 "Table 1 ‣ Artifacts hold global information. ‣ 2.1 Artifacts in the local features of DINOv2 ‣ 2 Problem Formulation ‣ Vision Transformers Need Registers").

Appendix H Qualitative Results
------------------------------

We trained three popular models: DeiT-III, OpenCLIP, DINOv2 with and without the introduction of register tokens. We observe in Fig. [19](https://arxiv.org/html/2309.16588v2#A8.F19 "Figure 19 ‣ Appendix H Qualitative Results ‣ Vision Transformers Need Registers") the attention maps in the last layer of the Vision Transformer, for all three cases. We see that our approach provides much cleaner attention maps, with considerably fewer artifacts, explaining the improvement on the downstream object discovery task mentioned in Sec. [3.3](https://arxiv.org/html/2309.16588v2#S3.SS3 "3.3 Object discovery ‣ 3 Experiments ‣ Vision Transformers Need Registers"). The feature maps are also visibly improved, as shown in Fig. [20](https://arxiv.org/html/2309.16588v2#A8.F20 "Figure 20 ‣ Appendix H Qualitative Results ‣ Vision Transformers Need Registers"). Finally, we also show the norm of the patch tokens in Fig. [21](https://arxiv.org/html/2309.16588v2#A8.F21 "Figure 21 ‣ Appendix H Qualitative Results ‣ Vision Transformers Need Registers"), and confirm that in all three models, artifact patches correspond to norm outliers.

Figure 19: Attention maps of models trained without and with registers on various images.

Figure 20:  First principal component of the feature maps output by models trained without and with registers on various images. The components are whitened and the colormap covers the range [−3⁢σ,+3⁢σ]3 𝜎 3 𝜎[-3\sigma,+3\sigma][ - 3 italic_σ , + 3 italic_σ ]. 

Figure 21:  Maps of token norms for models trained without and with registers on various images. The norm outliers are very visible for models trained without registers.

