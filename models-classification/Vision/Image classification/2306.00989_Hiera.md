# Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles

Chaitanya Ryali<sup>\*1</sup> Yuan-Ting Hu<sup>\*1</sup> Daniel Bolya<sup>\*1,2</sup> Chen Wei<sup>1,3</sup> Haoqi Fan<sup>1</sup>  
 Po-Yao Huang<sup>1</sup> Vaibhav Aggarwal<sup>1</sup> Arkabandhu Chowdhury<sup>1</sup> Omid Poursaeed<sup>1</sup>  
 Judy Hoffman<sup>2</sup> Jitendra Malik<sup>1</sup> Yanghao Li<sup>\*1</sup> Christoph Feichtenhofer<sup>\*1</sup>

## Abstract

Modern hierarchical vision transformers have added several vision-specific components in the pursuit of supervised classification performance. While these components lead to effective accuracies and attractive FLOP counts, the added complexity actually makes these transformers *slower* than their vanilla ViT counterparts. In this paper, we argue that this additional bulk is *unnecessary*. By pretraining with a strong visual pretext task (MAE), we can strip out all the bells-and-whistles from a state-of-the-art multi-stage vision transformer *without losing accuracy*. In the process, we create Hiera, an extremely simple hierarchical vision transformer that is *more accurate* than previous models while being *significantly faster* both at inference and during training. We evaluate Hiera on a variety of tasks for image and video recognition. Our code and models are available at <https://github.com/facebookresearch/hiera>.

## 1. Introduction

Since their introduction by Dosovitskiy et al. (2021) a few years ago, Vision Transformers (ViTs) have dominated several tasks in computer vision. While architecturally simple, their accuracy (Touvron et al., 2022) and ability to scale (Zhai et al., 2021) make them still a popular choice today. Moreover, their simplicity unlocks the use of powerful pretraining strategies such as MAE (He et al., 2022), which make ViTs computationally and data efficient to train.

However, this simplicity comes at a cost: by using the same spatial resolution and number of channels throughout the network, ViTs make inefficient use of their parameters. This is in contrast to prior “hierarchical” or “multi-scale” models (e.g., Krizhevsky et al. (2012); He et al. (2016)), which use fewer channels but higher spatial resolution in early stages with simpler features, and more channels but lower spatial resolution later in the model with more complex features.

Several domain specific vision transformers have been in-

<sup>\*</sup>Equal contribution <sup>1</sup>Meta AI, FAIR <sup>2</sup>Georgia Tech <sup>3</sup>Johns Hopkins University. Correspondence to: Chaitanya Ryali <chayryali@meta.com>.

Figure 1. Hiera cuts out expensive specialized operations (e.g., convs) from hierarchical transformers to create a simple, efficient, and accurate model that is fast across many image and video tasks. Above we compare to recent MAE-based works (Woo et al., 2023; Feichtenhofer et al., 2022). All speeds measured with A100, fp16.

troduced that employ this hierarchical design, such as Swin (Liu et al., 2021) or MViT (Fan et al., 2021). However, in the pursuit of state-of-the-art results using fully supervised training on ImageNet-1K (an area where ViT has historically struggled), these models have become more and more complicated as they add specialized modules (e.g., cross-shaped windows in CSWin (Dong et al., 2022), decomposed relative position embeddings in MViTv2 (Li et al., 2022c)). While these changes produce effective models with attractive floating point operation (FLOP) counts, under the hood the added complexity makes these models *slower* overall.

We argue that a lot of this bulk is actually *unnecessary*. Because ViTs lack inductive bias after their initial patchify operation, many of the changes proposed by subsequent vision specific transformers serve to manually add spatial biases. But why should we slow down our architecture to add these biases, if we could just train the model to learn them instead? In particular, MAE pretraining has shown to be a very effective tool to teach ViTs spatial reasoning, allowing pure vision transformers to obtain good results on detection (Li et al., 2022b), which was a task previously dominated by models like Swin or MViT. Moreover, MAE pretraining is *sparse* and can be 4 – 10 $\times$  as fast as normal supervised training, making it an already desirable alternative across many domains for more than just accuracy (He et al., 2022; Feichtenhofer et al., 2022; Huang et al., 2022b).**Figure 2. Hiera Setup.** Modern hierarchical transformers like Swin (Liu et al., 2021) or MViT (Li et al., 2022c) are more parameter efficient than vanilla ViTs (Dosovitskiy et al., 2021), but end up slower due to overhead from adding spatial bias through vision-specific modules like shifted windows or convs. In contrast, we design Hiera to be as simple as possible. To add spatial bias, we opt to *teach* it to the model using a strong pretext task like MAE (pictured here) instead. Hiera consists entirely of standard ViT blocks. For efficiency, we use local attention within “mask units” (Fig. 4, 5) for the first two stages and global attention for the rest. At each stage transition,  $Q$  and the skip connection have their features doubled by a linear layer and spatial dimension pooled by a  $2 \times 2$  maxpool. Hiera-B is shown here (see Tab. 2 for other configs).

We test this hypothesis with a simple strategy: using some implementation tricks (Fig. 4), take an existing hierarchical ViT (e.g., MViTv2) and carefully *remove* non-essential components while training with MAE (Tab. 1). After tuning the MAE task to this new architecture (Tab. 3), we find that we can actually simplify or remove *all* of the non-transformer components, while *increasing in accuracy*. The result is an extremely efficient model with no bells-and-whistles: no convolutions, no shifted or cross-shaped windows, no decomposed relative position embeddings. Just a pure, simple hierarchical ViT that is both *faster and more accurate* than prior work across several model sizes, domains, and tasks.

Our Simple Hierarchical Vision Transformer (Hiera) outperforms the SotA on **images** and *far exceeds* prior work on **video** while being *much faster* (Fig. 1) at every model scale (Fig. 3) and across extensive datasets and tasks (Sec. 5, 6).

## 2. Related Work

**Vision transformers (ViTs)** have attracted attention because of their massive success on several vision tasks including image classification (Dosovitskiy et al., 2021), video classification (Fan et al., 2021; Arnab et al., 2021; Bertasius et al., 2021), semantic segmentation (Ranftl et al., 2021), object detection (Carion et al., 2020; Li et al., 2022b), video object segmentation (Duke et al., 2021), 3D object detection (Misra et al., 2021) and 3D reconstruction (Bozic et al., 2021). The key difference between vanilla ViT (Dosovitskiy et al., 2021) and prior convolutional neural networks (CNNs) (LeCun et al., 1998) is that ViT partitions images into, e.g.,  $16 \times 16$  pixel, *non-overlapping* patches and flattens the spatial grid into a 1D sequence, whereas CNNs maintain this grid over multiple stages of the model, reducing the resolution in each stage and introducing inductive biases such as shift equivariance. Recently, the field has shown an increased interest in hybrid methods (Fan et al., 2021;

**Figure 3. Performance vs. prior work.** Hiera compared to B, L, and H variants of SotA models that use MAE-like pretraining. On **images**, Hiera is faster and more accurate than even the most recent SotA (He et al., 2022; Gao et al., 2022; Woo et al., 2023), offering 30-40% speed-up compared to the best model at every scale. On **video**, Hiera represents a new class of performance, significantly improving accuracy, while *being over  $2 \times$  faster* than popular ViT models. Marker size is proportional to FLOP count.

Liu et al., 2021; Li et al., 2022c; Dong et al., 2022; Wang et al., 2021) that combine transformers with convolution-like operations and the hierarchical stage structure of prior CNNs. This direction has shown success and has achieved state-of-the-art on various vision tasks. However, in practice these models are actually *slower* than their vanilla ViT counterparts and convs are not easily compatible with popular self-supervised tasks such as masked image modeling. We address both of these issues in the creation of Hiera.

**Masked pretraining** has emerged as a powerful self-supervised learning pretext task for learning visual representations (Vincent et al., 2010; Pathak et al., 2016; Chen et al., 2020; He et al., 2022; Bao et al., 2022; Xie et al., 2022; Houet al., 2022). Among previous works, Masked AutoEncoder (MAE, He et al. (2022)) takes advantage of vanilla ViTs, which allow any length of input, and thereby derives an efficient training regime using the *sparsity* of masked images. This greatly improves the training efficiency of masked pretraining, but adapting sparse training to hierarchical models is nontrivial, because the input is no longer laid out in a rigid 2D grid. There have been several attempts to enable hierarchical ViTs to use masked pretraining. MaskFeat (Wei et al., 2022) and SimMIM (Xie et al., 2022) replace masked patches with `[mask]` tokens, meaning most computation is wasted on non-visible tokens and training is incredibly slow. Huang et al. (2022a) introduce several techniques to enable sparsity in every component of the network, in the end creating a much more complicated model that doesn’t improve much in accuracy. UM-MAE (Li et al., 2022a) uses a special masking strategy to allow for sparsity, but this restriction significantly hurts accuracy. MCMAE (Gao et al., 2022) uses masked convolution in the first couple of stages which obtains high accuracy but significantly reduces the efficiency of the model overall. We bypass all of these complicated techniques and restrictions by designing our architecture specifically for *sparse* MAE pretraining, thereby creating a powerful yet simple model.

### 3. Approach

Our goal is to create a powerful and efficient multiscale vision transformer that is, above all, *simple*. We argue that we do not need any specialized modules like convolution (Fan et al., 2021), shifted windows (Liu et al., 2021), or attention bias (Graham et al., 2021; Li et al., 2022c) to obtain high accuracy on vision tasks. This may seem difficult, as these techniques add much needed spatial (and temporal) biases that vanilla transformers (Dosovitskiy et al., 2021) lack. However, we employ a different strategy. While prior work adds spatial bias through complicated architectural changes, we opt to keep the model simple and *learn* these biases through a strong pretext task instead. To show the efficacy of this idea, we devise a simple experiment: take an existing hierarchical vision transformer and ablate its bells-and-whistles while training with a strong pretext task.

For the pretext task, we use Masked Autoencoders (MAE, He et al. (2022)), which has been shown effective in teaching ViTs localization capabilities for downstream tasks (e.g., detection (Li et al., 2022b)) by having the network reconstruct masked input patches (Fig. 2). Note that MAE pretraining is *sparse*—that is, masked tokens are *deleted* instead of being overwritten like in other masked image modeling approaches (Wei et al., 2022; Xie et al., 2022). This makes pretraining efficient, but poses a problem for existing hierarchical models as it breaks the 2D grid that they rely on (Fig. 4b). Moreover, MAE masks out individual tokens,

(a) Use Mask Units instead of tokens. One Token, One Mask Unit.

(b) Problem: MAE *deletes* mask units. This **breaks the 2D grid**, causing errors for hierarchical models (e.g., w/ convs).

(c) MaskFeat: Fill with `[mask]`. Not sparse: **VERY** slow training.

(d) Baseline: Separate units & pad. Sparse, but padding has overhead.

(e) Hiera: Just set *kernel size* = *stride*. Sparse, no overhead, simple.

**Figure 4. MAE for Hierarchical Models.** MAE is not compatible with multi-stage models, but we can apply some simple tricks to remedy this. While MAE masks individual tokens, tokens in multi-stage transformers start very small (e.g.,  $4 \times 4$  pixels), doubling size in each stage. (a) Thus, we mask coarser “mask units” ( $32 \times 32$  pixels) instead of tokens directly. (b) For efficiency, MAE is *sparse*, meaning it *deletes* what it masks (a problem for spatial modules like convs). (c) Keeping masked tokens fixes this, but gives up the potential  $4 - 10 \times$  training speed-up of MAE. (d) As a baseline, we introduce a trick that treats mask units as separate entities for convs, solving the issue but requiring undesirable padding. (e) In Hiera, we side-step the problem entirely by changing the architecture so the kernels can’t overlap between mask units.

which are large  $16 \times 16$  patches for ViT, but only small  $4 \times 4$  patches for most hierarchical models (Fig. 4a).

To address both of these issues, we opt to distinguish tokens from “mask units”. As described in Fig. 4a, mask units are at the resolution we apply MAE masking, while tokens are the internal resolution of the model (like in Wei et al. (2022); Xie et al. (2022)). In our case, we mask  $32 \times 32$  pixel regions, meaning one mask unit is  $8 \times 8$  tokens at the start of the network. Once we have made this distinction, we can use a clever trick (Fig. 4d) to evaluate hierarchical models by treating mask units as contiguous, separate from other tokens. Thus, we can continue with our experiments and use MAE with an existing hierarchical vision transformer.

#### 3.1. Preparing MViTv2

We choose MViTv2 as our base architecture, as its small  $3 \times 3$  kernels are affected the least by the separate-and-pad trick described in Fig. 4d, though we likely could have chosen a different transformer and obtained a similar end result. We briefly review MViTv2 below.

**MViTv2** (Li et al., 2022c) is a *hierarchical model*. That is, it learns multi-scale representations over its four stages. It starts by modeling low level features with a small channel capacity but high spatial resolution, and then in each stage**Figure 5. Mask Unit Attention.** MViTv2 uses pooling attention (a) which performs global attention with a pooled version of  $K$  and  $V$ . This can get expensive for large inputs (e.g., for video), so we opt to replace this with “Mask Unit Attention” (b) which performs local attention within mask units (Fig. 4a). This has no overhead because we already group tokens into units for masking. We do not have to worry about shifting like in Swin (Liu et al., 2021), because we use global attention in stages 3 and 4 (Fig. 2).

trades channel capacity for spatial resolution to model more complex high-level features in deeper layers.

A key feature of MViTv2 is *pooling attention* (Fig. 5a), wherein features are *locally aggregated*—typically using  $3 \times 3$  convolution, before computing self-attention. In pooling attention,  $K$  and  $V$  are pooled to decrease computation in the first two stages, while  $Q$  is pooled to transition from one stage to the next by reducing spatial resolution. MViTv2 also features *decomposed relative position embeddings* instead of absolute ones and a *residual pooling* connection to skip between pooled  $Q$  tokens inside the attention blocks. Note that by default, pooling attention in MViTv2 contain convs with stride 1 even if no downsampling is required.

**Applying MAE.** Since MViTv2 downsamples by  $2 \times 2$  a total of three times (Fig. 2) and because it uses a token size of  $4 \times 4$  pixels, we employ a mask unit of size  $32 \times 32$ . This ensures that each mask unit corresponds to  $8^2$ ,  $4^2$ ,  $2^2$ ,  $1^2$  tokens in stages 1, 2, 3, 4 respectively, allowing each mask unit to cover at least one distinct token in each stage. Then as described in Fig. 4d, to make sure conv kernels do not bleed into deleted tokens, we shift the mask units to the batch dimension to separate them for pooling (effectively treating each mask unit as an “image”) and then undo the shift afterward to ensure that self-attention is still global.

### 3.2. Simplifying MViTv2

In this section we *remove* non-essential components of MViTv2 while training with MAE. In Tab. 1, we find that we can remove or otherwise simplify *all* of them and still maintain high accuracy for image classification on ImageNet-1K. We use MViTv2-L to ensure our changes work at scale.

**Relative Position Embeddings.** MViTv2 swaps the absolute position embeddings in Dosovitskiy et al. (2021) for more powerful relative ones added to attention in *each block*. Technically, we could implement a version of this that is compatible with sparse pretraining, but doing so would add

<table border="1">
<thead>
<tr>
<th rowspan="2">Setting</th>
<th colspan="2">Image</th>
<th colspan="2">Video</th>
</tr>
<tr>
<th>acc.</th>
<th>im/s</th>
<th>acc.</th>
<th>clip/s</th>
</tr>
</thead>
<tbody>
<tr>
<td>MViTv2-L Supervised</td>
<td>85.3</td>
<td>219.8</td>
<td>80.5</td>
<td>20.5</td>
</tr>
<tr>
<td><b>Hiera-L MAE</b></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>a. replace rel pos with absolute *</td>
<td><b>85.6</b></td>
<td>253.3</td>
<td><b>85.3</b></td>
<td>20.7</td>
</tr>
<tr>
<td>b. replace convs with maxpools *</td>
<td>84.4</td>
<td>99.9<sup>†</sup></td>
<td>84.1</td>
<td>10.4<sup>†</sup></td>
</tr>
<tr>
<td>c. delete stride=1 maxpools *</td>
<td>85.4</td>
<td>309.2</td>
<td>84.3</td>
<td>26.2</td>
</tr>
<tr>
<td>d. set kernel size equal to stride</td>
<td><b>85.7</b></td>
<td>369.8</td>
<td><b>85.5</b></td>
<td>29.4</td>
</tr>
<tr>
<td>e. delete q attention residuals</td>
<td>85.6</td>
<td>374.3</td>
<td><b>85.5</b></td>
<td>29.8</td>
</tr>
<tr>
<td>f. replace kv pooling with MU attn</td>
<td><b>85.6</b></td>
<td><b>531.4</b></td>
<td><b>85.5</b></td>
<td><b>40.8</b></td>
</tr>
</tbody>
</table>

**Table 1. Simplifying MViTv2.** MViTv2 employs several architectural tweaks to perform well on supervised training. By progressively removing them in Sec. 3.2, we find these bells-and-whistles are *unnecessary* when training with a strong pretext task (MAE). In the process, we create an extremely simple model (Fig. 2) that is accurate while being significantly faster. We report fp16 inference speed for ImageNet-1K and Kinetics-400 on an A100. Our final Hiera-L in gray. \*Requires the separate-and-pad trick described in Fig. 4d. <sup>†</sup>PyTorch’s maxpool3d interacts unfavorably with this.

a lot of complexity. Instead, we opt to start our study here by undoing this change and using absolute position embeddings instead. As shown in Tab. 1a, these relative position embeddings are not necessary when training with MAE. Further, absolute position embeddings are much faster.

**Removing Convolutions.** Next, we aim to remove the convs in the model, which are vision specific modules and add potentially unnecessary overhead. We first attempt to replace every conv layer with maxpools (shown by Fan et al. (2021) to be the next best option), which itself is fairly costly. The result (Tab. 1b) drops accuracy by over 1% on images, but this is to be expected: we’ve also replaced all of the extra stride=1 convs with maxpools, which impacts the features significantly (with padding and small mask units, this in effect performs a relu on every feature map). Once we delete those additional stride=1 maxpools (Tab. 1c), we nearly return to the accuracy we had before, while speeding up the model by 22% for images and 27% for video. At this point, the only pooling layers that remain are for  $Q$  at stage transitions and for  $KV$  pooling in the first two stages.

**Removing Overlap.** The remaining maxpool layers still have a kernel size of  $3 \times 3$ , necessitating the use of the separate-and-pad trick in Fig. 4d during both training and inference. However, as shown in Fig. 4e, we can avoid this problem entirely if we just do not let these maxpool kernels overlap. That is, if we set the kernel size equal to stride for each maxpool, we can use sparse MAE pretraining *without* the separate-and-pad trick. As shown in Tab. 1d, this speeds up the model by 20% on image and 12% on video while increasing accuracy, likely due to not having to pad.

**Removing the Attention Residual.** MViTv2 adds a residual connection in the attention layer between  $Q$  and the output to assist in learning its pooling attention. However,**Figure 6. Mask Unit Attn vs. Window Attn.** Window attention (a) performs local attention within a *fixed* size window. Doing so would potentially overlap with deleted tokens during sparse MAE pretraining. In contrast, Mask Unit attention (b) performs local attention within individual mask units, no matter their size.

so far we’ve minimized the number of layers, making attention easier to learn. Thus, we can safely remove it (Tab. 1e).

**Mask Unit Attention.** At this point, the only specialized module left is pooling attention. Pooling  $Q$  is necessary to maintain a hierarchical model, but  $KV$  pooling is only there to reduce the size of the attention matrix in the first two stages. We can remove this outright, but it would considerably increase the computational cost of the network. Instead, in Tab. 1f we replace it with an implementationally trivial alternative: local attention within a mask unit.

During MAE pretraining, we already have to separate out mask units at the start of the network (see Fig. 2). Thus the tokens are already neatly grouped by units once they arrive at attention. We can then simply perform local attention within these units with no overhead. While this “Mask Unit attention” is local instead of global like pooling attention (Fig. 5),  $K$  and  $V$  were only pooled in the first two stages, where global attention isn’t as useful. Thus, as shown in Tab. 1, this change has no impact on accuracy but increases throughput by quite a lot—up to 32% on video.

Note that mask unit attention is distinct from window attention because it adapts the window size to the size of mask units at the current resolution. Window attention would have a fixed size throughout the network, which would leak into deleted tokens after a downsample (see Fig. 6).

**Hiera.** The result of these changes is an extremely simple and efficient model, which we denote “Hiera”. Hiera is  $2.4\times$  faster on images and  $5.1\times$  faster on video than the MViTv2 we started with and is actually *more accurate* because of MAE. Furthermore, because Hiera supports sparse pretraining, the results in Tab. 1 are extremely fast to obtain. In fact, to obtain superior accuracy on images, Hiera-L is  $3\times$  faster to train than a supervised MViTv2-L (Fig. 7). For video, Wei et al. (2022) report 80.5% using a *cut down* version of MViTv2 with double the  $KV$  stride in the first 3 stages. Compared to this model, our Hiera-L obtains 85.5% in 800 pretrain epochs while being  $2.1\times$  faster to train (Fig. 7). All benchmarks in this paper are on an A100 with fp16 (as this setting is most useful in practice) unless noted otherwise.

**Figure 7. Training time.** Measured in half precision A100 days. Our Hiera is *significantly* faster to train than MViTv2 due to being more efficient and benefiting from sparse pretraining (as opposed to MaskFeat). Here, supervised uses 300 epochs for ImageNet-1K and 200 for Kinetics-400, while MaskFeat and MAE use 400 for pretraining on images and 800 on video followed by 50 epochs of finetuning for both. Note that Hiera-L at 200 epochs of pretraining (81.8) already outperforms MViTv2-L supervised (80.5) on video, making it  $5.6\times$  faster to obtain higher accuracy.

While we used Hiera-L for the experiments in this section, we can of course instantiate it in different sizes, e.g. Tab. 2.

<table border="1">
<thead>
<tr>
<th>model</th>
<th>#Channels</th>
<th>#Blocks</th>
<th>#Heads</th>
<th>FLOPs</th>
<th>Param</th>
</tr>
</thead>
<tbody>
<tr>
<td>Hiera-T</td>
<td>[96-192-384-768]</td>
<td>[1-2-7-2]</td>
<td>[1-2-4-8]</td>
<td>5G</td>
<td>28M</td>
</tr>
<tr>
<td>Hiera-S</td>
<td>[96-192-384-768]</td>
<td>[1-2-11-2]</td>
<td>[1-2-4-8]</td>
<td>6G</td>
<td>35M</td>
</tr>
<tr>
<td>Hiera-B</td>
<td>[96-192-384-768]</td>
<td>[2-3-16-3]</td>
<td>[1-2-4-8]</td>
<td>9G</td>
<td>52M</td>
</tr>
<tr>
<td>Hiera-B+</td>
<td>[112-224-448-896]</td>
<td>[2-3-16-3]</td>
<td>[2-4-8-16]</td>
<td>13G</td>
<td>70M</td>
</tr>
<tr>
<td>Hiera-L</td>
<td>[144-288-576-1152]</td>
<td>[2-6-36-4]</td>
<td>[2-4-8-16]</td>
<td>40G</td>
<td>214M</td>
</tr>
<tr>
<td>Hiera-H</td>
<td>[256-512-1024-2048]</td>
<td>[2-6-36-4]</td>
<td>[4-8-16-32]</td>
<td>125G</td>
<td>673M</td>
</tr>
</tbody>
</table>

**Table 2. Configuration for Hiera variants.** #Channels, #Blocks and #Heads specify the channel width, number of Hierablocks and heads in each block for the four stages, respectively. FLOPs are measured for image classification with  $224 \times 224$  input. The stage resolutions are  $[56^2, 28^2, 14^2, 7^2]$ . We introduce B+ for more direct comparison against prior work with slower B models.

## 4. MAE Ablations

In this section, we ablate MAE pretraining settings in Hiera for both images and video, using ImageNet-1K (IN1K, Deng et al. (2009)) and Kinetics-400 (K400, Kay et al. (2017)). Like in He et al. (2022); Feichtenhofer et al. (2022), we ablate using our large model, Hiera-L, to ensure that our method works at scale. We evaluate performance by finetuning. All metrics are top-1 accuracies using standard evaluation protocols—a single (resized) center crop on IN1K and 3 spatial  $\times$  5 temporal views on K400.

**Multi-Scale decoder.** While He et al. (2022); Feichtenhofer et al. (2022) use the tokens from the *last* block of the encoder as the input to the decoder, Hiera being *hierarchical* permits more flexibility: as in Gao et al. (2022), we can use *multi-scale* information by fusing representations from all stages, which brings large gains in both modalities (Tab. 3a).<table border="1">
<thead>
<tr>
<th>multi-scale</th>
<th>image</th>
<th>video</th>
<th>mask</th>
<th>image</th>
<th>mask</th>
<th>video</th>
<th>target</th>
<th>image</th>
<th>video</th>
</tr>
</thead>
<tbody>
<tr>
<td>✗</td>
<td>85.0</td>
<td>83.8</td>
<td>0.5</td>
<td>85.5</td>
<td>0.75</td>
<td>84.9</td>
<td>pixel</td>
<td>85.6</td>
<td>85.5</td>
</tr>
<tr>
<td>✓</td>
<td><b>85.6</b></td>
<td><b>85.5</b></td>
<td>0.6</td>
<td><b>85.6</b></td>
<td>0.9</td>
<td><b>85.5</b></td>
<td>HOG</td>
<td><b>85.7</b></td>
<td><b>86.1</b></td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td>0.7</td>
<td>85.3</td>
<td>0.95</td>
<td>84.4</td>
<td></td>
<td></td>
<td></td>
</tr>
</tbody>
</table>

(a) **Multi-Scale Decoder.** Hiera being *hierarchical*, using multi-scale information for decoding brings significant gains.

(b) **Mask ratio.** High masking ratios lead to good performance, with video benefiting from higher masking than image modality.

(c) **Reconstruction target.** Both pixel and HOG targets result in strong performance.

<table border="1">
<thead>
<tr>
<th>dpr</th>
<th>image</th>
<th>video</th>
<th>depth</th>
<th>image</th>
<th>video</th>
<th>epochs</th>
<th>image</th>
<th>video</th>
</tr>
</thead>
<tbody>
<tr>
<td>0.0</td>
<td>85.2</td>
<td>84.5</td>
<td>4</td>
<td>85.5</td>
<td>84.8</td>
<td>400</td>
<td>85.6</td>
<td>84.0</td>
</tr>
<tr>
<td>0.1</td>
<td>85.6</td>
<td>85.4</td>
<td>8</td>
<td><b>85.6</b></td>
<td><b>85.5</b></td>
<td>800</td>
<td>85.8</td>
<td>85.5</td>
</tr>
<tr>
<td>0.2</td>
<td><b>85.6</b></td>
<td><b>85.5</b></td>
<td>12</td>
<td>85.5</td>
<td>85.4</td>
<td>1600</td>
<td><b>86.1</b></td>
<td>86.4</td>
</tr>
<tr>
<td>0.3</td>
<td>85.5</td>
<td>85.2</td>
<td></td>
<td></td>
<td></td>
<td>3200</td>
<td><b>86.1</b></td>
<td><b>87.3</b></td>
</tr>
</tbody>
</table>

(d) **Drop path rate.** Surprisingly, we find drop path important during MAE pretraining, especially for video, unlike in He et al. (2022); Feichtenhofer et al. (2022).

(e) **Decoder depth.** We find that a deeper decoder than in Feichtenhofer et al. (2022) works better for video.

(f) **Pretraining schedule.** Our pretraining follows the same trend as He et al. (2022), benefiting significantly from longer training.

Table 3. **Ablating MAE pretraining with Hiera-L.** For each ablation, we use 400 (800) epochs of sparse MAE pretraining for IN1K (K400) and 50 epochs of dense finetuning unless otherwise noted. Our default<sup>†</sup> settings are marked in gray. For design choices not ablated here, we find the defaults in (He et al., 2022; Feichtenhofer et al., 2022) to be appropriate. † default pretraining length for the rest of the paper is 1600 (3200) epochs, unless otherwise noted.

**Masking ratio.** Feichtenhofer et al. (2022) find video to require a much higher masking ratio than images, suggesting higher information redundancy. We observe a similar trend in Tab. 3b with optimal masking ratios of 0.6 for images but 0.9 for video. Our optimal ratio for images, 0.6, is slightly lower than the 0.75 used in He et al. (2022). We expect this is due to increased difficulty of the pretext task from using a  $32 \times 32$  mask unit instead of  $16 \times 16$  as in He et al. (2022). Interestingly, we find the same 0.9 masking ratio to be appropriate for video as in Feichtenhofer et al. (2022). This could be because they actually find 0.95 to work optimally if allowed to train twice as long. With our increased task difficulty, 0.9 works out to be best.

**Reconstruction target.** We find (Tab. 3c) that both pixel (w/ norm) and HOG (Dalal & Triggs, 2005) targets result in strong performance. While HOG targets results in slightly better performance for the default number of pretraining epochs we use in ablations, we found that with longer training HOG targets achieve the same performance as pixel targets for video, but slightly worse for images.

**Droppath rate.** The original MAE pretraining recipes (He et al., 2022; Feichtenhofer et al., 2022) explicitly do not use drop path (Huang et al., 2016) during pretraining, instead opting to only do so while finetuning. However, our Hiera-L has twice the depth of a ViT-L model: 48 for Hiera-L vs. 24 for ViT-L. While each layer individually has a lower parameter count, due to the sheer depth of Hiera, there could be a significant benefit from drop path.

In Tab. 3d, we ablate applying drop path during pretraining (finetuning employs drop path by default) and find significant gains. This is surprising because it means that without

drop path, Hiera can overfit to the MAE task.

**Decoder depth.** We find a significant benefit from a deeper decoder than previous work use for video (Feichtenhofer et al., 2022), see Tab. 3e. This brings the decoder for video in line with images (He et al., 2022).

**Pretraining schedule.** Several masked image modeling approaches (He et al., 2022; Wei et al., 2022) have found benefits from longer pretraining schedules, often using up to 1600 epochs. In Tab. 3f, we observe the same trend for Hiera, increasing +0.5% over 400 epochs on IN1K. In fact, Hiera’s accuracy at 400ep is +0.7% higher than ViT-L MAE (84.9%) at the same number of epochs but only +0.2% higher at 1600 epochs—suggesting that Hiera is a more *efficient* learner. On K400, even with only 800 epochs of pretraining, Hiera outperforms the previous SotA result that uses 1600 epochs (85.2%). Gains from longer training saturate less quickly on video, with a large 0.9% gain from 800 epochs to 1600 epochs and beyond.

## 5. Video Results

We report our results on video recognition. All models input 16 frames of  $224^2$  pixels unless otherwise specified. For video, mask units are  $2 \times 32 \times 32$  px (i.e.,  $1 \times 8 \times 8$  tokens as before). The rest of the model is the same as for images.

**Kinetics-400,-600,-700.** In Tab. 4, we compare Hiera trained with MAE to the SotA on Kinetics-400 (Kay et al., 2017) at a system level. We compare to MViTv2-L (Li et al., 2022c) pretrained with MaskFeat (Wei et al., 2022) and ViT (Dosovitskiy et al., 2021) pretrained with MAE<table border="1">
<thead>
<tr>
<th>backbone</th>
<th>pretrain</th>
<th>acc.</th>
<th>FLOPs (G)</th>
<th>Param</th>
</tr>
</thead>
<tbody>
<tr>
<td>ViT-B</td>
<td>MAE</td>
<td>81.5</td>
<td>180×3×5</td>
<td>87M</td>
</tr>
<tr>
<td>Hiera-B</td>
<td>MAE</td>
<td><u>84.0</u></td>
<td><u>102</u>×3×5</td>
<td><u>51M</u></td>
</tr>
<tr>
<td>Hiera-B+</td>
<td>MAE</td>
<td><b>85.0</b></td>
<td><u>133</u>×3×5</td>
<td><u>69M</u></td>
</tr>
<tr>
<td>MViTv2-L</td>
<td>-</td>
<td>80.5</td>
<td><u>377</u>×1×10</td>
<td><u>218M</u></td>
</tr>
<tr>
<td>MViTv2-L</td>
<td>MaskFeat</td>
<td>84.3</td>
<td><u>377</u>×1×10</td>
<td><u>218M</u></td>
</tr>
<tr>
<td>ViT-L</td>
<td>MAE</td>
<td>85.2</td>
<td>597×3×5</td>
<td>305M</td>
</tr>
<tr>
<td>Hiera-L</td>
<td>MAE</td>
<td><b>87.3</b></td>
<td>413×3×5</td>
<td><b>213M</b></td>
</tr>
<tr>
<td>ViT-H</td>
<td>MAE</td>
<td>86.6</td>
<td>1192×3×5</td>
<td><b>633M</b></td>
</tr>
<tr>
<td>Hiera-H</td>
<td>MAE</td>
<td><b>87.8</b></td>
<td><b>1159</b>×3×5</td>
<td>672M</td>
</tr>
</tbody>
</table>

Table 4. **K400 results**. Hiera improves on previous SotA by a large amount, while being lighter and faster. FLOPs are reported as inference FLOPs × spatial crops × temporal clips.

<table border="1">
<thead>
<tr>
<th>backbone</th>
<th>pretrain</th>
<th>acc.</th>
<th>FLOPs (G)</th>
<th>Param</th>
</tr>
</thead>
<tbody>
<tr>
<td>MViTv2-L</td>
<td>Sup, IN-21K</td>
<td>85.8</td>
<td>377×1×10</td>
<td>218M</td>
</tr>
<tr>
<td>MViTv2-L</td>
<td>MaskFeat</td>
<td><u>86.4</u></td>
<td><u>377</u>×1×10</td>
<td><u>218M</u></td>
</tr>
<tr>
<td>Hiera-L</td>
<td>MAE</td>
<td><b>88.3</b></td>
<td>413×3×5</td>
<td><b>213M</b></td>
</tr>
<tr>
<td>Hiera-H</td>
<td>MAE</td>
<td><b>88.8</b></td>
<td>1159×3×5</td>
<td>672M</td>
</tr>
</tbody>
</table>

(a) **Kinetics-600** video classification

<table border="1">
<thead>
<tr>
<th>backbone</th>
<th>pretrain</th>
<th>acc.</th>
<th>FLOPs (G)</th>
<th>Param</th>
</tr>
</thead>
<tbody>
<tr>
<td>MViTv2-L</td>
<td>Sup, IN-21K</td>
<td>76.7</td>
<td>377×1×10</td>
<td>218M</td>
</tr>
<tr>
<td>MViTv2-L</td>
<td>MaskFeat</td>
<td><u>77.5</u></td>
<td><u>377</u>×1×10</td>
<td><u>218M</u></td>
</tr>
<tr>
<td>Hiera-L</td>
<td>MAE</td>
<td><b>80.3</b></td>
<td>413×3×5</td>
<td><b>213M</b></td>
</tr>
<tr>
<td>Hiera-H</td>
<td>MAE</td>
<td><b>81.1</b></td>
<td>1159×3×5</td>
<td>672M</td>
</tr>
</tbody>
</table>

(b) **Kinetics-700** video classification

Table 5. **K600 and K700 results**. Hiera improves over SotA by a large margin. FLOPs reported as inference FLOPs × spatial crops × temporal clips. Approaches using extra data are de-emphasized.

on video (Feichtenhofer et al., 2022; Tong et al., 2022). Hiera-L brings large gains (+2.1%) over previous SotA (Feichtenhofer et al., 2022; Tong et al., 2022), while using ~45% fewer flops, being ~43% smaller and 2.3× faster (Fig. 3). In fact, Hiera-L significantly outperforms (+0.7%) models one tier higher, while being 3× smaller and 3.5× faster. Hiera-L achieves a gain of +6.8% over the corresponding MViTv2-L supervised baseline. Going one tier up in size, Hiera-H improves performance over previous SotA by +1.2%, establishing a new SotA for 224<sup>2</sup> without external data. We show similarly large improvements over the art on K600 (+1.9%) and K700 (+2.8%) in Tab. 5, with our H models bringing even further gains.

**Something-Something-v2 (SSv2).** In Tab. 6, we compare our Hiera with the current art on SSv2 (Goyal et al., 2017b) at a system level: MViTv2-L (Li et al., 2022c) pretrained with MaskFeat (Wei et al., 2022) and ViT (Dosovitskiy et al., 2021) pretrained with MAE on video (Tong et al., 2022). When pretrained on K400, Hiera-L outperforms the runner-up method MaskFeat by +0.6%, but Hiera is dramatically more efficient, using 16 frames at 224<sup>2</sup> resolution vs. 40 frames at 312<sup>2</sup> resolution in MaskFeat, effectively using

<table border="1">
<thead>
<tr>
<th>backbone</th>
<th>pretrain</th>
<th>acc.</th>
<th>FLOPs (G)</th>
<th>Param</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="5"><i>K400 pretrain</i></td>
</tr>
<tr>
<td>ViT-L</td>
<td>supervised</td>
<td>55.7</td>
<td>598×3×1</td>
<td>304M</td>
</tr>
<tr>
<td>MViTv2-L<sub>40,312</sub></td>
<td>MaskFeat</td>
<td>74.4</td>
<td>2828×3×1</td>
<td><u>218M</u></td>
</tr>
<tr>
<td>ViT-L</td>
<td>MAE</td>
<td>74.0</td>
<td><u>597</u>×3×2</td>
<td>305M</td>
</tr>
<tr>
<td>Hiera-L</td>
<td>MAE</td>
<td><u>74.7</u></td>
<td><u>413</u>×3×1</td>
<td><u>213M</u></td>
</tr>
<tr>
<td>Hiera-L</td>
<td>MAE</td>
<td><b>75.0</b></td>
<td><b>413</b>×3×2</td>
<td><b>213M</b></td>
</tr>
<tr>
<td colspan="5"><i>SSv2 pretrain</i></td>
</tr>
<tr>
<td>ViT-L</td>
<td>MAE</td>
<td>74.3</td>
<td>597×3×2</td>
<td>305M</td>
</tr>
<tr>
<td>Hiera-L</td>
<td>MAE</td>
<td><u>74.9</u></td>
<td><u>413</u>×3×1</td>
<td><u>213M</u></td>
</tr>
<tr>
<td>Hiera-L</td>
<td>MAE</td>
<td><b>75.1</b></td>
<td><b>413</b>×3×2</td>
<td><b>213M</b></td>
</tr>
<tr>
<td>ViT-L<sub>32</sub></td>
<td>MAE</td>
<td>75.4</td>
<td>1436×3×1</td>
<td>305M</td>
</tr>
<tr>
<td>Hiera-L<sub>32</sub></td>
<td>MAE</td>
<td><b>76.5</b></td>
<td><b>1029</b>×3×1</td>
<td><b>213M</b></td>
</tr>
</tbody>
</table>

Table 6. **SSv2 results** pretrained on Kinetics-400 and SSv2. Hiera improves over SotA by a large margin. We report inference FLOPs × spatial crops × temporal clips.

3.4× fewer FLOPs. When pretrained on SSv2, Hiera-L achieves 75.1%, outperforming ViT-L pretrained with MAE, by +0.8%, while using ~45% fewer flops and being ~43% smaller. Our Hiera-L<sub>32</sub> model further achieves 76.5%, SotA among approaches trained only on SSv2.

**Transferring to action detection (AVA).** We evaluate transfer learning of K400/K600/K700 pretrained Hiera on action detection using AVA v2.2 dataset (Gu et al., 2018). In Tab. 7 we compare the pretrained Hiera with SotA methods, MViTv2 with MaskFeat (Wei et al., 2022) and ViT with MAE on video (Tong et al., 2022; Feichtenhofer et al., 2022) at system level, and report mean average precision (mAP). Our K400 pretrained Hiera-L outperforms an MAE pretrained ViT-L by +2.8% and an MViTv2-L<sub>40,312</sub> MaskFeat by +1.3% mAP while Hiera-L has fewer FLOPs and parameters. Our Hiera-H outperforms an MAE pretrained ViT-H by +3.0% mAP. We observe similar performance improvement of the K600/K700 pretrained Hiera as well. Specifically, the K700 pretrained Hiera-H outperforms an MAE pretrained ViT-H by +3.2, establishing a new SotA.

## 6. Image Results

We first evaluate performance on IN1K and then transfer to other image recognition, detection, and segmentation tasks.

### 6.1. Performance on ImageNet-1K

In Tab. 8, we perform a system-level comparison of Hiera trained with MAE to relevant prior work. First, we observe that the supervised MViTv2 baselines are already quite strong, with MViTv2-B (L) reaching 84.4 (85.3) top-1 accuracy—better than several approaches that use pre-training (e.g. ViT-B MAE). This showcases the significant benefits that convolutions give in the supervised setting, es-<table border="1">
<thead>
<tr>
<th>backbone</th>
<th>pretrain</th>
<th>mAP</th>
<th>FLOPs (G)</th>
<th>Param</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="5"><i>K400 pretrain</i></td>
</tr>
<tr>
<td>ViT-L</td>
<td>supervised</td>
<td>22.2</td>
<td>598</td>
<td>304M</td>
</tr>
<tr>
<td>MViTv2-L<sub>40,312</sub></td>
<td>MaskFeat</td>
<td><u>38.5</u></td>
<td>2828</td>
<td><u>218M</u></td>
</tr>
<tr>
<td>ViT-L</td>
<td>MAE</td>
<td>37.0</td>
<td><u>597</u></td>
<td>305M</td>
</tr>
<tr>
<td>Hiera-L</td>
<td>MAE</td>
<td><b>39.8</b></td>
<td><b>413</b></td>
<td><b>213M</b></td>
</tr>
<tr>
<td>ViT-H</td>
<td>MAE</td>
<td>39.5</td>
<td>1192</td>
<td><b>633M</b></td>
</tr>
<tr>
<td>Hiera-H</td>
<td>MAE</td>
<td><b>42.5</b></td>
<td><b>1158</b></td>
<td>672M</td>
</tr>
<tr>
<td colspan="5"><i>K600 pretrain</i></td>
</tr>
<tr>
<td>ViT-L</td>
<td>MAE</td>
<td>38.4</td>
<td>598</td>
<td>304M</td>
</tr>
<tr>
<td>MViTv2-L<sub>40,312</sub></td>
<td>MaskFeat</td>
<td><u>39.8</u></td>
<td>2828</td>
<td><u>218M</u></td>
</tr>
<tr>
<td>Hiera-L</td>
<td>MAE</td>
<td><b>40.7</b></td>
<td><b>413</b></td>
<td><b>213M</b></td>
</tr>
<tr>
<td>ViT-H</td>
<td>MAE</td>
<td>40.3</td>
<td>1193</td>
<td><b>632M</b></td>
</tr>
<tr>
<td>Hiera-H</td>
<td>MAE</td>
<td><b>42.8</b></td>
<td><b>1158</b></td>
<td>672M</td>
</tr>
<tr>
<td colspan="5"><i>K700 pretrain</i></td>
</tr>
<tr>
<td>ViT-L</td>
<td>MAE</td>
<td>39.5</td>
<td>598</td>
<td>304M</td>
</tr>
<tr>
<td>Hiera-L</td>
<td>MAE</td>
<td><b>41.7</b></td>
<td><b>413</b></td>
<td><b>213M</b></td>
</tr>
<tr>
<td>ViT-H</td>
<td>MAE</td>
<td>40.1</td>
<td>1193</td>
<td><b>632M</b></td>
</tr>
<tr>
<td>Hiera-H</td>
<td>MAE</td>
<td><b>43.3</b></td>
<td><b>1158</b></td>
<td>672M</td>
</tr>
</tbody>
</table>

Table 7. **AVA v2.2 results** pretrained on Kinetics. Hiera improves over SotA by a large margin. All inference FLOPs reported with a center crop strategy following Fan et al. (2021).

pecially at the *base* model size and *lower*. Remarkably, even at this size, Hiera-B *without* using any bells-and-whistles (e.g., convs), is able to reach **84.5%** (slightly) outperforming MViTv2-B; MCMAE-B achieves a higher accuracy, but the model is significantly heavier. Our Hiera-B+ model handily outperforms it in both speed (Fig. 3) and accuracy. Going *even smaller*, Hiera-S, -T demonstrate remarkably strong performance - in a scale regime where convolutions have historically dominated, consistent with our core premise that good spatial biases can be *learned*.

At our default scale, Hiera-L MAE reaches an accuracy of **86.1%**, a significant **+0.8%** gain over MViTv2-L; it also (slightly) outperforms ViT-L MAE, which is 42% larger and has  $1.6\times$  the FLOPs, by **+0.2%**. Note that while we adopted the MAE pretraining in this work due to its efficient sparse pretraining, Hiera-L is readily compatible with complementary, *orthogonal* approaches, e.g. using an EMA teacher (El-Noubi et al., 2021; Baevski et al., 2022).

## 6.2. Transfer learning experiments

Here, we perform transfer learning experiments on downstream classification, detection, and segmentation tasks.

**Classification on iNaturalists and Places.** In Tab. 9 we evaluate transfer learning performance on downstream iNaturalist (Van Horn et al., 2018) and Places (Zhou et al., 2014) datasets. We finetune the ImageNet-1K pretrained Hiera on iNaturalist 2017, 2018, and 2019, and Places 365. Hiera consistently outperforms ViT pretrained with MAE (He et al., 2022), indicating that our Hiera-L and Hiera-H architectures

<table border="1">
<thead>
<tr>
<th>backbone</th>
<th>pretrain</th>
<th>acc.</th>
<th>FLOPs (G)</th>
<th>Param</th>
</tr>
</thead>
<tbody>
<tr>
<td>Swin-T</td>
<td></td>
<td>81.3</td>
<td><b>5</b></td>
<td>29M</td>
</tr>
<tr>
<td>MViTv2-T</td>
<td></td>
<td><u>82.3</u></td>
<td><b>5</b></td>
<td><b>24M</b></td>
</tr>
<tr>
<td>Hiera-T</td>
<td>MAE</td>
<td><b>82.8</b></td>
<td><b>5</b></td>
<td>28M</td>
</tr>
<tr>
<td>Swin-S</td>
<td></td>
<td>83.0</td>
<td>9</td>
<td>50M</td>
</tr>
<tr>
<td>MViTv2-S</td>
<td></td>
<td><u>83.6</u></td>
<td><u>7</u></td>
<td><b>35M</b></td>
</tr>
<tr>
<td>Hiera-S</td>
<td>MAE</td>
<td><b>83.8</b></td>
<td><b>6</b></td>
<td><b>35M</b></td>
</tr>
<tr>
<td>ViT-B</td>
<td></td>
<td>82.3</td>
<td>18</td>
<td>87M</td>
</tr>
<tr>
<td>Swin-B</td>
<td></td>
<td>83.3</td>
<td>15</td>
<td>88M</td>
</tr>
<tr>
<td>MViTv2-B</td>
<td></td>
<td>84.4</td>
<td><u>10</u></td>
<td><b>52M</b></td>
</tr>
<tr>
<td>ViT-B</td>
<td>BEiT<sub>DALLE</sub></td>
<td>83.2</td>
<td>18</td>
<td>87M</td>
</tr>
<tr>
<td>ViT-B</td>
<td>MAE</td>
<td>83.6</td>
<td>18</td>
<td>87M</td>
</tr>
<tr>
<td>ViT-B</td>
<td>MaskFeat</td>
<td>84.0</td>
<td>18</td>
<td>87M</td>
</tr>
<tr>
<td>Swin-B</td>
<td>SimMIM</td>
<td>83.8</td>
<td>15</td>
<td>88M</td>
</tr>
<tr>
<td>MCMAE-B</td>
<td>MCMAE</td>
<td>85.0</td>
<td>28</td>
<td>88M</td>
</tr>
<tr>
<td>Hiera-B</td>
<td>MAE</td>
<td>84.5</td>
<td><b>9</b></td>
<td><b>52M</b></td>
</tr>
<tr>
<td>Hiera-B+</td>
<td>MAE</td>
<td><b>85.2</b></td>
<td>13</td>
<td>70M</td>
</tr>
<tr>
<td>ViT-L</td>
<td></td>
<td>82.6</td>
<td>62</td>
<td>304M</td>
</tr>
<tr>
<td>MViTv2-L</td>
<td></td>
<td>85.3</td>
<td>42</td>
<td>218M</td>
</tr>
<tr>
<td>ViT-L</td>
<td>BEiT<sub>DALLE</sub></td>
<td>85.2</td>
<td>62</td>
<td>304M</td>
</tr>
<tr>
<td>ViT-L</td>
<td>MAE</td>
<td>85.9</td>
<td>62</td>
<td>304M</td>
</tr>
<tr>
<td>ViT-L</td>
<td>MaskFeat</td>
<td>85.7</td>
<td>62</td>
<td>304M</td>
</tr>
<tr>
<td>Swin-L</td>
<td>SimMIM</td>
<td>85.4</td>
<td><b>36</b></td>
<td><b>197M</b></td>
</tr>
<tr>
<td>MCMAE-L</td>
<td>MCMAE</td>
<td><b>86.2</b></td>
<td>94</td>
<td>323M</td>
</tr>
<tr>
<td>Hiera-L</td>
<td>MAE</td>
<td>86.1</td>
<td>40</td>
<td>214M</td>
</tr>
<tr>
<td>ViT-H</td>
<td></td>
<td><u>83.1</u></td>
<td><u>167</u></td>
<td><b>632M</b></td>
</tr>
<tr>
<td>ViT-H</td>
<td>MAE</td>
<td><b>86.9</b></td>
<td><u>167</u></td>
<td><b>632M</b></td>
</tr>
<tr>
<td>Hiera-H</td>
<td>MAE</td>
<td><b>86.9</b></td>
<td><b>125</b></td>
<td><u>673M</u></td>
</tr>
</tbody>
</table>

Table 8. **ImageNet-1K** comparison to previous MIM approaches. We de-emphasize approaches using extra data and indicate the source of extra data.

are effective outside of just ImageNet.

**Object detection and segmentation on COCO.** We finetune Mask R-CNN (He et al., 2017) with different pretrained backbones on the COCO dataset (Lin et al., 2014). We report  $AP^{\text{box}}$  and  $AP^{\text{mask}}$  for object detection and instance segmentation. We utilize the training recipe following ViT-Det (Li et al., 2022b) and incorporate multi-scale features from Hiera with a Feature Pyramid Network (FPN, Lin et al. (2017)) as described in the original paper.

In Tab. 10, our Hiera with MAE pretraining demonstrates a strong scaling behavior when compared models with supervised pretraining such as MViTv2 (Li et al., 2022c), while being consistently faster. For example, Hiera-L is **+1.8**  $AP^{\text{box}}$  higher than MViTv2-L (55.0 vs. 53.2) with a **24%** reduction in inference time. Even when compared to MViTv2 using ImageNet-21K pretraining, Hiera-L still performs **+1.4**  $AP^{\text{box}}$  better than MViTv2-L.

When compared to the state-of-the-art method, ViTDet, our Hiera models achieve comparable results while having faster inference and a lower operation count. For example, Hiera-B shows **+0.6** higher  $AP^{\text{box}}$  than ViTDet-B with **34%** fewer parameters and **15%** lower inference time. Additionally, Hiera-B+ achieves **+1.9** boxAP improvements while having<table border="1">
<thead>
<tr>
<th>backbone</th>
<th>iNat17</th>
<th>iNat18</th>
<th>iNat19</th>
<th>Places365</th>
</tr>
</thead>
<tbody>
<tr>
<td>ViT-B</td>
<td>70.5</td>
<td>75.4</td>
<td>80.5</td>
<td>57.9</td>
</tr>
<tr>
<td>Hiera-B</td>
<td><u>73.3</u></td>
<td><u>77.9</u></td>
<td><u>83.0</u></td>
<td><u>58.9</u></td>
</tr>
<tr>
<td>Hiera-B+</td>
<td><b>74.7</b></td>
<td><b>79.9</b></td>
<td><b>83.1</b></td>
<td><b>59.2</b></td>
</tr>
<tr>
<td>ViT-L</td>
<td>75.7</td>
<td>80.1</td>
<td>83.4</td>
<td>59.4</td>
</tr>
<tr>
<td>Hiera-L</td>
<td><b>76.8</b></td>
<td><b>80.9</b></td>
<td><b>84.3</b></td>
<td><b>59.6</b></td>
</tr>
<tr>
<td>ViT-H</td>
<td>79.3</td>
<td>83.0</td>
<td><b>85.7</b></td>
<td>59.8</td>
</tr>
<tr>
<td>Hiera-H</td>
<td><b>79.6</b></td>
<td><b>83.5</b></td>
<td><b>85.7</b></td>
<td><b>60.0</b></td>
</tr>
<tr>
<td>ViT-H<sub>448</sub></td>
<td>83.4</td>
<td>86.8</td>
<td>88.3</td>
<td>60.3</td>
</tr>
<tr>
<td>Hiera-H<sub>448</sub></td>
<td><b>83.8</b></td>
<td><b>87.3</b></td>
<td><b>88.5</b></td>
<td><b>60.6</b></td>
</tr>
</tbody>
</table>

 Table 9. **Transfer learning** on iNaturalists and Places datasets.

<table border="1">
<thead>
<tr>
<th>backbone</th>
<th>pretrain</th>
<th>AP<sup>box</sup></th>
<th>AP<sup>mask</sup></th>
<th>FLOPs</th>
<th>params</th>
<th>time</th>
</tr>
</thead>
<tbody>
<tr>
<td>Swin-B</td>
<td>Sup, 21K</td>
<td>51.4</td>
<td>45.4</td>
<td>0.7T</td>
<td>109M</td>
<td>164ms</td>
</tr>
<tr>
<td>MViTv2-B</td>
<td>Sup, 21K</td>
<td>53.1</td>
<td>47.4</td>
<td>0.6T</td>
<td>73M</td>
<td>208ms</td>
</tr>
<tr>
<td>Swin-B</td>
<td>Sup</td>
<td>50.1</td>
<td>44.5</td>
<td>0.7T</td>
<td>109M</td>
<td><b>164ms</b></td>
</tr>
<tr>
<td>MViTv2-B</td>
<td>Sup</td>
<td><u>52.4</u></td>
<td><u>46.7</u></td>
<td><b>0.6T</b></td>
<td><b>73M</b></td>
<td>208ms</td>
</tr>
<tr>
<td>ViTDet-B</td>
<td>MAE</td>
<td>51.6</td>
<td>45.9</td>
<td>0.8T</td>
<td>111M</td>
<td>201ms</td>
</tr>
<tr>
<td>Hiera-B</td>
<td>MAE</td>
<td>52.2</td>
<td>46.3</td>
<td><b>0.6T</b></td>
<td><b>73M</b></td>
<td><u>173ms</u></td>
</tr>
<tr>
<td>Hiera-B+</td>
<td>MAE</td>
<td><b>53.5</b></td>
<td><b>47.3</b></td>
<td><b>0.6T</b></td>
<td>92M</td>
<td>192ms</td>
</tr>
<tr>
<td>Swin-L</td>
<td>Sup, 21K</td>
<td>52.4</td>
<td>46.2</td>
<td>1.1T</td>
<td>218M</td>
<td>243ms</td>
</tr>
<tr>
<td>MViTv2-L</td>
<td>Sup, 21K</td>
<td>53.6</td>
<td>47.5</td>
<td>1.3T</td>
<td>239M</td>
<td>447ms</td>
</tr>
<tr>
<td>MViTv2-L</td>
<td>Sup</td>
<td>53.2</td>
<td>47.1</td>
<td><u>1.3T</u></td>
<td><u>239M</u></td>
<td>447ms</td>
</tr>
<tr>
<td>ViTDet-L</td>
<td>MAE</td>
<td><b>55.6</b></td>
<td><b>49.2</b></td>
<td>1.9T</td>
<td>331M</td>
<td><u>396ms</u></td>
</tr>
<tr>
<td>Hiera-L</td>
<td>MAE</td>
<td><u>55.0</u></td>
<td><u>48.6</u></td>
<td><b>1.2T</b></td>
<td><b>236M</b></td>
<td><b>340ms</b></td>
</tr>
</tbody>
</table>

 Table 10. **COCO object detection and segmentation** using Mask-RCNN. All methods are following the training recipe from Li et al. (2022b) and pretrained on ImageNet-1K by default. Methods using ImageNet-21K pretraining are de-emphasized. Test time is measured on a single V100 GPU with full precision.

lower inference time and model complexity vs. ViTDet-B. For the large model, Hiera-L is consistently faster than ViTDet-L with only a slightly lower accuracy.

## 7. Conclusion

In this work, we create a simple hierarchical vision transformer by taking an existing one and removing all its bells-and-whistles while supplying the model with spatial bias through MAE pretraining. The resulting architecture, Hiera, is more effective than current work on image recognition tasks and surpasses the state-of-the-art on video tasks. We hope Hiera can allow future work to do more, faster.

## A. Implementation Details

A mask unit for video corresponds to a block of 2 frames  $\times 32 \text{ px} \times 32 \text{ px}$  (as opposed to images which use  $1 \times 32 \times 32$ ). Following Feichtenhofer et al. (2022), each token in Hiera on video corresponds to 2 frames of the input. Since the mask units also span 2 frames, the window sizes for Mask Unit Attention do not change for video (i.e.,  $1 \times 8 \times 8$  tokens in the first stage,  $1 \times 4 \times 4$  tokens in the second stage)—meaning we use exactly the same implementation for images and video (just the mask unit size is changed). We use learned spatial (separable spatio-temporal) position

embeddings for images (video). These are all the differences between Hiera for images and for video. The rest of the encoder is completely agnostic to spatio-temporal structure.

As in Wei et al. (2022), we remove  $Q$ -pooling before the last stage for MAE pretraining only. This is done so that MAE settings from prior work using ViT also work for Hiera with minimal modifications. This introduces little extra computation as stage 4 is small. If desired, by *design*, pretraining with Hiera can also work without removal of query pooling during pretraining, since a mask unit of  $1 \times 8 \times 8$  tokens would correspond to 1 distinct token in the last stage.

### A.1. Video Experiments

**Kinetics-400, -600, -700.** Our settings mainly follow Feichtenhofer et al. (2022). We report the pretraining and finetuning settings for our main results on the Kinetics-400 (Kay et al., 2017), -600 (Carreira et al., 2018) and -700 (Carreira et al., 2019) human action datasets in Tab. 11. Epochs are always reported as effective epochs (Feichtenhofer et al., 2022), i.e. accounting for repeated sampling. We use  $16 \times 4$  sampling as in Feichtenhofer et al. (2022).

**Something-Something-v2 (SSv2).** We evaluate Hiera-L on the SSv2 dataset (Goyal et al., 2017b). SSv2 is a dataset focusing on human-object interaction classification. We pretrain Hiera-L on either Kinetics 400 or SSv2 and finetune on SSv2. We report the top-1 classification accuracy in Tab. 6. We provide further details about the pretraining and finetuning settings on SSv2 in Tab. 12.

**AVA v2.2.** We perform transferring experiments on AVA v2.2 (Gu et al., 2018) for human action localization in video. We adopt a detection framework following (Feichtenhofer et al., 2019) for human action localization. We extract ROI features from the feature map of the last layer in Hiera and pool the ROI features via spatial max-pooling. We then use a linear classifier trained with cross entropy loss to predict the action class. We use the center crop for Hiera in the evaluation and report the mAP in Tab. 7. We use Kinetics pretrained and finetuned Hiera in the experiments. We provide details about the finetuning setting on AVA v2.2 in Tab. 13.

### A.2. Image Experiments

**ImageNet-1K.** Our settings mainly follow He et al. (2022). We report the pretraining and finetuning settings for our main results in Tab. 14.

**Transfer learning on iNaturalists and Places.** We conduct transfer learning experiments on classification datasets including iNaturalist2017, iNaturalist2018, iNat-<table border="1">
<thead>
<tr>
<th>config</th>
<th>value</th>
</tr>
</thead>
<tbody>
<tr>
<td>optimizer</td>
<td>AdamW (Loshchilov &amp; Hutter, 2019)</td>
</tr>
<tr>
<td>optimizer momentum</td>
<td><math>\beta_1, \beta_2=0.9, 0.95</math></td>
</tr>
<tr>
<td>weight decay</td>
<td>0.05</td>
</tr>
<tr>
<td>learning rate</td>
<td><math>8e-4</math> (B, B+, L); <math>- / 8e-4 / 3.2e-4</math> (H)</td>
</tr>
<tr>
<td>learning rate sch.</td>
<td>cosine decay (Loshchilov &amp; Hutter, 2017)</td>
</tr>
<tr>
<td>warmup epochs (Goyal et al., 2017a)</td>
<td>120</td>
</tr>
<tr>
<td>epochs</td>
<td>800 / 1600 / 3200</td>
</tr>
<tr>
<td>repeated sampling (Hoffer et al., 2020)</td>
<td>4</td>
</tr>
<tr>
<td>augmentation</td>
<td>hflip, crop [0.5, 1]</td>
</tr>
<tr>
<td>batch size</td>
<td>512</td>
</tr>
<tr>
<td>num. decoder blocks</td>
<td>8</td>
</tr>
<tr>
<td>num. decoder heads</td>
<td>8</td>
</tr>
<tr>
<td>mask ratio</td>
<td>0.9</td>
</tr>
<tr>
<td>drop path (Huang et al., 2016)</td>
<td>0.1 (B); 0.2 (B+, L); <math>- / 0.3 / 0.4</math> (H)</td>
</tr>
</tbody>
</table>

(a) Pretraining

<table border="1">
<thead>
<tr>
<th>config</th>
<th>value</th>
</tr>
</thead>
<tbody>
<tr>
<td>optimizer</td>
<td>AdamW</td>
</tr>
<tr>
<td>optimizer momentum</td>
<td><math>\beta_1, \beta_2=0.9, 0.999</math></td>
</tr>
<tr>
<td>weight decay</td>
<td>0.05</td>
</tr>
<tr>
<td>learning rate</td>
<td><math>8e-4</math> (B, B+, L), <math>4e-4</math> (H)</td>
</tr>
<tr>
<td>learning rate schedule</td>
<td>cosine decay</td>
</tr>
<tr>
<td>warmup epochs</td>
<td>10</td>
</tr>
<tr>
<td>epochs</td>
<td>150 (B, B+), 100 (L, H)</td>
</tr>
<tr>
<td>repeated sampling</td>
<td>2</td>
</tr>
<tr>
<td>augmentation</td>
<td>RandAug (7, 0.5) (Cubuk et al., 2020)</td>
</tr>
<tr>
<td>batch size</td>
<td>256</td>
</tr>
<tr>
<td>gradient clipping</td>
<td>5.0</td>
</tr>
<tr>
<td>mixup (Zhang et al., 2018)</td>
<td>0.8</td>
</tr>
<tr>
<td>cutmix (Yun et al., 2019)</td>
<td>1.0</td>
</tr>
<tr>
<td>label smoothing (Szegedy et al., 2016)</td>
<td>0.1</td>
</tr>
<tr>
<td>drop path</td>
<td>0.2 (B, B+, L), 0.3 (H)</td>
</tr>
<tr>
<td>dropout (Srivastava et al., 2014)</td>
<td>0.3 (B, B+), 0.5 (L, H)</td>
</tr>
<tr>
<td>layer-wise decay (Clark et al., 2020)</td>
<td><math>- / 0.85 / 0.8</math> (B, B+); <math>0.925 / 0.9 / 0.875</math> (L, H)</td>
</tr>
</tbody>
</table>

(b) Finetuning
 Table 11. Settings for Kinetics-400, -600, -700. Notation: setting corresponding to 800 / 1600 / 3200 epochs of pretraining.

uralist2019 (Van Horn et al., 2018) and Places365 (Zhou et al., 2014). Following (He et al., 2022), we adjust learning rate, training epochs on each dataset. We search the layer-wise decay among 0.875, 0.9 and 0.925, drop path rate between 0.1 to 0.5, and the dropout rate among 0.1, 0.2 and 0.3. For Hiera-H<sub>448</sub>, we set the learning rate decay of the positional embedding to 0.5 instead of following the layer-wise decay rule.

**COCO.** We use the Mask R-CNN (He et al., 2017) framework in Detectron2 (Wu et al., 2019) for object detection and instance segmentation experiments on the COCO dataset. Similar to ViTDet (Li et al., 2022b), we use 2 hidden convolution layers for the RPN and 4 hidden convolution layers for the RoI heads for Hiera and all comparison detection methods. These layers are followed by LayerNorm layers.

<table border="1">
<thead>
<tr>
<th>config</th>
<th>value</th>
</tr>
</thead>
<tbody>
<tr>
<td>optimizer</td>
<td>AdamW</td>
</tr>
<tr>
<td>optimizer momentum</td>
<td><math>\beta_1, \beta_2=0.9, 0.95</math></td>
</tr>
<tr>
<td>weight decay</td>
<td>0.05</td>
</tr>
<tr>
<td>learning rate</td>
<td><math>8e-4</math></td>
</tr>
<tr>
<td>learning rate schedule</td>
<td>cosine decay</td>
</tr>
<tr>
<td>warmup epochs</td>
<td>30</td>
</tr>
<tr>
<td>epochs</td>
<td>1600</td>
</tr>
<tr>
<td>augmentation</td>
<td>crop [0.5, 1]</td>
</tr>
<tr>
<td>batch size</td>
<td>1024</td>
</tr>
<tr>
<td>gradient clipping</td>
<td>0.02</td>
</tr>
<tr>
<td>num. decoder blocks</td>
<td>8</td>
</tr>
<tr>
<td>num. decoder heads</td>
<td>8</td>
</tr>
<tr>
<td>mask ratio</td>
<td>0.9</td>
</tr>
<tr>
<td>drop path</td>
<td>0.2</td>
</tr>
</tbody>
</table>

(a) Pretraining

<table border="1">
<thead>
<tr>
<th>config</th>
<th>values</th>
</tr>
</thead>
<tbody>
<tr>
<td>optimizer</td>
<td>SGD</td>
</tr>
<tr>
<td>weight decay</td>
<td><math>1e-4</math></td>
</tr>
<tr>
<td>learning rate</td>
<td>0.16 / 0.08</td>
</tr>
<tr>
<td>learning rate schedule</td>
<td>cosine decay</td>
</tr>
<tr>
<td>warmup epochs</td>
<td>3</td>
</tr>
<tr>
<td>epochs</td>
<td>40</td>
</tr>
<tr>
<td>augmentation</td>
<td>RandAug (7, 0.5)</td>
</tr>
<tr>
<td>batch size</td>
<td>256 / 128</td>
</tr>
<tr>
<td>mixup</td>
<td>0.8 / -</td>
</tr>
<tr>
<td>cutmix</td>
<td>1.0 / -</td>
</tr>
<tr>
<td>label smoothing</td>
<td>0.1 / -</td>
</tr>
<tr>
<td>drop path</td>
<td>0.1 / 0.2</td>
</tr>
<tr>
<td>dropout</td>
<td>0.5</td>
</tr>
<tr>
<td>layer-wise decay</td>
<td>0.875</td>
</tr>
</tbody>
</table>

(b) Finetuning
 Table 12. Settings for SSv2. Notation: setting corresponding to Hiera-L / L<sub>32</sub>.

<table border="1">
<thead>
<tr>
<th>config</th>
<th>values</th>
</tr>
</thead>
<tbody>
<tr>
<td>optimizer</td>
<td>SGD</td>
</tr>
<tr>
<td>weight decay</td>
<td><math>1e-8</math></td>
</tr>
<tr>
<td>learning rate</td>
<td>3.6</td>
</tr>
<tr>
<td>learning rate schedule</td>
<td>cosine decay</td>
</tr>
<tr>
<td>warmup epochs</td>
<td>5</td>
</tr>
<tr>
<td>epochs</td>
<td>30</td>
</tr>
<tr>
<td>batch size</td>
<td>128</td>
</tr>
<tr>
<td>drop path</td>
<td>0.4</td>
</tr>
<tr>
<td>dropout</td>
<td>0.5</td>
</tr>
<tr>
<td>layer-wise decay</td>
<td>0.875</td>
</tr>
</tbody>
</table>

 Table 13. Settings for AVA. Hiera-L and Hiera-H finetuning settings on AVA.

For the training recipe, we follow ViTDet to use input size as  $1024 \times 1024$  with large-scale jittering (LSJ) (Ghiasi et al., 2021). We don’t use the layer-wise decay during training. Additional hyperparamters can be found in Tab. 15.

### A.3. Speed Benchmarking

We use an NVIDIA A100 40GB GPU, PyTorch v1.12.1 and CUDA 11.4 to benchmark speed for all baselines and our approach, unless otherwise mentioned. Note that we did *not* use Flash Attention (Dao et al., 2022) or any other<table border="1">
<thead>
<tr>
<th>config</th>
<th>value</th>
</tr>
</thead>
<tbody>
<tr>
<td>optimizer</td>
<td>AdamW</td>
</tr>
<tr>
<td>optimizer momentum</td>
<td><math>\beta_1, \beta_2=0.9, 0.95</math></td>
</tr>
<tr>
<td>weight decay</td>
<td>0.05</td>
</tr>
<tr>
<td>learning rate</td>
<td><math>8e-4</math></td>
</tr>
<tr>
<td>learning rate sch.</td>
<td>cosine decay</td>
</tr>
<tr>
<td>warmup epochs</td>
<td>40</td>
</tr>
<tr>
<td>epochs</td>
<td>400 / 1600</td>
</tr>
<tr>
<td>augmentation</td>
<td>hflip, crop [0.2, 1]</td>
</tr>
<tr>
<td>batch size</td>
<td>4096</td>
</tr>
<tr>
<td>num. decoder blocks</td>
<td>8</td>
</tr>
<tr>
<td>num. decoder heads</td>
<td>16</td>
</tr>
<tr>
<td>mask ratio</td>
<td>0.6</td>
</tr>
<tr>
<td>drop path</td>
<td>0.0 (T, S); 0.2 (B, B+, L); 0.3 (H)</td>
</tr>
</tbody>
</table>

(a) Pretraining

<table border="1">
<thead>
<tr>
<th>config</th>
<th>value</th>
</tr>
</thead>
<tbody>
<tr>
<td>optimizer</td>
<td>AdamW</td>
</tr>
<tr>
<td>optimizer momentum</td>
<td><math>\beta_1, \beta_2=0.9, 0.999</math></td>
</tr>
<tr>
<td>weight decay</td>
<td>0.05</td>
</tr>
<tr>
<td>learning rate</td>
<td><math>2e-3</math> (T, S, B); <math>1e-3</math> (B+, L, H)</td>
</tr>
<tr>
<td>learning rate schedule</td>
<td>cosine decay</td>
</tr>
<tr>
<td>warmup epochs</td>
<td>5</td>
</tr>
<tr>
<td>epochs</td>
<td>300 (T); 200 (S); 100 (B, B+);<br/>50 (L, H)</td>
</tr>
<tr>
<td>augmentation</td>
<td>RandAug (9, 0.5)</td>
</tr>
<tr>
<td>batch size</td>
<td>1024</td>
</tr>
<tr>
<td>mixup</td>
<td>0.8</td>
</tr>
<tr>
<td>cutmix</td>
<td>1.0</td>
</tr>
<tr>
<td>label smoothing</td>
<td>0.1</td>
</tr>
<tr>
<td>drop path</td>
<td>0.1 (T, S, B, B+); 0.2 / 0.1 (L);<br/>0.3 (H)</td>
</tr>
<tr>
<td>layer-wise decay</td>
<td>0.65 (T, S); 0.7 (B, B+); 0.9 /<br/>0.85 (L); 0.85 (H)</td>
</tr>
</tbody>
</table>

(b) Finetuning
 Table 14. Settings for ImageNet-1K. Notation: setting corresponding to 400 / 1600 epochs of pretraining.

<table border="1">
<thead>
<tr>
<th>config</th>
<th>values</th>
</tr>
</thead>
<tbody>
<tr>
<td>optimizer</td>
<td>AdamW</td>
</tr>
<tr>
<td>optimizer momentum</td>
<td><math>\beta_1, \beta_2=0.9, 0.999</math></td>
</tr>
<tr>
<td>weight decay</td>
<td>0.1</td>
</tr>
<tr>
<td>learning rate</td>
<td><math>3.5e-5</math> (B, B+), <math>3e-5</math> (L)</td>
</tr>
<tr>
<td>learning rate schedule</td>
<td>step-wise decay</td>
</tr>
<tr>
<td>epochs</td>
<td>100</td>
</tr>
<tr>
<td>augmentation</td>
<td>LSJ [0.1, 2.0]</td>
</tr>
<tr>
<td>batch size</td>
<td>64</td>
</tr>
<tr>
<td>drop path</td>
<td>0.2 (B, B+), 0.4 (L)</td>
</tr>
</tbody>
</table>

 Table 15. Settings for COCO. Hiera finetuning settings on COCO.

attention speed-up mechanism in this paper, though they can be used to further increase speed. For each of the methods, we measure purely the model inference throughput. We compute the throughput with various batch sizes, and report the throughput with the optimal batch size. We use half precision (fp16) to run speed benchmarking unless otherwise specified. We set the input resolution to  $224 \times 224 \times 3$  for image benchmarking, and  $224 \times 224 \times 3$  with 16 frames as a clip for video benchmarking. To measure the training

<table border="1">
<thead>
<tr>
<th>size</th>
<th>ViT</th>
<th>MCMAE</th>
<th>ConvNextV2</th>
<th>Hiera</th>
</tr>
</thead>
<tbody>
<tr>
<td>T</td>
<td>-</td>
<td>-</td>
<td>1381</td>
<td><b>2758</b></td>
</tr>
<tr>
<td>S</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td><b>2211</b></td>
</tr>
<tr>
<td>B</td>
<td>1448</td>
<td>1069</td>
<td>646</td>
<td><b>1556</b></td>
</tr>
<tr>
<td>B+</td>
<td>-</td>
<td>936</td>
<td>-</td>
<td><b>1247</b></td>
</tr>
<tr>
<td>L</td>
<td>514</td>
<td>381</td>
<td>414</td>
<td><b>531</b></td>
</tr>
<tr>
<td>H</td>
<td>205</td>
<td>194</td>
<td>202</td>
<td><b>274</b></td>
</tr>
</tbody>
</table>

 Table 16. Image speed benchmarking on A100 fp16 (im/s).

<table border="1">
<thead>
<tr>
<th>size</th>
<th>ViT</th>
<th>Hiera</th>
</tr>
</thead>
<tbody>
<tr>
<td>B</td>
<td>47.1</td>
<td><b>133.6</b></td>
</tr>
<tr>
<td>B+</td>
<td>-</td>
<td><b>84.1</b></td>
</tr>
<tr>
<td>L</td>
<td>17.8</td>
<td><b>40.8</b></td>
</tr>
<tr>
<td>H</td>
<td>11.3</td>
<td><b>20.9</b></td>
</tr>
</tbody>
</table>

 Table 17. Video speed benchmarking on A100 fp16 (clip/s).

time, we measure the speed of a forward-backward pass on a single gpu and extrapolate the total training time according to the size of the dataset and the number of training epochs, ignoring dataloading and communication overheads when training with multiple GPUs.

We report the image benchmarking results on NVIDIA A100 with fp16 and compared with in ViT (Dosovitskiy et al., 2021), ConvNextV2 (Woo et al., 2023) and MCMAE (Gao et al., 2022) in Tab. 16. We provide video benchmarking results in Tab. 17.

## B. From Scratch Supervised Training

In the main paper, we show that we can replace the spatial biases offered by specialized modules in a hierarchical vision transformer with a strong pretext task like MAE (He et al., 2022), thereby *teaching* these spatial biases instead. This renders these bells-and-whistles unnecessary, and we remove them to construct an extremely fast and accurate vision transformer: Hiera.

However, we *do not* claim that these modules are unnecessary *in general*. In fact, here we intend to show the opposite: the reason these spatial biases were necessary in the first place is because they are required when training a vision transformer from scratch with classification. In Fig. 8, we show this by repeating the ablations in Tab. 1 on ImageNet-1K starting from an MViT-B model and ending at Hiera-B, but this time training on classification *from scratch*.

As expected, we see the *opposite* trend as we did when training with a strong pretext task: the bells-and-whistles *are* necessary when training in a classical supervised setting. This reiterates the fact that, by training with MAE, we are *replacing* the need to explicitly build spatial biases into the network’s architecture itself.

Note that this also has ramifications for downstream tasks:**Figure 8. Training on classification from scratch.** Here we repeat the experiment in Tab. 1 but without MAE pretraining, using MViTv2’s supervised recipe instead. As expected, the bells-and-whistles that Hiera removes are actually *necessary* when training from scratch—hence their introduction in prior work in the first place. Hiera *learns* spatial biases instead.

while prior specialized Vision Transformers like MViT or Swin act like convnets (e.g., you can just use a normal Mask R-CNN (He et al., 2017) head for detection), Hiera *acts like a ViT*. Thus, we recommend using transformer-based solutions for downstream tasks such as ViTDet (Li et al., 2022b) for detection instead of Mask R-CNN.## References

Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lučić, M., and Schmid, C. Vivit: A video vision transformer. In *ICCV*, 2021.

Baevski, A., Hsu, W.-N., Xu, Q., Babu, A., Gu, J., and Auli, M. data2vec: A general framework for self-supervised learning in speech, vision and language. In *ICML*, 2022.

Bao, H., Dong, L., and Wei, F. Beit: Bert pre-training of image transformers. *ICLR*, 2022.

Bertasius, G., Wang, H., and Torresani, L. Is space-time attention all you need for video understanding? In *ICML*, 2021.

Bozic, A., Palafox, P., Thies, J., Dai, A., and Nießner, M. Transformerfusion: Monocular rgb scene reconstruction using transformers. *NeurIPS*, 2021.

Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., and Zagoruyko, S. End-to-end object detection with transformers. In *ECCV*, 2020.

Carreira, J., Noland, E., Banki-Horvath, A., Hillier, C., and Zisserman, A. A short note about kinetics-600. *arXiv preprint arXiv:1808.01340*, 2018.

Carreira, J., Noland, E., Hillier, C., and Zisserman, A. A short note on the kinetics-700 human action dataset. *arXiv preprint arXiv:1907.06987*, 2019.

Chen, M., Radford, A., Child, R., Wu, J., Jun, H., Luan, D., and Sutskever, I. Generative pretraining from pixels. In *ICML*, 2020.

Clark, K., Luong, M.-T., Le, Q. V., and Manning, C. D. ELECTRA: Pre-training text encoders as discriminators rather than generators. In *ICLR*, 2020.

Cubuk, E. D., Zoph, B., Shlens, J., and Le, Q. V. RandAugment: Practical automated data augmentation with a reduced search space. In *CVPR*, 2020.

Dalal, N. and Triggs, B. Histograms of oriented gradients for human detection. In *CVPR*, 2005.

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Ré, C. Flashattention: Fast and memory-efficient exact attention with io-awareness. *arXiv preprint arXiv:2205.14135*, 2022.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In *CVPR*, 2009.

Dong, X., Bao, J., Chen, D., Zhang, W., Yu, N., Yuan, L., Chen, D., and Guo, B. Cswin transformer: A general vision transformer backbone with cross-shaped windows. In *CVPR*, 2022.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al. An image is worth 16x16 words: Transformers for image recognition at scale. In *ICLR*, 2021.

Duke, B., Ahmed, A., Wolf, C., Aarabi, P., and Taylor, G. W. Sstvos: Sparse spatiotemporal transformers for video object segmentation. In *CVPR*, 2021.

El-Noubby, A., Izacard, G., Touvron, H., Laptev, I., Jegou, H., and Grave, E. Are large-scale datasets necessary for self-supervised pre-training? *arXiv preprint arXiv:2112.10740*, 2021.

Fan, H., Xiong, B., Mangalam, K., Li, Y., Yan, Z., Malik, J., and Feichtenhofer, C. Multiscale vision transformers. In *ICCV*, 2021.

Feichtenhofer, C., Fan, H., Malik, J., and He, K. Slowfast networks for video recognition. In *ICCV*, 2019.

Feichtenhofer, C., Fan, H., Li, Y., and He, K. Masked autoencoders as spatiotemporal learners. *NeurIPS*, 2022.

Gao, P., Ma, T., Li, H., Lin, Z., Dai, J., and Qiao, Y. Mcmae: Masked convolution meets masked autoencoders. In *NeurIPS*, 2022.

Ghiasi, G., Cui, Y., Srinivas, A., Qian, R., Lin, T.-Y., Cubuk, E. D., Le, Q. V., and Zoph, B. Simple copy-paste is a strong data augmentation method for instance segmentation. In *CVPR*, 2021.

Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., and He, K. Accurate, large minibatch sgd: Training imagenet in 1 hour. *arXiv preprint arXiv:1706.02677*, 2017a.

Goyal, R., Ebrahimi Kahou, S., Michalski, V., Materzynska, J., Westphal, S., Kim, H., Haenel, V., Fruend, I., Yianilos, P., Mueller-Freitag, M., et al. The “something something” video database for learning and evaluating visual common sense. In *ICCV*, 2017b.

Graham, B., El-Noubby, A., Touvron, H., Stock, P., Joulin, A., Jégou, H., and Douze, M. Levit: a vision transformer in convnet’s clothing for faster inference. In *ICCV*, 2021.

Gu, C., Sun, C., Ross, D. A., Vondrick, C., Pantofaru, C., Li, Y., Vijayanarasimhan, S., Toderici, G., Ricco, S., Sukthankar, R., et al. AVA: A video dataset of spatio-temporally localized atomic visual actions. In *CVPR*, 2018.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In *CVPR*, 2016.He, K., Gkioxari, G., Dollár, P., and Girshick, R. Mask R-CNN. In *ICCV*, 2017.

He, K., Chen, X., Xie, S., Li, Y., Dollár, P., and Girshick, R. Masked autoencoders are scalable vision learners. In *CVPR*, 2022.

Hoffer, E., Ben-Nun, T., Hubara, I., Giladi, N., Hoefler, T., and Soudry, D. Augment your batch: Improving generalization through instance repetition. In *CVPR*, 2020.

Hou, Z., Sun, F., Chen, Y.-K., Xie, Y., and Kung, S.-Y. Milan: Masked image pretraining on language assisted representation. *arXiv preprint arXiv:2208.06049*, 2022.

Huang, G., Sun, Y., Liu, Z., Sedra, D., and Weinberger, K. Q. Deep networks with stochastic depth. In *ECCV*, 2016.

Huang, L., You, S., Zheng, M., Wang, F., Qian, C., and Yamasaki, T. Green hierarchical vision transformer for masked image modeling. In *NeurIPS*, 2022a.

Huang, P.-Y., Xu, H., Li, J., Baevski, A., Auli, M., Galuba, W., Metze, F., and Feichtenhofer, C. Masked autoencoders that listen. *NeurIPS*, 2022b.

Kay, W., Carreira, J., Simonyan, K., Zhang, B., Hillier, C., Vijayanarasimhan, S., Viola, F., Green, T., Back, T., Natsev, P., et al. The kinetics human action video dataset. *arXiv preprint arXiv:1705.06950*, 2017.

Krizhevsky, A., Sutskever, I., and Hinton, G. E. Imagenet classification with deep convolutional neural networks. In *NeurIPS*, 2012.

LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 1998.

Li, X., Wang, W., Yang, L., and Yang, J. Uniform masking: Enabling mae pre-training for pyramid-based vision transformers with locality. *arXiv preprint arXiv:2205.10063*, 2022a.

Li, Y., Mao, H., Girshick, R., and He, K. Exploring plain vision transformer backbones for object detection. In *ECCV*, 2022b.

Li, Y., Wu, C.-Y., Fan, H., Mangalam, K., Xiong, B., Malik, J., and Feichtenhofer, C. Mvitv2: Improved multiscale vision transformers for classification and detection. In *CVPR*, 2022c.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., and Zitnick, C. L. Microsoft COCO: Common objects in context. In *ECCV*, 2014.

Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., and Belongie, S. Feature pyramid networks for object detection. In *CVPR*, 2017.

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. Swin transformer: Hierarchical vision transformer using shifted windows. In *ICCV*, 2021.

Loshchilov, I. and Hutter, F. SGDR: Stochastic gradient descent with warm restarts. In *ICLR*, 2017.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In *ICLR*, 2019.

Misra, I., Girdhar, R., and Joulin, A. An end-to-end transformer model for 3d object detection. In *ICCV*, 2021.

Pathak, D., Krahenbuhl, P., Donahue, J., Darrell, T., and Efros, A. A. Context encoders: Feature learning by inpainting. In *CVPR*, 2016.

Ranftl, R., Bochkovskiy, A., and Koltun, V. Vision transformers for dense prediction. In *CVPR*, 2021.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. Dropout: A simple way to prevent neural networks from overfitting. *JMLR*, 2014.

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., and Wojna, Z. Rethinking the inception architecture for computer vision. In *CVPR*, 2016.

Tong, Z., Song, Y., Wang, J., and Wang, L. VideoMAE: Masked autoencoders are data-efficient learners for self-supervised video pre-training. In *NeurIPS*, 2022.

Touvron, H., Cord, M., and Jégou, H. Deit iii: Revenge of the vit. *ECCV*, 2022.

Van Horn, G., Mac Aodha, O., Song, Y., Cui, Y., Sun, C., Shepard, A., Adam, H., Perona, P., and Belongie, S. The iNaturalist species classification and detection dataset. In *CVPR*, 2018.

Vincent, P., Larochelle, H., Lajoie, I., Bengio, Y., Manzagol, P.-A., and Bottou, L. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. *JMLR*, 2010.

Wang, W., Xie, E., Li, X., Fan, D.-P., Song, K., Liang, D., Lu, T., Luo, P., and Shao, L. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In *ICCV*, 2021.

Wei, C., Fan, H., Xie, S., Wu, C.-Y., Yuille, A., and Feichtenhofer, C. Masked feature prediction for self-supervised visual pre-training. In *CVPR*, 2022.Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon, I. S., and Xie, S. Convnext v2: Co-designing and scaling convnets with masked autoencoders. *arXiv preprint arXiv:2301.00808*, 2023.

Wu, Y., Kirillov, A., Massa, F., Lo, W.-Y., and Girshick, R. Detectron2, 2019.

Xie, Z., Zhang, Z., Cao, Y., Lin, Y., Bao, J., Yao, Z., Dai, Q., and Hu, H. Simmim: A simple framework for masked image modeling. In *CVPR*, 2022.

Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., and Yoo, Y. Cutmix: Regularization strategy to train strong classifiers with localizable features. In *ICCV*, 2019.

Zhai, X., Kolesnikov, A., Houlsby, N., and Beyer, L. Scaling vision transformers. In *CVPR*, 2021.

Zhang, H., Cisse, M., Dauphin, Y. N., and Lopez-Paz, D. mixup: Beyond empirical risk minimization. In *ICLR*, 2018.

Zhou, B., Lapedriza, A., Xiao, J., Torralba, A., and Oliva, A. Learning deep features for scene recognition using places database. In *NeurIPS*, 2014.

