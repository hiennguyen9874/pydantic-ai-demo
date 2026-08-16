# ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth

Shariq Farooq Bhat  
KAUST

Reiner Birkl  
Intel

Diana Wofk  
Intel

Peter Wonka  
KAUST

Matthias Müller  
Intel

Figure 1. **Zero-shot transfer.** Our single multi-domain metric depth estimation model can be applied across domains, indoor or outdoor, simulated or real. **Top:** Input RGB. **Bottom:** Predicted depth. **From left to right:** iBims-1, DIML Outdoor, Hypersim, DIODE Indoor, vKITTI2, SUN-RGBD, DIODE Outdoor and DDAD.

## Abstract

This paper tackles the problem of depth estimation from a single image. Existing work either focuses on generalization performance disregarding metric scale, i.e. relative depth estimation, or state-of-the-art results on specific datasets, i.e. metric depth estimation. We propose the first approach that combines both worlds, leading to a model with excellent generalization performance while maintaining metric scale. Our flagship model, ZoeD-M12-NK, is pre-trained on 12 datasets using relative depth and fine-tuned on two datasets using metric depth. We use a lightweight head with a novel bin adjustment design called metric bins module for each domain. During inference, each input image is automatically routed to the appropriate head using a latent classifier. Our framework admits multiple configurations depending on the datasets used for relative depth pre-training and metric fine-tuning. Without pre-training, we can already significantly improve the state of the art (SOTA) on the NYU Depth v2 indoor dataset. Pre-training on twelve datasets and fine-tuning on the NYU Depth v2 indoor dataset, we can further improve SOTA for a total of 21% in terms of relative absolute error (REL). Finally, ZoeD-M12-NK is the first model that can jointly train on multiple datasets (NYU Depth v2 and KITTI) without a significant drop in performance and achieve unprecedented zero-shot generalization performance to eight unseen datasets from both indoor and outdoor domains.

The code and pre-trained models are publicly available at <https://github.com/isl-org/ZoeDepth>.

## 1. Introduction

Single-image depth estimation (SIDE) is a classic problem in computer vision with many recent contributions. There are two branches of work: metric depth estimation (MDE) and relative depth estimation (RDE). The dominant branch is MDE [5, 6, 26, 30, 50], where the goal is to estimate depth in absolute physical units, i.e. meters. The advantage of predicting metric depth is the practical utility for many downstream applications in computer vision and robotics, such as mapping, planning, navigation, object recognition, 3D reconstruction, and image editing. However, training a single metric depth estimation model across multiple datasets often deteriorates the performance, especially when the collection includes images with large differences in depth scale, e.g. indoor and outdoor images. As a result, current MDE models usually overfit to specific datasets and do not generalize well to other datasets.

The second branch of work, relative depth estimation [30, 33], deals with the large depth scale variation in multiple types of environments by factoring out the scale. As a result, disparity is sufficient for supervision; metric scale and camera parameters are not required and do not need to be consistent across datasets. In RDE, depth predictions per pixel are only consistent relative to each otherFigure 2. **ZoeDepth architecture.** An RGB image is fed into the MiDaS depth estimation framework [33]. The bottleneck and succeeding four hierarchy levels of the MiDaS decoder (at 1/32, 1/16, 1/8, 1/4 and 1/2 of the MiDaS in- and output resolution) are hooked into the metric bins module (see Fig. 3). The metric bins module computes the per-pixel depth bin centers that are linearly combined to output the metric depth. Different transformer backbones can be utilized for the MiDaS encoder; a state-of-the-art example is BEiT<sub>384</sub>-L [3].

across image frames and the scale factor is unknown. This allows methods to be trained on a diverse set of scenes and datasets, even 3D movies [33], enabling model generalizability across domains. The trade-off is that the predicted depth has no metric meaning, limiting the applications.

In this paper, we propose a two-stage framework that combines the two approaches (see Fig. 2). In the first stage, we train a common encoder-decoder architecture for relative depth estimation using the standard training scheme [32]. Our model first learns from a large variety of datasets in pre-training which leads to good generalization. In the second stage, we add heads for metric depth estimation to the encoder-decoder architecture and fine-tune them on metric depth datasets, using one light-weight metric head per domain (a metric head has less than 1% of the parameters of the backbone). During inference, an image is automatically routed to the appropriate head using a classifier on encoder features. Adding these domain-specific heads helps the model learn metric depth while benefiting from the relative depth pre-training. Our metric head design (dubbed metric bins module) is inspired by a recently introduced method for metric depth estimation [6] that estimates a set of depth values instead of a single depth value per pixel. Similarly, we estimate a set of depth values (bins) and subsequently transform this estimation at each layer of the decoder using a novel concept we call *attractors*.

Our framework is flexible and can be used in multiple different configurations. We specifically want to highlight three configurations, that improve the state-of-the-art (SOTA) in different categories of metric SIDE. These three configurations are the main contributions of this paper.

**Metric SIDE.** Without any relative pre-training, our model *ZoeD-X-N* is trained only on NYU Depth v2 [37]. This configuration validates the design of our metric bins module and demonstrates that it can already improve upon the current SOTA NeWCRFs [50] by 13.7% on indoor depth estimation without relative depth pre-training.

**Metric SIDE with relative depth pre-training.** By conducting relative depth pre-training on 12 datasets and then conducting metric fine tuning on NYU Depth v2, our model *ZoeD-M12-N* can further improve on *ZoeD-X-N* by 8.5%, leading to 21% improvement over current published SOTA. Existing architectures do not have an established way to benefit from relative depth pre-training at a competitive level.

**Universal Metric SIDE with automatic routing.** We make a step towards universal depth estimation in the wild. Our flagship architecture *ZoeD-M12-NK* uses relative pre-training on 12 datasets combined with metric fine-tuning on indoor and outdoor datasets, *i.e.* NYU Depth v2 and KITTI, jointly. We evaluate this setup by first showing that it significantly outperforms SOTA on datasets it was trained on (NYU and KITTI) when compared to other models that are also trained on these two datasets jointly; we achieve an overall improvement in absolute relative error (REL) of 24.3%. Second, our setup outperforms SOTA on 7 metric datasets it was not trained on, with up to 976.4% (~11x) improvement in metrics; this demonstrates its unprecedented zero-shot capabilities.

## 2. Related Work

### 2.1. Single-Image Depth Estimation (SIDE)

Supervised single-image depth estimation methods can be categorized into regressing metric depth [5, 6, 16, 26, 30, 50] and relative depth [21, 30, 32, 33]. Metric depth models are typically trained on singular datasets, are more prone to overfitting, and typically generalize poorly to unseen environments or across varying depth ranges. Relative depth models tend to generalize better as they can be trained on more diverse datasets with relative depth annotations using scale-invariant losses. Yet, their utility for downstream tasks requiring metric depth is limited, as relative depth models regress depth with unknown scale and shift. Recent works have sought to resolve metric information inregressed depth. For example, Yin *et al.* [49] recover 3D scene shape from a single image via a two-stage framework combining monocular depth estimation with 3D point cloud encoders that are trained to predict missing depth shift and focal length. Jun *et al.* [16] decompose metric depth into normalized depth and scale features and propose a multi-decoder network where a metric depth decoder leverages relative depth features from the gradient and normalized depth decoders. Universal depth prediction has also been investigated by the Robust Vision Challenge<sup>1</sup> that includes indoor and outdoor domains. A popular idea is to discretize the target depth interval and reformulate the continuous depth regression as a classification task [11, 23, 24, 34]. Ren *et al.* [34] propose a two-stage framework: first involving training a classifier to distinguish low-depth-range and high-depth-range images. Two separate networks are then trained for the respective depth ranges. We compare to the best publicly available model based on DORN [11].

## 2.2. Distribution learning for metric SIDE

Many conventional learning-based monocular depth estimation methods adopt encoder-decoder architectures with convolutional layers, and more recently, transformer blocks. Depth estimation is commonly treated as a per-pixel regression task. An evolving line of work seeks to reformulate depth estimation as a combined classification-regression problem that reasons about distributions of depth values across an image. AdaBins [5] extends standard encoder-decoder backbones with a transformer-based module that discretizes predicted depth ranges into bins, where bin widths are determined adaptively per image; the final depth estimation is computed as a linear combination of bin centers. LocalBins [6] builds on this concept by considering depth distributions within local neighborhoods of a given pixel instead of globally over the image, as well as computing bin embeddings in a multi-scale fashion across decoder layers. PixelBins [36] simplifies AdaBins by replacing transformer block with convolutions, reducing complexity. BinsFormer [26] incorporates an auxiliary scene classification query to guide bin generation and also utilizes a multi-scale strategy to refine adaptively-generated bins. PixelFormer [2] treats depth estimation as pixel queries that are refined via skip attention and that are used to predict bin centers without leveraging decoded features.

## 3. Methodology

In this section, we describe our architecture, design choices and training protocol in detail.

<sup>1</sup><http://www.robustvision.net/>

## 3.1. Overview

We use the MiDaS [33] training strategy for relative depth prediction. MiDaS uses a loss that is invariant to scale and shift. If multiple datasets are available, a multi-task loss that ensures pareto-optimality across the datasets is used. The MiDaS training strategy can be applied to many different network architectures. We use the DPT encoder-decoder architecture as our base model [32], but replace the encoder with more recent transformer-based backbones [3]. After pre-training the MiDaS model for relative depth prediction, we add one or more heads for metric depth estimation by attaching our proposed *metric bins module* to the decoder (see Fig. 2 for the overall architecture). The metric bins module outputs metric depth and follows the adaptive binning principle, originally introduced in [5] and subsequently modified by [2, 6, 26, 36]. In particular, we start out with the pixel-wise prediction design as in LocalBins [6] and propose modifications that further improve performance. Finally, we fine-tune the complete architecture end-to-end.

## 3.2. Architecture Details

We first review LocalBins, and then introduce our novel metric bins module with *attractor layers*, our bin aggregation strategy, and loss function.

**LocalBins review.** Our metric bins module is inspired by the LocalBins architecture proposed in [6]. LocalBins uses a standard encoder-decoder as the base model and attaches a module that takes the multi-scale features from the encoder-decoder as input and predicts the bin centers at every pixel. Final depth at a pixel is obtained by a linear combination of the bin centers weighted by the corresponding predicted probabilities. The LocalBins module first predicts  $N_{seed}$  different seed bins at each pixel position at the bottleneck. Each bin is then split into two at every decoder layer using splitter MLPs. The number of bin centers is doubled at every decoder layer and we end up with  $2^n N_{seed}$  bins at each pixel at the end of  $n$  decoder layers. Simultaneously, the probability scores (**p**) over  $N_{total} = 2^n N_{seed}$  bin centers (**c**) are predicted from the decoder features using softmax and the final depth at pixel  $i$  is obtained using:

$$d(i) = \sum_{k=1}^{N_{total}} p_i(k) c_i(k) \quad (1)$$

**Metric bins module.** The metric bins module takes multi-scale features from the MiDaS decoder as input and predicts the bin centers to be used for metric depth prediction (see Fig. 3). However, instead of starting with a small number of bins at the bottleneck and splitting them later, our metric bins module predicts all the bin centers at the bottleneck and adjusts them at subsequent decoder layers. This binFigure 3. **Metric Bins Module.** Five incoming channels, corresponding to different depth hierarchies (see Fig. 2), are converted to 1-dimensional bin embeddings (green boxes) by MLPs, in combination with upsampling and addition operations. The lowest bin embedding yields metric bin centers (blue, vertical lines; not representative of actual number 64), whereas the remaining embeddings provide attractors for their respective hierarchy levels (green dots). Going upwards in the metric bins module, the attractors pull the bin centers according to Eqs. (2) and (3).

adjustment is implemented via our newly proposed building block, called *attractor layers*.

**Attract instead of split.** LocalBins implements multi-scale refinement of the bins by splitting them conditioned on the multi-scale features. In contrast, we implement the multi-scale refinement of the bins by adjusting them, moving them left or right on the depth interval. Using the multi-scale features, we predict a set of points on the depth interval towards which the bin centers get attracted. More specifically, at the  $l^{th}$  decoder layer, an MLP takes the features at a pixel as input and predicts  $n_a$  attractor points  $\{a_k : k = 1, \dots, n_a\}$  for that pixel position. The adjusted bin center is  $c'_i = c_i + \Delta c_i$ , with the adjustment given by:

$$\Delta c_i = \sum_{k=1}^{n_a} \frac{a_k - c_i}{1 + \alpha |a_k - c_i|^\gamma} \quad (2)$$

where the hyperparameters  $\alpha$  and  $\gamma$  determine the attractor strength. We name this attractor variant *inverse attractor*. We also experiment with an exponential variant given by:

$$\Delta c_i = \sum_{k=1}^{n_a} (a_k - c_i) e^{-\alpha |a_k - c_i|^\gamma} \quad (3)$$

Our experiments suggest that the *inverse attractor* leads to better performance. We let the number of attractor points vary from one decoder layer to another, denoted together as a set  $\{n_a^l\}$ . We use  $N_{total} = 64$  bins and  $\{16, 8, 4, 1\}$  attractors. Please refer to Sec. 5.4 for various ablations.

The attracting strategy is preferred because it's a con-

tracting process while splitting is inherently dilative. Splitting adds extra constraints of newly produced bins summing up to the original bin width, while attractors adjust freely without such local constraints (only the total width is invariant). Intuitively, the prediction should get more refined and focused with decoder layers, which attractors achieve without dealing with any local constraints.

**Log-binomial instead of softmax.** To get the final metric depth prediction, the bin centers are linearly combined, weighted by their probability scores as per Eq. (1). Prior adaptive bins based models [2, 5, 6, 26] use a softmax to predict the probability distribution over the bin centers. The choice of softmax is mainly inspired from the discrete classification analogy. Although the softmax plays well with unordered classes, since the bins are inherently ordered, it intuitively makes sense to use an ordering-aware prediction of the probabilities. The softmax approach can result in vastly different probabilities for nearby bin centers ( $|p_i - p_{i+1}| \gg 0$ ). Inspired by Beckham and Pal [4], we use a binomial distribution instead to address this issue and correctly consider ordinal relationships between bins.

The binomial distribution has one parameter  $q$  which controls the placement of the mode. We concatenate the relative depth predictions with the decoder features and predict a 2-channel output ( $q$  - mode and  $t$  - temperature) from the decoder features to get the probability score over the  $k^{th}$  bin center by:

$$p(k; N, q) = \binom{N}{k} q^k (1 - q)^{N-k} \quad (4)$$

where  $N = N_{total}$  is the total number of bins. In practice, since we use large values of  $N$ , we take  $\log(p)$ , use Stirling's approximation [1] for factorials and apply softmax( $\{\log(p_k)/t\}_{k=1}^N$ ) to get normalized scores for numerical stability. The parameter  $t$  controls the temperature of the resulting distribution. The softmax normalization preserves the unimodality of the logits. Finally, the resulting probability scores and the bin centers from the metric bins module are used to obtain the final depth as per Eq. (1).

**Loss.** We use the scale-invariant log loss ( $\mathcal{L}_{pixel}$ ) for pixel-level supervision as in LocalBins [6]. Unlike LocalBins, we do not use the chamfer loss for bins due to the high memory requirement but only limited improvement.

### 3.3. Training strategies

As described previously, we have two stages for training: relative depth pre-training for the MiDaS backbone and metric depth fine-tuning for the prediction heads. We compare models with and without pre-training for relative depth as in [33]. We also explore different variations of fine-tuning, using a single dataset and multiple datasets; inthe case of multiple datasets, we also compare using a single head, *i.e.* metric bins module, to using multiple heads. Please refer to Sec. 4.2 for more details about the exact model definitions. In the supplement, we report results for additional variations.

**Metric fine-tuning on multiple datasets** Training a metric depth model on a mixture of datasets with a wide variety of scenes, for example from indoor and outdoor domains, is hard. The model not only has to handle images taken with different cameras and camera settings but also has to learn to adjust for the large variations in the overall scale of the scenes. Indoor scenes are usually limited to a maximum depth of 10 meters while outdoor scenes can have infinite depth (capped at 80 meters in most prior works). We hypothesize that a backbone pre-trained for relative depth estimation, alleviates the issues of fine-tuning on multiple datasets to some extent. We can also equip the model with multiple metric bins modules, one for each scene type (indoor versus outdoor). Different metric heads can be thought of as scene-type experts. Note that the base model is still common to all metric heads; the complete model with multiple heads is trained end-to-end. See Sec. 5.2 for a comparison of our model with single head and multiple heads.

**Routing to metric heads.** When the model has multiple metric heads, we need a router that chooses the metric head to use for a particular input. We employ commonly used routing mechanisms developed in other contexts, *e.g.*, see Fedus *et al.* [10] for a review. We explore three main variants: (R.1) *Labeled Router*: In this variant, we provide scene type labels (indoor or outdoor) to the model at both training and inference times and manually map from the scene type to the metric head. (R.2) *Trained Router*: Here, we train a classifier MLP that predicts the scene type of the input image based on the bottleneck features and then routes to the corresponding metric head. Therefore, this variant only needs scene-type labels during training. (R.3) *Auto Router*: In this setting, a router MLP (equivalent to a classifier in R.2) is used, but no labels are provided during either training or inference. Both the trainable router types, *Trained Router* and *Auto Router*, are trained end-to-end with the whole model. See Sec. 5.4 for a performance comparison of the discussed routing mechanisms.

## 4. Experimental Setup

### 4.1. Datasets

Our primary datasets for training ZoeDepth are NYU Depth v2 (N) for indoor and KITTI (K) for outdoor scenes. We refer to the combination of both datasets as (NK). For pre-training the relative depth backbone, we train on a mix of 12 datasets (M12) consisting of

the 10 datasets used in [32]: HRWSI [46], BlendMVS [47], ReDWeb [45], DIML-Indoor [17], 3D Movies [33], MegaDepth [25], WSVD [41], TartanAir [43], ApolloScape [15] and IRS [42], plus 2 additional datasets: KITTI [29] and NYU Depth v2 [37].

To demonstrate generalizability, we evaluate zero-shot performance on a number of real-world and synthetic datasets: SUN RGB-D [38], iBims [18], DIODE Indoor [40] and HyperSim [35] for the indoor domain; DDAD [12], DIML Outdoor [17], DIODE Outdoor [40] and Virtual KITTI 2 [7] for the outdoor domain. We provide further details about the datasets in the supplement.

### 4.2. Models

The models are named according to the following convention:  $ZoeD-\{RDPT\}-\{MFT\}$ , where  $ZoeD$  is the abbreviation for ZoeDepth,  $RDPT$  denotes the datasets used for relative depth pre-training (X denotes no pre-training) and  $MFT$  denotes the datasets used for metric depth fine-tuning. We train and evaluate the following models:  $ZoeD-X-N$ ,  $ZoeD-X-K$ ,  $ZoeD-M12-N$ ,  $ZoeD-M12-K$  and  $ZoeD-M12-NK$ . All models use the BEiT<sub>384</sub>-L backbone from timm [44] that was pre-trained on ImageNet. The models  $ZoeD-X-N$  and  $ZoeD-X-K$  are directly fine-tuned for metric depth on NYU Depth v2 and KITTI respectively without any pre-training for relative depth estimation.  $ZoeD-M12-N$  and  $ZoeD-M12-K$  additionally include pre-training for relative depth estimation on the M12 dataset mix before the fine-tuning stage for metric depth.  $ZoeD-M12-NK$  is also pre-trained on M12, but has two separate heads fine-tuned on both NYU Depth v2 and KITTI.  $ZoeD-M12-NK^\dagger$  is a variant of this model with a single head but otherwise the same pre-training and fine-tuning procedure. In the supplement, we provide further results for models trained on additional dataset combinations in pre-training and fine-tuning.

### 4.3. Evaluation Metrics

We evaluate in metric depth space  $\mathbf{d}$  by computing the absolute relative error (REL) =  $\frac{1}{M} \sum_{i=1}^M |\mathbf{d}_i - \hat{\mathbf{d}}_i|/\mathbf{d}_i$ , the root mean squared error (RMSE) =  $[\frac{1}{M} \sum_{i=1}^M |\mathbf{d}_i - \hat{\mathbf{d}}_i|^2]^{\frac{1}{2}}$ , the average  $\log_{10}$  error =  $\frac{1}{M} \sum_{i=1}^M |\log_{10} \mathbf{d}_i - \log_{10} \hat{\mathbf{d}}_i|$ , and the threshold accuracy  $\delta_n = \%$  of pixels s.t.  $\max(\mathbf{d}_i/\hat{\mathbf{d}}_i, \hat{\mathbf{d}}_i/\mathbf{d}_i) < 1.25^n$  for  $n = 1, 2, 3$ , where  $\mathbf{d}_i$  and  $\hat{\mathbf{d}}_i$  refer to ground truth and predicted depth at pixel  $i$ , respectively, and  $M$  is the total number of pixels in the image. We cap the evaluation depth at 10m for indoor datasets (8m for SUN RGB-D) and at 80m for outdoor datasets. Final model outputs are computed as the average of an image’s depth prediction and the prediction of its mirror image and are evaluated at ground truth resolution.

In addition, we define two metrics to measure relative improvement (RI) across datasets and metrics respectively. For  $M$  datasets  $D_i$  with  $i \in [1, M]$ , we compute  $mRI_D =$Figure 4. **Qualitative comparison on NYU Depth v2.** Our method consistently produces better predictions with much less error. When looking closely at the depth maps it can also be observed that our predictions are much sharper with clear edges.  $\Delta$  indicates square error ranging from lowest (dark blue) to highest (dark red) across predictions. Invalid regions are indicated as grey.

$\frac{1}{M} \sum_{i=1}^M \text{RI}_{D_i}$ . Similarly, for  $N$  metrics  $\theta_j$  with  $j \in [1, N]$ , we compute  $\text{mRI}_\theta = \frac{1}{N} \sum_{j=1}^N \text{RI}_{\theta_j}$ . For metrics where lower is better,  $\text{RI} = \frac{r-t}{r}$  and for metrics where higher is better  $\text{RI} = \frac{t-r}{r}$ , where  $r$  and  $t$  correspond to the reference and target scores respectively.

## 5. Results

### 5.1. Comparison to SOTA on NYU Depth V2

Our novel architecture beats SOTA without using any additional data for pre-training. To demonstrate this, we evaluate our model *ZoeD-X-N* on the popular metric depth estimation benchmark NYU Depth v2 [37]. *ZoeD-X-N* is

not pre-trained for relative depth; the backbone is initialized with the standard weights from ImageNet pre-training. Tab. 1 shows the performance of models on the official NYU Depth v2 test set. This model already outperforms NeWCRFs [50] by 13.7% (REL = 0.082), highlighting the contribution of our architecture design.

Next, we verify that our architecture can benefit from relative depth pre-training. Our corresponding model *ZoeD-M12-N* significantly outperforms the prior state-of-the-art NeWCRFs [50] by nearly 21% (REL = 0.075). Results are not just numerically better; the resulting depth maps also have much sharper boundaries (see Fig. 4). We believe this is the first demonstration of successful relative depth pre-training at a competitive level. While other architectures can also benefit from pre-training, some modifications are required. In the next section, we show one such modification by combining our base model with architecture building blocks proposed by other papers (see Tab. 2). This shows that while other architectures benefit from our larger backbone and relative depth pre-training, they are still not competitive with our complete framework.

### 5.2. Universal Metric SIDE

Here, we evaluate our progress towards a universal metric depth estimation framework by analyzing our model *ZoeD-M12-NK* which was trained across two different metric datasets and generalizes across indoor and outdoor domains. Models trained across multiple metric datasets usually perform worse or diverge. In contrast, our model *ZoeD-M12-NK* still outperforms the previous SOTA NeWCRFs [50] on NYU Depth v2 by 18.9% (REL = 0.077, Tab. 1). While *ZoeD-M12-NK* is not as good as our model (*ZoeD-M12-N*) fine-tuned only on NYU Depth v2, it provides a very attractive trade-off between performance and

<table border="1">
<thead>
<tr>
<th>Method</th>
<th><math>\delta_1 \uparrow</math></th>
<th><math>\delta_2 \uparrow</math></th>
<th><math>\delta_3 \uparrow</math></th>
<th>REL <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th><math>\log_{10} \downarrow</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>Eigen <i>et al.</i> [9]</td>
<td>0.769</td>
<td>0.950</td>
<td>0.988</td>
<td>0.158</td>
<td>0.641</td>
<td>–</td>
</tr>
<tr>
<td>Laina <i>et al.</i> [19]</td>
<td>0.811</td>
<td>0.953</td>
<td>0.988</td>
<td>0.127</td>
<td>0.573</td>
<td>0.055</td>
</tr>
<tr>
<td>Hao <i>et al.</i> [13]</td>
<td>0.841</td>
<td>0.966</td>
<td>0.991</td>
<td>0.127</td>
<td>0.555</td>
<td>0.053</td>
</tr>
<tr>
<td>DORN [11]</td>
<td>0.828</td>
<td>0.965</td>
<td>0.992</td>
<td>0.115</td>
<td>0.509</td>
<td>0.051</td>
</tr>
<tr>
<td>SharpNet [31]</td>
<td>0.836</td>
<td>0.966</td>
<td>0.993</td>
<td>0.139</td>
<td>0.502</td>
<td>0.047</td>
</tr>
<tr>
<td>Hu <i>et al.</i> [14]</td>
<td>0.866</td>
<td>0.975</td>
<td>0.993</td>
<td>0.115</td>
<td>0.530</td>
<td>0.050</td>
</tr>
<tr>
<td>Lee <i>et al.</i> [22]</td>
<td>0.837</td>
<td>0.971</td>
<td>0.994</td>
<td>0.131</td>
<td>0.538</td>
<td>–</td>
</tr>
<tr>
<td>Chen <i>et al.</i> [8]</td>
<td>0.878</td>
<td>0.977</td>
<td>0.994</td>
<td>0.111</td>
<td>0.514</td>
<td>0.048</td>
</tr>
<tr>
<td>BTS [20]</td>
<td>0.885</td>
<td>0.978</td>
<td>0.994</td>
<td>0.110</td>
<td>0.392</td>
<td>0.047</td>
</tr>
<tr>
<td>Yin <i>et al.</i> [48]</td>
<td>0.875</td>
<td>0.976</td>
<td>0.994</td>
<td>0.108</td>
<td>0.416</td>
<td>0.048</td>
</tr>
<tr>
<td>AdaBins [5]</td>
<td>0.903</td>
<td>0.984</td>
<td>0.997</td>
<td>0.103</td>
<td>0.364</td>
<td>0.044</td>
</tr>
<tr>
<td>LocalBins [6]</td>
<td>0.907</td>
<td>0.987</td>
<td><u>0.998</u></td>
<td>0.099</td>
<td>0.357</td>
<td>0.042</td>
</tr>
<tr>
<td>Jun <i>et al.</i> [16]</td>
<td>0.913</td>
<td>0.987</td>
<td><u>0.998</u></td>
<td>0.098</td>
<td>0.355</td>
<td>0.042</td>
</tr>
<tr>
<td>NeWCRFs [50]</td>
<td>0.922</td>
<td>0.992</td>
<td><u>0.998</u></td>
<td>0.095</td>
<td>0.334</td>
<td>0.041</td>
</tr>
<tr>
<td><b>ZoeD-X-N</b></td>
<td>0.946</td>
<td><u>0.994</u></td>
<td><b>0.999</b></td>
<td>0.082</td>
<td>0.294</td>
<td>0.035</td>
</tr>
<tr>
<td><b>ZoeD-M12-N</b></td>
<td><b>0.955</b></td>
<td><b>0.995</b></td>
<td><b>0.999</b></td>
<td><b>0.075</b></td>
<td><b>0.270</b></td>
<td><b>0.032</b></td>
</tr>
<tr>
<td><b>ZoeD-M12-NK</b></td>
<td><u>0.953</u></td>
<td><b>0.995</b></td>
<td><b>0.999</b></td>
<td><u>0.077</u></td>
<td><u>0.277</u></td>
<td><u>0.033</u></td>
</tr>
</tbody>
</table>

Table 1. **Quantitative comparison on NYU-Depth v2.** The reported numbers of prior art are from the corresponding original papers. Best results are in bold, second best are underlined.<table border="1">
<thead>
<tr>
<th>Method</th>
<th>NYU</th>
<th>KITTI</th>
<th>iBims-1</th>
<th>vKITTI-2</th>
<th>mRI<sub>D</sub></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6"><b>Baselines: no modification</b></td>
</tr>
<tr>
<td>DORN-X-NK<sup>†</sup></td>
<td>0.156</td>
<td>0.115</td>
<td>0.287</td>
<td>0.259</td>
<td>-45.7%</td>
</tr>
<tr>
<td>LocalBins-X-NK<sup>†</sup></td>
<td>0.245</td>
<td>0.133</td>
<td>0.296</td>
<td>0.265</td>
<td>-74.0%</td>
</tr>
<tr>
<td>PixelBins-X-NK<sup>†</sup></td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>NeWCRFs-X-NK<sup>†</sup></td>
<td>0.109</td>
<td>0.076</td>
<td>0.189</td>
<td>0.190</td>
<td>0.0%</td>
</tr>
<tr>
<td colspan="6"><b>Baselines: modified to use our pre-trained DPT-BEiT-L as backbone</b></td>
</tr>
<tr>
<td>DORN-M12-NK<sup>†</sup></td>
<td>0.110</td>
<td>0.081</td>
<td>0.242</td>
<td>0.215</td>
<td>-12.2%</td>
</tr>
<tr>
<td>LocalBins-M12-NK<sup>†</sup></td>
<td>0.086</td>
<td>0.071</td>
<td>0.221</td>
<td>0.121</td>
<td>11.8%</td>
</tr>
<tr>
<td>PixelBins-M12-NK<sup>†</sup></td>
<td>0.088</td>
<td>0.071</td>
<td>0.232</td>
<td>0.119</td>
<td>10.1%</td>
</tr>
<tr>
<td>NeWCRFs-M12-NK<sup>†</sup></td>
<td>0.088</td>
<td>0.073</td>
<td>0.233</td>
<td>0.124</td>
<td>8.7%</td>
</tr>
<tr>
<td colspan="6"><b>Ours: different configurations for fair comparison</b></td>
</tr>
<tr>
<td>ZoeD-X-NK<sup>†</sup></td>
<td>0.095</td>
<td>0.074</td>
<td><u>0.187</u></td>
<td>0.184</td>
<td>4.9%</td>
</tr>
<tr>
<td>ZoeD-M12-NK<sup>†</sup></td>
<td><u>0.081</u></td>
<td><u>0.061</u></td>
<td>0.210</td>
<td><u>0.112</u></td>
<td>18.8%</td>
</tr>
<tr>
<td><b>ZoeD-M12-NK</b></td>
<td><b>0.077</b></td>
<td><b>0.057</b></td>
<td><b>0.186</b></td>
<td><b>0.105</b></td>
<td><b>25.2%</b></td>
</tr>
</tbody>
</table>

Table 2. **Comparison with existing works when trained on NYU and KITTI.** Results are reported using the REL metric. The mRI<sub>D</sub> column denotes the mean relative improvement with respect to NeWCRFs across datasets. X in the model name, means no architecture change and no pre-training. M12 means that the model was pre-trained (using our base model based on the DPT architecture with the BEiT-L encoder). All models are fine-tuned on NYU and KITTI. † denotes a single metric head (shared); single-head training allows us to adapt prior models without major changes. Best results are in bold, second best are underlined. PixelBins [36] did not converge without modification. We also tried to train AdaBins [5] across both datasets, but despite our best effort and extensive hyperparameter tuning, it did not converge.

generalization across domains.

To underline the difficulty of cross-domain training, we perform a comparison to other models trained simultaneously on indoor and outdoor datasets (NK). First, we evaluate recent methods trained on NK without any architectural modifications and compare them with our method in Tab. 2. We find that existing works are unable to achieve competitive results in that setting. AdaBins [5] and PixelBins [36] fail to converge at all, while the SOTA NeWCRFs’ [50] performance degrades by nearly 15% (REL 0.095 to 0.109) on NYU (compare Tab. 2 with Tab. 1). These experiments confirm that previous models significantly degrade when being trained jointly on datasets from different domains. In contrast, we only observe an 8% drop (REL 0.075 to 0.081) while using the shared head, demonstrating our model’s ability to deal with different domains at once. This gap is further reduced to mere 2.6% (REL 0.075 vs 0.077) using our two-head model *ZoeD-M12-NK*, outperforming NeWCRFs [50] by 25.2% (mRI<sub>D</sub> in REL).

We conclude that previous models require changes to successfully train on multiple datasets. We conduct additional experiments where we improve previous models by incorporating part of our framework. Specifically, we use the same base model (DPT with a BEiT-L backbone), with relative pre-training on M12, and with fine-tuning on

NK (NYU and KITTI mixture) using a single metric head; only the design of the metric head varies. Tab. 2 shows that previous models still fall behind in this setting. Since DORN [11] and LocalBins [6] are light-weight and modular, they can be easily used in conjunction with our pre-trained relative depth model instead of our metric bins module. NeWCRFs [50] is originally a tightly-coupled decoder-focused design; however, to keep the base model exactly the same, we use an extra head with NeWCRFs layers that use DPT decoder features as multi-scale input. This increases the complexity significantly (~40M more parameters than ours) yet still underperforms when compared to pixel-wise bins-based methods: LocalBins, PixelBins and *ZoeD-M12-NK*. This suggests that bins-based architectures are better suited for multi-domain training and can better exploit relative depth pre-training. Our model performs best both on NYU and KITTI, as well as on iBims-1 and virtual KITTI-2 that have not been seen in training. These results indicate that our metric bins module exploits pre-training better than existing works, enabling improved domain adaptation and generalization (zero-shot performance). We investigate zero-shot performance in more detail next.

### 5.3. Zero-shot Generalization

We evaluate the generalization capabilities of our approach by comparing its zero-shot performance to prior works on eight unseen indoor and outdoor datasets without fine-tuning; we show qualitative results in Fig. 1 and report quantitative results in Tab. 3 and Tab. 4.

Tab. 3 reports zero-shot generalization on indoor datasets. Even with fine-tuning across both the indoor (NYU Depth v2) and outdoor (KITTI) domains, our model *ZoeD-M12-NK* demonstrates significantly better performance than previous state-of-the-art models. The mean relative improvement (mRI<sub>θ</sub>) ranges from 5.3% for HyperSim to 46.3% for DIODE Indoor. As expected, fine-tuning only on NYU Depth v2 so that the training and test domains are both indoor, *i.e.* *ZoeD-M12-N*, leads to an increase in mRI<sub>θ</sub> on all datasets. *ZoeD-X-N* scores lower in most datasets due to the lack of relative depth pre-training.

Tab. 4 reports zero-shot generalization on outdoor datasets. Similar as for the indoor datasets, pre-training on M12 is generally beneficial. *ZoeD-M12-NK* improves from 7.8% for Virtual KITTI 2 to 976.4% for DIML Outdoor over NeWCRFs [50]. On DDAD, *ZoeD-M12-NK* performs 12.8% worse while NeWCRFs [50] is best. The metrics in Tab. 4 and the rightmost image in Fig. 1 show the quality of our results. Overall, our framework is the top performer in 7 out of 8 datasets.

Probably the most interesting result is the high mRI<sub>θ</sub> value of 976.4% that *ZoeD-M12-NK* achieves on DIML Outdoor. All other models are fine-tuned only on KITTI with large depth ranges but the DIML Outdoor dataset con-<table border="1">
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="4">SUN RGB-D</th>
<th colspan="4">iBims-1 Benchmark</th>
<th colspan="4">DIODE Indoor</th>
<th colspan="4">HyperSim</th>
</tr>
<tr>
<th><math>\delta_1 \uparrow</math></th>
<th>REL <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th>mRI<math>_{\theta} \uparrow</math></th>
<th><math>\delta_1 \uparrow</math></th>
<th>REL <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th>mRI<math>_{\theta} \uparrow</math></th>
<th><math>\delta_1 \uparrow</math></th>
<th>REL <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th>mRI<math>_{\theta} \uparrow</math></th>
<th><math>\delta_1 \uparrow</math></th>
<th>REL <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th>mRI<math>_{\theta} \uparrow</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>BTS [20]</td>
<td>0.740</td>
<td>0.172</td>
<td>0.515</td>
<td>-14.2%</td>
<td>0.538</td>
<td>0.231</td>
<td>0.919</td>
<td>-6.9%</td>
<td>0.210</td>
<td>0.418</td>
<td>1.905</td>
<td>2.3%</td>
<td>0.225</td>
<td>0.476</td>
<td>6.404</td>
<td>-8.6%</td>
</tr>
<tr>
<td>AdaBins [5]</td>
<td>0.771</td>
<td>0.159</td>
<td>0.476</td>
<td>-7.0%</td>
<td>0.555</td>
<td>0.212</td>
<td>0.901</td>
<td>-2.1%</td>
<td>0.174</td>
<td>0.443</td>
<td>1.963</td>
<td>-7.2%</td>
<td>0.221</td>
<td>0.483</td>
<td>6.546</td>
<td>-10.5%</td>
</tr>
<tr>
<td>LocalBins [6]</td>
<td>0.777</td>
<td>0.156</td>
<td>0.470</td>
<td>-5.6%</td>
<td>0.558</td>
<td>0.211</td>
<td>0.880</td>
<td>-0.7%</td>
<td>0.229</td>
<td>0.412</td>
<td>1.853</td>
<td>7.1%</td>
<td>0.234</td>
<td>0.468</td>
<td>6.362</td>
<td>-6.6%</td>
</tr>
<tr>
<td>NeWCRFs [50]</td>
<td>0.798</td>
<td>0.151</td>
<td>0.424</td>
<td>0.0%</td>
<td>0.548</td>
<td>0.206</td>
<td>0.861</td>
<td>0.0%</td>
<td>0.187</td>
<td>0.404</td>
<td>1.867</td>
<td>0.0%</td>
<td>0.255</td>
<td>0.442</td>
<td>6.017</td>
<td>0.0%</td>
</tr>
<tr>
<td><b>ZoeD-X-N</b></td>
<td>0.857</td>
<td>0.124</td>
<td>0.363</td>
<td>13.2%</td>
<td><b>0.668</b></td>
<td>0.173</td>
<td>0.730</td>
<td>17.7%</td>
<td><b>0.400</b></td>
<td><b>0.324</b></td>
<td><b>1.581</b></td>
<td><b>49.7%</b></td>
<td>0.284</td>
<td>0.421</td>
<td>5.889</td>
<td>6.1%</td>
</tr>
<tr>
<td><b>ZoeD-M12-N</b></td>
<td><b>0.864</b></td>
<td><b>0.119</b></td>
<td><b>0.346</b></td>
<td><b>16.0%</b></td>
<td><u>0.658</u></td>
<td><b>0.169</b></td>
<td><b>0.711</b></td>
<td><b>18.5%</b></td>
<td>0.376</td>
<td><u>0.327</u></td>
<td><u>1.588</u></td>
<td>45.0%</td>
<td><b>0.292</b></td>
<td><b>0.410</b></td>
<td><b>5.771</b></td>
<td><b>8.6%</b></td>
</tr>
<tr>
<td><b>ZoeD-M12-NK</b></td>
<td>0.856</td>
<td><u>0.123</u></td>
<td><u>0.356</u></td>
<td><u>13.9%</u></td>
<td>0.615</td>
<td>0.186</td>
<td>0.777</td>
<td>10.6%</td>
<td><u>0.386</u></td>
<td>0.331</td>
<td>1.598</td>
<td><u>46.3%</u></td>
<td>0.274</td>
<td><u>0.419</u></td>
<td><u>5.830</u></td>
<td>5.3%</td>
</tr>
</tbody>
</table>

Table 3. Quantitative results for zero-shot transfer to four unseen indoor datasets. mRI $_{\theta}$  denotes the mean relative improvement with respect to NeWCRFs across all metrics ( $\delta_1$ , REL, RMSE). Evaluation depth is capped at 8m for SUN RGB-D, 10m for iBims and DIODE Indoor, and 80m for HyperSim. Best results are in bold, second best are underlined.

<table border="1">
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="4">Virtual KITTI 2</th>
<th colspan="4">DDAD</th>
<th colspan="4">DIML Outdoor</th>
<th colspan="4">DIODE Outdoor</th>
</tr>
<tr>
<th><math>\delta_1 \uparrow</math></th>
<th>REL <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th>mRI<math>_{\theta} \uparrow</math></th>
<th><math>\delta_1 \uparrow</math></th>
<th>REL <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th>mRI<math>_{\theta} \uparrow</math></th>
<th><math>\delta_1 \uparrow</math></th>
<th>REL <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th>mRI<math>_{\theta} \uparrow</math></th>
<th><math>\delta_1 \uparrow</math></th>
<th>REL <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th>mRI<math>_{\theta} \uparrow</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>BTS [20]</td>
<td>0.831</td>
<td>0.115</td>
<td>5.368</td>
<td>2.5%</td>
<td>0.805</td>
<td>0.147</td>
<td>7.550</td>
<td>-17.8%</td>
<td>0.016</td>
<td>1.785</td>
<td>5.908</td>
<td>24.3%</td>
<td>0.171</td>
<td>0.837</td>
<td>10.48</td>
<td>-4.8%</td>
</tr>
<tr>
<td>AdaBins [5]</td>
<td>0.826</td>
<td>0.122</td>
<td>5.420</td>
<td>0.0%</td>
<td>0.766</td>
<td>0.154</td>
<td>8.560</td>
<td>-26.7%</td>
<td>0.013</td>
<td>1.941</td>
<td>6.272</td>
<td>9.7%</td>
<td>0.161</td>
<td>0.863</td>
<td>10.35</td>
<td>-7.2%</td>
</tr>
<tr>
<td>LocalBins [6]</td>
<td>0.810</td>
<td>0.127</td>
<td>5.981</td>
<td>-5.3%</td>
<td>0.777</td>
<td>0.151</td>
<td>8.139</td>
<td>-23.2%</td>
<td><u>0.016</u></td>
<td>1.820</td>
<td>6.706</td>
<td>19.5%</td>
<td>0.170</td>
<td>0.821</td>
<td>10.27</td>
<td>-3.6%</td>
</tr>
<tr>
<td>NeWCRFs [50]</td>
<td>0.829</td>
<td>0.117</td>
<td>5.691</td>
<td>0.0%</td>
<td><b>0.874</b></td>
<td><b>0.119</b></td>
<td><b>6.183</b></td>
<td><b>0.0%</b></td>
<td>0.010</td>
<td>1.918</td>
<td>6.283</td>
<td>0.0%</td>
<td>0.176</td>
<td>0.854</td>
<td>9.228</td>
<td>0.0%</td>
</tr>
<tr>
<td><b>ZoeD-X-K</b></td>
<td>0.837</td>
<td>0.112</td>
<td>5.338</td>
<td>3.8%</td>
<td>0.790</td>
<td>0.137</td>
<td>7.734</td>
<td>-16.6%</td>
<td>0.005</td>
<td><u>1.756</u></td>
<td>6.180</td>
<td>-13.3%</td>
<td>0.242</td>
<td><u>0.799</u></td>
<td>7.806</td>
<td>19.8%</td>
</tr>
<tr>
<td><b>ZoeD-M12-K</b></td>
<td><b>0.864</b></td>
<td><b>0.100</b></td>
<td><b>4.974</b></td>
<td><b>10.5%</b></td>
<td><u>0.835</u></td>
<td><u>0.129</u></td>
<td>7.108</td>
<td><u>-9.3%</u></td>
<td>0.003</td>
<td>1.921</td>
<td>6.978</td>
<td>-27.1%</td>
<td><b>0.269</b></td>
<td>0.852</td>
<td><b>6.898</b></td>
<td><b>26.1%</b></td>
</tr>
<tr>
<td><b>ZoeD-M12-NK</b></td>
<td><u>0.850</u></td>
<td><u>0.105</u></td>
<td><u>5.095</u></td>
<td><u>7.8%</u></td>
<td>0.824</td>
<td>0.138</td>
<td>7.225</td>
<td>-12.8%</td>
<td><b>0.292</b></td>
<td><b>0.641</b></td>
<td><b>3.610</b></td>
<td><b>976.4%</b></td>
<td>0.208</td>
<td><b>0.757</b></td>
<td><u>7.569</u></td>
<td>15.8%</td>
</tr>
</tbody>
</table>

Table 4. Quantitative results for zero-shot transfer to four unseen outdoor datasets. mRI $_{\theta}$  denotes the mean relative improvement with respect to NeWCRFs across all metrics ( $\delta_1$ , REL, RMSE). Best results are in bold, second best are underlined.

tains mainly close-up images of outdoor scenarios making it more similar to an indoor dataset. Since *ZoeD-M12-NK* was also fine-tuned on NYU Depth v2 and automatically routes inputs to different heads, it seems to leverage its knowledge of the indoor domain to improve predictions. This is also supported by the low performance of *ZoeD-X-K* and *ZoeD-M12-K* which were only fine-tuned on KITTI. This result clearly shows the benefit of models fine-tuned across multiple domains for generalization to arbitrary datasets. We expect that defining more granular domains and fine-tuning a variant of our model with more than two heads across many metric datasets would lead to even better generalization performance.

## 5.4. Ablation Studies

In this section we study the importance of various design choices in our models.

**Backbones.** We study the effect of using different backbones for our base MiDaS model. The results are summarized in Fig. 5. We find that larger backbones with more parameters lead to improved performance, but our model still outperforms the previous state of the art when using the same backbone [28]. Further, the image classification performance of the backbone is highly correlated to the performance of our depth estimation model, *i.e.* lower absolute relative error (REL). Hence, our architecture can directly

benefit from new backbones as they get introduced in the future.

Figure 5. Backbone ablation study. There is a strong correlation between backbone performance in image classification and depth estimation. Larger backbones achieve lower absolute relative error (REL); with the same backbone and overall fewer parameters, our method still outperforms the current state-of-the-art NeWCRFs. The area of the circles is proportional to the number of parameters. The backbones shown are BEiT [3], Swin2 [27], Swin [28] and EfficientNet B5 [39], where L stands for large and B for base.

**Metric Bins Module.** We study the contribution to the overall performance by various variants of our metric bins module listed in Tab. 5. First, we remove our metric bins<table border="1">
<thead>
<tr>
<th colspan="3">Metric head type</th>
<th>REL ↓</th>
<th>RMSE ↓</th>
</tr>
<tr>
<th>Type</th>
<th>Variant</th>
<th>Config</th>
<th></th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">Naive head</td>
<td>-</td>
<td>-</td>
<td>0.096</td>
<td>0.335</td>
</tr>
<tr>
<td>Splitter</td>
<td>factor = 2</td>
<td>0.085</td>
<td>0.301</td>
</tr>
<tr>
<td rowspan="3">Metric bins</td>
<td>Exponential Attractor</td>
<td>{16,8,4,1}</td>
<td>0.086</td>
<td>0.305</td>
</tr>
<tr>
<td rowspan="3">Inverse Attractor</td>
<td>{8,8,8,8}</td>
<td>0.081</td>
<td>0.295</td>
</tr>
<tr>
<td>{16,2,2,16}</td>
<td>0.081</td>
<td>0.291</td>
</tr>
<tr>
<td>{1,4,8,16}</td>
<td><u>0.080</u></td>
<td><u>0.287</u></td>
</tr>
<tr>
<td></td>
<td>{16,8,4,1}</td>
<td><b>0.075</b></td>
<td><b>0.270</b></td>
</tr>
</tbody>
</table>

Table 5. **Metric head variants.** The “**Config**” column specifies the split factor in case of the splitter variant and the number of attractors  $\{n_a^l\}$  for attractor variants. The reported results are all based on *ZoeD-M12-N* evaluated on NYU Depth v2. Best results are in bold, second best are underlined.

module and attach a convolutional block to the decoder features from the base DPT model and directly predict the metric depth (standard regression). We call this variant *naive head*. Our best attractor variant performs about 21% better than the naive head. Notably, the metric bins with the splitter design as in [6] improves upon the naive head by 11.4%, which is consistent with the 10.8% improvement observed by [6] when comparing a naive Unet design with the splitter LocalBins design (refer to Tab. 3 in [6]). Next, we compare our novel attractor design with the splitter design of LocalBins [6]. Our best attractor variant performs 11.7% better. All the *inverse* attractor variants perform decisively better than the splitter variant while the *exponential* variant performs slightly worse.

**Routers.** As discussed in Sec. 3.3, we test the three variants for routing the relative features to metric heads. The results for the models with two metric heads, one for NYU Depth v2 and one for KITTI, are provided in Tab. 6. Out of the three variants, the Auto Router performs the worst. This is expected as in this case the router never sees any domain labels. Surprisingly, the Trained Router performs better on NYU Depth v2 than the Labeled Router, even though domain labels are unavailable during inference. We hypothesize that the domain-level discriminatory supervision may help in learning better representations. As we aim for a generic model without special requirements during inference, we choose the Trained Router and use it in all our multi-head models.

**Log Binomial.** We evaluate the effect of using a log binomial distribution by studying the performance of *ZoeD-M12-N* on NYU-Depth-v2 with log binomial and softmax probability heads. Consistent with [4], we observe that using log binomial (REL = 0.075) instead of softmax (REL = 0.077) leads to about 2% improvement. This highlights the importance of unimodal distributions for ordinal problems.

<table border="1">
<thead>
<tr>
<th rowspan="2">Variant</th>
<th colspan="2">Labels required</th>
<th colspan="2">REL ↓</th>
<th colspan="2">RMSE ↓</th>
</tr>
<tr>
<th>Train</th>
<th>Inference</th>
<th>NYU</th>
<th>KITTI</th>
<th>NYU</th>
<th>KITTI</th>
</tr>
</thead>
<tbody>
<tr>
<td>Labeled Router</td>
<td>✓</td>
<td>✓</td>
<td><u>0.080</u></td>
<td><b>0.057</b></td>
<td><u>0.290</u></td>
<td><u>2.452</u></td>
</tr>
<tr>
<td>Trained Router</td>
<td>✓</td>
<td>✗</td>
<td><b>0.077</b></td>
<td><b>0.057</b></td>
<td><b>0.277</b></td>
<td><b>2.362</b></td>
</tr>
<tr>
<td>Auto Router</td>
<td>✗</td>
<td>✗</td>
<td>0.102</td>
<td>0.075</td>
<td>0.377</td>
<td>2.584</td>
</tr>
</tbody>
</table>

Table 6. **Router variants.** The reported results are all based on *ZoeD-M12-NK* evaluated on NYU Depth v2 and KITTI. Best results are in bold, second best are underlined.

## 6. Conclusion

Our proposed framework, *ZoeDepth*, bridges the gap between relative and metric depth estimation. In the first stage, we pre-train an encoder-decoder architecture using relative depth on a collection of datasets. In the second stage, we add domain-specific heads based on our new metric bins module to the decoder and fine-tune the model on one or more datasets for metric depth prediction. Our proposed architecture decisively improves upon the state of the art for NYU Depth v2 (21% in REL) and also significantly improves upon the state of the art in zero-shot transfer. We expect that defining more granular domains beyond indoor and outdoor, and fine-tuning on more metric datasets can improve our results further. In future work, we would like to investigate a mobile architecture version of *ZoeDepth*, *e.g.*, for on-device photo editing, and extend our work to stereo-image depth estimation.## References

- [1] Miton Abramowitz. Stegun., ia (1972). handbook of mathematical functions. *Formulas, Graphs and Mathematical Tables*, 2002. [4](#)
- [2] Ashutosh Agarwal and Chetan Arora. Attention attention everywhere: Monocular depth prediction with skip attention. *arXiv preprint arXiv:2210.09071*, 2022. [3](#), [4](#)
- [3] Hangbo Bao, Li Dong, and Furu Wei. Beit: BERT pre-training of image transformers. *CoRR*, abs/2106.08254, 2021. [2](#), [3](#), [8](#), [12](#)
- [4] Christopher Beckham and Christopher Pal. Unimodal probability distributions for deep ordinal classification. In Doina Precup and Yee Whye Teh, editors, *Proceedings of the 34th International Conference on Machine Learning*, volume 70 of *Proceedings of Machine Learning Research*, pages 411–419. PMLR, 06–11 Aug 2017. [4](#), [9](#)
- [5] Shariq Farooq Bhat, Ibraheem Alhashim, and Peter Wonka. Adabins: Depth estimation using adaptive bins. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 4009–4018, 2021. [1](#), [2](#), [3](#), [4](#), [6](#), [7](#), [8](#), [13](#), [14](#), [15](#), [16](#), [20](#)
- [6] Shariq Farooq Bhat, Ibraheem Alhashim, and Peter Wonka. Localbins: Improving depth estimation by learning local distributions. In *European Conference on Computer Vision*, pages 480–496. Springer, 2022. [1](#), [2](#), [3](#), [4](#), [6](#), [7](#), [8](#), [9](#), [13](#), [14](#), [15](#), [16](#), [20](#)
- [7] Yohann Cabon, Naila Murray, and Martin Humenberger. Virtual kitti 2, 2020. [5](#), [12](#), [13](#), [15](#), [17](#)
- [8] Xiaotian Chen, Xuejin Chen, and Zheng-Jun Zha. Structure-aware residual pyramid network for monocular depth estimation. In *Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI-19*, pages 694–700. International Joint Conferences on Artificial Intelligence Organization, 7 2019. [6](#), [20](#)
- [9] David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. In *NIPS*, 2014. [6](#), [20](#)
- [10] William Fedus, Jeff Dean, and Barret Zoph. A review of sparse expert models in deep learning. *arXiv preprint arXiv:2209.01667*, 2022. [5](#)
- [11] Huan Fu, Mingming Gong, Chaohui Wang, Nematollah Batmanghelich, and Dacheng Tao. Deep ordinal regression network for monocular depth estimation. *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 2002–2011, 2018. [3](#), [6](#), [7](#), [20](#)
- [12] Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raventos, and Adrien Gaidon. 3d packing for self-supervised monocular depth estimation. In *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020. [5](#), [12](#), [13](#), [15](#), [18](#)
- [13] Zhixiang Hao, Yu Li, Shaodi You, and Feng Lu. Detail preserving depth estimation from a single image using attention guided networks. *2018 International Conference on 3D Vision (3DV)*, pages 304–313, 2018. [6](#), [20](#)
- [14] Junjie Hu, Mete Ozay, Yan Zhang, and Takayuki Okatani. Revisiting single image depth estimation: Toward higher resolution maps with accurate object boundaries. *2019 IEEE Winter Conference on Applications of Computer Vision (WACV)*, pages 1043–1051, 2018. [6](#), [20](#)
- [15] Xinyu Huang, Peng Wang, Xinjing Cheng, Dingfu Zhou, Qichuan Geng, and Ruigang Yang. The apolloscape open dataset for autonomous driving and its application. *IEEE transactions on pattern analysis and machine intelligence*, 42(10):2702–2719, 2019. [5](#), [13](#)
- [16] Jinyoung Jun, Jae-Han Lee, Chul Lee, and Chang-Su Kim. Depth map decomposition for monocular depth estimation. *arXiv preprint arXiv:2208.10762*, 2022. [2](#), [3](#), [6](#)
- [17] Youngjung Kim, Hyungjoo Jung, Dongbo Min, and Kwanghoon Sohn. Deep monocular depth estimation via integration of global and local predictions. *IEEE transactions on Image Processing*, 27(8):4131–4144, 2018. [5](#), [12](#), [13](#), [16](#), [18](#)
- [18] Tobias Koch, Lukas Liebel, Friedrich Fraundorfer, and Marco Körner. Evaluation of cnn-based single-image depth estimation methods. In *Proceedings ECCV 2018 Workshops*, 2019. [5](#), [12](#), [13](#), [15](#)
- [19] Iro Laina, Christian Rupprecht, Vasileios Belagiannis, Federico Tombari, and Nassir Navab. Deeper depth prediction with fully convolutional residual networks. *2016 Fourth International Conference on 3D Vision (3DV)*, pages 239–248, 2016. [6](#), [20](#)
- [20] Jin Han Lee, Myung-Kyu Han, Dong Wook Ko, and Il Hong Suh. From big to small: Multi-scale local planar guidance for monocular depth estimation. *arXiv preprint arXiv:1907.10326*, 2019. [6](#), [8](#), [13](#), [14](#), [15](#), [16](#), [20](#)
- [21] Jae-Han Lee and Chang-Su Kim. Monocular depth estimation using relative depth maps. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 9729–9738, 2019. [2](#)
- [22] Wonwoo Lee, Nohyoung Park, and Woontack Woo. Depth-assisted real-time 3d object detection for augmented reality. *ICAT’11*, 2:126–132, 2011. [6](#), [20](#)
- [23] Bo Li, Yuchao Dai, and Mingyi He. Monocular depth estimation with hierarchical fusion of dilated cnns and soft-weighted-sum inference. *Pattern Recognition*, 83:328–339, 2018. [3](#)
- [24] Ruibo Li, Ke Xian, Chunhua Shen, Zhiguo Cao, Hao Lu, and Lingxiao Hang. Deep attention-based classification network for robust depth prediction. In C.V. Jawahar, Hongdong Li, Greg Mori, and Konrad Schindler, editors, *Computer Vision – ACCV 2018*, pages 663–678, Cham, 2019. Springer International Publishing. [3](#)
- [25] Zhengqi Li and Noah Snavely. Megadepth: Learning single-view depth prediction from internet photos. In *Computer Vision and Pattern Recognition (CVPR)*, 2018. [5](#), [13](#)
- [26] Zhenyu Li, Xuyang Wang, Xianming Liu, and Junjun Jiang. Binsformer: Revisiting adaptive bins for monocular depth estimation. *arXiv preprint arXiv:2204.00987*, 2022. [1](#), [2](#), [3](#), [4](#)
- [27] Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, et al. Swin transformer v2: Scaling up capacity and resolution. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 12009–12019, 2022. [8](#), [12](#)[28] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 10012–10022, 2021. [8](#)

[29] Moritz Menze and Andreas Geiger. Object scene flow for autonomous vehicles. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, June 2015. [5](#), [12](#), [13](#)

[30] Alican Mertan, Damien Jade Duff, and Gozde Unal. Single image depth estimation: An overview. *Digital Signal Processing*, 123:103441, 2022. [1](#), [2](#)

[31] Michael Ramamonjisoa and Vincent Lepetit. Sharpnet: Fast and accurate recovery of occluding contours in monocular depth estimation. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops*, Oct 2019. [6](#), [20](#)

[32] René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 12179–12188, October 2021. [2](#), [3](#), [5](#)

[33] René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 2020. [1](#), [2](#), [3](#), [4](#), [5](#), [12](#), [13](#)

[34] Haoyu Ren, Mostafa El-Khamy, and Jungwon Lee. Deep robust single image depth estimation neural network using scene understanding. In *CVPR Workshops*, 2019. [3](#)

[35] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M. Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In *International Conference on Computer Vision (ICCV) 2021*, 2021. [5](#), [12](#), [13](#), [14](#), [17](#)

[36] Khalil Sarwari, Forrest Laine, and Claire Tomlin. Progress and proposals: A case study of monocular depth estimation. Master’s thesis, EECS Department, University of California, Berkeley, May 2021. [3](#), [7](#)

[37] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In *Computer Vision – ECCV 2012*, pages 746–760, Berlin, Heidelberg, 2012. Springer Berlin Heidelberg. [2](#), [5](#), [6](#), [12](#), [13](#)

[38] S. Song, S. P. Lichtenberg, and J. Xiao. Sun rgb-d: A rgb-d scene understanding benchmark suite. In *2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 567–576, 2015. [5](#), [12](#), [13](#), [14](#)

[39] Mingxing Tan and Quoc V. Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, *Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA*, volume 97 of *Proceedings of Machine Learning Research*, pages 6105–6114. PMLR, 2019. [8](#)

[40] Igor Vasiljevic, Nick Kolkin, Shanyi Zhang, Ruotian Luo, Haochen Wang, Falcon Z. Dai, Andrea F. Daniele, Mohammadreza Mostajabi, Steven Basart, Matthew R. Walter, and Gregory Shakhnarovich. DIODE: A Dense Indoor and Outdoor DEpth Dataset. *CoRR*, abs/1908.00463, 2019. [5](#), [12](#), [13](#), [14](#), [16](#), [19](#)

[41] Chaoyang Wang, Simon Lucey, Federico Perazzi, and Oliver Wang. Web stereo video supervision for depth prediction from dynamic scenes. In *2019 International Conference on 3D Vision (3DV)*, pages 348–357. IEEE, 2019. [5](#), [13](#)

[42] Qiang Wang, Shizhen Zheng, Qingsong Yan, Fei Deng, Kaiyong Zhao, and Xiaowen Chu. Irs: A large naturalistic indoor robotics stereo dataset to train deep models for disparity and surface normal estimation. *arXiv preprint arXiv:1912.09678*, 2019. [5](#), [13](#)

[43] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. Tartanair: A dataset to push the limits of visual slam. In *2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pages 4909–4916. IEEE, 2020. [5](#), [13](#)

[44] Ross Wightman. Pytorch image models. <https://github.com/rwightman/pytorch-image-models>, 2019. [5](#), [12](#)

[45] Ke Xian, Chunhua Shen, Zhiguo Cao, Hao Lu, Yang Xiao, Ruibo Li, and Zhenbo Luo. Monocular relative depth perception with web stereo data supervision. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 311–320, 2018. [5](#), [13](#)

[46] Ke Xian, Jianming Zhang, Oliver Wang, Long Mai, Zhe Lin, and Zhiguo Cao. Structure-guided ranking loss for single image depth prediction. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 611–620, 2020. [5](#), [13](#)

[47] Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren, Lei Zhou, Tian Fang, and Long Quan. Blendedmvs: A large-scale dataset for generalized multi-view stereo networks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 1790–1799, 2020. [5](#), [13](#)

[48] Wei Yin, Yifan Liu, Chunhua Shen, and Youliang Yan. Enforcing geometric constraints of virtual normal for depth prediction. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, October 2019. [6](#), [20](#)

[49] Wei Yin, Jianming Zhang, Oliver Wang, Simon Niklaus, Long Mai, Simon Chen, and Chunhua Shen. Learning to recover 3d scene shape from a single image. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 204–213, 2021. [3](#)

[50] Weihao Yuan, Xiaodong Gu, Zuozhuo Dai, Siyu Zhu, and Ping Tan. New crfs: Neural window fully-connected crfs for monocular depth estimation. *arXiv preprint arXiv:2203.01502*, 2022. [1](#), [2](#), [6](#), [7](#), [8](#), [13](#), [14](#), [15](#), [16](#), [17](#), [18](#), [19](#), [20](#)## A. Appendix

### A.1. Datasets Overview

We begin by providing a detailed overview of the properties of the datasets used for metric depth fine-tuning and evaluation of the new ZoeDepth architecture (see Fig. 2 in the main paper) in Table 7. These datasets consist of NYU Depth v2 [37] and KITTI [29] used for metric depth fine-tuning as well as respectively four in- and outdoor datasets to test for generalization performance (see Sec. 4.1 of the main paper). The indoor datasets consist of SUN RGB-D [38], the iBims benchmark [18], DIODE Indoor [40] and HyperSim [35]. For the outdoor datasets, we use DDAD [12], DIML Outdoor [17], DIODE Outdoor [40] and Virtual KITTI 2 [7].

All ZoeDepth architectures and prior works are evaluated by resizing the input to the training resolution. Zoe\*-N, Zoe\*-NK, and Zoe\*-K models are trained at resolutions  $384 \times 512$ ,  $384 \times 512$  and  $384 \times 768$  respectively. Predictions are resized to original ground truth resolution before evaluation.

### A.2. Training Details

In Table 8, we show various training strategies (see Section 3.3 of the main paper) applied to the ZoeDepth architecture. The training strategies differ by the datasets used for relative depth pre-training of the MiDaS [33] encoder-decoder, the datasets employed for metric depth fine-tuning in ZoeDepth and the number of used metric heads. Each combination of these options provided in Tab. 8 defines a different model of ZoeDepth. Results for these combinations are shown in Appendix A.3.

### A.3. Detailed Results

In Tables 3 and 4 of the main paper, we have provided quantitative results for zero-shot transfer to the four in- and outdoor datasets unseen during training, which are mentioned in Appendix A.1. These results are supplemented by the threshold accuracies  $\delta_2$  and  $\delta_3$  as well as the average  $\log_{10}$  error in Tables 9 to 16. Also, while the main paper only shows our models ZoeD-X-K, ZoeD-M12-K and ZoeD-M12-NK, Tables 9 to 16 contain the additional model ZoeD-NK-N. This model uses only the dataset combination of NYU Depth v2 (N) [37] and KITTI (K) [29] for the relative depth pre-training.

Figures 6 to 13 show metric depth maps computed with our ZoeDepth architecture for various example images of the in- and outdoor datasets described in Appendix A.1.

For the indoor datasets, NeWCRF shows a tendency to underestimate the depth, *e.g.*, the relatively bright images of NeWCRF in rows 3 and 4 of Figure 6 as well as rows 1 and 3 of Figure 7. Our models Zoe-M12-NK and Zoe-M12-N are much closer to the ground truth. Only row 4 of Figure 9

is an exception where our models share this behavior with NeWCRF.

For the outdoor datasets, rows 1 to 3 of Figure 12 clearly demonstrate the advantage of our model ZoeD-M12-NK with respect to NeWCRF. As explained in Section 5.3 of the main paper, ZoeD-M12-NK is not only fine-tuned on the outdoor dataset KITTI but also on the indoor dataset NYU Depth v2, which better reflects the low depth values observable in the RGB and ground truth images of Figure 12. The improved sharpness in predictions from our models when compared to NeWCRF, as mentioned in the caption of Figure 4 of the main paper, continues to hold across all 8 indoor and outdoor datasets.

### A.4. ZoeDepth with different backbones

We achieve the best performance for ZoeDepth when using the large BEiT<sub>384</sub>-L [3] backbone for the MiDaS encoder (see Fig. 2 of the main paper), which is responsible for the feature extraction of relative depth computation. According to Table 17, this transformer backbone causes ZoeDepth to consist of 345M parameters of which 344M parameters are consumed by MiDaS. In MiDaS, the BEiT<sub>384</sub>-L [3] backbone takes up 305M of the 344M parameters. The number of parameters of ZoeDepth is therefore mainly influenced by the chosen MiDaS encoder backbone.

Due to the modularity of our architecture, we can swap out the MiDaS encoder backbone. In Table 17, we compare the number of parameters of ZoeDepth for different backbones. The timm [44] repository, which provides the backbones, also offers the base BEiT<sub>384</sub>-B transformer. Utilizing this variant reduces the number of parameters of ZoeDepth from 345M to 112M. Another type of transformer with good performance is the hierarchical transformer Swin2 [27] based on shifted windows. When using the base and tiny variants Swin2-B and Swin2-T, the number of parameters of ZoeDepth drops to 102M and 42M, respectively. We report the results of all the aforementioned models evaluated on NYU Depth V2 in Table 18.<table border="1">
<thead>
<tr>
<th rowspan="2">Dataset</th>
<th rowspan="2">Domain</th>
<th rowspan="2">Type</th>
<th rowspan="2">Seen in Training?</th>
<th rowspan="2"># Train Samples</th>
<th rowspan="2"># Eval Samples</th>
<th colspan="2">Eval Depth [m]</th>
<th rowspan="2">Crop Method</th>
</tr>
<tr>
<th>Min</th>
<th>Max</th>
</tr>
</thead>
<tbody>
<tr>
<td>NYU Depth v2 [37]</td>
<td>Indoor</td>
<td>Real</td>
<td>✓</td>
<td>24k [20]</td>
<td>654</td>
<td>1e-3</td>
<td>10</td>
<td>Eigen</td>
</tr>
<tr>
<td>SUN RGB-D [38]</td>
<td>Indoor</td>
<td>Real</td>
<td>✗</td>
<td>-</td>
<td>5050</td>
<td>1e-3</td>
<td>8</td>
<td>Eigen</td>
</tr>
<tr>
<td>iBims-1 [18]</td>
<td>Indoor</td>
<td>Real</td>
<td>✗</td>
<td>-</td>
<td>100</td>
<td>1e-3</td>
<td>10</td>
<td>Eigen</td>
</tr>
<tr>
<td>DIODE Indoor [40]</td>
<td>Indoor</td>
<td>Real</td>
<td>✗</td>
<td>-</td>
<td>325</td>
<td>1e-3</td>
<td>10</td>
<td>Eigen</td>
</tr>
<tr>
<td>HyperSim [35]</td>
<td>Indoor</td>
<td>Synthetic</td>
<td>✗</td>
<td>-</td>
<td>7690</td>
<td>1e-3</td>
<td>80</td>
<td>Eigen</td>
</tr>
<tr>
<td>KITTI [29]</td>
<td>Outdoor</td>
<td>Real</td>
<td>✓</td>
<td>26k [20]</td>
<td>697</td>
<td>1e-3</td>
<td>80</td>
<td>Garg<sup>‡</sup></td>
</tr>
<tr>
<td>Virtual KITTI 2 [7]</td>
<td>Outdoor</td>
<td>Synthetic</td>
<td>✗</td>
<td>-</td>
<td>1701</td>
<td>1e-3</td>
<td>80</td>
<td>Garg<sup>‡</sup></td>
</tr>
<tr>
<td>DDAD [12]</td>
<td>Outdoor</td>
<td>Real</td>
<td>✗</td>
<td>-</td>
<td>3950</td>
<td>1e-3</td>
<td>80</td>
<td>Garg</td>
</tr>
<tr>
<td>DIML Outdoor [17]</td>
<td>Outdoor</td>
<td>Real</td>
<td>✗</td>
<td>-</td>
<td>500</td>
<td>1e-3</td>
<td>80</td>
<td>Garg</td>
</tr>
<tr>
<td>DIODE Outdoor [40]</td>
<td>Outdoor</td>
<td>Real</td>
<td>✗</td>
<td>-</td>
<td>446</td>
<td>1e-3</td>
<td>80</td>
<td>Garg</td>
</tr>
</tbody>
</table>

Table 7. Overview of datasets used in metric depth fine-tuning and evaluation of ZoeDepth architectures. For demonstrating zero-shot transfer, we evaluate across a total of 13165 indoor samples and 6597 outdoor samples. While HyperSim is predominantly an indoor dataset, there are several samples exhibiting depth ranges exceeding 10 m, so we relax the maximum evaluation depth up to 80 m. <sup>‡</sup>: To follow prior works [5, 50], we crop the sample and then use scaled Garg crop for evaluation. We verify the transforms by reproducing results obtained by using respective pre-trained checkpoints provided by prior works.

<table border="1">
<thead>
<tr>
<th>Model</th>
<th>Relative depth pre-training</th>
<th>Metric depth fine-tuning</th>
<th># metric heads</th>
</tr>
</thead>
<tbody>
<tr>
<td>ZoeD-X-N</td>
<td>✗</td>
<td>NYU</td>
<td>1</td>
</tr>
<tr>
<td>ZoeD-N-N</td>
<td>NYU</td>
<td>NYU</td>
<td>1</td>
</tr>
<tr>
<td>ZoeD-NK-N</td>
<td>NYU+KITTI</td>
<td>NYU</td>
<td>1</td>
</tr>
<tr>
<td>ZoeD-M12-N</td>
<td>M12</td>
<td>NYU</td>
<td>1</td>
</tr>
<tr>
<td>ZoeD-X-K</td>
<td>✗</td>
<td>KITTI</td>
<td>1</td>
</tr>
<tr>
<td>ZoeD-K-K</td>
<td>KITTI</td>
<td>KITTI</td>
<td>1</td>
</tr>
<tr>
<td>ZoeD-NK-K</td>
<td>NYU+KITTI</td>
<td>KITTI</td>
<td>1</td>
</tr>
<tr>
<td>ZoeD-M12-K</td>
<td>M12</td>
<td>KITTI</td>
<td>1</td>
</tr>
<tr>
<td>ZoeD-NK-NK<sup>†</sup></td>
<td>NYU+KITTI</td>
<td>NYU+KITTI</td>
<td>1</td>
</tr>
<tr>
<td>ZoeD-M12-NK<sup>†</sup></td>
<td>M12</td>
<td>NYU+KITTI</td>
<td>1</td>
</tr>
<tr>
<td>ZoeD-NK-NK</td>
<td>NYU+KITTI</td>
<td>NYU+KITTI</td>
<td>2</td>
</tr>
<tr>
<td>ZoeD-M12-NK</td>
<td>M12</td>
<td>NYU+KITTI</td>
<td>2</td>
</tr>
</tbody>
</table>

Table 8. Models are named according to the following convention: ZoeD-RDPT-MFT, where ZoeD is the abbreviation for ZoeDepth, RDPT denotes the datasets used for relative depth pre-training and MFT denotes the datasets used for metric depth fine-tuning. Models with an X do not use a relative depth pre-training. The collection M12 contains the datasets HRWSI [46], BlendedMVS [47], ReDWeb [45], DIML-Indoor [17], 3D Movies [33], MegaDepth [25], WSVD [41], TartanAir [43], ApolloScape [15], IRS [42], KITTI (K) [29] and NYU Depth v2 (N) [37].

<table border="1">
<thead>
<tr>
<th>Method</th>
<th><math>\delta_1 \uparrow</math></th>
<th><math>\delta_2 \uparrow</math></th>
<th><math>\delta_3 \uparrow</math></th>
<th>REL <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th><math>\log_{10} \downarrow</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>BTS [20]</td>
<td>0.740</td>
<td>0.933</td>
<td>0.980</td>
<td>0.172</td>
<td>0.515</td>
<td>0.075</td>
</tr>
<tr>
<td>AdaBins [5]</td>
<td>0.771</td>
<td>0.944</td>
<td>0.983</td>
<td>0.159</td>
<td>0.476</td>
<td>0.068</td>
</tr>
<tr>
<td>LocalBins [6]</td>
<td>0.777</td>
<td>0.949</td>
<td>0.985</td>
<td>0.156</td>
<td>0.470</td>
<td>0.067</td>
</tr>
<tr>
<td>NeWCRF [50]</td>
<td>0.798</td>
<td>0.967</td>
<td>0.992</td>
<td>0.151</td>
<td>0.424</td>
<td>0.064</td>
</tr>
<tr>
<td><b>ZoeD-X-N</b></td>
<td><u>0.857</u></td>
<td><u>0.979</u></td>
<td><b>0.995</b></td>
<td>0.124</td>
<td>0.363</td>
<td>0.054</td>
</tr>
<tr>
<td><b>ZoeD-NK-N</b></td>
<td><u>0.857</u></td>
<td>0.978</td>
<td><u>0.994</u></td>
<td>0.125</td>
<td><u>0.360</u></td>
<td>0.054</td>
</tr>
<tr>
<td><b>ZoeD-M12-N</b></td>
<td><b>0.864</b></td>
<td><b>0.982</b></td>
<td><b>0.995</b></td>
<td><b>0.119</b></td>
<td><b>0.346</b></td>
<td><b>0.052</b></td>
</tr>
<tr>
<td><b>ZoeD-M12-NK</b></td>
<td>0.856</td>
<td><u>0.979</u></td>
<td><b>0.995</b></td>
<td><u>0.123</u></td>
<td><u>0.356</u></td>
<td><u>0.053</u></td>
</tr>
</tbody>
</table>

Table 9. Zero-shot transfer to the SUN RGB-D dataset [38]. Best results are in bold, second best are underlined.

<table border="1">
<thead>
<tr>
<th>Method</th>
<th><math>\delta_1 \uparrow</math></th>
<th><math>\delta_2 \uparrow</math></th>
<th><math>\delta_3 \uparrow</math></th>
<th>REL <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th><math>\log_{10} \downarrow</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>BTS [20]</td>
<td>0.538</td>
<td>0.863</td>
<td>0.948</td>
<td>0.231</td>
<td>0.919</td>
<td>0.112</td>
</tr>
<tr>
<td>AdaBins [5]</td>
<td>0.555</td>
<td>0.873</td>
<td>0.960</td>
<td>0.212</td>
<td>0.901</td>
<td>0.107</td>
</tr>
<tr>
<td>LocalBins [6]</td>
<td>0.558</td>
<td>0.877</td>
<td>0.966</td>
<td>0.211</td>
<td>0.880</td>
<td>0.104</td>
</tr>
<tr>
<td>NeWCRF [50]</td>
<td>0.548</td>
<td>0.884</td>
<td>0.979</td>
<td>0.206</td>
<td>0.861</td>
<td>0.102</td>
</tr>
<tr>
<td><b>ZoeD-X-N</b></td>
<td><u>0.668</u></td>
<td><u>0.944</u></td>
<td><u>0.983</u></td>
<td>0.173</td>
<td><u>0.730</u></td>
<td><u>0.084</u></td>
</tr>
<tr>
<td><b>ZoeD-NK-N</b></td>
<td><b>0.671</b></td>
<td>0.939</td>
<td><u>0.983</u></td>
<td><u>0.172</u></td>
<td>0.735</td>
<td>0.084</td>
</tr>
<tr>
<td><b>ZoeD-M12-N</b></td>
<td>0.658</td>
<td><b>0.947</b></td>
<td><b>0.985</b></td>
<td><b>0.169</b></td>
<td><b>0.711</b></td>
<td><b>0.083</b></td>
</tr>
<tr>
<td><b>ZoeD-M12-NK</b></td>
<td>0.615</td>
<td>0.928</td>
<td>0.982</td>
<td>0.186</td>
<td>0.777</td>
<td>0.092</td>
</tr>
</tbody>
</table>

Table 10. Zero-shot transfer to the iBims-1 benchmark [18]. Best results are in bold, second best are underlined.Figure 6. Zero-shot transfer to the SUN RGB-D dataset [38]. Invalid regions are indicated in gray.

<table border="1">
<thead>
<tr>
<th>Method</th>
<th><math>\delta_1 \uparrow</math></th>
<th><math>\delta_2 \uparrow</math></th>
<th><math>\delta_3 \uparrow</math></th>
<th>REL <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th><math>\log_{10} \downarrow</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>BTS [20]</td>
<td>0.210</td>
<td>0.478</td>
<td>0.699</td>
<td>0.418</td>
<td>1.905</td>
<td>0.250</td>
</tr>
<tr>
<td>AdaBins [5]</td>
<td>0.174</td>
<td>0.438</td>
<td>0.658</td>
<td>0.443</td>
<td>1.963</td>
<td>0.270</td>
</tr>
<tr>
<td>LocalBins [6]</td>
<td>0.229</td>
<td>0.520</td>
<td>0.718</td>
<td>0.412</td>
<td>1.853</td>
<td>0.246</td>
</tr>
<tr>
<td>NeWCRF [50]</td>
<td>0.187</td>
<td>0.498</td>
<td>0.748</td>
<td>0.404</td>
<td>1.867</td>
<td>0.241</td>
</tr>
<tr>
<td><b>ZoeD-X-N</b></td>
<td><b>0.400</b></td>
<td><b>0.704</b></td>
<td>0.808</td>
<td><b>0.324</b></td>
<td><b>1.581</b></td>
<td><b>0.181</b></td>
</tr>
<tr>
<td><b>ZoeD-NK-N</b></td>
<td>0.365</td>
<td><u>0.696</u></td>
<td><u>0.819</u></td>
<td>0.335</td>
<td>1.604</td>
<td>0.188</td>
</tr>
<tr>
<td><b>ZoeD-M12-N</b></td>
<td>0.376</td>
<td><u>0.696</u></td>
<td><b>0.822</b></td>
<td><u>0.327</u></td>
<td><u>1.588</u></td>
<td>0.186</td>
</tr>
<tr>
<td><b>ZoeD-M12-NK</b></td>
<td><u>0.386</u></td>
<td>0.695</td>
<td>0.807</td>
<td>0.331</td>
<td>1.598</td>
<td><u>0.185</u></td>
</tr>
</tbody>
</table>

Table 11. Zero-shot transfer to the DIODE Indoor dataset [40]. Best results are in bold, second best are underlined.

<table border="1">
<thead>
<tr>
<th>Method</th>
<th><math>\delta_1 \uparrow</math></th>
<th><math>\delta_2 \uparrow</math></th>
<th><math>\delta_3 \uparrow</math></th>
<th>REL <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th><math>\log_{10} \downarrow</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>BTS [20]</td>
<td>0.225</td>
<td>0.419</td>
<td>0.582</td>
<td>0.476</td>
<td>6.404</td>
<td>0.329</td>
</tr>
<tr>
<td>AdaBins [5]</td>
<td>0.221</td>
<td>0.410</td>
<td>0.568</td>
<td>0.483</td>
<td>6.546</td>
<td>0.345</td>
</tr>
<tr>
<td>LocalBins [6]</td>
<td>0.234</td>
<td>0.432</td>
<td>0.594</td>
<td>0.468</td>
<td>6.362</td>
<td>0.320</td>
</tr>
<tr>
<td>NeWCRF [50]</td>
<td>0.255</td>
<td>0.464</td>
<td>0.638</td>
<td>0.442</td>
<td>6.017</td>
<td>0.283</td>
</tr>
<tr>
<td><b>ZoeD-X-N</b></td>
<td>0.284</td>
<td>0.502</td>
<td>0.692</td>
<td>0.421</td>
<td>5.889</td>
<td>0.267</td>
</tr>
<tr>
<td><b>ZoeD-NK-N</b></td>
<td><u>0.291</u></td>
<td><b>0.519</b></td>
<td><u>0.700</u></td>
<td>0.414</td>
<td>5.838</td>
<td><u>0.260</u></td>
</tr>
<tr>
<td><b>ZoeD-M12-N</b></td>
<td><b>0.292</b></td>
<td><u>0.514</u></td>
<td><b>0.706</b></td>
<td><b>0.410</b></td>
<td><b>5.771</b></td>
<td><b>0.257</b></td>
</tr>
<tr>
<td><b>ZoeD-M12-NK</b></td>
<td>0.274</td>
<td>0.494</td>
<td>0.696</td>
<td>0.419</td>
<td>5.830</td>
<td>0.262</td>
</tr>
</tbody>
</table>

Table 12. Zero-shot transfer to the HyperSim dataset [35]. Best results are in bold, second best are underlined.Figure 7. Zero-shot transfer to the iBims-1 benchmark [18]. Invalid regions are indicated in gray.

<table border="1">
<thead>
<tr>
<th>Method</th>
<th><math>\delta_1 \uparrow</math></th>
<th><math>\delta_2 \uparrow</math></th>
<th><math>\delta_3 \uparrow</math></th>
<th>REL <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th><math>\log_{10} \downarrow</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>BTS [20]</td>
<td>0.831</td>
<td>0.948</td>
<td>0.982</td>
<td>0.115</td>
<td>5.368</td>
<td>0.054</td>
</tr>
<tr>
<td>AdaBins [5]</td>
<td>0.826</td>
<td>0.947</td>
<td>0.98</td>
<td>0.122</td>
<td>5.42</td>
<td>0.057</td>
</tr>
<tr>
<td>LocalBins [6]</td>
<td>0.810</td>
<td>0.94</td>
<td>0.978</td>
<td>0.127</td>
<td>5.981</td>
<td>0.061</td>
</tr>
<tr>
<td>NeWCRF [50]</td>
<td>0.829</td>
<td>0.951</td>
<td>0.984</td>
<td>0.117</td>
<td>5.691</td>
<td>0.056</td>
</tr>
<tr>
<td><b>ZoeD-X-K</b></td>
<td>0.837</td>
<td>0.965</td>
<td><u>0.991</u></td>
<td>0.112</td>
<td>5.338</td>
<td>0.053</td>
</tr>
<tr>
<td><b>ZoeD-NK-K</b></td>
<td><u>0.855</u></td>
<td><u>0.970</u></td>
<td><b>0.992</b></td>
<td><u>0.101</u></td>
<td>5.102</td>
<td><u>0.048</u></td>
</tr>
<tr>
<td><b>ZoeD-M12-K</b></td>
<td><b>0.864</b></td>
<td><b>0.973</b></td>
<td><b>0.992</b></td>
<td><b>0.100</b></td>
<td><b>4.974</b></td>
<td><b>0.046</b></td>
</tr>
<tr>
<td><b>ZoeD-M12-NK</b></td>
<td>0.850</td>
<td>0.965</td>
<td><u>0.991</u></td>
<td>0.105</td>
<td><u>5.095</u></td>
<td>0.050</td>
</tr>
</tbody>
</table>

Table 13. Zero-shot transfer to the Virtual KITTI 2 dataset [7]. Best results are in bold, second best are underlined.

<table border="1">
<thead>
<tr>
<th>Method</th>
<th><math>\delta_1 \uparrow</math></th>
<th><math>\delta_2 \uparrow</math></th>
<th><math>\delta_3 \uparrow</math></th>
<th>REL <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th><math>\log_{10} \downarrow</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>BTS [20]</td>
<td>0.805</td>
<td>0.945</td>
<td>0.982</td>
<td>0.147</td>
<td>7.550</td>
<td>0.067</td>
</tr>
<tr>
<td>AdaBins [5]</td>
<td>0.766</td>
<td>0.918</td>
<td>0.972</td>
<td>0.154</td>
<td>8.560</td>
<td>0.074</td>
</tr>
<tr>
<td>LocalBins [6]</td>
<td>0.777</td>
<td>0.930</td>
<td>0.978</td>
<td>0.151</td>
<td>8.139</td>
<td>0.071</td>
</tr>
<tr>
<td>NeWCRF [50]</td>
<td><b>0.874</b></td>
<td><b>0.974</b></td>
<td><b>0.991</b></td>
<td><b>0.119</b></td>
<td><b>6.183</b></td>
<td><b>0.054</b></td>
</tr>
<tr>
<td><b>ZoeD-X-K</b></td>
<td>0.790</td>
<td>0.95</td>
<td>0.985</td>
<td>0.137</td>
<td>7.734</td>
<td>0.066</td>
</tr>
<tr>
<td><b>ZoeD-NK-K</b></td>
<td>0.824</td>
<td>0.957</td>
<td>0.987</td>
<td>0.134</td>
<td>7.249</td>
<td>0.062</td>
</tr>
<tr>
<td><b>ZoeD-M12-K</b></td>
<td><u>0.835</u></td>
<td><u>0.962</u></td>
<td><u>0.988</u></td>
<td><u>0.129</u></td>
<td><u>7.108</u></td>
<td><u>0.060</u></td>
</tr>
<tr>
<td><b>ZoeD-M12-NK</b></td>
<td>0.824</td>
<td>0.951</td>
<td>0.980</td>
<td>0.138</td>
<td>7.225</td>
<td>0.066</td>
</tr>
</tbody>
</table>

Table 14. Zero-shot transfer to the DDAD dataset [12]. Best results are in bold, second best are underlined.Figure 8. Zero-shot transfer to the DIODE Indoor dataset [40]. Invalid regions are indicated in gray.

<table border="1">
<thead>
<tr>
<th>Method</th>
<th><math>\delta_1 \uparrow</math></th>
<th><math>\delta_2 \uparrow</math></th>
<th><math>\delta_3 \uparrow</math></th>
<th>REL <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th><math>\log_{10} \downarrow</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>BTS [20]</td>
<td><u>0.016</u></td>
<td>0.042</td>
<td>0.123</td>
<td>1.785</td>
<td><u>5.908</u></td>
<td><u>0.428</u></td>
</tr>
<tr>
<td>AdaBins [5]</td>
<td>0.013</td>
<td>0.038</td>
<td>0.107</td>
<td>1.941</td>
<td>6.272</td>
<td>0.451</td>
</tr>
<tr>
<td>LocalBins [6]</td>
<td><u>0.016</u></td>
<td><u>0.044</u></td>
<td><u>0.124</u></td>
<td>1.82</td>
<td>6.706</td>
<td>0.434</td>
</tr>
<tr>
<td>NeWCRF [50]</td>
<td>0.010</td>
<td>0.032</td>
<td>0.094</td>
<td>1.918</td>
<td>6.283</td>
<td>0.449</td>
</tr>
<tr>
<td><b>ZoeD-X-K</b></td>
<td>0.005</td>
<td>0.022</td>
<td>0.096</td>
<td><u>1.756</u></td>
<td>6.180</td>
<td>0.429</td>
</tr>
<tr>
<td><b>ZoeD-NK-K</b></td>
<td>0.004</td>
<td>0.012</td>
<td>0.047</td>
<td>2.068</td>
<td>7.432</td>
<td>0.473</td>
</tr>
<tr>
<td><b>ZoeD-M12-K</b></td>
<td>0.003</td>
<td>0.010</td>
<td>0.048</td>
<td>1.921</td>
<td>6.978</td>
<td>0.455</td>
</tr>
<tr>
<td><b>ZoeD-M12-NK</b></td>
<td><b>0.292</b></td>
<td><b>0.562</b></td>
<td><b>0.697</b></td>
<td><b>0.641</b></td>
<td><b>3.610</b></td>
<td><b>0.213</b></td>
</tr>
</tbody>
</table>

Table 15. Zero-shot transfer to the DIML Outdoor dataset [17]. Best results are in bold, second best are underlined.

<table border="1">
<thead>
<tr>
<th>Method</th>
<th><math>\delta_1 \uparrow</math></th>
<th><math>\delta_2 \uparrow</math></th>
<th><math>\delta_3 \uparrow</math></th>
<th>REL <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th><math>\log_{10} \downarrow</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>BTS [20]</td>
<td>0.171</td>
<td>0.347</td>
<td>0.526</td>
<td>0.837</td>
<td>10.48</td>
<td>0.334</td>
</tr>
<tr>
<td>AdaBins [5]</td>
<td>0.161</td>
<td>0.329</td>
<td>0.529</td>
<td>0.863</td>
<td>10.35</td>
<td>0.318</td>
</tr>
<tr>
<td>LocalBins [6]</td>
<td>0.170</td>
<td>0.336</td>
<td>0.531</td>
<td><u>0.821</u></td>
<td>10.273</td>
<td>0.329</td>
</tr>
<tr>
<td>NeWCRF [50]</td>
<td>0.176</td>
<td>0.369</td>
<td>0.588</td>
<td>0.854</td>
<td>9.228</td>
<td>0.283</td>
</tr>
<tr>
<td><b>ZoeD-X-K</b></td>
<td><u>0.242</u></td>
<td>0.485</td>
<td>0.744</td>
<td><b>0.799</b></td>
<td>7.806</td>
<td>0.219</td>
</tr>
<tr>
<td><b>ZoeD-NK-K</b></td>
<td>0.241</td>
<td><u>0.505</u></td>
<td><u>0.759</u></td>
<td>0.892</td>
<td><u>7.489</u></td>
<td><u>0.216</u></td>
</tr>
<tr>
<td><b>ZoeD-M12-K</b></td>
<td><b>0.269</b></td>
<td><b>0.563</b></td>
<td><b>0.816</b></td>
<td>0.852</td>
<td><b>6.898</b></td>
<td><b>0.198</b></td>
</tr>
<tr>
<td><b>ZoeD-M12-NK</b></td>
<td>0.208</td>
<td>0.405</td>
<td>0.586</td>
<td>0.757</td>
<td>7.569</td>
<td>0.258</td>
</tr>
</tbody>
</table>

Table 16. Zero-shot transfer to the DIODE Outdoor dataset [40]. Best results are in bold, second best are underlined.Figure 9. Zero-shot transfer to the HyperSim dataset [35].

Figure 10. Zero-shot transfer to the Virtual KITTI 2 dataset [7]. Invalid regions are indicated in gray.Figure 11. Zero-shot transfer to the DDAD dataset [12]. Ground truth depth is too sparse to visualize here.

Figure 12. Zero-shot transfer to the DIML Outdoor dataset [17].Figure 13. Zero-shot transfer to the DIODE Outdoor dataset [40]. Invalid regions are indicated in gray.<table border="1">
<thead>
<tr>
<th>Method</th>
<th>Encoder</th>
<th># Params</th>
</tr>
</thead>
<tbody>
<tr>
<td>Eigen <i>et al.</i> [9]</td>
<td>-</td>
<td>141M</td>
</tr>
<tr>
<td>Laina <i>et al.</i> [19]</td>
<td>ResNet-50</td>
<td>64M</td>
</tr>
<tr>
<td>Hao <i>et al.</i> [13]</td>
<td>ResNet-101</td>
<td>60M</td>
</tr>
<tr>
<td>Lee <i>et al.</i> [22]</td>
<td>-</td>
<td>119M</td>
</tr>
<tr>
<td>Fu <i>et al.</i> [11]</td>
<td>ResNet-101</td>
<td>110M</td>
</tr>
<tr>
<td>SharpNet [31]</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>Hu <i>et al.</i> [14]</td>
<td>SENet-154</td>
<td>157M</td>
</tr>
<tr>
<td>Chen <i>et al.</i> [8]</td>
<td>SENet</td>
<td>210M</td>
</tr>
<tr>
<td>Yin <i>et al.</i> [48]</td>
<td>ResNeXt-101</td>
<td>114M</td>
</tr>
<tr>
<td>BTS [20]</td>
<td>DenseNet-161</td>
<td>47M</td>
</tr>
<tr>
<td>AdaBins [5]</td>
<td>EfficientNet-B5</td>
<td>78M</td>
</tr>
<tr>
<td>LocalBins [6]</td>
<td>EfficientNet-B5</td>
<td>74M</td>
</tr>
<tr>
<td>NeWCRFs [50]</td>
<td>Swin-L</td>
<td>270M</td>
</tr>
<tr>
<td><b>ZoeDepth (S-L)</b></td>
<td>Swin-L</td>
<td>212M</td>
</tr>
<tr>
<td><b>ZoeDepth (S2-T)</b></td>
<td>Swin2-T</td>
<td>42M</td>
</tr>
<tr>
<td><b>ZoeDepth (S2-B)</b></td>
<td>Swin2-B</td>
<td>102M</td>
</tr>
<tr>
<td><b>ZoeDepth (S2-L)</b></td>
<td>Swin2-L</td>
<td>214M</td>
</tr>
<tr>
<td><b>ZoeDepth (B-B)</b></td>
<td>Beit-B</td>
<td>112M</td>
</tr>
<tr>
<td><b>ZoeDepth (B-L)</b></td>
<td>Beit-L</td>
<td>345M</td>
</tr>
</tbody>
</table>

Table 17. Parameter comparison of ZoeDepth models with different backbones and state of the art models. Note that the number of parameters of ZoeDepth only varies with the backbone and is the same for all variants trained on different dataset combinations, *e.g.*, ZoeD-X-N, ZoeD-M12-N and ZoeD-M12-NK, *etc.*

<table border="1">
<thead>
<tr>
<th>Method</th>
<th><math>\delta_1 \uparrow</math></th>
<th><math>\delta_2 \uparrow</math></th>
<th><math>\delta_3 \uparrow</math></th>
<th>REL <math>\downarrow</math></th>
<th>RMSE <math>\downarrow</math></th>
<th><math>\log_{10} \downarrow</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>BTS [20]</td>
<td>0.885</td>
<td>0.978</td>
<td>0.994</td>
<td>0.110</td>
<td>0.392</td>
<td>0.047</td>
</tr>
<tr>
<td>AdaBins [5]</td>
<td>0.903</td>
<td>0.984</td>
<td>0.997</td>
<td>0.103</td>
<td>0.364</td>
<td>0.044</td>
</tr>
<tr>
<td>LocalBins [6]</td>
<td>0.907</td>
<td>0.987</td>
<td><u>0.998</u></td>
<td>0.099</td>
<td>0.357</td>
<td>0.042</td>
</tr>
<tr>
<td>NeWCRFs [50]</td>
<td>0.922</td>
<td>0.992</td>
<td><u>0.998</u></td>
<td>0.095</td>
<td>0.334</td>
<td>0.041</td>
</tr>
<tr>
<td><b>ZoeD-M12-N (S-L)</b></td>
<td>0.937</td>
<td>0.992</td>
<td><u>0.998</u></td>
<td>0.086</td>
<td>0.310</td>
<td>0.037</td>
</tr>
<tr>
<td><b>ZoeD-M12-N (S2-T)</b></td>
<td>0.899</td>
<td>0.982</td>
<td>0.995</td>
<td>0.106</td>
<td>0.371</td>
<td>0.045</td>
</tr>
<tr>
<td><b>ZoeD-M12-N (S2-B)</b></td>
<td>0.927</td>
<td>0.992</td>
<td><b>0.999</b></td>
<td>0.090</td>
<td>0.313</td>
<td>0.038</td>
</tr>
<tr>
<td><b>ZoeD-M12-N (S2-L)</b></td>
<td><u>0.943</u></td>
<td><u>0.993</u></td>
<td><b>0.999</b></td>
<td><u>0.083</u></td>
<td><u>0.296</u></td>
<td><u>0.035</u></td>
</tr>
<tr>
<td><b>ZoeD-M12-N (B-B)</b></td>
<td>0.922</td>
<td>0.990</td>
<td><u>0.998</u></td>
<td>0.093</td>
<td>0.329</td>
<td>0.040</td>
</tr>
<tr>
<td><b>ZoeD-M12-N (B-L)</b></td>
<td><b>0.955</b></td>
<td><b>0.995</b></td>
<td><b>0.999</b></td>
<td><b>0.075</b></td>
<td><b>0.270</b></td>
<td><b>0.032</b></td>
</tr>
</tbody>
</table>

Table 18. Results on the NYU Depth V2 dataset with different backbones. Best results are in bold, second best are underlined.

