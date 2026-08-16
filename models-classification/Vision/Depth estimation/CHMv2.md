# CHMv2: Improvements in Global Canopy Height Mapping using DINOv3

John Brandt<sup>1,\*</sup>, Seungeun Yi<sup>2,+</sup>, Jamie Tolan<sup>3,+</sup>, Xinyuan Li<sup>4</sup>, Peter Potapov<sup>1</sup>, Jessica Ertel<sup>1</sup>, Justine Spore<sup>1</sup>, Huy V. Vo<sup>2</sup>, Michaël Ramamonjisoa<sup>2</sup>, Patrick Labatut<sup>2</sup>, Piotr Bojanowski<sup>2</sup>, and Camille Couprie<sup>2+</sup>

<sup>1</sup>World Resources Institute, 10 G St NE #800, Washington, DC 20002, USA

<sup>2</sup>Fundamental AI Research (FAIR), Meta, 75002 Paris, France

<sup>3</sup>Meta, 1 Hacker Way, Menlo Park, CA 94025, USA

<sup>4</sup>University of Maryland, Department of Geography, College Park, MD 20742, USA

\* John.Brandt@wri.org

+these authors contributed equally to this work

## ABSTRACT

Accurate canopy height information is essential for quantifying forest carbon, monitoring restoration and degradation, and assessing habitat structure, yet high-fidelity measurements from airborne laser scanning (ALS) remain unevenly available globally. Here we present CHMv2, a global, meter-resolution canopy height map derived from high-resolution optical satellite imagery using a depth-estimation model built on DINOv3 and trained against ALS canopy height models. Compared to existing products, CHMv2 substantially improves accuracy, reduces bias in tall forests, and better preserves fine-scale structure such as canopy edges and gaps. These gains are enabled by a large expansion of geographically diverse training data, automated data curation and registration, and a loss formulation and data sampling strategy tailored to canopy height distributions. We validate CHMv2 against independent ALS test sets and against tens of millions of GEDI and ICESat-2 observations, demonstrating consistent performance across major forest biomes.

## 1 Background & Summary

Remote sensing assessments of canopy height and structure are important for forest land management, including the monitoring of forest and landscape restoration<sup>1</sup>, detecting forest degradation and regrowth<sup>2</sup>, and measuring above-ground biomass<sup>3</sup>. Wall-to-wall maps of canopy height provide a bridge between field-based measurements and landscape or national scale land management decisions and are increasingly needed to support climate mitigation accounting, restoration planning, biodiversity assessments, and emerging digital monitoring, reporting, and verification (dMRV) approaches for climate finance<sup>4</sup>. Remote-sensing derived canopy height datasets have been used to improve land cover mapping<sup>5</sup>, measure the carbon benefits of afforestation<sup>6</sup>, model forest fire fuel<sup>7</sup>, and estimate aboveground biomass<sup>8</sup> among other applications. Beyond percentile-based canopy height estimates (e.g., p95), fine-scale structural metrics such as gap fraction, edge density, and height heterogeneity are typically derived from ALS-based canopy height models and are not recoverable from medium-resolution products. In forest monitoring contexts, these structural metrics are commonly used as covariates or stratification layers to reduce uncertainty in biomass estimation, support sampling design for field verification, and characterize heterogeneous systems such as agroforestry and secondary forests, where canopy structure varies at sub-hectare scales<sup>9,10</sup>. Extending access to such structural information beyond regions with ALS coverage remains a key challenge for global monitoring.

Global canopy height datasets derived from observation satellite data have utilized a variety of methodological approaches, typically involving a fusion of optical satellite imagery with LiDAR observations derived from spaceborne instruments like Global Ecosystem Dynamics Investigation (GEDI)<sup>11</sup> and ICESat-2<sup>12</sup> or airborne laser scanning<sup>13</sup>. Among globally available datasets, Potapov *et al.*<sup>14</sup> fused GEDI RH95 (percentile of relative height) data with Landsat multispectral data using a locally calibrated regression tree ensemble algorithm to generate a 30-meter global canopy height product for 2020. Using Sentinel-2 data, Lang *et al.*<sup>15</sup> applied an ensemble of convolutional neural network (CNN) models to predict the GEDI RH98 value for**Figure 1.** Visual improvements from CHMv1 to CHMv2 in a disturbance area in the Amazon (top), an urban forest in Central Java, Indonesia (middle) and a plantation in the Ghanaian cocoa belt (bottom).

each 10-meter pixel. Despite the diversity in methods for canopy height mapping, these datasets share three limitations. First, they have difficulty capturing short or low-stature vegetation<sup>16</sup>. Second, they are not high enough resolution to map forest structure or complexity which is a key input to degradation, restoration, and biodiversity monitoring<sup>17</sup>. Third, the majority of global-scale, medium resolution height products rely solely on spaceborne LiDAR from GEDI or ICESat-2, rather than airborne LiDAR, which may limit their applicability for land management at granular spatial scales where the gold standard evaluation criteria is agreement with airborne LiDAR data<sup>18</sup>.

Recent breakthroughs in vision transformers (ViTs) and self-supervised learning (SSL) have substantially improved performance in dense prediction tasks, including monocular depth estimation. Self-supervised methods such as DINOv2<sup>19</sup> and more recently DINOv3<sup>20</sup> have demonstrated strong capabilities in extracting semantically meaningful representations from large corpora of unlabeled imagery. These representations have shown improved generalizability across different domains, enabling DINOv3 to establish state of the art results on a variety of image tasks within earth observation. For the task of canopy height mapping, the DINOv3 SSL features show clear accuracy improvements over strong baselines when measured on the Open-Canopy benchmark<sup>21</sup>. Features from SSL ViTs have also demonstrated improved generalization to new geographies, compared to U-Net models<sup>22</sup>, for Landsat-based canopy height mapping<sup>23</sup>.

While ALS provides high fidelity 3D structural information, its global availability is highly uneven. One recent review<sup>13</sup> found that 90% of studies fusing LiDAR data with earth observation data were focused in North America, Europe, and Asia.Because ALS coverage is uneven, models trained from ALS alone can inherit strong geographic priors and under-perform in underrepresented forest types. SSL representations provide a complementary signal by learning broad visual semantics and structure-related features from diverse imagery, improving robustness when supervised ALS training data are sparse or regionally biased. Moreover, ALS acquisitions are typically episodic in time, whereas archived RGB imagery enables retrospective canopy height estimation for historical baselines and change analyses in areas without repeated ALS coverage. While initiatives such as the Global Canopy Atlas<sup>18</sup> may alleviate the data coverage limitation, there remain significant operational challenges to fusing ALS data with optical imagery. Pairing ALS-derived canopy height models with optical imagery introduces several sources of noise. ALS and optical acquisitions often differ in date by months to years, leading to mismatched phenology or land use change between inputs and targets. Even when acquisitions are temporally close, geolocation error and viewing-geometry differences can induce local misregistration. Because ALS-derived CHMs provide fine-scale structural supervision, this misregistration disproportionately degrades the learning of detailed canopy structure. These challenges motivate automated data cleaning and registration procedures that can scale to millions of training pairs.

Tolan *et al.*<sup>24</sup> developed the first high resolution map of global canopy height (hereafter referred to as CHMv1) by training a depth estimation model using a frozen DINOv2<sup>19</sup> backbone as a fixed feature extractor, which reduces the amount of task-specific training required, with high resolution optical input and ALS-derived CHM output. The ALS data consisted of sparse LiDAR transects within the USA from NEON<sup>25</sup> and the model was applied globally with a low-resolution correction factor applied via a secondary CNN trained on GEDI footprints. The quality of this map varies widely across spatial scales, reference canopy height, and degree of heterogeneity in canopy structure. In a large-scale comparison to 3,458 ALS data reference transects, Fischer *et al.*<sup>18</sup> found that while CHMv1<sup>24</sup> compares favorably to medium-resolution global products, residual errors can still be limiting for sub-landscape analyses of canopy height variability. These results are consistent with the conclusions of Moudrý *et al.*<sup>26</sup>, which found that CHMv1<sup>24</sup> had significant underestimation of canopy height across three biodiversity areas in California, New Zealand, and Switzerland. Within a sparser forest system, D’Amico *et al.*<sup>27</sup> found that CHMv1 had comparably high accuracy for mapping canopy height of agroforestry tree systems in Italy.

Together, these limitations highlight the need for a globally consistent, meter-resolution canopy height dataset trained on more representative ALS coverage and supported by modern SSL backbones. Existing global products either lack high spatial resolution, rely on LiDAR supervision concentrated in a few regions, or exhibit substantial geographic bias. To address this gap, we develop a new global 1-meter canopy height dataset that builds directly on the CHMv1<sup>24</sup> framework but incorporates several key advances. We replace the DINOv2-H encoder with the more capable DINOv3 Sat-L backbone, expand and rigorously clean a geographically diverse ALS training corpus, and apply improved RGB-CHM registration to reduce label noise. We further introduce a loss formulation tailored to canopy height distributions and structural variability. Through ablations and evaluations across multiple benchmarks, we show that these improvements substantially reduce bias, increase accuracy across canopy height ranges, and enhance fine-scale structural fidelity relative to previous global products (Figure 1).

## 2 Methods

### 2.1 Overview of Workflow

This paper presents a global 1-meter canopy height map generated from high-resolution satellite imagery with a depth estimation model trained against ALS data. The imagery source is the same as in CHMv1<sup>24</sup>, with approximately 80% of images spanning 2018 to 2020. Compared to CHMv1<sup>24</sup>, we substantially expand and clean the training dataset, improve the model backbone and decoder, and identify an improved training curriculum (loss, dataset mix, optimization parameters). These differences yield substantial qualitative and quantitative accuracy improvements, which are confirmed via rigorous ablations and validation analyses.

### 2.2 Input Imagery

The satellite imagery resolution, acquisition dates, and preprocessing are discussed in detail in the CHMv1 manuscript<sup>24</sup>. In summary, we utilize Maxar Vivid2 mosaic imagery as input imagery for model training and inference. This dataset mosaics together imagery from multiple instruments (WorldView-2, WorldView-3, and Quickbird II) and observation dates, with a pixel size of 0.597 m GSD at the equator.## 2.3 Decoder training and validation datasets

**Figure 2.** Training and testing/validation data locations for NAIP-3DEP and SatLidar datasets.

### 2.3.1 NAIP-3DEP dataset

The 3D Elevation Program (3DEP)<sup>28</sup> collects extensive ALS and topographic data across the United States. Allred *et al.*<sup>29</sup> curated a dataset of approximately 22 million paired 3DEP ALS and NAIP optical chips across the contiguous United States with a  $256 \times 256$  meter chip size. We created a derivative dataset (NAIP-3DEP) based on post-processing, cleaning, and sampling of the Allred *et al.*<sup>29</sup> dataset. First, we selected approximately 360k NAIP-CHM pairs with CHM and NAIP acquisition dates within 60 days of each other. Next, building footprints were set to zero canopy height using the Microsoft US Building Footprint dataset<sup>30</sup>. Approximately 60k CHM tiles with striping, leaf-off acquisition, or evident land use change between input and output were then discarded with an automated method described in Section 2.4.1. Finally, we performed automatic registration of the optical and ALS data as outlined in Section 2.4.2. The remaining samples, about 300k, were split with geographic stratification into approximately 280k training and 20k validation patches. The dataset consists of  $427 \times 427$  NAIP RGB optical images at 0.6 m GSD and ALS-derived CHM at the same pixel resolution but with a 1 m GSD. The CHM data is sampled to 0.6 m GSD with a super-resolution model based on DSen2<sup>31</sup>. The locations of the train and validation data for NAIP-3DEP are displayed in Figure 2.

### 2.3.2 SatLidar dataset

We construct SatLidar v2, a curated and registered revision of the SatLidar v1 dataset introduced in the DINOv3<sup>20</sup> paper. SatLidar v1 consists of approximately one million  $512 \times 512$  images with ALS ground truths split into train/val/test splits with ratios 8/1/1. The splits include the Neon dataset used by CHMv1<sup>24</sup>. SatLidar v2 corresponds to a curated and registered version of SatLidar v1 where poor quality training samples have been discarded (Section 2.4.1) and input and output datasets have been registered (Section 2.4.2). It includes approximately 726K train, 89K val and 91K test samples. The locations of the train, val, and test data for SatLidar v2 are displayed in Figure 2.### 2.3.3 Multisource dataset

The multisource dataset is a test dataset consisting of 433 triplets of images from three different sources (Maxar, Maxar taken at a different date, and NAIP) with associated ALS ground truths. All images are located in the ‘WLOU’ NEON site (Colorado), with ALS observations acquired in 2019<sup>32</sup>. Similar to the other datasets, the multisource dataset was curated following classification with a linear probe, and the same alignment procedure we used for SatLidar v2.

### 2.3.4 NAIP Sea dataset

The datasets described in Sections 2.3.1 to 2.3.3 have spatial footprints determined by the locations of ALS transects, which are sparse or non-existent over open water. However, choppy water can look quite similar to forests with snow on the ground in high resolution imagery leading to erroneous height predictions over water. To remedy this, we randomly sample open water and shorelines within the United States in NAIP imagery based on the Global Lakes and Wetlands Database<sup>33</sup> and create canopy height target values of zero for approximately 3,500 samples. This dataset is included in our data mixes to improve data fidelity over water.

## 2.4 Training data curation

### 2.4.1 Data cleaning

We used DINOv3 as a few-shot anomaly detector to discard poor quality pairs in the training datasets<sup>34</sup>. A weak CHM model, trained on approximately 5k manually reviewed samples, produced preliminary RGB / CHM label / CHM prediction triptychs. We manually annotated "keep" and "discard" labels covering land-use mismatch and CHM sensor artifacts for a small subset ( $\approx 10k$ ) of our data stratified by tree cover and height metrics based on visual comparison of the input image, the weak prediction, and the label. A linear probe was trained on the L1 difference between frozen DINOv3 embeddings (prediction - label) for these samples. The linear probe was applied to all the training corpus and samples with below a 0.5 keep score were removed based on the receiver operating characteristic, excluding about 15% of all data. All removals in the validation and test set were manually visually inspected to avoid erroneous removal of difficult validation or test samples.

### 2.4.2 Data registration

Accurate spatial alignment of RGB imagery and CHM labels is essential for high quality canopy height mapping. Recent monocular depth estimation approaches, such as Depth Anything v2<sup>35</sup>, have found that models trained on synthetic data generate sharper depth predictions than models trained on non-synthetic data. Because optical and ALS data are collected at different times by different sensors, such paired datasets have large amounts of random misalignment due to geolocation and differences in viewing geometries. Without correcting for these misalignments, there is no consistent supervisory signal for detailed local 3D structure. Despite the importance of well aligned RGB and CHM data, reliable cross-domain registration remains challenging in forested systems. Many cross-domain registration pipelines for earth observation rely on photometric consistency, repeatable keypoints, or significant mutual information between domains<sup>36,37</sup>. These assumptions are infrequently met for canopy height mapping datasets because reflectance is not a consistent proxy for height (limiting mutual cross domain information) and forest structure, which is dominated by textured crowns, shadows, and view-illumination effects, is not a reliable source for keypoint identification.

A further complication is that the most reliable way to register images of different domains (CHM labels and RGB imagery) is to compare like with like via domain adaptation<sup>38,39</sup>. However, the practical application of domain adaptation to transforming optical images to a canopy-height representation is made difficult by the varying amount of mutual information between these domains. We found that supervised approaches were necessary for generalization, but such approaches require a reasonably accurate CHM prediction model for domain adaptation, which itself must be trained using well-aligned RGB–CHM pairs. If the initial supervision is misaligned, the model converges to blurred and conservative predictions that are unsuitable for reliable registration. To break this chicken-and-egg dependency, we bootstrap alignment using auxiliary structural cues from an independent tree detection model together with a weak CHM model trained on a carefully cleaned subset of the data (Figure 3a).

We correct local misalignment using tree detections as control points. A DINO DETR<sup>40</sup> detector trained on human annotations from MillionTrees<sup>41</sup> produces bounding boxes for individual trees for each input RGB image. Because these tree predictions are derived from a model trained on manually annotated images, they do not exhibit misalignment between input**(a)** Global best shift magnitude for SatLidar and NAIP-3DEP. NAIP imagery is much better registered, as is the ALS data. SatLidar has many input data sources and satellites, and the raw data is much more poorly aligned.

**(b)** Summary of our alignment methodology for local alignment (top) with bounding box proxies and global alignment with weak predictions (bottom).

**Figure 3.** Data registration methodology and analysis.

and label. Within each predicted tree box, we compute one or more canopy height "centers of mass" (COMs). The per-tree offsets from box center to COM are clustered with DBSCAN<sup>42</sup>, regularized with per-cluster medians to resolve disagreements, and interpolated to a dense warp field using a thin-plate radial basis function. Alignment is performed iteratively: offsets are estimated, a small warp is composed into a velocity field, and CHM predictions are re-measured until the incremental offset is smaller than 1 pixel.

We correct global misalignment by estimating a rigid 2D translation between predictions and labels with Fast Fourier Transform (FFT)-based cross-correlation<sup>43</sup>. For each tile, we first apply a lightweight "weak" canopy height model, trained on data with local misalignments corrected via the detection-based control points, to both the satellite imagery and the ALS CHM to place both inputs and labels in a common radiometric space suitable for automated alignment. We then detect high-canopypeaks and compute an initial alignment via FFT-based cross-correlation of peak masks, followed by a small local grid search for refinement. Candidate shifts are scored with IoU of high-canopy regions and a height "lift" term calculated as the median delta in canopy height of the shifted prediction at each ground truth peak. The shifts are only applied when they improve IoU or height lift related to a zero-shift baseline. Different steps of the registration process are illustrated in Figure 3b.

The final training dataset is composed of the original optical data with CHM data that has been first aligned globally with a per-tile offset, and then locally with a dense warp field, to the optical data. Overall, 85% of training samples with non-zero canopy height targets benefit from the alignment. The SatLidar dataset had much more severe misalignment (53% of samples with  $\geq 5$  px shift) than did the NAIP-3DEP dataset (6.4% of samples with  $\geq 5$  px shift) (Figure 3a). The alignment process improved the average IoU of the "weak" canopy height model for the per-tile p80 canopy height from 0.31 to 0.42.

## 2.5 Decoder loss functions

Canopy height mapping differs from conventional monocular depth estimation in both viewing geometry and depth statistics. Typical monocular depth tasks are trained on ground-based or oblique camera views, emphasize near-camera depth accuracy, and contain few zero-depth regions. In contrast, canopy height estimation is performed from a nadir, overhead viewing angle and includes large areas of zero or near-zero height, while simultaneously requiring accurate prediction of tall, spatially sparse structures. These characteristics alter the distribution of prediction errors and the relative importance of different error modes. We therefore evaluate several loss formulations with regards to MAE (Mean Absolute Error), block- $R^2$  with  $50 \times 50$  pixels blocks, bias, and high depth ( $\geq 30$  meter) mean bias error.

**SiLog loss.** We evaluate the SiLog loss<sup>44</sup> function commonly used in depth estimation approaches such as DepthAnythingv2<sup>35</sup>, DepthPro<sup>45</sup>, and CHMv1<sup>24</sup>. We find that training with SiLog loss alone for the task of canopy height mapping is insufficient because canopy height maps have a significant proportion of zero or near-zero values, which can dominate the objective in log space and reduce emphasis on errors in tall, spatially sparse canopy, leading to negative bias at high canopy height.

**Charbonnier loss.** Because training with SiLog loss resulted in large high-depth mean bias error, we evaluate model results when training with Charbonnier loss<sup>46</sup>, a smooth variant of the L1 loss. These models converge to trivial solutions by predicting the batch mean value. As such, we consider the effect of starting model training with SiLog loss to establish relative depth structure and gradually transitioning to Charbonnier loss during training to encourage the model to have correct predictions in linear space.

**Gradient loss.** When training depth estimation models with L1 variants, it is common to utilize a gradient loss with Sobel or Laplacian operators to penalize deviations in spatial structure<sup>35,45,47</sup>. Notably, MiDaS<sup>48</sup> introduced a multi-scale, scale- and shift-invariant gradient loss that operates in log-depth space and is evaluated across multiple spatial resolutions of both the prediction and ground-truth depth. However, per-pixel gradient loss does not always provide the expected benefits. For instance, DepthAnything v2<sup>35</sup> found that gradient matching loss improves depth sharpness when trained on synthetic data but fails to bring improvement when trained on labeled real datasets. Similarly, we find that directly applying MiDaS-style<sup>48</sup> gradient matching loss to our training paradigm does not improve sharpness. We hypothesize this is due to the persistence of small misalignment between the input and output that dampen the fine-grained supervision signal for gradient loss.

**Patch Gradient loss.** To remedy this, we construct a gradient loss that measures structural consistency across multiple spatial scales while remaining robust to local translation and scale differences (Algorithm 1). Following from Depth Pro<sup>45</sup>, prediction and target maps are pre-processed using a log-space mean-centering transformation to eliminate sensitivity to global scale or offset. Spatial derivatives are then computed using horizontal, vertical, and diagonal Sobel operators. At the pixel level, we compute the absolute difference between predicted and target gradients. To capture higher-order spatial context and reduce sensitivity to small misalignment, we apply a differentiable range pooling operator that estimates a min-max range of Sobel gradients within  $3 \times 3$  and  $5 \times 5$  patches using softmax with a temperature hyperparameter. The complete loss combines pixel-wise gradient magnitude, patch-wise gradient range loss, and a unit-direction consistency to align the gradient orientations between prediction and target. The losses are computed at multiple image resolutions ( $1.0 \times, 0.5 \times, 0.25 \times$ ) similar to MiDaS-style gradient loss<sup>48</sup>.**Final loss.** The final loss is the combination of SiLog loss, progressively annealed and replaced by a Charbonnier loss, with the progressive addition of the Patch Gradient loss at mid training. The parameters of this curriculum loss, that we denote Cur, are provided in the next section.

## 2.6 Decoder architecture, training details and data sampling

The DINOv3 paper experimented with the same decoder as CHMv1<sup>24</sup>, modified with a larger image resolution of  $448 \times 448$  instead of  $256 \times 256$ , resulting in a large boost in the performance of all models (from 0.6 to 0.8 block  $R^2$ ) on SatLidar v1. When introducing our new loss function to this setting, we encountered instability issues that motivated some of the architecture changes described below.

**Architecture changes.** We bring slight architecture changes to the decoder compared to CHMv1. The most important change is the modification of the binning strategy from a linear scale to a mix of linear and log scale. Compared to CHMv1, where the CHMs ground truths were divided by 10, we use a factor of 8. We use a maximum depth of 96m/8 for the decoder predictions instead of 80m/10. We use bias in the residual layers. In the up-sampling head of the decoder (UpConvHead), we use Kaiming initialization. We remove the extra  $1 \times 1$  convolution projection layer of the decoder. We increase the dimension of the final hidden layer of the UpConvHead from 32 to 128. We use intermediate backbone layers [5, 11, 17, 23] instead of [4, 11, 17, 23]. The backbone norm is set to true following the DINOv3 defaults.

**Training details.** We use a similar training loop to DINOv3 depth training<sup>49</sup>, with a learning rate of  $10^{-4}$ ,  $\beta_1 = 0.9$ ,  $\beta_2 = 0.99$ , weight decay of 0.001, 100k iterations, a total batch size of 16 (2 with 8 GPU), and cosine LR scheduler with 6k linear warmup iterations. In the curriculum loss, the SiLog loss is progressively replaced by Charbonnier loss during the first 30k iterations. The Patch Gradient loss is linearly warmed up with a weight from 0 to 0.075 between iterations 5k and 50k.

**Data sampling.** We sample data from the concatenation of our curated NAIP-3DEP and SatLidar v2 datasets. We also implement batch category sampling, to ensure each batch contains some proportion of small and tall trees. The category sampling ratios were determined based on an ablation study [10% of GT below 1m, 20% above 35m] for SatLidar and [10% of GT below 1m, 10% above 25m] for NAIP-3DEP.

## 3 Data Record

The CHMv2 dataset covers nearly the entirety of global land area (except Greenland and Antarctica) with canopy height values encoded in integer meters for each pixel. The 22.65 terabyte dataset is released in 213,109 cloud-optimized geotiffs (COGs) that are  $32,768 \times 32,768$  pixel in size, with a 1.2m pixel spacing at the equator. Each COG covers approximately 1,500 km<sup>2</sup> at the equator and about 65 km<sup>2</sup> near the poles. Pixels where the input imagery was occluded by clouds, as specified by the Maxar Vivid metadata, are encoded by the no data value. Each file is released in a Pseudo-Mercator projection (EPSG 3857).

The CHMv2 dataset is available via Amazon Web Services (AWS)<sup>1</sup> under the DINOv3 license.<sup>2</sup> It is also released on a Google Earth Engine viewer<sup>3</sup>. A global lookup file listing tile names and bounding boxes is included in the AWS repository. We additionally release a global GeoTIFF of input image acquisition date, where pixel values encode year minus 2000 (e.g., 18.25 indicates April 2018).

## 4 Technical Validation

### 4.1 Improvements versus CHMv1

The presented dataset has large overall performance gains relative to CHMv1<sup>24</sup> measured against our ALS-derived test datasets. On the SatLidar test dataset, the MAE improves from 4.3m to 3.0m, with  $R^2$  improvements from 0.53 to 0.86 (Table 2, Figure 4)

<sup>1</sup>[s3://dataforgood-fb-data/forests/v2/global/dinov3\\_global\\_chm\\_v2\\_ml3/chm/](https://dataforgood-fb-data/forests/v2/global/dinov3_global_chm_v2_ml3/chm/)

<sup>2</sup><https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md>

<sup>3</sup><https://meta-forest-monitoring-okw37.projects.earthengine.app/view/canopyheight>**Figure 4.** Compared to CHMv1 on NAIP-3DEP (top) and SatLidar v2 (bottom), CHMv2 exhibits greatly reduced biases, especially for trees above 30 m. The inset figures plot the 95th percentiles of CHM on  $50 \times 50$  crops (y axis) as a function of the 95th percentiles of Ground Truths (x axis). Scales are in meters.

compared to CHMv1. The CHMv2 model and data also has a marked improvement in mean bias error at high ( $\geq 30\text{m}$ ) height (Figure 4). We still note area for improvement in very high ( $\geq \text{p}98$ ) canopy estimation, which are still underestimated. When qualitatively compared with CHMv1, the presented dataset has noticeable improvements in crown delineation, characterization of canopy gaps and complex canopy structures, and improved sharpness of canopy edges (Figure 5). Finally, CHMv2 performs much better than CHMv1 on generating consistent CHM predictions across multiple sources of imagery of the same location (Figure 6).

## 4.2 Comparison with other global datasets

We compare CHMv2 against existing low-resolution global canopy height products, including Potapov *et al.*<sup>14</sup>, Paul *et al.*<sup>50</sup>, and Lang *et al.*<sup>15</sup>, by evaluating all products on the SatLidar v2 ALS-derived test set (Table 1). This provides a direct comparison between CHMv2 and prior global canopy height products on a shared meter-scale ALS reference. The compared low-resolution products are trained using spaceborne LiDAR supervision (e.g., GEDI) and are primarily evaluated in their original manuscripts against held-out GEDI footprints. In contrast, CHMv2 (and CHMv1) is trained solely on ALS-derived canopy height maps. We therefore evaluate all products against the same ALS-derived reference data to ensure a consistent comparison on our test distribution. CHMv2 achieves the lowest error on the SatLidar v2 ALS-derived test split (MAE = 3.0 m), improving over the evaluated low-resolution global products (4.9 - 8.4 m). A visual comparison of the evaluated datasets demonstrates the improved capability of CHMv2 for measuring forest structure when compared to ALS-derived canopy height (Figure 7).**Figure 5.** Qualitative improvements over CHMv1, in terms of sharpness and accuracy. Comparison on the NAIP-3DEP (two top lines) and SatLidar v2 (bottom lines) on  $256 \times 256$  samples. CHMv1 is without GEDI correction.**Figure 6.** Comparison of CHMv1 and CHMv2 on the Multisource dataset. CHMv1 is without GEDI correction.

<table border="1">
<thead>
<tr>
<th>Dataset</th>
<th>Resolution (m)</th>
<th>MAE</th>
</tr>
</thead>
<tbody>
<tr>
<td>Pauls <i>et al.</i><sup>50</sup></td>
<td>10</td>
<td>7.5</td>
</tr>
<tr>
<td>Lang <i>et al.</i><sup>15</sup></td>
<td>10</td>
<td>8.4</td>
</tr>
<tr>
<td>Potapov <i>et al.</i><sup>14</sup></td>
<td>30</td>
<td>4.9</td>
</tr>
<tr>
<td>CHMv1<sup>24</sup></td>
<td>1</td>
<td>4.3</td>
</tr>
<tr>
<td>CHMv2</td>
<td>1</td>
<td><b>3.0</b></td>
</tr>
</tbody>
</table>

**Table 1.** Comparison of global canopy height products to ground truth ALS-derived CHM data in the SatLidar v2 test split.

### 4.3 Comparison with GEDI and ICESat-2

To evaluate agreement between CHMv2 and canopy height from spaceborne LiDAR, we compiled reference observations from GEDI (2019–2022) and ICESat-2 (2018–2023). We used the footprint-level RH98 metric from the GEDI L2A product<sup>51</sup> and the 98th percentile canopy height for 20-m along-track segments from ICESat-2 ATLAS ATL08\_V7<sup>52</sup>. Only high-quality observations were retained. For GEDI, we selected full-power beam, nighttime, leaf-on shots with sensitivity  $\geq 0.95$ , the highest quality flag, and a difference between the DEM and lowest detected elevation  $< 150$  m. For ICESat-2, we selected strong-beam nighttime segments and excluded observations flagged for snow/ice, aerosols, or clouds.

To ensure geographically balanced sampling, we randomly extracted 10,000 GEDI observations and 5,000 ICESat-2 observations per  $1^\circ \times 1^\circ$  grid cell. For each reference observation, we computed the 98th percentile of CHMv2 within a 12 m radius (GEDI) or 7 m radius (ICESat-2). We excluded samples where tree cover loss or gain<sup>14,53</sup> indicated potential forest change between the Maxar acquisition and the ALS observation date. To reduce sensitivity to Maxar image quality and seasonality, we also excluded samples with cloud contamination, sun elevation  $< 45^\circ$ , or off-nadir angle  $> 25^\circ$ . Additional exclusions removed samples where (i) the tree-cover edge intersected the footprint, or (ii) the reference canopy height was inconsistent with existing land cover and tree height products<sup>14,54</sup>. These quality and consistency filters removed 66% of GEDI samples and 78% of ICESat-2 samples, reflecting strict screening for cloud contamination, acquisition geometry, and temporal mismatch between imagery and reference observations. For the remaining observations, reference height was set to zero within**Figure 7.** Canopy height profiles along a single 10-m wide transect comparing ALS reference measurements to five canopy height datasets. Gray points show ALS returns within the transect corridor; the black line shows the ALS-derived canopy height profile. ALS data is from Kalimantan, Indonesia ( $1.6412^\circ$ ,  $115.281^\circ$ ) and is  $\geq 200$  km from any observation within the SatLidar v2 dataset.urban, bare ground, and water classes<sup>54</sup> to reduce contamination from non-woody vertical structures.

(a) Comparison of CHMv2 98<sup>th</sup> percentile with GEDI Relative Height (RH) at the 98<sup>th</sup> percentile.

(b) Comparison of CHMv2 98<sup>th</sup> percentile canopy height (y-axis) against ICESat-2 98<sup>th</sup> percentile relative height (x-axis).

**Figure 8.** Comparison with GEDI and ICESat-2.

We evaluated agreement using 37M GEDI and 11M ICESat-2 reference samples. Global comparison against GEDI yielded an  $R^2$  of 0.70, MAE of 3.1 m, and RMSE of 6.4 m. Regional comparisons (Figure 8a) showed consistently high agreement across major forest biomes ( $R^2 = 0.68$ – $0.76$ ), with higher uncertainty in sparsely treed landscapes. Global comparison against ICESat-2 yielded similar agreement with an MAE of 2.9 m and RMSE of 5.63 m, but a lower  $R^2$  of 0.60. The lower  $R^2$  may partially be explained by the smaller footprint size of ICESat-2 introducing more random geolocation noise. Regional comparisons of ICESat-2 (Figure 8b) show strong agreement in boreal and temperate regions with slightly lower agreement in the tropics, while the regional comparisons with GEDI show relatively stronger agreement in the tropics.

GEDI and ICESat-2 provide the only globally consistent reference canopy height observations, but their estimates are not strict ground truth. Retrieval accuracy is affected by slope, canopy density<sup>55</sup>, and non-woody vertical structures<sup>56</sup>. Relative to airborne laser scanning, ICESat-2 generally exhibits higher canopy height uncertainty than GEDI<sup>55</sup>, but it provides coverage in high-latitude regions not observed by GEDI. Both sensors are also affected by geolocation uncertainty: mean GEDI geolocation error is  $\approx 10$  m<sup>57</sup>, while ICESat-2 error is  $\approx 2.5$ – $4.4$  m<sup>58</sup>. Given the mean Maxar geolocation accuracy of 8.7 m for this dataset, compound Maxar–lidar geolocation errors may substantially affect agreement within individual footprints.#### 4.4 Ablation results

Our ablation studies identify the relative contributions of training data quality, migration from DINOv2 to DINOv3, loss design, and training curriculum on our qualitative and quantitative results. We employ the same metrics as Tolan *et al.*<sup>24</sup> to measure ablation results, namely MAE, block- $R^2$ , bias, and Edge Error, which measures the difference between the Sobel gradient of the GT and prediction.

After comparing results using a ViT-L and larger models in the DINOv3 study<sup>20</sup> on SatLidar v1, we concluded that using larger backbones would lead to very similar results with significant compute overhead. Specifically, there was no performance improvement when using the CHMv1 ViT-H backbone instead of ViT-L. Therefore, we use only ViT-L backbones for CHMv1 and our approach in this section. We compared the different models without post-processing (no GEDI correction step<sup>24</sup>).

<table border="1">
<thead>
<tr>
<th rowspan="2">Backbone</th>
<th rowspan="2">Setting</th>
<th rowspan="2">Loss</th>
<th rowspan="2">Sampling</th>
<th rowspan="2">Train data</th>
<th colspan="4">SatLidar v2</th>
<th colspan="3">NAIP-3DEP</th>
</tr>
<tr>
<th>MAE</th>
<th><math>R^2</math></th>
<th>Bias</th>
<th>Edge</th>
<th>MAE</th>
<th><math>R^2</math></th>
<th>Edge</th>
</tr>
</thead>
<tbody>
<tr>
<td>CHMv1</td>
<td>Base</td>
<td>SiLog</td>
<td>Uniform</td>
<td>Neon train</td>
<td>4.3</td>
<td>0.53</td>
<td>2.6</td>
<td>0.67</td>
<td>2.2</td>
<td>0.54</td>
<td>0.77</td>
</tr>
<tr>
<td>DINOv3</td>
<td>Base</td>
<td>SiLog</td>
<td>Uniform</td>
<td>Neon train</td>
<td>3.9</td>
<td>0.64</td>
<td>2.2</td>
<td>0.66</td>
<td>2.0</td>
<td>0.59</td>
<td>0.72</td>
</tr>
<tr>
<td>DINOv3</td>
<td>Base</td>
<td>SiLog</td>
<td>Uniform</td>
<td>SatLidar v1 train</td>
<td>3.5</td>
<td>0.78</td>
<td>0.9</td>
<td>0.62</td>
<td>2.0</td>
<td>0.46</td>
<td>0.63</td>
</tr>
<tr>
<td>DINOv3</td>
<td>Base</td>
<td>SiLog</td>
<td>Uniform</td>
<td>SatLidar v2 train</td>
<td>3.5</td>
<td>0.76</td>
<td>0.5</td>
<td>0.58</td>
<td>2.1</td>
<td>0.52</td>
<td>0.63</td>
</tr>
<tr>
<td>DINOv3</td>
<td>Final</td>
<td>Cur.</td>
<td>Uniform</td>
<td>SatLidar v2 train</td>
<td>3.5</td>
<td>0.75</td>
<td>-0.5</td>
<td>0.55</td>
<td>2.2</td>
<td>0.48</td>
<td>0.58</td>
</tr>
<tr>
<td>DINOv3</td>
<td>Final</td>
<td>Cur.</td>
<td>Cat</td>
<td>3DEP, SatLidar v2 train</td>
<td>3.1</td>
<td>0.85</td>
<td>0.1</td>
<td>0.55</td>
<td>1.4</td>
<td><b>0.94</b></td>
<td><b>0.53</b></td>
</tr>
<tr>
<td>DINOv3</td>
<td>Final</td>
<td>Cur.</td>
<td>Cat</td>
<td>3DEP, SatLidar v2 trainval</td>
<td><b>3.0</b></td>
<td><b>0.86</b></td>
<td><b>0.0</b></td>
<td><b>0.53</b></td>
<td><b>1.4</b></td>
<td>0.93</td>
<td><b>0.53</b></td>
</tr>
</tbody>
</table>

**Table 2.** Main ablation from the CHMv1 settings to the CHMv2 settings, demonstrating improvements due to the DINOv3 backbone, training data diversity, registration, new architecture and loss. The base setting corresponds to the CHMv1 architecture and training parameters, and the final setting to the changes described in Section 2.6.

Table 2 summarizes the improvements from CHMv1 using DINOv2. The first two lines compare models trained on Neon, demonstrating improvements brought by training the DINOv3 backbone in all metrics, e.g. from 0.53 to 0.64 of  $R^2$ , and significantly sharper results on NAIP-3DEP. Second, the replacement of the Neon data by the more diverse SatLidar v1 data results in another large boost of  $R^2$  on SatLidar v2 (0.64 to 0.78); however, we also note a drop of overall accuracy on the out of domain performance on NAIP-3DEP, except an improved edge error. Third, the registration and quality filtering of the SatLidar dataset leads to a reduction of the edge error on SatLidar v2 and improvement in  $R^2$  on NAIP-3DEP. Changes in architecture, training setting, and loss function bring only improvements in terms of edge error. The combination of aerial and satellite datasets for training leads to drastic improvements in all metrics (e.g. from 0.48 to 0.94 of  $R^2$  on NAIP-3DEP, and from 0.75 to 0.85 on SatLidar v2). Our final model was trained on this data mix, augmented by the NAIP Sea dataset described above, and the val set of the SatLidar v2 dataset, leading to a small boost on SatLidar v2.

Table 3 details that training decoders with larger image resolution leads to important gains, e.g.  $R^2$  from 0.73 to 0.81. We also note the large improvement of bias using the DINOv3 backbone on the out of domain NAIP-3DEP aerial dataset.

<table border="1">
<thead>
<tr>
<th rowspan="2">Backbone</th>
<th rowspan="2">Decoder data</th>
<th colspan="3">SatLidar v1</th>
<th colspan="3">NAIP-3DEP</th>
</tr>
<tr>
<th>MAE</th>
<th><math>R^2</math></th>
<th>Bias</th>
<th>MAE</th>
<th><math>R^2</math></th>
<th>Bias</th>
</tr>
</thead>
<tbody>
<tr>
<td>CHMv1</td>
<td>Neon 256<math>\times</math></td>
<td>4.0</td>
<td>0.61</td>
<td>3.7</td>
<td>2.21</td>
<td>0.54</td>
<td>1.1</td>
</tr>
<tr>
<td>CHMv1</td>
<td>Sat v1 256<math>\times</math></td>
<td>3.6</td>
<td>0.73</td>
<td>1.2</td>
<td>2.21</td>
<td><b>0.59</b></td>
<td>0.5</td>
</tr>
<tr>
<td>CHMv1</td>
<td>Sat v1 448<math>\times</math></td>
<td>3.4</td>
<td><b>0.81</b></td>
<td>0.9</td>
<td>2.32</td>
<td>0.52</td>
<td>0.9</td>
</tr>
<tr>
<td>DINOv3</td>
<td>Sat v1 448<math>\times</math></td>
<td><b>3.2</b></td>
<td><b>0.81</b></td>
<td><b>0.8</b></td>
<td><b>2.17</b></td>
<td>0.52</td>
<td><b>0.0</b></td>
</tr>
</tbody>
</table>

**Table 3.** (a) Backbone ablation. Factors of improvements: **more diverse training data, larger decoder training resolution, improved DINOv3 backbone.** Loss: SiLog.

<table border="1">
<thead>
<tr>
<th rowspan="2">Backbone</th>
<th rowspan="2">Setting</th>
<th rowspan="2">Loss</th>
<th colspan="4">NAIP-3DEP</th>
</tr>
<tr>
<th>MAE</th>
<th><math>R^2</math></th>
<th>Bias</th>
<th>Edge</th>
</tr>
</thead>
<tbody>
<tr>
<td>DINOv3</td>
<td>Base</td>
<td>SiLog</td>
<td>1.61</td>
<td>0.82</td>
<td>1.0</td>
<td>0.55</td>
</tr>
<tr>
<td>DINOv3</td>
<td>100k iter</td>
<td>SiLog</td>
<td>1.55</td>
<td>0.84</td>
<td>0.9</td>
<td>0.54</td>
</tr>
<tr>
<td>DINOv3</td>
<td>Final</td>
<td>SiLog</td>
<td>1.55</td>
<td>0.84</td>
<td>0.9</td>
<td>0.54</td>
</tr>
<tr>
<td>DINOv3</td>
<td>Final</td>
<td>Cur.</td>
<td><b>1.38</b></td>
<td><b>0.94</b></td>
<td><b>0.1</b></td>
<td><b>0.53</b></td>
</tr>
</tbody>
</table>

**Table 4.** (b) Decoder parameters ablation with decoders trained on the NAIP-3DEP dataset with 416 $\times$  samples. Factors of improvement: **longer training and loss.**

Table 4 shows that moving from 38K to 100K training iterations is helpful. The architecture changes do not affect performance on this dataset; however, the loss brings a large boost in all metrics. As shown in Table 5, the architecture changes help or do not degrade performance on SatLidar v2. Table 6 explores different loss alternatives. Using a SiLog loss alone results in a large bias and relatively low  $R^2$ . Adding a gradient loss in an attempt to obtain sharper results mildly improves**Figure 9.** Impact of our loss, a Combination of SiLog, Charbonnier and Multi-scale Patch Gradient on the NAIP-3DEP results. Compared to SiLog loss, results are sharper and more accurate.

results but leads to larger edge error. Combining SiLog and Charbonnier greatly reduces the bias from 0.9 to 0.1; however results are still blurry as shown by the large 0.54 edge error. Adding a gradient loss term does not solve the problem, and neither does a patch gradient loss that creates grid artifacts. Our curriculum loss combines SiLog, Charbonnier and only starts enforcing a Patch Gradient loss at mid training, avoiding artifacts while yielding sharper predictions (Figure 9). Finally, Table 7 studies the importance of using category sampling and different aerial / satellite data ratios used in decoder training.

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="3">SatLidar v2</th>
</tr>
<tr>
<th>MAE</th>
<th>Bias</th>
<th><math>R^2</math></th>
</tr>
</thead>
<tbody>
<tr>
<td>Final Decoder arch. : setting F</td>
<td><b>3.15</b></td>
<td>-0.3</td>
<td><b>0.84</b></td>
</tr>
<tr>
<td>F + project</td>
<td><b>3.15</b></td>
<td>-0.3</td>
<td><b>0.84</b></td>
</tr>
<tr>
<td>F - bias in residual layers</td>
<td>3.16</td>
<td><b>-0.0</b></td>
<td>0.83</td>
</tr>
<tr>
<td>F - K. init in UpConvHead</td>
<td>3.20</td>
<td>-0.1</td>
<td>0.82</td>
</tr>
<tr>
<td>F - larger dim of UpConvHead</td>
<td>3.16</td>
<td><b>-0.0</b></td>
<td>0.83</td>
</tr>
<tr>
<td>F - backbone norm</td>
<td>3.16</td>
<td>-0.1</td>
<td>0.83</td>
</tr>
<tr>
<td>F - mixed bin mixing</td>
<td>3.28</td>
<td><b>-0.0</b></td>
<td>0.80</td>
</tr>
</tbody>
</table>

**Table 5.** Most architecture changes do not affect performance except bin mixing and Kaiming initialization. Models trained on SatLidar v2, 3DEP mix with Charbonnier and Gradient loss.

<table border="1">
<thead>
<tr>
<th rowspan="2">Loss</th>
<th colspan="5">NAIP-3DEP</th>
</tr>
<tr>
<th>MAE</th>
<th>Bias</th>
<th><math>R^2</math></th>
<th>Edge</th>
<th>No Artifacts</th>
</tr>
</thead>
<tbody>
<tr>
<td>SiLog (S)</td>
<td>1.55</td>
<td>0.9</td>
<td>0.84</td>
<td>0.536</td>
<td>✓</td>
</tr>
<tr>
<td>S + Grad</td>
<td>1.51</td>
<td>0.8</td>
<td>0.85</td>
<td>0.557</td>
<td>✓</td>
</tr>
<tr>
<td>S + Char</td>
<td>1.35</td>
<td><b>0.1</b></td>
<td><b>0.94</b></td>
<td>0.541</td>
<td>✓</td>
</tr>
<tr>
<td>S + Char + Grad</td>
<td><b>1.31</b></td>
<td>0.2</td>
<td><b>0.94</b></td>
<td>0.542</td>
<td>✓</td>
</tr>
<tr>
<td>S + Char + Patch Grad</td>
<td>1.42</td>
<td><b>0.1</b></td>
<td><b>0.94</b></td>
<td>0.507</td>
<td>✗</td>
</tr>
<tr>
<td>Cur. + Char + Patch Grad</td>
<td>1.38</td>
<td><b>0.1</b></td>
<td><b>0.94</b></td>
<td><b>0.527</b></td>
<td>✓</td>
</tr>
</tbody>
</table>

**Table 6.** Loss ablation on the NAIP-3DEP dataset. The combination of SiLog, Charbonnier and Patch gradient loss is the best compromise between accuracy and sharpness but suffers from artifacts, removed by the curriculum learning.

## 5 Usage Notes

CHMv2 can be used either as a global meter-scale canopy height product, or as a pretrained model that can be applied to user-provided high-resolution imagery. Because CHMv2 is derived from single-date optical imagery, users should account for variability in image acquisition date, viewing geometry, and atmospheric condition. Although we apply cloud mask filtering, localized artifacts (e.g., haze, missed clouds, or seams between neighboring acquisitions) may persist in some regions.

When using CHMv2 to compute canopy height statistics, we recommend masking non-vegetated areas such as open water, built-up areas, or bare ground using an independent land cover map. CHMv2 can support a range of downstream analyses,<table border="1">
<thead>
<tr>
<th rowspan="2">Category sampling</th>
<th rowspan="2">NAIP-3DEP</th>
<th rowspan="2">SatLidar v2</th>
<th colspan="2">SatLidar v2</th>
<th colspan="2">NAIP-3DEP</th>
<th colspan="2">Multisource</th>
</tr>
<tr>
<th>MAE</th>
<th><math>R^2</math></th>
<th>MAE</th>
<th><math>R^2</math></th>
<th>MAE</th>
<th><math>R^2</math></th>
</tr>
</thead>
<tbody>
<tr>
<td><b>x</b></td>
<td>50%</td>
<td>50%</td>
<td><b>3.07</b></td>
<td>0.83</td>
<td>1.37</td>
<td><b>0.93</b></td>
<td>2.98</td>
<td>0.77</td>
</tr>
<tr>
<td><b>✓</b></td>
<td>50%</td>
<td>50%</td>
<td><b>3.07</b></td>
<td><b>0.84</b></td>
<td>1.36</td>
<td><b>0.93</b></td>
<td>2.98</td>
<td>0.78</td>
</tr>
<tr>
<td><b>✓</b></td>
<td>30%</td>
<td>70%</td>
<td>3.10</td>
<td>0.83</td>
<td>1.38</td>
<td><b>0.93</b></td>
<td><b>2.94</b></td>
<td><b>0.82</b></td>
</tr>
<tr>
<td><b>✓</b></td>
<td>70%</td>
<td>30%</td>
<td>3.18</td>
<td>0.82</td>
<td><b>1.35</b></td>
<td><b>0.93</b></td>
<td>2.98</td>
<td>0.79</td>
</tr>
</tbody>
</table>

**Table 7.** Data mix ablation: using a 50-50 ratio of data source brings a good compromise. Category sampling shows slightly improved metrics. To complement this ablation, we computed that the category sampling improves the MAE of trees above 35 meters by at least two meters. Using DINOv3 viT-L backbone, Charbonnier + Gradient Loss.

including tree cover and land cover mapping, forest type segmentation, carbon and biomass estimation, assessments of forest structural diversity, and monitoring of tree-based land uses such as plantations and agroforestry systems.

In addition to the canopy height product, we release the trained CHMv2 model to enable inference on user-provided satellite or aerial imagery at approximately 0.6 m ground sampling distance (GSD). For applications requiring temporal change detection, users should preferentially compare imagery acquired under similar seasonal and illumination conditions.

## 5.1 Example applications

CHMv2 provides a complementary structural signal that can improve land use and land cover characterization when combined with spectral and texture features. For example, in agroforestry systems, height heterogeneity and multi-strata canopy structure can serve as indicators of shade tree presence and management intensity. Similarly, canopy height metrics can provide useful proxies for forest successional stage or stand development, supporting classification of secondary forest and inference about regrowth trajectories<sup>26,59</sup>. More broadly, CHMv2 can be used to derive spatial covariates for biomass estimation, stratification layers for sampling design, and structural metrics such as canopy height percentiles, gap fraction, and within-polygon height variability.

## 5.2 Limitations

**Temporal and acquisition constraints** CHMv2 is derived from single-date imagery, where the acquisition process selects the best available image within a target period (2017 -2020). This limits the direct use of the released CHMv2 data for attributing canopy height to a specified year of interest. To support change applications, we provide the image acquisition date associated with each prediction in the dataset metadata.

CHMv2 data users should account for possible local tree detection and height modeling errors due to input image quality issues. Cloud and haze contamination can preclude accurate CHM modeling, and about 10% of input images were affected by cloud presence. However, cloud masking is imperfect, and residual clouds may further degrade map quality. High off-nadir viewing geometry can also reduce fidelity by inducing map over-generalization and displacement of tree footprints (see example in Figure 6). We estimated that  $\approx 20\%$  of the Maxar images used for this study had off-nadir angles exceeding  $25^\circ$ . Low sun elevation may further affect CHM prediction. In temperate and boreal regions, low sun angles often correspond to winter-season imagery when deciduous trees are leafless; long shadows cast under these conditions may additionally reduce performance. Analysis of forested samples above  $40^\circ\text{N}$  using ICESat-2 reference data suggests that sun elevation is a strong predictor of map quality. CHMv2 values from input images with sun elevation  $< 45^\circ$  show lower agreement ( $R^2=0.42$ , RMSE=7.4 m) with ICESat-2 than areas with sun elevation  $\geq 50^\circ$  ( $R^2=0.6$ , RMSE=6.5 m). Approximately one-third of the Maxar images used for our map have sun elevations below  $45^\circ$ , indicating that a substantial portion of CHMv2 coverage in boreal and temperate forests may be affected by suboptimal acquisition conditions.

**Gaps and biases in training data** Although CHMv2 shows improved agreement relative to CHMv1, the geographic distribution of ALS training data is uneven and concentrated in a limited set of geographies. While our validation results demonstrate strong agreement with independent spaceborne LiDAR references (GEDI and ICESat-2), some land cover types and forest structures may exhibit reduced accuracy if they are poorly represented in the training distribution.---

**Algorithm 1** Gradient loss

---

```
1: Inputs: prediction  $P \in \mathbb{R}^{H \times W}$ , target  $T \in \mathbb{R}^{H \times W}$ , optional mask  $M$ 
2: Hyperparams: small  $\epsilon$ , weights  $\lambda_{\text{mag}}, \lambda_{\text{rng}}, \lambda_{\text{dir}}$  (defaults: 0.3, 0.6, 0.4)
3: Kernels: Sobel  $G_x, G_y$  ▷ standard  $3 \times 3$  filters
4: function GRADLOSS( $P, T, M$ )
5:   if  $M$  is not provided then
6:      $M \leftarrow \mathbf{1}_{H \times W}$ 
7:   end if
8:   (1) Log mean-center
9:    $\hat{P} \leftarrow \log(\max(P, \epsilon)) - \text{mean}(\log(\max(P, \epsilon)) | M)$ 
10:   $\hat{T} \leftarrow \log(\max(T, \epsilon)) - \text{mean}(\log(\max(T, \epsilon)) | M)$ 
11:  (2) Gradients
12:   $\mathbf{g}_P \leftarrow [G_x * \hat{P}, G_y * \hat{P}]$ ,  $\mathbf{g}_T \leftarrow [G_x * \hat{T}, G_y * \hat{T}]$ 
13:   $m_P \leftarrow \|\mathbf{g}_P\|_2$ ,  $m_T \leftarrow \|\mathbf{g}_T\|_2$ 
14:  (3) Pixel magnitude loss
15:   $L_{\text{mag}} \leftarrow \text{mean}_M [|m_P - m_T|]$ 
16:  (4)  $3 \times 3$  and  $5 \times 5$  range matching
17:   $r_P \leftarrow \text{maxpool}_{3 \times 3}(m_P) - \text{minpool}_{3 \times 3}(m_P)$ 
18:   $r_T \leftarrow \text{maxpool}_{3 \times 3}(m_T) - \text{minpool}_{3 \times 3}(m_T)$ 
19:   $L_{\text{rng}} \leftarrow \text{mean}_M [|r_P - r_T|]$ 
20:  (5) direction consistency
21:   $\hat{\mathbf{u}}_P \leftarrow \mathbf{g}_P / (m_P + \epsilon)$ ,  $\hat{\mathbf{u}}_T \leftarrow \mathbf{g}_T / (m_T + \epsilon)$ 
22:   $c \leftarrow \text{clamp}(\langle \hat{\mathbf{u}}_P, \hat{\mathbf{u}}_T \rangle, -1, 1)$ ,  $L_{\text{dir}} \leftarrow \text{mean}_M [1 - c]$ 
23:  (6) Combine
24:  return  $L \leftarrow \lambda_{\text{mag}} L_{\text{mag}} + \lambda_{\text{rng}} L_{\text{rng}} + \lambda_{\text{dir}} L_{\text{dir}}$ 
25: end function
26: (Optional multi-scale: replace lines 4–24 by a short loop)
27: for  $s \in \{1, \frac{1}{2}, \frac{1}{4}\}$  do downsample  $(P, T, M)$  by  $s$ , compute  $L^{(s)}$ , accumulate and average.
```

---

**Residual artifacts and failure modes** While CHMv2 improves robustness to input image quality, localized artifacts remain. These primarily include seamlines between neighboring acquisitions, missed haze or thin cloud, and disagreement in areas of extremely tall canopy between different image sources. Despite the significant improvement from CHMv1, CHMv2 still underestimates the upper tail of canopy height distributions, particularly for very tall forests and the crowns of emergent trees. Additionally, because CHMv2 estimates canopy height from a single input image (rather than a multi-date composite), terrain shadow may be a more important cause of measurement error in some regions.

### 5.3 Reproducibility statement and environmental impact

**Reproducibility statement** Our inference code with decoder weights is available in the DINOv3 repository. The pseudo code of our gradient loss appears in Algorithm 1.

**Environmental impact** We estimate the carbon footprint of decoder training using the calculations from<sup>19</sup>, with a Thermal Design Power (TDP) of the H100 GPU equal to 700W, a Power Usage Effectiveness (PUE) of 1.1, a carbon intensity factor of 0.385 kg CO<sub>2</sub> per KWh, a time of 150 trainings (ablations)  $\times$  3 hours  $\times$  8 GPUs = 3600 GPU hours. The 2772 kWh used to train the model is approximately equivalent to a CO<sub>2</sub> footprint of  $2772 \times 0.385 = 1.1\text{T}$  of CO<sub>2</sub>. We estimate the global map inference footprint to 3T of CO<sub>2</sub>.

## References

1. 1. Almeida, D. *et al.* Monitoring the structure of forest restoration plantations with a drone-lidar system. *Int. J. Appl. Earth Obs. Geoinformation* **79**, 192–198, DOI: <https://doi.org/10.1016/j.jag.2019.03.014> (2019).1. 2. Senf, C., Mori, A. S., Müller, J. & Seidl, R. The response of canopy height diversity to natural disturbances in two temperate forest landscapes. *Landsc. Ecol.* **35**, 2101–2112, DOI: [10.1007/s10980-020-01085-7](https://doi.org/10.1007/s10980-020-01085-7) (2020).
2. 3. Zhang, G. *et al.* Estimation of forest aboveground biomass in california using canopy height and leaf area index estimated from satellite data. *Remote. Sens. Environ.* **151**, 44–56, DOI: <https://doi.org/10.1016/j.rse.2014.01.025> (2014). Special Issue on 2012 ForestSAT.
3. 4. Brandt, M. *et al.* High-resolution sensors and deep learning models for tree resource monitoring. *Nat. Rev. Electr. Eng.* **2**, 13–26, DOI: [10.1038/s44287-024-00116-8](https://doi.org/10.1038/s44287-024-00116-8) (2025).
4. 5. Yuh, Y. G., Tracz, W., Matthews, H. D. & Turner, S. E. Application of machine learning approaches for land cover monitoring in northern cameroon. *Ecol. Informatics* **74**, 101955, DOI: <https://doi.org/10.1016/j.ecoinf.2022.101955> (2023).
5. 6. Xu, H., Yue, C., Zhang, Y., Liu, D. & Piao, S. Forestation at the right time with the right species can generate persistent carbon benefits in china. *Proc. Natl. Acad. Sci.* **120**, e2304988120, DOI: [10.1073/pnas.2304988120](https://doi.org/10.1073/pnas.2304988120) (2023). <https://www.pnas.org/doi/pdf/10.1073/pnas.2304988120>.
6. 7. Gale, M. G., Cary, G. J., Van Dijk, A. I. & Yebra, M. Forest fire fuel through the lens of remote sensing: Review of approaches, challenges and future directions in the remote sensing of biotic determinants of fire behaviour. *Remote. Sens. Environ.* **255**, 112282, DOI: <https://doi.org/10.1016/j.rse.2020.112282> (2021).
7. 8. Tamiminia, H., Salehi, B., Mahdianpari, M. & Goulden, T. State-wide forest canopy height and aboveground biomass map for new york with 10 m resolution, integrating gedi, sentinel-1, and sentinel-2 data. *Ecol. Informatics* **79**, 102404, DOI: <https://doi.org/10.1016/j.ecoinf.2023.102404> (2024).
8. 9. Haneda, L. E. *et al.* Straightforward model-based approach using only field data and open-source maps to improve carbon stock estimates for redd + projects. *Sci. Reports* DOI: [10.1038/s41598-026-37201-x](https://doi.org/10.1038/s41598-026-37201-x) (2026).
9. 10. Chen, Q. *et al.* Modeling and mapping agroforestry aboveground biomass in the brazilian amazon using airborne lidar data. *Remote. Sens.* **8**, DOI: [10.3390/rs8010021](https://doi.org/10.3390/rs8010021) (2016).
10. 11. Dubayah, R. *et al.* Gedi 11b geolocated waveform data global footprint level v001, DOI: [10.5067/GEDI/GEDI01\\_B.001](https://doi.org/10.5067/GEDI/GEDI01_B.001) (2020). Accessed 2025-12-01.
11. 12. Neuenschwander, A. L. *et al.* ATLAS/ICESat-2 L3A Land and Vegetation Height (ATL08), Version 7, DOI: [10.5067/ATLAS/ATL08.007](https://doi.org/10.5067/ATLAS/ATL08.007) (2025).
12. 13. Balestra, M. *et al.* Lidar data fusion to improve forest attribute estimates: A review. *Curr. For. Reports* **10**, 281–297, DOI: [10.1007/s40725-024-00223-7](https://doi.org/10.1007/s40725-024-00223-7) (2024).
13. 14. Potapov, P. *et al.* Mapping global forest canopy height through integration of GEDI and Landsat data. *Remote. Sens. Environ.* **253**, 112165, DOI: <https://doi.org/10.1016/j.rse.2020.112165> (2021).
14. 15. Lang, N., Jetz, W., Schindler, K. & Wegner, J. D. A high-resolution canopy height model of the earth, DOI: [10.48550/ARXIV.2204.08322](https://doi.org/10.48550/ARXIV.2204.08322) (2022).
15. 16. Hunter, M. O. *et al.* Global 30-m annual median vegetation height maps (2000–2022) based on icesat-2 data and machine learning. *Sci. Data* **12**, 1470, DOI: [10.1038/s41597-025-05739-6](https://doi.org/10.1038/s41597-025-05739-6) (2025).
16. 17. Moudrý, V. *et al.* Vegetation structure derived from airborne laser scanning to assess species distribution and habitat suitability: The way forward. *Divers. Distributions* **29**, 39–50, DOI: <https://doi.org/10.1111/ddi.13644> (2023).
17. 18. Fischer, F. J. *et al.* The global canopy atlas: analysis-ready maps of 3d structure for the world’s woody ecosystems, DOI: [10.1101/2025.08.31.673375](https://doi.org/10.1101/2025.08.31.673375) (2025). <https://www.biorxiv.org/content/early/2025/09/04/2025.08.31.673375.full.pdf>.
18. 19. Oquab, M. *et al.* Dinov2: Learning robust visual features without supervision (2023). [2304.07193](https://arxiv.org/abs/2304.07193).
19. 20. Siméoni, O. *et al.* Dinov3. *arXiv preprint arXiv:2508.10104* (2025).
20. 21. Fogel, F. *et al.* Open-canopy: Towards very high resolution forest monitoring. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, 1395–1406, DOI: [10.1109/CVPR52734.2025.00138](https://doi.org/10.1109/CVPR52734.2025.00138) (2025).
21. 22. Ronneberger, O., Fischer, P. & Brox, T. U-net: Convolutional networks for biomedical image segmentation. In Navab, N., Hornegger, J., Wells, W. M. & Frangi, A. F. (eds.) *Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015*, 234–241, DOI: [10.1007/978-3-319-24574-4\\_28](https://doi.org/10.1007/978-3-319-24574-4_28) (Springer International Publishing, Cham, 2015).
22. 23. da Silva, A. F. *et al.* Geospatial foundational model for canopy height estimates across kenya’s ecoregions. In *IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium*, 2853–2857, DOI: [10.1109/IGARSS53475.2024.10640630](https://doi.org/10.1109/IGARSS53475.2024.10640630) (2024).1. 24. Tolan, J. *et al.* Very high resolution canopy height maps from rgb imagery using self-supervised vision transformer and convolutional decoder trained on aerial lidar. *Remote. Sens. Environ.* **300**, 113888, DOI: [10.1016/j.rse.2023.113888](https://doi.org/10.1016/j.rse.2023.113888) (2024).
2. 25. National Ecological Observatory Network (NEON). Ecosystem structure (dp3.30015.001), DOI: [10.48443/JQCD-1N30](https://doi.org/10.48443/JQCD-1N30) (2025).
3. 26. Moudrý, V. *et al.* Comparison of three global canopy height maps and their applicability to biodiversity modeling: Accuracy issues revealed. *Ecosphere* **15**, e70026, DOI: <https://doi.org/10.1002/ecs2.70026> (2024).
4. 27. D’Amico, G. *et al.* Gedi and sentinel data integration for quantifying agroforestry tree height and stocks. *J. Environ. Manag.* **393**, 127197, DOI: <https://doi.org/10.1016/j.jenvman.2025.127197> (2025).
5. 28. U.S. Geological Survey. 3d elevation program (3dep) data. Accessed through OpenTopography (2021).
6. 29. Allred, B. W., McCord, S. E. & Morford, S. L. Canopy height model and naip imagery pairs across conus. *Sci. Data* **12**, 322, DOI: [10.1038/s41597-025-04655-z](https://doi.org/10.1038/s41597-025-04655-z) (2025).
7. 30. Bing Maps Team. Computer generated building footprints for the united states. GitHub (2018).
8. 31. Lanaras, C., Bioucas-Dias, J., Galliani, S., Baltsavias, E. & Schindler, K. Super-resolution of sentinel-2 images: Learning a globally applicable deep neural network. *ISPRS J. Photogramm. Remote. Sens.* **146**, 305–319, DOI: [10.1016/j.isprsjprs.2018.09.018](https://doi.org/10.1016/j.isprsjprs.2018.09.018) (2018).
9. 32. National Ecological Observatory Network (NEON). Ecosystem structure (dp3.30015.001) (2022).
10. 33. Lehner, B. & Döll, P. Development and validation of a global database of lakes, reservoirs and wetlands. *J. Hydrol.* **296**, 1–22, DOI: <https://doi.org/10.1016/j.jhydrol.2004.03.028> (2004).
11. 34. Zhai, G. *et al.* Foundation visual encoders are secretly few-shot anomaly detectors (2025). [2510.01934](https://doi.org/10.26434/ijoe.2025.01934).
12. 35. Yang, L. *et al.* Depth anything v2, DOI: [10.52202/079017-0688](https://doi.org/10.52202/079017-0688) (2024). [2406.09414](https://doi.org/10.26434/ijoe.2024.09414).
13. 36. Li, L. *et al.* Coarse-to-fine matching via cross fusion of satellite images. *Int. J. Appl. Earth Obs. Geoinformation* **125**, 103574, DOI: <https://doi.org/10.1016/j.jag.2023.103574> (2023).
14. 37. Han, Q. *et al.* A siamese network via cross-domain robust feature decoupling for multi-source remote sensing image registration. *Remote. Sens.* **17**, DOI: [10.3390/rs17040646](https://doi.org/10.3390/rs17040646) (2025).
15. 38. Mahapatra, D. & Ge, Z. Training data independent image registration using generative adversarial networks and domain adaptation. *Pattern Recognit.* **100**, 107109, DOI: <https://doi.org/10.1016/j.patcog.2019.107109> (2020).
16. 39. Shi, L., Zhao, R., Pan, B., Zou, Z. & Shi, Z. Unsupervised multimodal remote sensing image registration via domain adaptation. *IEEE Transactions on Geosci. Remote. Sens.* **61**, 1–11, DOI: [10.1109/TGRS.2023.3333889](https://doi.org/10.1109/TGRS.2023.3333889) (2023).
17. 40. Zhang, H. *et al.* Dino: Detr with improved denoising anchor boxes for end-to-end object detection (2022). [2203.03605](https://doi.org/10.26434/ijoe.2022.03605).
18. 41. Weinstein, B. Milliontrees: A benchmark dataset for airborne tree prediction. <https://milliontrees.idtrees.org/> (2025). Accessed: 2026-01-17; includes TreeBoxes, TreePoints, and TreePolygons datasets.
19. 42. Ester, M., Kriegel, H.-P., Sander, J., Xu, X. *et al.* A density-based algorithm for discovering clusters in large spatial databases with noise. In *kdd*, vol. 96, 226–231 (1996).
20. 43. Kidorf, H. & Piegorsch, W. A practical fast fourier transform (FFT)-based implementation for image correlation. In *Applications of Digital Image Processing VII*, vol. 0504 of *Proceedings of SPIE*, 350–357, DOI: [10.1117/12.944856](https://doi.org/10.1117/12.944856) (1984).
21. 44. Eigen, D., Puhrsch, C. & Fergus, R. Depth map prediction from a single image using a multi-scale deep network (2014). [1406.2283](https://doi.org/10.26434/ijoe.2014.2283).
22. 45. Bochkovskii, A. *et al.* Depth pro: Sharp monocular metric depth in less than a second (2025). [2410.02073](https://doi.org/10.26434/ijoe.2025.02073).
23. 46. Charbonnier, P., Blanc-Feraud, L., Aubert, G. & Barlaud, M. Two deterministic half-quadratic regularization algorithms for computed imaging. In *Proceedings of 1st International Conference on Image Processing*, vol. 2, 168–172 vol.2, DOI: [10.1109/ICIP.1994.413553](https://doi.org/10.1109/ICIP.1994.413553) (1994).
24. 47. Jiao, J., Cao, Y., Song, Y. & Lau, R. Look deeper into depth: Monocular depth estimation with semantic booster and attention-driven loss. In *Proceedings of the European Conference on Computer Vision (ECCV)*, DOI: [10.1007/978-3-030-01267-0\\_4](https://doi.org/10.1007/978-3-030-01267-0_4) (2018).
25. 48. Ranftl, R., Lasinger, K., Hafner, D. & Koltun, V. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. *IEEE Transactions on Pattern Analysis Mach. Intell.* **PP**, 1–1, DOI: [10.1109/TPAMI.2020.3019967](https://doi.org/10.1109/TPAMI.2020.3019967) (2020).1. 49. Ramamonjisoa, M. *et al.* DINOv3 depth code. <https://github.com/facebookresearch/dinov3/blob/main/dinov3/eval/depth/train.py> (2025).
2. 50. Pauls, J. *et al.* Estimating canopy height at scale (2024). [2406.01076](#).
3. 51. Dubayah, R. *et al.* Gedi l2a elevation and height metrics data global footprint level v002, DOI: [10.5067/GEDI/GEDI02\\_A\\_002](#) (2021). Accessed: 2025-12-01.
4. 52. Neuenschwander, A. *et al.* Ice, cloud, and land elevation satellite (ICESat-2) project algorithm theoretical basis document (ATBD) for land - vegetation along-track products (ATL08), version 7 (2025).
5. 53. Hansen, M. C. *et al.* High-resolution global maps of 21st-century forest cover change. *Science* **342**, 850–853, DOI: [10.1126/science.1244693](#) (2013).
6. 54. Zanaga, D. *et al.* ESA WorldCover 10 m 2021 v200 (2022).
7. 55. Liu, A., Cheng, X. & Chen, Z. Performance evaluation of gedi and icesat-2 laser altimeter data for terrain and canopy height retrievals. *Remote. Sens. Environ.* **264**, 112571, DOI: <https://doi.org/10.1016/j.rse.2021.112571> (2021).
8. 56. Chen, P. *et al.* Unveiling the performance and influential factors of GEDI L2A for building height retrieval. *GIsci Remote. Sens.* **62**, DOI: <https://doi.org/10.1080/15481603.2025.2498785> (2025).
9. 57. Tang, H. *et al.* Evaluating and mitigating the impact of systematic geolocation error on canopy height measurement performance of GEDI. *Remote. Sens. Environ.* **291**, 113571, DOI: <https://doi.org/10.1016/j.rse.2023.113571> (2023).
10. 58. Luthcke, S. B. *et al.* Icesat-2 pointing calibration and geolocation performance. *Earth Space Sci.* **8**, e2020EA001494, DOI: <https://doi.org/10.1029/2020EA001494> (2021). E2020EA001494 2020EA001494.
11. 59. Tian, L., Liao, L., Tao, Y., Wu, X. & Li, M. Forest age mapping using landsat time-series stacks data based on forest disturbance and empirical relationships between age and height. *Remote. Sens.* **15**, DOI: [10.3390/rs15112862](#) (2023).

## 6 Data Availability

The CHMv2 maps are available at [link](#). The 3DEP-NAIP dataset will be made available on Zenodo following manuscript publication. The ALS sources of the SatLidar dataset are listed in the DINOv3<sup>20</sup> paper. This work used the Microsoft Open Buildings dataset<sup>30</sup>, available under the Open Database License (ODbL) v1.0 license, to clean and curate the NAIP-3DEP dataset.

## 7 Code Availability

The CHMv2 backbone, decoder weights, and example inference code will be available in the DINOv3 GitHub repository.

## 8 Acknowledgments

We would like to thank the DINO team, Natacha Supper, Daniel Haziza and Christian Keller, Patrick Nease, and Laura McGorman for their precious help. We would also like to thank past team members for their contribution to this project: Tobias Tiecke, Tracy Johns, Benjamin Nosarzewski, Hung-I Yang, Guillaume Couairon, Sayantan Majumdar, Janaki Vamaraju, Theo Moutakanni and Brian White.

## 9 Authors information

### 9.1 Contributions

**J.B:** Training data collection, data registration and quality checking, loss design, architecture, sampling changes, analysis of results, paper writing, project coordination. **S.Y:** GPU Optimization, global Inference of maps, data and OSS preparation. **J.T:** Training data collection, global inference of maps, data processing for comparative analysis. **H.V:** Decoder training with data mix and sampler ablation. **M.R:** Software contributions, loss ablation. **P.L., P.B.:** Technical advice throughout theproject. **C.C.:** Data preparation, Decoder training, comparative evaluation, model selection, paper writing (ablation, training sections), project coordination. **J.S.:** Analysis of results, paper writing, comparative evaluation. **J.E.:** Analysis of results, paper writing. **P.P.:** Analysis of results, paper writing, evaluation with spaceborne LiDAR. **X.L.:** Analysis of results, evaluation with spaceborne LiDAR.

All authors read and approved the final manuscript.

## **9.2 Funding**

The World Resources Institute was supported by a contract from Meta and a grant from the Bezos Earth Fund. Meta and FAIR received no financial support for this research, the preparation of the manuscript, or the publication process.

## **9.3 Competing interests**

The authors declare no competing interests.

