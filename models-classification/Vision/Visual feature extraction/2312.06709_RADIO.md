Title: Agglomerative Vision Foundation Model Reduce All Domains Into One

URL Source: https://arxiv.org/html/2312.06709

Published Time: Thu, 02 May 2024 00:10:27 GMT

Markdown Content:
Mike Ranzinger∗, Greg Heinrich∗, Jan Kautz, Pavlo Molchanov 

NVIDIA 

{mranzinger,gheinrich,jkautz,pmolchanov}@nvidia.com

###### Abstract

A handful of visual foundation models (VFMs) have recently emerged as the backbones for numerous downstream tasks. VFMs like CLIP, DINOv2, SAM are trained with distinct objectives, exhibiting unique characteristics for various downstream tasks. We find that despite their conceptual differences, these models can be effectively merged into a unified model through multi-teacher distillation. We name this approach AM-RADIO (Agglomerative Model – Reduce All Domains Into One). This integrative approach not only surpasses the performance of individual teacher models but also amalgamates their distinctive features, such as zero-shot vision-language comprehension, detailed pixel-level understanding, and open vocabulary segmentation capabilities. Additionally, in pursuit of the most hardware-efficient backbone, we evaluated numerous architectures in our multi-teacher distillation pipeline using the same training recipe. This led to the development of a novel architecture (E-RADIO) that exceeds the performance of its predecessors and is at least 6x faster than the teacher models at matched resolution. Our comprehensive benchmarking process covers downstream tasks including ImageNet classification, semantic segmentation linear probing, COCO object detection and integration into LLaVa-1.5.

Code: [https://github.com/NVlabs/RADIO](https://github.com/NVlabs/RADIO).

![Image 1: [Uncaptioned image]](https://arxiv.org/html/2312.06709v5/x1.jpg)![Image 2: [Uncaptioned image]](https://arxiv.org/html/2312.06709v5/x2.png)![Image 3: [Uncaptioned image]](https://arxiv.org/html/2312.06709v5/x3.png)

Figure 1:  AM-RADIO is a framework to distill multiple pretrained vision foundation models, such as CLIP[[51](https://arxiv.org/html/2312.06709v5#bib.bib51)], DINOv2[[48](https://arxiv.org/html/2312.06709v5#bib.bib48)], SAM[[35](https://arxiv.org/html/2312.06709v5#bib.bib35)], into a single model that we call RADIO. As a result, a single vision foundation model agglomerates unique properties of the original models. This unifying approach obtains state-of-the-art feature representations in a single forward pass while also enabling unique properties such as zero-shot classification (CLIP) or open set instance segmentation (SAM) at negligible additional cost. 

Image description: (left) PCA feature visualization of different models. Our proposed RADIO model can process any resolution and aspect ratio, and produces semantically rich dense encodings; (middle) the overview of the AM-RADIO framework; (right) benchmarks on classification, segmentation, and vision-language modeling tasks, see section[5](https://arxiv.org/html/2312.06709v5#S5 "5 Results ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One"). 

**footnotetext: Equal contribution![Image 4: Refer to caption](https://arxiv.org/html/2312.06709v5/x4.png)

Figure 2: AM-RADIO - is a multi-teacher distillation framework that efficiently trains new vision foundation models of arbitrary architecture. It unifies unique attributes (like zero-shot text grounding, dense correspondence) of each teacher into a single model that even outperforms them on a majority of the tasks. 

1 Introduction
--------------

Model Params Resol-Throughput ImageNet1K Segmentation (linear)Vision-Language (LLaVa-1.5[[40](https://arxiv.org/html/2312.06709v5#bib.bib40)])SAM[[35](https://arxiv.org/html/2312.06709v5#bib.bib35)]
(M)ution Zero-shot k-NN ADE20k VOC GQA POPE TextVQA VQAv2 COCO
OpenCLIP-H/14 [[11](https://arxiv.org/html/2312.06709v5#bib.bib11)]632 224 503 77.19 81.10 40.04 68.03 57.94 83.61 50.48 72.24-
MetaCLIP-H/14 [[64](https://arxiv.org/html/2312.06709v5#bib.bib64)]632 224 486 80.51 82.12 35.39 62.62 60.57 84.76 53.65 75.71-
SigLIP-L/14 [[74](https://arxiv.org/html/2312.06709v5#bib.bib74)]428 384 241 82.61 85.16 40.53 70.31 57.70 84.85 56.65 71.94-
Intern-ViT-6B [[10](https://arxiv.org/html/2312.06709v5#bib.bib10)]5,902 224 63 83.20††78.43 47.20 76.85 60.18 84.02 52.45 76.75-
5,537 448 14††68.64 42.78 74.43 61.19 87.23 60.36 78.83-
DFN CLIP-H/14 [[19](https://arxiv.org/html/2312.06709v5#bib.bib19)]633 378 170 83.90 85.27 39.00 70.29 61.73 85.91 56.78 78.78-
OpenAI CLIP-L/14 [[51](https://arxiv.org/html/2312.06709v5#bib.bib51)]305 336 414 75.54 79.80 36.51 67.04 62.20 86.09 57.92 78.49-
DINOv2-g/14-reg [[14](https://arxiv.org/html/2312.06709v5#bib.bib14)]1,137 224 294†-83.41 48.68 82.78 61.88 85.62 47.18 76.23-
SAM-H/16 [[35](https://arxiv.org/html/2312.06709v5#bib.bib35)]637 1024 12-22.12 28.08 34.34 49.92 81.76 43.91 57.65 77.18
E-RADIO-L (Ours)391 512 468 80.73 83.89 48.22 81.64 61.70 85.07 51.47 76.73 76.31
RADIO-ViT-H/16 (Ours)653 432 158 82.93 86.06 51.34 84.71 63.01 86.20 56.32 79.28 76.23

Table 1: Comparison of vision foundation and RADIO models. “Zero-Shot” and k-NN are computed on ImageNet-1K. ADE20K[[77](https://arxiv.org/html/2312.06709v5#bib.bib77)] and VOC (PascalVOC2012) refer to linear probe semantic segmentation mIOU. GQA, POPE (popular), TextVQA, and VQAv2 are obtained via LLaVa 1.5[[40](https://arxiv.org/html/2312.06709v5#bib.bib40)] by replacing the vision encoder. COCO is the instance segmentation metric introduced by [[8](https://arxiv.org/html/2312.06709v5#bib.bib8)] to evaluate SAM[[35](https://arxiv.org/html/2312.06709v5#bib.bib35)] distillation. RADIO attains the best metrics on most benchmarks, and is competitive with the rest, while E-RADIO enables high quality results in resource constrained settings. Note that Zero-Shot and COCO use teacher’s decoder head that is not finetuned. Throughput computed using NVIDIA A100 GPU, stated resolution, and TensorRT v8601. *Denotes teachers used to train our final RADIO.†We failed to export DINOv2-g-reg to TensorRT, so we report DINOv2-g here, which should be fairly close. ††We were unable to get zero shot working using their model code.

Knowledge Distillation [[26](https://arxiv.org/html/2312.06709v5#bib.bib26)] has been a very successful and popular technique for transferring the knowledge of a “teacher” model (or ensemble of models) into a typically smaller “student” model. In the original formulation, both the student and the teacher operate on the same in-domain dataset, and the student simultaneously matches the logits of the teacher, and the ground truth labels. Instead of using labeled images, an alternative approach is to train the student model to match the features of the teacher model [[53](https://arxiv.org/html/2312.06709v5#bib.bib53), [28](https://arxiv.org/html/2312.06709v5#bib.bib28), [1](https://arxiv.org/html/2312.06709v5#bib.bib1), [25](https://arxiv.org/html/2312.06709v5#bib.bib25), [72](https://arxiv.org/html/2312.06709v5#bib.bib72), [56](https://arxiv.org/html/2312.06709v5#bib.bib56), [61](https://arxiv.org/html/2312.06709v5#bib.bib61)].

Instead of using a smaller student model, [[63](https://arxiv.org/html/2312.06709v5#bib.bib63)] employ an iterative learning procedure with a high-capacity model where a student of equal or greater capacity than the teacher is trained with heavy augmentation applied to the student. Once trained, they expand the dataset by pseudo-labeling new data using the trained student. They then make the student become the teacher, and repeat the process. An important finding in this work is that the student is capable of surpassing the performance of the teacher.

The authors of [[26](https://arxiv.org/html/2312.06709v5#bib.bib26)] explore the concept of ensemble distillation, where there are multiple teachers, each of which having restricted domain knowledge. [[78](https://arxiv.org/html/2312.06709v5#bib.bib78)] provides an overview of multi-teacher distillation, and proposes that instead of matching the summary of an ensemble of teachers, the student can match the features of each individual teacher via some learned non-shared mapping from the representation space of the student to each teacher. Of interest in their approach is that the student and teacher don’t need to share the same architecture, and also that treating teachers individually yields improved performance.

Recently, the concept of Foundation Models (FMs) [[3](https://arxiv.org/html/2312.06709v5#bib.bib3)] has emerged, with the general understanding that these models are large, general, and expensive to train. Through training on very large datasets they are broadly applicable to numerous downstream tasks. A seminal example of such models is CLIP [[51](https://arxiv.org/html/2312.06709v5#bib.bib51)], which trains on web-scale weakly supervised (image, caption) pairs, and results in exceptional zero-shot performances on a wide array of computer vision benchmarks. While CLIP is firmly a FM, another model, DINOv2 [[48](https://arxiv.org/html/2312.06709v5#bib.bib48)] has emerged with broad capabilities, often surpassing CLIP on dense tasks that require strong spatial features, such as ADE20k [[77](https://arxiv.org/html/2312.06709v5#bib.bib77)] and Pascal VOC [[18](https://arxiv.org/html/2312.06709v5#bib.bib18)]. Separately, SAM (Segment Anything) [[35](https://arxiv.org/html/2312.06709v5#bib.bib35)] is gaining popularity for its excellent open-vocabulary instance segmentation abilities, whose vision encoder we hypothesize has strong dense feature representations.

We introduce AM-RADIO with the goal of learning from multiple foundational models simultaneously. We observe that, when given a student model of sufficient capacity, it is often able to exceed any of its teachers on important axes. In addition to performing well on representative foundational benchmarks, by virtue of the training framework, our student models are able to mimic their teacher models, and thus are able to perform downstream tasks that are otherwise performed by the teachers. Examples of this include CLIP-ZeroShot applications, since the language model trained by CLIP is compatible with our student, and also Segment-Anything tasks, as the student is able to replace the vision encoder and interface with the already-trained mask decoders.

We also study the effect of using a more hardware-efficient model architecture. Most works on efficiency are not directly comparable as they use different training recipes, even when evaluated on the same dataset such as ImageNet-1k, and may be over-tuned. To this end, we evaluate more than 10 promising architectures under the same training recipe for a direct comparison. We reveal that CNN-like architectures are faster but struggle to distill ViT VFMs. This led us to the development of a novel hybrid architecture, E-RADIO, that exceeds the performance of its predecessors and is at least 6x faster than teacher models at matched resolution.

Our main contributions are as follows:

*   •We describe a general methodology for distilling multiple distinct foundation models into one, including models with incompatible input resolutions. 
*   •We show that these student models are able to outperform their teachers on representative benchmarks. 
*   •We demonstrate that these student models can either drop-in replace their teachers, or their features can be used directly in downstream applications such as providing visual encoding for LLaVA [[41](https://arxiv.org/html/2312.06709v5#bib.bib41), [40](https://arxiv.org/html/2312.06709v5#bib.bib40)]. 
*   •We benchmark a number of efficient architectures and propose a new architecture (E-RADIO) that allows for similar model quality at significant speedups. 

2 Related Work
--------------

Knowledge Distillation The underpinning of our work is based on the method of Knowledge Distillation [[26](https://arxiv.org/html/2312.06709v5#bib.bib26), [34](https://arxiv.org/html/2312.06709v5#bib.bib34), [4](https://arxiv.org/html/2312.06709v5#bib.bib4), [47](https://arxiv.org/html/2312.06709v5#bib.bib47), [5](https://arxiv.org/html/2312.06709v5#bib.bib5)] which aims to train a “student” model using soft targets produced by an already-trained “teacher” model, using the the teacher’s output logits as “soft” labels. Alternatively, distillation can be performed using intermediate network activations [[53](https://arxiv.org/html/2312.06709v5#bib.bib53), [28](https://arxiv.org/html/2312.06709v5#bib.bib28), [1](https://arxiv.org/html/2312.06709v5#bib.bib1), [25](https://arxiv.org/html/2312.06709v5#bib.bib25), [72](https://arxiv.org/html/2312.06709v5#bib.bib72), [56](https://arxiv.org/html/2312.06709v5#bib.bib56), [61](https://arxiv.org/html/2312.06709v5#bib.bib61)]. In general, due to the heterogeneous nature of the different teacher foundation models that we employ, we ignore any potential labels coming from the data, and we ignore the logits of teachers, and simply opt to match the feature representations of the teachers before any task-specific processing stages.

Multi-Teacher Distillation There is also a body of work that studies distilling a student model jointly from multiple teacher models simultaneously [[26](https://arxiv.org/html/2312.06709v5#bib.bib26), [42](https://arxiv.org/html/2312.06709v5#bib.bib42), [78](https://arxiv.org/html/2312.06709v5#bib.bib78), [71](https://arxiv.org/html/2312.06709v5#bib.bib71), [75](https://arxiv.org/html/2312.06709v5#bib.bib75), [68](https://arxiv.org/html/2312.06709v5#bib.bib68), [50](https://arxiv.org/html/2312.06709v5#bib.bib50), [69](https://arxiv.org/html/2312.06709v5#bib.bib69), [36](https://arxiv.org/html/2312.06709v5#bib.bib36), [2](https://arxiv.org/html/2312.06709v5#bib.bib2), [20](https://arxiv.org/html/2312.06709v5#bib.bib20)]. Because of the heterogeneous domains that our teacher models cover, we don’t apply approaches that marginalize teachers into a unified label, and instead map students to each teacher independently using teacher-specific projection heads from the unified student representation. Although the reason behind this method in [[78](https://arxiv.org/html/2312.06709v5#bib.bib78)] is different, we find the same overall strategy to be effective. While [[61](https://arxiv.org/html/2312.06709v5#bib.bib61)] doesn’t study matching the features of multiple teachers simultaneously, we are able to extend their paradigm via the different projection heads. To preserve drop-in compatibility with teacher frameworks, we eliminate the feature normalization in the loss function.

Distilling Foundation Models Foundation Models [[3](https://arxiv.org/html/2312.06709v5#bib.bib3)] are meant to be generalist models that are trained on massive amounts of data, and are typically resource intensive to train from scratch. In the vein of single-teacher distillation, [[48](https://arxiv.org/html/2312.06709v5#bib.bib48)] employ self-distillation to train their smaller variants from the larger teacher. [[61](https://arxiv.org/html/2312.06709v5#bib.bib61)] distills their model from a CLIP [[51](https://arxiv.org/html/2312.06709v5#bib.bib51)] teacher. Instead of focusing our energy on one teacher in particular, we instead grab high-quality versions of CLIP [[51](https://arxiv.org/html/2312.06709v5#bib.bib51)] (using OpenCLIP [[30](https://arxiv.org/html/2312.06709v5#bib.bib30)]), DINOv2 [[48](https://arxiv.org/html/2312.06709v5#bib.bib48)], and SAM [[35](https://arxiv.org/html/2312.06709v5#bib.bib35)]. Concurrently with our work, [[60](https://arxiv.org/html/2312.06709v5#bib.bib60)] describe a methodology for merging a CLIP model into a pretrained SAM model via distillation, which is, in spirit, quite similar to our approach. In contrast to theirs, we include DINOv2 and also simplify the objective to straightforward feature matching. Since we don’t rely on the student model to be pre-trained, it also gives us the flexibility to have the student be an architecture distinct from any teacher.

3 Knowledge Agglomeration
-------------------------

We propose a framework to train a vision foundation model from scratch via multi-teacher distillation as shown in Figure[2](https://arxiv.org/html/2312.06709v5#S0.F2 "Figure 2 ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One"). We demonstrate that each teacher brings unique properties to the foundational vision model, and the resulting trained model will agglomerate these attributes.

### 3.1 Overview

As an initial assumption, we expect that the teacher models are capable of representing a broad swath of images found on the internet, coming from datasets such as ImageNet (1k or 21k) [[15](https://arxiv.org/html/2312.06709v5#bib.bib15)], LAION-400M [[54](https://arxiv.org/html/2312.06709v5#bib.bib54)] or DataComp-1B [[21](https://arxiv.org/html/2312.06709v5#bib.bib21)]. With this in mind, we choose to study 3 seminal teacher model families: CLIP [[51](https://arxiv.org/html/2312.06709v5#bib.bib51)], DINOv2 [[48](https://arxiv.org/html/2312.06709v5#bib.bib48)], and SAM [[35](https://arxiv.org/html/2312.06709v5#bib.bib35)] as they have demonstrated outstanding performance over a broad range of tasks (as in CLIP), or specifically strong performance on downstream dense tasks, such as semantic segmentation under linear probe (as in DINOv2), or open-vocabulary segmentation (as in SAM). Because these teacher models come from such diverse domains, we omit any form of supplemental ground truth guidance and treat the aforementioned datasets simply as sources of images. To assess the quality of our models, we adopt a set of representative metrics across a few broad domains.

*   •Image level reasoning: (i) k-NN Top-1 accuracy on ImageNet-1K, and (ii) Zero-Shot accuracy using the CLIP teacher’s language model [[51](https://arxiv.org/html/2312.06709v5#bib.bib51)]. k-NN [[62](https://arxiv.org/html/2312.06709v5#bib.bib62), [9](https://arxiv.org/html/2312.06709v5#bib.bib9), [48](https://arxiv.org/html/2312.06709v5#bib.bib48)] embeds the model’s summary feature vector for every image in the training set, and then for each validation image, it uses a weighted sum of the k 𝑘 k italic_k nearest training vectors to elect a label. 
*   •Pixel-level visual tasks: segmentation mIOU on (i) ADE20K and (ii) Pascal VOC - under the linear probe setting, details in Section[5.3](https://arxiv.org/html/2312.06709v5#S5.SS3 "5.3 Semantic Segmentation Linear Probing ‣ 5 Results ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One"). 
*   •Large Vision-Language Models:  we plug our frozen vision encoder model into LLaVA-1.5 [[40](https://arxiv.org/html/2312.06709v5#bib.bib40)] and evaluate it on a wide set of tasks including GQA [[29](https://arxiv.org/html/2312.06709v5#bib.bib29)], TextVQA [[55](https://arxiv.org/html/2312.06709v5#bib.bib55)], ScienceQA [[46](https://arxiv.org/html/2312.06709v5#bib.bib46)] and VQAv2 [[23](https://arxiv.org/html/2312.06709v5#bib.bib23)]. Details in Section[5.4](https://arxiv.org/html/2312.06709v5#S5.SS4 "5.4 Visual Question Answering ‣ 5 Results ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One"). 
*   •SAM-COCO instance segmentation: From [[8](https://arxiv.org/html/2312.06709v5#bib.bib8)], we adopt their COCO instance segmentation methodology to evaluate our ability to replicate SAM visual features. 

Results on these tasks, both for teacher models and our AM-RADIO variants, are summarized in Table [1](https://arxiv.org/html/2312.06709v5#S1.T1 "Table 1 ‣ 1 Introduction ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One").

### 3.2 Adaptor Heads

We opt for simplicity in design of the adaptor heads, and leave alternative architectures as future work. To this end, we employ a simple 2-layer MLP, with a LayerNorm and GELU in between. The input dimension is the student embedding dimension, the intermediate dimension is the maximum embedding dimension of all teachers, and the output dimension matches the specific teacher. For each teacher, we employ two heads, one for the summary vector, and one for the spatial features.

### 3.3 Distillation Dataset Choice

In table [2](https://arxiv.org/html/2312.06709v5#S3.T2 "Table 2 ‣ 3.3 Distillation Dataset Choice ‣ 3 Knowledge Agglomeration ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One") we study the effect of different datasets on downstream metrics. While the highest image classification metrics are achieved using ImageNet-1K as the training dataset, we argue that it doesn’t fairly measure “zero shot” performance as the student directly learns the teacher features in the evaluation domain. For this reason, we opt for the DataComp-1B dataset.

Table 2: Ablation study on the choice of training dataset. We use MetaCLIP ViT-H/14[[16](https://arxiv.org/html/2312.06709v5#bib.bib16)] and DINOv2 ViT-g/14 teachers, and a ViT-L/14 student model with CPE[[33](https://arxiv.org/html/2312.06709v5#bib.bib33)]. Both “k-NN” and “Zero Shot” are for ImageNet-1k. ADE20k refers to mIOU linear probe on ADE20k.

### 3.4 Loss Formulation

Because we don’t have ground truth data for each teacher for each image, we instead opt to match the features coming from each teacher’s vision encoder. In particular, we distinguish between the summary feature vector and the spatial feature vectors for each teacher. The summary feature is computed differently based on the model. For CLIP and DINOv2, we use the “class token” as the summary feature vector, and we don’t match a summary for SAM.

Let f⁢(x|Θ 0)𝑓 conditional 𝑥 subscript Θ 0 f\left(x|\Theta_{0}\right)italic_f ( italic_x | roman_Θ start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT ) be the student vision encoder with parameters Θ 0 subscript Θ 0\Theta_{0}roman_Θ start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT, and y i s=h i(s)⁢(x′|Θ i(s))superscript subscript 𝑦 𝑖 𝑠 superscript subscript ℎ 𝑖 𝑠 conditional superscript 𝑥′superscript subscript Θ 𝑖 𝑠 y_{i}^{s}=h_{i}^{(s)}(x^{\prime}|\Theta_{i}^{(s)})italic_y start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_s end_POSTSUPERSCRIPT = italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_s ) end_POSTSUPERSCRIPT ( italic_x start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT | roman_Θ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_s ) end_POSTSUPERSCRIPT ) be the learned student head matching teacher summary features z i(s)=t i(s)⁢(x|Φ i)superscript subscript 𝑧 𝑖 𝑠 superscript subscript 𝑡 𝑖 𝑠 conditional 𝑥 subscript Φ 𝑖 z_{i}^{(s)}=t_{i}^{(s)}\left(x|\Phi_{i}\right)italic_z start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_s ) end_POSTSUPERSCRIPT = italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_s ) end_POSTSUPERSCRIPT ( italic_x | roman_Φ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) with student adaptor parameters Θ i(s)superscript subscript Θ 𝑖 𝑠\Theta_{i}^{(s)}roman_Θ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_s ) end_POSTSUPERSCRIPT and teacher parameters Φ i subscript Φ 𝑖\Phi_{i}roman_Φ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT.

x′superscript 𝑥′\displaystyle x^{\prime}italic_x start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT=f⁢(x|Θ 0);absent 𝑓 conditional 𝑥 subscript Θ 0\displaystyle=f\left(x|\Theta_{0}\right);= italic_f ( italic_x | roman_Θ start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT ) ;y i(s)superscript subscript 𝑦 𝑖 𝑠\displaystyle y_{i}^{(s)}italic_y start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_s ) end_POSTSUPERSCRIPT=h i(s)⁢(x′|Θ i(s));absent superscript subscript ℎ 𝑖 𝑠 conditional superscript 𝑥′superscript subscript Θ 𝑖 𝑠\displaystyle=h_{i}^{(s)}\left(x^{\prime}|\Theta_{i}^{(s)}\right);= italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_s ) end_POSTSUPERSCRIPT ( italic_x start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT | roman_Θ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_s ) end_POSTSUPERSCRIPT ) ;(1)
z i(s)superscript subscript 𝑧 𝑖 𝑠\displaystyle z_{i}^{(s)}italic_z start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_s ) end_POSTSUPERSCRIPT=t i(s)⁢(x|Φ i),absent superscript subscript 𝑡 𝑖 𝑠 conditional 𝑥 subscript Φ 𝑖\displaystyle=t_{i}^{(s)}\left(x|\Phi_{i}\right),= italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_s ) end_POSTSUPERSCRIPT ( italic_x | roman_Φ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) ,

L summary⁢(x)subscript 𝐿 summary 𝑥\displaystyle L_{\text{summary}}(x)italic_L start_POSTSUBSCRIPT summary end_POSTSUBSCRIPT ( italic_x )=∑i λ i⁢L cos⁢(y i(s),z i(s))absent subscript 𝑖 subscript 𝜆 𝑖 subscript 𝐿 cos superscript subscript 𝑦 𝑖 𝑠 superscript subscript 𝑧 𝑖 𝑠\displaystyle=\sum_{i}\lambda_{i}L_{\text{cos}}(y_{i}^{(s)},z_{i}^{(s)})= ∑ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_λ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_L start_POSTSUBSCRIPT cos end_POSTSUBSCRIPT ( italic_y start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_s ) end_POSTSUPERSCRIPT , italic_z start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_s ) end_POSTSUPERSCRIPT )(2)

We found empirically that cosine distance loss produced better models compared to L1, MSE, Smooth-L1 [[22](https://arxiv.org/html/2312.06709v5#bib.bib22)]. Additionally, supervising the spatial features of the model by matching the teacher was not only important for downstream dense tasks, but also improved the holistic quality of our model.

Table 3: Ablation over which teachers we supervise the spatial features. We use a ViT-L/14 student model and train on the LAION-400M dataset. Adding this loss term is always beneficial. DINOv2 appears to provide better spatial features than CLIP, but training the student to match both teachers produces the best results. We don’t ablate SAM as we solely want it for its spatial features.

For matching the spatial features, we employ a combination of cosine similarity and smooth L1. Similar to equation([2](https://arxiv.org/html/2312.06709v5#S3.E2 "Equation 2 ‣ 3.4 Loss Formulation ‣ 3 Knowledge Agglomeration ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One")) where we found that cosine similarity produced the best results, we found the same to be true for the spatial features. However, we want to allow our student model to be a drop-in replacement in the teacher frameworks, thus it’s important that we match the magnitude of the teacher vectors, and so we include smooth L1. In ([3](https://arxiv.org/html/2312.06709v5#S3.E3 "Equation 3 ‣ 3.4 Loss Formulation ‣ 3 Knowledge Agglomeration ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One")) we show the formulation of this loss. Let h i(v)⁢(x′|Θ i(v))superscript subscript ℎ 𝑖 𝑣 conditional superscript 𝑥′superscript subscript Θ 𝑖 𝑣 h_{i}^{(v)}(x^{\prime}|\Theta_{i}^{(v)})italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_v ) end_POSTSUPERSCRIPT ( italic_x start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT | roman_Θ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_v ) end_POSTSUPERSCRIPT ) be the learned student head for matching teacher feature vectors, and corresponding t i(v)⁢(x|Φ i(v))superscript subscript 𝑡 𝑖 𝑣 conditional 𝑥 superscript subscript Φ 𝑖 𝑣 t_{i}^{(v)}(x|\Phi_{i}^{(v)})italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_v ) end_POSTSUPERSCRIPT ( italic_x | roman_Φ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_v ) end_POSTSUPERSCRIPT ) be the teacher feature vectors, with x′=f⁢(x|Θ 0)superscript 𝑥′𝑓 conditional 𝑥 subscript Θ 0 x^{\prime}=f(x|\Theta_{0})italic_x start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT = italic_f ( italic_x | roman_Θ start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT ), then the spatial feature loss is:

L match⁢(x,y)subscript 𝐿 match 𝑥 𝑦\displaystyle L_{\text{match}}(x,y)italic_L start_POSTSUBSCRIPT match end_POSTSUBSCRIPT ( italic_x , italic_y )=α⁢L cos⁢(x,y)+β⁢L smooth-l1⁢(x,y)absent 𝛼 subscript 𝐿 cos 𝑥 𝑦 𝛽 subscript 𝐿 smooth-l1 𝑥 𝑦\displaystyle=\alpha L_{\text{cos}}(x,y)+\beta L_{\text{smooth-l1}}(x,y)= italic_α italic_L start_POSTSUBSCRIPT cos end_POSTSUBSCRIPT ( italic_x , italic_y ) + italic_β italic_L start_POSTSUBSCRIPT smooth-l1 end_POSTSUBSCRIPT ( italic_x , italic_y )(3)
L features⁢(x)subscript 𝐿 features 𝑥\displaystyle L_{\text{features}}(x)italic_L start_POSTSUBSCRIPT features end_POSTSUBSCRIPT ( italic_x )=∑i γ i L match(h i(v)(x′|Θ i(v)),t i(v)(x|Φ i(v)))\displaystyle=\sum_{i}\gamma_{i}L_{\text{match}}\left(h_{i}^{(v)}(x^{\prime}|% \Theta_{i}^{(v)}),t_{i}^{(v)}(x|\Phi_{i}^{(v}))\right)= ∑ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_γ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_L start_POSTSUBSCRIPT match end_POSTSUBSCRIPT ( italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_v ) end_POSTSUPERSCRIPT ( italic_x start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT | roman_Θ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_v ) end_POSTSUPERSCRIPT ) , italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_v ) end_POSTSUPERSCRIPT ( italic_x | roman_Φ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_v end_POSTSUPERSCRIPT ) ) )

We choose α=0.9 𝛼 0.9\alpha=0.9 italic_α = 0.9 and β=0.1 𝛽 0.1\beta=0.1 italic_β = 0.1 to mostly rely on the empirically better cosine distance, but to also match vector magnitudes.

#### 3.4.1 Loss Balancing

Due to the number of possible combinations of loss weights between the different teachers, and even which teachers, and possible formulations of loss functions, we mostly opted toward naive loss balancing with all teachers equally weighted for spatial features (γ i=1 subscript 𝛾 𝑖 1\gamma_{i}=1 italic_γ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = 1). For summary features, we have λ C⁢L⁢I⁢P=λ D⁢I⁢N⁢O=1 subscript 𝜆 𝐶 𝐿 𝐼 𝑃 subscript 𝜆 𝐷 𝐼 𝑁 𝑂 1\lambda_{CLIP}=\lambda_{DINO}=1 italic_λ start_POSTSUBSCRIPT italic_C italic_L italic_I italic_P end_POSTSUBSCRIPT = italic_λ start_POSTSUBSCRIPT italic_D italic_I italic_N italic_O end_POSTSUBSCRIPT = 1 and λ S⁢A⁢M=0 subscript 𝜆 𝑆 𝐴 𝑀 0\lambda_{SAM}=0 italic_λ start_POSTSUBSCRIPT italic_S italic_A italic_M end_POSTSUBSCRIPT = 0.

We did experiment with automatic loss balancing using predicted uncertainty [[12](https://arxiv.org/html/2312.06709v5#bib.bib12)], AdaLoss [[27](https://arxiv.org/html/2312.06709v5#bib.bib27)] (momentum 0.99) and separately with AMTML-KD [[42](https://arxiv.org/html/2312.06709v5#bib.bib42)], as ways to learn the balance of λ i subscript 𝜆 𝑖\lambda_{i}italic_λ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT and γ i subscript 𝛾 𝑖\gamma_{i}italic_γ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT. In the case of AMTML-KD, the model would always collapse its entire weight around the CLIP teacher and would yield worse results than naive manual balancing. Based on the results in table [4](https://arxiv.org/html/2312.06709v5#S3.T4 "Table 4 ‣ 3.4.1 Loss Balancing ‣ 3.4 Loss Formulation ‣ 3 Knowledge Agglomeration ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One"), there is very little advantage to the more exotic balancing schemes, so we opt for the "Naive" method throughout the rest of the paper.

Table 4: Loss term balancing methods comparison. We use a ViT-B/14 student, and CLIP+DINOv2 teachers. We found that AdaLoss produces the best results on the ImageNet tasks, but the worst on ADE20K.

4 Implementation Details
------------------------

Performing heterogeneous multi-teacher distillation is not trivial due to a mismatch in feature dimensions, input resolutions, concepts for loss computation, and downsampling ratios, as well as challenges in fitting multiple teachers into a single GPU.

General. We train all student models using the AdamW [[45](https://arxiv.org/html/2312.06709v5#bib.bib45)] optimizer, batch size 1024, cosine annealing learning rate schedule and base learning rate of 0.001 0.001 0.001 0.001. We train for 600k steps, resulting in 614M total examples seen. For our best student model, we train using DFN CLIP ViT-H/14 378px, OpenAI CLIP ViT-L/14 336px, DINOv2 ViT-g/14 224px, and SAM ViTDet-H 1024px. We apply random scale + cropping to both student and teacher inputs. We chose the DataComp-1B dataset due to it having the highest quality results of the web-scale datasets we had access to. We train in two stages, first with CLIP+DINOv2 for 300k steps at 256px, and second with CLIP+DINOv2 at 432px plus SAM at 1024px for 300k steps.

Student architecture. We study two settings for student model architecture:

*   •Standard ViT[[16](https://arxiv.org/html/2312.06709v5#bib.bib16)] architecture to match the architecture of teachers. Our best model is a ViT-H/16. 
*   •Efficient architecture variants prioritizing high throughput on GPUs. See Section[5.1](https://arxiv.org/html/2312.06709v5#S5.SS1 "5.1 Efficient Students ‣ 5 Results ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One"). 

Multi-scale Teachers. We choose ViT-H/16 architecture for our student model. To match resolution of SAM features, we feed the expected resolution of 1024 2 superscript 1024 2 1024^{2}1024 start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT. Given that our CLIP and DINOv2 teachers are patch-14 models, we opt to feed the student 432 2 superscript 432 2 432^{2}432 start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT inputs, as that is the same effective resolution as 378 2 superscript 378 2 378^{2}378 start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT for patch-14. We found that interpolating DINOv2 features doesn’t degrade results, so the teacher operates at 224px and we upsample the outputs to match the student.

Rank/Teacher Partitioning. We group teacher models by (batch_size, student_resolution), and then distribute the groups to different GPUs, such that each GPU processes a consistent batch size and input resolution. We also sample groups at different rates. For our training setups that include SAM, we train with 64 GPUs, half of which get the CLIP+DINOv2 group with batch size 32 per GPU and input resolution 432, and the other half get SAM with batch size 2 per GPU and input resolution 1024. This results in an effective batch size of 1,152. For CLIP+DINOv2 training, we use 32 GPUs, resulting in batch size 1024.

Multi-Resolution ViTs. Many of our student models use ViT [[16](https://arxiv.org/html/2312.06709v5#bib.bib16)] as the base vision architecture. Traditionally, ViTs use a learned position embedding for each input patch in an image, which in turn enforces that the model always operates at a constant resolution. We employ the Cropped Position Embedding (CPE) [[33](https://arxiv.org/html/2312.06709v5#bib.bib33)] augmentation with the number of positions being equal to 128 2 superscript 128 2 128^{2}128 start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT. The position embeddings are then randomly cropped and interpolated to match the number of input patches for the student model. Even when training with CLIP+DINOv2 at 224 resolution, we found that this technique results in a negligible drop (Table [5](https://arxiv.org/html/2312.06709v5#S4.T5 "Table 5 ‣ 4 Implementation Details ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One")) in summary metrics, but improved semantic segmentation linear probing mIOU. For heterogeneous-resolution students, this is a seamless technique that allows ViT to operate at arbitrary resolutions within some envelope. In addition to enabling arbitrary resolutions, as shown in figure [3](https://arxiv.org/html/2312.06709v5#S4.F3 "Figure 3 ‣ 4 Implementation Details ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One"), CPE reduces the noise artifacts in the position embeddings as compared to other ViT models [[67](https://arxiv.org/html/2312.06709v5#bib.bib67), [66](https://arxiv.org/html/2312.06709v5#bib.bib66), [6](https://arxiv.org/html/2312.06709v5#bib.bib6)].

Table 5: Comparing identical ViT-L/14 student models, with and without CPE [[33](https://arxiv.org/html/2312.06709v5#bib.bib33)] formulation. While the student only ever trains at 224 2 superscript 224 2 224^{2}224 start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT resolution, CPE allows us to generalize to 518 2 superscript 518 2 518^{2}518 start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT resolution, not only improving over non-CPE, but even outperforming DINOv2-g itself.

![Image 5: Refer to caption](https://arxiv.org/html/2312.06709v5/extracted/5569764/resources/pos_embed/vis_pos_embed.jpg)

(a)RADIO 2048px

![Image 6: Refer to caption](https://arxiv.org/html/2312.06709v5/extracted/5569764/resources/pos_embed/vis_dino_pos_embed.jpg)

(b)DINOv2-g-reg 518px

![Image 7: Refer to caption](https://arxiv.org/html/2312.06709v5/extracted/5569764/resources/pos_embed/vis_clip_pos_embed.jpg)

(c)DFN CLIP 378px

![Image 8: Refer to caption](https://arxiv.org/html/2312.06709v5/extracted/5569764/resources/pos_embed/vis_oaiclip_pos_embed.jpg)

(d)OpenAI CLIP 336px

Figure 3: PCA visualization of the position embeddings for various models. The CPE method not only allows RADIO to learn an arbitrarily large absolution position embedding map, but also goes a long way towards regularizing the space and eliminating high frequency artifacts. As seen with the other models, position embeddings normally have regular frequency patterns, leading to undesirable output artifacts from the ViT [[67](https://arxiv.org/html/2312.06709v5#bib.bib67), [66](https://arxiv.org/html/2312.06709v5#bib.bib66), [6](https://arxiv.org/html/2312.06709v5#bib.bib6)].

High-Resolution ViT Student. In SAM, they employ the ViTDet [[37](https://arxiv.org/html/2312.06709v5#bib.bib37)] architecture as a way to reduce the computational and memory burden of ViT models at high-resolution. We reformulate this arch instead into a training augmentation, where we sample a window size from a set of possible window sizes. This allows us to reduce the computational burden of training the student model with the SAM teacher, and, as we make the window size flexible, it provides an additional throughput scaling mechanism during inference. Table [8](https://arxiv.org/html/2312.06709v5#S5.T8 "Table 8 ‣ 5.2 Comparison with teachers ‣ 5 Results ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One") demonstrates our ability to replace SAM’s encoder. Separately, we found that high resolution training was unstable, so we apply spectral reparametrization [[73](https://arxiv.org/html/2312.06709v5#bib.bib73)] and a weight decay of 0.02 0.02 0.02 0.02 to prevent attention entropy collapse.

Student/Teacher Resolution Mismatch. When the student and teacher downsample images through their processing stack at different rates, it results in the output feature vectors having different resolutions. For example, if the teachers use a ViT-H/14 architecture and student a ViT-H/16, it means that the student outputs a 14 2 superscript 14 2 14^{2}14 start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT feature map, and the teachers a 16 2 superscript 16 2 16^{2}16 start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT feature map. For L features subscript 𝐿 features L_{\text{features}}italic_L start_POSTSUBSCRIPT features end_POSTSUBSCRIPT we bilinearly interpolate the outputs to match the larger resolution between the student and teacher features.

Table 6: Comparing identical ViT models, with CLS token and average pooling summarization. 

Feature Summarization. In [3.4](https://arxiv.org/html/2312.06709v5#S3.SS4 "3.4 Loss Formulation ‣ 3 Knowledge Agglomeration ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One") we explained how teacher summary features are extracted using the “class token” of their respective ViT models. We now turn our attention to the summarization of student features. ViTs have 2 options: (i) a separate summarization “CLS” token or (ii) average pooling patch tokens. We evaluate both options in Table[6](https://arxiv.org/html/2312.06709v5#S4.T6 "Table 6 ‣ 4 Implementation Details ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One"). We observe that average pooling improves summary loss, but has a more significant detrimental effect on the feature loss. Given the importance of the latter we choose to use separate CLS tokens.

5 Results
---------

In this section, we analyze models obtained with the proposed AM-RADIO framework. First, we touch upon backbone efficiency, then compare with the original teachers (CLIP, DINOv2, SAM), and benchmark models under vision question answering in the LLaVa framework. We will see that the proposed models outperform the original teachers in multiple metrics, including throughput. Results are shown in Figure[1](https://arxiv.org/html/2312.06709v5#S0.F1 "Figure 1 ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One") and Table[1](https://arxiv.org/html/2312.06709v5#S1.T1 "Table 1 ‣ 1 Introduction ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One").

![Image 9: Refer to caption](https://arxiv.org/html/2312.06709v5/x5.png)

Figure 4:  All models followed the same training protocol. The results from three benchmarks show that RADIO and E-RADIO models outperform others in efficiency. This under-performance in other models might be due to overfitting architectures on supervised ImageNet-1K training. E-RADIO notably delivers results 10 times faster and with a 20% improvement over teacher models. We study E-RADIO at 224px resolution, with a window size of 7. 

### 5.1 Efficient Students

Table 7: Comparison of backbones. Throughput is measured using TensorRT 9.0.1 on A100 in mixed FP16/FP32 precision at batch size 128 on 224 2 superscript 224 2 224^{2}224 start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT px resolution. Sorted by descending throughput order. FD loss is the Feature Distillation training loss against the DINOv2 teacher, it exhibits high correlation with the ADE20k mIoU. Bolded models form the speed/quality Pareto front. 

We aim to find an efficient model architecture to speed up the inference of VFM. There are a number of architectural designs aimed at high throughput on GPU devices. We use our distillation framework to evaluate several backbones with no change in training hyperparameters.

Upon reviewing the literature on efficient vision backbones focused for high GPU throughput, we pick the following list of architectures: EfficientNetV2 [[58](https://arxiv.org/html/2312.06709v5#bib.bib58)], ResNetv2 [[57](https://arxiv.org/html/2312.06709v5#bib.bib57)], RegNetY [[52](https://arxiv.org/html/2312.06709v5#bib.bib52)], FasterViT [[24](https://arxiv.org/html/2312.06709v5#bib.bib24)], EfficientViT [[8](https://arxiv.org/html/2312.06709v5#bib.bib8)], ConvNext [[44](https://arxiv.org/html/2312.06709v5#bib.bib44)], NFNet [[7](https://arxiv.org/html/2312.06709v5#bib.bib7)], SwinV2 [[43](https://arxiv.org/html/2312.06709v5#bib.bib43)], MaxViT [[59](https://arxiv.org/html/2312.06709v5#bib.bib59)], PoolformerV2 [[70](https://arxiv.org/html/2312.06709v5#bib.bib70)] and MViTV2 [[38](https://arxiv.org/html/2312.06709v5#bib.bib38)]. We train all the backbones via distillation on the ImageNet-21k dataset, using OpenCLIP ViT-H/14 (laion2B-s32B-b79K) and DINOv2 g/14 as teachers. Results are compiled in Table [7](https://arxiv.org/html/2312.06709v5#S5.T7 "Table 7 ‣ 5.1 Efficient Students ‣ 5 Results ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One").

We observe that many models lag behind teachers. Additionally, CNN-like models are significantly faster than ViTs, while the latter are more accurate. The relatively low performance of existing efficient backbones on the dense ADE20k segmentation task is not unexpected since all of them apply a spatial dimension reduction factor of 32 for final feature maps of size 7 2 superscript 7 2 7^{2}7 start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT for input resolution of 224 2 superscript 224 2 224^{2}224 start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT px, thus hardly capable of capturing fine-grain spatial information.

E-RADIO: To overcome this issue, we propose a novel hybrid architecture, named E-RADIO (Efficient RADIO). This design borrows ideas from existing literature and includes an input stem with strided convolutions to downsample the input image by 4x. It then proceeds with 2 stages of YOLOv8 C2f convolution blocks and 2 stages of transformer. For the transformer variant we pick windowed attention (like in SWIN[[43](https://arxiv.org/html/2312.06709v5#bib.bib43)]), and interleave local windowed attention with “global” windowed attention as done in[[24](https://arxiv.org/html/2312.06709v5#bib.bib24)] and ViTDet [[37](https://arxiv.org/html/2312.06709v5#bib.bib37)]. To perform “global” attention we first downsample the feature map by 2x, apply windowed attention, and then upsample the feature maps back to the original resolution. Up-/down-sampling is performed by strided convolution with a kernel size 3x3 and a stride of 2. The last idea is borrowed from EdgeViT[[49](https://arxiv.org/html/2312.06709v5#bib.bib49)], which uses local-global-local attention. See Appendix for details. Finally, E-RADIO upsamples final feature maps by 2x via a deconvolutional layer and adds them to feature maps from the third stage, resulting in only a 16x spatial resolution reduction. Such upsampling gives an improvement in dense task while being only 10% slower. Results of E-RADIO in Table[7](https://arxiv.org/html/2312.06709v5#S5.T7 "Table 7 ‣ 5.1 Efficient Students ‣ 5 Results ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One") demonstrate that the proposed architecture significantly outperforms the competition, and can be seen as an efficient replacement for the much slower full ViT.

### 5.2 Comparison with teachers

A comprehensive set of results is presented in Table [1](https://arxiv.org/html/2312.06709v5#S1.T1 "Table 1 ‣ 1 Introduction ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One"). We notice that MetaCLIP is better than OpenCLIP, and DFN CLIP better than MetaCLIP. DINOv2 provides important properties for dense tasks: ADE20k and VOC. Our E-RADIO-L model is significantly faster than all ViT models. At the same time, it strongly outperforms MetaCLIP on most metrics at matched throughput, while also enabling Zero-shot capability that is absent in DINOv2 and SAM. Our full model, ViT-H/16, is as fast as the teachers but outperforms them on 6 out of 9 tasks, demonstrating the efficiency of the proposed distillation framework.

Drop-In SAM Replacement. Following [[8](https://arxiv.org/html/2312.06709v5#bib.bib8)], we use their evaluation harness to compute the mIOU for instance segmentation using pretrained SAM with vision encoder replaced by our model. Table [8](https://arxiv.org/html/2312.06709v5#S5.T8 "Table 8 ‣ 5.2 Comparison with teachers ‣ 5 Results ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One") shows the results of the COCO Instance Segmentation task using the baseline SAM models and RADIO.

Table 8: We substitute SAM’s vision encoder with our RADIO model. RADIO aligns with SAM’s features just before the encoder’s neck layer. We also examine the impact of varying ViTDet window sizes. Differences in throughput owe to the fact that RADIO doesn’t use relative positional embeddings and we reduced shuffling with our patch reordering algorithm (in appendix). Throughput is computed on an NVIDIA A100 GPU using TensorRT and batch size 16. †This is the same model, just with a different window size setting. 

### 5.3 Semantic Segmentation Linear Probing

We train a linear head on top of the frozen features of the teachers and students alike and evaluate performance in the MMSeg [[13](https://arxiv.org/html/2312.06709v5#bib.bib13)] framework using the mIoU metric on ADE20k and PascalVOC2012 datasets. We use a training and evaluation crop size of 512 for RADIO, 518 for DINOv2, and the native resolution for the others. We use the “slide” evaluation mode with a stride of 2 3 2 3\frac{2}{3}divide start_ARG 2 end_ARG start_ARG 3 end_ARG the crop size. We train the linear head for 160k steps using a total batch size of 16, a base learning rate of 10−3 superscript 10 3 10^{-3}10 start_POSTSUPERSCRIPT - 3 end_POSTSUPERSCRIPT and the AdamW optimizer.

### 5.4 Visual Question Answering

We replace the vision encoder in a LLaVA 1.5[[40](https://arxiv.org/html/2312.06709v5#bib.bib40)] setup with our own encoder. A 2-layer MLP is used to project frozen visual features into the language token space. Under the default LLaVA 1.5 settings, we pretrain a multimodal projection MLP and then run instruction tuning to finetune a Vicuna 7B-1.5 model[[76](https://arxiv.org/html/2312.06709v5#bib.bib76)]. We evaluate models using the validation sets of GQA [[29](https://arxiv.org/html/2312.06709v5#bib.bib29)], TextVQA [[55](https://arxiv.org/html/2312.06709v5#bib.bib55)], POPE [[39](https://arxiv.org/html/2312.06709v5#bib.bib39)] (popular), and we score the model on the Test-Dev set of VQAv2 [[23](https://arxiv.org/html/2312.06709v5#bib.bib23)] using EvalAI[[65](https://arxiv.org/html/2312.06709v5#bib.bib65)]. We use the vision encoder’s native input resolution, resizing the long edge and padding the short edge. Experimental results are compiled in Table [1](https://arxiv.org/html/2312.06709v5#S1.T1 "Table 1 ‣ 1 Introduction ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One"). Owing to the increased input resolution flexibility of RADIO, we resize the long edge of the image to 432px aspect preserving, only padding to the nearest multiple of the patch size. This results in 462 462~{}462 462 tokens on average, versus the 576 576 576 576 tokens required by the 336px patch-14 encoders, a 20% reduction.

### 5.5 3D Awareness Probing

Following the work from [[17](https://arxiv.org/html/2312.06709v5#bib.bib17)], we probe our model’s ability to extract 3D features such as depth, surface normals and multi-view keypoint correspondance. Our results are summarized in Table [9](https://arxiv.org/html/2312.06709v5#S5.T9 "Table 9 ‣ 5.5 3D Awareness Probing ‣ 5 Results ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One") and show that our model’s performance is on par with the bigger DINOv2-g-14-reg[[14](https://arxiv.org/html/2312.06709v5#bib.bib14)] and significantly better than other comparably-sized teachers.

Table 9: Probing 3D Awareness: we use the code from [[17](https://arxiv.org/html/2312.06709v5#bib.bib17)] and evaluate our RADIO model and its teachers on monocular depth, surface normals and multi-view correspondance tasks, using the NAVI[[31](https://arxiv.org/html/2312.06709v5#bib.bib31)] dataset. For each task we report the accuracy, averaged over all thresholds. 

6 Conclusion and Key Insights
-----------------------------

![Image 10: Refer to caption](https://arxiv.org/html/2312.06709v5/extracted/5569764/resources/radiov2_mode_switch_plot.png)
![Image 11: Refer to caption](https://arxiv.org/html/2312.06709v5/extracted/5569764/resources/radiov2_mode_switch_small.jpg)

Figure 5: RADIO “mode switches” when resolution is increased. In the plot, we show the MSE error between the RADIO features coming from its DINOv2 head at different resolutions, versus the features actually produced by DINOv2 at 518px. We bilinearly interpolate the RADIO features to match the DINOv2 feature resolution. At 720px, there is a sudden jump in the error, which corresponds with a complete change in color space in the image.

Most VFMs have unique properties such as language grounding (CLIP), dense correspondences (DINOv2), and detailed segmentation (SAM), but also large holes in capability. Distillation allows uniting all these properties in a single model that often outperforms any of the teachers. We have also observed that better teachers yield better students, which allows RADIO to absorb and challenge the current SOTA foundation models at a given point in time.

Feature distillation loss. We observe the crucial importance of full feature distillation to boost the performance of the teacher in dense image understanding tasks, such as an 18% relative improvement on ADE20K.

SAM vs DINOv2. We find that, out of the box, SAM is not well-suited for downstream tasks, whereas DINOv2 significantly outperforms in zero- and few-shot tasks. For example, ADE20K segmentation via linear probing is 1.7x better with the latter, and the ImageNet1k k-NN metric is 4x better. SAM excels in detecting edges and segmenting objects but performs poorly in high-level object description and combining the semantics of multiple objects (Figure[4](https://arxiv.org/html/2312.06709v5#S5.F4 "Figure 4 ‣ 5 Results ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One")).

Dense features. As seen in figure [1](https://arxiv.org/html/2312.06709v5#S0.F1 "Figure 1 ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One"), RADIO is capable of producing high resolution and low-noise features. An issue we identified, however, shown in figure [5](https://arxiv.org/html/2312.06709v5#S6.F5 "Figure 5 ‣ 6 Conclusion and Key Insights ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One") is that RADIO appears to have a latent ‘low resolution’ and ‘high resolution’ mode, likely due to the partitioned training between CLIP+DINO and SAM objectives, which we intend to fix in future work.

Efficient backbone. Based on our analysis of distilling efficient backbones, we conclude that most model designs are overly tailored towards supervised training on ImageNet1K, and as a result, do not scale well to VFM settings. We designed a new vision backbone, E-RADIO, with a hybrid CNN-Transformer architecture that improves upon the Pareto frontier.

References
----------

*   Ahn et al. [2019] S. Ahn, S. Hu, A. Damianou, N.D. Lawrence, and Z. Dai. Variational information distillation for knowledge transfer. In _2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, pages 9155–9163, Los Alamitos, CA, USA, 2019. IEEE Computer Society. 
*   Asif et al. [2019] Umar Asif, Jianbin Tang, and Stefan Harrer. Ensemble knowledge distillation for learning improved and efficient networks. In _European Conference on Artificial Intelligence_, 2019. 
*   Awais et al. [2023] Muhammad Awais, Muzammal Naseer, Salman Khan, Rao Muhammad Anwer, Hisham Cholakkal, Mubarak Shah, Ming-Hsuan Yang, and Fahad Shahbaz Khan. Foundational models defining a new era in vision: A survey and outlook, 2023. 
*   Ba and Caruana [2014] Jimmy Ba and Rich Caruana. Do deep nets really need to be deep? In _Advances in Neural Information Processing Systems_, pages 2654–2662, 2014. 
*   Beyer et al. [2022] L. Beyer, X. Zhai, A. Royer, L. Markeeva, R. Anil, and A. Kolesnikov. Knowledge distillation: A good teacher is patient and consistent. In _2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, pages 10915–10924, Los Alamitos, CA, USA, 2022. IEEE Computer Society. 
*   Bolya et al. [2023] Daniel Bolya, Chaitanya Ryali, Judy Hoffman, and Christoph Feichtenhofer. Window attention is bugged: How not to interpolate position embeddings, 2023. 
*   Brock et al. [2021] Andrew Brock, Soham De, Samuel L. Smith, and Karen Simonyan. High-performance large-scale image recognition without normalization, 2021. 
*   Cai et al. [2023] Han Cai, Junyan Li, Muyan Hu, Chuang Gan, and Song Han. Efficientvit: Multi-scale linear attention for high-resolution dense prediction, 2023. 
*   Caron et al. [2021] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers, 2021. 
*   Chen et al. [2023] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. _arXiv preprint arXiv:2312.14238_, 2023. 
*   Cherti et al. [2022] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning, 2022. 
*   Cipolla et al. [2018] R. Cipolla, Y. Gal, and A. Kendall. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In _2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, pages 7482–7491, Los Alamitos, CA, USA, 2018. IEEE Computer Society. 
*   Contributors [2020] MMSegmentation Contributors. MMSegmentation: Openmmlab semantic segmentation toolbox and benchmark. [https://github.com/open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation), 2020. 
*   Darcet et al. [2023] Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers, 2023. 
*   Deng et al. [2009] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In _2009 IEEE Conference on Computer Vision and Pattern Recognition_, pages 248–255, 2009. 
*   Dosovitskiy et al. [2021] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale, 2021. 
*   El Banani et al. [2024] Mohamed El Banani, Amit Raj, Kevis-Kokitsi Maninis, Abhishek Kar, Yuanzhen Li, Michael Rubinstein, Deqing Sun, Leonidas Guibas, Justin Johnson, and Varun Jampani. Probing the 3D Awareness of Visual Foundation Models. In _CVPR_, 2024. 
*   Everingham et al. [2015] M. Everingham, S.M.A. Eslami, L. Van Gool, C.K.I. Williams, J. Winn, and A. Zisserman. The pascal visual object classes challenge: A retrospective. _International Journal of Computer Vision_, 111(1):98–136, 2015. 
*   Fang et al. [2023] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks, 2023. 
*   Fukuda et al. [2017] Takashi Fukuda, Masayuki Suzuki, Gakuto Kurata, Samuel Thomas, Jia Cui, and Bhuvana Ramabhadran. Efficient knowledge distillation from an ensemble of teachers. In _Interspeech_, 2017. 
*   Gadre et al. [2023] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, Eyal Orgad, Rahim Entezari, Giannis Daras, Sarah Pratt, Vivek Ramanujan, Yonatan Bitton, Kalyani Marathe, Stephen Mussmann, Richard Vencu, Mehdi Cherti, Ranjay Krishna, Pang Wei Koh, Olga Saukh, Alexander Ratner, Shuran Song, Hannaneh Hajishirzi, Ali Farhadi, Romain Beaumont, Sewoong Oh, Alex Dimakis, Jenia Jitsev, Yair Carmon, Vaishaal Shankar, and Ludwig Schmidt. Datacomp: In search of the next generation of multimodal datasets, 2023. 
*   Girshick [2015] Ross Girshick. Fast r-cnn. In _Proceedings of the IEEE international conference on computer vision_, pages 1440–1448, 2015. 
*   Goyal et al. [2017] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA matter: Elevating the role of image understanding in Visual Question Answering. In _Conference on Computer Vision and Pattern Recognition (CVPR)_, 2017. 
*   Hatamizadeh et al. [2023] Ali Hatamizadeh, Greg Heinrich, Hongxu Yin, Andrew Tao, Jose M. Alvarez, Jan Kautz, and Pavlo Molchanov. Fastervit: Fast vision transformers with hierarchical attention, 2023. 
*   Heo et al. [2019] B. Heo, J. Kim, S. Yun, H. Park, N. Kwak, and J. Choi. A comprehensive overhaul of feature distillation. In _2019 IEEE/CVF International Conference on Computer Vision (ICCV)_, pages 1921–1930, Los Alamitos, CA, USA, 2019. IEEE Computer Society. 
*   Hinton et al. [2015] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. _arXiv preprint arXiv:1503.02531_, 2015. 
*   Hu et al. [2019] Hanzhang Hu, Debadeepta Dey, Martial Hebert, and J.Andrew Bagnell. Learning anytime predictions in neural networks via adaptive loss balancing. In _Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence and Thirty-First Innovative Applications of Artificial Intelligence Conference and Ninth AAAI Symposium on Educational Advances in Artificial Intelligence_. AAAI Press, 2019. 
*   Huang and Wang [2017] Zehao Huang and Naiyan Wang. Like what you like: Knowledge distill via neuron selectivity transfer. _CoRR_, abs/1707.01219, 2017. 
*   Hudson and Manning [2019] Drew A. Hudson and Christopher D. Manning. GQA: a new dataset for compositional question answering over real-world images. _CoRR_, abs/1902.09506, 2019. 
*   Ilharco et al. [2021] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, 2021. 
*   Jampani et al. [2023] Varun Jampani, Kevis-Kokitsi Maninis, Andreas Engelhardt, Arjun Karpur, Karen Truong, Kyle Sargent, Stefan Popov, André Araujo, Ricardo Martin-Brualla, Kaushal Patel, Daniel Vlasic, Vittorio Ferrari, Ameesh Makadia, Ce Liu, Yuanzhen Li, and Howard Zhou. Navi: Category-agnostic image collections with high-quality 3d shape and pose annotations, 2023. 
*   Jocher et al. [2023] Glenn Jocher, Ayush Chaurasia, and Jing Qiu. Ultralytics yolov8, 2023. 
*   Kim et al. [2023] Dahun Kim, Anelia Angelova, and Weicheng Kuo. Region-aware pretraining for open-vocabulary object detection with vision transformers, 2023. 
*   Kim et al. [2018] Jangho Kim, SeongUk Park, and Nojun Kwak. Paraphrasing complex network: Network compression via factor transfer. In _Proceedings of the 32nd International Conference on Neural Information Processing Systems_, page 2765–2774, Red Hook, NY, USA, 2018. Curran Associates Inc. 
*   Kirillov et al. [2023] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything, 2023. 
*   Lan et al. [2018] Xu Lan, Xiatian Zhu, and Shaogang Gong. Knowledge distillation by on-the-fly native ensemble, 2018. 
*   Li et al. [2022a] Yanghao Li, Hanzi Mao, Ross Girshick, and Kaiming He. Exploring plain vision transformer backbones for object detection, 2022a. 
*   Li et al. [2022b] Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, and Christoph Feichtenhofer. Mvitv2: Improved multiscale vision transformers for classification and detection, 2022b. 
*   Li et al. [2023] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In _The 2023 Conference on Empirical Methods in Natural Language Processing_, 2023. 
*   Liu et al. [2023a] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023a. 
*   Liu et al. [2023b] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023b. 
*   Liu et al. [2020] Yuang Liu, Wei Zhang, and Jun Wang. Adaptive multi-teacher multi-level knowledge distillation. _Neurocomputing_, 415:106–113, 2020. 
*   Liu et al. [2022a] Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, Furu Wei, and Baining Guo. Swin transformer v2: Scaling up capacity and resolution, 2022a. 
*   Liu et al. [2022b] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s, 2022b. 
*   Loshchilov and Hutter [2019] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _International Conference on Learning Representations_, 2019. 
*   Lu et al. [2022] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In _The 36th Conference on Neural Information Processing Systems (NeurIPS)_, 2022. 
*   Mirzadeh et al. [2019] Seyed Iman Mirzadeh, Mehrdad Farajtabar, Ang Li, Nir Levine, Akihiro Matsukawa, and Hassan Ghasemzadeh. Improved knowledge distillation via teacher assistant. In _AAAI Conference on Artificial Intelligence_, 2019. 
*   Oquab et al. [2023] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2023. 
*   Pan et al. [2022] Junting Pan, Adrian Bulat, Fuwen Tan, Xiatian Zhu, Lukasz Dudziak, Hongsheng Li, Georgios Tzimiropoulos, and Brais Martinez. Edgevits: Competing light-weight cnns on mobile devices with vision transformers. In _ECCV_, 2022. 
*   Park and Kwak [2020] Seonguk Park and Nojun Kwak. Feature-level ensemble knowledge distillation for aggregating knowledge from multiple networks. In _European Conference on Artificial Intelligence_, 2020. 
*   Radford et al. [2021] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In _Proceedings of the 38th International Conference on Machine Learning_, pages 8748–8763. PMLR, 2021. 
*   Radosavovic et al. [2020] Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr Dollár. Designing network design spaces, 2020. 
*   Romero et al. [2014] Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua Bengio. Fitnets: Hints for thin deep nets. _CoRR_, abs/1412.6550, 2014. 
*   Schuhmann et al. [2021] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs, 2021. 
*   Singh et al. [2019] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, 2019. 
*   Sun et al. [2021] X. Sun, R. Panda, C. Chen, A. Oliva, R. Feris, and K. Saenko. Dynamic network quantization for efficient video inference. In _2021 IEEE/CVF International Conference on Computer Vision (ICCV)_, pages 7355–7365, Los Alamitos, CA, USA, 2021. IEEE Computer Society. 
*   Szegedy et al. [2016] Christian Szegedy, Sergey Ioffe, and Vincent Vanhoucke. Inception-v4, inception-resnet and the impact of residual connections on learning. _CoRR_, abs/1602.07261, 2016. 
*   Tan and Le [2021] Mingxing Tan and Quoc V. Le. Efficientnetv2: Smaller models and faster training. _CoRR_, abs/2104.00298, 2021. 
*   Tu et al. [2022] Zhengzhong Tu, Hossein Talebi, Han Zhang, Feng Yang, Peyman Milanfar, Alan Bovik, and Yinxiao Li. Maxvit: Multi-axis vision transformer, 2022. 
*   Wang et al. [2023] Haoxiang Wang, Pavan Kumar Anasosalu Vasu, Fartash Faghri, Raviteja Vemulapalli, Mehrdad Farajtabar, Sachin Mehta, Mohammad Rastegari, Oncel Tuzel, and Hadi Pouransari. Sam-clip: Merging vision foundation models towards semantic and spatial understanding, 2023. 
*   Wei et al. [2022] Yixuan Wei, Han Hu, Zhenda Xie, Zheng Zhang, Yue Cao, Jianmin Bao, Dong Chen, and Baining Guo. Contrastive learning rivals masked image modeling in fine-tuning via feature distillation, 2022. 
*   Wu et al. [2018] Zhirong Wu, Yuanjun Xiong, X Yu Stella, and Dahua Lin. Unsupervised feature learning via non-parametric instance discrimination. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_, 2018. 
*   Xie et al. [2020] Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V. Le. Self-training with noisy student improves imagenet classification. In _2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, pages 10684–10695, 2020. 
*   Xu et al. [2023] Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying clip data. 2023. 
*   Yadav et al. [2019] Deshraj Yadav, Rishabh Jain, Harsh Agrawal, Prithvijit Chattopadhyay, Taranjeet Singh, Akash Jain, Shiv Baran Singh, Stefan Lee, and Dhruv Batra. Evalai: Towards better evaluation systems for ai agents, 2019. 
*   Yang et al. [2023] Jiawei Yang, Boris Ivanovic, Or Litany, Xinshuo Weng, Seung Wook Kim, Boyi Li, Tong Che, Danfei Xu, Sanja Fidler, Marco Pavone, and Yue Wang. Emernerf: Emergent spatial-temporal scene decomposition via self-supervision, 2023. 
*   Yang et al. [2024] Jiawei Yang, Katie Z Luo, Jiefeng Li, Kilian Q Weinberger, Yonglong Tian, and Yue Wang. Denoising vision transformers, 2024. 
*   Yang et al. [2020] Ze Yang, Linjun Shou, Ming Gong, Wutao Lin, and Daxin Jiang. Model compression with two-stage multi-teacher knowledge distillation for web question answering system. In _Proceedings of the 13th International Conference on Web Search and Data Mining_, page 690–698, New York, NY, USA, 2020. Association for Computing Machinery. 
*   You et al. [2017] Shan You, Chang Xu, Chao Xu, and Dacheng Tao. Learning from multiple teacher networks. In _Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_, page 1285–1294, New York, NY, USA, 2017. Association for Computing Machinery. 
*   Yu et al. [2022] Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen Zhou, Xinchao Wang, Jiashi Feng, and Shuicheng Yan. Metaformer is actually what you need for vision, 2022. 
*   Yuan et al. [2020] Fei Yuan, Linjun Shou, Jian Pei, Wutao Lin, Ming Gong, Yan Fu, and Daxin Jiang. Reinforced multi-teacher selection for knowledge distillation, 2020. 
*   Zagoruyko and Komodakis [2017] Sergey Zagoruyko and Nikos Komodakis. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. In _5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings_. OpenReview.net, 2017. 
*   Zhai et al. [2023a] Shuangfei Zhai, Tatiana Likhomanenko, Etai Littwin, Dan Busbridge, Jason Ramapuram, Yizhe Zhang, Jiatao Gu, and Joshua M Susskind. Stabilizing transformer training by preventing attention entropy collapse. In _International Conference on Machine Learning_, pages 40770–40803. PMLR, 2023a. 
*   Zhai et al. [2023b] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. _arXiv preprint arXiv:2303.15343_, 2023b. 
*   Zhao et al. [2022] Haoran Zhao, Xin Sun, Junyu Dong, Changrui Chen, and Zihe Dong. Highlight every step: Knowledge distillation via collaborative teaching. _IEEE Transactions on Cybernetics_, 52(4):2070–2081, 2022. 
*   Zheng et al. [2023] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric.P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023. 
*   Zhou et al. [2017] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In _2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, pages 5122–5130, 2017. 
*   Zuchniak [2023] Konrad Zuchniak. Multi-teacher knowledge distillation as an effective method for compressing ensembles of neural networks, 2023. 

\thetitle

Supplementary Material

Appendix A E-RADIO architecture details
---------------------------------------

![Image 12: Refer to caption](https://arxiv.org/html/2312.06709v5/x6.png)

Figure 6: High level architecture of the ERADIO network architecture. Overall architecture is composed of multiple stages: 1) the stem, 2) 2 convolutional blocks from YOLOv8, 3) 2 transformer blocks with multi-resolution windowed self attention.

The architecture of E-RADIO is illustrated in Figure[6](https://arxiv.org/html/2312.06709v5#A1.F6 "Figure 6 ‣ Appendix A E-RADIO architecture details ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One"). It is a hybrid CNN-Transformer architecture. First 2 stages follow convolution paradigm and have the C2f architecture from YOLOv8 model[[32](https://arxiv.org/html/2312.06709v5#bib.bib32)]. The last 2 stages have the Transformer architecture with windowed attention and multi-resolution attention (MRA) structure. Every stage, except the last one, are followed by downsample block. We implement it as a strided convolution with 3x3 kernel and stride 2, followed by batch normalization layer.

### A.1 Multi-Resolution Attention

![Image 13: Refer to caption](https://arxiv.org/html/2312.06709v5/extracted/5569764/resources/eradio/eradio_mra_details.png)

Figure 7: Multi-resolution attention for E-RADIO

Standard transformers struggle to scale with high input image resolution because of quadratic complexity of the attention. SWIN[[43](https://arxiv.org/html/2312.06709v5#bib.bib43)] proposed to use windowed attention to reduce the complexity of attention. We reuse windowed attention in the E-RADIO. To address for missing communication between windows, SWIN introduced window shifting, unfortunately, it has non-negligible compute cost. Instead, we propose multi-resolution attention inspired by EdgeViT’s Local-Global-Local attention[[49](https://arxiv.org/html/2312.06709v5#bib.bib49)]. The idea is illustrated in Figure [7](https://arxiv.org/html/2312.06709v5#A1.F7 "Figure 7 ‣ A.1 Multi-Resolution Attention ‣ Appendix A E-RADIO architecture details ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One"). Every layer in the transformer will have a local windowed attention with optional subsampling via convolutional operator. For example, if susbampling is dissabled, then it is just a standard windowed attention. If the subsampling ratio is 2, then the feature map is downsampled by a factor of 2, windowed attention is performed, and then the feature map is upsampled to the original resolution with deconvolution. For FasterVIT2 models, we interleave subsampled attention with ratio 2 and the normal attention with no subsampling.

### A.2 Configurations

All models in the family follow the same configuration except the embedding dimension (hide dimension). We simply scale it up with bigger models. Other parameters:

*   •Input resolution is 224 
*   •In-stem contains 2 3x3 convolutions with stride 2 
*   •Total stages: 2 convolutional and 2 transformer 
*   •First stage takes input feature size of 56x56, has 3 layers with C2f structure from YOLO8[[32](https://arxiv.org/html/2312.06709v5#bib.bib32)]. 
*   •Second stage takes input feature size of 28x28, has 3 layers of C2f. 
*   •Third stage takes features of size 14x14, has 5x multi-resolution attention, window size 7. 
*   •Forth stage takes features of size 7x7, has 5x windowed attention of window size 7. 
*   •Embedding dimension for different model variants: XT - 64, T - 80, S - 96, B - 128, L - 192. The smallest XT and T models have [1, 3, 4, 5] layers for each of 4 stages. 
*   •Output features have resolution of 14x14 and are obtained by upsampling the features of stage 4 by 2x with deconvolution and adding to stage 3 features of size 14x14. 

Appendix B PCA Visualizations
-----------------------------

We visualize various models using PCA to reduce the model’s spatial feature dimensionality down to 3 dimensions, and directly map those to RGB. Most models are only able to handle square inputs at fixed resolutions, however DINOv2 and RADIO can handle arbitrary resolutions and aspect ratios, so we visualize them in both settings.

### B.1 Square Models

### B.2 Flexible Models

Appendix C ViTDet Augmentation
------------------------------

The following python code shows how the alternating window/global architecture of ViTDet [[37](https://arxiv.org/html/2312.06709v5#bib.bib37)] can be applied to a transformer. We take advantage of the fact that transformers are permutation invariant after position encodings have been applied, and thus it’s easy to organize the patch order such that contiguous chunks of patches belong to the same window. Once reordered in this way, alternating between windowed and global attention is achieved simply by absorbing the windows into the batch dimension or returning to the original shape respectively. We also enforce that the final transformer layer always applies global attention.

from einops import rearrange

def reorder_patches(patches:torch.Tensor,

patched_size:Tuple[int,int],

window_size:int):

p_idxs=torch.arange(patches.shape[1])

p_idxs=rearrange(p_idxs,’(wy y wx x)->(wy wx y x)’,

wy=patched_size[0]//window_size,y=window_size,

wx=patched_size[1]//window_size,x=window_size)

p_idxs=p_idxs.reshape(1,-1,1).expand_as(patches)

return torch.gather(patches,p_idxs),p_idxs

def vitdet_aug(blocks:nn.Sequential,

patches:torch.Tensor,

patched_size:Tuple[int,int],

window_sizes:List[int],

num_windowed:int):

B,T,C=patches.shape

window_size=sample(window_sizes)

sq_window_size=window_size**2

patches,p_idxs=reorder_patches(patches,patched_size,window_size)

period=num_windowed+1

for i,block in enumerate(blocks[:-1]):

if i%period==0:

patches=patches.reshape(B*sq_window_size,-1,C)

elif i%period==num_windowed:

patches=patches.reshape(B,T,C)

patches=block(patches)

patches=patches.reshape(B,T,C)

patches=blocks[-1](patches)

ret=torch.empty_like(patches)

ret=ret.scatter(dim=1,index=p_idxs,src=patches)

return ret

Appendix D Comparison with SAM-CLIP [[60](https://arxiv.org/html/2312.06709v5#bib.bib60)]
-----------------------------------------------------------------------------------------

Concurrently with our work, SAM-CLIP was introduced as a method of fusing SAM and CLIP into a single model. Due to the concurrency of effort, we don’t compare our model with the full suite of metrics demonstrated in their method, however, we do have some overlap in key metrics such as Zero-Shot ImageNet-1k, and ADE20k semantic segmentation via linear probing. We present the comparison in table [10](https://arxiv.org/html/2312.06709v5#A4.T10 "Table 10 ‣ Appendix D Comparison with SAM-CLIP [60] ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One"), however we note that there are enough differences between these two models that we can’t conclude one way or another what is the superior approach. Instead we’ll argue that DINOv2 does a better job of ADE20k linear probing than SAM, and thus our significantly higher quality on this metric is likely due to the inclusion of DINOv2, which is a key introduction with our approach.

Table 10: We compare our common key metrics with those demonstrated in SAM-CLIP [[60](https://arxiv.org/html/2312.06709v5#bib.bib60)]. We note that there are numerous differences between the two approaches, including model capacity and architecture. SAM-CLIP uses the ViT-B variant of SAM as a starting point, which implies it’s a ViTDet-B/16 architecture. As a result of this choice, their metrics are computed at a resolution of 1024. RADIO trains a vanilla ViT-H/14 from scratch, and as a result of the flexibility gained via the CPE method, we evaluate Zero-Shot ImageNet1k at a resolution of 432, and we run ADE20k linear probing at a resolution of 512 using the exact same weights. We note that Zero-Shot quality is largely determined by the quality of the CLIP teacher and the capacity of the student. We attribute our superior quality on ADE20k semantic segmentation largely to our inclusion of DINOv2 as a teacher.

Appendix E Automatic Loss Balancing
-----------------------------------

### E.1 Uncertainty

L⁢(x)=∑k 1 2⁢σ k 2⁢L k⁢(x)+log⁡σ k 𝐿 𝑥 subscript 𝑘 1 2 superscript subscript 𝜎 𝑘 2 subscript 𝐿 𝑘 𝑥 subscript 𝜎 𝑘 L(x)=\sum_{k}\frac{1}{2\sigma_{k}^{2}}L_{k}(x)+\log\sigma_{k}italic_L ( italic_x ) = ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT divide start_ARG 1 end_ARG start_ARG 2 italic_σ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT end_ARG italic_L start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ( italic_x ) + roman_log italic_σ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT(4)

where the σ k subscript 𝜎 𝑘\sigma_{k}italic_σ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT values are predicted by the student. In practice, the student predicts b:=log⁡σ k 2 assign 𝑏 superscript subscript 𝜎 𝑘 2 b:=\log\sigma_{k}^{2}italic_b := roman_log italic_σ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT for numerical stability, to avoid division by zero, and to regress unconstrained scalar values.

We make some minor modifications to ([4](https://arxiv.org/html/2312.06709v5#A5.E4 "Equation 4 ‣ E.1 Uncertainty ‣ Appendix E Automatic Loss Balancing ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One")) to make training a bit more stable in our setting. We replace the manual λ 𝜆\lambda italic_λ scalars with the learned uncertainty weights, and add the loss term for large uncertainties. Altogether, this yields:

λ k subscript 𝜆 𝑘\displaystyle\lambda_{k}italic_λ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT=e−b k 2 absent superscript 𝑒 subscript 𝑏 𝑘 2\displaystyle=\frac{e^{-b_{k}}}{2}= divide start_ARG italic_e start_POSTSUPERSCRIPT - italic_b start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_POSTSUPERSCRIPT end_ARG start_ARG 2 end_ARG(5)
L⁢(x)𝐿 𝑥\displaystyle L(x)italic_L ( italic_x )=∑k λ k⁢L k⁢(x)+b k 2 absent subscript 𝑘 subscript 𝜆 𝑘 subscript 𝐿 𝑘 𝑥 subscript 𝑏 𝑘 2\displaystyle=\sum_{k}\lambda_{k}L_{k}(x)+\frac{b_{k}}{2}= ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_λ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_L start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ( italic_x ) + divide start_ARG italic_b start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_ARG start_ARG 2 end_ARG

Let b i(s|v)⁢(x′|Θ i(s))superscript subscript 𝑏 𝑖 conditional 𝑠 𝑣 conditional superscript 𝑥′superscript subscript Θ 𝑖 𝑠 b_{i}^{(s|v)}(x^{\prime}|\Theta_{i}^{(s)})italic_b start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_s | italic_v ) end_POSTSUPERSCRIPT ( italic_x start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT | roman_Θ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_s ) end_POSTSUPERSCRIPT ) be a learned function predicting balance parameters for teacher i 𝑖 i italic_i and summary weight (s)𝑠(s)( italic_s ) or feature vector weight (v)𝑣(v)( italic_v ), we transform equation ([5](https://arxiv.org/html/2312.06709v5#A5.E5 "Equation 5 ‣ E.1 Uncertainty ‣ Appendix E Automatic Loss Balancing ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One")) slighty to:

ψ⁢(x)𝜓 𝑥\displaystyle\psi(x)italic_ψ ( italic_x )=log⁡(1+e x)absent 1 superscript 𝑒 𝑥\displaystyle=\log(1+e^{x})= roman_log ( 1 + italic_e start_POSTSUPERSCRIPT italic_x end_POSTSUPERSCRIPT )(6)
λ i(m)superscript subscript 𝜆 𝑖 𝑚\displaystyle\lambda_{i}^{(m)}italic_λ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_m ) end_POSTSUPERSCRIPT=e−b i(m)⁢(x′)absent superscript 𝑒 superscript subscript 𝑏 𝑖 𝑚 superscript 𝑥′\displaystyle=e^{-b_{i}^{(m)}(x^{\prime})}= italic_e start_POSTSUPERSCRIPT - italic_b start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_m ) end_POSTSUPERSCRIPT ( italic_x start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ) end_POSTSUPERSCRIPT
L⁢(x)𝐿 𝑥\displaystyle L(x)italic_L ( italic_x )=∑i∑m∈{s,v}λ i(m)⁢L i(m)⁢(x)+ψ⁢(b i(m)⁢(x′))absent subscript 𝑖 subscript 𝑚 𝑠 𝑣 superscript subscript 𝜆 𝑖 𝑚 superscript subscript 𝐿 𝑖 𝑚 𝑥 𝜓 superscript subscript 𝑏 𝑖 𝑚 superscript 𝑥′\displaystyle=\sum_{i}\sum_{m\in\{s,v\}}\lambda_{i}^{(m)}L_{i}^{(m)}(x)+\psi% \left(b_{i}^{(m)}(x^{\prime})\right)= ∑ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∑ start_POSTSUBSCRIPT italic_m ∈ { italic_s , italic_v } end_POSTSUBSCRIPT italic_λ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_m ) end_POSTSUPERSCRIPT italic_L start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_m ) end_POSTSUPERSCRIPT ( italic_x ) + italic_ψ ( italic_b start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_m ) end_POSTSUPERSCRIPT ( italic_x start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ) )

The function ψ⁢(x)𝜓 𝑥\psi(x)italic_ψ ( italic_x ) is the familiar “softplus” nonlinear activation function. We drop the division by 2 on the left because, assuming outputs are initially b∼𝒩⁢(0,σ 2)similar-to 𝑏 𝒩 0 superscript 𝜎 2 b\sim\mathcal{N}(0,\sigma^{2})italic_b ∼ caligraphic_N ( 0 , italic_σ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ), then the loss weights will initially have an expected value of 1, matching the naive weighting. On the right, we replace b k 2 subscript 𝑏 𝑘 2\frac{b_{k}}{2}divide start_ARG italic_b start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_ARG start_ARG 2 end_ARG with ψ⁢(x)𝜓 𝑥\psi(x)italic_ψ ( italic_x ) for a few reasons:

*   •When x⪆4 greater-than-or-approximately-equals 𝑥 4 x\gtrapprox 4 italic_x ⪆ 4, then ψ⁢(x)≈x 𝜓 𝑥 𝑥\psi(x)\approx x italic_ψ ( italic_x ) ≈ italic_x, yielding the same expression as before. 
*   •When x≈0 𝑥 0 x\approx 0 italic_x ≈ 0, then ψ′⁢(x)≈1 2 superscript 𝜓′𝑥 1 2\psi^{\prime}(x)\approx\frac{1}{2}italic_ψ start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ( italic_x ) ≈ divide start_ARG 1 end_ARG start_ARG 2 end_ARG, yielding the same expression as before. 
*   •When x<0 𝑥 0 x<0 italic_x < 0, which translates to a loss weight >1 absent 1>1> 1, ψ′⁢(x)→0→superscript 𝜓′𝑥 0\psi^{\prime}(x)\to 0 italic_ψ start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ( italic_x ) → 0, improving stability as the weight gets larger. 
*   •It has range (0,∞)0(0,\infty)( 0 , ∞ ) which aesthetically enforces the loss to be greater than zero. 

### E.2 AdaLoss

In addition to uncertainty auto-balancing, we also explored AdaLoss [[27](https://arxiv.org/html/2312.06709v5#bib.bib27)]. In this formulation, we have:

λ i(m)superscript subscript 𝜆 𝑖 𝑚\displaystyle\lambda_{i}^{(m)}italic_λ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_m ) end_POSTSUPERSCRIPT=1 𝔼⁢(L i(m))absent 1 𝔼 superscript subscript 𝐿 𝑖 𝑚\displaystyle=\frac{1}{\mathbb{E}(L_{i}^{(m)})}= divide start_ARG 1 end_ARG start_ARG blackboard_E ( italic_L start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_m ) end_POSTSUPERSCRIPT ) end_ARG(7)
L⁢(x)𝐿 𝑥\displaystyle L(x)italic_L ( italic_x )=∑i∑m∈{s,v}λ i(m)⁢L i(m)⁢(x)absent subscript 𝑖 subscript 𝑚 𝑠 𝑣 superscript subscript 𝜆 𝑖 𝑚 superscript subscript 𝐿 𝑖 𝑚 𝑥\displaystyle=\sum_{i}\sum_{m\in\{s,v\}}\lambda_{i}^{(m)}L_{i}^{(m)}(x)= ∑ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∑ start_POSTSUBSCRIPT italic_m ∈ { italic_s , italic_v } end_POSTSUBSCRIPT italic_λ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_m ) end_POSTSUPERSCRIPT italic_L start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_m ) end_POSTSUPERSCRIPT ( italic_x )

Appendix F Visual Question Answering Samples
--------------------------------------------

![Image 14: Refer to caption](https://arxiv.org/html/2312.06709v5/extracted/5569764/resources/attention-maps.png)

Figure 8: Visualization of the LLaVA attention maps over the visual features produced by a RADIO encoder. We use one sample image from the GQA[[29](https://arxiv.org/html/2312.06709v5#bib.bib29)] validation set and one associated question: "What color is the helmet in the middle of the image?". For each layer in the language model, we retrieve attention scores for all positions of the visual tokens, average them over all attention heads, and overlay corresponding heat maps with the input image. We can see that as we progress through the layers, the model’s attention focuses on the relevant part of the image. The model’s answer is "Blue".

![Image 15: Refer to caption](https://arxiv.org/html/2312.06709v5/x7.png)

Figure 9: Sample questions from the GQA[[29](https://arxiv.org/html/2312.06709v5#bib.bib29)] and their answers from our LLaVA models, using various image encoders. Answers are painted green when they match the ground truth, pink otherwise.

![Image 16: Refer to caption](https://arxiv.org/html/2312.06709v5/x8.png)

Figure 10: Sample questions from the GQA[[29](https://arxiv.org/html/2312.06709v5#bib.bib29)] and their answers from our LLaVA models, using various image encoders. Answers are painted green when they match the ground truth, pink otherwise.

![Image 17: Refer to caption](https://arxiv.org/html/2312.06709v5/x9.png)

Figure 11: Sample questions from the TextVQA [[55](https://arxiv.org/html/2312.06709v5#bib.bib55)] dataset and their answers from our LLaVA models, using various image encoders. Answers are painted green when they match the ground truth, pink otherwise.

![Image 18: Refer to caption](https://arxiv.org/html/2312.06709v5/x10.png)

Figure 12: Sample questions from the TextVQA [[55](https://arxiv.org/html/2312.06709v5#bib.bib55)] dataset and their answers from our LLaVA models, using various image encoders. Answers are painted green when they match the ground truth, pink otherwise.

![Image 19: Refer to caption](https://arxiv.org/html/2312.06709v5/x11.png)

Figure 13: Sample questions from the TextVQA [[55](https://arxiv.org/html/2312.06709v5#bib.bib55)] dataset and their answers from our LLaVA models, using various image encoders. Answers are painted green when they match the ground truth, pink otherwise.

Figures [9](https://arxiv.org/html/2312.06709v5#A6.F9 "Figure 9 ‣ Appendix F Visual Question Answering Samples ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One") to [13](https://arxiv.org/html/2312.06709v5#A6.F13 "Figure 13 ‣ Appendix F Visual Question Answering Samples ‣ AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One") show sample questions from our Visual Question Answering datasets, together with sample answers when using our vision encoders in a LLaVA setup.

