# Perception Encoder: The best visual embeddings are not at the output of the network

Daniel Bolya<sup>1,\*</sup>, Po-Yao Huang<sup>1,\*</sup>, Peize Sun<sup>1,\*</sup>, Jang Hyun Cho<sup>1,2,\*†</sup>, Andrea Madotto<sup>1,\*</sup>, Chen Wei<sup>1</sup>, Tengyu Ma<sup>1</sup>, Jiale Zhi<sup>1</sup>, Jathushan Rajasegaran<sup>1</sup>, Hanoona Rasheed<sup>3,†</sup>, Junke Wang<sup>4,†</sup>, Marco Monteiro<sup>1</sup>, Hu Xu<sup>1</sup>, Shiyu Dong<sup>5</sup>, Nikhila Ravi<sup>1</sup>, Daniel Li<sup>1</sup>, Piotr Dollár<sup>1</sup>, Christoph Feichtenhofer<sup>1</sup>

<sup>1</sup>Meta FAIR, <sup>2</sup>UT Austin, <sup>3</sup>MBZUAI, <sup>4</sup>Fudan University, <sup>5</sup>Meta Reality Labs

\*Joint first author, <sup>†</sup>Work done during internships at Meta

We introduce Perception Encoder (PE), a state-of-the-art vision encoder for image and video understanding trained via simple vision-language learning. Traditionally, vision encoders have relied on a variety of pretraining objectives, each tailored to specific downstream tasks such as classification, captioning, or localization. Surprisingly, after scaling our carefully tuned image pretraining recipe and refining with our robust video data engine, we find that contrastive vision-language training *alone* can produce strong, general embeddings for all of these downstream tasks. There is only one caveat: *these embeddings are hidden within the intermediate layers of the network*. To draw them out, we introduce two alignment methods: language alignment for multimodal language modeling, and spatial alignment for dense prediction. Together, our PE family of models achieves best-in-class results on a wide variety of tasks, including (1) zero-shot image and video classification and retrieval, simultaneously obtaining 86.6 average zero-shot ImageNet robustness and 76.9 zero-shot Kinetics-400 video classification; (2) document, image, and video Q&A, enabling 94.6 DocVQA, 80.9 InfographicVQA, and 82.7 PerceptionTest with an 8B LLM; and (3) spatial tasks such as detection, tracking, and depth estimation, setting a new COCO state-of-the-art of 66.0 box mAP. To foster further research, we release our models, code, and novel dataset of synthetically and human-annotated videos.

**Code:** [https://github.com/facebookresearch/perception\\_models](https://github.com/facebookresearch/perception_models)

**Dataset:** <https://ai.meta.com/datasets/pe-video/>

## 1 Introduction

For the last decade in computer vision, pretrained vision encoders have been the core building block for most applications requiring *perception*. From million-scale ImageNet [26] pretrained convolutional networks [42, 61, 81, 124, 131] to billion-scale web-pretrained transformers [19, 24, 29, 33, 54, 102, 130, 152, 158], the dominant strategy in vision has consistently been to adapt large-scale pretrained encoders to downstream tasks.

There are many pretraining objectives today, each with distinct characteristics and each yielding representations better suited for specific tasks: vision-language contrastive losses [106, 160] learn a global vision and language embedding well-suited for zero-shot classification and retrieval as well as provide vision-language alignment for open-world [69, 94] and generative tasks [108, 114]; captioning losses [37, 137] learn to predict image descriptions using a language decoder, which transfers well to downstream multimodal language model (MLLM) tasks; and spatially self-supervised losses [44, 98] learn dense spatial correspondences without language supervision, making them useful for tasks requiring precise localization like object detection.

Many works are now attempting to combine two or more of these techniques in different ways [19, 34, 35, 37, 45, 90, 110, 158]. While many have been successful, the complexity of these strategies grows exponentially with number of use cases, which can make scaling difficult. There has not yet been shown a *single, simple, and easily scalable* pretraining technique that can learn state-of-the-art features for all downstream tasks.

In this work we discover that *global vision-language contrastive learning alone* can be one such approach. After building a state-of-the-art contrastive model for image and video, we found a surprising result: *inside the model were specific features aligned to OCR, VQA, grounding, detection, depth estimation, and tracking*. Compared to the state-of-the-art models with captioning [37] and spatially self-supervised [98] pretraining, our**Figure 1** Perception Encoder (PE) is a family of large-scale vision encoder models with state-of-the-art performance on a large variety of vision tasks. By using a robust contrastive pretraining recipe and finetuning on synthetically aligned videos, PE not only outperforms all existing models on classification and retrieval (§2), but it also internally produces strong, general features that *scale* for downstream tasks (§3). PE unlocks the ability for large-scale contrastive pretraining to transfer to downstream tasks with alignment tuning to capitalize on those general features (§4, §5).

contrastive encoder has specific layers that, when used as frozen features, matches or exceeds the performance of the other two pretraining techniques *on tasks they should be the best at*. The only problem is—these features exist at *different layers* for each task. By exploiting this phenomenon with *alignment tuning*, we show it is possible to align these features to the end of the network in order to create state-of-the-art encoders for downstream MLLM and spatial tasks—all following the same easily scalable contrastive pretraining.

We begin by building PE<sub>core</sub> (Fig. 1, left), a large-scale contrastively pretrained model with state-of-the-art zero-shot performance on *both* images and video (§2). To accomplish this, we first focus on developing a strong *image-only* contrastive pretraining recipe to extract general knowledge from billion-scale image-text data. Keeping the data and training FLOPs fixed, this recipe significantly improves upon vanilla CLIP in both absolute performance and robustness (§2.1). We then use the resulting model as a frame-based encoder to develop a *video* data engine for generating well-aligned video captions. Finetuning on this synthetic video-text data substantially improves performance on *both image and video* classification and retrieval tasks (§2.2). Motivated by this success, we release a large portion of the data used to train the engine: PE Video Dataset (PVD), consisting of 1M diverse videos with 120K human-refined annotations (§2.3). Finally, we scale our robust image pretraining and well-aligned video finetuning strategy to 2B parameters to produce PE<sub>core</sub>G (§2.4), a single unified encoder that outperforms SigLIP2 [138] on zero-shot image tasks and InternVideo2 [146] on most zero-shot video tasks. We further transfer this power to smaller model scales through distillation.

With the strongest image and video recognition model in hand, we shift our focus to downstream tasks. Remarkably, despite being pretrained with CLIP loss, we find that the *intermediate layers* of PE<sub>core</sub>G can rival AIMv2-3B [37] on language tasks and DINOv2-g [98] on spatial tasks, both of which among the strongest pretrained models in their respective domains. Upon investigation, we attribute this capability to our robust image pretraining strategy, which appears to have unlocked the potential of contrastive pretraining to scale effectively for downstream tasks (§3). However, a challenge remains: the model does not naturally output these features, keeping them hidden internally. To address this, we introduce two *alignment tuning* methods (Fig. 1, right) to extract these strong, general features.

First, in §4, we investigate the most effective technique to align features to the end of the network by adapting to a large language model. This *language alignment* enables us to construct PE<sub>lang</sub>G, which individually outperforms all other popular vision encoders for MLLM tasks. Moreover, when paired with our Perception Language Model (PLM) [21], the combination rivals the latest state-of-the-art MLLMs, like InternVL3 [168].

Second, in §5, we identify a dichotomy in the layers optimal for spatial tasks. By visualizing the features and pinpointing the explicit reason for this dichotomy, we develop a straightforward *spatial alignment* approach: distilling *from the model’s own frozen features* to achieve most of the alignment, complemented by a novel use of SAM 2 [111] for *spatial correspondence* distillation to refine the process. The resulting PE<sub>spatial</sub>G not only outperforms other popular models in depth estimation, tracking, and semantic segmentation, but also sets a new absolute state-of-the-art on COCO [76] detection with a much simpler decoder.

With this family of checkpoints, Perception Encoder unlocks the potential to scale one simple pretraining method to solve many downstream vision tasks. We are releasing our models, code, and PE Video Dataset.## 2 Perception Encoder: Core

To build Perception Encoder (PE), we start by training a large-scale, robust, and highly performant vision-language contrastive model for image *and* video. We have two objectives: first, to enhance the scalability and data efficiency of contrastive training; and second, to create a unified model effective on both image and video.

These goals are somewhat conflicting: image-text data is plentiful and training on images is efficient, but video-text data is scarce and video training is expensive. Thus, we decouple image and video training into two stages. We first develop a strong *image* pretraining recipe (§2.1) with several regularization techniques to create a robust starting point. Then we use the resulting image model as a frame encoder to develop a *video data engine* (§2.2) supported by our novel human-refined video-text dataset (§2.3) to generate aligned captions for video clips. Finally, we finetune the image encoder on the resulting aligned video data (§2.4). Using our data engine design, this short finetuning step substantially improves *both* image and video performance.

### 2.1 Robust Image Pretraining

In the first stage of pretraining, we want to learn as much visual information as possible from a large set of image-text data. Notably, a unique quirk of contrastive training is the loss for a given sample depends on the other samples in the batch. Because each batch is different, there is potential to learn new information every time an example is sampled, even if that sample has been seen before. Thus, we find contrastive learning to benefit from a long training schedule. To exploit this, we design our pretraining recipe with high regularization, stability, and training efficiency in mind.

**Setup.** (Fig. 2.1) We track our changes on a vanilla CLIP model using an OpenCLIP [51] ViT-L/14 model at 224 resolution as a baseline. We keep the training budget fixed to around 1T GFLOPs (*i.e.*, a ZFLOP), and train on a fixed 2.3B image-text dataset curated using the MetaCLIP [152] text-only curation pipeline. For the baseline, we use a global batch size of 32K, class token, AdamW [83], and train for 12B samples seen. To assess the *generality* of the information learned during pretraining, we report not only zero-shot ImageNet val [26] results but also the average performance across a range of robustness metrics, including ImageNet val [26], ImageNet v2 [112], ObjectNet [4], ImageNet Adversarial [47], ImageNet Rendition [46], and ImageNet Sketch [143]. As observed with other pure CLIP models [33, 106, 152], the average robustness metric performance of this vanilla recipe is much lower than ImageNet val alone.

**Figure 2 Robust Image Pretraining.** We tune our pretraining recipe (§2.1) to maximize performance on a fixed set of data, starting with an OpenCLIP [51] ViT-L/14 model. We report cumulative zero-shot classification results for each modification. The inner bars show robustness evaluation, calculated as the average of 6 robustness benchmarks [4, 26, 46, 47, 112, 143], and the outer bars show ImageNet val [26] alone. Several changes significantly improve robustness, indicating that ImageNet val scales more with data, while robustness can scale with refined training techniques.

**Progressive Resolution.** (Fig. 2.2) To enable longer training, we first improve training efficiency. As shown in many works [70, 71, 79, 131, 136], vision encoders work well with a *progressively increasing* resolution schedule. Thus, we *halve* the training FLOPs while maintaining performance by evenly splitting the baseline 12B-sample run into 98, 154, and 224 resolution stages, with 4B samples per stage.

**Increasing Batch Size.** (Fig. 2.3) We use the extra budget to double the batch size from 32K to 64K, increasing the total samples seen from 12B to 24B. Larger batch size means a higher likelihood for there to be a non-trivially novel pair of samples, *i.e.*, hard negatives. This is akin to increasing the “task difficulty” of CLIP and improves ImageNet val by +0.6% and robustness by double of that, +1.1%.

**LAMB Optimizer.** (Fig. 2.4) We switch from AdamW to LAMB [156], which is known to stabilize large batch training. More importantly, LAMB allows us to train stably with a higher learning rate of  $2 \times 10^{-3}$  comparedto the original  $5 \times 10^{-4}$ . We observe that starting with a high learning rate is important to allow the model to adapt to different resolutions. These factors combine for +0.4% on ImageNet val and +0.7% on robustness.

**Increasing Final Resolution.** (Fig. 2.5) A classic finding is that parameters and resolution should be scaled together [36, 131]. Thus, we add a fourth 336 resolution stage at the end of training. To keep the training FLOPs the same, we adjust the training schedule to 10B samples at 98 resolution, 8B at 154, 4B at 224, and 2B at 336. While ImageNet val only increases by +0.5%, robustness improves threefold, rising by +1.4%.

**RoPE.** (Fig. 2.6) We add 2D RoPE [127] to each attention layer to improve extrapolation, keeping the original position embedding. 2D RoPE only improves ImageNet val by +0.3% but enhances robustness by +0.9%.

**Attention Pooling.** (Fig. 2.7) We follow [160] in constructing the CLIP embedding using an attention probing transformer block. Surprisingly, we found keeping the class token as an input to this block is important for small model performance. Together, this improves ImageNet val by +0.3% and robustness by +0.9%.

**Tuned Data Augmentation.** (Fig. 2.8) Despite training on billions of samples, we find data augmentation still important—especially for transfer to unlikely scenarios like in ObjectNet [4]. We add heavy random cropping, brightness/saturation jitter, and horizontal flip. Random cropping encourages using the entire caption, as not everything is in frame. Jitter helps low-light settings and documents. Horizontal flip improves natural images and does not hurt OCR (see §2.5). These improve robustness by +0.7%, notably, ObjectNet by +2.4%.

**Mask Regularization.** (Fig. 2.9) As regularization, we want the model to produce the same features if some patches are not visible. However, passing the CLIP gradients through masked images may negatively alter behavior on unmasked images. Thus, we convert MaskFeat [147] into a regularization loss by duplicating and masking 1/16th of the batch. At the output, the masked tokens are aligned to their unmasked counterparts by maximizing cosine similarity. Care is taken to ensure that the CLIP and masked gradients are disjoint.

**Scaling Behavior.** (Figs. 3 and 4) In Fig. 3, we show the performance of our recipe (Fig. 2.9) *vs.* the original CLIP recipe (Fig. 2.1) across S/14, B/14, and L/14 models. For each benchmark, our recipe scales around the same rate or better than the original CLIP recipe. On some difficult datasets like ObjectNet [4] and ImageNet Adversarial [47], our recipe shows distinctly better scaling. This indicates that the improvements in performance were not at the cost of scalability, meaning we can further benefit from scaling the model size.

**Figure 3 Scaling Behavior (Model Size).** Results before and after our recipe changes (Fig. 2) for S/14, B/14, and L/14 models. Our recipe improves scaling for difficult metrics like ObjectNet [4] and ImageNet Adversarial [47].

In Fig. 4, we additionally show the performance of our recipe *vs.* the original CLIP recipe across L/14 models trained with 120K steps (one-third schedule), 240K steps (two-thirds schedule), and 360K steps (full ablation schedule). All models are their own training runs with full learning rate annealing and the progressive resolution schedule adjusted proportionally. We see nearly linear trends for our recipe on most datasets. This suggests we can train longer for more performance, even at L scale and with 24B samples seen already.

**Figure 4 Scaling Behavior (Training Steps).** Results before and after our recipe changes for an L/14 model trained with 120K, 240K, and 360K steps, adjusting the learning rate and progressive resolution schedules accordingly. Despite our recipe being much stronger than the original, there is still room for further improvement by training longer.## 2.2 Bootstrapping a Video Data Engine with Perception Encoder

With a robust image pretraining recipe settled and its scaling behavior confirmed, our next step is to extend the image-only encoder to accommodate video and build a unified image-video model. Unlike web-scale image-text data, which comes in many cases with human-generated descriptive alt-text information, videos with aligned language annotation are inherently scarce. High-quality human-annotated captions for videos are even rarer. This scarcity presents a unique and significant challenge in training encoders capable of effectively processing video inputs.

Inspired by the recent success of image data engines [58, 64, 96, 111, 151], we extend this concept to develop a robust video data engine that generates well-aligned synthetic captions for a diverse set of videos, facilitating the training of a video encoder. This innovative approach represents the first large-scale exploration of its kind. In the following sections, we introduce the process of building our video data engine.

To bootstrap our contrastive video finetuning, we focus on synthesizing video captions. We build our data engine in three stages: (1) we create a strong baseline video captioner, which we call the Perception Language Model (PLM), described in [21]; (2) we add additional high quality video data with human-refined captions to further enhance the captioner’s quality; (3) we refine and summarize the generated video captions with an LLM to construct a large video dataset to use for the contrastive video finetuning of our Perception Encoder.

**Phase 1: Base Video Captioner (PLM).** We build our data engine on an early version of PLM [21], a multimodal large language model with PE as the vision encoder and Llama [82] as the language decoder. We train PLM on a large-scale collection of open-access image and video datasets [21]. In total, the training dataset consists of 64.7M images and videos covering natural images, charts, documents, exocentric and egocentric videos.

**Phase 2: PLM + Refined Data.** To further boost captioning performance, we collect a set of 265K videos (105K from PVD which we release, see §2.3), caption them with our base PLM model, and ask human raters to refine the captions<sup>1</sup>. We then finetune our base PLM model with this data, significantly improving captioning quality (see Tab. 1).

**Phase 3: LLM Summarization.** We synthesize the final aligned video captions by incorporating the PLM video captions, Llama 3.2 [82] image-only frame captions, and the existing video metadata of video titles and descriptions (Fig. 5). Similar to image alt-text, video metadata contains knowledge often not covered by the image and video captioning models. Thus, combining the two leads to more comprehensive captions. We summarize video captions, frame captions, and video metadata together using the Llama 3.3 70B model to provide the final captions. The prompt used to generate the summary can be found in Appendix A.1.

**Using the Engine.** Finally, we use the resulting data engine bootstrapped with an image-only checkpoint of PE to generate well-aligned, information-dense captions for a diverse set of 22M videos for contrastive finetuning.

**Training with Recaptioned Videos.** Our goal is to develop a unified image *and* video encoder. To encode videos using our existing image encoder, we uniformly sample  $N = 8$  frames from video clips and extract frame-level

**Figure 5 Video Data Engine.** To create aligned video-text data for contrastive training, we use a PE-based video captioner [21] to generate a holistic video caption and an image-level captioner [82] on sampled frames. We then provide those captions as well as the original video metadata to text-only LLM [82] to synthesize a single short, aligned caption optimal for contrastive training.

<table border="1">
<thead>
<tr>
<th rowspan="2">Captioner</th>
<th colspan="2">AuroraCap [13]</th>
<th colspan="2">VCG Diverse [87]</th>
<th>VCG Bench [86]</th>
</tr>
<tr>
<th>Score</th>
<th>Acc</th>
<th>Score</th>
<th>Acc</th>
<th>Score</th>
</tr>
</thead>
<tbody>
<tr>
<td>PLM</td>
<td>2.2</td>
<td>51.9</td>
<td>3.1</td>
<td>65.1</td>
<td>34.3</td>
</tr>
<tr>
<td>PLM + Human-Refined Data</td>
<td><b>3.4</b></td>
<td><b>71.1</b></td>
<td><b>3.6</b></td>
<td><b>79.4</b></td>
<td><b>35.2</b></td>
</tr>
</tbody>
</table>

**Table 1 Video Captioning.** We use an early version of PLM-8B [21], consisting of our image-only PE encoder and a Llama decoder, for captioning. Adding human-refined data greatly boosts captioning performance (higher is better).

<sup>1</sup>The annotators are instructed to remove, correct, and add information from the captions.embeddings with the image encoder. We then apply average pooling over these frame embeddings to obtain video embeddings, which are used for contrastive learning with encoded video captions by the text encoder. Despite being extremely simple, we find this technique surprisingly effective in producing a strong joint image-video encoder. We share this finding with previous studies [19, 84], which note that simple average pooling outperforms more complex pooling strategies like attention-based compression for video.

**Ablations.** In Tab. 2, we conduct an ablation study on the components of the video data engine by finetuning an intermediate image-only checkpoint on 17M of the 22M videos recaptioned by our video data engine. The results show that the video data engine significantly enhances zero-shot classification and retrieval performance for both image and video benchmarks, compared to the image-only baseline encoder (first row). Notably, using the video data engine’s video-level and frame-level captions provides significant improvements over relying solely on metadata such as video title and description (second row), highlighting the importance of building a robust video data engine to compensate for noise in web videos.

Our analysis reveals that the most critical components are the video metadata and PLM’s video caption; however, all components are necessary to achieve peak performance in our video data engine.

In Fig. 6, we investigate the impact of scaling recaptioned video data on a later checkpoint of the same image-only model as in Fig. 2. Notably, scaling synthetic video data demonstrates consistent improvement in both image and video benchmarks. Full results of this scaling experiment can be found in the Appendix 19.

In the top row, scaling synthetic video data consistently improves performance on image benchmarks, with monotonic improvements of +1.1% in ObjectNet and +1.6% in ImageNet Adversarial. ImageNet val and ImageNet v2 have smaller gains, with accuracy increases of 0.3% to 0.5%, plateauing at  $\sim 7M$  samples. We also observe a significant boost to zero-shot retrieval (here, COCO [76]) of +3.8% to +4.1% top-1 recall.

The video tasks listed in the bottom row demonstrate a consistent story. We observe a significant jump in performance between none and 3M videos across all video classification tasks, indicating that there is a domain gap for image-only models that hinders their ability to perform well on video out of the box. Further scaling synthetic video data leads to substantial performance gains in both video classification and retrieval. Video classification accuracy improves consistently by +5.6% to +11.7% without plateauing, while video retrieval shows significant improvements of +7.7 to +15.3 top-1 recall.

<table border="1">
<thead>
<tr>
<th rowspan="2">Title</th>
<th rowspan="2">Description</th>
<th rowspan="2">Video Caption</th>
<th rowspan="2">Frame Caption</th>
<th colspan="6">Image Zero-Shot</th>
<th colspan="4">Video Zero-Shot</th>
</tr>
<tr>
<th>Average Image</th>
<th>ImageNet val [26]</th>
<th>ImageNet v2 [112]</th>
<th>ObjectNet [4]</th>
<th>MS-COCO img→img [76]</th>
<th>MS-COCO img→vid [76]</th>
<th>Average Video</th>
<th>Kinetics 400 [55]</th>
<th>Kinetics 600 [55]</th>
<th>MSR-VTT img→vid [153]</th>
<th>MSR-VTT vid→vid [153]</th>
</tr>
</thead>
<tbody>
<tr>
<td>✓</td>
<td>✓</td>
<td></td>
<td></td>
<td>72.6</td>
<td>83.3</td>
<td>77.8</td>
<td>85.8</td>
<td>49.4</td>
<td>66.8</td>
<td>50.9</td>
<td>69.7</td>
<td>68.4</td>
<td>38.0</td>
<td>27.3</td>
</tr>
<tr>
<td>✓</td>
<td>✓</td>
<td></td>
<td></td>
<td>75.4</td>
<td>83.2</td>
<td>78.2</td>
<td>87.1</td>
<td>47.3</td>
<td>66.0</td>
<td>56.0</td>
<td>74.1</td>
<td>73.5</td>
<td>39.0</td>
<td>37.3</td>
</tr>
<tr>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td></td>
<td>78.2</td>
<td>83.5</td>
<td>78.4</td>
<td>86.8</td>
<td>56.0</td>
<td>74.3</td>
<td>60.9</td>
<td>73.8</td>
<td>73.4</td>
<td>47.6</td>
<td>48.8</td>
</tr>
<tr>
<td>✓</td>
<td>✓</td>
<td>✓*</td>
<td>✓</td>
<td>78.1</td>
<td>83.7</td>
<td>79.0</td>
<td>87.7</td>
<td>54.1</td>
<td>73.0</td>
<td>60.9</td>
<td>75.4</td>
<td>75.1</td>
<td>46.7</td>
<td>46.5</td>
</tr>
<tr>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>78.2</td>
<td>83.7</td>
<td>79.0</td>
<td>87.5</td>
<td>54.6</td>
<td>73.2</td>
<td>61.6</td>
<td>75.8</td>
<td>75.5</td>
<td>47.4</td>
<td>48.1</td>
</tr>
</tbody>
</table>

**Table 2 Video Data Engine Ablation.** We ablate our video data engine in Fig. 5 by finetuning on an in-development image-only version of PE by averaging the frame embeddings to create a single video CLIP embedding. Video captions are generated by PLM trained with or without\* human-refined data (see §2.3). Frame captions are generated by the Llama 3.2 vision model. Each component helps on different metrics, overall culminating in a huge boost to both image and video zero-shot performance.

**Figure 6 Video Data Scaling.** Finetuning on videos recaptioned by the PE video data engine from 0M (baseline image-only model) to 17M samples consistently improves both image and video performance, both classification and retrieval.**Figure 7 PE Video Dataset Example.** A sample from PVD, our released video-text dataset. Initial captions are generated by our video captioning model and then refined by human annotators. Annotators are instructed to add details and remove model hallucination. In this example, the model hallucination “a spoon” is removed; and more details such as “glass bowl” and the action “scraping” are added. See Appendix Fig. 18 for more.

These experiments highlight the quality of our video data engine and its ability to significantly improve encoder performance, even with only a relatively modest 17M videos compared to the billions of images seen during pretraining. Our video data engine is a vital component in build a strong, unified image-video encoder.

## 2.3 PE Video Dataset (PVD)

For the benefit of the community, we release a new video dataset: PE Video Dataset (PVD).<sup>2</sup> PVD comprises of 1M high-quality and diverse videos with accompanying tags and descriptions. The videos are motion-centered, covering both first-person and third-person views with a wide coverage of scenes.

We additionally select 120K of these videos with the highest degree of motion to annotate with detailed captions by generating synthetic captions using our video captioner (§2.2) and employing 200 annotators to verify and refine them. We ask the human annotators to improve the synthetic captions by removing any hallucinations, correcting words that describe the video inaccurately, eliminating repetitive or redundant words to make the caption more concise, and adding any missing actions being performed in the video.

We release two versions of annotations for the 120K PVD subset: (1) Human verified captions: extended summaries with an average length of 57.1 words that provide a high-level description of each video. These captions are suitable for CLIP-style training. (2) Long automated captions: detailed and fine-grained descriptions with an average length of 111.7 words that capture spatial and temporal events. These captions are ideal for fine-grained video understanding.

<table border="1">
<tbody>
<tr>
<td>Videos</td>
<td>998,862</td>
</tr>
<tr>
<td>Human Captions</td>
<td>118,862</td>
</tr>
<tr>
<td>Total Duration</td>
<td>4625 hrs</td>
</tr>
<tr>
<td>Duration (s)</td>
<td>16.7±9.8</td>
</tr>
<tr>
<td>Human Caption Length</td>
<td>57.1±25.4</td>
</tr>
<tr>
<td>Model Caption Length</td>
<td>111.7±43.2</td>
</tr>
</tbody>
</table>

**Table 3 PVD Statistics.**

In Fig. 7, we visualize a video example together with their model and human captions from PE Video Dataset (See Fig. 18 for more). The dataset statistics are summarized in Tab. 3. Finally, We use 105K of these refined samples to improve the data engine (§2.2 phase 2) and 15K as a high-quality video retrieval benchmark.

**PVD Benchmark.** We use 15K of the human-refined video-caption pairs as a held-out test set, which we introduce as a new video retrieval benchmark, PVD Benchmark, to evaluate finegrained video-caption alignment. We follow the format of MSR-VTT [153] to construct the benchmark. We select videos from 10 different categories, including hand actions, object interactions, food preparation, work activities, outdoor scenes, animals, water scenes, object handling, close-up shots, and nature scenes, with an overall average caption length of 51.7 words (see Appendix A.2.3 for statistics). We use PVD Benchmark to evaluate SigLIP [160], SigLIP2 [138], InternVL [19], and PE models, and the results can be found in Tab. 7.

<sup>2</sup>PVD available at <https://ai.meta.com/datasets/pe-video/>## 2.4 A Unified Encoder for Image and Video

Using a robust, scalable image pretraining recipe and video-pretraining data recaptioned by the proposed video data engine, in this section we present **PE<sub>core</sub>**, a unified image-and-video encoder.

**Model Architecture.** To capitalize on the promising scaling behavior observed in §2.1, we scale the largest PE<sub>core</sub> model to 2B parameters<sup>3</sup> (G scale). Tab. 4 shows the detailed model configuration of the vision and text transformers and the dimension of the output clip embedding space.

<table border="1">
<thead>
<tr>
<th>Scale</th>
<th>Tower</th>
<th>Params</th>
<th>Width</th>
<th>Depth</th>
<th>MLP</th>
<th>Heads</th>
<th>CLIP Dim</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">B</td>
<td>Vision</td>
<td>0.09B</td>
<td>768</td>
<td>12</td>
<td>3072</td>
<td>12</td>
<td rowspan="2">1024</td>
</tr>
<tr>
<td>Text</td>
<td>0.31B</td>
<td>1024</td>
<td>24</td>
<td>4096</td>
<td>16</td>
</tr>
<tr>
<td rowspan="2">L</td>
<td>Vision</td>
<td>0.32B</td>
<td>1024</td>
<td>24</td>
<td>4096</td>
<td>16</td>
<td rowspan="2">1024</td>
</tr>
<tr>
<td>Text</td>
<td>0.31B</td>
<td>1024</td>
<td>24</td>
<td>4096</td>
<td>16</td>
</tr>
<tr>
<td rowspan="2">G</td>
<td>Vision</td>
<td>1.88B</td>
<td>1536</td>
<td>50</td>
<td>8960</td>
<td>16</td>
<td rowspan="2">1280</td>
</tr>
<tr>
<td>Text</td>
<td>0.47B</td>
<td>1280</td>
<td>24</td>
<td>5120</td>
<td>20</td>
</tr>
</tbody>
</table>

**Table 4 PE Model Configurations.**

**Smaller Model Distillation.** To maximize the performance of smaller models (B and L scales in Tab. 4), we employ a distillation finetuning approach [49] using PE<sub>core</sub>G as the teacher. This process involves a short finetuning schedule where both the student and teacher models encode image and text inputs separately to compute image-to-text and text-to-image similarity distributions, similar to CLIP training [106]. The student’s distributions are then optimized to match those of the teacher by minimizing KL-divergence, distilling multimodal relational knowledge from the teacher into the student.

Notably, we find that using a smaller softmax temperature for the teacher’s distributions, specifically  $0.5 \times$  the temperature used for the student’s distribution, significantly enhances the effectiveness of knowledge distillation. By leveraging the strong embeddings provided by PE<sub>core</sub>G, our short distillation finetuning schedule significantly boosts the performance of both B and L scale models of PE<sub>core</sub> (see Appendix C.3).

**Model Training.** The training process of PE<sub>core</sub> involves three stages:

1. 1. *Image pretraining.* We scale up image pretraining to 5.4B publicly available image alt-text pairs curated with MetaCLIP [152] and a total of 86B samples seen to ensure convergence (58B for B and L). We use a global batch size of 131K, with progressive resolution from 98 to up to 448 depending on the model.
2. 2. *Image and video finetuning.* Following the initial pretraining, we subsequently finetune the model at max resolution with a short schedule for 50M samples on the image pretraining data (as cooldown) followed by 22M samples on the recaptioned videos with a smaller learning rate and batch size. The video captions are produced using the proposed video data engine (§2.2). For each video clip, we uniformly sample 8 frames, encode them, take their average to produce a single video embedding, and align them with the corresponding video captions using the same contrastive objective in image training.
3. 3. *Smaller model distillation.* We distill the 2B model (G scale) into smaller contrastive pretrained models at B and L scales under their final resolutions, using a short schedule that covers approximately 4B samples seen ( $\sim 8\%$  of the pretraining schedule) with a lower learning rate and no weight decay.

The detailed training configuration and setups are listed in Appendix B.1.1.

## 2.5 Core Results

**Zero-Shot Image Results.** In Tab. 5, we present PE<sub>core</sub>’s performance on zero-shot image benchmarks for classification and retrieval *vs.* the strongest existing models, including SigLIP2 [138] and proprietary models using JFT-3B [29], which is likely tuned for ImageNet. PE<sub>core</sub> outperforms all other contrastive models across the board on all zero-shot tasks, including the highly competitive average of zero-shot ImageNet robustness metrics [4, 26, 46, 47, 112, 143]. This marks a significant achievement, as we are the first to accomplish this in over 3 years without access to Google’s internal JFT-3B [29] or WebLI [17] datasets. And *at the same time*, PE<sub>core</sub> also exceeds the existing state-of-the-art on image-text retrieval and significantly improves on fine-grained classification—the first to simultaneously hold state-of-the-art on all common zero-shot categories.

By harnessing the power of our video data engine, training with a relatively small dataset of 22M videos and their corresponding synthetic captions leads to substantial *gains in image benchmarks*, with average general image classification improving by  $+0.6\%$  with emphasis on more difficult benchmarks (notably  $+1.2\%$

<sup>3</sup>We employ the setup described in §2.1 except for the additional class token (only used for L and B). Interestingly, we find using the same high learning rate ( $2 \times 10^{-3}$ ) to perform well for G. We also did not find scaling the text encoder to be beneficial.<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Encoder Params</th>
<th rowspan="2">Resolution</th>
<th rowspan="2">Data</th>
<th colspan="8">Zero-Shot Classification</th>
<th colspan="8">Zero-Shot Fine-Grained Classification</th>
<th colspan="6">Zero-Shot Retrieval</th>
</tr>
<tr>
<th>Avg Class.</th>
<th>ImageNet val [26]</th>
<th>ImageNet v2 [112]</th>
<th>ObjectNet IN Classes [4]</th>
<th>ImageNet Adversarial [47]</th>
<th>ImageNet Realitions [46]</th>
<th>ImageNet Sketch [143]</th>
<th>Avg Fine.</th>
<th>Food 10/ [9]</th>
<th>Flowers Oxford [97]</th>
<th>Pets Oxford [100]</th>
<th>Cars Stanford [59]</th>
<th>Aircrafts FGVC [88]</th>
<th>Countries 2/ [133]</th>
<th>Scenes SUN397 [150]</th>
<th>Satellite RESMC [20]</th>
<th>Avg Retrieval</th>
<th>MS-COCO <math>tr \rightarrow img</math> [76]</th>
<th>MS-COCO <math>img \rightarrow tr</math> [76]</th>
<th>Flickr-30k <math>tr \rightarrow img</math> [157]</th>
<th>Flickr-30k <math>img \rightarrow tr</math> [157]</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="26"><i>Proprietary</i></td>
</tr>
<tr>
<td>BASIC [102]</td>
<td>2.4B</td>
<td>224</td>
<td>6.6B</td>
<td>84.3</td>
<td>85.7</td>
<td>80.6</td>
<td>82.3</td>
<td>85.6</td>
<td>95.7</td>
<td>76.1</td>
<td>-</td>
<td>95.1</td>
<td>91.2</td>
<td>97.9</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>76.2</td>
<td>72.7</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>CoCa [158]</td>
<td>1.0B</td>
<td>576</td>
<td>4.8B</td>
<td>85.7</td>
<td>86.3</td>
<td>80.7</td>
<td>82.7</td>
<td>90.2</td>
<td>96.5</td>
<td>77.6</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>72.6</td>
<td>51.2</td>
<td>66.3</td>
<td>80.4</td>
<td>92.5</td>
</tr>
<tr>
<td>LiT-22B [24]</td>
<td>21.7B</td>
<td>224</td>
<td>15B</td>
<td>-</td>
<td>85.9</td>
<td>80.9</td>
<td>87.6</td>
<td>90.1</td>
<td>96.0</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td colspan="26"><i>B Scale</i></td>
</tr>
<tr>
<td>SigLIP-B/16<sup>†</sup> [160]</td>
<td>0.1B</td>
<td>224</td>
<td>10B</td>
<td>69.9</td>
<td>76.2</td>
<td>69.5</td>
<td>70.7</td>
<td>45.1</td>
<td>90.2</td>
<td>67.9</td>
<td>69.5</td>
<td>91.6</td>
<td>85.2</td>
<td>94.2</td>
<td>90.8</td>
<td>44.0</td>
<td>15.9</td>
<td>70.0</td>
<td>64.6</td>
<td>69.8</td>
<td>47.2</td>
<td>64.5</td>
<td>77.9</td>
<td>89.6</td>
</tr>
<tr>
<td>SigLIP2-B/16<sup>†</sup> [138]</td>
<td>0.1B</td>
<td>224</td>
<td>10B</td>
<td>73.1</td>
<td>78.2</td>
<td>71.4</td>
<td><b>73.6</b></td>
<td>55.0</td>
<td><b>91.7</b></td>
<td><b>68.9</b></td>
<td>73.1</td>
<td><b>92.8</b></td>
<td>85.7</td>
<td><b>95.4</b></td>
<td><b>93.4</b></td>
<td>54.8</td>
<td>19.2</td>
<td>72.7</td>
<td>71.1</td>
<td>73.7</td>
<td><b>52.1</b></td>
<td>68.9</td>
<td>80.7</td>
<td>93.0</td>
</tr>
<tr>
<td><b>PE<sub>core</sub>B</b></td>
<td>0.1B</td>
<td>224</td>
<td>5.4B</td>
<td><b>73.2</b></td>
<td><b>78.4</b></td>
<td><b>71.7</b></td>
<td>71.9</td>
<td><b>62.4</b></td>
<td>88.7</td>
<td>66.1</td>
<td><b>75.0</b></td>
<td>92.5</td>
<td><b>86.5</b></td>
<td>94.6</td>
<td>92.1</td>
<td><b>57.0</b></td>
<td><b>30.5</b></td>
<td><b>74.0</b></td>
<td><b>72.7</b></td>
<td><b>74.3</b></td>
<td>50.9</td>
<td><b>71.0</b></td>
<td><b>80.8</b></td>
<td><b>94.4</b></td>
</tr>
<tr>
<td colspan="26"><i>L Scale</i></td>
</tr>
<tr>
<td>SigLIP-L/16<sup>†</sup> [160]</td>
<td>0.3B</td>
<td>384</td>
<td>10B</td>
<td>80.7</td>
<td>82.1</td>
<td>75.9</td>
<td>80.9</td>
<td>76.5</td>
<td>95.0</td>
<td>73.6</td>
<td>74.4</td>
<td>95.6</td>
<td>89.4</td>
<td><b>96.8</b></td>
<td>94.8</td>
<td>53.2</td>
<td>24.7</td>
<td>72.5</td>
<td>67.9</td>
<td>74.7</td>
<td>52.8</td>
<td>70.5</td>
<td>82.6</td>
<td>92.9</td>
</tr>
<tr>
<td>SigLIP2-L/16<sup>†</sup> [138]</td>
<td>0.3B</td>
<td>384</td>
<td>10B</td>
<td>83.3</td>
<td>83.1</td>
<td>77.4</td>
<td>84.4</td>
<td>84.3</td>
<td><b>95.7</b></td>
<td><b>75.5</b></td>
<td>78.4</td>
<td>96.1</td>
<td><b>90.0</b></td>
<td>96.4</td>
<td><b>95.8</b></td>
<td>67.0</td>
<td>31.6</td>
<td>74.8</td>
<td>75.5</td>
<td>76.7</td>
<td>55.3</td>
<td>71.4</td>
<td>85.0</td>
<td>95.2</td>
</tr>
<tr>
<td><b>PE<sub>core</sub>L</b></td>
<td>0.3B</td>
<td>336</td>
<td>5.4B</td>
<td><b>83.9</b></td>
<td><b>83.5</b></td>
<td><b>77.9</b></td>
<td><b>84.7</b></td>
<td><b>89.0</b></td>
<td>95.2</td>
<td>73.4</td>
<td><b>80.0</b></td>
<td><b>96.2</b></td>
<td>87.2</td>
<td>96.4</td>
<td>93.7</td>
<td><b>67.8</b></td>
<td><b>45.6</b></td>
<td><b>77.4</b></td>
<td><b>75.7</b></td>
<td><b>78.8</b></td>
<td><b>57.1</b></td>
<td><b>75.9</b></td>
<td><b>85.5</b></td>
<td><b>96.6</b></td>
</tr>
<tr>
<td colspan="26"><i>Unbounded Scale</i></td>
</tr>
<tr>
<td>DFN-H<sup>†</sup> [33]</td>
<td>0.6B</td>
<td>378</td>
<td>5B</td>
<td>81.6</td>
<td>84.3</td>
<td>78.3</td>
<td>79.6</td>
<td>79.6</td>
<td>93.6</td>
<td>73.3</td>
<td>80.5</td>
<td>96.2</td>
<td><b>91.6</b></td>
<td>96.8</td>
<td><b>96.0</b></td>
<td>72.5</td>
<td>37.9</td>
<td>77.4</td>
<td>75.9</td>
<td>75.8</td>
<td>55.6</td>
<td>71.8</td>
<td>82.1</td>
<td>93.6</td>
</tr>
<tr>
<td>InternVL-C [19]</td>
<td>5.5B</td>
<td>224</td>
<td>5B</td>
<td>82.5</td>
<td>83.2</td>
<td>77.3</td>
<td>80.6</td>
<td>83.8</td>
<td>95.7</td>
<td>74.3</td>
<td>76.4</td>
<td>95.3</td>
<td>85.8</td>
<td>96.3</td>
<td>94.4</td>
<td>53.3</td>
<td>35.1</td>
<td>76.3</td>
<td>74.4</td>
<td>78.6</td>
<td><b>58.6</b></td>
<td>74.9</td>
<td>85.0</td>
<td>95.7</td>
</tr>
<tr>
<td>EVA 18B [130]</td>
<td>17.5B</td>
<td>224</td>
<td>2B</td>
<td>83.6</td>
<td>83.8</td>
<td>77.9</td>
<td>82.2</td>
<td>87.3</td>
<td>95.7</td>
<td>74.7</td>
<td>78.8</td>
<td>95.8</td>
<td>86.0</td>
<td>96.1</td>
<td>94.9</td>
<td>59.7</td>
<td>43.1</td>
<td>77.7</td>
<td><b>76.9</b></td>
<td>77.5</td>
<td>56.2</td>
<td>73.6</td>
<td>83.3</td>
<td><b>96.7</b></td>
</tr>
<tr>
<td>EVA 18B+ [130]</td>
<td>17.5B</td>
<td>336</td>
<td>2B</td>
<td>84.1</td>
<td>83.9</td>
<td>78.2</td>
<td>83.6</td>
<td>88.9</td>
<td>95.6</td>
<td>74.3</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>SigLIP2-g-opt<sup>†</sup> [138]</td>
<td>1.1B</td>
<td>384</td>
<td>10B</td>
<td>86.2</td>
<td>85.0</td>
<td>79.8</td>
<td>88.0</td>
<td>90.5</td>
<td><b>96.6</b></td>
<td><b>77.4</b></td>
<td>81.0</td>
<td><b>97.0</b></td>
<td>91.5</td>
<td><b>97.8</b></td>
<td>95.9</td>
<td>73.6</td>
<td>40.1</td>
<td>76.3</td>
<td>75.9</td>
<td>78.0</td>
<td>56.1</td>
<td>72.8</td>
<td><b>86.0</b></td>
<td>95.4</td>
</tr>
<tr>
<td><b>PE<sub>core</sub>G</b> (image only)</td>
<td>1.9B</td>
<td>448</td>
<td>5.4B</td>
<td>86.0</td>
<td>85.2</td>
<td><b>80.2</b></td>
<td>87.1</td>
<td>91.2</td>
<td>96.1</td>
<td>76.1</td>
<td>82.7</td>
<td>96.6</td>
<td>91.0</td>
<td>96.4</td>
<td>94.6</td>
<td>76.7</td>
<td>57.3</td>
<td>77.5</td>
<td>71.8</td>
<td>74.9</td>
<td>53.1</td>
<td>70.9</td>
<td>81.6</td>
<td>93.9</td>
</tr>
<tr>
<td><b>PE<sub>core</sub>G</b></td>
<td>1.9B</td>
<td>448</td>
<td>5.4B</td>
<td><b>86.6</b></td>
<td><b>85.4</b></td>
<td><b>80.2</b></td>
<td><b>88.2</b></td>
<td><b>92.6</b></td>
<td>96.5</td>
<td>76.5</td>
<td><b>83.7</b></td>
<td>96.9</td>
<td>91.4</td>
<td>96.9</td>
<td>94.7</td>
<td><b>78.2</b></td>
<td><b>57.6</b></td>
<td><b>78.5</b></td>
<td>75.8</td>
<td><b>78.9</b></td>
<td>58.1</td>
<td><b>75.4</b></td>
<td>85.7</td>
<td>96.2</td>
</tr>
</tbody>
</table>

**Table 5 Zero-Shot Image Results.** Image zero-shot performance of PE<sub>core</sub> compared to the state-of-the-art for both proprietary and open models. PE<sub>core</sub>G is the first vision encoder to outperform the best models trained on the proprietary JFT-3B [29] and WebLI [17] on general classification. Moreover at all model sizes, PE<sub>core</sub> obtains state-of-the-art results across general classification, retrieval, and finegrained classification. <sup>†</sup>Re-evaluated: DFN by [130]; SigLIP and SigLIP2 by us with the same benchmark settings if not reported in [138] (see Appendix B.1.2).

ObjectNet, +1.4% ImageNet Adversarial) and fine-grained classification by +1.0% on average. Furthermore, due to the high level of detail and alignment of our synthetic captions, zero-shot retrieval is significantly boosted by +3.6% on average. These results emphasize that training with well-aligned video text data does not just improve video performance—it creates a strictly better model for both videos *and* images.

**Zero-Shot Video Results.** We assess the performance of PE<sub>core</sub> on zero-shot video benchmarks by employing the same model as a frame-based video encoder, utilizing 8 uniformly sampled frames, as described in §2.2.

We present the corresponding video results in Tab. 6. Our base image encoder already outperforms all other image-only encoders on both zero-shot classification and retrieval, including SigLIP2-g-opt. With video finetuning, PE<sub>core</sub>G significantly outperforms even native video models that use full temporal attention on video classification, and nearly matches the state-of-the-art on video retrieval using a simple frame-level encoder. This result underscores the importance of our video data engine, resulting in +3.9% on average zero-shot video classification, and a massive +11.1% on retrieval. Moreover, PE<sub>core</sub> does this with much less video data compared to other video-based approaches like InternVideo2 [146] and VideoPrism [164], highlighting the benefits of a joint image-video encoder.

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Encoder Params</th>
<th rowspan="2">Resolution</th>
<th rowspan="2"># Frames</th>
<th rowspan="2">Video Data</th>
<th colspan="8">Zero-Shot Classification</th>
<th colspan="8">Zero-Shot Retrieval</th>
</tr>
<tr>
<th>Avg Class.</th>
<th>Kinetics 400 [55]</th>
<th>Kinetics 600 [55]</th>
<th>Kinetics 700 [55]</th>
<th>UCF 10/ [136]</th>
<th>HMDB 57 [62]</th>
<th>Avg Retrieval</th>
<th>MSR-VTT <math>tr \rightarrow video</math> [76]</th>
<th>MSR-VTT <math>video \rightarrow tr</math> [76]</th>
<th>MSVD <math>tr \rightarrow video</math> [157]</th>
<th>MSVD <math>video \rightarrow tr</math> [157]</th>
<th>ActivityNet <math>tr \rightarrow video</math> [157]</th>
<th>ActivityNet <math>video \rightarrow tr</math> [157]</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="17"><i>B Scale</i></td>
</tr>
<tr>
<td>CLIP [106]</td>
<td>0.1B</td>
<td>224</td>
<td>8</td>
<td>n/a</td>
<td>54.3</td>
<td>58.4</td>
<td>55.1</td>
<td>46.1</td>
<td>68.9</td>
<td>43.2</td>
<td>29.2</td>
<td>30.4</td>
<td>24.2</td>
<td>40.5</td>
<td>57.2</td>
<td>9.1</td>
<td>13.2</td>
</tr>
<tr>
<td>CLIP4CLIP [84]</td>
<td>0.1B</td>
<td>224</td>
<td>12</td>
<td>n/a</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>32.0</td>
<td>-</td>
<td>38.5</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>SigLIP2-B/16<sup>†</sup> [138]</td>
<td>0.1B</td>
<td>224</td>
<td>8</td>
<td>n/a</td>
<td>57.3</td>
<td>58.7</td>
<td>55.0</td>
<td>48.4</td>
<td>82.0</td>
<td>42.3</td>
<td>39.9</td>
<td>38.5</td>
<td>30.1</td>
<td>49.0</td>
<td>67.2</td>
<td>28.6</td>
<td>25.8</td>
</tr>
<tr>
<td><b>PE<sub>core</sub>B</b></td>
<td>0.1B</td>
<td>224</td>
<td>8</td>
<td>22M</td>
<td><b>63.9</b></td>
<td><b>65.6</b></td>
<td><b>65.1</b></td>
<td><b>55.8</b></td>
<td><b>84.6</b></td>
<td><b>48.2</b></td>
<td><b>49.9</b></td>
<td><b>47.6</b></td>
<td><b>47.3</b></td>
<td><b>50.4</b></td>
<td><b>76.7</b></td>
<td><b>39.0</b></td>
<td><b>38.4</b></td>
</tr>
<tr>
<td colspan="17"><i>L Scale</i></td>
</tr>
<tr>
<td>UMT-L [67]</td>
<td>0.3B</td>
<td>224</td>
<td>8</td>
<td>25M</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>47.1</td>
<td>40.7</td>
<td>37.1</td>
<td>49.0</td>
<td>74.5</td>
<td>41.9</td>
<td>39.4</td>
</tr>
<tr>
<td>SigLIP2-L/16<sup>†</sup> [138]</td>
<td>0.3B</td>
<td>384</td>
<td>8</td>
<td>n/a</td>
<td>64.1</td>
<td>65.3</td>
<td>62.5</td>
<td>56.8</td>
<td>86.7</td>
<td>49.3</td>
<td>44.7</td>
<td>41.5</td>
<td>31.4</td>
<td>53.7</td>
<td>74.2</td>
<td>35.9</td>
<td>31.5</td>
</tr>
<tr>
<td><b>PE<sub>core</sub>L</b></td>
<td>0.3B</td>
<td>336</td>
<td>8</td>
<td>22M</td>
<td><b>71.4</b></td>
<td><b>73.4</b></td>
<td><b>72.7</b></td>
<td><b>65.3</b></td>
<td><b>87.1</b></td>
<td><b>58.5</b></td>
<td><b>54.8</b></td>
<td><b>50.3</b></td>
<td><b>50.1</b></td>
<td><b>57.2</b></td>
<td><b>82.4</b></td>
<td><b>46.4</b></td>
<td><b>42.1</b></td>
</tr>
<tr>
<td colspan="17"><i>Unbounded Scale</i></td>
</tr>
<tr>
<td>InternVL [19]</td>
<td>5.5B</td>
<td>224</td>
<td>8</td>
<td>n/a</td>
<td>-</td>
<td>69.1</td>
<td>68.9</td>
<td>60.6</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>44.7</td>
<td>40.2</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>InternVideo2 [146]</td>
<td>1.0B</td>
<td>224</td>
<td>8</td>
<td>102M</td>
<td>70.7</td>
<td>73.1</td>
<td>72.8</td>
<td>64.9</td>
<td>88.8</td>
<td>53.9</td>
<td><b>59.9</b></td>
<td><b>51.9</b></td>
<td>50.9</td>
<td>58.1</td>
<td>83.3</td>
<td><b>60.4</b></td>
<td><b>54.8</b></td>
</tr>
<tr>
<td>VideoPrism-g* [164]</td>
<td>1.1B</td>
<td>288</td>
<td>16</td>
<td>619M</td>
<td>-</td>
<td>76.4</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>39.7</td>
<td><b>71.0</b></td>
<td>-</td>
<td>-</td>
<td>52.7</td>
<td>50.3</td>
</tr>
<tr>
<td>SigLIP2-g-opt<sup>†</sup> [138]</td>
<td>1.1B</td>
<td>384</td>
<td>8</td>
<td>n/a</td>
<td>68.2</td>
<td>69.8</td>
<td>67.0</td>
<td>61.8</td>
<td>90.7</td>
<td>51.8</td>
<td>46.6</td>
<td>43.1</td>
<td>34.2</td>
<td>55.8</td>
<td>74.6</td>
<td>38.3</td>
<td>33.4</td>
</tr>
<tr>
<td><b>PE<sub>core</sub>G</b> (image only)</td>
<td>1.9B</td>
<td>448</td>
<td>8</td>
<td>n/a</td>
<td>70.9</td>
<td>73.1</td>
<td>72.2</td>
<td>64.3</td>
<td>89.5</td>
<td>55.5</td>
<td>47.6</td>
<td>44.3</td>
<td>35.2</td>
<td>54.3</td>
<td>73.9</td>
<td>41.4</td>
<td>36.3</td>
</tr>
<tr>
<td><b>PE<sub>core</sub>G</b></td>
<td>1.9B</td>
<td>448</td>
<td>8</td>
<td>22M</td>
<td><b>74.8</b></td>
<td><b>76.9</b></td>
<td><b>76.1</b></td>
<td><b>69.1</b></td>
<td><b>90.7</b></td>
<td><b>61.1</b></td>
<td>58.7</td>
<td>51.2</td>
<td>49.9</td>
<td><b>59.7</b></td>
<td><b>85.4</b></td>
<td>54.7</td>
<td>51.2</td>
</tr>
</tbody>
</table>

**Table 6 Zero-Shot Video Results.** Video performance of PE<sub>core</sub> compared to recent video and image encoders. PE<sub>core</sub> obtains state-of-the-art in video classification and comparable performance on retrieval benchmarks while using only 22M videos. \*Proprietary models. <sup>†</sup>SigLIP2 are evaluated by us with the same zero-shot prompts frame embedding averaging strategy (as in [19, 84, 106]). See Appendix B.1.2.<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Encoder Params</th>
<th rowspan="2">Resolution</th>
<th rowspan="2">Data</th>
<th colspan="4">Zero-Shot Classification</th>
<th colspan="4">Zero-Shot Retrieval</th>
</tr>
<tr>
<th>ObjectNet [4]<br/>IN Overlap (115)</th>
<th>ObjectNet [4]<br/>All Classes (315)</th>
<th>iNaturalist<br/>2017 [140]</th>
<th>Dollar St.<br/>58 [39, 113]</th>
<th>TextCaps<br/>img→vid [122]</th>
<th>TextCaps Flip<br/>img→vid [122]</th>
<th>PVD Bench<br/>tex→vid</th>
<th>PVD Bench<br/>vid→vid</th>
</tr>
</thead>
<tbody>
<tr>
<td>SigLIP2-B/16 [138]</td>
<td>0.1B</td>
<td>224</td>
<td>10B</td>
<td><b>73.6</b></td>
<td><b>59.1</b></td>
<td>16.9</td>
<td><b>55.9</b></td>
<td>72.0</td>
<td>69.8</td>
<td>53.9</td>
<td>60.1</td>
</tr>
<tr>
<td><b>PE<sub>core</sub>B</b></td>
<td>0.1B</td>
<td>224</td>
<td>5.4B</td>
<td>71.9</td>
<td>58.3</td>
<td><b>25.9</b></td>
<td>52.1</td>
<td><b>72.3</b></td>
<td><b>71.9</b></td>
<td><b>59.8</b></td>
<td><b>61.1</b></td>
</tr>
<tr>
<td>SigLIP2-L/16 [138]</td>
<td>0.3B</td>
<td>384</td>
<td>10B</td>
<td><b>84.4</b></td>
<td>73.2</td>
<td>26.7</td>
<td>57.6</td>
<td>78.0</td>
<td>76.2</td>
<td>61.9</td>
<td><b>67.1</b></td>
</tr>
<tr>
<td><b>PE<sub>core</sub>L</b></td>
<td>0.3B</td>
<td>336</td>
<td>5.4B</td>
<td><b>84.7</b></td>
<td><b>74.3</b></td>
<td><b>35.3</b></td>
<td><b>59.6</b></td>
<td><b>78.5</b></td>
<td><b>78.3</b></td>
<td><b>64.7</b></td>
<td>65.2</td>
</tr>
<tr>
<td>InternVL-C [19]</td>
<td>5.5B</td>
<td>224</td>
<td>5B</td>
<td>80.6</td>
<td>67.2</td>
<td>19.4</td>
<td>58.2</td>
<td>72.3</td>
<td>67.8</td>
<td>63.4</td>
<td>65.1</td>
</tr>
<tr>
<td>SigLIP2-g-opt [138]</td>
<td>1.1B</td>
<td>384</td>
<td>10B</td>
<td>88.0</td>
<td>78.1</td>
<td>31.5</td>
<td>59.3</td>
<td><b>78.8</b></td>
<td>76.9</td>
<td>62.5</td>
<td>67.1</td>
</tr>
<tr>
<td><b>PE<sub>core</sub>G</b></td>
<td>1.9B</td>
<td>448</td>
<td>5.4B</td>
<td><b>88.2</b></td>
<td><b>79.0</b></td>
<td><b>41.1</b></td>
<td><b>62.3</b></td>
<td><b>78.8</b></td>
<td><b>78.7</b></td>
<td><b>77.0</b></td>
<td><b>76.6</b></td>
</tr>
</tbody>
</table>

**Table 7 Additional Zero-Shot Results.** We present several additional zero-shot benchmarks from existing datasets and our own PVD (§2.3) to address evaluation gaps left by standard benchmarks.

**Additional Zero-Shot Benchmarks.** We further evaluate PE<sub>core</sub> on an additional set of zero-shot classification and retrieval benchmarks we construct in Tab. 7 to address key gaps in common benchmarks. For comparison, we also evaluate SigLIP2 [138] and InternVL-C [19] on these benchmarks.

First, we note that the version of ObjectNet [4] that is standard to benchmark robustness (e.g., in Tab. 5) is *not* the full set. ObjectNet consists of 313 classes of objects in challenging and uncommon orientations, locations, and viewpoints. However, the standard version used for benchmarking is a 113 class subset of classes that overlap with ImageNet-1k [26]. Naturally, benchmarking in this way rewards performing well on ImageNet classes over generality. To remove this bias, we construct the full ObjectNet set with all classes and compare to the reduced ObjectNet set in Tab. 7. Surprisingly, we find that while PE<sub>core</sub>G performs +7.6% over InternVL-C and only +0.2% over SigLIP2-g-opt on the reduced ObjectNet set, it performs +11.8% over InternVL-C and +0.9% over SigLIP2-g-opt on the full set of classes, highlighting PE’s generality.

Next, we include iNaturalist [140] as a *zero-shot* benchmark because of its level of specificity with 2,101 fine-grained long-tail classes. PE<sub>core</sub>G outperforms the next best SigLIP2-g-opt model by +9.6%, emphasizing PE’s long tail knowledge. We then evaluate PE’s cultural diversity on Dollar Street [113]<sup>4</sup>, which consists of images of under-represented populations. Here too we find PE<sub>core</sub>G to outperform existing methods, with +3.0% over SigLIP2-g-opt. Further, we test OCR performance by setting up TextCaps [122] as a retrieval dataset. Notably, PE<sub>core</sub> performs on par or better than SigLIP, which is known for good OCR performance. This is potentially surprising, as the horizontal flip augmentation we used during robust pretraining (§2.1) is typically thought to hurt OCR performance. However, instead it seems to have given PE<sub>core</sub> the ability to read backwards: we test the same TextCaps retrieval but with all images horizontally flipped. Other models suffer from this, but PE<sub>core</sub>G’s performance only drops by 0.1%. Finally, we evaluate PE<sub>core</sub>G on the PVD benchmark (§2.3), a challenging video retrieval task on 15K diverse and human-refined videos. Here, PE<sub>core</sub>G significantly outperforms InternVL [19] by +13.6% on text→video and +9.5% to SigLIP2 [138] on video→text.

**Frozen Encoder Probing Results.** To compare against models that are not capable of zero-shot classification, we additionally evaluate PE<sub>core</sub> using k nearest neighbors (following [98]), linear probing (following [19]), and attention probing (following [37]) on top of the ImageNet-1k [26] train set. We present these results in Tab. 8 and compare to other encoders using their reported numbers. In every case, PE<sub>core</sub>G outperforms all existing open encoders, including those with significantly more parameters.

**Summary.** PE<sub>core</sub>, a unified image-video encoder, achieves state-of-the-art performance across zero-shot classification and retrieval on both images and videos on a wide variety of benchmarks. This synergy is made possible by our robust image pretraining recipe (§2.1) and powerful video data engine (§2.2), which together enable the model to effectively leverage the strengths of both image and video data at scale.

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Encoder Params</th>
<th rowspan="2">Resolution</th>
<th rowspan="2">Data</th>
<th colspan="3">Encoder Probing</th>
</tr>
<tr>
<th>ImageNet [26]<br/>KNN</th>
<th>ImageNet [26]<br/>Linear</th>
<th>ImageNet [26]<br/>Attention</th>
</tr>
</thead>
<tbody>
<tr>
<td>DINOv2-g [98]</td>
<td>1.1B</td>
<td>224</td>
<td>145M</td>
<td>83.5</td>
<td>86.5</td>
<td>87.2<sup>†</sup></td>
</tr>
<tr>
<td>RADIOv2.5-g [45]</td>
<td>1.1B</td>
<td>518</td>
<td>-</td>
<td>85.3</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>AIMv2 3B [37]</td>
<td>2.7B</td>
<td>448</td>
<td>7.2B</td>
<td>-</td>
<td>-</td>
<td>89.5</td>
</tr>
<tr>
<td>InternVL-C [19]</td>
<td>5.5B</td>
<td>224</td>
<td>5B</td>
<td>-</td>
<td>88.2</td>
<td>-</td>
</tr>
<tr>
<td>EVA 18B [130]</td>
<td>17.5B</td>
<td>224</td>
<td>2B</td>
<td>-</td>
<td>88.9</td>
<td>-</td>
</tr>
<tr>
<td><b>PE<sub>core</sub>G</b></td>
<td>1.9B</td>
<td>448</td>
<td>5.4B</td>
<td><b>86.8</b></td>
<td><b>89.5</b></td>
<td><b>89.8</b></td>
</tr>
</tbody>
</table>

**Table 8 Encoder Probing Results.** We evaluate PE<sub>core</sub>G’s frozen features using the typical probing methods to compare to models without zero-shot support. <sup>†</sup>from [37].

<sup>4</sup>We use the version provided by [39] and re-evaluate all models to ensure a fair comparison.### 3 General Features in a Contrastive Disguise

PE<sub>core</sub> puts up strong results on the tasks contrastive encoders are known for, like zero-shot classification and retrieval. But while those tasks are useful, they are only a small part of the vision ecosystem. What *really matters* is whether or not the features learned with our pretraining recipe are useful to downstream tasks.

Today’s common wisdom in the vision community cites that different pretraining methods result in features useful for different tasks: e.g., contrastive for classification, captioning for language modeling, and self-supervised learning for spatial tasks. To see how PE<sub>core</sub> stacks up against against models with different pretraining techniques, we compare its *frozen features* to the state-of-the-art large-scale models for captioning (AIMv2-3B [37]) and self-supervised learning (DINOv2-g [98]) on a variety of downstream tasks.

**Layerwise Feature Analysis.** We summarize the results of our frozen feature analysis in Fig. 8 for several downstream benchmarks in 3 categories: classification, language modeling, and spatial tasks. For classification, we probe each model using a randomly initialized cross attention transformer block. For language alignment, we use the Perception Language Model (PLM) [21] frozen encoder evaluation setup, learning a projector and finetuning a decoder-only LLM (see §4), and for spatial tasks we train with several different decoders (ViTDet [72] Mask-RCNN [43] with Absolute Win [7] for detection, DPT [109] for depth, and zero-shot feature correspondence for tracking [52]). For each experiment, we sweep over the layers of the model as the optimal features are not necessarily the last [18]. In each case, we use an equivalent image size (window size for detection) of 32 × 32 tokens. In each plot, we normalize performance by the maximum and minimum performance across models on that task.

**An Alignment Problem.** This analysis reveals several insights. First, as expected, AIMv2 performs well at classification and the best at visual Q&A language tasks. Similarly, DINOv2 performs the well on spatial tasks like detection, depth, and even performs the best at grounding through an LLM. Then as already established by other works: DINOv2 lacks performance on OCR tasks [134]. This is no secret, but what is interesting is that its performance *peaks in the middle of the network* and then drops significantly by the end. And so does the performance of other models for other downstream tasks (AIMv2: tracking, grounding, detection; DINOv2: VQ&A, grounding).

**Figure 8 Layer Analysis.** Evaluating intermediate layers as frozen features across tasks for different pretraining methods: captioning (AIMv2-3B [37], left), spatially self-supervised (DINOv2-g [98], middle), and our contrastive recipe (PE<sub>core</sub>G, right). Vertical lines denote the best layer and horizontal lines the best performance across models. As expected, AIMv2 performs well on language but not spatial, and DINOv2 performs well on spatial but not language. But surprisingly, *intermediate layers* of PE<sub>core</sub>G perform well on *both* language modeling and spatial tasks.PE<sub>core</sub> exhibits similar behavior, but with unexpected results. Unlike the others, in earlier layers of the network PE<sub>core</sub> *performs well on all tasks, often matching or exceeding the leading models*. Remarkably, PE has intermediate layers that perform near to or on par with AIMv2 for language tasks and DINOv2 for spatial tasks, despite being trained with contrastive loss. Depth estimation is particularly noteworthy, as contrastive encoders are not typically considered state-of-the-art in that area.

However, in almost all cases this strong performance *diminishes rapidly* towards the end of the network. In fact, the performance of PE<sub>core</sub> in the final layer is *abysmal* for certain tasks, such as LLM-based grounding (the reason for which will become apparent in §5). This behavior is less pronounced the closer the downstream task is to the pretraining method, suggesting an *alignment problem*. Specifically, a well-tuned large-scale contrastive model can learn general embeddings in the process of fitting its objective, *but it fails to output them*. Therefore, to reveal these embeddings, the model must be subsequentially aligned to downstream tasks.

**Analysis.** The finding that pure CLIP models possess features which match the performance of state-of-the-art pretraining methods in their specialized domains is new. In fact, recent work [31] has shown the opposite—that CLIP models fail to scale on downstream tasks. We next investigate how our approach yields these results.

To start, we perform layerwise frozen feature analysis on COCO detection. PE<sub>core</sub> was particularly “peaky” on this task in Fig. 8, with its best layer on par with DINOv2, but last layer significantly worse. We already ablated each change we made from vanilla CLIP in Fig. 2 using a ViT-L/14 model. So to retrace our steps, we run frozen feature analysis on those checkpoints. For efficiency, we perform this experiment at a lower resolution and only sample even layers. In Fig. 9, we report COCO box mAP for the last and best layers for each cumulative ablation, along with the index of the best layer. Further, we plot the layerwise performance for each change in Fig. 10.

Surprisingly, the simple changes we made in §2.1 to construct our pretraining recipe overall improved the best layer’s performance by *almost 10 mAP* over vanilla CLIP! Some changes like high resolution (5) and RoPE (6) improving spatial features is to be expected, but unexpectedly data augmentation (8) and *especially* progressive resolution (2) help considerably. It is possible that contrastive pretraining is prone to overfit to the “global” nature of the task through “global tokens” [23]. However, as the model cannot maintain global tokens in the same place due to the resolution progressively changing, it is forced to be more robust. Also of note is that both progressive resolution (2) and attention pooling (7) move the argmax layer deeper into the network (rightmost column of Fig. 9). Attention pooling in particular alters the whole shape of the layerwise performance curve (Fig. 10), while the other changes typically only raise or lower it.

**Figure 9 The Downstream Effects of Robust Pretraining.** The ViT-L/14 checkpoints from Fig. 2 evaluated as frozen features on COCO [76] using Mask R-CNN [43]. We report the last layer performance, best layer performance, and the best layer’s index.

**Figure 10 Layer Analysis** corresponding to the results presented in Fig. 9.

Potentially more interesting is what did not improve performance: specifically, increasing the batch size (3) and using LAMB with a high learning rate (4). Both of these changes explicitly help the model fit the CLIP loss better, which after a certain point may not improve the general features. Moreover, while the best layer overall improved significantly, the last layer performance stagnated after (2). This suggests that constructing the global CLIP token requires a substantial “decoder” (in this case, 6 layers for the final L/14 model). Although the features of this decoder are beneficial for some tasks (e.g., Visual Q&A as shown in Fig. 8), they are not general. Nevertheless, this does not prevent the model from learning general features; it merely limits their expression in the output.**Scaling Behavior.** Finding a simple, easily scalable vision pretraining method that produces generally useful features has been the white whale of the vision community for a while. Evidently, our robust recipe can enable contrastive pretraining to produce general features. So that begs the question, “does it scale?”

We can answer this question in the same way: by performing frozen feature layer analysis of our S/14, B/14, and L/14 scaling ablation checkpoints from Fig. 3. We report the result of that analysis in Fig. 11. We also include our final PE<sub>core</sub>G model using the same setup, but note this is an estimate as our ablation and final schedules are different.

Immediately, we see a stark contrast between the scaling behavior of the vanilla CLIP recipe and ours. While the vanilla recipe quickly plateaus at L scale (300M), the best layer of our robust pretraining recipe demonstrates scaling to G scale (2B) and potentially beyond—despite being trained with a decidedly non-spatially aligned global contrastive loss. However, this is the *best* layer. The *last* layer performance still stagnates for both the vanilla recipe and ours. This may be why prior work [31] finds contrastive pretraining to not scale for downstream tasks—CLIP loss obfuscates its general features even with our recipe, placing them several layers deep.

However, this is just for a single spatial task. To see whether the trend is consistent, we repeat this scaling analysis on a wide variety of downstream language modeling tasks using the same frozen evaluation setup as Fig. 8 and report the results in Fig. 12. Surprisingly, the simple change in pretraining recipe improves scaling for most language tasks as well—including output-side grounding (RefCOCO). Note that in this benchmarking setup, the LLM never sees videos during training so the Video Q&A per-layer results are noisy. Yet, the best layer trend is still the same.

Clearly, contrastive pretraining with our robust recipe produces strong general features that scale. However, these features are not going to be much use stuck in the middle of the network. To remedy this, in the remaining sections we will discuss methods for *aligning* these general features to the output of the network for both language modeling and spatial tasks.

**Figure 11 The Downstream Scalability of Robust Pretraining.** Left: frozen feature layer analysis of the S/14, B/14, and L/14 models from Fig. 3 using the same setup as Fig. 9. Right: scaling behavior of the *best* layer for each model. Note: G is our final model and has a different schedule.

**Figure 12 Further Scalability Analysis.** We repeat the analysis from Fig. 11 on a wide range of downstream tasks by adapting to a language model. Each category is an average of several downstream tasks (see §4).## 4 Perception Encoder: Language Alignment

In §3 we have seen that  $PE_{\text{core}}$  already possesses useful features for vision-language modeling. In this section, we *lift* these features through *alignment tuning* to construct a new encoder,  $PE_{\text{lang}}$ , specialized for multimodal large language models (MLLMs). Our principle is to design not only the most performant, but also the most *general* vision encoder for use in MLLM development. To this end, we want a single language-aligned encoder that performs well across language models, across input resolutions, and for a wide variety of MLLM tasks.

**MLLM Evaluation Tasks.** In this section, our main testbed is to adapt vision encoders to MLLMs and test on various MLLM tasks. We evaluate the downstream performance of each MLLM across five task categories: (1) *OCR, Chart, Document Q&A* on ChartQA [165], DocVQA [91], InfoVQA [92] and AI2D [57]; (2) *Visual Q&A* on TextVQA [125], OK-VQA [118], POPE [73], and VQAv2 [40]; (3) *Captioning* on Flickr [157], COCO [76], and No Cap [1]; (4) *Video Understanding* on VideoMME [38], STAR [148], TGIF-QA [53], EgoSchema [89], MVBench [68], and PerceptionTest [105]; and finally (5) *Grounding* on RefCOCO [56].

### 4.1 Language Alignment Method

We begin by searching for the optimal language alignment method. We design our alignment tuning based on the *midtraining* stage of Perception Language Model (PLM) [21], which is to adapt  $PE_{\text{core}}$  to a pretrained decoder-only LLM (Llama 3 [82]) connected by a vision projector. We start with “warmup” training stage with autoregressive next-token prediction loss on 1M image-text samples from pretraining, where everything but the projector is frozen. Then, we proceed to finetune all parameters on 70M data samples [21] covering natural images, documents/charts/diagrams, and videos, using the same next-token prediction loss. After completing this language alignment, we extract the vision encoder from the model and refer to it as  $PE_{\text{lang}}$ .

To arrive at the optimal training configuration presented in PLM [21], we first conduct ablation studies using a 20M subset of the data. In Tab. 9, we ablate the LLM sizes, training parameters, vision projector types, output layers to project, and encoder regularization. We evaluate across OCR Q&A, Captioning, Visual Q&A, and Video Q&A and find the best configuration.

**LLM Setup.** We explore different *scales* (1B or 3B parameters) and *freezing* weights of the LLM. We observe that going from 1B to 3B parameters increases average score by 1.6 points (76.5→78.1). Unfreezing the LLM boosts this number to 78.4.

**Vision Projector.** Using a *2-layer MLP* vision projector instead of a *linear layer* improves the average score from 77.2 to 78.1, while only adding few parameters (13.5M → 27M).

**PE Output Layer.** As shown in §3,  $PE_{\text{coreG}}$  has intermediate layers that perform significantly better than the last layer when used as features for certain tasks. However, it is not clear if that same behavior applies when finetuning. We test applying the projector to layers 41, 47, and 50 (the last layer), and find that layer 47 works best. Incidentally, this is also the optimal layer for frozen VQ&A in Fig. 8.

**PE Regularization.** We apply LayerScale [135] and DropPath [50] to the vision encoder during the alignment, for stabilizing training. This improves the 78.1 average score to 79.9 (+1.8 points). Unfreezing the LLM boosts this number further to 80.1. We choose this configuration (last row) as our final alignment setup.

To construct  $PE_{\text{lang}}$ , we scale this recipe up the 70M samples mentioned above (more details in [21]). In summary, we use a pretrained Llama3.2 3B, unfrozen, with a 2-layer MLP as a vision projector on top of layer  $PE_{\text{coreG}}$  layer 47 (with the last 3 discarded) and regularize the encoder with LayerScale and DropPath. Compared to the 20M sample ablation setting in Tab. 9, the final  $PE_{\text{lang}}$  trained on 70M total samples gives another +2.1 points to 82.2 on the average across OCR Q&A, Captioning, Visual Q&A, and Video Q&A.

**Effects.** The goal of alignment tuning is to *lift* the strong features found in intermediate layers of  $PE_{\text{core}}$  described in §3 to the end of the network. To see if we actually accomplished that, we perform the same layerwise

<table border="1">
<thead>
<tr>
<th>LLM scale</th>
<th>LLM unfrozen</th>
<th>Regularization?</th>
<th>Projector</th>
<th>Layer</th>
<th>Avg.</th>
<th>OCR Q&amp;A<br/>Average of 4</th>
<th>Captioning<br/>Average of 3</th>
<th>Visual Q&amp;A<br/>Average of 4</th>
<th>Video Q&amp;A<br/>Average of 6</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="10"><i>LLM Setup</i></td>
</tr>
<tr>
<td>1B</td>
<td></td>
<td></td>
<td>MLP</td>
<td>47</td>
<td>76.5</td>
<td>60.7</td>
<td>115.1</td>
<td>76.0</td>
<td>54.0</td>
</tr>
<tr>
<td>3B</td>
<td></td>
<td></td>
<td>MLP</td>
<td>47</td>
<td>78.1</td>
<td>65.9</td>
<td>115.7</td>
<td>76.6</td>
<td>54.1</td>
</tr>
<tr>
<td>3B</td>
<td>✓</td>
<td></td>
<td>MLP</td>
<td>47</td>
<td>78.4</td>
<td>65.8</td>
<td>117.6</td>
<td>76.3</td>
<td>53.7</td>
</tr>
<tr>
<td colspan="10"><i>Vision Projector</i></td>
</tr>
<tr>
<td>3B</td>
<td></td>
<td></td>
<td>Linear</td>
<td>47</td>
<td>77.2</td>
<td>64.5</td>
<td>114.1</td>
<td>76.5</td>
<td>53.7</td>
</tr>
<tr>
<td>3B</td>
<td></td>
<td></td>
<td>MLP</td>
<td>47</td>
<td>78.1</td>
<td>65.9</td>
<td>115.7</td>
<td>76.6</td>
<td>54.1</td>
</tr>
<tr>
<td colspan="10"><i>PE Output Layer</i></td>
</tr>
<tr>
<td>3B</td>
<td></td>
<td></td>
<td>MLP</td>
<td>50</td>
<td>75.9</td>
<td>56.6</td>
<td>116.7</td>
<td>76.5</td>
<td>53.7</td>
</tr>
<tr>
<td>3B</td>
<td></td>
<td></td>
<td>MLP</td>
<td>47</td>
<td>78.1</td>
<td>65.9</td>
<td>115.7</td>
<td>76.6</td>
<td>54.1</td>
</tr>
<tr>
<td>3B</td>
<td></td>
<td></td>
<td>MLP</td>
<td>41</td>
<td>76.9</td>
<td>65.5</td>
<td>112.8</td>
<td>75.4</td>
<td>53.9</td>
</tr>
<tr>
<td colspan="10"><i>PE Regularization</i></td>
</tr>
<tr>
<td>3B</td>
<td></td>
<td>✓</td>
<td>MLP</td>
<td>47</td>
<td>79.9</td>
<td>69.0</td>
<td>117.5</td>
<td>77.4</td>
<td>55.6</td>
</tr>
<tr>
<td>3B</td>
<td>✓</td>
<td>✓</td>
<td>MLP</td>
<td>47</td>
<td><b>80.1</b></td>
<td>68.7</td>
<td>118.3</td>
<td>77.0</td>
<td>56.3</td>
</tr>
</tbody>
</table>

**Table 9 Language Alignment.** We find the best configuration to language align  $PE_{\text{coreG}}$  using autoregressive language training.analysis as in Fig. 8 on our final PE<sub>lang</sub>G model and compare it to the original PE<sub>core</sub>G checkpoint it was initialized from. We present the results of this analysis in Fig. 13, and immediately we see that language alignment was a success: across all categories, the performing layer for the aligned model was the last, no matter the performance of the original checkpoint. Notably, our PE<sub>lang</sub> training mix did *not* contain grounding data, which means that this significantly lifted grounding performance is entirely due to the strong intermediate grounding features in PE<sub>core</sub> now being aligned to the end of the network. Moreover, specific domains such as OCR Q&A that *were* represented in the training mix see a significant boost to performance compared to even the best layer of PE<sub>core</sub>, which was already strong. Thus, with an order of magnitude fewer samples compared to pretraining, we were able to *language align* PE<sub>core</sub>G to create a single, strong encoder for all visual language modeling tasks. Following this success, we align PE<sub>core</sub>L in a similar manner to construct PE<sub>lang</sub>L (see [21]).

## 4.2 Comparisons with Existing Vision Encoders

We compare PE<sub>core</sub> and PE<sub>lang</sub> with other vision encoders that are popular choices in MLLM literature: MetaCLIP [152], SigLIP2 [138], CLIP [106], AIMv2 [37], DINOv2 [98], and InternViT2.5 [18]. Overall, these encoders span several different pretraining losses (e.g., contrastive, captioning, self-supervised, and mixed supervision), encoder sizes (from 300M to 6B parameters), and resolutions (from 224 to 512). *For all vision encoders, we find the best intermediate layers to train MLLM for fair comparison* (more in Appendix B.2).

**MLLM Benchmarking Setup.** We connect each vision encoder, including PE<sub>lang</sub>, to a language decoder with a fresh 2-layer MLP projector. Similar to the alignment stage, we first train only the projector on a subset of 1M image-text pairs from pretraining. Then, we train both the projector and LLM on 2.6M visual Q&A pairs,

**Figure 13 Language Alignment.** We analyze how language alignment changes the internal features of PE. Similar to our PE<sub>core</sub> analysis in Fig. 12, we extract PE<sub>lang</sub> and adapt each layer to a new LLM.

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Encoder Params</th>
<th rowspan="2">Resolution<br/>Patch Size</th>
<th colspan="5">OCR / Chart / Doc. Q&amp;A</th>
<th colspan="5">Visual Q&amp;A</th>
<th colspan="5">Captioning</th>
<th colspan="5">Video</th>
</tr>
<tr>
<th>Avg. OCR QA</th>
<th>ChartQA<br/>Acc. [165]</th>
<th>DocVQA<br/>Acc. [91]</th>
<th>Info. QA<br/>Acc. [92]</th>
<th>AI2D<br/>Acc. [57]</th>
<th>Avg. VQA</th>
<th>TextVQA<br/>Acc. [125]</th>
<th>OK-VQA<br/>Acc. [118]</th>
<th>POPE<br/>Acc. [73]</th>
<th>VQAv2<br/>Acc. [40]</th>
<th>Avg. Cap.</th>
<th>Flicker<br/>CIDEr [157]</th>
<th>COCO<br/>CIDEr [76]</th>
<th>No Cap<br/>CIDEr [1]</th>
<th>Avg. Ground.<br/>RefCOCO/g++ [56]</th>
<th>Avg. Video</th>
<th>VideoMMME<br/>Acc. [38]</th>
<th>STAR<br/>Acc. [48]</th>
<th>TGIF-QA<br/>Acc. [35]</th>
<th>EgoSchema<br/>Acc. [89]</th>
<th>MVBench<br/>Acc. [68]</th>
<th>PerceptionTest<br/>Acc. [105]</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="25"><i>256 Tokens per Image</i></td>
</tr>
<tr>
<td>MetaCLIP-L [152]</td>
<td>0.3B</td>
<td>224/14</td>
<td>44.9</td>
<td>47.9</td>
<td>33.0</td>
<td>28.7</td>
<td>70.2</td>
<td>68.4</td>
<td>47.6</td>
<td>62.5</td>
<td>86.9</td>
<td>76.5</td>
<td>110.5</td>
<td>87.5</td>
<td>130.0</td>
<td>114.1</td>
<td>60.6</td>
<td>53.9</td>
<td>46.1</td>
<td>51.0</td>
<td>66.4</td>
<td>58.6</td>
<td>49.4</td>
<td>51.9</td>
</tr>
<tr>
<td>MetaCLIP-G [152]</td>
<td>1.8B</td>
<td>224/14</td>
<td>44.8</td>
<td>47.6</td>
<td>33.1</td>
<td>27.9</td>
<td>70.6</td>
<td>68.8</td>
<td>48.2</td>
<td>63.5</td>
<td>86.5</td>
<td>76.9</td>
<td>111.1</td>
<td>86.5</td>
<td>132.1</td>
<td>114.8</td>
<td>60.5</td>
<td>53.1</td>
<td>45.0</td>
<td>50.7</td>
<td>66.4</td>
<td>56.0</td>
<td>48.7</td>
<td>51.9</td>
</tr>
<tr>
<td><b>PE<sub>lang</sub> G<sup>†</sup></b></td>
<td>1.7B*</td>
<td>224/14</td>
<td>53.7</td>
<td>61.3</td>
<td>47.1</td>
<td>32.2</td>
<td>74.1</td>
<td>71.8</td>
<td>55.1</td>
<td>65.3</td>
<td>86.8</td>
<td>79.8</td>
<td>116.4</td>
<td>91.0</td>
<td>136.9</td>
<td>121.2</td>
<td>65.7</td>
<td>55.5</td>
<td>47.3</td>
<td>55.7</td>
<td>68.9</td>
<td>59.6</td>
<td>48.6</td>
<td>52.9</td>
</tr>
<tr>
<td colspan="25"><i>576 Tokens per Image</i></td>
</tr>
<tr>
<td>CLIP [106]</td>
<td>0.3B</td>
<td>336/14</td>
<td>53.5</td>
<td>61.7</td>
<td>49.5</td>
<td>32.8</td>
<td>70.1</td>
<td>72.7</td>
<td>60.7</td>
<td>63.9</td>
<td>87.3</td>
<td>78.9</td>
<td>113.3</td>
<td>92.0</td>
<td>132.9</td>
<td>115.0</td>
<td>65.0</td>
<td>54.2</td>
<td>46.3</td>
<td>52.1</td>
<td>68.6</td>
<td>57.4</td>
<td>48.5</td>
<td>52.3</td>
</tr>
<tr>
<td>AIMv2-L [37]</td>
<td>0.3B</td>
<td>336/14</td>
<td>53.3</td>
<td>61.6</td>
<td>48.0</td>
<td>32.1</td>
<td>71.4</td>
<td>73.7</td>
<td>62.7</td>
<td>64.3</td>
<td>87.7</td>
<td>80.1</td>
<td>115.2</td>
<td>90.9</td>
<td>135.6</td>
<td>119.2</td>
<td>63.3</td>
<td>52.5</td>
<td>44.3</td>
<td>50.9</td>
<td>67.5</td>
<td>54.4</td>
<td>44.9</td>
<td>53.2</td>
</tr>
<tr>
<td>AIMv2 L Dist. [37]</td>
<td>0.3B</td>
<td>336/14</td>
<td>53.7</td>
<td>61.1</td>
<td>49.4</td>
<td>31.5</td>
<td>72.7</td>
<td>74.1</td>
<td>62.8</td>
<td>64.8</td>
<td>88.3</td>
<td>80.3</td>
<td>117.8</td>
<td>94.7</td>
<td>137.5</td>
<td>121.2</td>
<td>62.6</td>
<td>53.8</td>
<td>44.3</td>
<td>52.4</td>
<td>65.0</td>
<td>57.4</td>
<td>50.0</td>
<td>53.6</td>
</tr>
<tr>
<td>SigLIP2-so [138]</td>
<td>0.4B</td>
<td>384/16</td>
<td>58.9</td>
<td>69.0</td>
<td>58.3</td>
<td>35.2</td>
<td>73.1</td>
<td>76.8</td>
<td>69.8</td>
<td>72.7</td>
<td>88.7</td>
<td>81.6</td>
<td>116.5</td>
<td>92.1</td>
<td>137.7</td>
<td>119.8</td>
<td>67.4</td>
<td>54.5</td>
<td>45.5</td>
<td>53.1</td>
<td>67.2</td>
<td>57.6</td>
<td>49.3</td>
<td>54.5</td>
</tr>
<tr>
<td>SigLIP2-g-opt [138]</td>
<td>1.1B</td>
<td>384/16</td>
<td>56.2</td>
<td>63.1</td>
<td>55.3</td>
<td>34.0</td>
<td>72.4</td>
<td>77.0</td>
<td>70.3</td>
<td>66.7</td>
<td>89.6</td>
<td>81.6</td>
<td>117.7</td>
<td>94.9</td>
<td>137.8</td>
<td>120.3</td>
<td>66.5</td>
<td>53.9</td>
<td>46.2</td>
<td>53.9</td>
<td>66.6</td>
<td>53.8</td>
<td>48.5</td>
<td>54.7</td>
</tr>
<tr>
<td><b>PE<sub>lang</sub> G<sup>†</sup></b></td>
<td>1.7B*</td>
<td>336/14</td>
<td>66.9</td>
<td>76.8</td>
<td>73.6</td>
<td>41.1</td>
<td>76.1</td>
<td>76.2</td>
<td>68.5</td>
<td>66.0</td>
<td>89.1</td>
<td>81.3</td>
<td>119.7</td>
<td>96.1</td>
<td>139.6</td>
<td>123.4</td>
<td>68.9</td>
<td>58.1</td>
<td>48.7</td>
<td>58.9</td>
<td>70.5</td>
<td>61.8</td>
<td>52.7</td>
<td>55.9</td>
</tr>
<tr>
<td colspan="25"><i>1024 Tokens per Image</i></td>
</tr>
<tr>
<td>InternViT 2.5 L [18]</td>
<td>0.3B</td>
<td>448/14</td>
<td>60.6</td>
<td>74.1</td>
<td>59.2</td>
<td>35.9</td>
<td>73.1</td>
<td>74.2</td>
<td>65.4</td>
<td>64.4</td>
<td>87.6</td>
<td>79.6</td>
<td>112.3</td>
<td>88.4</td>
<td>133.7</td>
<td>114.9</td>
<td>66.9</td>
<td>50.6</td>
<td>45.2</td>
<td>44.8</td>
<td>62.7</td>
<td>54.2</td>
<td>46.0</td>
<td>50.5</td>
</tr>
<tr>
<td>SigLIP2-so [138]</td>
<td>0.4B</td>
<td>512/16</td>
<td>63.3</td>
<td>72.1</td>
<td>69.3</td>
<td>39.0</td>
<td>72.7</td>
<td>77.9</td>
<td>74.8</td>
<td>66.0</td>
<td>89.0</td>
<td><b>81.8</b></td>
<td>117.4</td>
<td>93.5</td>
<td>138.3</td>
<td>120.2</td>
<td>69.6</td>
<td>55.8</td>
<td>46.2</td>
<td>55.4</td>
<td>67.0</td>
<td><b>62.0</b></td>
<td>50.0</td>
<td>54.5</td>
</tr>
<tr>
<td><b>PE<sub>core</sub> L</b></td>
<td>0.3B</td>
<td>448/14</td>
<td>59.4</td>
<td>68.7</td>
<td>62.5</td>
<td>36.6</td>
<td>69.7</td>
<td>74.7</td>
<td>67.7</td>
<td>64.3</td>
<td>88.3</td>
<td>78.7</td>
<td>112.7</td>
<td>89.6</td>
<td>133.4</td>
<td>114.9</td>
<td>59.7</td>
<td>50.9</td>
<td>41.7</td>
<td>51.2</td>
<td>61.6</td>
<td>52.6</td>
<td>47.4</td>
<td>50.6</td>
</tr>
<tr>
<td><b>PE<sub>lang</sub> L</b></td>
<td>0.3B</td>
<td>448/14</td>
<td>71.1</td>
<td>81.0</td>
<td>81.9</td>
<td>46.4</td>
<td>75.0</td>
<td>77.1</td>
<td>73.0</td>
<td>65.5</td>
<td>89.3</td>
<td>80.8</td>
<td>117.3</td>
<td>94.3</td>
<td>137.3</td>
<td>120.1</td>
<td>70.5</td>
<td>56.5</td>
<td>47.0</td>
<td>57.2</td>
<td>68.0</td>
<td>59.8</td>
<td>52.3</td>
<td>54.7</td>
</tr>
<tr>
<td>DINOv2-g [98]</td>
<td>1.1B</td>
<td>448/14</td>
<td>30.0</td>
<td>19.6</td>
<td>14.7</td>
<td>24.2</td>
<td>61.5</td>
<td>61.0</td>
<td>19.3</td>
<td>60.4</td>
<td>88.6</td>
<td>75.8</td>
<td>109.4</td>
<td>86.5</td>
<td>131.6</td>
<td>110.1</td>
<td>64.9</td>
<td>49.5</td>
<td>39.7</td>
<td>52.1</td>
<td>60.1</td>
<td>46.8</td>
<td>47.4</td>
<td>50.8</td>
</tr>
<tr>
<td>AIMv2 3B [37]</td>
<td>2.7B</td>
<td>448/14</td>
<td>48.9</td>
<td>40.5</td>
<td>53.9</td>
<td>33.9</td>
<td>67.2</td>
<td>73.0</td>
<td>64.1</td>
<td>64.0</td>
<td>85.2</td>
<td>78.9</td>
<td>115.7</td>
<td>93.8</td>
<td>135.2</td>
<td>118.1</td>
<td>36.1</td>
<td>54.6</td>
<td>45.1</td>
<td>54.5</td>
<td>66.7</td>
<td>55.4</td>
<td>51.7</td>
<td>54.3</td>
</tr>
<tr>
<td>InternViT2.5-6B [18]</td>
<td>5.5B</td>
<td>448/14</td>
<td>59.9</td>
<td>72.3</td>
<td>59.4</td>
<td>35.2</td>
<td>72.5</td>
<td>75.5</td>
<td>68.9</td>
<td>64.9</td>
<td>88.2</td>
<td>80.2</td>
<td>115.0</td>
<td>92.2</td>
<td>136.3</td>
<td>116.3</td>
<td>68.0</td>
<td>49.6</td>
<td>44.5</td>
<td>47.0</td>
<td>62.6</td>
<td>45.8</td>
<td>48.9</td>
<td>48.5</td>
</tr>
<tr>
<td><b>PE<sub>core</sub> G</b></td>
<td>1.9B</td>
<td>448/14</td>
<td>60.8</td>
<td>69.9</td>
<td>65.4</td>
<td>36.7</td>
<td>71.1</td>
<td>73.3</td>
<td>65.9</td>
<td>60.7</td>
<td>88.4</td>
<td>78.0</td>
<td>112.5</td>
<td>91.6</td>
<td>133.6</td>
<td>112.4</td>
<td>66.6</td>
<td>52.0</td>
<td>42.3</td>
<td>53.1</td>
<td>62.9</td>
<td>51.4</td>
<td>48.8</td>
<td>53.6</td>
</tr>
<tr>
<td><b>PE<sub>lang</sub> G</b></td>
<td>1.7B*</td>
<td>448/14</td>
<td><b>72.4</b></td>
<td><b>80.5</b></td>
<td><b>84.4</b></td>
<td><b>48.3</b></td>
<td><b>76.4</b></td>
<td><b>78.1</b></td>
<td><b>75.2</b></td>
<td>65.4</td>
<td><b>90.1</b></td>
<td><b>81.8</b></td>
<td><b>120.1</b></td>
<td><b>96.6</b></td>
<td><b>140.0</b></td>
<td><b>123.6</b></td>
<td><b>71.3</b></td>
<td><b>58.0</b></td>
<td><b>48.0</b></td>
<td><b>60.1</b></td>
<td><b>69.4</b></td>
<td><b>62.0</b></td>
<td><b>52.4</b></td>
<td><b>56.0</b></td>
</tr>
</tbody>
</table>

**Table 10 MLLM Results with Llama 3.1 8B.** We compare various vision encoders at their native resolution using Llama 3.1-instruct 8B [82] as the language model. The table compares models of similar class in number of vision tokens and parameters. PE<sub>lang</sub> shows strong performance across all benchmarks, including against models 3× its size. \*PE<sub>lang</sub> has 1.7B parameters since we discard the last 3 layers during language alignment. <sup>†</sup>Interpolated without extra training.image captions, and image grounding samples (see Appendix B.2 for details). We benchmark at the native resolution of each encoder (with higher resolution tiling results in Appendix C.4). Finally, we ablate over two language decoders, Llama 3.1 8B [82] and QwenLM 2.5 7B [155], to measure generalization across LLMs.

**Results.** Tab. 10 shows benchmarks results for native resolution input across existing encoders, PE<sub>core</sub> and PE<sub>lang</sub>. Notably, AIMv2 [37], InternViT2.5 [18], SigLIP2 [138] and PE<sub>lang</sub> are trained jointly with a language decoder using next token prediction objective, and thus they perform better overall compared to the base contrastive and self-supervised models across all the metrics. However, PE<sub>lang</sub> uses a fraction of the training FLOPs for language alignment tuning, while significantly outperforming all vision encoders by large margin (an average of +3.5 points for G and +2.0 points for L). Similarly, when tiling with 4 tiles and 1 thumbnail (see Appendix Tab. 30), both PE<sub>lang</sub>L and PE<sub>lang</sub>G outperform all existing vision encoders, including InternViT2.5 [18], which was specifically pretrained in a tiling setting and with grounding data. Appendix C.4, shows a breakdown of the RefCOCO results, as well as results for tiling with higher resolution.

**Transferability.** As PE<sub>lang</sub> is aligned with Llama 3.2-instruct 3B, we conduct a separate set of experiments to check if our model performs well with a different base LLM. In Tab. 11 we repeat the native resolution comparison with QwenLM 2.5 7B [155]. Interestingly, PE<sub>lang</sub> not only outperforms all vision encoders in this setting, but it also outperforms InternViT2.5 [18], which is specifically aligned to QwenLM 2 [154] throughout midtraining. In fact, PE<sub>lang</sub>G with QwenLM even improves its performance with Llama in some cases like with OCR Q&A and video benchmarks, emphasizing the generality of our language alignment.

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Encoder Params</th>
<th rowspan="2">Resolution<br/>Patch Size</th>
<th colspan="6">OCR / Chart / Doc. Q&amp;A</th>
<th colspan="4">Visual Q&amp;A</th>
<th colspan="4">Captioning</th>
<th rowspan="2">Avg. Ground.<br/>RefCOCO/gt [56]</th>
<th colspan="6">Video</th>
</tr>
<tr>
<th>Avg. OCR QA</th>
<th>ChartQA<br/>Acc. [165]</th>
<th>DocVQA<br/>Acc. [91]</th>
<th>Info. QA<br/>Acc. [92]</th>
<th>A12D<br/>Acc. [57]</th>
<th>Avg. VQA</th>
<th>TextVQA<br/>Acc. [125]</th>
<th>OK-VQA<br/>Acc. [118]</th>
<th>POPE<br/>Acc. [73]</th>
<th>VQAV2<br/>Acc. [40]</th>
<th>Avg. Cap.</th>
<th>Flicker<br/>CIDEr [157]</th>
<th>COCO<br/>CIDEr [76]</th>
<th>No Cap<br/>CIDEr [1]</th>
<th>Avg. Video</th>
<th>VideoMME<br/>Acc. [38]</th>
<th>STAR<br/>Acc. [148]</th>
<th>TGIF-QA<br/>Acc. [53]</th>
<th>EgoSchema<br/>Acc. [89]</th>
<th>MVBench<br/>Acc. [68]</th>
<th>PerceptionTest<br/>Acc. [105]</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="24"><i>576 Tokens per Image</i></td>
</tr>
<tr>
<td>SigLIP2-so [138]</td>
<td>0.4B</td>
<td>384/16</td>
<td>60.5</td>
<td>72.0</td>
<td>59.1</td>
<td>36.7</td>
<td>74.3</td>
<td>76.2</td>
<td>69.0</td>
<td>65.4</td>
<td>89.2</td>
<td>81.1</td>
<td>116.3</td>
<td>91.6</td>
<td>137.3</td>
<td>120.0</td>
<td>70.0</td>
<td>57.0</td>
<td>51.3</td>
<td>55.8</td>
<td>66.0</td>
<td>61.0</td>
<td>51.9</td>
<td>55.7</td>
</tr>
<tr>
<td>SigLIP2-g-opt [138]</td>
<td>1.1B</td>
<td>384/16</td>
<td>60.8</td>
<td>71.0</td>
<td>60.4</td>
<td>36.7</td>
<td>75.2</td>
<td>76.8</td>
<td>70.3</td>
<td>65.6</td>
<td>89.5</td>
<td>81.8</td>
<td>118.8</td>
<td>96.4</td>
<td>139.0</td>
<td>121.1</td>
<td>69.9</td>
<td>58.3</td>
<td>52.0</td>
<td>57.6</td>
<td>68.1</td>
<td>62.0</td>
<td>52.8</td>
<td>57.4</td>
</tr>
<tr>
<td><b>PE<sub>lang</sub> G<sup>†</sup></b></td>
<td>1.7B*</td>
<td>336/14</td>
<td>66.8</td>
<td>77.5</td>
<td>72.4</td>
<td>41.1</td>
<td>76.4</td>
<td>76.0</td>
<td>67.9</td>
<td>65.4</td>
<td>89.1</td>
<td>81.5</td>
<td>118.8</td>
<td>94.6</td>
<td>139.5</td>
<td>122.3</td>
<td>70.1</td>
<td>60.2</td>
<td>54.6</td>
<td>61.7</td>
<td>69.8</td>
<td>63.6</td>
<td>54.3</td>
<td>57.2</td>
</tr>
<tr>
<td colspan="24"><i>1024 Tokens per Image</i></td>
</tr>
<tr>
<td>InternViT2.5 [18]</td>
<td>0.3B</td>
<td>448/14</td>
<td>60.3</td>
<td>75.4</td>
<td>61.1</td>
<td>36.2</td>
<td>68.4</td>
<td>74.2</td>
<td>65.6</td>
<td>63.7</td>
<td>87.8</td>
<td>79.5</td>
<td>112.1</td>
<td>88.5</td>
<td>133.5</td>
<td>114.1</td>
<td>68.1</td>
<td>55.8</td>
<td>50.3</td>
<td>54.7</td>
<td>66.6</td>
<td>59.0</td>
<td>50.6</td>
<td>53.8</td>
</tr>
<tr>
<td>SigLIP2-so [138]</td>
<td>0.4B</td>
<td>512/16</td>
<td>66.3</td>
<td>77.2</td>
<td>71.9</td>
<td>42.4</td>
<td>73.9</td>
<td><b>77.9</b></td>
<td>74.2</td>
<td>65.6</td>
<td>89.9</td>
<td>81.8</td>
<td>117.1</td>
<td>93.0</td>
<td>138.0</td>
<td>120.3</td>
<td>70.5</td>
<td>55.9</td>
<td>50.3</td>
<td>57.3</td>
<td>67.2</td>
<td>62.6</td>
<td>50.3</td>
<td>47.4</td>
</tr>
<tr>
<td><b>PE<sub>core</sub> L</b></td>
<td>0.3B</td>
<td>448/14</td>
<td>63.5</td>
<td>73.9</td>
<td>67.4</td>
<td>40.5</td>
<td>72.2</td>
<td>75.7</td>
<td>69.2</td>
<td>64.0</td>
<td>89.4</td>
<td>80.2</td>
<td>113.3</td>
<td>88.7</td>
<td>135.2</td>
<td>115.9</td>
<td>66.5</td>
<td>57.3</td>
<td>49.6</td>
<td>57.8</td>
<td>67.7</td>
<td>60.8</td>
<td>52.3</td>
<td>55.5</td>
</tr>
<tr>
<td><b>PE<sub>lang</sub> L</b></td>
<td>0.3B</td>
<td>448/14</td>
<td>70.2</td>
<td>80.6</td>
<td>80.7</td>
<td>46.0</td>
<td>73.5</td>
<td>76.8</td>
<td>72.8</td>
<td>64.1</td>
<td>89.4</td>
<td>81.0</td>
<td>116.4</td>
<td>93.4</td>
<td>137.6</td>
<td>118.1</td>
<td>70.4</td>
<td>58.3</td>
<td>51.6</td>
<td>59.8</td>
<td>67.4</td>
<td>62.2</td>
<td>53.4</td>
<td>55.4</td>
</tr>
<tr>
<td>DINOv2 [98]</td>
<td>1.1B</td>
<td>448/14</td>
<td>31.3</td>
<td>21.7</td>
<td>14.7</td>
<td>24.6</td>
<td>64.3</td>
<td>61.0</td>
<td>18.9</td>
<td>59.5</td>
<td>88.9</td>
<td>76.9</td>
<td>110.1</td>
<td>87.3</td>
<td>132.1</td>
<td>110.8</td>
<td>69.3</td>
<td>54.3</td>
<td>46.9</td>
<td>56.5</td>
<td>63.4</td>
<td>56.8</td>
<td>49.7</td>
<td>52.2</td>
</tr>
<tr>
<td>AIMv2 3B [37]</td>
<td>2.7B</td>
<td>448/14</td>
<td>66.0</td>
<td>76.7</td>
<td>70.5</td>
<td>41.4</td>
<td>75.2</td>
<td><b>77.9</b></td>
<td>74.2</td>
<td><b>66.2</b></td>
<td>89.4</td>
<td><b>81.9</b></td>
<td><b>119.2</b></td>
<td><b>96.4</b></td>
<td>139.2</td>
<td>122.0</td>
<td>67.6</td>
<td>56.3</td>
<td>45.9</td>
<td>58.0</td>
<td>67.8</td>
<td>60.8</td>
<td>51.4</td>
<td>53.9</td>
</tr>
<tr>
<td>InternViT2.5 [18]</td>
<td>5.5B</td>
<td>448/14</td>
<td>64.2</td>
<td>78.2</td>
<td>65.3</td>
<td>39.6</td>
<td>73.6</td>
<td>76.4</td>
<td>70.1</td>
<td>64.5</td>
<td>89.3</td>
<td>81.7</td>
<td>117.6</td>
<td>95.9</td>
<td>138.4</td>
<td>118.6</td>
<td><b>72.8</b></td>
<td>56.1</td>
<td>50.3</td>
<td>59.1</td>
<td>67.3</td>
<td>56.6</td>
<td>51.1</td>
<td>52.2</td>
</tr>
<tr>
<td><b>PE<sub>core</sub> G</b></td>
<td>1.9B</td>
<td>448/14</td>
<td>64.8</td>
<td>75.9</td>
<td>68.8</td>
<td>41.6</td>
<td>72.9</td>
<td>75.2</td>
<td>67.9</td>
<td>62.4</td>
<td>89.7</td>
<td>80.7</td>
<td>113.1</td>
<td>91.7</td>
<td>135.2</td>
<td>112.3</td>
<td>70.5</td>
<td>57.0</td>
<td>48.7</td>
<td>58.3</td>
<td>66.9</td>
<td>60.8</td>
<td>52.9</td>
<td>54.5</td>
</tr>
<tr>
<td><b>PE<sub>lang</sub> G</b></td>
<td>1.7B*</td>
<td>448/14</td>
<td><b>72.9</b></td>
<td><b>81.6</b></td>
<td><b>83.7</b></td>
<td><b>49.5</b></td>
<td><b>76.7</b></td>
<td><b>77.9</b></td>
<td><b>74.9</b></td>
<td>64.5</td>
<td><b>90.3</b></td>
<td><b>81.9</b></td>
<td>118.9</td>
<td>94.6</td>
<td><b>139.8</b></td>
<td><b>122.3</b></td>
<td>72.1</td>
<td><b>60.4</b></td>
<td><b>54.1</b></td>
<td><b>62.5</b></td>
<td><b>68.3</b></td>
<td><b>66.6</b></td>
<td><b>54.2</b></td>
<td><b>56.8</b></td>
</tr>
</tbody>
</table>

**Table 11 MLLM Results with QwenLM 2.5 7B.** Same setting as Tab. 10, but with QwenLM2.5 7B [155] as the language model. Although PE<sub>lang</sub> is aligned to Llama3.2 3B, the language alignment transfers well to a different language model.

**System-Level MLLM Comparison.** In Tab. 12, we conduct a system-level comparison to the state-of-the-art open-access MLLMs: LLaVA-OneVision 7B [66], Gemma3 12B [132], Molmo-D 7B [25], Qwen2 VL 7B [144], InternVL 2.5 8B [18] and the very recent InternVL 3 8B [168]. Each baseline uses a contrastively pretrained ViT (SigLIP-so400M [160], CLIP-L [106], DFN-H [33], and InternViT 2.5 300M [18]). For our PLM-8B we use PE<sub>lang</sub>G as the vision encoder with 36 tiles for images and 32 frames for video and Llama 3.1-instruct 8B as the language decoder (more details in [21]). We show numbers from their respective works or evaluate them ourselves if they are not reported (except for Gemma and InternVL 3). PLM-8B outperforms all other models tested, emphasizing that PE<sub>lang</sub>G can be used to drive strong results across a wide range of tasks.

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Encoder</th>
<th colspan="6">OCR / Chart / Doc. Q&amp;A</th>
<th colspan="4">Visual Q&amp;A</th>
<th colspan="4">Captioning</th>
<th rowspan="2">Avg. Video</th>
<th colspan="6">Video</th>
</tr>
<tr>
<th>Avg. OCR QA</th>
<th>ChartQA<br/>Acc. [165]</th>
<th>DocVQA<br/>Acc. (test) [91]</th>
<th>Info. QA<br/>Acc. (test) [92]</th>
<th>A12D<br/>w/o mask [57]</th>
<th>Avg. VQA</th>
<th>TextVQA<br/>Acc. [125]</th>
<th>OK-VQA<br/>Acc. [118]</th>
<th>POPE<br/>Acc. [73]</th>
<th>VQAV2<br/>Acc. (val) [40]</th>
<th>Avg. Cap.</th>
<th>Flicker<br/>CIDEr [157]</th>
<th>COCO<br/>CIDEr [76]</th>
<th>No Cap<br/>CIDEr [1]</th>
<th>VideoMME<br/>Acc. [38]</th>
<th>STAR<br/>Acc. [148]</th>
<th>TGIF-QA<br/>Acc. [53]</th>
<th>EgoSchema<br/>(test) Acc. [89]</th>
<th>MVBench<br/>Acc. [68]</th>
<th>PerceptionTest<br/>Acc. (test) [105]</th>
</tr>
</thead>
<tbody>
<tr>
<td>LLaVA-OV 7B [66]</td>
<td>SigLIP-so400M</td>
<td>81.4</td>
<td>80.0</td>
<td>86.7</td>
<td>68.8</td>
<td>90.1</td>
<td>79.9</td>
<td>77.3</td>
<td><b>69.6</b></td>
<td>89.2</td>
<td>83.5</td>
<td>79.5</td>
<td>55.7</td>
<td>70.7</td>
<td>112.1</td>
<td>63.8</td>
<td>57.7</td>
<td>66.0</td>
<td>77.2</td>
<td>65.2</td>
<td>57.1</td>
<td>58.1</td>
</tr>
<tr>
<td>Gemma3 12B [132]</td>
<td>SigLIP-so400M</td>
<td>-</td>
<td>75.7</td>
<td>87.1</td>
<td>64.9</td>
<td>-</td>
<td>-</td>
<td>67.7</td>
<td>-</td>
<td>-</td>
<td>71.6</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>54.9</td>
</tr>
<tr>
<td>Qwen2 VL 7B [144]</td>
<td>DFN-H</td>
<td>86.6</td>
<td>83.6</td>
<td>94.5</td>
<td>76.5</td>
<td>91.7</td>
<td>80.9</td>
<td>83.6</td>
<td>67.9</td>
<td>88.3</td>
<td>83.8</td>
<td>93.7</td>
<td>79.9</td>
<td>102.5</td>
<td>98.7</td>
<td>67.7</td>
<td>62.9</td>
<td>67.3</td>
<td>81.8</td>
<td>65.4</td>
<td>61.6</td>
<td>66.9</td>
</tr>
<tr>
<td>InternVL 2.5 8B [18]</td>
<td>InternViT 2.5-300M</td>
<td>87.0</td>
<td>84.6</td>
<td>93.0</td>
<td>77.6</td>
<td><b>92.8</b></td>
<td>79.9</td>
<td>79.3</td>
<td>69.2</td>
<td>90.6</td>
<td>80.6</td>
<td>113.0</td>
<td>96.5</td>
<td>125.8</td>
<td>116.7</td>
<td>72.9</td>
<td>60.6</td>
<td>77.6</td>
<td>91.3</td>
<td>66.2</td>
<td>72.6</td>
<td>68.9</td>
</tr>
<tr>
<td>InternVL 3 8B [168]</td>
<td>InternViT 2.5-300M</td>
<td>87.2</td>
<td><b>86.6</b></td>
<td>92.7</td>
<td>76.8</td>
<td>92.6</td>
<td>-</td>
<td>80.2</td>
<td>-</td>
<td><b>91.1</b></td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td><b>66.3</b></td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>75.4</td>
<td>-</td>
</tr>
<tr>
<td><b>PLM-8B</b></td>
<td><b>PE<sub>lang</sub>G</b></td>
<td><b>88.4</b></td>
<td>85.5</td>
<td><b>94.6</b></td>
<td><b>80.9</b></td>
<td>92.7</td>
<td><b>82.9</b></td>
<td><b>86.5</b></td>
<td><b>69.6</b></td>
<td>89.9</td>
<td><b>85.6</b></td>
<td><b>127.4</b></td>
<td><b>105.6</b></td>
<td><b>146.7</b></td>
<td><b>129.9</b></td>
<td><b>77.9</b></td>
<td>58.3</td>
<td><b>84.9</b></td>
<td><b>95.5</b></td>
<td><b>68.8</b></td>
<td><b>77.1</b></td>
<td><b>82.7</b></td>
</tr>
</tbody>
</table>

**Table 12 MLLM System-Level Comparison.** We show a system-level comparison between PLM-8B based on PE<sub>lang</sub>G and popular open-access models of similar LLM scale using existing encoders. We report test set results where specified.## 5 Perception Encoder: Spatial Alignment

While language alignment with a pretrained LLM decoder is well-established, the best way to spatially align a model is not obvious. As shown in §3,  $PE_{\text{core}}$  already has features that perform well for spatial tasks. However, the layer that performs the best for higher level spatial tasks like detection or depth estimation (layer  $\sim 40$ ) is vastly different than the layer that performs the best for a pure spatial task like tracking (layer  $\sim 30$ ). While we were able to ignore this disparity during language alignment by aligning to an LLM decoder that could do all tasks, classical spatial tasks have decoders that come in all shapes and sizes. It would be impractical to simply align the model using all downstream decoders mirroring language alignment. Thus, we must first answer the question, what is happening in the features at those layers to make them useful for spatial tasks?

### 5.1 Core Feature Analysis

We begin by analyzing the spatial properties of the features for  $PE_{\text{coreG}}$  in the range of layers where it performed optimally for zero-shot tracking in §3. In Fig. 14, we plot (1) the pairwise feature cosine similarity between the pink token and all others, (2) the head average attention map for that token, and (3) the full attention matrix ( $HW \times HW$ ).

**Figure 14  $PE_{\text{coreG}}$  Feature Analysis.** To understand the dichotomy between optimal  $PE_{\text{core}}$  features for spatial tasks observed in Fig. 8, we analyze the spatial properties of the features between layers 30 and 34.

An 18 Layer Decoder. Remarkably, the cause for the tracking performance peak at layer 32 is abundantly clear from observing the visualizations. Up until layer 32, the attention maps remain local. However, that changes abruptly at layer 33, at which point several tokens in the background of the image become “global” tokens. As shown by the vertical lines in the full attention matrix, starting from layer 33 every token attends to them. Thus, every layer 33 and up become part of a *decoder* for global information.

This is not a new phenomenon. Recent work [23] shows this happening in all modern vision transformers above L scale. But notably these “global tokens” are not necessarily harmful. Given the optimal layer for most tasks in Fig. 8 lies within the global token region, the information they aggregate is useful downstream. However, tracking in §3 is zero-shot and relies purely on spatial correspondences, meaning it cannot make use of the global tokens. This explains why tracking peaks right before their introduction, while tasks that rely on semantic understanding or have larger decoders that can benefit from them do well with the later layers.

### 5.2 Spatial Alignment Method

Given the analysis in §5.1, we have two objectives in creating a spatial alignment method: (1) we must preserve the optimal *semantic information* of the model (including the global tokens) that peaks around layer 40, and (2) we must do so while emphasizing *local alignment* in service of spatial tasks with shallow decoders. The first can be easily achieved by aligning with the model’s own features (e.g., with MaskFeat [147]), but the second is more challenging. To accomplish this, we employ the Segment Anything Model (SAM) 2.1 [111] in a novel way to enforce spatial correspondence information in  $PE$ .

**Retaining Semantics.** To retain the strong semantic features from  $PE_{\text{core}}$ , we finetune the model with itself as a teacher. Specifically, we train the model to minimize the cosine similarity between its *last layer* and the frozen layer 41 features of its initialization (a layer around the peak for many tasks in Fig. 8). On its own this would be a tautology, so we apply heavy regularization to the student: DropPath [50] and LayerScale [135] similar to language alignment, as well as performing MaskFeat [147] with 75% masking. We keep the teacherfixed in contrast to other state-of-the-art spatial models, which all employ an EMA teacher [98, 138]. This could potentially help, but we opt for simplicity.

**Encouraging Locality.** While we could “retain” locality by self-distilling from layer 32 features, that may be less effective as we are already distilling another layer of the model. Instead, we turn to a model that is explicitly tuned for locality: SAM [58, 111]. Notably, several works [110, 116, 119] have shown SAM to *not* be an effective teacher when distilling from multiple sources (though recently [45] has shown it can help with some tricks). However, upon observation of the raw features of SAM 2.1-L (Fig. 15), the main problem may be the same one we are currently trying to solve: *SAM has global tokens as well!* In this case, they appear as dark spots in a grid-like arrangement across all examples in Fig. 15 raw features.

**Figure 15 SAM 2.1 Feature Similarity.** The cosine similarity between the pink marked token and all others for SAM 2.1-L [111] features *vs.* our proposed mask logit features.

Using the features of a model that itself has global tokens to mitigate the effect of global tokens is dubious at best. But, we don’t have to use SAM’s *features* to learn locality. At its core, SAM is a model that transforms points into spatially contiguous masks of select object. If what we want is smooth, locally consistent features, we can use the *mask predictions* themselves. Specifically, we query SAM 2.1-L with 1024 points arranged in a  $32 \times 32$  grid. For each point, SAM returns a  $H \times W$  mask logit the size of the image, which it normally would threshold and NMS. However, we instead concatenate those logits into a  $H \times W \times 1024$  tensor and use *that* as the feature map for alignment. This explicitly produces locally well-aligned features compared to the underlying feature space and has no spatial artifacts caused by global tokens, as shown in Fig. 15.

Then to align, we distill the spatial correspondences between tokens by computing their pairwise cosine similarity for both the student and the teacher (creating a  $HW \times HW$  matrix for each) and aligning them with MSE loss. Unlike SAM’s underlying feature space (which [45] shows may be brittle to interpolation), the mask logit features are robust to interpolation, so we simply interpolate them down and train at the PE<sub>core</sub> model’s original 448px resolution. Finally, like for self-distillation we add the same masking and regularization. For both teachers, we apply loss to all tokens and add no extra parameters other than LayerScale.

**Effects.** Again, the goal of alignment is to *lift* the strong features already learned by the core model as shown in §3. Thus, like we did for language alignment in §4.1, we perform layerwise frozen feature analysis on spatial tasks in Fig. 16. This time, we evaluate the original PE<sub>core</sub>G checkpoint as well PE<sub>core</sub>G aligned to its own layer 41, to SAM 2.1 mask logits, and finally both. We denote aligning to both as PE<sub>spatial</sub>G.

Aligning purely based on the original model’s layer 41 features performs well on detection, depth, and semantic segmentation, but falls short for zero-shot tracking, where precise locality is necessary to define boundaries between objects. In contrast, aligning to SAM 2.1 mask logits lowers last layer performance on every task except for tracking, where it significantly improves performance. Understandably, this is because the mask logits have little semantics (see Fig. 17). Thus, the optimal approach is to combine both teachers. As a result, PE<sub>spatial</sub>G not only lifts the features for all tasks to the end of the network, but it also improves over self-alignment alone. Notably, PE<sub>spatial</sub>G’s tracking performance is lower than the SAM-aligned model, but it is still ahead of other methods while being a generally good model, see §5.3.

**Figure 16 Spatial Alignment.** We analyze how our two spatial alignment methods individually change the internal features of PE<sub>core</sub>G. Then we combine both alignment methods to create PE<sub>spatial</sub>G (see Appendix B.3.1).**Last Layer Feature Visualization.** In Fig. 17, we visualize the last layer features for the  $PE_{core}G$  and the 3 aligned models, with similar colors denoting similar features. In the first column, we see why the last layer performance of  $PE_{core}$  is so poor: while the last layer features contain information about the salient objects, they seem to have lost spatial coherence. Aligning to the model’s own layer 41 features fixes this, but its spatial quality is lacking. In contrast, the model aligned to SAM 2.1 mask logits has locally clear features, but without semantics (similar objects have dissimilar features, see row 1 cats and row 2 cows).  $PE_{spatial}$ , using both teachers at once, retains the semantics of  $PE_{core}$  while producing high quality spatial features.

**Figure 17 Last Layer Visualization** for the models in Fig. 16 using 3 dimensional PCA to map features to LCh color space (see Appendix B.3.2). More examples in Appendix C.5.

<table border="1">
<thead>
<tr>
<th rowspan="2">Encoder</th>
<th rowspan="2">Params</th>
<th rowspan="2">Resolution</th>
<th colspan="3">Tracking</th>
<th colspan="3">Segmentation</th>
<th colspan="3">Depth</th>
</tr>
<tr>
<th>DAVIS (<math>\uparrow</math>) [104]</th>
<th>ADE20k (<math>\uparrow</math>) [167]</th>
<th>NYU (<math>\downarrow</math>) [123]</th>
<th>Best</th>
<th>Last</th>
<th>Idx</th>
<th>Best</th>
<th>Last</th>
<th>Idx</th>
</tr>
</thead>
<tbody>
<tr>
<td>OAI CLIP-L [106]</td>
<td>0.3B</td>
<td>224/14</td>
<td>39.4</td>
<td>37.1</td>
<td>17/24</td>
<td>39.4</td>
<td>38.3</td>
<td>19/24</td>
<td>.366</td>
<td>.397</td>
<td>19/24</td>
</tr>
<tr>
<td>AIMv2-3B [37]</td>
<td>2.7B</td>
<td>448/14</td>
<td>54.7</td>
<td>29.3</td>
<td>13/24</td>
<td>41.6</td>
<td>31.9</td>
<td>20/24</td>
<td>.311</td>
<td>.326</td>
<td>16/24</td>
</tr>
<tr>
<td>SigLIP-so [160]</td>
<td>0.4B</td>
<td>384/14</td>
<td>48.7</td>
<td>36.3</td>
<td>16/27</td>
<td>40.1</td>
<td>38.3</td>
<td>22/27</td>
<td>.339</td>
<td>.369</td>
<td>21/27</td>
</tr>
<tr>
<td>SigLIP2-so [138]</td>
<td>0.4B</td>
<td>512/16</td>
<td>51.4</td>
<td>45.3</td>
<td>15/27</td>
<td>44.0</td>
<td>42.9</td>
<td>24/27</td>
<td>.306</td>
<td>.329</td>
<td>25/27</td>
</tr>
<tr>
<td>SigLIP2-g-opt [138]</td>
<td>1.1B</td>
<td>384/16</td>
<td>43.5</td>
<td>38.8</td>
<td>32/40</td>
<td>42.1</td>
<td>41.3</td>
<td>34/40</td>
<td>.302</td>
<td>.324</td>
<td>34/40</td>
</tr>
<tr>
<td>DINOv2-L [98]</td>
<td>0.3B</td>
<td>448/14</td>
<td>58.7</td>
<td>58.2</td>
<td>23/24</td>
<td>47.3</td>
<td>47.3</td>
<td>24/24</td>
<td>.297</td>
<td>.308</td>
<td>23/24</td>
</tr>
<tr>
<td>DINOv2-g [98]</td>
<td>1.1B</td>
<td>448/14</td>
<td>58.5</td>
<td>58.5</td>
<td>40/40</td>
<td>48.7</td>
<td>48.4</td>
<td>37/40</td>
<td>.279</td>
<td>.290</td>
<td>27/40</td>
</tr>
<tr>
<td><math>PE_{core}G</math></td>
<td>1.9B</td>
<td>448/14</td>
<td>56.8</td>
<td>42.8</td>
<td>32/50</td>
<td>41.5</td>
<td>38.6</td>
<td>44/50</td>
<td><b>.249</b></td>
<td><b>.309</b></td>
<td>39/50</td>
</tr>
<tr>
<td><math>PE_{spatial}G</math></td>
<td>1.9B</td>
<td>448/14</td>
<td><b>61.5</b></td>
<td><b>61.5</b></td>
<td>50/50</td>
<td><b>49.3</b></td>
<td><b>48.9</b></td>
<td>49/50</td>
<td>.262</td>
<td><b>.275</b></td>
<td>46/50</td>
</tr>
</tbody>
</table>

**Table 13 Frozen Feature Dense Prediction** including zero-shot tracking, semantic segmentation and depth estimation. We report best and last layer performance, along with which layer was best for each model. See Appendix B.3.3 for experimental settings.

### 5.3 Comparisons with Existing Vision Encoders

**Frozen Feature Dense Prediction.** In Tab. 13, we compare different vision encoder’s frozen features on three dense prediction tasks: DAVIS tracking [104] (J&F) following the training-free setting from [52, 107], ADE20k semantic segmentation [167] (mIoU) linear probing, and NYU depth estimation [123] (RMSE) with a DPT head [109]. For each model, we report both its best layer and last layer performance. Across the board,  $PE_{spatial}$  performs outperforms other state-of-the-art spatial models, with its best features being much better aligned to the last layer than the  $PE_{core}$  it started from. Notably, SigLIP2, which during pretraining combines spatial, captioning, and contrastive losses [138] is *not* aligned well to the last layer in comparison.

**End-to-End Finetuning Detection and Segmentation.** In Tab. 14, we compare  $PE_{core}$  and  $PE_{spatial}$  with other popular vision encoders in the standard *full-finetuning* ViTDet [72] Mask-RCNN [43] setting using COCO [76] and LVIS [41] as benchmarks. In this controlled experiment,  $PE_{spatial}$  is state-of-the-art among various vision backbones. This is significant, as contrastive encoders (especially large ones like MetaCLIP-G [152]) usually perform very poorly on detection, with smaller models often performing better. Typically, encoders only scale for detection if using spatial pretraining or a significant amount of detection data [98] is used to align them directly to downstream tasks. In contrast,  $PE_{spatial}$  *uses no detection data for alignment*, making it general.

**System-Level Detection.** In Tab. 15, we provide a system-level end-to-end finetuning comparison *vs.* the absolute state-of-the-art in COCO detection. With only Object365 [120] as extra detection data,  $PE_{spatial}$  can match the performance of more complex models tuned for detection, while only using a simple DETR-style decoder [12, 99].  $PE_{spatial}$  marks the first general, contrastively pretrained model to accomplish this.

<table border="1">
<thead>
<tr>
<th rowspan="2">Encoder</th>
<th rowspan="2">Params</th>
<th rowspan="2">Pretrain Resolution</th>
<th colspan="2">LVIS [41]</th>
<th colspan="2">COCO [76]</th>
</tr>
<tr>
<th>AP<sub>box</sub></th>
<th>AP<sub>mask</sub></th>
<th>AP<sub>box</sub></th>
<th>AP<sub>mask</sub></th>
</tr>
</thead>
<tbody>
<tr>
<td>OAI CLIP-L [106]</td>
<td>0.3B</td>
<td>224/14</td>
<td>45.0</td>
<td>41.9</td>
<td>54.0</td>
<td>47.5</td>
</tr>
<tr>
<td>MetaCLIP-G [152]</td>
<td>1.8B</td>
<td>224/14</td>
<td>45.1</td>
<td>41.9</td>
<td>53.2</td>
<td>46.7</td>
</tr>
<tr>
<td>SigLIP-so [160]</td>
<td>0.4B</td>
<td>224/14</td>
<td>45.0</td>
<td>41.9</td>
<td>54.4</td>
<td>47.6</td>
</tr>
<tr>
<td>MAE-L [44]</td>
<td>0.3B</td>
<td>224/14</td>
<td>46.1</td>
<td>43.9</td>
<td>55.6</td>
<td>49.3</td>
</tr>
<tr>
<td>EVA02-L [35]</td>
<td>0.3B</td>
<td>224/14</td>
<td>49.3</td>
<td>45.2</td>
<td>54.9</td>
<td>48.2</td>
</tr>
<tr>
<td>SigLIP2-so [138]</td>
<td>0.4B</td>
<td>512/16</td>
<td>49.3</td>
<td>45.6</td>
<td>56.0</td>
<td>49.4</td>
</tr>
<tr>
<td>SigLIP2-g-opt [138]</td>
<td>1.1B</td>
<td>384/16</td>
<td>52.9</td>
<td>48.5</td>
<td>57.1</td>
<td>50.2</td>
</tr>
<tr>
<td>DINOv2-L [98]</td>
<td>0.3B</td>
<td>518/14</td>
<td>46.7</td>
<td>43.5</td>
<td>55.7</td>
<td>49.0</td>
</tr>
<tr>
<td>DINOv2-g [98]</td>
<td>1.1B</td>
<td>518/14</td>
<td>51.5</td>
<td>47.3</td>
<td>57.2</td>
<td>50.0</td>
</tr>
<tr>
<td><math>PE_{core}G</math></td>
<td>1.9B</td>
<td>448/14</td>
<td>51.9</td>
<td>47.9</td>
<td>57.0</td>
<td>49.8</td>
</tr>
<tr>
<td><math>PE_{spatial}G</math></td>
<td>1.9B</td>
<td>448/14</td>
<td><b>54.2</b></td>
<td><b>49.3</b></td>
<td><b>57.8</b></td>
<td><b>50.3</b></td>
</tr>
</tbody>
</table>

**Table 14 End-to-End Finetuning Detection and Segmentation** using Mask R-CNN [43] and ViTDet [72] in a controlled setting. Details in Appendix B.3.4.

<table border="1">
<thead>
<tr>
<th>Encoder</th>
<th>Params</th>
<th>Detector</th>
<th>COCO AP<sub>box</sub></th>
</tr>
</thead>
<tbody>
<tr>
<td>SwinV2-G [80]</td>
<td>3.0B</td>
<td>HTC++ [14]</td>
<td>62.5</td>
</tr>
<tr>
<td>Swin-L [79]</td>
<td>0.3B</td>
<td>DINO [161]</td>
<td>63.2</td>
</tr>
<tr>
<td>EVA02-L [35]</td>
<td>0.3B</td>
<td>Cascade [11]</td>
<td>64.1</td>
</tr>
<tr>
<td>InterImage-G [145]</td>
<td>3.0B</td>
<td>DINO [161]</td>
<td>65.3</td>
</tr>
<tr>
<td>EVA02-L [35]</td>
<td>0.3B</td>
<td>CoDETR [169]</td>
<td>65.9</td>
</tr>
<tr>
<td><math>PE_{spatial}G</math></td>
<td>1.9B</td>
<td>DETA [99]</td>
<td><b>66.0</b></td>
</tr>
</tbody>
</table>

**Table 15 System-Level Comparison on Detection.** Comparing to the leading results on COCO [76] val2017. See Appendix B.3.5 for training recipe.## 6 Related Work

Learning vision-semantic representations has long been the leading approach for developing foundational models in perception. By aligning visual and textual representations, these models excel not only in vision tasks such as zero-shot image classification and image-text retrieval [51, 106, 117], open-vocabulary detection [63, 94, 95] and segmentation [22, 28], but also serve as the basis for multi-modal large language models (MLLMs) [3, 5, 78, 93, 101, 134].

**Contrastive Language-Image Pretraining.** The early works of Virtex [27], ICMLM [115], and ConVIRT [163] developed the techniques for learning through contrastive objectives between vision and language modalities. Subsequently, vision encoders such as CLIP [51, 106] and ALIGN [54] scaled these techniques to much larger datasets and model sizes, popularizing vision-language contrastive learning. A series of open-weight contrastive models have been developed to enhance the performance and robustness of CLIP [33, 71, 117, 129, 152, 160]. For instance, SigLIP [160] replaces the traditional softmax with a sigmoid function in contrastive learning, while FLIP [74] employs masking techniques to expedite the training process. We are among this effort and build a state-of-the-art open Perception Encoder (PE) (§2.1). Other objectives that have proven useful for building visual encoders include captioning loss, which learns to predict image descriptions using a language model decoder and transfers well to downstream multi-modal language modeling tasks [37, 137]. Many works are now attempting to combine two or more objectives to address different downstream tasks through pretraining with multiple objectives [37, 158] or training sequentially [19, 66].

**Efficient Training.** Various axes of efficient training of clip models have been explored. BASIC [102] and LAION [117] explored scaling the batch size up to 160K, and shows the benefits of large batch sizes during training. EVA-CLIP [130] uses LAMB optimizer [156] for large batch training of clip models. Rotary positional embedding (RoPE) [127] has been successfully adopted in large language models. In vision transformers [2, 48] adopted 2D rotary positional embeddings. For data engine, a series of works focus on large-scale sourcing and filtering through efficient data curation [33, 39, 117, 152] and explore recaptioning training images using MLLMs or VLMs [32, 64, 96, 151]. We extend these concepts to build a video data engine and scale our model to function as one strong model for both image and video (§2.2).

**Best Embedding Layer Inside the Network.** Typically, most vision encoders rely on the last layer to extract features for the task it is trained on. However, when trained on proxy or self-supervised tasks, the last layer is often not the ideal candidate for other tasks [8, 15, 16, 30, 85, 107, 121, 128, 142, 159, 166]. For example, when using image colorization as pretraining objective, [162, 166] showed that the middle layers were better at image classification compared to last layers. Subsequently, in iGPT [15], when trained for next token prediction, intermediate layers performed better at image classification. AIMv1 [30] also showed similar behavior for image based next token prediction with patch normalized MSE loss. Toto [107] showed this can be extended for next token prediction in videos, and intermediate layers are best for image classification, video classification, tracking and robotics. REPA [159] showed this behavior for image generation models, where the intermediate layers of SiT [85] has better linear probing accuracy compared to earlier or later layers. In CLIP models, CLIPer [128] identified that early layers in CLIP possess good spatial understanding. In contrast to these lines of work, in this paper, we first show this behavior is not limited to one class of encoders. Specifically, we show this behavior exists in a spatially self-supervised model [98], generative captioning model [37], and also in our own PE. Then we study this behavior for PE encoder in depth, and show it is possible for CLIP training to produce rich spatial and semantic features in intermediate layers (§3).

**Alignment Tuning.** We explore alignment tuning for language (§4) and for spatial understanding (§5). For language alignment, we focus on adapting to multimodal large language models (MLLMs); for spatial alignment, we employ self-distillation of the models own features combined with a teacher for locality. In MLLM literature, *midtraining*—i.e., a middle stage of training used to exploit large-scale multimodal data—has been actively studied. LLaVA-OneVision [66], InternVL series [18, 19], QwenVL series [3, 144], and several other leading MLLMs [82, 132] adopt this paradigm. Our PE<sub>lang</sub> can be seen as a variant of midtraining, but with one critical difference in principle: our goal is *not* to build the best MLLM, but to make the vision encoder the most *general*. Throughout §4, we benchmark our PE<sub>lang</sub> across different language models, input resolution, on various tasks for image and video to show this generality. For spatial tasks, we utilize the hidden embeddingsin the intermediate layers. Recently, several works showed the effectiveness of distilling teacher model via representation alignment with cosine similarity. REPA [159] distilled an early layer features of DINO for image diffusion models, RADIO [110] used multi-teacher distillation (DINO, CLIP and SAM). The key idea is to borrow semantic understanding (*e.g.*, CLIP) and spatial understanding (*e.g.*, SAM, DINO) of a pretrained vision encoders. In our PE<sub>spatial</sub>, we exploit the intermediate features of PE<sub>core</sub> for semantics, and a novel way to use SAM for spatial understanding.

## 7 Conclusion

We have presented Perception Encoders (PE), a family of best-in-class foundation models comprising PE<sub>core</sub>, PE<sub>lang</sub>, and PE<sub>spatial</sub>. We have shown that PE<sub>core</sub> can outperform models trained with WebLI and JFT-3B, which were previously the undisputed leaders in zero-shot image recognition, while also excelling in zero-shot video recognition. We have demonstrated that PE<sub>lang</sub> can be used to build a multimodal language model [21] that is at the forefront of the field in terms of performance. We have established that PE<sub>spatial</sub> can match the long-standing state-of-the-art in object detection with a significantly simpler decoder. Throughout all of this, one conclusion is abundantly clear: Perception Encoder unlocks the potential to scale simple contrastive vision-language pretraining to address a wide range of downstream vision tasks.

**Additional Contributors and Acknowledgments.** We would like to thank Abhimanyu Dubey, Adel Ahmadyan, Andrew Westbury, Arkabandhu Chowdhury, Azita Shokrpour, Babak Damavandi, Chay Ryali, Cyprien de Lichy, Didac Suris Coll-Vinent, Dong Wang, Filip Radenovic, George Orlin, Han Zou, Harry Tran, Jitendra Malik, Joelle Pineau, Joseph Greer, Kavya Srinet, Kirmani Ahmed, Laura Gustafson, Lu Zhang, Muhammad Maaz, Natalia Neverova, Nicolas Carion, Oleksandr Maksymets, Ramya Raghavendra, Romy Luo, Ronghang Hu, Sam Doud, Sasha Mitts, Sean Bell, Shane Moon, Shuming Hu, Soerian Lieve, Stephane Kasriel, Valentin Gabeur, Vanessa Stark, Vignesh Ramanathan, Vivian Lee, Xuan Hu, Yang Li, and Ziyang Wang for their contributions and support for the project. And we thank you, the reader, for reading this far.# A Video Data Engine

## A.1 Video Caption

### LLM Summarization prompt

LLM Summarization prompt 72 tokens

Create a concise caption of a video using the provided metadata, video caption, and frame captions.

TASK: Extract key information from the captions and combine it into an alt text format using single phrase or set of phrases that includes all relevant details.

Steps to Follow:

1. 1. Review the metadata (title and description) for general context, you can rely it for entity names but do not rely on it as the primary source of information for your caption.
2. 2. Blend title / description with video caption and frame captions for the main storyline
3. 3. Extract the most relevant and concise information.
4. 4. Combine extracted information into a alt text format using short phrase or set of phrases with approximately 120 tokens, considering special characters like comma as part of the token count.
5. 5. Prioritize including all key information over sentence structure or grammar.
6. 6. Minimize the use of special characters and focus of key information.

What to Avoid:

- - Avoid adding or inferring information not present in the original metadata and captions.
- - Avoid using complex sentence structures or prioritizing sentence flow.

Create a concise caption of the video based on the metadata, video caption, and frame captions.

## A.2 PE Video Dataset Details

PE Video is a dataset that we collected and curated from a licensed data source. The videos are high-resolution and high-quality with a focus on motion. The total number of videos is 1M. Among these, 120K videos have human-refined video captions, and we selected 15K from the 120K videos as a benchmark.

### A.2.1 Video Data Filtering Pipeline

The goal of video data filtering is to identify videos that contain motions such as object motion, camera motion, interaction between objects, human actions, sequences of actions, and manipulation of objects, while rejecting videos with static scenes, like landscapes, or those that are artificial or highly edited.

To achieve this, we created a video filtering pipeline consisting of the following steps:

**Step 1:** Compute motion features. For each video, we compute a list of features from video frames, including frames per second (fps), number of frames, number of I-frames, motion vector magnitude, and motion vector variance, using off-the-shelf tools like OpenCV [10].

**Step 2:** Extract video frame features. For each video, we uniformly sample three frames and encode them using a DINOv2 model [98] and a SigLIP model [160].

**Step 3:** LLM Features. For each video, we also run a multimodal large language model (LLM) like Llava-Onevision QwenLM 2 0.5B [66] to extract MLLM features. We composed a list of 26 questions and performed MLLM inference on the videos. The questions can be found here in §A.2.2.

**Step 4:** Video Quality Scoring. We combine all the features collected so far and use a random forest model to predict a score between 0 and 5. To train the model, we manually annotated approximately 1,000 videos with scores between 0 and 5. A low score indicates that the video is almost static and can be nearly summarized by a single frame, while a high score indicates that there are multiple temporal events in the video, requiring several frames to accurately caption it. We use these annotated videos as training data to fit a random forest model for video quality score prediction.

**Step 5:** We apply k-means clustering to the videos and rank them within each cluster. By selecting the top-ranked videos from each cluster, we effectively reduce the number of duplicated videos in the final dataset.## A.2.2 LLM Feature Extraction

### LLM Feature extraction question list

Is the camera capturing the scene static? Reply yes or no.  
Is the camera capturing the scene moving? Reply yes or no.  
Is the video capturing a landscape? Reply yes or no.  
Is the video capturing a static scene? Reply yes or no.  
Is the scene captured from a distance? Reply yes or no.  
Is the video captured with a drone? Reply yes or no.  
Is the video computer-generated? Reply yes or no.  
Is the video content abstract? Reply yes or no.  
Is there something moving through the scene? Reply yes or no.  
Is there someone doing something in the video? Reply yes or no.  
Are there several things moving in the video? Reply yes or no.  
Is there an object that is being manipulated? Reply yes or no.  
Are there animals in the video? Reply yes or no.  
Is the scene mostly static? Reply yes or no.  
Are things occluding each other in this video? Reply yes or no.  
Is there something obstructing the view apart from the watermark? Reply yes or no.  
Is there a large number of things in the video? Reply yes or no.  
Are there more than 5 different objects in the video? Reply yes or no.  
Is it hard to keep track of some entities because they are moving so much? Reply yes or no.  
Is someone looking at a phone, a tablet or a computer screen? Reply yes or no.  
Are they looking at a phone, a tablet or a computer screen during the whole video? Reply yes or no.  
Are there several moving persons in this video? Reply yes or no.  
Are there several moving animals in this video? Reply yes or no.  
Are there several objects in this video? Reply yes or no.  
Are there several similar-looking objects in the video? Reply yes or no.  
Do they look similar? Reply yes or no.

We use LLaVA-OneVision [78] model to extract LLM features from the videos. For each video, we prompt with 26 different questions to extract features ranging from, “is the video a landscape video?” to, “are there any moving objects in the video?” The features are then used by a random forest model to determine the video quality score.

## A.2.3 PVD Benchmark Distribution

<table border="1"><thead><tr><th>Category</th><th>Number of videos</th><th>Avg. Caption Length</th></tr></thead><tbody><tr><td><i>Hand Actions</i></td><td>2143</td><td>54.2</td></tr><tr><td><i>Object Interactions</i></td><td>1864</td><td>42.6</td></tr><tr><td><i>Food Preparation</i></td><td>1691</td><td>56.8</td></tr><tr><td><i>Work Activities</i></td><td>1689</td><td>47.8</td></tr><tr><td><i>Outdoor Scenes</i></td><td>1558</td><td>50.7</td></tr><tr><td><i>Animals</i></td><td>1423</td><td>50.9</td></tr><tr><td><i>Water Scenes</i></td><td>1337</td><td>44.6</td></tr><tr><td><i>Object Handling</i></td><td>1307</td><td>51.6</td></tr><tr><td><i>Close-up Shots</i></td><td>1122</td><td>45.1</td></tr><tr><td><i>Nature Scenes</i></td><td>866</td><td>38.4</td></tr></tbody></table>

**Table 16** PVD Benchmark Statistics. We created a dataset of 15K videos together with human-verified captions. The videos are motion-centered, covering both first-person and third-person views with a wide coverage of scenes.**Category: Hand Actions**

**Caption:** The video captures a closeup shot of person typing on a keyboard. The camera moves from the left side of the keyboard to the right, an animation of the revolving globe and some numbers can be seen in the frame and the video ends.

**Category: Object Interactions**

**Caption:** The video shows a black and white spiral that is spinning. The spiral is made up of alternating black and white stripes that are evenly spaced and symmetrical.

**Category: Food Preparation**

**Caption:** The video shows a person cutting an green color item into small pieces. They are using a knife to slice the pickle into thin pieces, and then chopping those pieces into smaller cubes. The person is working on a wooden cutting board, and the Hands are visible from the left side of the frame with pink nail paint on their nails.

**Category: Work Activities**

**Caption:** The video shows a person using a shovel to clean the ashes from a fireplace. They are scooping up the ashes and removing them from the fireplace.

**Category: Outdoor Scenes**

**Caption:** The video shows a tall, pointed structure in the middle of a field, and the structure is surrounded by trees and other vegetation. The field is divided into sections, with some areas covered in green grass and others covered in white material. The video shows the structure and the field from a distance, with the camera moving around it.

**Category: Animals**

**Caption:** The video shows a white and gray adult cat and two kittens. The adult cat is grooming the kitten closest to it with its tongue, and the kitten is looking around. A hand reaches out from the frame's upper left to pet the two kittens.

**Category: Water Scenes**

**Caption:** The video shows a large school of fish swimming in a water body towards the right frame. The camera too pans a little to the right.

**Category: Object Handling**

**Caption:** The video shows a person putting a bowl of something into an oven. The person then closes the oven door. The background is blurry.

**Category: Close-up Shots**

**Caption:** The video shows a white counter with two brown buckets and a yellow bucket. Then a person's right hand wearing a green glove enters the frame from top right side and place a yellow flower near to yellow watering can. The person then places the flower, in front of the buckets and exits the frame. In the background is a brown wall, and the camera is static throughout the clip.

**Category: Nature Scenes**

**Caption:** The video shows a pile of branches and leaves on fire in a field. The fire is burning brightly, with flames licking at the edges of the pile. The smoke from the fire rises into the air, billowing up into the sky.

**Figure 18 More PE Video Dataset Examples.** For each of the ten categories, we randomly pick one video and show its video caption. The captions were generated by our video data pipeline and then refined by human annotators.## B Implementation Details

### B.1 PE Core

We provide additional implementation details for building PE<sub>core</sub>. Our implementation is based on OpenCLIP<sup>5</sup>.

#### B.1.1 Architecture and Training Setups

**Model Architecture.** Following CLIP, PE<sub>core</sub> comprises a Transformer-based [141] vision and a text encoder. We employ customized Transformer configurations as detailed in Tab. 17. For pooling, we use an attention pooling block in the style of SigLIP [160] with 8 heads from the last-layer feature to construct image and video embeddings. Regarding positional embedding, we use 2D RoPE [127] for relative positional embeddings and 2D learnable absolute positional embeddings (abs) the same size as the model’s input resolution. We interpolate positional embeddings to enable support for various resolutions beyond the default. The text context length is 72 for G-scale and 32 for B and L-scale models. Originally a bug, we find it optimal to *not disable the class token* when using attention pooling for smaller models. Thus, the B and L models use a class token, then the attention pooling layer probes all features at once (class token included). Finally, we use an input mean and standard deviation of (0.5, 0.5, 0.5) for simplicity.

<table border="1">
<thead>
<tr>
<th>Scale</th>
<th>Tower</th>
<th>Params</th>
<th>Width</th>
<th>Depth</th>
<th>MLP</th>
<th>Heads</th>
<th>CLIP Dim</th>
<th>Pooling</th>
<th>Positional Embedding</th>
<th>Resolution &amp; Context Len</th>
<th>Patch Size</th>
<th>Class Token Register</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">B</td>
<td>Vision</td>
<td>0.09B</td>
<td>768</td>
<td>12</td>
<td>3072</td>
<td>12</td>
<td>1024</td>
<td>Attn Pool</td>
<td>RoPE+Abs</td>
<td>224</td>
<td>16</td>
<td>✓</td>
</tr>
<tr>
<td>Text</td>
<td>0.31B</td>
<td>1024</td>
<td>24</td>
<td>4096</td>
<td>16</td>
<td>-</td>
<td>EOS Token</td>
<td>Abs</td>
<td>32</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td rowspan="2">L</td>
<td>Vision</td>
<td>0.32B</td>
<td>1024</td>
<td>24</td>
<td>4096</td>
<td>16</td>
<td>1024</td>
<td>Attn Pool</td>
<td>RoPE+Abs</td>
<td>336</td>
<td>14</td>
<td>✓</td>
</tr>
<tr>
<td>Text</td>
<td>0.31B</td>
<td>1024</td>
<td>24</td>
<td>4096</td>
<td>16</td>
<td>-</td>
<td>EOS Token</td>
<td>Abs</td>
<td>32</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td rowspan="2">G</td>
<td>Vision</td>
<td>1.88B</td>
<td>1536</td>
<td>50</td>
<td>8960</td>
<td>16</td>
<td>1280</td>
<td>Attn Pool</td>
<td>RoPE+Abs</td>
<td>448</td>
<td>14</td>
<td>✗</td>
</tr>
<tr>
<td>Text</td>
<td>0.47B</td>
<td>1280</td>
<td>24</td>
<td>5120</td>
<td>20</td>
<td>-</td>
<td>EOS Token</td>
<td>Abs</td>
<td>72</td>
<td>-</td>
<td>-</td>
</tr>
</tbody>
</table>

**Table 17** PE Model Configurations with full details.

**PE Core Training.** As discussed in §2.4, the training of PE<sub>core</sub> involves three stages: 1) image pretraining; 2) image and video finetuning; and 3) an additional model distillation for smaller models. These three stages work together to develop a robust and effective PE<sub>core</sub> model.

We first provide training recipes for 1) image pretraining in Tab. 18 and 2) video finetuning in Tab. 19.

<table border="1">
<thead>
<tr>
<th>config</th>
<th>values</th>
<th>config</th>
<th>values</th>
<th>config</th>
<th>values</th>
</tr>
</thead>
<tbody>
<tr>
<td>optimizer</td>
<td>LAMB</td>
<td>optimizer</td>
<td>LAMB</td>
<td>optimizer</td>
<td>LAMB</td>
</tr>
<tr>
<td><math>\beta_1, \beta_2</math></td>
<td>(0.9, 0.95)</td>
<td><math>\beta_1, \beta_2</math></td>
<td>(0.9, 0.95)</td>
<td><math>\beta_1, \beta_2</math></td>
<td>(0.9, 0.95)</td>
</tr>
<tr>
<td>weight decay</td>
<td>0.05</td>
<td>weight decay</td>
<td>0.05</td>
<td>weight decay</td>
<td>0.05</td>
</tr>
<tr>
<td>learning rate</td>
<td>2e-3</td>
<td>learning rate</td>
<td>1e-6</td>
<td>learning rate</td>
<td>1e-6</td>
</tr>
<tr>
<td>batch size</td>
<td>131,072</td>
<td>batch size</td>
<td>4096</td>
<td>batch size</td>
<td>16384</td>
</tr>
<tr>
<td>warm-up steps</td>
<td>2K</td>
<td>warm-up steps</td>
<td>2K</td>
<td>warm-up steps</td>
<td>2K</td>
</tr>
<tr>
<td>training steps</td>
<td>443K (B, L) / 656K (G)</td>
<td>training steps</td>
<td>5.4K</td>
<td>training steps</td>
<td>269K</td>
</tr>
<tr>
<td>data quantity</td>
<td>5.4B</td>
<td>data quantity</td>
<td>22M</td>
<td>data quantity</td>
<td>5.4B</td>
</tr>
<tr>
<td>samples seen</td>
<td>58B (B, L) / 86B (G)</td>
<td>samples seen</td>
<td>22M</td>
<td>samples seen</td>
<td>4.4B</td>
</tr>
<tr>
<td>max logit scale</td>
<td>100</td>
<td>max logit scale</td>
<td>100</td>
<td>max logit scale</td>
<td>100</td>
</tr>
<tr>
<td>mask reg ratio</td>
<td>0.4</td>
<td>number of frames</td>
<td>8</td>
<td>teacher logit scale</td>
<td>200 (§C.3)</td>
</tr>
<tr>
<td>mask reg batch</td>
<td>8192</td>
<td></td>
<td></td>
<td>data aug</td>
<td>None</td>
</tr>
<tr>
<td>progressive res</td>
<td>112-160-224 (B)<br/>98-154-224-336 (L)<br/>98-154-224-336-448 (G)</td>
<td>data aug</td>
<td>aspect jitter ar(0.75, 1.33)<br/>rand crop s(0.08, 1)<br/>color jitter j(0.32, 0, 0.32, 0)<br/>hflip p(0.5)</td>
<td></td>
<td></td>
</tr>
<tr>
<td>data aug</td>
<td>aspect jitter ar(0.75, 1.33)<br/>rand crop s(0.08, 1)<br/>color jitter j(0.32, 0, 0.32, 0)<br/>hflip p(0.5)</td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
</tbody>
</table>

**Table 18** Image Pretraining.

**Table 19** Video Finetuning.

**Table 20** Distillation.

After training the largest G-scale model, we train the smaller models with image pretraining, then distill with image distillation in Tab. 20, then finally apply video finetuning at the end.

<sup>5</sup>[https://github.com/mlfoundations/open\\_clip](https://github.com/mlfoundations/open_clip)## B.1.2 Zero-Shot Classification and Retrieval

**Zero-Shot Evaluation on Images and Videos.** We use CLIPBench<sup>6</sup> for zero-shot classification and retrieval benchmarking. The benchmark datasets and splits are obtained from the original dataset websites or HuggingFace. We extend the CLIPBench zero-shot evaluation to include video datasets such as MSR-VTT and Kinetics, and will release our model checkpoints, evaluation code, and scripts for reproducibility.

**Prompt Design.** For zero-shot image-text and video-text retrieval, we rely solely on the original captions without any additional prompts. In contrast, for zero-shot classification, we utilize task-specific prompts graciously provided by the InternVL [19] authors. All additional prompts will be released.

For example, we employ specific prompts for zero-shot image classification on various ImageNet benchmarks (e.g., ImageNet val, ImageNet v2) and video classification on Kinetics datasets (e.g., K400, K600, K700).

### Zero-Shot Image Classification Prompts – ImageNet

a bad photo of a {c}. a photo of many {c}. a sculpture of a {c}. a photo of the hard to see {c}. a low resolution photo of the {c}. a rendering of a {c}. graffiti of a {c}. a bad photo of the {c}. a cropped photo of the {c}. a tattoo of a {c}. the embroidered {c}. a photo of a hard to see {c}. a bright photo of a {c}. a photo of a clean {c}. a photo of a dirty {c}. a dark photo of the {c}. a drawing of a {c}. a photo of my {c}. the plastic {c}. a photo of the cool {c}. a close-up photo of a {c}. a black and white photo of the {c}. a painting of the {c}. a painting of a {c}. a pixelated photo of the {c}. a sculpture of the {c}. a bright photo of the {c}. a cropped photo of a {c}. a plastic {c}. a photo of the dirty {c}. a jpeg corrupted photo of a {c}. a blurry photo of the {c}. a photo of the {c}. a good photo of the {c}. a rendering of the {c}. a {c} in a video game. a photo of one {c}. a doodle of a {c}. a close-up photo of the {c}. a photo of a {c}. the origami {c}. the {c} in a video game. a sketch of a {c}. a doodle of the {c}. a origami {c}. a low resolution photo of a {c}. the toy {c}. a rendition of the {c}. a photo of the clean {c}. a photo of a large {c}. a rendition of a {c}. a photo of a nice {c}. a photo of a weird {c}. a blurry photo of a {c}. a cartoon {c}. art of a {c}. a sketch of the {c}. a embroidered {c}. a pixelated photo of a {c}. itap of the {c}. a jpeg corrupted photo of the {c}. a good photo of a {c}. a plushie {c}. a photo of the nice {c}. a photo of the small {c}. a photo of the weird {c}. the cartoon {c}. art of the {c}. a drawing of the {c}. a photo of the large {c}. a black and white photo of a {c}. the plushie {c}. a dark photo of a {c}. itap of a {c}. graffiti of the {c}. a toy {c}. itap of my {c}. a photo of a cool {c}. a photo of a small {c}. a tattoo of the {c}.

### Zero-Shot Video Classification Prompts – Kinetics

a photo of {c}. a photo of a person {c}. a photo of a person using {c}. a photo of a person doing {c}. a photo of a person during {c}. a photo of a person performing {c}. a photo of a person practicing {c}. a video of {c}. a video of a person {c}. a video of a person using {c}. a video of a person doing {c}. a video of a person during {c}. a video of a person performing {c}. a video of a person practicing {c}. a example of {c}. a example of a person {c}. a example of a person using {c}. a example of a person doing {c}. a example of a person during {c}. a example of a person performing {c}. a example of a person practicing {c}. a demonstration of {c}. a demonstration of a person {c}. a demonstration of a person using {c}. a demonstration of a person doing {c}. a demonstration of a person during {c}. a demonstration of a person performing {c}. a demonstration of a person practicing {c}.

**Evaluation Method.** Several works use different input transformations for different datasets when evaluating zero-shot performance (e.g., [33, 130, 138, 160]). To be as fair as possible, we follow [130] in evaluating with two transformations—center crop and non aspect ratio preserving resize (“squash”)—and report the max between the two for all models and all datasets we evaluate. Additionally, ObjectNet has a red border around every image to facilitate deduplication, which we remove for evaluation. Finally, we follow [19] in using *retrieval reweighting* (DSL), applying the softmax score distribution to the similarities used for retrieval:

$$\text{scores} = \text{scores} * \text{softmax}(\text{scores}, \text{dim}=0) \quad (1)$$

This slightly improves retrieval for most models, so we do it for all models we evaluate for fairness. Notably, we were able to reproduce the reported numbers for most papers with these techniques, but for cases where we could not, we default to the reported number.

## B.2 PE: Language Alignment

We provide details of the MLLM experimental setup in § 4. We describe *data*, *model*, and *training* separately.

**Data.** Our MLLM training contains *warmup* data and *supervised finetuning* (*SFT*) data. Our warmup data is a 1M subset image-text pairs of our PE<sub>core</sub> pretraining dataset. For SFT data, we use a diverse data

<sup>6</sup>[https://github.com/LAION-AI/CLIP\\_benchmark](https://github.com/LAION-AI/CLIP_benchmark)mix consisting of 2.6M unique samples. This dataset is composed of 1.7M <sup>7</sup> visual QAs samples from the Cauldron [65], 0.5M grounded QA pairs from Visual Genome [60], Flickr-Entities [103] and Densely Captioned Images [139], 0.1M image-captioning pairs from COCO [76] and 0.3M text-only samples. This comprehensive data mix allows us to thoroughly assess our model’s capabilities in various MLLM tasks.

**Model.** As described in § 4.1, we use a simple vision-language model architecture where a vision encoder and a pretrained decoder-only LLM are connected by a vision projector. For all tables, we use either Llama3.1-instruct 8B or QwenLM 2.5-instruct 7B as a language model, and 2-layer MLP as a vision projector. For fair comparison, we use the native resolution for image input. During inference, we evaluate the models on video tasks in *zeroshot* manner: We concatenate all video frames into a sequence and feed to language model, without seeing video samples during SFT. For all video tasks, we use 8 frames with the same native resolution of height and width. For  $PE_{\text{core}}$  and  $PE_{\text{lang}}$ , this makes  $448 \times 448 \times 8$  input and  $32 \times 32 \times 8$  vision tokens.

**Training.** MLLM training consists of *warmup* and *supervised finetuning (SFT)* stages. In both stages, we freeze vision encoder and train vision projector and LLM. During warmup stage, we use a global batch size of 128 with a learning rate of  $1 \times 10^{-4}$ . We gradually increase the learning rate from  $1 \times 10^{-6}$  to  $1 \times 10^{-4}$  over 120 steps, and follow a cosine learning rate decay schedule to train a total of 8,000 steps. During SFT stage, we use a global batch size 256 with a learning rate of  $1 \times 10^{-5}$ . Similar to the warmup, we gradually increase the learning rate from  $1 \times 10^{-7}$  to  $1 \times 10^{-5}$  over 300 steps, and follow a cosine learning rate decay schedule to train a total of 12.5K steps. We truncate text-sequences longer than 2,048 tokens on top the visual tokens. This makes the maximum sequence length to be  $(\text{num. vision tokens}) + 2,048$ . With  $448 \times 448$  input resolution and patch size of 14, we set the maximum sequence length to  $1,024 + 2,048 = 3,072$ . To represent bounding boxes on output side for image grounding tasks, we simply use text tokens to represent each bounding box: each coordinate is normalized between 000 and 999, in “[x, y, x, y]” box format for top-left and bottom-right corners (*e.g.*, [012, 122, 633, 782]).

For all baselines, we search for the **best** intermediate layer features to adapt to LLM. We search over  $\{-1, -2, -4, -6, -8, -10, -12, -14, -16, -18, -20, -40\}$  layers (counting from last) and report the best result in average over OCR/Chart/Document Q&A, Visual Q&A, Image Captioning and Video Understanding.

## B.3 PE: Spatial Alignment

### B.3.1 Training Details

**Loss Functions.** For self-aligning to frozen  $PE_{\text{coreG}}$  layer 41 features ( $L_{\text{core}}$ ), we minimize cosine similarity:

$$L_{\text{core}} = \frac{1}{n_{\text{tok}}} \sum \left( \frac{(S_{50})(T_{41})^T}{\|S_{50}\| \cdot \|T_{41}\|} \right) \quad (2)$$

where  $S_{50}$  denotes the last layer features of the student,  $T_{41}$  denotes frozen layer 41 features from  $PE_{\text{coreG}}$ , and  $n_{\text{tok}}$  represents the number of tokens. Note that we chose 41 fairly arbitrarily (it is layer 40 when written with indexing from 0). Judging by Fig. 8, any layer around 40 should work (and 39 may be slightly better).

For the encouraging locality loss ( $L_{\text{loc}}$ ), we compute the pairwise cosine similarity between a model’s own tokens and itself. This forms a “spatial correspondence map” for what tokens should be considered similar. We then compute the same for the student, and minimize the difference between the two with MSE loss:

$$L_{\text{loc}} = \frac{1}{n_{\text{tok}}^2} \sum \left( \frac{(S_{50})(S_{50})^T}{\|S_{50}\|^2} - \frac{(T_{\text{SAM}})(T_{\text{SAM}})^T}{\|T_{\text{SAM}}\|^2} \right)^2 \quad (3)$$

where  $T_{\text{SAM}}$  denotes the “SAM Mask Logits” constructed in §5.2. We also find using a temperature ( $t$ ) on the SAM teacher’s pairwise cosine similarity term ( $x$ ) useful:  $e^{t(x-1)}$ . The full loss is  $L_{\text{spatial}} = L_{\text{core}} + L_{\text{loc}}$ .

**Hyperparameters.** In Tab. 21 we show the training hyperparameters for spatial alignment, finetuned on top of the initial  $PE_{\text{coreG}}$  checkpoint. Then in Tab. 22 and Tab. 23, we show the settings for the two teachers and losses. Note that when running the teachers, we run them on the exact same image as the student (same data

<sup>7</sup>We excluded multi-images samples.aug and all). Additionally, because the SAM 2.1 teacher operates at a resolution of 1024, we upsample the image, generate the mask logits, and then downsample the result. Both teachers are frozen.

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
<td>LAMB</td>
</tr>
<tr>
<td><math>\beta_1, \beta_2</math></td>
<td>(0.9, 0.95)</td>
</tr>
<tr>
<td>weight decay</td>
<td>0.05</td>
</tr>
<tr>
<td>learning rate</td>
<td>5e-4</td>
</tr>
<tr>
<td>batch size</td>
<td>12,288</td>
</tr>
<tr>
<td>warm-up steps</td>
<td>0</td>
</tr>
<tr>
<td>training steps</td>
<td>24K</td>
</tr>
<tr>
<td>data quantity</td>
<td>5.4B (PE<sub>core</sub> PT Data)</td>
</tr>
<tr>
<td>samples seen</td>
<td>300M</td>
</tr>
<tr>
<td>resolution</td>
<td>448</td>
</tr>
<tr>
<td>mask ratio</td>
<td>0.75</td>
</tr>
<tr>
<td>mask size</td>
<td>2 × 2 tokens</td>
</tr>
<tr>
<td>droppath</td>
<td>0.4</td>
</tr>
<tr>
<td>layerscale</td>
<td>0.1</td>
</tr>
<tr>
<td>data aug</td>
<td>aspect jitter ar(0.75, 1.33)<br/>color jitter j(0.32, 0, 0.32, 0)<br/>hflip p(0.5)</td>
</tr>
</tbody>
</table>

**Table 21 Spatial Alignment.**

<table border="1">
<thead>
<tr>
<th>config</th>
<th>values</th>
</tr>
</thead>
<tbody>
<tr>
<td>model</td>
<td>SAM 2.1-L</td>
</tr>
<tr>
<td>layer</td>
<td>mask logits</td>
</tr>
<tr>
<td>resolution</td>
<td>1024 (interp → 448)</td>
</tr>
<tr>
<td>loss</td>
<td>Eq. 3</td>
</tr>
<tr>
<td>loss weight</td>
<td>1</td>
</tr>
<tr>
<td>temperature</td>
<td>20</td>
</tr>
<tr>
<td>sample points</td>
<td>32 × 32 (1024)</td>
</tr>
<tr>
<td>pred iou threshold</td>
<td>0</td>
</tr>
<tr>
<td>stability score threshold</td>
<td>0</td>
</tr>
<tr>
<td>mask threshold</td>
<td>0</td>
</tr>
</tbody>
</table>

**Table 22 SAM 2.1 Teacher.**

<table border="1">
<thead>
<tr>
<th>config</th>
<th>values</th>
</tr>
</thead>
<tbody>
<tr>
<td>model</td>
<td>PE<sub>core</sub>G</td>
</tr>
<tr>
<td>layer</td>
<td>41</td>
</tr>
<tr>
<td>resolution</td>
<td>448</td>
</tr>
<tr>
<td>loss</td>
<td>Eq. 2</td>
</tr>
<tr>
<td>loss weight</td>
<td>1</td>
</tr>
</tbody>
</table>

**Table 23 PE<sub>core</sub>G Teacher.**

### B.3.2 Visualization Method

To visualize the features in Fig. 17 and Fig. 20, our goal is to map a 1536-dimensional space down to 3 dimensions to view how the model encodes each token in relation to each other. One naive approach would be to apply PCA with 3 dimensions across all token in the image. However, we find this alone can be misleading.

Specifically, if the model has rich semantics, it should be the case that most of those 1536 features have some useful information in them. Some of that information could be spatially contiguous, some of it not. We want PCA to only select the *spatially contiguous* information, since we are trying to evaluate the spatial quality of the features. However, naively applying PCA will not necessarily do that, especially for models with information aggregated in “global tokens” (§5.1). Despite these tokens carrying important information, they are not spatially contiguous. Thus, if PCA dedicates a large portion of its 3 dimensions to global tokens, the features will *look* like their spatial quality is bad, despite the features containing good spatial information.

So, how do we select for only the *spatially contiguous* information to visualize? The answer is simple: by definition, the spatially contiguous information will be... spatially contiguous. To keep the spatially contiguous information while lowering the impact of the global tokens, we can simply apply a low pass filter to the features (specifically, a gaussian blur with kernel size 3 and a  $\sigma$  of 1). To retain the detail of the original features, we can average the two together. Thus, to visualize features, we use the 3D PCA of the of the following.  $x$  denotes the model’s output features, and  $g(x)$  denotes gaussian blur.

$$0.5x + 0.5g(x, k = 3, \sigma = 1) \quad (4)$$

We show the impact of this in Fig. 19. Blurring the features make them appear more detailed! In reality, that information was always there, just PCA did not show it. Thus, great care must be taken when visualizing high dimensional feature spaces. If they were easy to map to 3 dimensions—you wouldn’t need 1536 of them!

**Figure 19 Feature Visualization Ablation.** With raw features (top row), PCA misses spatially contiguous parts of the feature space and instead focuses on global tokens (which carry information but are not spatially coherent). By applying a simple low pass filter (bottom row), we can reveal spatial information that PCA originally missed (see column 2: with raw features, the background looks like a mess, with the low pass filter the tiles become visible).Then, to map the PCA dimensions to RBG pixel values, we map each PCA component to a corresponding channel in LCh color space, then convert those LCh colors to RGB to get the final image. Note that we use LCh instead of RGB directly for aesthetic reasons, and also because LCh is a cylindrical color space—where smooth changes to the values look like smooth changes in colors to humans—and thus is easier to discern.

### B.3.3 Frozen Feature Dense Prediction

We discuss the detailed settings of the results for dense prediction with frozen features in Tab. 13. Each model is evaluated with its native resolution up to 448 or 448 (whichever is optimal).

**Zero-Shot Tracking.** We evaluate our pretrained models on label propagation task using the protocols in [52, 107] on DAVIS dataset [104]. This evaluation does not require any finetuning or probing, therefore preserves the spatial features in the model. Following Toto [107], we use the features from the last  $n = 7$  frames to find the nearest neighbor patch in the current frame, and then propagate the masks from the previous frames to the current frame. Note that this evaluation method does not require any training.

**Semantic Segmentation.** For semantic segmentation, we evaluate our pretrained models on ADE20K [167] semantic segmentation task. We use a linear layer and convolutional layer to map intermediate spatial features to segmentation masks following [98]. The models are evaluated and then features are resized to  $518 \times 518$ . We only use features from single layer. The probing layers are finetuned with AdamW [83] with a learning rate of 0.001.

**Depth Estimation.** For depth estimation on NYUv2 [123], we follow [75, 98]. We use a DPT-head [109] on top of our frozen pretrained model and use only single layer features. We scale the size of the DPT-head for each models based on the hidden size for each architecture. Because NYU is a small dataset and the models we evaluate are large, we observe the results for most models are noisy and prone to overfitting. Thus, for fair comparison we train *all models* for 20 epochs and for *all models* take the lowest validation loss over all epochs.

**Frozen Detection.** For the frozen feature detection results presented in §3, we evaluated using Mask R-CNN [43] as a probe. We used a resolution of 1024 for Fig. 8 and 768 for the remaining experiments in §3. Because the backbones were frozen, we did not add any global attention and instead simply tiled the input image with a window size of 32 for the 1024px experiments and 24 for the 768px experiments. All models were interpolated to patch 16. Finally, the backbones were frozen and only the FPN and R-CNN heads trained for 15 epochs on COCO with a stepwise decay LR without drop path.

### B.3.4 End-to-End Finetuning Detection and Segmentation

We provide a detailed discussion of settings of end-to-end finetuning on detection and segmentation presented in Tab. 14. The hyperparameters can be found in Tab. 24. We find that the default 100-epoch protocol in ViTDet [72, 149] causes overfitting problems in COCO experiments especially for billion-level parameter vision encoders, so we tune the training epochs, learning rate, drop path and learning rate decay accordingly.

The LVIS experiment setting is the same as COCO except all L-size models use learning rate of  $2e-4$  and all g-size and G-size models use 75 epochs.

<table border="1">
<thead>
<tr>
<th>config</th>
<th>values</th>
<th>model</th>
<th>lr</th>
<th>epochs</th>
<th>drop path</th>
<th>lr decay</th>
<th>layers</th>
<th>global window index</th>
<th>window size</th>
</tr>
</thead>
<tbody>
<tr>
<td>optimizer</td>
<td>AdamW</td>
<td>OpenAI CLIP-L</td>
<td>1e-4</td>
<td>100</td>
<td>0.4</td>
<td>0.8</td>
<td>24</td>
<td>(5, 11, 17, 23)</td>
<td>14</td>
</tr>
<tr>
<td>optimizer momentum</td>
<td>(0.9, 0.999)</td>
<td>MetaCLIP-L</td>
<td>1e-4</td>
<td>100</td>
<td>0.4</td>
<td>0.8</td>
<td>24</td>
<td>(5, 11, 17, 23)</td>
<td>14</td>
</tr>
<tr>
<td>weight decay</td>
<td>0.1</td>
<td>MetaCLIP-G</td>
<td>5e-5</td>
<td>75</td>
<td>0.5</td>
<td>0.9</td>
<td>48</td>
<td>(11, 23, 35, 47)</td>
<td>14</td>
</tr>
<tr>
<td>learning rate</td>
<td>→</td>
<td>SigLIP-so</td>
<td>1e-4</td>
<td>100</td>
<td>0.4</td>
<td>0.8</td>
<td>27</td>
<td>(2, 10, 18, 26)</td>
<td>14</td>
</tr>
<tr>
<td>learning rate schedule</td>
<td>Step-wise decay</td>
<td>EVA02-L</td>
<td>1e-4</td>
<td>100</td>
<td>0.4</td>
<td>0.8</td>
<td>24</td>
<td>(5, 11, 17, 23)</td>
<td>14</td>
</tr>
<tr>
<td>learning rate decay</td>
<td>→</td>
<td>MAE-L</td>
<td>1e-4</td>
<td>100</td>
<td>0.4</td>
<td>0.8</td>
<td>24</td>
<td>(5, 11, 17, 23)</td>
<td>14</td>
</tr>
<tr>
<td>batch size</td>
<td>64</td>
<td>SigLIP2-so</td>
<td>1e-4</td>
<td>100</td>
<td>0.4</td>
<td>0.8</td>
<td>27</td>
<td>(2, 10, 18, 26)</td>
<td>14</td>
</tr>
<tr>
<td>image size</td>
<td><math>1024 \times 1024</math></td>
<td>SigLIP2-g</td>
<td>5e-5</td>
<td>75</td>
<td>0.5</td>
<td>0.9</td>
<td>40</td>
<td>(9, 19, 29, 39)</td>
<td>14</td>
</tr>
<tr>
<td>augmentation</td>
<td>LSJ [0.1, 2.0]</td>
<td>DINOv2-L</td>
<td>1e-4</td>
<td>100</td>
<td>0.4</td>
<td>0.8</td>
<td>24</td>
<td>(5, 11, 17, 23)</td>
<td>32</td>
</tr>
<tr>
<td>epochs</td>
<td>→</td>
<td>DINOv2-g</td>
<td>5e-5</td>
<td>36</td>
<td>0.5</td>
<td>0.9</td>
<td>40</td>
<td>(9, 19, 29, 39)</td>
<td>32</td>
</tr>
<tr>
<td>drop path</td>
<td>→</td>
<td><b>PE<sub>core</sub>G</b></td>
<td>5e-5</td>
<td>75</td>
<td>0.5</td>
<td>0.9</td>
<td>50</td>
<td>(12, 24, 36, 49)</td>
<td>32</td>
</tr>
<tr>
<td>postional embedding</td>
<td>abswin [7]</td>
<td><b>PE<sub>spatial</sub>G</b></td>
<td>5e-5</td>
<td>36</td>
<td>0.5</td>
<td>0.9</td>
<td>50</td>
<td>(12, 24, 36, 49)</td>
<td>32</td>
</tr>
<tr>
<td>patch size</td>
<td>16</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>window size</td>
<td>→</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>global window index</td>
<td>→</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
</tbody>
</table>

Table 24 Settings for End-to-End Finetuning Detection and Segmentation.### B.3.5 System-Level Comparison on Detection

We describe our implementation for system-level comparison to the state-of-the-arts on COCO object detection in Tab 15. Our implementation is based on the DETA repository<sup>8</sup>. We replace the vision encoder with our PE<sub>spatial</sub> and maintain the same hyperparameters as in the end-to-end finetuning settings, while keeping the detector unchanged. The training process consists of three stages:

1. 1. **Initial Training:** Train on Objects365 for 12 epochs with an image resolution of  $1024 \times 1024$ , a total batch size of 256, and a learning rate of  $2e-4$ , which is divided by 10 at the 10th epoch.
2. 2. **Increasing Resolution:** Continue training on Objects365 for 6 epochs with a resolution of  $1536 \times 1536$ , a total batch size of 128, and a learning rate of  $5e-5$ , which is divided by 10 at the 5th epoch.
3. 3. **Finetuning:** Finetune on COCO dataset for 12 epochs with an image resolution of  $1728 \times 1728$ , a total batch size of 64, and a learning rate of  $5e-5$ , which is divided by 10 at the 8th epoch.
4. 4. **Further Increasing Resolution:** Further finetune on COCO dataset for 3 epochs with a resolution of  $1824 \times 1824$ , a total batch size of 64. To save GPU memory, we use SGD optimizer instead of Adam, with a learning rate of  $5e-3$ , which is divided by 10 at the 2th epoch.

We apply a series of test-time augmentation techniques to further improve the performance, detailed in Tab. 25.

<table border="1">
<thead>
<tr>
<th>Test-Time Aug</th>
<th>AP<sub>box</sub></th>
</tr>
</thead>
<tbody>
<tr>
<td>No TTA</td>
<td>65.2</td>
</tr>
<tr>
<td>+ More Queries</td>
<td>65.3</td>
</tr>
<tr>
<td>+ SoftNMS [6]</td>
<td>65.8</td>
</tr>
<tr>
<td>+ Flip Aug</td>
<td>65.8</td>
</tr>
<tr>
<td>+ Multiscale Aug</td>
<td><b>66.0</b></td>
</tr>
</tbody>
</table>

**Table 25** Test-Time Aug for system-level comparison on COCO in Tab. 15.

## C Additional Results

### C.1 PE<sub>core</sub>: Robust Image Pretraining

In Tab. 26, we present the raw data for the robustness metrics in Fig. 2. Across the board, each change improved almost all metrics (with the exception of progressive resolution slightly hurting the average and mask regularization slightly hurting ImageNet Adversarial). The fact that there were no tradeoffs to these changes, indicate that their improvements to the features are general. This could be why most of these changes improved performance for downstream tasks as well.

Note that in §2.1, we only discuss changes that we know to work. There are several changes that we have tried that do not work (i.e., do not improve performance or lower performance). For instance: average pooling instead of using a class token, increasing the text tower size, using hue or contrast jitter, and maintaining the same resolution throughout training but dropping tokens instead of progressive resolution (FLIP-style).

We also find increasing batch size and increasing training iterations for an L scale model to have equivalent effects. This is in contrast to the batch size scaling observed by [160], but it is possible that this difference is down to a hyperparameter issue.

<table border="1">
<thead>
<tr>
<th rowspan="2">Step</th>
<th colspan="7">Zero-Shot Classification</th>
</tr>
<tr>
<th>Avg Class.</th>
<th>ImageNet val [26]</th>
<th>ImageNet v2 [112]</th>
<th>ObjectNet IV Classes [4]</th>
<th>ImageNet Adversarial [47]</th>
<th>ImageNet Renditions [46]</th>
<th>ImageNet Sketch [143]</th>
</tr>
</thead>
<tbody>
<tr>
<td>1 Baseline</td>
<td>75.3</td>
<td>78.9</td>
<td>71.9</td>
<td>73.7</td>
<td>68.3</td>
<td>91.1</td>
<td>67.8</td>
</tr>
<tr>
<td>2 Progressive Resolution</td>
<td>75.1</td>
<td>78.9</td>
<td>71.8</td>
<td>72.4</td>
<td>69.9</td>
<td>90.5</td>
<td>67.0</td>
</tr>
<tr>
<td>3 High Batch Size</td>
<td>76.2</td>
<td>79.5</td>
<td>72.8</td>
<td>74.1</td>
<td>71.8</td>
<td>91.0</td>
<td>68.1</td>
</tr>
<tr>
<td>4 LAMB and High LR</td>
<td>76.9</td>
<td>79.9</td>
<td>73.3</td>
<td>74.3</td>
<td>73.5</td>
<td>91.5</td>
<td>68.6</td>
</tr>
<tr>
<td>5 High Resolution (336)</td>
<td>78.3</td>
<td>80.4</td>
<td>73.8</td>
<td>75.6</td>
<td>79.2</td>
<td>92.0</td>
<td>68.8</td>
</tr>
<tr>
<td>6 2D RoPE</td>
<td>79.2</td>
<td>80.7</td>
<td>74.1</td>
<td>77.4</td>
<td>80.9</td>
<td>92.7</td>
<td>69.4</td>
</tr>
<tr>
<td>7 Attention Pooling</td>
<td>80.1</td>
<td>81.0</td>
<td>74.8</td>
<td>78.4</td>
<td>82.9</td>
<td>93.4</td>
<td>69.9</td>
</tr>
<tr>
<td>8 Data Augmentation</td>
<td>80.8</td>
<td>81.1</td>
<td>75.2</td>
<td>80.8</td>
<td>83.1</td>
<td>93.5</td>
<td>71.2</td>
</tr>
<tr>
<td>9 Mask Regularization</td>
<td>80.9</td>
<td>81.3</td>
<td>75.3</td>
<td>80.9</td>
<td>82.8</td>
<td>93.8</td>
<td>71.2</td>
</tr>
</tbody>
</table>

**Table 26** Robust Image Pretraining Full Results. Raw results for the robustness metrics in Fig. 2. Almost every change improves every metric, but some metrics are improved more than others (e.g., ObjectNet and ImageNet-A).

<sup>8</sup><https://github.com/jozhang97/DETA>

