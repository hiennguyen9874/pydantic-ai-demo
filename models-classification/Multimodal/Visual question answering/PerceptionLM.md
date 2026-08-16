# PerceptionLM: Open-Access Data and Models for Detailed Visual Understanding

Jang Hyun Cho<sup>1,2,\*†</sup>, Andrea Madotto<sup>1,\*</sup>, Effrosyni Mavroudi<sup>1,\*</sup>, Triantafyllos Afouras<sup>1,\*</sup>, Tushar Nagarajan<sup>1,\*</sup>, Muhammad Maaz<sup>3,\*†</sup>, Yale Song<sup>1,\*</sup>, Tengyu Ma<sup>1,\*</sup>, Shuming Hu<sup>1,\*</sup>, Suyog Jain<sup>1</sup>, Miguel Martin<sup>1</sup>, Huiyu Wang<sup>1</sup>, Hanoona Rasheed<sup>3,†</sup>, Peize Sun<sup>1</sup>, Po-Yao Huang<sup>1</sup>, Daniel Bolya<sup>1</sup>, Nikhila Ravi<sup>1</sup>, Shashank Jain<sup>4</sup>, Tammy Stark<sup>4</sup>, Shane Moon<sup>4</sup>, Babak Damavandi<sup>4</sup>, Vivian Lee<sup>1</sup>, Andrew Westbury<sup>1</sup>, Salman Khan<sup>3</sup>, Philipp Krähenbühl<sup>2</sup>, Piotr Dollár<sup>1</sup>, Lorenzo Torresani<sup>1,\*</sup>, Kristen Grauman<sup>1,2,\*</sup>, Christoph Feichtenhofer<sup>1,\*</sup>

<sup>1</sup>Meta FAIR <sup>2</sup>UT Austin <sup>3</sup>MBZUAI <sup>4</sup>Meta Reality Labs

\*Joint first author <sup>†</sup>Work done during internships at Meta <sup>\*</sup>Project lead

## Abstract

Vision-language models are integral to computer vision research, yet many high-performing models remain closed-source, obscuring their data, design and training recipe. The research community has responded by using distillation from black-box models to label training data, achieving strong benchmark results, at the cost of measurable scientific progress. However, without knowing the details of the teacher model and its data sources, scientific progress remains difficult to measure. In this paper, we study building a Perception Language Model (PLM) in a fully open and reproducible framework for transparent research in image and video understanding. We analyze standard training pipelines without distillation from proprietary models and explore large-scale synthetic data to identify critical data gaps, particularly in detailed video understanding. To bridge these gaps, we release 2.8M human-labeled instances of fine-grained video question-answer pairs and spatio-temporally grounded video captions. Additionally, we introduce PLM–VideoBench, a suite for evaluating challenging video understanding tasks focusing on the ability to reason about “what”, “where”, “when”, and “how” of a video. We make our work fully reproducible by providing data, training recipes, code & models.

**GitHub:** [https://github.com/facebookresearch/perception\\_models](https://github.com/facebookresearch/perception_models)

## 1 Introduction

Vision-language models (VLMs) are now a key part of computer vision research and are widely used in both academia and industry. Many of the strongest performing VLMs are *closed-source*, meaning their design, training methods, and the data they use are not publicly shared. To stay competitive, the research community has started to catch up to the proprietary models by using a straightforward approach — *distillation from black-box models* [1, 2, 3, 4, 5], where proprietary models are directly used to label training data [3, 6, 7], directly leading to strong benchmark results.

Although distillation will unlock strong performance, there are two main issues for basic research. First, it makes it hard to track scientific progress. Specifically, we cannot tell if better results on benchmarks are due to advances in model design or training, or simply because the proprietary *teacher* models were trained on the evaluation sets of widely used benchmarks or *internal data* collected to resemble them — this information is not available. Second, the heavy reliance on distillation leads to a fundamental misunderstanding of the effectiveness of current methods for training VLMs *from scratch*. Several key questions remain unanswered, including the significance of each training stage,**Fine-Grained QA (FGQA)**

**Question**  
In which direction does the person fold the paper?

**Answer**  
Diagonally, pulling the top right corner towards the center of the middle crease of the yellow paper.

**Spatio-Temporal Captions (STC)**

The dog looks at the person throwing the ball

The dog is out of frame

The dog brings back the ball to the person.

**Perception Language Model**

- + Perception Encoder - 0.3B / 2B
- + Llama 3 - 1B / 3B / 8B
- + Image & Multi-image & Video
- + Largest video-language data
- + Reproducible training

Figure 1: We introduce the largest collection of manually annotated fine-grained activity QA and spatiotemporal captioning data (left panel). Together with this data, we train and release PLM —open and fully reproducible models to facilitate research in vision-language model training (right panel).

the influence of synthetic data , the *data gaps* that the research community should prioritize, and which of these gaps are currently being artificially addressed by distillation from proprietary models.

To better understand these challenges, we develop the Perception Language Model (PLM), a fully open and *reproducible* model for transparent research in image and video understanding (Fig. 1 right). PLM consists of a vision encoder with a small scale ( $<8\text{B}$  parameters) LLM decoder. We start by an analysis of standard training pipelines with available data, without any proprietary model distillation. We investigate large-scale synthetic data and establish key *scaling laws* to identify critical data gaps that limit *video understanding* performance, especially for *spatio-temporal reasoning* and *fine-grained understanding* tasks.

To fill these gaps, we create 2.8M high-quality human-labeled instances of fine-grained video QA and spatio-temporally grounded video captions, see Fig. 1. This release is nearly an order of magnitude larger than the largest existing video datasets of each type [8, 9]. Our model, dataset and benchmark push the boundaries of video understanding, and provide a foundation for reproducible and transparent training and evaluation of VLM research. Across 40 image and video benchmarks, we achieve comparable performance with existing state-of-the-art open-weight models (*e.g.*, InternVL2.5 [10]), *without* distilling from proprietary models, and greatly outperform fully open models (*i.e.*, Molmo [11]).

## 2 Related Work

**Vision-Language Models.** Building on the strengths of large language models (LLMs), several vision-language models (VLMs) have recently been proposed for image understanding [1, 12, 13, 14, 15, 16, 17, 18, 19], video understanding [20, 21, 22, 23, 24, 25, 26, 27] and joint understanding of both images and videos [10, 28, 29, 30]. These works employ several modeling advancements such as dynamic high resolution inputs [12], adaptive token compression [25, 31], and multimodal positional embeddings [30].

**Open source, open data VLMs.** Training data is a key component in developing powerful VLMs. Many existing approaches train on proprietary data that is not released to the community [32, 33, 34, 35, 36] or on data generated using proprietary models (*e.g.*, GPT4o) [3], effectively distilling the *closed* models. Doing so make measuring scientific progress difficult and limits research on how to train VLMs ground-up. Molmo [11] proposes a class of open-data models, however, they are image VLMs trained on relatively small-scale data, limiting their performance as our experiments will show.

**VLM Benchmarks.** Several benchmarks have been proposed to assess the capabilities of VLMs. Popular image benchmarks cover broad perception and reasoning [37, 38, 39, 40, 41, 42, 43, 44, 19, 45, 46, 47, 48] as well as capabilities like image captioning [49, 50, 51], document/diagram understanding [52, 53, 54, 55, 56, 57, 58, 59, 60, 61], mathematical reasoning [62, 63, 64], visual grounding [65, 66] and hallucination [67, 68]. Popular video benchmarks cover video question answering [20, 8, 69, 70, 71, 72, 73, 74, 75, 76, 77, 22, 78, 79, 80], video captioning [81, 82, 83, 84, 85, 86, 87], and hallucination in videos [88, 89]. Many of these video benchmarks remain *image-centric* — they have questions that can be answered with a few frames. Video-centric reasoning in benchmarks has been relatively neglected with benchmarks proposed only recently for long video understanding [90, 91, 92, 93, 94, 95, 96, 97, 98] and fine-grained, temporal reasoning [99, 100, 101, 102, 103]. We introduce PLM-VideoBench— a benchmark suite aimed at the core, video-centric capabilities that current benchmarks neglect, namely fine-grained activity understanding and spatio-temporally grounded reasoning.

### 3 PLM: Overview

In this section, we overview the *model*, *training stages* and *training data* involved in the development of PLM. Please refer to Fig. 8 for a detailed overview and Appendix A for additional details.

**Model.** PLM consists of a vision encoder and language decoder, where a pre-trained Perception Encoder (PE) [104] is connected to the Llama 3 [13] language decoder (1B, 3B, or 8B parameters) with a 2-layer MLP *projector*. We use PE L/14 for Llama3.2 1B and 3B, and PE G/14 for Llama3.1 8B. For image input, PLM incorporates dynamic tiling to support high resolution images for up to 36 tiles of  $448^2$  resolution, where each tile undergoes  $2 \times 2$  average pooling to compress the visual tokens. For video input, PLM uses 32 frames at  $448^2$  resolution, where the same pooling is applied across the spatial dimensions of each video frame.

<table border="1">
<thead>
<tr>
<th></th>
<th>Stage 1<br/>Warmup</th>
<th>Stage 2<br/>Midtraining</th>
<th>Stage 3<br/>SFT</th>
</tr>
</thead>
<tbody>
<tr>
<td>Modality</td>
<td>Image</td>
<td>Image + Video</td>
<td>Image + Video</td>
</tr>
<tr>
<td>Data</td>
<td>1M Synthetic</td>
<td>72M Mix</td>
<td>19M Mix</td>
</tr>
<tr>
<td>Training</td>
<td>Projector</td>
<td>Full</td>
<td>Full</td>
</tr>
<tr>
<td>Downsampling</td>
<td>-</td>
<td><math>2 \times 2</math></td>
<td><math>2 \times 2</math></td>
</tr>
<tr>
<td>Tiles/Frames</td>
<td>1/-</td>
<td>16/16</td>
<td>36/32</td>
</tr>
</tbody>
</table>

Table 1: Summary of three training stages to train PLM. See Appendix Table 7 and Table 8 for data splits.

**Data.** The data used to train the PLM consists of *synthetic* and *human-annotated* samples. Synthetic data enhances the *general* capabilities of PLM, while *human-annotated* data broadens these capabilities to encompass more complex tasks. Synthetic data is sourced from a diverse array of image and video datasets, covering fundamental VLM capabilities such as OCR, chart/document/diagram understanding, image/video captioning, and visual question answering.

We design data engines for each data modality (*e.g.*, natural images, charts, documents, figures, egocentric and exocentric videos) to efficiently scale up, creating  $\sim 66.1M$  samples (§4). The synthetic data can be noisy, but is available at large scale; on the other hand, human-annotated data provides rich, high-quality supervision for image and video tasks. Here, we combine existing human annotations of diverse image and video sources, with our own collected human-annotated data, specifically geared towards *fine-grained video understanding* and *spatio-temporally grounded reasoning* (§5).

**Training stages.** PLM trains in three stages:

**1. Projector warm-up.** First, we freeze the vision encoder and LLM and only train the vision projector on a small amount of synthetic image data. This *warms-up* the newly initialized parameters in the projector and improves stability for later stages. We use 1M images from SA-1B [105] with the image captions generated by our data engine (§4).

**2. Large-scale midtraining with synthetic data.** Next, we train PLM on diverse domains of images and videos *at scale*, using a maximum of 16 tiles for images and 16 frames for videos. PLM sees around 64.7M images and videos with synthetically generated captions and question-answer pairs. We employ our data engine to scale up synthetic data generation (see §4).

**3. Supervised fine-tuning with human-annotated data.** Finally, we train PLM with higher image resolutions and more video frames, using up to 36 tiles for images and 32 frames for videos. In this stage, we tackle more challenging video tasks, including *fine-grained QA* and *spatio-temporally grounded reasoning*.

<table border="1">
<thead>
<tr>
<th></th>
<th>Samples</th>
<th>Type</th>
<th>Stage</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="4"><i>Our Human-annotated (2.87M)</i></td>
</tr>
<tr>
<td>PLM-FGQA</td>
<td>2.4M</td>
<td>Fine-grained</td>
<td>3</td>
</tr>
<tr>
<td>PLM-STC</td>
<td>476.2K</td>
<td>R(D)Cap + RTL</td>
<td>3</td>
</tr>
<tr>
<td colspan="4"><i>Our Synthetic (66.1M)</i></td>
</tr>
<tr>
<td>Natural Images</td>
<td>15.9M</td>
<td>Caption</td>
<td>1,2,3</td>
</tr>
<tr>
<td>Charts &amp; Documents</td>
<td>31.9M</td>
<td>Caption</td>
<td>2,3</td>
</tr>
<tr>
<td>Videos Mix</td>
<td>17.5M</td>
<td>Mix.</td>
<td>2,3</td>
</tr>
<tr>
<td>Ego4D</td>
<td>880K</td>
<td>Cap. + QA</td>
<td>2,3</td>
</tr>
<tr>
<td colspan="4"><i>Existing Open Source (6.52M)</i></td>
</tr>
<tr>
<td>Image (92 datasets)</td>
<td>5.6M</td>
<td>Diverse</td>
<td>2,3</td>
</tr>
<tr>
<td>Video (27 datasets)</td>
<td>920K</td>
<td>Diverse</td>
<td>2,3</td>
</tr>
</tbody>
</table>

Table 2: Summary of the data mix for training PLM. See Table 9 for the full data blend.

Table 1 shows an overview of our training setup for each stage. Appendix A.1 provides the complete training recipe for each stage, including hyperparameters and data sources.## 4 Synthetic Data Generation and Scaling

The predominant paradigm for VLM training is to generate synthetic annotations as cheap alternatives to human-labeled data [1, 106, 30, 107, 10, 11, 15]. Although seemingly promising to get the best results on benchmarks, the majority of such data shared in the community is *derived from proprietary models*. This trend makes it hard to decouple scientific progress from proprietary distillation impact. In this section, we explore the efficacy of the current paradigm for VLM training in a *transparent* manner. We design our data engine entirely from *open-source* models and scale the synthetic data generation to around 66.1M samples of images and videos. We establish the scaling laws of training from synthetic data on standard VLM tasks, including image, OCR/document, and video tasks.

### 4.1 Data Engine

Our data engine is designed to target *base* capabilities of VLMs for image and video understanding.

**Image Data Engine.** We generate short and long captions, as well as question-answer pairs, for natural images and those containing documents, diagrams, and text recognizable by optical character recognition (OCR). We prompt openly accessible Llama 3 [13] model to produce factual, detailed image captions while minimizing hallucinations. To create *informative* question-answer pairs, we utilize OCR data, captions, and other *metadata*, which are fed into the prompt of a text-only LLM.

**Video Data Engine.** For videos, we first use an off-the-shelf scene detector [108] to extract video clips of approximately 30 seconds duration. Then, we extract the keyframes and generate frame-level captions using Llama 3, and video captions using our initial PLM trained with Stage 1 and Stage 3 data as shown in Table 2. We then employ an LLM to refine the frame-level and video captions by incorporating existing video metadata (*e.g.*, action labels, time tags) into a cohesive, detailed video-level caption. Similarly, we generate question-answer pairs from the video-level captions.

The resulting synthetic data is large-scale and diverse – 66.1M samples carefully curated from a variety of image and video sources including natural images, in-the-wild text, chart, figures, documents, egocentric and exocentric videos. Additional details are in Appendix J.

### 4.2 Scaling Laws with Synthetic Data

We examine scaling properties of our synthetic data under controlled setup and establish *scaling laws*.

Figure 2: **Synthetic Scaling Plots.** Relationship between Average Error across benchmarks and training compute (in floating-point operations) for various PLM models. We report average errors across Video QA tasks [75, 72, 90, 8, 70, 71], OCR QA tasks [109, 53, 56, 57], and Natural Images tasks [45, 110, 111, 68, 40, 112]. Model’s performance using only human-labeled data subset are reported (No Synth.) as well as the actual power-law fit of each subcategory.

**Setup.** To establish power-law relationship between compute and *validation-set errors* of downstream benchmarks, we vary the scale of synthetic data, language model decoders (1B, 3B, and 8B), vision encoders (300M and 2B), and resolution/number of frames. For each configuration, we train a model with the 66.1M synthetic data from our data engine and 6.5M publicly available human-labeled data, following stage 2 training described in §3. At every 2M samples, we evaluate PLM on three categories of downstream benchmarks (*VideoQA*, *OCR QA*, *Natural QA*), constructed from 20 vision-language understanding benchmarks that provide a comprehensive and general evaluation ofmulti-modal large language models. We compute the *pareto frontier* of these data points and fit a power law relationship:  $\text{Err.} = (\beta \times \text{FLOP})^\alpha$  and compare the exponents  $\alpha$  of the power function as *scalability* of each setup, where a smaller  $\alpha$  implies better scaling.

**Scaling with decoder size.** Fig. 2 shows the scaling behavior of PLM across various LLM sizes. We show validation-set errors and training compute on a logarithmic scale, with the black linear line representing the power-law relationship between them. Different colors (green, turquoise, and blue) represent different language model scales (1B, 3B, 8B) while keeping the vision encoder size constant at 300M. As described in the setup section above, we show the power law fit of the pareto frontier in each benchmark category. We also show the results of PLM only trained on 4M *human-labeled* datasets as baselines, denoted with horizontal lines of each color. The gap from the horizontal line to the data point marks the impact of the synthetic data. Interestingly, all three categories of benchmarks demonstrate clear power-law relationship between compute and average benchmark errors, with the power law exponent ( $\alpha$ ) of  $-0.15$ ,  $-0.20$ , and  $-0.11$  for Video QA, OCR QA, and Natural Image QA, respectively. In Appendix B, we provide more details and extend the analysis to (1) *scaling the encoder size*, and (2) *scaling the image resolution and video frames*.

**Limitation of synthetic data.** In Fig. 3, we evaluate stage 2 on an extended set of video benchmarks. Specifically, we show the result of 7 *challenging video tasks* on fine-grained activity understanding [97, 100, 89, 101, 99], temporal grounding [113] and long-video reasoning [92]. Unlike generic, high-level understanding (*e.g.*, “what is happening in this video”), the “challenging” tasks require a thorough understanding of video in space and time, and fine-grained semantic details. As shown, the challenging video tasks (“HardQA” in lavender, plum, magenta) show a poor scaling trend ( $-0.03$ ) compared to general video QA ( $-0.15$ ). The stark difference between the two power law fits shows that *scaling synthetic data is only effective for established, base tasks*. Extending VLMs to these more challenging, complex tasks still remain unsolved. Next, we address this challenge with high-quality human-annotated video data, **PLM-FGQA** and **PLM-STC**.

Figure 3: **Limitation of synthetic data.** Challenging video tasks (HardQA [97, 100, 89, 101, 99, 113, 92]) do not scale well with synthetic data.

## 5 Human-annotated High Quality Data

As shown in Fig. 3, the current paradigm with synthetic data has run out of steam. Training from tens of millions of synthetically annotated data hardly improves our model on new, *challenging* video benchmarks. Beyond standard VLM tasks, these benchmarks focus on advanced capabilities such as fine-grained activity understanding, temporal grounding, and long video understanding. Perhaps, the knowledge that these benchmarks examine is simply not present in the initial training set of our data engine nor in existing human-annotated data. Our community lacks high quality datasets for detailed visual understanding to start from, that covers *what*, *where*, *when*, and *how* of activities in video. To address this gap, we introduce two large-scale, human-annotated video datasets:

**PLM-FGQA** is a fine-grained video QA dataset collected by asking human annotators to watch a short video segment and answer model-generated questions which focus on “*what*” activities humans perform and “*how*” they perform these activities. Question types include fine-grained recognition (action and object), fine-grained temporal perception (direction of movements, repetition counts, hand pose etc.), and fine-grained spatial understanding (object locations and spatial relationships). We use a multi-stage data engine to first extract video segments with salient actions from untrimmed videos through temporal clustering and shot-detection. Next, we generate questions and answers using either a text-only LLM or an early version of PLM. Finally, we refine the answers by asking humans to verify or replace them if they are incorrect, resulting in a high-quality QA pairs.

Overall, we collect 2.4M question answer pairs from various open-access video datasets [114, 115, 116, 117, 118, 83] spanning over 780k unique video clips from diverse domains (*e.g.*, cooking, DIY, carpentry, automotive and bike repair) and viewpoints (egocentric and third-person); refer to Fig. 13 for domain statistics. This is nearly 8 times larger than the size of the largest existing human-annotatedFigure 4: **Overview PLM-FGQA**. Examples of question-answer pairs from PLM-FGQA, focusing on fine-grained human activity understanding. PLM-FGQA is approximately 8 times larger than the largest existing human-annotated video QA dataset and addresses a wide range of fine-grained question types that are scarce in existing video QA datasets, such as ones that cover *direction of movement*, *object states*, *locations* and *spatial relations*.

video QA dataset in the community [91]. Moreover, as illustrated by the breakdown of question types<sup>1</sup> in Fig. 4 (top-right), PLM-FGQA contains a large number of annotations about fine-grained details that have been largely missing in existing training video QA datasets [119, 69, 71, 76, 20, 120, 121, 122, 123]. Please refer to Table 16 for comparison with existing datasets Table 17 for dataset examples and Appendix G for further details.

**PLM-STC** is a spatio-temporal video captioning dataset that offers detailed activity descriptions for each video. It includes timestamps (“*when*”) of each activity and focuses on specific subjects identified by a masklet (“*where*”). We employ a two-stage annotation process to improve efficiency in collecting PLM-STC. In the first stage, annotators select interesting objects that exhibit significant motion changes in the video and use SAM 2 [124] to generate initial mask tablets, which they then refine to ensure high-quality spatial-temporal segmentation. For segments where the subject is out of frame, we automatically supplement “out of frame” caption. In the second stage, a separate set of annotators write temporally localized descriptions of the highlighted subject focusing on the *changes in action across time in relation to the whole video*.

Figure 5: **Overview of PLM-STC**. Examples of spatio-temporally grounded captions from PLM-STC, the first dataset to associate each caption both with a temporal interval as well as a high-fps sequence of segmentation masks of the subject - *i.e.*, masklets (compared to just a temporal interval or a sparse sequence of bounding boxes).

Overall, we collect 194.2K spatio-temporal captions as the first existing large-scale dense video-region captioning dataset. We convert these spatio-temporal captions into three tasks for training: RCap (194.2K): Given the video region and timestamps, the model generates a caption; RTLoc (194.2K): Given the video region and caption, the model localizes the action; and RDCap (122.3K): Given the video region, the model generates dense, localized captions. In total, we construct 194.2K + 194.2K + 122.3K = 522.7K samples, of which 476.2K are used for training and the rest for constructing

<sup>1</sup>obtained with LLM-based tagging.PLM–VideoBench. Please refer to Fig. 5 for dataset examples, Table 19 for comparison with existing datasets, Table 20 for dataset statistics and Appendix H for further details.

## 5.1 PLM–VideoBench

Our high-quality human-annotated data offers VLMs to train for broader range of capabilities for holistic video understanding. However, existing video benchmarks are not adequately equipped to evaluate these. To this end, we introduce PLM–VideoBench, a novel benchmark focusing on specific activities (what) and their execution details (how) within spatio-temporal contexts (where and when).

Figure 6: **PLM-Video Dataset** includes fine-grained video QA (FGQA), open-ended QA in videos recorded using smart glasses (SGQA), Spatio-Temporal Captions (STC) post-processed into video region captioning (RCap), video region temporal localization (RTLoc) and video region dense captioning (RDCap) tasks.

**Fine-Grained Question Answering (FGQA).** In this task, a model must answer a multiple-choice question (MCQ) that probes nuanced, fine-grained activity understanding (*e.g.*, painting “vertically” vs. “horizontally” in Fig. 6, first). We report multi-binary accuracy (MBAcc) [99] where each question is split into multiple binary choice questions. Our test set consists of 4,371 question-answer pairs. For more information, including statistics on video clips, segment duration, question types, and benchmark construction, see Table 18 and §G.2.

**Smart Glasses Question Answering (SGQA).** In this task, a model must answer open-ended questions about activities and objects visible in an egocentric video stream recorded by a smart-glasses device (see Fig. 6, second). The questions are designed to simulate real-world scenarios where a user would ask for assistance from their smart glasses. We manually collect the videos using commercially available smart glasses, providing a completely new, unique dataset that reflects modern use-cases such as online AI video assistance and activity coaching. For evaluation, we use LLM-judge accuracy with an open-access model (Llama3.3 70B). The test set consists of 665 human-annotated question-answer pairs. See Appendix I for more details.

**Video Region Captioning (RCap).** In this task, a model must generate a detailed description of an event involving a subject of interest in the video. Given a region masklet and a specified time interval, the model is required to output a caption that accurately describes the event occurring within that interval. Compared to traditional video captioning [125, 83, 84] where the aim is to generate a *video-level* caption, the goal is to generate a *region-level* caption tied to a specific subject (*e.g.*, a person, object or animal) (see Fig. 6, third). The test set contains 10,060 human-annotated instances and we report LLM-judge accuracy with Llama3.3 70B. See Appendix C.3 for details.

**Region Temporal Localization (RTLoc).** In this task, a model must identify the precise time interval within the video when the specified event takes place for the given subject. Given a video, a region masklet and a text description of the event, the model is required to output the start and end timestamps that correspond to the occurrence of the event (see Fig. 6 fourth). Notably, this task is the inverse of RCap — instead of generating the caption, the model receives it as input and generates the corresponding time interval. We filter the test set to include only the captions that are unambiguously localized, *i.e.*, they map to a single time window in the video. As a result, the test set size is reduced to 7,910 instances compared to RCap. We report average recall@1 over IoU thresholds (0.3, 0.5, 0.7, 0.9). See Appendix C.3 for details.**Region Dense Video Captioning (RDCap).** In this task, a model must generate a detailed description of all events involving a specific subject of interest (*e.g.*, person, animal, or object) in a video. Given a video and a region masklet, the model must produce a sequence of (*start*, *end*, *caption*) tuples that cover the entire duration of the video, including periods when the subject is not visible (see Fig. 6, last). This task is a composition of RTLoc and RCap, requiring the model to produce both temporal windows for events as well as captions directly from the video. The test set contains 2,620 samples and we report the SODA score [126] which uses an LLM judge. See Appendix C.3 for details.

## 6 Experiments

We first overview the baselines and evaluation setting (§6.1). We then compare benchmark results of PLMs with the baselines on a broad collection of image (§6.2) and video (§6.3) tasks as well as on our PLM-VideoBench (§6.4). Finally, we provide analyses on data and model ablations (§6.5).

### 6.1 Setup

We compare PLMs against the following two classes of baselines:

- • **Proprietary models** such as GPT-4o [33] (gpt-4o-2024-11-20), Gemini-Pro 1.5 [34] and Gemini-Flash 2.0 [35]. We use API calls to evaluate these models.
- • **Open-access models** such as Molmo-O [11], LLaVA-OneVision [28], Qwen2.5-VL [106] and InternVL2.5 [10] — state-of-the-art *open-access* models, for which model scale, architecture and inference code are available. We use the official inference code for all models.

**Inference protocol.** For mask inputs in PLM-VideoBench, we overlay a colored box on the video frames to specify the regions. We report validation set performance unless specified (in brackets) under the benchmark name. Metrics marked with † use LLM as a judge. Complete implementation details including inference hyper-parameters, task prompts, judge prompts and proprietary model evaluation protocol can be found in Appendix C.4.

### 6.2 Image Benchmark Results

We evaluate PLM on a total of 20 image benchmarks. **Charts, Diagrams and Documents:** answer questions that require parsing images of documents and diagrams; **Image Captioning:** generate a short/detailed caption, **Perception and Reasoning:** answer questions of varying difficulty about objects, actions, functional correspondence, multi-view reasoning, spatial layout etc. and **Hallucination:** evaluate robustness to *hallucinated* details. More details are in Appendix C.1.

Table 3 shows our results. Overall, PLM shows strong performance on a wide spectrum of image benchmarks with *solely from open-access data with a white-box data engine*. Additionally, we report

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="5">Charts, Diagrams and Documents</th>
<th colspan="5">Perception and Reasoning</th>
<th colspan="4">Hard Perception</th>
<th colspan="2">Halluc.</th>
</tr>
<tr>
<th>DocVQA<br/>(test) acc [53]</th>
<th>ChartQA<br/>acc [54]</th>
<th>TextVQA<br/>acc [52]</th>
<th>InfoQA<br/>(test) acc [56]</th>
<th>AI2D<br/>(info_mask)<br/>acc [55]</th>
<th>OCRBatch<br/>acc [57]</th>
<th>MMU<br/>(val) acc [37]</th>
<th>VQW2<br/>(val) acc [111]</th>
<th>OK-VQA<br/>acc [39]</th>
<th>VizWiz<br/>acc [40]</th>
<th>SEED<br/>(image) acc [58]</th>
<th>BLINK<br/>(multi-image)<br/>acc [44]</th>
<th>CV-Bench<br/>acc [19]</th>
<th>RealWorldQA<br/>acc [45]</th>
<th>VSR<br/>acc [127]</th>
<th>POPE<br/>acc [68]</th>
</tr>
</thead>
<tbody>
<tr>
<td>GPT-4o [33]</td>
<td>92.8*</td>
<td>85.7*</td>
<td>75.3</td>
<td>80.7*</td>
<td>94.2*</td>
<td>810</td>
<td>70.7*</td>
<td>-</td>
<td>63.9</td>
<td>-</td>
<td>77.1*</td>
<td>68.0*</td>
<td>72.5</td>
<td>73.9</td>
<td>78.0</td>
<td>87.2*</td>
</tr>
<tr>
<td>Gemini 1.5 Pro [35]</td>
<td>94.0</td>
<td>84.2</td>
<td>74.8</td>
<td>81.0*</td>
<td>95.7</td>
<td>830</td>
<td>63.2</td>
<td>-</td>
<td>63.9</td>
<td>-</td>
<td>77.8</td>
<td>59.8</td>
<td>81.0</td>
<td>66.3</td>
<td>76.1</td>
<td>88.2*</td>
</tr>
<tr>
<td>Gemini 2.0 Flash [35]</td>
<td>93.0</td>
<td>84.8</td>
<td>80.2</td>
<td>81.0</td>
<td>94.0</td>
<td>792</td>
<td>69.9*</td>
<td>-</td>
<td>57.8</td>
<td>-</td>
<td>77.0</td>
<td>64.4</td>
<td>82.3</td>
<td>71.9</td>
<td>74.8</td>
<td>-</td>
</tr>
<tr>
<td colspan="18"><b>1B scale</b></td>
</tr>
<tr>
<td>Qwen2VL-2B [30]</td>
<td>90.1*</td>
<td>75.3</td>
<td>80.3</td>
<td><b>65.5*</b></td>
<td>84.6*</td>
<td><b>809*</b></td>
<td><b>41.1*</b></td>
<td>80.0</td>
<td>59.7</td>
<td><b>67.4</b></td>
<td>72.9</td>
<td>44.4*</td>
<td>17.3</td>
<td>62.6*</td>
<td>73.0</td>
<td>87.2</td>
</tr>
<tr>
<td>InternVL2.5-1B [10]</td>
<td>84.8*</td>
<td>75.9*</td>
<td>72.0*</td>
<td>56.0*</td>
<td>77.8*</td>
<td>785*</td>
<td>40.9*</td>
<td>72.2</td>
<td>51.5</td>
<td>47.4</td>
<td>71.3</td>
<td>42.4</td>
<td>42.1</td>
<td>58.3</td>
<td>65.4</td>
<td><b>90.2</b></td>
</tr>
<tr>
<td>PLM-1B</td>
<td><b>90.7</b></td>
<td><b>78.6</b></td>
<td><b>82.1</b></td>
<td>63.0</td>
<td><b>84.9</b></td>
<td>807</td>
<td>34.8</td>
<td><b>81.7</b></td>
<td><b>61.0</b></td>
<td>59.7</td>
<td><b>76.3</b></td>
<td><b>46.8</b></td>
<td><b>73.8</b></td>
<td><b>67.1</b></td>
<td><b>68.8</b></td>
<td>88.4</td>
</tr>
<tr>
<td colspan="18"><b>3B scale</b></td>
</tr>
<tr>
<td>Qwen2.5 VL-3B [106]</td>
<td><b>93.9*</b></td>
<td>83.1</td>
<td>79.3*</td>
<td><b>77.1*</b></td>
<td>90.2</td>
<td>797*</td>
<td>53.1*</td>
<td>80.8</td>
<td>63.2</td>
<td><b>71.9</b></td>
<td>73.1</td>
<td>47.6*</td>
<td>54.4</td>
<td>65.4*</td>
<td>78.5</td>
<td>88.2</td>
</tr>
<tr>
<td>InternVL2.5-4B [10]</td>
<td>91.6*</td>
<td>84.0*</td>
<td>79.3</td>
<td>72.1*</td>
<td>90.5*</td>
<td>828*</td>
<td><b>52.3*</b></td>
<td>80.9</td>
<td>64.0</td>
<td>61.8</td>
<td>75.6</td>
<td>50.8*</td>
<td>55.9</td>
<td>64.6</td>
<td>80.0</td>
<td><b>91.0</b></td>
</tr>
<tr>
<td>PLM-3B</td>
<td>93.8</td>
<td><b>84.3</b></td>
<td><b>84.3</b></td>
<td>74.6</td>
<td><b>90.9</b></td>
<td><b>830</b></td>
<td>41.2</td>
<td><b>84.3</b></td>
<td><b>66.8</b></td>
<td>64.0</td>
<td><b>78.5</b></td>
<td><b>55.4</b></td>
<td><b>81.4</b></td>
<td><b>72.4</b></td>
<td><b>80.4</b></td>
<td>88.7</td>
</tr>
<tr>
<td colspan="18"><b>8B scale</b></td>
</tr>
<tr>
<td>Molmo-7B-O [11]</td>
<td>90.8*</td>
<td>80.4*</td>
<td>80.4*</td>
<td>70.0*</td>
<td>90.7*</td>
<td>-</td>
<td>39.3*</td>
<td>85.3*</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>67.5*</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>LLaVA-OV-7B [28]</td>
<td>86.7</td>
<td>80.0</td>
<td>77.3</td>
<td>68.8</td>
<td>90.1</td>
<td>656</td>
<td>48.9</td>
<td>83.5</td>
<td><b>69.6</b></td>
<td>63.4</td>
<td>76.4</td>
<td>49.4</td>
<td>75.0</td>
<td>66.7</td>
<td>78.1</td>
<td>89.2</td>
</tr>
<tr>
<td>Qwen2.5VL-7B [106]</td>
<td><b>95.7*</b></td>
<td><b>87.3*</b></td>
<td>84.9*</td>
<td><b>82.6*</b></td>
<td><b>93.0</b></td>
<td>864*</td>
<td>58.6*</td>
<td>70.1</td>
<td>61.0</td>
<td>73.5</td>
<td>73.2</td>
<td><b>56.4*</b></td>
<td>11.9</td>
<td>69.8</td>
<td>80.3</td>
<td>87.2</td>
</tr>
<tr>
<td>InternVL2.5-8B [10]</td>
<td>93.0*</td>
<td>84.8*</td>
<td>79.3</td>
<td>77.6*</td>
<td>92.8*</td>
<td>823</td>
<td><b>56.0*</b></td>
<td>80.6</td>
<td>69.2</td>
<td>64.3</td>
<td>77.6</td>
<td>54.8*</td>
<td>53.9</td>
<td>70.1*</td>
<td>80.0</td>
<td><b>90.6*</b></td>
</tr>
<tr>
<td>PLM-8B</td>
<td>94.6</td>
<td>85.5</td>
<td><b>86.5</b></td>
<td>80.9</td>
<td>92.7</td>
<td><b>870</b></td>
<td>46.1</td>
<td><b>85.6</b></td>
<td><b>69.6</b></td>
<td>67.0</td>
<td><b>79.3</b></td>
<td>56.0</td>
<td><b>81.3</b></td>
<td><b>75.0</b></td>
<td><b>82.8</b></td>
<td>89.9</td>
</tr>
</tbody>
</table>

Table 3: **Image benchmarks.** PLM versus proprietary models and open-access baselines of comparable scale. Cells with \* are reported numbers from literature, and the remaining are reproduced using official code.<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="3">VCap.</th>
<th colspan="4">Video QA</th>
<th colspan="4">Fine-grained Video QA</th>
<th colspan="2">T.Loc.</th>
<th colspan="2">Halluc.</th>
</tr>
<tr>
<th>DREAM-1K<br/><i>F1</i> [86]</th>
<th>MVBench<br/><i>acc</i> [70]</th>
<th>NEXT-QA<br/><i>acc</i> [69]</th>
<th>PerceptionTest<br/><i>(test)</i> <i>acc</i> [71]</th>
<th>STAR<br/><i>acc</i> [72]</th>
<th>Video-MME<br/><i>acc</i> [73]</th>
<th>ActivityNet-QA<br/><i>acc</i><sup>†</sup> [76]</th>
<th>EgoSchema<br/><i>(test)</i> <i>acc</i> [90]</th>
<th>TemporalBench<br/><i>MBAcc</i> [99]</th>
<th>TOMATO<br/><i>acc</i> [100]</th>
<th>MotionBench<br/><i>(dev)</i> <i>acc</i> [101]</th>
<th>TempCompass<br/><i>(MCQ)</i> <i>acc</i> [102]</th>
<th>CG-Bench<br/><i>(clue)</i> <i>acc</i> [97]</th>
<th>Charades-STA<br/><i>mIOU</i> [113]</th>
<th>VideoHalluc<br/><i>overall</i> <i>acc</i> [88]</th>
<th>EventHalluc<br/><i>(binary)</i> <i>acc</i> [89]</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="17"><b>Proprietary</b></td>
</tr>
<tr>
<td>GPT-4o [33]</td>
<td>-</td>
<td>64.6*</td>
<td>79.1</td>
<td>-</td>
<td>70.4</td>
<td>71.9*</td>
<td>-</td>
<td>72.2*</td>
<td>38.5*</td>
<td>37.7*</td>
<td>55.9</td>
<td>74.5</td>
<td>58.3*</td>
<td>38.6</td>
<td>56.4</td>
<td>91.9*</td>
</tr>
<tr>
<td>Gemini 1.5 Pro [35]</td>
<td>-</td>
<td>60.5*</td>
<td>81.6</td>
<td>65.9</td>
<td>-</td>
<td>75.0*</td>
<td>56.7*</td>
<td>71.2*</td>
<td>34.7</td>
<td>32.0</td>
<td>56.1</td>
<td>75.6</td>
<td>50.1*</td>
<td>34.2</td>
<td>56.0</td>
<td>80.9</td>
</tr>
<tr>
<td>Gemini 2.0 Flash [35]</td>
<td>-</td>
<td>60.7</td>
<td>81.9</td>
<td>-</td>
<td>-</td>
<td>70.3*</td>
<td>-</td>
<td>71.5*</td>
<td>27.6</td>
<td>32.8</td>
<td>56.1</td>
<td>76.9</td>
<td>47.0*</td>
<td>29.8</td>
<td>60.1</td>
<td>81.6</td>
</tr>
<tr>
<td colspan="17"><b>1B scale</b></td>
</tr>
<tr>
<td>Qwen2VL-2B [30]</td>
<td>26.8</td>
<td>63.2*</td>
<td>76.4</td>
<td>53.9*</td>
<td>67.3</td>
<td><b>55.6*</b></td>
<td>38.4</td>
<td>27.0</td>
<td>13.1</td>
<td>25.7</td>
<td>46.9</td>
<td>62.3</td>
<td>42.8</td>
<td>0.3</td>
<td>34.9</td>
<td>59.9</td>
</tr>
<tr>
<td>InternVL2.5-1B [10]</td>
<td>27.7</td>
<td>64.8</td>
<td>74.3</td>
<td>59.4</td>
<td>73.0</td>
<td>50.3*</td>
<td>60.7</td>
<td>55.7</td>
<td><b>27.7</b></td>
<td>25.0</td>
<td>45.0</td>
<td>56.4</td>
<td>40.9</td>
<td>0.8</td>
<td>31.0</td>
<td>38.9</td>
</tr>
<tr>
<td>PLM-1B</td>
<td><b>34.3</b></td>
<td><b>70.1</b></td>
<td><b>80.3</b></td>
<td><b>72.7</b></td>
<td><b>83.7</b></td>
<td>49.2</td>
<td><b>62.5</b></td>
<td><b>60.4</b></td>
<td>18.2</td>
<td><b>25.5</b></td>
<td><b>52.2</b></td>
<td><b>64.6</b></td>
<td><b>43.6</b></td>
<td><b>55.2</b></td>
<td><b>49.2</b></td>
<td><b>79.5</b></td>
</tr>
<tr>
<td colspan="17"><b>3B scale</b></td>
</tr>
<tr>
<td>Qwen2.5 VL-3B [106]</td>
<td>20.3</td>
<td>67.0</td>
<td>76.8</td>
<td>66.9*</td>
<td>63.0</td>
<td>61.5*</td>
<td>59.2</td>
<td>64.8*</td>
<td>17.2</td>
<td>23.5</td>
<td>49.2</td>
<td>63.0</td>
<td>45.7</td>
<td>38.8*</td>
<td>45.2</td>
<td>53.5</td>
</tr>
<tr>
<td>InternVL2.5-4B [10]</td>
<td>29.2</td>
<td>71.7</td>
<td>82.5</td>
<td>67.9</td>
<td>77.2</td>
<td><b>62.3*</b></td>
<td>64.1</td>
<td>66.6</td>
<td><b>23.7</b></td>
<td>27.4</td>
<td>52.7</td>
<td>65.2</td>
<td><b>52.0</b></td>
<td>8.4</td>
<td>49.6</td>
<td>66.3</td>
</tr>
<tr>
<td>PLM-3B</td>
<td><b>37.4</b></td>
<td><b>74.7</b></td>
<td><b>83.4</b></td>
<td><b>79.3</b></td>
<td><b>84.8</b></td>
<td>54.9</td>
<td><b>66.2</b></td>
<td><b>66.9</b></td>
<td>23.4</td>
<td><b>30.9</b></td>
<td><b>60.4</b></td>
<td><b>69.3</b></td>
<td>47.2</td>
<td><b>57.7</b></td>
<td><b>55.5</b></td>
<td><b>76.5</b></td>
</tr>
<tr>
<td colspan="17"><b>8B scale</b></td>
</tr>
<tr>
<td>LLaVA-OV-7B [28]</td>
<td>28.0</td>
<td>57.1</td>
<td>81.0</td>
<td>58.1</td>
<td>66.0</td>
<td>57.7</td>
<td>60.5</td>
<td>45.4</td>
<td>19.5</td>
<td>27.6</td>
<td>53.7</td>
<td>67.8</td>
<td>41.2</td>
<td>12.1</td>
<td>34.7</td>
<td>61.1</td>
</tr>
<tr>
<td>Qwen2.5VL-7B [106]</td>
<td>23.3</td>
<td>69.6*</td>
<td>80.0</td>
<td>70.5*</td>
<td>68.1</td>
<td><b>65.5*</b></td>
<td>63.7</td>
<td>65.0*</td>
<td>24.5</td>
<td>24.6</td>
<td>51.1</td>
<td>71.7*</td>
<td>49.8</td>
<td>43.6*</td>
<td>50.1</td>
<td>61.1</td>
</tr>
<tr>
<td>InternVL2.5-8B [10]</td>
<td>28.5</td>
<td>72.6</td>
<td><b>85.5</b></td>
<td>68.9*</td>
<td>77.6</td>
<td>64.2*</td>
<td>66.1</td>
<td>66.2*</td>
<td>24.3</td>
<td>29.4</td>
<td>53.5</td>
<td>68.3*</td>
<td><b>53.1</b></td>
<td>14.3</td>
<td>57.1</td>
<td>60.2</td>
</tr>
<tr>
<td>PLM-8B</td>
<td><b>35.9</b></td>
<td><b>77.1</b></td>
<td>84.1</td>
<td><b>82.7</b></td>
<td><b>84.9</b></td>
<td>58.3</td>
<td><b>67.3</b></td>
<td><b>68.8</b></td>
<td><b>28.3</b></td>
<td><b>33.2</b></td>
<td><b>61.4</b></td>
<td><b>72.7</b></td>
<td>46.4</td>
<td><b>58.6</b></td>
<td><b>57.7</b></td>
<td><b>77.3</b></td>
</tr>
</tbody>
</table>

Table 4: **Video benchmark results.** PLM versus proprietary models and open-access baselines of comparable scale. Cells with \* are reported numbers from literature and the remaining are reproduced using official code.

Image Grounding task results on RefCOCO+/g [65] datasets in Appendix Table 14, and show that PLM outperforms both specialist models as well as the VLM baselines in all model scales.

### 6.3 Video Benchmark Results

We evaluate PLM on a total of 25 video benchmarks. We divide these into the following categories. **Video Captioning:** generate a short caption for a video, or a dense description of all events; **Short video QA:** answer a question about a short video (few seconds to a minute), either by selecting from a list of options, or providing a free-form answer; **Long video QA:** answer a question as before, about a much longer video (minutes to hours); **Fine-grained QA:** answer detailed questions about spatial location, motion, temporal information etc.; and **Hallucination:** evaluate the robustness of video models to *hallucinated* details about objects and events.

Table 4 shows video captioning, video QA, fine-grained video QA, and video hallucination results. We achieve strong results on widely adopted benchmarks, despite only using open-access data mix free from proprietary model artifacts, outperforming both the open-access and proprietary models.

Further, we achieve competitive performance on the majority of challenging benchmarks, such as EgoSchema (68.8 %), MotionBench (61.4 %), TOMATO (33.2 %), TempCompass (72.7 %), TemporalBench (28.3 &), Charades-STA (58.6 %), and more. All our model scales show strong performance against both proprietary models as well as open-access baselines of same scale.

Lastly, we also show that PLMs at all scale greatly outperform existing approaches on captioning tasks and hallucination detection tasks, owing to our focus on detailed, fine-grained spatio-temporal annotations in our human-annotated data collection.

### 6.4 PLM-VideoBench Results

We report the result on our proposed benchmark PLM-VideoBench from §5.1 in Table 5. We evaluate our PLM as well as (proprietary and open-access) baselines. In addition, we provide human performance of each subtask in the first row. The results show a significant gap between the baselines and PLM. Proprietary baselines and open-source baselines alike perform reasonably on FGQA tasks, though still 6.5 points lower than PLM (61.2 vs 67.7). On SGQA, where the video sources and the question-answer pairs are unseen to all models, PLM performs reasonably well, yet 2.1 points short from open-access best (InternVL2.5) and far from the best proprietary model

<table border="1">
<thead>
<tr>
<th>Model</th>
<th>FGQA<br/><i>MBAcc</i></th>
<th>SGQA<br/><i>acc</i><sup>†</sup></th>
<th>RDCap<br/><i>SODA</i><sup>†</sup></th>
<th>RCap<br/><i>score</i><sup>†</sup></th>
<th>RTLoc<br/><i>meanR</i></th>
<th>Avg.</th>
</tr>
</thead>
<tbody>
<tr>
<td>Human perf.</td>
<td>90.9</td>
<td>67.9</td>
<td>66.6</td>
<td>53.9</td>
<td>67.8</td>
<td>73.9</td>
</tr>
<tr>
<td colspan="7"><b>Proprietary</b></td>
</tr>
<tr>
<td>GPT-4o [33]</td>
<td>61.2</td>
<td>63.7</td>
<td>20.9</td>
<td>35.7</td>
<td>33.1</td>
<td>51.6</td>
</tr>
<tr>
<td>Gemini 1.5 Pro [35]</td>
<td>57.1</td>
<td>49.9</td>
<td>14.4</td>
<td>33.1</td>
<td>27.6</td>
<td>44.0</td>
</tr>
<tr>
<td>Gemini 2.0 Flash [35]</td>
<td>58.7</td>
<td>44.8</td>
<td>13.2</td>
<td>30.9</td>
<td>27.6</td>
<td>42.5</td>
</tr>
<tr>
<td colspan="7"><b>Open-access</b></td>
</tr>
<tr>
<td>LLaVA-OV-7B [28]</td>
<td>40.2</td>
<td>41.5</td>
<td>4.7</td>
<td>24.4</td>
<td>13.9</td>
<td>32.0</td>
</tr>
<tr>
<td>Qwen2VL-7B [30]</td>
<td>49.2</td>
<td>44.5</td>
<td>4.1</td>
<td>17.6</td>
<td>15.1</td>
<td>35.3</td>
</tr>
<tr>
<td>Qwen2.5VL-7B [106]</td>
<td>49.8</td>
<td>43.0</td>
<td>2.5</td>
<td>21.5</td>
<td>10.7</td>
<td>34.8</td>
</tr>
<tr>
<td>InternVL2-8B [10]</td>
<td>47.7</td>
<td>45.9</td>
<td>1.2</td>
<td>21.5</td>
<td>11.6</td>
<td>35.0</td>
</tr>
<tr>
<td>InternVL2.5-8B [10]</td>
<td>53.7</td>
<td><b>48.3</b></td>
<td>5.7</td>
<td>26.1</td>
<td>8.8</td>
<td>38.5</td>
</tr>
<tr>
<td>PLM-8B</td>
<td><b>67.7</b></td>
<td>46.2</td>
<td><b>52.8</b></td>
<td><b>46.6</b></td>
<td><b>59.1</b></td>
<td><b>55.6</b></td>
</tr>
</tbody>
</table>

Table 5: **PLM-VideoBench results.** We evaluate PLM against baselines and report breakdowns. We report human performance in the first row.(GPT-4o). On spatio-temporal tasks (RDCap, DCap, RTLoc), open source baselines are unable to perform grounded reasoning and default to repeating the same caption for every time interval. Proprietary models perform reasonably well, yet far from the human performance. In all sub-tasks of PLM-VideoBench, PLM shows competitive performance compared to proprietary and open-access baselines. Results for all model scales are in Appendix D.

Note that the human performance varies based on the nature of the task and evaluation metrics. For example, FGQA human scores are naturally higher than RCap because the task is structured (select the correct option vs. open-ended) and the metric is objective (accuracy vs. LLM-judge accuracy).

## 6.5 Ablation Studies

**Setup.** We perform an ablation study to assess the importance of each of our proposed data, both synthetic and human-annotated. We start with PLM 3B after stage 2 training, and finetune on 4M short image and video SFT data mix<sup>2</sup> for the data ablation. We evaluate and report average video benchmark performance across five categories — video captioning, short video QA, fine-grained QA, and video hallucination, as well as spatial and temporal tasks, PLM-VideoBench and three image categories — image OCR, image captioning, and image perception. Full details are in Appendix A.3.

<table border="1">
<thead>
<tr>
<th rowspan="2">PLM-Synth.</th>
<th rowspan="2">PLM-STC</th>
<th rowspan="2">PLM-FGQA</th>
<th colspan="4">PLM-VideoBench</th>
<th colspan="4">Video Tasks</th>
<th colspan="4">Image Tasks</th>
</tr>
<tr>
<th>Total Average</th>
<th>PLM-FGQA<br/>w/BAcc</th>
<th>PLM-SGQA<br/>acc<sup>1</sup></th>
<th>PLM-ST<br/>3 metric avg.</th>
<th>Fine-Grained QA<br/>5 benchmark avg.</th>
<th>Video Cap.<br/>Dream 1K</th>
<th>Video QA<br/>5 benchmark avg.</th>
<th>Video Hallu.<br/>2 benchmark avg.</th>
<th>Spatio&amp;Temp.<br/>4 benchmark avg.</th>
<th>Image OCR<br/>6 benchmark avg.</th>
<th>Image Cap.<br/>3 benchmark avg.</th>
<th>Image Rec.<br/>5 benchmark avg.</th>
</tr>
</thead>
<tbody>
<tr>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>48.5</td>
<td>39.7</td>
<td>34.4</td>
<td>6.6</td>
<td>42.2</td>
<td>24.0</td>
<td>67.5</td>
<td>64.9</td>
<td>50.6</td>
<td>76.0</td>
<td>64.3</td>
<td>63.3</td>
</tr>
<tr>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>54.3</td>
<td>49.8</td>
<td>35.9</td>
<td>14.7</td>
<td>48.8</td>
<td>29.9</td>
<td>73.2</td>
<td>73.3</td>
<td>56.1</td>
<td><b>84.0</b></td>
<td>65.9</td>
<td>65.5</td>
</tr>
<tr>
<td>✓</td>
<td>✓</td>
<td>✗</td>
<td>57.9</td>
<td>49.9</td>
<td>36.2</td>
<td>42.1</td>
<td>48.6</td>
<td>32.3</td>
<td>73.9</td>
<td>74.2</td>
<td>62.9</td>
<td>83.8</td>
<td>67.5</td>
<td>65.0</td>
</tr>
<tr>
<td>✓</td>
<td>✗</td>
<td>✓</td>
<td>56.7</td>
<td>62.9</td>
<td>43.2</td>
<td>15.2</td>
<td>50.1</td>
<td>30.4</td>
<td>74.1</td>
<td><b>76.3</b></td>
<td>58.3</td>
<td>83.7</td>
<td>64.0</td>
<td><b>65.6</b></td>
</tr>
<tr>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td><b>61.2</b></td>
<td><b>63.6</b></td>
<td><b>44.0</b></td>
<td><b>42.2</b></td>
<td><b>50.2</b></td>
<td><b>34.3</b></td>
<td><b>74.6</b></td>
<td><b>76.3</b></td>
<td><b>64.3</b></td>
<td>83.7</td>
<td><b>74.2</b></td>
<td>65.4</td>
</tr>
</tbody>
</table>

**Table 6: Ablation.** We show the impact of individual data components in PLM training. For this ablation, we use a reduced the SFT datamix consists of 4M open-access image and video data. Results are aggregated validation-set performance over selected benchmarks in each category of tasks, details in Appendix A.3.

**Figure 7:** HardQA improves with PLM data.

**Discussion.** First, we observe that stage 2 synthetic data training boosts model performance across the board. Moreover, adding our PLM-STC data further improves a variety of benchmarks, including PLM-STC (+27.4 points), video captioning (+2.4 points), and most importantly, spatial and temporal tasks (+6.8 points). Adding our PLM-FGQA data improves a distinct set of categories for fine-grained activity understanding; PLM-FGQA (+13.1 points), PLM-SGQA (+7.3 points), Fine-grained video tasks (+1.3 points), video hallucination tasks (+3.0 points), and spatial and temporal tasks (+2.2 points). Using our human-annotated data altogether results in the best performance overall. Further in Fig.7, we show that our human-annotated data improves upon HardQA [97, 100, 89, 101, 99, 113, 92], effectively addressing the limitations of synthetic data discussed in §4.2.

## 7 Conclusion

This work presents Perception Language Model (PLM), a fully-reproducible vision-language model to transparently tackle visual perception tasks without distillation of private black-box models. We trained PLM using data from existing open-access datasets and synthetic samples generated by our data engine. We identified gaps in detailed video understanding capabilities that cannot be filled with synthetic data. In response, we collected 2.8M human-labels for fine-grained video question answering and spatio-temporally grounded captioning, and created a new benchmark, PLM-VideoBench, to evaluate these capabilities. We hope our open dataset, benchmark, and models will foster transparent research in visual perception.

<sup>2</sup>3.8M datamix: TextQA 500K, Image QA 2.8M, and Video QA 500K. Each detail can be found in Tab. 9.# Appendix

## Table of Contents

---

<table><tr><td><b>A</b></td><td><b>PLM Training Details</b></td><td><b>12</b></td></tr><tr><td>A.1</td><td>PLM Training Setting . . . . .</td><td>12</td></tr><tr><td>A.2</td><td>PLM Training Datamix . . . . .</td><td>13</td></tr><tr><td>A.3</td><td>Ablation Experiment Details . . . . .</td><td>14</td></tr><tr><td><b>B</b></td><td><b>Synthetic Scaling Experiments</b></td><td><b>14</b></td></tr><tr><td><b>C</b></td><td><b>VLM Benchmark Details</b></td><td><b>16</b></td></tr><tr><td>C.1</td><td>Image Benchmarks . . . . .</td><td>16</td></tr><tr><td>C.2</td><td>Video Benchmarks . . . . .</td><td>17</td></tr><tr><td>C.3</td><td>PLM-VideoBench . . . . .</td><td>17</td></tr><tr><td>C.4</td><td>Evaluation Protocols . . . . .</td><td>18</td></tr><tr><td><b>D</b></td><td><b>Additional PLM-VideoBench Results</b></td><td><b>19</b></td></tr><tr><td><b>E</b></td><td><b>Baseline Implementation Details</b></td><td><b>19</b></td></tr><tr><td><b>F</b></td><td><b>Additional Results</b></td><td><b>20</b></td></tr><tr><td>F.1</td><td>Comparison with LLaMA-3V . . . . .</td><td>20</td></tr><tr><td>F.2</td><td>Image Captioning . . . . .</td><td>20</td></tr><tr><td>F.3</td><td>Image Grounding . . . . .</td><td>21</td></tr><tr><td>F.4</td><td>Long Video Understanding . . . . .</td><td>21</td></tr><tr><td><b>G</b></td><td><b>PLM-FGQA: Fine-grained QA</b></td><td><b>22</b></td></tr><tr><td>G.1</td><td>Annotation process: Data Engine . . . . .</td><td>22</td></tr><tr><td>G.2</td><td>FGQA PLM-VideoBench Construction . . . . .</td><td>27</td></tr><tr><td><b>H</b></td><td><b>PLM-STC Details</b></td><td><b>28</b></td></tr><tr><td>H.1</td><td>Annotation Process . . . . .</td><td>28</td></tr><tr><td>H.2</td><td>PLM-STC Benchmark . . . . .</td><td>30</td></tr><tr><td><b>I</b></td><td><b>Smart Glasses Data</b></td><td><b>30</b></td></tr><tr><td>I.1</td><td>Data collection and annotation . . . . .</td><td>30</td></tr><tr><td>I.2</td><td>SGQA Benchmark . . . . .</td><td>31</td></tr><tr><td><b>J</b></td><td><b>Synthetic Data Engine</b></td><td><b>31</b></td></tr><tr><td><b>K</b></td><td><b>Qualitative Results</b></td><td><b>35</b></td></tr><tr><td><b>L</b></td><td><b>Limitations and Future Work</b></td><td><b>39</b></td></tr><tr><td><b>M</b></td><td><b>Broader Impact</b></td><td><b>39</b></td></tr></table>

---## A PLM Training Details

The diagram illustrates the data sources and tasks for the Perception Language Model (PLM). It is organized into four main sections: Image - Synthetic, Video - Synthetic, Video - Human, and Open Source.

- **Image - Synthetic:**
  - Documents: QA (PDFAcc, UCSF, 301M)
  - Charts: Caption, QA (ArxivCap, 1.8M)
  - Natural Images: Captions (SAIB, Objects365, OpenImages, 15.9M)
  - **Total: 47.8M (Ours)**
- **Video - Synthetic:**
  - Exocentric Videos: Captioning (YT-Temporal, 14M), MCQs (YT-Temporal, 3M), Others (500k)
  - Egocentric Videos: Captions (Ego4D, 180k), QA (Ego4D, 700k)
  - **Total: 18.4M (Ours)**
- **Video - Human:**
  - Fine-grained: Activity QA (11M), How2 QA (1.3M)
  - Spatio-Temporal: PLM-RTLoc (180k), PLM-RCap (180k), PLM-RDCap (177k)
  - **Total: 2.9M (Ours)**
- **Open Source:**
  - Images: Single & Multi, 92 Datasets (Total 5.6M)
  - Videos: QA, MCQs, Caption, 27 Datasets (Total 679k), Grounding (41 k)
  - **Total: 6.52M**

**Total 69.1M PLM Image and Video Data** and **6.52M Existing Data** are combined for the **Perception Language Model (PLM)**.

The PLM is trained on various tasks, each shown with an image/video and a Q&A pair:

- **Image QA:** Q: Which Place is shown in the image? A: A serene lake scene with a monument featuring a man on horseback.
- **Image OCR:** Q: What location does this map represent? A: The map is of Kansai International Airport (KIX), which is located in Japan.
- **Multi-image:** Q: Spot the difference between these two images. A: The color of the pole on the dog's right changes from black to light red.
- **Video QA:** Q: What is the person doing in the video? A: Person is wearing ring in left hand and cutting the onions with a knife.
- **Video Cap.:** Q: Briefly caption the video. A: The video shows a person making avocado sandwich by mashing the avocado and spreading it on bread.
- **PLM-FGQA:** Q: How does a person pour the tea into the mug? A: The person moves the teapot up and down three times while pouring the tea into the mug.
- **PLM-RTLoc:** Q: Localize the event "Person facing towards the camera is dancing." for the shaded region? A: [0, 13]
- **PLM-RCap:** Q: Describe the highlighted subject in the video between frames [0, 13]. A: A person facing towards the camera is dancing.
- **PLM-RDCap:** Q: Provide a dense caption for the shaded region. A: [0, 12] Kid is playing hide and seek and hides behind a tree. [13, 16] The kid reappears.

Figure 8: The figure provides an overview of the datasets used in the paper. PLM is trained with 47.8M synthetic image and 18.4M synthetic video, and 2.9M human-labeled video samples. Our data enables PLM to perform a variety of tasks, including standard tasks like Image, Multi-image, and Video QA, as well as *new video tasks* such as Fine-grained QA (FGQA), Region Temporal Localization (RTLoc), Region Captioning (RCap), and Region Detailed Captioning (RDCap).

In this section, we describe the training details of PLM. In §A.1 we describe exact details of training setting such as hyper-parameters and implementation details. In §A.2 we describe our datamix for both synthetically generated and human-annotated parts.

### A.1 PLM Training Setting

For all three stages, we use AdamW optimizer [128] with weight decay of 0.05 and use FSDP [129] with FlashAttention2 [130] for overall implementation based on PyTorch [131].

**Stage 1 training.** In stage 1, we use a subset of SA-1B [105] paired with detailed captions generated by our data engine (§4.1). We use total 1M samples to train PLM with next token prediction loss, with vision encoder and LLM parameters frozen. This stage is commonly known as *warm-up* stage. We use learning rate  $1 \times 10^{-4}$  for all model scale with global batch size of 512 and  $448 \times 448$  resolution. We use the Perception Encoder [104] L/14 variant for the 1B and 3B PLM models, and the G/14 variant for the 8B PLM model.

**Stage 2 training.** In Stage 2, we train on a total of 72.5M samples. Of these, 66M consist of images and videos with synthetically generated annotations produced by our data engine. The remaining 6.5M samples are a subset of human-annotated images and videos from open-source datasets, which are included in our final datamix described in §A.2. We train with global batch size of 2048, learning rate of  $4 \times 10^{-5}$ , weight decay of 0.05 for the full set of parameters (vision encoder, projector, and LLM). For both image and video input, we use  $448 \times 448$  resolution for each tile/frame, which effectively generate 1024 vision tokens. We apply  $2 \times 2$  spatial average pooling to reduce this to 256. We use dynamic tiling with a thumbnail to support any resolution and aspect ratio, similar to prior work [12], and uniform sampling of video frames after preprocessing the videos to 1 fps. We set the maximum number of tiles/frames to be 16, which results in maximum of  $(16 + 1) \times 256 = 4352$  and  $16 \times 256 = 4096$  vision tokens respectively for images and videos. We train the model with a sequence length of 6144 allowing a maximum of 2048 tokens for the text modality.

**Stage 3 training.** In stage 3, we use total of 19.1M high-quality datamix spanning over multiple image, video, and text modalities. We describe this datamix in §A.2. In this stage, we use global batch size of 1024, learning rate of  $1 \times 10^{-5}$  for 8B and  $4 \times 10^{-5}$  for 1B and 3B PLM models. Wetrain the full set of parameters for all scales. Similar to stage 2, we adapt dynamic tiling and uniform frame sampling for up to 36 tiles for image and 32 frames for video, with  $2 \times 2$  spatial average pooling, which generates  $(36 + 1) \times 256 = 9472$  vision tokens for image and  $32 \times 256 = 8192$  vision tokens for video. For all modalities, we use 11264 maximum training sequence length.

## A.2 PLM Training Datamix

Table 9 presents the full data mix used across all training stages apart from our manually collected data in §5. This contains annotations from existing public datasets as well as synthetically generated data (see §4). We filter and include a wide variety of existing datasets spanning across images (captioning, QA, grounding), videos (captioning, QA, temporal localization, region captioning and dense captioning) and text-only datasets to preserve the text-instruction following capabilities of our model. Most importantly, we filter out *every* dataset that contains annotations generated by proprietary models. Table 7 and Table 8 shows the exact number of samples for each datasets in Stage 2 and Stage 3 respectively. Marjory of the data in stage 2 are synthetic, with a focus on captioning samples, since they carry the dense information about the image or video. In stage 3, we have one third of the data, mostly focusing on human annotated samples, covering a large variety of tasks.

<table border="1">
<thead>
<tr>
<th>Dataset</th>
<th>Num Samples</th>
<th>Type</th>
<th>Dataset</th>
<th>Num Samples</th>
<th>Type</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="3"><i>Image Synthetic</i></td>
<td colspan="3"><i>Image Synthetic</i></td>
</tr>
<tr>
<td>PDFAcc (QA) [132]</td>
<td>12M</td>
<td>QA</td>
<td>PDFAcc (QA) [132]</td>
<td>2M</td>
<td>QA</td>
</tr>
<tr>
<td>PDFAcc (Cap) [132]</td>
<td>12M</td>
<td>Cap.</td>
<td>ArxivCap [134]</td>
<td>1.5M</td>
<td>Cap./QA</td>
</tr>
<tr>
<td>UCSF [133]</td>
<td>6M</td>
<td>QA</td>
<td>SA1B [105]</td>
<td>800K</td>
<td>Cap.</td>
</tr>
<tr>
<td>ArxivCap [134]</td>
<td>1.8M</td>
<td>Cap./QA</td>
<td>Object365 [135]</td>
<td>300K</td>
<td>Cap.</td>
</tr>
<tr>
<td>SA1B [105]</td>
<td>10M</td>
<td>Cap.</td>
<td>OpenImages [136]</td>
<td>300K</td>
<td>Cap.</td>
</tr>
<tr>
<td>Object365 [135]</td>
<td>3.5M</td>
<td>Cap.</td>
<td>DocVQA [53]</td>
<td>100K</td>
<td>QA</td>
</tr>
<tr>
<td>OpenImages [136]</td>
<td>1.8M</td>
<td>Cap.</td>
<td>InfographicVQA [56]</td>
<td>50K</td>
<td>QA</td>
</tr>
<tr>
<td>DocVQA [53]</td>
<td>50K</td>
<td>QA</td>
<td>PixmoCap [11]</td>
<td>500K</td>
<td>Cap</td>
</tr>
<tr>
<td>InfographicVQA [56]</td>
<td>20K</td>
<td>QA</td>
<td colspan="3"><i>Video Synthetic</i></td>
</tr>
<tr>
<td>PixmoCap [11]</td>
<td>600K</td>
<td>Cap</td>
<td>YT-1B (QA) [137]</td>
<td>300K</td>
<td>MCQA</td>
</tr>
<tr>
<td colspan="3"><i>Video Synthetic</i></td>
<td>Ego4D (Cap.) [115]</td>
<td>180K</td>
<td>Cap.</td>
</tr>
<tr>
<td>YT-1B (Cap.) [137]</td>
<td>14M</td>
<td>Cap.</td>
<td>Ego4D (QA) [115]</td>
<td>700K</td>
<td>QA</td>
</tr>
<tr>
<td>YT-1B (QA) [137]</td>
<td>3M</td>
<td>MCQA</td>
<td>Spoken Moments [138]</td>
<td>449K</td>
<td>Cap.</td>
</tr>
<tr>
<td>Ego4D (Cap.) [115]</td>
<td>180K</td>
<td>Cap.</td>
<td>Charades [139]</td>
<td>8K</td>
<td>Cap.</td>
</tr>
<tr>
<td>Ego4D (QA) [115]</td>
<td>700K</td>
<td>QA</td>
<td>Kinetics710 [121]</td>
<td>40K</td>
<td>Cap.</td>
</tr>
<tr>
<td>Spoken Moments [138]</td>
<td>449K</td>
<td>Cap.</td>
<td>DiDeMo [140]</td>
<td>7.5K</td>
<td>Cap.</td>
</tr>
<tr>
<td>Charades [139]</td>
<td>8K</td>
<td>Cap.</td>
<td colspan="3"><i>Text Synthetic</i></td>
</tr>
<tr>
<td>Kinetics710 [121]</td>
<td>40K</td>
<td>Cap.</td>
<td>NaturalReasoning [141]</td>
<td>1M</td>
<td>QA</td>
</tr>
<tr>
<td>DiDeMo [140]</td>
<td>7.5K</td>
<td>Cap.</td>
<td colspan="3"><i>Human Annotated</i></td>
</tr>
<tr>
<td colspan="3"><i>Text Synthetic</i></td>
<td>Image QA [9]</td>
<td>2.8M</td>
<td>QA</td>
</tr>
<tr>
<td>NaturalReasoning [141]</td>
<td>1M</td>
<td>QA</td>
<td>Image Cap [9]</td>
<td>36K</td>
<td>QA</td>
</tr>
<tr>
<td colspan="3"><i>Human Annotated</i></td>
<td>Image Grnd. [9]</td>
<td>1.4M</td>
<td>QA</td>
</tr>
<tr>
<td>Image QA [9]</td>
<td>2.8M</td>
<td>QA</td>
<td>Image Misc. [9]</td>
<td>1.4M</td>
<td>QA</td>
</tr>
<tr>
<td>Video QA [9]</td>
<td>570K</td>
<td>QA</td>
<td>Video QA [9]</td>
<td>570K</td>
<td>QA</td>
</tr>
<tr>
<td>Video TL [9]</td>
<td>16K</td>
<td>Temp. Loc.</td>
<td>Video Cap. [9]</td>
<td>315K</td>
<td>QA</td>
</tr>
<tr>
<td>Video Dense Cap. [9]</td>
<td>10K</td>
<td>Dense Cap.</td>
<td>Video TL [9]</td>
<td>16K</td>
<td>TL</td>
</tr>
<tr>
<td>Text QA [9]</td>
<td>2M</td>
<td>Mix</td>
<td>Video Dense Cap. [9]</td>
<td>10K</td>
<td>DCap.</td>
</tr>
<tr>
<td>Total</td>
<td>72.5M</td>
<td></td>
<td>Video Region Captioning [9]</td>
<td>15K</td>
<td>Cap.</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td>Text QA [9]</td>
<td>1.5M</td>
<td>Mix</td>
</tr>
</tbody>
</table>

Table 7: PLM Stage 2 training data mix.

<table border="1">
<tbody>
<tr>
<td colspan="3"><i>Human Annotated (Our)</i></td>
</tr>
<tr>
<td>PLM FGQA</td>
<td>2.4M</td>
<td>QA</td>
</tr>
<tr>
<td>PLM STC</td>
<td>476K</td>
<td>Cap./TL</td>
</tr>
<tr>
<td>Total</td>
<td>19.1M</td>
<td></td>
</tr>
</tbody>
</table>

Table 8: PLM Stage 3 training data mix.<table border="1">
<thead>
<tr>
<th colspan="2">Image QA</th>
<th colspan="2">M4 Instruct (Filtered)</th>
<th colspan="2">Grounding</th>
<th colspan="2">Video Temporal Loc.</th>
</tr>
<tr>
<th>Dataset</th>
<th>Size</th>
<th>Dataset</th>
<th>Size</th>
<th>Dataset</th>
<th>Size</th>
<th>Dataset</th>
<th>Size</th>
</tr>
</thead>
<tbody>
<tr>
<td>DVQA [142]</td>
<td>222222</td>
<td>STAR [72]</td>
<td>3032</td>
<td>VisualGenome [66]</td>
<td>154792</td>
<td>HiREST [199]</td>
<td>7919</td>
</tr>
<tr>
<td>PlotQA [143]</td>
<td>157070</td>
<td>NEXT-QA [69]</td>
<td>3870</td>
<td>FLickr Entities [196]</td>
<td>296332</td>
<td>Charades [139]</td>
<td>7566</td>
</tr>
<tr>
<td>MapQA [144]</td>
<td>42761</td>
<td>VISION [180]</td>
<td>9900</td>
<td>DCI (Region Caption) [193]</td>
<td>304912</td>
<td>DiDeMo [140]</td>
<td>435</td>
</tr>
<tr>
<td>OCRVQA [145]</td>
<td>167646</td>
<td>Flintstones/SV [181]</td>
<td>22341</td>
<td>RefCOCO/g/* [197]</td>
<td>212923</td>
<td><b>Total</b></td>
<td><b>15920</b></td>
</tr>
<tr>
<td>Localized Narratives [146]</td>
<td>199998</td>
<td>ImageCoDe [182]</td>
<td>16594</td>
<td>VCR [60]</td>
<td>85577</td>
<td></td>
<td></td>
</tr>
<tr>
<td>FigureQA [147]</td>
<td>119999</td>
<td>VizWiz [40]</td>
<td>4900</td>
<td><b>Total</b></td>
<td><b>1398690</b></td>
<td></td>
<td></td>
</tr>
<tr>
<td>Hateful Memes [148]</td>
<td>9713</td>
<td>MIT-States (State Coherence) [183]</td>
<td>1900</td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>CLEVR [149]</td>
<td>73181</td>
<td>MIT-States (Prop. Coherence) [183]</td>
<td>1900</td>
<td colspan="2">Image Synth.</td>
<td colspan="2">Video Region Captioning</td>
</tr>
<tr>
<td>CLEVR v1.0 [149]</td>
<td>70000</td>
<td>WebQA [184]</td>
<td>9338</td>
<td>Dataset</td>
<td>Size</td>
<td>Dataset</td>
<td>Size</td>
</tr>
<tr>
<td>IconQA [150]</td>
<td>116514</td>
<td>Birds-to-Words [185]</td>
<td>14281</td>
<td>DocVQA [53]</td>
<td>50170</td>
<td>HC-STVG [200]</td>
<td>10131</td>
</tr>
<tr>
<td>TextVQA [112]</td>
<td>21953</td>
<td>AESOP [186]</td>
<td>6915</td>
<td>InfographicVQA [56]</td>
<td>21660</td>
<td>ViDLN (UVO subset) [123]</td>
<td>5296</td>
</tr>
<tr>
<td>GeomVerse [151]</td>
<td>11162</td>
<td>RecipeQA (Img. Coherence) [187]</td>
<td>8699</td>
<td>PDFAcc (Cap.) [132]</td>
<td>12024670</td>
<td><b>Total</b></td>
<td><b>15427</b></td>
</tr>
<tr>
<td>RobuT (wikisql) [152]</td>
<td>80757</td>
<td>CLEVR-Change [188]</td>
<td>3885</td>
<td>PDFAcc (QA) [132]</td>
<td>12024670</td>
<td colspan="2">Video Dense Cap.</td>
</tr>
<tr>
<td>WebSight [153]</td>
<td>10000</td>
<td>IEdit [189]</td>
<td>3456</td>
<td>UCSF [133]</td>
<td>5953490</td>
<td>Dataset</td>
<td>Size</td>
</tr>
<tr>
<td>VisualFW [154]</td>
<td>15961</td>
<td>ChartQA [109]</td>
<td>45820</td>
<td>ArxivCap [134]</td>
<td>1859680</td>
<td>ActivityNet [125]</td>
<td>8859</td>
</tr>
<tr>
<td>TallyQA [155]</td>
<td>100050</td>
<td>DocVQA [53]</td>
<td>69562</td>
<td>SAIB [105]</td>
<td>9834573</td>
<td>YouCook [83]</td>
<td>1039</td>
</tr>
<tr>
<td>Robut (WTQ) [152]</td>
<td>42495</td>
<td>InfographicVQA [56]</td>
<td>32661</td>
<td>Object365 [135]</td>
<td>3484584</td>
<td><b>Total</b></td>
<td><b>9898</b></td>
</tr>
<tr>
<td>DaTikz [156]</td>
<td>47974</td>
<td>TextVQA [112]</td>
<td>69170</td>
<td>OpenImages [136]</td>
<td>1740864</td>
<td></td>
<td></td>
</tr>
<tr>
<td>CocoQA [157]</td>
<td>46287</td>
<td>TextCaps [167]</td>
<td>21324</td>
<td>PixmoCap [11]</td>
<td>584650</td>
<td></td>
<td></td>
</tr>
<tr>
<td>ChartQA [109]</td>
<td>27395</td>
<td>VisualMRC [171]</td>
<td>24456</td>
<td></td>
<td></td>
<td colspan="2">Video Synth.</td>
</tr>
<tr>
<td>VQAv2 [111]</td>
<td>82772</td>
<td>WTQ [190]</td>
<td>16885</td>
<td><b>Total</b></td>
<td><b>47579011</b></td>
<td>Dataset</td>
<td>Size</td>
</tr>
<tr>
<td>Chart2Text [158]</td>
<td>35946</td>
<td></td>
<td></td>
<td colspan="2">Video QA</td>
<td>Spoken Moments [138]</td>
<td>449044</td>
</tr>
<tr>
<td>VisText [159]</td>
<td>35995</td>
<td>HME100k [191]</td>
<td>74492</td>
<td>Dataset</td>
<td>Size</td>
<td>Charades [139]</td>
<td>7919</td>
</tr>
<tr>
<td>FinQA [160]</td>
<td>5276</td>
<td>chrome_writing [163]</td>
<td>8825</td>
<td>EgoQA [119]</td>
<td>7813</td>
<td>Kinetics710 [121]</td>
<td>39949</td>
</tr>
<tr>
<td>DocVQA [53]</td>
<td>12089</td>
<td>OK-VQA [110]</td>
<td>27536</td>
<td>NEXT-QA (instruct) [69]</td>
<td>34114</td>
<td>DiDeMo [140]</td>
<td>7566</td>
</tr>
<tr>
<td>STVQA [161]</td>
<td>18684</td>
<td>VQA-RAD [172]</td>
<td>1793</td>
<td>NEXT-QA (MCQ) [69]</td>
<td>34114</td>
<td>Ego4D (Cap.) [115]</td>
<td>183029</td>
</tr>
<tr>
<td>TAT-QA [162]</td>
<td>2199</td>
<td></td>
<td></td>
<td>PerceptionTest [71]</td>
<td>2403</td>
<td>Ego4D (QA) [115]</td>
<td>703935</td>
</tr>
<tr>
<td>RenderedText [163]</td>
<td>10435</td>
<td><b>Total</b></td>
<td><b>2796145</b></td>
<td>ActivityNetQA [76]</td>
<td>23530</td>
<td>YT-1B (Cap.) [137]</td>
<td>14792983</td>
</tr>
<tr>
<td>RAVEN [164]</td>
<td>31418</td>
<td></td>
<td></td>
<td>VideoInstruct (human) [20]</td>
<td>25803</td>
<td>YT-1B (QA) [137]</td>
<td>3383670</td>
</tr>
<tr>
<td>IAM [165]</td>
<td>7549</td>
<td colspan="2">Image Cap.</td>
<td>CLEVRER (MC) [120]</td>
<td>42620</td>
<td><b>Total</b></td>
<td><b>19568095</b></td>
</tr>
<tr>
<td>A-OKVQA [39]</td>
<td>17720</td>
<td>Dataset</td>
<td>Size</td>
<td>CLEVRER (QA) [120]</td>
<td>40000</td>
<td></td>
<td></td>
</tr>
<tr>
<td>TabMWP [166]</td>
<td>45439</td>
<td>DOCCI [192]</td>
<td>13362</td>
<td>Kinetics710 [121]</td>
<td>39949</td>
<td>Dataset</td>
<td>Size</td>
</tr>
<tr>
<td>CocoQA [157]</td>
<td>9009</td>
<td>DCI [193]</td>
<td>7599</td>
<td>SSv2 (classification) [122]</td>
<td>40000</td>
<td>no_robots [201]</td>
<td>9485</td>
</tr>
<tr>
<td>TextCaps [167]</td>
<td>21953</td>
<td>Altogether [194]</td>
<td>15166</td>
<td>ViDLN [123]</td>
<td>43126</td>
<td>MathQA [202]</td>
<td>29837</td>
</tr>
<tr>
<td>Screen2Words [168]</td>
<td>16713</td>
<td></td>
<td></td>
<td>ViDLN (QA) [123]</td>
<td>75090</td>
<td>LIMA [203]</td>
<td>1030</td>
</tr>
<tr>
<td>VSR [169]</td>
<td>2157</td>
<td colspan="2">Image Misc.</td>
<td>HowQA [8]</td>
<td>45731</td>
<td>GSM8k (socratic) [204]</td>
<td>7473</td>
</tr>
<tr>
<td>TQA [170]</td>
<td>9742</td>
<td>AI2d [55]</td>
<td>12413</td>
<td>STAR [72]</td>
<td>35297</td>
<td>GSM8k [204]</td>
<td>7473</td>
</tr>
<tr>
<td>Robut (SQA) [152]</td>
<td>12769</td>
<td>COCO cap. [49]</td>
<td>41413</td>
<td>Memento [198]</td>
<td>40060</td>
<td>FLAN [205]</td>
<td>1386050</td>
</tr>
<tr>
<td>VisualMRC [171]</td>
<td>3027</td>
<td>GQA-Balanced [195]</td>
<td>943000</td>
<td>Memento-MultiImage [198]</td>
<td>40060</td>
<td>Dolly15k [206]</td>
<td>15011</td>
</tr>
<tr>
<td>ScienceQA [61]</td>
<td>9947</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>Magpie Pro (MT) [207]</td>
<td>300000</td>
</tr>
<tr>
<td>VQA-RAD [172]</td>
<td>313</td>
<td><b>Total</b></td>
<td><b>1369526</b></td>
<td></td>
<td></td>
<td>Magpie Pro [207]</td>
<td>300000</td>
</tr>
<tr>
<td>InfographicVQA [56]</td>
<td>2118</td>
<td></td>
<td></td>
<td colspan="2">Video Cap.</td>
<td><b>Total</b></td>
<td><b>2056359</b></td>
</tr>
<tr>
<td>Hitab [173]</td>
<td>4995</td>
<td></td>
<td></td>
<td>VATex (en caption) [84]</td>
<td>259910</td>
<td></td>
<td></td>
</tr>
<tr>
<td>AI2D [55]</td>
<td>4863</td>
<td></td>
<td></td>
<td>Charades (caption) [139]</td>
<td>11593</td>
<td></td>
<td></td>
</tr>
<tr>
<td>Inter-GPS [174]</td>
<td>2555</td>
<td></td>
<td></td>
<td>ActivityNet (captions) [125]</td>
<td>33375</td>
<td></td>
<td></td>
</tr>
<tr>
<td>diagram_image_to_text [175]</td>
<td>70939</td>
<td></td>
<td></td>
<td>YouCook2 [83]</td>
<td>10537</td>
<td></td>
<td></td>
</tr>
<tr>
<td>MIMIC-IT (CGD) [176]</td>
<td>15233</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>Multihier [177]</td>
<td>136799</td>
<td></td>
<td></td>
<td><b>Total</b></td>
<td><b>315215</b></td>
<td></td>
<td></td>
</tr>
<tr>
<td>NLVR2 [178]</td>
<td>56081</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>RAVEN (Multi-image) [164]</td>
<td>19340</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
</tbody>
</table>

Table 9: **PLM training datamix**. Our mix includes synthetic and manually annotated data across a combination of **image data** (QA, captioning, OCR, Visual grounding), **video data** (captioning, grounded captioning, dense captioning, temporal localization) and **text-only data**. Importantly, all data is publicly accessible, and *not* generated by proprietary models.

### A.3 Ablation Experiment Details

We provide additional details about the ablation experiment in §6.5. We report benchmark average scores across 5 categories, along with the average across all of them. We select a representative set of benchmarks from the full set of image and video benchmarks in §6.2 and §6.3 that report comparable scores so the average results are meaningful. For Video captioning we select Dream 1K and report the LLM-judge score with Llama3.3 70B as judge. for Short Video QA, and Finegrained QA, we select benchmarks that report MCQ accuracy (and exclude open-ended QA). For Hallucination, we include both benchmarks. For Spatial and Temporal tasks, we select BLINK, CVBench, VSR, and Charades-STA. For Image Perception, we choose SEED, MMMU, VQAv2, OK-VQA, and VizWiz. We train the ablation setup of SFT with the exactly matching hyperparameters as our final run; only difference is the size of the SFT datamix.

## B Synthetic Scaling Experiments

In this section we provide additional results to the synthetic scaling experiments in §4.2. We report aggregate benchmark accuracies across three categories — Video QA, OCR QA and Image QA — by selecting representative benchmarks from each category. For VideoQA, these are STAR [72], EgoSchema [90], MVBench [70], VideoMME [75] and PerceptionTest [71]; For OCR QA, these are ChartQA [109], DocVQA [53], InfographicsQA [56], TextVQA [112] and OCRBench [57]; and for Natural Image QA, these are RealworldQA [45], OKVQA [110], VQAv2 [111], and VizWiz [40].

**Scaling with encoder size.** After investigating the impact of the LLM decoder in Fig. 2, we examine the impact of increasing the vision encoder size from 300M (PE Large) to 2B (PE Giant) for each language model scale next. In Fig. 9, we overlay the new power-law with the 2B vision encoder (black dashed) line onto the 300M (red dashed) line. Notably, we find that the larger vision encoder (300M → 2B) leads to greater scaling trend on video QA benchmarks. Quantitatively, the power lawFigure 9: **Scaling with encoder size.** Scaling trends of PE-G vs. PE-L vision encoders. Larger encoders scale better in Video QA tasks while similar scaling in OCR and Natural QA is seen.

fit has improved from  $-0.15$  to  $-0.19$ . The two lines intersect around 8B scale with PE-G, proving that 8B and larger PLM will benefit more with larger vision encoder. We use PE-L for 1B and 3B LLM scale and PE-G for 8B scale by default.

Figure 10: **Scaling with input size.** Scaling trends of training with 16 tiles/frames vs. 8 tiles/frames. Higher input size scales better in Video QA and OCR QA tasks while similar trend is seen for Natural QA.

**Scaling with input size.** In Fig. 10, we show the impact of increasing the input size to VLM through higher image resolution and more video frames. In this setting, each scale of PLM trains with *dynamic tiling* for image input and *uniform sampling* for video input with maximum 8 or 16 tiles/frames per sample. In each plot, the average error of PLM trained with 16 tiles/frames are plotted. All models use  $2 \times 2$  spatial average pooling before input to LLM, and each tile/frame has  $448 \times 448$  resolution. Similar to Fig. 2, we show power law fit with a **black** dashed line, and compare to 8 tiles/frames training denoted with **red** dashed line. Notably, we find out that on Video QA and OCR QA benchmarks, PLM shows better scalability with training with higher input size. This means *with the same FLOP counts at  $10^{13}$ , training with 16 frames makes 2.0 points of metric error lower than 8 frames counterpart (32.2 vs 30.2)*. Similar trends are observed with OCR QA going from 8 tiles max. to 16 tiles max. Notably, higher resolution did not make a difference for Natural QA tasks. We chose the 16 max-tiles and frames to be our final training setting for stage 2 PLM.

In Fig. 11, we show the breakdown of the scaling trend shown in §4.2. “H” stands for *human only* (i.e., no synthetic) baseline. From the breakdown, the most notable point is the scalability in OCR, Chart, Document QA tasks. In each benchmark, synthetic data makes more than 10 points of improvement on every model scale, compared to “no synthetic” baselines. Moreover, there is no sign of saturation; the performance will most likely improve with more synthetic data. We hypothesize that OCR, Chart, Document QA tasks reduce to “translation” task — a set of pixels has one-to-one mapping to text space. Remaining tasks exhibit clean power-law relationship between metric error and FLOPs. The last plot shows scaling trend on average over all benchmarks, which shows a close power-law relationship.Figure 11: **Synthetic Scaling Plots**. Relationship between Average Error and training compute (in floating-point operations) for various 1B, 3B, 8B PLM with L14 vision encoder. Each plot reports the individual error in VideoMME [75], STAR [72], EgoSchema [90], How2QA [8], MVBench [70], Perception-Test [71], ChartQA [109], DocVQA [53], InfoVQA [56], OCRBench [57], RealworldQA [45], OKVQA [110], VQAv2 [111], VizWiz [40], and TextVQA [112]. Finally, we report Avg. All, which average over all the metrics.

## C VLM Benchmark Details

In this section, we provide details about all the image and video benchmarks considered in §6 including composition and evaluation metrics for image benchmarks (§C.1), video benchmarks (§C.2) and our PLM–VideoBench (§C.3). We also describe evaluation protocol for all these benchmarks including inference parameters and prompts (§C.4). Pointers to evaluation code are linked where available.

### C.1 Image Benchmarks

**Image captioning** We evaluate on single image captioning and grounded image captioning benchmarks like COCO [49], nocaps [50] and Flickr [51]. We report CIDEr as the evaluation metric.

**Perception and reasoning** We evaluate on broad, general purpose VQA benchmarks like MMMU [37], VQAv2 [111], MMBench [38], OK-VQA [39], VizWiz [40] as well as hard perception benchmarks like BLINK [44], CV-Bench [19], RealWorldQA [45], and VSR [127]. For all MCQ benchmarks, we report accuracy of selecting the correct option.**Charts, diagrams and documents** We evaluate on benchmarks for reasoning over various types of charts, graphs, diagrams, infographics etc. Specifically, DocVQA [53], ChartQA [54], TextVQA [52], InfographicsVQA [56], AI2D [55], OCRBench [57], and SEED [58]. We report accuracy of selecting the correct option.

**Image Hallucination** Finally, we evaluate on benchmarks that evaluate robustness of models to hallucinated details in questions such as HallusionBench [67] and POPE [68]. For HallusionBench we report the *aAcc* metric (code) which accounts for correctness and consistency using an LLM judge.

## C.2 Video Benchmarks

**Video captioning** We evaluate on short-video captioning benchmarks, namely YouCook2 [83] and VATEX [84] as well as recent detailed video captioning benchmarks — DREAM-1k [86] and AuroraCap-VDC [87]. For YouCook2 and VATEX, we report CIDEr score [208]. For DREAM-1k we report AutoDQ F1-score (code) and for AuroraCap-VDC we report the VDC accuracy (code) following the author’s proposed metric.

**Short video QA** We evaluate on multiple-choice (MCQ) benchmarks such as How2QA [8], NExt-QA [69], PerceptionTest [71], STAR [72], TGIF-QA [73], TVQA [74], Video-MME [75] and TVBench [80]. We report accuracy of selecting the correct option. We also evaluate on open-ended question answering benchmarks (w/o options) such as ActivityNet-QA [76] (code), MMBench-Video [79] (code) and VCGBench-Diverse [22]. We report LLM-judge scores/accuracies for these benchmarks. For VCGBench-Diverse, we report the average of 5 LLM-judge scores (code).

**Long video QA** We evaluate on popular long-video benchmarks such as EgoSchema [90], LVBench [92], LongVideoBench [94] and MLVU [96]. We report accuracy of selecting the correct option.

**Fine-grained video QA** We evaluate on benchmarks for fine-grained spatial, temporal and detail reasoning in videos such as TemporalBench [99], TOMATO [100], MotionBench [101], TempCompas [102] and CG-Bench [97]. We report accuracy of selecting the correct option. For TemporalBench, we report the *multi-binary accuracy* (MBAcc) (code) proposed by the authors to reduce bias in evaluation.

**Hallucination** We evaluate on benchmarks that evaluate robustness of models to hallucinated details in questions such as VideoHallucer [88] and EventHallusion [89]. We report accuracy of selecting the correct option.

## C.3 PLM-VideoBench

We evaluate on our suite of benchmarks for fine-grained and spatio-temporal reasoning in videos. These include:

**Fine-grained QA (FGQA)** We report multi-binary accuracy (MBAcc) following prior work [99]. In short, this entails presenting the model multiple independent, binary-choice questions about the same video (in our case, three questions) and requiring the model to gets all of them correct, to count towards accuracy. This sets a higher bar for models, and combats bias in multiple-choice question benchmarks that prior work identifies.

**SmartGlasses-QA (SGQA)** We report LLM-judge accuracy of the predicted answer compared to the ground truth answer. We follow existing LLM judge prompts from ActivityNetQA (code). The prompt is repeated below for completeness.

**Video Region Captioning (PLM-RCap)** We use an LLM-judge to generate the similarity scores between predicted and ground truth captions. The prompt is below.

**Dense Video Region Captioning (PLM-RDCap)** We adapt the SODA metric [126] from dense video captioning literature for this task. To compute this metric, we use the same LLM-judge fromabove to generate the pairwise similarity scores between predicted and ground truth captions, which is then fed to the standard metric computation routine.

**Region Temporal Localization (PLM-RTLoc)** We report standard temporal localization metrics, namely Mean Recall@1, averaged over a range of IoU thresholds [0.3, 0.5, 0.7, 0.9].

## C.4 Evaluation Protocols

**Common evaluation protocol.** For video benchmark evaluations, we sample 32 frames uniformly from the full video unless otherwise specified. For uniformity and consistency across benchmarks, we implement all LLM-judge evaluations using LLama3.3-70B-Instruct [13], following LLM judge prompts from popular evaluation frameworks [209, 210] where available. Outputs from all models are generated via greedy sampling (temperature 0).

### SG-QA judge prompt

You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here’s how you can accomplish the task:

#### ##INSTRUCTIONS:

- - Focus on the meaningful match between the predicted answer and the correct answer.
- - Consider synonyms or paraphrases as valid matches.
- - Evaluate the correctness of the prediction compared to the answer.

Please evaluate the following video-based question-answer pair:

Question: [question]

Correct Answer: [target]

Predicted Answer: [candidate]

Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING. DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. For example, your response should look like this: {"pred": "yes", "score": 4.8}.

### PLM-RCap judge prompt

Your task is to compare a given pair of captions and provide a single score indicating how correct the pred is compared to GT, on a scale from 0 to 10. Focus on meaning and context, not exact word matches. Penalize missing and incorrect information, with lower scores for more significant errors. High scores require accurate conveyance of all key GT information. Respond with only the score, starting your response with the number and including no additional text. Output format: [score].

**PLM-VideoBench inference prompts.** Table 10 contains example inference prompt examples for each PLM-VideoBench task. Note that some variation exists between instances in the benchmark. For example, for RCap a prompt may be “What is happening to the subject in the region highlighted by the red rectangle ...” instead of “Give a detailed description of the events occurring in the region marked by the red rectangle ...”, however they convey the same underlying instruction and information.

Proprietary models like GPT-4o and Gemini require more careful prompting to ensure that the output formatting is respected. For example, we append instructions to prevent model hallucinations (*e.g.*, “You must use these frames to answer the question; do not rely on any external knowledge or commonsense.”), to prevent refusals to answer (*e.g.*, “Even if the information in these separate frames is not enough to answer the question, please try your best to guess an answer which you think would be the most possible one based on the question. Do not generate answer such as *not possible to determine*”) and in-context examples to help guide the model towards the correct output format. Model- and benchmark-specific inference prompts will be released along with our code for full reproducibility.<table border="1">
<thead>
<tr>
<th>Task</th>
<th>Prompt</th>
</tr>
</thead>
<tbody>
<tr>
<td>FGQA</td>
<td>Question: [question] \n Options: \n (A) [option1] \n (B) [option2] \n Only give the best option.</td>
</tr>
<tr>
<td>SGQA</td>
<td>The following question is asked by the camera wearer at the end of the video. Provide a detailed answer even if unsure. Try to answer in around 20-30 words. Now answer the following question based on the video content: [question]</td>
</tr>
<tr>
<td>RDCap</td>
<td>Create a dense caption of the subject’s actions within the red rectangles, including action frames ids and brief descriptions. For each item use the format [start, end]: [description] separated by a newline, where start and end are frame numbers between 0 and 31 in this 32 frame video.</td>
</tr>
<tr>
<td>RCap</td>
<td>Give a detailed description of the events occurring in the region marked by the red rectangle within frames ([start frame], [end frame]) in this 32 frame video</td>
</tr>
<tr>
<td>RTLoc</td>
<td>Given the region marked by the red rectangle in the video, please provide the start and end frame of when '[event]' happens. Use the format (start, end), where start and end are frame numbers between 0 and 31 in this 32 frame video.</td>
</tr>
</tbody>
</table>

Table 10: **PLM-VideoBench task prompts**. Items in square brackets are placeholders filled in for each benchmark instance.

## D Additional PLM-VideoBench Results

We present benchmarking results across all model scales (1B, 3B, 8B) in Table 11, to supplement the 8B model results in the main paper (Table 5). Our approach consistently outperforms baselines across all scales, including proprietary models whose model scale is unknown.

<table border="1">
<thead>
<tr>
<th>Model</th>
<th>FGQA<br/><i>MBAcc</i></th>
<th>SGQA<br/><i>acc†</i></th>
<th>RDCap<br/><i>SODA†</i></th>
<th>RCap<br/><i>score†</i></th>
<th>RTLoc<br/><i>meanR</i></th>
<th>Avg.</th>
</tr>
</thead>
<tbody>
<tr>
<td>Human perf.</td>
<td>90.9</td>
<td>67.9</td>
<td>66.6</td>
<td>53.9</td>
<td>67.8</td>
<td>70.9</td>
</tr>
<tr>
<td colspan="7"><b>Proprietary</b></td>
</tr>
<tr>
<td>GPT-4o [33]</td>
<td>61.2</td>
<td>63.7</td>
<td>20.9</td>
<td>35.7</td>
<td>33.1</td>
<td>51.6</td>
</tr>
<tr>
<td>Gemini 1.5 Pro [35]</td>
<td>57.1</td>
<td>49.9</td>
<td>14.4</td>
<td>33.1</td>
<td>27.6</td>
<td>44.0</td>
</tr>
<tr>
<td>Gemini 2.0 Flash [35]</td>
<td>58.7</td>
<td>44.8</td>
<td>13.2</td>
<td>30.9</td>
<td>27.6</td>
<td>42.5</td>
</tr>
<tr>
<td colspan="7"><b>1B scale</b></td>
</tr>
<tr>
<td>Qwen2VL-2B [30]</td>
<td>39.0</td>
<td>38.5</td>
<td>0.9</td>
<td>18.1</td>
<td>10.8</td>
<td>29.1</td>
</tr>
<tr>
<td>InternVL2-1B [10]</td>
<td>35.8</td>
<td>28.9</td>
<td>0.3</td>
<td>17.2</td>
<td>2.7</td>
<td>23.8</td>
</tr>
<tr>
<td>InternVL2.5-1B [10]</td>
<td>42.3</td>
<td>39.6</td>
<td>6.7</td>
<td>23.6</td>
<td>1.6</td>
<td>30.8</td>
</tr>
<tr>
<td>PLM-1B</td>
<td><b>57.6</b></td>
<td><b>40.9</b></td>
<td><b>50.3</b></td>
<td><b>40.9</b></td>
<td><b>57.7</b></td>
<td>49.4</td>
</tr>
<tr>
<td colspan="7"><b>3B scale</b></td>
</tr>
<tr>
<td>Qwen2.5 VL-3B [106]</td>
<td>43.7</td>
<td>45.1</td>
<td>0.3</td>
<td>17.2</td>
<td>13.9</td>
<td>33.1</td>
</tr>
<tr>
<td>InternVL2-4B [10]</td>
<td>43.2</td>
<td>41.7</td>
<td>0.5</td>
<td>19.9</td>
<td>9.6</td>
<td>30.3</td>
</tr>
<tr>
<td>InternVL2.5-4B [10]</td>
<td>50.0</td>
<td><b>49.2</b></td>
<td>4.9</td>
<td>25.9</td>
<td>15.4</td>
<td>35.3</td>
</tr>
<tr>
<td>PLM-3B</td>
<td><b>67.1</b></td>
<td>38.8</td>
<td><b>53.1</b></td>
<td><b>45.0</b></td>
<td><b>58.2</b></td>
<td>53.0</td>
</tr>
<tr>
<td colspan="7"><b>8B scale</b></td>
</tr>
<tr>
<td>LLaVA-OV-7B [28]</td>
<td>40.2</td>
<td>41.5</td>
<td>4.7</td>
<td>24.4</td>
<td>13.9</td>
<td>32.0</td>
</tr>
<tr>
<td>Qwen2VL-7B [30]</td>
<td>49.2</td>
<td>44.5</td>
<td>4.1</td>
<td>17.6</td>
<td>15.1</td>
<td>35.3</td>
</tr>
<tr>
<td>Qwen2.5VL-7B [106]</td>
<td>49.8</td>
<td>43.0</td>
<td>2.5</td>
<td>21.5</td>
<td>10.7</td>
<td>34.8</td>
</tr>
<tr>
<td>InternVL2-8B [10]</td>
<td>47.7</td>
<td>45.9</td>
<td>1.2</td>
<td>21.5</td>
<td>11.6</td>
<td>35.0</td>
</tr>
<tr>
<td>InternVL2.5-8B [10]</td>
<td>53.7</td>
<td><b>48.3</b></td>
<td>5.7</td>
<td>26.1</td>
<td>8.8</td>
<td>38.5</td>
</tr>
<tr>
<td>PLM-8B</td>
<td><b>67.7</b></td>
<td>46.2</td>
<td><b>52.8</b></td>
<td><b>46.6</b></td>
<td><b>59.1</b></td>
<td><b>55.6</b></td>
</tr>
</tbody>
</table>

Table 11: **PLM-VideoBench results** across all model scales to supplement results in Table 5.

## E Baseline Implementation Details

We provide baseline-specific implementation details for all models in §6.1 of the main paper.

**Proprietary baselines** We evaluate the GPT and Gemini family of models. For GPT-4o, we use the GPT-4o-2024-11-20 checkpoint. We feed 32 uniformly sampled frames regardless of video length, loaded at *high* image quality setting. For Gemini, we evaluate Gemini-1.5-Pro and Gemini-2.0-Flash. For VQA tasks, we input the video (without audio) which is processed internally at 1 fps. Forspatio-temporal tasks (RCap, RDCap, and RTLoc) we use the same inputs as for open-source models and GPT-4o. We evaluate these models using API call.

**Open-source models** We evaluate InternVL, Qwen, Molmo and Llava-OV models. We follow official implementation and preprocessing pipelines for each. Specifically, we evaluate InternVL2 and InternVL2.5 ([code](#)); QwenVL2 and QwenVL2.5 ([code](#)); Molmo-O-0924 ([code](#)) and Llava-OV ([code](#)). For QwenVL, we sample frames at 1 fps from videos. For InternVL2, we use 12 tiles per image as this more closely matches the reported results.

**Human performance baseline.** In Table 5, we report human performance on PLM-VideoBench. For each task, we present annotators with the test sets and collect answers for each instance given the standard task prompt. Given the difficulty of RDCap, we reuse our data annotation pipeline in §H to collect new dense captions independently, rather than providing the standard task instruction.

## F Additional Results

### F.1 Comparison with LLaMA-3V

<table border="1">
<thead>
<tr>
<th>Model</th>
<th>Avg.</th>
<th>DocVQA<br/>(test) acc [53]</th>
<th>ChartQA<br/>acc [54]</th>
<th>TextVQA<br/>acc [52]</th>
<th>InfoQA<br/>(test) acc [56]</th>
<th>A2D<br/>(w/o mask)<br/>acc [55]</th>
<th>MMMU<br/>(val) acc [37]</th>
<th>VQAV2<br/>(val) acc [111]</th>
</tr>
</thead>
<tbody>
<tr>
<td>LLaMA 3.2V (11B) [13]</td>
<td>73.0</td>
<td>88.4</td>
<td>83.4</td>
<td>79.7</td>
<td>63.6</td>
<td>91.1</td>
<td>50.7</td>
<td>75.2</td>
</tr>
<tr>
<td>LLaMA 3.2V (90B) [13]</td>
<td>76.6</td>
<td>90.1</td>
<td>85.5</td>
<td>82.3</td>
<td>67.2</td>
<td>92.3</td>
<td>60.3</td>
<td>78.1</td>
</tr>
<tr>
<td>PLM (1B)</td>
<td>67.1</td>
<td>90.7</td>
<td>78.6</td>
<td>82.1</td>
<td>63.0</td>
<td>84.9</td>
<td>34.8</td>
<td>81.7</td>
</tr>
<tr>
<td>PLM (3B)</td>
<td>74.4</td>
<td>93.8</td>
<td>84.3</td>
<td>84.3</td>
<td>74.6</td>
<td>90.9</td>
<td>41.2</td>
<td>84.3</td>
</tr>
<tr>
<td>PLM (8B)</td>
<td>76.2</td>
<td>94.6</td>
<td>86.5</td>
<td>86.5</td>
<td>80.9</td>
<td>92.7</td>
<td>46.1</td>
<td>85.6</td>
</tr>
</tbody>
</table>

Table 12: **PLM versus LLaMA-3V on Image Benchmarks:** Note that we use LLaMA-3V-90B [13] for generating image captions in our synthetic data engine.

### F.2 Image Captioning

<table border="1">
<thead>
<tr>
<th>Model</th>
<th>COCO<br/>(karpathy)<br/><i>CIDEr</i> [49]</th>
<th>Nocap<br/><i>CIDEr</i> [50]</th>
<th>Flickr<br/><i>CIDEr</i> [51]</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="4"><b>Proprietary</b></td>
</tr>
<tr>
<td>GPT-4o [33]</td>
<td>74.4</td>
<td>76.6</td>
<td>71.7</td>
</tr>
<tr>
<td>Gemini 1.5 Pro [35]</td>
<td>70.6</td>
<td>71.1</td>
<td>68.2</td>
</tr>
<tr>
<td>Gemini 2.0 Flash [35]</td>
<td>84.8</td>
<td>85.0</td>
<td>66.6</td>
</tr>
<tr>
<td colspan="4"><b>1B scale</b></td>
</tr>
<tr>
<td>Qwen2VL-2B [30]</td>
<td>107.1</td>
<td>101.2</td>
<td>86.0</td>
</tr>
<tr>
<td>InternVL2.5-1B [10]</td>
<td>122.6</td>
<td>110.5</td>
<td>86.1</td>
</tr>
<tr>
<td>PLM-1B</td>
<td><b>138.6</b></td>
<td><b>124.2</b></td>
<td><b>100.5</b></td>
</tr>
<tr>
<td colspan="4"><b>3B scale</b></td>
</tr>
<tr>
<td>Qwen2.5 VL-3B [106]</td>
<td>101.7</td>
<td>105.5</td>
<td>77.5</td>
</tr>
<tr>
<td>InternVL2.5-4B [10]</td>
<td>125.4</td>
<td>117.1</td>
<td>87.4</td>
</tr>
<tr>
<td>PLM-3B</td>
<td><b>144.9</b></td>
<td><b>126.5</b></td>
<td><b>98.0</b></td>
</tr>
<tr>
<td colspan="4"><b>8B scale</b></td>
</tr>
<tr>
<td>LLaVA-OV-7B [28]</td>
<td>112.1</td>
<td>70.7</td>
<td>55.7</td>
</tr>
<tr>
<td>Qwen2.5VL-7B [106]</td>
<td>36.8</td>
<td>32.7</td>
<td>34.9</td>
</tr>
<tr>
<td>InternVL2.5-8B [10]</td>
<td>125.8</td>
<td>116.7</td>
<td>96.5</td>
</tr>
<tr>
<td>PLM-8B</td>
<td><b>146.7</b></td>
<td><b>129.9</b></td>
<td><b>105.6</b></td>
</tr>
</tbody>
</table>

Table 13: **Image Captioning benchmarks.** PLM versus proprietary models and open-access baselines of comparable scale on Image Captioning benchmarks.### F.3 Image Grounding

<table border="1">
<thead>
<tr>
<th>Model</th>
<th>RefCOCO<br/><i>val</i></th>
<th>RefCOCO<br/><i>testA</i></th>
<th>RefCOCO<br/><i>testB</i></th>
<th>RefCOCO+<br/><i>val</i></th>
<th>RefCOCO+<br/><i>testA</i></th>
<th>RefCOCO+<br/><i>testB</i></th>
<th>RefCOCOg<br/><i>val</i></th>
<th>RefCOCOg<br/><i>test</i></th>
<th>Avg.</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="10"><b>Specialists</b></td>
</tr>
<tr>
<td>GroundingDINO [211]</td>
<td>90.6</td>
<td>93.2</td>
<td>88.2</td>
<td>88.2</td>
<td>89.0</td>
<td>75.9</td>
<td>86.1</td>
<td>87.0</td>
<td>86.6</td>
</tr>
<tr>
<td>UNINEXT-H [212]</td>
<td>92.6</td>
<td>94.3</td>
<td>91.5</td>
<td>85.2</td>
<td>89.6</td>
<td>79.8</td>
<td>88.7</td>
<td>89.4</td>
<td>88.9</td>
</tr>
<tr>
<td>ONE-PEACE [213]</td>
<td>90.6</td>
<td>93.2</td>
<td>88.2</td>
<td>88.2</td>
<td>89.0</td>
<td>75.9</td>
<td>86.1</td>
<td>87.0</td>
<td>86.6</td>
</tr>
<tr>
<td colspan="10"><b>1B scale</b></td>
</tr>
<tr>
<td>PLM-1B</td>
<td>88.5</td>
<td>91.5</td>
<td>84.8</td>
<td>83.2</td>
<td>88.6</td>
<td>76.5</td>
<td>86.0</td>
<td>86.4</td>
<td>85.7</td>
</tr>
<tr>
<td colspan="10"><b>3B scale</b></td>
</tr>
<tr>
<td>Qwen2.5 VL-3B [106]</td>
<td>89.1</td>
<td>91.7</td>
<td>84.0</td>
<td>82.4</td>
<td>88.0</td>
<td>74.1</td>
<td>85.2</td>
<td>85.7</td>
<td>85.0</td>
</tr>
<tr>
<td>PLM-3B</td>
<td><b>93.3</b></td>
<td><b>94.9</b></td>
<td><b>89.5</b></td>
<td><b>89.8</b></td>
<td><b>93.6</b></td>
<td><b>84.2</b></td>
<td><b>90.8</b></td>
<td><b>90.9</b></td>
<td><b>90.9</b></td>
</tr>
<tr>
<td colspan="10"><b>8B scale</b></td>
</tr>
<tr>
<td>Cube-LLM [214]</td>
<td>90.9</td>
<td>92.6</td>
<td><b>87.9</b></td>
<td>83.9</td>
<td>89.2</td>
<td>77.4</td>
<td>86.6</td>
<td>87.2</td>
<td>87.0</td>
</tr>
<tr>
<td>Qwen2VL-7B [30]</td>
<td><b>91.7</b></td>
<td>93.6</td>
<td>87.3</td>
<td>85.8</td>
<td>90.5</td>
<td>79.5</td>
<td>87.3</td>
<td>87.8</td>
<td>87.9</td>
</tr>
<tr>
<td>Qwen2.5VL-7B [106]</td>
<td>89.1</td>
<td>91.7</td>
<td>84.0</td>
<td>82.4</td>
<td>88.0</td>
<td>74.1</td>
<td>85.2</td>
<td>85.7</td>
<td>85.0</td>
</tr>
<tr>
<td>InternVL2-8B [10]</td>
<td>87.1</td>
<td>91.1</td>
<td>80.7</td>
<td>79.8</td>
<td>87.9</td>
<td>71.4</td>
<td>82.7</td>
<td>82.7</td>
<td>82.9</td>
</tr>
<tr>
<td>InternVL2.5-8B [10]</td>
<td>90.3</td>
<td><b>94.5</b></td>
<td>85.9</td>
<td>85.2</td>
<td><b>91.5</b></td>
<td>78.8</td>
<td>86.7</td>
<td>87.6</td>
<td>87.6</td>
</tr>
<tr>
<td>PLM-8B</td>
<td>90.6</td>
<td>91.8</td>
<td>85.9</td>
<td><b>87.3</b></td>
<td>91.3</td>
<td><b>81.1</b></td>
<td><b>88.8</b></td>
<td><b>89.2</b></td>
<td><b>88.2</b></td>
</tr>
</tbody>
</table>

Table 14: Image Grounding results on RefCOCO+/+g. PLM performs competitively compared to the baselines across all model scales, and outperforms specialist models for the image grounding task.

### F.4 Long Video Understanding

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="3">Long Video QA</th>
</tr>
<tr>
<th>LYBench<br/><i>acc</i> [92]</th>
<th>LongVideoBench<br/><i>(val) acc</i> [94]</th>
<th>MLYU<br/><i>(dev) Mavg</i> [96]</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="4"><b>Proprietary</b></td>
</tr>
<tr>
<td>GPT-4o [33]</td>
<td>37.2</td>
<td>66.7*</td>
<td>67.4</td>
</tr>
<tr>
<td>Gemini 1.5 Pro [35]</td>
<td>33.1*</td>
<td>64.0*</td>
<td>69.9</td>
</tr>
<tr>
<td>Gemini 2.0 Flash [35]</td>
<td>-</td>
<td>61.6*</td>
<td>69.5</td>
</tr>
<tr>
<td colspan="4"><b>1B scale</b></td>
</tr>
<tr>
<td>Qwen2VL-2B [30]</td>
<td><b>42.0</b></td>
<td>47.9</td>
<td><b>62.7</b></td>
</tr>
<tr>
<td>InternVL2-1B [10]</td>
<td>31.4</td>
<td>43.3*</td>
<td>52.0</td>
</tr>
<tr>
<td>InternVL2.5-1B [10]</td>
<td>35.3</td>
<td>47.9</td>
<td>57.3*</td>
</tr>
<tr>
<td>PLM-1B</td>
<td>40.0</td>
<td><b>52.3</b></td>
<td>58.9</td>
</tr>
<tr>
<td colspan="4"><b>3B scale</b></td>
</tr>
<tr>
<td>Qwen2.5 VL-3B [106]</td>
<td><b>43.3*</b></td>
<td>54.2*</td>
<td>68.2</td>
</tr>
<tr>
<td>InternVL2-4B [10]</td>
<td>34.0</td>
<td>53.0*</td>
<td>59.9*</td>
</tr>
<tr>
<td>InternVL2.5-4B [10]</td>
<td>40.1</td>
<td>56.3</td>
<td><b>68.3*</b></td>
</tr>
<tr>
<td>PLM-3B</td>
<td>40.4</td>
<td><b>57.9</b></td>
<td>65.0</td>
</tr>
<tr>
<td colspan="4"><b>8B scale</b></td>
</tr>
<tr>
<td>LLaVA-OV-7B [28]</td>
<td>38.8</td>
<td>55.7</td>
<td>64.6</td>
</tr>
<tr>
<td>Qwen2VL-7B [30]</td>
<td><b>46.0</b></td>
<td>55.8</td>
<td>69.8*</td>
</tr>
<tr>
<td>Qwen2.5VL-7B [106]</td>
<td>45.3*</td>
<td>56.0*</td>
<td><b>70.2*</b></td>
</tr>
<tr>
<td>InternVL2-8B [10]</td>
<td>37.0</td>
<td>55.4</td>
<td>64.0*</td>
</tr>
<tr>
<td>InternVL2.5-8B [10]</td>
<td>43.2*</td>
<td><b>60.0*</b></td>
<td>68.9</td>
</tr>
<tr>
<td>PLM-8B</td>
<td>44.5</td>
<td>56.9</td>
<td>66.4</td>
</tr>
</tbody>
</table>

Table 15: Results on long video understanding tasks. We compare PLM with open-access baselines and proprietary models of comparable scale, and report results over 3 long video QA benchmarks. Cells with \* are reported numbers from literature. The remaining are reproduced using official code.## G PLM-FGQA: Fine-grained QA

We present PLM-FGQA Fine-grained QA (FGQA), a video dataset focused on “how” actions are performed, capturing nuanced fine-grained details through specially designed questions and carefully annotated answers. Due to the scarcity of fine-grained video Q&A data, see Table 16, we built a data engine to enable the collection of our 2.4M Q&A dataset, PLM-FGQA.

<table border="1">
<thead>
<tr>
<th>Dataset</th>
<th>Year</th>
<th>#Q&amp;As</th>
<th>Dataset</th>
<th>Year</th>
<th>#Q&amp;As</th>
</tr>
</thead>
<tbody>
<tr>
<td>MovieQA</td>
<td>2016</td>
<td>6462</td>
<td>STAR</td>
<td>2021</td>
<td>60000</td>
</tr>
<tr>
<td>MSRVTT-QA</td>
<td>2017</td>
<td>243690</td>
<td>CLEVRER</td>
<td>2023</td>
<td>82620</td>
</tr>
<tr>
<td>TGIF-QA</td>
<td>2017</td>
<td>165165</td>
<td>EgoQA</td>
<td>2024</td>
<td>19000</td>
</tr>
<tr>
<td>MSVD-QA</td>
<td>2017</td>
<td>51000</td>
<td>PerceptionTest</td>
<td>2024</td>
<td>44146</td>
</tr>
<tr>
<td>TVQA</td>
<td>2018</td>
<td>152545</td>
<td>VideoInstruct</td>
<td>2024</td>
<td>25803</td>
</tr>
<tr>
<td>ActivityNetQA</td>
<td>2019</td>
<td>58000</td>
<td>MoVQA</td>
<td>2024</td>
<td>21953</td>
</tr>
<tr>
<td>How2QA</td>
<td>2020</td>
<td>44007</td>
<td>CinePile</td>
<td>2024</td>
<td>303828</td>
</tr>
<tr>
<td>NexT-QA</td>
<td>2021</td>
<td>52044</td>
<td>Sports-QA</td>
<td>2025</td>
<td>94000</td>
</tr>
<tr>
<td>PLM-FGQA</td>
<td>2025</td>
<td>2379067</td>
<td></td>
<td></td>
<td></td>
</tr>
</tbody>
</table>

Table 16: Comparison of our PLM-FGQA dataset with existing video-QA datasets.

### G.1 Annotation process: Data Engine

Our data engine is built upon the following modules: (1) Temporal Segment Generation, (2) Question Generation, (3) Answer Generation, (4) Human Annotation (answer verification/manual answer annotation), (5) Quality Control, as illustrated in Figure 12. Next, we describe each module in detail, and finally also provide additional details about the extra steps we took for forming the FG-QA component of PLM-VideoBench out of these annotations.

```

graph LR
    Video[Video] --> TSG[Temporal Segment Generation]
    TSG --> QG[Question Generation (LLM)]
    QG --> AG[Answer Generation (PLM)]
    AG --> HVC[Human Verification & Correction]
    HVC --> FGQA[FGQA]
  
```

Figure 12: Data engine used to collect the PLM-FGQA dataset.

#### G.1.1 Temporal Segment Generation

We source the video data that serves as a basis for our annotations from publicly available datasets. Based on the video sources and the type of existing annotations, we split the videos into three distinct categories.

**Videos with existing ground-truth segment annotations:** We directly adopt segments with their human-annotated action annotations from the following datasets: Ego4d Goal-Step[215], Ego4D Moments[115], EgoExo4D [116], HT-Step[216, 217], COIN [117], CrossTask [118], and YouCook2 [83]. All those sources provide video segment boundaries accompanied by some form of textual action descriptions, and are therefore readily usable with the rest of the pipeline.

**Unedited videos of physical activities:** For physical activities videos (*e.g.* basketball, dancing, soccer), actions are usually atomic and short (*e.g.* dribble, dance move, kick) and therefore require precise temporal localization. To source videos for these scenarios we used data from EgoExo4D [116] that contains temporally well-aligned and precise narrations; we obtained segments of 2-3 seconds centered around narration timings, and used the anchor narrations directly as the action description.

**Raw, untrimmed videos in-the-wild without temporal segment annotations.** We source a very large part of our data from untrimmed instructional videos in the large-scale HT100M dataset [114] which we first need to segment before use. The goal is to obtain video clips that contain meaningful, salient actions, and also caption the resulting segments with concise but accurate action descriptions. We describe the automatic segmentation and captioning module in the following.

The automatic segmentation and captioning pipeline involves the following three stages:Figure 13: Distribution of question types (left) and video sources (right) in the FGQA component of PLM-VideoBench.

**Temporal segment proposal.** Given untrimmed long videos, the first step is to identify semantically coherent segments within them. Inspired by prior work on unsupervised action proposal and segmentation, we leverage visual feature clustering to generate temporal segment proposals, and use shot-boundary detection results to further refine the segment boundaries. We extract clip-level visual features[218] using a sliding window with temporal stride of 1 second. We then compute the pairwise similarity between neighborhood features and detect the class-agnostic action boundaries using a boundary detection kernel (similar to those used in literature[219, 220]). Finally, since the detected segments are usually over-segmented, we perform a bottom-up agglomerate clustering approach to group adjacent segments into clusters, using a segment duration prior of 10 seconds. We also leverage shot boundary detection[221] to obtain precise moments of scene changes: we refine the boundaries of the segment proposals by aligning them to the detected shot boundaries when they’re sufficiently close ( $\leq 1$  second).

**Segment filtering and ranking.** How-to videos often include a lot of content that is irrelevant to the demonstration of the activity at hand, such as the instructor explaining what they are about to do or showcasing tools and ingredients. It is therefore important to detect and filter segments with such uninformative content. To that end we rank candidate segments according to relevance using a series of heuristics and learned models, described bellow.

*a. Talking head detection.* A common mode in instructional videos is instructors talking into the camera, describing objects or explaining actions they’re about to take. To detect and remove such segments, we employ an Active Speaker Detection (ASD) pipeline[222], which we run densely on every video and combine resulting talking head tracks, to produce an ASD score for every segment.

*b. Hand-object interaction (HOI) detection.* The presence of hand-object interaction (HOI) can be a good indicator of visually groundable actions. We leverage the temporal selection strategy[223] to filter out the segment proposals that contain hand-object interaction. We first employ an off-the-shelf robust HOI detector[224] to densely extract HOI regions within a proposed segment. The HOI score is then calculated by measuring the likelihood of hand-object interaction in the segment and the averaged probability of all the detected hands.

*c. ASR groundability.* HT100M videos contain timestamped ASR captions, which are speech transcriptions of the audio instructions. It is desirable to rank candidate segments based on how likely their ASR content is to their video content. The hypothesis here is that segments containing ASR transcriptions that align well to the video content, are more likely to be visual-information rich. Moreover since the action labeling pipeline (described next) relies on ASR metadata for producing descriptions, higher ASR groundability scores make it likelier to produce good quality segment descriptions. For every candidate segment, we compute an ASR-groundability score by computing video-text alignment scores[218] for each ASR caption within the segment and then averaging the ones that are above a threshold (we use 0.5).

*d. Relevance classification.* The above heuristics work well for the clear-cut cases they are tailored for, but in practice we found that they struggle with more nuanced segments (*e.g.* instructor fiddling with an object and describing it rather than using it). To improve the detection of those cases, we manually labelled a small amount of segments that passed through the other filters and trained a binary classifier to classify them as “relevant” or “irrelevant”; to that end we trained a simple 2-layer MLP classifieron top of temporally pooled video representations with a logistic loss for binary classification. We deployed the trained model to provide a relevance score for all the candidate segments.

We combined the scores resulting from all the modules described above and determined cutoff thresholds, based on a small manually annotated validation set. In production, we keep all the segments that have relevance scores above those thresholds.

**Segment captioning** We follow a two-step process to obtain action labels for each unlabeled segment: In the first step, a collection of off-the-shelf perception models are used to extract individual image-level captions, video-level captions, and object detections from the segment. The output of all perception models is then fed as text into an LLM to generate long, fine-grained captions. At the second step, the detailed captions are fused with the ASR content of the segment, to obtain a concise action description. Specifically, we query an LLM (Llama 3.3 70B [13]) with the following prompt:

**Segment to action labels prompt**

Detailed description: [fine grained caption] ASR transcription: [asr caption]. Given the detailed description above, identify the specific action performed as part of the activity [task name]. Your response must not be the same as the activity [task name] and needs to be a specific substep within the activity [task name]. Please also supply a rationale for your answer.

The extracted labeled video segments obtained through the above process serve as the foundation for the subsequent Q&A generation.

### G.1.2 Automatic Question Generation

We automatically generate questions about the fine-grained details of the way activities are executed in the video. Our questions are generated with a variety of prompts and models which lead to increased question diversity and specificity. In Table 17 we present the question types and sample questions per question type. Here, we summarize how these questions are generated automatically with an ensemble with models and prompts:

**LLM-based action-conditioned question generation** Given a segment, its action name (*e.g., cut potatoes*), a task name (*e.g., How to make sweet potato gratin*) and optionally other metadata about the segment (for example, recognized objects [? ]), we generate questions that can elicit descriptions of fine-grained details by raters with an LLM. We use tailored prompts for generating questions that cover *how* the activity is executed (tools, object locations, object states, direction of movements, hand pose), and the spatial arrangement of objects.

**Activity FG question generation prompt**

I am learning how to [action name] while [task name]. Ask me [N] most relevant questions that reveal the details of the way the step is executed in my environment, *e.g.*, (a) part location, (b) types of tools/ingredients used, (c) direction of movements, (d) how are objects held, (e) object states at the beginning of the step, (f) object state at the end of the step. The questions must be answerable by visually observing the activity, without reading instructions or trying out. Please indicate the type of question from (a) to (f) for each question asked at the beginning of the question.

**Spatial FG question generation prompt**

Imagine I have no common sense or understanding of the 3D real world. I am trying to [task name] and am at the step where I am [action name]. There's [object list] when I'm [action name]. Ask me [N] questions about the 3D position of objects, relative location between objects, distance between objects, spatial relationship using prepositions like above, below, next to, etc. that I might want to know. The questions must be answerable by only visually observing me performing activity, without reading instructions or trying out.

We explicitly encourage the LLM to provide questions that can be answered solely based on the video frames, in contrast to questions that are focused on external knowledge or non-groundable concepts or judging the execution of the step (*e.g., avoid questions like is the pan hot enough to add the oil?, what tool is typically used to loosen the axle nut*). The rationale for this is to collect as many Q&A pairs that a model cannot answer just based on external knowledge/language prior, but they ratherrequire vision perception to be answered. Note that these questions are generated without visual input, hence they are not instance-specific and might not be answerable given the video segment.

**VLM-based instance-specific question generation** After collecting a first set of Q&As using the LLM-generated questions, we bootstrap a VLM Question Generator model, which takes as input the video segment, question types and optionally the task name, and generates a set of instance-specific visual questions. The VLM Question Generator model is obtained by supervised fine-tuning of PLM with a question generation instruction-tuning dataset which consists of triplets (video, prompt, response), where the prompt includes the instruction to generate questions based on question types and the response includes example questions to be generated for the given video. Due to the lack of such a dataset with fine-grained question, we synthetically generated it by utilizing the Q&A pairs obtained based on the LLM-generated questions. Specifically, for each video segment, we use an LLM to (1) decompose existing Q&A pairs into multiple Q&A pairs, with each new question focusing on one detail of the original answer; (2) tag question types for the generated questions based on an expanded list of question types; and (3) generate a (prompt, response) pair for the segment. This resulted in  $\sim 600k$  training instances.

#### VLM Question Generator training sample

```
Generate 3 different questions that reveal the fine-grained details of the way the activity is executed. In particular, focus on these question types: fine-grained object locations, hand pose, object/repetition counts, generating at least one question per type. Write each question in a separate line, e.g., Q1. first question.  
Q2. second question.  
...  
QN. N-th question.  
Response:  
Q1. Where are the tomatoes positioned prior to being cut?  
Q2. How is the person grasping the tomato with their left hand?  
Q3. How many tomatoes did the person use in the segment?
```

**LLM-based follow-up question generation** This final set of questions aims to increase coverage of video details and generate highly fine-grained questions by leveraging the already collected Q&A pairs for each segment and feed them to an LLM that generates “follow-up” questions that are more detailed and challenging than the initial questions.

#### Follow-up question generation prompt

```
I have the following information gathered about the video: [list of previous Q&A samples] Utilizing information and details from all the provided Q&A pairs (make sure to specialize questions based on the already corrected answers, e.g., using referring expressions), ask [N] most relevant and interesting, visual questions that we can ask annotators in order to reveal NEW, rich, additional fine-grained details about the video that we don't know yet, in particular about the following question types: 'tools/ingredients', 'object counts', 'repetition counts', 'direction of movement', 'hand pose', 'fine-grained object locations', 'spatial relations', 'initial state/end state', 'action happened before/after', 'clothes wearing', 'body pose', 'main action in the video', 'temporal extent of action', 'sizes'. The questions should be specific and have a specific answer. Avoid generic questions that can be very tedious to answer, e.g., how many objects are there in the scene. Also, do not generate questions that start with "Is ..." and then list options. Prefer open-ended questions, e.g., starting with "How". [... More examples & formatting ...]
```

### G.1.3 Automatic Answer Generation

The next step of the data engine aims to produce correct and comprehensive answers to the generated questions. We obtain automatic answers to the generated questions using a version of PLM that has been fine-tuned with extra privileged information of various forms as input. The privileged information includes textual annotations from the metadata available with the candidate training videos and feature embeddings extracted from off-the-shelf models. Useful textual metadata include the video title, ASR captions or written descriptions, video-level task name (inferred by an LLM using the title and captions), and any existing QAs for that video. Off-the-shelf embeddings include frame-level features extracted denseley at 1 fps; we use an open-vocabulary object detection model, OWLv2 [225], for embedding object detection information and CLIP ViT-L14 embeddings [226]<table border="1">
<thead>
<tr>
<th>Question Type</th>
<th>Sample Questions</th>
</tr>
</thead>
<tbody>
<tr>
<td>Action Recognition</td>
<td>What is the process being performed on the sandpaper?<br/>What is the action shown?</td>
</tr>
<tr>
<td>Action Sequence</td>
<td>What does the person do after brewing the tea?<br/>What does the person do before marking the vinyl with a pencil?</td>
</tr>
<tr>
<td>Counting Problems</td>
<td>What is the quantity of universal down cleaner being poured into the task area?<br/>How many branches does the person cut in total?<br/>How many times does the person spray Greased Lightning onto the ketchup spill?</td>
</tr>
<tr>
<td>Movement Direction</td>
<td>In what direction is the black welding tool pointing while the person is working on the metal joint?<br/>How does the person chop the garlic with the knife?</td>
</tr>
<tr>
<td>Object Attributes</td>
<td>What is the color of the seatpost shown in the video segment?<br/>What is the shape of the tube at the end of the step?<br/>What is the size of the knife being used to chop the spring onions?</td>
</tr>
<tr>
<td>Object Location</td>
<td>Where does the person put the honey bottle away?<br/>Where does the person position the clothes before ironing?</td>
</tr>
<tr>
<td>Object Recognition</td>
<td>What type of roller and paint are being used?<br/>What does the person place on top of the smooth half of the egg carton?<br/>What was the person initially holding in their left hand?</td>
</tr>
<tr>
<td>Object State</td>
<td>How would you describe the sink at the beginning of the cleaning process?<br/>What is the state of the nematode after mixing it with water and sponge?</td>
</tr>
<tr>
<td>Other</td>
<td>At what point in the video is the person seen holding the wires?</td>
</tr>
<tr>
<td>Pose</td>
<td>How are the woman’s legs positioned while she is sitting?<br/>How bent is the left elbow during the activity?</td>
</tr>
<tr>
<td>Spatial Relations</td>
<td>How far is the bias tape maker from the right edge of the ironing board?<br/>What is the spatial relationship between the bowls and the Brussels sprouts on the kitchen countertop?</td>
</tr>
<tr>
<td>Speed/Force</td>
<td>How would you describe the consistency of pressure applied during sanding?<br/>How fast does the person initially push the stone?</td>
</tr>
</tbody>
</table>

Table 17: PLM–FGQA question types and sample questions

for scene classification information. We incorporate the textual annotations directly into language prompts using the following template:

Automatic answer generation prompt

A video is showing a task [video level task name], specifically the part where [ASR caption]. Here is what we already know about the video: [existing question-answer pairs]. Answer this question in detail: [question]

The off-the-shelf embeddings are incorporated into the PLM input via an additional Perceiver-IO[227] tokenizer, which summarizes the embeddings at the segment level.

We fine-tune the answer generator on 1M manually annotated QA pairs. After fine-tuning, we deploy the trained answer generator with privileged information access on the unlabelled questions produced in the previous step, to produce automatic answers.

#### G.1.4 Human Annotation

After obtaining segments and generating questions and automatic answers, we employ human annotators to obtain high-quality answers. Our answer annotations include the following:

- • **Human-verified answers:** Raters are provided with the model-generated answer and are asked to accept or reject the answer. They can reject questions for being irrelevant or unanswerable, and answers for being factually incorrect or lacking details. Accepted question-answer pairs proceed without changes, while rejected ones are handled differently:question-related rejections (irrelevant or unanswerable) are discarded, whereas answer-related rejections (factually incorrect or lacking details) are marked for correction in the next phase. 17.8% of the total training samples are human-verified automatic answers.

- • **Human annotated answers:** Raters answer the questions from scratch by ensuring to cover all the relevant details within the temporal segment. They receive reference information, such as video-level task names and ASR captions, and may use online resources like WikiHow for additional context. Questions that cannot be answered based on the video segment (for example, due to some false premise) are rejected (with an explanation). These manually annotated answers make up 82.2% of the PLM-FGQA training split, and 100% of the evaluation set.

**Quality Control.** Data quality is crucial for model success. We followed several strategies to monitor and enhance annotation quality: *annotation Certification* - we reviewed a small sample of annotations from each rater before they could work in production queues, ensuring that annotators met high-quality standards before advancing to production; *golden Examples* - annotators were provided with high-quality annotation examples, highlighting common error patterns and offering acceptable answers. *targeted and Dual QA* - we conducted daily audits, including vendor auditing and our own sampled quality control. In total, 13% of the training set was audited, and 100% of the samples in PLM-VideoBench underwent quality control.

## G.2 FGQA PLM-VideoBench Construction

<table border="1">
<thead>
<tr>
<th></th>
<th>Train</th>
<th>Test</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="3"><b>Sources stats</b></td>
</tr>
<tr>
<td>Total Videos</td>
<td>767k</td>
<td>3.6k</td>
</tr>
<tr>
<td>Unique Source Videos</td>
<td>251k</td>
<td>1.9</td>
</tr>
<tr>
<td>Average Duration (sec.)</td>
<td>9.8</td>
<td>12.3</td>
</tr>
<tr>
<td colspan="3"><b>Annotations stats</b></td>
</tr>
<tr>
<td>Number of QA Pairs</td>
<td>2.4M</td>
<td>4.2k</td>
</tr>
<tr>
<td>Number Question Types</td>
<td>12</td>
<td>12</td>
</tr>
<tr>
<td>Question Length (avg/max)</td>
<td>12/114</td>
<td>12.3/56</td>
</tr>
<tr>
<td>Answer Length (avg/max)</td>
<td>13.3/911</td>
<td>14.1/62</td>
</tr>
<tr>
<td>Annotation Type</td>
<td>Human</td>
<td>Human</td>
</tr>
<tr>
<td>Open-Domain</td>
<td>Yes</td>
<td>Yes</td>
</tr>
</tbody>
</table>

Table 18: Statistics of the PLM-FGQA training and test data. The test split refers to the FGQA module of PLM-VideoBench.

The FG-QA component of PLM-VideoBench is formed from a held-out portion of PLM-FGQA. We refine this set and transform it into a challenging MCQ-based benchmark by (1) generating MCQs, (2) filtering out samples that can be answered by text-only (blind) LLMs, (3) performing human verification of negatives, and (4) balancing the distribution of question types and domains. The statistics of the dataset are summarized in Table 18. In more detail the steps we followed are:

*MCQ Generation:* To transform QAs into challenging MCQs for evaluation, instead of generating random incorrect answers, we prompt LLMs to produce hard negatives that are semantically close to the correct answer. We use the following prompt which was designed to generate distractors that differ from the correct answer by only a single detail. In effect this enables evaluation to assess fine-grained reasoning about object attributes and tool distinctions.

*Filtering Text-Only Answers:* To ensure that video-based reasoning is required, we test whether a text-only LLM can answer the question correctly without seeing the video. If a question can be answered correctly from text alone, we remove or modify it to emphasize visual and temporal grounding.

*Human Verification of Negatives:* Automatically generated negatives may sometimes be factually true despite being labeled as incorrect. To address this, we perform human verification, where annotators review distractors to confirm that they are both plausible yet definitively incorrect given the video context. MCQs with ambiguous distractors are removed.*Balancing Question Types:* Finally, after the above postprocessing and filtering is done, we rebalance the test set, to make sure that the question type and domain distributions are approximately uniform, by undersampling over-represented question types and domains.

**Note on the evaluation metric.** We report the multi-binary accuracy (MBAcc) [99] to evaluate on the FG-QA task. This accuracy is calculated by comparing the correct answer to each distractor individually. Specifically, for each question, we generate a series of binary questions, where the correct answer is compared with one distractor at a time. A prediction is considered correct only if the correct answer is consistently selected across all binary comparisons. We preferred this metric to vanilla MCQ accuracy as it greatly reduces the predictability of automatically-generated MCQs.

#### MCQ generation prompt

Here is a question and answer pair about a video:

Q: [question]

A: [answer]

You need to transform this into a high-quality multiple-choice question. To do this, first rephrase the given correct answer and then provide n distractor answers. The n incorrect answers should be reasonable and valid responses to the question, but should have a different meaning than the correct answer. You generate an incorrect answer from the correct one by changing a single detail, e.g. an object or verb/action that is relevant to what's being asked. Make the incorrect answers realistic, plausible and similar enough to the correct answer so that it is very difficult for someone to distinguish between them with prior knowledge alone. Finding the correct answer should also require visual information about the scene. The distractor answers should answer the question, but should be incorrect but in a non-obvious way. When changing a single detail to create the distractors, make sure that this detail is the main point of the question. For example, if the question is about the color of an object, then the distractor should change the color of the object and not the kind of object.

Here are some examples of good distractors (desired) and bad distractors (to be avoided):

Q: What is the person wearing on their hands while applying varnish?

A: The person is wearing white gloves on their hands while applying varnish with a brush.

Good distractors:

- The person is wearing black gloves on their hands while applying varnish with a brush.

Bad distractors:

- The person is wearing black gloves on their hands while applying paint with a roller.

... More examples & formatting ...

## H PLM-STC Details

We present PLM Spatio-Temporal Captions (PLM-STC), a novel dataset aimed at training and evaluating VLMs for spatial-temporal reasoning. We collected pairs of mask tublets for objects in videos, along with their corresponding detailed temporal descriptions. The annotations are collected on top of the SA-V [124] videos, which are diverse and high-quality. We excluded the test set videos from SA-V, to avoid any data cross contamination. Table 20 provides statistics about the dataset, such as number of total samples, training/val/test splits, object types, and time-segment duration. PLM-STC, is not only novel, but also larger and higher quality compared to existing datasets, see Table 19. In Fig. 5 (right), we show an example of our spatio-temporal captions, describing a little girl (highlighted in **blue**): *(frame 0-81): A little girl moves back as beluga whale approaches her face. (frame 82-85): Out of frame. (frame 86-98): She tries to feed the whale.*

We describe the overall annotation process in Appendix H.1, and how we build the three sub-tasks in Appendix H.2.

### H.1 Annotation Process

The annotation process is summarized in Figure 14. The annotation process involves three stages: *Object Selection and Tracking, Temporal Segmentation and Captioning and Verification and Quality Control.*<table border="1">
<thead>
<tr>
<th>Dataset</th>
<th>Spatial Type</th>
<th>Year</th>
<th>#Videos</th>
<th>Regions</th>
<th>Temp. Seg.</th>
<th>Captions?</th>
</tr>
</thead>
<tbody>
<tr>
<td>DAVIS16-RVOS [228]</td>
<td>Segmentation</td>
<td>2018</td>
<td>50</td>
<td>50</td>
<td>-</td>
<td>No</td>
</tr>
<tr>
<td>DAVIS17-RVOS [229]</td>
<td>Segmentation</td>
<td>2018</td>
<td>90</td>
<td>205</td>
<td>-</td>
<td>No</td>
</tr>
<tr>
<td>YouCook2-BB [83]</td>
<td>BBox</td>
<td>2018</td>
<td>647</td>
<td>-</td>
<td>4.3K</td>
<td>No</td>
</tr>
<tr>
<td>A2D Sentence [230]</td>
<td>Segmentation</td>
<td>2018</td>
<td>3.7K</td>
<td>4.8K</td>
<td>-</td>
<td>No</td>
</tr>
<tr>
<td>J-HMDB Sentence [231]</td>
<td>Segmentation</td>
<td>2018</td>
<td>928</td>
<td>928</td>
<td>-</td>
<td>No</td>
</tr>
<tr>
<td>ActivityNet Entities [232]</td>
<td>BBox</td>
<td>2019</td>
<td>14.3K</td>
<td>1.5M</td>
<td>52K</td>
<td>No</td>
</tr>
<tr>
<td>VidSTG [9]</td>
<td>BBox</td>
<td>2020</td>
<td>6.9K</td>
<td>44.8K</td>
<td>-</td>
<td>No</td>
</tr>
<tr>
<td>Refer-Youtube-VOS [233]</td>
<td>Segmentation</td>
<td>2020</td>
<td>3.9K</td>
<td>7.5K</td>
<td>-</td>
<td>No</td>
</tr>
<tr>
<td>HC-STVG [234]</td>
<td>BBox</td>
<td>2021</td>
<td>16K</td>
<td>16K</td>
<td>-</td>
<td>No</td>
</tr>
<tr>
<td>VLN [123]</td>
<td>Mouse Trace</td>
<td>2023</td>
<td>50K</td>
<td>43.1K</td>
<td>43.1K</td>
<td>Yes</td>
</tr>
<tr>
<td>MeVis [235]</td>
<td>Segmentation</td>
<td>2023</td>
<td>2K</td>
<td>8.8K</td>
<td>-</td>
<td>No</td>
</tr>
<tr>
<td><b>PLM-STC</b></td>
<td>Segmentation</td>
<td>2025</td>
<td>45.7K</td>
<td>122.3K</td>
<td>194.2K</td>
<td>Yes</td>
</tr>
</tbody>
</table>

Table 19: Spatio-Temporal-Captioning datasets comparison.

```

graph LR
    Video[Video] --> OST[Object Selection & Tracking]
    OST --> TSC[Temporal Seg. & Captioning]
    TSC --> VQC[Verification & Quality Control]
    VQC --> STC[Spatio Temporal Captions]
  
```

Figure 14: PLM-STC Annotation pipeline.

### H.1.1 Object Selection and Tracking

Annotators select interesting objects with significant motion changes in the video and use SAM 2 [124] to generate initial mask tublets, which they then refine to ensure high-quality spatial-temporal segmentation. We instructed the annotators by defining interesting regions in video footage as those with the presence of significant, dynamic actions performed by subjects, which can be human, animal, or object. These regions involve multiple major actions that evolve over time, rather than static or insignificant actions. We provided annotators with examples of interesting regions, such as one featuring a person making a sandwich, a dog chasing a cat, or a kite getting stuck in a tree. The goal for the annotator is to identify regions with high delta, where the subject performs a sequence of significant activities that change over time, such as a person entering a room, sitting down, and then drinking from a glass. By focusing on these dynamic and evolving actions, annotators can effectively select regions worthy of captioning. Finally, annotators are provided with several examples of good and bad annotations.

### H.1.2 Temporal Segmentation and Captioning

Based on the selected mask tublets, another set of annotators provides time segments for each action and fills in the caption within each time segment. The annotators are instructed to focus on capturing major actions, avoiding minor details or unnecessary movements. When writing captions for each segment, they must ensure clarity in describing the subject’s movements and directionality. Additionally, the annotators are advised to avoid making assumptions about the subject’s actions or adding details not clearly visible, sticking only to what is directly observable in the frame. As in the previous task, the annotators are provided with several examples of good and bad annotations to guide their work.

### H.1.3 Verification and Quality Control

A final set of annotators manually verifies the tublets and time-segment captions to ensure accuracy and consistency. For mask refinement, we re-run the same pipeline as §H.1.1, while not letting the annotators choose the interesting object, but only refine the quality of the mask. For captioning refinement, the annotators are tasked with three objectives: 1) *Redundancy*: eliminate any repeating or redundant information to ensure the caption is concise; 2) *Accuracy*: verify that every word in the caption accurately describes a fact present in the video, correcting or removing any incorrect information; and 3) *Actions*: add missing major action information to the caption while preserving existing atomic actions, ensuring the caption effectively conveys the key events in the video.<table border="1">
<thead>
<tr>
<th></th>
<th>All</th>
<th>Train</th>
<th>Val</th>
<th>Test</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="5"><b>Dataset stats</b></td>
</tr>
<tr>
<td>Number of Videos</td>
<td>45.2K</td>
<td>42.0K</td>
<td>804</td>
<td>2.3K</td>
</tr>
<tr>
<td>Spatio Temporal Caption</td>
<td>127.8K</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>Temporal Caption</td>
<td>198.7K</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td colspan="5"><b>Tube’s categories</b></td>
</tr>
<tr>
<td>Person</td>
<td>104.5K</td>
<td>99.6K</td>
<td>861</td>
<td>2.4K</td>
</tr>
<tr>
<td>Animal</td>
<td>16.8K</td>
<td>13.2K</td>
<td>550</td>
<td>1.7K</td>
</tr>
<tr>
<td>Object/things</td>
<td>6.4K</td>
<td>4.4K</td>
<td>436</td>
<td>1.2K</td>
</tr>
<tr>
<td colspan="5"><b>Temporal captions per Tube</b></td>
</tr>
<tr>
<td>1 caption per tube</td>
<td>78.9K</td>
<td>73.9K</td>
<td>842</td>
<td>2.4K</td>
</tr>
<tr>
<td>2 caption per tube</td>
<td>30.9K</td>
<td>27.8K</td>
<td>566</td>
<td>1.7K</td>
</tr>
<tr>
<td>3 or more Caption per tube</td>
<td>16.38K</td>
<td>14.15K</td>
<td>421</td>
<td>1.2K</td>
</tr>
<tr>
<td colspan="5"><b>Tasks stats</b></td>
</tr>
<tr>
<td>Region Detailed Captioning (RDCap)</td>
<td>122.3K</td>
<td>117.2K</td>
<td>2.5K</td>
<td>2.6K</td>
</tr>
<tr>
<td>Region Captioning (RCap)</td>
<td>194.2K</td>
<td>179.5K</td>
<td>4.6K</td>
<td>10.1K</td>
</tr>
<tr>
<td>Region Temporal Localization (RTLoc)</td>
<td>192.0K</td>
<td>179.5K</td>
<td>4.6K</td>
<td>7.9K</td>
</tr>
</tbody>
</table>

Table 20: PLM–STC dataset statistics. Note the for RTLoc, we filter the test set to include only the captions that are unambiguously localized, *i.e.*, they map to a single time window in the video. As a result, the test set size is reduced to 7,910 instances compared to RCap.

## H.2 PLM–STC Benchmark

We utilize the collected data to train and evaluate the PLM on three challenging tasks that are essential for video perception. Firstly, we created a balanced validation and test split by the combination of tube categories and number of caption per tube while making sure no video overlaps with the training set. This is done to make sure we evaluate all the categories presents in the dataset equally. Then, we process the data for each task:

**Dense Video Region Captioning (RDCap).** This comprehensive task combines both “what” and “when” aspects. The model takes the video and the tubelets as input and outputs the full time-segment captions. We also assign an *out of frame* caption to temporal segments for which the subject does not appear in the video to ensure dense temporal coverage of events across the video duration.

**Video Region Captioning (RCap).** This task involves describing “what” activities are performed within a specific time frame by the objects in the tubelets. The model receives the video, the tubelets, and the temporal region as input and outputs the corresponding captions. We filter out events that refer to the subject when it is out-of-frame to avoid evaluating trivial captions.

**Region Temporal Localization (RTLoc).** This task requires the model to localize “when” specific events occur in relation to a given tubelet. The input includes the video, the tubelet, and the caption, while the output is the start and end frames indicating when the captioned event occurs. Like RCap, we filter out out-of-frame events, as well as ambiguous events that may be localized to multiple time segments. For example, if the subject opens the door twice, the event text are guaranteed to be unique (*e.g.*, referring to the first and second time they opened the door) or dropped entirely if ambiguous (*e.g.*, if the text only mentions the action).

These tasks are designed to both improve and evaluate the model’s capabilities, with the same input-output format applied during both training and evaluation. Figure 6 illustrate an examples of the task, including the prompt used to train and evaluate the PLM.

## I Smart Glasses Data

### I.1 Data collection and annotation

We collected the source videos for PLM-SGQA using commercial smart glasses, which enable participants to capture egocentric videos in a hands-free manner. Participants are presented with 14 categories of popular scenarios, such as shopping, cooking, and walking in a neighborhood, and are instructed to ask questions about their surroundings as if interacting with a multi-modal assistant that shares their visual perspective. Specifically, participants are asked to ask questions spontaneously,

