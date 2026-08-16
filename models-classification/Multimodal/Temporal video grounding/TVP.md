Title: Text-Visual Prompting for Efficient 2D Temporal Video Grounding

URL Source: https://arxiv.org/html/2303.04995

Markdown Content:
Yimeng Zhang 1,2 1 2{}^{1,2}start_FLOATSUPERSCRIPT 1 , 2 end_FLOATSUPERSCRIPT, Xin Chen 2 2{}^{2}start_FLOATSUPERSCRIPT 2 end_FLOATSUPERSCRIPT, Jinghan Jia 1 1{}^{1}start_FLOATSUPERSCRIPT 1 end_FLOATSUPERSCRIPT, Sijia Liu 1 1{}^{1}start_FLOATSUPERSCRIPT 1 end_FLOATSUPERSCRIPT, Ke Ding 2 2{}^{2}start_FLOATSUPERSCRIPT 2 end_FLOATSUPERSCRIPT

1 1{}^{1}start_FLOATSUPERSCRIPT 1 end_FLOATSUPERSCRIPT Michigan State University, 2 2{}^{2}start_FLOATSUPERSCRIPT 2 end_FLOATSUPERSCRIPT Applied ML, Intel 

{zhan1853, jiajingh, liusiji5}@msu.edu, 

{xin.chen, ke.ding}@intel.com

###### Abstract

In this paper, we study the problem of temporal video grounding(TVG), which aims to predict the starting/ending time points of moments described by a text sentence within a long untrimmed video. Benefiting from fine-grained 3D visual features, the TVG techniques have achieved remarkable progress in recent years. However, the high complexity of 3D convolutional neural networks (CNNs) makes extracting dense 3D visual features time-consuming, which calls for intensive memory and computing resources. Towards efficient TVG, we propose a novel t ext-v isual p rompting (TVP) framework, which incorporates optimized perturbation patterns (that we call ‘prompts’) into both visual inputs and textual features of a TVG model. In sharp contrast to 3D CNNs, we show that TVP allows us to effectively co-train vision encoder and language encoder in a 2D TVG model and improves the performance of crossmodal feature fusion using only low-complexity sparse 2D visual features. Further, we propose a T emporal-D istance IoU(TDIoU) loss for efficient learning of TVG. Experiments on two benchmark datasets, Charades-STA and ActivityNet Captions datasets, empirically show that the proposed TVP significantly boosts the performance of 2D TVG (e.g., 9.79% improvement on Charades-STA and 30.77% improvement on ActivityNet Captions) and achieves 5×5\times 5 × inference acceleration over TVG using 3D visual features. Codes are available at [Open.Intel](https://github.com/intel/TVP).

1 Introduction
--------------

![Image 1: Refer to caption](https://arxiv.org/html/extracted/5151594/fig/fig1_all_final.png)

Figure 1: The architecture and performance comparison among TVG methods: a) 3D TVG methods[[14](https://arxiv.org/html/2303.04995#bib.bib14), [67](https://arxiv.org/html/2303.04995#bib.bib67), [62](https://arxiv.org/html/2303.04995#bib.bib62), [61](https://arxiv.org/html/2303.04995#bib.bib61), [64](https://arxiv.org/html/2303.04995#bib.bib64), [71](https://arxiv.org/html/2303.04995#bib.bib71), [69](https://arxiv.org/html/2303.04995#bib.bib69), [34](https://arxiv.org/html/2303.04995#bib.bib34), [43](https://arxiv.org/html/2303.04995#bib.bib43), [16](https://arxiv.org/html/2303.04995#bib.bib16), [73](https://arxiv.org/html/2303.04995#bib.bib73), [60](https://arxiv.org/html/2303.04995#bib.bib60), [18](https://arxiv.org/html/2303.04995#bib.bib18)], b) 2D TVG methods[[1](https://arxiv.org/html/2303.04995#bib.bib1), [7](https://arxiv.org/html/2303.04995#bib.bib7)], and c) TVP-based 2D TVG(Ours), d) overall performance comparison. Ours is the most efficient(least inference time) and achieves competitive performance compared to 3D TVG methods. In contrast to existing TVG methods, which utilize dense video features extracted by non-trainable offline 3D CNNs and textual features, our proposed framework utilizes a trainable 2D CNN as the vision encoder to extract features from sparsely-sampled video frames with a universal set of frame-aware visual prompts and adds text prompts in textual feature space for end-to-end regression-based modeling.

In recent years, we have witnessed great progress on temporal video grounding(TVG)[[30](https://arxiv.org/html/2303.04995#bib.bib30), [74](https://arxiv.org/html/2303.04995#bib.bib74)]. One key to this success comes from the fine-grained dense 3D visual features extracted by 3D convolutional neural networks (CNNs) (e.g., C3D[[56](https://arxiv.org/html/2303.04995#bib.bib56)] and I3D[[3](https://arxiv.org/html/2303.04995#bib.bib3)]) since TVG tasks demand spatial-temporal context to locate the temporal interval of the moments described by the text query. However, due to the high cost of the dense 3D feature extraction, most existing TVG models only take these 3D visual features extracted by offline 3D CNNs as inputs instead of co-training during TVG model training.

Although models using 3D visual features(that we call ‘3D methods or models’) outperform these using the 2D features(that we call ‘2D methods or models’), a unique advantage of 2D methods is that extracting 2D visual features can significantly reduce the cost in TVG tasks [[14](https://arxiv.org/html/2303.04995#bib.bib14), [74](https://arxiv.org/html/2303.04995#bib.bib74), [30](https://arxiv.org/html/2303.04995#bib.bib30), [62](https://arxiv.org/html/2303.04995#bib.bib62), [61](https://arxiv.org/html/2303.04995#bib.bib61), [75](https://arxiv.org/html/2303.04995#bib.bib75), [69](https://arxiv.org/html/2303.04995#bib.bib69), [34](https://arxiv.org/html/2303.04995#bib.bib34), [15](https://arxiv.org/html/2303.04995#bib.bib15)]. An efficient and lightweight solution with reasonable performance is also demanded in computer vision, NLP, and video-language tasks [[79](https://arxiv.org/html/2303.04995#bib.bib79), [77](https://arxiv.org/html/2303.04995#bib.bib77), [41](https://arxiv.org/html/2303.04995#bib.bib41), [23](https://arxiv.org/html/2303.04995#bib.bib23), [76](https://arxiv.org/html/2303.04995#bib.bib76), [19](https://arxiv.org/html/2303.04995#bib.bib19), [78](https://arxiv.org/html/2303.04995#bib.bib78), [68](https://arxiv.org/html/2303.04995#bib.bib68), [38](https://arxiv.org/html/2303.04995#bib.bib38), [80](https://arxiv.org/html/2303.04995#bib.bib80)]. As discussed above, the methods employing 3D video features are challenging to be employed in practical applications. It thus has significant practical and economic value to develop compact 2D solutions for TVG tasks. In this paper, we ask: {tcolorbox}How to advance 2D TVG methods so as to achieve comparable results to 3D TVG methods?

To address this problem, we propose a novel t ext-v isual p rompting(TVP) framework for training TVG models using 2D visual features. As shown in Fig.[1](https://arxiv.org/html/2303.04995#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Text-Visual Prompting for Efficient 2D Temporal Video Grounding"), for existing 2D TVG and 3D TVG methods, they all utilize offline pretrained vision encoders and language encoders to perform feature extraction. In contrast, our proposed TVP framework is end-to-end trainable. Furthermore, benefiting from text-visual prompting and cross-modal pretraining on large-scale image-text datasets, our proposed framework could achieve comparable performance to 3D TVG methods with significant inference time acceleration.

Conventionally, TVG methods consist of three stages: ① extracting feature from visual and text inputs; ② multi-modal feature fusion; ③ cross-modal modelling. In contrast to conventional methods, TVP incorporates optimized input perturbation patterns (that we call ‘prompts’) into both visual inputs and textual features of a TVG model. We apply trainable parameters in the textual features as text prompts and develop a universal set of frame-aware patterns as visual prompts. Specially, we sample a fixed number of frames from a video and optimize text prompts for the input query sentence and a set of visual prompts for frames with different temporal locations during training. During testing, the same set of optimized visual prompts and textual prompts are applied to all test-time videos. We refer readers to Fig.[2](https://arxiv.org/html/2303.04995#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Text-Visual Prompting for Efficient 2D Temporal Video Grounding") for illustrations of visual prompts and text prompts introduced. To the best of our knowledge, our work makes the first attempt to utilize prompt learning to successfully improve the performance of regression-based TVG tasks using 2D visual features.

Compared to 3D CNNs, 2D CNNs loses spatiotemporal information of the video during feature extraction. Inspired by the success of transformers on the vision-language tasks[[35](https://arxiv.org/html/2303.04995#bib.bib35), [22](https://arxiv.org/html/2303.04995#bib.bib22), [47](https://arxiv.org/html/2303.04995#bib.bib47), [44](https://arxiv.org/html/2303.04995#bib.bib44), [54](https://arxiv.org/html/2303.04995#bib.bib54), [55](https://arxiv.org/html/2303.04995#bib.bib55), [9](https://arxiv.org/html/2303.04995#bib.bib9)] and the recent application of prompt learning to transformers in both vision and language domains[[25](https://arxiv.org/html/2303.04995#bib.bib25), [37](https://arxiv.org/html/2303.04995#bib.bib37), [32](https://arxiv.org/html/2303.04995#bib.bib32), [40](https://arxiv.org/html/2303.04995#bib.bib40), [2](https://arxiv.org/html/2303.04995#bib.bib2), [27](https://arxiv.org/html/2303.04995#bib.bib27)], we choose transformer as our base TVG model and propose to utilize prompts to compensate for the lack of spatiotemporal information in 2D visual features. Furthermore, we develop a T emporal-D istance IoU (TDIoU) loss for training our proposed framework. There are two aspects that distinguish our proposed framework from existing works. First, our proposed framework is designed to boost the performance of the regression-based TVG methods utilizing 2D CNNs as the vision encoder, not for transfer learning[[21](https://arxiv.org/html/2303.04995#bib.bib21), [26](https://arxiv.org/html/2303.04995#bib.bib26), [2](https://arxiv.org/html/2303.04995#bib.bib2)]Second, our proposed framework utilizes 2D CNN to extract visual features from sparsely-sampled video frames, which requires less memory and is easier to be applied in practical applications compared to 3D methods [[62](https://arxiv.org/html/2303.04995#bib.bib62), [61](https://arxiv.org/html/2303.04995#bib.bib61), [75](https://arxiv.org/html/2303.04995#bib.bib75), [60](https://arxiv.org/html/2303.04995#bib.bib60), [69](https://arxiv.org/html/2303.04995#bib.bib69), [34](https://arxiv.org/html/2303.04995#bib.bib34)], especially for long videos. Furthermore, thanks to the compact 2D CNN as the vision encoder, our proposed framework could implement the language encoder and visual encoder co-training for better multimodal feature fusion. In summary, the contributions of this work are unfolded below:

*   •
We propose an effective and efficient framework to train 2D TVG models, in which we leverage TVP (text-visual prompting) to improve the utility of sparse 2D visual features without resorting to costly 3D features. To the best of our knowledge, it is the first work to expand the application of prompt learning for resolving TVG problems. Our method outperforms all of 2D methods and achieves competitive performance to 3D TVG methods.

*   •
Technology-wise, we integrate visual prompt with text prompt to co-improve the effectiveness of 2D visual features. On top of that, we propose TDIoU (temporal-distance IoU)-based prompt-model co-training method to obtain high-accuracy 2D TVG models.

*   •
Experiment-wise, we show the empirical success of our proposal to boost the performance of 2D TVG on Charades-STA and ActivityNet Captions datasets, e.g., 9.79% improvement in Charades-STA, and 30.77%percent 30.77 30.77\%30.77 % in ActivityNet-Captions together with 5×{5\times}5 × inference time acceleration over 3D TVG methods.

![Image 2: Refer to caption](https://arxiv.org/html/extracted/5151594/fig/fig2_tex.png)

(a)Text Prompts

![Image 3: Refer to caption](https://arxiv.org/html/extracted/5151594/fig/fig2_vid.png)

(b)Frame-aware Visual Prompts

Figure 2: Text-visual prompting illustration. (a) Text prompts are directly applied in the feature space. (b) A set of visual prompts are applied to video frames in order. 

2 Related Work
--------------

Video Temporal Grounding (TVG).The objective of the TVG is to predict the starting/ending time points of target moments within an untrimmed video, which is described by a text sentence. Early TVG solutions[[14](https://arxiv.org/html/2303.04995#bib.bib14), [20](https://arxiv.org/html/2303.04995#bib.bib20), [39](https://arxiv.org/html/2303.04995#bib.bib39), [70](https://arxiv.org/html/2303.04995#bib.bib70), [64](https://arxiv.org/html/2303.04995#bib.bib64), [7](https://arxiv.org/html/2303.04995#bib.bib7), [62](https://arxiv.org/html/2303.04995#bib.bib62)] mainly employ two-stage “propose-and-rank” pipeline: ① Propose: utilize sliding windows or proposal network to generate proposal candidates from the input video. ② Rank: the proposed candidates would be ranked according to the text query, and then the proposal with the highest ranking would be the final prediction decision. In contrast to proposal-based methods, regression-based methods[[67](https://arxiv.org/html/2303.04995#bib.bib67), [16](https://arxiv.org/html/2303.04995#bib.bib16), [69](https://arxiv.org/html/2303.04995#bib.bib69)] directly predict the starting/ending time points of the target moments without ranking massive proposal candidates. Thus, regression-based methods are much faster than proposal-based methods, which is one reason why our work focuses on the regression-based TVG. Furthermore, reinforcement learning (RL)-based methods formulate the TVG task as a sequence of decisions to make [[18](https://arxiv.org/html/2303.04995#bib.bib18), [60](https://arxiv.org/html/2303.04995#bib.bib60)]. In particular, they train an agent to control the movement of a window by shifting or scaling. During training, the agent would be rewarded or punished based on whether the window is close to the target moment after an adjustment.

Temporal Action Detection (TAD). TAD aims to determine whether predefined actions occur in a video and to predict the corresponding time intervals during which these actions occur[[53](https://arxiv.org/html/2303.04995#bib.bib53), [63](https://arxiv.org/html/2303.04995#bib.bib63), [56](https://arxiv.org/html/2303.04995#bib.bib56), [48](https://arxiv.org/html/2303.04995#bib.bib48), [59](https://arxiv.org/html/2303.04995#bib.bib59), [12](https://arxiv.org/html/2303.04995#bib.bib12), [13](https://arxiv.org/html/2303.04995#bib.bib13)]. Different from TVG, the input of TAD is only a video. In other words, TAD only requires a semantic understanding of videos. Compared to TAD, TVG is more challenging since it requires a semantic understanding of both videos and natural languages. Furthermore, TVG needs to process the multimodal interaction between videos and natural languages.

Text Prompting. Prompting has recently achieved great success in the domain of natural language processing [[46](https://arxiv.org/html/2303.04995#bib.bib46), [49](https://arxiv.org/html/2303.04995#bib.bib49), [50](https://arxiv.org/html/2303.04995#bib.bib50), [51](https://arxiv.org/html/2303.04995#bib.bib51), [25](https://arxiv.org/html/2303.04995#bib.bib25), [52](https://arxiv.org/html/2303.04995#bib.bib52), [37](https://arxiv.org/html/2303.04995#bib.bib37), [32](https://arxiv.org/html/2303.04995#bib.bib32), [58](https://arxiv.org/html/2303.04995#bib.bib58), [40](https://arxiv.org/html/2303.04995#bib.bib40)]. Text prompting is a process that leverages a data-agnostic perturbation operation applied to text inputs or their embeddings to improve the performance of the downstream task. The simplest way is to construct an input context template originating from human contemplation [[46](https://arxiv.org/html/2303.04995#bib.bib46), [49](https://arxiv.org/html/2303.04995#bib.bib49), [50](https://arxiv.org/html/2303.04995#bib.bib50), [51](https://arxiv.org/html/2303.04995#bib.bib51)]. Although the manually-crafted context templates are simple and interpretable, they are typically not the optimal input prompts. To tackle this issue, other work has focused on searching the optimal prompting in the discrete input space [[25](https://arxiv.org/html/2303.04995#bib.bib25), [52](https://arxiv.org/html/2303.04995#bib.bib52), [58](https://arxiv.org/html/2303.04995#bib.bib58)] or in the language model’s embedding space [[37](https://arxiv.org/html/2303.04995#bib.bib37), [32](https://arxiv.org/html/2303.04995#bib.bib32), [40](https://arxiv.org/html/2303.04995#bib.bib40)].

Visual Prompting. Inspired by the idea of prompt learning in NLP [[37](https://arxiv.org/html/2303.04995#bib.bib37)], visual prompting(VP) was first proposed by Bahng et. al.[[2](https://arxiv.org/html/2303.04995#bib.bib2)] to reprogram a source vision model (e.g., ImageNet-pretrained classifier) to accomplish downstream target tasks (e.g., CIFAR-10 image classification). VP shares almost the same idea with the model reprogramming technology in the vision domain [[6](https://arxiv.org/html/2303.04995#bib.bib6), [11](https://arxiv.org/html/2303.04995#bib.bib11), [57](https://arxiv.org/html/2303.04995#bib.bib57), [81](https://arxiv.org/html/2303.04995#bib.bib81), [65](https://arxiv.org/html/2303.04995#bib.bib65), [72](https://arxiv.org/html/2303.04995#bib.bib72), [4](https://arxiv.org/html/2303.04995#bib.bib4), [5](https://arxiv.org/html/2303.04995#bib.bib5)], which incorporates a universal input perturbation into testing data so as to improve a desired performance metric, e.g., target task accuracy, robustness, and fairness.

Multi-Modal Prompting. Although visual prompting and text prompting have recently attracted much attention, they are under-explored in the multi-modal learning, especially on the temporal video grounding task. The existing works [[2](https://arxiv.org/html/2303.04995#bib.bib2), [27](https://arxiv.org/html/2303.04995#bib.bib27), [66](https://arxiv.org/html/2303.04995#bib.bib66)] mainly focus on integrating text and visual prompts with the CLIP (Contrastive Language–Image Pretrained) model to improve downstream tasks with imagery data. The problem of multi-modal prompting in the video understanding task has not been studied. In this paper, we for the first time develop the text-visual prompting technique to improve the performance of temporal video grounding using 2D visual features.

3 Methods
---------

In this section, we begin with the problem formulation of regression-based TVG. Then we demonstrate the design of TVP (text-visual prompts) and present the overview of our proposed TVP framework.

### 3.1 Problem Definition

Let 𝐯∈ℝ N vid×C×H×W 𝐯 superscript ℝ subscript 𝑁 vid 𝐶 𝐻 𝑊\mathbf{v}\in\mathbb{R}^{N_{\mathrm{vid}}\times C\times H\times W}bold_v ∈ blackboard_R start_POSTSUPERSCRIPT italic_N start_POSTSUBSCRIPT roman_vid end_POSTSUBSCRIPT × italic_C × italic_H × italic_W end_POSTSUPERSCRIPT be an untrimmed vid eo consisting of a sequence of N vid subscript 𝑁 vid N_{\mathrm{vid}}italic_N start_POSTSUBSCRIPT roman_vid end_POSTSUBSCRIPT video frames, and 𝐬∈ℝ N tex 𝐬 superscript ℝ subscript 𝑁 tex\mathbf{s}\in\mathbb{R}^{N_{\mathrm{tex}}}bold_s ∈ blackboard_R start_POSTSUPERSCRIPT italic_N start_POSTSUBSCRIPT roman_tex end_POSTSUBSCRIPT end_POSTSUPERSCRIPT be a tex t query consisting of a sequence of N tex subscript 𝑁 tex N_{\mathrm{tex}}italic_N start_POSTSUBSCRIPT roman_tex end_POSTSUBSCRIPT language tokens. Here, the video-query pair (𝐯,𝐬)𝐯 𝐬(\mathbf{v},\mathbf{s})( bold_v , bold_s ) belongs to a video-language dataset 𝒟 𝒟\mathcal{D}caligraphic_D. Given 𝐯 𝐯\mathbf{v}bold_v and 𝐬 𝐬\mathbf{s}bold_s, TVG aims to predict the time interval 𝐓^=(t^sta,t^end)^𝐓 subscript^𝑡 sta subscript^𝑡 end{\mathbf{\hat{T}}}=(\hat{t}_{\mathrm{sta}},\hat{t}_{\mathrm{end}})over^ start_ARG bold_T end_ARG = ( over^ start_ARG italic_t end_ARG start_POSTSUBSCRIPT roman_sta end_POSTSUBSCRIPT , over^ start_ARG italic_t end_ARG start_POSTSUBSCRIPT roman_end end_POSTSUBSCRIPT ) of the target video moments described by the query 𝐬 𝐬\mathbf{s}bold_s. The TVG model that fuses the vision-language modalities can be described as:

𝐓^=f⁢(g tex⁢(𝐬),g vid⁢(𝐯)),^𝐓 𝑓 subscript 𝑔 tex 𝐬 subscript 𝑔 vid 𝐯\displaystyle\mathbf{\hat{T}}=f(~{}g_{\mathrm{tex}}(\mathbf{s}),~{}g_{\mathrm{% vid}}(\mathbf{v})~{}),\vspace*{-3mm}over^ start_ARG bold_T end_ARG = italic_f ( italic_g start_POSTSUBSCRIPT roman_tex end_POSTSUBSCRIPT ( bold_s ) , italic_g start_POSTSUBSCRIPT roman_vid end_POSTSUBSCRIPT ( bold_v ) ) ,(1)

where f 𝑓 f italic_f denotes TVG model, and g vid subscript 𝑔 vid g_{\mathrm{vid}}italic_g start_POSTSUBSCRIPT roman_vid end_POSTSUBSCRIPT and g tex subscript 𝑔 tex g_{\mathrm{tex}}italic_g start_POSTSUBSCRIPT roman_tex end_POSTSUBSCRIPT represent vision encoder and language encoder, respectively.

![Image 4: Refer to caption](https://arxiv.org/html/x1.png)

Figure 3: Overview of our proposed TVP (text-visual prompting) framework for 2D TVG (temporal video grounding). The whole process contains four phases: ❶ Video frame preprocessing: uniformly sample frames from input video and apply a set of frame-aware visual prompts to the sampled frames in order; ❷ Feature extraction: 2D CNN extracts features from sampled video frames with visual prompts, and the language encoder extracts textual features. In addition, the visual features would be spatially downsampled and temporally fused by max pooling and mean pooling, respectively. ❸ Multimodal feature processing: after spatial downsampling and temporal fusion, the 2D visual features would be integrated into the prompted textual features. ❹ Crossmodal fusion: the multimodal features would be processed by a 12-layer transformer encoder, and MLP would predict the starting/ending time points of the target moment.

### 3.2 TDIoU Loss Function

Conventionally, the TVG model can be learned by minimizing the temporal IoU loss ℒ tIoU subscript ℒ tIoU\mathcal{L}_{\mathrm{tIoU}}caligraphic_L start_POSTSUBSCRIPT roman_tIoU end_POSTSUBSCRIPT defined below:

ℒ tIoU=(1−𝐓^⁢(𝜽)⁢⋂𝐓 𝐓^⁢(𝜽)⁢⋃𝐓),subscript ℒ tIoU 1^𝐓 𝜽 𝐓^𝐓 𝜽 𝐓\displaystyle\mathcal{L}_{\mathrm{tIoU}}=\left(1-\frac{\mathbf{\hat{T}}(% \boldsymbol{\theta})\bigcap\mathbf{T}}{{\mathbf{\hat{T}}(\boldsymbol{\theta})}% \bigcup\mathbf{T}}\right),\vspace*{-3mm}caligraphic_L start_POSTSUBSCRIPT roman_tIoU end_POSTSUBSCRIPT = ( 1 - divide start_ARG over^ start_ARG bold_T end_ARG ( bold_italic_θ ) ⋂ bold_T end_ARG start_ARG over^ start_ARG bold_T end_ARG ( bold_italic_θ ) ⋃ bold_T end_ARG ) ,(2)

where for ease of notation let 𝜽 𝜽\boldsymbol{\theta}bold_italic_θ denote all the trainable parameters involved in ([1](https://arxiv.org/html/2303.04995#S3.E1 "1 ‣ 3.1 Problem Definition ‣ 3 Methods ‣ Text-Visual Prompting for Efficient 2D Temporal Video Grounding")), and 𝐓=(t sta,t end)𝐓 subscript 𝑡 sta subscript 𝑡 end\mathbf{T}=(t_{\mathrm{sta}},t_{\mathrm{end}})bold_T = ( italic_t start_POSTSUBSCRIPT roman_sta end_POSTSUBSCRIPT , italic_t start_POSTSUBSCRIPT roman_end end_POSTSUBSCRIPT ) is the label (i.e., the ground-truth time interval) of the target moment associated with the input video-query pair (𝐯,𝐬)𝐯 𝐬(\mathbf{v},\mathbf{s})( bold_v , bold_s ). The rationale behind ([2](https://arxiv.org/html/2303.04995#S3.E2 "2 ‣ 3.2 TDIoU Loss Function ‣ 3 Methods ‣ Text-Visual Prompting for Efficient 2D Temporal Video Grounding")) is to maximize the overlapping between the predicted time interval and its ground truth.

However, for non-overlapping cases, the temporal IoU loss ℒ tIoU subscript ℒ tIoU\mathcal{L}_{\mathrm{tIoU}}caligraphic_L start_POSTSUBSCRIPT roman_tIoU end_POSTSUBSCRIPT would encounter a gradient vanishing problem. Inspired by [[82](https://arxiv.org/html/2303.04995#bib.bib82)], we develop a novel TDIoU (Temporal-Distance IoU) loss for training our proposed TVG models by incorporating the normalized central time point distance and duration difference between the predicted video clips and the target video clips. We elaborate on the proposed loss below.

Dis tance Loss ℒ dis subscript ℒ normal-dis\mathcal{L}_{\mathrm{dis}}caligraphic_L start_POSTSUBSCRIPT roman_dis end_POSTSUBSCRIPT. To avoid the gradient vanishing problem caused by the non-overlapping case, we involve distance loss ℒ dis subscript ℒ dis\mathcal{L}_{\mathrm{dis}}caligraphic_L start_POSTSUBSCRIPT roman_dis end_POSTSUBSCRIPT to directly minimize the normalized central time point distance. In addition, we add a threshold α 1 subscript 𝛼 1\alpha_{1}italic_α start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT to prevent oscillation in the later training phase. The distance loss is then given by:

ℒ dis=max⁡(|(t sta+t end)/2−(t^sta+t^end)/2||𝐓^⁢⋃𝐓|,α 1),subscript ℒ dis subscript 𝑡 sta subscript 𝑡 end 2 subscript^𝑡 sta subscript^𝑡 end 2^𝐓 𝐓 subscript 𝛼 1\displaystyle\mathcal{L}_{\mathrm{dis}}=\max\left(\frac{|\left(t_{\mathrm{sta}% }+t_{\mathrm{end}}\right)/2-\left(\hat{t}_{\mathrm{sta}}+\hat{t}_{\mathrm{end}% }\right)/2|}{|\mathbf{\hat{T}}\bigcup\mathbf{T}|},~{}\alpha_{1}\right),\vspace% *{-3mm}caligraphic_L start_POSTSUBSCRIPT roman_dis end_POSTSUBSCRIPT = roman_max ( divide start_ARG | ( italic_t start_POSTSUBSCRIPT roman_sta end_POSTSUBSCRIPT + italic_t start_POSTSUBSCRIPT roman_end end_POSTSUBSCRIPT ) / 2 - ( over^ start_ARG italic_t end_ARG start_POSTSUBSCRIPT roman_sta end_POSTSUBSCRIPT + over^ start_ARG italic_t end_ARG start_POSTSUBSCRIPT roman_end end_POSTSUBSCRIPT ) / 2 | end_ARG start_ARG | over^ start_ARG bold_T end_ARG ⋃ bold_T | end_ARG , italic_α start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT ) ,(3)

where recall that 𝐓=(t sta,t end)𝐓 subscript 𝑡 sta subscript 𝑡 end\mathbf{T}=(t_{\mathrm{sta}},t_{\mathrm{end}})bold_T = ( italic_t start_POSTSUBSCRIPT roman_sta end_POSTSUBSCRIPT , italic_t start_POSTSUBSCRIPT roman_end end_POSTSUBSCRIPT ), 𝐓^^𝐓\mathbf{\hat{T}}over^ start_ARG bold_T end_ARG is predicted by the TVG model ([1](https://arxiv.org/html/2303.04995#S3.E1 "1 ‣ 3.1 Problem Definition ‣ 3 Methods ‣ Text-Visual Prompting for Efficient 2D Temporal Video Grounding")), and we choose α 1=0.2 subscript 𝛼 1 0.2\alpha_{1}=0.2 italic_α start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT = 0.2 in experiments.

Dur ation Loss ℒ dur subscript ℒ normal-dur\mathcal{L}_{\mathrm{dur}}caligraphic_L start_POSTSUBSCRIPT roman_dur end_POSTSUBSCRIPT. The introduction of distance loss ℒ dis subscript ℒ dis\mathcal{L}_{\mathrm{dis}}caligraphic_L start_POSTSUBSCRIPT roman_dis end_POSTSUBSCRIPT avoids the gradient vanishing problem but only considers the central time point distance. Yet, this may not be precise enough. For example, even if the central time points are completely overlapped, the duration of two video clips may not be identical. Inspired by the above, we propose the duration loss:

ℒ dur=max⁡(|𝐓−𝐓^⁢(𝜽)||𝐓|,α 2),subscript ℒ dur 𝐓^𝐓 𝜽 𝐓 subscript 𝛼 2\displaystyle\mathcal{L}_{\mathrm{dur}}=\max\left(\frac{|\mathbf{T}-\mathbf{% \hat{T}(\boldsymbol{\theta})}|}{|\mathbf{T}|},~{}\alpha_{2}\right),\vspace*{-3mm}caligraphic_L start_POSTSUBSCRIPT roman_dur end_POSTSUBSCRIPT = roman_max ( divide start_ARG | bold_T - over^ start_ARG bold_T end_ARG ( bold_italic_θ ) | end_ARG start_ARG | bold_T | end_ARG , italic_α start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT ) ,(4)

where α 2 subscript 𝛼 2\alpha_{2}italic_α start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT is the precision tolerance threshold and set by 0.4 in our experiments.

Finally, the proposed Temporal-Distance IoU (TDIoU) loss is given by

ℒ=ℒ tIoU+β 1⁢ℒ dis+β 2⁢ℒ dur,ℒ subscript ℒ tIoU subscript 𝛽 1 subscript ℒ dis subscript 𝛽 2 subscript ℒ dur\displaystyle\mathcal{L}=\mathcal{L}_{\mathrm{tIoU}}+\beta_{1}\mathcal{L}_{% \mathrm{dis}}+\beta_{2}\mathcal{L}_{\mathrm{dur}},caligraphic_L = caligraphic_L start_POSTSUBSCRIPT roman_tIoU end_POSTSUBSCRIPT + italic_β start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT caligraphic_L start_POSTSUBSCRIPT roman_dis end_POSTSUBSCRIPT + italic_β start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT caligraphic_L start_POSTSUBSCRIPT roman_dur end_POSTSUBSCRIPT ,(5)

where β 1>0 subscript 𝛽 1 0\beta_{1}>0 italic_β start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT > 0 and β 2>0 subscript 𝛽 2 0\beta_{2}>0 italic_β start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT > 0 are regularization parameters.

### 3.3 Text-Visual Prompt Design

Inspired by the application of prompts on transformers[[21](https://arxiv.org/html/2303.04995#bib.bib21), [37](https://arxiv.org/html/2303.04995#bib.bib37), [2](https://arxiv.org/html/2303.04995#bib.bib2), [36](https://arxiv.org/html/2303.04995#bib.bib36)], we propose jointly text-visual prompting to boost the performance of our models, in which prompts are optimized perturbation patterns. To improve data processing efficiency, we uniformly sam ple video frames from the untrimmed video 𝐯 𝐯\mathbf{v}bold_v to obtain 𝐯 sam∈ℝ N sam×C×H×W subscript 𝐯 sam superscript ℝ subscript 𝑁 sam 𝐶 𝐻 𝑊\mathbf{v}_{\mathrm{sam}}\in\mathbb{R}^{N_{\mathrm{sam}}\times C\times H\times W}bold_v start_POSTSUBSCRIPT roman_sam end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_N start_POSTSUBSCRIPT roman_sam end_POSTSUBSCRIPT × italic_C × italic_H × italic_W end_POSTSUPERSCRIPT, where N sam subscript 𝑁 sam N_{\mathrm{sam}}italic_N start_POSTSUBSCRIPT roman_sam end_POSTSUBSCRIPT is the number of sampled video frames. In addition, we introduce a set of frame-aware v isual p rompts 𝜹 vp∈ℝ N sam×d vp subscript 𝜹 vp superscript ℝ subscript 𝑁 sam subscript 𝑑 vp\boldsymbol{\delta}_{\mathrm{vp}}\in\mathbb{R}^{N_{\mathrm{sam}}\times d_{% \mathrm{vp}}}bold_italic_δ start_POSTSUBSCRIPT roman_vp end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_N start_POSTSUBSCRIPT roman_sam end_POSTSUBSCRIPT × italic_d start_POSTSUBSCRIPT roman_vp end_POSTSUBSCRIPT end_POSTSUPERSCRIPT in the pixel space of sampled video frames 𝐯 sam subscript 𝐯 sam\mathbf{v}_{\mathrm{sam}}bold_v start_POSTSUBSCRIPT roman_sam end_POSTSUBSCRIPT, and introduce t ext p rompts 𝜹 tp∈ℝ N tp×d tp subscript 𝜹 tp superscript ℝ subscript 𝑁 tp subscript 𝑑 tp\boldsymbol{\delta}_{\mathrm{tp}}\in\mathbb{R}^{N_{\mathrm{tp}}\times d_{% \mathrm{tp}}}bold_italic_δ start_POSTSUBSCRIPT roman_tp end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_N start_POSTSUBSCRIPT roman_tp end_POSTSUBSCRIPT × italic_d start_POSTSUBSCRIPT roman_tp end_POSTSUBSCRIPT end_POSTSUPERSCRIPT in the textual feature space. By incorporating video frame sampling and text-visual prompts into the TVG model ([1](https://arxiv.org/html/2303.04995#S3.E1 "1 ‣ 3.1 Problem Definition ‣ 3 Methods ‣ Text-Visual Prompting for Efficient 2D Temporal Video Grounding")), we obtain:

(t^sta,t^end)=f⁢(𝜹 tp,g tex⁢(𝐬),g vid⁢(𝐯 sam+𝜹 vp)).subscript^𝑡 sta subscript^𝑡 end 𝑓 subscript 𝜹 tp subscript 𝑔 tex 𝐬 subscript 𝑔 vid subscript 𝐯 sam subscript 𝜹 vp\displaystyle(\hat{t}_{\mathrm{sta}},\hat{t}_{\mathrm{end}})=f(~{}\boldsymbol{% \delta}_{\mathrm{tp}},~{}g_{\mathrm{tex}}(\mathbf{s}),~{}g_{\mathrm{vid}}(% \mathbf{v}_{\mathrm{sam}}+\boldsymbol{\delta}_{\mathrm{vp}})~{}).\vspace*{-3mm}( over^ start_ARG italic_t end_ARG start_POSTSUBSCRIPT roman_sta end_POSTSUBSCRIPT , over^ start_ARG italic_t end_ARG start_POSTSUBSCRIPT roman_end end_POSTSUBSCRIPT ) = italic_f ( bold_italic_δ start_POSTSUBSCRIPT roman_tp end_POSTSUBSCRIPT , italic_g start_POSTSUBSCRIPT roman_tex end_POSTSUBSCRIPT ( bold_s ) , italic_g start_POSTSUBSCRIPT roman_vid end_POSTSUBSCRIPT ( bold_v start_POSTSUBSCRIPT roman_sam end_POSTSUBSCRIPT + bold_italic_δ start_POSTSUBSCRIPT roman_vp end_POSTSUBSCRIPT ) ) .(6)

Given a pre-trained 2D TVG model f 𝑓 f italic_f, the objective of text-visual prompting (TVP) is to learn a universal set of visual prompts 𝜹 vp subscript 𝜹 vp\boldsymbol{\delta}_{\mathrm{vp}}bold_italic_δ start_POSTSUBSCRIPT roman_vp end_POSTSUBSCRIPT and text prompts 𝜹 tp subscript 𝜹 tp\boldsymbol{\delta}_{\mathrm{tp}}bold_italic_δ start_POSTSUBSCRIPT roman_tp end_POSTSUBSCRIPT to be integrated into sampled video frames and textual features, respectively. Specially, a set of different visual prompts are applied to uniformly-sampled frames of one untrimmed video in order. During training, only the set of visual prompts and text prompts are updated through backpropagation. During finetuning, prompts are frozen, and the parameters of the TVG model and encoders are updated. During testing, the set of optimized visual prompts and the optimized text prompts are applied to all test-time video-query pairs.

### 3.4 Framework

Inspired by the success of transformers in vision-language tasks, we choose ClipBERT[[31](https://arxiv.org/html/2303.04995#bib.bib31)] as the base model for 2D TVG. Extended from ClipBERT, the input of our regression-based TVG model would be describable sentences and uniformly sampled frames of one untrimmed video as shown in Fig.[3](https://arxiv.org/html/2303.04995#S3.F3 "Figure 3 ‣ 3.1 Problem Definition ‣ 3 Methods ‣ Text-Visual Prompting for Efficient 2D Temporal Video Grounding"). Then, the predicted starting and ending time points of the target video clip would be model outputs. As described in Algorithm[1](https://arxiv.org/html/2303.04995#alg1 "Algorithm 1 ‣ 3.4 Framework ‣ 3 Methods ‣ Text-Visual Prompting for Efficient 2D Temporal Video Grounding"), there are four phases of our proposed TVP framework: ❶ Video frame preprocessing: We obtain sparsely-sampled frames 𝐯 sam subscript 𝐯 sam\mathbf{v}_{\mathrm{sam}}bold_v start_POSTSUBSCRIPT roman_sam end_POSTSUBSCRIPT from one input untrimmed video 𝐯 𝐯\mathbf{v}bold_v, and apply universal frame-aware visual prompts 𝜹 vp subscript 𝜹 vp\boldsymbol{\delta}_{\mathrm{vp}}bold_italic_δ start_POSTSUBSCRIPT roman_vp end_POSTSUBSCRIPT on top of frames at the padding location. ❷ Feature extraction: 2D vision encoder (first 5 ConvBlock of ResNet-50) g vid subscript 𝑔 vid g_{\mathrm{vid}}italic_g start_POSTSUBSCRIPT roman_vid end_POSTSUBSCRIPT and language encoder (a trainable word embedding layer) g tex subscript 𝑔 tex g_{\mathrm{tex}}italic_g start_POSTSUBSCRIPT roman_tex end_POSTSUBSCRIPT would extract features from the prompted frames 𝐯 sam′subscript superscript 𝐯′sam\mathbf{v}^{\prime}_{\mathrm{sam}}bold_v start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT roman_sam end_POSTSUBSCRIPT and textual inputs 𝐬 𝐬\mathbf{s}bold_s, respectively. ❸ Multimodal feature processing: Following the setting of Pixel-BERT[[22](https://arxiv.org/html/2303.04995#bib.bib22)], the 2D visual features 𝐐 vid subscript 𝐐 vid\mathbf{Q}_{\mathrm{vid}}bold_Q start_POSTSUBSCRIPT roman_vid end_POSTSUBSCRIPT are downsampled spatially by a 2×2 2 2 2\times 2 2 × 2 max-pooling layer and fused temporally by a mean-pooling layer. Then, text prompts 𝜹 tp subscript 𝜹 tp\boldsymbol{\delta}_{\mathrm{tp}}bold_italic_δ start_POSTSUBSCRIPT roman_tp end_POSTSUBSCRIPT are integrated into textual features 𝐐 tex subscript 𝐐 tex\mathbf{Q}_{\mathrm{tex}}bold_Q start_POSTSUBSCRIPT roman_tex end_POSTSUBSCRIPT. In addition, trainable 2D visual position embeddings 𝐌 2⁢D subscript 𝐌 2 D\mathbf{M}_{\mathrm{2D}}bold_M start_POSTSUBSCRIPT 2 roman_D end_POSTSUBSCRIPT and textual position embeddings 𝐌 pos subscript 𝐌 pos\mathbf{M}_{\mathrm{pos}}bold_M start_POSTSUBSCRIPT roman_pos end_POSTSUBSCRIPT are applied to the processed 2D visual features 𝐐 vid′subscript superscript 𝐐′vid\mathbf{Q}^{\prime}_{\mathrm{vid}}bold_Q start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT roman_vid end_POSTSUBSCRIPT and prompted textual features 𝐐 tex′subscript superscript 𝐐′tex\mathbf{Q}^{\prime}_{\mathrm{tex}}bold_Q start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT roman_tex end_POSTSUBSCRIPT, respectively [[31](https://arxiv.org/html/2303.04995#bib.bib31), [10](https://arxiv.org/html/2303.04995#bib.bib10)]. Afterwards, the processed and position-encoded 2D visual features 𝐐 vid′′subscript superscript 𝐐′′vid\mathbf{Q}^{\prime\prime}_{\mathrm{vid}}bold_Q start_POSTSUPERSCRIPT ′ ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT roman_vid end_POSTSUBSCRIPT are flattened and integrated into prompted and position-encoded textual features 𝐐 tex′′subscript superscript 𝐐′′tex\mathbf{Q}^{\prime\prime}_{\mathrm{tex}}bold_Q start_POSTSUPERSCRIPT ′ ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT roman_tex end_POSTSUBSCRIPT. Moreover, type embeddings 𝐌 type subscript 𝐌 type\mathbf{M}_{\mathrm{type}}bold_M start_POSTSUBSCRIPT roman_type end_POSTSUBSCRIPT would be added to the integrated multimodal features 𝐐 all subscript 𝐐 all\mathbf{Q}_{\mathrm{all}}bold_Q start_POSTSUBSCRIPT roman_all end_POSTSUBSCRIPT to indicate the source type of features. ❹ Crossmodal fusion: A 12-layer transformer[[10](https://arxiv.org/html/2303.04995#bib.bib10)] is utilized for crossmodal fusion on 𝐐 all subscript 𝐐 all\mathbf{Q}_{\mathrm{all}}bold_Q start_POSTSUBSCRIPT roman_all end_POSTSUBSCRIPT, and then multilayer perceptron (MLP) ending with sigmoid function is used as the prediction head to process the last-layer c ross m odal representation 𝐐 CM subscript 𝐐 CM\mathbf{Q}_{\mathrm{CM}}bold_Q start_POSTSUBSCRIPT roman_CM end_POSTSUBSCRIPT of the transformer for generating the predicted starting/ending time points (t^sta,t^sta)subscript^𝑡 sta subscript^𝑡 sta(\hat{t}_{\mathrm{sta}},\hat{t}_{\mathrm{sta}})( over^ start_ARG italic_t end_ARG start_POSTSUBSCRIPT roman_sta end_POSTSUBSCRIPT , over^ start_ARG italic_t end_ARG start_POSTSUBSCRIPT roman_sta end_POSTSUBSCRIPT ) of the target moments described by the text query input.

Algorithm 1 Overview of TVP framework

1:vision encoder

g vid subscript 𝑔 vid g_{\mathrm{vid}}italic_g start_POSTSUBSCRIPT roman_vid end_POSTSUBSCRIPT
, language encoder

g tex subscript 𝑔 tex g_{\mathrm{tex}}italic_g start_POSTSUBSCRIPT roman_tex end_POSTSUBSCRIPT
, position embeddings

𝐌 pos subscript 𝐌 pos\mathbf{M}_{\mathrm{pos}}bold_M start_POSTSUBSCRIPT roman_pos end_POSTSUBSCRIPT
, 2D position embeddings

𝐌 2⁢D subscript 𝐌 2 D\mathbf{M}_{\mathrm{2D}}bold_M start_POSTSUBSCRIPT 2 roman_D end_POSTSUBSCRIPT
, type embeddings

𝐌 type subscript 𝐌 type\mathbf{M}_{\mathrm{type}}bold_M start_POSTSUBSCRIPT roman_type end_POSTSUBSCRIPT
, transformer

f 𝑓 f italic_f
, prediction head

M⁢L⁢P 𝑀 𝐿 𝑃 MLP italic_M italic_L italic_P
, visual prompts

𝜹 vp subscript 𝜹 vp\boldsymbol{\delta}_{\mathrm{vp}}bold_italic_δ start_POSTSUBSCRIPT roman_vp end_POSTSUBSCRIPT
, text prompts

𝜹 tp subscript 𝜹 tp\boldsymbol{\delta}_{\mathrm{tp}}bold_italic_δ start_POSTSUBSCRIPT roman_tp end_POSTSUBSCRIPT

2:Predicted time interval

𝐓^=(t^sta,t^end)^𝐓 subscript^𝑡 sta subscript^𝑡 end\mathbf{\hat{T}}=(\hat{t}_{\mathrm{sta}},\hat{t}_{\mathrm{end}})over^ start_ARG bold_T end_ARG = ( over^ start_ARG italic_t end_ARG start_POSTSUBSCRIPT roman_sta end_POSTSUBSCRIPT , over^ start_ARG italic_t end_ARG start_POSTSUBSCRIPT roman_end end_POSTSUBSCRIPT )

3:Phase ❶: Video frame preprocessing

4:

𝐯 sam←←subscript 𝐯 sam absent\mathbf{v}_{\mathrm{sam}}\leftarrow bold_v start_POSTSUBSCRIPT roman_sam end_POSTSUBSCRIPT ←
uniformly sample video frames from an untrimmed video

𝐯 𝐯\mathbf{v}bold_v

5:

𝐯 sam′←←subscript superscript 𝐯′sam absent\mathbf{v}^{\prime}_{\mathrm{sam}}\leftarrow bold_v start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT roman_sam end_POSTSUBSCRIPT ←
apply visual prompts

𝜹 vp subscript 𝜹 vp\boldsymbol{\delta}_{\mathrm{vp}}bold_italic_δ start_POSTSUBSCRIPT roman_vp end_POSTSUBSCRIPT
to the sampled video frames

𝐯 sam subscript 𝐯 sam\mathbf{v}_{\mathrm{sam}}bold_v start_POSTSUBSCRIPT roman_sam end_POSTSUBSCRIPT

6:Phase ❷: Feature Extraction

7:

𝐐 vid=g vid⁢(𝐯 sam′)←subscript 𝐐 vid subscript 𝑔 vid subscript superscript 𝐯′sam←absent\mathbf{Q}_{\mathrm{vid}}=g_{\mathrm{vid}}(\mathbf{v}^{\prime}_{\mathrm{sam}})\leftarrow bold_Q start_POSTSUBSCRIPT roman_vid end_POSTSUBSCRIPT = italic_g start_POSTSUBSCRIPT roman_vid end_POSTSUBSCRIPT ( bold_v start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT roman_sam end_POSTSUBSCRIPT ) ←
extracting 2D visual features

8:

𝐐 tex=g tex⁢(𝐬)←subscript 𝐐 tex subscript 𝑔 tex 𝐬←absent\mathbf{Q}_{\mathrm{tex}}=g_{\mathrm{tex}}(\mathbf{s})\leftarrow bold_Q start_POSTSUBSCRIPT roman_tex end_POSTSUBSCRIPT = italic_g start_POSTSUBSCRIPT roman_tex end_POSTSUBSCRIPT ( bold_s ) ←
extracting textual features

9:Phase ❸: Multimodal feature processing

10:

𝐐 vid′←←subscript superscript 𝐐′vid absent\mathbf{Q}^{\prime}_{\mathrm{vid}}\leftarrow bold_Q start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT roman_vid end_POSTSUBSCRIPT ←
apply spatial downsampling and temporal fusion to 2D visual features

𝐐 vid subscript 𝐐 vid\mathbf{Q}_{\mathrm{vid}}bold_Q start_POSTSUBSCRIPT roman_vid end_POSTSUBSCRIPT

11:

𝐐 tex′←←subscript superscript 𝐐′tex absent\mathbf{Q}^{\prime}_{\mathrm{tex}}\leftarrow bold_Q start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT roman_tex end_POSTSUBSCRIPT ←
apply text prompts

𝜹 tp subscript 𝜹 tp\boldsymbol{\delta}_{\mathrm{tp}}bold_italic_δ start_POSTSUBSCRIPT roman_tp end_POSTSUBSCRIPT
to textual features

𝐐 tex subscript 𝐐 tex\mathbf{Q}_{\mathrm{tex}}bold_Q start_POSTSUBSCRIPT roman_tex end_POSTSUBSCRIPT

12:

𝐐 vid′′←←subscript superscript 𝐐′′vid absent\mathbf{Q}^{\prime\prime}_{\mathrm{vid}}\leftarrow bold_Q start_POSTSUPERSCRIPT ′ ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT roman_vid end_POSTSUBSCRIPT ←
add 2D visual position embeddings

𝐌 2⁢D subscript 𝐌 2 D\mathbf{M}_{\mathrm{2D}}bold_M start_POSTSUBSCRIPT 2 roman_D end_POSTSUBSCRIPT
on the processed 2D visual features

𝐐 vid′subscript superscript 𝐐′vid\mathbf{Q}^{\prime}_{\mathrm{vid}}bold_Q start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT roman_vid end_POSTSUBSCRIPT

13:

𝐐 tex′′←←subscript superscript 𝐐′′tex absent\mathbf{Q}^{\prime\prime}_{\mathrm{tex}}\leftarrow bold_Q start_POSTSUPERSCRIPT ′ ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT roman_tex end_POSTSUBSCRIPT ←
add position embeddings

𝐌 pos subscript 𝐌 pos\mathbf{M}_{\mathrm{pos}}bold_M start_POSTSUBSCRIPT roman_pos end_POSTSUBSCRIPT
to prompted textual features

𝐐 tex′subscript superscript 𝐐′tex\mathbf{Q}^{\prime}_{\mathrm{tex}}bold_Q start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT roman_tex end_POSTSUBSCRIPT

14:

𝐐 all←←subscript 𝐐 all absent\mathbf{Q}_{\mathrm{all}}\leftarrow bold_Q start_POSTSUBSCRIPT roman_all end_POSTSUBSCRIPT ←
integrate the processed and position-encoded textual features

𝐐 tex′′subscript superscript 𝐐′′tex\mathbf{Q}^{\prime\prime}_{\mathrm{tex}}bold_Q start_POSTSUPERSCRIPT ′ ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT roman_tex end_POSTSUBSCRIPT
and the processed and position-encoded 2D visual features

𝐐 vid′′subscript superscript 𝐐′′vid\mathbf{Q}^{\prime\prime}_{\mathrm{vid}}bold_Q start_POSTSUPERSCRIPT ′ ′ end_POSTSUPERSCRIPT start_POSTSUBSCRIPT roman_vid end_POSTSUBSCRIPT

15:

𝐐 all+𝐌 type←←subscript 𝐐 all subscript 𝐌 type absent\mathbf{Q}_{\mathrm{all}}+\mathbf{M}_{\mathrm{type}}\leftarrow bold_Q start_POSTSUBSCRIPT roman_all end_POSTSUBSCRIPT + bold_M start_POSTSUBSCRIPT roman_type end_POSTSUBSCRIPT ←
add type embeddings

𝐌 type subscript 𝐌 type\mathbf{M}_{\mathrm{type}}bold_M start_POSTSUBSCRIPT roman_type end_POSTSUBSCRIPT
to the integrated multimodal features

𝐐 all subscript 𝐐 all\mathbf{Q}_{\mathrm{all}}bold_Q start_POSTSUBSCRIPT roman_all end_POSTSUBSCRIPT

16:Phase ❹: Crossmodal fusion

17:

𝐐 CM=f⁢(𝐐 all+𝐌 type)←subscript 𝐐 CM 𝑓 subscript 𝐐 all subscript 𝐌 type←absent\mathbf{Q}_{\mathrm{CM}}=f(\mathbf{Q}_{\mathrm{all}}+\mathbf{M}_{\mathrm{type}% })\leftarrow bold_Q start_POSTSUBSCRIPT roman_CM end_POSTSUBSCRIPT = italic_f ( bold_Q start_POSTSUBSCRIPT roman_all end_POSTSUBSCRIPT + bold_M start_POSTSUBSCRIPT roman_type end_POSTSUBSCRIPT ) ←
implement c ross m odal fusion through transformer

f 𝑓 f italic_f

18:

(t^sta,t^end)=M⁢L⁢P⁢(𝐐 CM)←subscript^𝑡 sta subscript^𝑡 end 𝑀 𝐿 𝑃 subscript 𝐐 CM←absent(\hat{t}_{\mathrm{sta}},\hat{t}_{\mathrm{end}})=MLP(\mathbf{Q}_{\mathrm{CM}})\leftarrow( over^ start_ARG italic_t end_ARG start_POSTSUBSCRIPT roman_sta end_POSTSUBSCRIPT , over^ start_ARG italic_t end_ARG start_POSTSUBSCRIPT roman_end end_POSTSUBSCRIPT ) = italic_M italic_L italic_P ( bold_Q start_POSTSUBSCRIPT roman_CM end_POSTSUBSCRIPT ) ←
prediction head generates the predicted time interval according to crossmodal representation

𝐐 CM subscript 𝐐 CM\mathbf{Q}_{\mathrm{CM}}bold_Q start_POSTSUBSCRIPT roman_CM end_POSTSUBSCRIPT

4 Experiments
-------------

In this section, we demonstrate the effectiveness of our proposed TVP framework on Charades-STA and ActivityNet Captions datasets.

Table 1:  Statistics of TVG benchmark datasets (Charades-STA and ActivityNet Captions datasets). 

### 4.1 Experiment Setup

Datasets. The evaluations are implemented on two standard benchmark datasets for TVG task, Charades-STA[[14](https://arxiv.org/html/2303.04995#bib.bib14)] and ActivityNet Captions[[28](https://arxiv.org/html/2303.04995#bib.bib28)]. Tab.[1](https://arxiv.org/html/2303.04995#S4.T1 "Table 1 ‣ 4 Experiments ‣ Text-Visual Prompting for Efficient 2D Temporal Video Grounding") summarizes the details of both datasets. Charades-STA dataset contains 6,672 6 672 6,672 6 , 672 videos and 16,124 16 124 16,124 16 , 124 text queries in total. The average length of videos is 30.6⁢s 30.6 𝑠 30.6s 30.6 italic_s, and the average length of text query is 7.2⁢w⁢o⁢r⁢d⁢s 7.2 𝑤 𝑜 𝑟 𝑑 𝑠 7.2~{}words 7.2 italic_w italic_o italic_r italic_d italic_s . The average length of moments corresponding to the text query is 8.1⁢s 8.1 𝑠 8.1s 8.1 italic_s. Following the same dataset split as [[14](https://arxiv.org/html/2303.04995#bib.bib14)] for fair comparisons, there are 12,408 12 408 12,408 12 , 408 video-query pairs for training and 3,720 3 720 3,720 3 , 720 pairs for testing. ActivityNet Captions dataset contains 14,926 14 926 14,926 14 , 926 videos and 71,953 71 953 71,953 71 , 953 text queries in total. The average length of videos is 117.6⁢s 117.6 𝑠 117.6s 117.6 italic_s, and the average length of text query is 14.4⁢w⁢o⁢r⁢d⁢s 14.4 𝑤 𝑜 𝑟 𝑑 𝑠 14.4~{}words 14.4 italic_w italic_o italic_r italic_d italic_s. The average length of moments corresponding to the text query is 37.1⁢s 37.1 𝑠 37.1s 37.1 italic_s. ActivityNet Captions dataset is split into training set, validation set, and testing set in a 2:1:1:2 1:1 2:1:1 2 : 1 : 1 ratio. Since the testing set is withheld for competition, only a training set and two validation sets (val1 and val2) can be accessed publicly. For fair comparisons, we evaluate our proposed framework on val1.

Baselines. We compare our proposal with 15 15 15 15 baseline methods: ① Proposal-based: CTRL[[14](https://arxiv.org/html/2303.04995#bib.bib14)], MCN[[1](https://arxiv.org/html/2303.04995#bib.bib1)], SAP[[7](https://arxiv.org/html/2303.04995#bib.bib7)], BPNet[[62](https://arxiv.org/html/2303.04995#bib.bib62)], LPNet[[61](https://arxiv.org/html/2303.04995#bib.bib61)], QSPN[[64](https://arxiv.org/html/2303.04995#bib.bib64)], MAN[[71](https://arxiv.org/html/2303.04995#bib.bib71)]; ② Proposal-free: ABLR[[67](https://arxiv.org/html/2303.04995#bib.bib67)], DRN[[69](https://arxiv.org/html/2303.04995#bib.bib69)], CPNet[[34](https://arxiv.org/html/2303.04995#bib.bib34)], DEBUG[[43](https://arxiv.org/html/2303.04995#bib.bib43)], ExCL[[16](https://arxiv.org/html/2303.04995#bib.bib16)], VSLNet[[73](https://arxiv.org/html/2303.04995#bib.bib73)]; ③ Reinforcement learning: TSP-PRL[[60](https://arxiv.org/html/2303.04995#bib.bib60)], TripNet[[18](https://arxiv.org/html/2303.04995#bib.bib18)].

Evaluation metrics. Following [[14](https://arxiv.org/html/2303.04995#bib.bib14)], we adopt Acc(R@1, IoU=m) as the performance evaluation metric, which represents the percentage accuracy of top-1 1 1 1 predicted moments whose tIoU (temporal IoU) with the ground-truth moment is larger than m 𝑚 m italic_m. By convention, we consider the following tIoU threshold values m={0.3,0.5,0.7}𝑚 0.3 0.5 0.7 m=\{0.3,0.5,0.7\}italic_m = { 0.3 , 0.5 , 0.7 }.

Crossmodal pretraining setup. Our 2D vision encoder (ResNet-50) is initialized with the weight from grid-feat[[24](https://arxiv.org/html/2303.04995#bib.bib24)], which can extract effective grid features from visual inputs. In addition, both the language encoder and 12-layer transformer are initialized with the BERT-base model weight[[10](https://arxiv.org/html/2303.04995#bib.bib10)], which are pretrained on English Wikipedia and BookCorpus[[83](https://arxiv.org/html/2303.04995#bib.bib83)]. Thanks to the compact 2D vision encoder, TVP (our proposal) is able to directly utilize image-text pairs for end-to-end training. Since the benefits of cross-modal pretraining has been demonstrated by [[22](https://arxiv.org/html/2303.04995#bib.bib22), [44](https://arxiv.org/html/2303.04995#bib.bib44), [55](https://arxiv.org/html/2303.04995#bib.bib55)], our base model is pretrained on two large-scale image-text datasets, which are Visual Genome Captions[[29](https://arxiv.org/html/2303.04995#bib.bib29)] and COCO Captions[[8](https://arxiv.org/html/2303.04995#bib.bib8)]. To be more specific, image-text matching[[55](https://arxiv.org/html/2303.04995#bib.bib55), [44](https://arxiv.org/html/2303.04995#bib.bib44)] and masked language modeling[[10](https://arxiv.org/html/2303.04995#bib.bib10)] are employed for cross-modal pretraining.

Implementation setup. For video inputs, we uniformly sample N sam subscript 𝑁 sam N_{\mathrm{sam}}italic_N start_POSTSUBSCRIPT roman_sam end_POSTSUBSCRIPT frames from a video (N sam=48 subscript 𝑁 sam 48 N_{\mathrm{sam}}=48 italic_N start_POSTSUBSCRIPT roman_sam end_POSTSUBSCRIPT = 48 for Charades-STA and N sam=64 subscript 𝑁 sam 64 N_{\mathrm{sam}}=64 italic_N start_POSTSUBSCRIPT roman_sam end_POSTSUBSCRIPT = 64 for ActivityNet Captions). In addition, all video frames are resized to have a maximum longer side of 448 448 448 448 with an original aspect ratio, and then the frames are zero-padded to 448×448 448 448 448\times 448 448 × 448. The default visual prompt sizes for both dataset are 96 96 96 96. The default text prompt sizes are 10 10 10 10 and 20 20 20 20 for Charades-STA and ActivityNet Captions, respectively. We utilize the first 5 ConvBlocks of ResNet-50 as the 2D vision encoder and a trainable embedding layer as the language encoder for both Charades-STA and ActivityNet Captions datasets. For text queries, all word tokens are maintained after lower-case conversion and tokenization. We use AdamW[[42](https://arxiv.org/html/2303.04995#bib.bib42)] for end-to-end model training, with β 1=1.0 subscript 𝛽 1 1.0\beta_{1}=1.0 italic_β start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT = 1.0, β 2=0.1 subscript 𝛽 2 0.1\beta_{2}=0.1 italic_β start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT = 0.1, α 1=0.2 subscript 𝛼 1 0.2\alpha_{1}=0.2 italic_α start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT = 0.2, α 2=0.4 subscript 𝛼 2 0.4\alpha_{2}=0.4 italic_α start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT = 0.4. Initial learning rates are 1⁢e−1 1 𝑒 1 1e-1 1 italic_e - 1 and 5⁢e−7 5 𝑒 7 5e-7 5 italic_e - 7 for prompt training and model finetuning, respectively. In addition, the learning rate linearly decays to 0 0 with the first 10%percent 10 10\%10 % training step for warmup. Our experiments are implemented in PyTorch[[45](https://arxiv.org/html/2303.04995#bib.bib45)], and models and prompts are finetuned separately for 12 12 12 12 epochs with the mixed precision on 8 NVIDIA V100 GPUs.

### 4.2 Experiment Results

Table 2:  Performance comparison of different thresholds m 𝑚 m italic_m on the Charades-STA dataset. 

Table 3:  Performance comparison of different thresholds m 𝑚 m italic_m on the ActivityNet Captions dataset. 

Effectiveness of TVP on Charades-STA. The performance comparisons with SOTA methods on the Charades-STA dataset are summarized in Tab.[2](https://arxiv.org/html/2303.04995#S4.T2 "Table 2 ‣ 4.2 Experiment Results ‣ 4 Experiments ‣ Text-Visual Prompting for Efficient 2D Temporal Video Grounding"). Our proposed TVP framework can achieve competitive performance at all tIoU thresholds m 𝑚 m italic_m in the case of utilizing 2D visual features extracted by ResNet-50, and reach the highest score at m=0.3 𝑚 0.3 m=0.3 italic_m = 0.3. Compared to the 2D TVG methods using VGG as the vision encoder, our proposed framework could achieve around 2.5×2.5\times 2.5 × and 2.7×2.7\times 2.7 × performance gain at thresholds 0.5 0.5 0.5 0.5 and 0.7 0.7 0.7 0.7, respectively. Furthermore, we can find that for our base model only one of visual prompts and text prompts can achieve up to 7.37%percent 7.37 7.37\%7.37 % and 9.60%percent 9.60 9.60\%9.60 % improvement at tIoU thresholds m=0.3 𝑚 0.3 m=0.3 italic_m = 0.3 and m=0.5 𝑚 0.5 m=0.5 italic_m = 0.5. The combination of text and visual prompts can not only achieves 7.55%percent 7.55 7.55\%7.55 % and 9.79%percent 9.79 9.79\%9.79 % improvements at tIoU thresholds m=0.3 𝑚 0.3 m=0.3 italic_m = 0.3 and m=0.5 𝑚 0.5 m=0.5 italic_m = 0.5, but also improve the performance by 8.14%percent 8.14 8.14\%8.14 % at m=0.7 𝑚 0.7 m=0.7 italic_m = 0.7. This demonstrates the effectiveness and necessity of the joint text-visual prompting.

![Image 5: Refer to caption](https://arxiv.org/html/x2.png)

(a)Charades-STA

![Image 6: Refer to caption](https://arxiv.org/html/x3.png)

(b)ActivityNet Captions

Figure 4: Impact of sampled frame numbers.

Effectiveness of TVP on ActivityNet Captions. We focus on the performance comparisons with 3D TVG methods on ActivityNet since there are no results of 2D TVG method reported on ActivityNet Captions. The results of multiple methods on ActivityNet Captions datasets are reported in Tab.[3](https://arxiv.org/html/2303.04995#S4.T3 "Table 3 ‣ 4.2 Experiment Results ‣ 4 Experiments ‣ Text-Visual Prompting for Efficient 2D Temporal Video Grounding"). Even on this more challenging dataset, our proposed method still has achieved competitive performance compared to 3D TVG methods. Different from the performance of TVP on Charades-STA dataset, text prompts or visual prompts can achieve a significant performance boost on the base model over all IoU thresholds m 𝑚 m italic_m alone (5.73%percent 5.73 5.73\%5.73 % at m=0.3 𝑚 0.3 m=0.3 italic_m = 0.3, 8.04%percent 8.04 8.04\%8.04 % at m=0.5 𝑚 0.5 m=0.5 italic_m = 0.5, 27.43%percent 27.43 27.43\%27.43 % at m=0.7 𝑚 0.7 m=0.7 italic_m = 0.7 ) , and the text-visual prompt combination could further boost the performance (6.14%percent 6.14 6.14\%6.14 % at m=0.3 𝑚 0.3 m=0.3 italic_m = 0.3, 8.17%percent 8.17 8.17\%8.17 % at m=0.5 𝑚 0.5 m=0.5 italic_m = 0.5, 30.77%percent 30.77 30.77\%30.77 % at m=0.7 𝑚 0.7 m=0.7 italic_m = 0.7). It is worth noting that the performance gap over m=0.7 𝑚 0.7 m=0.7 italic_m = 0.7 between 2D TVG methods and 3D TVG methods is narrowed significantly.

In summary, through the experimental results on Charades-STA and ActivityNet Captions datasets, we can find that our proposed TVP framework could achieve competitive performance overall tIoU thresholds on Charades-STA and ActivityNet Captions by improving the utility of sparse 2D visual features. Thanks to the lightweight 2D vision encoder, cotraining language encoder and vision encoder on large-scale image-text datasets can be performed, which benefits the base model to achieve good performance. Furthermore, the combination of text and visual prompts can achieve better results than any single kind of prompts on both datasets, which again proves the importance of crossmodal training.

Video frame sampling effect.Fig.[4](https://arxiv.org/html/2303.04995#S4.F4 "Figure 4 ‣ 4.2 Experiment Results ‣ 4 Experiments ‣ Text-Visual Prompting for Efficient 2D Temporal Video Grounding") demonstrates the performance of base model with different number N sam subscript 𝑁 sam N_{\mathrm{sam}}italic_N start_POSTSUBSCRIPT roman_sam end_POSTSUBSCRIPT of sampled video frames as visual inputs. For Charades dataset, the base model performance keeps increasing before N sam subscript 𝑁 sam N_{\mathrm{sam}}italic_N start_POSTSUBSCRIPT roman_sam end_POSTSUBSCRIPT reaches 48, but when it exceeds 48, performance starts to degrade. This is because frequent background changes harm the performance of object re-identification in videos, which are noisy for object motion analysis [[17](https://arxiv.org/html/2303.04995#bib.bib17)].

For ActivityNet Caption dataset, base model performance continues to improve even when sampled frame number N sam subscript 𝑁 sam N_{\mathrm{sam}}italic_N start_POSTSUBSCRIPT roman_sam end_POSTSUBSCRIPT exceeds 48, due to the longer average video length in ActivityNet Captions dataset. Balancing the frame number and batch size for training, we choose N sam=64 subscript 𝑁 sam 64 N_{\mathrm{sam}}=64 italic_N start_POSTSUBSCRIPT roman_sam end_POSTSUBSCRIPT = 64 for ActivityNet Captions.

Table 4:  The performance comparison of different visual prompt sizes on Charades-STA dataset. 

Visual Prompt Size Acc(R@1, IoU=m 𝑚 m italic_m)Prompt + Frame
m 𝑚 m italic_m=0.3 m 𝑚 m italic_m=0.5 m 𝑚 m italic_m=0.7
0 61.29 40.43 19.89![Image 7: [Uncaptioned image]](https://arxiv.org/html/x4.png)
16 61.29 40.43 20.00![Image 8: [Uncaptioned image]](https://arxiv.org/html/x5.png)
32 61.94 39.78 19.35![Image 9: [Uncaptioned image]](https://arxiv.org/html/x6.png)
48 63.66 42.37 20.00![Image 10: [Uncaptioned image]](https://arxiv.org/html/x7.png)
72 63.87 43.66 19.78![Image 11: [Uncaptioned image]](https://arxiv.org/html/x8.png)
96 65.38 44.31 20.22![Image 12: [Uncaptioned image]](https://arxiv.org/html/x9.png)
128 64.73 43.66 19.78![Image 13: [Uncaptioned image]](https://arxiv.org/html/x10.png)

Table 5:  The performance comparison of different text prompt sizes on Charades-STA dataset. 

TVP performance vs. prompt size. As shown in Tab.[4](https://arxiv.org/html/2303.04995#S4.T4 "Table 4 ‣ 4.2 Experiment Results ‣ 4 Experiments ‣ Text-Visual Prompting for Efficient 2D Temporal Video Grounding"), we can find that when visual prompts are small, they cannot bring changes to the base model, and when visual prompts are too large, the performance starts to decrease. This is because key information within video frames might be removed. However, the text prompts can bring significant performance boost even when the text prompt size is small as shown in Tab.[5](https://arxiv.org/html/2303.04995#S4.T5 "Table 5 ‣ 4.2 Experiment Results ‣ 4 Experiments ‣ Text-Visual Prompting for Efficient 2D Temporal Video Grounding"), which is because the textual features has a smaller dimension compared to visual features, and also the text prompts are directly optimized in feature space during training.

Table 6:  The performance comparison of different visual prompt operations (‘remove’, ‘add’, ‘replace’) with fixed visual prompt size p=96 𝑝 96 p=96 italic_p = 96 on Charades-STA and ActivityNet Captions datasets. 

TVP performance vs. visual prompt operation. Visual prompt is first proposed by [[2](https://arxiv.org/html/2303.04995#bib.bib2)], where visual prompts are added to the image for transfer learning on classification tasks. In contrast, our proposed prompting framework is designed to compensate for the spatiotemporal information loss in 2D visual features. Due to the differences in the task, we try two different prompt operation strategies, ‘replace’ and ‘add’. ‘add’ is to add the visual prompts to the pixel value of the video frame at the corresponding padding locations. ‘replace’ is to replace the pixel values of video frames with visual prompts at corresponding padding locations. ‘remove’ is in order to study the impact of removing the pixel values at the padding location. As shown in Tab.[6](https://arxiv.org/html/2303.04995#S4.T6 "Table 6 ‣ 4.2 Experiment Results ‣ 4 Experiments ‣ Text-Visual Prompting for Efficient 2D Temporal Video Grounding"), ‘add’ or ‘remove’ prompt operations have limited effects on the base model. However, ‘replace’ does boost the base model performance.

![Image 14: Refer to caption](https://arxiv.org/html/x11.png)

Figure 5:  Inference time comparison. (a) inference time comparison between 2D vision encoder (ResNet-50) and 3D vision encoder (C3D). (b)inference time comparison between the vision encoder and the other modules of the 2D TVG model, where the sampled frame number for our TVP framework is 1.2×1.2\times 1.2 × the length of the video in seconds. 

TVP achieves inference efficiency. As shown in Fig.[5](https://arxiv.org/html/2303.04995#S4.F5 "Figure 5 ‣ 4.2 Experiment Results ‣ 4 Experiments ‣ Text-Visual Prompting for Efficient 2D Temporal Video Grounding"), we can find that the inference time required for visual feature extraction accounts for more than half of the inference time of the whole model, while the inference time required for the 3D vision encoder is more than 5×5\times 5 × compared to the 2D vision encoder, and even more than the time required for the whole TVG model using 2D vision encoder, which fully demonstrates the feasibility of accelerating the overall inference speed by reducing the complexity of the vision encoder. Need to note that if there are multiple model weights for different sampled frame number settings and model weights can be adopted adaptively for different lengths of videos, the inference speed for short videos should increase, and the prediction results for long videos will be further improved.

Ablation studies. Through Tab. [7](https://arxiv.org/html/2303.04995#S4.T7 "Table 7 ‣ 4.2 Experiment Results ‣ 4 Experiments ‣ Text-Visual Prompting for Efficient 2D Temporal Video Grounding"), we can find that the addition of either distance loss ℒ dis subscript ℒ dis\mathcal{L}_{\mathrm{dis}}caligraphic_L start_POSTSUBSCRIPT roman_dis end_POSTSUBSCRIPT or duration loss ℒ dur subscript ℒ dur\mathcal{L}_{\mathrm{dur}}caligraphic_L start_POSTSUBSCRIPT roman_dur end_POSTSUBSCRIPT will result in a performance increase, but the combination of the two will result in a significant performance increase (11.34%percent 11.34 11.34\%11.34 % at m=0.3 𝑚 0.3 m=0.3 italic_m = 0.3, 35.26%percent 35.26 35.26\%35.26 % at m=0.5 𝑚 0.5 m=0.5 italic_m = 0.5, 68.27%percent 68.27 68.27\%68.27 % at m=0.7 𝑚 0.7 m=0.7 italic_m = 0.7, ), especially over tIoU thresholds m=0.5 𝑚 0.5 m=0.5 italic_m = 0.5 and m=0.7 𝑚 0.7 m=0.7 italic_m = 0.7. This demonstrates that distance loss ℒ dis subscript ℒ dis\mathcal{L}_{\mathrm{dis}}caligraphic_L start_POSTSUBSCRIPT roman_dis end_POSTSUBSCRIPT and duration loss ℒ dur subscript ℒ dur\mathcal{L}_{\mathrm{dur}}caligraphic_L start_POSTSUBSCRIPT roman_dur end_POSTSUBSCRIPT could provide more precise training guides compared to only using temporal IoU loss ℒ tIoU subscript ℒ tIoU\mathcal{L}_{\mathrm{tIoU}}caligraphic_L start_POSTSUBSCRIPT roman_tIoU end_POSTSUBSCRIPT. Furthermore, we posit that prompting may encode additional spatial-temporal supervision to help the model trainer to escape from bad local optima as shown in Fig.[6](https://arxiv.org/html/2303.04995#S4.F6 "Figure 6 ‣ 4.2 Experiment Results ‣ 4 Experiments ‣ Text-Visual Prompting for Efficient 2D Temporal Video Grounding"), where fine-tuning w/ prompts yields a flatter loss landscape than the one w/o prompts.

Table 7:  The performance comparison of different loss designs on Charades-STA dataset. 

![Image 15: Refer to caption](https://arxiv.org/html/x12.png)![Image 16: Refer to caption](https://arxiv.org/html/x13.png)

Figure 6: Loss landscape visualization in 2D plane: Finetuning w/o prompts (left) and using prompts (right); see [[33](https://arxiv.org/html/2303.04995#bib.bib33)] for implementation. 

5 Conclusion
------------

In this paper, we propose text-visual prompting to boost the performance of 2D TVG methods by compensating for the lack of spatiotemporal information in 2D visual features. In contrast to 3D TVG methods, TVP allows us to effectively co-train vision encoder and language encoder in a 2D TVG model and improves the performance of cross- modal feature fusion using only low-complexity sparse 2D visual features. The effectiveness of our proposed TVP(text-visual prompting) framework has been demonstrated on two standard datasets, Charades-STA and ActivityNet. Our models outperform all 2D models significantly, and also achieve comparable performance to 3D models. What is more, we achieve over 5×5\times 5 × inference speedup over TVG methods of using 3D visual features.

References
----------

*   [1] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In Proceedings of the IEEE international conference on computer vision, pages 5803–5812, 2017. 
*   [2] H Bahng, A Jahanian, S Sankaranarayanan, and P Isola. Exploring visual prompts for adapting large-scale models. arXiv preprint arXiv:2203.17274, page 2022, 2022. 
*   [3] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6299–6308, 2017. 
*   [4] Aochuan Chen, Peter Lorenz, Yuguang Yao, Pin-Yu Chen, and Sijia Liu. Visual prompting for adversarial robustness. arXiv preprint arXiv:2210.06284, 2022. 
*   [5] Aochuan Chen, Yuguang Yao, Pin-Yu Chen, Yihua Zhang, and Sijia Liu. Understanding and improving visual prompting: A label-mapping perspective. arXiv preprint arXiv:2211.11635, 2022. 
*   [6] Pin-Yu Chen. Model reprogramming: Resource-efficient cross-domain machine learning. arXiv:2202.10629, 2022. 
*   [7] Shaoxiang Chen and Yu-Gang Jiang. Semantic proposal for activity localization in videos via sentence query. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 8199–8206, 2019. 
*   [8] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015. 
*   [9] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. Uniter: Universal image-text representation learning. In European conference on computer vision, pages 104–120. Springer, 2020. 
*   [10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 
*   [11] Gamaleldin F Elsayed, Ian Goodfellow, et al. Adversarial reprogramming of neural networks. arXiv:1806.11146, 2018. 
*   [12] Christoph Feichtenhofer. X3d: Expanding architectures for efficient video recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020. 
*   [13] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6202–6211, 2019. 
*   [14] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query. In Proceedings of the IEEE international conference on computer vision, pages 5267–5275, 2017. 
*   [15] Yanjun Gao, Lulu Liu, Jason Wang, Xin Chen, Huayan Wang, and Rui Zhang. Evoquer: Enhancing temporal grounding with video-pivoted backquery generation. arXiv preprint arXiv:2109.04600, 2021. 
*   [16] Soham Ghosh, Anuva Agarwal, Zarana Parekh, and Alexander Hauptmann. Excl: Extractive clip localization using natural language descriptions. arXiv preprint arXiv:1904.02755, 2019. 
*   [17]Xinqian Gu, Hong Chang, Bingpeng Ma, Hongkai Zhang, and Xilin Chen. Appearance-preserving 3d convolution for video-based person re-identification. In ECCV, 2020. 
*   [18] Meera Hahn, Asim Kadav, James M Rehg, and Hans Peter Graf. Tripping through time: Efficient localization of activities in videos. arXiv preprint arXiv:1904.09936, 2019. 
*   [19] Xiaochen Han, Bo Wu, Zheng Shou, Xiao-Yang Liu, Yimeng Zhang, and Linghe Kong. Tensor fista-net for real-time snapshot compressive imaging. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 10933–10940, 2020. 
*   [20] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with temporal language. arXiv preprint arXiv:1809.01337, 2018. 
*   [21] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for nlp. In International Conference on Machine Learning, pages 2790–2799. PMLR, 2019. 
*   [22] Zhicheng Huang, Zhaoyang Zeng, Bei Liu, Dongmei Fu, and Jianlong Fu. Pixel-bert: Aligning image pixels with text by deep multi-modal transformers. arXiv preprint arXiv:2004.00849, 2020. 
*   [23] Jinghan Jia, Mingyi Hong, Yimeng Zhang, Mehmet Akçakaya, and Sijia Liu. On the robustness of deep learning-based mri reconstruction to image transformations. arXiv preprint arXiv:2211.04930, 2022. 
*   [24] Huaizu Jiang, Ishan Misra, Marcus Rohrbach, Erik Learned-Miller, and Xinlei Chen. In defense of grid features for visual question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10267–10276, 2020. 
*   [25]Zhengbao Jiang, Frank F Xu, Jun Araki, and Graham Neubig. How can we know what language models know? Transactions of the Association for Computational Linguistics, 8:423–438, 2020. 
*   [26] Chen Ju, Tengda Han, Kunhao Zheng, Ya Zhang, and Weidi Xie. Prompting visual-language models for efficient video understanding. arXiv preprint arXiv:2112.04478, 2021. 
*   [27] Muhammad Uzair Khattak, Hanoona Rasheed, Muhammad Maaz, Salman Khan, and Fahad Shahbaz Khan. Maple: Multi-modal prompt learning. arXiv preprint arXiv:2210.03117, 2022. 
*   [28] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In Proceedings of the IEEE international conference on computer vision, pages 706–715, 2017. 
*   [29] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123(1):32–73, 2017. 
*   [30] Xiaohan Lan, Yitian Yuan, Xin Wang, Zhi Wang, and Wenwu Zhu. A survey on temporal sentence grounding in videos. ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 2021. 
*   [31] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L Berg, Mohit Bansal, and Jingjing Liu. Less is more: Clipbert for video-and-language learning via sparse sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7331–7341, 2021. 
*   [32] Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691, 2021. 
*   [33] Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, and Tom Goldstein. Visualizing the loss landscape of neural nets. Advances in neural information processing systems, 2018. 
*   [34] Kun Li, Dan Guo, and Meng Wang. Proposal-free video grounding with contextual pyramid network. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 1902–1910, 2021. 
*   [35] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. Visualbert: A simple and performant baseline for vision and language. arXiv preprint arXiv:1908.03557, 2019. 
*   [36] Muheng Li, Lei Chen, Yueqi Duan, Zhilan Hu, Jianjiang Feng, Jie Zhou, and Jiwen Lu. Bridge-prompt: Towards ordinal action understanding in instructional videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. 
*   [37] Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190, 2021. 
*   [38] Yanyu Li, Pu Zhao, Geng Yuan, Xue Lin, Yanzhi Wang, and Xin Chen. Pruning-as-search: Efficient neural architecture search via channel pruning and structural reparameterization. arXiv preprint arXiv:2206.01198, 2022. 
*   [39] Meng Liu, Xiang Wang, Liqiang Nie, Qi Tian, Baoquan Chen, and Tat-Seng Chua. Cross-modal moment localization in videos. In Proceedings of the 26th ACM international conference on Multimedia, pages 843–851, 2018. 
*   [40] Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, and Jie Tang. Gpt understands, too. arXiv preprint arXiv:2103.10385, 2021. 
*   [41] Xiao-Yang Liu, Yimeng Zhang, Yukang Liao, and Ling Jiang. Dynamic updating of the knowledge base for a large-scale question answering system. ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP), 2020. 
*   [42] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 
*   [43] Chujie Lu, Long Chen, Chilie Tan, Xiaolin Li, and Jun Xiao. Debug: A dense bottom-up grounding approach for natural language video localization. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5144–5153, 2019. 
*   [44] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. Advances in neural information processing systems, 32, 2019. 
*   [45] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 2019. 
*   [46] Fabio Petroni, Tim Rocktäschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H Miller, and Sebastian Riedel. Language models as knowledge bases? arXiv preprint arXiv:1909.01066, 2019. 
*   [47] Di Qi, Lin Su, Jia Song, Edward Cui, Taroon Bharti, and Arun Sacheti. Imagebert: Cross-modal pre-training with large-scale weak-supervised image-text data. arXiv preprint arXiv:2001.07966, 2020. 
*   [48] Zhaofan Qiu, Ting Yao, and Tao Mei. Learning spatio-temporal representation with pseudo-3d residual networks. In proceedings of the IEEE International Conference on Computer Vision, 2017. 
*   [49] Timo Schick and Hinrich Schütze. Exploiting cloze questions for few shot text classification and natural language inference. arXiv preprint arXiv:2001.07676, 2020. 
*   [50] Timo Schick and Hinrich Schütze. Few-shot text generation with pattern-exploiting training. arXiv preprint arXiv:2012.11926, 2020. 
*   [51] Timo Schick and Hinrich Schütze. It’s not just size that matters: Small language models are also few-shot learners. arXiv preprint arXiv:2009.07118, 2020. 
*   [52] Taylor Shin, Yasaman Razeghi, Robert L Logan IV, Eric Wallace, and Sameer Singh. Autoprompt: Eliciting knowledge from language models with automatically generated prompts. arXiv preprint arXiv:2010.15980, 2020. 
*   [53] Karen Simonyan and Andrew Zisserman. Two-stream convolutional networks for action recognition in videos. Advances in neural information processing systems, 2014. 
*   [54] Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. Vl-bert: Pre-training of generic visual-linguistic representations. arXiv preprint arXiv:1908.08530, 2019. 
*   [55] Hao Tan and Mohit Bansal. Lxmert: Learning cross-modality encoder representations from transformers. arXiv preprint arXiv:1908.07490, 2019. 
*   [56] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri. Learning spatiotemporal features with 3d convolutional networks. In Proceedings of the IEEE international conference on computer vision, pages 4489–4497, 2015. 
*   [57] Yun-Yun Tsai et al. Transfer learning without knowing: Reprogramming black-box machine learning models with scarce data and limited resources. arXiv:2007.08714, 2020. 
*   [58] Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, and Sameer Singh. Universal adversarial triggers for attacking and analyzing nlp. arXiv preprint arXiv:1908.07125, 2019. 
*   [59] Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. Non-local neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2018. 
*   [60] Jie Wu, Guanbin Li, Si Liu, and Liang Lin. Tree-structured policy based progressive reinforcement learning for temporally language grounding in video. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 12386–12393, 2020. 
*   [61] Shaoning Xiao, Long Chen, Jian Shao, Yueting Zhuang, and Jun Xiao. Natural language video localization with learnable moment proposals. arXiv preprint arXiv:2109.10678, 2021. 
*   [62] Shaoning Xiao, Long Chen, Songyang Zhang, Wei Ji, Jian Shao, Lu Ye, and Jun Xiao. Boundary proposal network for two-stage natural language video localization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 2986–2994, 2021. 
*   [63] Saining Xie, Chen Sun, Jonathan Huang, Zhuowen Tu, and Kevin Murphy. Rethinking spatiotemporal feature learning: Speed-accuracy trade-offs in video classification. In Proceedings of the European conference on computer vision (ECCV), 2018. 
*   [64] Huijuan Xu, Kun He, Bryan A Plummer, Leonid Sigal, Stan Sclaroff, and Kate Saenko. Multilevel language and vision integration for text-to-clip retrieval. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 9062–9069, 2019. 
*   [65] Chao-Han Huck Yang, Yun-Yun Tsai, et al. Voice2series: Reprogramming acoustic models for time series classification. In ICML. PMLR, 2021. 
*   [66] Yuan Yao, Ao Zhang, Zhengyan Zhang, Zhiyuan Liu, Tat-Seng Chua, and Maosong Sun. Cpt: Colorful prompt tuning for pre-trained vision-language models. arXiv preprint arXiv:2109.11797, 2021. 
*   [67] Yitian Yuan, Tao Mei, and Wenwu Zhu. To find where you talk: Temporal sentence localization in video with attention based location regression. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 9159–9166, 2019. 
*   [68] Ofir Zafrir, Ariel Larey, Guy Boudoukh, Haihao Shen, and Moshe Wasserblat. Prune once for all: Sparse pre-trained language models. arXiv preprint arXiv:2111.05754, 2021. 
*   [69] Runhao Zeng, Haoming Xu, Wenbing Huang, Peihao Chen, Mingkui Tan, and Chuang Gan. Dense regression network for video grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10287–10296, 2020. 
*   [70] Yawen Zeng, Da Cao, Xiaochi Wei, Meng Liu, Zhou Zhao, and Zheng Qin. Multi-modal relational graph for cross-modal video moment retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2215–2224, 2021. 
*   [71] Da Zhang, Xiyang Dai, Xin Wang, Yuan-Fang Wang, and Larry S Davis. Man: Moment alignment network for natural language moment retrieval via iterative graph adjustment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1247–1257, 2019. 
*   [72] Guanhua Zhang, Yihua Zhang, Yang Zhang, Wenqi Fan, Qing Li, Sijia Liu, and Shiyu Chang. Fairness reprogramming. arXiv preprint arXiv:2209.10222, 2022. 
*   [73] Hao Zhang, Aixin Sun, Wei Jing, and Joey Tianyi Zhou. Span-based localizing network for natural language video localization. arXiv preprint arXiv:2004.13931, 2020. 
*   [74] Hao Zhang, Aixin Sun, Wei Jing, and Joey Tianyi Zhou. The elements of temporal sentence grounding in videos: A survey and future directions. arXiv preprint arXiv:2201.08071, 2022. 
*   [75] Songyang Zhang, Houwen Peng, Jianlong Fu, and Jiebo Luo. Learning 2d temporal adjacent networks for moment localization with natural language. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 12870–12877, 2020. 
*   [76] Yimeng Zhang, Akshay Karkal Kamath, Qiucheng Wu, Zhiwen Fan, Wuyang Chen, Zhangyang Wang, Shiyu Chang, Sijia Liu, and Cong Hao. Data-model-circuit tri-design for ultra-light video intelligence on edge devices. In Proceedings of the 28th Asia and South Pacific Design Automation Conference, pages 745–750, 2023. 
*   [77] Yimeng Zhang, Xiao-Yang Liu, Bo Wu, and Anwar Walid. Video synthesis via transform-based tensor neural network. In Proceedings of the 28th ACM International Conference on Multimedia, pages 2454–2462, 2020. 
*   [78] Yimeng Zhang, Yuguang Yao, Jinghan Jia, Jinfeng Yi, Mingyi Hong, Shiyu Chang, and Sijia Liu. How to robustify black-box ml models? a zeroth-order optimization perspective. arXiv preprint arXiv:2203.14195, 2022. 
*   [79] Yihua Zhang, Yuguang Yao, Parikshit Ram, Pu Zhao, Tianlong Chen, Mingyi Hong, Yanzhi Wang, and Sijia Liu. Advancing model pruning via bi-level optimization. arXiv preprint arXiv:2210.04092, 2022. 
*   [80] Yihua Zhang, Guanhua Zhang, Prashant Khanduri, Mingyi Hong, Shiyu Chang, and Sijia Liu. Revisiting and advancing fast adversarial training through the lens of bi-level optimization. In International Conference on Machine Learning, pages 26693–26712. PMLR, 2022. 
*   [81] Yang Zheng, Xiaoyi Feng, et al. Why adversarial reprogramming works, when it fails, and how to tell the difference. arXiv:2108.11673, 2021. 
*   [82] Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, and Dongwei Ren. Distance-iou loss: Faster and better learning for bounding box regression. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 12993–13000, 2020. 
*   [83] Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In Proceedings of the IEEE international conference on computer vision, pages 19–27, 2015.

