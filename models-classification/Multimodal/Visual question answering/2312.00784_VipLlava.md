Title: Making Large Multimodal Models Understand Arbitrary Visual Prompts

URL Source: https://arxiv.org/html/2312.00784

Published Time: Tue, 30 Apr 2024 00:10:34 GMT

Markdown Content:
Mu Cai 1 Haotian Liu 1 Dennis Park 2 Siva Karthik Mustikovela 2

Gregory P. Meyer 2 Yuning Chai 2 Yong Jae Lee 1,2

1 University of Wisconsin–Madison 2 Cruise LLC 

[https://vip-llava.github.io](https://vip-llava.github.io/)

###### Abstract

While existing large vision-language multimodal models focus on whole image understanding, there is a prominent gap in achieving region-specific comprehension. Current approaches that use textual coordinates or spatial encodings often fail to provide a user-friendly interface for visual prompting. To address this challenge, we introduce a novel multimodal model capable of decoding arbitrary (free-form) visual prompts. This allows users to intuitively mark images and interact with the model using natural cues like a “red bounding box” or “pointed arrow”. Our simple design directly overlays visual markers onto the RGB image, eliminating the need for complex region encodings, yet achieves state-of-the-art performance on region-understanding tasks like Visual7W, PointQA, and Visual Commonsense Reasoning benchmark. Furthermore, we present ViP-Bench, a comprehensive benchmark to assess the capability of models in understanding visual prompts across multiple dimensions, enabling future research in this domain. Code, data, and model are publicly available.

1 Introduction
--------------

Large language models (LLMs) like ChatGPT[[32](https://arxiv.org/html/2312.00784v2#bib.bib32)], GPT4[[33](https://arxiv.org/html/2312.00784v2#bib.bib33)], and Bard[[12](https://arxiv.org/html/2312.00784v2#bib.bib12)] have recently gained significant attention for their strong reasoning and generalization capabilities, and their ability to chat in a human-like manner. In particular, models such as GPT-4V(ision)[[31](https://arxiv.org/html/2312.00784v2#bib.bib31)], which incorporate visual information, have demonstrated human-level perception and reasoning capabilities[[50](https://arxiv.org/html/2312.00784v2#bib.bib50)]. This has spurred the development of similar open-source models that aim to replicate or even surpass the proprietary models’ performance.

Despite their capabilities, current models, including seminal ones like LLaVA[[24](https://arxiv.org/html/2312.00784v2#bib.bib24), [23](https://arxiv.org/html/2312.00784v2#bib.bib23)] and MiniGPT-4[[56](https://arxiv.org/html/2312.00784v2#bib.bib56)], focus predominantly on whole-image understanding; in other words, they lack the capability to process _region-specific_ information in complex scenes. This limitation becomes particularly apparent when attempting to describe specific objects within an image using only language prompts, which can be difficult when there is ambiguity (e.g., when there are multiple people in the image, and the question relates to a specific person), as shown in Figure[1](https://arxiv.org/html/2312.00784v2#S1.F1 "Figure 1 ‣ 1 Introduction ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts").

To address this issue, recent work explores spatial references in multimodal models. Existing efforts have primarily focused on using textual representations of coordinates[[53](https://arxiv.org/html/2312.00784v2#bib.bib53), [9](https://arxiv.org/html/2312.00784v2#bib.bib9), [5](https://arxiv.org/html/2312.00784v2#bib.bib5), [4](https://arxiv.org/html/2312.00784v2#bib.bib4)], learned positional embeddings[[52](https://arxiv.org/html/2312.00784v2#bib.bib52), [34](https://arxiv.org/html/2312.00784v2#bib.bib34), [55](https://arxiv.org/html/2312.00784v2#bib.bib55)], or ROI features[[52](https://arxiv.org/html/2312.00784v2#bib.bib52), [37](https://arxiv.org/html/2312.00784v2#bib.bib37)]. However, they often lack user-friendliness, as they are limited to fixed-format visual references like bounding boxes and the spatial coordinates of a mask contour. Most of these approaches, including those by Zhang et al.[[52](https://arxiv.org/html/2312.00784v2#bib.bib52)] and Chen et al.[[5](https://arxiv.org/html/2312.00784v2#bib.bib5)], only employ bounding box inputs for visual referrals. While effective in structured scenarios, this method proves less versatile in natural, user-driven interactions where the visual prompts may not conform to clean geometric shapes.

![Image 1: Refer to caption](https://arxiv.org/html/2312.00784v2/x1.png)

Figure 1: Main Idea of ViP-LLaVA. We directly overlay diverse visual prompts (e.g., arrows, boxes, circles, scribbles) onto the original image, and then feed the corresponding visual features along with text embeddings into the large multimodal model for conversational assistance. Here we show an example using a red arrow. 

In this paper, we propose a simple yet highly effective solution to this problem: a large multimodal model that can process _arbitrary visual prompts_. This allows a user to intuitively mark up images and interact using natural cues such as a “red bounding box” or “pointed arrow”. Our model recognizes these visual prompts, offering a user-friendly way to integrate visual references into the language dialogue. Based on our own observation and prior work[[38](https://arxiv.org/html/2312.00784v2#bib.bib38)], which shows that CLIP can understand visual markers, we _directly inject the visual prompts into the original image space_ without any additional region-specific model designs. Although our approach is deceptively simple, it yields an unexpected benefit: our model sets new state-of-the-art performances on tasks demanding precise region-specific perception and complex reasoning. It surpasses the capabilities of existing related models with specialized region encoding techniques, as evidenced by our superior performance on region reasoning tasks on Visual7W[[57](https://arxiv.org/html/2312.00784v2#bib.bib57)] and PointQA[[29](https://arxiv.org/html/2312.00784v2#bib.bib29)].

To further support research in this area, we introduce _ViP-Bench_, a benchmark for evaluating multimodal models’ region understanding capabilities with arbitrary visual prompts. By collecting a diverse set of 303 images and questions, we provide a comprehensive assessment of visual understanding capabilities across six aspects at the region level: recognition, OCR, knowledge, math, object relationship reasoning, and language generation. We believe that ViP-Bench will provide a solid foundation for future research into multimodal models with arbitrary visual prompts.

In summary, our main contributions are:

*   •We introduce a novel multimodal model for intuitive interaction with images using natural language and arbitrary visual prompts, enhancing user accessibility and model flexibility. 
*   •We develop a visual referal approach that overlays visual prompts directly onto images, simplifying the model’s architecture without compromising performance. 
*   •Our model, ViP-LLaVA, achieves state-of-the-art results on region understanding tasks on established benchmarks, surpassing specialized region encoding models. 
*   •We introduce ViP-Bench, a benchmark for evaluating visual prompt interpretation, setting a foundational platform for future research. 

2 Related Work
--------------

#### Advancements in Large Multimodal Models.

Large language models like ChatGPT[[32](https://arxiv.org/html/2312.00784v2#bib.bib32)], GPT4[[33](https://arxiv.org/html/2312.00784v2#bib.bib33)], and LLaMA[[41](https://arxiv.org/html/2312.00784v2#bib.bib41)] have shown impressive reasoning and generalization capabilities. The landscape of LLMs has been markedly transformed by the recent introduction of models that integrate visual information, such as GPT-4V(ision)[[31](https://arxiv.org/html/2312.00784v2#bib.bib31)]. Building upon open-source LLMs[[41](https://arxiv.org/html/2312.00784v2#bib.bib41), [43](https://arxiv.org/html/2312.00784v2#bib.bib43)], a vast number of multimodal vision-language models have made significant strides, spearheaded by LLaVA[[24](https://arxiv.org/html/2312.00784v2#bib.bib24), [23](https://arxiv.org/html/2312.00784v2#bib.bib23)] and MiniGPT-4[[56](https://arxiv.org/html/2312.00784v2#bib.bib56)], which combine LLaMA’s[[41](https://arxiv.org/html/2312.00784v2#bib.bib41)] language prowess with a CLIP[[36](https://arxiv.org/html/2312.00784v2#bib.bib36)] based image encoder. While these models excel at whole-image understanding, a key challenge has been region-specific comprehension within complex visual scenes. This has led to the exploration of spatial referrals in multimodal contexts. Existing models utilize textual coordinate representations[[53](https://arxiv.org/html/2312.00784v2#bib.bib53), [9](https://arxiv.org/html/2312.00784v2#bib.bib9), [5](https://arxiv.org/html/2312.00784v2#bib.bib5), [4](https://arxiv.org/html/2312.00784v2#bib.bib4)], learned positional embeddings[[52](https://arxiv.org/html/2312.00784v2#bib.bib52), [34](https://arxiv.org/html/2312.00784v2#bib.bib34), [55](https://arxiv.org/html/2312.00784v2#bib.bib55)], or Region of Interest (ROI) features[[52](https://arxiv.org/html/2312.00784v2#bib.bib52)] to anchor language to specific image regions. However, they often employ rigid visual referral formats that are not as intuitive for users.

#### Visual Prompting as a User-Friendly Solution.

Our focus is on making the interaction with multimodal models more natural and intuitive. Traditional models have employed regular shapes for visual prompting, but our research is motivated by the need for a system that can interpret a wider range of visual prompts. For example, in visual perception, interactive segmentation methods have been proposed that can take in points or scribbles[[58](https://arxiv.org/html/2312.00784v2#bib.bib58), [17](https://arxiv.org/html/2312.00784v2#bib.bib17)]. Drawing inspiration from recent findings that show GPT-4V’s ability to understand a variety of markers[[46](https://arxiv.org/html/2312.00784v2#bib.bib46)], we advocate for a model that can handle arbitrary visual cues, such as scribbles and arrows. In our model, ViP-LLaVA, we overlay these visual prompts directly onto the image canvas. This is accomplished by fine-tuning on a dataset specifically designed for arbitrary visual prompt instructions.

#### Evaluating LMM’s Region Understanding Capabilities.

Existing works[[5](https://arxiv.org/html/2312.00784v2#bib.bib5), [52](https://arxiv.org/html/2312.00784v2#bib.bib52), [34](https://arxiv.org/html/2312.00784v2#bib.bib34), [47](https://arxiv.org/html/2312.00784v2#bib.bib47)] evaluates the model’s region understanding capabilities on regional multichoice[[51](https://arxiv.org/html/2312.00784v2#bib.bib51), [57](https://arxiv.org/html/2312.00784v2#bib.bib57), [29](https://arxiv.org/html/2312.00784v2#bib.bib29)] or captioning[[49](https://arxiv.org/html/2312.00784v2#bib.bib49), [18](https://arxiv.org/html/2312.00784v2#bib.bib18)] tasks with metrics such as accuracy, recall, and CIDer[[42](https://arxiv.org/html/2312.00784v2#bib.bib42)]. However, these metrics fall short when it comes to evaluating visual dialogue for large multimodal models in an open-world setting. To evaluate LMM’s capability in engaging in visual conversations for _image-level_ understanding, two families of evaluation are proposed: multiple-choice[[25](https://arxiv.org/html/2312.00784v2#bib.bib25)] or using GPT4 as a judge for free-form answers[[50](https://arxiv.org/html/2312.00784v2#bib.bib50), [24](https://arxiv.org/html/2312.00784v2#bib.bib24)]. However, a gap still exists in the evaluation of LMM’s capabilities for comprehending arbitrary visual prompts. To address this, we introduce ViP-Bench, a comprehensive benchmark tailored to evaluate how well the LMMs can interpret various visual prompts across multiple dimensions, including recognition, OCR, knowledge, math, relationship reasoning, and language generation.

3 Approach
----------

Our research hinges on the premise that a large multimodal model should not only perceive the visual content of an image but also interpret arbitrary visual markers as part of the user interaction. In this section, we describe our approach that achieves this goal, highlighting the pivotal role of CLIP in understanding visual markers and the construction of a new instruction tuning dataset tailored to train ViP-LLaVA to understand arbitrary visual prompts.

### 3.1 Visual Prompt Embedding via CLIP

In contrast to prior work on region understanding[[34](https://arxiv.org/html/2312.00784v2#bib.bib34), [52](https://arxiv.org/html/2312.00784v2#bib.bib52)] which constructs a new module to process visual prompts, we leverage CLIP’s[[36](https://arxiv.org/html/2312.00784v2#bib.bib36)] existing capabilities to encode both the image and superimposed visual markers. Specifically, CLIP’s proficiency in aligning visual and textual data makes it an ideal candidate for this task, as recent studies[[38](https://arxiv.org/html/2312.00784v2#bib.bib38)] suggest that it inherently pays attention to marked regions including circles, rectangles, _etc_. As shown in our experiments, we further demonstrate that CLIP can focus the model’s attention on a wider variety of visual prompts such as arrows and arbitrary scribbles. To utilize this functionality, we composite the visual prompts 𝐏 v subscript 𝐏 v\mathbf{P}_{\mathrm{v}}bold_P start_POSTSUBSCRIPT roman_v end_POSTSUBSCRIPT onto the original image 𝐗 v subscript 𝐗 v\mathbf{X}_{\mathrm{v}}bold_X start_POSTSUBSCRIPT roman_v end_POSTSUBSCRIPT through alpha blending, creating a merged representation that highlights the areas of interest:

𝐗^v=α⋅𝐏 v+(1−α)⋅𝐗 v,subscript^𝐗 v⋅𝛼 subscript 𝐏 v⋅1 𝛼 subscript 𝐗 v\hat{\mathbf{X}}_{\mathrm{v}}=\alpha\cdot\mathbf{P}_{\mathrm{v}}+(1-\alpha)% \cdot\mathbf{X}_{\mathrm{v}},over^ start_ARG bold_X end_ARG start_POSTSUBSCRIPT roman_v end_POSTSUBSCRIPT = italic_α ⋅ bold_P start_POSTSUBSCRIPT roman_v end_POSTSUBSCRIPT + ( 1 - italic_α ) ⋅ bold_X start_POSTSUBSCRIPT roman_v end_POSTSUBSCRIPT ,(1)

where α∈[0,1]𝛼 0 1\alpha\in[0,1]italic_α ∈ [ 0 , 1 ] denotes the transparency level of the visual prompt, 𝐗 v subscript 𝐗 v\mathbf{X}_{\mathrm{v}}bold_X start_POSTSUBSCRIPT roman_v end_POSTSUBSCRIPT is the image, and 𝐏 v subscript 𝐏 v\mathbf{P}_{\mathrm{v}}bold_P start_POSTSUBSCRIPT roman_v end_POSTSUBSCRIPT is the image with the visual prompt. Note that we only perform alpha blending for pixels underlying the visual prompt. The composite image 𝐗^v subscript^𝐗 v\hat{\mathbf{X}}_{\mathrm{v}}over^ start_ARG bold_X end_ARG start_POSTSUBSCRIPT roman_v end_POSTSUBSCRIPT is then fed into the multimodal model.

![Image 2: Refer to caption](https://arxiv.org/html/2312.00784v2/x2.png)

Figure 2: Model Architecture. After alpha blending the visual prompts onto the original image, we feed the resulting image into the visual encoder to obtain multi-level visual features. Those features are concatenated and fed into the LayerNorm and MLP layers to form the visual tokens. Then visual tokens and text instruction tokens are fed into the large language model to produce the language response in an auto-regressive manner. The frozen and trainable modules during instruction tuning are annotated. 

To effectively recognize the visual prompts, we balance low-level and high-level visual features in ViP-LLaVA.

To address the tendency of CLIP’s deeper features to overlook low-level details[[54](https://arxiv.org/html/2312.00784v2#bib.bib54)], we selectively extract features from multiple CLIP layers. Specifically, we use one early layer (6-th) to encode detailed geometric shapes and four deeper layers (15, 18, 21, 24-th) to capture broader semantic information. These multi-level features are then concatenated, normalized using LayerNorm[[2](https://arxiv.org/html/2312.00784v2#bib.bib2)] for training stability, and finally passed through an MLP layer. This process ensures ViP-LLaVA effectively integrates diverse visual cues, a strategy validated through our ablation studies detailed in Sec.[5.4](https://arxiv.org/html/2312.00784v2#S5.SS4 "5.4 Ablation Studies ‣ 5 Experiments ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts").

Our design’s simplicity of directly overlaying visual prompts offers several advantages. It reduces model complexity by bypassing additional processing modules and aligns closely with natural human interactions, as users often employ diverse and spontaneous visual markers. This flexibility allows ViP-LLaVA to interpret a wide range of user-generated visual cues, enhancing its applicability in real-world scenarios.

To train ViP-LLaVA, we perform autoregressive language modeling; _i.e_., we maximize the likelihood of generating the tokens of the ground-truth answer 𝐗 a subscript 𝐗 a\mathbf{X}_{\mathrm{a}}bold_X start_POSTSUBSCRIPT roman_a end_POSTSUBSCRIPT:

P⁢(𝐗 a∣𝐗^v,𝐗 instruct)=∏i=1 L P 𝜽⁢(x i∣𝐗^v,𝐗 instruct,𝐗 a,<i)𝑃 conditional subscript 𝐗 a subscript^𝐗 v subscript 𝐗 instruct superscript subscript product 𝑖 1 𝐿 subscript 𝑃 𝜽 conditional subscript 𝑥 𝑖 subscript^𝐗 v subscript 𝐗 instruct subscript 𝐗 a absent 𝑖 P(\mathbf{X}_{\mathrm{a}}\mid\hat{\mathbf{X}}_{\mathrm{v}},\mathbf{X}_{\text{% instruct}})=\prod_{i=1}^{L}P_{\boldsymbol{\theta}}(x_{i}\mid\hat{\mathbf{X}}_{% \mathrm{v}},\mathbf{X}_{\text{instruct}},\mathbf{X}_{\mathrm{a},<i})italic_P ( bold_X start_POSTSUBSCRIPT roman_a end_POSTSUBSCRIPT ∣ over^ start_ARG bold_X end_ARG start_POSTSUBSCRIPT roman_v end_POSTSUBSCRIPT , bold_X start_POSTSUBSCRIPT instruct end_POSTSUBSCRIPT ) = ∏ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_L end_POSTSUPERSCRIPT italic_P start_POSTSUBSCRIPT bold_italic_θ end_POSTSUBSCRIPT ( italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∣ over^ start_ARG bold_X end_ARG start_POSTSUBSCRIPT roman_v end_POSTSUBSCRIPT , bold_X start_POSTSUBSCRIPT instruct end_POSTSUBSCRIPT , bold_X start_POSTSUBSCRIPT roman_a , < italic_i end_POSTSUBSCRIPT )(2)

where 𝜽 𝜽\boldsymbol{\theta}bold_italic_θ represents the trainable parameters, 𝐗 instruct subscript 𝐗 instruct\mathbf{X}_{\text{instruct}}bold_X start_POSTSUBSCRIPT instruct end_POSTSUBSCRIPT is the text instruction, L 𝐿 L italic_L is the sequence length of the answer 𝐗 a subscript 𝐗 a\mathbf{X}_{\mathrm{a}}bold_X start_POSTSUBSCRIPT roman_a end_POSTSUBSCRIPT, and 𝐗 a,<i subscript 𝐗 a absent 𝑖\mathbf{X}_{\mathrm{a},<i}bold_X start_POSTSUBSCRIPT roman_a , < italic_i end_POSTSUBSCRIPT denotes all the answer tokens before the current prediction token x i subscript 𝑥 𝑖 x_{i}italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT, where i 𝑖 i italic_i denotes the steps during text token generation. Here we omit system messages from the equation for clarity, even though they are part of the conditioning. Figure[2](https://arxiv.org/html/2312.00784v2#S3.F2 "Figure 2 ‣ 3.1 Visual Prompt Embedding via CLIP ‣ 3 Approach ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") shows our model architecture.

This training objective enables the model to generate contextually accurate responses by comprehending the visual content, language instruction, and the overlaid prompts. It fosters the model’s ability to interpret visual markers in unison with the image, thereby enhancing its proficiency in addressing complex, region-specific language inquiries. This capability is crucial for tasks requiring nuanced understanding of both the visual elements and user intentions conveyed through arbitrary visual prompts.

### 3.2 Visual Prompting Design

To train the model to recognize and interpret arbitrary visual prompts, we develop a new visual prompt instruction tuning dataset, as there are no prior datasets with arbitrary visual prompts and instruction-output text pairs that we can use.

Our dataset comprises a diverse collection of 520k image-text pairs marked with visual prompts, sourced from publicly available datasets, including (1) single region reasoning data: 80k referring comprehension and generation data from RefCOCOg[[49](https://arxiv.org/html/2312.00784v2#bib.bib49)], and 37k object counting data from PointQA-LookTwice[[29](https://arxiv.org/html/2312.00784v2#bib.bib29)], (2) two-region reasoning data: 80k triplet relationship data from Visual Genome[[18](https://arxiv.org/html/2312.00784v2#bib.bib18)], (3) multi-region reasoning data: 30k grounded image captioning data from Flicker 30k Entities[[35](https://arxiv.org/html/2312.00784v2#bib.bib35)], 213K data from Visual Commonsense Reasoning dataset[[51](https://arxiv.org/html/2312.00784v2#bib.bib51)], and 82k data from Visual7W[[57](https://arxiv.org/html/2312.00784v2#bib.bib57)]. Note that all those data are collected from the training split of the aforementioned datasets.

We automatically annotate each image with various visual prompts. For the data that only comes with bounding box annotations, we sample the visual prompts from three possible categories: rectangle, ellipse, and arrow. Here we make sure that the head of the arrow lies within [(−W 2,−H 2),(W 2,H 2)]𝑊 2 𝐻 2 𝑊 2 𝐻 2[(-\frac{W}{2},-\frac{H}{2}),(\frac{W}{2},\frac{H}{2})][ ( - divide start_ARG italic_W end_ARG start_ARG 2 end_ARG , - divide start_ARG italic_H end_ARG start_ARG 2 end_ARG ) , ( divide start_ARG italic_W end_ARG start_ARG 2 end_ARG , divide start_ARG italic_H end_ARG start_ARG 2 end_ARG ) ] space, where W,H 𝑊 𝐻 W,H italic_W , italic_H are the width and height of the image, respectively. For ellipse, the lengths along the semi-major and semi-minor axes are inherited from the bounding box size, where we enlarge the ellipse with a ratio between [1,1.5]1 1.5[1,1.5][ 1 , 1.5 ]. On the other hand, for regions that come with ground truth pixel-level mask annotations, we annotate each region with visual prompts sampled from the following 8 possibilities: rectangle, ellipse, point, triangle, mask, mask contour, arrow, and scribble created using Bézier curves; see Figure[3](https://arxiv.org/html/2312.00784v2#S3.F3 "Figure 3 ‣ 3.2 Visual Prompting Design ‣ 3 Approach ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"). We make sure that the head of the arrow, entire point, triangle, and scribble lies within the provided mask. These annotations simulate natural human interactions with images, where users often use spontaneous markers to highlight areas of interest.

For scribbles, we simulate human-like drawings using Bézier curves[[8](https://arxiv.org/html/2312.00784v2#bib.bib8)]. This process begins by randomly selecting three points within the object mask, which serve as the anchors for the quadratic Bézier curve. The generated Bézier curve is then composited onto the image using the previously mentioned alpha blending technique to produce a merged image with the scribble serving as a visual prompt.

![Image 3: Refer to caption](https://arxiv.org/html/2312.00784v2/x3.png)

Figure 3: Visualization of Visual Prompt Types. From top-left to bottom-right: mask contour, ellipse, bounding box, triangle, scribble, point, arrow, and mask. Note that the prompts not only have diverse shapes, but they also have diverse colors, transparency values, widths, scales, and directions.

Humans naturally use various markers to highlight objects within their environment. For instance, in educational settings, teachers often use arrows or underlining to draw students’ attention to specific parts of an image or text. Similarly, in everyday communication, people might circle items in a photograph to point out something of interest or use scribbles to obscure sensitive information before sharing. Through our design, we create a visual instruction following dataset that mirrors the way humans visually interact with objects, thus fostering a more intuitive and natural interaction with the model.

### 3.3 Optional Region-level Instruction Tuning Data

Our training data comes from two sources: (i) region-level visual prompting data described in Section[3.2](https://arxiv.org/html/2312.00784v2#S3.SS2 "3.2 Visual Prompting Design ‣ 3 Approach ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"), and (ii) image-level data devoid of visual prompts, sourced from LLaVA-1.5[[23](https://arxiv.org/html/2312.00784v2#bib.bib23)]. This strategy enables ViP-LLaVA to engage in human-like conversations, primarily due to the image-level LLaVA instruction data from Liu _et al_.[[24](https://arxiv.org/html/2312.00784v2#bib.bib24)]. Optionally, to further enhance ViP-LLaVA’s capability in multimodal conversations at the region-level, we design region-specific instruction data with the help of GPT-4V.

Prior approaches like Shikra[[5](https://arxiv.org/html/2312.00784v2#bib.bib5)] attempted to generate region-level instruction data using text-only models like GPT4. However, this method is inherently limiting, particularly in object-level tasks where the model, lacking visual context, cannot accurately reference multiple objects of the same class within a single scene. To overcome this, we develop an instruction data curation method using GPT-4V. Unlike text-only models, GPT-4V can interpret visual prompts displayed in images[[46](https://arxiv.org/html/2312.00784v2#bib.bib46)]. Our method involves feeding two images into GPT-4V: the original image and a modified version with annotated visual prompts. Alongside these images, we provide the model with the ground-truth (text) annotation in the original dataset and system messages. This process is used to curate <visual prompt, text prompt, text output> triplets for the images in our dataset described in Section[3.2](https://arxiv.org/html/2312.00784v2#S3.SS2 "3.2 Visual Prompting Design ‣ 3 Approach ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts").

We introduce specific textual representations such as <within red mask> and (<within red box>, <within blue box>) to guide GPT-4V in recognizing the visual prompts in both single-region and multi-region settings. During training, we replace these phrases with the set of eight possible visual prompts described in Section[3.2](https://arxiv.org/html/2312.00784v2#S3.SS2 "3.2 Visual Prompting Design ‣ 3 Approach ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"), significantly enhancing the dataset’s versatility. In total, we curate 13k high-quality region-level instruction data points, comprised of 7k single-region and 6k multi-region instances. In the supplementary, we provide specific details of the system messages, input text prompts, and generated text outputs.

Although ViP-LLaVA works well even without this enriched data for standard visual reasoning benchmarks, we find that it helps to further improve the model’s ability to have human-like conversations in open-world settings.

4 ViP-Bench for Evaluation
--------------------------

In order to rigorously evaluate the capabilities of multimodal models in interpreting and responding to visual reasoning queries, we introduce ViP-Bench, a benchmarking suite for evaluating multimodal region-understanding capabilities under various visual prompts. ViP-Bench consists of 303 unique image-question pairs, where images are collected from MM-Vet[[50](https://arxiv.org/html/2312.00784v2#bib.bib50)], MMBench[[25](https://arxiv.org/html/2312.00784v2#bib.bib25)], and Visual Genome[[18](https://arxiv.org/html/2312.00784v2#bib.bib18)]. Each pair consists of an image coupled with a diverse visual reasoning question designed to test a model’s understanding and interpretation capabilities. We reuse the questions in MM-Vet[[50](https://arxiv.org/html/2312.00784v2#bib.bib50)] and MMBench[[25](https://arxiv.org/html/2312.00784v2#bib.bib25)] (but make minor adjustments so that they take into account the region-specific visual prompts), while in Visual Genome, we design the questions and answers by ourselves. We use bounding boxes and masks produced by the Segment Anything Model(SAM)[[17](https://arxiv.org/html/2312.00784v2#bib.bib17)] to annotate the location of the objects.

Key to the design of ViP-Bench is its comprehensive coverage of six crucial aspects of visual understanding at the region level: recognition, OCR (Optical Character Recognition), knowledge, math, object relationship reasoning, and language generation. This range ensures a holistic assessment of a model’s performance in various facets of region-level visual reasoning.

ViP-Bench employs a similar grading mechanism as MM-Vet[[50](https://arxiv.org/html/2312.00784v2#bib.bib50)]. We employ the GPT-4 text model, a state-of-the-art language model, to evaluate the responses of multimodal models. Specifically, we feed the response from the multimodal model, the human annotated answer, and several in-context scoring examples to GPT-4. The responses are scored by GPT-4 on a scale from 0 to 10, offering a quantitative measure of the multimodal model’s proficiency in understanding and interpreting visual data. This grading system provides a standardized framework for comparing the performance of different models.

ViP-Bench is meticulously annotated by humans. This process involved seven rounds of validation to ensure the accuracy and relevance of the object boxes/masks, questions, and answers. Such rigorous annotation guarantees the reliability of the benchmark as a tool for model evaluation. An illustrative example in Table[6](https://arxiv.org/html/2312.00784v2#S6.T6 "Table 6 ‣ Overfitting Concerns in Region-Level LMMs. ‣ 6 ViP-Bench Evaluation Results ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") showcases a scenario where a leading model like GPT-4V misinterprets object localization under ViP-Bench, highlighting the challenges in current multimodal understanding. We present additional visualizations and statistics of ViP-Bench in the supp.

Through ViP-Bench, we provide a valuable tool for the research community, aiding in the development and refinement of multimodal models. By offering a comprehensive and challenging testbed, we believe ViP-Bench can set the stage for future advancements in the field of visual reasoning and multimodal interaction.

Table 1: Comparison of methods in terms of generality and accuracy on Visual7W[[57](https://arxiv.org/html/2312.00784v2#bib.bib57)] test set.

5 Experiments
-------------

In this section, we compare ViP-LLaVA to state-of-the-art multimodal models, including those that explicitly design region-specific modules, perform in-depth analysis to assess ViP-LLaVA’s capabilities, and perform ablation studies.

Table 2: Comparison of methods in terms of generality and accuracy on PointQA-LookTwice[[29](https://arxiv.org/html/2312.00784v2#bib.bib29)] test set. †zero-shot eval.

### 5.1 Training Setup

#### Model.

For the visual model, we choose CLIP-336px[[36](https://arxiv.org/html/2312.00784v2#bib.bib36)] to preserve more information from the raw pixel space. We use Vicuna v1.5[[43](https://arxiv.org/html/2312.00784v2#bib.bib43)] as the language encoder. For the multimodal connector, a 2-layer MLP is utilized. Ablations on more LLM backbones are shown in Supp.[A.5](https://arxiv.org/html/2312.00784v2#A1.SS5 "A.5 Impact of Different LLMs on Image-level and Region-level Benchmarks ‣ Appendix A In-Depth Analysis ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts").

#### Training and data.

During the initial stage of training, we employ 558k BLIP[[6](https://arxiv.org/html/2312.00784v2#bib.bib6), [24](https://arxiv.org/html/2312.00784v2#bib.bib24)] captioned image-text pairs to pretrain the multimodal connector. The second stage utilizes LLaVA v1.5[[23](https://arxiv.org/html/2312.00784v2#bib.bib23)] instruction data alongside our region-level visual prompting dataset from Section[3.2](https://arxiv.org/html/2312.00784v2#S3.SS2 "3.2 Visual Prompting Design ‣ 3 Approach ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"). Both stages train the model for 1 epoch, with an overall training time of around 20/40 hours for the 7B/13B model using 8 NVIDIA A100 GPUs. Finally, we mix the 13k GPT-4V instruction data with 13k sampled data from stage 2 to get 26k stage 3 training data, and then fine-tune our stage-2 model (referred to as ViP-LLaVA-Base) for one epoch to get our model ViP-LLaVA, which requires approximately 0.5 hours for the 7B model and 1 hour for the 13B model on 8 NVIDIA A100 GPUs.

#### Visual prompts.

ViP-LLaVA uses 8 visual prompts: rectangles, ellipses, points, scribbles, triangles, masks, mask contours, and arrows. Their attributes, such as color, thickness, and alpha value for alpha blending (in [0.5, 1]) are randomized. The arrow’s direction and length are randomized, with the endpoint remaining within the mask. For referencing specific regions, we replace the <region> text with the color and shape description, such as red scribble. The visual prompt type and associated attributes for each region are randomly assigned during training.

### 5.2 Evaluation on Region Reasoning Benchmarks

We first quantitatively evaluate ViP-LLaVA on three region reasoning benchmarks.

#### Visual7W.

The Visual7W dataset[[57](https://arxiv.org/html/2312.00784v2#bib.bib57)] tests models’ spatial perception by requiring them to match text descriptions with the correct bounding boxes from a set of choices. We differentiate between ‘generalist’ models, which are not specifically trained on the target dataset, and ‘specialist’ models, which are. For a fair comparison, we use image overlays as visual prompts for the LLaVA model and textual coordinates for Shikra’s text prompts. The results in Table[1](https://arxiv.org/html/2312.00784v2#S4.T1 "Table 1 ‣ 4 ViP-Bench for Evaluation ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") shows ViP-LLaVA-7B outperforming recent state-of-the-art methods, including GPT4RoI[[52](https://arxiv.org/html/2312.00784v2#bib.bib52)] and Shikra[[5](https://arxiv.org/html/2312.00784v2#bib.bib5)], despite having fewer parameters, and ViP-LLaVA-13B producing even higher gains. ViP-LLaVA overlays bounding boxes directly onto the image, creating an immediate link between the image and spatial locations. This contrasts with other methods that rely on external embeddings from either textual or newly learned embedding spaces to reference specific regions, proving less effective in this context.

Table 3: Validation Accuracy on VCR[[51](https://arxiv.org/html/2312.00784v2#bib.bib51)] dataset.

#### PointQA-LookTwice.

PointQA[[29](https://arxiv.org/html/2312.00784v2#bib.bib29)] presents a dataset where queries are based on either a specific point or a bounding box within an image. We evaluate ViP-LLaVA under the broad-question scenario using the bounding box type, typified by the prompt How many of these are there? This requires the model to first correctly identify the object within the given region and subsequently enumerate instances of the same category across the image—essentially a test of object recognition followed by class-specific counting. In line with our methodology for Visual7W, we use the image overlaid with the bounding box for LLaVA, while for Shikra, we incorporate the bounding box coordinates into the text prompt. Table[2](https://arxiv.org/html/2312.00784v2#S5.T2 "Table 2 ‣ 5 Experiments ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") shows ViP-LLaVA’s superior performance on this intricate task, surpassing other multimodal contenders. Our method of overlaying visual prompts ensures the object remains unobscured, effectively combining the original image pixels with visual cues to enhance object recognition and counting accuracy.

#### Visual Commonsense Reasoning.

The Visual Commonsense Reasoning (VCR) dataset[[51](https://arxiv.org/html/2312.00784v2#bib.bib51)] is a challenging benchmark designed to evaluate a model’s capabilities in high-level cognition and commonsense reasoning in the context of visual information. The dataset presents multiple-choice questions that require an understanding of the scene depicted in an image. Each question (Q) is paired with four potential answers (A), where the model must not only select the correct answer but also provide a rationale (R) that justifies its choice, demonstrating the model’s ability to comprehend and rationalize visual elements within a given context.

We finetune ViP-LLaVA-Base-7B on VCR, similar to the protocol in GPT4RoI[[52](https://arxiv.org/html/2312.00784v2#bib.bib52)]. As shown in Table[3](https://arxiv.org/html/2312.00784v2#S5.T3 "Table 3 ‣ Visual7W. ‣ 5.2 Evaluation on Region Reasoning Benchmarks ‣ 5 Experiments ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"), our approach exhibits state-of-the-art performance on the validation set, illustrating its proficiency in visual commonsense reasoning. This success highlights our approach’s dual strengths: adeptness in perception tasks and effectiveness in multi-region reasoning. By integrating visual prompts directly into the image, our model more effectively associates spatial locations with semantic understanding, facilitating a better interaction between spatial and semantic reasoning.

![Image 4: Refer to caption](https://arxiv.org/html/2312.00784v2/x4.png)

Figure 4: ViP-LLaVA model is able to infer correspondences between multiple objects in the image.

![Image 5: Refer to caption](https://arxiv.org/html/2312.00784v2/x5.png)

Figure 5: ViP-LLaVA is able to understand the direction of arrows.

### 5.3 In-depth Analysis

#### Region reasoning with arbitrary human drawings at test time.

ViP-LLaVA, when presented with arbitrarily drawn enclosed regions or arrows by a user, can accurately describe, shown in Figure[4](https://arxiv.org/html/2312.00784v2#S5.F4 "Figure 4 ‣ Visual Commonsense Reasoning. ‣ 5.2 Evaluation on Region Reasoning Benchmarks ‣ 5 Experiments ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") and[6](https://arxiv.org/html/2312.00784v2#S5.F6 "Figure 6 ‣ Generalization to other attributes. ‣ 5.3 In-depth Analysis ‣ 5 Experiments ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts").

#### Multi-region understanding capabilities.

ViP-LLaVA demonstrates robust multi-region understanding, able to dissect complex visual scenes and infer relationships between various elements. As shown in Figure[4](https://arxiv.org/html/2312.00784v2#S5.F4 "Figure 4 ‣ Visual Commonsense Reasoning. ‣ 5.2 Evaluation on Region Reasoning Benchmarks ‣ 5 Experiments ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"), ViP-LLaVA is able to infer correspondences between multiple objects in the image, and make the correct reasoning that the red and blue circles both include the train.

#### Arrow direction understanding.

ViP-LLaVA is able to understand arrows. Here we conduct an ablation study of the arrow direction. Given two arrows that have the same body yet different heads, as shown in Figure[5](https://arxiv.org/html/2312.00784v2#S5.F5 "Figure 5 ‣ Visual Commonsense Reasoning. ‣ 5.2 Evaluation on Region Reasoning Benchmarks ‣ 5 Experiments ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"), ViP-LLaVA is able to understand the direction of the arrows, making correction descriptions about the respective regions.

#### Generalization to other attributes.

ViP-LLaVA also generalizes to untrained attributes, like varying visual prompt thickness or location, showcasing its adaptability beyond what was seen during training. See the supplementary material for examples of different thicknesses.

Figure[6](https://arxiv.org/html/2312.00784v2#S5.F6 "Figure 6 ‣ Generalization to other attributes. ‣ 5.3 In-depth Analysis ‣ 5 Experiments ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") further shows that ViP-LLaVA is able to conduct OCR first, and then make correspondences between different regions to make a correct prediction about the content of each part.

Table 4: Single layer vs.multi-layer CLIP image encodings.

![Image 6: Refer to caption](https://arxiv.org/html/2312.00784v2/x6.png)

Figure 6: ViP-LLaVA is able to understand the text markers in the image, and reason about them.

Table 5: ViP-Bench Evaluation Results. This table presents the performance of various models under ViP-Bench, utilizing different visual prompt formats. The evaluation includes both synthesized and human-drawn prompts, providing insights into the models’ maximum potential and real-world applicability, respectively. Formats include VP (visual prompts), Coor (coordinates as visual prompts), Dis (discrete positional tokens for vocabulary expansion), and ROI (CLIP region of interest features with positional embedding). The assessed dimensions are Recognition (Rec), OCR, Knowledge (Know), Math, Relationship (Rel), and Language Generation (Lang).

### 5.4 Ablation Studies

#### Impact of overlaying visual prompts on visual information.

To assess whether overlaying visual prompts on images obscures visual information, we conduct a comparison by inputting visual tokens from both the original and overlayed images into ViP-LLaVA-Base-7B. Using the VCR dataset, we evaluate the accuracy of the QA task with and without the additional visual tokens from the original image. Results on the VCR validation split shows an accuracy of 81.63% with the original image and overlaid image tokens, compared to 82.47% with the overlaid image tokens only. The similar accuracies suggest that the overlaid prompts do not detract from the visual information processed by our model.

#### Influence of CLIP multi-layer features.

We next explore the impact of using multi-layer visual features from CLIP as opposed to single-layer features, specifically focusing on the second-last layer as implemented in LLaVA[[24](https://arxiv.org/html/2312.00784v2#bib.bib24), [23](https://arxiv.org/html/2312.00784v2#bib.bib23)]. Our ablation study in Table[4](https://arxiv.org/html/2312.00784v2#S5.T4 "Table 4 ‣ Generalization to other attributes. ‣ 5.3 In-depth Analysis ‣ 5 Experiments ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") reveals a marked improvement in performance, particularly in scenarios involving multiple visual prompts, as in the Visual7W and VCR datasets. This indicates that leveraging multi-layer visual features significantly enhances the model’s ability to localize and recognize visual prompts within images.

6 ViP-Bench Evaluation Results
------------------------------

Finally, we evaluate on ViP-Bench using a set of image-level and region-level LMMs, including InstructBLIP[[6](https://arxiv.org/html/2312.00784v2#bib.bib6)], GPT-4V[[31](https://arxiv.org/html/2312.00784v2#bib.bib31)], LLaVA v1.5[[24](https://arxiv.org/html/2312.00784v2#bib.bib24)], Qwen-VL[[3](https://arxiv.org/html/2312.00784v2#bib.bib3)], Shikra[[5](https://arxiv.org/html/2312.00784v2#bib.bib5)], GPT4ROI[[52](https://arxiv.org/html/2312.00784v2#bib.bib52)] and Kosmos-2[[34](https://arxiv.org/html/2312.00784v2#bib.bib34)]. For open-source models, we evaluate with greedy decoding (temperature=0). As shown in Table[5](https://arxiv.org/html/2312.00784v2#S5.T5 "Table 5 ‣ Generalization to other attributes. ‣ 5.3 In-depth Analysis ‣ 5 Experiments ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"), we first see that the performance of all models, including GPT-4V, is far from perfect, demonstrating the difficulty of ViP-Bench. An illustrative case in Table[6](https://arxiv.org/html/2312.00784v2#S6.T6 "Table 6 ‣ Overfitting Concerns in Region-Level LMMs. ‣ 6 ViP-Bench Evaluation Results ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") depicts a scenario where GPT-4V and LLaVA incorrectly predict object localization. Overall, ViP-LLaVA outperforms other models, except GPT-4V, demonstrating greater adaptability to various visual perception and reasoning tasks. By training on images overlaid with visual prompts, ViP-LLaVA becomes adept at understanding arbitrary visual cues and mimicks the natural human method of referring to objects in images. This enables it not only to better identify and interpret visual prompts but also to integrate these prompts into its reasoning process, enhancing its overall comprehension and response accuracy.

#### Visual prompting is superior to other representations.

In zero-shot evaluation, when visual prompts are represented as a simple list of four textual numerical values, models like Qwen-VL and LLaVA underperform compared to ViP-LLaVA. This underscores the effectiveness of visual prompts over basic textual representations.

#### Language tasks: A challenge for current LMMs.

The ViP-Bench results reveal that, compared to GPT-4V, open-source LMMs show a significant gap in OCR, math, and language generation tasks, while they perform decently in recognition, knowledge, and object relationship reasoning. This suggests that future VLM developments should prioritize enhancing language reasoning capabilities. For OCR, the results indicate a need for higher resolution inputs or a more robust backbone model, moving beyond the existing capabilities of models like CLIP.

#### Overfitting Concerns in Region-Level LMMs.

Current region-level LMMs, including Shikra[[5](https://arxiv.org/html/2312.00784v2#bib.bib5)], GPT4ROI[[52](https://arxiv.org/html/2312.00784v2#bib.bib52)] and Kosmos-2[[34](https://arxiv.org/html/2312.00784v2#bib.bib34)], tend to struggle with tasks involving mathematics, relationship reasoning, and language generation. This trend suggests a potential overfitting issue with these models to existing public region-level datasets, which predominantly feature brief descriptions.

Table 6: An example in ViP-Bench where GPT-4V makes a wrong prediction. The correct answer should be Object 2.

7 Conclusion
------------

In summary, ViP-LLaVA shows that visual prompts are promising for region-specific image understanding. By integrating arbitrary visual prompts, we bridge the gap between user-friendly interfaces and the precision required for region comprehension. ViP-LLaVA’s intuitive design leverages natural linguistic interactions coupled with visual markers, simplifying the process of image annotation while enhancing the clarity of visual references. Our state-of-the-art performance on established benchmarks including Visual7W, PointQA, and VCR, underlines the efficacy of ViP-LLaVA. Notably, the introduction of ViP-Bench as a comprehensive evaluative platform sets a new standard for assessing multimodal models’ region reasoning abilities. ViP-LLaVA establishes a foundation for further exploration in the field of intelligent visual systems. We believe that ViP-LLaVA can motivate how visual and linguistic modalities are integrated, enabling more sophisticated and nuanced human-machine interactions.

#### Acknowledgements.

This work was supported in part by NSF CAREER IIS2150012, and Institute of Information & communications Technology Planning & Evaluation(IITP) grants funded by the Korea government(MSIT) (No. 2022-0-00871, Development of AI Autonomy and Knowledge Enhancement for AI Agent Collaboration). (No. RS2022-00187238, Development of Large Korean Language Model Technology for Efficient Pre-training), and Microsoft Accelerate Foundation Models Research Program.

References
----------

*   Abdin et al. [2024] Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. _arXiv preprint arXiv:2404.14219_, 2024. 
*   Ba et al. [2016] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. _arXiv preprint arXiv:1607.06450_, 2016. 
*   Bai et al. [2023] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. _arXiv preprint arXiv:2308.12966_, 2023. 
*   Chen et al. [2023a] Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechu Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. Minigpt-v2: large language model as a unified interface for vision-language multi-task learning. _arXiv preprint arXiv:2310.09478_, 2023a. 
*   Chen et al. [2023b] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. _arXiv preprint arXiv:2306.15195_, 2023b. 
*   Dai et al. [2023] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023. 
*   Denkowski and Lavie [2014] Michael Denkowski and Alon Lavie. Meteor universal: Language specific translation evaluation for any target language. In _Proceedings of the EACL 2014 Workshop on Statistical Machine Translation_, 2014. 
*   Farin [2014] Gerald Farin. _Curves and Surfaces for Computer-Aided Geometric Design: A Practical Guide_. Academic Press, 2014. 
*   Ferraiolo et al. [2000] Jon Ferraiolo, Fujisawa Jun, and Dean Jackson. _Scalable vector graphics (SVG) 1.0 specification_. iuniverse Bloomington, 2000. 
*   Fu et al. [2023] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. _arXiv preprint arXiv:2306.13394_, 2023. 
*   Gan et al. [2020] Zhe Gan, Yen-Chun Chen, Linjie Li, Chen Zhu, Yu Cheng, and Jingjing Liu. Large-scale adversarial training for vision-and-language representation learning. _Advances in Neural Information Processing Systems_, 33:6616–6628, 2020. 
*   Google [2023] Google. Google bard. [https://bard.google.com/chat/](https://bard.google.com/chat/), 2023. 
*   Goyal et al. [2017] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In _Proceedings of the IEEE conference on computer vision and pattern recognition_, pages 6904–6913, 2017. 
*   Gurari et al. [2018] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In _Proceedings of the IEEE conference on computer vision and pattern recognition_, pages 3608–3617, 2018. 
*   Hu et al. [2017] Ronghang Hu, Marcus Rohrbach, Jacob Andreas, Trevor Darrell, and Kate Saenko. Modeling relationships in referential expressions with compositional modular networks. In _Proceedings of the IEEE conference on computer vision and pattern recognition_, pages 1115–1124, 2017. 
*   Hudson and Manning [2019] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In _CVPR_, 2019. 
*   Kirillov et al. [2023] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. _arXiv preprint arXiv:2304.02643_, 2023. 
*   Krishna et al. [2017] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. _International journal of computer vision_, 123:32–73, 2017. 
*   Li et al. [2023a] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. _arXiv preprint arXiv:2307.16125_, 2023a. 
*   Li et al. [2020] Gen Li, Nan Duan, Yuejian Fang, Ming Gong, and Daxin Jiang. Unicoder-vl: A universal encoder for vision and language by cross-modal pre-training. In _Proceedings of the AAAI conference on artificial intelligence_, pages 11336–11344, 2020. 
*   Li et al. [2023b] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. _arXiv preprint arXiv:2305.10355_, 2023b. 
*   Lin et al. [2014] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In _Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13_, pages 740–755. Springer, 2014. 
*   Liu et al. [2023a] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023a. 
*   Liu et al. [2023b] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _arXiv:2304.08485_, 2023b. 
*   Liu et al. [2023c] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? _arXiv preprint arXiv:2307.06281_, 2023c. 
*   Lu et al. [2019] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In _Advances in Neural Information Processing Systems_. Curran Associates, Inc., 2019. 
*   Lu et al. [2020] Jiasen Lu, Vedanuj Goswami, Marcus Rohrbach, Devi Parikh, and Stefan Lee. 12-in-1: Multi-task vision and language representation learning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_, pages 10437–10446, 2020. 
*   Lu et al. [2022] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. _Advances in Neural Information Processing Systems_, 2022. 
*   Mani et al. [2020] Arjun Mani, Nobline Yoo, Will Hinthorn, and Olga Russakovsky. Point and ask: Incorporating pointing into visual question answering. _arXiv preprint arXiv:2011.13681_, 2020. 
*   Meta [2024] Meta. Llama-3. [https://ai.meta.com/blog/meta-llama-3/](https://ai.meta.com/blog/meta-llama-3/), 2024. 
*   OpenAI [2023a] OpenAI. Gpt-4v(ision) system card. [https://cdn.openai.com/papers/GPTV_System_Card.pdf](https://cdn.openai.com/papers/GPTV_System_Card.pdf), 2023a. 
*   OpenAI [2023b] OpenAI. Chatgpt. [https://openai.com/blog/chatgpt/](https://openai.com/blog/chatgpt/), 2023b. 
*   OpenAI [2023c] OpenAI. Gpt-4 technical report. 2023c. 
*   Peng et al. [2023] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. _arXiv preprint arXiv:2306.14824_, 2023. 
*   Plummer et al. [2015] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In _Proceedings of the IEEE international conference on computer vision_, pages 2641–2649, 2015. 
*   Radford et al. [2021] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_, pages 8748–8763. PMLR, 2021. 
*   Rasheed et al. [2023] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Erix Xing, Ming-Hsuan Yang, and Fahad S Khan. Glamm: Pixel grounding large multimodal model. _arXiv preprint arXiv:2311.03356_, 2023. 
*   Shtedritski et al. [2023] Aleksandar Shtedritski, Christian Rupprecht, and Andrea Vedaldi. What does clip know about a red circle? visual prompt engineering for vlms. _ICCV_, 2023. 
*   Singh et al. [2019] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_, pages 8317–8326, 2019. 
*   Su et al. [2019] Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. Vl-bert: Pre-training of generic visual-linguistic representations. _arXiv preprint arXiv:1908.08530_, 2019. 
*   Touvron et al. [2023] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_, 2023. 
*   Vedantam et al. [2015] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In _Proceedings of the IEEE conference on computer vision and pattern recognition_, pages 4566–4575, 2015. 
*   Vicuna [2023] Vicuna. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. [https://vicuna.lmsys.org/](https://vicuna.lmsys.org/), 2023. 
*   Wu et al. [2022] Jialian Wu, Jianfeng Wang, Zhengyuan Yang, Zhe Gan, Zicheng Liu, Junsong Yuan, and Lijuan Wang. Grit: A generative region-to-text transformer for object understanding. _arXiv preprint arXiv:2212.00280_, 2022. 
*   Yang et al. [2023a] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. _arXiv preprint arXiv:2310.11441_, 2023a. 
*   Yang et al. [2023b] Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v (ision). _arXiv preprint arXiv:2309.17421_, 2023b. 
*   You et al. [2023] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. _arXiv preprint arXiv:2310.07704_, 2023. 
*   Yu et al. [2021] Fei Yu, Jiji Tang, Weichong Yin, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. Ernie-vil: Knowledge enhanced vision-language representations through scene graphs. In _Proceedings of the AAAI Conference on Artificial Intelligence_, pages 3208–3216, 2021. 
*   Yu et al. [2016] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In _Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14_, pages 69–85. Springer, 2016. 
*   Yu et al. [2023] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. _arXiv preprint arXiv:2308.02490_, 2023. 
*   Zellers et al. [2019] Rowan Zellers, Yonatan Bisk, Ali Farhadi, and Yejin Choi. From recognition to cognition: Visual commonsense reasoning. In _The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, 2019. 
*   Zhang et al. [2023] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-of-interest. _arXiv preprint arXiv:2307.03601_, 2023. 
*   Zhao et al. [2023] Liang Zhao, En Yu, Zheng Ge, Jinrong Yang, Haoran Wei, Hongyu Zhou, Jianjian Sun, Yuang Peng, Runpei Dong, Chunrui Han, et al. Chatspot: Bootstrapping multimodal llms via precise referring instruction tuning. _arXiv preprint arXiv:2307.09474_, 2023. 
*   Zhou et al. [2022] Chong Zhou, Chen Change Loy, and Bo Dai. Extract free dense labels from clip. In _European Conference on Computer Vision (ECCV)_, 2022. 
*   Zhou et al. [2023] Qiang Zhou, Chaohui Yu, Shaofeng Zhang, Sitong Wu, Zhibing Wang, and Fan Wang. Regionblip: A unified multi-modal pre-training framework for holistic and regional comprehension, 2023. 
*   Zhu et al. [2023] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_, 2023. 
*   Zhu et al. [2016] Yuke Zhu, Oliver Groth, Michael Bernstein, and Li Fei-Fei. Visual7W: Grounded Question Answering in Images. In _IEEE Conference on Computer Vision and Pattern Recognition_, 2016. 
*   Zou et al. [2023] Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li, Jianfeng Gao, and Yong Jae Lee. Segment everything everywhere all at once. _NeurIPS_, 2023. 

Supplementary Material

This supplementary document extends our main paper by providing additional results and in-depth analyses that were not included in the main manuscript due to space limitations. In Section[A](https://arxiv.org/html/2312.00784v2#A1 "Appendix A In-Depth Analysis ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"), we offer both qualitative and quantitative analyses on topics such as visual prompt generation, effect of the instruction data, arrow direction understanding, perform under each visual prompt, and impacts of different LLMs on the conventional vision-language model benchmarks, thus providing a comprehensive examination of our research. In Section[B](https://arxiv.org/html/2312.00784v2#A2 "Appendix B Training Details ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"), we include more training details. Section[C](https://arxiv.org/html/2312.00784v2#A3 "Appendix C Additional Ablation Studies ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") delves into further ablation studies, illuminating the design and efficacy of our approach. Additional experimental results, including a focus on region captioning, are discussed in Section[D](https://arxiv.org/html/2312.00784v2#A4 "Appendix D Additional Experimental Results ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"). The exploration of visual prompt augmentation’s potential is presented in Section[E](https://arxiv.org/html/2312.00784v2#A5 "Appendix E Potential of Visual Prompt Augmentation ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"). Lastly, Section[F](https://arxiv.org/html/2312.00784v2#A6 "Appendix F Further Insights into ViP-Bench ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") comprises detailed statistics and visualizations from the ViP-Bench dataset.

Table 7: Three samples requiring different capability integrations. 

Appendix A In-Depth Analysis
----------------------------

### A.1 Generalization to New Attributes

ViP-LLaVA, having been trained on eight types of visual prompts—namely mask contour, ellipse, bounding box, triangle, scribble, point, arrow, and mask—exhibits notable generalization capabilities. In the main paper, we show that ViP-LLaVA can understand human drawn visual prompts, as shown in Figure[7](https://arxiv.org/html/2312.00784v2#A1.F7 "Figure 7 ‣ A.1 Generalization to New Attributes ‣ Appendix A In-Depth Analysis ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"). As demonstrated in Figures[8](https://arxiv.org/html/2312.00784v2#A1.F8 "Figure 8 ‣ A.1 Generalization to New Attributes ‣ Appendix A In-Depth Analysis ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") and Figure 5 of the main paper, ViP-LLaVA adeptly handles visual prompts with varying thicknesses and diverse markers, even though it was not explicitly trained on such variations. Furthermore, it effectively interprets text markers as visual prompts, a feature inspired by the Set-of-Mark[[45](https://arxiv.org/html/2312.00784v2#bib.bib45)].

Figures[9](https://arxiv.org/html/2312.00784v2#A1.F9 "Figure 9 ‣ A.1 Generalization to New Attributes ‣ Appendix A In-Depth Analysis ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"), [10](https://arxiv.org/html/2312.00784v2#A1.F10 "Figure 10 ‣ A.1 Generalization to New Attributes ‣ Appendix A In-Depth Analysis ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"), and [11](https://arxiv.org/html/2312.00784v2#A1.F11 "Figure 11 ‣ A.1 Generalization to New Attributes ‣ Appendix A In-Depth Analysis ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") present qualitative examples. In Figure[9](https://arxiv.org/html/2312.00784v2#A1.F9 "Figure 9 ‣ A.1 Generalization to New Attributes ‣ Appendix A In-Depth Analysis ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"), ViP-LLaVA accurately localizes objects tagged with the digits “1”, “2”, and “3”, and generates precise descriptions for each. Figure[10](https://arxiv.org/html/2312.00784v2#A1.F10 "Figure 10 ‣ A.1 Generalization to New Attributes ‣ Appendix A In-Depth Analysis ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") showcases the model’s ability to recognize digit markers and describe the color of vehicles accurately, despite the markers displaying counterfactual colors relative to the actual vehicle colors. Figure[11](https://arxiv.org/html/2312.00784v2#A1.F11 "Figure 11 ‣ A.1 Generalization to New Attributes ‣ Appendix A In-Depth Analysis ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") illustrates the model’s competency in localizing a lemon within a scene densely populated with markers.

![Image 7: Refer to caption](https://arxiv.org/html/2312.00784v2/x7.png)

Figure 7: ViP-LLaVA is able to recognize a tiny region specified by the red contour.

![Image 8: Refer to caption](https://arxiv.org/html/2312.00784v2/x8.png)

Figure 8: ViP-LLaVA is able to recognize visual prompts with different thickness not seen during training.

![Image 9: Refer to caption](https://arxiv.org/html/2312.00784v2/x9.png)

Figure 9: ViP-LLaVA successfully localizes objects tagged with digits and generates accurate object descriptions.

![Image 10: Refer to caption](https://arxiv.org/html/2312.00784v2/x10.png)

Figure 10: The model effectively localizes digit markers and accurately predicts vehicle colors, uninfluenced by the marker colors.

![Image 11: Refer to caption](https://arxiv.org/html/2312.00784v2/x11.png)

Figure 11: Demonstration of the model’s ability to localize an object in a densely marked scene.

### A.2 Effect of Optional GPT-4V Region-Level Instruction Data

![Image 12: Refer to caption](https://arxiv.org/html/2312.00784v2/x12.png)

Figure 12: Curation process of region-level instruction data. This figure delineates the workflow where both original and annotated images, along with corresponding text prompts, are integrated into the GPT-4V model, facilitating the generation of detailed instruction data for region-specific tasks.

As mentioned in Section 3.3 of the main paper, incorporating GPT-4V as an additional source of instruction data can enhance ViP-LLaVA’s performance. An example of the curation process is shown in Figure[12](https://arxiv.org/html/2312.00784v2#A1.F12 "Figure 12 ‣ A.2 Effect of Optional GPT-4V Region-Level Instruction Data ‣ Appendix A In-Depth Analysis ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"). For this purpose, we combine 13K data entries from the original stage 2 instruction dataset with an equal number of GPT-4V region-level instruction data entries, forming a comprehensive 26K-entry stage 3 fine-tuning dataset. We fine-tune our stage-2 model for one epoch, which requires approximately 0.5 hours for the 7B model and 1 hour for the 13B model on 8 NVIDIA A100 GPUs. As shown in Table[8](https://arxiv.org/html/2312.00784v2#A1.T8 "Table 8 ‣ A.2 Effect of Optional GPT-4V Region-Level Instruction Data ‣ Appendix A In-Depth Analysis ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"), the fine-tuned model, designated as ViP-LLaVA, demonstrates improvements across nearly all datasets for both the 7B and 13B models, underscoring the efficacy of the GPT-4V instruction data curation process. Notably, even without the GPT-4V instruction data, ViP-LLaVA outperforms contemporary methods on benchmarks such as Visual7W, PointQA-LookTwice, and ViP-Bench. The inclusion of GPT-4V instruction data further amplifies this performance advantage.

Table 8: Comparative performance analysis of the use of GPT-4V data in the 7B and 13B models. † Indicates that GPT4ROI specifically trained a specialist model on Visual7W.

### A.3 Understanding Arrow Direction

To rigorously evaluate ViP-LLaVA’s capacity for interpreting arrow directions, we next construct a challenging dataset of examples derived from the COCO validation set[[22](https://arxiv.org/html/2312.00784v2#bib.bib22)]. Specifically, we generate multiple scenarios with arrows: each arrow originates from the center of one object’s bounding box and points towards the center of another, and vice versa. These visualizations are depicted in Figure[13](https://arxiv.org/html/2312.00784v2#A1.F13 "Figure 13 ‣ A.3 Understanding Arrow Direction ‣ Appendix A In-Depth Analysis ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"). The typical prompt used is as follows: Determine whether object A (category1) or object B (category2) is at the head of the arrow, with the other object representing the tail. It is important to note that we ensure each pair of objects belong to distinct categories. A total of 3520 such paired examples are collected and analyzed. Impressively, ViP-LLaVA-13B achieves an accuracy of 90.28%, demonstrating a robust understanding of arrow directionality and ruling out the possibility of random guessing.

![Image 13: Refer to caption](https://arxiv.org/html/2312.00784v2/x13.png)

Figure 13: An illustration from our arrow direction understanding dataset. Panels (a) and (b) display two arrows. These arrows share a similar body but differ in their heads. In this example, the multimodal model is required to discern whether the arrow points to the orange or the bowl in both images.

### A.4 Performance across Different Visual Prompts

During training, we leverage eight types of visual prompts. Here we study the performance of under each kind of visual prompt on downstream tasks. Note that we duplicate the region-level training data by eight time during instruction fine-tuning.

Results on VCR, Visual7W, PointQA, and ViP-Bench shown in Table[9](https://arxiv.org/html/2312.00784v2#A1.T9 "Table 9 ‣ A.4 Performance across Different Visual Prompts ‣ Appendix A In-Depth Analysis ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") show our 7B model’s consistent accuracy with varied visual prompts, with “Point" and “Ellipse" owning marginally better performance than others.

Table 9: Performance under different visual prompts in VCR, Visual7W, PointQA and ViP-Bench on the 7B model of ViP-LLaVA.

### A.5 Impact of Different LLMs on Image-level and Region-level Benchmarks

Here we study the impact of Large Language Model backbones on both whole image-level and region-level vision-language benchmarks. Specifically, we employ Vicuna-1.5-7B, Vicuna-1.5-13B[[43](https://arxiv.org/html/2312.00784v2#bib.bib43)], Llama-3-8B[[30](https://arxiv.org/html/2312.00784v2#bib.bib30)] and Phi-3-mini 3.8B[[1](https://arxiv.org/html/2312.00784v2#bib.bib1)] as the language model backbone for both LLaVA-1.5[[23](https://arxiv.org/html/2312.00784v2#bib.bib23)] and ViP-LLaVA while keeping all other configurations and hyper-parameters the same. Results for these two types of LMMs are shown in Table[10](https://arxiv.org/html/2312.00784v2#A1.T10 "Table 10 ‣ A.5 Impact of Different LLMs on Image-level and Region-level Benchmarks ‣ Appendix A In-Depth Analysis ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") and Table[11](https://arxiv.org/html/2312.00784v2#A1.T11 "Table 11 ‣ A.5 Impact of Different LLMs on Image-level and Region-level Benchmarks ‣ Appendix A In-Depth Analysis ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"), respectively. The corresponding radar plots are shown in Figure[14](https://arxiv.org/html/2312.00784v2#A1.F14 "Figure 14 ‣ A.5 Impact of Different LLMs on Image-level and Region-level Benchmarks ‣ Appendix A In-Depth Analysis ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") (a) and (b).

Benchmark abbreviations are due to spatial constraints. Whole image-level benchmarks are from the official LLaVA-1.5 pipeline, including MMB: MMBench[[25](https://arxiv.org/html/2312.00784v2#bib.bib25)]; MMB CN CN{}^{\text{CN}}start_FLOATSUPERSCRIPT CN end_FLOATSUPERSCRIPT: MMBench-Chinese[[25](https://arxiv.org/html/2312.00784v2#bib.bib25)]; LLaVA W W{}^{\text{W}}start_FLOATSUPERSCRIPT W end_FLOATSUPERSCRIPT: LLaVA-Bench (In-the-Wild)[[24](https://arxiv.org/html/2312.00784v2#bib.bib24)]; POPE[[21](https://arxiv.org/html/2312.00784v2#bib.bib21)]; SQA I I{}^{\text{I}}start_FLOATSUPERSCRIPT I end_FLOATSUPERSCRIPT: ScienceQA-IMG[[28](https://arxiv.org/html/2312.00784v2#bib.bib28)]; MM-Vet[[50](https://arxiv.org/html/2312.00784v2#bib.bib50)]; VisWiz[[14](https://arxiv.org/html/2312.00784v2#bib.bib14)]; MME[[10](https://arxiv.org/html/2312.00784v2#bib.bib10)]; VQA T T{}^{\text{T}}start_FLOATSUPERSCRIPT T end_FLOATSUPERSCRIPT: TextVQA[[39](https://arxiv.org/html/2312.00784v2#bib.bib39)]; VQA-v2[[13](https://arxiv.org/html/2312.00784v2#bib.bib13)]; GQA[[16](https://arxiv.org/html/2312.00784v2#bib.bib16)]; SEED I I{}^{\text{I}}start_FLOATSUPERSCRIPT I end_FLOATSUPERSCRIPT: SEED-Bench-1[[19](https://arxiv.org/html/2312.00784v2#bib.bib19)] Image subset. Region-level Benchmarks include: V7W: Visual7W[[57](https://arxiv.org/html/2312.00784v2#bib.bib57)]; PointQA: PointQA-LookTwice[[29](https://arxiv.org/html/2312.00784v2#bib.bib29)]; ViP-B Bbox Bbox{}^{\text{Bbox}}start_FLOATSUPERSCRIPT Bbox end_FLOATSUPERSCRIPT: ViP-Bench with the tight bounding box condition; ViP-B Human Human{}^{\text{Human}}start_FLOATSUPERSCRIPT Human end_FLOATSUPERSCRIPT: ViP-Bench with the human annotated visual prompts. Some interesting findings:

*   •Recent LLMs, Llama-3 and Phi-3, are more capable in tasks that require more language and commonsense reasoning. For example, both Llama-3-8B and Phi-3-mini-3.8B receive better performance than Vicuna-1.5-13B on MMBench and ScienceQA. Specifically, under the LLaVA-1.5 framework, Llama-3-8B receives a 7.7 and 5.7 performance boost compared to Vicuna-1.5-13B. These LLMs, Llama-3-8B and Phi-3-mini-3.8B, own fewer parameters but are trained on 15T and 3.3T tokens, respectively, which is much larger than the previous paradigm. As a result, the language understanding and reasoning capability is significantly improved. 
*   •Llama-3-8B and Phi-3-mini-3.8B do not bring a performance boost for tasks that primarily require visual understanding capability rather than language reasoning capability. For example, Vicuna-1.5-13B still performs better on MME, TextVQA, GQA in Table[10](https://arxiv.org/html/2312.00784v2#A1.T10 "Table 10 ‣ A.5 Impact of Different LLMs on Image-level and Region-level Benchmarks ‣ Appendix A In-Depth Analysis ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"), Visual7W, and PointQA in Table[11](https://arxiv.org/html/2312.00784v2#A1.T11 "Table 11 ‣ A.5 Impact of Different LLMs on Image-level and Region-level Benchmarks ‣ Appendix A In-Depth Analysis ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"). Our results indicate that better visual representation is critical for tasks that mainly require core visual understanding capability. 
*   •Phi-3-mini-3.8B shows poor performance on zero-shot vision-language tasks such as VizWiz, while it performs on par with Vicuna-1.5-7B on most tasks. Our results indicate that the generation capability of Phi-3-mini-3.8B for zero-shot tasks is limited. 
*   •Overall, Llama-3-8B performs better than Vicuna-1.5-7B while performing on par with Vicuna-1.5-13B; Phi-3-mini-3.8B performs similarly to Vicuna-1.5-7B on most tasks while underperforming Vicuna-1.5-13B on average. 
*   •ViP-LLaVA maintains the performance of LLaVA-1.5 across various whole image understanding benchmarks. 

![Image 14: Refer to caption](https://arxiv.org/html/2312.00784v2/x14.png)

Figure 14: The effects of different LLMs on LLaVA-1.5 and ViP-LLaVA.

Table 10: Comprehensive benchmarking of LLaVA-1.5[[23](https://arxiv.org/html/2312.00784v2#bib.bib23)] under different LLM backbones.

Table 11: Impact on conventional vision-language model benchmarks and region-level benchmarks under different LLM backbones for ViP-LLaVA.

Appendix B Training Details
---------------------------

Our approach involves three stages:

*   •Pretraining the MLP projector with frozen language and image encoder; 
*   •Instruction tuning and further fine-tuning with frozen image encoder but trainable language model and MLP connector. 

Appendix C Additional Ablation Studies
--------------------------------------

### C.1 Effects of Input Resolution and LLM

To ensure a fair comparison, we conduct ablation studies using the same image encoder (CLIP ViT-L from Radford et al.[[36](https://arxiv.org/html/2312.00784v2#bib.bib36)]), input resolution (224 pixels), and language model (Vicuna v1.1[[43](https://arxiv.org/html/2312.00784v2#bib.bib43)]) as employed by GPT4ROI[[52](https://arxiv.org/html/2312.00784v2#bib.bib52)]. Table[12](https://arxiv.org/html/2312.00784v2#A3.T12 "Table 12 ‣ C.1 Effects of Input Resolution and LLM ‣ Appendix C Additional Ablation Studies ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") presents the results of this analysis. Despite utilizing the same underlying technologies, ViP-LLaVA consistently outperforms on the ViP-Bench evaluations and achieves comparable results on the Visual7W dataset, notwithstanding the fact that GPT4ROI was specifically fine-tuned for Visual7W. These results further reinforce the potential of visual prompting as a more effective approach for region-specific referencing compared to embedding coordinates directly into the language model.

Table 12: Ablation study focusing on the impact of input resolution and language model. All models listed use the Vicuna 7B language model. † Indicates GPT4ROI specifically trained on the Visual7W dataset. VP: visual prompts; ROI: CLIP region of interest (ROI) features and positional embedding.

### C.2 Comparing Visual Prompts with Coordinates

To rigorously evaluate the effectiveness of visual prompts versus coordinate-based region referring formats, we next replace visual prompts with textual coordinates embedded in language descriptions. We train a 7B model using identical data and training schedules. The results, as shown in Table[13](https://arxiv.org/html/2312.00784v2#A3.T13 "Table 13 ‣ C.2 Comparing Visual Prompts with Coordinates ‣ Appendix C Additional Ablation Studies ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"), indicate that visual prompts significantly outperform coordinate formats on the PointQA-LookTwice and ViP-Bench@Box datasets. Performance on the Visual7W dataset remains comparable between the two formats. These comparisons highlight the superiority of visual prompts as a more effective format for region-specific referencing in complex visual tasks.

Table 13: Performance comparison between visual prompts and coordinate formats under ViP-LLaVA-Base-7B. VP: visual prompts; Coor: coordinates as visual prompts.

### C.3 Effects of Splitting Overlaid Images into Two Separate Image

We conduct rigorous ablation study to split the overlaid image into the source image and the image with overlaid cue, where the number of visual tokens are doubled, as shown in Figure[15](https://arxiv.org/html/2312.00784v2#A3.F15 "Figure 15 ‣ C.3 Effects of Splitting Overlaid Images into Two Separate Image ‣ Appendix C Additional Ablation Studies ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"). Specifically, we train 7B models under such two settings. Results in Table[14](https://arxiv.org/html/2312.00784v2#A3.T14 "Table 14 ‣ C.3 Effects of Splitting Overlaid Images into Two Separate Image ‣ Appendix C Additional Ablation Studies ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") shows that those two settings perform comparably.

![Image 15: Refer to caption](https://arxiv.org/html/2312.00784v2/x15.png)

Figure 15: We separate the overlaid images into the original image along with the visual prompts with white background.

Table 14: Performance comparison between different model formats under 7B conditions.

Appendix D Additional Experimental Results
------------------------------------------

### D.1 Region Captioning

Expanding upon the region perception and reasoning tasks discussed in the main paper, we further evaluate ViP-LLaVA’s region captioning capabilities on the RefCOCOg dataset[[49](https://arxiv.org/html/2312.00784v2#bib.bib49)]. This involves fine-tuning the ViP-LLaVA-Base-7B for one epoch subsequent to stage 2 training. As Table[15](https://arxiv.org/html/2312.00784v2#A4.T15 "Table 15 ‣ D.1 Region Captioning ‣ Appendix D Additional Experimental Results ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") illustrates, ViP-LLaVA-Base-7B demonstrates strong performance in region captioning, as evidenced by its scores in both CIDEr[[42](https://arxiv.org/html/2312.00784v2#bib.bib42)] and METEOR[[7](https://arxiv.org/html/2312.00784v2#bib.bib7)] metrics. These results indicate that visual prompting is not only effective for region-specific referencing and reasoning tasks but also shows promising potential in generating precise and contextually relevant captions for specific image regions.

Table 15: Performance of region captioning on the RefCOCOg dataset. The table demonstrates ViP-LLaVA’s effectiveness in generating accurate and contextually relevant captions for specific regions within images.

### D.2 Assessment of GPT-4 as a Judge

To evaluate the consistency of ViP-LLaVA-Base-7B, we employ the GPT-4 text model as a judge, conducting five separate assessments. The observed variance in the overall score is a minimal 0.1, indicating stable performance by the GPT-4 judge across multiple evaluations.

Appendix E Potential of Visual Prompt Augmentation
--------------------------------------------------

A key advantage of ViP-LLaVA approach is the ability to very easily employ prompt augmentation during testing. This entails using various sets of visual prompts and aggregating the predictions for a more accurate final answer. For instance, we can modify the prompt from “the woman within a red rectangle” to “the woman marked with a red scribble”, along with corresponding changes in the overlaid image. As shown in Table[16](https://arxiv.org/html/2312.00784v2#A5.T16 "Table 16 ‣ Appendix E Potential of Visual Prompt Augmentation ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"), ViP-LLaVA-Base-7B achieves further improvements through visual prompt augmentation. This process is lossless, unlike textual coordinate representation, where e.g., perturbing coordinates can reduce localization accuracy.

Table 16: Comparison of performance with and without visual prompt ensembling at test time using ViP-LLaVA-Base-7B.

Appendix F Further Insights into ViP-Bench
------------------------------------------

### F.1 Statistics of ViP-Bench

Table[17](https://arxiv.org/html/2312.00784v2#A6.T17 "Table 17 ‣ F.1 Statistics of ViP-Bench ‣ Appendix F Further Insights into ViP-Bench ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") presents the statistical breakdown of ViP-Bench. The majority of examples focus on recognition capabilities, with a notable proportion (89 examples) requiring Optical Character Recognition (OCR). The proportion of each capability and the combined capabilities are shown in Figure[16](https://arxiv.org/html/2312.00784v2#A6.F16 "Figure 16 ‣ F.1 Statistics of ViP-Bench ‣ Appendix F Further Insights into ViP-Bench ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") and Figure[17](https://arxiv.org/html/2312.00784v2#A6.F17 "Figure 17 ‣ F.1 Statistics of ViP-Bench ‣ Appendix F Further Insights into ViP-Bench ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") respectively.

Table 17: Statistics of ViP-Bench across various categories.

![Image 16: Refer to caption](https://arxiv.org/html/2312.00784v2/extracted/5562405/figs/corrected_individual_capabilities_chart.png)

Figure 16: ViP-Bench proportion of capabilities. The proportion of each capability. The sum of the proportion is larger than 100% because some samples have more than one capability.

![Image 17: Refer to caption](https://arxiv.org/html/2312.00784v2/extracted/5562405/figs/corrected_combined_capabilities_chart.png)

Figure 17: ViP-Bench proportion of capability integrations. 

### F.2 Visualizations of ViP-Bench

Figure[18](https://arxiv.org/html/2312.00784v2#A6.F18 "Figure 18 ‣ F.2 Visualizations of ViP-Bench ‣ Appendix F Further Insights into ViP-Bench ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") showcases examples from ViP-Bench, comparing synthesized and human-annotated visual prompts. Panel (a) illustrates tight bounding boxes as synthesized prompts, while panel (b) features human-annotated bounding boxes, highlighting the diversity in human-driven region referring methods. The text prompt that we use to evaluate ViP-Bench performance using GPT4 text model is similar to that used in MM-Vet, which is shown in Table[18](https://arxiv.org/html/2312.00784v2#A6.T18 "Table 18 ‣ F.3 Examples of capability requirements. ‣ Appendix F Further Insights into ViP-Bench ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts"). Some examples are shown in Table[7](https://arxiv.org/html/2312.00784v2#A0.T7 "Table 7 ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts").

![Image 18: Refer to caption](https://arxiv.org/html/2312.00784v2/x16.png)

Figure 18: Comparison of synthesized and human-annotated visual prompts in ViP-Bench. Panel (a) displays synthesized tight bounding boxes, and panel (b) shows diverse human annotations.

### F.3 Examples of capability requirements.

Table[7](https://arxiv.org/html/2312.00784v2#A0.T7 "Table 7 ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") presents a selection of examples from our benchmark, demonstrating the diverse capabilities required to complete various tasks, whether they involve single-region or multi-region analysis.

Table 18: Few-shot prompt for evaluating model outputs using GPT-4 text model, where 𝒬 𝒬\mathcal{Q}caligraphic_Q is a sample’s question, 𝒢 𝒢\mathcal{G}caligraphic_G is the ground truth and 𝒫 𝒫\mathcal{P}caligraphic_P is the model output for the sample. In the prompt, there are examples with short and long open-ended answers, enabling the evaluation of diverse answer styles. Taking the prompt filled with 𝒬 𝒬\mathcal{Q}caligraphic_Q, 𝒢 𝒢\mathcal{G}caligraphic_G and 𝒫 𝒫\mathcal{P}caligraphic_P, GPT-4 will generate a soft grading score from 0 to 1. 

### F.4 Failure cases of GPT-4V

Tables[19](https://arxiv.org/html/2312.00784v2#A6.T19 "Table 19 ‣ F.4 Failure cases of GPT-4V ‣ Appendix F Further Insights into ViP-Bench ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") to [23](https://arxiv.org/html/2312.00784v2#A6.T23 "Table 23 ‣ F.4 Failure cases of GPT-4V ‣ Appendix F Further Insights into ViP-Bench ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") display various instances where GPT-4V encountered challenges on ViP-Bench. For instance, Table[19](https://arxiv.org/html/2312.00784v2#A6.T19 "Table 19 ‣ F.4 Failure cases of GPT-4V ‣ Appendix F Further Insights into ViP-Bench ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") illustrates a case where both GPT-4V and LLaVA-1.5 incorrectly interpret a yellow scribble, with GPT-4V mistaking a yellow circle for the scribble, leading to erroneous responses. In contrast, ViP-LLaVA accurately answers the questions. Another example in Table[23](https://arxiv.org/html/2312.00784v2#A6.T23 "Table 23 ‣ F.4 Failure cases of GPT-4V ‣ Appendix F Further Insights into ViP-Bench ‣ ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts") (a) shows GPT-4V incorrectly identifying a person marked by a pink point as holding ski poles and LLaVA-1.5 as holding a green flag, while ViP-LLaVA successfully makes the correct prediction.

Table 19: Failure cases for GPT-4V on ViP-Bench.

Table 20: Failure cases for GPT-4V on ViP-Bench.

Table 21: Failure cases for GPT-4V on ViP-Bench.

Table 22: Failure cases for GPT-4V on ViP-Bench.

Table 23: Failure cases for GPT-4V on ViP-Bench.

