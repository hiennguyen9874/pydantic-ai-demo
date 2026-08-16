Title: Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution

URL Source: https://arxiv.org/html/2409.12191

Published Time: Fri, 26 Dec 2025 09:57:38 GMT

Markdown Content:
Peng Wang* Shuai Bai* Sinan Tan* Shijie Wang* Zhihao Fan* Jinze Bai*†

 Keqin Chen Xuejing Liu Jialin Wang Wenbin Ge Yang Fan Kai Dang Mengfei Du 

 Xuancheng Ren Rui Men Dayiheng Liu Chang Zhou Jingren Zhou Junyang Lin†

Qwen Team Alibaba Group

###### Abstract

We present the Qwen2-VL Series, an advanced upgrade of the previous Qwen-VL models that redefines the conventional predetermined-resolution approach in visual processing. Qwen2-VL introduces the Naive Dynamic Resolution mechanism, which enables the model to dynamically process images of varying resolutions into different numbers of visual tokens. This approach allows the model to generate more efficient and accurate visual representations, closely aligning with human perceptual processes. The model also integrates Multimodal Rotary Position Embedding (M-RoPE), facilitating the effective fusion of positional information across text, images, and videos. We employ a unified paradigm for processing both images and videos, enhancing the model’s visual perception capabilities. To explore the potential of large multimodal models, Qwen2-VL investigates the scaling laws for large vision-language models (LVLMs). By scaling both the model size-with versions at 2B, 8B, and 72B parameters-and the amount of training data, the Qwen2-VL Series achieves highly competitive performance. Notably, the Qwen2-VL-72B model achieves results comparable to leading models such as GPT-4o and Claude3.5-Sonnet across various multimodal benchmarks, outperforming other generalist models. Code is available at [https://github.com/QwenLM/Qwen2-VL](https://github.com/QwenLM/Qwen2-VL).

††footnotetext: ∗Equal core contribution, †Corresponding author
Introduction
------------

In the realm of artificial intelligence, Large Vision-Language Models (LVLMs) represent a significant leap forward, building upon the strong textual processing capabilities of traditional large language models. These advanced models now encompass the ability to interpret and analyze a broader spectrum of data, including images, audio, and video. This expansion of capabilities has transformed LVLMs into indispensable tools for tackling a variety of real-world challenges. Recognized for their unique capacity to condense extensive and intricate knowledge into functional representations, LVLMs are paving the way for more comprehensive cognitive systems. By integrating diverse data forms, LVLMs aim to more closely mimic the nuanced ways in which humans perceive and interact with their environment. This allows these models to provide a more accurate representation of how we engage with and perceive our environment

Recent advancements in large vision-language models (LVLMs) (Li et al., [2023c](https://arxiv.org/html/2409.12191v2#bib.bib48); Liu et al., [2023b](https://arxiv.org/html/2409.12191v2#bib.bib54); Dai et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib22); Zhu et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib120); Huang et al., [2023a](https://arxiv.org/html/2409.12191v2#bib.bib34); Bai et al., [2023b](https://arxiv.org/html/2409.12191v2#bib.bib11); Liu et al., [2023a](https://arxiv.org/html/2409.12191v2#bib.bib53); Wang et al., [2023b](https://arxiv.org/html/2409.12191v2#bib.bib99); OpenAI., [2023](https://arxiv.org/html/2409.12191v2#bib.bib71); Team et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib93)) have led to significant improvements in a short span. These models (OpenAI, [2023](https://arxiv.org/html/2409.12191v2#bib.bib70); Touvron et al., [2023a](https://arxiv.org/html/2409.12191v2#bib.bib94), [b](https://arxiv.org/html/2409.12191v2#bib.bib95); Chiang et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib21); Bai et al., [2023a](https://arxiv.org/html/2409.12191v2#bib.bib10)) generally follow a common approach of visual encoder→\rightarrow cross-modal connector→\rightarrow LLM. This setup, combined with next-token prediction as the primary training method and the availability of high-quality datasets (Liu et al., [2023a](https://arxiv.org/html/2409.12191v2#bib.bib53); Zhang et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib118); Chen et al., [2023b](https://arxiv.org/html/2409.12191v2#bib.bib14); Li et al., [2023b](https://arxiv.org/html/2409.12191v2#bib.bib47)), has driven much of the progress. Additional factors like larger model architectures (Alayrac et al., [2022](https://arxiv.org/html/2409.12191v2#bib.bib1)), higher-resolution images (Li et al., [2023a](https://arxiv.org/html/2409.12191v2#bib.bib46), [d](https://arxiv.org/html/2409.12191v2#bib.bib51)), and advanced techniques such as mixture-of-expert models (MoE) (Wang et al., [2023b](https://arxiv.org/html/2409.12191v2#bib.bib99); Ye et al., [2023b](https://arxiv.org/html/2409.12191v2#bib.bib108)), model ensembles (Lin et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib52)), and more sophisticated connectors (Ye et al., [2023a](https://arxiv.org/html/2409.12191v2#bib.bib107)) between visual and textual modalities have also played a key role in enhancing LVLMs’ ability to process complex visual and textual information more effectively.

However, current large vision-language models (LVLMs) are typically constrained by a fixed image input size. Standard LVLMs encode input images to a fixed resolution (e.g., 224×224), often by either downsampling or upsampling the images (Zhu et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib120); Huang et al., [2023a](https://arxiv.org/html/2409.12191v2#bib.bib34)), or by employing a scale-then-padding approach (Liu et al., [2023b](https://arxiv.org/html/2409.12191v2#bib.bib54), [a](https://arxiv.org/html/2409.12191v2#bib.bib53)). While this one-size-fits-all strategy enables processing of images at consistent resolutions, it also limits the model’s ability to capture information at different scales, particularly leading to a significant loss of detailed information in high-resolution images. Consequently, such models fall short of perceiving visual information with the same sensitivity to scale and detail as human vision.

Additionally, most LVLMs rely on a static, frozen CLIP-style (Radford et al., [2021](https://arxiv.org/html/2409.12191v2#bib.bib78)) vision encoder, raising concerns about whether the visual representations produced by such pre-trained models are adequate, particularly for complex reasoning tasks and processing intricate details within images. Recent works (Bai et al., [2023b](https://arxiv.org/html/2409.12191v2#bib.bib11); Ye et al., [2023a](https://arxiv.org/html/2409.12191v2#bib.bib107)) have attempted to address these limitations by fine-tuning the vision transformer (ViT) during the LVLM training process, which has shown to yield improved results. To further enhance the model’s adaptability to varying resolutions, we introduce dynamic resolution training in the LVLM training process. Specifically, we employ a 2D Rotary Position Embedding (RoPE) in the ViT, thus allowing the model to better capture information across different spatial scales.

![Image 1: Refer to caption](https://arxiv.org/html/2409.12191v2/images/qwen2_vl_example.jpg)

Figure 1: Qwen2-VL capabilities: Multilingual image text understanding, code/math reasoning, video analysis, live chat, agent potential, and more. See Appendix for details.

When it comes to video content, which is essentially a sequence of frames, many existing models continue to treat it as an independent modality. However, understanding the dynamic nature of reality, as manifested in videos, is crucial for models aiming to grasp the complexities of the real world. Unlike text, which is inherently one-dimensional, the real-world environment exists in three dimensions. The use of one-dimensional position embeddings in current models significantly limits their ability to model three-dimensional space and temporal dynamics effectively. To bridge this gap, we have developed Multimodal Rotary Position Embedding (M-RoPE), which employs separate components to represent temporal and spatial information. This enables the model to naturally comprehend dynamic content, such as videos or streaming data, improving its ability to understand and interact with the world.

Furthermore, compared to the scaling of large language models (LLMs), current LVLMs are still in the early stages of exploring the impact of scaling in terms of training data and model parameters. The exploration of scaling laws for LVLMs—how increases in model and data size affect performance—remains an open and promising area of research.

In this work, we introduce the newest addition to the large vision-language models of the Qwen family: Qwen2-VL series, which comprises three open-weight models with total parameter counts of 2 billion, 8 billion, and 72 billion. As shown in Figure [1](https://arxiv.org/html/2409.12191v2#S1.F1 "Figure 1 ‣ Introduction ‣ Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution"), the key advances in Qwen2-VL include:

*   •State-of-the-art understanding across various resolutions and aspect ratios: Qwen2-VL achieves leading performance on visual benchmarks, including DocVQA, InfoVQA, RealWorldQA, MTVQA, MathVista, and others. 
*   •Comprehension of extended-duration videos (20 min+): Qwen2-VL is capable of understanding videos over 20 minutes in length, enhancing its ability to perform high-quality video-based question answering, dialogue, content creation, and more. 
*   •Robust agent capabilities for device operation: With advanced reasoning and decision-making abilities, Qwen2-VL can be integrated with devices such as mobile phones, robots, etc., enabling autonomous operation based on visual inputs and text instructions. 
*   •Multilingual support: To serve a global audience, beyond English and Chinese, Qwen2-VL now supports multilingual context understanding within images, including most European languages, Japanese, Korean, Arabic, Vietnamese, and others. 

Table 1: Model descriptions of Qwen2-VL.

Model Name Vision Encoder LLM Model Description
Qwen2-VL-2B 675M 1.5B The most efficient model, designed to run on-device. It delivers adequate performance for most scenarios with limited resources.
Qwen2-VL-7B 675M 7.6B The performance-optimized model in terms of cost, significantly upgraded for text recognition and video understanding capabilities. It delivers significant performance across a broad range of visual tasks.
Qwen2-VL-72B 675M 72B The most capable model, further improvements in visual reasoning, instruction-following, decision-making, and agent capabilities. It delivers optimal performance on most complex tasks.

![Image 2: Refer to caption](https://arxiv.org/html/2409.12191v2/images/qwen2_vl_frame.jpg)

Figure 2: Qwen2-VL is capable of accurately identifying and comprehending the content within images, regardless of their clarity, resolution, or extreme aspect ratios.

Approach
--------

The Qwen2-VL series consists of models of 3 sizes, which are Qwen2-VL-2B, Qwen2-VL-7B and Qwen2-VL-72B. Table [1](https://arxiv.org/html/2409.12191v2#S1.T1 "Table 1 ‣ Introduction ‣ Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution") lists the hyper-parameters and important information. Notably, Qwen2-VL employs a 675M parameter ViT across various-sized LLMs, ensuring that the computational load of the ViT remains constant regardless of the scale of the LLM.

### 2.1 Model Architecture

Figure [2](https://arxiv.org/html/2409.12191v2#S1.F2 "Figure 2 ‣ Introduction ‣ Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution") illustrates the comprehensive structure of Qwen2-VL. We have retained the Qwen-VL (Bai et al., [2023b](https://arxiv.org/html/2409.12191v2#bib.bib11)) framework, which integrates vision encoders and language models. For various scale adaptations, we have implemented a Vision Transformer (ViT) (Dosovitskiy et al., [2021](https://arxiv.org/html/2409.12191v2#bib.bib26)) with approximately 675 million parameters, adept at handling both image and video inputs. In terms of language processing, we have opted for the more powerful Qwen2 (Yang et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib104)) series of language models. To further enhance the model’s ability to effectively perceive and comprehend visual information in videos, we introduced several key upgrades:

##### Naive Dynamic Resolution

![Image 3: Refer to caption](https://arxiv.org/html/2409.12191v2/images/mrope.png)

Figure 3: A demonstration of M-RoPE. By decomposing rotary embedding into temporal, height, and width components, M-RoPE can explicitly model the positional information of text, images, and video in LLM.

A key architectural improvement in Qwen2-VL is the introduction of naive dynamic resolution support (Dehghani et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib25)). Unlike Qwen-VL, Qwen2-VL can now process images of any resolution, dynamically converting them into a variable number of visual tokens.1 1 1 This technology was previously implemented in the internal iterations, Qwen-VL Plus and Qwen-VL MAX. We have further upgraded it in Qwen2-VL. To support this feature, we modified ViT by removing the original absolute position embeddings and introducing 2D-RoPE (Su et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib91); Su, [2021](https://arxiv.org/html/2409.12191v2#bib.bib89)) to capture the two-dimensional positional information of images. At the inference stage, images of varying resolutions are packed into a single sequence, with the packed length controlled to limit GPU memory usage. Furthermore, to reduce the visual tokens of each image, a simple MLP layer is employed after the ViT to compress adjacent 2×2 2\times 2 tokens into a single token, with the special ¡—vision_start—¿ and ¡—vision_end—¿ tokens placed at the beginning and end of the compressed visual tokens. As a result, an image with a resolution of 224×224 224\times 224, encoded with a ViT using patch_size=14, will be compressed to 66 tokens before entering LLM.

##### Multimodal Rotary Position Embedding (M-RoPE)

Another key architectural enhancement is the innovation of Multimodal Rotary Position Embedding (M-RoPE). Unlike the traditional 1D-RoPE in LLMs, which is limited to encoding one-dimensional positional information, M-RoPE effectively models the positional information of multimodal inputs. This is achieved by deconstructing the original rotary embedding into three components: temporal, height, and width. For text inputs, these components utilize identical position IDs, making M-RoPE functionally equivalent to 1D-RoPE (Su, [2024](https://arxiv.org/html/2409.12191v2#bib.bib90)). When processing images, the temporal IDs of each visual token remain constant, while distinct IDs are assigned to the height and width components based on the token’s position in the image. For videos, which are treated as sequences of frames, the temporal ID increments for each frame, while the height and width components follow the same ID assignment pattern as images. In scenarios where the model’s input encompasses multiple modalities, position numbering for each modality is initialized by incrementing the maximum position ID of the preceding modality by one. An illustration of M-RoPE is shown in Figure [3](https://arxiv.org/html/2409.12191v2#S2.F3 "Figure 3 ‣ Naive Dynamic Resolution ‣ 2.1 Model Architecture ‣ Approach ‣ Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution"). M-RoPE not only enhances the modeling of positional information but also reduces the value of position IDs for images and videos, enabling the model to extrapolate to longer sequences during inference.

##### Unified Image and Video Understanding

Qwen2-VL employs a mixed training regimen incorporating both image and video data, ensuring proficiency in image understanding and video comprehension. To preserve video information as completely as possible, we sampled each video at two frames per second. Additionally, we integrated 3D convolutions (Carreira and Zisserman, [2017](https://arxiv.org/html/2409.12191v2#bib.bib12)) with a depth of two to process video inputs, allowing the model to handle 3D tubes instead of 2D patches, thus enabling it to process more video frames without increasing the sequence length (Arnab et al., [2021](https://arxiv.org/html/2409.12191v2#bib.bib8)). For consistency, each image is treated as two identical frames. To balance the computational demands of long video processing with overall training efficiency, we dynamically adjust the resolution of each video frame, limiting the total number of tokens per video to 16384. This training approach strikes a balance between the model’s ability to comprehend long videos and training efficiency.

### 2.2 Training

Following Qwen-VL (Bai et al., [2023b](https://arxiv.org/html/2409.12191v2#bib.bib11)), we adopt a three-stage training methodology. In the first stage, we focus exclusively on training the Vision Transformer (ViT) component, utilizing a vast corpus of image-text pairs to enhance semantic understanding within the Large Language Model (LLM). In the second stage, we unfreeze all parameters and train with a wider range of data for more comprehensive learning. In the final stage, we lock the ViT parameters and perform exclusive fine-tuning of the LLM using instructional datasets.

The model is pre-trained on a diverse dataset that includes image-text pairs, optical character recognition (OCR) data, interleaved image-text articles, visual question answering datasets, video dialogues, and image knowledge datasets. Our data sources primarily comprise cleaned web pages, open-source datasets, and synthetic data. The cutoff date for our data knowledge is June 2023. This diverse data composition is instrumental in developing a robust multimodal understanding capability.

During the initial pre-training phase, Qwen2-VL is exposed to a corpus of around 600 billion tokens. The LLM component of Qwen2-VL is initialized using the parameters from Qwen2 (Yang et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib104)), while the vision encoder of Qwen2-VL is initialized with the ViT derived from DFN. However, the fixed position embedding in the original DFN’s ViT (Fang et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib28)) is replaced by RoPE-2D. This pre-training phase primarily focuses on learning image-text relationships, textual content recognition within images through OCR, and image classification tasks. Such foundational training is instrumental in enabling the model to develop a robust understanding of core visual-textual correlations and alignments.

The second pre-training phase marks a significant progression, involving an additional 800 billion tokens of image-related data. This stage introduces a higher volume of mixed image-text content, facilitating a more nuanced understanding of the interplay between visual and textual information. The incorporation of visual question answering datasets refines the model’s capacity to respond to image-related queries. Moreover, the inclusion of multitasking datasets is pivotal in developing the model’s ability to navigate diverse tasks concurrently, a skill of paramount importance when dealing with complex, real-world datasets. Concurrently, purely textual data continues to play a crucial role in maintaining and advancing the model’s linguistic proficiency.

Throughout the pre-training stages, Qwen2-VL processes a cumulative total of 1.4 trillion tokens. Specifically, these tokens encompass not only text tokens but also image tokens. During the training process, however, we only provide supervision for the text tokens. This exposure to extensive and diverse linguistic and visual scenarios ensures that the model develops a deep understanding of the intricate relationships between visual and textual information, thereby laying a robust foundation for various multimodal tasks.

During the instruction fine-tuning phase, we employ the ChatML (Openai, [2024](https://arxiv.org/html/2409.12191v2#bib.bib72)) format to construct instruction-following data. This dataset encompasses not only pure text-based dialogue data but also multimodal conversational data. The multimodal components include image question-answering, document parsing, multi-image comparison, video comprehension, video stream dialogue, and agent-based interactions. Our comprehensive approach to data construction aims to enhance the model’s capability to understand and execute a wide range of instructions across various modalities. By incorporating diverse data types, we seek to develop a more versatile and robust language model capable of handling complex, multimodal tasks in addition to traditional text-based interactions.

#### 2.2.1 Data Format.

In line with Qwen-VL, Qwen2-VL also employs special tokens to distinguish vision and text inputs. Tokens ¡—vision_start—¿ and ¡—vision_end—¿ are inserted at the start and end of the image feature sequence to demarcate the image content.

##### Dialogue Data.

In terms of dialogue format, we construct our instruction tuning dataset using the ChatML format, where each interaction’s statement is marked with two special tokens (¡—im_start—¿ and ¡—im_end—¿) to facilitate dialogue termination. The sections marked in blue indicate the supervised parts.

##### Visual Grounding.

To endow the model with visual grounding capabilities, bounding box coordinates are normalized within [0, 1000) and represented as ”(X top left,Y top left),(X bottom right,Y bottom right)(X_{\text{top left}},Y_{\text{top left}}),(X_{\text{bottom right}},Y_{\text{bottom right}})”. Tokens ¡—box_start—¿ and ¡—box_end—¿ are utilized to demarcate bounding box text. To accurately link bounding boxes with their textual descriptions, we introduce tokens ¡—object_ref_start—¿ and ¡—object_ref_end—¿ to indicate the content that the bounding box references, thereby allowing the model to effectively interpret and generate precise descriptions of specific regions.

##### Visual Agent.

To develop Qwen2-VL as a general-purpose VL-Agent, we treat various agent tasks, such as UI Operations, Robotic Control, Games, and Navigation, as sequential decision-making problems, enabling Qwen2-VL to accomplish tasks through multi-step action execution. For each task, we first define a set of permissible actions and keywords pattern (underline) for function call (Qwen Team, [2024](https://arxiv.org/html/2409.12191v2#bib.bib77)). Qwen2-VL then analyzes the observations, performs reasoning and planning, executes the selected actions, and interacts with the environment to acquire new observations. This cycle repeats iteratively until the task is successfully completed. By integrating various tools and leveraging the vision perception capabilities of large vision-language models (LVLMs), Qwen2-VL is able to iteratively execute increasingly complex tasks involving real-world visual interactions.

### 2.3 Multimodal Model Infrastructure

The Qwen2-VL models were trained on Alibaba Cloud’s PAI-Lingjun Intelligent Computing Service (Alibaba-Cloud, [2024c](https://arxiv.org/html/2409.12191v2#bib.bib4)) with its scalable computing, auto resuming and straggler detection.

##### Storage.

We use Alibaba Cloud’s ultra-speed CPFS (Cloud Parallel File Storage) (Alibaba-Cloud, [2024a](https://arxiv.org/html/2409.12191v2#bib.bib2)) to build a storage system of Qwen2-VL pre-training and post-training. We decoupled the text data and vision data storage. We simply store text data on CPFS and use mmap for efficient access. For vision data, we use Alibaba Cloud’s OSS (Object Storage Service) (Alibaba-Cloud, [2024b](https://arxiv.org/html/2409.12191v2#bib.bib3)) for persistent storage. During training, we accessed vision data through OSS’s python-client concurrently and tuned the concurrency and retrying parameters to avoid reaching the QPS (queries per second) limit. We also found that video data decoding is a main bottleneck, especially for long videos. After several attempts with open-source (FFmpeg-Developers, [2024](https://arxiv.org/html/2409.12191v2#bib.bib29)) and in-house software failed, we opted for a caching decoding technique. Checkpointing saves each GPU’s optimizer and model states on CPFS.

##### Parallelism.

We use 3D parallelism which combines data parallelism (DP) (Li et al., [2020](https://arxiv.org/html/2409.12191v2#bib.bib50)), tensor parallelism (TP) (Krizhevsky et al., [2012](https://arxiv.org/html/2409.12191v2#bib.bib44); Shoeybi et al., [2019](https://arxiv.org/html/2409.12191v2#bib.bib83)) and pipeline parallelism (PP) (Huang et al., [2019](https://arxiv.org/html/2409.12191v2#bib.bib36); Narayanan et al., [2021](https://arxiv.org/html/2409.12191v2#bib.bib67); Lamy-Poirier, [2023](https://arxiv.org/html/2409.12191v2#bib.bib45)) to scale Qwen2-VL model training. We also leverage deepspeed’s zero-1 redundancy optimizer (Rajbhandari et al., [2020](https://arxiv.org/html/2409.12191v2#bib.bib79)) to shard states for memory saving. Sequence parallelism (SP) (Korthikanti et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib43)) with selective checkpointing activation (Chen et al., [2016](https://arxiv.org/html/2409.12191v2#bib.bib17)) was leveraged to reduce memory usage. When enabling TP training, we always shard the vision encoder and large language models together but not the vision merger due to its relatively few parameters. We found the TP training would result in different model shared-weights due to the convolution operator’s non-deterministic behavior 2 2 2[https://pytorch.org/docs/stable/notes/randomness.html](https://pytorch.org/docs/stable/notes/randomness.html). We resolved this issue by performing offline reduction of the shared weights, thereby avoiding an additional all-reduce communication step. This approach resulted in only a minimal impact on performance. We leverage 1F1B PP (Narayanan et al., [2021](https://arxiv.org/html/2409.12191v2#bib.bib67)) for Qwen2-VL 72B training. We combine the vision encoder, vision adapter and several LLM’s decoder layers into one stage, and evenly split the remaining decoder layers. Note that the vision and text sequence lengths are dynamic for each data point. We broadcast the dynamic sequence lengths before initiating the 1F1B process and access the shape information using batch indices. We also implemented an interleaved 1F1B PP (Narayanan et al., [2021](https://arxiv.org/html/2409.12191v2#bib.bib67)) but found it is slower than the standard 1F1B setting.

##### Software.

We use PyTorch (Paszke et al., [2019](https://arxiv.org/html/2409.12191v2#bib.bib74); Ansel et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib6)) version 2.1.2 with CUDA 11.8 (Nvidia, [2024b](https://arxiv.org/html/2409.12191v2#bib.bib69)) for training. Additionally, we leverage flash-attention (Dao et al., [2022](https://arxiv.org/html/2409.12191v2#bib.bib24); Dao, [2024](https://arxiv.org/html/2409.12191v2#bib.bib23); Shah et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib82)) for efficient training in both the vision encoder and the LLM. We also utilize fused operators (Nvidia, [2024a](https://arxiv.org/html/2409.12191v2#bib.bib68)) such as LayerNorm (Ba et al., [2016](https://arxiv.org/html/2409.12191v2#bib.bib9)), RMSNorm (Zhang and Sennrich, [2019](https://arxiv.org/html/2409.12191v2#bib.bib115)), and Adam (Loshchilov and Hutter, [2019](https://arxiv.org/html/2409.12191v2#bib.bib58)). Besides this, we leverage the overlap of communication and computation during matrix multiplication in our training process.

Experiments
-----------

In this section, we first evaluate the model’s performance by conducting a comparative analysis across a variety of visual benchmarks, demonstrating the advantages of our approach. Subsequently, we carry out a detailed examination of specific capabilities, including general visual perception, document understanding, multilingual recognition in images, video comprehension, and agent abilities. Finally, we present an ablation study to investigate several key components of our approach.

Table 2: Performance Comparison of Qwen2-VL Models and State-of-the-art.

Benchmark Previous SoTA Claude-3.5 Sonnet GPT-4o Qwen2-VL-72B Qwen2-VL-7B Qwen2-VL-2B
MMMU val(Yue et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib111))66.1 (X.AI, [2024b](https://arxiv.org/html/2409.12191v2#bib.bib101))68.3 69.1 64.5 54.1 41.1
DocVQA test(Mathew et al., [2021](https://arxiv.org/html/2409.12191v2#bib.bib66))94.1 (Chen et al., [2024c](https://arxiv.org/html/2409.12191v2#bib.bib20))95.2 92.8 96.5 94.5 90.1
InfoVQA test(Mathew et al., [2021](https://arxiv.org/html/2409.12191v2#bib.bib66))82.0 (Chen et al., [2024c](https://arxiv.org/html/2409.12191v2#bib.bib20))--84.5 76.5 65.5
AI2D (Kembhavi et al., [2016](https://arxiv.org/html/2409.12191v2#bib.bib40))87.6 (Chen et al., [2024c](https://arxiv.org/html/2409.12191v2#bib.bib20))80.2(94.7)84.6(94.2)88.1 83.0 74.7
ChartQA test(Masry et al., [2022](https://arxiv.org/html/2409.12191v2#bib.bib65))88.4 (Chen et al., [2024c](https://arxiv.org/html/2409.12191v2#bib.bib20))90.8 85.7 88.3 83.0 73.5
TextVQA val(Singh et al., [2019](https://arxiv.org/html/2409.12191v2#bib.bib87))84.4 (Chen et al., [2024c](https://arxiv.org/html/2409.12191v2#bib.bib20))--85.5 84.3 79.7
OCRBench (Liu et al., [2023e](https://arxiv.org/html/2409.12191v2#bib.bib57))852 (Yao et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib106))788 736 877 866 809
MTVQA (Tang et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib92))23.2 (Team et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib93))25.7 27.8 30.9 25.6 18.1
VCR en easy(Zhang et al., [2024c](https://arxiv.org/html/2409.12191v2#bib.bib119))84.7 (Chen et al., [2024c](https://arxiv.org/html/2409.12191v2#bib.bib20))63.9 91.6 91.9 89.7 81.5
VCR zh easy(Zhang et al., [2024c](https://arxiv.org/html/2409.12191v2#bib.bib119))22.1 (Chen et al., [2024c](https://arxiv.org/html/2409.12191v2#bib.bib20))1.0 14.9 65.4 59.9 46.2
RealWorldQA (X.AI, [2024a](https://arxiv.org/html/2409.12191v2#bib.bib100))72.2 (Chen et al., [2024c](https://arxiv.org/html/2409.12191v2#bib.bib20))60.1 75.4 77.8 70.1 62.9
MME sum(Fu et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib30))2414.7 (Chen et al., [2024c](https://arxiv.org/html/2409.12191v2#bib.bib20))1920.0 2328.7 2482.7 2326.8 1872.0
MMBench-EN test(Liu et al., [2023d](https://arxiv.org/html/2409.12191v2#bib.bib56))86.5(Chen et al., [2024c](https://arxiv.org/html/2409.12191v2#bib.bib20))79.7 83.4 86.5 83.0 74.9
MMBench-CN test(Liu et al., [2023d](https://arxiv.org/html/2409.12191v2#bib.bib56))86.3 (Chen et al., [2024c](https://arxiv.org/html/2409.12191v2#bib.bib20))80.7 82.1 86.6 80.5 73.5
MMBench-V1.1 test(Liu et al., [2023d](https://arxiv.org/html/2409.12191v2#bib.bib56))85.5 (Chen et al., [2024c](https://arxiv.org/html/2409.12191v2#bib.bib20))78.5 82.2 85.9 80.7 72.2
MMT-Bench test(Ying et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib109))63.4 (Chen et al., [2024b](https://arxiv.org/html/2409.12191v2#bib.bib19))-65.5 71.7 63.7 54.5
MMStar (Chen et al., [2024a](https://arxiv.org/html/2409.12191v2#bib.bib15))67.1 (Chen et al., [2024c](https://arxiv.org/html/2409.12191v2#bib.bib20))62.2 63.9 68.3 60.7 48.0
MMVet GPT-4-Turbo(Yu et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib110))67.5 (OpenAI., [2023](https://arxiv.org/html/2409.12191v2#bib.bib71))66.0 69.1 74.0 62.0 49.5
HallBench avg(Guan et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib32))55.2 (Chen et al., [2024c](https://arxiv.org/html/2409.12191v2#bib.bib20))49.9 55.0 58.1 50.6 41.7
MathVista testmini(Lu et al., [2024a](https://arxiv.org/html/2409.12191v2#bib.bib61))69.0 (X.AI, [2024b](https://arxiv.org/html/2409.12191v2#bib.bib101))67.7 63.8 70.5 58.2 43.0
MathVision (Wang et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib96))30.3 (OpenAI, [2023](https://arxiv.org/html/2409.12191v2#bib.bib70))-30.4 25.9 16.3 12.4
MMMU-Pro (Yue et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib112))46.9 (Team et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib93))51.5 51.9 46.2 43.5 37.6

Table 3: Performance of Qwen2-VL and GPT-4o on internal multilingual OCR benchmarks.

Language Korean Japanese French German Italian Russian Vietnamese Arabic
GPT-4o 87.8 88.3 89.7 88.3 74.1 96.8 72.0 75.9
Qwen2-VL-72B 94.5 93.4 94.1 91.5 89.8 97.2 73.0 70.7

Table 4: Performance of Qwen2-VL and other models on video benchmarks.

Benchmark Previous SoTA Gemini 1.5-Pro GPT-4o Qwen2-VL-72B Qwen2-VL-7B Qwen2-VL-2B
MVBench (Li et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib49))69.6--73.6 67.0 63.2
PerceptionTest test(Patraucean et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib75))66.9--68.0 62.3 53.9
EgoSchema test(Mangalam et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib63))62.0 63.2 72.2 77.9 66.7 54.9
Video-MME(wo/w subs)(Fu et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib31))66.3/69.6 75.0/81.3 71.9/77.2 71.2/77.8 63.3/69.0 55.6/60.4

Table 5: Performance Comparison of Qwen2-VL-72B across various agent benchmarks and GPT-4o. SR, GC, TM and EM are short for success rate, goal-condition success, type match and exact match. ALFRED, R2R and REVERIE are performance in valid-unseen.

Benchmark Metric Previous SoTA GPT-4o Qwen2-VL-72B
General FnCall TM-90.2 93.1
EM-50.0 53.2
UI Operations AITZ (Zhang et al., [2024b](https://arxiv.org/html/2409.12191v2#bib.bib117))TM 83.0 (Hong et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib33))70.0 89.6
EM 47.7 (Zhan and Zhang, [2023](https://arxiv.org/html/2409.12191v2#bib.bib114))35.3 72.1
Card Games Number Line (Zhai et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib113))SR 89.4 (Zhai et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib113))91.5 100.0
BlackJack (Zhai et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib113))SR 40.2 (Zhai et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib113))34.5 42.6
EZPoint (Zhai et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib113))SR 50.0 (Zhai et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib113))85.5 100.0
Point24 (Zhai et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib113))SR 2.6 (Liu et al., [2023b](https://arxiv.org/html/2409.12191v2#bib.bib54))3.0 4.5
Robotic Control ALFRED (Shridhar et al., [2020a](https://arxiv.org/html/2409.12191v2#bib.bib84))SR 67.7 (Lu et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib59))-67.8
GC 75.3 (Lu et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib59))-75.8
Navigation R2R (Anderson et al., [2018](https://arxiv.org/html/2409.12191v2#bib.bib5))SR 79.0(Chen et al., [2022](https://arxiv.org/html/2409.12191v2#bib.bib16))43.7 51.7
REVERIE (Qi et al., [2020](https://arxiv.org/html/2409.12191v2#bib.bib76))SR 61.0(Sigurdsson et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib86))31.6 31.0

### 3.1 Compare to SOTAs

We evaluate the visual capabilities of our model through various visual benchmarks, video tasks, and agent-based assessments. Qwen2-VL demonstrates highly competitive performance at the same scale, achieving new state-of-the-art (SoTA) results. Overall, our 72B model consistently delivers top-tier performance across most evaluation metrics, frequently surpassing even closed-source models such as GPT-4o (OpenAI, [2024](https://arxiv.org/html/2409.12191v2#bib.bib73)) and Claude 3.5-Sonnet (Anthropic, [2024](https://arxiv.org/html/2409.12191v2#bib.bib7)). Notably, it exhibits a significant advantage in document understanding tasks. However, in the MMMU (Yue et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib111)) benchmark, our model still lags behind GPT-4o to some extent, indicating that Qwen2-VL-72B has room for improvement when handling more complex and challenging problem sets.

### 3.2 Quantitative Results

In this section, we present an extensive evaluation of the Qwen2-VL series across an array of datasets, offering a comprehensive understanding of the model’s capabilities in various aspects.

#### 3.2.1 General Visual Question Answering

To rigorously assess our models’ capabilities in general visual question answering tasks, we conduct extensive evaluations across a diverse array of state-of-the-art benchmarks: RealWorldQA (X.AI, [2024a](https://arxiv.org/html/2409.12191v2#bib.bib100)), MMStar (Chen et al., [2024a](https://arxiv.org/html/2409.12191v2#bib.bib15)), MMVet (Yu et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib110)), MMT-Bench (Ying et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib109)), MMBench (Liu et al., [2023d](https://arxiv.org/html/2409.12191v2#bib.bib56)), MMbench-1.1 (Liu et al., [2023d](https://arxiv.org/html/2409.12191v2#bib.bib56)), MME (Fu et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib30)), and HallusionBench (Guan et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib32)). The Qwen2-VL series exhibits exceptional performance across these benchmarks, with the 72B model consistently achieving or surpassing state-of-the-art results, while the 7B and 2B variants also demonstrate robust capabilities. On RealWorldQA, which evaluates real-world spatial comprehension, Qwen2-VL-72B achieves a score of 77.8, surpassing both the previous state-of-the-art (72.2) and formidable baselines such as GPT-4o (75.4), thus demonstrating superior understanding of physical environments. For MMStar, a benchmark designed to assess genuine multimodal capabilities through visually indispensable samples, Qwen2-VL-72B attains 68.3, outperforming the previous best of 67.1 and highlighting its proficiency in integrating visual and textual information. On MMVet, which evaluates the integration of core vision-language capabilities across 16 complex multimodal tasks, Qwen2-VL-72B achieves a remarkable 74.0, significantly outperforming strong competitors including GPT-4V (67.5) and showcasing its versatility in addressing diverse multimodal challenges. In the MMT-Bench evaluation, which assesses advanced reasoning and instruction following across 32 core meta-tasks and 162 subtasks in multimodal understanding, Qwen2-VL-72B achieves 71.7, markedly surpassing the previous best (63.4) and demonstrating its prowess in applying expert knowledge and executing deliberate visual recognition, localization, reasoning, and planning. On MMBench, which evaluates fine-grained abilities across 20 dimensions, Qwen2-VL-72B exhibits strong performance, achieving 86.5 on the English test set, matching the state-of-the-art, and 86.6 on the Chinese test set, establishing a new benchmark. For MME, which measures a wide spectrum of perception and cognition abilities across 14 subtasks, Qwen2-VL-72B achieves a cumulative score of 2482.7, significantly outperforming the previous best (2414.7), underscoring its advanced capabilities in both visual perception and high-level cognition tasks.

These comprehensive results underscore the Qwen2-VL series’ exceptional proficiency in general visual question answering tasks. The models demonstrate advanced capabilities in real-world spatial comprehension, genuine multimodal integration, complex reasoning, instruction following, and a broad range of perception and cognition tasks. The consistent superior performance across diverse benchmarks, particularly the outstanding results of the 72B model, positions the Qwen2-VL series as a leading solution in the field of visual question answering. Our models excel in handling visually indispensable tasks, integrating core vision-language capabilities, and demonstrating expertise across diverse multimodal scenarios, ranging from fundamental perception tasks to complex reasoning and planning. This exhaustive evaluation highlights the Qwen2-VL series’ versatility and effectiveness in addressing the multifaceted challenges posed by state-of-the-art multimodal benchmarks, thereby setting a new standard for large vision-language models.

#### 3.2.2 Document and Diagrams Reading

We tested our model’s OCR and document and diagram comprehension on DocVQA (Mathew et al., [2021](https://arxiv.org/html/2409.12191v2#bib.bib66)), ChartQA (Masry et al., [2022](https://arxiv.org/html/2409.12191v2#bib.bib65)),InfoVQA (Mathew et al., [2021](https://arxiv.org/html/2409.12191v2#bib.bib66)), TextVQA (Singh et al., [2019](https://arxiv.org/html/2409.12191v2#bib.bib87)),AI2D (Kembhavi et al., [2016](https://arxiv.org/html/2409.12191v2#bib.bib40)) datasets. The DocVQA/InfoVQA/ChartQA dataset focuses on the model’s ability to comprehend text in documents/high-resolution infographics/charts, while the TextVQA dataset examines the ability to comprehend text in naturalistic images. The OCRBench dataset is a a dataset of mixed tasks, which focuses on mathematical formula parsing and information extraction in addition to the text-based VQA. The AI2D dataset focuses on multiple-choice questions on scientific diagrams containing text. In addition, we also tested the OCR and formula recognition capabilities of our model on OCRBench (Liu et al., [2023e](https://arxiv.org/html/2409.12191v2#bib.bib57)), as well as the multilingual OCR capabilities of our model on the MTVQA (Tang et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib92)) dataset.

The experimental results show that our model achieves SoTA level in several metrics, including DocVQA, InfoVQA, TextVQA and OCRBench, demonstrating that our model has good comprehension of textual content in images from multiple domains.

#### 3.2.3 Multilingual Text Recognition and Understanding

In particular, our model surpasses all existing general-purpose LVLMs in multilingual OCR. Our model not only outperforms existing LVLMs (including proprietary models such as GPT-4o, Claude 3.5 Sonnet, etc.) on the public-available MTVQA dataset, it also outperforms GPT-4o on the in-house internal benchmark across all foreign languages except Arabic (Table [3](https://arxiv.org/html/2409.12191v2#S3.T3 "Table 3 ‣ Experiments ‣ Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution")).

#### 3.2.4 Mathematical Reasoning

We’ve conducted experiments on the MathVista (Lu et al., [2024a](https://arxiv.org/html/2409.12191v2#bib.bib61)) and MathVision (Wang et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib96)) datasets to assess mathematical reasoning capabilities. MathVista is a comprehensive benchmark featuring 6,141 diverse examples of mathematical and visual tasks. The MathVision dataset comprises 3,040 math problems embedded in visual contexts from actual math competitions, covering 16 mathematical disciplines and varying in difficulty across five levels. These challenges underscore the necessity for LVLMs to exhibit strong visual comprehension, a deep understanding of mathematics, and sound logical reasoning skills. The Qwen2-VL series has demonstrated superior performance on MathVista, achieving a 70.5 outperforming other LVLMs. Additionally, it has set a new open-source benchmark on MathVision with 25.9.

#### 3.2.5 Referring Expression Comprehension

Regarding visual localization task, we evaluate Qwen2-VL on RefCOCO, RefCOCO+, and RefCOCOg datasets (Kazemzadeh et al., [2014](https://arxiv.org/html/2409.12191v2#bib.bib39); Mao et al., [2016](https://arxiv.org/html/2409.12191v2#bib.bib64)). The results, as depicted in Table [6](https://arxiv.org/html/2409.12191v2#S3.T6 "Table 6 ‣ 3.2.5 Referring Expression Comprehension ‣ 3.2 Quantitative Results ‣ Experiments ‣ Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution"), demonstrate that Qwen2-VL attains top-tier results among generalist models. Benefiting from a more rational structure design, Qwen2-VL is able to perceive details in high-resolution images, leading to significant improvements over Qwen-VL. The superiority of these models in comparison to both generalist and specialized models highlights their potential for advancing the field of visual localization and their capacity for real-world implementation in tasks requiring precise visual understanding.

Table 6: Performance Comparison on Referring Expression Comprehension Task.

Type Model RefCOCO RefCOCO+RefCOCOg
val test-A test-B val test-A test-B val test
Generalist OFA-L (Wang et al., [2022](https://arxiv.org/html/2409.12191v2#bib.bib97))80.0 83.7 76.4 68.3 76.0 61.8 67.6 67.6
Shikra (Chen et al., [2023a](https://arxiv.org/html/2409.12191v2#bib.bib13))87.0 90.6 80.2 81.6 87.4 72.1 82.3 82.2
Qwen-VL (Bai et al., [2023b](https://arxiv.org/html/2409.12191v2#bib.bib11))89.4 92.3 85.3 83.1 88.3 77.2 85.6 85.5
Ferretv2 (Zhang et al., [2024a](https://arxiv.org/html/2409.12191v2#bib.bib116))92.6 95.0 88.9 87.4 92.1 81.4 89.4 90.0
CogVLM (Wang et al., [2023b](https://arxiv.org/html/2409.12191v2#bib.bib99))92.8 94.8 89.0 88.7 92.9 83.4 89.8 90.8
InternVL2 2b(Chen et al., [2024c](https://arxiv.org/html/2409.12191v2#bib.bib20))82.3 88.2 75.9 73.5 82.8 63.3 77.6 78.3
InternVL2 8b(Chen et al., [2024c](https://arxiv.org/html/2409.12191v2#bib.bib20))87.1 91.1 80.7 79.8 87.9 71.4 82.7 82.7
InternVL2 76b(Chen et al., [2024c](https://arxiv.org/html/2409.12191v2#bib.bib20))92.2 94.8 88.4 88.8 93.1 82.8 89.5 90.3
Qwen2-VL 2b 87.6 90.6 82.3 79.0 84.9 71.0 81.2 80.3
Qwen2-VL 7b 91.7 93.6 87.3 85.8 90.5 79.5 87.3 87.8
Qwen2-VL 72b 93.2 95.3 90.7 90.1 93.8 85.6 89.9 90.4
Specialist G-DINO-L (Liu et al., [2023c](https://arxiv.org/html/2409.12191v2#bib.bib55))90.6 93.2 88.2 82.8 89.0 75.9 86.1 87.0
UNINEXT-H (Yan et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib102))92.6 94.3 91.5 85.2 89.6 79.8 88.7 89.4
ONE-PEACE (Wang et al., [2023a](https://arxiv.org/html/2409.12191v2#bib.bib98))92.6 94.2 89.3 88.8 92.2 83.2 89.2 89.3

#### 3.2.6 Video Understanding

We evaluate our models on various video understanding tasks, with related benchmarks covering short videos of a few seconds to long videos of up to one hour. Table [4](https://arxiv.org/html/2409.12191v2#S3.T4 "Table 4 ‣ Experiments ‣ Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution") presents the performance of Qwen2-VL and baseline models. Overall, Qwen2-VL demonstrates strong results across 2B, 7B, and 72B sizes, with Qwen2-VL-72B achieving the best performance on MVBench (Li et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib49)), PerceptionTest (Patraucean et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib75)), and EgoSchema (Mangalam et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib63)). This showcases Qwen2-VL’s superior capabilities in video understanding tasks, and scaling up Qwen2-VL yields significant improvements. For the challenging Video-MME benchmark (Fu et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib31)), which includes videos up to one hour, it is noteworthy that we limited the maximum number of frames extracted per video to 768 during evaluation, potentially impacting performance on longer videos. Future work will focus on extending Qwen2-VL to support longer sequences, thereby accommodating longer videos.

#### 3.2.7 Visual Agent

Qwen2-VL is evaluated first for its ability to interact with the environment via function calls and then for its capacity to complete complex sequential decision tasks through multiple rounds of interaction. The implementation is based on the Qwen-Agent framework (Qwen Team, [2024](https://arxiv.org/html/2409.12191v2#bib.bib77)).

##### Function Calling

Unlike function calling in LLMs (Yan et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib103); Srinivasan et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib88); Chen et al., [2023c](https://arxiv.org/html/2409.12191v2#bib.bib18)), function calling in LVLMs often involves extracting information from visual cues. Due to the absence of public benchmarks for evaluating the capabilities of LVLMs in function calling, we constructed our internal evaluation dataset.

To construct the evaluation dataset, we undertook the following procedures (Chen et al., [2023c](https://arxiv.org/html/2409.12191v2#bib.bib18)): Scene Categorization, Image Collection, Image Content Extraction, and Question/Functions/Arguments Generation. Firstly, we classified scenes into categories based on different visual applications. Subsequently, we downloaded and meticulously selected high-quality, representative images from the internet for each category. Thereafter, utilizing an advanced LVLM (Bai et al., [2023b](https://arxiv.org/html/2409.12191v2#bib.bib11)), we analyzed each image to extract key visual elements and textual information. Finally, based on the content information from the images, we used an advanced LLM (Yang et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib104)) to generate a series of questions that required specific functions to answer, along with specifying the input parameters needed for these function calls.

Similar to the function calling evaluation method in LLMs (Yan et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib103)), we designed two metrics to evaluate the accuracy of the function selection and the correctness of the arguments input. Specifically, Type Match(TM), is calculated as the ratio of times the model successfully invoked the correct function to the total number of calls attempted. Exact Match(EM), for each function calling, we checked whether the arguments passed to the function exactly matched those recorded in the image’s content information, calculating this correctness ratio.

As shown in Table [5](https://arxiv.org/html/2409.12191v2#S3.T5 "Table 5 ‣ Experiments ‣ Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution"), the performance of Qwen2-VL in both Type Match(93.1 vs. 90.2) and Exact Match(53.2 vs. 50.0) over GPT-4o substantiates the efficacy of Qwen2-VL’s capability in function calling, thereby underscoring its significant potential for application expansion through external tool integration.

The evaluation results demonstrated that GPT-4o underperformed, primarily due to two factors: in scenarios where uncertainty arises, GPT-4o demonstrates a conservative approach by avoiding using external tools. The Optical Character Recognition (OCR) capability of GPT-4o is outperformed by Qwen2-VL, particularly in the context of Chinese characters.

##### UI Operations/Games/Robotics/Navigation

To assess Qwen2-VL’s ability to generally handle complex tasks, we conduct evaluations across multiple VL agent tasks, including mobile operations (Zhang et al., [2024b](https://arxiv.org/html/2409.12191v2#bib.bib117); Rawles et al., [2024b](https://arxiv.org/html/2409.12191v2#bib.bib81); Lu et al., [2024b](https://arxiv.org/html/2409.12191v2#bib.bib62); Rawles et al., [2024a](https://arxiv.org/html/2409.12191v2#bib.bib80)), robotic control (Kolve et al., [2017](https://arxiv.org/html/2409.12191v2#bib.bib42); Shridhar et al., [2020a](https://arxiv.org/html/2409.12191v2#bib.bib84); Inoue and Ohashi, [2022](https://arxiv.org/html/2409.12191v2#bib.bib37); Lu et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib59); Jiang et al., [2022](https://arxiv.org/html/2409.12191v2#bib.bib38); Huang et al., [2023b](https://arxiv.org/html/2409.12191v2#bib.bib35)), card games (Zhai et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib113)), and vision-language navigation (Anderson et al., [2018](https://arxiv.org/html/2409.12191v2#bib.bib5); Qi et al., [2020](https://arxiv.org/html/2409.12191v2#bib.bib76)). As these tasks need multiple actions to complete tasks, we keep the history (observation, action) through Qwen2-VL supports a 32K context length, then append each new observation image after every action, enabling continuous reasoning about subsequent steps.

UI Operations: we evaluate Qwen2-VL using the AITZ task (Zhang et al., [2024b](https://arxiv.org/html/2409.12191v2#bib.bib117)), which constructs a core clean test set derived from AITW (Rawles et al., [2024b](https://arxiv.org/html/2409.12191v2#bib.bib81)). Based on common operation patterns of phone, we define actions such as tap, input and swipe (Rawles et al., [2024b](https://arxiv.org/html/2409.12191v2#bib.bib81)) for Qwen2-VL to interact with on-screen icons for task completion. For example, when Qwen2-VL is tasked with finding a pizza restaurant nearby by Google Maps, it should input ”pizza” in the search term, swipe to select the appropriate restaurant, and tap the corresponding link. Following the AITZ setting, we report both type match (correctness of tap, input, or swipe) and exact match (correctness of tap location, input text, or swipe direction). With the support of grounding capability on UI, Qwen2-VL surpasses GPT-4 and previous SoTA (Zhang et al., [2024b](https://arxiv.org/html/2409.12191v2#bib.bib117); Zhan and Zhang, [2023](https://arxiv.org/html/2409.12191v2#bib.bib114)).

Robotic Control: we evaluate Qwen2-VL on the ALFRED task (Shridhar et al., [2020a](https://arxiv.org/html/2409.12191v2#bib.bib84)) in AI2THOR (Kolve et al., [2017](https://arxiv.org/html/2409.12191v2#bib.bib42)). The task requires agent to perform complex household tasks, such as toasting bread and slicing an apple to prepare a meal. To work in the virtual environment, we define high-level actions (GotoLocation, Pickup, PutDown, Open, Close, Clean, Heat, Cool, Slice) (Shridhar et al., [2020b](https://arxiv.org/html/2409.12191v2#bib.bib85)) as the action set. Moreover, agent needs to localize objects for manipulation (e.g., it can only pick up an apple if the apple is recognized). To improve the accuracy of manipulation, we integrate SAM (Kirillov et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib41)). ALFRED task reports task success rate (SR) (e.g., preparing dinner) and sub-goal completion metrics (GC) (e.g., whether the bread is toasted or the apple is sliced). Qwen2-VL slightly outperforms the previously specialized model ThinkBot (Lu et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib59)) on the valid-unseen set.

Card Games: we leverage the card game environment from RL4VLM (Zhai et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib113)) to assess Qwen2-VL’s performance in a series of card-based games: Number Line, BlackJack, EZPoint, and Point24. Each game presents distinct challenges: (1) reaching a target number using +1 or -1 operations, (2) drawing or holding cards to compete against the dealer, (3) applying basic arithmetic operations to reach a total of 12, and (4) using arithmetic operations to achieve a total of 24. We report the success rate of the tasks. They not only evaluate agent capabilities but also require strong OCR skills to recognize these cards and understand the progression of the game. Qwen2-VL demonstrates superior performance across all tasks.

Vision-Language Navigation: we evaluate Qwen2-VL on the Vision-and-Language Navigation (VLN) task using the R2R (Anderson et al., [2018](https://arxiv.org/html/2409.12191v2#bib.bib5)) and REVERIE (Qi et al., [2020](https://arxiv.org/html/2409.12191v2#bib.bib76)). In VLN, the model must autonomously determine the next location based on instruction, current observations. We report the success rate (SR) of VLM in reaching the predetermined destination for this task. The performance of Qwen2-VL is comparable to that of GPT-4o, but both models fall significantly behind current specialized VLN models (Chen et al., [2022](https://arxiv.org/html/2409.12191v2#bib.bib16); Sigurdsson et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib86)). We attribute this gap to the incomplete and unstructured map information generated by the model from multiple images. Accurately modeling maps and locations in a 3D environment remains a major challenge for multimodal models.

### 3.3 Ablation Study

In this section, we present ablation studies on image dynamic resolution, M-RoPE, and model scale. These experiments aim to provide insights into the impact of these key components on our model’s performance.

#### 3.3.1 Dynamic Resolution

Table 7: Qwen2-VL-7B under fixed/dynamic image tokens. Adjusting image sizes only results in small perturbations in performance, demonstrating the robustness to varying image sizes. Moreover, the dynamic resolution strategy achieves top-tier performance while consuming fewer tokens on average, demonstrating the efficiency of our model.

Strategy Average Image Tokens InfoVQA val RealWorldQA OCRBench MMMU
Fixed Image Tokens 64 28.85 56.47 572 53.33
576 65.72 65.88 828 52.78
1600 74.99 69.54 824 52.89
3136 77.27 70.59 786 53.44
Dynamic Image Tokens 1924 75.89 70.07 866 53.44

![Image 4: Refer to caption](https://arxiv.org/html/2409.12191v2/images/qwen2-vl/minpixels_resolution.png)

Figure 4: Qwen2-VL-7B with different min_pixels. Small images are upscaled to surpass a specified min_pixels threshold before input into the model. Increasing the image size within a reasonable range shows enhanced performance on perceptual tasks like InfoVQA, HallusionBench, and OCRBench.

As shown in Table [7](https://arxiv.org/html/2409.12191v2#S3.T7 "Table 7 ‣ 3.3.1 Dynamic Resolution ‣ 3.3 Ablation Study ‣ Experiments ‣ Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution"), we compare the performance between dynamic resolution and fixed resolution. For fixed resolution, we resize the images to ensure a constant number of image tokens being input to the model, rather than resizing to a specific height and width, as this would distort the original aspect ratio. For dynamic resolution, we only set min_pixels=100×28×28=100\times 28\times 28 and max_pixels=16384×28×28=16384\times 28\times 28, allowing the number of image tokens depend primarily on the image’s native resolution. It can be observed that adjusting image sizes only results in small perturbations in performance, demonstrating the model robustness to varying image sizes. Moreover, dynamic resolution approach is more efficient. We can observe that no single fixed resolution achieves optimal performance across all benchmarks. In contrast, the dynamic resolution approach consistently achieves top-tier performance while consuming fewer tokens on average.

Additionally, we observe that merely increasing the image size does not always lead to improved performance. It is more important to choose an appropriate resolution for different images. As detailed in Figure [4](https://arxiv.org/html/2409.12191v2#S3.F4 "Figure 4 ‣ 3.3.1 Dynamic Resolution ‣ 3.3 Ablation Study ‣ Experiments ‣ Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution"), we upscale small images to surpass a specified min_pixels threshold. Evaluations on upscaled images shows enhanced performance on perceptual tasks like InfoVQA, HallusionBench, and OCRBench. We attribute these gains to increased computational load. However, for OCRBench, a too-high min_pixels value leads to a severe performance decline. This is likely because OCRBench contains numerous extremely small images, and excessive enlargement causes these images to deviate from the training data distribution, turning them into out-of-distribution samples. In contrast, the effect of increasing min_pixels on the MMMU benchmark is negligible. We hypothesize that the performance bottleneck in MMMU is more related to the model’s reasoning capability rather than image resolution.

#### 3.3.2 M-RoPE

Table 8: Ablation studies of M-RoPE. Compared to 1D-RoPE, using M-RoPE achieves better performance in downstream tasks, particularly in video benchmarks. RWQ means RealworldQA.

Image Benchmarks Video Benchmarks
MathVista MMB MMStar RWQ DocVQA ChartQA InfoVQA TextVQA PerceptionTest NextQA STAR
1D-RoPE 39.2 58.6 36.7 54.5 82.5 68.0 50.8 71.3 46.6 43.9 55.5
M-RoPE 43.4 60.6 36.7 53.7 82.8 68.4 50.3 71.8 47.4 46.0 57.9

![Image 5: Refer to caption](https://arxiv.org/html/2409.12191v2/x1.png)

Figure 5: Evaluate the length extrapolation capability of Qwen2-VL-72B on Video-MME Medium Video. With the help of M-RoPE, the model demonstrated robust performance when the inference length exceeded the maximum training length of 16384 tokens.

In this subsection, we demonstrate the effectiveness of M-RoPE. First, we validate its capability on various downstream tasks. We employ Qwen2-1.5B and ViT-L as the backbone and report the results of the pre-trained models. As shown in Table [8](https://arxiv.org/html/2409.12191v2#S3.T8 "Table 8 ‣ 3.3.2 M-RoPE ‣ 3.3 Ablation Study ‣ Experiments ‣ Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution"), compared to 1D-RoPE, using M-RoPE achieves better performance in downstream tasks, particularly in video benchmarks. Furthermore, we assess the length extrapolation capability of M-RoPE on Video-MME medium-length videos. Figure [5](https://arxiv.org/html/2409.12191v2#S3.F5 "Figure 5 ‣ 3.3.2 M-RoPE ‣ 3.3 Ablation Study ‣ Experiments ‣ Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution") illustrates the performance of Qwen2-VL-72B at different inference lengths. Leveraging M-RoPE, the model demonstrates robust results across various inference lengths. Notably, despite limiting the maximum tokens per video to 16K during training, the model still exhibits exceptional performance at a maximum inference length of 80K tokens.

#### 3.3.3 Model Scaling

![Image 6: Refer to caption](https://arxiv.org/html/2409.12191v2/images/scale.jpg)

Figure 6: Model Performance Scaling Across Capabilities and Training Progress. As model size and the volume of training data increase, performance consistently improves across a range of capabilities and benchmarks.

We evaluate the performance of models of varying scales across multiple capability dimensions. Specifically, we categorize these dimensions into complex college-level problem-solving, mathematical abilities, document and table comprehension, general scenario question-answering, and video comprehension. The overall capability of a model is assessed by averaging its scores across different benchmarks associated with each dimension.

In particular, we use the MMMU (Yue et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib111)) benchmark to represent college-level problem-solving ability, while the average scores from MathVista (Lu et al., [2024a](https://arxiv.org/html/2409.12191v2#bib.bib61)) and MathVision (Wang et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib96)) serve as indicators of mathematical ability. For general scenario question-answering, we compute the average score across the RealWorldQA (X.AI, [2024a](https://arxiv.org/html/2409.12191v2#bib.bib100)), MMBench-V1.1 (Liu et al., [2023d](https://arxiv.org/html/2409.12191v2#bib.bib56)), MMT-Bench (Ying et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib109)), HallBench (Guan et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib32)), MMVet (Yu et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib110)), and MMStar (Chen et al., [2024a](https://arxiv.org/html/2409.12191v2#bib.bib15)) benchmarks. Document and table comprehension capability is reflected through the average score from benchmarks like DocVQA (Mathew et al., [2021](https://arxiv.org/html/2409.12191v2#bib.bib66)), InfoVQA (Mathew et al., [2021](https://arxiv.org/html/2409.12191v2#bib.bib66)), ChartQA (Masry et al., [2022](https://arxiv.org/html/2409.12191v2#bib.bib65)), TextVQA (Singh et al., [2019](https://arxiv.org/html/2409.12191v2#bib.bib87)), OCRBench (Liu et al., [2023e](https://arxiv.org/html/2409.12191v2#bib.bib57)), and MTVQA (Tang et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib92)). Lastly, video comprehension ability is measured by averaging scores across MVBench (Li et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib49)), PerceptionTest (Patraucean et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib75)), EgoSchema (Mangalam et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib63)), and Video-MME (Fu et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib31)).

As illustrated in Figure [6](https://arxiv.org/html/2409.12191v2#S3.F6 "Figure 6 ‣ 3.3.3 Model Scaling ‣ 3.3 Ablation Study ‣ Experiments ‣ Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution")(a), there is a consistent improvement in performance with increasing model size, particularly with respect to mathematical abilities, which show a positive correlation with the number of model parameters. On the other hand, for optical character recognition (OCR)-related tasks, even smaller-scale models exhibit relatively strong performance.

As shown in Figure [6](https://arxiv.org/html/2409.12191v2#S3.F6 "Figure 6 ‣ 3.3.3 Model Scaling ‣ 3.3 Ablation Study ‣ Experiments ‣ Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution")(b), we visualize the relationship between model performance and the number of training tokens during the second stage of pretraining for Qwen2-VL-7B. As the number of training tokens increases, the model performance improves; however, performance on vision question answering (VQA) tasks exhibits some fluctuation. In contrast, for tasks such as AI2D (Kembhavi et al., [2016](https://arxiv.org/html/2409.12191v2#bib.bib40)) and InfoVQA (Mathew et al., [2021](https://arxiv.org/html/2409.12191v2#bib.bib66))—both of which involve understanding textual and graphical information in images—the model performance shows steady improvement as training data is augmented.

Conclusion
----------

We have presented the Qwen2-VL series, the versatile large vision-language models, including three open-weight models with total parameter counts of 2, 8, and 72 billion. Qwen2-VL matches the performance of top-tier models like GPT-4o and Claude3.5-Sonnet in a range of multimodal scenarios, surpassing all other open-weight LVLM models. Qwen2-VL series introduces naive dynamic resolution and multimodal rotary position embedding (M-RoPE) to fuse information across modals effectively and be capable of understanding videos over 20 minutes in length. With advanced reasoning and decision-making abilities, Qwen2-VL can be integrated with devices such as mobile phones, robots, etc. Furthermore, Qwen2-VL now supports understanding multilingual texts within images, including most European languages, Japanese, Korean, Arabic, Vietnamese, and others.

We have made the Qwen2-VL model weights openly accessible, which enables researchers and developers to harness the full potential in a variety of applications and research projects. We aim to advance AI technologies and enhance their beneficial effects on society by dedicating ourselves to these endeavors.

Acknowledgements
----------------

We express our gratitude to Juan Zhu, Fan Hong, Jie Zhang, Yong Li of Alibaba Cloud’s PAI team (Alibaba-Cloud, [2024c](https://arxiv.org/html/2409.12191v2#bib.bib4)) for supporting the training infrastructure of Qwen2-VL. This work was also supported by Qwen LLM team (Yang et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib104)), and we especially thank Na Ni, Yichang Zhang, Jianxin Ma, Bowen Yu, Zheren Fu for their data contribution and insightful discussion.

References
----------

*   Alayrac et al. (2022) Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In _NeurIPS_, 2022. 
*   Alibaba-Cloud (2024a) Alibaba-Cloud. Cloud parallel file storage (cpfs), 2024a. URL [https://www.alibabacloud.com/en/product/cpfs](https://www.alibabacloud.com/en/product/cpfs). 
*   Alibaba-Cloud (2024b) Alibaba-Cloud. Object storage service (oss), 2024b. URL [https://www.alibabacloud.com/en/product/object-storage-service](https://www.alibabacloud.com/en/product/object-storage-service). 
*   Alibaba-Cloud (2024c) Alibaba-Cloud. Pai-lingjun intelligent computing service, 2024c. URL [https://www.alibabacloud.com/en/product/pai-lingjun](https://www.alibabacloud.com/en/product/pai-lingjun). 
*   Anderson et al. (2018) Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In _CVPR_, 2018. 
*   Ansel et al. (2024) Jason Ansel, Edward Z. Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky, Bin Bao, Peter Bell, David Berard, Evgeni Burovski, Geeta Chauhan, Anjali Chourdia, Will Constable, Alban Desmaison, Zachary DeVito, Elias Ellison, Will Feng, Jiong Gong, Michael Gschwind, Brian Hirsh, Sherlock Huang, Kshiteej Kalambarkar, Laurent Kirsch, Michael Lazos, Mario Lezcano, Yanbo Liang, Jason Liang, Yinghai Lu, C. K. Luk, Bert Maher, Yunjie Pan, Christian Puhrsch, Matthias Reso, Mark Saroufim, Marcos Yukio Siraichi, Helen Suk, Shunting Zhang, Michael Suo, Phil Tillet, Xu Zhao, Eikan Wang, Keren Zhou, Richard Zou, Xiaodong Wang, Ajit Mathews, William Wen, Gregory Chanan, Peng Wu, and Soumith Chintala. Pytorch 2: Faster machine learning through dynamic python bytecode transformation and graph compilation. In _ASPLOS_, 2024. 
*   Anthropic (2024) Anthropic. Claude 3.5 sonnet, 2024. URL [https://www.anthropic.com/news/claude-3-5-sonnet](https://www.anthropic.com/news/claude-3-5-sonnet). 
*   Arnab et al. (2021) Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić, and Cordelia Schmid. Vivit: A video vision transformer. In _ICCV_, 2021. 
*   Ba et al. (2016) Lei Jimmy Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. _arXiv:1607.06450_, 2016. 
*   Bai et al. (2023a) Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. _arXiv:2309.16609_, 2023a. 
*   Bai et al. (2023b) Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. _arXiv:2308.12966_, 2023b. 
*   Carreira and Zisserman (2017) Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In _CVPR_, 2017. 
*   Chen et al. (2023a) Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. _arXiv:2306.15195_, 2023a. 
*   Chen et al. (2023b) Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. _arXiv:2311.12793_, 2023b. 
*   Chen et al. (2024a) Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? _arXiv:2403.20330_, 2024a. 
*   Chen et al. (2022) Shizhe Chen, Pierre-Louis Guhur, Makarand Tapaswi, Cordelia Schmid, and Ivan Laptev. Think global, act local: Dual-scale graph transformer for vision-and-language navigation. In _CVPR_, 2022. 
*   Chen et al. (2016) Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. _arXiv:1604.06174_, 2016. 
*   Chen et al. (2023c) Zehui Chen, Weihua Du, Wenwei Zhang, Kuikun Liu, Jiangning Liu, Miao Zheng, Jingming Zhuo, Songyang Zhang, Dahua Lin, Kai Chen, et al. T-eval: Evaluating the tool utilization capability step by step. _arXiv:2312.14033_, 2023c. 
*   Chen et al. (2024b) Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. _arXiv:2404.16821_, 2024b. 
*   Chen et al. (2024c) Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. Internvl2: Better than the best—expanding performance boundaries of open-source multimodal models with the progressive scaling strategy, 2024c. URL [https://internvl.github.io/blog/2024-07-02-InternVL-2.0](https://internvl.github.io/blog/2024-07-02-InternVL-2.0). 
*   Chiang et al. (2023) Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, 2023. URL [https://lmsys.org/blog/2023-03-30-vicuna/](https://lmsys.org/blog/2023-03-30-vicuna/). 
*   Dai et al. (2023) Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. _arXiv:2305.06500_, 2023. 
*   Dao (2024) Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. In _ICLR_, 2024. 
*   Dao et al. (2022) Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. In _NeurIPS_, 2022. 
*   Dehghani et al. (2024) Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, et al. Patch n’pack: Navit, a vision transformer for any aspect ratio and resolution. In _NeurIPS_, 2024. 
*   Dosovitskiy et al. (2021) Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In _ICLR_, 2021. 
*   Dubey et al. (2024) Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. _arXiv:2407.21783_, 2024. 
*   Fang et al. (2023) Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks. _arXiv:2309.17425_, 2023. 
*   FFmpeg-Developers (2024) FFmpeg-Developers. ffmpeg tool, 2024. URL [http://ffmpeg.org/](http://ffmpeg.org/). 
*   Fu et al. (2023) Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. _arXiv:2306.13394_, 2023. 
*   Fu et al. (2024) Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. _arXiv:2405.21075_, 2024. 
*   Guan et al. (2023) Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination & visual illusion in large vision-language models. _arXiv:2310.14566_, 2023. 
*   Hong et al. (2023) Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. _arXiv:2312.08914_, 2023. 
*   Huang et al. (2023a) Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Qiang Liu, et al. Language is not all you need: Aligning perception with language models. _arXiv:2302.14045_, 2023a. 
*   Huang et al. (2023b) Siyuan Huang, Zhengkai Jiang, Hao Dong, Yu Qiao, Peng Gao, and Hongsheng Li. Instruct2act: Mapping multi-modality instructions to robotic actions with large language model. _arXiv:2305.11176_, 2023b. 
*   Huang et al. (2019) Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Xu Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, Yonghui Wu, and Zhifeng Chen. Gpipe: Efficient training of giant neural networks using pipeline parallelism. In _NeurIPS_, 2019. 
*   Inoue and Ohashi (2022) Yuki Inoue and Hiroki Ohashi. Prompter: Utilizing large language model prompting for a data efficient embodied instruction following. _arXiv:2211.03267_, 2022. 
*   Jiang et al. (2022) Yunfan Jiang, Agrim Gupta, Zichen Zhang, Guanzhi Wang, Yongqiang Dou, Yanjun Chen, Li Fei-Fei, Anima Anandkumar, Yuke Zhu, and Linxi Fan. Vima: General robot manipulation with multimodal prompts. _arXiv:2210.03094_, 2022. 
*   Kazemzadeh et al. (2014) Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In _EMNLP_, 2014. 
*   Kembhavi et al. (2016) Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In _ECCV_, 2016. 
*   Kirillov et al. (2023) Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In _ICCV_, 2023. 
*   Kolve et al. (2017) Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Matt Deitke, Kiana Ehsani, Daniel Gordon, Yuke Zhu, et al. Ai2-thor: An interactive 3d environment for visual ai. _arXiv:1712.05474_, 2017. 
*   Korthikanti et al. (2023) Vijay Anand Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing activation recomputation in large transformer models. In _MLSys_, 2023. 
*   Krizhevsky et al. (2012) Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. In _NeurIPS_, 2012. 
*   Lamy-Poirier (2023) Joel Lamy-Poirier. Breadth-first pipeline parallelism. In _MLSys_, 2023. 
*   Li et al. (2023a) Bo Li, Peiyuan Zhang, Jingkang Yang, Yuanhan Zhang, Fanyi Pu, and Ziwei Liu. Otterhd: A high-resolution multi-modality model. _arXiv:2311.04219_, 2023a. 
*   Li et al. (2023b) Chen Li, Yixiao Ge, Dian Li, and Ying Shan. Vision-language instruction tuning: A review and analysis. _arXiv:2311.08172_, 2023b. 
*   Li et al. (2023c) Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. _arXiv:2301.12597_, 2023c. 
*   Li et al. (2024) Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In _CVPR_, 2024. 
*   Li et al. (2020) Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar, Pieter Noordhuis, Teng Li, Adam Paszke, Jeff Smith, Brian Vaughan, Pritam Damania, et al. Pytorch distributed: Experiences on accelerating data parallel training. In _VLDB_, 2020. 
*   Li et al. (2023d) Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. _arXiv:2311.06607_, 2023d. 
*   Lin et al. (2023) Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, Jiaming Han, Siyuan Huang, Yichi Zhang, Xuming He, Hongsheng Li, and Yu Jiao Qiao. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. _arXiv:2311.07575_, 2023. 
*   Liu et al. (2023a) Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. _arXiv:2310.03744_, 2023a. 
*   Liu et al. (2023b) Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _arXiv:2304.08485_, 2023b. 
*   Liu et al. (2023c) Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chun yue Li, Jianwei Yang, Hang Su, Jun-Juan Zhu, and Lei Zhang. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. _arXiv:2303.05499_, 2023c. 
*   Liu et al. (2023d) Yuan Liu, Haodong Duan, Bo Li Yuanhan Zhang, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player? _arXiv:2307.06281_, 2023d. 
*   Liu et al. (2023e) Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xucheng Yin, Cheng lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: On the hidden mystery of ocr in large multimodal models. _arXiv:2305.07895_, 2023e. 
*   Loshchilov and Hutter (2019) Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _ICLR_, 2019. 
*   Lu et al. (2023) Guanxing Lu, Ziwei Wang, Changliu Liu, Jiwen Lu, and Yansong Tang. Thinkbot: Embodied instruction following with thought chain reasoning. _arXiv:2312.07062_, 2023. 
*   Lu et al. (2021) Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In _ACL_, 2021. 
*   Lu et al. (2024a) Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In _ICLR_, 2024a. 
*   Lu et al. (2024b) Quanfeng Lu, Wenqi Shao, Zitao Liu, Fanqing Meng, Boxuan Li, Botong Chen, Siyuan Huang, Kaipeng Zhang, Yu Qiao, and Ping Luo. Gui odyssey: A comprehensive dataset for cross-app gui navigation on mobile devices. _arXiv:2406.08451_, 2024b. 
*   Mangalam et al. (2023) Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. In _NeurIPS_, 2023. 
*   Mao et al. (2016) Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In _CVPR_, 2016. 
*   Masry et al. (2022) Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. _arXiv:2203.10244_, 2022. 
*   Mathew et al. (2021) Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In _WACV_, 2021. 
*   Narayanan et al. (2021) Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, and Matei Zaharia. Efficient large-scale language model training on GPU clusters using megatron-lm. In _SC_, 2021. 
*   Nvidia (2024a) Nvidia. Apex, 2024a. URL [https://github.com/NVIDIA/apex](https://github.com/NVIDIA/apex). 
*   Nvidia (2024b) Nvidia. Cuda, 2024b. URL [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit). 
*   OpenAI (2023) OpenAI. Gpt-4 technical report. _arXiv:2303.08774_, 2023. 
*   OpenAI. (2023) OpenAI. Gpt-4v(ision) system card, 2023. URL [https://openai.com/research/gpt-4v-system-card](https://openai.com/research/gpt-4v-system-card). 
*   Openai (2024) Openai. Chatml documents, 2024. URL [https://github.com/openai/openai-python/blob/main/chatml.md](https://github.com/openai/openai-python/blob/main/chatml.md). 
*   OpenAI (2024) OpenAI. Hello gpt-4o, 2024. URL [https://openai.com/index/hello-gpt-4o](https://openai.com/index/hello-gpt-4o). 
*   Paszke et al. (2019) Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Z. Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In _NeurIPS_, 2019. 
*   Patraucean et al. (2024) Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Perception test: A diagnostic benchmark for multimodal video models. In _NeurIPS_, 2024. 
*   Qi et al. (2020) Yuankai Qi, Qi Wu, Peter Anderson, Xin Wang, William Yang Wang, Chunhua Shen, and Anton van den Hengel. Reverie: Remote embodied visual referring expression in real indoor environments. In _CVPR_, 2020. 
*   Qwen Team (2024) Alibaba Group Qwen Team. Qwen-agent framework, 2024. URL [https://github.com/QwenLM/Qwen-Agent](https://github.com/QwenLM/Qwen-Agent). 
*   Radford et al. (2021) Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _ICML_, 2021. 
*   Rajbhandari et al. (2020) Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: memory optimizations toward training trillion parameter models. In _SC_, 2020. 
*   Rawles et al. (2024a) Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, et al. Androidworld: A dynamic benchmarking environment for autonomous agents. _arXiv:2405.14573_, 2024a. 
*   Rawles et al. (2024b) Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, and Timothy Lillicrap. Androidinthewild: A large-scale dataset for android device control. In _NeurIPS_, 2024b. 
*   Shah et al. (2024) Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. _arXiv:2407.08608_, 2024. 
*   Shoeybi et al. (2019) Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. _arXiv:1909.08053_, 2019. 
*   Shridhar et al. (2020a) Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. Alfred: A benchmark for interpreting grounded instructions for everyday tasks. In _CVPR_, 2020a. 
*   Shridhar et al. (2020b) Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning. _arXiv:2010.03768_, 2020b. 
*   Sigurdsson et al. (2023) Gunnar A Sigurdsson, Jesse Thomason, Gaurav S Sukhatme, and Robinson Piramuthu. Rrex-bot: Remote referring expressions with a bag of tricks. In _IROS_, 2023. 
*   Singh et al. (2019) Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In _CVPR_, 2019. 
*   Srinivasan et al. (2023) Venkat Krishna Srinivasan, Zhen Dong, Banghua Zhu, Brian Yu, Damon Mosk-Aoyama, Kurt Keutzer, Jiantao Jiao, and Jian Zhang. Nexusraven: a commercially-permissive language model for function calling. In _NeurIPS Workshop_, 2023. 
*   Su (2021) Jianlin Su. Transformer upgrade path: 4. rotary position encoding for two-dimensional positions, 2021. URL [https://www.spaces.ac.cn/archives/8397](https://www.spaces.ac.cn/archives/8397). 
*   Su (2024) Jianlin Su. Transformer upgrade path: 17. insights into multimodal positional encoding, 2024. URL [https://spaces.ac.cn/archives/10040](https://spaces.ac.cn/archives/10040). 
*   Su et al. (2024) Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. In _Neurocomputing_, 2024. 
*   Tang et al. (2024) Jingqun Tang, Qi Liu, Yongjie Ye, Jinghui Lu, Shu Wei, Chunhui Lin, Wanqing Li, Mohamad Fitri Faiz Bin Mahmood, Hao Feng, Zhen Zhao, Yanjie Wang, Yuliang Liu, Hao Liu, Xiang Bai, and Can Huang. Mtvqa: Benchmarking multilingual text-centric visual question answering. _arXiv:2405.11985_, 2024. 
*   Team et al. (2023) Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: A family of highly capable multimodal models. _arXiv:2312.11805_, 2023. 
*   Touvron et al. (2023a) Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _arXiv:2302.13971_, 2023a. 
*   Touvron et al. (2023b) Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv:2307.09288_, 2023b. 
*   Wang et al. (2024) Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. _arXiv:2402.14804_, 2024. 
*   Wang et al. (2022) Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In _ICML_, 2022. 
*   Wang et al. (2023a) Peng Wang, Shijie Wang, Junyang Lin, Shuai Bai, Xiaohuan Zhou, Jingren Zhou, Xinggang Wang, and Chang Zhou. One-peace: Exploring one general representation model toward unlimited modalities. _arXiv:2305.11172_, 2023a. 
*   Wang et al. (2023b) Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. _arXiv:2311.03079_, 2023b. 
*   X.AI (2024a) X.AI. Grok-1.5 vision preview. [https://x.ai/blog/grok-1.5v](https://x.ai/blog/grok-1.5v), 2024a. 
*   X.AI (2024b) X.AI. Grok-2 beta release. [https://x.ai/blog/grok-2](https://x.ai/blog/grok-2), 2024b. 
*   Yan et al. (2023) B. Yan, Yi Jiang, Jiannan Wu, D. Wang, Ping Luo, Zehuan Yuan, and Huchuan Lu. Universal instance perception as object discovery and retrieval. In _CVPR_, 2023. 
*   Yan et al. (2024) Fanjia Yan, Huanzhi Mao, Charlie Cheng-Jie Ji, Tianjun Zhang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. Berkeley function calling leaderboard, 2024. URL [https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html). 
*   Yang et al. (2024) An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. _arXiv:2407.10671_, 2024. 
*   Yang et al. (2023) Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v (ision). _arXiv:2309.17421_, 2023. 
*   Yao et al. (2024) Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. _arXiv:2408.01800_, 2024. 
*   Ye et al. (2023a) Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. _arXiv:2304.14178_, 2023a. 
*   Ye et al. (2023b) Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Haowei Liu, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. _arXiv:2311.04257_, 2023b. 
*   Ying et al. (2024) Kaining Ying, Fanqing Meng, Jin Wang, Zhiqian Li, Han Lin, Yue Yang, Hao Zhang, Wenbo Zhang, Yuqi Lin, Shuo Liu, Jiayi Lei, Quanfeng Lu, Runjian Chen, Peng Xu, Renrui Zhang, Haozhe Zhang, Peng Gao, Yali Wang, Yu Qiao, Ping Luo, Kaipeng Zhang, and Wenqi Shao. Mmt-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask agi. _arXiv:2404.16006_, 2024. 
*   Yu et al. (2024) Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. In _ICML_, 2024. 
*   Yue et al. (2023) Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. _arXiv:2311.16502_, 2023. 
*   Yue et al. (2024) Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Ming Yin, Botao Yu, Ge Zhang, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. _arXiv preprint arXiv:2409.02813_, 2024. 
*   Zhai et al. (2024) Yuexiang Zhai, Hao Bai, Zipeng Lin, Jiayi Pan, Shengbang Tong, Yifei Zhou, Alane Suhr, Saining Xie, Yann LeCun, Yi Ma, et al. Fine-tuning large vision-language models as decision-making agents via reinforcement learning. _arXiv:2405.10292_, 2024. 
*   Zhan and Zhang (2023) Zhuosheng Zhan and Aston Zhang. You only look at screens: Multimodal chain-of-action agents. _arXiv:2309.11436_, 2023. 
*   Zhang and Sennrich (2019) Biao Zhang and Rico Sennrich. Root mean square layer normalization. In _NeurIPS_, 2019. 
*   Zhang et al. (2024a) Haotian Zhang, Haoxuan You, Philipp Dufter, Bowen Zhang, Chen Chen, Hong-You Chen, Tsu-Jui Fu, William Yang Wang, Shih-Fu Chang, Zhe Gan, and Yinfei Yang. Ferret-v2: An improved baseline for referring and grounding with large language models. _arXiv:2404.07973_, 2024a. 
*   Zhang et al. (2024b) Jiwen Zhang, Jihao Wu, Yihua Teng, Minghui Liao, Nuo Xu, Xiao Xiao, Zhongyu Wei, and Duyu Tang. Android in the zoo: Chain-of-action-thought for gui agents. _arXiv:2403.02713_, 2024b. 
*   Zhang et al. (2023) Pan Zhang, Xiaoyi Dong Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Shuangrui Ding, Songyang Zhang, Haodong Duan, Hang Yan, et al. Internlm-xcomposer: A vision-language large model for advanced text-image comprehension and composition. _arXiv:2309.15112_, 2023. 
*   Zhang et al. (2024c) Tianyu Zhang, Suyuchen Wang, Lu Li, Ge Zhang, Perouz Taslakian, Sai Rajeswar, Jie Fu, Bang Liu, and Yoshua Bengio. Vcr: Visual caption restoration. _arXiv:2406.06462_, 2024c. 
*   Zhu et al. (2023) Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv:2304.10592_, 2023. 

Appendix A Model Capabilities and Qualitative Examples
------------------------------------------------------

In this section, we present some practical examples of our Qwen2-VL.

### A.1 General Chat and OCR

The Qwen2-VL models are now more adept at accurately describing and identifying complex information within images, as well as providing detailed background and answering related questions. Besides, the text processing capabilities of the Qwen2-VL models have seen significant improvements, particularly concerning the recognition of Chinese and English text within images.

Figure 7: When presented with an image of cubes of different colors, the models identify their layout and the color of each cube.

Figure 8: The model displays an adeptness in recognizing flowers in photographs.

Figure 9: Literary writing in multiple languages based on visual stimuli.

Figure 10: The model displays an adeptness in recognizing multilingual texts in images.

Figure 11: Recognition of dense Chinese text. The model is capable of directly converting lengthy, closely packed Chinese characters into standard English with accuracy and fluency.

Figure 12: The model displays an adeptness in recognizing multilingual texts in images. Image source: (Yang et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib105))

Figure 13: The model displays an adeptness in recognizing multilingual texts in images.

### A.2 Information extraction and Visual Reasoning

A notable advancement in the Qwen2-VL models is their enhanced visual reasoning capability. This advancement allows the models to interpret and comprehend complex representations such as flowcharts, diagrams, and other symbolic systems.

Figure 14: The model displays an adeptness in solving mathematical problems. Image source: Lu et al. ([2021](https://arxiv.org/html/2409.12191v2#bib.bib60))

Figure 15: Solving a math problem. The model calculates the surface area and volume of these figures with step-by-step explanation 

Figure 16: The model displays an adeptness in solving algorithmic problems.

Figure 17: The model displays an adeptness in recognize content from the web page.

Figure 18: The model displays an adeptness in OCR and mathematical reasoning.

Figure 19: The model displays an adeptness in OCR and following formats.

Figure 20: The model displays an adeptness in OCR and following formats.

Figure 21: The model displays an adeptness in large images OCR.

### A.3 Video Understanding

Figure 22: The model displays an adeptness in recognizing multi-round video chat.

Figure 23: The model displays an adeptness in recognizing multi-video understanding.

### A.4 Visual Agent Capability

The Qwen2-VL also excels in location and agent tasks.

Figure 24: Our models were able to locate specific elements within images, such as identifying the red car accurately. 

Figure 25: Our model is capable of annotating and posing questions regarding the content present in webpage screenshots, demonstrating its potential as a visual agent. Image source: (Yang et al., [2023](https://arxiv.org/html/2409.12191v2#bib.bib105))

Figure 26: The model identified the destination and arrival time in the image, called the 24-hour weather query function, correctly input the destination, and extracted the weather at the arrival time from the query results, thus enabling it to answer the user’s question.

Figure 27: The model identified the steps in the flowchart, drafted the simulation code as required while omitting the details, and then successfully executed it through the code interpreter. Image source: (Dubey et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib27))

Figure 28: The model analyzed the table, and wrote the code to calculate the mean of each row in the table and to plot the results as a bar chart, which was successfully executed and visualized by the code interpreter. Image source: (Dubey et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib27))

Figure 29: The model understood the formula, implemented the code as required, and successfully executed it in the code interpreter to obtain the results. Image source: (Dubey et al., [2024](https://arxiv.org/html/2409.12191v2#bib.bib27))

Figure 30: Qwen2-VL as an agent understands the query with respect to UI operation, utilizes the pre-defined actions in system message, and fulfill the task step-by-step.

Figure 31: Qwen2-VL recognizes these cards and utilizes Hit and Stand to play the blackjack.

