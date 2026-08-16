Title: Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models

URL Source: https://arxiv.org/html/2507.08128

Published Time: Wed, 30 Jul 2025 00:10:42 GMT

Markdown Content:
Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models
===============

1.   [1 Introduction](https://arxiv.org/html/2507.08128v2#S1 "In Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
2.   [2 Related Work](https://arxiv.org/html/2507.08128v2#S2 "In Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
3.   [3 Methodology](https://arxiv.org/html/2507.08128v2#S3 "In Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
    1.   [3.1 Audio Flamingo 3 Architecture](https://arxiv.org/html/2507.08128v2#S3.SS1 "In 3 Methodology ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")

4.   [4 Audio Flamingo 3 Training Data](https://arxiv.org/html/2507.08128v2#S4 "In Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
    1.   [4.1 AudioSkills-XL: Expanding AudioSkills with Reasoning-Focused QAs](https://arxiv.org/html/2507.08128v2#S4.SS1 "In 4 Audio Flamingo 3 Training Data ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
    2.   [4.2 LongAudio-XL: Expanding LongAudio with Long Speech QA](https://arxiv.org/html/2507.08128v2#S4.SS2 "In 4 Audio Flamingo 3 Training Data ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
    3.   [4.3 AF-Think: Towards flexible, on-demand reasoning](https://arxiv.org/html/2507.08128v2#S4.SS3 "In 4 Audio Flamingo 3 Training Data ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
    4.   [4.4 AF-Chat: Multi-turn Multi-audio Chat Data](https://arxiv.org/html/2507.08128v2#S4.SS4 "In 4 Audio Flamingo 3 Training Data ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")

5.   [5 Audio Flamingo 3 Training Strategy](https://arxiv.org/html/2507.08128v2#S5 "In Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
6.   [6 Experiments](https://arxiv.org/html/2507.08128v2#S6 "In Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
    1.   [6.1 Audio Understanding and Reasoning Evaluation](https://arxiv.org/html/2507.08128v2#S6.SS1 "In 6 Experiments ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
    2.   [6.2 Chat and TTS Evaluation](https://arxiv.org/html/2507.08128v2#S6.SS2 "In 6 Experiments ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
    3.   [6.3 Ablation Studies](https://arxiv.org/html/2507.08128v2#S6.SS3 "In 6 Experiments ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")

7.   [7 Conclusion, Limitations and Future Work](https://arxiv.org/html/2507.08128v2#S7 "In Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
8.   [A AF-Whisper](https://arxiv.org/html/2507.08128v2#A1 "In Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
    1.   [A.1 Training Details](https://arxiv.org/html/2507.08128v2#A1.SS1 "In Appendix A AF-Whisper ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
    2.   [A.2 Training Datasets](https://arxiv.org/html/2507.08128v2#A1.SS2 "In Appendix A AF-Whisper ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")

9.   [B AudioSkills-XL](https://arxiv.org/html/2507.08128v2#A2 "In Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
    1.   [B.1 Skill-Wise Breakdown](https://arxiv.org/html/2507.08128v2#A2.SS1 "In Appendix B AudioSkills-XL ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
        1.   [B.1.1 Music Reasoning](https://arxiv.org/html/2507.08128v2#A2.SS1.SSS1 "In B.1 Skill-Wise Breakdown ‣ Appendix B AudioSkills-XL ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
        2.   [B.1.2 Music Knowledge](https://arxiv.org/html/2507.08128v2#A2.SS1.SSS2 "In B.1 Skill-Wise Breakdown ‣ Appendix B AudioSkills-XL ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
        3.   [B.1.3 Sound Reasoning](https://arxiv.org/html/2507.08128v2#A2.SS1.SSS3 "In B.1 Skill-Wise Breakdown ‣ Appendix B AudioSkills-XL ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")

10.   [C LongAudio-XL](https://arxiv.org/html/2507.08128v2#A3 "In Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
11.   [D AF-Think](https://arxiv.org/html/2507.08128v2#A4 "In Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
12.   [E AF-Chat](https://arxiv.org/html/2507.08128v2#A5 "In Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
    1.   [E.1 Human Study for AF-Chat-test](https://arxiv.org/html/2507.08128v2#A5.SS1 "In Appendix E AF-Chat ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
    2.   [E.2 Clustering for constructing AF-Chat](https://arxiv.org/html/2507.08128v2#A5.SS2 "In Appendix E AF-Chat ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")

13.   [F Prompts](https://arxiv.org/html/2507.08128v2#A6 "In Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
14.   [G AF3 Training Datasets](https://arxiv.org/html/2507.08128v2#A7 "In Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
15.   [H AF3 Training Details](https://arxiv.org/html/2507.08128v2#A8 "In Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
16.   [I Streaming TTS System Architecture and Training Details](https://arxiv.org/html/2507.08128v2#A9 "In Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
    1.   [I.1 Neural Audio Codec](https://arxiv.org/html/2507.08128v2#A9.SS1 "In Appendix I Streaming TTS System Architecture and Training Details ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
    2.   [I.2 Text-to-Speech (TTS) Module](https://arxiv.org/html/2507.08128v2#A9.SS2 "In Appendix I Streaming TTS System Architecture and Training Details ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")
    3.   [I.3 Training Data and Processing](https://arxiv.org/html/2507.08128v2#A9.SS3 "In Appendix I Streaming TTS System Architecture and Training Details ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")

17.   [J Qualitative Examples](https://arxiv.org/html/2507.08128v2#A10 "In Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")

![Image 1: [Uncaptioned image]](https://arxiv.org/html/2507.08128)Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models
============================================================================================================================================================

Arushi Goel★1, Sreyan Ghosh★12, Jaehyeon Kim 1, Sonal Kumar 2, Zhifeng Kong 1, Sang-gil Lee 1, 

Chao-Han Huck Yang 1, Ramani Duraiswami 2, Dinesh Manocha 2, Rafael Valle 1, Bryan Catanzaro 1

NVIDIA, USA 1, University of Maryland, College Park, USA 2

★Equal contribution. Alphabetically ordered. 

Correspondence: arushig@nvidia.com, sreyang@umd.edu 

Project:[https://research.nvidia.com/labs/adlr/AF3/](https://research.nvidia.com/labs/adlr/AF3/)

###### Abstract

We present Audio Flamingo 3 (AF3), a fully open state-of-the-art (SOTA) large audio-language model that advances reasoning and understanding across speech, sound, and music. AF3 introduces: (i) AF-Whisper, a unified audio encoder trained using a novel strategy for joint representation learning across all 3 modalities of speech, sound, and music; (ii) flexible, on-demand thinking, allowing the model to do chain-of-thought-type reasoning before answering; (iii) multi-turn, multi-audio chat; (iv) long audio understanding and reasoning (including speech) up to 10 minutes; and (v) voice-to-voice interaction. To enable these capabilities, we propose several large-scale training datasets curated using novel strategies, including AudioSkills-XL, LongAudio-XL, AF-Think, and AF-Chat, and train AF3 with a novel five-stage curriculum-based training strategy. Trained on only open-source audio data, AF3 achieves new SOTA results on over 20+ (long) audio understanding and reasoning benchmarks, surpassing both open-weight and closed-source models trained on much larger datasets.

1 Introduction
--------------

![Image 2: Refer to caption](https://arxiv.org/html/x2.png)

Figure 1: AF3 vs. prior SOTA LALMs (values normalized and WER=100-WER). AF3 outperforms most open-source/weights (e.g., Qwen2.5-Omni) and closed (e.g., Gemini 2.5 Pro) LALMs while being fully open.

Audio—including speech, sounds, and music—is central to human perception and interaction. It enables us to understand our surroundings, engage in conversations, express emotions, interpret videos, and enjoy music. For AI systems to approach artificial general intelligence (AGI)([morris2024position,](https://arxiv.org/html/2507.08128v2#bib.bib88)), they must similarly develop the ability to comprehend and reason over diverse audio signals. While Large Language Models (LLMs) excel at language-based reasoning, their audio comprehension remains limited — both in accessibility and capability[hurst2024gpt](https://arxiv.org/html/2507.08128v2#bib.bib54); [touvron2023llama](https://arxiv.org/html/2507.08128v2#bib.bib106). Extending LLMs to process and reason over audio is essential for building truly context-aware, intelligent agents.

Audio-Language Models (ALMs) extend the capabilities of LMs to the auditory domain. Early works such as CLAP[elizalde2022clap](https://arxiv.org/html/2507.08128v2#bib.bib33) align audio and text in a shared embedding space, enabling them with tasks like retrieval([oncescu2021audio,](https://arxiv.org/html/2507.08128v2#bib.bib89)). More recently, the emergence of Large ALMs (LALMs)—decoder-only language models augmented with audio understanding[chu2023qwenaudio](https://arxiv.org/html/2507.08128v2#bib.bib20); [chu2024qwenaudio2](https://arxiv.org/html/2507.08128v2#bib.bib19); [team2023gemini](https://arxiv.org/html/2507.08128v2#bib.bib105)—has unlocked powerful capabilities, including open-ended audio question-answering (AQA) that demands both reasoning and world knowledge([sakshi2024mmau,](https://arxiv.org/html/2507.08128v2#bib.bib101)). These capabilities have further enabled tasks like audio analysis[drossos2020clotho](https://arxiv.org/html/2507.08128v2#bib.bib32); [kim2019audiocaps](https://arxiv.org/html/2507.08128v2#bib.bib60), conversational assistants[daniel2018toward](https://arxiv.org/html/2507.08128v2#bib.bib24), etc.

| Models | Audio Understanding | Voice | Multi-turn Chat | Long Audio (>>>30 secs) | Open-Source |
| --- |
|  | Sound | Music | Speech | In | Out* | Single A | Multiple A | Speech | Sound | Music | Model | Data | Code |
| LTU | ✓ | ✓ | × | × | × | × | × | × | × | × | ✓ | ✓ | ✓ |
| LTU-AS | ✓ | ✓ | ✓ | × | × | × | × | × | × | × | ✓ | ✓ | ✓ |
| GAMA | ✓ | ✓ | × | × | × | × | × | × | × | × | ✓ | ✓ | ✓ |
| SALMONN | ✓ | ✓ | ✓ | × | × | × | × | × | × | × | ✓ | ✓ | ✓ |
| MuLLaMa | × | ✓ | × | × | × | × | × | × | × | × | ✓ | ✓ | ✓ |
| Phi-4-mm | ✓ | ✓ | ✓ | × | × | × | × | ✓ | ✓ | ✓ | ✓ | × | × |
| Qwen-Audio | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | × | ✓ | × | × |
| Qwen2-Audio | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | × | ✓ | × | × |
| Qwen2.5-Omni | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | × | × |
| GPT-4o Audio | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × |
| Gemini 2.0 / 2.5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | × | × | × |
| Audio Flamingo | ✓ | ✓ | × | × | × | ✓ | × | × | × | × | ✓ | ✓ | ✓ |
| Audio Flamingo 2 | ✓ | ✓ | × | × | × | × | × | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| Audio Flamingo 3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Table 1: Comparison of various LALMs in terms of capabilities and openness. AF3 stands out as the most capable and open model to date, achieving SOTA results across benchmarks (A in Chat stands for Audio). *Voice-out is powered by our novel streaming TTS implementation, which is also applicable to other LALMs.

However, existing models still fall short in key areas critical to AGI, such as expert-level reasoning[morris2024position](https://arxiv.org/html/2507.08128v2#bib.bib88); [sakshi2024mmau](https://arxiv.org/html/2507.08128v2#bib.bib101), multi-turn and multi-audio dialogue[goel2024audio](https://arxiv.org/html/2507.08128v2#bib.bib44), and long audio understanding[kong2025audioflamingo2](https://arxiv.org/html/2507.08128v2#bib.bib40). We identify two core limitations: (i) most LALMs are trained primarily on short audio for recognition tasks rather than ones that require deliberate reasoning; and (ii) in turn, they lack exposure to the skill sets required for complex tasks. Additionally, most LALMs that support all three modalities of speech, sound, and music are closed-source: while some have publicly released models weights[chu2023qwenaudio](https://arxiv.org/html/2507.08128v2#bib.bib20); [chu2024qwenaudio2](https://arxiv.org/html/2507.08128v2#bib.bib19); [abouelenin2025phi](https://arxiv.org/html/2507.08128v2#bib.bib1), they offer limited to no information about their data, code, or recipes (more details in [Table˜1](https://arxiv.org/html/2507.08128v2#S1.T1 "In 1 Introduction ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")).

Main Contributions. To address these issues, we introduce Audio Flamingo 3 (AF3), a fully open-source 1 1 1 By fully open, we mean that the model’s weights, training data, and code will be publicly released, with full transparency about the training methodology. Due to the licensing and scope of the training data used in the work, all releases will be under a research-only license. LALM with state-of-the-art performance in audio understanding and reasoning across 20+ benchmarks. In addition, AF3 brings several novel capabilities, including multi-turn, multi-audio chat, on-demand thinking, voice-to-voice interaction, and long-context audio reasoning (up to 10 minutes). We propose three core innovations to enable these capabilities: (i) Data: We focus on curating high-quality data at scale and propose (a) AudioSkills-XL: a large-scale dataset of 8M diverse AQA pairs, (b) LongAudio-XL: large-scale dataset of 1.25M diverse audio QA pairs for long audio reasoning; (c) AF-Chat: a multi-turn multi-audio chat dataset curated using a novel algorithm with 75k instances and (d) AF-Think: a dataset with 250k+ AQA pairs with short length prefixes to encourage CoT-type reasoning before arriving at the answer (ii) AF-Whisper: We train AF-Whisper, a unified audio encoder pretrained using a novel strategy on large-scale audio-caption pairs, capable of learning general-purpose representations across speech, sounds, and music; and (iii) Learning Curriculum: We train AF3 with a five-stage curriculum-based training strategy that progressively increases context length and task complexity. In summary, our main contributions are:

*   •We introduce Audio Flamingo 3 (AF3), the most open and capable foundational LALM to date. AF3 introduces key capabilities including: (i) long-context audio QA (extending beyond sounds as in([kong2025audioflamingo2,](https://arxiv.org/html/2507.08128v2#bib.bib40)) and including speech), and (ii) flexible, on-demand thinking, enabling the model to generate concise, CoT-style reasoning steps when prompted. AF3 achieves state-of-the-art performance on 20+ audio understanding and reasoning benchmarks. 
*   •We also present AF3-Chat, a fine-tuned variant of AF3 designed for multi-turn, multi-audio chat and voice-to-voice interaction. 
*   •We propose novelties in data curation, audio encoder representation learning, and training strategies. Being fully open, we release our code, training recipes, and 4 new datasets to promote research in this space. 

2 Related Work
--------------

Audio Language Models. The rapid progress of LLMs has catalyzed the development of multimodal LLMs (MLLMs) capable of understanding and reasoning across diverse data modalities, including audio. Within this space, ALMs specifically target reasoning over auditory inputs such as speech, sounds, and music. ALMs typically follow two main architectural paradigms: (i) Encoder-only ALMs, which learn a joint embedding space for audio and text, enabling tasks like cross-modal retrieval. Representative models include CLAP([elizalde2022clap,](https://arxiv.org/html/2507.08128v2#bib.bib33)), Wav2CLIP([wu2021wav2clip,](https://arxiv.org/html/2507.08128v2#bib.bib112)), and AudioCLIP([guzhov2021audioclip,](https://arxiv.org/html/2507.08128v2#bib.bib48)). (ii) Encoder-decoder ALMs, also referred to as LALMs, which use decoder-only LLMs augmented with an audio encoder. Notable examples include LTU([gong2023ltu,](https://arxiv.org/html/2507.08128v2#bib.bib46)), LTU-AS([gong2023ltu-as,](https://arxiv.org/html/2507.08128v2#bib.bib45)), SALMONN([tang2023salmonn,](https://arxiv.org/html/2507.08128v2#bib.bib104)), Pengi([deshmukh2023pengi,](https://arxiv.org/html/2507.08128v2#bib.bib27)), Audio Flamingo([kong2024audioflamingo,](https://arxiv.org/html/2507.08128v2#bib.bib65)), Audio Flamingo 2([kong2025audioflamingo2,](https://arxiv.org/html/2507.08128v2#bib.bib40)), AudioGPT([huang2023audiogpt,](https://arxiv.org/html/2507.08128v2#bib.bib53)), GAMA([ghosh2024gama,](https://arxiv.org/html/2507.08128v2#bib.bib41)), Qwen-Audio([chu2023qwenaudio,](https://arxiv.org/html/2507.08128v2#bib.bib20)), and Qwen2-Audio([chu2024qwenaudio2,](https://arxiv.org/html/2507.08128v2#bib.bib19)). These LALMs have significantly improved performance on core audio understanding tasks such as automatic speech recognition (ASR)([radford2022whisper,](https://arxiv.org/html/2507.08128v2#bib.bib96)), audio captioning([kim2019audiocaps,](https://arxiv.org/html/2507.08128v2#bib.bib60)), and acoustic scene classification([chen2022beats,](https://arxiv.org/html/2507.08128v2#bib.bib16)). More importantly, they have enabled new capabilities such as open-ended audio question answering, which requires complex reasoning and external world knowledge.

Despite these advancements, current LALMs fall short in supporting various capabilities, including multi-turn, multi-audio chat, long-context audio comprehension, etc. Moreover, most LALMs are limited to specific audio types, lacking the ability to unify understanding across speech, sounds, and music. Finally, the most advanced LALMs remain only partially open, releasing model checkpoints without accompanying training code or data. This lack of transparency limits reproducibility and impedes scientific progress by obscuring the development process.

Reasoning and Long-Context Understanding. Recent progress in LLMs has increasingly emphasized long-context understanding. In the vision-language space, substantial strides have been made in modeling long videos[chen2024longvila](https://arxiv.org/html/2507.08128v2#bib.bib17). In the audio domain, AF2 marked the first step toward long-context audio comprehension, though it is limited to sounds and music.

Parallel efforts have aimed to enhance reasoning in LLMs and MLLMs through improved reasoning datasets[sakshi2024mmau](https://arxiv.org/html/2507.08128v2#bib.bib101); [weck2024muchomusic](https://arxiv.org/html/2507.08128v2#bib.bib110), advancements in multimodal perception[xu2024llava](https://arxiv.org/html/2507.08128v2#bib.bib116); [team2023gemini](https://arxiv.org/html/2507.08128v2#bib.bib105), and emerging paradigms like chain-of-thought (CoT) prompting([ma2025audio,](https://arxiv.org/html/2507.08128v2#bib.bib80)), which encourages models to "think before answering." In developing AF3, we combine these advances—integrating controlled reasoning supervision, long-context training, and modality diversity—to equip the model with strong reasoning capabilities and long-context comprehension, including speech.

3 Methodology
-------------

### 3.1 Audio Flamingo 3 Architecture

In this section, we discuss our proposed architecture for Audio Flamingo 3 as shown in [Figure˜2](https://arxiv.org/html/2507.08128v2#S3.F2 "In 3.1 Audio Flamingo 3 Architecture ‣ 3 Methodology ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"). AF3 consists of i) AF-Whisper: an audio encoder with sliding window feature extraction, ii) audio projector, iii) an LLM, and iv) a streaming TTS. We provide details of each component below.

AF-Whisper Audio Encoder. Prior work in audio representation learning typically treats speech, sounds, and music as separate modalities, and LALMs often rely on distinct encoders for each[tang2023salmonn](https://arxiv.org/html/2507.08128v2#bib.bib104); [ghosh2024gama](https://arxiv.org/html/2507.08128v2#bib.bib41). Using separate encoders for LALMs increases model complexity, introduces frame-rate mismatches, and can lead to training instability. To address this, we propose AF-Whisper, a unified audio encoder trained with a simple yet effective representation learning strategy to model all three audio types.

As illustrated in [Figure˜2](https://arxiv.org/html/2507.08128v2#S3.F2 "In 3.1 Audio Flamingo 3 Architecture ‣ 3 Methodology ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), we start with the pre-trained Whisper large-v3 encoder[radford2022whisper](https://arxiv.org/html/2507.08128v2#bib.bib96), attach it to a standard Transformer decoder, and train using the audio captioning task with the next-token-prediction objective. To achieve this, we generate a natural language caption for each audio, describing its speech, sound, and music content. First, we pool several datasets and then prompt GPT-4.1 to generate the audio caption. For prompting, we use available metadata for each sample, which includes transcripts, ambient sound descriptions, and music attributes. For samples lacking any of the 3 metadata, we synthesize it using AF2([kong2025audioflamingo2,](https://arxiv.org/html/2507.08128v2#bib.bib40)) or Whisper-Large-v3 ASR([radford2022whisper,](https://arxiv.org/html/2507.08128v2#bib.bib96)). All datasets used for training are detailed in Section[A.2](https://arxiv.org/html/2507.08128v2#A1.SS2 "A.2 Training Datasets ‣ Appendix A AF-Whisper ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"). We choose Whisper as the backbone due to its existing speech understanding capabilities and its dense, high-resolution audio features, which are more informative than those from models like CLAP([elizalde2022clap,](https://arxiv.org/html/2507.08128v2#bib.bib33)). We connect it with a Transformer decoder using cross-attention (similar to RECAP[liu2023recap](https://arxiv.org/html/2507.08128v2#bib.bib77) and AF2([kong2025audioflamingo2,](https://arxiv.org/html/2507.08128v2#bib.bib40))) with 24 layers, 8 attention heads, and 1024 hidden size.

Feature Extraction. Given an audio input A A italic_A, we first resample it to 16kHz mono. The raw waveform is then transformed into a 128-channel mel-spectrogram using a window size of 25ms and a hop size of 10ms. This mel-spectrogram is processed by AF-Whisper, producing hidden representations, denoted as h a=f a​(A)h_{a}=f_{a}(A)italic_h start_POSTSUBSCRIPT italic_a end_POSTSUBSCRIPT = italic_f start_POSTSUBSCRIPT italic_a end_POSTSUBSCRIPT ( italic_A ), where h a∈ℝ N×d h_{a}\in\mathbb{R}^{N\times d}italic_h start_POSTSUBSCRIPT italic_a end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_N × italic_d end_POSTSUPERSCRIPT. As shown in [Figure˜2](https://arxiv.org/html/2507.08128v2#S3.F2 "In 3.1 Audio Flamingo 3 Architecture ‣ 3 Methodology ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), each audio is processed in 30-second chunks of non-overlapping sliding windows, and N N italic_N or the temporal resolution depends on the length of the audio and the maximum number of sliding windows (which varies according to the stage of training). AF-Whisper produces audio features at a frame rate of 50Hz, and we further apply a pooling layer with a stride of two similar to [chu2024qwenaudio2](https://arxiv.org/html/2507.08128v2#bib.bib19). d d italic_d denotes the hidden dimension, which is 1280.

![Image 3: Refer to caption](https://arxiv.org/html/x3.png)

Figure 2: Overview of Audio Flamingo 3, AF-Whisper training, and five-stage curriculum training.

Audio Adaptor. To align the audio modality with the text embedding space of the LLM, we introduce audio adaptor layers, denoted by 𝒜(.)\mathcal{A}(.)caligraphic_A ( . ). Specifically, the encoded hidden representations h a h_{a}italic_h start_POSTSUBSCRIPT italic_a end_POSTSUBSCRIPT from AF-Whisper are passed through these adaptor layers to produce embeddings: a=𝒜​(h a)a=\mathcal{A}(h_{a})italic_a = caligraphic_A ( italic_h start_POSTSUBSCRIPT italic_a end_POSTSUBSCRIPT ). These resulting embeddings serve as prompts to the LLM, alongside the textual instruction.

Large Language Model (LLM). We employ Qwen-2.5-7B([yang2024qwen2,](https://arxiv.org/html/2507.08128v2#bib.bib118)) as our backbone, a decoder-only causal LLM with 7B parameters, 36 hidden layers, and 16 attention heads.

Streaming TTS. To enable voice-to-voice interaction, we employ a TTS module for streaming speech generation, supporting streaming inputs and outputs. Our TTS module employs a decoder-only transformer architecture: it predicts the subsequent audio token conditioned on incoming subword text tokens from the LLM and the history of previously generated audio tokens. Similar streaming TTS techniques have been explored with LLMs([xie2024mini,](https://arxiv.org/html/2507.08128v2#bib.bib115)) (for voice-out on LLM outputs), but not in the context of LALMs (which we define as models designed to perceive and reason over diverse audio inputs). Since not a core novelty of our work, we provide more details, including training and architecture, in Appendix[I](https://arxiv.org/html/2507.08128v2#A9 "Appendix I Streaming TTS System Architecture and Training Details ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models").

4 Audio Flamingo 3 Training Data
--------------------------------

We present detailed statistics for all datasets used to train AF3 in Table[11](https://arxiv.org/html/2507.08128v2#A7.T11 "Table 11 ‣ Appendix G AF3 Training Datasets ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"). AF3 has a total of 5 stages of training, where each stage employs a unique combination of datasets with unique weights (number of passes over that dataset for that particular stage). For Stages 1 and 2, we use open-source, recognition-focused foundational datasets converted to QA format. In the following sub-sections, we introduce our four novel skill-focused and unique datasets, each accompanied by custom data curation strategies, used in Stages 3, 3.5, and 4, which form a core contribution of this work.

### 4.1 AudioSkills-XL: Expanding AudioSkills with Reasoning-Focused QAs

Audio QA pairs derived from foundational benchmarks focused on recognition tasks (e.g., ASR, acoustic event classification) are insufficient for training models in expert-level reasoning[sakshi2024mmau](https://arxiv.org/html/2507.08128v2#bib.bib101). Therefore, in Stage 3 fine-tuning, we prioritize the development of reasoning and problem-solving abilities by curating large-scale, high-quality Audio QA data. Inspired by AF2, we limit this stage to short audio clips (≤\leq≤30s) and defer long audio reasoning to later stages. We expand the AudioSkills dataset[kong2025audioflamingo2](https://arxiv.org/html/2507.08128v2#bib.bib40) by 4.5M new Audio QA pairs (majorly multiple-choice questions (MCQ)-based) to create AudioSkills-XL, a high-quality corpus containing 8M Audio QA pairs, using two strategies:

(1) We expand coverage of existing reasoning skills and introduce new ones using additional audio sources, increasing the dataset by 3.5M QA pairs: (a) For sounds, we incorporate data from YouTube8M and synthetic sources. (b) For music, we include Music4All[santana2020music4all](https://arxiv.org/html/2507.08128v2#bib.bib102) and the Million Song Dataset[Bertin-Mahieux2011](https://arxiv.org/html/2507.08128v2#bib.bib8). For YouTube8M, we adapt captions from AudioSetCaps[bai2024audiosetcapsnipsws](https://arxiv.org/html/2507.08128v2#bib.bib6) and generate QA using GPT-4.1 with general reasoning prompts from AF2. Additionally, we introduce new reasoning skills and design corresponding prompts to support them. For music, we generate data for novel skills (as AudioSkills was focused more on sounds; details in Table[6](https://arxiv.org/html/2507.08128v2#A2.T6 "Table 6 ‣ Appendix B AudioSkills-XL ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")) and go beyond captions - we leverage metadata such as song titles, artist names, album names, etc (see Fig.[4](https://arxiv.org/html/2507.08128v2#A2.F4 "Figure 4 ‣ B.1.2 Music Knowledge ‣ B.1 Skill-Wise Breakdown ‣ Appendix B AudioSkills-XL ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") for full list) to generate more complex, reasoning-focused QAs. We also use this metadata to generate rich music captions for Stage 1 and 2 pre-training (see Fig.[4](https://arxiv.org/html/2507.08128v2#A2.F4 "Figure 4 ‣ B.1.2 Music Knowledge ‣ B.1 Skill-Wise Breakdown ‣ Appendix B AudioSkills-XL ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")), demonstrating how text-based knowledge can enhance audio understanding, particularly in knowledge-driven domains like music. This method can be seen as synthetic knowledge generation, where we leverage text-based knowledge to enrich audio understanding and enable models to acquire domain-specific knowledge from unlabeled audios in the wild. Our analysis shows that LLMs like GPT-4.1 hold substantial world knowledge about music, and that metadata improves QA quality significantly.

(2) We augment AudioSkills with 1M speech QA samples using YouTube8M([abu2016youtube,](https://arxiv.org/html/2507.08128v2#bib.bib2)), LibriSpeech([panayotov2015librispeech,](https://arxiv.org/html/2507.08128v2#bib.bib92)) (read speech), GigaSpeech([chen2021gigaspeech,](https://arxiv.org/html/2507.08128v2#bib.bib14)) (conversational), and VoxCeleb2([chung2018voxceleb2,](https://arxiv.org/html/2507.08128v2#bib.bib21)) (interviews). From YouTube8M, we introduce a new task: Speech-in-Sound QA, where the model must reason over both speech content and ambient sounds to understand complex auditory scenes. To create these QAs, we create Speech-in-Sound-Caps, a new dataset with ≈\approx≈2M speech-aware auditory scene captions from YouTube8M. To curate this, we first filter the dataset for English speech (using AF2) and transcribe the spoken content with Whisper-Large-v3. We then generate two types of descriptions: one capturing sound events and another summarizing speech characteristics such as tone, emotion, and pitch (both using AF2 and custom prompts; see Appendix[26](https://arxiv.org/html/2507.08128v2#A10.F26 "Figure 26 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")). Finally, we prompt GPT-4.1 to synthesize a speech-aware scene caption. These captions significantly improve the quality of final audio captions (compared to only using sound information) by providing a more holistic representation of the audio. For LibriSpeech and GigaSpeech, we concatenate shorter segments into clips of 15–30 seconds, selecting information-dense segments filtered by prompting an LLM. To move beyond basic spoken content understanding common in most current datasets([zhao2023librisqa,](https://arxiv.org/html/2507.08128v2#bib.bib121)), we design five distinct types of speech QA that require diverse reasoning skills (explained in the next subsection).

### 4.2 LongAudio-XL: Expanding LongAudio with Long Speech QA

To our knowledge, Long Speech QA (i.e., audio ≥\geq≥ 30 seconds) has not been explored in prior work, despite its relevance to real-world applications such as long-form conversation understanding, meeting summarization, and narrative comprehension. To bridge this gap, we extend the existing LongAudio dataset([kong2025audioflamingo2,](https://arxiv.org/html/2507.08128v2#bib.bib40)) (focused on sounds and music) by incorporating over 1M reasoning-focused QA examples from long-form speech (30s-10min). We curate audios from diverse sources including: Single-speaker speech: LibriSpeech (audiobooks)([panayotov2015librispeech,](https://arxiv.org/html/2507.08128v2#bib.bib92)), EuroParl([koehn2005europarl,](https://arxiv.org/html/2507.08128v2#bib.bib62)), VoxPopuli (parliamentary debates)([wang2021voxpopuli,](https://arxiv.org/html/2507.08128v2#bib.bib107)) and Multi-speaker conversations: Spotify Podcasts([clifton-etal-2020-100000,](https://arxiv.org/html/2507.08128v2#bib.bib23)), Switchboard([godfrey1992switchboard,](https://arxiv.org/html/2507.08128v2#bib.bib43)), Fisher (dyadic calls)([cieri2004fisher,](https://arxiv.org/html/2507.08128v2#bib.bib22)), MELD([poria2018meld,](https://arxiv.org/html/2507.08128v2#bib.bib94)), DailyTalk([lee2023dailytalk,](https://arxiv.org/html/2507.08128v2#bib.bib71)), MMDialog (natural dialogues)([feng2022mmdialog,](https://arxiv.org/html/2507.08128v2#bib.bib35)). We merge consecutive short segments in chronological order to construct longer, coherent audios. We construct QAs across a wide range of skills, as illustrated in [Figure˜3](https://arxiv.org/html/2507.08128v2#S4.F3 "In 4.2 LongAudio-XL: Expanding LongAudio with Long Speech QA ‣ 4 Audio Flamingo 3 Training Data ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"):

![Image 4: Refer to caption](https://arxiv.org/html/x4.png)

Figure 3: Examples from AudioSkill-XL, LongAudio-XL, AF-Think, and AF-Chat. We include additional examples in Appendix[B](https://arxiv.org/html/2507.08128v2#A2 "Appendix B AudioSkills-XL ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") and [C](https://arxiv.org/html/2507.08128v2#A3 "Appendix C LongAudio-XL ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), featuring novel music reasoning QAs mentioned in detail in Section[B.1.2](https://arxiv.org/html/2507.08128v2#A2.SS1.SSS2 "B.1.2 Music Knowledge ‣ B.1 Skill-Wise Breakdown ‣ Appendix B AudioSkills-XL ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models").

1.   1.Sarcasm Identification: Inferring sarcasm by analyzing content, tone, and emotional cues. 
2.   2.Emotional State Reasoning:i) Identification: Determine the speaker’s emotion at a specific utterance. ii) Causal Reasoning: Identify the reason behind a speaker’s emotional state using conversational context. iii) Emotion Flip: Explain shifts in a speaker’s emotional state during the conversation. 
3.   3.Topic Relationship Reasoning: Understand how two ideas or topics are related within the overall discourse. 
4.   4.Information Extraction (IE):i) Needle QA: Targeted QA on specific utterances or parts of the speech (e.g., entity or fact extraction, general knowledge linkage). ii) Causal QA: Identify causes for a particular utterance in context. iii) Response QA: Extract how one speaker responds to another’s statement. iv) Topic QA: Identify the main topic of the speech or conversation. 
5.   5.Summarization: Generate a concise summary of the speech content. 
6.   6.Order:i) Temporal Order: Understanding the sequential order of topics in the speech; ii) Temporal Attribute: Understanding how topics change over time; iii) Temporal Referring: Resolve references to specific time points (e.g., "at the end") iv) Temporal Grounding: Identify when in the audio a specific topic was discussed. 

### 4.3 AF-Think: Towards flexible, on-demand reasoning

Recent studies show that making an LLM “think”, similar to chain-of-thought (CoT) prompting([wei2022chain,](https://arxiv.org/html/2507.08128v2#bib.bib111)), can improve reasoning performance in LLMs[guo2025deepseek](https://arxiv.org/html/2507.08128v2#bib.bib47), especially for complex tasks like coding and math (e.g., DeepSeek-R1, OpenAI-o1). Visual MLLMs have also benefited from this paradigm[xu2024llava](https://arxiv.org/html/2507.08128v2#bib.bib116); [wang2025multimodal](https://arxiv.org/html/2507.08128v2#bib.bib109). In the audio domain, early attempts such as Audio-CoT[ma2025audio](https://arxiv.org/html/2507.08128v2#bib.bib80), Audio-Reasoner[xie2025audio](https://arxiv.org/html/2507.08128v2#bib.bib114), and R1-AQA[li2025reinforcement](https://arxiv.org/html/2507.08128v2#bib.bib73) have explored CoT-style reasoning, but often yield limited gains and involve complex or inefficient training procedures. Moreover, consistent with findings in[li2025reinforcement](https://arxiv.org/html/2507.08128v2#bib.bib73), we observe that deep, explicit thinking does not always improve performance in audio understanding tasks.

In AF3, we adopt a lightweight thinking mechanism with two key modifications: (i) We create AF-Think, a dataset of 250k MCQ-based QAs with short, controlled thought preceding the answer. This additional thinking serve as a prefix to the answer and are limited to an average of approximately 40 words, providing concise yet effective context for audio QA (example in [Figure˜3](https://arxiv.org/html/2507.08128v2#S4.F3 "In 4.2 LongAudio-XL: Expanding LongAudio with Long Speech QA ‣ 4 Audio Flamingo 3 Training Data ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")). (ii) Instead of explicitly post-training for CoT, we add a special suffix to QA prompts (highlighted in [Figure˜3](https://arxiv.org/html/2507.08128v2#S4.F3 "In 4.2 LongAudio-XL: Expanding LongAudio with Long Speech QA ‣ 4 Audio Flamingo 3 Training Data ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")). We include AF-Think in the Stage 3.5 training mixture, upweighted relative to standard QA data. This allows AF3 to think only when prompted, offering flexible, on-demand additional reasoning.

To generate AF-Think, we first sample a subset of multiple-choice reasoning QAs from AudioSkills-XL and LongAudio-XL (originally with just the correct option as the answer). Next, we prompt Gemini 2.0 Flash with the input audio, the question, and the answer to generate short thinking prefixes. We found Gemini to hallucinate less and generate more accurate reasoning when guided by the ground-truth answer, rather than producing CoT from scratch. We restrict this process to only high-quality datasets and filter out noisy instances.

### 4.4 AF-Chat: Multi-turn Multi-audio Chat Data

While single-turn single-audio QA training equips LALMs to reason over individual audio inputs, enabling free-form, multi-turn, multi-audio conversations requires a dedicated chat alignment tuning stage, akin to the instruction-tuning phases used for LLMs[zhou2023lima](https://arxiv.org/html/2507.08128v2#bib.bib122). Chat becomes significantly more complex when multiple audio inputs must be integrated across turns, requiring the model to track context, reason over relationships between past and current inputs, and generate coherent follow-ups. Despite its importance and chat being the most used application of LLMs, this capability remains underexplored in LALMs primarily due to the absence of open, high-quality training data.

To address this gap, we introduce AF-Chat, a high-quality fine-tuning dataset consisting of 75k multi-turn, multi-audio chat instances. On average, each dialogue includes 4.6 audio clips and 6.2 dialogue turns, with a range of 2–8 audio clips and 2–10 turns. To construct this dataset, we draw from Speech-in-Sound Caps (for speech and sounds), and Music4All and MSD (for music). We follow a two-step curation process: First, for each seed audio, we identify its top 8 most semantically similar and dissimilar clips using a combination of captions, NV-Embed-v2[lee2024nv](https://arxiv.org/html/2507.08128v2#bib.bib68) embeddings, and FAISS-based clustering([douze2024faiss,](https://arxiv.org/html/2507.08128v2#bib.bib31)) (details in Appendix[E.2](https://arxiv.org/html/2507.08128v2#A5.SS2 "E.2 Clustering for constructing AF-Chat ‣ Appendix E AF-Chat ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")). For every dialogue, we restrict the audios to this pool. This targeted clustering yields significantly higher-quality dialogues than random audio selection by ensuring each instance is grounded in a diverse yet semantically coherent audio pool.

Next, we prompt GPT-4.1 using carefully designed expert exemplars (Fig.[36](https://arxiv.org/html/2507.08128v2#A10.F36 "Figure 36 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") and [35](https://arxiv.org/html/2507.08128v2#A10.F35 "Figure 35 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")) to generate natural, multi-turn chat sessions under the following constraints: (i) the model may choose any subset of the similar/dissimilar audios (up to 10 turns), prioritizing conversation quality; (ii) not all turns require a new audio—follow-up and clarification questions are encouraged; and (iii) later turns may refer back to earlier audios or responses to simulate real conversational grounding. The design of AF-Chat is informed by extensive internal human studies to reflect how users naturally interact with audio-language models. As a result, it provides rich, diverse supervision for aligning LALMs to handle complex, contextual, and naturalistic audio conversations. Finally, we select 200 high-quality samples for the test set, known as AF-Chat-test, and ensure that the audios in these instances have audio clips that were not seen during training.

5 Audio Flamingo 3 Training Strategy
------------------------------------

AF3 is trained using a five-stage strategy designed to progressively enhance its capabilities by increasing audio context length, improving data quality, and diversifying tasks. A full list of datasets used at each stage is provided in Appendix[11](https://arxiv.org/html/2507.08128v2#A7.T11 "Table 11 ‣ Appendix G AF3 Training Datasets ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models").

Stage 1: Alignment pre-training. For this stage, we train only the audio adaptor layers while keeping the audio encoder and LLM frozen. This step aligns encoder representations with the language model. Stage 2: Encoder Tuning. The main purpose of this stage is to adapt AF-Whisper to diverse datasets and broaden and improve its audio understanding capabilities. We fine-tune both the audio encoder and adaptor while keeping the LLM frozen. In both Stages 1 and 2, the audio context length is limited to 30 seconds, and training uses recognition-focused datasets (e.g., classification, captioning, and ASR). Stage 3: Full Fine-Tuning. The primary purpose of this stage is to emphasize reasoning and skill acquisition by the LALM. As mentioned earlier, since skill-specific data is easy to scale on short audios, we still stick to short audios in this stage and use high-quality foundational and QA datasets and our proposed AudioSkills-XL. However, we increase the audio context length up to 2.5 minutes now to accommodate the moderately long audios in AudioSkills. The resulting model at the end of Stage 3.5 is referred to as AF3. Stage 3.5: Context Extension and Thinking. This stage focuses on extending context length and encouraging CoT-style reasoning. In addition to the Stage 3 data mixture, we incorporate LongAudio-XL and AF-Think. We adopt LoRA-based training([hu2022lora,](https://arxiv.org/html/2507.08128v2#bib.bib51))—similar to LTU and GAMA—by freezing the model’s original weights and training LoRA adapters for the LLM. This approach allows end-users to flexibly enhance the model’s reasoning and long-context understanding capabilities on demand. Stage 4: Chat and Voice Fine-Tuning. This stage focuses on enabling multi-turn, interactive, and voice-based dialogue. We fine-tune the entire model on our proposed AF-Chat dataset to equip AF3 with conversational audio understanding and response generation capabilities. The resulting model at the end of Stage 4 is referred to as AF3-Chat.

6 Experiments
-------------

Table 2: Comparison of AF3 with other LALMs on various benchmarks (WER ↓ (Word Error Rate), ACC ↑ (Accuracy), and GPT4o ↑ (GPT evaluation)). We report scores for only the top-performing prior LALM. +Think refers to AF3 with additional thinking. We highlight closed source, open weights, and open source models.

Task Dataset Prior SOTA Metrics Results
Audio Understanding and Reasoning MMAU-v05.15.25 (test)Sound | Music | Speech | Avg Qwen2.5-O ACC ↑76.77 | 67.33 | 68.90 | 71.00
Audio Flamingo 3 75.83 | 74.47 | 66.97 | 72.42
+Think 75.27 | 74.60 | 69.60 | 73.16
MMAU-v05.15.25 (test-mini)Sound | Music | Speech | Avg Qwen2.5-O ACC ↑78.10 | 65.90 | 70.60 | 71.50
Audio Flamingo 3 79.58 | 73.95 | 66.37 | 73.30
+Think 79.88 | 76.55 | 66.37 | 74.26
MMAR Qwen2.5-O ACC ↑56.7
Audio Flamingo 3 58.5
+Think 60.1
MMSU Gemini-1.5-Pro ACC ↑60.7
Audio Flamingo 3 61.4
+Think 62.3
ClothoAQA unanimous | non-binary Qwen2.5-O | Qwen2.5-O ACC ↑89.2 | 52.6
Audio Flamingo 3 91.1 | 56.2
Audio Captioning Clotho-v2 | AudioCaps Audio Flamingo 2 | Audio Flamingo 2 CIDEr ↑0.46 | 0.58
Audio Flamingo 3 0.50 | 0.70
Audio Entailment Clotho | AudioCaps Audio Flamingo 2 | Audio Flamingo 2 ACC ↑92.5 | 93.3
Audio Flamingo 3 93.3 | 95.0
IEMOCAP Qwen2-A-Inst ACC ↑59.2
Audio Flamingo 3 63.8
CochlScene Pengi ACC ↑91.6
Audio Flamingo 3 93.2
NonSpeech7k Audio Flamingo 2 ACC ↑84.3
Audio Flamingo 3 85.9
CMM Hallucination Gemini 2.5 Pro ACC ↑82.0
Audio Flamingo 3 86.5
CompA-R-test Audio Flamingo 2 ACC ↑96.4
Audio Flamingo 3 98.0
MusicAVQA Qwen2.5-O ACC ↑73.4
Audio Flamingo 3 76.7
NSynth Source | Instrument Pengi | Qwen-A ACC ↑62.0 | 78.8
Audio Flamingo 3 65.5 | 78.9
Music Instruct Long Audio Flamingo 2 ACC ↑90.2
Audio Flamingo 3 92.7
MuchoMusic Qwen2-A-Inst ACC ↑46.2
Audio Flamingo 3 47.4
+Think 47.6
LibriSQA Gemini 2.5 Pro GPT4o ↑8.7
Audio Flamingo 3 8.7
LongAudioBench Gemini 2.5 Pro GPT4o ↑60.4
Audio Flamingo 3 68.6
+Speech (ours)Gemini 2.5 Pro GPT4o ↑66.2
Audio Flamingo 3 72.9
Automatic Speech Recognition(ASR)LibriSpeech (en)test-clean | test-other Phi-4-mm | Qwen2.5-O WER ↓1.67 | 3.4
Audio Flamingo 3 1.57 | 3.13
SPGISpeech (en)Qwen2-A-Inst WER ↓3.0
Audio Flamingo 3 1.86
TEDLIUM (en)Phi-4-mm WER ↓2.9
Audio Flamingo 3 3.5
GigaSpeech (en)Phi-4-mm WER ↓9.78
Audio Flamingo 3 10.27
Common Voice 15 (en)Phi-4-mm WER ↓7.61
Audio Flamingo 3 7.4
VoxPopuli (en)Phi-4-mm WER ↓5.91
Audio Flamingo 3 5.55

Experimental Setup. We train AF3 on 128 NVIDIA A100 GPUs, each with 80GB of memory. Details about batch size, learning rates, and optimizers for each stage of training are in [Appendix˜H](https://arxiv.org/html/2507.08128v2#A8 "Appendix H AF3 Training Details ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models").

Baselines. We evaluate our model against recent SOTA LALMs, including GAMA([ghosh2024gama,](https://arxiv.org/html/2507.08128v2#bib.bib41)), Audio Flamingo([kong2024audioflamingo,](https://arxiv.org/html/2507.08128v2#bib.bib65)), Audio Flamingo 2([kong2025audioflamingo2,](https://arxiv.org/html/2507.08128v2#bib.bib40)), Qwen-A(udio)([chu2023qwenaudio,](https://arxiv.org/html/2507.08128v2#bib.bib20)), Qwen2-A(udio)([chu2024qwenaudio2,](https://arxiv.org/html/2507.08128v2#bib.bib19)), Qwen2-A(udio)-(Inst)ruct, Qwen2.5-O(mni)([xu2025qwen2,](https://arxiv.org/html/2507.08128v2#bib.bib117)), R1-AQA[li2025reinforcement](https://arxiv.org/html/2507.08128v2#bib.bib73), Pengi([deshmukh2023pengi,](https://arxiv.org/html/2507.08128v2#bib.bib27)), Phi-4-mm([abouelenin2025phi,](https://arxiv.org/html/2507.08128v2#bib.bib1)), Baichun Audio([li2025baichuan,](https://arxiv.org/html/2507.08128v2#bib.bib75)), Step-Audio-Chat([huang2025step,](https://arxiv.org/html/2507.08128v2#bib.bib52)), LTU([gong2023ltu,](https://arxiv.org/html/2507.08128v2#bib.bib46)), LTU-AS([gong2023ltu-as,](https://arxiv.org/html/2507.08128v2#bib.bib45)), SALMONN([tang2023salmonn,](https://arxiv.org/html/2507.08128v2#bib.bib104)), AudioGPT([huang2023audiogpt,](https://arxiv.org/html/2507.08128v2#bib.bib53)), and Gemini (2.0 Flash, 1.5 Pro, 2.5 Flash and 2.5 Pro)([team2023gemini,](https://arxiv.org/html/2507.08128v2#bib.bib105)) (note we do not evaluate Gemini on ASR benchmarks due to low rate limits), as well as GPT-4o-audio([hurst2024gpt,](https://arxiv.org/html/2507.08128v2#bib.bib54)). For LongAudioBench, for models that do not support longer audio, we follow the cascaded approach for evaluation proposed by([kong2025audioflamingo2,](https://arxiv.org/html/2507.08128v2#bib.bib40)). For Table[3](https://arxiv.org/html/2507.08128v2#S6.T3 "Table 3 ‣ 6 Experiments ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), we only compare against open LALMs. All results reported in the tables correspond to the best-performing model. Evaluation for voice-to-voice capabilities is beyond our scope.

Evaluation Datasets. We evaluate AF3 on a variety of tasks and benchmarks, including audio classification (CochlScene([jeong2022cochlscene,](https://arxiv.org/html/2507.08128v2#bib.bib57)), NSynth (Source and Instrument)([engel2017neural,](https://arxiv.org/html/2507.08128v2#bib.bib34)), NonSpeech7k([rashid2023nonspeech7k,](https://arxiv.org/html/2507.08128v2#bib.bib99)), IEMOCAP([busso2008iemocap,](https://arxiv.org/html/2507.08128v2#bib.bib11))), audio QA (ClothoAQA([lipping2022clotho,](https://arxiv.org/html/2507.08128v2#bib.bib76)), MusicAVQA([li2022learning,](https://arxiv.org/html/2507.08128v2#bib.bib74)), Music Instruct([deng2023musilingo,](https://arxiv.org/html/2507.08128v2#bib.bib26)), LibriSQA([zhao2023librisqa,](https://arxiv.org/html/2507.08128v2#bib.bib121))), reasoning-focused audio QA (MMAU([sakshi2024mmau,](https://arxiv.org/html/2507.08128v2#bib.bib101)) (v05.15.25), MuchoMusic (perceptual version)([zang2025you,](https://arxiv.org/html/2507.08128v2#bib.bib120); [weck2024muchomusic,](https://arxiv.org/html/2507.08128v2#bib.bib110)), MMAR([ma2025mmarchallengingbenchmarkdeep,](https://arxiv.org/html/2507.08128v2#bib.bib81)), MMSU([wang2025mmsu,](https://arxiv.org/html/2507.08128v2#bib.bib108)), CompA-R-test([ghoshcompa,](https://arxiv.org/html/2507.08128v2#bib.bib42)), Audio Entailment([deshmukh2025audio,](https://arxiv.org/html/2507.08128v2#bib.bib29))), multimodal hallucination detection (CMM([leng2024curse,](https://arxiv.org/html/2507.08128v2#bib.bib72))), audio captioning (Clotho-v2([drossos2020clotho,](https://arxiv.org/html/2507.08128v2#bib.bib32)), AudioCaps([kim2019audiocaps,](https://arxiv.org/html/2507.08128v2#bib.bib60))), ASR (Librispeech (clean and other)([panayotov2015librispeech,](https://arxiv.org/html/2507.08128v2#bib.bib92)), SPGISpeech([o2021spgispeech,](https://arxiv.org/html/2507.08128v2#bib.bib90)), TEDLIUM([rousseau2012ted,](https://arxiv.org/html/2507.08128v2#bib.bib100); [hernandez2018ted,](https://arxiv.org/html/2507.08128v2#bib.bib49)), GigaSpeech (Large)([chen2021gigaspeech,](https://arxiv.org/html/2507.08128v2#bib.bib14)), Common Voice 15([commonvoice:2020,](https://arxiv.org/html/2507.08128v2#bib.bib5)) and Voxpopuli([wang2021voxpopuli,](https://arxiv.org/html/2507.08128v2#bib.bib107))) and long audio captioning and QA (LongAudioBench – which we augment with 2.5k human-annotated long-speech QA instances). For evaluating chat capabilities, we conduct a human study of model outputs on AF-Chat-test (more details in [Appendix˜E](https://arxiv.org/html/2507.08128v2#A5 "Appendix E AF-Chat ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")) and compare only with Qwen2-Audio. Each annotator is asked to rate the response of the model for every turn on a scale of 1-5 for factuality, usefulness, and depth. We report results averaged across all instances across all turns. Furthermore, we evaluate the voice-text capabilities of our AF3-Chat model on two datasets, OpenAudioBench([li2025baichuan,](https://arxiv.org/html/2507.08128v2#bib.bib75)) and VoiceBench([chen2024voicebench,](https://arxiv.org/html/2507.08128v2#bib.bib18)). These benchmarks consist of voice queries (synthetically generated speech from text queries) and assess aspects such as instruction following, question answering, trivia knowledge, and reasoning. Finally, we evaluate our speech generation module using zero-shot TTS evaluation on the English subset of the SEED benchmark([anastassiou2024seed,](https://arxiv.org/html/2507.08128v2#bib.bib4)). All baseline results reported in this work are based on our own evaluations; we did not rely on results from prior literature (in some cases, we were unable to reproduce the numbers as originally reported). To calculate accuracy, we use either exact string matching with the ground truth or CLAP-based retrieval following([deshmukh2023pengi,](https://arxiv.org/html/2507.08128v2#bib.bib27)), implemented with open-source AF-CLAP([kong2025audioflamingo2,](https://arxiv.org/html/2507.08128v2#bib.bib40)). For MCQ, AF3 typically outputs only the selected option. In cases where the model provides more verbose or open-ended responses (e.g., with thinking mode), we apply multiple regex patterns to extract the chosen option.

Table 3: Comparison of AF3 with open LALMs on AF-Chat, voice-text and TTS benchmarks. WER ↓ (Word Error Rate), SIM ↑ (Similarity), and GPT4o ↑ (GPT evaluation) indicate that lower or higher is better.

Task Dataset Model Metrics Results
Multi-audio chat AF-Chat-test Factuality | Usefulness | Depth Qwen2.5-O GPT4o ↑2.4 | 2.7 | 3.2
AF3-Chat 3.6 | 3.4 | 3.9
Voice-Text OpenAudioBench alpaca-eval | llama-questions |trivia-qa Qwen2-A-Inst GPT4o ↑57.19 | 69.67 | 40.30
Qwen2.5-O 72.76 | 75.33 | 57.06
AF3-Chat 76.26 | 80.33 | 53.05
VoiceBench AlpacaEval | AdvBench |OpenBookQA | Commoneval Qwen2-A-Inst GPT4o ↑3.69 | 98.85 | 49.01 | 3.40
Qwen2.5-O 4.33 | 99.62 | 79.12 | 3.84
AF3-Chat 4.19 | 98.26 | 66.81 | 3.40
Speech Generation SEED (test-en)Content Cons. | Speaker Sim. | Inf. Time Qwen2.5-O WER ↓ | SIM ↑ | Time ↓2.72 | 0.63 | 14.62s (1.26s)
AF3-Chat 2.02 | 0.61 |  5.94s (0.02s)

### 6.1 Audio Understanding and Reasoning Evaluation

AF3 is the strongest and fully open-source LALM. [Table˜2](https://arxiv.org/html/2507.08128v2#S6.T2 "In 6 Experiments ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") shows AF3 outperforming previous SOTA open-weight and closed-source models across a wide range of audio understanding and reasoning benchmarks. AF3 sets new highs on MMAU (72.42) (note for Qwen2.5-Omni on MMAU we report the “parsed score” for fair evaluation), ClothoAQA (91.1), Clotho Entailment (92.9), and CMM Hallucination (86.7). On tasks like NSynth and MusicInstruct, it shows significant gains, highlighting strong sound and music understanding. For LongAudioBench (sound and speech), AF3 outperforms Gemini 2.5 Pro by a wide margin, demonstrating its strength in long-context reasoning. We also evaluate AF3 with thinking prompts (+Think) on reasoning-heavy benchmarks like MMAU and MuchoMusic, observing a performance boost. Although the thinking mode is activated after Stage 3.5 only when using our specific thinking prompt, the checkpoint remains usable without it. We report average scores of 73.16 and 74.26 on MMAU-test and MMAU-test-mini, respectively. Additionally, AF3 achieves state-of-the-art ASR results on LibriSpeech, SPGISpeech, and VoxPopuli—even compared to dedicated ASR models—despite not being trained on large-scale ASR datasets like many open-weight models. We illustrate a demo of AF3’s capabilities in Fig.[14](https://arxiv.org/html/2507.08128v2#A10.F14 "Figure 14 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models").

### 6.2 Chat and TTS Evaluation

Multi-turn multi-audio chat evaluation.  On AF-Chat-test AF3-Chat shows a relative improvement of 30% over Qwen2.5-Omni, thereby showing the capability of effectively handling extended dialog turns, allowing for deeper contextual reasoning and more accurate references to multiple audio inputs.

Voice-Text and Speech Generation Evaluation. [Table˜3](https://arxiv.org/html/2507.08128v2#S6.T3 "In 6 Experiments ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") evaluates AF3-Chat on two key tasks: voice-to-text and text-to-speech generation. In the voice-to-text setting (spoken QA), AF3-Chat achieves strong gains across all of OpenAudioBench, surpassing Qwen2.5-Omni. On VoiceBench, which tests spoken QA robustness across AdvBench, CommonEval, and OpenBookQA, AF3-Chat performs comparably to Qwen2.5-Omni and Qwen2-Audio Chat. For TTS (evaluated on SEED test-en), AF3-Chat shows improved performance with a lower WER of 2.02 (vs. 2.72 for Qwen2.5-Omni) and a speaker similarity score of 0.61, closely matching Qwen2.5’s 0.63.

Furthermore, AF3-Chat exhibits significant advantages in generation speed. For a 10-second audio generation on an A100 GPU, AF3-Chat’s text-to-audio token generation is 5.94 seconds with an additional 0.02 seconds for waveform synthesis. In comparison, the Talker model of Qwen2.5-Omni requires 14.62 seconds for token generation and an additional 1.26 seconds for waveform synthesis. This efficiency allows our streaming text-to-speech to achieve a time-to-first-token of 0.15 seconds and an inter-token latency of 0.06 seconds (both including waveform synthesis), producing a 10-second audio clip in 6.68 seconds.

### 6.3 Ablation Studies

In this section, we ablate our key components (using just 10% of the training data) to support the paper’s main claims.

Evaluating AF-Whisper as a Unified Encoder.[Table˜4](https://arxiv.org/html/2507.08128v2#S6.T4 "In 6.3 Ablation Studies ‣ 6 Experiments ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") compares AF3 trained with our unified AF-Whisper encoder against a dual-encoder setup using CLAP for sounds/music and Whisper-v3 for speech([elizalde2022clap,](https://arxiv.org/html/2507.08128v2#bib.bib33); [radford2022whisper,](https://arxiv.org/html/2507.08128v2#bib.bib96)). AF-Whisper outperforms the dual-encoder model under the same data budget, demonstrating its effectiveness as a single encoder for sound, music, and speech.

AudioSkills-XL: A Key Dataset for Performance Gains.: To measure the impact of AudioSkills-XL, we ablate it from Stage 3 of training and compare results to the full setup. As shown in [Table˜4](https://arxiv.org/html/2507.08128v2#S6.T4 "In 6.3 Ablation Studies ‣ 6 Experiments ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), removing AudioSkills-XL causes a significant performance drop—particularly on MMAU—underscoring its role in improving generalization and robustness. These findings highlight the value of large-scale, skill-targeted audio QA data for fine-tuning multi-modal models.

Table 4: Comparison of AF3 w/ 10% data, w/o AF-Whisper and w/o AudioSkills-XL.

| Model | MMAU-Sound | MMAU-Music | MMAU-Speech | Librispeech-clean | Librispeech-other |
| --- | --- | --- | --- | --- | --- |
|  | ACC ↑ | ACC ↑ | ACC ↑ | WER ↓ | WER ↓ |
| w/ 10% data | 66.7 | 65.9 | 57.4 | 2.0 | 4.1 |
| + w/o AF-Whisper | 63.7 | 68.3 | 45.2 | 3.7 | 7.2 |
| w/o AudioSkills-XL | 56.1 | 42.1 | 14.3 | 1.6 | 3.6 |
| Audio Flamingo 3 | 75.8 | 74.4 | 66.9 | 1.5 | 3.1 |

7 Conclusion, Limitations and Future Work
-----------------------------------------

In this paper, we introduce Audio Flamingo 3, the most capable and open LALM. Our model leverages a custom Whisper, novel data curation techniques, and a 5-stage curriculum learning strategy. Audio Flamingo 3 not only achieves SOTA performance in audio understanding and reasoning but also introduces capabilities, including multi-turn multi-audio chat, on-demand thinking, and voice chat. We detail our practices, including architecture, training, inference, and the evaluation pipeline, and open-source two large datasets. For future work, we aim to address current limitations, including: (1) mitigating the need for a cascaded system for voice chat, (2) making AF3 multi-lingual, and (3) reducing dependency on closed-source models for synthetic data.

References
----------

*   (1) A.Abouelenin, A.Ashfaq, A.Atkinson, H.Awadalla, N.Bach, J.Bao, A.Benhaim, M.Cai, V.Chaudhary, C.Chen, et al. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras. arXiv preprint arXiv:2503.01743, 2025. 
*   (2) S.Abu-El-Haija, N.Kothari, J.Lee, P.Natsev, G.Toderici, B.Varadarajan, and S.Vijayanarasimhan. Youtube-8m: A large-scale video classification benchmark. arXiv preprint arXiv:1609.08675, 2016. 
*   (3) A.Agostinelli, T.I. Denk, Z.Borsos, J.Engel, M.Verzetti, A.Caillon, Q.Huang, A.Jansen, A.Roberts, M.Tagliasacchi, et al. Musiclm: Generating music from text. arXiv preprint arXiv:2301.11325, 2023. 
*   (4) P.Anastassiou, J.Chen, J.Chen, Y.Chen, Z.Chen, Z.Chen, J.Cong, L.Deng, C.Ding, L.Gao, et al. Seed-tts: A family of high-quality versatile speech generation models. arXiv preprint arXiv:2406.02430, 2024. 
*   (5) R.Ardila, M.Branson, K.Davis, M.Henretty, M.Kohler, J.Meyer, R.Morais, L.Saunders, F.M. Tyers, and G.Weber. Common voice: A massively-multilingual speech corpus. In Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020), pages 4211–4215, 2020. 
*   (6) J.Bai, H.Liu, M.Wang, D.Shi, W.Wang, M.D. Plumbley, W.-S. Gan, and J.Chen. Audiosetcaps: Enriched audio captioning dataset generation using large audio language models. In Audio Imagination: NeurIPS 2024 Workshop AI-Driven Speech, Music, and Sound Generation, 2024. 
*   (7) P.Barros, N.Churamani, E.Lakomkin, H.Siqueira, A.Sutherland, and S.Wermter. The omg-emotion behavior dataset. In 2018 International Joint Conference on Neural Networks (IJCNN), pages 1–7. IEEE, 2018. 
*   (8) T.Bertin-Mahieux, D.P. Ellis, B.Whitman, and P.Lamere. The million song dataset. In Proceedings of the 12th International Conference on Music Information Retrieval (ISMIR 2011), 2011. 
*   (9) T.Bertin-Mahieux, D.P. Ellis, B.Whitman, and P.Lamere. The million song dataset. In Ismir, volume 2, page 10, 2011. 
*   (10) R.M. Bittner, J.Salamon, M.Tierney, M.Mauch, C.Cannam, and J.P. Bello. Medleydb: A multitrack dataset for annotation-intensive mir research. In Ismir, volume 14, pages 155–160, 2014. 
*   (11) C.Busso, M.Bulut, C.-C. Lee, A.Kazemzadeh, E.Mower, S.Kim, J.N. Chang, S.Lee, and S.S. Narayanan. Iemocap: Interactive emotional dyadic motion capture database. Language resources and evaluation, 42:335–359, 2008. 
*   (12) M.Cartwright, J.Cramer, A.E.M. Mendez, Y.Wang, H.-H. Wu, V.Lostanlen, M.Fuentes, G.Dove, C.Mydlarz, J.Salamon, et al. Sonyc-ust-v2: An urban sound tagging dataset with spatiotemporal context. arXiv preprint arXiv:2009.05188, 2020. 
*   (13) C.Chen, P.Peng, A.Baid, Z.Xue, W.-N. Hsu, D.Harwath, and K.Grauman. Action2sound: Ambient-aware generation of action sounds from egocentric videos. In European Conference on Computer Vision, pages 277–295. Springer, 2024. 
*   (14) G.Chen, S.Chai, G.Wang, J.Du, W.-Q. Zhang, C.Weng, D.Su, D.Povey, J.Trmal, J.Zhang, et al. Gigaspeech: An evolving, multi-domain asr corpus with 10,000 hours of transcribed audio. arXiv preprint arXiv:2106.06909, 2021. 
*   (15) H.Chen, W.Xie, A.Vedaldi, and A.Zisserman. Vggsound: A large-scale audio-visual dataset. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 721–725. IEEE, 2020. 
*   (16) S.Chen, Y.Wu, C.Wang, S.Liu, D.Tompkins, Z.Chen, and F.Wei. Beats: Audio pre-training with acoustic tokenizers, 2022. 
*   (17) Y.Chen, F.Xue, D.Li, Q.Hu, L.Zhu, X.Li, Y.Fang, H.Tang, S.Yang, Z.Liu, et al. Longvila: Scaling long-context visual language models for long videos. arXiv preprint arXiv:2408.10188, 2024. 
*   (18) Y.Chen, X.Yue, C.Zhang, X.Gao, R.T. Tan, and H.Li. Voicebench: Benchmarking llm-based voice assistants. arXiv preprint arXiv:2410.17196, 2024. 
*   (19) Y.Chu, J.Xu, Q.Yang, H.Wei, X.Wei, Z.Guo, Y.Leng, Y.Lv, J.He, J.Lin, C.Zhou, and J.Zhou. Qwen2-audio technical report, 2024. 
*   (20) Y.Chu, J.Xu, X.Zhou, Q.Yang, S.Zhang, Z.Yan, C.Zhou, and J.Zhou. Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models, 2023. 
*   (21) J.S. Chung, A.Nagrani, and A.Zisserman. Voxceleb2: Deep speaker recognition. arXiv preprint arXiv:1806.05622, 2018. 
*   (22) C.Cieri, D.Miller, and K.Walker. The fisher corpus: A resource for the next generations of speech-to-text. In LREC, volume 4, pages 69–71, 2004. 
*   (23) A.Clifton, S.Reddy, Y.Yu, A.Pappu, R.Rezapour, H.Bonab, M.Eskevich, G.Jones, J.Karlgren, B.Carterette, and R.Jones. 100,000 podcasts: A spoken English document corpus. In Proceedings of the 28th International Conference on Computational Linguistics, pages 5903–5917, Barcelona, Spain (Online), Dec. 2020. International Committee on Computational Linguistics. 
*   (24) F.Daniel, M.Matera, V.Zaccaria, and A.Dell’Orto. Toward truly personal chatbots: on the development of custom conversational assistants. In Proceedings of the 1st international workshop on software engineering for cognitive services, pages 31–36, 2018. 
*   (25) M.Defferrard, K.Benzi, P.Vandergheynst, and X.Bresson. Fma: A dataset for music analysis. arXiv preprint arXiv:1612.01840, 2016. 
*   (26) Z.Deng, Y.Ma, Y.Liu, R.Guo, G.Zhang, W.Chen, W.Huang, and E.Benetos. Musilingo: Bridging music and text with pre-trained language models for music captioning and query response. arXiv preprint arXiv:2309.08730, 2023. 
*   (27) S.Deshmukh, B.Elizalde, R.Singh, and H.Wang. Pengi: An audio language model for audio tasks, 2023. 
*   (28) S.Deshmukh, B.Elizalde, and H.Wang. Audio retrieval with wavtext5k and clap training. arXiv preprint arXiv:2209.14275, 2022. 
*   (29) S.Deshmukh, S.Han, H.Bukhari, B.Elizalde, H.Gamper, R.Singh, and B.Raj. Audio entailment: Assessing deductive reasoning for audio understanding. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 23769–23777, 2025. 
*   (30) S.Doh, K.Choi, J.Lee, and J.Nam. Lp-musiccaps: Llm-based pseudo music captioning. arXiv preprint arXiv:2307.16372, 2023. 
*   (31) M.Douze, A.Guzhva, C.Deng, J.Johnson, G.Szilvasy, P.-E. Mazaré, M.Lomeli, L.Hosseini, and H.Jégou. The faiss library. arXiv preprint arXiv:2401.08281, 2024. 
*   (32) K.Drossos, S.Lipping, and T.Virtanen. Clotho: An audio captioning dataset. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 736–740. IEEE, 2020. 
*   (33) B.Elizalde, S.Deshmukh, M.Al Ismail, and H.Wang. Clap: Learning audio concepts from natural language supervision, 2022. 
*   (34) J.Engel, C.Resnick, A.Roberts, S.Dieleman, M.Norouzi, D.Eck, and K.Simonyan. Neural audio synthesis of musical notes with wavenet autoencoders. In International conference on machine learning, pages 1068–1077. PMLR, 2017. 
*   (35) J.Feng, Q.Sun, C.Xu, P.Zhao, Y.Yang, C.Tao, D.Zhao, and Q.Lin. Mmdialog: A large-scale multi-turn dialogue dataset towards multi-modal open-domain conversation. arXiv preprint arXiv:2211.05719, 2022. 
*   (36) E.Fonseca, X.Favory, J.Pons, F.Font, and X.Serra. Fsd50k: an open dataset of human-labeled sound events. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 30:829–852, 2021. 
*   (37) E.Fonseca, J.Pons, X.Favory, F.Font, D.Bogdanov, A.Ferraro, S.Oramas, A.Porter, and X.Serra. Freesound datasets: A platform for the creation of open audio datasets. In ISMIR, pages 486–493, 2017. 
*   (38) P.Foster, S.Sigtia, S.Krstulovic, J.Barker, and M.D. Plumbley. Chime-home: A dataset for sound source recognition in a domestic environment. In 2015 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), pages 1–5. IEEE, 2015. 
*   (39) J.F. Gemmeke, D.P. Ellis, D.Freedman, A.Jansen, W.Lawrence, R.C. Moore, M.Plakal, and M.Ritter. Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 776–780. IEEE, 2017. 
*   (40) S.Ghosh, Z.Kong, S.Kumar, S.Sakshi, J.Kim, W.Ping, R.Valle, D.Manocha, and B.Catanzaro. Audio flamingo 2: An audio-language model with long-audio understanding and expert reasoning abilities, 2025. 
*   (41) S.Ghosh, S.Kumar, A.Seth, C.K.R. Evuru, U.Tyagi, Sakshi, O.Nieto, R.Duraiswami, and D.Manocha. Gama: A large audio-language model with advanced audio understanding and complex reasoning abilities, 2024. 
*   (42) S.Ghosh, A.Seth, S.Kumar, U.Tyagi, C.K.R. Evuru, S.Ramaneswaran, S.Sakshi, O.Nieto, R.Duraiswami, and D.Manocha. Compa: Addressing the gap in compositional reasoning in audio-language models. In The Twelfth International Conference on Learning Representations. 
*   (43) J.J. Godfrey, E.C. Holliman, and J.McDaniel. Switchboard: Telephone speech corpus for research and development. In Acoustics, speech, and signal processing, ieee international conference on, volume 1, pages 517–520. IEEE Computer Society, 1992. 
*   (44) A.Goel, Z.Kong, R.Valle, and B.Catanzaro. Audio dialogues: Dialogues dataset for audio and music understanding. arXiv preprint arXiv:2404.07616, 2024. 
*   (45) Y.Gong, A.H. Liu, H.Luo, L.Karlinsky, and J.Glass. Joint audio and speech understanding, 2023. 
*   (46) Y.Gong, H.Luo, A.H. Liu, L.Karlinsky, and J.Glass. Listen, think, and understand, 2023. 
*   (47) D.Guo, D.Yang, H.Zhang, J.Song, R.Zhang, R.Xu, Q.Zhu, S.Ma, P.Wang, X.Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 
*   (48) A.Guzhov, F.Raue, J.Hees, and A.Dengel. Audioclip: Extending clip to image, text and audio, 2021. 
*   (49) F.Hernandez, V.Nguyen, S.Ghannay, N.Tomashenko, and Y.Esteve. Ted-lium 3: Twice as much data and corpus repartition for experiments on speaker adaptation. In Speech and Computer: 20th International Conference, SPECOM 2018, Leipzig, Germany, September 18–22, 2018, Proceedings 20, pages 198–208. Springer, 2018. 
*   (50) S.Hershey, D.P. Ellis, E.Fonseca, A.Jansen, C.Liu, R.C. Moore, and M.Plakal. The benefit of temporally-strong labels in audio event classification. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 366–370. IEEE, 2021. 
*   (51) E.J. Hu, Y.Shen, P.Wallis, Z.Allen-Zhu, Y.Li, S.Wang, L.Wang, W.Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 
*   (52) A.Huang, B.Wu, B.Wang, C.Yan, C.Hu, C.Feng, F.Tian, F.Shen, J.Li, M.Chen, et al. Step-audio: Unified understanding and generation in intelligent speech interaction. arXiv preprint arXiv:2502.11946, 2025. 
*   (53) R.Huang, M.Li, D.Yang, J.Shi, X.Chang, Z.Ye, Y.Wu, Z.Hong, J.Huang, J.Liu, Y.Ren, Z.Zhao, and S.Watanabe. Audiogpt: Understanding and generating speech, music, sound, and talking head, 2023. 
*   (54) A.Hurst, A.Lerer, A.P. Goucher, A.Perelman, A.Ramesh, A.Clark, A.Ostrow, A.Welihinda, A.Hayes, A.Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 
*   (55) M.M. Islam, N.Ho, X.Yang, T.Nagarajan, L.Torresani, and G.Bertasius. Video recap: Recursive captioning of hour-long videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18198–18208, 2024. 
*   (56) J.James, L.Tian, and C.I. Watson. An open source emotional speech corpus for human robot interaction applications. In Interspeech, pages 2768–2772, 2018. 
*   (57) I.-Y. Jeong and J.Park. Cochlscene: Acquisition of acoustic scene data using crowdsourcing. In 2022 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), pages 17–21. IEEE, 2022. 
*   (58) X.Ju, Y.Gao, Z.Zhang, Z.Yuan, X.Wang, A.Zeng, Y.Xiong, Q.Xu, and Y.Shan. Miradata: A large-scale video dataset with long durations and structured captions. Advances in Neural Information Processing Systems, 37:48955–48970, 2024. 
*   (59) W.Kang, X.Yang, Z.Yao, F.Kuang, Y.Yang, L.Guo, L.Lin, and D.Povey. Libriheavy: A 50,000 hours asr corpus with punctuation casing and context. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 10991–10995. IEEE, 2024. 
*   (60) C.D. Kim, B.Kim, H.Lee, and G.Kim. Audiocaps: Generating captions for audios in the wild. In NAACL-HLT, 2019. 
*   (61) J.Kim, T.Moon, K.Lee, and J.Cho. Efficient generative modeling with residual vector quantization-based tokens. arXiv preprint arXiv:2412.10208, 2024. 
*   (62) P.Koehn. Europarl: A parallel corpus for statistical machine translation. In Proceedings of machine translation summit x: papers, pages 79–86, 2005. 
*   (63) A.S. Koepke, A.-M. Oncescu, J.F. Henriques, Z.Akata, and S.Albanie. Audio retrieval with natural language queries: A benchmark study. IEEE Transactions on Multimedia, 25:2675–2685, 2022. 
*   (64) Y.Koizumi, H.Zen, S.Karita, Y.Ding, K.Yatabe, N.Morioka, M.Bacchiani, Y.Zhang, W.Han, and A.Bapna. Libritts-r: A restored multi-speaker text-to-speech corpus. INTERSPEECH 2023, 2023. 
*   (65) Z.Kong, A.Goel, R.Badlani, W.Ping, R.Valle, and B.Catanzaro. Audio flamingo: A novel audio language model with few-shot learning and dialogue abilities, 2024. 
*   (66) R.Kumar, P.Seetharaman, A.Luebs, I.Kumar, and K.Kumar. High-fidelity audio compression with improved rvqgan. In Thirty-seventh Conference on Neural Information Processing Systems. 
*   (67) E.Law, K.West, M.Mandel, M.Bay, and J.Downie. Evaluation of algorithms using games: the case of music annotation. In Proceedings of the 11th International Society for Music Information Retrieval Conference (ISMIR). Utrecht, the Netherlands, 2010. 
*   (68) C.Lee, R.Roy, M.Xu, J.Raiman, M.Shoeybi, B.Catanzaro, and W.Ping. Nv-embed: Improved techniques for training llms as generalist embedding models. arXiv preprint arXiv:2405.17428, 2024. 
*   (69) D.Lee, C.Kim, S.Kim, M.Cho, and W.-S. Han. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11523–11532, 2022. 
*   (70) K.Lee, D.W. Kim, J.Kim, S.Chung, and J.Cho. DiTTo-TTS: Diffusion transformers for scalable text-to-speech without domain-specific factors. In The Thirteenth International Conference on Learning Representations, 2025. 
*   (71) K.Lee, K.Park, and D.Kim. Dailytalk: Spoken dialogue dataset for conversational text-to-speech. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023. 
*   (72) S.Leng, Y.Xing, Z.Cheng, Y.Zhou, H.Zhang, X.Li, D.Zhao, S.Lu, C.Miao, and L.Bing. The curse of multi-modalities: Evaluating hallucinations of large multimodal models across language, visual, and audio. arXiv preprint arXiv:2410.12787, 2024. 
*   (73) G.Li, J.Liu, H.Dinkel, Y.Niu, J.Zhang, and J.Luan. Reinforcement learning outperforms supervised fine-tuning: A case study on audio question answering. arXiv preprint arXiv:2503.11197, 2025. 
*   (74) G.Li, Y.Wei, Y.Tian, C.Xu, J.-R. Wen, and D.Hu. Learning to answer questions in dynamic audio-visual scenarios. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19108–19118, 2022. 
*   (75) T.Li, J.Liu, T.Zhang, Y.Fang, D.Pan, M.Wang, Z.Liang, Z.Li, M.Lin, G.Dong, et al. Baichuan-audio: A unified framework for end-to-end speech interaction. arXiv preprint arXiv:2502.17239, 2025. 
*   (76) S.Lipping, P.Sudarsanam, K.Drossos, and T.Virtanen. Clotho-aqa: A crowdsourced dataset for audio question answering. In 2022 30th European Signal Processing Conference (EUSIPCO), pages 1140–1144. IEEE, 2022. 
*   (77) S.Liu, H.J. Cho, M.Freedman, X.Ma, and J.May. Recap: retrieval-enhanced context-aware prefix encoder for personalized dialogue response generation. arXiv preprint arXiv:2306.07206, 2023. 
*   (78) S.Liu, A.S. Hussain, C.Sun, and Y.Shan. Music understanding llama: Advancing text-to-music generation with question answering and captioning. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 286–290. IEEE, 2024. 
*   (79) Z.Liu, H.Mao, C.-Y. Wu, C.Feichtenhofer, T.Darrell, and S.Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976–11986, 2022. 
*   (80) Z.Ma, Z.Chen, Y.Wang, E.S. Chng, and X.Chen. Audio-cot: Exploring chain-of-thought reasoning in large audio language model. arXiv preprint arXiv:2501.07246, 2025. 
*   (81) Z.Ma, Y.Ma, Y.Zhu, C.Yang, Y.-W. Chao, R.Xu, W.Chen, Y.Chen, Z.Chen, J.Cong, K.Li, K.Li, S.Li, X.Li, X.Li, Z.Lian, Y.Liang, M.Liu, Z.Niu, T.Wang, Y.Wang, Y.Wang, Y.Wu, G.Yang, J.Yu, R.Yuan, Z.Zheng, Z.Zhou, H.Zhu, W.Xue, E.Benetos, K.Yu, E.-S. Chng, and X.Chen. Mmar: A challenging benchmark for deep reasoning in speech, audio, music, and their mix, 2025. 
*   (82) L.Martinez-Lucas, M.Abdelwahab, and C.Busso. The msp-conversation corpus. Interspeech 2020, 2020. 
*   (83) X.Mei, C.Meng, H.Liu, Q.Kong, T.Ko, C.Zhao, M.D. Plumbley, Y.Zou, and W.Wang. Wavcaps: A chatgpt-assisted weakly-labelled audio captioning dataset for audio-language multimodal research. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2024. 
*   (84) J.Melechovsky, Z.Guo, D.Ghosal, N.Majumder, D.Herremans, and S.Poria. Mustango: Toward controllable text-to-music generation. arXiv preprint arXiv:2311.08355, 2023. 
*   (85) A.Mesaros, T.Heittola, and T.Virtanen. A multi-device dataset for urban acoustic scene classification. arXiv preprint arXiv:1807.09840, 2018. 
*   (86) J.Moon, Y.Kong, and K.H. Chon. Language-independent sleepy speech detection. In 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), pages 1981–1984. IEEE, 2022. 
*   (87) I.M. Morato and A.Mesaros. Diversity and bias in audio captioning datasets. In Detection and Classication of Acoustic Scenes and Events, pages 90–94, 2021. 
*   (88) M.R. Morris, J.Sohl-Dickstein, N.Fiedel, T.Warkentin, A.Dafoe, A.Faust, C.Farabet, and S.Legg. Position: Levels of agi for operationalizing progress on the path to agi. In Forty-first International Conference on Machine Learning, 2024. 
*   (89) A.-M. Oncescu, A.Koepke, J.F. Henriques, Z.Akata, and S.Albanie. Audio retrieval with natural language queries. arXiv preprint arXiv:2105.02192, 2021. 
*   (90) P.K. O’Neill, V.Lavrukhin, S.Majumdar, V.Noroozi, Y.Zhang, O.Kuchaiev, J.Balam, Y.Dovzhenko, K.Freyberg, M.D. Shulman, et al. Spgispeech: 5,000 hours of transcribed financial audio for fully formatted end-to-end speech recognition. arXiv preprint arXiv:2104.02014, 2021. 
*   (91) Z.Ouyang, J.-C. Wang, D.Zhang, B.Chen, S.Li, and Q.Lin. Mqad: A large-scale question answering dataset for training music large language models. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2025. 
*   (92) V.Panayotov, G.Chen, D.Povey, and S.Khudanpur. Librispeech: an asr corpus based on public domain audio books. In 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 5206–5210. IEEE, 2015. 
*   (93) W.Peebles and S.Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023. 
*   (94) S.Poria, D.Hazarika, N.Majumder, G.Naik, E.Cambria, and R.Mihalcea. Meld: A multimodal multi-party dataset for emotion recognition in conversations. arXiv preprint arXiv:1810.02508, 2018. 
*   (95) V.Pratap, Q.Xu, A.Sriram, G.Synnaeve, and R.Collobert. Mls: A large-scale multilingual dataset for speech research. arXiv preprint arXiv:2012.03411, 2020. 
*   (96) A.Radford, J.W. Kim, T.Xu, G.Brockman, C.McLeavey, and I.Sutskever. Robust speech recognition via large-scale weak supervision, 2022. 
*   (97) Z.Rafii, A.Liutkus, F.-R. Stöter, S.I. Mimilakis, and R.Bittner. The musdb18 corpus for music separation. 2017. 
*   (98) M.A. Rahman, Z.I.A. Hakim, N.H. Sarker, B.Paul, and S.A. Fattah. Sonics: Synthetic or not–identifying counterfeit songs. arXiv preprint arXiv:2408.14080, 2024. 
*   (99) M.M. Rashid, G.Li, and C.Du. Nonspeech7k dataset: Classification and analysis of human non-speech sound. IET Signal Processing, 17(6):e12233, 2023. 
*   (100) A.Rousseau, P.Deléglise, and Y.Esteve. Ted-lium: an automatic speech recognition dedicated corpus. In LREC, pages 125–129, 2012. 
*   (101) S.Sakshi, U.Tyagi, S.Kumar, A.Seth, R.Selvakumar, O.Nieto, R.Duraiswami, S.Ghosh, and D.Manocha. Mmau: A massive multi-task audio understanding and reasoning benchmark. arXiv preprint arXiv:2410.19168, 2024. 
*   (102) I.A.P. Santana, F.Pinhelli, J.Donini, L.Catharin, R.B. Mangolin, V.D. Feltrim, M.A. Domingues, et al. Music4all: A new music database and its applications. In 2020 International Conference on Systems, Signals and Image Processing (IWSSIP), pages 399–404. IEEE, 2020. 
*   (103) H.Siuzdak. Vocos: Closing the gap between time-domain and fourier-based neural vocoders for high-quality audio synthesis. In The Twelfth International Conference on Learning Representations. 
*   (104) C.Tang, W.Yu, G.Sun, X.Chen, T.Tan, W.Li, L.Lu, Z.Ma, and C.Zhang. Salmonn: Towards generic hearing abilities for large language models, 2023. 
*   (105) G.Team, R.Anil, S.Borgeaud, J.-B. Alayrac, J.Yu, R.Soricut, J.Schalkwyk, A.M. Dai, A.Hauth, K.Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 
*   (106) H.Touvron, T.Lavril, G.Izacard, X.Martinet, M.-A. Lachaux, T.Lacroix, B.Rozière, N.Goyal, E.Hambro, F.Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 
*   (107) C.Wang, M.Riviere, A.Lee, A.Wu, C.Talnikar, D.Haziza, M.Williamson, J.Pino, and E.Dupoux. Voxpopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation. arXiv preprint arXiv:2101.00390, 2021. 
*   (108) D.Wang, J.Wu, J.Li, D.Yang, X.Chen, T.Zhang, and H.Meng. Mmsu: A massive multi-task spoken language understanding and reasoning benchmark. arXiv preprint arXiv:2506.04779, 2025. 
*   (109) Y.Wang, S.Wu, Y.Zhang, S.Yan, Z.Liu, J.Luo, and H.Fei. Multimodal chain-of-thought reasoning: A comprehensive survey. arXiv preprint arXiv:2503.12605, 2025. 
*   (110) B.Weck, I.Manco, E.Benetos, E.Quinton, G.Fazekas, and D.Bogdanov. Muchomusic: Evaluating music understanding in multimodal audio-language models. arXiv preprint arXiv:2408.01337, 2024. 
*   (111) J.Wei, X.Wang, D.Schuurmans, M.Bosma, F.Xia, E.Chi, Q.V. Le, D.Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 
*   (112) H.-H. Wu, P.Seetharaman, K.Kumar, and J.P. Bello. Wav2clip: Learning robust audio representations from clip, 2021. 
*   (113) Y.Wu, K.Chen, T.Zhang, Y.Hui, T.Berg-Kirkpatrick, and S.Dubnov. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023. 
*   (114) Z.Xie, M.Lin, Z.Liu, P.Wu, S.Yan, and C.Miao. Audio-reasoner: Improving reasoning capability in large audio language models. arXiv preprint arXiv:2503.02318, 2025. 
*   (115) Z.Xie and C.Wu. Mini-omni: Language models can hear, talk while thinking in streaming. arXiv preprint arXiv:2408.16725, 2024. 
*   (116) G.Xu, P.Jin, L.Hao, Y.Song, L.Sun, and L.Yuan. Llava-o1: Let vision language models reason step-by-step. arXiv preprint arXiv:2411.10440, 2024. 
*   (117) J.Xu, Z.Guo, J.He, H.Hu, T.He, S.Bai, K.Chen, J.Wang, Y.Fan, K.Dang, et al. Qwen2. 5-omni technical report. arXiv preprint arXiv:2503.20215, 2025. 
*   (118) A.Yang, B.Yang, B.Zhang, B.Hui, B.Zheng, B.Yu, C.Li, D.Liu, F.Huang, H.Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024. 
*   (119) Y.Yuan, D.Jia, X.Zhuang, Y.Chen, Z.Chen, Y.Wang, Y.Wang, X.Liu, X.Kang, M.D. Plumbley, et al. Sound-vecaps: Improving audio generation with visually enhanced captions. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2025. 
*   (120) Y.Zang, S.O’Brien, T.Berg-Kirkpatrick, J.McAuley, and Z.Novack. Are you really listening? boosting perceptual awareness in music-qa benchmarks. arXiv preprint arXiv:2504.00369, 2025. 
*   (121) Z.Zhao, Y.Jiang, H.Liu, Y.Wang, and Y.Wang. Librisqa: Advancing free-form and open-ended spoken question answering with a novel dataset and framework. arXiv preprint arXiv:2308.10390, 2023. 
*   (122) C.Zhou, P.Liu, P.Xu, S.Iyer, J.Sun, Y.Mao, X.Ma, A.Efrat, P.Yu, L.Yu, et al. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36:55006–55021, 2023. 

Appendix
--------

Appendix A AF-Whisper
---------------------

### A.1 Training Details

We train AF-Whisper on 128 NVIDIA A100 80GB GPUs. During training, we use an effective batch size of 1024, the AdamW optimizer (learning rate = 10−4 10^{-4}10 start_POSTSUPERSCRIPT - 4 end_POSTSUPERSCRIPT, weight decay = 0.1), and train using fp16 precision. We train for 5 epochs on the complete dataset and sample instances randomly from the entire pool for each batch.

### A.2 Training Datasets

Table[5](https://arxiv.org/html/2507.08128v2#A1.T5 "Table 5 ‣ A.2 Training Datasets ‣ Appendix A AF-Whisper ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") lists the datasets used to train AF-Whisper. For each dataset, we follow the same process outlined in Section 3 of the main paper: generating transcripts, spoken language characteristics, and audio captions. When available, we incorporate gold-standard metadata for these elements (for e.g., transcripts for LibriSpeech or captions for AudioCaps). GPT-4.1 is prompted to produce the final caption using a format similar to Fig.[50](https://arxiv.org/html/2507.08128v2#A10.F50 "Figure 50 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), with a modified exemplar. For extracting spoken language characteristics using AF2, we use the following prompt: “There is a human speaking in the audio. Describe in detail the characteristics of the spoken utterance, including pitch, emotion, mood, speed, and other speech dynamics.”

Table 5: Statistics of audio-caption datasets used for AF-Whisper training.

| Dataset | #Audio-Text Pairs |
| --- |
| GigaSpeech (L)[[14](https://arxiv.org/html/2507.08128v2#bib.bib14)] | 2,266,371 |
| Speech-in-Sound Captions∗[[2](https://arxiv.org/html/2507.08128v2#bib.bib2)] | 1,999,959 |
| SPGISpeech[[90](https://arxiv.org/html/2507.08128v2#bib.bib90)] | 1,966,109 |
| Sound-VECaps[[119](https://arxiv.org/html/2507.08128v2#bib.bib119)] | 1,657,029 |
| Million Songs Dataset[[9](https://arxiv.org/html/2507.08128v2#bib.bib9)] | 1,169,997 |
| Common Voice 15[[5](https://arxiv.org/html/2507.08128v2#bib.bib5)] | 1,109,689 |
| MiraData[[58](https://arxiv.org/html/2507.08128v2#bib.bib58)] | 748,320 |
| Action2sound∗[[13](https://arxiv.org/html/2507.08128v2#bib.bib13)] | 306,602 |
| NSynth[[34](https://arxiv.org/html/2507.08128v2#bib.bib34)] | 289,205 |
| LibriSpeech[[92](https://arxiv.org/html/2507.08128v2#bib.bib92)] | 281,241 |
| Freesound[[37](https://arxiv.org/html/2507.08128v2#bib.bib37)] | 256,695 |
| AudioSet Strong∗[[50](https://arxiv.org/html/2507.08128v2#bib.bib50)] | 216,622 |
| VGGSound[[15](https://arxiv.org/html/2507.08128v2#bib.bib15)] | 185,161 |
| VoxPopuli (en)[[107](https://arxiv.org/html/2507.08128v2#bib.bib107)] | 177,019 |
| FMA[[25](https://arxiv.org/html/2507.08128v2#bib.bib25)] | 106,412 |
| Video Recap[[55](https://arxiv.org/html/2507.08128v2#bib.bib55)] | 64,627 |
| CochlScene[[57](https://arxiv.org/html/2507.08128v2#bib.bib57)] | 60,855 |
| Music4All[[102](https://arxiv.org/html/2507.08128v2#bib.bib102)] | 109269 |
| Switchboard[[43](https://arxiv.org/html/2507.08128v2#bib.bib43)] | 76,652 |
| FSD50k[[36](https://arxiv.org/html/2507.08128v2#bib.bib36)] | 40,966 |
| MACS[[87](https://arxiv.org/html/2507.08128v2#bib.bib87)] | 31,675 |
| BBC 2 2 2[https://sound-effects.bbcrewind.co.uk/](https://sound-effects.bbcrewind.co.uk/) | 31,201 |
| MagnaTagATune[[67](https://arxiv.org/html/2507.08128v2#bib.bib67)] | 25,863 |
| SoundDescs[[63](https://arxiv.org/html/2507.08128v2#bib.bib63)] | 23,085 |
| Clotho[[32](https://arxiv.org/html/2507.08128v2#bib.bib32)] | 19,195 |
| TAU-Urban[[85](https://arxiv.org/html/2507.08128v2#bib.bib85)] | 14,400 |
| MusicCaps[[3](https://arxiv.org/html/2507.08128v2#bib.bib3)] | 5,479 |
| WavText5K[[28](https://arxiv.org/html/2507.08128v2#bib.bib28)] | 4,347 |
| SONICS[[98](https://arxiv.org/html/2507.08128v2#bib.bib98)] | 1,602 |
| SoundBible 3 3 3[https://soundbible.com/](https://soundbible.com/) | 935 |
| MUSDB18[[97](https://arxiv.org/html/2507.08128v2#bib.bib97)] | 276 |
| Medleydb-Pitch[[10](https://arxiv.org/html/2507.08128v2#bib.bib10)] | 103 |
| Total | 13,246,961 |

Appendix B AudioSkills-XL
-------------------------

Table[6](https://arxiv.org/html/2507.08128v2#A2.T6 "Table 6 ‣ Appendix B AudioSkills-XL ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") provides all details, including statistics and references to prompts we used for generating AudioSkills-XL.

Table 6: Detailed statistics of AudioSkills-XL, categorized into individual reasoning types, together with details on open-source datasets, additional meta-data, and prompts used for QA generation. * indicates that these types are further categorized into skills, and we elaborate on this in Section[B.1](https://arxiv.org/html/2507.08128v2#A2.SS1 "B.1 Skill-Wise Breakdown ‣ Appendix B AudioSkills-XL ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"). Rows not grayed out are the contributions of this paper. Speech QA types are the same as LongAudio-XL and explained in Section 4.2, with examples in Figure 3 and more examples in Appendix[C](https://arxiv.org/html/2507.08128v2#A3 "Appendix C LongAudio-XL ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models").

| Question Type | Size | Datasets Used | Meta-Data Used | Prompt Reference |
| --- | --- | --- | --- | --- |
| Temporal | 188K | Table 14 in[[40](https://arxiv.org/html/2507.08128v2#bib.bib40)] | Table 14 in[[40](https://arxiv.org/html/2507.08128v2#bib.bib40)] | Table 14 in[[40](https://arxiv.org/html/2507.08128v2#bib.bib40)] |
| + ours | 350K | Synthetic Data | - | pythonic |
| Attribute Identification | 201K | Table 14 in[[40](https://arxiv.org/html/2507.08128v2#bib.bib40)] | Table 14 in[[40](https://arxiv.org/html/2507.08128v2#bib.bib40)] | Table 14 in[[40](https://arxiv.org/html/2507.08128v2#bib.bib40)] |
| Counting | 50K | Table 14 in[[40](https://arxiv.org/html/2507.08128v2#bib.bib40)] | Table 14 in[[40](https://arxiv.org/html/2507.08128v2#bib.bib40)] | Table 14 in[[40](https://arxiv.org/html/2507.08128v2#bib.bib40)] |
| Contextual Sound Event Reasoning | 982K | Table 14 in[[40](https://arxiv.org/html/2507.08128v2#bib.bib40)] | Table 14 in[[40](https://arxiv.org/html/2507.08128v2#bib.bib40)] | Table 14 in[[40](https://arxiv.org/html/2507.08128v2#bib.bib40)] |
| Contextual Speech Event Reasoning | 1,272K | Table 14 in[[40](https://arxiv.org/html/2507.08128v2#bib.bib40)] | Table 14 in[[40](https://arxiv.org/html/2507.08128v2#bib.bib40)] | Table 14 in[[40](https://arxiv.org/html/2507.08128v2#bib.bib40)] |
| Information Extraction | 858K | Table 14 in[[40](https://arxiv.org/html/2507.08128v2#bib.bib40)] | Table 14 in[[40](https://arxiv.org/html/2507.08128v2#bib.bib40)] | Table 14 in[[40](https://arxiv.org/html/2507.08128v2#bib.bib40)] |
| General Reasoning | 704K | Table 14 in[[40](https://arxiv.org/html/2507.08128v2#bib.bib40)] | Table 14 in[[40](https://arxiv.org/html/2507.08128v2#bib.bib40)] | Table 14 in[[40](https://arxiv.org/html/2507.08128v2#bib.bib40)] |
| + ours (only sound) | 300K | YouTube8M | caption | Fig.[34](https://arxiv.org/html/2507.08128v2#A10.F34 "Figure 34 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") |
| Sound Reasoning* (ours) | 300K | YouTube8M | caption | Fig.[49](https://arxiv.org/html/2507.08128v2#A10.F49 "Figure 49 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"),[48](https://arxiv.org/html/2507.08128v2#A10.F48 "Figure 48 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"),[47](https://arxiv.org/html/2507.08128v2#A10.F47 "Figure 47 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") |
| Music Knowledge* (ours) | 1,000K | MusicBench, Music4All, MSD | captions, dataset-specific meta-data | Fig.[31](https://arxiv.org/html/2507.08128v2#A10.F31 "Figure 31 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"),[28](https://arxiv.org/html/2507.08128v2#A10.F28 "Figure 28 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") |
| Music Reasoning* (ours) | 1,000K | MusicBench, Music4All, MSD | captions, dataset-specific meta-data | Fig.[29](https://arxiv.org/html/2507.08128v2#A10.F29 "Figure 29 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"),[32](https://arxiv.org/html/2507.08128v2#A10.F32 "Figure 32 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"),[33](https://arxiv.org/html/2507.08128v2#A10.F33 "Figure 33 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") |
| Speech-in-Sound QA (ours) | 1,739K | Speech-in-Sound Caps (YouTube8M) | Caption, Transcripts, Speech Characteristics | Fig.[26](https://arxiv.org/html/2507.08128v2#A10.F26 "Figure 26 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")[50](https://arxiv.org/html/2507.08128v2#A10.F50 "Figure 50 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") |
| Speech QA* (ours) | 200K | LibriSpeech, GigaSpeech, VoxCeleb2 | Transcripts | Fig.[19](https://arxiv.org/html/2507.08128v2#A10.F19 "Figure 19 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"),[18](https://arxiv.org/html/2507.08128v2#A10.F18 "Figure 18 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"),[16](https://arxiv.org/html/2507.08128v2#A10.F16 "Figure 16 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"),[25](https://arxiv.org/html/2507.08128v2#A10.F25 "Figure 25 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") |

### B.1 Skill-Wise Breakdown

#### B.1.1 Music Reasoning

Genre and Style: Focuses on the model’s ability to infer musical genre or stylistic influences by analyzing instrumentation, arrangement, and production characteristics.

Mood and Expression: Focuses on how well the model interprets the emotional tone or affective content conveyed by the music, such as melancholy, uplifting, or aggressive moods.

Temporal Relations Between Elements: Focuses on the model’s understanding of structural evolution within the music over time, including transitions in energy, tempo, or instrumentation across different sections.

Functional Context: Focuses on the model to link the music with real-world settings or usage contexts (e.g., movie scenes, events), requiring understanding of appropriateness and intent.

Lyrics: Focuses on interpretation of lyrical themes and content where applicable, often demanding a blend of semantic understanding and musical context awareness.

Historical and Cultural Context: Focuses on whether the model can connect musical elements to their broader cultural or historical origins (e.g., jazz fusion, protest music), relying on external world knowledge.

Music Texture: Focuses on knowledge of the audio’s timbral and sonic character by evaluating aspects such as the layering of instruments, vocal texture, and overall audio quality. This skill captures how dense, sparse, smooth, or gritty a piece sounds, requiring models to interpret descriptive attributes and production characteristics.

Melody: Focuses on understanding the primary musical contour or thematic tune in the audio. Melody-based QAs evaluate recognition of pitch movement, vocal/instrumental phrasing, and stylistic traits such as ornamentation or melodic structure, encouraging indirect inference over simple labeling.

Rhythm and Tempo: Focuses on the temporal structure of the music, including pulse, beat, speed, and time signature. These questions test whether the model can identify rhythmic complexity, tempo changes, and groove characteristics that define a track’s pacing or drive.

Harmony and Chords: Focuses on the models’ ability to reason about harmonic progressions and chordal structures that shape the emotional and tonal qualities of the audio. This includes interpreting transitions, key relationships, and compositional patterns in harmony using indirect reasoning from musical cues.

General Complex Reasoning QA: Evaluates the model’s ability to perform multi-dimensional inference on short music segments by combining musical knowledge, perceptual cues, and contextual understanding. These questions are grounded in rich musical attributes, such as dynamics, structure, genre fusion, narrative cues, emotional evolution, and historical style, and require the model to synthesize diverse information to arrive at the correct answer. This category tests higher-order music comprehension across expressive, structural, technical, and cultural dimensions, aiming to emulate how humans make sense of music beyond surface-level tagging.

#### B.1.2 Music Knowledge

Instrumentation: Focuses on the model’s ability to recognize the instruments used in the music and how their timbre, arrangement, or presence contributes to the overall sound and suitability for various contexts.

Performance: Focuses on understanding of the vocal or instrumental delivery, including vocal tone, articulation, expression, or the presence of unique performance techniques.

Sound Texture: Focuses on the density and layering of sound, such as sparse vs. rich textures, acoustic vs. electronic timbres, and how these contribute to the sonic identity of the piece.

Metre and Rhythm: Focuses on the temporal structure of the piece, including rhythmic patterns, tempo consistency or variation, and the use of syncopation or groove, which are essential for identifying genre or compositional style.

Melody: Focuses on how the model interprets the musical contour and phrasing of the primary tune, including vocal stylings, tonal range, and melodic progression.

Dynamics and Expression: Focuses on the model’s sensitivity to dynamic shifts (e.g., soft to loud passages), expressive techniques, and emotional delivery throughout the performance.

Harmony: Focuses on the model’s ability to recognize chord progressions, harmonic structure, and tonal relationships, which contribute to the music’s emotional or stylistic impact.

![Image 5: Refer to caption](https://arxiv.org/html/x5.png)

Figure 4: Examples of Music Reasoning and Knowledge Questions from AudioSkills-XL. Additionally, we also illustrate examples of music captions generated for audios in Music4All by prompting GPT-4.1 with metadata obtained from the dataset.

#### B.1.3 Sound Reasoning

Speech-in-Sound QA: Focuses on reasoning over spoken content in addition to ambient sounds or music to answer complex questions about the input audio, including scene interpretation, action reasoning, etc.

Eco-Acoustic Sounds QA: Focuses on the model’s ability to interpret natural environmental conditions based on ambient audio cues. This includes reasoning over weather phenomena such as thunderstorms, snowfall, or rain using non-speech acoustic indicators like wind, water, or animal sounds.

Acoustic Scene Reasoning: Evaluates the model’s capability to infer real-world environments from ambient and structural sound patterns. These include background music, reverberation, crowd noise, and electronic elements, enabling scene classification (e.g., arcade, mall, theater) from complex audio mixes.

Sound-Based Event Reasoning: Focuses on identifying and reasoning over specific audio features or events, such as musical motifs, instrument timbres, or recurring sonic patterns, to infer event types or characteristic actions.

![Image 6: Refer to caption](https://arxiv.org/html/x6.png)

Figure 5: Examples of Sound Reasoning QA, together with the metadata used for generating them.

![Image 7: Refer to caption](https://arxiv.org/html/x7.png)

Figure 6: Examples of Speech-in-Sound Caps and QA, together with the metadata used for generating them.

![Image 8: Refer to caption](https://arxiv.org/html/x8.png)

Figure 7: Examples of general audio QA generated as part of AudioSkills. We generate this as we find models struggle to say a “No” while responding to questions.

Appendix C LongAudio-XL
-----------------------

Tables[9](https://arxiv.org/html/2507.08128v2#A5.T9 "Table 9 ‣ E.2 Clustering for constructing AF-Chat ‣ Appendix E AF-Chat ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") and[10](https://arxiv.org/html/2507.08128v2#A5.T10 "Table 10 ‣ E.2 Clustering for constructing AF-Chat ‣ Appendix E AF-Chat ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") present detailed skill-wise statistics for LongAudio-XL, including the source datasets and the minimum, maximum, and average durations of the audio samples.

Below, we also show some examples form LongAudio-Xl in Fig[8](https://arxiv.org/html/2507.08128v2#A3.F8 "Figure 8 ‣ Appendix C LongAudio-XL ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")

![Image 9: Refer to caption](https://arxiv.org/html/x9.png)

Figure 8: Examples of LongAudio-XL.

Appendix D AF-Think
-------------------

Table[7](https://arxiv.org/html/2507.08128v2#A4.T7 "Table 7 ‣ Appendix D AF-Think ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") provides all details, including statistics and prompts for generating AF-Think.

Table 7: Detailed statistics of AF-Think. Most speech QA examples in this benchmark involve reasoning about ambient sounds in addition to spoken content. As our analysis shows, this added requirement increases task complexity, necessitating deeper inference to answer questions accurately.

| Modality Type | Size | Datasets Used | Meta-Data Used | Prompt Reference |
| --- | --- | --- | --- | --- |
| Speech | 100K | Speech-in-Sound QA, LongAudio-XL | transcripts, generated QAs | Fig.[38](https://arxiv.org/html/2507.08128v2#A10.F38 "Figure 38 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"),[42](https://arxiv.org/html/2507.08128v2#A10.F42 "Figure 42 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") |
| Sound only | 50K | AudioSkills-XL (AudioSet-SL, Youtube8M) | captions, QAs, dataset-specific meta-data | Fig.[41](https://arxiv.org/html/2507.08128v2#A10.F41 "Figure 41 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"),[37](https://arxiv.org/html/2507.08128v2#A10.F37 "Figure 37 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") |
| Music | 100K | AudioSkills-XL (Music4All, MSD) | captions, QAs, dataset-specific meta-data | Fig.[39](https://arxiv.org/html/2507.08128v2#A10.F39 "Figure 39 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"),[40](https://arxiv.org/html/2507.08128v2#A10.F40 "Figure 40 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") |

Below, we also provide several examples from AF-Think in Fig.[9](https://arxiv.org/html/2507.08128v2#A4.F9 "Figure 9 ‣ Appendix D AF-Think ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"):

![Image 10: Refer to caption](https://arxiv.org/html/x10.png)

Figure 9: Examples of AF-Think, for music, speech and sounds.

Appendix E AF-Chat
------------------

Table[8](https://arxiv.org/html/2507.08128v2#A5.T8 "Table 8 ‣ Appendix E AF-Chat ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") provides all details, including statistics and prompts for generating AF-Chat.

Table 8: Detailed statistics of AF-Chat.

| Modality Type | Size | Datasets Used | Meta-Data Used | Prompt Reference |
| --- | --- | --- | --- | --- |
| Sound & Speech | 35K | YouTube8M | captions, transcripts, speech characteristics | Fig.[35](https://arxiv.org/html/2507.08128v2#A10.F35 "Figure 35 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") |
| Music | 40K | Music4All, MSD | captions, dataset-specific meta-data | Fig.[36](https://arxiv.org/html/2507.08128v2#A10.F36 "Figure 36 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") |

Below, in Fig.[10](https://arxiv.org/html/2507.08128v2#A5.F10 "Figure 10 ‣ Appendix E AF-Chat ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") and [11](https://arxiv.org/html/2507.08128v2#A5.F11 "Figure 11 ‣ Appendix E AF-Chat ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") we also provide 2 examples from AF-Chat, one for each sound and music.

![Image 11: Refer to caption](https://arxiv.org/html/x11.png)

Figure 10: Example of AF-Chat for sound and speech.

![Image 12: Refer to caption](https://arxiv.org/html/x12.png)

Figure 11: Example of AF-Chat for music.

### E.1 Human Study for AF-Chat-test

The human verification process has been approved by our institution’s Institutional Review Board (IRB). For the human study, we hire 4 Ph.D. students proficient in audio research, including music. For each instance in each test-set dialogue, the students were asked to rate the output of the model on a scale of 1-5 across Factuality (how correct the response is), Usefulness (how useful the response is with respect to the context of the conversation), and Depth (how detailed the response is). For reference, we, the authors of the paper, provide responses scored 1-5 across the 3 aspects. The final score provided in Table 3 is an average of scores across all instances.

### E.2 Clustering for constructing AF-Chat

To construct high-quality multi-turn, multi-audio dialogues for AF-Chat, we implement a targeted clustering strategy that ensures each dialogue is grounded in a semantically diverse but coherent audio context. Rather than sampling audio clips at random, which often leads to incoherent or loosely connected conversations, we curate each dialogue from a controlled pool of semantically related audio samples.

Specifically, for each seed audio, we retrieve its top 8 most semantically similar and top 8 most dissimilar clips from the dataset. Similarity is computed using captions, NV-Embed-v2 embeddings of the captions, and FAISS-based similarity search[[31](https://arxiv.org/html/2507.08128v2#bib.bib31)] of the embeddings.

For speech and environmental sounds, we use clips from Speech-in-Sound Caps. For music, we source from Music4All and the Million Song Dataset (MSD). Once the 16-candidate pool is formed (8 similar + 8 dissimilar), we restrict the dialogue construction process to this subset. GPT-4.1 is then prompted to construct multi-turn conversations (up to 10 turns) using any combination of these audio clips. This ensures:

1.   1.Topical consistency across turns using similar clips. 
2.   2.Diversity and contrast through the inclusion of dissimilar audio. 
3.   3.Clear referential structure, as questions may depend on or refer back to earlier audio. 

![Image 13: Refer to caption](https://arxiv.org/html/x13.png)

Figure 12: Examples of audio clusters obtained after clustering (Section[E.1](https://arxiv.org/html/2507.08128v2#A5.SS1 "E.1 Human Study for AF-Chat-test ‣ Appendix E AF-Chat ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")), used for constructing AF-Chat.

Our clustering strategy was informed by a preliminary human study (participant details similar to Section[E.1](https://arxiv.org/html/2507.08128v2#A5.SS1 "E.1 Human Study for AF-Chat-test ‣ Appendix E AF-Chat ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models")), where participants engaged in multi-audio, multi-turn conversations with an LALM, focused on tasks such as sound design and music information retrieval. We observed that participants naturally gravitated toward using either highly similar or strongly contrasting audio clips within a dialogue. This behavioral insight motivated our use of similar and dissimilar audio clustering.

Empirically, this approach produced dialogues that were more natural, coherent, and diverse compared to those built from randomly selected audio pools. Moreover, AF3-Chat, when trained on this clustered dataset, outperformed the variant trained on randomly selected audio clips, both in terms of response relevance and conversational depth.

Table 9: Detailed skill-wise and dataset-wise statistics of LongAudio-XL.

| QA Type | Dataset | #Instances | Min Dur.(s) | Max Dur.(s) | Avg. Dur.(s) |
| --- | --- | --- |
| Order | VoxPopuli[[107](https://arxiv.org/html/2507.08128v2#bib.bib107)] | 16,926 | 1.87 | 294.80 | 89.55 |
|  | LibriSpeech[[92](https://arxiv.org/html/2507.08128v2#bib.bib92)] | 2,340 | 16.02 | 147.59 | 82.01 |
|  | MELD[[94](https://arxiv.org/html/2507.08128v2#bib.bib94)] | 4,135 | 1.06 | 108.01 | 30.42 |
|  | IEMOCAP[[11](https://arxiv.org/html/2507.08128v2#bib.bib11)] | 599 | 82.00 | 542.00 | 272.45 |
|  | EuroParl[[62](https://arxiv.org/html/2507.08128v2#bib.bib62)] | 11,885 | 2.59 | 176.14 | 69.34 |
|  | Fisher[[22](https://arxiv.org/html/2507.08128v2#bib.bib22)] | 25,962 | 33.34 | 240.00 | 136.84 |
|  | Switchboard[[43](https://arxiv.org/html/2507.08128v2#bib.bib43)] | 2,702 | 22.81 | 148.96 | 87.38 |
|  | MultiDialog[[85](https://arxiv.org/html/2507.08128v2#bib.bib85)] | 27,927 | 1.31 | 499.33 | 135.10 |
|  | VoxCeleb2[[21](https://arxiv.org/html/2507.08128v2#bib.bib21)] | 12,855 | 8.00 | 1273.60 | 71.12 |
| Emotion Ident. | IEMOCAP[[11](https://arxiv.org/html/2507.08128v2#bib.bib11)] | 300 | 82.00 | 542.00 | 272.22 |
|  | MELD[[94](https://arxiv.org/html/2507.08128v2#bib.bib94)] | 1,847 | 1.78 | 108.01 | 33.20 |
| Emotion Causal Reason. | IEMOCAP[[11](https://arxiv.org/html/2507.08128v2#bib.bib11)] | 300 | 82.00 | 542.00 | 272.22 |
|  | MELD[[94](https://arxiv.org/html/2507.08128v2#bib.bib94)] | 1,850 | 1.57 | 108.01 | 33.13 |
| Emotion Flip Reason. | IEMOCAP[[11](https://arxiv.org/html/2507.08128v2#bib.bib11)] | 299 | 82.00 | 542.00 | 272.62 |
|  | MELD[[94](https://arxiv.org/html/2507.08128v2#bib.bib94)] | 1,807 | 1.64 | 108.01 | 33.57 |
| Topic Relation. Reason. | VoxPopuli[[107](https://arxiv.org/html/2507.08128v2#bib.bib107)] | 13,651 | 3.58 | 240.44 | 97.14 |
|  | LibriSpeech[[92](https://arxiv.org/html/2507.08128v2#bib.bib92)] | 1,165 | 16.02 | 147.59 | 82.11 |
|  | MELD[[94](https://arxiv.org/html/2507.08128v2#bib.bib94)] | 1,518 | 1.89 | 108.01 | 34.45 |
|  | IEMOCAP[[11](https://arxiv.org/html/2507.08128v2#bib.bib11)] | 188 | 82.00 | 542.00 | 270.99 |
|  | EuroParl[[62](https://arxiv.org/html/2507.08128v2#bib.bib62)] | 9,381 | 7.97 | 176.14 | 70.14 |
|  | Fisher[[22](https://arxiv.org/html/2507.08128v2#bib.bib22)] | 20,453 | 33.34 | 240.00 | 136.10 |
|  | Switchboard[[43](https://arxiv.org/html/2507.08128v2#bib.bib43)] | 998 | 24.58 | 148.96 | 90.05 |
|  | MultiDialog[[85](https://arxiv.org/html/2507.08128v2#bib.bib85)] | 14,906 | 5.11 | 499.33 | 135.35 |
|  | DailyTalk[[71](https://arxiv.org/html/2507.08128v2#bib.bib71)] | 3,141 | 8.05 | 103.66 | 35.32 |
|  | VoxCeleb2[[21](https://arxiv.org/html/2507.08128v2#bib.bib21)] | 5,414 | 8.51 | 1193.60 | 78.96 |
| Sarcasm Ident. | IEMOCAP[[11](https://arxiv.org/html/2507.08128v2#bib.bib11)] | 299 | 82.00 | 542.00 | 271.58 |
|  | MELD[[94](https://arxiv.org/html/2507.08128v2#bib.bib94)] | 1,958 | 1.10 | 108.01 | 31.82 |
| Summarization | VoxPopuli[[107](https://arxiv.org/html/2507.08128v2#bib.bib107)] | 13,913 | 2.12 | 294.80 | 91.38 |
|  | LibriSpeech[[92](https://arxiv.org/html/2507.08128v2#bib.bib92)] | 1,057 | 16.02 | 147.59 | 83.15 |
|  | MELD[[94](https://arxiv.org/html/2507.08128v2#bib.bib94)] | 2,803 | 1.84 | 108.01 | 32.92 |
|  | IEMOCAP[[11](https://arxiv.org/html/2507.08128v2#bib.bib11)] | 300 | 82.00 | 542.00 | 272.22 |
|  | EuroParl[[62](https://arxiv.org/html/2507.08128v2#bib.bib62)] | 8,905 | 6.62 | 176.14 | 70.03 |
|  | Fisher[[22](https://arxiv.org/html/2507.08128v2#bib.bib22)] | 15,500 | 0.33 | 240.00 | 135.60 |
|  | Switchboard[[43](https://arxiv.org/html/2507.08128v2#bib.bib43)] | 1,346 | 24.58 | 148.96 | 87.60 |
|  | MultiDialog[[85](https://arxiv.org/html/2507.08128v2#bib.bib85)] | 20,838 | 1.93 | 499.33 | 135.73 |
|  | DailyTalk[[71](https://arxiv.org/html/2507.08128v2#bib.bib71)] | 7,218 | 8.05 | 103.66 | 31.42 |
|  | VoxCeleb2[[21](https://arxiv.org/html/2507.08128v2#bib.bib21)] | 5,894 | 7.94 | 1193.60 | 70.87 |
|  | Spotify Podcasts[[23](https://arxiv.org/html/2507.08128v2#bib.bib23)] | 103920 | 0.06 | 18206.44 | 2002.99 |
| Needle QA (IE) | DailyTalk[[71](https://arxiv.org/html/2507.08128v2#bib.bib71)] | 13,563 | 5.72 | 103.66 | 31.12 |
|  | EuroParl[[62](https://arxiv.org/html/2507.08128v2#bib.bib62)] | 18,426 | 6.57 | 176.14 | 70.10 |
|  | Fisher[[22](https://arxiv.org/html/2507.08128v2#bib.bib22)] | 37,779 | 18.59 | 240.00 | 135.99 |
|  | IEMOCAP[[11](https://arxiv.org/html/2507.08128v2#bib.bib11)] | 542 | 82.00 | 542.00 | 272.03 |
|  | LibriSpeech[[92](https://arxiv.org/html/2507.08128v2#bib.bib92)] | 2,248 | 16.02 | 147.59 | 82.82 |
|  | Spotify Podcasts[[23](https://arxiv.org/html/2507.08128v2#bib.bib23)] | 103920 | 0.06 | 18206.44 | 2002.99 |
| Response QA (IE) | VoxPopuli[[107](https://arxiv.org/html/2507.08128v2#bib.bib107)] | 13,913 | 2.12 | 294.80 | 91.38 |
|  | MELD[[94](https://arxiv.org/html/2507.08128v2#bib.bib94)] | 1,660 | 1.57 | 108.01 | 31.83 |
|  | IEMOCAP[[11](https://arxiv.org/html/2507.08128v2#bib.bib11)] | 177 | 82.00 | 542.00 | 272.52 |
|  | MultiDialog[[85](https://arxiv.org/html/2507.08128v2#bib.bib85)] | 13,505 | 1.95 | 499.33 | 135.40 |
|  | DailyTalk[[71](https://arxiv.org/html/2507.08128v2#bib.bib71)] | 4,516 | 5.72 | 103.66 | 30.91 |
|  | Switchboard[[43](https://arxiv.org/html/2507.08128v2#bib.bib43)] | 862 | 22.81 | 148.96 | 88.75 |

Table 10: Detailed skill-wise and dataset-wise statistics of LongAudio-XL.

| QA Type | Dataset | #Instances | Min Dur. (s) | Max Dur.(s) | Avg. Dur.(s) |
| --- | --- | --- | --- | --- | --- |
| Causal QA (IE) | VoxPopuli[[107](https://arxiv.org/html/2507.08128v2#bib.bib107)] | 12,264 | 4.10 | 240.44 | 92.88 |
|  | LibriSpeech[[92](https://arxiv.org/html/2507.08128v2#bib.bib92)] | 1,166 | 16.02 | 147.59 | 82.04 |
|  | MELD[[94](https://arxiv.org/html/2507.08128v2#bib.bib94)] | 2,957 | 1.27 | 108.01 | 31.74 |
|  | IEMOCAP[[11](https://arxiv.org/html/2507.08128v2#bib.bib11)] | 298 | 82.00 | 542.00 | 273.10 |
|  | EuroParl[[62](https://arxiv.org/html/2507.08128v2#bib.bib62)] | 7,457 | 7.97 | 176.14 | 70.24 |
|  | Fisher[[22](https://arxiv.org/html/2507.08128v2#bib.bib22)] | 19,335 | 37.17 | 240.00 | 135.87 |
|  | Switchboard[[43](https://arxiv.org/html/2507.08128v2#bib.bib43)] | 1,352 | 22.81 | 148.96 | 87.40 |
|  | MultiDialog[[85](https://arxiv.org/html/2507.08128v2#bib.bib85)] | 20,811 | 3.17 | 499.33 | 135.62 |
|  | DailyTalk[[71](https://arxiv.org/html/2507.08128v2#bib.bib71)] | 7,368 | 8.05 | 103.66 | 31.15 |
|  | VoxCeleb2[[21](https://arxiv.org/html/2507.08128v2#bib.bib21)] | 6,171 | 8.06 | 1193.60 | 71.08 |

Appendix F Prompts
------------------

We provide all prompting templates used across our datasets and QA types in Figures[15](https://arxiv.org/html/2507.08128v2#A10.F15 "Figure 15 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [16](https://arxiv.org/html/2507.08128v2#A10.F16 "Figure 16 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [17](https://arxiv.org/html/2507.08128v2#A10.F17 "Figure 17 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [18](https://arxiv.org/html/2507.08128v2#A10.F18 "Figure 18 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [19](https://arxiv.org/html/2507.08128v2#A10.F19 "Figure 19 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [20](https://arxiv.org/html/2507.08128v2#A10.F20 "Figure 20 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [21](https://arxiv.org/html/2507.08128v2#A10.F21 "Figure 21 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [22](https://arxiv.org/html/2507.08128v2#A10.F22 "Figure 22 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [23](https://arxiv.org/html/2507.08128v2#A10.F23 "Figure 23 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [24](https://arxiv.org/html/2507.08128v2#A10.F24 "Figure 24 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [25](https://arxiv.org/html/2507.08128v2#A10.F25 "Figure 25 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [26](https://arxiv.org/html/2507.08128v2#A10.F26 "Figure 26 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [27](https://arxiv.org/html/2507.08128v2#A10.F27 "Figure 27 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [28](https://arxiv.org/html/2507.08128v2#A10.F28 "Figure 28 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [29](https://arxiv.org/html/2507.08128v2#A10.F29 "Figure 29 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [30](https://arxiv.org/html/2507.08128v2#A10.F30 "Figure 30 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [31](https://arxiv.org/html/2507.08128v2#A10.F31 "Figure 31 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [32](https://arxiv.org/html/2507.08128v2#A10.F32 "Figure 32 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [33](https://arxiv.org/html/2507.08128v2#A10.F33 "Figure 33 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [34](https://arxiv.org/html/2507.08128v2#A10.F34 "Figure 34 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [35](https://arxiv.org/html/2507.08128v2#A10.F35 "Figure 35 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [36](https://arxiv.org/html/2507.08128v2#A10.F36 "Figure 36 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [37](https://arxiv.org/html/2507.08128v2#A10.F37 "Figure 37 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [38](https://arxiv.org/html/2507.08128v2#A10.F38 "Figure 38 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [39](https://arxiv.org/html/2507.08128v2#A10.F39 "Figure 39 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [40](https://arxiv.org/html/2507.08128v2#A10.F40 "Figure 40 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [41](https://arxiv.org/html/2507.08128v2#A10.F41 "Figure 41 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [42](https://arxiv.org/html/2507.08128v2#A10.F42 "Figure 42 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [43](https://arxiv.org/html/2507.08128v2#A10.F43 "Figure 43 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [44](https://arxiv.org/html/2507.08128v2#A10.F44 "Figure 44 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [45](https://arxiv.org/html/2507.08128v2#A10.F45 "Figure 45 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [46](https://arxiv.org/html/2507.08128v2#A10.F46 "Figure 46 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [47](https://arxiv.org/html/2507.08128v2#A10.F47 "Figure 47 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [48](https://arxiv.org/html/2507.08128v2#A10.F48 "Figure 48 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), [49](https://arxiv.org/html/2507.08128v2#A10.F49 "Figure 49 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), and [50](https://arxiv.org/html/2507.08128v2#A10.F50 "Figure 50 ‣ Appendix J Qualitative Examples ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models").

Appendix G AF3 Training Datasets
--------------------------------

Table[11](https://arxiv.org/html/2507.08128v2#A7.T11 "Table 11 ‣ Appendix G AF3 Training Datasets ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models") summarizes all datasets used to train AF3, including total hours, number of audio-QA pairs, and the number of epochs (passes over the dataset) used at each training stage. Similar to [[40](https://arxiv.org/html/2507.08128v2#bib.bib40)], we convert all foundational datasets (captioning, classification, etc.) into QA formats, using the same set of prompts for each task mentioned in [[40](https://arxiv.org/html/2507.08128v2#bib.bib40)].

Table 11: List of fine pre-training and fine-tuning datasets together with their training composition.

Dataset Hours Num. Pairs St. 1 St. 2 St. 3 St. 3.5 St. 4
AudioSkills-XL (Uurs)-9700K-2.0 2.0-
LongAudioXL (Ours)-1000K 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0-
AF-Think (Ours)-250K 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 2.0 2.0 2.0-
AF-Chat (Ours)-75K----1.0 1.0 1.0
CompA-R[[42](https://arxiv.org/html/2507.08128v2#bib.bib42)]159 hrs 350k-2.0 2.0--
MusicBench[[84](https://arxiv.org/html/2507.08128v2#bib.bib84)]115.5 hrs 686k-1.0 1.0--
Mu-LLAMA[[78](https://arxiv.org/html/2507.08128v2#bib.bib78)]62.9 hrs 70k 1.0 2.0 2.0--
Salmonn AQA[[104](https://arxiv.org/html/2507.08128v2#bib.bib104)]800 hrs 270k-1.0 1.0--
ClothoAQA[[76](https://arxiv.org/html/2507.08128v2#bib.bib76)]7.4 hrs 9.7K-8.0 8.0 8.0 8.0 8.0 8.0--
OpenAQA[[46](https://arxiv.org/html/2507.08128v2#bib.bib46)]693.2 hrs 1959.8K-1.0 1.0 1.0 1.0 1.0 1.0--
Clotho-v2[[32](https://arxiv.org/html/2507.08128v2#bib.bib32)]24.0 hrs 19.2K 1.0 2.0 2.0 2.0 2.0 2.0 2.0--
MACS[[87](https://arxiv.org/html/2507.08128v2#bib.bib87)]10.9 hrs 17.3K-1.0 1.0 1.0 1.0 1.0 1.0--
FSD50k[[36](https://arxiv.org/html/2507.08128v2#bib.bib36)]80.8 hrs 41.0K 1.0 1.0 1.0 1.0 1.0 1.0 1.0--
CochlScene[[57](https://arxiv.org/html/2507.08128v2#bib.bib57)]169.0 hrs 60.9K-1.0 1.0 1.0 1.0 1.0 1.0--
NonSpeech 7k[[99](https://arxiv.org/html/2507.08128v2#bib.bib99)]6.2 hrs 6.3K-4.0 4.0 4.0 4.0 4.0 4.0--
Chime-home[[38](https://arxiv.org/html/2507.08128v2#bib.bib38)]5.0 hrs 4.5K-1.0 1.0 1.0 1.0 1.0 1.0--
Sonyc-UST[[12](https://arxiv.org/html/2507.08128v2#bib.bib12)]34.9 hrs 27.9K-1.0 1.0 1.0 1.0 1.0 1.0--
Emov-DB[[86](https://arxiv.org/html/2507.08128v2#bib.bib86)]7.8 hrs 6.8K-1.0 1.0 1.0 1.0 1.0 1.0--
JL-Corpus[[56](https://arxiv.org/html/2507.08128v2#bib.bib56)]1.4 hrs 2.4K-6.0 6.0 6.0 6.0 6.0 6.0--
Tess 1.6 hrs 2.8K-2.0 2.0 2.0 2.0 2.0 2.0-
OMGEmotion[[7](https://arxiv.org/html/2507.08128v2#bib.bib7)]3.0 hrs 1.7K-3.0 3.0 3.0 3.0 3.0 3.0--
MusicAVQA audio-only[[74](https://arxiv.org/html/2507.08128v2#bib.bib74)]77.1 hrs 5.7K-6.0 6.0 6.0 6.0 6.0 6.0--
MusicQA[[91](https://arxiv.org/html/2507.08128v2#bib.bib91)]62.9 hrs 70K-1.0 1.0 1.0 1.0 1.0 1.0--
LP-MusicCaps MSD[[30](https://arxiv.org/html/2507.08128v2#bib.bib30)]5805.7 hrs 1331.8K 1.0 1.0 1.0 1.0 1.0 1.0 1.0--
LP-MusicCaps MTT[[30](https://arxiv.org/html/2507.08128v2#bib.bib30)]126.4 hrs 46.9K 1.0 1.0 1.0 1.0 1.0 1.0 1.0--
LP-MusicCaps MC[[30](https://arxiv.org/html/2507.08128v2#bib.bib30)]7.4 hrs 7.9K 1.0 2.0 2.0 2.0 2.0 2.0 2.0--
MusicCaps[[3](https://arxiv.org/html/2507.08128v2#bib.bib3)]7.4 hrs 2.6K 1.0 6.0 6.0 6.0 6.0 6.0 6.0--
NSynth[[34](https://arxiv.org/html/2507.08128v2#bib.bib34)]321.3 hrs 289.2K-8.0 8.0 8.0 8.0 8.0 8.0--
MusDB-HQ[[97](https://arxiv.org/html/2507.08128v2#bib.bib97)]29.1 hrs 10.2K-2.0 2.0 2.0 2.0 2.0 2.0--
FMA[[25](https://arxiv.org/html/2507.08128v2#bib.bib25)]860.7 hrs 104.2K-1.0 1.0 1.0 1.0 1.0 1.0--
Laion630k BBCSoundEffects[[113](https://arxiv.org/html/2507.08128v2#bib.bib113)]456.9 hrs 15.1K 1.0 1.0 1.0-1.0 1.0 1.0--
Laion630k Freesound[[113](https://arxiv.org/html/2507.08128v2#bib.bib113)]2494.8 hrs 306.5K 1.0 1.0 1.0-1.0 1.0 1.0--
SoundDescs[[63](https://arxiv.org/html/2507.08128v2#bib.bib63)]749.7 hrs 23.1K 1.0 1.0 1.0-1.0 1.0 1.0--
WavCaps[[83](https://arxiv.org/html/2507.08128v2#bib.bib83)]3793.3 hrs 402.6 K 1.0 1.0 1.0-1.0 1.0 1.0--
AudioSet[[39](https://arxiv.org/html/2507.08128v2#bib.bib39)]2617.8 hrs 950.8K 1.0-1.0 1.0 1.0--
WavText5K[[28](https://arxiv.org/html/2507.08128v2#bib.bib28)]23.8 hrs 4.3K 1.0 1.0 1.0-1.0 1.0 1.0--
MSP-Podcast[[82](https://arxiv.org/html/2507.08128v2#bib.bib82)]73.9 hrs 45.1K 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0--
MELD[[94](https://arxiv.org/html/2507.08128v2#bib.bib94)]8.7 hrs 32.9K 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0--
MusicAVQA audio-visual[[74](https://arxiv.org/html/2507.08128v2#bib.bib74)]142.4 hrs 17.9K 1.0 1.0 1.0 6.0 6.0 6.0 6.0 6.0 6.0--
Music4All Captions (ours)910.5 hrs 55.6K 1.0 1.0 1.0-1.0 1.0 1.0--
MSD Captions (ours)15449.9 hrs 55.6K 1.0 1.0 1.0-1.0 1.0 1.0--
Speech-in-Sound Captions (ours)6227.6 hrs 1999959 1.0 1.0 1.0-1.0 1.0 1.0--
LibriSpeech[[92](https://arxiv.org/html/2507.08128v2#bib.bib92)]960 hrs 281.2K 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0--
Switchboard[[43](https://arxiv.org/html/2507.08128v2#bib.bib43)]109.9 hrs 76.6K 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0--
GigaSpeech (L)[[14](https://arxiv.org/html/2507.08128v2#bib.bib14)]2499.8 hrs 2266.3K 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0--
Common Voice 15[[5](https://arxiv.org/html/2507.08128v2#bib.bib5)]1752.1 hrs 1109.6K 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0--
VoxPopuli (en)[[107](https://arxiv.org/html/2507.08128v2#bib.bib107)]501.8 hrs 177K 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0--
TEDLIUM (en)[[49](https://arxiv.org/html/2507.08128v2#bib.bib49)]472.3 hrs 68K 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0--
SPGISpeech[[90](https://arxiv.org/html/2507.08128v2#bib.bib90)]4999.8 hrs 1966.1K 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0--
VoiceAssistant400K[[115](https://arxiv.org/html/2507.08128v2#bib.bib115)]684 hrs 470K----1.0 1.0 1.0

Appendix H AF3 Training Details
-------------------------------

In this section, we present the training settings of our models across all 5 stages, each with specific configurations. Details are in [Table˜12](https://arxiv.org/html/2507.08128v2#A8.T12 "In Appendix H AF3 Training Details ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models").

| Settings | Stage1 | Stage2 | Stage3 | Stage3.5 | Stage4 |
| --- |
| per device batch size | 64 | 16 | 4 | 4 | 2 |
| learning rate | 1e-3 | 2e-5 | 2e-5 | 5e-5 | 5e-5 |
| learning schedule | Cosine decay |
| warm up ratio | 0.03 |
| weight decay | 0.0 |
| epoch | 1 | 1 | 1 | 2 | 2 |
| bf16 | ✓ | ✓ | ✓ | ✓ | ✓ |
| grad accumulate | 1 | 2 | 4 | 4 | 8 |
| DeepSpeed stage | Zero3 |
| GPUs | 128×\times×A100 |

Table 12: Training settings across stages.

Appendix I Streaming TTS System Architecture and Training Details
-----------------------------------------------------------------

To enable voice output capabilities within our system, we incorporate a text-to-speech (TTS) module that operates on subword text tokens. For efficient and simplified streaming speech synthesis, our TTS module employs a decoder-only architecture.

![Image 14: Refer to caption](https://arxiv.org/html/x14.png)

(a)Streaming TTS system architecture.

![Image 15: Refer to caption](https://arxiv.org/html/x15.png)

(b)Iterative unmasking of RVQ audio tokens.

Figure 13: Streaming TTS is enabled by autoregressive audio token generation coupled with a neural audio codec decoder. (a) The streaming TTS system predicts audio tokens conditioned on incoming subword text tokens (e.g., from the main AF3 model) and the history of previously generated audio tokens; these audio tokens are then decoded into voice output by the neural audio codec. (b) The iterative audio token unmasking process relies on an MLP block. This block takes partially masked RVQ tokens and transformer decoder output as input, predicts a cumulative embedding vector, which is subsequently quantized into progressively more unmasked RVQ tokens.

As illustrated in Fig.[13](https://arxiv.org/html/2507.08128v2#A9.F13 "Figure 13 ‣ Appendix I Streaming TTS System Architecture and Training Details ‣ Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models"), the TTS module predicts the subsequent audio token conditioned on incoming subword text tokens from the main AF3 model and the history of previously generated audio tokens. These audio tokens are then decoded into voice output by the neural audio codec. This design simplifies the speech generation pipeline and minimizes latency, which are critical for real-time speech streaming.

### I.1 Neural Audio Codec

We utilize a fully causal convolutional neural audio codec for efficient streaming audio decoding, following [[61](https://arxiv.org/html/2507.08128v2#bib.bib61), [103](https://arxiv.org/html/2507.08128v2#bib.bib103)].

Encoder. Input audio is first resampled to 44.1 kHz. It is then converted into Short-Time Fourier Transform(STFT) parameters using a hop size of 8 and a window size of 32. This STFT representation is processed by an initial 1x1 convolutional layer to produce 384-dimensional hidden embeddings. Following this, the signal undergoes three downsampling stages. Each stage consists of three causal 1D-ConvNeXt blocks[[79](https://arxiv.org/html/2507.08128v2#bib.bib79), [103](https://arxiv.org/html/2507.08128v2#bib.bib103)] followed by a strided convolutional layer for downsampling. These strided convolutional layers use a stride and kernel size of 8. Each such layer doubles the hidden dimension, except for the final one, which produces a 512-dimensional output. The encoded output sequence is 4096 times shorter than the raw waveform, corresponding to approximately 10.8 frames per second.

Quantization. The encoded output is quantized into audio tokens using Residual Vector Quantization (RVQ)[[69](https://arxiv.org/html/2507.08128v2#bib.bib69), [66](https://arxiv.org/html/2507.08128v2#bib.bib66)]. The number of RVQ levels is set to 72.

Decoder. The decoder mirrors the encoder’s architecture symmetrically, employing 1D transposed convolutional layers for upsampling and causal 1D-ConvNeXt blocks. The final convolutional layer reconstructs the STFT parameters, which are then transformed back into a raw audio waveform via an inverse STFT(iSTFT) similar to Vocos[[103](https://arxiv.org/html/2507.08128v2#bib.bib103)].

Training. The codec is trained using a combination of adversarial training and a mel-spectrogram reconstruction loss, following methodologies from DAC[[66](https://arxiv.org/html/2507.08128v2#bib.bib66)].

### I.2 Text-to-Speech (TTS) Module

Architecture and Operation. The TTS module’s transformer decoder processes a sequence formed by concatenating subword text tokens (from the main AF3 model) and previously generated audio tokens. The resulting hidden states from the transformer serve as conditional input to a multi-layer perceptron(MLP) block. This MLP block then iteratively predicts progressively higher levels of the RVQ tokens, a technique inspired by[[61](https://arxiv.org/html/2507.08128v2#bib.bib61)]. In practice, we employ 4 iteration steps during inference. A key aspect contributing to the system’s simplicity and low latency is that the model is designed to generate an audio token whenever a text token is emitted by the AF3 model, without requiring explicit alignments between text and speech.

Training and Configuration. During training, the transformer decoder utilizes teacher-forcing with ground-truth audio tokens. The MLP block is trained to estimate the parameters of a mixture-of-gaussians distribution where the number of mixtures is 1024. The objective is to maximize the log-likelihood of predicting the cumulative RVQ token embedding, following[[61](https://arxiv.org/html/2507.08128v2#bib.bib61)]. The decoder-only transformer has a configuration similar to DiT-XL[[93](https://arxiv.org/html/2507.08128v2#bib.bib93), [70](https://arxiv.org/html/2507.08128v2#bib.bib70)]. The MLP block consists of 3 layers, totaling 644 M parameters.

### I.3 Training Data and Processing

The models are trained on a comprehensive dataset comprising LibriTTS-R[[64](https://arxiv.org/html/2507.08128v2#bib.bib64)], LibriHeavy[[59](https://arxiv.org/html/2507.08128v2#bib.bib59)], the English portion of Multilingual LibriSpeech[[95](https://arxiv.org/html/2507.08128v2#bib.bib95)], and proprietary datasets, totaling approximately 100,000 hours of audio. To support long-form speech generation, audio segments from the same speaker are randomly concatenated to create training samples with durations ranging from 1 to 120 seconds.

Appendix J Qualitative Examples
-------------------------------

![Image 16: Refer to caption](https://arxiv.org/html/x16.png)

Figure 14: Demonstration of AF3’s capabilities on an audio captioning task. We prompt AF3 with an unseen audio clip—extracted from the Superman 2025 trailer ([https://www.youtube.com/watch?v=2woCZg5QdVE](https://www.youtube.com/watch?v=2woCZg5QdVE))—captured in the wild. The model accurately identifies and describes background sounds, spoken content, speaker turns, and transcriptions, demonstrating strong audio understanding. Beyond this example, AF3 supports significantly more complex reasoning tasks. We invite readers to explore these capabilities via our public demo: [https://huggingface.co/spaces/nvidia/audio-flamingo-3](https://huggingface.co/spaces/nvidia/audio-flamingo-3).

![Image 17: Refer to caption](https://arxiv.org/html/x17.png)

Figure 15: Prompt used for generating Topic Relationship QA for LongAudioXL.

![Image 18: Refer to caption](https://arxiv.org/html/x18.png)

Figure 16: Prompt used for generating Needle QA (Information Extraction type) for LongAudioXL.

![Image 19: Refer to caption](https://arxiv.org/html/x19.png)

Figure 17: Prompt used for generating Topic QA (Information Extraction type) for LongAudioXL.

![Image 20: Refer to caption](https://arxiv.org/html/x20.png)

Figure 18: Prompt used for generating Order QA for LongAudioXL.

![Image 21: Refer to caption](https://arxiv.org/html/x21.png)

Figure 19: Prompt used for generating Causal QA for LongAudioXL.

![Image 22: Refer to caption](https://arxiv.org/html/x22.png)

Figure 20: Prompt used for generating Summarization QA (Summary QA) for LongAudioXL.

![Image 23: Refer to caption](https://arxiv.org/html/x23.png)

Figure 21: Prompt used for generating Emotion Flip QA (Emotional State Reasoning type) for LongAudioXL.

![Image 24: Refer to caption](https://arxiv.org/html/x24.png)

Figure 22: Prompt used for generating Causal Reasoning (Emotional State Reasoning type) for LongAudioXL.

![Image 25: Refer to caption](https://arxiv.org/html/x25.png)

Figure 23: Prompt used for generating Sarcasm Identification QA for LongAudioXL.

![Image 26: Refer to caption](https://arxiv.org/html/x26.png)

Figure 24: Prompt used for generating Identification QA (Emotional State Reasoning type) for LongAudioXL.

![Image 27: Refer to caption](https://arxiv.org/html/x27.png)

Figure 25: Prompt used for generating ResponseQA (Information Extraction type) for LongAudioXL.

![Image 28: Refer to caption](https://arxiv.org/html/x28.png)

Figure 26: Prompt used for generating Speech-in-Sound QA for AudioSkills-XL.

![Image 29: Refer to caption](https://arxiv.org/html/x29.png)

Figure 27: Prompt used for generating captions for Million Songs Dataset. Noisy captions for the prompt are generated using AF2.

![Image 30: Refer to caption](https://arxiv.org/html/x30.png)

Figure 28: Prompt used for generating Music Knowledge QA from Million Songs Dataset for AudioSkills-XL. Noisy captions for the prompt are generated using AF2.

![Image 31: Refer to caption](https://arxiv.org/html/x31.png)

Figure 29: Prompt used for generating Music Reasoning QA from Million Songs Dataset for AudioSkills-XL. Noisy captions for the prompt are generated using AF2.

![Image 32: Refer to caption](https://arxiv.org/html/x32.png)

Figure 30: Prompt used for generating captions for Music4All. Noisy captions for the prompt are generated using AF2.

![Image 33: Refer to caption](https://arxiv.org/html/x33.png)

Figure 31: Prompt used for generating Music Knowledge QA from Music4All for AudioSkills-XL. Noisy captions for the prompt are generated using AF2.

![Image 34: Refer to caption](https://arxiv.org/html/x34.png)

Figure 32: Prompt used for generating General Open-Ended Complex Reasoning QA for Music Reasoning QA from Music4All for AudioSkills-XL.

![Image 35: Refer to caption](https://arxiv.org/html/x35.png)

Figure 33: Prompt used for generating Music Reasoning QA from Music4All for AudioSkills-XL. Noisy captions for the prompt are generated using AF2.

![Image 36: Refer to caption](https://arxiv.org/html/x36.png)

Figure 34: Prompt used for generating Yes-No QA (part of General Reasoning+ours (only sound) from AudioSet-SL) for AudioSkills-XL.

![Image 37: Refer to caption](https://arxiv.org/html/x37.png)

Figure 35: Prompt used for generating multi-turn, multi-audio chat/dialogues (speech and sounds) for AF-Chat.

![Image 38: Refer to caption](https://arxiv.org/html/x38.png)

Figure 36: Prompt used for generating multi-turn, multi-audio chat/dialogues (music) for AF-Chat.

![Image 39: Refer to caption](https://arxiv.org/html/x39.png)

Figure 37: Prompt 1 used for generating CoT-style reasoning focused on speech and ambient sounds (input instances sampled from Speech-in-Sound Caps, which is curated using YouTube8M) for AF-Think.

![Image 40: Refer to caption](https://arxiv.org/html/x40.png)

Figure 38: Prompt 2 used for generating CoT-style reasoning focused on SpeechQAs (input instances randomly sampled from LongAudio-XL speech subset) for AF-Think.

![Image 41: Refer to caption](https://arxiv.org/html/x41.png)

Figure 39: Prompt 3 used for generating CoT-style reasoning focused on music (input instances sampled from our Music Knowledge and Reasoning subset of AudioSkills-XL) for AF-Think. This focuses on open-ended QA.

![Image 42: Refer to caption](https://arxiv.org/html/x42.png)

Figure 40: Prompt 4 used for generating CoT-style reasoning focused on music (input instances sampled from our Music Knowledge and Reasoning subset of AudioSkills-XL) for AF-Think. This focuses on MCQ-based QA.

![Image 43: Refer to caption](https://arxiv.org/html/x43.png)

Figure 41: Prompt 5 used for generating CoT-style reasoning focused on ambient sounds only (input instances sampled from our Sound Reasoning subset of AudioSkills-XL, which is curated from YouTube8M) for AF-Think. This focuses on MCQ-based QA.

![Image 44: Refer to caption](https://arxiv.org/html/x44.png)

Figure 42: Prompt 6 used for generating CoT-style reasoning focused on speech and ambient sounds (input instances sampled from Speech-in-Sound Caps, which is curated using YouTube8M) for AF-Think. This focuses on MCQ-based QA.

![Image 45: Refer to caption](https://arxiv.org/html/x45.png)

Figure 43: Prompt 1 used for Music Reasoning for AudioSkills-XL. The QAs are focused on music texture reasoning.

![Image 46: Refer to caption](https://arxiv.org/html/x46.png)

Figure 44: Prompt 2 used for Music Reasoning for AudioSkills-XL. The QAs are focused on melody reasoning.

![Image 47: Refer to caption](https://arxiv.org/html/x47.png)

Figure 45: Prompt 3 used for Music Reasoning for AudioSkills-XL. The QAs are focused on rhythm and tempo reasoning.

![Image 48: Refer to caption](https://arxiv.org/html/x48.png)

Figure 46: Prompt 4 used for Music Reasoning for AudioSkills-XL. The QAs are focused on harmony and chord reasoning.

![Image 49: Refer to caption](https://arxiv.org/html/x49.png)

Figure 47: Prompt 1 used for Sound Reasoning for AudioSkills-XL. The QAs are focused on - eco-acoustic sound reasoning.

![Image 50: Refer to caption](https://arxiv.org/html/x50.png)

Figure 48: Prompt 1 used for Sound Reasoning for AudioSkills-XL. The QAs are focused on -Acoustic Scene Reasoning.

![Image 51: Refer to caption](https://arxiv.org/html/x51.png)

Figure 49: Prompt 1 used for Sound Reasoning for AudioSkills-XL. The QAs are focused on -Sound-Based Event Reasoning.

![Image 52: Refer to caption](https://arxiv.org/html/x52.png)

Figure 50: Prompt used for generating Speech-in-Sound captions used in pre-training and further used in generating other QAs.

Generated on Mon Jul 28 22:54:54 2025 by [L a T e XML![Image 53: Mascot Sammy](blob:http://localhost/70e087b9e50c3aa663763c3075b0d6c5)](http://dlmf.nist.gov/LaTeXML/)

