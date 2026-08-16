Title: Qwen2-Audio Technical Report

URL Source: https://arxiv.org/html/2407.10759

Published Time: Tue, 16 Jul 2024 01:29:46 GMT

Markdown Content:
\doparttoc\faketableofcontents

Yunfei Chu∗† Jin Xu∗† Qian Yang∗ Haojie Wei 

Xipin Wei  Zhifang Guo  Yichong Leng  Yuanjun Lv  Jinzheng He 

Junyang Lin  Chang Zhou† Jingren Zhou 

 Qwen Team, Alibaba Group 
Code & Demo & Models: [https://github.com/QwenLM/Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio)

###### Abstract

We introduce the latest progress of Qwen-Audio, a large-scale audio-language model called Qwen2-Audio, which is capable of accepting various audio signal inputs and performing audio analysis or direct textual responses with regard to speech instructions. In contrast to complex hierarchical tags, we have simplified the pre-training process by utilizing natural language prompts for different data and tasks, and have further expanded the data volume. We have boosted the instruction-following capability of Qwen2-Audio and implemented two distinct audio interaction modes for voice chat and audio analysis. In the voice chat mode, users can freely engage in voice interactions with Qwen2-Audio without text input. In the audio analysis mode, users could provide audio and text instructions for analysis during the interaction. Note that we do not use any system prompts to switch between voice chat and audio analysis modes. Qwen2-Audio is capable of intelligently comprehending the content within audio and following voice commands to respond appropriately. For instance, in an audio segment that simultaneously contains sounds, multi-speaker conversations, and a voice command, Qwen2-Audio can directly understand the command and provide an interpretation and response to the audio. Additionally, DPO has optimized the model’s performance in terms of factuality and adherence to desired behavior. According to the evaluation results from AIR-Bench, Qwen2-Audio outperformed previous SOTAs, such as Gemini-1.5-pro, in tests focused on audio-centric instruction-following capabilities. Qwen2-Audio is open-sourced with the aim of fostering the advancement of the multi-modal language community.

††footnotetext: ∗Equal contribution, †Corresponding author![Image 1: Refer to caption](https://arxiv.org/html/2407.10759v1/x1.png)

Figure 1: Performance of Qwen2-Audio, Qwen-Audio and previous top-tiers from LALMs such as SpeechT5(Ao et al., [2021](https://arxiv.org/html/2407.10759v1#bib.bib2)), SpeechNet(Chen et al., [2021](https://arxiv.org/html/2407.10759v1#bib.bib6)), SpeechLLaMA(Wu et al., [2023a](https://arxiv.org/html/2407.10759v1#bib.bib35)), SALMONN(Tang et al., [2024](https://arxiv.org/html/2407.10759v1#bib.bib31)), Whisper(Radford et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib26)) Pengi(Deshmukh et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib11)), and SpeechVerse(Das et al., [2024](https://arxiv.org/html/2407.10759v1#bib.bib10)). We demonstrate the test set results across the 10 datasets covering Automatic Speech Recognition(ASR), Speech-to-Text Translation(S2TT), Speech Emotion Recognition(SER), Vocal Sound Classification(VSC), and instruction-following benchmark (Yang et al., [2024](https://arxiv.org/html/2407.10759v1#bib.bib37)). The results of ASR datasets, such as Librispeech and Aishell2 refer to 1 - WER%. The results of CoVoST2 is the average BLEU score of seven translation directions (en-de, de-en, en-zh, zh-en, es-en, fr-en and it-en). The results of the AIR-Bench chat benchmark encompass four dimensions: speech, sound, music, and mixed. Scores for each dimension are automatically assessed by GPT-4, with values ranging from 0 to 10. Qwen2-Audio achieves remarkable performance without requiring any task-specific fine-tuning, surpassing its counterparts.

Introduction
------------

Audio serves as a crucial medium for interaction and communication among humans and other living beings, carrying rich information content. A comprehensive understanding of various forms of audio signals is paramount to achieving Artificial General Intelligence (AGI). Recently, significant advancements have been made in the development of large audio-language models (LALMs)(Chu et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib7); Das et al., [2024](https://arxiv.org/html/2407.10759v1#bib.bib10); Kong et al., [2024](https://arxiv.org/html/2407.10759v1#bib.bib18); Tang et al., [2024](https://arxiv.org/html/2407.10759v1#bib.bib31); OpenAI, [2024](https://arxiv.org/html/2407.10759v1#bib.bib21)), demonstrating remarkable achievements in comprehending diverse speech signals, performing speech signal analysis, and complex reasoning.

In this report, we develop Qwen2-Audio, with a primary focus on enhancing its instruction-following capabilities. Qwen2-Audio is a Large Audio-Language Model (LALM) designed to process both audio and text inputs to generate textual outputs. Compared to previous models, Qwen2-Audio significantly scales up the training dataset. To reduce the gap between pre-training and post-training stages, we simplify the pre-training process by directly using natural language prompts for various data and tasks, as illustrated in figure[2](https://arxiv.org/html/2407.10759v1#S2.F2 "Figure 2 ‣ Methodology ‣ Qwen2-Audio Technical Report"). Following the practices in Large Language Models (LLMs)(OpenAI, [2023](https://arxiv.org/html/2407.10759v1#bib.bib20); Qwen, [2023](https://arxiv.org/html/2407.10759v1#bib.bib25)), we further conduct instruction tuning and direct preference optimization to align the model’s outputs with human preferences.

Qwen2-Audio operates in two distinct modes: Audio Analysis and Voice Chat. These two modes are differentiated by their functionality, but there is no need for users to distinguish between them during use. In the audio analysis mode, users can leverage Qwen2-Audio to analyze a diverse range of audio types, including speech, sound, music, or various mixed audio forms. Commands can be issued either through audio or text, and Qwen2-Audio will autonomously discern the command segments within the audio. Conversely, in voice chat mode, users can interact with Qwen2-Audio as if it were a conversational agent, engaging in unrestricted dialogue. Audio interaction is available, and users can switch to text interaction at any moment they choose. For instance, if a user inputs an audio clip where the initial part is the sound of typing on a keyboard, followed by the user asking "What is this sound?" in spoken language, Qwen2-Audio is expected to respond directly with "This is the sound of a keyboard."

As shown in Figure[1](https://arxiv.org/html/2407.10759v1#S0.F1 "Figure 1 ‣ Qwen2-Audio Technical Report"), extensive evaluation demonstrates that Qwen2-Audio, without any task-specific fine-tuning, outperforms previous LALMs across a diverse range of tasks. Among them, Qwen2-Audio achieves state-of-the-art performance on the test set of Aishell2, FLUERS-zh, VocalSound and AIR-Bench chat benchmark.

Methodology
-----------

![Image 2: Refer to caption](https://arxiv.org/html/2407.10759v1/x2.png)

Figure 2: The overview of three-stage training process of Qwen2-Audio.

#### Model Architecture

The training process of Qwen2-Audio is depicted in Figure[2](https://arxiv.org/html/2407.10759v1#S2.F2 "Figure 2 ‣ Methodology ‣ Qwen2-Audio Technical Report"), which contains an audio encoder and a large language model. Given the paired data (𝒂,𝒙)𝒂 𝒙(\bm{a},\bm{x})( bold_italic_a , bold_italic_x ), where the 𝒂 𝒂\bm{a}bold_italic_a and 𝒙 𝒙\bm{x}bold_italic_x denote the audio sequences and text sequences, the training objective is to maximize the next text token probability as

𝒫 θ⁢(x t|𝒙<t,Encoder ϕ⁢(𝒂)),subscript 𝒫 𝜃 conditional subscript 𝑥 𝑡 subscript 𝒙 absent 𝑡 subscript Encoder italic-ϕ 𝒂\mathcal{P}_{\theta}(x_{t}|\bm{x}_{<t},\text{Encoder}_{\phi}(\bm{a})),caligraphic_P start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( italic_x start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT | bold_italic_x start_POSTSUBSCRIPT < italic_t end_POSTSUBSCRIPT , Encoder start_POSTSUBSCRIPT italic_ϕ end_POSTSUBSCRIPT ( bold_italic_a ) ) ,(1)

conditioning on audio representations and previous text sequences 𝒙<t subscript 𝒙 absent 𝑡\bm{x}_{<t}bold_italic_x start_POSTSUBSCRIPT < italic_t end_POSTSUBSCRIPT, where θ 𝜃\theta italic_θ and ϕ italic-ϕ\phi italic_ϕ denote the trainable parameters of the LLM and audio encoder respectively.

Different from Qwen-Audio, the initialization of the audio encoder of Qwen2-Audio is based on the Whisper-large-v3 model(Radford et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib26)). To preprocess the audio data, we resamples it to a frequency of 16kHz and converts the raw waveform into 128-channel mel-spectrogram using a window size of 25ms and a hop size of 10ms. Additionally, a pooling layer with a stride of two is incorporated to reduce the length of the audio representation. As a result, each frame of the encoder output approximately corresponds to a 40ms segment of the original audio signal. Qwen2-Audio still incorporates the large language model Qwen-7B(Bai et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib4)) as its foundational component. The total parameters of Qwen2-Audio is 8.2B parameters.

#### Pre-training

At the pre-training stage, we replace the hierarchical tags(Chu et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib7)) with the natural language prompts. As shown in Figure[2](https://arxiv.org/html/2407.10759v1#S2.F2 "Figure 2 ‣ Methodology ‣ Qwen2-Audio Technical Report"). We find that using language prompts can improve better generalization ability and better instruction following ability.

![Image 3: Refer to caption](https://arxiv.org/html/2407.10759v1/extracted/5729934/images/pretrain_hours.png)

Figure 3: Statistics (hours) of pre-training dataset.

#### Supervised Fine-tuning

The thorough pretraining of Qwen2-Audio has equipped the model with a comprehensive understanding of audio content. Building upon this, we employ instruction-based fine-tuning techniques to improve the ability of the model to align with human intent, resulting in an interactive chat model. Our prelimilary study emphasizes the critical influence of the quality and complexity of SFT data on the model’s performance. Accordingly, a meticulously curated set of high-quality SFT data was collected, with rigorous quality control procedures implemented.

We consider two distinct modes for human interactions:

*   •Audio Analysis: In the audio analysis mode, users are afforded the flexibility to have Qwen2-Audio analyze a diverse array of audio. User instructions can be given either through audio or text. This mode is often used for offline analysis of audio files. 
*   •Voice Chat: In the voice chat mode, users are encouraged to engage in voice conversations with Qwen2-Audio, asking a wide range of questions. Please feel free to consider it your voice chat assistant. This mode is often used for online interaction with LALMs. 

For consistency and model uniformity, both interaction modes were jointly trained, thus users will not experience mode differentiation during use, nor is it necessary to switch between different modes using separate system prompts. The two modes are seamlessly integrated in actual use.

#### Direct Preference Optimization

We employ DPO(Rafailov et al., [2024](https://arxiv.org/html/2407.10759v1#bib.bib27)) to further optimize models to follow human preferences. By obtaining the dataset 𝒟 𝒟\mathcal{D}caligraphic_D with the triplet data (𝒙,𝒚 𝒘,𝒚 𝒍)𝒙 subscript 𝒚 𝒘 subscript 𝒚 𝒍(\bm{x},\bm{y_{w}},\bm{y_{l}})( bold_italic_x , bold_italic_y start_POSTSUBSCRIPT bold_italic_w end_POSTSUBSCRIPT , bold_italic_y start_POSTSUBSCRIPT bold_italic_l end_POSTSUBSCRIPT ), where 𝒙 𝒙\bm{x}bold_italic_x is the input sequence with input audio, and 𝒚 𝒘 subscript 𝒚 𝒘\bm{y_{w}}bold_italic_y start_POSTSUBSCRIPT bold_italic_w end_POSTSUBSCRIPT and 𝒚 𝒍 subscript 𝒚 𝒍\bm{y_{l}}bold_italic_y start_POSTSUBSCRIPT bold_italic_l end_POSTSUBSCRIPT are the human-annotated good and bad responses respectively, we optimize the model 𝒫 θ subscript 𝒫 𝜃\mathcal{P}_{\theta}caligraphic_P start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT as follows:

ℒ DPO⁢(𝒫 θ;𝒫 ref)=−𝔼(𝒙,𝒚 𝒘,𝒚 𝒍)∼𝒟⁢[log⁡σ⁢(β⁢log⁡𝒫 θ⁢(𝒚 𝒘∣𝒙)𝒫 ref⁢(𝒚 𝒘∣𝒙)−β⁢log⁡𝒫 θ⁢(𝒚 𝒍∣𝒙)𝒫 ref⁢(𝒚 𝒍∣𝒙))],subscript ℒ DPO subscript 𝒫 𝜃 subscript 𝒫 ref subscript 𝔼 similar-to 𝒙 subscript 𝒚 𝒘 subscript 𝒚 𝒍 𝒟 delimited-[]𝜎 𝛽 subscript 𝒫 𝜃 conditional subscript 𝒚 𝒘 𝒙 subscript 𝒫 ref conditional subscript 𝒚 𝒘 𝒙 𝛽 subscript 𝒫 𝜃 conditional subscript 𝒚 𝒍 𝒙 subscript 𝒫 ref conditional subscript 𝒚 𝒍 𝒙\mathcal{L}_{\text{DPO}}(\mathcal{P}_{\theta};\mathcal{P}_{\text{ref}})=-% \mathbb{E}_{(\bm{x},\bm{y_{w}},\bm{y_{l}})\sim\mathcal{D}}\left[\log\sigma% \left(\beta\log\frac{\mathcal{P}_{\theta}(\bm{y_{w}}\mid\bm{x})}{\mathcal{P}_{% \text{ref}}(\bm{y_{w}}\mid\bm{x})}-\beta\log\frac{\mathcal{P}_{\theta}(\bm{y_{% l}}\mid\bm{x})}{\mathcal{P}_{\text{ref}}(\bm{y_{l}}\mid\bm{x})}\right)\right],caligraphic_L start_POSTSUBSCRIPT DPO end_POSTSUBSCRIPT ( caligraphic_P start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ; caligraphic_P start_POSTSUBSCRIPT ref end_POSTSUBSCRIPT ) = - blackboard_E start_POSTSUBSCRIPT ( bold_italic_x , bold_italic_y start_POSTSUBSCRIPT bold_italic_w end_POSTSUBSCRIPT , bold_italic_y start_POSTSUBSCRIPT bold_italic_l end_POSTSUBSCRIPT ) ∼ caligraphic_D end_POSTSUBSCRIPT [ roman_log italic_σ ( italic_β roman_log divide start_ARG caligraphic_P start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( bold_italic_y start_POSTSUBSCRIPT bold_italic_w end_POSTSUBSCRIPT ∣ bold_italic_x ) end_ARG start_ARG caligraphic_P start_POSTSUBSCRIPT ref end_POSTSUBSCRIPT ( bold_italic_y start_POSTSUBSCRIPT bold_italic_w end_POSTSUBSCRIPT ∣ bold_italic_x ) end_ARG - italic_β roman_log divide start_ARG caligraphic_P start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( bold_italic_y start_POSTSUBSCRIPT bold_italic_l end_POSTSUBSCRIPT ∣ bold_italic_x ) end_ARG start_ARG caligraphic_P start_POSTSUBSCRIPT ref end_POSTSUBSCRIPT ( bold_italic_y start_POSTSUBSCRIPT bold_italic_l end_POSTSUBSCRIPT ∣ bold_italic_x ) end_ARG ) ] ,(2)

where 𝒫 ref subscript 𝒫 ref\mathcal{P}_{\text{ref}}caligraphic_P start_POSTSUBSCRIPT ref end_POSTSUBSCRIPT denotes the reference model initialized with 𝒫 θ subscript 𝒫 𝜃\mathcal{P}_{\theta}caligraphic_P start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT, σ 𝜎\sigma italic_σ represents sigmoid function and β 𝛽\beta italic_β is a hyperparameter. Figure[2](https://arxiv.org/html/2407.10759v1#S2.F2 "Figure 2 ‣ Methodology ‣ Qwen2-Audio Technical Report") illustrates the three-stage training process of Qwen2-Audio.

Table 1: Summary of Evaluation Benchmarks for Qwen2-Audio.

Task Description Dataset Split Metric
ASR Automatic Speech Recognition Fleurs(Conneau et al., [2022](https://arxiv.org/html/2407.10759v1#bib.bib9))dev | test WER
Aishell2(Du et al., [2018](https://arxiv.org/html/2407.10759v1#bib.bib13))test
Librispeech(Panayotov et al., [2015](https://arxiv.org/html/2407.10759v1#bib.bib22))dev | test
Common Voice(Ardila et al., [2020](https://arxiv.org/html/2407.10759v1#bib.bib3))dev | test
S2TT Speech-to-Text Translation CoVoST2(Wang et al., [2020](https://arxiv.org/html/2407.10759v1#bib.bib32))test BLEU 1 1 1 https://github.com/mjpost/sacrebleu(Papineni et al., [2002](https://arxiv.org/html/2407.10759v1#bib.bib23))
SER Speech Emotion Recognition Meld(Poria et al., [2019](https://arxiv.org/html/2407.10759v1#bib.bib24))test ACC
VSC Vocal Sound Classification VocalSound(Gong et al., [2022](https://arxiv.org/html/2407.10759v1#bib.bib16))test ACC
AIR-Bench(Yang et al., [2024](https://arxiv.org/html/2407.10759v1#bib.bib37))Chat-Benchmark-Speech Fisher(Cieri et al., [2004](https://arxiv.org/html/2407.10759v1#bib.bib8)) SpokenWOZ(Si et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib29)) IEMOCAP(Si et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib29)) Common voice(Ardila et al., [2020](https://arxiv.org/html/2407.10759v1#bib.bib3))dev | test GPT-4 Eval
Chat-Benchmark-Sound Clotho(Drossos et al., [2020](https://arxiv.org/html/2407.10759v1#bib.bib12))dev | test GPT-4 Eval
Chat-Benchmark-Music MusicCaps(Agostinelli et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib1))dev | test GPT-4 Eval
Chat-Benchmark-Mixed-Audio Common voice(Ardila et al., [2020](https://arxiv.org/html/2407.10759v1#bib.bib3)) AudioCaps(Kim et al., [2019](https://arxiv.org/html/2407.10759v1#bib.bib17)) MusicCaps(Agostinelli et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib1))dev | test GPT-4 Eval

Experiments
-----------

### 3.1 Evaluation

In practice, we have found that many previous test datasets are highly limited and cannot adequately reflect performance in real-world scenarios, such as some SLU (Spoken Language Understanding) and SER (Speech Emotion Recognition) datasets. Therefore, we mainly evaluated performance directly on AIR-Bench. We discovered that the scores from AIR-Bench align more closely with the actual user interaction experience. Meanwhile, in order to assess the universal understanding capabilities of Qwen2-Audio, as shown in Table[1](https://arxiv.org/html/2407.10759v1#S2.T1 "Table 1 ‣ Direct Preference Optimization ‣ Methodology ‣ Qwen2-Audio Technical Report"), we still perform a comprehensive evaluation that encompasses various tasks, namely Automatic Speech Recognition (ASR), Speech-to-Text Translation (S2TT), Speech Emotion Recognition (SER), Vocal Sound Classification (VSC). The evaluation is conducted across 13 datasets. The evaluation datasets are rigorously excluded from the training data to avoid data leakage. The models we compare include open-source models and callable APIs, such as Gemini.

Table 2: The results of Automatic Speech Recognition(ASR), Speech-to-Text Translation(S2TT), Speech Emotion Recognition(SER), Vocal Sound Classification(VSC), and AIR-Bench chat benchmark. Note that for Qwen2-Audio, the results for Fleurs are zero-shot, whereas the results for Common Voice are not zero-shot.

Task Dataset Model Performance
Metrics Results
ASR Librispeech dev-clean | dev-other |test-clean | test-other SpeechT5(Ao et al., [2021](https://arxiv.org/html/2407.10759v1#bib.bib2))WER↓↓\downarrow↓2.1 | 5.5 | 2.4 | 5.8
SpeechNet(Chen et al., [2021](https://arxiv.org/html/2407.10759v1#bib.bib6))- | - | 30.7 | -
SLM-FT(Wang et al., [2023b](https://arxiv.org/html/2407.10759v1#bib.bib34))- | - | 2.6 | 5.0
SALMONN(Tang et al., [2024](https://arxiv.org/html/2407.10759v1#bib.bib31))- | - | 2.1 | 4.9
SpeechVerse(Das et al., [2024](https://arxiv.org/html/2407.10759v1#bib.bib10))- | - | 2.1 | 4.4
Qwen-Audio(Chu et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib7))1.8 | 4.0 | 2.0 | 4.2
Qwen2-Audio 1.3 | 3.4 | 1.6 | 3.6
Common Voice 15 en | zh | yue | fr Whisper-large-v3(Radford et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib26))WER↓↓\downarrow↓9.3 | 12.8 | 10.9 | 10.8
Qwen2-Audio 8.6 | 6.9 | 5.9 | 9.6
Fleurs zh Whisper-large-v3(Radford et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib26))WER↓↓\downarrow↓7.7
Qwen2-Audio 7.5
Aishell2 Mic | iOS | Android MMSpeech-base(Zhou et al., [2022](https://arxiv.org/html/2407.10759v1#bib.bib39))WER↓↓\downarrow↓4.5 | 3.9 | 4.0
Paraformer-large(Gao et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib15))- | 2.9 | -
Qwen-Audio(Chu et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib7))3.3 | 3.1 | 3.3
Qwen2-Audio 3.0 | 3.0 | 2.9
S2TT CoVoST2 en-de | de-en |en-zh | zh-en SALMONN(Tang et al., [2024](https://arxiv.org/html/2407.10759v1#bib.bib31))BLEU↑↑\uparrow↑18.6 | - | 33.1 | -
SpeechLLaMA(Wu et al., [2023a](https://arxiv.org/html/2407.10759v1#bib.bib35))- | 27.1 | - | 12.3
BLSP(Wang et al., [2023a](https://arxiv.org/html/2407.10759v1#bib.bib33))14.1 | - | - | -
Qwen-Audio(Chu et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib7))25.1 | 33.9 | 41.5 | 15.7
Qwen2-Audio 29.9 | 35.2 | 45.2 | 24.4
CoVoST2 es-en | fr-en | it-en |SpeechLLaMA(Wu et al., [2023a](https://arxiv.org/html/2407.10759v1#bib.bib35))BLEU↑↑\uparrow↑27.9 | 25.2 | 25.9
Qwen-Audio(Chu et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib7))39.7 | 38.5 | 36.0
Qwen2-Audio 40.0 | 38.5 | 36.3
SER Meld WavLM-large(Chen et al., [2022](https://arxiv.org/html/2407.10759v1#bib.bib5))ACC↑↑\uparrow↑0.542
Qwen-Audio(Chu et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib7))0.557
Qwen2-Audio 0.553
VSC VocalSound CLAP(Elizalde et al., [2022](https://arxiv.org/html/2407.10759v1#bib.bib14))ACC↑↑\uparrow↑0.4945
Pengi(Deshmukh et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib11))0.6035
Qwen-Audio(Chu et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib7))0.9289
Qwen2-Audio 0.9392
AIR-Bench (Yang et al., [2024](https://arxiv.org/html/2407.10759v1#bib.bib37))Chat Benchmark Speech | Sound | Music | Mixed-Audio SALMONN(Tang et al., [2024](https://arxiv.org/html/2407.10759v1#bib.bib31)) BLSP(Wang et al., [2023a](https://arxiv.org/html/2407.10759v1#bib.bib33)) Pandagpt(Su et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib30)) Macaw-LLM(Lyu et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib19)) SpeechGPT(Zhang et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib38)) Next-gpt(Wu et al., [2023b](https://arxiv.org/html/2407.10759v1#bib.bib36)) Qwen-Audio(Chu et al., [2023](https://arxiv.org/html/2407.10759v1#bib.bib7)) Gemini-1.5-pro(Reid et al., [2024](https://arxiv.org/html/2407.10759v1#bib.bib28)) Qwen2-Audio GPT-4↑↑\uparrow↑6.16 | 6.28 | 5.95 | 6.08 6.17 | 5.55 | 5.08 | 5.33 3.58 | 5.46 | 5.06 | 4.25 0.97 | 1.01 | 0.91 | 1.01 1.57 | 0.95 | 0.95 | 4.13 3.86 | 4.76 | 4.18 | 4.13 6.47 | 6.95 | 5.52 | 6.08 6.97 | 5.49 | 5.06 | 5.27 7.18 | 6.99 | 6.79 | 6.77

### 3.2 Main Results

In this section, we present a comprehensive evaluation of the Qwen2-Audio model, assessing its performance across various tasks without any task-specific fine-tuning. We begin by examining its English Automatic Speech Recognition (ASR) results, as depicted in Table[2](https://arxiv.org/html/2407.10759v1#S3.T2 "Table 2 ‣ 3.1 Evaluation ‣ Experiments ‣ Qwen2-Audio Technical Report"), where Qwen2-Audio exhibits superior performance compared to previous multi-task learning models. Specifically, it achieves a 1.6% and 3.6% WER on the librispeech test-clean and test-other datasets, respectively. Compared with Whisper-large-v3 on Fleurs zh subset, we achieve better results than Whisper-large-v3. One point to note is that Qwen2-Audio is not evaluated in a zero-shot manner on the Common Voice 15 dataset, whereas Whisper’s results are obtained in a zero-shot fashion. However, on the Fleurs dataset, both Qwen2-Audio and Whisper are evaluated in a zero-shot manner. Furthermore, we evaluate Qwen2-Audio’s speech translation performance on the CoVoST2 dataset. The results reveal that Qwen2-Audio outperforms the baselines by a substantial margin across all seven translation directions. For sound, we analyze the performance of Qwen2-Audio on SER, and VSC, as summarized in Table[2](https://arxiv.org/html/2407.10759v1#S3.T2 "Table 2 ‣ 3.1 Evaluation ‣ Experiments ‣ Qwen2-Audio Technical Report"). Across these tasks, Qwen2-Audio consistently outperforms the baselines by a significant margin.

Lastly, to objectively evaluate the chat capabilities of Qwen2-Audio, we measured its performance on the chat benchmark of the AIR-Bench(Yang et al., [2024](https://arxiv.org/html/2407.10759v1#bib.bib37)). Note that since Gemini-1.5(Reid et al., [2024](https://arxiv.org/html/2407.10759v1#bib.bib28))2 2 2[https://console.cloud.google.com/vertex-ai/generative/multimodal/create](https://console.cloud.google.com/vertex-ai/generative/multimodal/create) cannot correctly return some test samples due to its SAFETY reasons during testing, the number of samples of Gemini-1.5 on AIR-Bench-chat has been reduced by about 1/5. As shown in table[2](https://arxiv.org/html/2407.10759v1#S3.T2 "Table 2 ‣ 3.1 Evaluation ‣ Experiments ‣ Qwen2-Audio Technical Report"), Qwen2-Audio demonstrates state-of-the-art (SOTA) instruction-following capabilities across speech, sound music and mixed-Audio subsets. It shows substantial improvements compared to Qwen-Audio and significantly outperforms other LALMs.

Cases
-----

![Image 4: Refer to caption](https://arxiv.org/html/2407.10759v1/x3.png)

Figure 4: Example showing Qwen2-Audio’s capability in free chat around speech.

![Image 5: Refer to caption](https://arxiv.org/html/2407.10759v1/x4.png)

Figure 5: Example showing Qwen2-Audio’s capability in free chat around speech.

![Image 6: Refer to caption](https://arxiv.org/html/2407.10759v1/x5.png)

Figure 6: Example showing Qwen2-Audio’s capability in free chat around speech and nature sound.

![Image 7: Refer to caption](https://arxiv.org/html/2407.10759v1/x6.png)

Figure 7: Example showing Qwen2-Audio’s capability in speech analysis.

![Image 8: Refer to caption](https://arxiv.org/html/2407.10759v1/x7.png)

Figure 8: Example showing Qwen2-Audio’s capability in sound analysis.

![Image 9: Refer to caption](https://arxiv.org/html/2407.10759v1/x8.png)

Figure 9: Example showing Qwen2-Audio’s capability in music analysis.

![Image 10: Refer to caption](https://arxiv.org/html/2407.10759v1/x9.png)

Figure 10: Example showing Qwen2-Audio’s robustness in mixed audio analysis.

Conclusion
----------

In this paper, we present Qwen2-Audio, which builds upon Qwen-Audio’s capability to analyze various types of audio while also being endowed with voice interaction abilities. During the pre-training stage, we utilized natural language prompts for different data and tasks and have further expanded the data volume. In the SFT phase, we enhanced Qwen2-Audio’s alignment with human interaction by increasing the quantity, quality, and complexity of SFT data, thereby enabling seamless voice and text interactions. Additionally, we improved Qwen2-Audio’s response quality through the DPO stage. Objective metrics tested on diverse benchmarks demonstrate Qwen2-Audio’s proficiency in audio understanding and dialogue capabilities. The cases presented within the paper also illustrate Qwen2-Audio’s fluent and flexible voice interaction capability.

Acknowledgements
----------------

We express our gratitude to Jinze Bai, Shuai Bai, Peng Wang, Sinan Tan, Shijie Wang, Kai Dang for their insightful discussion.

References
----------

*   Agostinelli et al. (2023) Andrea Agostinelli, Timo I Denk, Zalán Borsos, Jesse Engel, Mauro Verzetti, Antoine Caillon, Qingqing Huang, Aren Jansen, Adam Roberts, Marco Tagliasacchi, et al. Musiclm: Generating music from text. _arXiv preprint arXiv:2301.11325_, 2023. 
*   Ao et al. (2021) Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, et al. Speecht5: Unified-modal encoder-decoder pre-training for spoken language processing. _arXiv:2110.07205_, 2021. 
*   Ardila et al. (2020) R.Ardila, M.Branson, K.Davis, M.Henretty, M.Kohler, J.Meyer, R.Morais, L.Saunders, F.M. Tyers, and G.Weber. Common voice: A massively-multilingual speech corpus. In _Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020)_, pages 4211–4215, 2020. 
*   Bai et al. (2023) Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. _arXiv preprint arXiv:2309.16609_, 2023. 
*   Chen et al. (2022) Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, Xiangzhan Yu, and Furu Wei. Wavlm: Large-scale self-supervised pre-training for full stack speech processing. _IEEE J. Sel. Top. Signal Process._, 2022. 
*   Chen et al. (2021) Yi-Chen Chen, Po-Han Chi, Shu-wen Yang, Kai-Wei Chang, Jheng-hao Lin, Sung-Feng Huang, Da-Rong Liu, Chi-Liang Liu, Cheng-Kuang Lee, and Hung-yi Lee. Speechnet: A universal modularized model for speech processing tasks. _arXiv:2105.03070_, 2021. 
*   Chu et al. (2023) Yunfei Chu, Jin Xu, Xiaohuan Zhou, Qian Yang, Shiliang Zhang, Zhijie Yan, Chang Zhou, and Jingren Zhou. Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models. _arXiv preprint arXiv:2311.07919_, 2023. 
*   Cieri et al. (2004) Christopher Cieri, David Miller, and Kevin Walker. The fisher corpus: A resource for the next generations of speech-to-text. In _LREC_, volume 4, pages 69–71, 2004. 
*   Conneau et al. (2022) Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera, and Ankur Bapna. Fleurs: Few-shot learning evaluation of universal representations of speech. _2022 IEEE Spoken Language Technology Workshop (SLT)_, pages 798–805, 2022. URL [https://api.semanticscholar.org/CorpusID:249062909](https://api.semanticscholar.org/CorpusID:249062909). 
*   Das et al. (2024) Nilaksh Das, Saket Dingliwal, Srikanth Ronanki, Rohit Paturi, David Huang, Prashant Mathur, Jie Yuan, Dhanush Bekal, Xing Niu, Sai Muralidhar Jayanthi, et al. Speechverse: A large-scale generalizable audio language model. _arXiv preprint arXiv:2405.08295_, 2024. 
*   Deshmukh et al. (2023) Soham Deshmukh, Benjamin Elizalde, Rita Singh, and Huaming Wang. Pengi: An audio language model for audio tasks. _CoRR_, 2023. 
*   Drossos et al. (2020) Konstantinos Drossos, Samuel Lipping, and Tuomas Virtanen. Clotho: an audio captioning dataset. In _2020 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2020, Barcelona, Spain, May 4-8, 2020_. IEEE, 2020. 
*   Du et al. (2018) Jiayu Du, Xingyu Na, Xuechen Liu, and Hui Bu. AISHELL-2: transforming mandarin ASR research into industrial scale. abs/1808.10583, 2018. 
*   Elizalde et al. (2022) Benjamin Elizalde, Soham Deshmukh, Mahmoud Al Ismail, and Huaming Wang. CLAP: learning audio concepts from natural language supervision. abs/2206.04769, 2022. 
*   Gao et al. (2023) Zhifu Gao, Zerui Li, Jiaming Wang, Haoneng Luo, Xian Shi, Mengzhe Chen, Yabin Li, Lingyun Zuo, Zhihao Du, Zhangyu Xiao, and Shiliang Zhang. Funasr: A fundamental end-to-end speech recognition toolkit. _CoRR_, abs/2305.11013, 2023. 
*   Gong et al. (2022) Yuan Gong, Jin Yu, and James R. Glass. Vocalsound: A dataset for improving human vocal sounds recognition. In _IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2022, Virtual and Singapore, 23-27 May 2022_, pages 151–155. IEEE, 2022. doi: 10.1109/ICASSP43922.2022.9746828. URL [https://doi.org/10.1109/ICASSP43922.2022.9746828](https://doi.org/10.1109/ICASSP43922.2022.9746828). 
*   Kim et al. (2019) Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee, and Gunhee Kim. Audiocaps: Generating captions for audios in the wild. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_, 2019. 
*   Kong et al. (2024) Zhifeng Kong, Arushi Goel, Rohan Badlani, Wei Ping, Rafael Valle, and Bryan Catanzaro. Audio flamingo: A novel audio language model with few-shot learning and dialogue abilities. _arXiv preprint arXiv:2402.01831_, 2024. 
*   Lyu et al. (2023) Chenyang Lyu, Minghao Wu, Longyue Wang, Xinting Huang, Bingshuai Liu, Zefeng Du, Shuming Shi, and Zhaopeng Tu. Macaw-llm: Multi-modal language modeling with image, audio, video, and text integration. _CoRR_, abs/2306.09093, 2023. 
*   OpenAI (2023) OpenAI. Gpt-4 technical report, 2023. 
*   OpenAI (2024) OpenAI. Gpt-4o, 2024. URL [https://openai.com/index/hello-gpt-4o/](https://openai.com/index/hello-gpt-4o/). 
*   Panayotov et al. (2015) Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: An ASR corpus based on public domain audio books. In _2015 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2015, South Brisbane, Queensland, Australia, April 19-24, 2015_. IEEE, 2015. 
*   Papineni et al. (2002) Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In _Proceedings of the 40th annual meeting of the Association for Computational Linguistics_, 2002. 
*   Poria et al. (2019) Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. MELD: A multimodal multi-party dataset for emotion recognition in conversations. In _Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers_. Association for Computational Linguistics, 2019. 
*   Qwen (2023) Qwen. Introducing qwen-7b: Open foundation and human-aligned models (of the state-of-the-arts), 2023. URL [https://github.com/QwenLM/Qwen-7B](https://github.com/QwenLM/Qwen-7B). 
*   Radford et al. (2023) Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In _International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA_, 2023. 
*   Rafailov et al. (2024) Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. _Advances in Neural Information Processing Systems_, 36, 2024. 
*   Reid et al. (2024) Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. _arXiv preprint arXiv:2403.05530_, 2024. 
*   Si et al. (2023) Shuzheng Si, Wentao Ma, Yuchuan Wu, Yinpei Dai, Haoyu Gao, Ting-En Lin, Hangyu Li, Rui Yan, Fei Huang, and Yongbin Li. Spokenwoz: A large-scale speech-text benchmark for spoken task-oriented dialogue in multiple domains. _arXiv preprint arXiv:2305.13040_, 2023. 
*   Su et al. (2023) Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. Pandagpt: One model to instruction-follow them all. _arXiv:2305.16355_, 2023. 
*   Tang et al. (2024) Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun MA, and Chao Zhang. SALMONN: Towards generic hearing abilities for large language models. In _The Twelfth International Conference on Learning Representations_, 2024. URL [https://openreview.net/forum?id=14rn7HpKVk](https://openreview.net/forum?id=14rn7HpKVk). 
*   Wang et al. (2020) Changhan Wang, Anne Wu, and Juan Miguel Pino. Covost 2: A massively multilingual speech-to-text translation corpus. abs/2007.10310, 2020. URL [https://arxiv.org/abs/2007.10310](https://arxiv.org/abs/2007.10310). 
*   Wang et al. (2023a) Chen Wang, Minpeng Liao, Zhongqiang Huang, Jinliang Lu, Junhong Wu, Yuchen Liu, Chengqing Zong, and Jiajun Zhang. Blsp: Bootstrapping language-speech pre-training via behavior alignment of continuation writing. _arXiv:2309.00916_, 2023a. 
*   Wang et al. (2023b) Mingqiu Wang, Wei Han, Izhak Shafran, Zelin Wu, Chung-Cheng Chiu, Yuan Cao, Yongqiang Wang, Nanxin Chen, Yu Zhang, Hagen Soltau, Paul K. Rubenstein, Lukas Zilka, Dian Yu, Zhong Meng, Golan Pundak, Nikhil Siddhartha, Johan Schalkwyk, and Yonghui Wu. SLM: bridge the thin gap between speech and text foundation models. abs/2310.00230, 2023b. 
*   Wu et al. (2023a) Jian Wu, Yashesh Gaur, Zhuo Chen, Long Zhou, Yimeng Zhu, Tianrui Wang, Jinyu Li, Shujie Liu, Bo Ren, Linquan Liu, and Yu Wu. On decoder-only architecture for speech-to-text and large language model integration. abs/2307.03917, 2023a. 
*   Wu et al. (2023b) Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal LLM. _CoRR_, abs/2309.05519, 2023b. 
*   Yang et al. (2024) Qian Yang, Jin Xu, Wenrui Liu, Yunfei Chu, Ziyue Jiang, Xiaohuan Zhou, Yichong Leng, Yuanjun Lv, Zhou Zhao, Chang Zhou, and Jingren Zhou. Air-bench: Benchmarking large audio-language models via generative comprehension. In _ACL_, 2024. 
*   Zhang et al. (2023) Dong Zhang, Shimin Li, Xin Zhang, Jun Zhan, Pengyu Wang, Yaqian Zhou, and Xipeng Qiu. Speechgpt: Empowering large language models with intrinsic cross-modal conversational abilities. _CoRR_, abs/2305.11000, 2023. 
*   Zhou et al. (2022) Xiaohuan Zhou, Jiaming Wang, Zeyu Cui, Shiliang Zhang, Zhijie Yan, Jingren Zhou, and Chang Zhou. Mmspeech: Multi-modal multi-task encoder-decoder pre-training for speech recognition. abs/2212.00500, 2022.

