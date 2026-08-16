Title: Structural Embedding Alignment for Multimodal Large Language Model

URL Source: https://arxiv.org/html/2405.20797

Published Time: Tue, 18 Jun 2024 01:52:43 GMT

Markdown Content:
\useunder

\ul

Shiyin Lu 1 Yang Li 1 Qing-Guo Chen 1 Zhao Xu 1

Weihua Luo 1 Kaifu Zhang 1 Han-Jia Ye 2,3

1 AI Business, Alibaba Group 2 School of Artificial Intelligence, Nanjing University 

3 National Key Laboratory for Novel Software Technology, Nanjing University 

[https://github.com/AIDC-AI/Ovis](https://github.com/AIDC-AI/Ovis)

###### Abstract

Current Multimodal Large Language Models (MLLMs) typically integrate a pre-trained LLM with another pre-trained vision transformer through a connector, such as an MLP, endowing the LLM with visual capabilities. However, the misalignment between two embedding strategies in MLLMs — the structural textual embeddings based on an embedding look-up table and the continuous embeddings generated directly by the vision encoder — makes challenges for a more seamless fusion of visual and textual information. We propose Ovis, a novel MLLM architecture designed to structurally align visual and textual embeddings. Ovis integrates an additional learnable visual embedding table into the visual encoder’s process. To capture rich visual semantics, each image patch indexes the visual embedding table multiple times, resulting in a final visual embedding that is a probabilistic combination of the indexed embeddings. This structural approach mirrors the method used for generating textual embeddings. Empirical evaluations on various multimodal benchmarks show that Ovis outperforms open-source MLLMs of similar parameter scales and even surpasses the proprietary model Qwen-VL-Plus overall. These results highlight the potential of Ovis’ structured visual representation for advancing MLLM architectural design and promoting more effective multimodal learning.

1 Introduction
--------------

The development of Large Language Models (LLMs) is advancing rapidly[[64](https://arxiv.org/html/2405.20797v2#bib.bib64), [65](https://arxiv.org/html/2405.20797v2#bib.bib65), [6](https://arxiv.org/html/2405.20797v2#bib.bib6), [60](https://arxiv.org/html/2405.20797v2#bib.bib60), [58](https://arxiv.org/html/2405.20797v2#bib.bib58), [59](https://arxiv.org/html/2405.20797v2#bib.bib59)], illuminating the path toward Artificial General Intelligence (AGI). These sophisticated models excel at understanding and generating text with remarkable proficiency[[76](https://arxiv.org/html/2405.20797v2#bib.bib76), [77](https://arxiv.org/html/2405.20797v2#bib.bib77), [14](https://arxiv.org/html/2405.20797v2#bib.bib14), [74](https://arxiv.org/html/2405.20797v2#bib.bib74)]. However, to approach the complexity and versatility of human intelligence, LLMs must transcend mere textual comprehension. The ability to interpret and understand visual information becomes a critical feature on this journey toward AGI. Consequently, there has been a surge of interest in developing Multimodal Large Language Models (MLLMs) — models that meld the power of language comprehension and visual perception[[37](https://arxiv.org/html/2405.20797v2#bib.bib37), [38](https://arxiv.org/html/2405.20797v2#bib.bib38), [18](https://arxiv.org/html/2405.20797v2#bib.bib18), [92](https://arxiv.org/html/2405.20797v2#bib.bib92), [48](https://arxiv.org/html/2405.20797v2#bib.bib48), [41](https://arxiv.org/html/2405.20797v2#bib.bib41), [40](https://arxiv.org/html/2405.20797v2#bib.bib40), [4](https://arxiv.org/html/2405.20797v2#bib.bib4), [86](https://arxiv.org/html/2405.20797v2#bib.bib86)].

Instead of directly training the entire MLLMs, current open-source MLLMs primarily derive their visual ability from a pre-trained LLM and a pre-trained vision encoder. The visual and textual components have different tokenization and embedding strategies. Textual embeddings are indexed from the LLM’s embedding look-up table, where each “word” is mapped to an embedding, via one-hot textual tokens. In contrast, visual embeddings are generated directly by the vision encoder in an unstructured manner. To align the dimensions between these two types of embeddings, cross-modal connectors such as MLPs project embeddings into a joint space, allowing all embeddings to serve as inputs to the LLM[[48](https://arxiv.org/html/2405.20797v2#bib.bib48), [47](https://arxiv.org/html/2405.20797v2#bib.bib47), [49](https://arxiv.org/html/2405.20797v2#bib.bib49), [41](https://arxiv.org/html/2405.20797v2#bib.bib41), [40](https://arxiv.org/html/2405.20797v2#bib.bib40)]. Although this architecture only aligns the dimensions of visual and textual embeddings, it has shown promising performance across various vision-language tasks. Nevertheless, the inherent discrepancy in tokenization and embedding strategies may lead to a potential limitation in the connector-based architecture, so an intuitive question is

> Could we achieve further improvement in MLLMs if we generate visual embeddings in a structured manner to match the textual embedding strategy in LLMs?

We propose a novel MLLM architecture, dubbed “Ovis”, which assimilates the insights from LLMs to establish structured embeddings of visual input. As illustrated in[Figure 1](https://arxiv.org/html/2405.20797v2#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Ovis: Structural Embedding Alignment for Multimodal Large Language Model"), Ovis introduces an additional learnable visual embedding look-up table to transform continuous visual tokens, thus paralleling the structural integrity of its textual counterpart. [Figure 2](https://arxiv.org/html/2405.20797v2#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Ovis: Structural Embedding Alignment for Multimodal Large Language Model") demonstrates that Ovis outperforms open-source MLLMs within the same parameter tier across various benchmarks, and Ovis-14B also surpasses the high-resource proprietary model Qwen-VL-Plus overall.

![Image 1: Refer to caption](https://arxiv.org/html/2405.20797v2/x1.png)

Figure 1: Comparison between different embedding strategies in MLLM. In the connector-based approach (a), the connector transforms the visual embeddings into the same dimensional as the textual embedding, where the latter is indexed from a textual embedding table. As illustrated in (b), our Ovis leverages an additional visual embedding table to produce structural visual embeddings and align the embedding strategies of two modalities. 

![Image 2: Refer to caption](https://arxiv.org/html/2405.20797v2/x2.png)

(a)7B tier

![Image 3: Refer to caption](https://arxiv.org/html/2405.20797v2/x3.png)

(b)14B tier and Qwen-VL-Plus

Figure 2: Ovis outperforms open-source MLLMs within the same parameter tier in various benchmarks, and Ovis-14B also surpasses the high-resource proprietary model Qwen-VL-Plus overall.

In particular, Ovis incorporates a visual embedding table whose rows correspond to unique visual words, representing distinct visual patterns. Given the continuous token of a visual patch output by the visual encoder[[22](https://arxiv.org/html/2405.20797v2#bib.bib22)], Ovis first maps the token into a probabilistic token, revealing its similarity among the entire visual vocabulary set. The probabilistic token captures the rich semantics within a single visual patch, which may contain patterns from multiple visual words, effectively treating the visual token as if it were sampled from the visual embedding table based on the distribution. Ovis subsequently indexes the visual embedding table multiple times based on the probabilistic token, resulting in a final visual embedding that is a combination of the indexed embeddings, i.e., the expectation of embeddings over the whole embedding table. Therefore, Ovis aligns the visual embedding strategy with the structured nature of its textual counterpart.

The optimization of the visual embedding table and the parameters for generating the probabilistic tokens significantly influence the performance of the MLLM. Instead of using an additional autoencoder with vector quantization over images and various other losses, as utilized in previous methods[[78](https://arxiv.org/html/2405.20797v2#bib.bib78), [23](https://arxiv.org/html/2405.20797v2#bib.bib23), [33](https://arxiv.org/html/2405.20797v2#bib.bib33)], Ovis leverages a joint textual generation loss and optimizes the parameters in a three-stage manner. This learning process of Ovis avoids the risk of falling short in vision-language tasks due to the absence of textual guidance.

We implement Ovis with open-source vision transformer and LLM models as backbones and evaluate its performance in diverse multimodal benchmarks. The outcomes demonstrate that Ovis outperforms popular open-source MLLMs within the same parameter tier in the majority of these benchmarks. Specifically, Ovis-8B exhibits a large margin over its competitors, and Ovis-14B consistently surpasses the compared open-source MLLMs. Impressively, Ovis-14B also performs better than the high-resource proprietary model Qwen-VL-Plus [[4](https://arxiv.org/html/2405.20797v2#bib.bib4)] overall, and its performance is even on par with the stronger proprietary model Qwen-VL-Max [[4](https://arxiv.org/html/2405.20797v2#bib.bib4)] in the general multimodal benchmarks MMStar [[12](https://arxiv.org/html/2405.20797v2#bib.bib12)] and MMBench [[50](https://arxiv.org/html/2405.20797v2#bib.bib50)] and several specialized multimodal benchmarks including MathVista [[55](https://arxiv.org/html/2405.20797v2#bib.bib55)], HallusionBench [[46](https://arxiv.org/html/2405.20797v2#bib.bib46)], and RealWorldQA [[84](https://arxiv.org/html/2405.20797v2#bib.bib84)]. These results underscore the superiority and potential of the Ovis architecture. We believe that the demonstrated effectiveness and advantages of Ovis will enhance further investigations into MLLM architectural designs, moving beyond the confines of the connector-based architecture.

2 Related Work
--------------

#### Large Language Models.

In recent years, the development of Large Language Models (LLMs) has significantly advanced the field of natural language processing. The debut of GPT-3 [[6](https://arxiv.org/html/2405.20797v2#bib.bib6)] marked a notable surge in performance, especially in few-shot and zero-shot learning scenarios, underscoring the substantial promise of LLMs. This potential was further demonstrated by subsequent enhancements in models such as ChatGPT [[58](https://arxiv.org/html/2405.20797v2#bib.bib58)], GPT-4 [[59](https://arxiv.org/html/2405.20797v2#bib.bib59)], Gemini [[75](https://arxiv.org/html/2405.20797v2#bib.bib75), [68](https://arxiv.org/html/2405.20797v2#bib.bib68)], and Claude [[2](https://arxiv.org/html/2405.20797v2#bib.bib2)]. Concurrently, open-source models have been rapidly evolving, including the LLaMA [[76](https://arxiv.org/html/2405.20797v2#bib.bib76), [77](https://arxiv.org/html/2405.20797v2#bib.bib77)] series, Vicuna [[14](https://arxiv.org/html/2405.20797v2#bib.bib14)], Baichuan [[85](https://arxiv.org/html/2405.20797v2#bib.bib85)], Qwen [[3](https://arxiv.org/html/2405.20797v2#bib.bib3)], Mistral [[32](https://arxiv.org/html/2405.20797v2#bib.bib32)], and Yi [[88](https://arxiv.org/html/2405.20797v2#bib.bib88)]. Notably, the open-source models Llama3 [[56](https://arxiv.org/html/2405.20797v2#bib.bib56)] and Mistral-MOE [[57](https://arxiv.org/html/2405.20797v2#bib.bib57)] have approached and, in some cases, surpassed the performance of closed-source models. Despite these advancements, LLMs inherently lack the capability to process or interpret multimodal data, limiting their application in scenarios requiring an understanding of more than just textual information.

#### Multimodal Large Language Models.

Multimodal Large Language Models (MLLMs) enhance the capabilities of LLMs by not only understanding and generating text but also interpreting and relating visual elements to textual descriptions [[87](https://arxiv.org/html/2405.20797v2#bib.bib87)]. Most open-source MLLMs consist of several components, namely a vision encoder [[66](https://arxiv.org/html/2405.20797v2#bib.bib66), [73](https://arxiv.org/html/2405.20797v2#bib.bib73), [24](https://arxiv.org/html/2405.20797v2#bib.bib24), [90](https://arxiv.org/html/2405.20797v2#bib.bib90)], a connector, and an LLM. The type of the connector can be roughly divided into three categories. The cross-attention-based methods isolate and integrate visual and text modalities within the LLM, as seen in models like Flamingo [[1](https://arxiv.org/html/2405.20797v2#bib.bib1)] and CogVLM [[79](https://arxiv.org/html/2405.20797v2#bib.bib79)]. The query-based methods query visual embeddings via a transformer-like architecture and send the obtained visual embeddings along with the text to the LLM, exemplified by Blip-2 [[38](https://arxiv.org/html/2405.20797v2#bib.bib38)], Instruct-Blip [[18](https://arxiv.org/html/2405.20797v2#bib.bib18)], and Qwen-VL [[4](https://arxiv.org/html/2405.20797v2#bib.bib4)]. The projection-based methods directly project the visual embeddings, align them to the text modality, and uniformly feed the mixed embeddings into the LLM for understanding and generation. This approach is used by models such as LLaVA [[48](https://arxiv.org/html/2405.20797v2#bib.bib48)], Mini-GPT4 [[92](https://arxiv.org/html/2405.20797v2#bib.bib92)], DeepSeek-VL [[51](https://arxiv.org/html/2405.20797v2#bib.bib51)], and Mini-Gemini [[40](https://arxiv.org/html/2405.20797v2#bib.bib40)]. In addition to architecture design, current MLLM research focuses on high-resolution capabilities [[44](https://arxiv.org/html/2405.20797v2#bib.bib44), [21](https://arxiv.org/html/2405.20797v2#bib.bib21), [80](https://arxiv.org/html/2405.20797v2#bib.bib80)], miniaturization of MLLMs [[31](https://arxiv.org/html/2405.20797v2#bib.bib31), [15](https://arxiv.org/html/2405.20797v2#bib.bib15), [42](https://arxiv.org/html/2405.20797v2#bib.bib42)], specialized models (e.g., medical MLLMs [[36](https://arxiv.org/html/2405.20797v2#bib.bib36)], document MLLMs [[10](https://arxiv.org/html/2405.20797v2#bib.bib10), [30](https://arxiv.org/html/2405.20797v2#bib.bib30)]), and the integration of other modalities [[83](https://arxiv.org/html/2405.20797v2#bib.bib83), [61](https://arxiv.org/html/2405.20797v2#bib.bib61), [52](https://arxiv.org/html/2405.20797v2#bib.bib52)]. Our Ovis serves as a new MLLM architecture that departs from the connector-based framework and involves a novel visual tokenizer for structured visual embeddings.

#### Visual Tokenization.

Tokenizing visual input has been explored in various visual tasks[[35](https://arxiv.org/html/2405.20797v2#bib.bib35)]. VQVAE[[78](https://arxiv.org/html/2405.20797v2#bib.bib78)] encodes visual input into discrete latent variables, combining the principles of variational autoencoders with vector quantization. This approach facilitates the generation of high-quality and diverse outputs, making it effective for tasks such as image generation and compression. Based on VQVAE, VQGAN[[23](https://arxiv.org/html/2405.20797v2#bib.bib23)] incorporates the adversarial training framework of PatchGAN [[19](https://arxiv.org/html/2405.20797v2#bib.bib19)], enhancing the realism of generated images. Leveraging a visual tokenization strategy similar to VQVAE, BEIT[[5](https://arxiv.org/html/2405.20797v2#bib.bib5), [63](https://arxiv.org/html/2405.20797v2#bib.bib63)] uses discrete visual tokens during its pre-training phase. In this phase, portions of the input image are masked, and the model predicts the discrete tokens for these masked patches, similar to the masked language modeling in BERT[[20](https://arxiv.org/html/2405.20797v2#bib.bib20)]. Due to the lack of joint modeling with the linguistic modality, there has been scant work combining discretized visual tokens with MLLMs. The discretization of visual tokens has been investigated to link visual output to the input of diffusion models[[28](https://arxiv.org/html/2405.20797v2#bib.bib28), [29](https://arxiv.org/html/2405.20797v2#bib.bib29), [33](https://arxiv.org/html/2405.20797v2#bib.bib33)], where additional reconstruction loss and decoders are used during training. A recent method [[62](https://arxiv.org/html/2405.20797v2#bib.bib62)] employs a linear head layer to tokenize visual information, which differs from our approach. Specifically, the head layer in [[62](https://arxiv.org/html/2405.20797v2#bib.bib62)] is trained solely on vision data in a distilled manner, whereas we optimize the visual head layer using gradients backward from the LLM on vision-language data. Additionally, we propose learning a distinct visual embedding table tailored specifically for visual information, rather than directly using the LLM’s textual embedding table to retrieve embeddings for visual tokens as done in [[62](https://arxiv.org/html/2405.20797v2#bib.bib62)].

3 Ovis
------

In this section, we first review the differences in visual and textual embedding strategies in MLLMs. We then introduce our proposed architecture, Ovis, which incorporates a linear mapping for probabilistic tokens and an additional visual embedding look-up table within LLM.

![Image 4: Refer to caption](https://arxiv.org/html/2405.20797v2/x4.png)

(a)Overall Architecture

![Image 5: Refer to caption](https://arxiv.org/html/2405.20797v2/x5.png)

(b)Visual Tokenizer

![Image 6: Refer to caption](https://arxiv.org/html/2405.20797v2/x6.png)

(c)From visual patch to visual embedding

Figure 3: Illustration of Ovis. Figure (a) shows the whole architecture of Ovis, which contains two embedding tables for visual and textual inputs. Figure (b) illustrates how a visual patch is first mapped to a probabilistic token. Figure (c) demonstrates that the probabilistic token helps select multiple embeddings from the embedding table and output their weighted combination.

### 3.1 Difference between Visual and Textual Tokens

Both images and texts are input into the MLLM, and they have diverse tokenization strategies.

Let ℐ∈ℝ C×W×H ℐ superscript ℝ 𝐶 𝑊 𝐻\mathcal{I}\in\mathbb{R}^{C\times W\times H}caligraphic_I ∈ blackboard_R start_POSTSUPERSCRIPT italic_C × italic_W × italic_H end_POSTSUPERSCRIPT be the pixel value tensor of an image, where C,W,H 𝐶 𝑊 𝐻 C,W,H italic_C , italic_W , italic_H denote the channel number, width, and height of the image, respectively. The image is first divided into a sequence of visual patches {𝒫 i∈ℝ C×w×h}i=1,2,…,n subscript subscript 𝒫 𝑖 superscript ℝ 𝐶 𝑤 ℎ 𝑖 1 2…𝑛\{\mathcal{P}_{i}\in\mathbb{R}^{C\times w\times h}\}_{i=1,2,\ldots,n}{ caligraphic_P start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_C × italic_w × italic_h end_POSTSUPERSCRIPT } start_POSTSUBSCRIPT italic_i = 1 , 2 , … , italic_n end_POSTSUBSCRIPT, where w 𝑤 w italic_w and h ℎ h italic_h denote the width and height of the patch, respectively, and n=⌈W w⌉⁢⌈H h⌉𝑛 𝑊 𝑤 𝐻 ℎ n=\lceil\frac{W}{w}\rceil\lceil\frac{H}{h}\rceil italic_n = ⌈ divide start_ARG italic_W end_ARG start_ARG italic_w end_ARG ⌉ ⌈ divide start_ARG italic_H end_ARG start_ARG italic_h end_ARG ⌉ is the number of patches. Given a pre-trained vision transformer (ViT) backbone g θ subscript 𝑔 𝜃 g_{\theta}italic_g start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT with parameters θ 𝜃\theta italic_θ, we then transform the patches into a sequence of visual representations {𝒓 i∈ℝ d}i=1 n superscript subscript subscript 𝒓 𝑖 superscript ℝ 𝑑 𝑖 1 𝑛\{{\bm{r}}_{i}\in\mathbb{R}^{d}\}_{i=1}^{n}{ bold_italic_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_d end_POSTSUPERSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_n end_POSTSUPERSCRIPT.

For the textual input, let {𝒕 i}i=1 m superscript subscript subscript 𝒕 𝑖 𝑖 1 𝑚\{\bm{t}_{i}\}_{i=1}^{m}{ bold_italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_m end_POSTSUPERSCRIPT be the input sequence of textual tokens, which are further processed by an LLM f ϕ subscript 𝑓 italic-ϕ f_{\phi}italic_f start_POSTSUBSCRIPT italic_ϕ end_POSTSUBSCRIPT, parameterized by ϕ italic-ϕ\phi italic_ϕ. In MLLM, both visual ({𝒓 i}i=1 n superscript subscript subscript 𝒓 𝑖 𝑖 1 𝑛\{{\bm{r}}_{i}\}_{i=1}^{n}{ bold_italic_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_n end_POSTSUPERSCRIPT) and textual ({𝒕 i}i=1 m superscript subscript subscript 𝒕 𝑖 𝑖 1 𝑚\{\bm{t}_{i}\}_{i=1}^{m}{ bold_italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_m end_POSTSUPERSCRIPT) tokens should be transformed into the same form, and then LLM processes all tokens into an output sequence of textual tokens. We use λ 𝜆\lambda italic_λ to denote the index of the image indicator token, i.e., 𝒕 λ=<image>subscript 𝒕 𝜆<image>\bm{t}_{\lambda}=\text{\textless image\textgreater}bold_italic_t start_POSTSUBSCRIPT italic_λ end_POSTSUBSCRIPT = <image>, and the multimodal input tokens become

[𝒕 1,…,𝒕 λ−1,<image>,…,𝒕 m].subscript 𝒕 1…subscript 𝒕 𝜆 1<image>…subscript 𝒕 𝑚[\bm{t}_{1},\ldots,\bm{t}_{\lambda-1},\text{\textless image\textgreater},% \ldots,\bm{t}_{m}]\;.[ bold_italic_t start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , … , bold_italic_t start_POSTSUBSCRIPT italic_λ - 1 end_POSTSUBSCRIPT , <image> , … , bold_italic_t start_POSTSUBSCRIPT italic_m end_POSTSUBSCRIPT ] .(1)

Since the visual and textual tokens have diverse dimensions, it is difficult to substitute <image> directly with {𝒓 i}i=1 n superscript subscript subscript 𝒓 𝑖 𝑖 1 𝑛\{{\bm{r}}_{i}\}_{i=1}^{n}{ bold_italic_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_n end_POSTSUPERSCRIPT. Previous approaches introduce additional linear projection[[48](https://arxiv.org/html/2405.20797v2#bib.bib48)], MLP[[47](https://arxiv.org/html/2405.20797v2#bib.bib47)], or transformer[[1](https://arxiv.org/html/2405.20797v2#bib.bib1), [79](https://arxiv.org/html/2405.20797v2#bib.bib79), [38](https://arxiv.org/html/2405.20797v2#bib.bib38), [4](https://arxiv.org/html/2405.20797v2#bib.bib4)] architectures to map visual tokens into the same form as textual ones.

### 3.2 Probabilistic Visual Tokens

Instead of using continuous visual tokens in[Equation 1](https://arxiv.org/html/2405.20797v2#S3.E1 "1 ‣ 3.1 Difference between Visual and Textual Tokens ‣ 3 Ovis ‣ Ovis: Structural Embedding Alignment for Multimodal Large Language Model"), we align the internal tokenization strategies between images and texts to inspire the potential of the MLLM.

To mimic the discrete textual tokens, we use a linear head ℝ K×d superscript ℝ 𝐾 𝑑\mathbb{R}^{K\times d}blackboard_R start_POSTSUPERSCRIPT italic_K × italic_d end_POSTSUPERSCRIPT to transform the concrete visual tokens. Assuming K 𝐾 K italic_K is the visual vocabulary size, i.e., the number of unique visual words, then given a visual token 𝒓 i subscript 𝒓 𝑖{\bm{r}}_{i}bold_italic_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT, we first transform 𝒓 i subscript 𝒓 𝑖{\bm{r}}_{i}bold_italic_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT into a (K−1)𝐾 1(K-1)( italic_K - 1 )-dimensional probability simplex Δ K superscript Δ 𝐾\Delta^{K}roman_Δ start_POSTSUPERSCRIPT italic_K end_POSTSUPERSCRIPT by a linear projection followed by a softmax normalization:

𝒗 i=softmax⁢(𝑾⁢𝒓 i)𝑾∈ℝ K×d.formulae-sequence subscript 𝒗 𝑖 softmax 𝑾 subscript 𝒓 𝑖 𝑾 superscript ℝ 𝐾 𝑑\bm{v}_{i}=\text{softmax}(\bm{W}{\bm{r}}_{i})\quad\bm{W}\in\mathbb{R}^{K\times d}.bold_italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = softmax ( bold_italic_W bold_italic_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) bold_italic_W ∈ blackboard_R start_POSTSUPERSCRIPT italic_K × italic_d end_POSTSUPERSCRIPT .(2)

We set 𝒗 i∈Δ K subscript 𝒗 𝑖 superscript Δ 𝐾\bm{v}_{i}\in\Delta^{K}bold_italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∈ roman_Δ start_POSTSUPERSCRIPT italic_K end_POSTSUPERSCRIPT as a kind of probabilistic token, which is a probability distribution over the visual vocabulary containing K 𝐾 K italic_K visual words. If 𝒓 i subscript 𝒓 𝑖{\bm{r}}_{i}bold_italic_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT is more related to certain patterns, the corresponding elements in 𝒗 i subscript 𝒗 𝑖\bm{v}_{i}bold_italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT should be larger.

Remark. Given a visual embedding table, we associate each visual word with its prototype {𝒘 i∈ℝ d}i=1 K superscript subscript subscript 𝒘 𝑖 superscript ℝ 𝑑 𝑖 1 𝐾\{\bm{w}_{i}\in\mathbb{R}^{d}\}_{i=1}^{K}{ bold_italic_w start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_d end_POSTSUPERSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_K end_POSTSUPERSCRIPT. To match a continuous visual token with the K 𝐾 K italic_K visual words in the embedding table, we leverage the inner product to calculate their similarity value. [Equation 2](https://arxiv.org/html/2405.20797v2#S3.E2 "2 ‣ 3.2 Probabilistic Visual Tokens ‣ 3 Ovis ‣ Ovis: Structural Embedding Alignment for Multimodal Large Language Model") is the normalized similarity between 𝒓 i subscript 𝒓 𝑖{\bm{r}}_{i}bold_italic_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT and all visual words.

### 3.3 Visual Embedding Table

In LLMs, it is a common practice to employ a textual embedding table, which maps each word in the vocabulary to an embedding vector. For each textual token 𝒕 i subscript 𝒕 𝑖\bm{t}_{i}bold_italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT in the one-hot form, its embedding T i∈ℝ d′subscript 𝑇 𝑖 superscript ℝ superscript 𝑑′T_{i}\in\mathbb{R}^{d^{\prime}}italic_T start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_d start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT end_POSTSUPERSCRIPT is the row of the textual embedding table indicated by the non-zero index in 𝒕 i subscript 𝒕 𝑖\bm{t}_{i}bold_italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT.

Analogously, we introduce an additional visual embedding table, where each visual word (each row) is associated with an embedding vector 𝒆 k∈ℝ d′subscript 𝒆 𝑘 superscript ℝ superscript 𝑑′{\bm{e}}_{k}\in\mathbb{R}^{d^{\prime}}bold_italic_e start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_d start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT end_POSTSUPERSCRIPT with d′superscript 𝑑′d^{\prime}italic_d start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT being the embedding dimension. To make the embeddings of visual and textual tokens have compatible shapes, we simply set the dimension of the visual embedding table to be the same as that of the textual embedding table.

Accordingly, the embedding of each visual token 𝒗 i subscript 𝒗 𝑖\bm{v}_{i}bold_italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT can be derived based on the probabilistic token:

V i=∑k=1 K v i,k⁢𝒆 k∈ℝ d′,subscript 𝑉 𝑖 superscript subscript 𝑘 1 𝐾 subscript 𝑣 𝑖 𝑘 subscript 𝒆 𝑘 superscript ℝ superscript 𝑑′V_{i}=\sum_{k=1}^{K}v_{i,k}{\bm{e}}_{k}\in\mathbb{R}^{d^{\prime}}\;,italic_V start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_k = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_K end_POSTSUPERSCRIPT italic_v start_POSTSUBSCRIPT italic_i , italic_k end_POSTSUBSCRIPT bold_italic_e start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ∈ blackboard_R start_POSTSUPERSCRIPT italic_d start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT end_POSTSUPERSCRIPT ,(3)

where v i,k subscript 𝑣 𝑖 𝑘 v_{i,k}italic_v start_POSTSUBSCRIPT italic_i , italic_k end_POSTSUBSCRIPT denotes the k 𝑘 k italic_k-th component of 𝒗 i subscript 𝒗 𝑖\bm{v}_{i}bold_italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT. On the other hand, since 𝒗 i∈Δ K subscript 𝒗 𝑖 superscript Δ 𝐾\bm{v}_{i}\in\Delta^{K}bold_italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∈ roman_Δ start_POSTSUPERSCRIPT italic_K end_POSTSUPERSCRIPT, the above formula can be rewritten as V i=𝔼 k∼𝒗 i⁢[𝒆 k]subscript 𝑉 𝑖 subscript 𝔼 similar-to 𝑘 subscript 𝒗 𝑖 delimited-[]subscript 𝒆 𝑘 V_{i}=\mathbb{E}_{k\sim\bm{v}_{i}}[{\bm{e}}_{k}]italic_V start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = blackboard_E start_POSTSUBSCRIPT italic_k ∼ bold_italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_POSTSUBSCRIPT [ bold_italic_e start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ], which is an expectation of the visual word’s embedding, with the visual word drawn from 𝒗 i subscript 𝒗 𝑖\bm{v}_{i}bold_italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT. In other words, we assume that the visual embedding could be sampled from the discrete visual embedding table based on the probabilistic token 𝒗 i subscript 𝒗 𝑖\bm{v}_{i}bold_italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT of the patch.

Remark. Considering the polysemous nature of a visual patch 𝒓 i subscript 𝒓 𝑖{\bm{r}}_{i}bold_italic_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT, assigning it only one visual word from the embedding table indexed by arg⁡max j∈{1,…,K}⁡𝒗 i⁢j subscript 𝑗 1…𝐾 subscript 𝒗 𝑖 𝑗\arg\max_{j\in\{1,\ldots,K\}}\bm{v}_{ij}roman_arg roman_max start_POSTSUBSCRIPT italic_j ∈ { 1 , … , italic_K } end_POSTSUBSCRIPT bold_italic_v start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT may neglect the rich semantics within the patch. To address this, we link the patch with multiple visual words at a time, as indicated by the non-zero elements in 𝒗 i subscript 𝒗 𝑖\bm{v}_{i}bold_italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT, which represent the correlation between the patch and K 𝐾 K italic_K visual words. We then use the weighted combination V i subscript 𝑉 𝑖 V_{i}italic_V start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT of the selected visual words as the final patch embedding. In other words, multiple visual embeddings are indexed from the embedding table based on the values in 𝒗 i subscript 𝒗 𝑖\bm{v}_{i}bold_italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT, and the weighted average of these embeddings serves as the final output of the visual embedding module. Experiments in[Appendix E](https://arxiv.org/html/2405.20797v2#A5 "Appendix E Sparsity of Ovis’ Visual Tokenizer ‣ Ovis: Structural Embedding Alignment for Multimodal Large Language Model") validate the sparsity of 𝒗 i subscript 𝒗 𝑖\bm{v}_{i}bold_italic_v start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT. The visual embedding V i subscript 𝑉 𝑖 V_{i}italic_V start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT captures the rich visual semantics and keeps its generation process similar to its textual counterpart simultaneously.

### 3.4 Training Strategy of Ovis

Both visual and textual embeddings are combined as the input into the LLM. In particular, we feed the following multimodal input embedding sequence

[T 1,…,T λ−1,V 1,…,V n,T λ+1,…,T m]subscript 𝑇 1…subscript 𝑇 𝜆 1 subscript 𝑉 1…subscript 𝑉 𝑛 subscript 𝑇 𝜆 1…subscript 𝑇 𝑚[T_{1},\ldots,T_{\lambda-1},V_{1},\ldots,V_{n},T_{\lambda+1},\ldots,T_{m}][ italic_T start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , … , italic_T start_POSTSUBSCRIPT italic_λ - 1 end_POSTSUBSCRIPT , italic_V start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , … , italic_V start_POSTSUBSCRIPT italic_n end_POSTSUBSCRIPT , italic_T start_POSTSUBSCRIPT italic_λ + 1 end_POSTSUBSCRIPT , … , italic_T start_POSTSUBSCRIPT italic_m end_POSTSUBSCRIPT ](4)

to the LLM. All tokens not only have the same dimensionality but are also generated in a similar manner with embedding tables. LLM will output a textual token sequence o 1,…,o l subscript 𝑜 1…subscript 𝑜 𝑙 o_{1},\ldots,o_{l}italic_o start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , … , italic_o start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT.

Ovis is trained in a three-stage fashion and optimized throughout with the textual generation loss, i.e., the cross-entropy between the textual tokens of Ovis output and that of the ground-truth text. The stages differ in their trainable parameters and the types of training data.

Stage 1. We freeze all parameters of LLM, as well as most parameters in the visual encoder g 𝑔 g italic_g, an open-source pre-trained ViT backbone. We randomly re-initialize the parameters within the last block of g 𝑔 g italic_g, and using visual caption datasets such as COYO [[7](https://arxiv.org/html/2405.20797v2#bib.bib7)] to train the re-initialized parameters as well as the projection 𝑾 𝑾\bm{W}bold_italic_W and the visual embedding table {𝒆 k}k=1 K superscript subscript subscript 𝒆 𝑘 𝑘 1 𝐾\{{\bm{e}}_{k}\}_{k=1}^{K}{ bold_italic_e start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_k = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_K end_POSTSUPERSCRIPT of Ovis. For each image in the caption dataset, we construct a training sample with input as “<image>’s caption: ” and label it as “CAPTION”, where CAPTION denotes the image’s caption.

Stage 2. In this stage, we advance the training of Ovis’s 𝑾 𝑾\bm{W}bold_italic_W, the visual embedding table {𝒆 k}k=1 K superscript subscript subscript 𝒆 𝑘 𝑘 1 𝐾\{{\bm{e}}_{k}\}_{k=1}^{K}{ bold_italic_e start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_k = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_K end_POSTSUPERSCRIPT, and all parameters in the vision encoder g 𝑔 g italic_g. The LLM is still frozen. In contrast to the caption samples used in the first stage, we leverage visual description datasets such as ShareGPT4V-Pretrain [[11](https://arxiv.org/html/2405.20797v2#bib.bib11)], which consist of training samples structured as dialogues that describe images.

Stage 3. After endowing Ovis with visual perception capabilities through training in Stage 1 and Stage 2, this stage focuses on multimodal instruction learning. The goal is to equip Ovis with the ability to follow multimodal instructions. To this end, we unfreeze the LLM module and train Ovis’s entire set of parameters on multimodal instruction datasets, such as LLaVA-Finetune [[47](https://arxiv.org/html/2405.20797v2#bib.bib47)].

4 Experiments
-------------

In this section, we provide empirical results to demonstrate the effectiveness of the proposed MLLM architecture Ovis.1 1 1 Qualitative results are presented in [Appendix A](https://arxiv.org/html/2405.20797v2#A1 "Appendix A Qualitative Results ‣ Ovis: Structural Embedding Alignment for Multimodal Large Language Model").

### 4.1 Experimental Setup

#### Implementation Details.

Ovis encompasses three configurations: the LLM module, the ViT backbone, and the visual vocabulary size. We incorporate popular open-source LLMs (Qwen1.5-Chat [[3](https://arxiv.org/html/2405.20797v2#bib.bib3)] and Llama3-Instruct [[56](https://arxiv.org/html/2405.20797v2#bib.bib56)]) and ViTs (Clip-ViT-L/14@336px [[66](https://arxiv.org/html/2405.20797v2#bib.bib66)]) into Ovis. The visual vocabulary size is set to 2 17=131,072 superscript 2 17 131 072 2^{17}=131,072 2 start_POSTSUPERSCRIPT 17 end_POSTSUPERSCRIPT = 131 , 072, a value comparable to LLMs’ textual vocabulary size. To facilitate community use and future innovation, the Ovis architecture and its training code are built upon the widely-used Transformers [[81](https://arxiv.org/html/2405.20797v2#bib.bib81)] and DeepSpeed [[67](https://arxiv.org/html/2405.20797v2#bib.bib67)] packages. We detail the training hyper-parameters for each stage in [Table 4](https://arxiv.org/html/2405.20797v2#A2.T4 "Table 4 ‣ Appendix B Training Details ‣ Ovis: Structural Embedding Alignment for Multimodal Large Language Model") of [Appendix B](https://arxiv.org/html/2405.20797v2#A2 "Appendix B Training Details ‣ Ovis: Structural Embedding Alignment for Multimodal Large Language Model").

#### Training Datasets.

Ovis is trained predominantly on open-source datasets, supplemented by a smaller proportion of in-house datasets. The datasets employed can be categorized into three groups: visual captions, visual descriptions, and multimodal instructions, which are respectively utilized in Stage 1, 2, and 3 of the training process. The visual caption dataset is extracted from the COYO dataset based on the similarity between the image and its caption. We leveraged the COYO dataset’s provided “clip-similarity-vitb32” and “clip-similarity-vitl14” scores for this purpose. Specifically, we select all entries from the COYO dataset with both similarity metrics exceeding 0.36 0.36 0.36 0.36. The visual description datasets and multimodal instruction datasets are all converted into the same format as LLaVA-Finetune [[47](https://arxiv.org/html/2405.20797v2#bib.bib47)]. Our in-house datasets are available at [https://huggingface.co/datasets/AIDC-AI/Ovis-dataset](https://huggingface.co/datasets/AIDC-AI/Ovis-dataset). We describe the constructions and present several samples from the in-house datasets in [Appendix C](https://arxiv.org/html/2405.20797v2#A3 "Appendix C In-house Visual Description Dataset ‣ Ovis: Structural Embedding Alignment for Multimodal Large Language Model") and [Appendix D](https://arxiv.org/html/2405.20797v2#A4 "Appendix D In-house Visual Instruction Dataset ‣ Ovis: Structural Embedding Alignment for Multimodal Large Language Model"). Statistics of the training dataset are reported in [Table 5](https://arxiv.org/html/2405.20797v2#A2.T5 "Table 5 ‣ Appendix B Training Details ‣ Ovis: Structural Embedding Alignment for Multimodal Large Language Model") of [Appendix B](https://arxiv.org/html/2405.20797v2#A2 "Appendix B Training Details ‣ Ovis: Structural Embedding Alignment for Multimodal Large Language Model").

### 4.2 Main Results

Table 1: Comparison with popular open-source and proprietary MLLMs on general multimodal benchmarks. MMBench is shortened to MMB due to width limitations. MMMU-V and MMMU-T denote the validation and test splits, respectively. The GPT4V-HR denotes GPT4V in the high-resolution mode. Within each parameter tier, the top-performing model is highlighted in bold, while the runner-up is marked with an underscore.

Table 2: Comparison with popular open-source and proprietary MLLMs on specialized multimodal benchmarks. The GPT4V-HR denotes GPT4V in the high-resolution mode. We report the sum of perception and cognition scores for MME and the QuestionAcc score for HallusionBench. Within each parameter tier, the top-performing and runner-up models are in bold and underscored, respectively.

MLLM MathVista-Mini MME HallusionBench RealWorldQA
open-source models in 7B tier
InstructBLIP-7B [[18](https://arxiv.org/html/2405.20797v2#bib.bib18)]24.4 1391 53.6 36.9
LLaVA-v1.5-7B [[47](https://arxiv.org/html/2405.20797v2#bib.bib47)]25.6 1808 48.8 54.8
ShareGPT4V-7B [[11](https://arxiv.org/html/2405.20797v2#bib.bib11)]26.5\ul 1915 48.8 54.9
Monkey [[41](https://arxiv.org/html/2405.20797v2#bib.bib41)]33.5 1760 55.1 51.6
Monkey-Chat [[41](https://arxiv.org/html/2405.20797v2#bib.bib41)]35.9 1887\ul 58.4 52.4
DeepSeek-VL-7B [[51](https://arxiv.org/html/2405.20797v2#bib.bib51)]36.9 1765 53.9 54.2
Yi-VL-6B [[88](https://arxiv.org/html/2405.20797v2#bib.bib88)]29.7\ul 1915 55.7 53.5
Qwen-VL-Chat [[4](https://arxiv.org/html/2405.20797v2#bib.bib4)]34.9 1860 56.4 49.3
Mini-Gemini-7B [[40](https://arxiv.org/html/2405.20797v2#bib.bib40)]-1839--
Mini-Gemini-HD-7B [[40](https://arxiv.org/html/2405.20797v2#bib.bib40)]-1865--
LLaVA-Next-Vicuna-7B [[49](https://arxiv.org/html/2405.20797v2#bib.bib49)]31.5 1769 47.2 57.8
LLaVA-Next-Mistral-7B [[49](https://arxiv.org/html/2405.20797v2#bib.bib49)]34.6 1821 47.9 60.0
LLaVA-Llama3-8B [[17](https://arxiv.org/html/2405.20797v2#bib.bib17)]40.0 1826 48.6 56.7
Ovis-Qwen1.5-7B 41.4 1882 56.4 60.0
Ovis-Llama3-8B\ul 40.8 2009 61.1\ul 57.9
open-source models in 14B tier
PandaGPT-13B [[72](https://arxiv.org/html/2405.20797v2#bib.bib72)]25.0 1072 43.1 32.8
LLaVA-v1.5-13B [[47](https://arxiv.org/html/2405.20797v2#bib.bib47)]27.7 1781 45.3 55.3
ShareGPT4V-13B [[11](https://arxiv.org/html/2405.20797v2#bib.bib11)]29.3 1853 48.8 57.0
Mini-Gemini-13B [[40](https://arxiv.org/html/2405.20797v2#bib.bib40)]-1887--
Mini-Gemini-HD-13B [[40](https://arxiv.org/html/2405.20797v2#bib.bib40)]-\ul 1917--
LLaVA-Next-Vicuna-13B [[49](https://arxiv.org/html/2405.20797v2#bib.bib49)]\ul 34.1 1746\ul 51.5\ul 57.6
Ovis-Qwen1.5-14B 43.4 1961 57.6 62.7
proprietary models
GPT4V [[59](https://arxiv.org/html/2405.20797v2#bib.bib59)]51.4 2038 60.1 61.4
GPT4V-HR [[59](https://arxiv.org/html/2405.20797v2#bib.bib59)]54.7 2070 62.1 68.0
Gemini-Pro [[75](https://arxiv.org/html/2405.20797v2#bib.bib75)]46.5 2149 63.7 60.4
Qwen-VL-Plus [[4](https://arxiv.org/html/2405.20797v2#bib.bib4)]37.6 2230 57.6 44.6
Qwen-VL-Max [[4](https://arxiv.org/html/2405.20797v2#bib.bib4)]43.4 2282 57.7 61.3

We evaluate Ovis across a variety of benchmarks, covering both general multimodal capabilities benchmarks (MMMU [[89](https://arxiv.org/html/2405.20797v2#bib.bib89)], MMBench-EN [[50](https://arxiv.org/html/2405.20797v2#bib.bib50)], MMBench-CN [[50](https://arxiv.org/html/2405.20797v2#bib.bib50)], and MMStar [[12](https://arxiv.org/html/2405.20797v2#bib.bib12)]), as well as benchmarks for more specialized multimodal tasks (MathVista-Mini [[54](https://arxiv.org/html/2405.20797v2#bib.bib54)], MME [[25](https://arxiv.org/html/2405.20797v2#bib.bib25)], HallusionBench [[46](https://arxiv.org/html/2405.20797v2#bib.bib46)], and RealWorldQA [[84](https://arxiv.org/html/2405.20797v2#bib.bib84)]). The evaluation is performed using the VLMEvalKit package [[16](https://arxiv.org/html/2405.20797v2#bib.bib16)]. The comparison between Ovis’ benchmark performance with that of popular open-source MLLMs and leading proprietary models is summarized in [Table 1](https://arxiv.org/html/2405.20797v2#S4.T1 "Table 1 ‣ 4.2 Main Results ‣ 4 Experiments ‣ Ovis: Structural Embedding Alignment for Multimodal Large Language Model") and [Table 2](https://arxiv.org/html/2405.20797v2#S4.T2 "Table 2 ‣ 4.2 Main Results ‣ 4 Experiments ‣ Ovis: Structural Embedding Alignment for Multimodal Large Language Model"), where the benchmark scores of the compared models are mainly derived from VLMEvalKit for consistency. MLLMs with no specialized multimodal benchmark performance reported are not included in [Table 2](https://arxiv.org/html/2405.20797v2#S4.T2 "Table 2 ‣ 4.2 Main Results ‣ 4 Experiments ‣ Ovis: Structural Embedding Alignment for Multimodal Large Language Model") for conciseness.

It can be seen that Ovis-8B outperforms the open-source models of similar size across the majority of benchmarks. Ovis-14B not only excels in all benchmarks but also surpasses the high-resource proprietary model Qwen-VL-Plus in most benchmarks. In the vision-indispensable multi-modal benchmark MMStar, Ovis-8B exhibits a large margin over the compared open-source MLLMs, highlighting its advantage in utilizing visual information. Ovis also achieves leading results in the highly challenging college-level MMMU benchmark, demonstrating strong visual comprehension and reasoning abilities. The MMBench-EN and MMBench-CN benchmarks differ only in the language [[50](https://arxiv.org/html/2405.20797v2#bib.bib50)]. While Ovis’ training dataset contains very few non-English samples, Ovis performs well in both versions. Ovis-14B achieves consistently outstanding performance in MMBench-EN and MMBench-CN, suggesting that Ovis’ advantage in multimodal capabilities is not limited to English but can extend to another language like Chinese as well.

Turning attention to the specialized multimodal benchmarks, we find that Ovis enjoys better multimodal capabilities in math and logical reasoning than open-source competitors, as demonstrated by its notable performance in the MathVista-Mini benchmark. While Ovis only employs a 336px ViT backbone and is not equipped with high-resolution-boosted techniques such as the dynamic high resolution used in LLaVA-Next [[49](https://arxiv.org/html/2405.20797v2#bib.bib49)] and the dual vision encoders used in Mini-Gemini-HD [[40](https://arxiv.org/html/2405.20797v2#bib.bib40)], Ovis exhibits impressive performance in the RealWorldQA benchmark, which is comprised of real-world visual tasks with high-resolution images (e.g., 1080P). Notably, Ovis-14B’s RealWorldQA score is even higher than the leading proprietary model GPT4V, illustrating its outstanding multimodal capabilities in solving practical visual tasks. In the MME and hallucination benchmarks, Ovis-8B and Ovis-14B perform the best within the 7B and 14B tier, respectively. This implies that Ovis’ strong visual understanding and reasoning abilities are accompanied by a lower rate of hallucination, a highly desirable trait for the deployment of MLLMs in critical scenarios such as medicine.

### 4.3 Ablation Study

Table 3: Comparison between Ovis and the conventional connector-based architecture, both employing Qwen1.5-7B-Chat and Clip-ViT-L/14@336px as backbones and trained on the same datasets. Due to width limitations, MMBench-EN and MMBench-CN are merged into a single column, as are MMMU-V and MMMU-T. MathVista-Mini, HallusionBench, and RealWorldQA are shortened to Math, HB, and RWQA, respectively.

To further elucidate the advantages of Ovis’ architectural design, we conduct a comparative experiment between Ovis-7B and a connector-based MLLM utilizing identical LLM and ViT backbones as Ovis-7B. Following [[47](https://arxiv.org/html/2405.20797v2#bib.bib47)], we implement the connector as a two-layer MLP with GELU activation. The hidden size of the MLP is configured to match Ovis-7B’s visual vocabulary size, ensuring parity in parameter count between the connector-based MLLM and Ovis-7B. We train the connector-based MLLM on the same datasets as Ovis-7B, adhering to the training paradigm outlined in [[47](https://arxiv.org/html/2405.20797v2#bib.bib47)]. The experimental results are summarized in [Table 3](https://arxiv.org/html/2405.20797v2#S4.T3 "Table 3 ‣ 4.3 Ablation Study ‣ 4 Experiments ‣ Ovis: Structural Embedding Alignment for Multimodal Large Language Model"). Remarkably, Ovis consistently outperforms the connector-based architecture across all benchmark evaluations, achieving an impressive 8.8% performance margin on average. Given the identical parameter counts, backbones, and training datasets, the results compellingly advocate for the efficacy of Ovis’ architectural design.

5 Conclusion
------------

We emphasize the necessity of structurally aligning visual embeddings with the textual counterparts, considering their different tokenization and embedding strategies in MLLMs. In Ovis, we introduce an additional visual embedding look-up table. Image patches are mapped into probabilistic tokens, which then index the visual embedding table and are transformed into a structural manner similar to textual embeddings. Empirical evaluations across various multimodal benchmarks validate Ovis’ effectiveness, demonstrating that it outperforms open-source MLLMs of similar parameter scales as well as the proprietary model Qwen-VL-Plus.

6 Broader Impact and Limitations
--------------------------------

#### Broader Impact.

As a powerful multimodal large language model architecture, Ovis has the potential to benefit a wide range of users through enhanced interactions between visual content and textual analysis. However, it is crucial to acknowledge the potential negative impacts associated with Ovis, such as the risk of hallucination, wherein Ovis may generate misleading or incorrect information, potentially leading to misinformation. Furthermore, Ovis also suffers from biases and potential harms, a common issue among generative models. These potential adverse effects could be mitigated through content moderation mechanisms and transparent model developments.

#### Limitations.

While Ovis has demonstrated promising performance, its efficacy in handling visual tasks with high-resolution images is limited due to the absence of high-resolution-boosted techniques. Moreover, Ovis is trained solely with single-image samples, posing challenges when confronted with scenarios requiring visual understanding across multiple images. Considerable research efforts have been dedicated to these areas [[49](https://arxiv.org/html/2405.20797v2#bib.bib49), [40](https://arxiv.org/html/2405.20797v2#bib.bib40), [43](https://arxiv.org/html/2405.20797v2#bib.bib43)], primarily within the connector-based framework. Drawing inspiration from these researches, we plan to enhance Ovis’ capacity to better handle high-resolution images and process multi-image inputs in future iterations.

References
----------

*   Alayrac et al. [2022] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. _NeurIPS_, 2022. 
*   anthropic [2024] anthropic. Introducing the next generation of claude. Technical report, anthropic, 2024. URL [https://www.anthropic.com/news/claude-3-family](https://www.anthropic.com/news/claude-3-family). 
*   Bai et al. [2023a] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. _arXiv:2309.16609_, 2023a. 
*   Bai et al. [2023b] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. _arXiv:2308.12966_, 2023b. 
*   Bao et al. [2021] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. _arXiv:2106.08254_, 2021. 
*   Brown et al. [2020] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. _NeurIPS_, 2020. 
*   Byeon et al. [2022] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. [https://github.com/kakaobrain/coyo-dataset](https://github.com/kakaobrain/coyo-dataset), 2022. 
*   Changpinyo et al. [2021] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12M: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In _CVPR_, 2021. 
*   Chen et al. [2024a] Guiming Hardy Chen, Shunian Chen, Ruifei Zhang, Junying Chen, Xiangbo Wu, Zhiyi Zhang, Zhihong Chen, Jianquan Li, Xiang Wan, and Benyou Wang. Allava: Harnessing gpt4v-synthesized data for a lite vision-language model. _arXiv:2402.11684_, 2024a. 
*   Chen et al. [2024b] Jinyue Chen, Lingyu Kong, Haoran Wei, Chenglong Liu, Zheng Ge, Liang Zhao, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Onechart: Purify the chart structural extraction via one auxiliary token. _arXiv:2404.09987_, 2024b. 
*   Chen et al. [2023] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. _arXiv:2311.12793_, 2023. 
*   Chen et al. [2024c] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? _arXiv:2403.20330_, 2024c. 
*   Chen et al. [2024d] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, Ji Ma, Jiaqi Wang, Xiaoyi Dong, Hang Yan, Hewei Guo, Conghui He, Botian Shi, Zhenjiang Jin, Chao Xu, Bin Wang, Xingjian Wei, Wei Li, Wenjian Zhang, Bo Zhang, Pinlong Cai, Licheng Wen, Xiangchao Yan, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. _arXiv:2404.16821_, 2024d. 
*   Chiang et al. [2023] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90% chatgpt quality, 2023. URL [https://vicuna.lmsys.org](https://vicuna.lmsys.org/). 
*   Chu et al. [2024] Xiangxiang Chu, Limeng Qiao, Xinyu Zhang, Shuang Xu, Fei Wei, Yang Yang, Xiaofei Sun, Yiming Hu, Xinyang Lin, Bo Zhang, et al. Mobilevlm v2: Faster and stronger baseline for vision language model. _arXiv:2402.03766_, 2024. 
*   Contributors [2023a] OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models. [https://github.com/open-compass/opencompass](https://github.com/open-compass/opencompass), 2023a. 
*   Contributors [2023b] XTuner Contributors. Xtuner: A toolkit for efficiently fine-tuning llm. [https://github.com/InternLM/xtuner](https://github.com/InternLM/xtuner), 2023b. 
*   Dai et al. [2023] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. _arXiv:2305.06500_, 2023. 
*   Demir and Unal [2018] Ugur Demir and Gozde Unal. Patch-based image inpainting with generative adversarial networks, 2018. 
*   Devlin et al. [2018] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. _arXiv:1810.04805_, 2018. 
*   Dong et al. [2024] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, et al. Internlm-xcomposer2-4khd: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd. _arXiv:2404.06512_, 2024. 
*   Dosovitskiy et al. [2020] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. _arXiv:2010.11929_, 2020. 
*   Esser et al. [2021] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_, pages 12873–12883, 2021. 
*   Fang et al. [2023] Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva-02: A visual representation for neon genesis. _arXiv:2303.11331_, 2023. 
*   Fu et al. [2023a] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Zhenyu Qiu, Wei Lin, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. _arXiv:2306.13394_, 2023a. 
*   Fu et al. [2023b] Chaoyou Fu, Renrui Zhang, Zihan Wang, Yubo Huang, Zhengye Zhang, Longtian Qiu, Gaoxiang Ye, Yunhang Shen, Mengdan Zhang, Peixian Chen, Sirui Zhao, Shaohui Lin, Deqiang Jiang, Di Yin, Peng Gao, Ke Li, Hongsheng Li, and Xing Sun. A challenger to gpt-4v? early explorations of gemini in visual expertise. _arXiv preprint arXiv:2312.12436_, 2023b. 
*   Gao et al. [2023] Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, et al. G-llava: Solving geometric problem with multi-modal large language model. _aarXiv:2312.11370_, 2023. 
*   Ge et al. [2023a] Yuying Ge, Yixiao Ge, Ziyun Zeng, Xintao Wang, and Ying Shan. Planting a seed of vision in large language model. _arXiv preprint arXiv:2307.08041_, 2023a. 
*   Ge et al. [2023b] Yuying Ge, Sijie Zhao, Ziyun Zeng, Yixiao Ge, Chen Li, Xintao Wang, and Ying Shan. Making llama see and draw with seed tokenizer. _arXiv:2310.01218_, 2023b. 
*   Hu et al. [2024a] Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, et al. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. _arXiv:2403.12895_, 2024a. 
*   Hu et al. [2024b] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. _arXiv:2404.06395_, 2024b. 
*   Jiang et al. [2023] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. _arXiv:2310.06825_, 2023. 
*   Jin et al. [2023] Yang Jin, Kun Xu, Liwei Chen, Chao Liao, Jianchao Tan, Bin Chen, Chenyi Lei, An Liu, Chengru Song, Xiaoqiang Lei, et al. Unified language-vision pretraining with dynamic discrete visual tokenization. _arXiv:2309.04669_, 2023. 
*   Kim et al. [2022] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In _European Conference on Computer Vision (ECCV)_, 2022. 
*   Kingma and Welling [2013] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. _arXiv:1312.6114_, 2013. 
*   Li et al. [2023a] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. _arXiv:2306.00890_, 2023a. 
*   Li et al. [2022] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In _ICML_, 2022. 
*   Li et al. [2023b] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. _arXiv:2301.12597_, 2023b. 
*   Li et al. [2024a] Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong Feng, Lingpeng Kong, and Qi Liu. Multimodal arxiv: A dataset for improving scientific comprehension of large vision-language models, 2024a. 
*   Li et al. [2024b] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. _arXiv:2403.18814_, 2024b. 
*   Li et al. [2023c] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. _arXiv:2311.06607_, 2023c. 
*   Lin et al. [2024] Bin Lin, Zhenyu Tang, Yang Ye, Jiaxi Cui, Bin Zhu, Peng Jin, Junwu Zhang, Munan Ning, and Li Yuan. Moe-llava: Mixture of experts for large vision-language models. _arXiv:2401.15947_, 2024. 
*   Lin et al. [2023] Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models, 2023. 
*   Liu et al. [2024a] Chaohu Liu, Kun Yin, Haoyu Cao, Xinghua Jiang, Xin Li, Yinsong Liu, Deqiang Jiang, Xing Sun, and Linli Xu. Hrvda: High-resolution visual document assistant. _arXiv:2404.06918_, 2024a. 
*   Liu et al. [2023a] Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Aligning large multi-modal model with robust instruction tuning. _arXiv:2306.14565_, 2023a. 
*   Liu et al. [2024b] Fuxiao Liu, Tianrui Guan, Zongxia Li, Lichang Chen, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: You see what you think? or you think what you see? an image-context reasoning benchmark challenging for gpt-4v (ision), llava-1.5, and other multi-modality models. In _CVPR_, 2024b. 
*   Liu et al. [2023b] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. _arXiv:2310.03744_, 2023b. 
*   Liu et al. [2023c] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _arXiv:2304.08485_, 2023c. 
*   Liu et al. [2024c] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024c. URL [https://llava-vl.github.io/blog/2024-01-30-llava-next/](https://llava-vl.github.io/blog/2024-01-30-llava-next/). 
*   Liu et al. [2023d] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? _arXiv:2307.06281_, 2023d. 
*   Lu et al. [2024a] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Yaofeng Sun, et al. Deepseek-vl: towards real-world vision-language understanding. _arXiv:2403.05525_, 2024a. 
*   Lu et al. [2023] Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision, language, audio, and action. _arXiv:2312.17172_, 2023. 
*   Lu et al. [2022] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. _Advances in Neural Information Processing Systems_, 35:2507–2521, 2022. 
*   Lu et al. [2024b] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In _International Conference on Learning Representations (ICLR)_, 2024b. 
*   Lu et al. [2024c] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In _ICLR_, 2024c. 
*   meta [2024] meta. Build the future of ai with meta llama 3. Technical report, meta, 2024. URL [https://llama.meta.com/llama3/](https://llama.meta.com/llama3/). 
*   Mistral [2024] Mistral. Mistral-8×22b. Technical report, Mistral, 2024. URL [https://mistral.ai/news/mixtral-8x22b/](https://mistral.ai/news/mixtral-8x22b/). 
*   OpenAI [2023a] OpenAI. Chatgpt: A language model for conversational ai. Technical report, OpenAI, 2023a. URL [https://www.openai.com/research/chatgpt](https://www.openai.com/research/chatgpt). 
*   OpenAI [2023b] OpenAI. Gpt-4 technical report. _arXiv:2303.08774_, 2023b. 
*   Ouyang et al. [2022] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. _NeurIPS_, 2022. 
*   Panagopoulou et al. [2023] Artemis Panagopoulou, Le Xue, Ning Yu, Junnan Li, Dongxu Li, Shafiq Joty, Ran Xu, Silvio Savarese, Caiming Xiong, and Juan Carlos Niebles. X-instructblip: A framework for aligning x-modal instruction-aware representations to llms and emergent cross-modal reasoning. _arXiv:2311.18799_, 2023. 
*   Peng et al. [2024] Tianshuo Peng, Zuchao Li, Lefei Zhang, Hai Zhao, Ping Wang, and Bo Du. Multi-modal auto-regressive modeling via visual words. _arXiv:2403.07720_, 2024. 
*   Peng et al. [2022] Z Peng, L Dong, H Bao, Q Ye, and F Wei. Beit v2: Masked image modeling with vector-quantized visual tokenizers. arxiv 2022. _arXiv:2208.06366_, 2022. 
*   Radford et al. [2018] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018. 
*   Radford et al. [2019] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. _OpenAI blog_, 1(8):9, 2019. 
*   Radford et al. [2021] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _ICML_, 2021. 
*   Rasley et al. [2020] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In _Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_, KDD ’20, page 3505–3506, New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450379984. doi: 10.1145/3394486.3406703. URL [https://doi.org/10.1145/3394486.3406703](https://doi.org/10.1145/3394486.3406703). 
*   Reid et al. [2024] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. _arXiv:2403.05530_, 2024. 
*   Russakovsky et al. [2015] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. _International Journal of Computer Vision_, 115(3):211–252, 2015. 
*   Schuhmann et al. [2022] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. _NeurIPS_, 2022. 
*   Singh et al. [2019] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_, pages 8317–8326, 2019. 
*   Su et al. [2023] Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. Pandagpt: One model to instruction-follow them all. _arXiv:2305.16355_, 2023. 
*   Sun et al. [2023] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. _arXiv:2303.15389_, 2023. 
*   Taori et al. [2023] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Stanford alpaca: An instruction-following llama model, 2023. 
*   Team et al. [2023] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. _arXiv:2312.11805_, 2023. 
*   Touvron et al. [2023a] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. _arXiv:2302.13971_, 2023a. 
*   Touvron et al. [2023b] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv:2307.09288_, 2023b. 
*   Van Den Oord et al. [2017] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. _Advances in neural information processing systems_, 30, 2017. 
*   Wang et al. [2023] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. _arXiv:2311.03079_, 2023. 
*   Wei et al. [2023] Haoran Wei, Lingyu Kong, Jinyue Chen, Liang Zhao, Zheng Ge, Jinrong Yang, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Vary: Scaling up the vision vocabulary for large vision-language models. _arXiv:2312.06109_, 2023. 
*   Wolf et al. [2020] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations_, pages 38–45, Online, October 2020. Association for Computational Linguistics. URL [https://www.aclweb.org/anthology/2020.emnlp-demos.6](https://www.aclweb.org/anthology/2020.emnlp-demos.6). 
*   Wu et al. [2023a] Haoning Wu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Annan Wang, Kaixin Xu, Chunyi Li, Jingwen Hou, Guangtao Zhai, Geng Xue, Wenxiu Sun, Qiong Yan, and Weisi Lin. Q-instruct: Improving low-level visual abilities for multi-modality foundation models, 2023a. 
*   Wu et al. [2023b] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal llm. _arXiv:2309.05519_, 2023b. 
*   x.ai [2024] x.ai. Grok-1.5v. [https://x.ai/blog/grok-1.5v](https://x.ai/blog/grok-1.5v), 2024. 
*   Yang et al. [2023a] Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, et al. Baichuan 2: Open large-scale language models. _arXiv:2309.10305_, 2023a. 
*   Yang et al. [2023b] Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v (ision). _arXiv:2309.17421_, 2023b. 
*   Yin et al. [2023] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. _arXiv preprint arXiv:2306.13549_, 2023. 
*   Young et al. [2024] Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, et al. Yi: Open foundation models by 01. ai. _arXiv:2403.04652_, 2024. 
*   Yue et al. [2024] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In _Proceedings of CVPR_, 2024. 
*   Zhai et al. [2023] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_, pages 11975–11986, 2023. 
*   Zhang et al. [2024] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? _arXiv preprint arXiv:2403.14624_, 2024. 
*   Zhu et al. [2023] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv:2304.10592_, 2023. 

Appendix A Qualitative Results
------------------------------

As shown in [Figure 4](https://arxiv.org/html/2405.20797v2#A1.F4 "Figure 4 ‣ Appendix A Qualitative Results ‣ Ovis: Structural Embedding Alignment for Multimodal Large Language Model") and [Figure 5](https://arxiv.org/html/2405.20797v2#A1.F5 "Figure 5 ‣ Appendix A Qualitative Results ‣ Ovis: Structural Embedding Alignment for Multimodal Large Language Model"), Ovis-Llama3-8B performs well in various multimodal tasks, where the images and prompts are sourced from literature.

![Image 7: Refer to caption](https://arxiv.org/html/2405.20797v2/x8.png)

(b)Reasoning [[75](https://arxiv.org/html/2405.20797v2#bib.bib75)]

![Image 8: Refer to caption](https://arxiv.org/html/2405.20797v2/x9.png)

(c)Coding [[84](https://arxiv.org/html/2405.20797v2#bib.bib84)]

Figure 4: Qualitative results with Ovis-Llama3-8B, part I

![Image 9: Refer to caption](https://arxiv.org/html/2405.20797v2/x10.png)

(a)Math [[91](https://arxiv.org/html/2405.20797v2#bib.bib91)]

![Image 10: Refer to caption](https://arxiv.org/html/2405.20797v2/x11.png)

(b)Science [[13](https://arxiv.org/html/2405.20797v2#bib.bib13)]

![Image 11: Refer to caption](https://arxiv.org/html/2405.20797v2/x12.png)

(c)Cooking scenario [[75](https://arxiv.org/html/2405.20797v2#bib.bib75)]

Figure 5: Qualitative results with Ovis-Llama3-8B, part II

Appendix B Training Details
---------------------------

Table 4: Training hyper-parameters

Hyper-parameter Stage 1 Stage 2 Stage 3
batch size 8192 1024 1024
learning rate (Ovis-Qwen1.5-7B/14B)1e-4 1e-4 2e-5
learning rate (Ovis-Llama3-8B)1e-4 1e-4 1e-5
learning rate schedule cosine cosine cosine
learning rate warm-up ratio 0.1 0.1 0.05
weight decay 0 0 0
grad norm clipping 1.0 1.0 1.0
epoch 1 1 1
optimizer AdamW AdamW AdamW
float precision bfloat16 bfloat16 bfloat16
deepspeed configuration (Ovis-7B/8B)zero2 zero3 zero3
deepspeed configuration (Ovis-14B)zero3 zero3 zero3
training hours (Ovis-7B/8B, 128 H100 GPUs)6 2 7
training hours (Ovis-14B, 128 H100 GPUs)10 6 21

Table 5: Statistics of the training dataset with ⋆⋆\star⋆ denoting in-house data

Dataset Category Dataset Name#Samples Total Size
Visual Caption COYO-10M [[7](https://arxiv.org/html/2405.20797v2#bib.bib7)]10M 10M
Visual Description LLaVA-Pretrain [[47](https://arxiv.org/html/2405.20797v2#bib.bib47)]558K 2M
ShareGPT4V-Pretrain [[11](https://arxiv.org/html/2405.20797v2#bib.bib11)]82K
ALLaVA-Caption-Laion-4V [[9](https://arxiv.org/html/2405.20797v2#bib.bib9)]485K
ALLaVA-Caption-Vflan-4V [[9](https://arxiv.org/html/2405.20797v2#bib.bib9)]203K
Laion-Description⋆11K
CC12M-Description⋆1M
Multimodal Instruction ScienceQA-Train-Val [[53](https://arxiv.org/html/2405.20797v2#bib.bib53)]17K 3M
TextVQA-Train [[71](https://arxiv.org/html/2405.20797v2#bib.bib71)]35K
ALLaVA-Instruct-Laion-4V [[9](https://arxiv.org/html/2405.20797v2#bib.bib9)]485K
ALLaVA-Instruct-Vflan-4V [[9](https://arxiv.org/html/2405.20797v2#bib.bib9)]203K
ArXivQA [[39](https://arxiv.org/html/2405.20797v2#bib.bib39)]100K
Q-Instruct [[82](https://arxiv.org/html/2405.20797v2#bib.bib82)]198K
LLaVA-Finetune [[47](https://arxiv.org/html/2405.20797v2#bib.bib47)]665K
Geo [[27](https://arxiv.org/html/2405.20797v2#bib.bib27)]177K
LRV-Instruction [[45](https://arxiv.org/html/2405.20797v2#bib.bib45)]300K
Chart-Instruction [[45](https://arxiv.org/html/2405.20797v2#bib.bib45)]43K
Synthdog-EN-OCR [[34](https://arxiv.org/html/2405.20797v2#bib.bib34)]200K
ALLaVA-Evol-Instruct [[9](https://arxiv.org/html/2405.20797v2#bib.bib9)]143K
CC12M-QA⋆387K

Appendix C In-house Visual Description Dataset
----------------------------------------------

We sample images from the Laion [[70](https://arxiv.org/html/2405.20797v2#bib.bib70)] and CC12M [[8](https://arxiv.org/html/2405.20797v2#bib.bib8)] datasets, which cover various categories, including nature, lifestyle, humanities, architecture, cartoons, and abstract art, as shown in Figure[6](https://arxiv.org/html/2405.20797v2#A3.F6 "Figure 6 ‣ Appendix C In-house Visual Description Dataset ‣ Ovis: Structural Embedding Alignment for Multimodal Large Language Model"). For each image, we call the Gemini-Pro or GPT-4V API with a unified prompt to generate the image’s descriptions. The unified prompt explicitly requires the API to reply with concise and clear visual information about the image, as well as performing OCR recognition if relevant, while avoiding embellishments and interpretations.

![Image 12: Refer to caption](https://arxiv.org/html/2405.20797v2/x13.png)

Figure 6: Samples from our in-house visual description dataset

Appendix D In-house Visual Instruction Dataset
----------------------------------------------

We create visual instruction samples for images from the CC12M dataset [[8](https://arxiv.org/html/2405.20797v2#bib.bib8)] in a similar way to [[9](https://arxiv.org/html/2405.20797v2#bib.bib9)], using Gemini-Pro and GPT-4V to carry out a self-questioning and answering task. This method produces diverse questions and high-quality answers, as illustrated in Figure[7](https://arxiv.org/html/2405.20797v2#A4.F7 "Figure 7 ‣ Appendix D In-house Visual Instruction Dataset ‣ Ovis: Structural Embedding Alignment for Multimodal Large Language Model").

![Image 13: Refer to caption](https://arxiv.org/html/2405.20797v2/x14.png)

Figure 7: Samples from our in-house visual instruction dataset

Appendix E Sparsity of Ovis’ Visual Tokenizer
---------------------------------------------

To assess the sparsity of Ovis’ visual tokenizer, we conduct an experiment using 10,000 images sampled from the ImageNet-1K dataset [[69](https://arxiv.org/html/2405.20797v2#bib.bib69)]. Each image is tokenized by the visual tokenizer of Ovis-Llama3-8B, resulting in a sequence of visual tokens, each of which is a probability distribution over the visual vocabulary. We then employ thresholds of 1e-4, 1e-5, and 1e-6 to categorize the probability values and calculate the ratio of values falling within each interval across the 10,000 images. As shown in [Figure 8](https://arxiv.org/html/2405.20797v2#A5.F8 "Figure 8 ‣ Appendix E Sparsity of Ovis’ Visual Tokenizer ‣ Ovis: Structural Embedding Alignment for Multimodal Large Language Model"), the probability distributions characterized by the visual tokens are highly sparse, with only 0.22% probability values exceeding the threshold of 1e-4.

![Image 14: Refer to caption](https://arxiv.org/html/2405.20797v2/x15.png)

Figure 8: Statistics of the probability distributions characterized by the visual tokens across 10,000 images from ImageNet-1K. The visual tokens are obtained using Ovis-Llama3-8B’s visual tokenizer.

