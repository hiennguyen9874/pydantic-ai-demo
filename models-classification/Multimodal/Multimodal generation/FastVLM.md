Title: FastVLM: Efficient Vision Encoding for Vision Language Models

URL Source: https://arxiv.org/html/2412.13303

Published Time: Mon, 19 May 2025 00:08:32 GMT

Markdown Content:
FastVLM: Efficient Vision Encoding for Vision Language Models
===============

1.   [1 Introduction](https://arxiv.org/html/2412.13303v2#S1 "In FastVLM: Efficient Vision Encoding for Vision Language Models")
2.   [2 Related Works](https://arxiv.org/html/2412.13303v2#S2 "In FastVLM: Efficient Vision Encoding for Vision Language Models")
3.   [3 Architecture](https://arxiv.org/html/2412.13303v2#S3 "In FastVLM: Efficient Vision Encoding for Vision Language Models")
    1.   [3.1 FastViT as VLM Image Encoder](https://arxiv.org/html/2412.13303v2#S3.SS1 "In 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")
        1.   [3.1.1 Multi-Scale Features](https://arxiv.org/html/2412.13303v2#S3.SS1.SSS1 "In 3.1 FastViT as VLM Image Encoder ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")

    2.   [3.2 FastViTHD: High Resolution Encoder for VLM](https://arxiv.org/html/2412.13303v2#S3.SS2 "In 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")
        1.   [3.2.1 Vision Encoder - Language Decoder Interplay](https://arxiv.org/html/2412.13303v2#S3.SS2.SSS1 "In 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")
        2.   [3.2.2 Static vs. Dynamic Input Resolution](https://arxiv.org/html/2412.13303v2#S3.SS2.SSS2 "In 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")
        3.   [3.2.3 Comparison with Token Pruning & Downsampling](https://arxiv.org/html/2412.13303v2#S3.SS2.SSS3 "In 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")

4.   [4 Experiments](https://arxiv.org/html/2412.13303v2#S4 "In FastVLM: Efficient Vision Encoding for Vision Language Models")
    1.   [4.1 Comparison with state-of-the-art](https://arxiv.org/html/2412.13303v2#S4.SS1 "In 4 Experiments ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")

5.   [5 Conclusion](https://arxiv.org/html/2412.13303v2#S5 "In FastVLM: Efficient Vision Encoding for Vision Language Models")
6.   [A [Training Setup](https://arxiv.org/html/2412.13303)](https://arxiv.org/html/2412.13303v2#S1a "In FastVLM: Efficient Vision Encoding for Vision Language Models")
7.   [B [Architecture Details](https://arxiv.org/html/2412.13303)](https://arxiv.org/html/2412.13303v2#S2a "In FastVLM: Efficient Vision Encoding for Vision Language Models")
    1.   [B.1 Naive Scaling](https://arxiv.org/html/2412.13303v2#S2.SS1 "In B Architecture Details ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")

8.   [C Additional Results](https://arxiv.org/html/2412.13303v2#S3a "In FastVLM: Efficient Vision Encoding for Vision Language Models")
    1.   [C.1 [Dynamic Resolution (AnyRes) Results](https://arxiv.org/html/2412.13303)](https://arxiv.org/html/2412.13303v2#S3.SS1a "In C Additional Results ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")
    2.   [C.2 [CVBench and MathVista Results](https://arxiv.org/html/2412.13303)](https://arxiv.org/html/2412.13303v2#S3.SS2a "In C Additional Results ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")

9.   [D [Datasets](https://arxiv.org/html/2412.13303)](https://arxiv.org/html/2412.13303v2#S4a "In FastVLM: Efficient Vision Encoding for Vision Language Models")
    1.   [D.1 Pretraining Datasets](https://arxiv.org/html/2412.13303v2#S4.SS1a "In D Datasets ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")
    2.   [D.2 Visual Instruction Tuning Datasets](https://arxiv.org/html/2412.13303v2#S4.SS2 "In D Datasets ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")
    3.   [D.3 [Evaluations](https://arxiv.org/html/2412.13303)](https://arxiv.org/html/2412.13303v2#S4.SS3 "In D Datasets ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")

10.   [E [Qualitative Analysis](https://arxiv.org/html/2412.13303)](https://arxiv.org/html/2412.13303v2#S5a "In FastVLM: Efficient Vision Encoding for Vision Language Models")

FastVLM: Efficient Vision Encoding for Vision Language Models
=============================================================

Pavan Kumar Anasosalu Vasu⋆†Fartash Faghri⋆Chun-Liang Li⋆Cem Koc⋆Nate True Albert Antony Gokul Santhanam James Gabriel Peter Grasch Oncel Tuzel⋆Hadi Pouransari⋆†Apple 

{panasosaluvasu,fartash,chunliang_li,cem_koc,otuzel,mpouransari}@apple.com

⋆Core authors; †Project lead 

###### Abstract

Scaling the input image resolution is essential for enhancing the performance of Vision Language Models (VLMs), particularly in text-rich image understanding tasks. However, popular visual encoders such as ViTs become inefficient at high resolutions due to the large number of tokens and high encoding latency. At different operational resolutions, the vision encoder of a VLM can be optimized along two axes: reducing encoding latency and minimizing the number of visual tokens passed to the LLM, thereby lowering overall latency. Based on a comprehensive efficiency analysis of the interplay between image resolution, vision latency, token count, and LLM size, we introduce FastVLM—a model that achieves an optimized trade-off between resolution, latency, and accuracy. FastVLM incorporates FastViTHD, a novel hybrid vision encoder designed to output fewer tokens and significantly reduce encoding time for high-resolution images. Unlike previous methods, FastVLM achieves the optimal balance between visual token count and image resolution solely by scaling the input image, eliminating the need for additional token pruning and simplifying the model design. In the LLaVA-1.5 setup, FastVLM achieves 3.2×\times× improvement in time-to-first-token (TTFT) while maintaining similar performance on VLM benchmarks compared to prior works. Compared to LLaVa-OneVision at the highest resolution (1152×\times×1152), FastVLM achieves better performance on key benchmarks like SeedBench, MMMU and DocVQA, using the same 0.5B LLM, but with 85×\times× faster TTFT and a vision encoder that is 3.4×\times× smaller. Code and models are available at [https://github.com/apple/ml-fastvlm](https://github.com/apple/ml-fastvlm).

1 Introduction
--------------

![Image 1: Refer to caption](https://arxiv.org/html/x1.png)

(a)Qwen2-0.5B

![Image 2: Refer to caption](https://arxiv.org/html/x2.png)

(b)Vicuna-7B

Figure 1: FastVLM is more than 3×\times× faster than prior work. Comparison of commonly used vision encoders for VLMs with (a) Qwen2[[86](https://arxiv.org/html/2412.13303v2#bib.bib86)] 0.5B LLM and (b) Vicuna 7B[[98](https://arxiv.org/html/2412.13303v2#bib.bib98)] LLM. All the vision encoders are CLIP[[69](https://arxiv.org/html/2412.13303v2#bib.bib69)] pretrained. For a fair comparison all models are trained using LLaVA-1.5[[53](https://arxiv.org/html/2412.13303v2#bib.bib53)] setup with the vision encoders made trainable for resolution adaptation, see [Sec.4](https://arxiv.org/html/2412.13303v2#S4 "4 Experiments ‣ FastVLM: Efficient Vision Encoding for Vision Language Models") for more details. Marker size for each model corresponds to number of parameters of the vision encoder. The x 𝑥 x italic_x-axis is the sum of vision encoder latency and LLM prefilling time. All models are benchmarked on an M1 Macbook Pro. 

Vision Language Models (VLMs) enable visual understanding alongside textual inputs. VLMs are often built by passing visual tokens from a pretrained vision backbone to a pretrained Large Language Model (LLM) through a projection layer. Previous works[[54](https://arxiv.org/html/2412.13303v2#bib.bib54), [53](https://arxiv.org/html/2412.13303v2#bib.bib53)] have explored various training and fine-tuning strategies for these three components: the vision backbone, the projection layer, and the LLM, which is typically a decoder-only transformer[[84](https://arxiv.org/html/2412.13303v2#bib.bib84)] model.

Several studies[[61](https://arxiv.org/html/2412.13303v2#bib.bib61), [28](https://arxiv.org/html/2412.13303v2#bib.bib28), [66](https://arxiv.org/html/2412.13303v2#bib.bib66)] highlight image resolution as a key factor in VLM performance, especially for text and chart-rich data. However, increasing image resolution presents multiple challenges. First, pretrained vision encoders may not support high-resolution images, as this would make pretraining inefficient. To address this, one approach is to continuously pretrain the vision backbone to adapt it for high resolutions[[6](https://arxiv.org/html/2412.13303v2#bib.bib6)]. Alternatively, tiling strategies, such as Sphinx[[52](https://arxiv.org/html/2412.13303v2#bib.bib52)], S2[[72](https://arxiv.org/html/2412.13303v2#bib.bib72)], and AnyRes[[53](https://arxiv.org/html/2412.13303v2#bib.bib53)], divide images into subregions, with each subregion processed independently by the backbone.

A further challenge is the runtime computational cost associated with high-resolution inference. Both single high-resolution inference and multiple inferences at lower resolution (the tiling strategy) result in significant latency when generating visual tokens. Additionally, high-resolution images naturally produce more tokens, which increases the LLM prefilling time (the LLM forward pass time on all tokens in the context, including visual tokens), thereby further increasing the time-to-first-token (TTFT), which is the sum of the vision encoder latency and the LLM prefilling time.

In this work, we study VLM design and training from a runtime efficiency perspective. We explore the optimization landscape as image resolution increases, aiming to improve accuracy-latency trade-off, where latency includes both the vision encoder inference time and the LLM prefilling time. Using extensive experiments with different LLM sizes and resolutions, we establish the Pareto optimal curve for a specific vision backbone, showing the best accuracy achievable within a given runtime budget (TTFT) based on different choices of resolution and LLM size.

We start by exploring the use of a hybrid convolutional-transformer architecture FastViT[[82](https://arxiv.org/html/2412.13303v2#bib.bib82)], pretrained with MobileCLIP[[83](https://arxiv.org/html/2412.13303v2#bib.bib83)], as a vision backbone for the VLM setup ([Section 3.1](https://arxiv.org/html/2412.13303v2#S3.SS1 "3.1 FastViT as VLM Image Encoder ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")). We demonstrate the potential of this hybrid backbone, which generates visual tokens over 4×\times× faster than a ViT model while achieving higher overall VLM accuracy with multi-scale features ([Section 3.1.1](https://arxiv.org/html/2412.13303v2#S3.SS1.SSS1 "3.1.1 Multi-Scale Features ‣ 3.1 FastViT as VLM Image Encoder ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")).

However, further architectural optimization is possible when the primary goal is a high-resolution VLM (rather than embedding generation as in MobileCLIP-pretrained FastViT). We introduce a new hybrid vision encoder, FastViTHD, specifically designed for efficient VLM performance on high-resolution images ([Section 3.2](https://arxiv.org/html/2412.13303v2#S3.SS2 "3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")), and use it as the vision backbone to obtain FastVLM through visual instruction tuning. FastVLM demonstrates a significantly improved accuracy-latency trade-off over VLMs based on ViTs, convolutional encoders, and our previously discussed hybrid FastViT for different input image resolutions and LLM sizes ([Figure 1(a)](https://arxiv.org/html/2412.13303v2#S1.F1.sf1 "In Figure 1 ‣ 1 Introduction ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), [Figs.1(b)](https://arxiv.org/html/2412.13303v2#S1.F1.sf2 "In Figure 1 ‣ 1 Introduction ‣ FastVLM: Efficient Vision Encoding for Vision Language Models") and[4](https://arxiv.org/html/2412.13303v2#S3.F4 "Figure 4 ‣ 3.2.1 Vision Encoder - Language Decoder Interplay ‣ 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")). In particular, FastVLM outperforms several prior works while being smaller, faster, and trained with less data ([Table 6](https://arxiv.org/html/2412.13303v2#S3.T6 "In 3.2.3 Comparison with Token Pruning & Downsampling ‣ 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")). Compared to LLaVA-OneVision[[45](https://arxiv.org/html/2412.13303v2#bib.bib45)] operating at the highest possible resolution (1152×\times×1152), FastVLM obtains comparable performance with the same 0.5B LLM, but with 85×\times× faster TTFT and a 3.4×\times× smaller vision encoder.

The following is a summary of our contributions:

*   •We show that hybrid vision backbones outperform ViTs in VLMs, and introduce additional architectural interventions, such as multi-scale vision features, to further improve VLM performance while maintaining efficiency. 
*   •We design and pretrain a new hybrid architecture, FastViTHD, optimized for efficient VLM performance with high resolution input for FastVLM. In a controlled experimental setup, where only the vision backbone is changed, we show that FastViTHD outperforms its ViT-based and convolution-based counterparts when used in VLMs: achieving 3.2×\times× faster TTFT and 3.6×\times× smaller size than SigLIP-SO400M[[94](https://arxiv.org/html/2412.13303v2#bib.bib94)], and 2.3×\times× faster TTFT and 1.7×\times× smaller size than ConvNeXT[[28](https://arxiv.org/html/2412.13303v2#bib.bib28)]. We further demonstrate that FastVLM scales effectively as more visual instruction tuning data becomes available. 
*   •We systematically study the VLM accuracy-latency trade-off by considering both the vision backbone latency and the LLM prefilling time on actual hardware benchmarks. Our results demonstrate an improved resolution-latency-accuracy trade-off achieved by FastVLM, measured on-device rather than estimates. 

2 Related Works
---------------

Large Multimodal Models. With the emergence of large language models[[79](https://arxiv.org/html/2412.13303v2#bib.bib79), [86](https://arxiv.org/html/2412.13303v2#bib.bib86), [77](https://arxiv.org/html/2412.13303v2#bib.bib77), [98](https://arxiv.org/html/2412.13303v2#bib.bib98), [68](https://arxiv.org/html/2412.13303v2#bib.bib68)] and large pretrained vision models, such as CLIP[[69](https://arxiv.org/html/2412.13303v2#bib.bib69)], trained on web-scale image-text datasets, several multimodal architectures have been proposed to encode images aligned with a large language model (LLM) to enable the interpretation of visual signals. Earlier works like Frozen[[80](https://arxiv.org/html/2412.13303v2#bib.bib80)] and Florence[[1](https://arxiv.org/html/2412.13303v2#bib.bib1), [2](https://arxiv.org/html/2412.13303v2#bib.bib2)] used a cross-attention mechanism where the image embeddings are fused with text embeddings in intermediate layers of the LLM. More recently, auto-regressive architectures have gained popularity where the image embedding is fed alongside text as input to an LLM. Some prominent works that use this architecture are LLaVA[[54](https://arxiv.org/html/2412.13303v2#bib.bib54), [53](https://arxiv.org/html/2412.13303v2#bib.bib53), [55](https://arxiv.org/html/2412.13303v2#bib.bib55)], mPLUG-Owl[[89](https://arxiv.org/html/2412.13303v2#bib.bib89), [90](https://arxiv.org/html/2412.13303v2#bib.bib90), [88](https://arxiv.org/html/2412.13303v2#bib.bib88)], InstructBLIP[[20](https://arxiv.org/html/2412.13303v2#bib.bib20)], BLIP-3[[85](https://arxiv.org/html/2412.13303v2#bib.bib85)], SPHINX[[52](https://arxiv.org/html/2412.13303v2#bib.bib52)], MiniGPT-4[[99](https://arxiv.org/html/2412.13303v2#bib.bib99)], VILA[[50](https://arxiv.org/html/2412.13303v2#bib.bib50)], MM1[[66](https://arxiv.org/html/2412.13303v2#bib.bib66)], Qwen-VL[[4](https://arxiv.org/html/2412.13303v2#bib.bib4)], InternVL[[15](https://arxiv.org/html/2412.13303v2#bib.bib15), [16](https://arxiv.org/html/2412.13303v2#bib.bib16)] and Cambrian-1[[78](https://arxiv.org/html/2412.13303v2#bib.bib78)]. Recently, Fuyu[[5](https://arxiv.org/html/2412.13303v2#bib.bib5)] and EVE[[23](https://arxiv.org/html/2412.13303v2#bib.bib23)] introduced a simplified architecture that passes raw images directly to the LLM decoder. Chameleon[[76](https://arxiv.org/html/2412.13303v2#bib.bib76)] introduced early fusion mixed-modal models where images are tokenized using a pretrained codebook. While skipping the image encoder is an intriguing approach, the performance of this new class of models lags behind architectures that use a pretrained image encoder.

Efficient Image Encoding. CLIP[[69](https://arxiv.org/html/2412.13303v2#bib.bib69)] pretrained vision transformers[[24](https://arxiv.org/html/2412.13303v2#bib.bib24)] are widely used for encoding images in vision-language models, with popular choices including SigLIP[[94](https://arxiv.org/html/2412.13303v2#bib.bib94)], EVA-CLIP[[75](https://arxiv.org/html/2412.13303v2#bib.bib75)], InternViT[[15](https://arxiv.org/html/2412.13303v2#bib.bib15)] and DFN-CLIP[[26](https://arxiv.org/html/2412.13303v2#bib.bib26)]. To enhance performance, recent works[[36](https://arxiv.org/html/2412.13303v2#bib.bib36), [78](https://arxiv.org/html/2412.13303v2#bib.bib78), [73](https://arxiv.org/html/2412.13303v2#bib.bib73)] employ ensembles of vision encoders trained with different objectives. These works are orthogonal to our work as they can benefit from using an efficient vision encoder among the ensemble of vision encoders. Since ViT-based architectures are a popular choice for VLMs, inefficiencies arise from the number of visual tokens outputted by the encoder, prompting methods like LLaVA-PruMerge[[70](https://arxiv.org/html/2412.13303v2#bib.bib70)] and Matryoshka-based token sampling[[32](https://arxiv.org/html/2412.13303v2#bib.bib32), [7](https://arxiv.org/html/2412.13303v2#bib.bib7)] to dynamically prune tokens. Other approaches[[20](https://arxiv.org/html/2412.13303v2#bib.bib20), [9](https://arxiv.org/html/2412.13303v2#bib.bib9), [18](https://arxiv.org/html/2412.13303v2#bib.bib18), [19](https://arxiv.org/html/2412.13303v2#bib.bib19)] reduce tokens using perceiver-style resamplers or pooling techniques. Rather than using an isotropic architecture like ViT and then designing custom resamplers and projectors, hierarchical architectures can be a simpler design choice. Hierarchical backbones like ConvNeXT[[57](https://arxiv.org/html/2412.13303v2#bib.bib57)] and FastViT[[82](https://arxiv.org/html/2412.13303v2#bib.bib82)] produce fewer tokens as they downsample the input tensor at every stage of compute. Recently, ConvLLaVA[[28](https://arxiv.org/html/2412.13303v2#bib.bib28)] was introduced that uses a pure-convolutional vision encoder to encode images for a VLM. In our work, we introduce an improved convolution-transformer hybrid architecture for VLMs and discuss the Pareto optimal operating points when this architecture is scaled to higher input resolutions.

3 Architecture
--------------

![Image 3: Refer to caption](https://arxiv.org/html/x3.png)

Figure 2: Overview of the FastVLM architecture. FastVLM consists of our novel vision encoder, FastViTHD, trained using the same setup as LLaVA. The FastViTHD architecture is designed for low latency at high resolution, by utilizing additional self-attention layers, and downsampling to generate 4×\times× fewer tokens than FastViT, and 16×\times× fewer tokens than ViT-L/14 at resolution 336. 

In this section, we first explore the adoption of the FastViT hybrid vision encoder for vision-language modeling. We then introduce architectural interventions to improve performance on VLM tasks. We present FastViTHD, a new hybrid vision encoder designed for an efficient high-resolution VLM. We provide comprehensive ablations to demonstrate the optimality of FastViTHD over FastViT and prior works for different LLMs and input resolutions. [Figure 2](https://arxiv.org/html/2412.13303v2#S3.F2 "In 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models") illustrates the overall architecture of FastVLM and FastViTHD. The training setup for all results in this section follows the same configuration as LLaVA-1.5[[53](https://arxiv.org/html/2412.13303v2#bib.bib53)] with Vicuna-7B[[98](https://arxiv.org/html/2412.13303v2#bib.bib98)] as the LLM decoder, unless mentioned otherwise. See [Sec.4](https://arxiv.org/html/2412.13303v2#S4 "4 Experiments ‣ FastVLM: Efficient Vision Encoding for Vision Language Models") for more details.

### 3.1 FastViT as VLM Image Encoder

VLMs such as LLaVA have three main components: an image encoder, a vision-language projector, and a large language model (LLM). Both the performance and runtime efficiency of a VLM highly depend on its vision backbone. Encoding images at high resolution is essential for achieving strong performance across various VLM benchmarks, especially for text-rich tasks. Therefore, a vision encoder with scalable resolution is particularly beneficial for VLMs. We identify hybrid vision encoders (convolutional layers followed by transformer blocks) as an ideal candidate for VLMs, as their convolutional component enables native resolution scaling, and their transformer blocks further refine high-quality visual tokens for consumption by the LLM.

We use a CLIP-pretrained hybrid vision encoder, specifically the MCi2 image encoder from MobileCLIP[[83](https://arxiv.org/html/2412.13303v2#bib.bib83)], which has 35.7M parameters and is based on the FastViT architecture. For simplicity, we refer to this encoder as “FastViT” throughout the rest of the paper. As shown in [Tab.1](https://arxiv.org/html/2412.13303v2#S3.T1 "In 3.1 FastViT as VLM Image Encoder ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), using FastViT at its CLIP-pretrained resolution (256×\times×256) alone does not yield a strong VLM. The main advantage of a hybrid encoder like FastViT lies in its favorable image resolution scaling characteristics, meaning it generates 5.2×\times× fewer tokens than the ViT architecture with a patch size of 14. The token reduction gives significant advantage to VLMs, as it reduces the prefilling time and time-to-first-token (TTFT) of the transformer decoders. When the input resolution of FastViT is scaled to 768×\times×768, it produces the same number of visual tokens as ViT-L/14 with an input resolution of 336×\times×336 but achieves better performance on VLM benchmarks. This performance gap is even more pronounced on text-rich benchmarks like TextVQA and DocVQA, despite both architectures producing the same number of visual tokens. Moreover, even with the same token count at higher resolution, it encodes images much faster due to efficient convolution layers.

| Image | Input | #Visual | Latency | GQA | TextVQA | POPE | DocVQA | Seed | Avg-5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Encoder | Res. | Tokens | Enc.(ms)↓↓\downarrow↓ | Bench I |
| ViT-L/14 | 336 | 576 | 127.4 | 62.0 | 58.2 | 85.9 | 28.1 | 66.1 | 60.1 |
| ViT-L/14† | 336 | 576 | 127.4 | 63.5 | 59.2 | 86.3 | 28.7 | 68.6 | 61.2 |
| FastViT | 256 | 64 | 3.0 | 60.2 | 51.6 | 82.9 | 15.8 | 61.5 | 54.4 |
| FastViT | 768 | 576 | 34.5 | 62.7 | 62.3 | 86.5 | 34.4 | 67.1 | 62.6 |

Table 1: FastViT has higher accuracy than ViT-L/14 at near 4×\times× lower latency. To scale resolution up to 768, FastViT is made trainable during Stage-2 training of LLaVA-1.5 setup. †To have a fair comparison, we also report the performance of ViT-L/14 finetuned during Stage-2 training of LLaVA-1.5. All latencies are reported in milliseconds. See [Sec.4](https://arxiv.org/html/2412.13303v2#S4 "4 Experiments ‣ FastVLM: Efficient Vision Encoding for Vision Language Models") for details. 

| Image | Multi | Pool | GQA | TextVQA | POPE | DocVQA | Seed | Avg-5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Encoder | Scale | Type | Bench I |
| FastViT |  | - | 62.7 | 62.3 | 86.5 | 34.4 | 67.1 | 62.6 |
| FastViT | ✓ | AvgPool | 63.0 | 62.2 | 86.2 | 35.1 | 66.9 | 62.7 |
| FastViT | ✓ | DWConv | 63.0 | 62.5 | 86.8 | 34.7 | 67.4 | 62.9 |

Table 2: Pushing FastViT VLM performance using multi-scale features and pooling strategies. These modifications slightly improve FastViT. Training setup is LLaVA-1.5 with Vicuna 7B. 

#### 3.1.1 Multi-Scale Features

Typical convolutional and hybrid architectures split up the computations into 4 distinct stages with a downsampling operation between them. While the VLM relies on features from the penultimate layer, features in earlier stages of the network extract information at different granularity. Aggregating information from multiple scales can complement high-level features from the penultimate layer.

The architecture for multiple scale feature extraction is shown in [Fig.2](https://arxiv.org/html/2412.13303v2#S3.F2 "In 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"). We ablate between 2 designs to pool features from different stages, i.e. AvgPooling and 2D Depthwise convolutions. From [Tab.2](https://arxiv.org/html/2412.13303v2#S3.T2 "In 3.1 FastViT as VLM Image Encoder ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), we find that using depthwise convolutions results in better performance.

![Image 4: Refer to caption](https://arxiv.org/html/x4.png)

Figure 3: Novel scaling strategy of FastViTHD lowers latency at various image resolutions. FastViT-Naive, a naive scaling of the FastViT architecture, and our proposed FastViTHD have the same number of parameters. ConvNeXt-L is provided for reference. All models are benchmarked on M1 Macbook Pro and trained with LLaVA-1.5 setup and Vicuna 7B. Note that the y 𝑦 y italic_y-axis is in log scale.

### 3.2 FastViTHD: High Resolution Encoder for VLM

While FastViT with the introduced model interventions performs well as an image encoder that is 8.7×\times× smaller than ViT-L/14, previous studies[[15](https://arxiv.org/html/2412.13303v2#bib.bib15), [47](https://arxiv.org/html/2412.13303v2#bib.bib47)] have demonstrated that increasing the scale of the image encoder improves its generalization capabilities.

Hybrid architectures[[17](https://arxiv.org/html/2412.13303v2#bib.bib17), [92](https://arxiv.org/html/2412.13303v2#bib.bib92)] typically scale the number of self-attention layers and width in a 4-stage design, but this has drawbacks. From [Fig.3](https://arxiv.org/html/2412.13303v2#S3.F3 "In 3.1.1 Multi-Scale Features ‣ 3.1 FastViT as VLM Image Encoder ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), simply scaling-up the number of self-attention layers in stages 3 and 4 of FastViT, as done in prior works, is suboptimal and slower than ConvNeXT-L. To mitigate this, we introduce an extra stage with a downsampling layer, ensuring self-attention operates on tensors downsampled by a factor of 32, rather than 16 as in recent models like ViTamin, see [Fig.2](https://arxiv.org/html/2412.13303v2#S3.F2 "In 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"). More details on the naive scaling approach can be found in Sec.[B](https://arxiv.org/html/2412.13303v2#supp_arc_details "Architecture Details ‣ B Architecture Details ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"). Our design reduces image encoding latency and generates 4×\times× fewer tokens for the compute-intensive LLM decoder, thereby decreasing the time-to-first-token (TTFT). The architecture schematic is shown in [Fig.2](https://arxiv.org/html/2412.13303v2#S3.F2 "In 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), and we call this model FastViTHD.

The model architecture consists of 5 stages, as shown in [Fig.2](https://arxiv.org/html/2412.13303v2#S3.F2 "In 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), with the first three stages utilizing RepMixer[[82](https://arxiv.org/html/2412.13303v2#bib.bib82)] blocks and the last two stages employing multi-headed self-attention[[24](https://arxiv.org/html/2412.13303v2#bib.bib24)] blocks. The model depth at each stage is [2, 12, 24, 4, 2], and the embedding dimensions for each stage are [96, 192, 384, 768, 1536]. The MLP expansion ratio for the ConvFFN layers is set to 4.0. The model has 125.1M parameters, which is 3.5×\times× larger than the largest FastViT variant from MobileCLIP, but is still smaller than popular ViT alternatives.

| Image | Encoder | Input | Latency | Zero-Shot | Avg Perf. | Avg Perf. |
| --- | --- | --- | --- | --- | --- | --- |
| Encoder | Size(M)↓↓\downarrow↓ | Res. | Enc.(ms)↓↓\downarrow↓ | ImageNet | Retrieval | on 38 tasks |
| ViT-L/14[[27](https://arxiv.org/html/2412.13303v2#bib.bib27)] | 304 | 224 | 47.2 | 79.2 | 60.8 | 66.3 |
| ViTamin-L[[12](https://arxiv.org/html/2412.13303v2#bib.bib12)] | 333 | 224 | 38.1 | 80.8 | 60.3 | 66.7 |
| ConvNeXt-L | 200 | 320 | 34.4 | 76.8 | 64.8 | 63.9 |
| FastViTHD | 125 | 224 | 6.8 | 78.3 | 67.7 | 66.3 |

Table 3: FastViTHD achieves competitive results on CLIP benchmarks at significantly lower latency. We follow the same setup described in[[12](https://arxiv.org/html/2412.13303v2#bib.bib12)] to report average retrieval performance and setup described in[[27](https://arxiv.org/html/2412.13303v2#bib.bib27)] to report average performance on 38 tasks. All models are benchmarked on M1 Macbook Pro. 

We follow the CLIP pretraining setup of [[83](https://arxiv.org/html/2412.13303v2#bib.bib83)] using the DataCompDR-1B dataset to pretrain FastViTHD before employing it for FastVLM training. [Table 3](https://arxiv.org/html/2412.13303v2#S3.T3 "In 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models") shows that FastViTHD, despite being 2.4×\times× smaller and 6.9×\times× faster than ViT-L/14, achieves comparable average performance across 38 multi-modal zero-shot tasks[[27](https://arxiv.org/html/2412.13303v2#bib.bib27)]. In comparison to ViTamin[[12](https://arxiv.org/html/2412.13303v2#bib.bib12)], a hybrid transformer architecture built for VLMs, FastViTHD delivers superior average retrieval performance while being 2.7×\times× smaller and 5.6×\times× faster. In [Tab.4](https://arxiv.org/html/2412.13303v2#S3.T4 "In 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), we compare FastViTHD with other CLIP-pretrained hierarchical backbones, i.e. ConvNeXT-L and ConvNeXT-XXL, for VLM tasks after LLaVA-1.5 training. FastViTHD performs as well as ConvNeXT-XXL while being 6.8×\times× smaller and 3.3×\times× faster.

| Image | Input | Latency | #Visual | GQA | TextVQA | POPE | DocVQA | Seed | Avg-5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Encoder | Res. | Enc.(ms)↓↓\downarrow↓ | Tokens | Bench I |
| FastViTHD | 256 | 10.1 | 16 | 60.6 | 53.1 | 82.3 | 17.4 | 63.7 | 55.5 |
| C.N-L | 320 | 34.4 | 100 | 61.9 | 55.5 | 85.3 | 21.3 | 64.6 | 57.7 |
| C.N-XXL | 256 | 89.9 | 64 | 62.7 | 56.3 | 85.3 | 21.6 | 65.6 | 58.3 |
| FastViTHD | 512 | 33.5 | 64 | 63.0 | 59.3 | 86.4 | 25.7 | 67.1 | 60.4 |
| FastViTHD | 768 | 122.6 | 144 | 62.4 | 62.9 | 87.7 | 32.9 | 68.2 | 62.8 |
| C.N-L | 512 | 71.9 | 256 | 61.8 | 61.0 | 86.3 | 30.8 | 66.8 | 61.3 |
| C.N-XXL | 512 | 397.1 | 256 | 62.3 | 65.1 | 87.7 | 36.2 | 68.4 | 63.9 |
| FastViTHD | 1024 | 235.6 | 256 | 63.1 | 64.4 | 88.1 | 35.6 | 68.5 | 63.9 |

Table 4: FastViTHD achieves higher accuracy than ConvNeXT while having lower latency at a higher resolution. The models are grouped based on the total number of visual tokens produced for the LLM to process. “C.N” stands for ConvNeXT. Training setup is LLaVA-1.5 with Vicuna 7B. 

#### 3.2.1 Vision Encoder - Language Decoder Interplay

![Image 5: Refer to caption](https://arxiv.org/html/x5.png)

Figure 4: FastViTHD improves the Pareto-Optimal curve for accuracy versus time to first token compared with FastViT. Comparison of FastViT and FastViTHD backbones paired with Qwen2[[86](https://arxiv.org/html/2412.13303v2#bib.bib86)] family (chat variant) LLMs of varying sizes and different image resolutions (annotated for each point). The Pareto-optimal curve is highlighted for the two vision backbones. Training setup is LLaVA-1.5. Note that the x 𝑥 x italic_x-axis is in log scale.

![Image 6: Refer to caption](https://arxiv.org/html/x6.png)

Figure 5: Vision latency dominates at high resolution. Breakdown of FastVLM’s time to first token for varying image resolutions. Vision encoder is FastViTHD and LLM is Qwen2-1.5B.

![Image 7: Refer to caption](https://arxiv.org/html/x7.png)

Figure 6: Dynamic input resolution (AnyRes) is only optimal at the highest resolution when using fewer tiles (2×\times×2). The vision encoder is FastViTHD. The tile grid size is specified in parenthesis. Training setup is LLaVA-1.5 with Vicuna 7B. Note that the x 𝑥 x italic_x-axis is in log scale.

The accuracy-latency trade-off in a VLM is influenced by several factors. On one hand, the overall performance of the VLM depends on (1) the input image resolution, (2) the quantity and quality of visual tokens, and (3) the capability of the LLM. On the other hand, the total latency (time to first token generation) of a VLM is determined by (1) the latency of the vision encoder and (2) the prefilling time of the LLM. The latter is affected by both the number of tokens produced by the vision encoder and the size of the LLM.

Due to the complex optimization landscape of VLMs, claims regarding the optimality of a vision encoder must be verified across various pairs of (Resolution, LLM). Here, we empirically demonstrate the optimality of FastViTHD over FastViT. For each vision encoder, we consider three LLMs, Qwen2[[86](https://arxiv.org/html/2412.13303v2#bib.bib86)]-0.5B/1.5B/7B, along with a range of input image resolutions. For each (Resolution, LLM) pair, we conduct LLaVA-1.5[[53](https://arxiv.org/html/2412.13303v2#bib.bib53)] pretraining and visual instruction tuning, and evaluate the resulting model over a range of tasks. The results are presented in [Fig.4](https://arxiv.org/html/2412.13303v2#S3.F4 "In 3.2.1 Vision Encoder - Language Decoder Interplay ‣ 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models").

First, we observe that for a vision encoder, the Pareto-optimal curve (highlighted in [Fig.4](https://arxiv.org/html/2412.13303v2#S3.F4 "In 3.2.1 Vision Encoder - Language Decoder Interplay ‣ 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")), which represents the maximum achievable performance for a given runtime budget (TTFT), consists of varying sizes of LLMs. Specifically, pairing high resolution with a small LLM is suboptimal as a small LLM cannot effectively utilize that many tokens, and TTFT will be dominated by the latency of the vision encoder (see [Fig.5](https://arxiv.org/html/2412.13303v2#S3.F5 "In 3.2.1 Vision Encoder - Language Decoder Interplay ‣ 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")).

Second, the Pareto-optimal curve for FastViTHD in [Fig.4](https://arxiv.org/html/2412.13303v2#S3.F4 "In 3.2.1 Vision Encoder - Language Decoder Interplay ‣ 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models") is significantly better than that of FastViT. For a given runtime budget, considering all possible (Resolution, LLM) pairs, we achieve significantly better performance (an improvement of over 2.5 points on the Average-5 metric) with FastViTHD. Similarly, FastViTHD can reach a target VLM performance up to 3×\times× faster. It is important to note that in previous sections, we demonstrated that a FastViT-based VLM already represents a significant improvement over ViT-based VLMs, and yet FastViTHD provides substantial gains over FastViT.

#### 3.2.2 Static vs. Dynamic Input Resolution

There are two ways to scale input resolution: adjusting the model’s input resolution directly or tiling the image and setting the encoder’s resolution to the tile size. The tiled inference (AnyRes) was introduced in prior works [[52](https://arxiv.org/html/2412.13303v2#bib.bib52), [54](https://arxiv.org/html/2412.13303v2#bib.bib54)] to enable ViT models to process high resolution images. Since FastViTHD is designed to run inference efficiently on high input resolutions, we analyze the optimal operating point for various resolutions using the two strategies. From [Fig.6](https://arxiv.org/html/2412.13303v2#S3.F6 "In 3.2.1 Vision Encoder - Language Decoder Interplay ‣ 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), we see that setting the model’s input resolution directly to the desired resolution offers the best accuracy-latency tradeoff, with dynamic resolution benefiting only at extreme resolutions like 1536×\times×1536, due to memory bandwidth limitations. If dynamic resolution is desired, using a setting with fewer tiles exhibits better accuracy-latency tradeoff. Further discussion on this setup is presented in Sec. [C.1](https://arxiv.org/html/2412.13303v2#supp_anyres_results "Dynamic Resolution (AnyRes) Results ‣ C.1 Dynamic Resolution (AnyRes) Results ‣ C Additional Results ‣ FastVLM: Efficient Vision Encoding for Vision Language Models").

#### 3.2.3 Comparison with Token Pruning & Downsampling

We further compare the performance of FastViTHD operating at different resolutions to popular token pruning methods in literature. From [Tab.5](https://arxiv.org/html/2412.13303v2#S3.T5 "In 3.2.3 Comparison with Token Pruning & Downsampling ‣ 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), we find that VLMs achieve better accuracy to latency trade-off using a hierarchical backbone as opposed to using token pruning methods on isotropic architectures like ViT. By simply training the VLMs at lower input resolution, FastViTHD achieves visual token counts as low as 16, while improving over recent token pruning methods. Interestingly, even the most effective token pruning methods, such as those proposed by[[7](https://arxiv.org/html/2412.13303v2#bib.bib7), [32](https://arxiv.org/html/2412.13303v2#bib.bib32), [33](https://arxiv.org/html/2412.13303v2#bib.bib33), [87](https://arxiv.org/html/2412.13303v2#bib.bib87)], perform worse than FastViTHD trained at a lower input resolution of 256×\times×256.

Model Input#Visual GQA SQA Text-POPE VQA Seed
Res.Tokens VQA v2 Bench
ViT-L/14 M 3[[7](https://arxiv.org/html/2412.13303v2#bib.bib7)]336 9 58.0--83.4-55.4
ViT-L/14 MQT[[32](https://arxiv.org/html/2412.13303v2#bib.bib32)]336 16 57.6 67.5-80.8 71.1-
FastViTHD 256 16 60.6 69.2 53.1 82.3 74.7 58.8
ViT-L/14 PruMerge[[70](https://arxiv.org/html/2412.13303v2#bib.bib70)]336 40-68.5 56.0 76.3 72.0-
ViT-L/14 PruMerge+[[70](https://arxiv.org/html/2412.13303v2#bib.bib70)]336 40-68.3 57.1 84.0 76.8-
ViT-L/14 M 3[[7](https://arxiv.org/html/2412.13303v2#bib.bib7)]336 36 60.3--85.5-58.0
FastV[[14](https://arxiv.org/html/2412.13303v2#bib.bib14)]336 64 46.1 51.1 47.8 48.0 55.0 51.9
SparseVLM[[97](https://arxiv.org/html/2412.13303v2#bib.bib97)]336 64 52.7 62.2 51.8 75.1 68.2 51.1
VisionZip[[87](https://arxiv.org/html/2412.13303v2#bib.bib87)]336 64 55.1 69.0 55.5 77.0 62.9 52.2
VisionZip‡[[87](https://arxiv.org/html/2412.13303v2#bib.bib87)]336 64 57.0 68.8 56.0 80.9 74.2 53.4
DynamicLLaVA I[[33](https://arxiv.org/html/2412.13303v2#bib.bib33)]336 115 61.4 69.1 57.0 85.0 78.0-
DynamicLLaVA I|T[[33](https://arxiv.org/html/2412.13303v2#bib.bib33)]336 115 61.3 68.6 56.5 85.9 77.9-
FastViTHD 512 64 63.0 68.9 59.3 86.4 78.0 61.8
ViT-L/14 M 3[[7](https://arxiv.org/html/2412.13303v2#bib.bib7)]336 144 61.3--87.0-59.7
ViT-L/14 MQT[[32](https://arxiv.org/html/2412.13303v2#bib.bib32)]336 144 61.4 67.6-83.9 76.4-
FastV[[14](https://arxiv.org/html/2412.13303v2#bib.bib14)]336 192 52.7 67.3 52.5 64.8 67.1 57.1
SparseVLM[[97](https://arxiv.org/html/2412.13303v2#bib.bib97)]336 192 57.6 69.1 56.1 83.6 75.6 55.8
VisionZip[[87](https://arxiv.org/html/2412.13303v2#bib.bib87)]336 192 59.3 68.9 57.3 85.3 76.8 56.4
VisionZip‡[[87](https://arxiv.org/html/2412.13303v2#bib.bib87)]336 192 60.1 68.2 57.8 84.9 77.4 57.1
FastViTHD 768 144 62.4 67.6 62.9 87.7 78.9 62.5
ViT-L/14 MQT[[32](https://arxiv.org/html/2412.13303v2#bib.bib32)]336 256 61.6 67.5-84.4 76.8-
FastViTHD 1024 256 63.1 67.4 64.4 88.1 79.2-

Table 5: FastViTHD more effectively reduces tokens compared with token pruning methods. The models are grouped based on total number of visual tokens. “-” indicates that performance was not reported in the respective paper. All models presented in this table are trained using LLaVA-1.5 setup with Vicuna 7B. ‡- indicates further finetuning as reported in[[87](https://arxiv.org/html/2412.13303v2#bib.bib87)]. I - indicates vision only sparsification and I|T indicates vision-language sparsification, as reported in[[33](https://arxiv.org/html/2412.13303v2#bib.bib33)]. 

Row Method Vision LLM Data (M)Input#Visual Vis. Enc.TTFT GQA SQA Text POPE LLaVA MM-VQA Doc MMMU Seed
Ann.Encoder(PT+IT)Res.Tokens Size(M)↓↓\downarrow↓(ms)↓↓\downarrow↓VQA Bench W Vet v2 VQA Bench I
0.5B Model Comparison
R1 nanoLLaVA ViT-SO400M Qw.1.5-384 729 430 535 54.8 59.0 46.7 84.1--70.8-30.4-
R2 LLaVAOV[[45](https://arxiv.org/html/2412.13303v2#bib.bib45)]∗ViT-SO400M Qw.2 4.5+3.2 1152 7290 430 14124-67.2---29.1-70.0 31.4 65.5
R3 FastVLM (Ours)FastViTHD Qw.2 15+1.1 1024 256 125 166 61.6 61.4 57.4 87.4 56.0 31.8 77.0 61.0 30.9 65.6
R4 FastVLM (Ours)FastViTHD Qw.2 15+12.5 1024 256 125 166 63.1 81.5 62.9 86.6 66.7 29.8 78.8 70.4 32.9 69.2
R5 FastVLM (Ours)FastViTHD Qw.2 15+23.1 1024 256 125 166 62.7 82.0 65.8 86.2-37.5 78.6 79.1 33.6 69.3
1-2B Model Comparison
R6 MobileVLMv2[[19](https://arxiv.org/html/2412.13303v2#bib.bib19)]ViT-L/14 ML.1.2+3.6 336 144 304 458 59.3 66.7 52.1 84.3------
R7 FastVLM (Ours)FastViTHD Qw.2 15+1.1 768 144 125 152 63.9 75.8 64.4 87.2 65.2 35.4 79.4 61.3 34.9 71.7
R8 FastVLM (Ours)FastViTHD Qw.2 15+12.5 768 144 125 152 64.2 87.9 66.4 88.3 68.5 41.1 80.1 69.4 38.1 72.4
R9 DeepSeekVL[[58](https://arxiv.org/html/2412.13303v2#bib.bib58)]ViT-SO400M DS.-384 576 430----87.6-34.8--32.2 66.7
R10 MM1[[66](https://arxiv.org/html/2412.13303v2#bib.bib66)]∗ViT-H-3000+1.5 1344 720 632--62.3 68.2 87.4 67.5 39.4-68.4 33.2 65.6
R11 FastVLM (Ours)FastViTHD Qw.2 15+1.1 1024 256 125 233 64.2 74.8 66.0 88.0 66.9 37.6 79.9 67.7 33.1 71.4
R12 FastVLM (Ours)FastViTHD Qw.2 15+12.5 1024 256 125 233 64.3 89.9 69.0 87.8 68.0 43.6 80.7 75.6 39.1 73.1
R13 FastVLM (Ours)FastViTHD Qw.2 15+23.1 1024 256 125 233 64.3 92.9 71.2 87.8-51.0 80.9 84.2 38.0 72.6
7B Model Comparison
R14 InstructBLIP[[20](https://arxiv.org/html/2412.13303v2#bib.bib20)]ViT-g/14 Vic.129+1.2 224 32 1012 302 49.2 60.5 50.1-60.9 26.2--30.6-
R15 FastVLM (Ours)FastViTHD Vic.0.5+0.6 256 16 125 150 60.6 69.2 53.1 82.3 60.4 27.5 74.7 17.4 36.2 63.7
R16 FastVLM (Ours)FastViTHD Vic.15+1.1 256 16 125 150 62.1 75.7 57.2 83.9 64.0 31.5 77.3 29.8 37.6 68.8
R17 MobileVLMv2[[19](https://arxiv.org/html/2412.13303v2#bib.bib19)]ViT-L/14 Vic.1.2+3.6 336 144 304 460 62.6 74.8 62.3 85.3------
R18 ConvLLaVA[[28](https://arxiv.org/html/2412.13303v2#bib.bib28)]ConvNeXT-L Vic.4.9+0.6 768 144 200 496--59.1 87.3-44.8-44.8 36.3 68.8
R19 FastVLM (Ours)FastViTHD Vic.0.5+0.6 768 144 125 387 62.4 67.6 62.9 87.7 63.8 31.5 78.9 32.9 34.9 68.2
R20 FastVLM (Ours)FastViTHD Vic.0.5+1.1 768 144 125 387 63.2 73.5 67.5 86.3 63.9 33.0 79.1 57.3 36.9 69.9
R21 FastVLM (Ours)FastViTHD Vic.15+1.1 768 144 125 387 65.0 78.7 69.4 87.5 67.0 42.2 81.3 65.5 37.0 73.7
R22 FastVLM (Ours)FastViTHD Qw.2 15+1.1 768 144 125 446 65.6 85.9 69.5 87.2 73.0 41.3 81.3 66.9 43.6 75.3
R23 FastVLM (Ours)FastViTHD Qw.2 15+12.5 768 144 125 446 64.7 96.0 71.9 87.4 73.6 49.4 81.1 75.3 44.6 74.3
R24 Qwen-VL[[4](https://arxiv.org/html/2412.13303v2#bib.bib4)]ViT-G/14 Qw.1400+50 448 256 1844-59.3 67.1 63.8---79.5 65.1--
R25 Qwen-VL-Chat[[4](https://arxiv.org/html/2412.13303v2#bib.bib4)]ViT-G/14 Qw.1400+50 448 256 1844-57.5 68.2 61.5---78.2 62.6--
R26 ConvLLaVA[[28](https://arxiv.org/html/2412.13303v2#bib.bib28)]ConvNeXT-L Vic.4.9+0.6 1024 256 200 1157--62.5 87.7-44.4-48.5 35.1 69.3
R27 FastVLM (Ours)FastViTHD Vic.0.5+0.6 1024 256 125 577 63.1 67.4 64.4 88.1 64.8 31.7 79.2 35.6 35.1 68.5
R28 FastVLM (Ours)FastViTHD Vic.0.5+1.1 1024 256 125 577 63.3 74.1 67.4 87.1 66.5 32.4 79.3 62.8 37.3 69.9
R29 FastVLM (Ours)FastViTHD Vic.15+1.1 1024 256 125 577 65.2 80.3 70.6 87.2 71.5 40.1 81.6 72.4 36.7 73.5
R30 FastVLM (Ours)FastViTHD Qw.2 15+1.1 1024 256 125 641 65.8 84.9 72.1 87.8 75.8 44.1 81.7 73.3 46.2 75.1
R31 LLaVA-1.5[[53](https://arxiv.org/html/2412.13303v2#bib.bib53)]ViT-L/14 Vic.0.5+0.6 336 576 304 1297 62.0 70.4 58.2 85.9 59.6 31.1 76.6 28.1 35.3 66.1
R32 MobileVLMv2[[19](https://arxiv.org/html/2412.13303v2#bib.bib19)]ViT-L/14 Vic.1.2+3.6 336 576 304 1297 64.6 74.8 66.8 86.1------
R33 ShareGPT4V[[13](https://arxiv.org/html/2412.13303v2#bib.bib13)]ViT-L/14 Vic.1.2+0.7 336 576 304 1297 63.3 68.4 60.4 85.7 72.6 37.6 80.6--69.7
R34 ViTamin[[12](https://arxiv.org/html/2412.13303v2#bib.bib12)]ViTamin-L Vic.0.5+0.6 384 576 333 1308 61.6 67.6 59.8 85.5 66.1 33.6 78.9---
R35 ConvLLaVA[[28](https://arxiv.org/html/2412.13303v2#bib.bib28)]ConvNeXT-L Vic.4.9+0.6 1536 576 200 2740--65.8 87.3-45.9-59.0 35.8 70.2
R36 VILA[[50](https://arxiv.org/html/2412.13303v2#bib.bib50)]ViT-L/14 L-2 50+1 336 576 304 1297 62.3 68.2 64.4 85.5 69.7 34.9 79.9--62.8
R37 LLaVA-FlexAttn[[46](https://arxiv.org/html/2412.13303v2#bib.bib46)]ViT-L/14 Vic.0.5+0.6 1008 576 304-62.2-48.9 85.9-29.4 78.7---
R38 MM1[[66](https://arxiv.org/html/2412.13303v2#bib.bib66)]∗ViT-H-3000+1.5 1344 720 632--72.6 72.8 86.6 81.5 42.1 82.8 76.8 37.0 69.9
R39 LLaVA-NeXT†∗ViT-L/14 L-3-672 2880 304 20347 65.2 72.8 64.6-80.1--78.2 41.7 72.7
R40 FastVLM (Ours)FastViTHD Qw.2 15+6.5 1024 256 125 641 66.0 87.4 73.1 87.3 72.4 47.6 82.3 78.7 42.8 75.9
R41 FastVLM (Ours)FastViTHD Qw.2 15+12.5 1024 256 125 641 65.2 95.7 73.4 86.9 71.1 48.4 81.6 82.7 47.3 74.1
R42 FastVLM (Ours)FastViTHD Qw.2 15+23.1 1024 256 125 641 64.1 96.8 76.1 87.2-56.3 81.9 88.0 44.0 74.5
VLMs with Multiple Vision Encoders and 8B LLM
ConvNeXT-L 1536 200
R43 MiniGemini-HD†ViT-L/14 L-3 1.5+1.5 672 2880 304 21832 64.5 75.1 70.2----74.6 37.3 73.2
ViT-SO400M 384 430
ConvNeXt-XXL 1024 846
DINOv2-ViT-L/14 518 304
R44 Cambrian-1[[78](https://arxiv.org/html/2412.13303v2#bib.bib78)]ViT-L/14 L-3 2.5+7 336 576 304 5085 64.6 80.4 71.7----77.8 42.7 74.7

Table 6: VLM evaluations and comparison with recent methods. The models are grouped based on total number of visual tokens. “-” indicates that performance was not reported in the respective paper. For the dataset column, “-” indicates that the dataset size for pretraining (“PT”) or instruction tuning (“IT”) is not explicitly mentioned in the respective paper. For methods that have more than 2 stages of training, we report the total samples used for all the pretraining stages as part of “PT”. “TTFT” means time to first token (the sum of the vision encoder latency and the LLM prefilling time), we report latency only for models that are publicly available and in a format favorable to MLX[[31](https://arxiv.org/html/2412.13303v2#bib.bib31)] “Vic.” refers to Vicuna[[98](https://arxiv.org/html/2412.13303v2#bib.bib98)], “Qw.2” refers to Qwen2[[86](https://arxiv.org/html/2412.13303v2#bib.bib86)] and “Qw.” refers to Qwen[[3](https://arxiv.org/html/2412.13303v2#bib.bib3)]. “L-2” refers to LLaMA-2. “L-3” refers to LLaMA-3. “ML.” refers to MobileLLaMA[[18](https://arxiv.org/html/2412.13303v2#bib.bib18), [19](https://arxiv.org/html/2412.13303v2#bib.bib19)]. “DS.” refers to DeepSeek LLM[[21](https://arxiv.org/html/2412.13303v2#bib.bib21)]. ∗ For input resolution and visual tokens, we report the highest supported resolution by the respective models as some models like LLaVA-OneVision[[45](https://arxiv.org/html/2412.13303v2#bib.bib45)] and MM1[[66](https://arxiv.org/html/2412.13303v2#bib.bib66)] use dynamic input resolution. FastVLM models using dynamic resolution employs a simple 2×\times×2 grid, with tile size set to 1024. †- performance numbers reported from[[78](https://arxiv.org/html/2412.13303v2#bib.bib78)]. For VLMs that use multiple vision encoders, the size of each encoder is listed independently, for TTFT, the latency from each encoder is summed up. 

4 Experiments
-------------

Training Setup. For all the ablations presented in [Sec.3](https://arxiv.org/html/2412.13303v2#S3 "3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), we follow the 2-stage setup described in LLaVA-1.5[[53](https://arxiv.org/html/2412.13303v2#bib.bib53)] with Vicuna-7B[[98](https://arxiv.org/html/2412.13303v2#bib.bib98)] as the LLM decoder, unless mentioned otherwise. During the first stage, only the projector is trained using LLaVA-558K alignment dataset for one epoch, with a batch size of 256 and a learning rate of 10−3 superscript 10 3 10^{-3}10 start_POSTSUPERSCRIPT - 3 end_POSTSUPERSCRIPT. At this stage, the input image resolution matches the backbone pretraining resolution (e.g., 256 for FastViT and 224 for FastViTHD). In the second stage, we use LLaVA-665K supervised finetuning dataset, training the models for one epoch and tuning all the modules, i.e., vision encoder, projector and the LLM. At this stage, the input image resolution is set to the target resolution.

In [Sec.4](https://arxiv.org/html/2412.13303v2#S4 "4 Experiments ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), we present results with different LLM decoders, primarily with Qwen2-0.5B/1.5B/7B model family[[86](https://arxiv.org/html/2412.13303v2#bib.bib86)] (chat variant) and Vicuna-7B model[[98](https://arxiv.org/html/2412.13303v2#bib.bib98)]. We report results in two training setups, the first one is the 2-Stage setup introduced in LLaVA-1.5. For the second training setup, we follow the current trend in literature[[43](https://arxiv.org/html/2412.13303v2#bib.bib43), [66](https://arxiv.org/html/2412.13303v2#bib.bib66)] of training the VLMs in 3 stages, i.e. Stage 1 for training the connector, Stage 1.5 for resolution scaling and Stage 2 for visual instruction tuning. Information on datasets used in these stages can be found in Sec.[D](https://arxiv.org/html/2412.13303v2#supp_datasets_info "Datasets ‣ D Datasets ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"). In this setup, the input image resolution is set to the backbone pretraining resolution for Stage 1 and adjusted to the target resolution for the following two stages. In both setups, the vision encoder and LLM are frozen only in stage 1, while all modules are finetuned in the remaining stages. For the best setup, we further finetune the model from Stage 2 with high quality instruction tuning dataset with chain-of-thought reasoning from[[30](https://arxiv.org/html/2412.13303v2#bib.bib30)] and call this Stage 3. Details of this setup is elaborated in Sec.[A](https://arxiv.org/html/2412.13303v2#supp_training_setup "Training Setup ‣ A Training Setup ‣ FastVLM: Efficient Vision Encoding for Vision Language Models") and Sec.[D](https://arxiv.org/html/2412.13303v2#supp_datasets_info "Datasets ‣ D Datasets ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"). We publicly release R4, R12, and R41 checkpoints from Stage 2 training, as well as the R5, R13, and R42 checkpoints from Stage 3 training, as part of our open-sourced codebase.

All FastVLM models reported in the paper are trained on a single node with 8× NVIDIA H100-80GB GPUs. Stage 1 training of VLM is quick, taking roughly 30 minutes to train with a Qwen2-7B decoder. Stage 1.5 and Stage 2 training runs are dependent on input resolution. For an input resolution of 1024×\times×1024, Stage 1.5 takes 77 hours and Stage 2 takes 8 hours. The reported wall clock times correspond to the following datasets used in these stages: 15 million samples in Stage 1.5 and 1.1 million samples in Stage 2.

Evaluation. We evaluate the models on the mainstream benchmarks of GQA[[34](https://arxiv.org/html/2412.13303v2#bib.bib34)], ScienceQA[[59](https://arxiv.org/html/2412.13303v2#bib.bib59)], TextVQA[[74](https://arxiv.org/html/2412.13303v2#bib.bib74)], POPE[[48](https://arxiv.org/html/2412.13303v2#bib.bib48)], LLaVA-in-the-wild[[54](https://arxiv.org/html/2412.13303v2#bib.bib54)], VQAv2[[29](https://arxiv.org/html/2412.13303v2#bib.bib29)], MMVet[[91](https://arxiv.org/html/2412.13303v2#bib.bib91)], MMMU[[93](https://arxiv.org/html/2412.13303v2#bib.bib93)], DocVQA[[65](https://arxiv.org/html/2412.13303v2#bib.bib65)] and SeedBench[[42](https://arxiv.org/html/2412.13303v2#bib.bib42)]. For GQA, ScienceQA, TextVQA, POPE and LLaVA-in-the-wild benchmarks, we use the official evaluation from LLaVA[[54](https://arxiv.org/html/2412.13303v2#bib.bib54)]. For the remaining evaluations we use lmms-eval[[96](https://arxiv.org/html/2412.13303v2#bib.bib96)] library v0.2.2. We use the default settings for all the evaluations and lmms-eval defaults to 0613 version of GPT for evaluations that rely on GPT as a judge.

For ablations presented in [Sec.3](https://arxiv.org/html/2412.13303v2#S3 "3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), we report GQA, TextVQA, POPE, DocVQA and SeedBench. GQA and SeedBench are general knowledge benchmarks, DocVQA and TextVQA represent text-rich evaluations and POPE is a hallucination benchmark. Together these benchmarks provide diversity and are quick to evaluate for ablations. Most importantly, they exhibit lower variance to different initializations and under probabilistic decoding setting. We report the variance for all the evals for different initialization in Sec. [D.3](https://arxiv.org/html/2412.13303v2#supp_eval_variance "Evaluations ‣ D.3 Evaluations ‣ D Datasets ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"). The standard deviation across the 5 selected metrics is less than 0.5. We call the average of these 5 benchmarks Avg-5, and use it as a reliable signal for our analysis. The empirical standard deviation estimate for Avg-5 is 0.1.

Benchmarking. We benchmark all the models on a MacBook Pro with the M1 Max chip and 32GB RAM. The image encoder is converted to a Core ML package file using coremltools v7.2 and benchmarked on the neural engine using XCode 15.4 (15F31d). The LLM is benchmarked on the MacBook Pro GPU using MLX[[31](https://arxiv.org/html/2412.13303v2#bib.bib31)]. The model is first converted using mlx_lm.convert tool, which converts the models on huggingface to the MLX format and casts the tensors to FP16. The prefilling latency is estimated using mlx_lm.cache_prompt tool[[31](https://arxiv.org/html/2412.13303v2#bib.bib31)]. Time-To-First-Token (TTFT) is estimated by adding the image encoding latency at a specific resolution to the LLM prefilling latency for the associated visual tokens.

### 4.1 Comparison with state-of-the-art

In [Tab.6](https://arxiv.org/html/2412.13303v2#S3.T6 "In 3.2.3 Comparison with Token Pruning & Downsampling ‣ 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), we compare FastVLM with recently published methods. The training setup can vary widely between works. For each, we report the LLM decoder and the sizes of the instruction tuning and pretraining datasets used to train the respective VLMs, to facilitate a fair comparison.

Hierarchical Backbones. When we compare FastVLM (R20) with ConvLLaVA[[28](https://arxiv.org/html/2412.13303v2#bib.bib28)] (R18), with the same LLM and similar training data size, our model obtains +8.4% better performance on TextVQA and +12.5% better performance on DocVQA while being 22% faster. The gap widens at higher resolution, where FastVLM (R28 and R29) achieves superior performance on wide range of benchmarks while being 2×\times× faster than ConvLLaVA (R26), with the same LLM decoder.

Dataset Scaling. When scaling the pretraining dataset by incorporating an intermediate pretraining stage for resolution scaling with 15M samples, FastVLM (R21) matches or surpasses MM1[[66](https://arxiv.org/html/2412.13303v2#bib.bib66)] (R38) across a wide range of benchmarks. Remarkably, FastVLM achieves this performance while generating 5×\times× fewer visual tokens. With an input resolution of 1024×\times×1024 and a larger instruction tuning dataset of size 12.5M, FastVLM (R41) outperforms MM1 (R38) and LLaVA-NeXT (R39) across various benchmarks, including text-rich evaluations, like TextVQA and DocVQA, which are sensitive to input resolution and number of visual tokens. The gap widens when further scale up input resolution using AnyRes, more details in Sec. [C.1](https://arxiv.org/html/2412.13303v2#supp_anyres_results "Dynamic Resolution (AnyRes) Results ‣ C.1 Dynamic Resolution (AnyRes) Results ‣ C Additional Results ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"). We provide details of the dataset splits in Sec.[D](https://arxiv.org/html/2412.13303v2#supp_datasets_info "Datasets ‣ D Datasets ‣ FastVLM: Efficient Vision Encoding for Vision Language Models").

Multiple Vision Encoders. Recently, MiniGemini[[49](https://arxiv.org/html/2412.13303v2#bib.bib49)] and Cambrian-1[[78](https://arxiv.org/html/2412.13303v2#bib.bib78)] introduced models that rely on multiple vision encoders. In [Tab.6](https://arxiv.org/html/2412.13303v2#S3.T6 "In 3.2.3 Comparison with Token Pruning & Downsampling ‣ 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), we compare FastVLM (R40), which uses a single vision encoder with methods that use multiple encoders and trained on similarly scaled visual instruction tuning dataset. In Cambrian-1[[78](https://arxiv.org/html/2412.13303v2#bib.bib78)] (R44), vision encoding contributes 3.2×\times× more than LLM prefilling to the total time-to-first-token of approximately 5 seconds (detailed breakdown is provided in [Tab.10](https://arxiv.org/html/2412.13303v2#S2.T10 "In B.1 Naive Scaling ‣ B Architecture Details ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")). FastVLM (R40) outperforms Cambrian-1 (R44) when trained on a similar visual instruction tuning dataset, while being 7.9×\times× faster. By scaling the instruction tuning dataset to 12.5M, FastVLM (R41) achieves superior performance over Cambrian-1 (R44) with 2.3×\times× fewer visual tokens, even on text-rich evaluations (see [Tab.11](https://arxiv.org/html/2412.13303v2#S3.T11 "In C Additional Results ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")) that are sensitive to the number of visual tokens.

Effect of Decoder. VLM performance also depends on the quality of LLM, as demonstrated in prior studies, like[[44](https://arxiv.org/html/2412.13303v2#bib.bib44)]. By switching from Vicuna-7B (R21, R29) to Qwen2[[86](https://arxiv.org/html/2412.13303v2#bib.bib86), [77](https://arxiv.org/html/2412.13303v2#bib.bib77)] models (R22, R30), we see improvement in performance across all the benchmarks. The improvements are significant on MMVet, LLaVA-in-the-wild and MMMU benchmarks. With Qwen2-0.5B as the LLM decoder, FastVLM (R4) outperforms LLaVA-OneVision[[45](https://arxiv.org/html/2412.13303v2#bib.bib45)] (R2) while being 85×\times× faster. This result underscores the quality of our vision encoder, as both models use the same LLM decoder, while FastViTHD is 3.4×\times× smaller compared to SigLIP-SO400M[[94](https://arxiv.org/html/2412.13303v2#bib.bib94)].

Row Method Vision LLM#Visual Vis. Enc.ChartQA OCRBench TextVQA DocVQA InfoVQA MMMU AI2D SQA Seed
Ann.Encoder Tokens Size(M)↓↓\downarrow↓Bench I
0.5B Model Comparison
R1 SmolVLM2-0.5B[[62](https://arxiv.org/html/2412.13303v2#bib.bib62)]†ViT-B SmolLM2 1088∗93 62.8 61.0 60.2 70.5 25.5 33.7 59.2 80.0 62.2
R2 FastVLM-0.6B (Ours)FastViTHD Qw.2 256 125 71.4 55.8 65.8 79.1 43.3 33.6 66.0 82.0 69.3
1-3B Model Comparison
R3 SmolVLM2-2.2B[[62](https://arxiv.org/html/2412.13303v2#bib.bib62)]†ViT-SO400M SmolLM2 2106∗430 68.7 72.9 73.0 80.0 37.8 42.0 70.0 89.6 71.3
R4 MolmoE A1B-7B[[22](https://arxiv.org/html/2412.13303v2#bib.bib22)]†ViT-SO400M OLMoE.1728 430 48.0 54.7 61.5 77.7-33.9 71.0 87.5 67.9
R5 MM1.5-1B[[95](https://arxiv.org/html/2412.13303v2#bib.bib95)]ViT-H-1440 632 67.2 60.5 72.5 81.0 50.5 35.8 59.3 82.1 70.2
R6 FlorenceVL-3B[[11](https://arxiv.org/html/2412.13303v2#bib.bib11)]DaViT Phi-3 576 770 70.7 63.0 69.1 82.1 51.3 41.8 73.8 84.6 70.6
R7 FastVLM-1.7B (Ours)FastViTHD Qw.2 256 125 69.5 61.2 71.2 84.2 49.6 38.0 76.6 92.9 72.6

Table 7: Comparison with concurrent works. Models are ordered in the descending order of the number of visual tokens. The number of visual tokens listed for each model is based on the highest supported input resolution for the respective models. ∗ - For the number of visual tokens generated by SmolVLM2, we estimated it based on the model configs as there is no explicit mention of this information in the paper. The reported visual tokens correspond to the highest resolution supported by the respective model. † - models are evaluated using VLMEvalKit[[25](https://arxiv.org/html/2412.13303v2#bib.bib25)]. The rest of the models are evaluated using lmms-eval[[96](https://arxiv.org/html/2412.13303v2#bib.bib96)] library. The FastVLM model used in this table is R5 and R13 from [Tab.6](https://arxiv.org/html/2412.13303v2#S3.T6 "In 3.2.3 Comparison with Token Pruning & Downsampling ‣ 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models")

Concurrent works. In [Tab.7](https://arxiv.org/html/2412.13303v2#S4.T7 "In 4.1 Comparison with state-of-the-art ‣ 4 Experiments ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), we compare FastVLM with concurrent works like SmolVLM2[[62](https://arxiv.org/html/2412.13303v2#bib.bib62)] and FlorenceVL[[11](https://arxiv.org/html/2412.13303v2#bib.bib11)]. FastVLM (R7) performs competitively against SmolVLM2 (R3), particularly on text-rich evaluations such as ChartQA and TextVQA, while surpassing it on the DocVQA and InfoVQA benchmarks. FastVLM (R7) accomplishes this using 8.2×\times× fewer visual tokens than SmolVLM2 (R3). Additionally, FastVLM (R7) outperforms FlorenceVL (R6) on benchmarks such as TextVQA and DocVQA, while using 2.3×\times× fewer visual tokens and a 6.2×\times× smaller vision encoder. On knowledge based evaluations like AI2D and ScienceQA, FastVLM (R7) outperforms all other competing models. At the smallest scale, FastVLM (R2) outperforms a similar sized SmolVLM2 model (R1) on a wide range of benchmarks while using 4.3×\times× fewer visual tokens.

5 Conclusion
------------

In this work, we introduced FastVLM, which leverages the FastViTHD vision backbone for efficient encoding of high-resolution inputs. FastViTHD has a hybrid architecture, is pretrained on reinforced image-text data, and outputs a substantially reduced number of visual tokens with minimal accuracy sacrifice. FastVLM has competitive performance with prior works across a wide range of VLM benchmarks while improving efficiency in both time-to-first-token and the number of parameters in the vision backbone. Rigorous benchmarking on an M1 MacBook Pro demonstrates that FastVLM achieves a state-of-the-art resolution-latency-accuracy trade-off compared to existing works.

Acknowledgement
---------------

We thank David Koski, Christopher Webb, Matt Biddulph, Nick Henderson, Megan Maher Welsh, Jamie Cheng, and Jerremy Holland for their valuable contributions to prototyping the demo app. We are also grateful to Angelos Katharopoulos and Awni Hannun for their insightful feedback and suggestions on benchmarking. Additionally, we thank Jen-Hao Rick Chang and Vishwanath Sindagi for their helpful feedback and recommendations.

References
----------

*   Alayrac et al. [2022] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karén Simonyan. Flamingo: a visual language model for few-shot learning. _Advances in neural information processing systems_, 2022. 
*   Awadalla et al. [2023] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, and Ludwig Schmidt. Openflamingo: An open-source framework for training large autoregressive vision-language models. _arXiv preprint arXiv:2308.01390_, 2023. 
*   Bai et al. [2023] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. _arXiv preprint arXiv:2309.16609_, 2023. 
*   Bai et al. [202k] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. _arXiv preprint arXiv:2308.12966_, 202k. 
*   Bavishi et al. [2023] Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, and Sağnak Taşırlar. Introducing our multimodal models, 2023. 
*   Beyer et al. [2024] Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, Thomas Unterthiner, Daniel Keysers, Skanda Koppula, Fangyu Liu, Adam Grycner, Alexey Gritsenko, Neil Houlsby, Manoj Kumar, Keran Rong, Julian Eisenschlos, Rishabh Kabra, Matthias Bauer, Matko Bošnjak, Xi Chen, Matthias Minderer, Paul Voigtlaender, Ioana Bica, Ivana Balazevic, Joan Puigcerver, Pinelopi Papalampidi, Olivier Henaff, Xi Xiong, Radu Soricut, Jeremiah Harmsen, and Xiaohua Zhai. Paligemma: A versatile 3b vlm for transfer, 2024. 
*   Cai et al. [2024] Mu Cai, Jianwei Yang, Jianfeng Gao, and Yong Jae Lee. Matryoshka multimodal models. _arXiv preprint arXiv:2405.17430_, 2024. 
*   ”Cao and Xiao [”2022”] Jie ”Cao and Jing” Xiao. ”an augmented benchmark dataset for geometric question answering through dual parallel text encoding”. In _”Proceedings of the 29th International Conference on Computational Linguistics”_, ”2022”. 
*   Cha et al. [2024] Junbum Cha, Wooyoung Kang, Jonghwan Mun, and Byungseok Roh. Honeybee: Locality-enhanced projector for multimodal llm. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, 2024. 
*   Changpinyo et al. [2021] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 3558–3568, 2021. 
*   Chen et al. [2024a] Jiuhai Chen, Jianwei Yang, Haiping Wu, Dianqi Li, Jianfeng Gao, Tianyi Zhou, and Bin Xiao. Florence-vl: Enhancing vision-language models with generative vision encoder and depth-breadth fusion. _arXiv preprint arXiv:2412.04424_, 2024a. 
*   Chen et al. [2024b] Jieneng Chen, Qihang Yu, Xiaohui Shen, Alan Yuille, and Liang-Chieh Chen. Vitamin: Designing scalable vision models in the vision-language era. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, 2024b. 
*   Chen et al. [2023a] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. _arXiv preprint arXiv:2311.12793_, 2023a. 
*   Chen et al. [2024c] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models, 2024c. 
*   Chen et al. [2023b] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. _arXiv preprint arXiv:2312.14238_, 2023b. 
*   Chen et al. [2024d] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. _arXiv preprint arXiv:2404.16821_, 2024d. 
*   Chenglin Yang et al [2023] Chenglin Yang et al. MOAT: Alternating mobile convolution and attention brings strong vision models. In _ICLR_, 2023. 
*   Chu et al. [2023] Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu, Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang, Xiaolin Wei, et al. Mobilevlm: A fast, reproducible and strong vision language assistant for mobile devices. _arXiv preprint arXiv:2312.16886_, 2023. 
*   Chu et al. [2024] Xiangxiang Chu, Limeng Qiao, Xinyu Zhang, Shuang Xu, Fei Wei, Yang Yang, Xiaofei Sun, Yiming Hu, Xinyang Lin, Bo Zhang, et al. Mobilevlm v2: Faster and stronger baseline for vision language model. _arXiv preprint arXiv:2402.03766_, 2024. 
*   Dai et al. [2023] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023. 
*   DeepSeek-AI [2024] DeepSeek-AI. Deepseek llm: Scaling open-source language models with longtermism. _arXiv preprint arXiv:2401.02954_, 2024. 
*   Deitke et al. [2024] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, Jiasen Lu, Taira Anderson, Erin Bransom, Kiana Ehsani, Huong Ngo, YenSung Chen, Ajay Patel, Mark Yatskar, Chris Callison-Burch, Andrew Head, Rose Hendrix, Favyen Bastani, Eli VanderBilt, Nathan Lambert, Yvonne Chou, Arnavi Chheda, Jenna Sparks, Sam Skjonsberg, Michael Schmitz, Aaron Sarnat, Byron Bischoff, Pete Walsh, Chris Newell, Piper Wolters, Tanmay Gupta, Kuo-Hao Zeng, Jon Borchardt, Dirk Groeneveld, Jen Dumas, Crystal Nam, Sophie Lebrecht, Caitlin Wittlif, Carissa Schoenick, Oscar Michel, Ranjay Krishna, Luca Weihs, Noah A. Smith, Hannaneh Hajishirzi, Ross Girshick, Ali Farhadi, and Aniruddha Kembhavi. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. _arXiv preprint arXiv:2409.17146_, 2024. 
*   Diao et al. [2024] Haiwen Diao, Yufeng Cui, Xiaotong Li, Yueze Wang, Huchuan Lu, and Xinlong Wang. Unveiling encoder-free vision-language models. _arXiv preprint arXiv:2406.11832_, 2024. 
*   Dosovitskiy et al. [2020] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. _arXiv preprint arXiv:2010.11929_, 2020. 
*   Duan et al. [2024] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In _Proceedings of the 32nd ACM International Conference on Multimedia_, 2024. 
*   Fang et al. [2023] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks. _arXiv preprint arXiv:2309.17425_, 2023. 
*   Gadre et al. [2023] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. _arXiv preprint arXiv:2304.14108_, 2023. 
*   Ge et al. [2024] Chunjiang Ge, Sijie Cheng, Ziming Wang, Jiale Yuan, Yuan Gao, Jun Song, Shiji Song, Gao Huang, and Bo Zheng. Convllava: Hierarchical backbones as visual encoder for large multimodal models, 2024. 
*   Goyal et al. [2017] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In _Proceedings of the IEEE conference on computer vision and pattern recognition_, pages 6904–6913, 2017. 
*   Guo et al. [2024] Jarvis Guo, Tuney Zheng, Yuelin Bai, Bo Li, Yubo Wang, King Zhu, Yizhi Li, Graham Neubig, Wenhu Chen, and Xiang Yue. Mammoth-vl: Eliciting multimodal reasoning with instruction tuning at scale. _arXiv preprint arXiv:2412.05237_, 2024. 
*   Hannun et al. [2023] Awni Hannun, Jagrit Digani, Angelos Katharopoulos, and Ronan Collobert. MLX: Efficient and flexible machine learning on apple silicon, 2023. 
*   Hu et al. [2024] Wenbo Hu, Zi-Yi Dou, Liunian Harold Li, Amita Kamath, Nanyun Peng, and Kai-Wei Chang. Matryoshka query transformer for large vision-language models, 2024. 
*   Huang et al. [2024] Wenxuan Huang, Zijie Zhai, Yunhang Shen, Shaoshen Cao, Fei Zhao, Xiangfeng Xu, Zheyu Ye, and Shaohui Lin. Dynamic-llava: Efficient multimodal large language models via dynamic vision-language context sparsification, 2024. 
*   Hudson and Manning [2019] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_, pages 6700–6709, 2019. 
*   Kafle et al. [2018] Kushal Kafle, Scott Cohen, Brian Price, and Christopher Kanan. Dvqa: Understanding data visualizations via question answering. In _CVPR_, 2018. 
*   Karamcheti et al. [2024] Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic vlms: Investigating the design space of visually-conditioned language models. In _International Conference on Machine Learning (ICML)_, 2024. 
*   Kembhavi et al. [2016] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images, 2016. 
*   Kim et al. [2022] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In _European Conference on Computer Vision (ECCV)_, 2022. 
*   Kirillov et al. [2023] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything. _arXiv preprint arXiv:2304.02643_, 2023. 
*   Krishna et al. [2017] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A. Shamma, Michael S. Bernstein, and Li Fei-Fei. Visual genome: Connecting language and vision using crowdsourced dense image annotations. _International Journal of Computer Vision_, 2017. 
*   Laurençon et al. [2024] Hugo Laurençon, Andrés Marafioti, Victor Sanh, and Léo Tronchon. Building and better understanding vision-language models: insights and future directions., 2024. 
*   Li et al. [2023a] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. _arXiv preprint arXiv:2307.16125_, 2023a. 
*   Li et al. [2024a] Bo Li, Hao Zhang, Kaichen Zhang, Dong Guo, Yuanhan Zhang, Renrui Zhang, Feng Li, Ziwei Liu, and Chunyuan Li. Llava-next: What else influences visual instruction tuning beyond data?, 2024a. 
*   Li et al. [2024b] Bo Li, Kaichen Zhang, Hao Zhang, Dong Guo, Renrui Zhang, Feng Li, Yuanhan Zhang, Ziwei Liu, and Chunyuan Li. Llava-next: Stronger llms supercharge multimodal capabilities in the wild, 2024b. 
*   Li et al. [2024c] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. _arXiv preprint arXiv:2408.03326_, 2024c. 
*   Li et al. [2025] Junyan Li, Delin Chen, Tianle Cai, Peihao Chen, Yining Hong, Zhenfang Chen, Yikang Shen, and Chuang Gan. Flexattention for efficient high-resolution vision-language models. In _European Conference on Computer Vision_, pages 286–302. Springer, 2025. 
*   Li et al. [2024d] Kevin Y. Li, Sachin Goyal, Joao D. Semedo, and J.Zico Kolter. Inference optimal vlms need only one visual token but larger models, 2024d. 
*   Li et al. [2023b] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. _arXiv preprint arXiv:2305.10355_, 2023b. 
*   Li et al. [2023c] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. _arXiv preprint arXiv:2403.18814_, 2023c. 
*   Lin et al. [2023a] Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models, 2023a. 
*   Lin et al. [2014] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In _European Conference on Computer Vision_, pages 740–755, 2014. 
*   Lin et al. [2023b] Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. _arXiv preprint arXiv:2311.07575_, 2023b. 
*   Liu et al. [2023a] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023a. 
*   Liu et al. [2023b] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In _Advances in Neural Information Processing Systems (NeurIPS)_, 2023b. 
*   Liu et al. [2024a] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024a. 
*   Liu et al. [2024b] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xucheng Yin, Cheng lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: On the hidden mystery of ocr in large multimodal models, 2024b. 
*   Liu et al. [2022] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, 2022. 
*   Lu et al. [2024a] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, Yaofeng Sun, Chengqi Deng, Hanwei Xu, Zhenda Xie, and Chong Ruan. Deepseek-vl: Towards real-world vision-language understanding, 2024a. 
*   Lu et al. [2022] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. _Advances in Neural Information Processing Systems_, 2022. 
*   Lu et al. [2024b] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In _International Conference on Learning Representations (ICLR)_, 2024b. 
*   Luo et al. [2024] Gen Luo, Yiyi Zhou, Yuxin Zhang, Xiawu Zheng, Xiaoshuai Sun, and Rongrong Ji. Feast your eyes: Mixture-of-resolution adaptation for multimodal large language models. _arXiv preprint arXiv:2403.03003_, 2024. 
*   Marafioti et al. [2025] Andrés Marafioti, Orr Zohar, Miquel Farré, Merve Noyan, Elie Bakouch, Pedro Cuenca, Cyril Zakka, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, Vaibhav Srivastav, Joshua Lochner, Hugo Larcher, Mathieu Morlon, Lewis Tunstall, Leandro von Werra, and Thomas Wolf. Smolvlm: Redefining small and efficient multimodal models. _arXiv preprint arXiv:2504.05299_, 2025. 
*   Masry et al. [”2022”] Ahmed Masry, Do Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In _”Findings of the Association for Computational Linguistics: ACL 2022”_, ”2022”. 
*   Mathew et al. [2021a] Minesh Mathew, Viraj Bagal, Rubèn Pérez Tito, Dimosthenis Karatzas, Ernest Valveny, and C.V Jawahar. Infographicvqa, 2021a. 
*   Mathew et al. [2021b] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In _Proceedings of the IEEE/CVF winter conference on applications of computer vision_, pages 2200–2209, 2021b. 
*   McKinzie et al. [2024] Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, Anton Belyi, Haotian Zhang, Karanjeet Singh, Doug Kang, Ankur Jain, Hongyu Hè, Max Schwarzer, Tom Gunter, Xiang Kong, Aonan Zhang, Jianyu Wang, Chong Wang, Nan Du, Tao Lei, Sam Wiseman, Guoli Yin, Mark Lee, Zirui Wang, Ruoming Pang, Peter Grasch, Alexander Toshev, and Yinfei Yang. Mm1: Methods, analysis & insights from multimodal llm pre-training, 2024. 
*   Mishra et al. [2019] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In _ICDAR_, 2019. 
*   OpenAI [2023] OpenAI. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_, 2023. 
*   Radford et al. [2021] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_, pages 8748–8763. PMLR, 2021. 
*   Shang et al. [2024] Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. _arXiv preprint arXiv:2403.15388_, 2024. 
*   Sharma et al. [2018] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, pages 2556–2565, 2018. 
*   Shi et al. [2024a] Baifeng Shi, Ziyang Wu, Maolin Mao, Xin Wang, and Trevor Darrell. When do we not need larger vision models? In _European Conference on Computer Vision (ECCV)_, 2024a. 
*   Shi et al. [2024b] Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, Humphrey Shi, Bryan Catanzaro, Andrew Tao, Jan Kautz, Zhiding Yu, and Guilin Liu. Eagle: Exploring the design space for multimodal llms with mixture of encoders. _arXiv:2408.15998_, 2024b. 
*   Singh et al. [2019] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_, pages 8317–8326, 2019. 
*   Sun et al. [2023] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. _arXiv preprint arXiv:2303.15389_, 2023. 
*   Team [2024a] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models, 2024a. 
*   Team [2024b] Qwen Team. Qwen2.5: A party of foundation models, 2024b. 
*   Tong et al. [2024] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms, 2024. 
*   Touvron et al. [2023] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023. 
*   Tsimpoukelli et al. [2021] Maria Tsimpoukelli, Jacob Menick, Serkan Cabi, SM Eslami, Oriol Vinyals, and Felix Hill. Multimodal few-shot learning with frozen language models. _Conference on Neural Information Processing Systems (NeurIPS)_, 2021. 
*   Vasu et al. [2023a] Pavan Kumar Anasosalu Vasu, James Gabriel, Jeff Zhu, Oncel Tuzel, and Anurag Ranjan. Mobileone: An improved one millisecond mobile backbone. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, 2023a. 
*   Vasu et al. [2023b] Pavan Kumar Anasosalu Vasu, James Gabriel, Jeff Zhu, Oncel Tuzel, and Anurag Ranjan. Fastvit: A fast hybrid vision transformer using structural reparameterization. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_, 2023b. 
*   Vasu et al. [2024] Pavan Kumar Anasosalu Vasu, Hadi Pouransari, Fartash Faghri, Raviteja Vemulapalli, and Oncel Tuzel. Mobileclip: Fast image-text models through multi-modal reinforced training. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, 2024. 
*   Vaswani et al. [2017] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _Advances in Neural Information Processing Systems_, 2017. 
*   Xue et al. [2024] Le Xue, Manli Shu, Anas Awadalla, Jun Wang, An Yan, Senthil Purushwalkam, Honglu Zhou, Viraj Prabhu, Yutong Dai, Michael S. Ryoo, Shrikant Kendre, Jieyu Zhang, Can Qin, Shu Zhang, Chia-Chih Chen, Ning Yu, Juntao Tan, Tulika Manoj Awalgaonkar, Shelby Heinecke, Huan Wang, Yejin Choi, Ludwig Schmidt, Zeyuan Chen, Silvio Savarese, Juan Carlos Niebles, Caiming Xiong, and Ran Xu. xgen-mm (BLIP-3): A family of open large multimodal models. _CoRR_, abs/2408.08872, 2024. 
*   Yang et al. [2024a] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2 technical report. _arXiv preprint arXiv:2407.10671_, 2024a. 
*   Yang et al. [2024b] Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip: Longer is better but not necessary in vision language models, 2024b. 
*   Ye et al. [2024] Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl3: Towards long image-sequence understanding in multi-modal large language models, 2024. 
*   Ye et al. [2023a] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, Chaoya Jiang, Chenliang Li, Yuanhong Xu, Hehong Chen, Junfeng Tian, Qian Qi, Ji Zhang, and Fei Huang. mplug-owl: Modularization empowers large language models with multimodality, 2023a. 
*   Ye et al. [2023b] Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration, 2023b. 
*   Yu et al. [2023] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. _arXiv preprint arXiv:2308.02490_, 2023. 
*   Yu et al. [2024] Weihao Yu, Chenyang Si, Pan Zhou, Mi Luo, Yichen Zhou, Jiashi Feng, Shuicheng Yan, and Xinchao Wang. Metaformer baselines for vision. _IEEE Transactions on Pattern Analysis and Machine Intelligence_, 2024. 
*   Yue et al. [2024] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 9556–9567, 2024. 
*   Zhai et al. [2023] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. _International Conference on Computer Vision (ICCV)_, 2023. 
*   Zhang et al. [2024a] Haotian Zhang, Mingfei Gao, Zhe Gan, Philipp Dufter, Nina Wenzel, Forrest Huang, Dhruti Shah, Xianzhi Du, Bowen Zhang, Yanghao Li, Sam Dodge, Keen You, Zhen Yang, Aleksei Timofeev, Mingze Xu, Hong-You Chen, Jean-Philippe Fauconnier, Zhengfeng Lai, Haoxuan You, Zirui Wang, Afshin Dehghan, Peter Grasch, and Yinfei Yang. Mm1.5: Methods, analysis & insights from multimodal llm fine-tuning, 2024a. 
*   Zhang et al. [2024b] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmms-eval: Reality check on the evaluation of large multimodal models, 2024b. 
*   Zhang et al. [2024c] Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, et al. Sparsevlm: Visual token sparsification for efficient vision-language model inference. _arXiv preprint arXiv:2410.04417_, 2024c. 
*   Zheng et al. [2023] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric.P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023. 
*   Zhu et al. [2023] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_, 2023. 

\thetitle

Supplementary Material

A [Training Setup](https://arxiv.org/html/2412.13303)
-----------------------------------------------------

For experiments presented in [Tab.1](https://arxiv.org/html/2412.13303v2#S3.T1 "In 3.1 FastViT as VLM Image Encoder ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), [Tab.2](https://arxiv.org/html/2412.13303v2#S3.T2 "In 3.1 FastViT as VLM Image Encoder ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), [Tab.4](https://arxiv.org/html/2412.13303v2#S3.T4 "In 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), [Tab.5](https://arxiv.org/html/2412.13303v2#S3.T5 "In 3.2.3 Comparison with Token Pruning & Downsampling ‣ 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), we perform 2-stage training with the hyperparameters listed in [Tab.8](https://arxiv.org/html/2412.13303v2#S1.T8 "In A Training Setup ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"). The model is trained for a single epoch in all the stages. Note, in [Tab.5](https://arxiv.org/html/2412.13303v2#S3.T5 "In 3.2.3 Comparison with Token Pruning & Downsampling ‣ 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), we do not re-train other token pruning works, we simply report the performance of the respective methods as they adhere to the 2-stage training setup described in [Tab.8](https://arxiv.org/html/2412.13303v2#S1.T8 "In A Training Setup ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), which was originally introduced in LLaVA-1.5[[53](https://arxiv.org/html/2412.13303v2#bib.bib53)].

To showcase our model’s performance in the presence of additional dataset, we scale both pretraining and instruction tuning datasets in [Sec.4](https://arxiv.org/html/2412.13303v2#S4 "4 Experiments ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"). For results presented in R15, R19, R20, R27, R28 in [Tab.6](https://arxiv.org/html/2412.13303v2#S3.T6 "In 3.2.3 Comparison with Token Pruning & Downsampling ‣ 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), we still perform 2-stage training described in [Tab.8](https://arxiv.org/html/2412.13303v2#S1.T8 "In A Training Setup ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), for R20 and R28, we use instruction tuning dataset of size 1.1 million samples in Stage-2. For results presented in R3, R4, R7, R8, R11, R12, R16, R21, R22, R23, R29, R30, R40 and R41 we scale-up both instruction tuning dataset and pretraining dataset. We also introduce and additional stage of pretraining with the scaled-up dataset as described in [Tab.9](https://arxiv.org/html/2412.13303v2#S1.T9 "In A Training Setup ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"). Details of 1.1 million, 6.5 million and 12.5 million instruction tuning dataset is presented in Sec.[D](https://arxiv.org/html/2412.13303v2#supp_datasets_info "Datasets ‣ D Datasets ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"). For results presented in R5, R13 and R42, we introduce additional finetuning on high quality instruction tuning dataset from[[30](https://arxiv.org/html/2412.13303v2#bib.bib30)] in Stage 3. We publicly release R4, R12, and R41 checkpoints from Stage 2 training, as well as the R5, R13, and R42 checkpoints from Stage 3 training, as part of our open-sourced codebase.

|  | Stage-1 | Stage-2 |
| --- |
| Data | LLaVA-1.5 558K | LLaVA-1.5 665k |
| Learning Rate | 1e-3 | 2e-5 |
| Batch size | 256 | 128 |
| LR. schedule | cosine decay | cosine decay |
| LR. warmup ratio | 0.03 | 0.03 |
| Optimizer | AdamW | AdamW |
| Trainable | Projector | Full |
| modules | Model |

Table 8: 2-Stage training setup used in ablations for [Sec.3](https://arxiv.org/html/2412.13303v2#S3 "3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models").

|  | Stage-1 | Stage-1.5 | Stage-2 | Stage 3 |
| --- |
| Data | LLaVA-1.5 558K | Recap-CC3M + | 1.1M / 6.5M | 10.6M |
| Recap-CC12M[[43](https://arxiv.org/html/2412.13303v2#bib.bib43)] | 12.5M |
| Learning Rate | 1e-3 | 2e-5 | 2e-5 | 2e-5 |
| Batch size | 256 | 128 | 128 | 128 |
| LR. schedule | cosine decay | cosine decay | cosine decay | cosine decay |
| LR. warmup ratio | 0.03 | 0.03 | 0.03 | 0.03 |
| Optimizer | AdamW | AdamW | AdamW | AdamW |
| Trainable | Projector | Full | Full | Full |
| modules | Model | Model | Model |

Table 9: 4-Stage training setup. We report performance after Stage-2 and Stage-3 of training in [Tab.6](https://arxiv.org/html/2412.13303v2#S3.T6 "In 3.2.3 Comparison with Token Pruning & Downsampling ‣ 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"). Please note, both Stage-2 and Stage-3 are visual instruction tuning stages.

B [Architecture Details](https://arxiv.org/html/2412.13303)
-----------------------------------------------------------

The patch embedding layers shown in [Fig.2](https://arxiv.org/html/2412.13303v2#S3.F2 "In 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), consists of 7×\times×7 depthwise convolutions with[[81](https://arxiv.org/html/2412.13303v2#bib.bib81)] style train-time overparameterization, followed by 1×\times×1 pointwise convolution. The stride for 7×\times×7 depthwise convolution is set to 2 in order to downsample the input tensor. In[[83](https://arxiv.org/html/2412.13303v2#bib.bib83)], squeeze-excite layers were incorporated into this block; however, we found them to negatively impact inference latency, especially for high image resolutions, so we opted not to include them in our model. We use the same ConvFFN layer defined in[[82](https://arxiv.org/html/2412.13303v2#bib.bib82)], i.e. 7×\times×7 depthwise convolutions preceding a typical FFN layer. The stem downsamples the input tensor by factor of 4 on each side, and each patch embedding layer downsamples the input tensor by a factor 2. Although recent architectures like ViTamin[[12](https://arxiv.org/html/2412.13303v2#bib.bib12)] recommend an overall downsampling factor of only 16, FastViTHD incorporates an additional patch embedding layer compared to FastViT, resulting in an overall downsampling factor of 64× for the input tensor. In each stage, we increase the number of channels by a factor of 2 as done in FastViT and other convolutional and hybrid transformer architectures. This results in a Stage-5 with the widest MLP layers in the architecture, performing self-attention on an input tensor which is downsampled by a factor of 64.

### B.1 Naive Scaling

In order to scale the model size of FastViT, we simply increased the embedding dimensions per stage to [128, 256, 512, 1024], and set the number of layers per stage to [2, 12, 16, 6]. Patch embedding layers in each stage use squeeze-excite layers and the MLP expansion ratio is set to 3.0, following the design in[[83](https://arxiv.org/html/2412.13303v2#bib.bib83)].

Row Method Vision LLM Input#Visual Vis. Enc.Vision Enc.LLM
Ann.Encoder Res.Tokens Size(M)↓↓\downarrow↓Latency(ms)↓↓\downarrow↓Prefilling(ms)↓↓\downarrow↓
0.5B Model Comparison
R1 nanoLLaVA ViT-SO400M Qw.1.5 384 729 430 272.1 263.3
R2 LLaVAOV[[45](https://arxiv.org/html/2412.13303v2#bib.bib45)]∗ViT-SO400M Qw.2 1152 7290 430 2721.4 11402.4
R3 FastVLM (Ours)FastViTHD Qw.2 1024 256 125 116.3 50.5
R3 FastVLM (Ours)∗FastViTHD Qw.2 2048 1280 125 581.5 336.4
1-2B Model Comparison
R4 MobileVLMv2[[19](https://arxiv.org/html/2412.13303v2#bib.bib19)]ViT-L/14 ML.336 144 304 127.4 458
R5 FastVLM (Ours)FastViTHD Qw.2 768 144 125 54.8 97.1
R6 DeepSeekVL[[58](https://arxiv.org/html/2412.13303v2#bib.bib58)]ViT-SO400M DS.384 576 430 272.1-
R7 MM1[[66](https://arxiv.org/html/2412.13303v2#bib.bib66)]∗ViT-H-1344 720 632--
R8 FastVLM (Ours)FastViTHD Qw.2 1024 256 125 116.3 116.1
R8 FastVLM (Ours)∗FastViTHD Qw.2 2048 1280 125 581.5 681.7
7B Model Comparison
R9 InstructBLIP[[20](https://arxiv.org/html/2412.13303v2#bib.bib20)]ViT-g/14 Vic.224 32 1012 149.5 152.1
R11 FastVLM (Ours)FastViTHD Vic.256 16 125 6.8 143.4
R12 MobileVLMv2[[19](https://arxiv.org/html/2412.13303v2#bib.bib19)]ViT-L/14 Vic.336 144 304 127.4 332.1
R13 ConvLLaVA[[28](https://arxiv.org/html/2412.13303v2#bib.bib28)]ConvNeXT-L Vic.768 144 200 164.3 332.1
R14 FastVLM (Ours)FastViTHD Vic.768 144 125 54.8 332.1
R17 FastVLM (Ours)FastViTHD Qw.2 768 144 125 54.8 391.2
R20 ConvLLaVA[[28](https://arxiv.org/html/2412.13303v2#bib.bib28)]ConvNeXT-L Vic.1024 256 200 696.1 461.1
R26 LLaVA-1.5[[53](https://arxiv.org/html/2412.13303v2#bib.bib53)]ViT-L/14 Vic.336 576 304 127.4 1170.0
R27 MobileVLMv2[[19](https://arxiv.org/html/2412.13303v2#bib.bib19)]336 576 304 127.4 1170.0
R28 ShareGPT4V[[13](https://arxiv.org/html/2412.13303v2#bib.bib13)]336 576 304 127.4 1170.0
R29 ViTamin[[12](https://arxiv.org/html/2412.13303v2#bib.bib12)]ViTamin-L Vic.384 576 333 137.6 1170.0
R30 ConvLLaVA[[28](https://arxiv.org/html/2412.13303v2#bib.bib28)]ConvNeXT-L Vic.1536 576 200 1569.7 1170.0
R31 VILA[[50](https://arxiv.org/html/2412.13303v2#bib.bib50)]ViT-L/14 L-2 336 576 304 127.4 1169.5
R33 MM1[[66](https://arxiv.org/html/2412.13303v2#bib.bib66)]∗ViT-H-1344 720 632--
R34 LLaVA-NeXT∗ViT-L/14 L-3 672 2880 304 637.0 19709.7
R21 FastVLM (Ours)FastViTHD Vic.1024 256 125 116.3 461.1
R36 FastVLM (Ours)FastViTHD Qw.2 1024 256 125 116.3 524.5
R36 FastVLM (Ours)∗FastViTHD Qw.2 2048 1280 125 581.5 3139.5
VLMs with Multiple Vision Encoders and 8B LLM
ConvNeXT-L 1536 200 1569.7
R35 MiniGemini-HD ViT-L/14 L-3 672 2880 304 552.6 19709.7
ViT-SO400M 384 430 272.1
ConvNeXt-XXL 1024 846 2290.4
DINOv2-ViT-L/14 518 304 1171.5
R36 Cambrian-1[[78](https://arxiv.org/html/2412.13303v2#bib.bib78)]ViT-L/14 L-3 336 576 304 127.4 1223.6

Table 10: Breakdown of prefilling latencies for recent methods. The models are grouped based on total number of visual tokens. For models that were difficult to export or unavailable, we mark them as ’-’ in the table. “Vic.” refers to Vicuna[[98](https://arxiv.org/html/2412.13303v2#bib.bib98)], “Qw.2” refers to Qwen2[[86](https://arxiv.org/html/2412.13303v2#bib.bib86)] and “Qw.” refers to Qwen[[3](https://arxiv.org/html/2412.13303v2#bib.bib3)]. “L-2” refers to LLaMA-2. “L-3” refers to LLaMA-3. “ML.” refers to MobileLLaMA[[18](https://arxiv.org/html/2412.13303v2#bib.bib18), [19](https://arxiv.org/html/2412.13303v2#bib.bib19)]. “DS.” refers to DeepSeek LLM[[21](https://arxiv.org/html/2412.13303v2#bib.bib21)]. ∗ For input resolution and visual tokens, we report the highest supported resolution by the respective models as some models like LLaVA-OneVision[[45](https://arxiv.org/html/2412.13303v2#bib.bib45)] and MM1[[66](https://arxiv.org/html/2412.13303v2#bib.bib66)] use dynamic input resolution. FastVLM models using dynamic resolution employs a simple 2×\times×2 grid, with tile size set to 1024. For VLMs that use multiple vision encoders, the size of each encoder is listed independently, for TTFT, the latency from each encoder is summed up. 

C Additional Results
--------------------

We present the performance of FastVLM on text-rich benchmarks under various training settings in [Tab.11](https://arxiv.org/html/2412.13303v2#S3.T11 "In C Additional Results ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"). FastVLM surpasses MM1 and Cambrian-1 across a wide range of benchmarks by scaling up pretraining and instruction tuning datasets. This result highlights the quality of visual tokens produced by FastViTHD, as FastVLM is able to achieve these improvements with 2.8×\times× less visual tokens than MM1 and with a vision encoder that is 5.1×\times× smaller.

Row Method Vision LLM Data (M)Input#Visual Vis. Enc.TTFT ChartQA OCRBench TextVQA DocVQA InfoVQA
Ann.Encoder(PT+IT)Res.Tokens Size(M)↓↓\downarrow↓(ms)↓↓\downarrow↓
0.5B Model Comparison
R1 LLaVAOV[[45](https://arxiv.org/html/2412.13303v2#bib.bib45)]∗ViT-SO400M Qw.2 4.5+3.2 1152 7290 430 14124 61.4--70.0 46.3
R2 FastVLM (Ours)FastViTHD Qw.2 15+12.5 1024 256 125 166 63.4 54.9 62.9 70.4 35.8
R3 FastVLM (Ours)FastViTHD Qw.2 15+23.1 1024 256 125 166 71.4 55.8 65.8 79.1 43.3
R4 FastVLM (Ours)∗FastViTHD Qw.2 15+12.5 2048 1280 125 918 68.8 59.0 65.4 82.1 49.3
R5 FastVLM (Ours)∗FastViTHD Qw.2 15+23.1 2048 1280 125 918 64.2 55.7 65.5 85.3 53.9
1-2B Model Comparison
R6 MM1[[66](https://arxiv.org/html/2412.13303v2#bib.bib66)]∗ViT-H-3000+1.5 1344 720 632-61.8 56.6 68.2 68.4 38.5
R7 FastVLM (Ours)FastViTHD Qw.2 15+12.5 1024 256 125 233 69.6 62.9 69.0 75.6 41.7
R8 FastVLM (Ours)FastViTHD Qw.2 15+23.1 1024 256 125 233 69.5 61.2 71.2 84.2 49.6
R9 FastVLM (Ours)∗FastViTHD Qw.2 15+12.5 2048 1280 125 1263 76.4 63.2 71.5 87.6 60.0
R10 FastVLM (Ours)∗FastViTHD Qw.2 15+12.5 2048 1280 125 1263 71.2 62.4 72.9 90.2 62.5
7B Model Comparison
R11 MM1[[66](https://arxiv.org/html/2412.13303v2#bib.bib66)]∗ViT-H-3000+1.5 1344 720 632-72.6 62.6 72.8 76.8 45.5
R12 LLaVA-NeXT†∗ViT-L/14 L-3-672 2880 304 20347 69.5 49.0 64.6 72.6-
R13 Cambrian-1[[78](https://arxiv.org/html/2412.13303v2#bib.bib78)]ViT-L/14 L-3 2.5+7 336 576 304 5085 73.3 62.4 71.7 77.8-
ViT-SO400M 384 430
ConvNeXt-XXL 1024 846
DINOv2-ViT-L/14 518 304
R14 FastVLM (Ours)FastViTHD Vic.0.5+0.6 768 144 125 387 17.1 30.0 62.9 32.9 28.7
R15 FastVLM (Ours)FastViTHD Vic.0.5+1.1 768 144 125 387 59.1 38.4 67.5 57.3 29.7
R16 FastVLM (Ours)FastViTHD Vic.15+1.1 768 144 125 387 65.4 45.3 69.4 65.5 32.0
R17 FastVLM (Ours)FastViTHD Qw.2 15+1.1 768 144 125 446 69.3 45.9 69.5 66.9 34.3
R18 FastVLM (Ours)FastViTHD Qw.2 15+11.9 768 144 125 446 74.2 59.0 72.8 72.0 44.3
R19 FastVLM (Ours)FastViTHD Vic.0.5+0.6 1024 256 125 577 19.2 29.3 64.4 35.6 28.9
R20 FastVLM (Ours)FastViTHD Vic.0.5+1.1 1024 256 125 577 61.0 38.3 67.4 62.8 32.0
R21 FastVLM (Ours)FastViTHD Vic.15+1.1 1024 256 125 577 66.9 47.1 70.6 72.4 34.7
R22 FastVLM (Ours)FastViTHD Qw.2 15+1.1 1024 256 125 641 71.0 49.7 72.1 73.3 37.5
R23 FastVLM (Ours)FastViTHD Qw.2 15+6.5 1024 256 125 641 76.6 52.9 73.1 78.7 44.2
R24 FastVLM (Ours)FastViTHD Qw.2 15+11.9 1024 256 125 641 77.0 63.3 74.8 78.9 49.7
R25 FastVLM (Ours)FastViTHD Qw.2 15+12.5 1024 256 125 641 77.5 65.7 73.4 82.7 51.2
R26 FastVLM (Ours)FastViTHD Qw.2 15+23.1 1024 256 125 641 78.1 67.3 76.1 88.0 55.1
R27 FastVLM (Ours)∗FastViTHD Qw.2 15+12.5 2048 1280 125 3721 82.4 67.3 76.6 92.3 68.3
R28 FastVLM (Ours)∗FastViTHD Qw.2 15+23.1 2048 1280 125 3721 74.6 65.7 77.1 94.5 71.1

Table 11: Comparison with recent methods on text-rich benchmarks. The models are grouped based on total number of visual tokens. “-” indicates that performance was not reported in the respective paper. For the dataset column, “-” indicates that the dataset size for pretraining (“PT”) or instruction tuning (“IT”) is not explicitly mentioned in the respective paper. For methods that have more than 2 stages of training, we report the total samples used for all the pretraining stages as part of “PT”. “TTFT” means time to first token (the sum of the vision encoder latency and the LLM prefilling time), we report latency only for models that are publicly available and in a format favorable to MLX[[31](https://arxiv.org/html/2412.13303v2#bib.bib31)] “Vic.” refers to Vicuna[[98](https://arxiv.org/html/2412.13303v2#bib.bib98)], “Qw.2” refers to Qwen2[[86](https://arxiv.org/html/2412.13303v2#bib.bib86)]. “L-3” refers to LLaMA-3. * - For input resolution and visual tokens, we report the highest supported resolution by the respective models that use dynamic input resolution. †- performance numbers reported from[[78](https://arxiv.org/html/2412.13303v2#bib.bib78)]. For VLMs that use multiple vision encoders, the size of each encoder is listed independently, for TTFT, the latency from each encoder is summed up. 

### C.1 [Dynamic Resolution (AnyRes) Results](https://arxiv.org/html/2412.13303)

From [Fig.6](https://arxiv.org/html/2412.13303v2#S3.F6 "In 3.2.1 Vision Encoder - Language Decoder Interplay ‣ 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), it is evident that VLMs prefer visual encoding with fewer semantic breaks. Variants with more tiles typically underperform compared to those with fewer tiles and a static resolution. To scale up input resolution, we train variants of FastVLM with support for dynamic input resolution, where we use a tile size of 1024×\times×1024 and use a simple 2×\times×2 grid. This enables the model to process a peak input resolution of 2048×\times×2048 using only 4 tiles, unlike models like InternVL2[[16](https://arxiv.org/html/2412.13303v2#bib.bib16)] which uses roughly 36 tiles to process images of resolution 2688×\times×2688. We report performance of FastVLM (R4, R5, R9, R10, R27, R28) with dynamic resolution support on text-rich benchmarks in [Tab.11](https://arxiv.org/html/2412.13303v2#S3.T11 "In C Additional Results ‣ FastVLM: Efficient Vision Encoding for Vision Language Models").

### C.2 [CVBench and MathVista Results](https://arxiv.org/html/2412.13303)

Results in [Tab.12](https://arxiv.org/html/2412.13303v2#S3.T12 "In C.2 CVBench and MathVista Results ‣ C Additional Results ‣ FastVLM: Efficient Vision Encoding for Vision Language Models") show that, in comparison to Cambrian-1[[78](https://arxiv.org/html/2412.13303v2#bib.bib78)], FastVLM is significantly better on MathVista[[60](https://arxiv.org/html/2412.13303v2#bib.bib60)] and competitive on CVBench[[78](https://arxiv.org/html/2412.13303v2#bib.bib78)], even though we have a single backbone and significantly fewer tokens. Results on both CVBench and MathVista benchmarks further improve as we scale the SFT dataset by including LLaVA-OneVision[[45](https://arxiv.org/html/2412.13303v2#bib.bib45)].

| Method | LLM | Data (M) | Input | #Visual | Latency | CVBench | CVBench | Math |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decoder | (PT+IT) | Res. | Tokens | Enc.(ms) | 2D | 3D | Vista |
| Cambrian-1 | LLama3-8B | 2.5+7 | Mult. | 576 | 3861.4 | 72.3 | 72.0 | 49.0 |
| FastVLM | Qwen2-7B | 15+6.5 | 1024 | 256 | 116.3 | 71.8 | 69.6 | 57.6 |
| FastVLM | Qwen2-7B | 15+11.9 | 768 | 144 | 54.8 | 75.9 | 79.3 | 64.6 |
| FastVLM | Qwen2-7B | 15+11.9 | 1024 | 256 | 116.3 | 76.7 | 80.9 | 64.8 |

Table 12: Evaluation on CVBench and MathVista. “6.5M” instruction tuning dataset is from Cambrian-1. “11.9” instruction tuning dataset is concatenation of Cambrian-1 and LLaVA-OneVision datasets. 

D [Datasets](https://arxiv.org/html/2412.13303)
-----------------------------------------------

### D.1 Pretraining Datasets

For Stage-1 training, we only use LLaVA-1.5 558K[[53](https://arxiv.org/html/2412.13303v2#bib.bib53)] dataset. For Stage-1.5 training, we use densely captioned versions of CC3M[[71](https://arxiv.org/html/2412.13303v2#bib.bib71)] and CC12M[[10](https://arxiv.org/html/2412.13303v2#bib.bib10)] introduced in[[43](https://arxiv.org/html/2412.13303v2#bib.bib43)]. The total size of this dataset is 15 million image-text pairs. We generated 300 generic questions, such as “What is in this photo?”. For each (image, dense-caption) pair, we randomly selected a generic question to form a triplet of (question, image, dense-caption). With a 0.5 probability, we placed the image’s special token <image> either before or after the question. From recent works like[[43](https://arxiv.org/html/2412.13303v2#bib.bib43), [78](https://arxiv.org/html/2412.13303v2#bib.bib78), [66](https://arxiv.org/html/2412.13303v2#bib.bib66)] and our results in[Tab.6](https://arxiv.org/html/2412.13303v2#S3.T6 "In 3.2.3 Comparison with Token Pruning & Downsampling ‣ 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), scaling dataset in Scale-1.5 is beneficial to improve the performance of VLM across a wide range of evaluations. Even though FastViTHD is smaller than ViT-L/14 and ViT-H used in [[43](https://arxiv.org/html/2412.13303v2#bib.bib43), [66](https://arxiv.org/html/2412.13303v2#bib.bib66)] respectively, we see similar scaling trends.

### D.2 Visual Instruction Tuning Datasets

We use 3 different version of instruction tuning datasets. The smallest scale is LLaVA-1.5 665K dataset[[53](https://arxiv.org/html/2412.13303v2#bib.bib53)]. We further scale up this dataset by including training splits of the following datasets; AI2D[[37](https://arxiv.org/html/2412.13303v2#bib.bib37)], ScienceQA[[59](https://arxiv.org/html/2412.13303v2#bib.bib59)], ChartQA[[63](https://arxiv.org/html/2412.13303v2#bib.bib63)], COCO[[51](https://arxiv.org/html/2412.13303v2#bib.bib51)], DocVQA[[65](https://arxiv.org/html/2412.13303v2#bib.bib65)], DVQA[[35](https://arxiv.org/html/2412.13303v2#bib.bib35)], GeoQA+[[8](https://arxiv.org/html/2412.13303v2#bib.bib8)], OCRVQA[[67](https://arxiv.org/html/2412.13303v2#bib.bib67)], SegmentAnything[[39](https://arxiv.org/html/2412.13303v2#bib.bib39)], SynthDoG-EN[[38](https://arxiv.org/html/2412.13303v2#bib.bib38)], TextVQA[[74](https://arxiv.org/html/2412.13303v2#bib.bib74)] and Visual Genome[[40](https://arxiv.org/html/2412.13303v2#bib.bib40)]. The conversational data for the listed datasets is sourced from[[15](https://arxiv.org/html/2412.13303v2#bib.bib15)]. The total number of samples in this dataset is 1.1 million and is referred to as “1.1M” in all the tables. We further scale-up instruction tuning dataset using image-based conversational data from Cambrian-7M[[78](https://arxiv.org/html/2412.13303v2#bib.bib78)], which amounts to 5.4 million samples. Filtered Cambrian-7M[[78](https://arxiv.org/html/2412.13303v2#bib.bib78)] is merged with “1.1M ” dataset to obtain “6.5M” instruction tuning dataset. We then append all available single-image instruction tuning data open-sourced by LLaVA-OneVision[[45](https://arxiv.org/html/2412.13303v2#bib.bib45)] to “6.5M” to obtain “11.9M” instruction tuning dataset. We then include roughly 0.6M samples from DocMatix[[41](https://arxiv.org/html/2412.13303v2#bib.bib41)] dataset to obtain “12.5M” instruction tuning dataset. From [Tab.6](https://arxiv.org/html/2412.13303v2#S3.T6 "In 3.2.3 Comparison with Token Pruning & Downsampling ‣ 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), we see further improvements in VLM benchmarks when instruction tuning dataset is scaled, following trends exhibited by image encoders much bigger than FastViTHD. The best performing models are achieved when Stage 2 models are further fine-tuned on the MammothVL instruction tuning dataset[[30](https://arxiv.org/html/2412.13303v2#bib.bib30)]. Specifically, we filter the corpus to include only single-image instruction tuning data, resulting in “10.6M” samples.

### D.3 [Evaluations](https://arxiv.org/html/2412.13303)

In addition to evaluations listed in [Sec.4](https://arxiv.org/html/2412.13303v2#S4 "4 Experiments ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), we report performance of FastVLM on ChartQA[[63](https://arxiv.org/html/2412.13303v2#bib.bib63)], OCRBench[[56](https://arxiv.org/html/2412.13303v2#bib.bib56)] and InfoVQA[[64](https://arxiv.org/html/2412.13303v2#bib.bib64)] to compare FastVLM against recent methods on text-rich benchmarks. In [Tab.13](https://arxiv.org/html/2412.13303v2#S4.T13 "In D.3 Evaluations ‣ D Datasets ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), report performance of FastViT model (with architectural interventions) from multiple training runs and compute the standard deviation of metrics reported in [Tab.6](https://arxiv.org/html/2412.13303v2#S3.T6 "In 3.2.3 Comparison with Token Pruning & Downsampling ‣ 3.2 FastViTHD: High Resolution Encoder for VLM ‣ 3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"). As described in [Sec.4](https://arxiv.org/html/2412.13303v2#S4 "4 Experiments ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), for ablations we are interested in benchmarks that are quick to evaluate and exhibit lower variance to different initializations. From [Tab.13](https://arxiv.org/html/2412.13303v2#S4.T13 "In D.3 Evaluations ‣ D Datasets ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), GQA, TextVQA, POPE, DocVQA and SeedBench fit the criteria. While VQAv2 also exhibits lower variance it is substantially larger and takes long time to evaluate. The standard deviation across the selected metrics is below 0.5, so we use the average of these metrics as a reliable indicator for our analysis in [Sec.3](https://arxiv.org/html/2412.13303v2#S3 "3 Architecture ‣ FastVLM: Efficient Vision Encoding for Vision Language Models").

|  | GQA | SQA | TextVQA | POPE | LLaVA | MMVet | VQAv2 | DocVQA | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Bench W | Bench I |
|  | 62.69 | 64.25 | 60.71 | 85.8 | 59.4 | 29.6 | 77.27 | 27.57 | 53.31 |
|  | 62.68 | 64.95 | 60.61 | 86.1 | 60.1 | 31.6 | 77.39 | 28.37 | 53.55 |
|  | 62.69 | 65.64 | 60.68 | 85.3 | 61.4 | 31.1 | 77.31 | 28.26 | 53.46 |
| Std. | 0.0047 | 0.57 | 0.041 | 0.33 | 0.83 | 0.85 | 0.049 | 0.35 | 0.099 |

Table 13:  VLM benchmarks across three independent runs with frozen FastViT image encoder. Training setup is LLaVA-1.5 with Vicuna 7B as LLM. Standard deviation across runs is listed in the bottom row. 

E [Qualitative Analysis](https://arxiv.org/html/2412.13303)
-----------------------------------------------------------

We analyzed failures across benchmarks and found: In text-rich benchmarks (e.g., DocVQA, ChartQA), failures occur when text is too small or precise alignment is needed (e.g., reading tables). In majority of cases where the text is too small, simply using higher input resolution can reduce errors as shown in [Tab.15](https://arxiv.org/html/2412.13303v2#S5.T15 "In E Qualitative Analysis ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"). Some cases require broader general knowledge to obtain a correct response as seen in [Tab.14](https://arxiv.org/html/2412.13303v2#S5.T14 "In E Qualitative Analysis ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"), in such cases using a bigger LLM reduces failures. Some cases require reasoning about a higher resolution image, in which case using a bigger LLM can decrease failures as seen in [Tab.16](https://arxiv.org/html/2412.13303v2#S5.T16 "In E Qualitative Analysis ‣ FastVLM: Efficient Vision Encoding for Vision Language Models"). Some failures result from misjudgment, where correct responses are misclassified by the LLM judge or incorrect labels, we ignore these cases in our analysis.

When to Use a Larger LLM
![Image 8: [Uncaptioned image]](https://arxiv.org/html/extracted/6443519/figures/fa_examples/mmmu_llm_diff_example_1.png)
Dataset: MMMU[[93](https://arxiv.org/html/2412.13303v2#bib.bib93)]
User What is the common term for the yellow area surrounding the site of an infection? Options: [”I don’t know and I don’t want to guess”, ’Corona’, ’Border’, ’Halo’, ’Toxin zone’]
Ground Truth D
FastVLM-0.5B @ 256 E ✗
FastVLM-0.5B @ 1024 E ✗
FastVLM-1.5B @ 256 D ✓
FastVLM-1.5B @ 1024 D ✓
![Image 9: [Uncaptioned image]](https://arxiv.org/html/extracted/6443519/figures/fa_examples/mmmu_llm_diff_example_2.jpeg)
Dataset: MMMU[[93](https://arxiv.org/html/2412.13303v2#bib.bib93)]
User The sinoatrial (SA) node is indicated by ___. Options: [’A’, ’B’, ’C’, ’D’, ’E’]
Ground Truth A
FastVLM-0.5B @ 256 E ✗
FastVLM-0.5B @ 1024 D ✗
FastVLM-1.5B @ 256 A ✓
FastVLM-1.5B @ 1024 A ✓

Table 14: Cases that require broader general knowledge, in which case a larger LLM is preferred.

When to Use Higher Resolution
![Image 10: [Uncaptioned image]](https://arxiv.org/html/extracted/6443519/figures/fa_examples/GQA_res_diff_example_2.png)
Dataset: GQA[[34](https://arxiv.org/html/2412.13303v2#bib.bib34)]
User What is sitting inside the bowls?
Ground Truth squash
FastVLM-0.5B @ 256 Sculpture ✗
FastVLM-0.5B @ 1024 Squash ✓
FastVLM-1.5B @ 256 Potato ✗
FastVLM-1.5B @ 1024 Squash ✓
![Image 11: [Uncaptioned image]](https://arxiv.org/html/extracted/6443519/figures/fa_examples/chartqa_res_diff_example_2.png)
Dataset: ChartQA[[63](https://arxiv.org/html/2412.13303v2#bib.bib63)]
User What was the highest expenditure on foreign military aid in 2009/10?
Ground Truth 3781
FastVLM-0.5B @ 256 Germany ✗
FastVLM-0.5B @ 1024 3781 ✓
FastVLM-1.5B @ 256 275 ✗
FastVLM-1.5B @ 1024 3781 ✓

Table 15: Cases where increasing the resolution is sufficient to obtain a better response.

When to Use Higher Resolution and a Larger LLM
![Image 12: [Uncaptioned image]](https://arxiv.org/html/extracted/6443519/figures/fa_examples/chartqa_llm_res_diff_example_1.png)
Dataset: ChartQA[[63](https://arxiv.org/html/2412.13303v2#bib.bib63)]
User What was the value of the commercial property market in 2016?
Ground Truth 883
FastVLM-0.5B @ 256 10000 ✗
FastVLM-0.5B @ 1024 871 ✗
FastVLM-1.5B @ 256 1100 ✗
FastVLM-1.5B @ 1024 883 ✓
![Image 13: [Uncaptioned image]](https://arxiv.org/html/extracted/6443519/figures/fa_examples/docvqa_llm_res_diff_example_1.png)
Dataset: DocVQA[[65](https://arxiv.org/html/2412.13303v2#bib.bib65)]
User Under which department ‘Stockroom’ is organized?
Ground Truth Research Service Department
FastVLM-0.5B @ 256 Department of Chemistry ✗
FastVLM-0.5B @ 1024 Library ✗
FastVLM-1.5B @ 256 Research Department ✗
FastVLM-1.5B @ 1024 Research Service Department ✓

Table 16: Cases where increasing the resolution alone is not sufficient to obtain a better response. Larger LLM in combination with higher resolution is required to obtain a correct response.

Generated on Thu May 15 21:52:58 2025 by [L a T e XML![Image 14: Mascot Sammy](blob:http://localhost/70e087b9e50c3aa663763c3075b0d6c5)](http://dlmf.nist.gov/LaTeXML/)

