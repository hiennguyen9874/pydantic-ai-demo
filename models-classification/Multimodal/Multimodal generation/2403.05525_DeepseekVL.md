Title: DeepSeek-VL: Towards Real-World Vision-Language Understanding

URL Source: https://arxiv.org/html/2403.05525

Published Time: Tue, 12 Mar 2024 01:44:04 GMT

Markdown Content:
\reportnumber

001 \correspondingauthor*{}^{*}start_FLOATSUPERSCRIPT * end_FLOATSUPERSCRIPT Equal contribution. 

††{}^{\dagger}start_FLOATSUPERSCRIPT † end_FLOATSUPERSCRIPT Work done during the internship at DeepSeek-AI. 

‡‡{}^{\ddagger}start_FLOATSUPERSCRIPT ‡ end_FLOATSUPERSCRIPT Project lead.

Wen Liu*1 1{}^{1}start_FLOATSUPERSCRIPT 1 end_FLOATSUPERSCRIPT Bo Zhang*1⁣‡1‡{}^{1{\ddagger}}start_FLOATSUPERSCRIPT 1 ‡ end_FLOATSUPERSCRIPT Bingxuan Wang 1⁣†1†{}^{1{\dagger}}start_FLOATSUPERSCRIPT 1 † end_FLOATSUPERSCRIPT Kai Dong 1 1{}^{1}start_FLOATSUPERSCRIPT 1 end_FLOATSUPERSCRIPT Bo Liu 1⁣†1†{}^{1{\dagger}}start_FLOATSUPERSCRIPT 1 † end_FLOATSUPERSCRIPT Jingxiang Sun 1⁣†1†{}^{1{\dagger}}start_FLOATSUPERSCRIPT 1 † end_FLOATSUPERSCRIPT Tongzheng Ren 1⁣†1†{}^{1{\dagger}}start_FLOATSUPERSCRIPT 1 † end_FLOATSUPERSCRIPT Zhuoshu Li 1 1{}^{1}start_FLOATSUPERSCRIPT 1 end_FLOATSUPERSCRIPT Hao Yang 1⁣†1†{}^{1{\dagger}}start_FLOATSUPERSCRIPT 1 † end_FLOATSUPERSCRIPT Yaofeng Sun 1 1{}^{1}start_FLOATSUPERSCRIPT 1 end_FLOATSUPERSCRIPT Chengqi Deng 1 1{}^{1}start_FLOATSUPERSCRIPT 1 end_FLOATSUPERSCRIPT Hanwei Xu 1 1{}^{1}start_FLOATSUPERSCRIPT 1 end_FLOATSUPERSCRIPT Zhenda Xie 1 1{}^{1}start_FLOATSUPERSCRIPT 1 end_FLOATSUPERSCRIPT Chong Ruan 1 1{}^{1}start_FLOATSUPERSCRIPT 1 end_FLOATSUPERSCRIPT 1 1{}^{1}start_FLOATSUPERSCRIPT 1 end_FLOATSUPERSCRIPT DeepSeek-AI 

{neal, liuwen, bo}@deepseek.com

[https://github.com/deepseek-ai/DeepSeek-VL](https://github.com/deepseek-ai/DeepSeek-VL)

###### Abstract

We present DeepSeek-VL, an open-source Vision-Language (VL) Model designed for real-world vision and language understanding applications. Our approach is structured around three key dimensions:

•Data Construction: We strive to ensure our data is diverse, scalable and extensively covers real-world scenarios including web screenshots, PDFs, OCR, charts, and knowledge-based content (expert knowledge, textbooks), aiming for a comprehensive representation of practical contexts. Further, we create a use case taxonomy from real user scenarios and construct an instruction-tuning dataset accordingly. The fine-tuning with this dataset substantially improves the model’s user experience in practical applications.

•Model Architecture: Considering efficiency and the demands of most real-world scenarios, DeepSeek-VL incorporates a hybrid vision encoder that efficiently processes high-resolution images (1024 x 1024) within a fixed token budget, while maintaining a relatively low computational overhead. This design choice ensures the model’s ability to capture critical semantic and detailed information across various visual tasks.

•Training Strategy: We posit that a proficient Vision-Language Model should, foremost, possess strong language abilities. To ensure the preservation of LLM capabilities during pretraining, we investigate an effective VL pretraining strategy by integrating LLM training from the beginning and carefully managing the competitive dynamics observed between vision and language modalities. Starting with a focus on text, we gradually adjust the ratio to facilitate a balanced integration of both modalities.

The DeepSeek-VL family (both 1.3B and 7B models) showcases superior user experiences as a vision-language chatbot in real-world applications, achieving state-of-the-art or competitive performance across a wide range of visual-language benchmarks at the same model size while maintaining robust performance on language-centric benchmarks. We have made both 1.3B and 7B models publicly accessible to foster innovations based on this foundation model.

###### Contents

1.   [1 Introduction](https://arxiv.org/html/2403.05525v2#S1 "1 Introduction ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding")
2.   [2 Data Construction](https://arxiv.org/html/2403.05525v2#S2 "2 Data Construction ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding")
    1.   [2.1 Vision-Language pretraining Data](https://arxiv.org/html/2403.05525v2#S2.SS1 "2.1 Vision-Language pretraining Data ‣ 2 Data Construction ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding")
    2.   [2.2 Supervised Fine-tuning Data](https://arxiv.org/html/2403.05525v2#S2.SS2 "2.2 Supervised Fine-tuning Data ‣ 2 Data Construction ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding")

3.   [3 Approach](https://arxiv.org/html/2403.05525v2#S3 "3 Approach ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding")
    1.   [3.1 Architecture](https://arxiv.org/html/2403.05525v2#S3.SS1 "3.1 Architecture ‣ 3 Approach ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding")
    2.   [3.2 Training Pipelines](https://arxiv.org/html/2403.05525v2#S3.SS2 "3.2 Training Pipelines ‣ 3 Approach ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding")
        1.   [3.2.1 Stage 1: Training Vision-Language Adaptor](https://arxiv.org/html/2403.05525v2#S3.SS2.SSS1 "3.2.1 Stage 1: Training Vision-Language Adaptor ‣ 3.2 Training Pipelines ‣ 3 Approach ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding")
        2.   [3.2.2 Stage 2: Joint Vision-Language pretraining](https://arxiv.org/html/2403.05525v2#S3.SS2.SSS2 "3.2.2 Stage 2: Joint Vision-Language pretraining ‣ 3.2 Training Pipelines ‣ 3 Approach ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding")
        3.   [3.2.3 Stage 3: Supervised Fine-tuning](https://arxiv.org/html/2403.05525v2#S3.SS2.SSS3 "3.2.3 Stage 3: Supervised Fine-tuning ‣ 3.2 Training Pipelines ‣ 3 Approach ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding")

    3.   [3.3 Hyperparameters and Infrastructures](https://arxiv.org/html/2403.05525v2#S3.SS3 "3.3 Hyperparameters and Infrastructures ‣ 3 Approach ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding")

4.   [4 Evaluation](https://arxiv.org/html/2403.05525v2#S4 "4 Evaluation ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding")
    1.   [4.1 Public Multimodal Benchmarks Evaluation](https://arxiv.org/html/2403.05525v2#S4.SS1 "4.1 Public Multimodal Benchmarks Evaluation ‣ 4 Evaluation ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding")
    2.   [4.2 Public Language Benchmarks Evaluation](https://arxiv.org/html/2403.05525v2#S4.SS2 "4.2 Public Language Benchmarks Evaluation ‣ 4 Evaluation ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding")
    3.   [4.3 Human Evaluation](https://arxiv.org/html/2403.05525v2#S4.SS3 "4.3 Human Evaluation ‣ 4 Evaluation ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding")
    4.   [4.4 Ablation Study](https://arxiv.org/html/2403.05525v2#S4.SS4 "4.4 Ablation Study ‣ 4 Evaluation ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding")

5.   [5 Conclusion, Limitation, and Future Work](https://arxiv.org/html/2403.05525v2#S5 "5 Conclusion, Limitation, and Future Work ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding")
6.   [A Appendix](https://arxiv.org/html/2403.05525v2#A1 "Appendix A Appendix ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding")

1 Introduction
--------------

The remarkable success of large language models (LLMs)(OpenAI, [2022](https://arxiv.org/html/2403.05525v2#bib.bib57), [2023a](https://arxiv.org/html/2403.05525v2#bib.bib58); Anthropic, [2023](https://arxiv.org/html/2403.05525v2#bib.bib4); Google, [2023](https://arxiv.org/html/2403.05525v2#bib.bib22)) has fueled the demand for a versatile interface that can handle multiple modalities beyond language. In response to this growing demand, we have seen an emergence of Large Multimodal Models (LMMs) like GPT-4V(OpenAI, [2023b](https://arxiv.org/html/2403.05525v2#bib.bib59)) and Gemini(Team et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib69)), which serve as versatile assistants capable of comprehending and acting upon instructions that span vision and language. These models exhibit considerable promise in executing complex, diverse real-world tasks, enabling more natural and human-like interactions.

![Image 1: Refer to caption](https://arxiv.org/html/2403.05525v2/x1.png)

Figure 1: DeepSeek-VL possesses general multimodal understanding capabilities, capable of processing logical diagrams, web pages, formula recognition, scientific literature, natural images, and embodied intelligence in complex scenarios.

Recently, there has been a surge of open-source large multimodal models aimed at narrowing the gap with proprietary counterparts. Substantial strides have been made, especially in benchmark performance, yet a significant divide persists between the majority of open-source models and state-of-the-art closed-source models(OpenAI, [2023b](https://arxiv.org/html/2403.05525v2#bib.bib59); Bavishi et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib7); Team et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib69); Bai et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib6)) when it comes to real-world performance and user experience. It remains challenging for the open-source community to develop models with robust general multimodal capabilities for real-world applications.

The performance gap between the most open-source models and the proprietary models is largely pronounced in real-world scenarios, primarily due to the following reasons:

*   •Many open-source solutions allocate a significant proportion of computational resources to the instruction tuning phase. However, the experience of training powerful language models underscores the importance of extensive pretraining in the development of general intelligence. To imbue multimodal models with rich world knowledge, there should be an emphasis on comprehensive pretraining that leverages a broad spectrum of vision-language data. 
*   •A common practice is to amalgamate various academic datasets during instruction tuning. While such an approach may yield good benchmark results, it often falls short in providing an authentic real-world usage experience. 
*   •In terms of model architecture, prior works mostly adapt a vision transformer, typically text-aligned, to a pre-trained language model. However, most of these models operate on a relatively low resolution, e.g., 336×\times×336 or 448×\times× 448. The intricacies of complex real-world scenarios, such as optical character recognition or tiny object discernment, demand high-resolution processing capability. 
*   •While some models(Sun et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib67); Wang et al., [2023b](https://arxiv.org/html/2403.05525v2#bib.bib76); 01-ai, [2024](https://arxiv.org/html/2403.05525v2#bib.bib1); Lin et al., [2023a](https://arxiv.org/html/2403.05525v2#bib.bib40)) have begun to exploit pretraining, they often overlook the preservation of language skills. Often, there is a degradation of language capability after prolonged multimodal training. Since we aim for a generalist that possesses strong capabilities in both modalities, there should be a training strategy that well preserves the language capability when developing the new modality ability. 

In light of these, we present DeepSeek-VL, an open-source large multimodal model, which is built upon the DeepSeek language model series. We develop the model in the pursuit of adept performance in real-world scenarios, which involves extensive pretraining, careful data curation based on a use case taxonomy, model architecture design for high-resolution processing, and a training strategy that balances the multi-modalities. On top of these, we develop a training methodology that steers the model scaling, from 1B to 7B. These comprehensive explorations bring a significant performance advantage in practical settings, compared to other large multimodal models (LMMs) of similar size.

DeepSeek-VL’s pretraining dataset is compiled from a variety of sources, including but not limited to Common Crawl, Web Code, E-books, Educational Materials, and arXiv Articles. This collection thoroughly encompasses real-world scenarios such as web screenshots, PDFs, OCR, charts, and knowledge-based content (expertise, textbooks), aiming for a broad and practical representation while remaining scalable.

While our pretraining data encompasses a wide array of world knowledge, we meticulously curate our instruction-tuning dataset to reflect real-world usage scenarios. To achieve this, we manually gather authentic test cases for GPT-4V and Gemini from the Internet. These cases have been systematically organized into a comprehensive taxonomy. We use this structured taxonomy to choose prompts for each test image, ensuring a practical and relevant instruction tuning dataset. This taxonomy is also used to create an evaluation dataset that effectively assesses real-world performance.

The visual module is designed to optimize the utilization of high-resolution visual inputs while remaining within a fixed token budget to manage inference costs effectively. As such, we employ a hybrid vision encoder, which combines a text-aligned encoder for coarse semantic extraction at 384×384 384 384 384\times 384 384 × 384 resolution with a high-resolution encoder that captures detailed visual information at 1024×1024 1024 1024 1024\times 1024 1024 × 1024 resolution. By fusing these two encoders, our hybrid approach efficiently condenses a 1024×1024 resolution image (which suffices in most use cases) into 576 tokens. This token count strikes a balance between rich visual representation and token economy, making it feasible for both text-image interleaving and multi-turn inference scenarios.

During the pretraining of multimodal models, a common challenge encountered is the potential degradation of language capabilities when the training process is overly reliant on vision-language data. Our research reveals that maintaining a significant proportion of language data—specifically, at least 70%—is essential to preserve the integrity of language knowledge within the model. This balance is critical for achieving a robust multimodal capability that does not compromise language performance. Moreover, we introduce a novel “modality warm-up” strategy. This approach carefully adjusts the ratio of modalities during training, gradually incorporating more vision-language data. The careful tuning of the modality ratio along with the warm-up strategy results in a balanced performance of both modalities.

When iterating on our model, We conduct experiments on a small scale before scaling to a larger model size. However, a smaller model, e.g., 1B model, cannot demonstrate reasonable performance on benchmarks(Schaeffer et al., [2024](https://arxiv.org/html/2403.05525v2#bib.bib61)) and faithfully reflect the model’s performance. We adopt two approaches to address this. First, we modify the evaluation protocol from multi-choice to compare the perplexity of options. Also, to prevent the instruction following ability from becoming the bottleneck, we mix a small proportion of instruction tuning data during the pretraining phase. In this way, we can achieve reasonable performance using the 1B model and more accurately measure the impact of each iteration during the experiment.

Through extensive evaluations of general vision and language benchmarks, the DeepSeek-VL family showcases superior user experiences in real-world applications and achieves state-of-the-art or competitive performance across a wide range of visual-language benchmarks at the same model size, while maintaining robust language-centric performance. To foster innovation and enable a wide range of applications, we have made two versions of our ours, 1.3B and 7B, publicly accessible, in the hope of facilitating the needs of varying computational capabilities.

2 Data Construction
-------------------

A diverse and large dataset is the most important ingredient of visual language model training. Our dataset can be divided into two parts: Vision-Language pretraining Data and Vision-Language Supervised Fine-Tuning Data. VL pretraining Data is composed of visual-text data from various sources, aimed at enhancing the model’s fundamental cross-modal understanding capabilities; while VL Supervised Fine-Tuning Data has a relatively smaller size and aims to teach the model to complete specific downstream tasks. By design, VL pretraining Data is used to warm up the vision-language adaptor in training stage 1 and jointly pretrain the vision-language model in stage 2, and VL Supervised Fine-Tuning Data is exploited in training stage 3, i.e., vision language supervised fine-tuning.

### 2.1 Vision-Language pretraining Data

Table 1: Summary of datasets used in the joint vision and language pretraining stage.

Category Dataset Ratio
Interleaved image-text MMC4(Zhu et al., [2024](https://arxiv.org/html/2403.05525v2#bib.bib90))13.1%
Wikipedia EN& CN([Foundation,](https://arxiv.org/html/2403.05525v2#bib.bib19))
Wikihow(Yang et al., [2021](https://arxiv.org/html/2403.05525v2#bib.bib78))
in-house PDF and Epub textbooks
Image caption Capsfusion(Yu et al., [2023a](https://arxiv.org/html/2403.05525v2#bib.bib80))11.1%
TaiSu(Liu et al., [2022b](https://arxiv.org/html/2403.05525v2#bib.bib45))
Detailed Caption(echo840, [2024](https://arxiv.org/html/2403.05525v2#bib.bib18))
Table and chart Chart2text(Kantharaj et al., [2022](https://arxiv.org/html/2403.05525v2#bib.bib28))2.1%
Geo170K(Gao et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib20))
Ureader(Ye et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib79))
Unichart(Masry et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib54))
M-paper(Hu et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib26))
ScienceQA(Lu et al., [2022b](https://arxiv.org/html/2403.05525v2#bib.bib51))
ScreenQA(Hsiao et al., [2022](https://arxiv.org/html/2403.05525v2#bib.bib25))
SciGraphQA-295K(Li and Tajbakhsh, [2023](https://arxiv.org/html/2403.05525v2#bib.bib36))
Paper2figure100k(Rodriguez et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib60))
Widget Captioning(Li et al., [2020](https://arxiv.org/html/2403.05525v2#bib.bib37))
Screen2words(Wang et al., [2021](https://arxiv.org/html/2403.05525v2#bib.bib74))
Refexp(Mao et al., [2016](https://arxiv.org/html/2403.05525v2#bib.bib53))
Web Code Websight(HuggingFaceM4, [2024](https://arxiv.org/html/2403.05525v2#bib.bib27))0.4%
python plots scraped from GitHub notebook
Scene text OCR ArT(Chng et al., [2019](https://arxiv.org/html/2403.05525v2#bib.bib13))1.2%
MLT-17(Nayef et al., [2017](https://arxiv.org/html/2403.05525v2#bib.bib56))
LSVT(Sun et al., [2019](https://arxiv.org/html/2403.05525v2#bib.bib68))
UberText(Zhang et al., [2017](https://arxiv.org/html/2403.05525v2#bib.bib87))
Coco-text(Veit et al., [2016](https://arxiv.org/html/2403.05525v2#bib.bib73))
RCTW-17(Shi et al., [2017](https://arxiv.org/html/2403.05525v2#bib.bib63))
ReCTS(Zhang et al., [2019](https://arxiv.org/html/2403.05525v2#bib.bib86))
TextOCR(Singh et al., [2021](https://arxiv.org/html/2403.05525v2#bib.bib65))
OpenVINO(Krylov et al., [2021](https://arxiv.org/html/2403.05525v2#bib.bib32))
HierText(Long et al., [2022](https://arxiv.org/html/2403.05525v2#bib.bib48))
Document OCR arXiv rendered markdown(Blecher et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib9))2.1%
Text-only corpus DeepSeek-LLM 2T text copus(DeepSeek-AI, [2024](https://arxiv.org/html/2403.05525v2#bib.bib16))70.0%

The pretraining dataset utilized in our study encompasses a diverse range of publicly accessible sources, in addition to a selection of proprietary data. We provide a comprehensive overview of the data sources employed during the joint vision and language pretraining stage in Table[1](https://arxiv.org/html/2403.05525v2#S2.T1 "Table 1 ‣ 2.1 Vision-Language pretraining Data ‣ 2 Data Construction ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding"). Such a dataset can facilitate LLM’s comprehension of the entities portrayed in the images.

Furthermore, we present a detailed breakdown of the complete dataset, which is organized into the following categories:

Interleaved image-text data enable the models to have a better capability for in-context learning of multi-modality inputs, and we utilize three public datasets MMC4(Zhu et al., [2024](https://arxiv.org/html/2403.05525v2#bib.bib90)), Wiki(Burns et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib10)), Wikihow(Yang et al., [2021](https://arxiv.org/html/2403.05525v2#bib.bib78)) and Epub textbooks.

Image caption data come from three high-quality image-text paired datasets: Capsfusion(Yu et al., [2023a](https://arxiv.org/html/2403.05525v2#bib.bib80)), TaiSu(Liu et al., [2022b](https://arxiv.org/html/2403.05525v2#bib.bib45)) and Detailed Caption(echo840, [2024](https://arxiv.org/html/2403.05525v2#bib.bib18)).

Table and chart data enable the models to learn the capability for general table and chart image understanding. It encompasses a diverse range of public data sources, including Chart2text(Kantharaj et al., [2022](https://arxiv.org/html/2403.05525v2#bib.bib28)), Geo170K(Gao et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib20)), Unichart(Masry et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib54)), Ureader(Ye et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib79)), M-paper(Hu et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib26)), ScienceQA(Lu et al., [2022b](https://arxiv.org/html/2403.05525v2#bib.bib51)), ScreenQA(Hsiao et al., [2022](https://arxiv.org/html/2403.05525v2#bib.bib25)), SciGraphQA-295K(Li and Tajbakhsh, [2023](https://arxiv.org/html/2403.05525v2#bib.bib36)), Paper2figure100k(Rodriguez et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib60)), Widget Captioning(Li et al., [2020](https://arxiv.org/html/2403.05525v2#bib.bib37)), Screen2words(Wang et al., [2021](https://arxiv.org/html/2403.05525v2#bib.bib74)), and Refexp(Mao et al., [2016](https://arxiv.org/html/2403.05525v2#bib.bib53)).

Web Code data empowers models with the capability to reconstruct code from graphical interfaces or visual plots. Leveraging Websight(HuggingFaceM4, [2024](https://arxiv.org/html/2403.05525v2#bib.bib27)) for UI Inverse Rendering, we adopted a strategy akin to that used in MATCHA(Liu et al., [2022a](https://arxiv.org/html/2403.05525v2#bib.bib42)) for visual plots inverse rendering. This involved the processing of approximately 1.46 million Jupyter notebooks from the Stack dataset(Kocetkov et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib30)). By extracting these notebooks and collating all diagrams along with their corresponding preceding code segments, we succeeded in curating a collection featuring 2 million pairs of images and codes. For better data quality, we filter 1.1 million instances, each comprising a singular image coupled with a minimum of 5 lines of code, to constitute our primary training dataset.

Document Optical Character Recognition (OCR) data facilitates the recognition of optical characters at the document level, even in challenging real-world scenarios. To the best of our knowledge, there is currently no publicly available large-scale dataset encompassing both English and Chinese documents. Despite the existence of the publicly accessible small-scale dataset Latex-OCR(Blecher, [2024](https://arxiv.org/html/2403.05525v2#bib.bib8)), we additionally constructed a comprehensive English and Chinese document OCR dataset. It is comprised of two parts: 1): arXiv Articles: We collected source code and compiled PDFs from 1.4 million arXiv articles. Utilizing pre-processing tools from Nougat(Blecher et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib9)), we rendered these articles into paired images and texts; 2): E-books and Educational Materials: We cleaned 860K English and 180K Chinese e-books from Anna’s Archive(Anna’s Archive, [2024](https://arxiv.org/html/2403.05525v2#bib.bib3)) alongside millions of K-12 education exam questions. Subsequently, we employed HTML rendering tools([Kulkarni and Truelsen,](https://arxiv.org/html/2403.05525v2#bib.bib33)) to convert these HTML files with different templates into paired image and text formats.

Scene text OCR data augment the capability of the model to recognize and extract text from images in which the text is integrated into the environment. The dataset is composed of multiple public datasets, including ArT(Chng et al., [2019](https://arxiv.org/html/2403.05525v2#bib.bib13)), MLT-17(Nayef et al., [2017](https://arxiv.org/html/2403.05525v2#bib.bib56)), LSVT(Sun et al., [2019](https://arxiv.org/html/2403.05525v2#bib.bib68)), UberText(Zhang et al., [2017](https://arxiv.org/html/2403.05525v2#bib.bib87)), Coco-text(Veit et al., [2016](https://arxiv.org/html/2403.05525v2#bib.bib73)), RCTW-17(Shi et al., [2017](https://arxiv.org/html/2403.05525v2#bib.bib63)), ReCTS(Zhang et al., [2019](https://arxiv.org/html/2403.05525v2#bib.bib86)), TextOCR(Singh et al., [2021](https://arxiv.org/html/2403.05525v2#bib.bib65)), OpenVINO(Krylov et al., [2021](https://arxiv.org/html/2403.05525v2#bib.bib32)) and HierText(Long et al., [2022](https://arxiv.org/html/2403.05525v2#bib.bib48)).

Text-only corpus serves to maintain proficiency in language-centric tasks. In this study, we employ the same text corpus with DeepSeek-LLM(DeepSeek-AI, [2024](https://arxiv.org/html/2403.05525v2#bib.bib16)).

Table 2: Summary of data used in our joint vision and language supervised fine-tuning stage.

Class Dataset Ratio
In-house Data SFT data based on taxonomy (Figure[3](https://arxiv.org/html/2403.05525v2#S2.T3 "Table 3 ‣ 2.1 Vision-Language pretraining Data ‣ 2 Data Construction ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding"))10.5%
General Multi-modality ShareGPT4V(Chen et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib12))35.5%
LAION-GPTV(LAION, [2023](https://arxiv.org/html/2403.05525v2#bib.bib34))
LVIS-Instruct4V(Wang et al., [2023a](https://arxiv.org/html/2403.05525v2#bib.bib75))
textOCR-GPT4V(Carter, [2024](https://arxiv.org/html/2403.05525v2#bib.bib11))
LLaVA1.6-GPT4V(Liu et al., [2024a](https://arxiv.org/html/2403.05525v2#bib.bib43))
IconQA(Lu et al., [2021](https://arxiv.org/html/2403.05525v2#bib.bib49))
Table and chart Ureader(Ye et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib79))4.1%
Geo170K(Gao et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib20))
ScienceQA(Lu et al., [2022b](https://arxiv.org/html/2403.05525v2#bib.bib51))
Web Code Screen-to-code(Abi, [2024](https://arxiv.org/html/2403.05525v2#bib.bib2))2.0%
ScreenQA(Hsiao et al., [2022](https://arxiv.org/html/2403.05525v2#bib.bib25))
Text-only SFT DeepSeek-LLM(DeepSeek-AI, [2024](https://arxiv.org/html/2403.05525v2#bib.bib16))47.9%

Main Category Description Secondary Category Tertiary Category
Recognition This part of the use cases mainly examines the understanding and description ability of large models for image content, which does not require high knowledge reserve and reasoning ability of the model, and some tasks can be completed using traditional machine learning models.Global Description Theme Description, Event/Behavior Description, Location/Scene Description, Emotion/Mood Description, Style Recognition, Food Recognition, Others
Local Description Pointing Description, Position Description, Person Recognition, Object Attribute Description, Logo Recognition, Counting, Currency Recognition
OCR and Transcription Printed Text Transcription, Handwritten Text Transcription, Specified Format Transcription, Specified Language Transcription
Conversion This type of use case requires the model to be able to describe and recognize image content, and use specific knowledge (e.g., code knowledge, prompt engineering knowledge) to convert image content into another form.Image to Code UI to Code, Chart to Code, Photo to SVG/p64 Encoding, Formula to Code, Flowchart to Code
Image to Text Image to Prompt, Text Summary, Image-based Creation, Text Interpretation
Analysis This type of use case requires the model to use specific knowledge and logical ability to make reasonable analysis and understanding based on image content, and describe the image according to instructions.Data Chart Analysis Graph Interpretation, Table Interpretation
Professional Chart Analysis Circuit Diagram, Flowchart, Map, Music Score, Financial Chart, Floor Plan, Others
Professional Image Analysis Sensor Image, Biological and Medical Image, Voiceprint Image, Point Cloud Image
Encyclopedia Knowledge Analysis Art and Culture Knowledge, Natural Environment Knowledge, Food/Clothing/Housing/Transportation Related Knowledge, Entertainment Related Knowledge, Historical Knowledge
Commonsense Reasoning This type of use case mainly tests the model’s understanding and mastery of common sense in life, which requires reasoning based on the interpretation and analysis of image content combined with common sense.Relationship Reasoning Interpersonal Relationship, Spatial Relationship, Size Relationship, Species Relationship
Function Reasoning Hardware Function Reasoning, Software Function Reasoning
Environment Reasoning Environment State Analysis, Environment-based Behavior Reasoning, Embodied Intelligence
Anomaly Reasoning Identifying Anomalies in Images, Defect Detection, Accident Judgment
Humor Reasoning-
Other Commonsense Reasoning State Reasoning, Cause Reasoning, Attribute Comparison, Optical Illusion, Fun Games, Intention Interpretation, Behavior Prediction
Logical Reasoning This type of use case requires the model to combine the understanding of images, comprehensively use domain knowledge and logical reasoning ability to complete corresponding tasks.Mathematical Reasoning Algebra and Operation, Plane Geometry, Solid Geometry
Other Logical Reasoning Physics, Chemistry, Biology, Code, IQ Questions
Evaluation This type of use case requires the model to evaluate the image content according to specific criteria.-Reality Evaluation, Similarity Evaluation, Aesthetic Evaluation, Open-ended Evaluation, Improvement Suggestions
Multi-graph This type of use case examines the model’s ability to analyze and understand multiple images.Temporal Sequence Understanding Event Prediction, Image Sequencing, Behavior Analysis
Multi-graph Comparison Attribute Comparison, Image-Text Matching, Finding Associations, Spotting Differences, Image Discrimination
Safety This type of use case examines the model’s performance in terms of safety.-Suggestive Questioning, Counterfactual Questioning, Prompt Injection

Table 3: Our taxonomy for the in-house SFT data. The categories covered by our high-quality in-house multi-modality SFT data are comprehensively represented in this taxonomy.

### 2.2 Supervised Fine-tuning Data

The supervised fine-tuning datasets utilized in our study encompass a diverse range of multi-modality and language data sources, including well-known open-source shared gpt4v datasets such as ShareGPT4V(Chen et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib12)), LAION-GPTV(LAION, [2023](https://arxiv.org/html/2403.05525v2#bib.bib34)), LVIS-Instruct4V(Wang et al., [2023a](https://arxiv.org/html/2403.05525v2#bib.bib75)), textOCR-GPT4V(Carter, [2024](https://arxiv.org/html/2403.05525v2#bib.bib11)), LLaVA1.6-GPT4V(Liu et al., [2024a](https://arxiv.org/html/2403.05525v2#bib.bib43)) and IconQA(Lu et al., [2021](https://arxiv.org/html/2403.05525v2#bib.bib49)). Additionally, we incorporate partial table and chart data extracted from pretraining datasets such as Ureader(Ye et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib79)), ScreenQA(Hsiao et al., [2022](https://arxiv.org/html/2403.05525v2#bib.bib25)), Geo170K(Gao et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib20)), and ScienceQA(Lu et al., [2022b](https://arxiv.org/html/2403.05525v2#bib.bib51)). Moreover, we integrate the UI Code dataset obtained from Screen-to-code(Abi, [2024](https://arxiv.org/html/2403.05525v2#bib.bib2)) tasks. To enhance the quality of our multi-modality SFT data, we have also curated a portion of high-quality in-house multi-modality SFT data, some of which are in the Chinese language. Our in-house instruction-tuning dataset is meticulously designed to reflect real-world usage scenarios and cover a wide range of tasks. We start by collecting a diverse set of authentic test cases for GPT-4V and Gemini from various online sources. These test cases are then carefully analyzed and organized into a comprehensive taxonomy, which encompasses multiple categories, such as recognition, conversion, analysis, reasoning, evaluation, and safety, as detailed in Table[3](https://arxiv.org/html/2403.05525v2#S2.T3 "Table 3 ‣ 2.1 Vision-Language pretraining Data ‣ 2 Data Construction ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding"). This structured taxonomy serves as a guideline for selecting representative prompts for each test image, ensuring that our instruction-tuning dataset is both practical and relevant to real-world applications. Moreover, this taxonomy is also employed to construct a balanced and comprehensive evaluation dataset, which allows us to effectively assess the model’s performance across different tasks and categories. By following this systematic approach, we ensure that the categories covered by our in-house multi-modality SFT data are well-aligned with the taxonomy and representative of real-world usage scenarios. Furthermore, we include the text-only SFT data employed in DeepSeek-LLM(DeepSeek-AI, [2024](https://arxiv.org/html/2403.05525v2#bib.bib16)) as part of our joint vision and language SFT data.

3 Approach
----------

### 3.1 Architecture

Our system contains three modules: a hybrid vision encoder, a vision adaptor, and a language model. We introduce each part in this section.

Hybrid Vision Encoder. We employ SigLIP as the vision encoder to extract high-level semantic feature representations from visual inputs. However, we observe that a single SigLIP encoder struggles to address all real-world questions comprehensively. Vision encoders in the CLIP family, including SigLIP, are primarily designed for semantic visual representations but are challenged by ambiguous encoding, resulting in visually distinct images being encoded as similar due to what is referred to as "CLIP-blind pairs"Tong et al. ([2024](https://arxiv.org/html/2403.05525v2#bib.bib70)). Meanwhile, the CLIP family of models is limited by its relatively low-resolution inputs (e.g., 224 x 224, 336 x 336, 384 x 384, 512 x 512), which hinders their ability to handle tasks requiring more detailed low-level features like dense OCR and visual grounding task.

To address these limitations, recent researches(Wei et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib77); Tong et al., [2024](https://arxiv.org/html/2403.05525v2#bib.bib70); Lin et al., [2023b](https://arxiv.org/html/2403.05525v2#bib.bib41)) have advocated for the integration of additional vision-only self-supervised encoders, to enhance the visual grounding capabilities of multi-modality models. Building upon previous motivations, we additionally utilize a vision-only encoder based on the SAM-B(Kirillov et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib29)), a pre-trained ViTDet(Li et al., [2022](https://arxiv.org/html/2403.05525v2#bib.bib38)) image encoder to process low-level features, which accepts high-resolution 1024 x 1024 image inputs. In addition to the SAM-B encoder, we retain the SigLIP-L vision encoder with low-resolution 384 x 384 image inputs. Consequently, our hybrid vision encoder combines the SAM-B and SigLIP-L encoders, efficiently encoding high-resolution 1024 x 1024 images while preserving both semantic and detailed information. Specifically, a high-resolution SAM-B vision encoder first resizes the image into 1024 x 1024 and results in a 64 x 64 x 256 feature map.

In the case of a high-resolution feature map of size, 64 x 64 x 256 generated by SAM-B, the VL Adaptor initially interpolates it into a size of 96 x 96 x 256. Subsequently, it employs two convolutional layers with a stride of 2, producing a feature map of 24 x 24 x 1024, and reshapes it to 576 x 1024. Alongside this, the low-resolution feature map of size 576 x 1024 generated by SigLIP-L is concatenated with the high-resolution features, resulting in 576 visual tokens with 2048 dimensions. These visual tokens possess a substantial capacity for enhancing high-level semantic visual recognition and low-level visual grounding tasks. Then they undergo GeLU activation and are directed through an embedding layer to establish a connection with the language model.

![Image 2: Refer to caption](https://arxiv.org/html/2403.05525v2/x2.png)

Figure 2: Visualization results. DeepSeek-VL is capable of capturing tiny object and giving organized explanations.

Vision-Language Adaptor. We employ a two-layer hybrid MLP to bridge the vision encoder and the LLM. Initially, distinct single-layer MLPs are used to process high-resolution features and low-resolution features separately. Subsequently, these features are concatenated along their dimensions and then transformed into the LLM’s input space through another layer of MLP.

Language Model. Our language model is built upon DeepSeek LLM(DeepSeek-AI, [2024](https://arxiv.org/html/2403.05525v2#bib.bib16)) whose micro design largely follows the design of LLaMA(Touvron et al., [2023a](https://arxiv.org/html/2403.05525v2#bib.bib71), [b](https://arxiv.org/html/2403.05525v2#bib.bib72)), adopting a Pre-Norm structure with RMSNorm(Zhang and Sennrich, [2019](https://arxiv.org/html/2403.05525v2#bib.bib84)) function and using SwiGLU(Shazeer, [2020](https://arxiv.org/html/2403.05525v2#bib.bib62)) as the activation function for the Feed-Forward Network (FFN), with an intermediate layer dimension of 8 3⁢d m⁢o⁢d⁢e⁢l 8 3 subscript 𝑑 𝑚 𝑜 𝑑 𝑒 𝑙\frac{8}{3}d_{model}divide start_ARG 8 end_ARG start_ARG 3 end_ARG italic_d start_POSTSUBSCRIPT italic_m italic_o italic_d italic_e italic_l end_POSTSUBSCRIPT. It also incorporates Rotary Embedding(Su et al., [2024](https://arxiv.org/html/2403.05525v2#bib.bib66)) for positional encoding and uses the same tokenizer with DeepSeek-LLM. We introduce a family of DeepSeek-VL models. Given our objective of conducting joint pretraining with multimodal and language, we select an intermediate checkpoint from DeepSeek’s pretrained models to continue pretraining.

Specifically, the DeepSeek-VL-1B model is constructed based on the DeekSeek-LLM-1B model, which underwent training with an approximate corpus of 500 billion text tokens. And the DeekSeek-VL-7B model is developed leveraging the DeepSeek-LLM-7B model trained with an estimated 2 trillion text tokens.

![Image 3: Refer to caption](https://arxiv.org/html/2403.05525v2/x3.png)

Figure 3: Our training pipelines consist of three stages. Stage 1 involves training the Vision-Language (VL) adaptor while keeping the hybrid vision encoder and language model fixed. Stage 2 is the crucial part of the joint vision and language pretraining, where both VL adaptor and language model are trainable. Stage 3 is the supervised fine-tuning phase, during which the low-resolution vision encoder SigLIP-L, VL adaptor, and language model will be trained.

### 3.2 Training Pipelines

We train our DeepSeek-VL in three consecutive stages as shown in Figure[3](https://arxiv.org/html/2403.05525v2#S3.F3 "Figure 3 ‣ 3.1 Architecture ‣ 3 Approach ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding"): vision-language adaptor warmup, joint vision-language pretraining, and supervised fine-tuning. We currently focus on visual understanding capabilities and only calculate the next token prediction loss on the language part.

#### 3.2.1 Stage 1: Training Vision-Language Adaptor

The primary objective of this stage is to establish a conceptual link between visual and linguistic elements within the embedding space, thereby facilitating the comprehensive understanding of depicted entities in the images by the Large Language Model (LLM). Consistent with prior research conducted by LLaVA(Liu et al., [2024b](https://arxiv.org/html/2403.05525v2#bib.bib44)) and Instruct-BLIP(Dai et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib15)), we adopt a similar approach in which both the vision encoder and the LLM remain frozen during this stage, while solely allowing the trainable parameters within the vision-language adaptor. We utilize a dataset comprising 1.25 million image-text paired captions obtained from ShareGPT4V, along with 2.5 million Document OCR rendering pairs to train the VL adaptor.

Nevertheless, compared to Large Language Models (LLMs), vision-language adaptors (e.g., a 2-layer MLP) have a significantly smaller parameter capacity. This limitation in model capacity restricts the capabilities that can be learned during this stage. A natural question arises: Can the law of data scaling be effective at this stage? To address this question, we conducted a simple experiment in Table[8](https://arxiv.org/html/2403.05525v2#S4.T8 "Table 8 ‣ 4.4 Ablation Study ‣ 4 Evaluation ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding"). The results demonstrate that expanding the data scale at this stage does not provide benefits and may even lead to inferior performance. Consequently, we proceed to unfreeze the Large Language Model (LLM) and investigate efficient vision-language pretraining approaches during stage 2.

#### 3.2.2 Stage 2: Joint Vision-Language pretraining

In this stage, we explore effective pretraining strategies which can be considered as an additional stage to enable Large Language Models (LLMs) to comprehend multimodal inputs. We keep the vision encoder frozen and optimize the language model and VL adaptor.

Initially, we attempt to directly train the LLM with multimodal data. However, we find while the metrics for multimodal performance incrementally improved, there is a stark and severe decline in language metrics as illustrated in Figure[4](https://arxiv.org/html/2403.05525v2#S3.F4 "Figure 4 ‣ 3.2.2 Stage 2: Joint Vision-Language pretraining ‣ 3.2 Training Pipelines ‣ 3 Approach ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding") (Multimodal:Language=100%:0%),. This underscores the inherent challenge in directly conducting multimodal pretraining on the foundation of an LLM, revealing a critical trade-off between enhancing multimodal abilities and preserving linguistic proficiency.

We hypothesize that the observed phenomenon stems from two primary factors: firstly, the majority of multimodal corpora, are overly simplistic and exhibit a significant divergence from the complexity and distribution of linguistic data. Secondly, there appears to be a competitive dynamic between multimodal and linguistic modalities, leading to what can be described as catastrophic forgetting of language capabilities within the LLM.

![Image 4: Refer to caption](https://arxiv.org/html/2403.05525v2/x4.png)

Figure 4: Comparative performance results on different modality fusion ratio on training stage 2. An excessively large proportion of multimodal data (multimodal:language=100%:0%) leads to significant forgetting of language capabilities in LLMs. A suitable ratio (multimodal:language=70%:30%) can effectively mitigate the issue of language forgetting while simultaneously enhancing the model’s multimodal abilities. 

Joint Language-multimodal Training To address this challenge, we devise a straightforward yet effective joint language-multimodal training strategy. During training, we not only engage in multimodal data training but also incorporate a large proportion of language data into the training. This approach aims to balance the training focus, mitigating the adverse effects observed. We conduct experiments on the DeepSeek-VL 1B model in Figure[4](https://arxiv.org/html/2403.05525v2#S3.F4 "Figure 4 ‣ 3.2.2 Stage 2: Joint Vision-Language pretraining ‣ 3.2 Training Pipelines ‣ 3 Approach ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding") to explore the impact of varying the modality mixing ratios.

The analysis of the graph yields several key conclusions: (1). Integrating language data significantly alleviates the decline in language capabilities, demonstrating a substantial improvement in the model’s linguistic performance. (2). The inclusion of language data does not lead to a significant loss in multimodal performance, indicating that the model retains its multimodal processing abilities. (3). The performance of different modalities is strongly correlated with their respective proportions in the training dataset, substantiating the competitive relationship between the two modalities. Ultimately, we opt for a training ratio of language to multimodal data of roughly 7:3 for our final model. This ratio enables the model to maintain its language capabilities while simultaneously achieving better pretraining on multimodal data, effectively balancing the development of both language and multimodal proficiencies.

Scaling Vision-Language Pretraining Nevertheless, the pretraining stage of the model incurs a substantial computational cost, and performing iterations on the 7B model requires an excessive amount of computing power and time. One suitable strategy involves conducting experiments on a smaller model, specifically the 1.3B model, and subsequently scaling it up to the 7B model. Fortunately, we have observed that a significant portion of the outcomes obtained from the 1.3B models can be effectively transferred to the 7B model through the utilization of SFT (e.g., the encoder design). However, during the stage 2 training phase, we have encountered considerable fluctuations in the generative metrics of the 1.3B model, rendering it challenging to supervise the training process effectively. And this has been discussed in Schaeffer et al. ([2024](https://arxiv.org/html/2403.05525v2#bib.bib61)), "sharp and unpredictable changes might be induced by the researcher’s choice of measurement, even though the model family’s per-token error rate changes smoothly, continuously and predictably with increasing scale." Subsequent experiments have led us to identify the root causes of this issue: the limited capacity of the 1.3B model and the absence of SFT data within the training dataset, both of which hinder the model’s ability to accurately follow instructions. Even when the model possesses knowledge of the correct options, it struggles to generate them precisely.

To mitigate these challenges, we adopte a dual-pronged approach. Firstly, we employ the Multi-choice PPL methodology to monitor the model’s progress. This involves inputting not only the prompt and image into the network but also all the answer associated with the question. Subsequently, we calculate the PPL for each answer position (e.g., A, B, C, D) and select the option deemed correct by the model as the final answer. Secondly, we introduce SFT data into the training dataset at a minimal proportion, allowing the model to acquire some proficiency in following instructions. The combination of these two approaches ensures the maintenance of stable training metrics for the 1.3B model and bring better performance after stage3.

#### 3.2.3 Stage 3: Supervised Fine-tuning

In this phase, we finetune the pretrained DeepSeek-VL model with instruction-based fine-tuning to bolster its ability to follow instructions and engage in dialogue, culminating in the creation of the interactive DeepSeek-VL-Chat model. We optimize the language model, VL adaptor, and hybrid vision encoder with the vision-language SFT data as shown in Table[2](https://arxiv.org/html/2403.05525v2#S2.T2 "Table 2 ‣ 2.1 Vision-Language pretraining Data ‣ 2 Data Construction ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding"), SAM-B remains frozen due to the limited GPU memory. We only supervise answers and special tokens and mask the system and user prompts. To guarantee the model’s comprehensive proficiency in dialogue, we utilize a blend of multimodal data and pure text dialogue data used in DeepSeek-LLM. This approach ensures the model’s versatility across various dialogue scenarios.

![Image 5: Refer to caption](https://arxiv.org/html/2403.05525v2/x5.png)

Figure 5: Visualization results. DeepSeek-VL can understand Python code and provide detailed and organized explanations.

### 3.3 Hyperparameters and Infrastructures

The detailed hyperparameters of all stages are illustrated in Table[4](https://arxiv.org/html/2403.05525v2#S3.T4 "Table 4 ‣ 3.3 Hyperparameters and Infrastructures ‣ 3 Approach ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding"). We train and evaluate our DeepSeek-VL with HAI-LLM(High-flyer, [2023](https://arxiv.org/html/2403.05525v2#bib.bib24)), a lightweight and efficient distributed training framework. Since we use visual encoders to convert images into embedding vectors and then treat image embeddings and text embeddings uniformly, we can easily adapt pipeline parallelism to VL model training: all we need to do is to view visual encoders and text embedding as a single module and take it as the first layer of the resulting model. This very first layer has a complicated model structure and precludes standard tensor parallelism technique, but luckily it requires relatively small computation compared to upper standard transformer blocks. We therefore simply recompute the visual encoder forward pass in all tensor parallel ranks. The existence of visual encoders also leads to non-uniform execution time across model layers, so we re-divide model layers between pipeline parallelism ranks to achieve better load balance and throughput. The upper layers of DeepSeek-VL are exactly the same as those in DeepSeek-LLM. With such minor modification, we can now perform canonical 3D parallelism techniques as in Megatron(Shoeybi et al., [2019](https://arxiv.org/html/2403.05525v2#bib.bib64); Narayanan et al., [2021](https://arxiv.org/html/2403.05525v2#bib.bib55); Korthikanti et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib31)) and overlap computation and communication as in DeepSeek-LLM(DeepSeek-AI, [2024](https://arxiv.org/html/2403.05525v2#bib.bib16)). DeepSeek-VL-7B consumed 5 days on a cluster of 64 nodes, each comprising 8 Nvidia A100 GPUs, while DeepSeek-VL-1B consumed 7 days on a setup involving 16 nodes.

Table 4: Detailed hyperparameters of our DeepSeek-VL.

4 Evaluation
------------

### 4.1 Public Multimodal Benchmarks Evaluation

We evaluate our models on a series of public benchmarks:

Multimodal comprehensive understanding datasets: MMMU(Yue et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib82)), CMMMU(Zhang et al., [2024](https://arxiv.org/html/2403.05525v2#bib.bib85)), MMBench(Liu et al., [2023a](https://arxiv.org/html/2403.05525v2#bib.bib46)), MMBench-CN(Liu et al., [2023a](https://arxiv.org/html/2403.05525v2#bib.bib46)), SeedBench(Li et al., [2023a](https://arxiv.org/html/2403.05525v2#bib.bib35)) and MMV(Yu et al., [2023b](https://arxiv.org/html/2403.05525v2#bib.bib81)). We compare DeepSeek-VL with competitors on MMB/MMC-dev as current official test download link is no longer active.

Chart/table understanding datasets: OCRBench(Liu et al., [2023b](https://arxiv.org/html/2403.05525v2#bib.bib47));

Table 5: The comparison between different multi-modal models. The top half are proprietary models, while the bottom are open-source models.

Table 6: The comparison between tiny multi-modal models.

Hallucination datasets: POPE(Li et al., [2023b](https://arxiv.org/html/2403.05525v2#bib.bib39));

Scientific problem datasets: ScienceQA(Lu et al., [2022a](https://arxiv.org/html/2403.05525v2#bib.bib50)) and MathVista(Lu et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib52)).

We apply generation-based evaluation with greedy decoding. The generation-based evaluation here refers to letting the model generate free texts and parsing results from generated texts. The comparative results, as illustrated in Table[5](https://arxiv.org/html/2403.05525v2#S4.T5 "Table 5 ‣ 4.1 Public Multimodal Benchmarks Evaluation ‣ 4 Evaluation ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding"), show that DeepSeek-VL-7B surpasses most open-source models of similar size across a wide range of benchmarks.

DeepSeek-VL outperforms open-source models of similar size in benchmarks such as MMB, MMC, and SEEDbench, even approaching proprietary models (DeepSeek-VL vs. GPT-4V = 70.4 vs. 71.6 on seedbench), demonstrating its powerful natural image comprehension capability. The model also surpasses all open-source models in mathematical logic, but still lags significantly behind proprietary models like GPT-4V (36.1 vs. 47.8 on MathVista). This difference could be attributed to the variance in base model sizes.

Furthermore, as shown in Table[6](https://arxiv.org/html/2403.05525v2#S4.T6 "Table 6 ‣ 4.1 Public Multimodal Benchmarks Evaluation ‣ 4 Evaluation ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding"), DeepSeek-VL-1.3B significantly outperforms models of comparable size. It demonstrates superior performance compared to leading open-source models in the MMB benchmark test, while utilizing only close to half the parameters (1.3B vs. 2.7B), indicating its robust natural image comprehension capability. DeepSeek-VL-1.3B even achieves comparable results to 7B open-source models on MathVista, further validating the powerful logical understanding capabilities of the DeepSeek-VL family.

### 4.2 Public Language Benchmarks Evaluation

We evaluate our models on the following public language benchmarks:

Multi-subject multiple-choice datasets including MMLU(Hendrycks et al., [2020](https://arxiv.org/html/2403.05525v2#bib.bib23)).

Language understanding and reasoning datasets including HellaSwag(Zellers et al., [2019](https://arxiv.org/html/2403.05525v2#bib.bib83)).

Language modeling datasets including Pile(Gao et al., [2020](https://arxiv.org/html/2403.05525v2#bib.bib21)).

Table 7: The performance on language benchmarks. 

Math datasets including GSM8K(Cobbe et al., [2021](https://arxiv.org/html/2403.05525v2#bib.bib14)).

Code datasets including MBPP(Austin et al., [2021](https://arxiv.org/html/2403.05525v2#bib.bib5)).

Standardized exams including AGIEval(Zhong et al., [2023](https://arxiv.org/html/2403.05525v2#bib.bib89)).

We apply perplexity-based evaluation to datasets that require answers to be chosen from several options. These datasets include HellaSwag and MMLU. The perplexity-based evaluation here refers to calculating the perplexity of each option and selecting the lowest one as the model prediction. Perplexity-based evaluation helps to distinguish subtle probability difference between model predictions and avoids discontinuity of exact match style evaluation. We apply generation-based evaluation with greedy decoding for GSM8K and AGIEval. The generation-based evaluation here refers to letting the model generate free texts and parsing results from generated texts. We apply language-modeling-based evaluation for Pile-test, which means calculating the bits-per-byte on the test corpus. And the results are illustrated in Table[7](https://arxiv.org/html/2403.05525v2#S4.T7 "Table 7 ‣ 4.2 Public Language Benchmarks Evaluation ‣ 4 Evaluation ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding")

It can be observed that across the majority of language benchmarks, DeepSeek-VL performs comparably to, or even surpasses, DeepSeek-7B. For instance, it achieves scores of 68.4 vs. 68.5 on HellaSwag, which serves as a general benchmark for evaluating general language ability. DeepSeek-VL outperforms DeepSeek-7B on metrics such as MMLU and AGIEval, indicating that multimodal training methods may even aid in language tasks. Nevertheless, DeepSeek-VL-7B shows a certain degree of decline in mathematics (GSM8K), which suggests that despite efforts to promote harmony between vision and language modalities, there still exists a competitive relationship between them. This could be attributed to the limited model capacity (7B), and larger models might alleviate this issue significantly. Overall, DeepSeek-VL strives to achieve the goal of minimizing declines in language capability while addressing these challenges.

### 4.3 Human Evaluation

To further explore the capabilities of our DeepSeek-VL, we independently construct a dataset for manual evaluation. This dataset comprises 100 questions, divided into seven categories, each encompassing specific tasks. These categories and tasks are same as our taxonomy for the in-house SFT data, as shown in Table[3](https://arxiv.org/html/2403.05525v2#S2.T3 "Table 3 ‣ 2.1 Vision-Language pretraining Data ‣ 2 Data Construction ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding"). This approach ensures that the tasks we test are universal and encompass the majority of use cases for multimodal models.

Moreover, based on the categories and tasks described in existing reports, we collect similar image materials and developed prompts. The sources for these image materials include royalty-free image communities and photographs taken by the researchers. This methodical collection and prompt formulation process ensures our dataset is both comprehensive and representative of real-world multimodal model applications.

![Image 6: Refer to caption](https://arxiv.org/html/2403.05525v2/x6.png)

Figure 6: Human evaluation results on InternLM-XComposer2-VL(Dong et al., [2024](https://arxiv.org/html/2403.05525v2#bib.bib17)), CogVLM(Wang et al., [2023b](https://arxiv.org/html/2403.05525v2#bib.bib76)), DeepSeek-VL and GPT-4V(OpenAI, [2023b](https://arxiv.org/html/2403.05525v2#bib.bib59)).

We compare our DeepSeek-VL-7B with InternLM-XComposer2-VL, CogVLM and GPT-4V as shown in Figure[6](https://arxiv.org/html/2403.05525v2#S4.F6 "Figure 6 ‣ 4.3 Human Evaluation ‣ 4 Evaluation ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding") (and we also provide visualization results in Appendix[A](https://arxiv.org/html/2403.05525v2#A1 "Appendix A Appendix ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding")). GPT-4V demonstrates exceptional performance across most dimensions. All open-source models are still far behind GPT-4V in logical reasoning, highlighting the necessity of scaling up the size of Large Language Models (LLMs). DeepSeek-VL-7B achieves better results in overall performance, reaching outcomes close to GPT-4V in Recognition, Conversion, and Commonsense Reasoning.

![Image 7: Refer to caption](https://arxiv.org/html/2403.05525v2/x7.png)

Figure 7: GPT-4V-based Evaluation Results of DeepSeek-VL vs. Other Models: The chart depicts results from a GPT-4V-based assessment across 99 test samples, demonstrating DeepSeek-VL’s favorable outcomes against both open-source and proprietary models.

In addition, we conduct a comparative assessment using GPT-4V to evaluate the performance of DeepSeek-VL against other models across a set of 99 test samples designed for human evaluation. Following (Zheng et al., [2024](https://arxiv.org/html/2403.05525v2#bib.bib88)), we show GPT-4V the question and the answers from two different models and ask GPT-4V to determine which one is better or declare a tie. The results indicate a preference for DeepSeek-VL’s responses in the majority of cases, as GPT-4V tends to rate the quality of DeepSeek-VL’s answers more favorably. As illustrated in Figure[7](https://arxiv.org/html/2403.05525v2#S4.F7 "Figure 7 ‣ 4.3 Human Evaluation ‣ 4 Evaluation ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding"), DeepSeek-VL is judged to be superior in over 60% of instances when compared to open-source multimodal models, including Fuyu-8B, CogVLM-17B, and InternLM-XComposer2-VL. Moreover, in comparison with other proprietary models, such as GPT-4V itself, DeepSeek-VL demonstrates comparably exceptional performance.

### 4.4 Ablation Study

Scale Up Projector Training We expand the dataset for stage 1 (projector warmup) and subsequently apply supervised fine-tuning. The results, depicted in Figure[8](https://arxiv.org/html/2403.05525v2#S4.T8 "Table 8 ‣ 4.4 Ablation Study ‣ 4 Evaluation ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding"), demonstrate that augmenting the training data volume does not enhance performance at this stage. This implies that the projector’s capacity is inherently constrained, rendering it incapable of capturing the extensive knowledge necessary for multimodal tasks.

Table 8: Comparative directly SFT performance results on scaling up stage 1 data. The results demonstrate that expanding the data scale at this stage does not yield benefits, or even results in worse performance.

Table 9: Analysis of model performance across training stages.

Training Stage In Table[9](https://arxiv.org/html/2403.05525v2#S4.T9 "Table 9 ‣ 4.4 Ablation Study ‣ 4 Evaluation ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding"), we examine the contributions of each stage to the model’s performance. It’s evident that combining stage 1, stage 2, and stage 3 yields significantly better results across all metrics compared to combining stage 1 and stage 3 alone, demonstrating the effectiveness of multimodal pretraining. Additionally, the combination of stage 2 and stage 3 still slightly lags behind the combined performance of stage 1, stage 2, and stage 3, indicating that vision-language adaptor warmup stage remains meaningful.

Modality Group Training When mixing language and multimodal data, we observe that directly blending them at the batch level significantly reduces training efficiency. This inefficiency arises because each batch gradient backpropagation process waits for the slowest sample to complete. As a result, the predominantly faster-to-process pure language data ends up waiting for the multimodal samples to finish, leading to a decrease in overall training efficiency.

![Image 8: Refer to caption](https://arxiv.org/html/2403.05525v2/x8.png)

Figure 8: Comparative analysis of modality warmup on language (Pile-test) and multimodal (MMBench and MMBench _ _\_ _ CN) benchmarks demonstrates that modality grouping consistently surpasses the non-grouped modality approach in language tasks, while simultaneously preserving performance on multimodal tasks on training stage 2 (Multimodal:Language=60%:40%). 

To address this issue, we experiment with grouping different modalities of data at each global step, sampling distinct modalities separately. This approach involves organizing the training data so that batches are composed either entirely of language data or entirely of multimodal data at different training steps, rather than mixing them within the same batch.

The results are shown in Figure[8](https://arxiv.org/html/2403.05525v2#S4.F8 "Figure 8 ‣ 4.4 Ablation Study ‣ 4 Evaluation ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding"), we observe that this method does not compromise the model’s performance while enhancing the model’s training efficiency by 20%. This strategy effectively circumvents the bottleneck caused by the disparate processing times between modalities, optimizing the training workflow.

Modality Warmup Considering that our approach involves multimodal training on the foundation of a language model, directly mixing multimodal data in a fixed proportion from the outset can destabilize the model. To counteract this issue, we propose a simple yet effective modality warm-up strategy. Initially, we set the language data ratio to 1, and then gradually decrease it to the target ratio for the final model training (e.g., 0.7).

![Image 9: Refer to caption](https://arxiv.org/html/2403.05525v2/x9.png)

Figure 9: Comparative performance results on language (Pile-test) and multimodal (MMBench and MMBench_CN) benchmarks for modality warmup. Modality warmup consistently matches or surpasses the performance of approaches without modality warmup across all evaluated tasks on training stage 2 (Multimodal:Language=60%:40%).

Our experiments, as illustrated in Figure[9](https://arxiv.org/html/2403.05525v2#S4.F9 "Figure 9 ‣ 4.4 Ablation Study ‣ 4 Evaluation ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding"), demonstrate that this strategy effectively prevents a significant decline in language capabilities at the beginning of training, while also yielding comparatively superior outcomes in the final phases for both the language and multimodal domains. This gradual adaptation enables the model to more seamlessly adjust to the incorporation of multimodal data, thereby improving overall training stability and performance.

Vision Encoder Selection In order to better acquire and utilize image information, we compare the training loss of different vision encoders under our training settings except for reducing training steps of stage 2 to 8000 for efficiency. As illustrated in Figure[10](https://arxiv.org/html/2403.05525v2#S4.F10 "Figure 10 ‣ 4.4 Ablation Study ‣ 4 Evaluation ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding"), the incorporation of vision-only self-supervised encoders has been found to significantly enhance performance on training loss. To more effectively process high-resolution images, our research ultimately adopts a hybrid vision encoder strategy, combining SigLIP with SAM for our model’s implementation.

![Image 10: Refer to caption](https://arxiv.org/html/2403.05525v2/x10.png)

Figure 10: Comparative analysis of different vision encoders on training losses in stage 2.

Vision-Language Adaptor Design To improve the efficiency of extracting information from the visual encoder while adhering to current token length constraints, adjustments can be made to the Vision-Language adaptor in two main ways: the method used to combine visual features and the design of the MLP adaptor.

Previous studies (Tong et al., [2024](https://arxiv.org/html/2403.05525v2#bib.bib70)) have indicated that combining visual features along the sequence dimension can lead to better model performance, although this comes with the trade-off of increased computational requirements due to a longer sequence of visual feature tokens. As demonstrated in the top section of Table[10](https://arxiv.org/html/2403.05525v2#S4.T10 "Table 10 ‣ 4.4 Ablation Study ‣ 4 Evaluation ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding"), reducing the sequence length by stacking visual features along the image’s width or height dimensions before sequence concatenation, in order to keep the sequence length constant, does not achieve better results compared to simply merging them along the embedding dimension in most metrics. In terms of the adaptor architecture, employing separate MLP adaptors for each vision feature encoder allows for more precise adjustments to the specific values and distribution patterns of visual features, facilitating smoother model training. Conversely, using a shared MLP adaptor for different vision encoders contributes to adequate feature fusion. We adopt a mixed strategy and report stable and improved performance, as outlined in the lower section of Table[10](https://arxiv.org/html/2403.05525v2#S4.T10 "Table 10 ‣ 4.4 Ablation Study ‣ 4 Evaluation ‣ DeepSeek-VL: Towards Real-World Vision-Language Understanding").

Architecture MMB MMC SEED POPE ScienceQA MMMU OCRB Average
Sequence Concatenation:
Token Pooling - W 61.2 59.6 61.6 86.5 57.7 31.6 304 55.5
Token Pooling - H 59.9 58.3 61.6 83.8 55.0 32.0 291 54.2
Embedding Concatenation:
Hybrid MLP 61.7 60.1 62.9 87.8 56.6 31.3 309 55.9
Shared MLP 62.0 58.9 62.5 86.6 54.7 30.2 318 55.2
Separate MLP 57.5 58.7 63.1 86.5 56.6 29.0 299 54.5

Table 10: Comparison of different adaptor architectures using SigLIP and SAM as hybrid vision encoder, Hybrid MLP are used for sequence concatenation experiments. Bolded entries represent the best results, while underlined entries denote the second-best results. For calculating the average score, we divide the OCRBench by the total number of questions.

5 Conclusion, Limitation, and Future Work
-----------------------------------------

In this technical report, we have introduced DeepSeek-VL, a series of Multimodal Large Language Models, available in scales of 1.3B and 6.7B parameters. This report has unveiled the limitations inherent in the predominant projector-based pretraining methodologies, setting the stage for the innovative approach adopted by DeepSeek-VL. By prioritizing a joint vision and language (VL) pretraining phase, DeepSeek-VL transcends traditional models by ensuring that the integration of multimodal data does not compromise the linguistic capabilities of the Large Language Models (LLMs). This is achieved through a strategic warm-up data ratio and the introduction of a hybrid vision encoder, which together enable the efficient processing of high-resolution images without losing sight of semantic richness.

The incorporation of a hybrid vision encoder, capable of handling 1024 x 1024 images within a constrained token budget, underscores our commitment to preserving the nuanced details and semantic integrity across diverse tasks. As a result, DeepSeek-VL emerges as a pioneering model that not only meets but exceeds the standards set by generalist models in its class. It showcases exceptional performance across a wide range of visually-centric benchmarks while sustaining formidable proficiency in language-centric evaluations.

In making DeepSeek-VL publicly available, we aim to catalyze further innovation and exploration within the research community, providing a robust foundation upon which future studies can build. This gesture of openness is intended to facilitate the collective advancement of our understanding and capabilities in handling multimodal data.

Looking ahead, we are excited to announce plans to scale up DeepSeek-VL to larger sizes, incorporating Mixture of Experts (MoE) technology. This forthcoming expansion promises to further enhance the model’s efficiency and effectiveness, opening up new horizons for research and application in the field of AI.

References
----------

*   01-ai (2024) 01-ai. Yi-34B vision language model. [https://huggingface.co/01-ai/Yi-VL-34B](https://huggingface.co/01-ai/Yi-VL-34B), 2024. 
*   Abi (2024) Abi. Screenshot to code. [https://github.com/abi/screenshot-to-code](https://github.com/abi/screenshot-to-code), 2024. 
*   Anna’s Archive (2024) Anna’s Archive. Anna’s archive. [https://annas-archive.org/](https://annas-archive.org/), 2024. 
*   Anthropic (2023) Anthropic. Introducing Claude, 2023. URL [https://www.anthropic.com/index/introducing-claude](https://www.anthropic.com/index/introducing-claude). 
*   Austin et al. (2021) J.Austin, A.Odena, M.Nye, M.Bosma, H.Michalewski, D.Dohan, E.Jiang, C.Cai, M.Terry, Q.Le, et al. Program synthesis with large language models. _arXiv preprint arXiv:2108.07732_, 2021. 
*   Bai et al. (2023) J.Bai, S.Bai, S.Yang, S.Wang, S.Tan, P.Wang, J.Lin, C.Zhou, and J.Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. _arXiv preprint arXiv:2308.12966_, 2023. 
*   Bavishi et al. (2023) R.Bavishi, E.Elsen, C.Hawthorne, M.Nye, A.Odena, A.Somani, and S.Taşırlar. Introducing our multimodal models, 2023. URL [https://www.adept.ai/blog/fuyu-8b](https://www.adept.ai/blog/fuyu-8b). 
*   Blecher (2024) L.Blecher. Latex-ocr. GitHub repository, 2024. URL [https://github.com/lukas-blecher/LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR). 
*   Blecher et al. (2023) L.Blecher, G.Cucurull, T.Scialom, and R.Stojnic. Nougat: Neural optical understanding for academic documents. _arXiv preprint arXiv:2308.13418_, 2023. 
*   Burns et al. (2023) A.Burns, K.Srinivasan, J.Ainslie, G.Brown, B.A. Plummer, K.Saenko, J.Ni, and M.Guo. A suite of generative tasks for multi-level multimodal webpage understanding. In _The 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)_, 2023. URL [https://openreview.net/forum?id=rwcLHjtUmn](https://openreview.net/forum?id=rwcLHjtUmn). 
*   Carter (2024) J.Carter. Textocr-gpt4v. [https://huggingface.co/datasets/jimmycarter/textocr-gpt4v](https://huggingface.co/datasets/jimmycarter/textocr-gpt4v), 2024. 
*   Chen et al. (2023) L.Chen, J.Li, X.Dong, P.Zhang, C.He, J.Wang, F.Zhao, and D.Lin. Sharegpt4v: Improving large multi-modal models with better captions. _arXiv preprint arXiv:2311.12793_, 2023. 
*   Chng et al. (2019) C.K. Chng, Y.Liu, Y.Sun, C.C. Ng, C.Luo, Z.Ni, C.Fang, S.Zhang, J.Han, E.Ding, et al. Icdar2019 robust reading challenge on arbitrary-shaped text-rrc-art. In _2019 International Conference on Document Analysis and Recognition (ICDAR)_, pages 1571–1576. IEEE, 2019. 
*   Cobbe et al. (2021) K.Cobbe, V.Kosaraju, M.Bavarian, M.Chen, H.Jun, L.Kaiser, M.Plappert, J.Tworek, J.Hilton, R.Nakano, et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_, 2021. 
*   Dai et al. (2023) W.Dai, J.Li, D.Li, A.M.H. Tiong, J.Zhao, W.Wang, B.Li, P.Fung, and S.Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023. 
*   DeepSeek-AI (2024) DeepSeek-AI. Deepseek llm: Scaling open-source language models with longtermism. _arXiv preprint arXiv:2401.02954_, 2024. URL [https://github.com/deepseek-ai/DeepSeek-LLM](https://github.com/deepseek-ai/DeepSeek-LLM). 
*   Dong et al. (2024) X.Dong, P.Zhang, Y.Zang, Y.Cao, B.Wang, L.Ouyang, X.Wei, S.Zhang, H.Duan, M.Cao, et al. Internlm-xcomposer2: Mastering free-form text-image composition and comprehension in vision-language large model. _arXiv preprint arXiv:2401.16420_, 2024. 
*   echo840 (2024) echo840. Detailed caption dataset. [https://huggingface.co/datasets/echo840/Detailed_Caption](https://huggingface.co/datasets/echo840/Detailed_Caption), 2024. 
*   (19) W.Foundation. Wikimedia downloads. URL [https://dumps.wikimedia.org](https://dumps.wikimedia.org/). 
*   Gao et al. (2023) J.Gao, R.Pi, J.Zhang, J.Ye, W.Zhong, Y.Wang, L.Hong, J.Han, H.Xu, Z.Li, et al. G-llava: Solving geometric problem with multi-modal large language model. _arXiv preprint arXiv:2312.11370_, 2023. 
*   Gao et al. (2020) L.Gao, S.Biderman, S.Black, L.Golding, T.Hoppe, C.Foster, J.Phang, H.He, A.Thite, N.Nabeshima, et al. The Pile: An 800GB dataset of diverse text for language modeling. _arXiv preprint arXiv:2101.00027_, 2020. 
*   Google (2023) Google. An important next step on our AI journey, 2023. URL [https://blog.google/technology/ai/bard-google-ai-search-updates/](https://blog.google/technology/ai/bard-google-ai-search-updates/). 
*   Hendrycks et al. (2020) D.Hendrycks, C.Burns, S.Basart, A.Zou, M.Mazeika, D.Song, and J.Steinhardt. Measuring massive multitask language understanding. _arXiv preprint arXiv:2009.03300_, 2020. 
*   High-flyer (2023) High-flyer. Hai-llm: 高效且轻量的大模型训练工具, 2023. URL [https://www.high-flyer.cn/en/blog/hai-llm](https://www.high-flyer.cn/en/blog/hai-llm). 
*   Hsiao et al. (2022) Y.-C. Hsiao, F.Zubach, M.Wang, et al. Screenqa: Large-scale question-answer pairs over mobile app screenshots. _arXiv preprint arXiv:2209.08199_, 2022. 
*   Hu et al. (2023) A.Hu, Y.Shi, H.Xu, J.Ye, Q.Ye, M.Yan, C.Li, Q.Qian, J.Zhang, and F.Huang. mplug-paperowl: Scientific diagram analysis with the multimodal large language model. _arXiv preprint arXiv:2311.18248_, 2023. 
*   HuggingFaceM4 (2024) HuggingFaceM4. Websight dataset. [https://huggingface.co/datasets/HuggingFaceM4/WebSight](https://huggingface.co/datasets/HuggingFaceM4/WebSight), 2024. 
*   Kantharaj et al. (2022) S.Kantharaj, R.T. Leong, X.Lin, A.Masry, M.Thakkar, E.Hoque, and S.Joty. Chart-to-text: A large-scale benchmark for chart summarization. In S.Muresan, P.Nakov, and A.Villavicencio, editors, _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, pages 4005–4023, Dublin, Ireland, May 2022. Association for Computational Linguistics. [10.18653/v1/2022.acl-long.277](https://arxiv.org/doi.org/10.18653/v1/2022.acl-long.277). URL [https://aclanthology.org/2022.acl-long.277](https://aclanthology.org/2022.acl-long.277). 
*   Kirillov et al. (2023) A.Kirillov, E.Mintun, N.Ravi, H.Mao, C.Rolland, L.Gustafson, T.Xiao, S.Whitehead, A.C. Berg, W.-Y. Lo, et al. Segment anything. _arXiv preprint arXiv:2304.02643_, 2023. 
*   Kocetkov et al. (2023) D.Kocetkov, R.Li, L.B. Allal, J.Li, C.Mou, C.M. Ferrandis, Y.Jernite, M.Mitchell, S.Hughes, T.Wolf, D.Bahdanau, L.von Werra, and H.de Vries. The stack: 3 tb of permissively licensed source code. In _Transactions on Machine Learning Research_, 2023. 
*   Korthikanti et al. (2023) V.A. Korthikanti, J.Casper, S.Lym, L.McAfee, M.Andersch, M.Shoeybi, and B.Catanzaro. Reducing activation recomputation in large transformer models. _Proceedings of Machine Learning and Systems_, 5, 2023. 
*   Krylov et al. (2021) I.Krylov, S.Nosov, and V.Sovrasov. Open images v5 text annotation and yet another mask text spotter. In _Asian Conference on Machine Learning_, pages 379–389. PMLR, 2021. 
*   (33) A.Kulkarni and J.Truelsen. wkhtmltopdf. [https://wkhtmltopdf.org/](https://wkhtmltopdf.org/). Project maintained by Ashish Kulkarni, originally created by Jakob Truelsen. Accessed: 2024-02-22. 
*   LAION (2023) LAION. Gpt-4v dataset. [https://huggingface.co/datasets/laion/gpt4v-dataset](https://huggingface.co/datasets/laion/gpt4v-dataset), 2023. 
*   Li et al. (2023a) B.Li, R.Wang, G.Wang, Y.Ge, Y.Ge, and Y.Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. _arXiv preprint arXiv:2307.16125_, 2023a. 
*   Li and Tajbakhsh (2023) S.Li and N.Tajbakhsh. Scigraphqa: A large-scale synthetic multi-turn question-answering dataset for scientific graphs, 2023. 
*   Li et al. (2020) Y.Li, G.Li, L.He, J.Zheng, H.Li, and Z.Guan. Widget captioning: Generating natural language description for mobile user interface elements. _arXiv preprint arXiv:2010.04295_, 2020. 
*   Li et al. (2022) Y.Li, H.Mao, R.Girshick, and K.He. Exploring plain vision transformer backbones for object detection. In _European Conference on Computer Vision_, pages 280–296. Springer, 2022. 
*   Li et al. (2023b) Y.Li, Y.Du, K.Zhou, J.Wang, W.X. Zhao, and J.-R. Wen. Evaluating object hallucination in large vision-language models. _arXiv preprint arXiv:2305.10355_, 2023b. 
*   Lin et al. (2023a) J.Lin, H.Yin, W.Ping, Y.Lu, P.Molchanov, A.Tao, H.Mao, J.Kautz, M.Shoeybi, and S.Han. Vila: On pre-training for visual language models. _arXiv preprint arXiv:2312.07533_, 2023a. 
*   Lin et al. (2023b) Z.Lin, C.Liu, R.Zhang, P.Gao, L.Qiu, H.Xiao, H.Qiu, C.Lin, W.Shao, K.Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. _arXiv preprint arXiv:2311.07575_, 2023b. 
*   Liu et al. (2022a) F.Liu, F.Piccinno, S.Krichene, C.Pang, K.Lee, M.Joshi, Y.Altun, N.Collier, and J.M. Eisenschlos. Matcha: Enhancing visual language pretraining with math reasoning and chart derendering. _arXiv preprint arXiv:2212.09662_, 2022a. 
*   Liu et al. (2024a) H.Liu, C.Li, Y.Li, B.Li, Y.Zhang, S.Shen, and Y.J. Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024a. URL [https://llava-vl.github.io/blog/2024-01-30-llava-next/](https://llava-vl.github.io/blog/2024-01-30-llava-next/). 
*   Liu et al. (2024b) H.Liu, C.Li, Q.Wu, and Y.J. Lee. Visual instruction tuning. _Advances in neural information processing systems_, 36, 2024b. 
*   Liu et al. (2022b) Y.Liu, G.Zhu, B.Zhu, Q.Song, G.Ge, H.Chen, G.Qiao, R.Peng, L.Wu, and J.Wang. Taisu: A 166m large-scale high-quality dataset for chinese vision-language pre-training. In S.Koyejo, S.Mohamed, A.Agarwal, D.Belgrave, K.Cho, and A.Oh, editors, _Advances in Neural Information Processing Systems_, volume 35, pages 16705–16717. Curran Associates, Inc., 2022b. URL [https://proceedings.neurips.cc/paper_files/paper/2022/file/6a386d703b50f1cf1f61ab02a15967bb-Paper-Datasets_and_Benchmarks.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/6a386d703b50f1cf1f61ab02a15967bb-Paper-Datasets_and_Benchmarks.pdf). 
*   Liu et al. (2023a) Y.Liu, H.Duan, Y.Zhang, B.Li, S.Zhang, W.Zhao, Y.Yuan, J.Wang, C.He, Z.Liu, et al. Mmbench: Is your multi-modal model an all-around player? _arXiv preprint arXiv:2307.06281_, 2023a. 
*   Liu et al. (2023b) Y.Liu, Z.Li, H.Li, W.Yu, M.Huang, D.Peng, M.Liu, M.Chen, C.Li, L.Jin, et al. On the hidden mystery of ocr in large multimodal models. _arXiv preprint arXiv:2305.07895_, 2023b. 
*   Long et al. (2022) S.Long, S.Qin, D.Panteleev, A.Bissacco, Y.Fujii, and M.Raptis. Towards end-to-end unified scene text detection and layout analysis. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, 2022. 
*   Lu et al. (2021) P.Lu, L.Qiu, J.Chen, T.Xia, Y.Zhao, W.Zhang, Z.Yu, X.Liang, and S.-C. Zhu. Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning. _arXiv preprint arXiv:2110.13214_, 2021. 
*   Lu et al. (2022a) P.Lu, S.Mishra, T.Xia, L.Qiu, K.-W. Chang, S.-C. Zhu, O.Tafjord, P.Clark, and A.Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In _The 36th Conference on Neural Information Processing Systems (NeurIPS)_, 2022a. 
*   Lu et al. (2022b) P.Lu, S.Mishra, T.Xia, L.Qiu, K.-W. Chang, S.-C. Zhu, O.Tafjord, P.Clark, and A.Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. _Advances in Neural Information Processing Systems_, 35:2507–2521, 2022b. 
*   Lu et al. (2023) P.Lu, H.Bansal, T.Xia, J.Liu, C.Li, H.Hajishirzi, H.Cheng, K.-W. Chang, M.Galley, and J.Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. _arXiv preprint arXiv:2310.02255_, 2023. 
*   Mao et al. (2016) J.Mao, J.Huang, A.Toshev, O.Camburu, A.L. Yuille, and K.Murphy. Generation and comprehension of unambiguous object descriptions. In _Proceedings of the IEEE conference on computer vision and pattern recognition_, pages 11–20, 2016. 
*   Masry et al. (2023) A.Masry, P.Kavehzadeh, X.L. Do, E.Hoque, and S.Joty. Unichart: A universal vision-language pretrained model for chart comprehension and reasoning. _arXiv preprint arXiv:2305.14761_, 2023. 
*   Narayanan et al. (2021) D.Narayanan, M.Shoeybi, J.Casper, P.LeGresley, M.Patwary, V.Korthikanti, D.Vainbrand, P.Kashinkunti, J.Bernauer, B.Catanzaro, et al. Efficient large-scale language model training on gpu clusters using megatron-lm. In _Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis_, pages 1–15, 2021. 
*   Nayef et al. (2017) N.Nayef, F.Yin, I.Bizid, H.Choi, Y.Feng, D.Karatzas, Z.Luo, U.Pal, C.Rigaud, J.Chazalon, et al. Icdar2017 robust reading challenge on multi-lingual scene text detection and script identification-rrc-mlt. In _2017 14th IAPR international conference on document analysis and recognition (ICDAR)_, volume 1, pages 1454–1459. IEEE, 2017. 
*   OpenAI (2022) OpenAI. Chatgpt: Optimizing language models for dialogue. 2022. URL [https://openai.com/blog/chatgpt](https://openai.com/blog/chatgpt). 
*   OpenAI (2023a) OpenAI. GPT-4 technical report. _arXiv_, 2023a. 
*   OpenAI (2023b) R.OpenAI. Gpt-4v(ision) system card. 2023b. 
*   Rodriguez et al. (2023) J.A. Rodriguez, D.Vazquez, I.Laradji, M.Pedersoli, and P.Rodriguez. Ocr-vqgan: Taming text-within-image generation. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_, pages 3689–3698, 2023. 
*   Schaeffer et al. (2024) R.Schaeffer, B.Miranda, and S.Koyejo. Are emergent abilities of large language models a mirage? _Advances in Neural Information Processing Systems_, 36, 2024. 
*   Shazeer (2020) N.Shazeer. Glu variants improve transformer. _arXiv preprint arXiv:2002.05202_, 2020. 
*   Shi et al. (2017) B.Shi, C.Yao, M.Liao, M.Yang, P.Xu, L.Cui, S.Belongie, S.Lu, and X.Bai. Icdar2017 competition on reading chinese text in the wild (rctw-17). In _2017 14th iapr international conference on document analysis and recognition (ICDAR)_, volume 1, pages 1429–1434. IEEE, 2017. 
*   Shoeybi et al. (2019) M.Shoeybi, M.Patwary, R.Puri, P.LeGresley, J.Casper, and B.Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. _arXiv preprint arXiv:1909.08053_, 2019. 
*   Singh et al. (2021) A.Singh, G.Pang, M.Toh, J.Huang, W.Galuba, and T.Hassner. Textocr: Towards large-scale end-to-end reasoning for arbitrary-shaped scene text. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_, pages 8802–8812, 2021. 
*   Su et al. (2024) J.Su, M.Ahmed, Y.Lu, S.Pan, W.Bo, and Y.Liu. Roformer: Enhanced transformer with rotary position embedding. _Neurocomputing_, 568:127063, 2024. 
*   Sun et al. (2023) Q.Sun, Q.Yu, Y.Cui, F.Zhang, X.Zhang, Y.Wang, H.Gao, J.Liu, T.Huang, and X.Wang. Generative pretraining in multimodality. _arXiv preprint arXiv:2307.05222_, 2023. 
*   Sun et al. (2019) Y.Sun, Z.Ni, C.-K. Chng, Y.Liu, C.Luo, C.C. Ng, J.Han, E.Ding, J.Liu, D.Karatzas, et al. Icdar 2019 competition on large-scale street view text with partial labeling-rrc-lsvt. In _2019 International Conference on Document Analysis and Recognition (ICDAR)_, pages 1557–1562. IEEE, 2019. 
*   Team et al. (2023) G.Team, R.Anil, S.Borgeaud, Y.Wu, J.-B. Alayrac, J.Yu, R.Soricut, J.Schalkwyk, A.M. Dai, A.Hauth, et al. Gemini: a family of highly capable multimodal models. _arXiv preprint arXiv:2312.11805_, 2023. 
*   Tong et al. (2024) S.Tong, Z.Liu, Y.Zhai, Y.Ma, Y.LeCun, and S.Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. _arXiv preprint arXiv:2401.06209_, 2024. 
*   Touvron et al. (2023a) H.Touvron, T.Lavril, G.Izacard, X.Martinet, M.-A. Lachaux, T.Lacroix, B.Rozière, N.Goyal, E.Hambro, F.Azhar, et al. LLaMA: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_, 2023a. 
*   Touvron et al. (2023b) H.Touvron, L.Martin, K.Stone, P.Albert, A.Almahairi, Y.Babaei, N.Bashlykov, S.Batra, P.Bhargava, S.Bhosale, D.Bikel, L.Blecher, C.Canton-Ferrer, M.Chen, G.Cucurull, D.Esiobu, J.Fernandes, J.Fu, W.Fu, B.Fuller, C.Gao, V.Goswami, N.Goyal, A.Hartshorn, S.Hosseini, R.Hou, H.Inan, M.Kardas, V.Kerkez, M.Khabsa, I.Kloumann, A.Korenev, P.S. Koura, M.Lachaux, T.Lavril, J.Lee, D.Liskovich, Y.Lu, Y.Mao, X.Martinet, T.Mihaylov, P.Mishra, I.Molybog, Y.Nie, A.Poulton, J.Reizenstein, R.Rungta, K.Saladi, A.Schelten, R.Silva, E.M. Smith, R.Subramanian, X.E. Tan, B.Tang, R.Taylor, A.Williams, J.X. Kuan, P.Xu, Z.Yan, I.Zarov, Y.Zhang, A.Fan, M.Kambadur, S.Narang, A.Rodriguez, R.Stojnic, S.Edunov, and T.Scialom. Llama 2: Open foundation and fine-tuned chat models. _CoRR_, abs/2307.09288, 2023b. [10.48550/arXiv.2307.09288](https://arxiv.org/doi.org/10.48550/arXiv.2307.09288). URL [https://doi.org/10.48550/arXiv.2307.09288](https://doi.org/10.48550/arXiv.2307.09288). 
*   Veit et al. (2016) A.Veit, T.Matera, L.Neumann, J.Matas, and S.Belongie. Coco-text: Dataset and benchmark for text detection and recognition in natural images. _arXiv preprint arXiv:1601.07140_, 2016. 
*   Wang et al. (2021) B.Wang, G.Li, X.Zhou, Z.Chen, T.Grossman, and Y.Li. Screen2words: Automatic mobile ui summarization with multimodal learning. In _The 34th Annual ACM Symposium on User Interface Software and Technology_, pages 498–510, 2021. 
*   Wang et al. (2023a) J.Wang, L.Meng, Z.Weng, B.He, Z.Wu, and Y.-G. Jiang. To see is to believe: Prompting gpt-4v for better visual instruction tuning. _arXiv preprint arXiv:2311.07574_, 2023a. 
*   Wang et al. (2023b) W.Wang, Q.Lv, W.Yu, W.Hong, J.Qi, Y.Wang, J.Ji, Z.Yang, L.Zhao, X.Song, et al. Cogvlm: Visual expert for pretrained language models. _arXiv preprint arXiv:2311.03079_, 2023b. 
*   Wei et al. (2023) H.Wei, L.Kong, J.Chen, L.Zhao, Z.Ge, J.Yang, J.Sun, C.Han, and X.Zhang. Vary: Scaling up the vision vocabulary for large vision-language models. _arXiv preprint arXiv:2312.06109_, 2023. 
*   Yang et al. (2021) Y.Yang, A.Panagopoulou, Q.Lyu, L.Zhang, M.Yatskar, and C.Callison-Burch. Visual goal-step inference using wikihow. _arXiv preprint arXiv:2104.05845_, 2021. 
*   Ye et al. (2023) J.Ye, A.Hu, H.Xu, Q.Ye, M.Yan, G.Xu, C.Li, J.Tian, Q.Qian, J.Zhang, et al. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. _arXiv preprint arXiv:2310.05126_, 2023. 
*   Yu et al. (2023a) Q.Yu, Q.Sun, X.Zhang, Y.Cui, F.Zhang, Y.Cao, X.Wang, and J.Liu. Capsfusion: Rethinking image-text data at scale. _arXiv preprint arXiv:2310.20550_, 2023a. 
*   Yu et al. (2023b) W.Yu, Z.Yang, L.Li, J.Wang, K.Lin, Z.Liu, X.Wang, and L.Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. _arXiv preprint arXiv:2308.02490_, 2023b. 
*   Yue et al. (2023) X.Yue, Y.Ni, K.Zhang, T.Zheng, R.Liu, G.Zhang, S.Stevens, D.Jiang, W.Ren, Y.Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. _arXiv preprint arXiv:2311.16502_, 2023. 
*   Zellers et al. (2019) R.Zellers, A.Holtzman, Y.Bisk, A.Farhadi, and Y.Choi. HellaSwag: Can a machine really finish your sentence? In A.Korhonen, D.R. Traum, and L.Màrquez, editors, _Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers_, pages 4791–4800. Association for Computational Linguistics, 2019. [10.18653/v1/p19-1472](https://arxiv.org/doi.org/10.18653/v1/p19-1472). URL [https://doi.org/10.18653/v1/p19-1472](https://doi.org/10.18653/v1/p19-1472). 
*   Zhang and Sennrich (2019) B.Zhang and R.Sennrich. Root mean square layer normalization. _Advances in Neural Information Processing Systems_, 32, 2019. 
*   Zhang et al. (2024) G.Zhang, X.Du, B.Chen, Y.Liang, T.Luo, T.Zheng, K.Zhu, Y.Cheng, C.Xu, S.Guo, et al. Cmmmu: A chinese massive multi-discipline multimodal understanding benchmark. _arXiv preprint arXiv:2401.11944_, 2024. 
*   Zhang et al. (2019) R.Zhang, Y.Zhou, Q.Jiang, Q.Song, N.Li, K.Zhou, L.Wang, D.Wang, M.Liao, M.Yang, et al. Icdar 2019 robust reading challenge on reading chinese text on signboard. In _2019 international conference on document analysis and recognition (ICDAR)_, pages 1577–1581. IEEE, 2019. 
*   Zhang et al. (2017) Y.Zhang, L.Gueguen, I.Zharkov, P.Zhang, K.Seifert, and B.Kadlec. Uber-text: A large-scale dataset for optical character recognition from street-level imagery. In _SUNw: Scene Understanding Workshop - CVPR 2017_, Hawaii, U.S.A., 2017. URL [http://sunw.csail.mit.edu/abstract/uberText.pdf](http://sunw.csail.mit.edu/abstract/uberText.pdf). 
*   Zheng et al. (2024) L.Zheng, W.-L. Chiang, Y.Sheng, S.Zhuang, Z.Wu, Y.Zhuang, Z.Lin, Z.Li, D.Li, E.Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. _Advances in Neural Information Processing Systems_, 36, 2024. 
*   Zhong et al. (2023) W.Zhong, R.Cui, Y.Guo, Y.Liang, S.Lu, Y.Wang, A.Saied, W.Chen, and N.Duan. AGIEval: A human-centric benchmark for evaluating foundation models. _CoRR_, abs/2304.06364, 2023. [10.48550/arXiv.2304.06364](https://arxiv.org/doi.org/10.48550/arXiv.2304.06364). URL [https://doi.org/10.48550/arXiv.2304.06364](https://doi.org/10.48550/arXiv.2304.06364). 
*   Zhu et al. (2024) W.Zhu, J.Hessel, A.Awadalla, S.Y. Gadre, J.Dodge, A.Fang, Y.Yu, L.Schmidt, W.Y. Wang, and Y.Choi. Multimodal c4: An open, billion-scale corpus of images interleaved with text. _Advances in Neural Information Processing Systems_, 36, 2024. 

Appendix A Appendix
-------------------

![Image 11: Refer to caption](https://arxiv.org/html/2403.05525v2/x11.png)

Figure 11: Visualization results. DeepSeek-VL can understand children’s programming diagrams from the real world and provide detailed and organized explanations.

![Image 12: Refer to caption](https://arxiv.org/html/2403.05525v2/x12.png)

Figure 12: Visualization results. DeepSeek-VL has strong understanding capabilities for code and charts in the real world.

![Image 13: Refer to caption](https://arxiv.org/html/2403.05525v2/x13.png)

Figure 13: Visualization results. DeepSeek-VL possesses extensive knowledge of the real world.

![Image 14: Refer to caption](https://arxiv.org/html/2403.05525v2/x14.png)

Figure 14: Visualization results. DeepSeek-VL is capable of accurately reading the contents of real-world tables.

