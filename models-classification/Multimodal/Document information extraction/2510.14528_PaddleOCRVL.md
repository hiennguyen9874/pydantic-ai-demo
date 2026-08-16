Title: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model

URL Source: https://arxiv.org/html/2510.14528

Published Time: Wed, 26 Nov 2025 01:23:21 GMT

Markdown Content:
PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model
===============

1.   [1 Introduction](https://arxiv.org/html/2510.14528v4#S1 "In PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
2.   [2 PaddleOCR-VL](https://arxiv.org/html/2510.14528v4#S2 "In PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
    1.   [2.1 Architecture](https://arxiv.org/html/2510.14528v4#S2.SS1 "In 2 PaddleOCR-VL ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
        1.   [2.1.1 Layout Analysis](https://arxiv.org/html/2510.14528v4#S2.SS1.SSS1 "In 2.1 Architecture ‣ 2 PaddleOCR-VL ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
        2.   [2.1.2 Element-level Recognition](https://arxiv.org/html/2510.14528v4#S2.SS1.SSS2 "In 2.1 Architecture ‣ 2 PaddleOCR-VL ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")

    2.   [2.2 Training Recipe](https://arxiv.org/html/2510.14528v4#S2.SS2 "In 2 PaddleOCR-VL ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
        1.   [2.2.1 Layout Analysis](https://arxiv.org/html/2510.14528v4#S2.SS2.SSS1 "In 2.2 Training Recipe ‣ 2 PaddleOCR-VL ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
        2.   [2.2.2 Element-level Recognition](https://arxiv.org/html/2510.14528v4#S2.SS2.SSS2 "In 2.2 Training Recipe ‣ 2 PaddleOCR-VL ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")

3.   [3 Dataset](https://arxiv.org/html/2510.14528v4#S3 "In PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
    1.   [3.1 Data Curation](https://arxiv.org/html/2510.14528v4#S3.SS1 "In 3 Dataset ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
    2.   [3.2 Automatic Data Annotation](https://arxiv.org/html/2510.14528v4#S3.SS2 "In 3 Dataset ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
    3.   [3.3 Hard Cases Mining](https://arxiv.org/html/2510.14528v4#S3.SS3 "In 3 Dataset ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")

4.   [4 Evaluation](https://arxiv.org/html/2510.14528v4#S4 "In PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
    1.   [4.1 Page-level Evaluation](https://arxiv.org/html/2510.14528v4#S4.SS1 "In 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
        1.   [OmniDocBench v1.5](https://arxiv.org/html/2510.14528v4#S4.SS1.SSS0.Px1 "In 4.1 Page-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
        2.   [OmniDocBench v1.0](https://arxiv.org/html/2510.14528v4#S4.SS1.SSS0.Px2 "In 4.1 Page-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
        3.   [olmOCR-Bench](https://arxiv.org/html/2510.14528v4#S4.SS1.SSS0.Px3 "In 4.1 Page-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")

    2.   [4.2 Element-level Evaluation](https://arxiv.org/html/2510.14528v4#S4.SS2 "In 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
        1.   [4.2.1 Text Recognition](https://arxiv.org/html/2510.14528v4#S4.SS2.SSS1 "In 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
            1.   [OmniDocBench-OCR-block:](https://arxiv.org/html/2510.14528v4#S4.SS2.SSS1.Px1 "In 4.2.1 Text Recognition ‣ 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
            2.   [In-house-OCR:](https://arxiv.org/html/2510.14528v4#S4.SS2.SSS1.Px2 "In 4.2.1 Text Recognition ‣ 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
            3.   [Ocean-OCR-Handwritten:](https://arxiv.org/html/2510.14528v4#S4.SS2.SSS1.Px3 "In 4.2.1 Text Recognition ‣ 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")

        2.   [4.2.2 Table Recognition.](https://arxiv.org/html/2510.14528v4#S4.SS2.SSS2 "In 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
            1.   [OmniDocBench-Table-block:](https://arxiv.org/html/2510.14528v4#S4.SS2.SSS2.Px1 "In 4.2.2 Table Recognition. ‣ 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
            2.   [In-house-Table:](https://arxiv.org/html/2510.14528v4#S4.SS2.SSS2.Px2 "In 4.2.2 Table Recognition. ‣ 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")

        3.   [4.2.3 Formula Recognition.](https://arxiv.org/html/2510.14528v4#S4.SS2.SSS3 "In 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
            1.   [OmniDocBench-Formula-block](https://arxiv.org/html/2510.14528v4#S4.SS2.SSS3.Px1 "In 4.2.3 Formula Recognition. ‣ 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
            2.   [In-house-Formula:](https://arxiv.org/html/2510.14528v4#S4.SS2.SSS3.Px2 "In 4.2.3 Formula Recognition. ‣ 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")

        4.   [4.2.4 Chart Recognition.](https://arxiv.org/html/2510.14528v4#S4.SS2.SSS4 "In 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
            1.   [In-house-Chart:](https://arxiv.org/html/2510.14528v4#S4.SS2.SSS4.Px1 "In 4.2.4 Chart Recognition. ‣ 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")

    3.   [4.3 Inference Performance](https://arxiv.org/html/2510.14528v4#S4.SS3 "In 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")

5.   [5 Conclusion](https://arxiv.org/html/2510.14528v4#S5 "In PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
6.   [A Training Dataset Details](https://arxiv.org/html/2510.14528v4#A1 "In PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
    1.   [A.1 Text](https://arxiv.org/html/2510.14528v4#A1.SS1 "In Appendix A Training Dataset Details ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
    2.   [A.2 Table](https://arxiv.org/html/2510.14528v4#A1.SS2 "In Appendix A Training Dataset Details ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
    3.   [A.3 Formula](https://arxiv.org/html/2510.14528v4#A1.SS3 "In Appendix A Training Dataset Details ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
    4.   [A.4 Chart](https://arxiv.org/html/2510.14528v4#A1.SS4 "In Appendix A Training Dataset Details ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")

7.   [B Supported Languages](https://arxiv.org/html/2510.14528v4#A2 "In PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
8.   [C Inference Performance on Different Hardware Configurations](https://arxiv.org/html/2510.14528v4#A3 "In PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
9.   [D Real-world Samples](https://arxiv.org/html/2510.14528v4#A4 "In PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
    1.   [D.1 Comprehensive Document Parsing](https://arxiv.org/html/2510.14528v4#A4.SS1 "In Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
    2.   [D.2 Layout Detection](https://arxiv.org/html/2510.14528v4#A4.SS2 "In Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
    3.   [D.3 Reading Order](https://arxiv.org/html/2510.14528v4#A4.SS3 "In Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
    4.   [D.4 Text Recognition](https://arxiv.org/html/2510.14528v4#A4.SS4 "In Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
        1.   [D.4.1 Multilingual Text Recognition](https://arxiv.org/html/2510.14528v4#A4.SS4.SSS1 "In D.4 Text Recognition ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
        2.   [D.4.2 Handwriting Text Recognition](https://arxiv.org/html/2510.14528v4#A4.SS4.SSS2 "In D.4 Text Recognition ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
        3.   [D.4.3 Vertical Text Recognition](https://arxiv.org/html/2510.14528v4#A4.SS4.SSS3 "In D.4 Text Recognition ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")

    5.   [D.5 Table Recognition](https://arxiv.org/html/2510.14528v4#A4.SS5 "In Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
    6.   [D.6 Formula Recognition](https://arxiv.org/html/2510.14528v4#A4.SS6 "In Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
    7.   [D.7 Chart Recognition](https://arxiv.org/html/2510.14528v4#A4.SS7 "In Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")

10.   [E Compare with Others](https://arxiv.org/html/2510.14528v4#A5 "In PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
    1.   [E.1 Layout Detection](https://arxiv.org/html/2510.14528v4#A5.SS1 "In Appendix E Compare with Others ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
    2.   [E.2 Text Recognition](https://arxiv.org/html/2510.14528v4#A5.SS2 "In Appendix E Compare with Others ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
        1.   [E.2.1 Multilingual Text Recognition](https://arxiv.org/html/2510.14528v4#A5.SS2.SSS1 "In E.2 Text Recognition ‣ Appendix E Compare with Others ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
        2.   [E.2.2 Handwriting Text Recognition](https://arxiv.org/html/2510.14528v4#A5.SS2.SSS2 "In E.2 Text Recognition ‣ Appendix E Compare with Others ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
        3.   [E.2.3 Vertical Text Recognition](https://arxiv.org/html/2510.14528v4#A5.SS2.SSS3 "In E.2 Text Recognition ‣ Appendix E Compare with Others ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")

    3.   [E.3 Table Recognition](https://arxiv.org/html/2510.14528v4#A5.SS3 "In Appendix E Compare with Others ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
    4.   [E.4 Formula Recognition](https://arxiv.org/html/2510.14528v4#A5.SS4 "In Appendix E Compare with Others ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")
    5.   [E.5 Chart Recognition](https://arxiv.org/html/2510.14528v4#A5.SS5 "In Appendix E Compare with Others ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")

\reportnumber
001

PaddleOCR-VL: Boosting Multilingual Document Parsing 

via a 0.9B Ultra-Compact Vision-Language Model
=====================================================================================================

Cheng Cui Ting Sun  Suyin Liang  Tingquan Gao  Zelun Zhang  Jiaxuan Liu 

Xueqing Wang Changda Zhou  Hongen Liu  Manhui Lin  Yue Zhang  Yubo Zhang 

Handong Zheng Jing Zhang  Jun Zhang  Yi Liu  Dianhai Yu  Yanjun Ma 

PaddlePaddle Team, Baidu Inc.

paddleocr@baidu.com

![Image 1: [Uncaptioned image]](https://arxiv.org/html/x1.png)Source Code: [https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

![Image 2: [Uncaptioned image]](https://arxiv.org/html/x2.png)Models & Online Demo: [https://huggingface.co/PaddlePaddle](https://huggingface.co/PaddlePaddle)

###### Abstract

In this report, we propose PaddleOCR-VL, a SOTA and resource-efficient model tailored for document parsing. Its core component is PaddleOCR-VL-0.9B, a compact yet powerful vision-language model (VLM) that integrates a NaViT-style dynamic resolution visual encoder with the ERNIE-4.5-0.3B language model to enable accurate element recognition. This innovative model efficiently supports 109 languages and excels in recognizing complex elements (e.g., text, tables, formulas, and charts), while maintaining minimal resource consumption. Through comprehensive evaluations on widely used public benchmarks and in-house benchmarks, PaddleOCR-VL achieves SOTA performance in both page-level document parsing and element-level recognition. It significantly outperforms existing solutions, exhibits strong competitiveness against top-tier VLMs, and delivers fast inference speeds. These strengths make it highly suitable for practical deployment in real-world scenarios.

![Image 3: Refer to caption](https://arxiv.org/html/images/paddleocr-vl_metrics.png)

Figure 1:  Performance of PaddleOCR-VL on OmniDocBench v1.0 and v1.5. 

Contents

1 Introduction
--------------

Documents serve as core information carriers, with their complexity and volume growing at an exponential rate, making document parsing an indispensable key technology. The primary goal of document parsing [li2025monkeyocr, niu2025mineru2, feng2025dolphin, liu2025points] is to enable deep structural and semantic understanding of a document’s layout. Specifically, it involves recognizing distinct text blocks and columns, distinguishing formulas, tables, charts, and images, determining the correct reading order, and detecting key elements (e.g., footnotes and image captions); these capabilities collectively lay a solid foundation for efficient information retrieval and data management. Furthermore, advanced document parsing enables large language models (LLMs) [ernie2025technicalreport, yang2025qwen3, achiam2023gpt], especially when combined with Retrieval-Augmented Generation (RAG) [lewis2020retrieval], to access high-quality knowledge and enhance their practical applications.

The inherent complexity of modern documents presents unique challenges: they often combine dense text, complex tables or chart, mathematical expressions, multiple languages and handwritten texts, with diverse layout structures. Recent research [li2025monkeyocr, wang2024mineru, cui2025paddleocr, Docling_Team_Docling, poznanski2025olmocr] in the field of document parsing primarily following two technological approaches. The first approach [wang2024mineru, cui2025paddleocr] employs pipeline methodologies based on specialized, modular expert models. Although these methods offer strong performance, they are increasingly hindered by integration complexity, cumulative error propagation, and inherent limitations when handling highly complex documents. Secondly, end-to-end approaches [poznanski2025olmocr, nassar2025smoldocling, MinerU2] leveraging multimodal models aim to simplify the workflow and enable joint optimization. However, these methods often struggle with correct text order and can even generate hallucinations when faced with lengthy or complex layouts, while also incurring substantial computational overhead for long sequence outputs, thereby restricting their practical deployment.

To address these advancements and challenges, we present PaddleOCR-VL, a high-performance, resource-efficient document parsing solution based on a vision-language model. This innovation paves the way for the widespread application of multimodal document parsing, particularly in resource-constrained environments. PaddleOCR-VL combines a robust layout analysis model with a compact yet powerful vision-language model, PaddleOCR-VL-0.9B.

Firstly, PaddleOCR-VL performs layout detection and reading order prediction to obtain the positional coordinates and reading order of elements (text blocks, tables, formulas, and charts). Compared to multimodal methods that rely on grounding and sequence output (e.g., MinerU2.5 [niu2025mineru2], Dolphin [feng2025dolphin]), our method offers faster inference speeds, lower training costs, and easier extensibility for new layout categories. Subsequently, the elements are segmented based on their positions and fed into PaddleOCR-VL-0.9B for recognition. This vision-language model is specifically designed for resource-efficient inference and excels at element recognition within document parsing. By integrating a NaViT-style [dehghani2023patch] dynamic high-resolution visual encoder with the lightweight ERNIE-4.5-0.3B [ernie2025technicalreport] language model, we have significantly enhanced the model’s dense text recognition capabilities and decoding efficiency.

To train a powerful multimodal model, we have developed a high-quality training data construction pipeline. We collected over 30 million training samples through public data acquisition and data synthesis. We meticulously designed prompt engineering to guide the automatic labeling by general large models, based on the recognition results of expert models. Simultaneously, We performed data cleaning to remove low-quality or inconsistent annotations, such as those caused by model hallucinations. We designed an evaluation engine, which is an assessment collection that categorizes each element into more detailed categories. Through this automated evaluation, we can analyze the current model’s training performance across different types. This allows us to conduct targeted hard sample mining based on element types and to construct similar challenging examples through data synthesis. Finally, we incorporated manual annotation for a small number of corner cases to complete the construction of the training data.

Comprehensive benchmarking on the public benchmarks, including OmniDocBench v1.0, v1.5 [ouyang2025omnidocbench] and olmOCR-Bench [poznanski2025olmocr], and in-house ones demonstrate that PaddleOCR-VL achieves SOTA performance in document parsing task, significantly outperforming existing pipeline-based solutions and exhibiting strong competitiveness against leading vision-language models (VLMs). Moreover, PaddleOCR-VL is optimized for efficiency, delivering substantially lower latency and higher throughput than competing approaches.

PaddleOCR-VL actively addresses current challenges in document processing with a high-performance, resource-efficient multimodal document parsing solution. Its key contributions include:

*   •Compact yet Powerful VLM Architecture: We present a novel vision-language model that is specifically designed for resource-efficient inference, achieving outstanding performance in element recognition. By integrating a NaViT-style dynamic high-resolution visual encoder with the lightweight ERNIE-4.5-0.3B language model, we significantly enhance the model’s recognition capabilities and decoding efficiency. This integration maintains high accuracy while reducing computational demands, making it well-suited for efficient and practical document processing applications. 
*   •High-quality Data Construction Methodology: We propose a systematic and comprehensive methodology for constructing high-quality datasets, providing a solid train data foundation for efficient and robust document parsing. This methodology not only enables us to construct high-quality data on demand, but also provides a new perspective on the automated generation of high-quality data. 
*   •SOTA Performance Document Parsing: PaddleOCR-VL achieves state-of-the-art performance in document parsing task. It excels in recognizing complex document elements, such as text, tables, formulas, and charts, making it suitable for a wide range of challenging content types, including handwritten text and historical documents. Supporting 109 languages, including major global languages and those with diverse scripts like Russian, Arabic, and Hindi, PaddleOCR-VL is highly applicable to multilingual and globalized document processing scenarios. 

2 PaddleOCR-VL
--------------

### 2.1 Architecture

PaddleOCR-VL decomposes the complex task of document parsing into a two stages, as illustrated in Figure [2](https://arxiv.org/html/2510.14528v4#S2.F2 "Figure 2 ‣ 2.1 Architecture ‣ 2 PaddleOCR-VL ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model"). The first stage, PP-DocLayoutV2, is responsible for layout analysis, where it localizes semantic regions and predicts their reading order. Subsequently, the second stage, PaddleOCR-VL-0.9B, leverages these layout predictions to perform fine-grained recognition of diverse content, including text, tables, formulas, and charts. Finally, a lightweight post-processing module aggregates the outputs from both stages and formats the final document into structured Markdown and JSON.

![Image 4: Refer to caption](https://arxiv.org/html/images/PaddleOCR-VL.png)

Figure 2:  The overview of PaddleOCR-VL. 

#### 2.1.1 Layout Analysis

Considering that end-to-end approaches based on VLM rely on long-sequence autoregressive processes, which result in high latency and memory consumption, and increase the risk of unstable layout analysis and hallucinations—problems that are particularly pronounced in multi-column or mixed text–graphic layouts—we employ a dedicated lightweight model for layout analysis, focusing specifically on element detection, classification, and reading order prediction.

Specifically, we decouple the layout analysis process by introducing an independent model, PP-DocLayoutV2, dedicated solely to this task. PP-DocLayoutV2 consists of an object detection model (RT-DETR [zhao2024detrs]) for elements localization and classification, as well as a lightweight pointer network [hou2024relation] with six transformer layers to accurately predict the reading order of layout elements.

This separation enables us to fully leverage the advanced capabilities of the vision model, which typically requires lower input image resolution, and contains significantly fewer parameters. As a result, it achieves stable and accurate layout analysis, without the instability issues that may arise in end-to-end approaches.

![Image 5: Refer to caption](https://arxiv.org/html/images/PP-DocLayoutV2.png)

Figure 3:  Architecture of layout analysis model. 

Architecturally, PP‑DocLayoutV2 is composed of two sequentially connected networks, as shown in Figure [3](https://arxiv.org/html/2510.14528v4#S2.F3 "Figure 3 ‣ 2.1.1 Layout Analysis ‣ 2.1 Architecture ‣ 2 PaddleOCR-VL ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model"). The first is an RT‑DETR-based [zhao2024detrs] detection model that performs layout element detection and classification. The detected bounding boxes and class labels are then passed to a subsequent pointer network, which is responsible for ordering these layout elements.

Specifically, we first apply per-class thresholds to select foreground proposals for the ordering network. The selected proposals are embedded using absolute 2D positional encodings and class label embeddings. Additionally, the encoder attention incorporates a geometric bias mechanism from Relation‑DETR [hou2024relation] to explicitly model pairwise geometric relationships among elements. The pairwise relation head linearly projects element representations into query and key vectors, then computes bilinear similarities to produce pairwise logits, resulting in an N×N N\times N matrix that represents the relative order between each pair of elements. Finally, a deterministic win‑accumulation decoding algorithm recovers a topologically consistent reading order for the detected layout elements.

In comparison to other specialized models, such as LayoutReader [wang2021layoutreader], our model achieves higher performance with fewer parameters by efficiently extending RT-DETR [zhao2024detrs] with a pointer network.

#### 2.1.2 Element-level Recognition

We systematically explore architecture configurations optimized for high accuracy and low computational overhead, and propose the PaddleOCR-VL-0.9B as shown in Figure [4](https://arxiv.org/html/2510.14528v4#S2.F4 "Figure 4 ‣ 2.1.2 Element-level Recognition ‣ 2.1 Architecture ‣ 2 PaddleOCR-VL ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model").

![Image 6: Refer to caption](https://arxiv.org/html/images/PaddleOCR-VL-0.9B.png)

Figure 4:  Architecture of PaddleOCR-VL-0.9B. 

We adopted an architectural style inspired by LLaVA [liu2023visual], integrating a pre-trained vision encoder with a dynamic resolution preprocessor, a randomly initialized 2-layer MLP projector, and a pre-trained large language model. Our architecture achieves a balance the scale of vision and language models to optimize performance in multi-elements recognition tasks.

Compared to earlier document parsing models based on fixed-resolution or tiling-based approaches [liu2025points, MinerU2, wei2024general], our approach utilizes native dynamic high-resolution preprocessing. For the vision encoder, we employed a NaViT-style [dehghani2023patch] encoder initialized from Keye-VL’s [team2025kwai] vision model, which support native-resolution inputs. This design enables the vision-language model to handle images of arbitrary resolution without distortion, yielding fewer hallucinations and stronger performance on text-intensive tasks.

The projector is a randomly initialized 2-layer MLP with GELU [hendrycks2016gaussian] activation, incorporating a merge size of 2 to efficiently bridge visual features from the encoder to the language model’s embedding space.

In auto-regressive language models, the entire sequence is generated by predicting one token at a time. This approach means that the size of the decoder is directly linked to the overall inference latency, so a smaller model will decode faster. With this in mind, we use the ERNIE-4.5-0.3B [ernie2025technicalreport] model, an open-source language model that balances a relatively small number of parameters with strong inference efficiency. In our implementation, we further enhance positional representation by incorporating a 3D-RoPE[bai2025qwen2].

The combination of NaViT [dehghani2023patch] with ERNIE-4.5-0.3B [ernie2025technicalreport] has led to significant performance improvements in documents parsing, achieving minimal memory usage and faster inference speed.

### 2.2 Training Recipe

The following sections introduce the training details of these two modules: PP-DocLayoutV2 for layout analysis and PaddleOCR-VL-0.9B for element recognition.

#### 2.2.1 Layout Analysis

We employ the PP-DocLayoutV2 model to perform layout element localization, classification, and reading order prediction. PP-DocLayoutV2 extends RT-DETR [zhao2024detrs] by incorporating an additional pointer network [hou2024relation], which is responsible for predicting the reading order of detected elements. The training process adopts a two-stage strategy: we first train the core RT-DETR [zhao2024detrs] model for layout detection and classification. Afterward, we freeze its parameters and independently train the pointer network for reading order prediction.

For the first stage, we follow the training strategy of RT-DETR [zhao2024detrs]. Specifically, we initialize the model with PP-DocLayout_Plus-L [sun2025pp] pretrained weights and train it for 100 epochs on our self-constructed dataset comprising over 20,000 high-quality samples.

For the second stage, specifically, the model outputs a matrix representing the pairwise ordering relationships between any two elements, and the Generalized Cross Entropy Loss [zhang2018generalized] is computed with respect to the ground truth labels, as this loss function demonstrates increased robustness in scenarios where pre-annotated data are mixed into the dataset. We utilize a constant learning rate 2e-4 and the AdamW optimizer to train 200 epochs.

#### 2.2.2 Element-level Recognition

As described in Section [2.1.2](https://arxiv.org/html/2510.14528v4#S2.SS1.SSS2 "2.1.2 Element-level Recognition ‣ 2.1 Architecture ‣ 2 PaddleOCR-VL ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model"), PaddleOCR-VL-0.9B consists of three modules: a vision encoder, a projector, and a language model. We adopt a post-adaptation strategy using pre-trained models. Specifically, the vision model is initialized with Keye-VL’s weights, and the language model is initialized with ERNIE-4.5-0.3B’s weights. The model is trained based on the ERNIEKit [ERNIEkit] repository and the training methodology for our VLM is divided into two stages, as outlined in Table [1](https://arxiv.org/html/2510.14528v4#S2.T1 "Table 1 ‣ 2.2.2 Element-level Recognition ‣ 2.2 Training Recipe ‣ 2 PaddleOCR-VL ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model").

Stages Stage 1 Stage 2
Training Samples 29M 2.7M
Max Resolution 1280 ×\times 28 ×\times 28 2048 ×\times 28 ×\times 28
Sequence length 16384 16384
Trainable components All All
Batch sizes 128 128
Data Augmentation Yes Yes
Maximum LR 5×10−5 5\times 10^{-5}5×10−6 5\times 10^{-6}
Minimum LR 5×10−6 5\times 10^{-6}5×10−7 5\times 10^{-7}
Epoch 1 2

Table 1: Training settings in stage 1 and stage 2.

Stage 1: The initial stage focuses on pre-training alignment, where the model learns to associate visual information from images with corresponding textual representations. This crucial step is performed on a massive dataset comprising 29 million high-quality image-text pairs. During this phase, which runs for one epoch, the model is trained to establish a coherent understanding between diverse visual inputs and their semantic textual content. The training utilizes a batch size of 128, a sequence length of 16384, and supports a maximum image resolution of 1280×\times 28×\times 28, with data augmentation enabled to improve robustness. For optimization, the learning rate is scheduled between a maximum of 5×10−5 5\times 10^{-5} and a minimum of 5×10−6 5\times 10^{-6}. The primary objective is to align the feature spaces of the vision encoder and the language model, enabling them to jointly process multimodal information effectively. This large-scale pre-training allows the model to capture intricate visual patterns, common textual structures, and their interdependencies across a vast range of contexts, laying a strong foundation for subsequent specialized tasks.

Stage 2: Following pre-training, the model undergoes instruction fine-tuning to adapt its general multimodal understanding to specific downstream elements recognition tasks. This stage utilizes a meticulously curated dataset of 2.7 million samples, which is intentionally designed to be highly rich and diverse in its distribution. The training is conducted over two epochs, maintaining the 128 batch size and 16384 sequence length, but increasing the maximum resolution to 2048×\times 28×\times 28 to handle more detailed inputs. A finer learning rate is adopted, with the maximum and minimum values set to 5×10−6 5\times 10^{-6} and 5×10−7 5\times 10^{-7}, to carefully adjust the model on specialized data. The richness of this dataset encompasses a wide variety of document types, languages, writing systems, and visual complexities pertinent to real-world scenarios. During this fine-tuning phase, the model is trained with explicit instructions for four types of tasks:

1.   1.OCR: This task fine-tunes the model to accurately identify and extract textual content from images, encompassing individual characters, words, text lines, text blocks and simple layout structure of page-level texts. 
2.   2.Table Recognition: The model learns to parse tabular structures within documents. This involves accurately extracting cell contents, identifying rows and columns, and recognize the logical relationships between different table elements, ultimately generating structured representations based on OTSL [lysak2023optimized] format. 
3.   3.Formula Recognition: This instruction focuses on enabling the model to recognize and interpret mathematical and scientific formulas. It aims to convert their visual representation into a structured L a T e X format and distinguishes between inline \(…\) and display \[…\] equations. 
4.   4.Chart Recognition: This task trains the model to recognition information from various types of charts, such as bar charts, line graphs, and pie charts and convert Markdown format tables. 

3 Dataset
---------

To build our high-quality and diverse training dataset, we propose a systematic methodology for constructing such datasets. As illustrated in Figure [5](https://arxiv.org/html/2510.14528v4#S3.F5 "Figure 5 ‣ 3 Dataset ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model"), we gather a diverse set of data from multiple sources to ensure comprehensive coverage. High-quality labels are then generated through automated annotation using large models, which guarantees precision and consistency. Additionally, we refine the training data by integrating challenging examples, which enhances the model’s performance and robustness. Each of these crucial steps is detailed in the following sections.

![Image 7: Refer to caption](https://arxiv.org/html/images/data.png)

Figure 5:  The construction process of training data for PaddleOCR-VL-0.9B. 

### 3.1 Data Curation

To ensure the breadth and diversity of the dataset, data is collected from four main sources: open-source dataset, synthesizing dataset, network accessible dataset, and in-house dataset.

1.   1.Open Source Dataset: As the foundation of our dataset, we systematically aggregated and curated a wide array of established public datasets. For textual content, we sourced data from the canonical dataset CASIA-HWDB [liu2011casia]. Our mathematical expression data is derived from UniMER-1M [unimernet] and MathWriting [MathWriting]. To ensure comprehensive coverage of data visualizations, we incorporated a rich spectrum of chart and graph datasets, including ChartQA [chartqa], PlotQA [plotqa], Chart2Text [chart2text], DVQA [dvqa], Unichart [masry2023unichart], Beagle [beagle], ChartINFO [chartinfo], visText [tang2023vistext], and ExcelChart [excelchart]. Each of these sources underwent an initial filtering and cleaning protocol to rectify or discard noisy and low-quality annotations. 
2.   2.Data Synthesizing Dataset: Due to the naturally imbalanced distribution of public data, we employed a data synthesizing strategy to produce large volumes of missing data types at low cost, providing our proposed model with the unbiased document parsing performance. 
3.   3.Network Accessible Dataset: To improve model generalization and robustness against the complexities of unstructured real-world documents, we amassed an extensive corpus of publicly accessible data harvested from the Internet. This public collection was deliberately curated to encompass a rich spectrum of document types and visual styles. It includes academic papers, newspapers, formal scientific journal articles, scanned handwritten documents, diverse examination papers, and slides, etc. The integration of these varied sources proved instrumental in significantly broadening the stylistic, structural, and domain diversity of our training data, thereby mitigating the risk of overfitting to clean, canonical datasets. 
4.   4.In-house Dataset: Through years of research in the field of OCR, we have accumulated extensive datasets with diverse data types across all tasks of document parsing. We incorporate all in-house datasets into training with precisely controlled proportions, which have become unnecessary factors that enable our models to achieve outstanding performance. 

### 3.2 Automatic Data Annotation

After acquiring the raw data, we utilize an automatic data annotations process for large-scale labeling. Initially, we employ the expert model, PP-StructureV3, to conduct preliminary processing on the data, generating pseudo labels that may contain some inaccuracies. Subsequently, through prompt engineering, we create prompts that include the original images and their associated pseudo labels, which are then submitted to more advanced multimodal large language models, ERNIE-4.5-VL [ernie2025technicalreport] and Qwen2.5VL [bai2025qwen2]. These sophisticated models refine and enhance the initial results by analyzing the image content, resulting in improved labels. Finally, to ensure the quality of the labels, the system performs a hallucination filtering step, which eliminates any potentially incorrect content generated by the large models, thereby producing reliable and high-quality labels.

### 3.3 Hard Cases Mining

To overcome performance bottlenecks in specific complex scenarios, we propose a hard case mining process for targeted performance improvement. We firstly develop a eval engine for various types. We created substantial evaluation data with precisely labeled data obtained through manual annotation. Theses evaluation datasets are categorized into several types: text data includes 23 categories such as Chinese, English, printed, handwritten, Japanese, Latin, and emojis; table data includes 20 categories such as limited tables, unlimited tables, handwritten tables, checklists, invoices, and rotated tables; formula data includes 4 categories such as Chinese and English formulas, handwritten and printed, simple, and complex; chart data includes 11 categories such as Chinese and English charts, line charts, and bar charts, sourced from diverse origins to cover different document. By inference on this evaluation set and using corresponding professional metrics (e.g., EditDist for Text, TEDS [teds] for Tables, RMS-F1 [deplot] for Charts, and BLEU [bleu_score] for Formulas), we can accurately identify hard cases where the model performs poorly. Finally, for these identified weaknesses, the system utilizes a rich set of resources (such as Font Library, CSS Library, Corpus) and rendering tools (like XeLaTeX and web browsers) to synthetically generate a large volume of new, high-quality hard cases.

4 Evaluation
------------

To thoroughly assess the effectiveness of PaddleOCR-VL, we compared it against leading general vision language models and specialized document parsing models across multiple public benchmarks and in-house benchmarks. We conducted comprehensive performance comparisons in two aspects: page-level document parsing and element-level recognition, which are detailed in Sections [4.1](https://arxiv.org/html/2510.14528v4#S4.SS1 "4.1 Page-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") and [4.2](https://arxiv.org/html/2510.14528v4#S4.SS2 "4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model"). Page-level involves analyzing entire pages of a document to parsing their overall content, structure and layout, while element-level is dedicated exclusively to assessing the recognition of specific elements, such as text, tables, formulas, and charts, within the document.

### 4.1 Page-level Evaluation

This section details the evaluation of end-to-end document parsing capabilities using the following three benchmarks, aiming to measure its overall performance in real-world document scenarios.

##### OmniDocBench v1.5

To comprehensively evaluate the document parsing capabilities, we conducted extensive experiments on the OmniDocBench v1.5 [niu2025mineru2] benchmark. It is an expansion of version v1.0, adding 374 new documents for a total of 1,355 document pages. It features a more balanced distribution of data in both Chinese and English, as well as a richer inclusion of formulas and other elements. The evaluation method has been updated, with formulas assessed using the CDM method. The overall metric is a weighted combination of the metrics for text, formulas, and tables.

Table [2](https://arxiv.org/html/2510.14528v4#S4.T2 "Table 2 ‣ OmniDocBench v1.5 ‣ 4.1 Page-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") demonstrate that PaddleOCR-VL achieves SOTA performance, outperforming existing pipeline tools, general VLMs, and other specialized document parsing models across all key metrics. Specifically, our model achieves a top-ranking overall score of 92.86, surpassing the next best model, MinerU2.5-1.2B (90.67). Moreover, our model establishes new SOTA results in the sub-tasks, including the lowest Text-Edit distance [lcvenshtcin1966binary] of 0.035, the highest Formula-CDM score of 91.22, the leading scores of 90.89 and 94.76 in Table-TEDS and Table-TEDS-S, and the best readering ordering scores of 0.043, respectively. These results underscore its superior accuracy in text recognition, formula recognition, and complex table structure analysis.

Model Type Methods Parameters Overall↑\uparrow Text Edit↓\downarrow Formula CDM↑\uparrow Table TEDS↑\uparrow Table TEDS-S↑\uparrow Reading Order Edit↓\downarrow
Pipeline Tools Marker-1.8.2 [vik2024marker]-71.30 0.206 76.66 57.88 71.17 0.250
Mineru2-pipeline [MinerU2]-75.51 0.209 76.55 70.90 79.11 0.225
PP-StructureV3 [cui2025paddleocr]-86.73 0.073 85.79 81.68 89.48 0.073
General VLMs GPT-4o [achiam2023gpt]-75.02 0.217 79.70 67.07 76.09 0.148
InternVL3-76B [zhu2025internvl3]76B 80.33 0.131 83.42 70.64 77.74 0.113
InternVL3.5-241B [wang2025internvl35]241B 82.67 0.142 87.23 75.00 81.28 0.125
Qwen2.5-VL-72B [bai2025qwen2]72B 87.02 0.094 88.27 82.15 86.22 0.102
Gemini-2.5 Pro [gemini25]-88.03 0.075 85.82 85.71 90.29 0.097
Specialized VLMs Dolphin [feng2025dolphin]322M 74.67 0.125 67.85 68.70 77.77 0.124
OCRFlux-3B [OCRFlux2025]3B 74.82 0.193 68.03 75.75 80.23 0.202
Mistral OCR [mistral]-78.83 0.164 82.84 70.03 78.04 0.144
POINTS-Reader [liu2025points]3B 80.98 0.134 79.20 77.13 81.66 0.145
olmOCR-7B [poznanski2025olmocr]7B 81.79 0.096 86.04 68.92 74.77 0.121
MinerU2-VLM [MinerU2]0.9B 85.56 0.078 80.95 83.54 87.66 0.086
Nanonets-OCR-s [Nanonets-OCR-S]3B 85.59 0.093 85.90 80.14 85.57 0.108
MonkeyOCR-pro-1.2B [li2025monkeyocr]1.9B 86.96 0.084 85.02 84.24 89.02 0.130
MonkeyOCR-3B [li2025monkeyocr]3.7B 87.13 0.075 87.45 81.39 85.92 0.129
dots.ocr [dotsocr]3B 88.41 0.048 83.22 86.78 90.62 0.053
MonkeyOCR-pro-3B [li2025monkeyocr]3.7B 88.85 0.075 87.25 86.78 90.63 0.128
MinerU2.5 [niu2025mineru2]1.2B 90.67 0.047 88.46 88.22 92.38 0.044
PaddleOCR-VL 0.9B 92.86 0.035 91.22 90.89 94.76 0.043

Table 2: Comprehensive evaluation of document parsing on OmniDocBench v1.5. Results are reported by OmniDocBench [ouyang2025omnidocbench] unless Ours.

##### OmniDocBench v1.0

A publicly available benchmark dataset specifically is designed to evaluate real-world document parsing capabilities. It comprises 981 PDF pages, spanning 9 distinct document types, 4 layout styles, and 3 language categories.

Based on the experimental results presented in Table [3](https://arxiv.org/html/2510.14528v4#S4.T3 "Table 3 ‣ OmniDocBench v1.0 ‣ 4.1 Page-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model"), PaddleOCR-VL demonstrates superior performance with an average overall edit distance of 0.115, demonstrating its superior capability in document parsing. The model excels in formula edit distance (0.241 EN, 0.316 ZH), and achieves the SOTA performance (0.062) and a comparable SOTA performance (0.041) for Chinese and English text edit distance respectively, showcasing its accuracy in handling textual and formulaic data. Although the model exhibits slightly lower performance in the English Table TEDS (88.0), this can be largely attributed to typo-related annotation errors in OmniDocBench v1.0. Nevertheless, it demonstrates a clear advantage in the Chinese Table TEDS (92.14). Regarding the reading order edit distance, the model achieves the best performance in Chinese (0.063) and a comparable SOTA result in English (0.045), emphasizing its capability to maintain structural integrity and logical document flow.

Method Type Methods AvgOverall Edit↓\downarrow Overall Edit↓\downarrow Text Edit↓\downarrow Formula Edit↓\downarrow Table TEDS↑\uparrow Table Edit↓\downarrow Reading Order Edit↓\downarrow
EN ZH EN ZH EN ZH EN ZH EN ZH EN ZH
Pipeline Tools Docling-2.14.0 [Docling_Team_Docling]0.749 0.589 0.909 0.416 0.987 0.999 1 61.3 25.0 0.627 0.810 0.313 0.837
OpenParse-0.7.0 [open-parse]0.730 0.646 0.814 0.681 0.974 0.996 1 64.8 27.5 0.284 0.639 0.595 0.641
Unstructured-0.17.2 [unstructured]0.651 0.586 0.716 0.198 0.481 0.999 1 0 0.1 1 0.998 0.145 0.387
Pix2Text-1.1.2.3 [Pix2Text]0.424 0.320 0.528 0.138 0.356 0.276 0.611 73.6 66.2 0.584 0.645 0.281 0.499
Marker-1.7.1 [vik2024marker]0.397 0.296 0.497 0.085 0.293 0.374 0.688 67.6 54.0 0.609 0.678 0.116 0.329
Mathpix [mathpix]0.278 0.191 0.364 0.105 0.381 0.306 0.454 77.0 67.1 0.243 0.320 0.108 0.304
MinerU-pipeline [wang2024mineru]0.203 0.162 0.244 0.072 0.111 0.313 0.581 77.4 79.5 0.166 0.150 0.097 0.136
PP-StructureV3 [cui2025paddleocr]0.176 0.145 0.206 0.058 0.088 0.295 0.535 77.2 83.9 0.159 0.109 0.069 0.091
General VLMs InternVL2-76B [InternVL]0.442 0.440 0.443 0.353 0.290 0.543 0.701 63.0 60.2 0.547 0.555 0.317 0.228
GPT-4o [achiam2023gpt]0.316 0.233 0.399 0.144 0.409 0.425 0.606 72.0 62.9 0.234 0.329 0.128 0.251
InternVL3-78B [zhu2025internvl3]0.257 0.218 0.296 0.117 0.210 0.380 0.533 69.0 73.9 0.279 0.282 0.095 0.161
Qwen2.5-VL-72B [bai2025qwen2]0.238 0.214 0.261 0.092 0.180 0.315 0.434 81.4 83.0 0.341 0.262 0.106 0.168
Gemini2.5-Pro [gemini25]0.180 0.148 0.212 0.055 0.168 0.356 0.439 85.8 86.4 0.130 0.119 0.049 0.121
Specialized VLMs Nougat [blecher2023nougat]0.713 0.452 0.973 0.365 0.998 0.488 0.941 39.9 0.0 0.572 1 0.382 0.954
SmolDocling-256M [nassar2025smoldocling]0.655 0.493 0.816 0.262 0.838 0.753 0.997 44.9 16.5 0.729 0.907 0.227 0.522
olmOCR-7B [poznanski2025olmocr]0.398 0.326 0.469 0.097 0.293 0.455 0.655 68.1 61.3 0.608 0.652 0.145 0.277
GOT [wei2024general]0.349 0.287 0.411 0.189 0.315 0.360 0.528 53.2 47.2 0.459 0.520 0.141 0.280
OCRFlux-3B [OCRFlux2025]0.294 0.238 0.349 0.112 0.256 0.447 0.716 69.0 80.0 0.269 0.162 0.126 0.263
Nanonets-OCR-s [Nanonets-OCR-S]0.289 0.283 0.295 0.134 0.231 0.518 0.546 76.8 79.4 0.343 0.201 0.135 0.200
Dolphin [feng2025dolphin]0.259 0.205 0.313 0.092 0.204 0.447 0.606 76.1 66.9 0.193 0.282 0.088 0.160
MinerU2-VLM [MinerU2]0.186 0.133 0.238 0.045 0.115 0.273 0.506 82.1 83.4 0.150 0.209 0.066 0.122
MonkeyOCR-pro-1.2B [li2025monkeyocr]0.184 0.146 0.221 0.068 0.118 0.272 0.452 81.3 85.5 0.149 0.134 0.093 0.179
MonkeyOCR-pro-3B [li2025monkeyocr]0.172 0.138 0.206 0.067 0.107 0.246 0.421 81.5 87.5 0.139 0.111 0.100 0.185
dots.ocr [dotsocr]0.143 0.125 0.160 0.032 0.066 0.329 0.416 88.6 89.0 0.099 0.092 0.040 0.067
MinerU2.5 [niu2025mineru2]0.143 0.111 0.174 0.050 0.074 0.258 0.473 88.3 89.2 0.089 0.083 0.045 0.068
PaddleOCR-VL 0.115 0.105 0.126 0.041 0.062 0.241 0.316 88.0 92.1 0.093 0.062 0.045 0.063

Table 3: Comprehensive evaluation of document parsing on OmniDocBench v1.0. Results are reported by OmniDocBench [ouyang2025omnidocbench] unless MinerU2.5 and Ours.

##### olmOCR-Bench

olmOCR-Bench [poznanski2025olmocr] includes 1,402 PDF documents and 7,010 test cases, addressing diverse document types and extraction challenges. It offers a detailed evaluation framework for PDF content extraction by assessing tools and models through simple, clear, and machine-verifiable unit tests. This approach avoids biased evaluations and soft metric comparisons, allowing for the detection of subtle but significant extraction errors.

Table [4](https://arxiv.org/html/2510.14528v4#S4.T4 "Table 4 ‣ olmOCR-Bench ‣ 4.1 Page-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") highlights the outstanding performance of PaddleOCR-VL in the olmOCR-Bench evaluation, achieving the highest overall score of 80.0 ± 1.0. It excels in various categories, leading in ArXiv (85.7), Headers and Footers (97.0) and securing second place in Multi-column text (79.9), Long Tiny Text (85.7). These results highlight the proposed model’s capability to effectively manage diverse document types, reinforcing its status as a top solution in document parsing and its reliability in complex OCR tasks.

Methods Unit Test Pass Rate ↑\uparrow
Overall ArXiv Old Scans Math Tables Old Scans Headers and Footers Multi column Long Tiny Text Base
GOT [wei2024general]48.3 ± 1.1 52.7 52.0 0.2 22.1 93.6 42.0 29.9 94.0
Gemini Flash 2 (No Anchor) [gemini25]57.8 ± 1.1 32.1 56.3 61.4 27.8 48.0 58.7 84.4 94.0
MinerU-pipeline [wang2024mineru]61.5 ± 1.1 75.4 47.4 60.9 17.3 96.6 59.0 39.1 96.6
Gemini Flash 2 (Anchored) [gemini25]63.8 ± 1.2 54.5 56.1 72.1 34.2 64.7 61.5 71.5 95.6
Nanonets-OCR-s [Nanonets-OCR-S]64.5 ± 1.1 67.0 68.6 77.7 39.5 40.7 69.9 53.4 99.3
Qwen2.5-VL-7B (No Anchor) [bai2025qwen2]65.5 ± 1.2 63.1 65.7 67.3 38.6 73.6 68.3 49.1 98.3
GPT-4o (No Anchor) [achiam2023gpt]68.9 ± 1.1 51.5 75.5 69.1 40.9 94.2 68.9 54.1 96.7
GPT-4o (Anchored) [achiam2023gpt]69.9 ± 1.1 53.5 74.5 70.0 40.7 93.8 69.3 60.6 96.8
Marker-1.8.2 [vik2024marker]70.1 ± 1.1 76.0 57.9 57.6 27.8 84.9 72.9 84.6 99.1
olmOCR v0.1.75 (No Anchor) [poznanski2025olmocr]74.7 ± 1.1 71.5 71.4 71.4 42.8 94.1 77.7 71.0 97.8
olmOCR v0.1.75 (Anchored) [poznanski2025olmocr]75.5 ± 1.0 74.9 71.2 71.0 42.2 94.5 78.3 73.3 98.3
MonkeyOCR-pro-3B [li2025monkeyocr]75.8 ± 1.0 83.8 68.8 74.6 36.1 91.2 76.6 80.1 95.3
MinerU2.5 [niu2025mineru2]77.5 ± 1.0 81.1 74.0 85.1 33.8 96.3 65.5 89.8 94.4
dots.ocr [dotsocr]79.1 ± 1.0 82.1 64.2 88.3 40.9 94.1 82.4 81.2 99.5
PaddleOCR-VL 80.0 ± 1.0 85.7 71.0 84.1 37.8 97.0 79.9 85.7 98.5

Table 4: Comprehensive evaluation of document parsing on olmOCR-Bench. Results are reported by olmOCR-Bench [poznanski2025olmocr] unless MinerU2.5 and Ours.

### 4.2 Element-level Evaluation

This section centers on evaluating the element-level capabilities of PaddleOCR VL 0.9B. We thoroughly assessed four tasks: text, tables, formulas, and charts using both public competition data and in-house data.

#### 4.2.1 Text Recognition

For text recognition, we utilize three benchmarks to validate the effectiveness of models based on the edit distance metric.

##### OmniDocBench-OCR-block:

From the 1355 images of OmniDocBench v1.5, we extracted all text-related sub-images based on layout detection labels, removing any with null annotations. This process resulted in a total of 17,148 block-level images. This evaluation set is named OmniDocBench-OCR-block, with the ground truth still sourced from OmniDocBench. This evaluation set can more accurately assess the model’s text recognition performance on without being affected by layout detection. We use the average normalized edit distance for evaluation.

In Table [5](https://arxiv.org/html/2510.14528v4#S4.T5 "Table 5 ‣ OmniDocBench-OCR-block: ‣ 4.2.1 Text Recognition ‣ 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model"), we present a comprehensive comparison of performance across various document types using different models. Our model, PaddleOCR-VL, consistently demonstrates superior performance, achieving the lowest error rates in almost all categories. Specifically, PaddleOCR-VL achieves the best results in the PPT2PDF (0.049), Academic Literature (0.021), Book (0.045), Colorful Textbook (0.081), Exam Paper (0.115), Magazine (0.020), Newspaper (0.034), Note (0.081), and Research Report (0.033) categories. These results highlight PaddleOCR-VL’s robust and versatile capability in handling diverse document types, establishing it as the leading method in the OmniDocBench-OCR-block performance evaluation.

Methods Edit Distance ↓\downarrow
PPT2PDF Academic 

Literature Book Colorful 

Textbook Exam 

Paper Magazine Newspaper Note Research 

Report
Qwen2.5-VL-72B [bai2025qwen2]0.054 0.023 0.061 0.084 0.195 0.032 0.056 0.118 0.040
MonkeyOCR-pro-3B [li2025monkeyocr]0.058 0.021 0.064 0.096 0.116 0.023 0.058 0.124 0.052
MinerU2.5 [niu2025mineru2]0.195 0.089 0.111 0.234 0.194 0.147 0.056 0.142 0.094
Dolphin [feng2025dolphin]0.237 0.095 0.135 0.347 0.248 0.233 0.121 0.309 0.213
PaddleOCR-VL 0.049 0.021 0.045 0.081 0.115 0.020 0.034 0.081 0.033

Table 5: Overall Comparison of OmniDocBench-OCR-block Performance.

##### In-house-OCR:

This is our self-built line-level text evaluation dataset which contains 107452 samples with high-quality labels. The dataset includes various text types such as handwritten Chinese, handwritten English, printed Chinese, printed English, traditional Chinese, ancient texts, general scenarios, Pinyin, obscure characters, vertical text, single characters, emojis, and artistic fonts. It also comprises evaluation sets for 109 languages, such as Latin and Japanese.

Table [6](https://arxiv.org/html/2510.14528v4#S4.T6 "Table 6 ‣ In-house-OCR: ‣ 4.2.1 Text Recognition ‣ 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") provides a detailed evaluation of performance across multiple languages and text types. In the Multilingual Metrics (Table [6(a)](https://arxiv.org/html/2510.14528v4#S4.T6.st1 "In Table 6 ‣ In-house-OCR: ‣ 4.2.1 Text Recognition ‣ 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")), the model demonstrates outstanding accuracy with the lowest edit distances in all evaluated scripts: Arabic(0.122), Korean(0.052), Tamil(0.043), Greek(0.135), Thai(0.081), Telugu (0.114), Devanagari (0.097), Cyrillic (0.109), Latin (0.013), and Japanese (0.096), indicating superior capability in handling diverse languages. Similarly, in the Text Type Metrics (Table [6(b)](https://arxiv.org/html/2510.14528v4#S4.T6.st2 "In Table 6 ‣ In-house-OCR: ‣ 4.2.1 Text Recognition ‣ 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")), it excels in various text types, achieving the lowest error rates in categories like Handwritten CN (0.089), Handwritten EN (0.042), Printed CN (0.035), Printed EN (0.016), Traditional Chinese (0.048), Ancient Texts(0.198), General Scene (0.067), Pinyin (0.113), Rare Characters (0.001), Vertical Text (0.005), Single Characters (0.027), Emoji (0.057), and Art Font (0.165). These impressive results underscore the model’s robust performance and versatility, establishing it as the leading OCR solution in this benchmark comparison.

Methods Edit Distance ↓\downarrow
Arabic Korean Tamil Greek Thai Telugu Devanagari Cyrillic Latin Japanese
Qwen2.5-VL-72B [bai2025qwen2]0.405 0.056 0.389 0.165 0.194 0.758 0.164 0.220 0.021 0.181
Dolphin [feng2025dolphin]0.682 0.699 0.912 0.691 0.709 0.832 0.818 0.549 0.037 0.309
MonkeyOCR-pro-3B [li2025monkeyocr]0.601 0.182 0.921 0.449 0.876 0.909 0.896 0.387 0.036 0.262
MinerU2.5 [niu2025mineru2]0.978 0.917 0.957 0.661 0.880 0.937 0.915 0.832 0.063 0.588
PaddleOCR-VL 0.122 0.052 0.043 0.135 0.081 0.011 0.097 0.109 0.013 0.086

(a)Multilingual Metrics.

Methods Edit Distance ↓\downarrow
Hand- 

written 

CN Hand- 

written 

EN Printed 

CN Printed 

EN Trad. 

Chinese Ancient 

Texts General 

Scene Pinyin Rare 

Char.Vertical 

Text Single 

Char.Emoji Art 

Font
Dolphin [feng2025dolphin]0.236 0.145 0.074 0.025 0.095 0.218 0.113 0.183 0.092 0.190 0.202 0.225 0.230
MonkeyOCR-pro-3B [li2025monkeyocr]0.253 0.071 0.048 0.023 0.295 0.529 0.144 0.165 0.063 0.086 0.110 0.184 0.263
Qwen2.5-VL-72B [bai2025qwen2]0.188 0.047 0.037 0.018 0.100 0.387 0.122 0.186 0.034 0.090 0.041 0.134 0.220
MinerU2.5 [niu2025mineru2]0.370 0.088 0.041 0.023 0.232 0.950 0.179 0.256 0.048 0.962 0.097 0.174 0.337
PaddleOCR-VL 0.089 0.042 0.035 0.016 0.048 0.198 0.067 0.113 0.001 0.005 0.027 0.057 0.165

(b)Text Type Metrics.

Table 6: Comparison of In-house-OCR Edit Distance Performance.

##### Ocean-OCR-Handwritten:

This is a line and paragraph levels handwritten evaluation dataset designed for comprehensive handwriting recognition assessment. It contains 400 samples, evenly divided into four subsets of 100 images each. The dataset covers both real and synthetic handwriting in Chinese and English. Real samples are collected from established handwriting datasets such as CASIA-HWDB [liu2011casia], GNHK [lee2021gnhk], and BRUSH [kotani2020generating], while synthetic samples are generated to simulate diverse writing styles, character densities, and layouts. The benchmark aims to provide balanced and fine-grained evaluation for handwritten text recognition across different scripts and writing conditions.

Table [7](https://arxiv.org/html/2510.14528v4#S4.T7 "Table 7 ‣ Ocean-OCR-Handwritten: ‣ 4.2.1 Text Recognition ‣ 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") presents a comparison of OCR performance for handwritten English and Chinese text on the Ocean-OCR-Bench. Our model demonstrates superior performance across all metrics in both languages. For English, it achieves the best edit distance of 0.118 and excels in F1-score, Precision, Recall, BLEU, and METEOR, establishing itself as the leading model. In Chinese, PaddleOCR-VL sets a benchmark with an edit distance of 0.034 and leads in all other metrics, showcasing its outstanding precision and reliability.

Methods Edit Distance ↓\downarrow F1-score ↑\uparrow Precision↑\uparrow Recall↑\uparrow BLEU↑\uparrow METEOR↑\uparrow
EN ZH EN ZH EN ZH EN ZH EN ZH EN ZH
InternVL2.5-4B [InternVL]0.197 0.240 0.661 0.741 0.674 0.754 0.655 0.734 0.406 0.473 0.652 0.687
MiniCPM-V2.6-8B [yao2024minicpm]0.147 0.175 0.727 0.810 0.747 0.811 0.714 0.812 0.443 0.583 0.727 0.774
Qwen2-VL-7B [qwen2]0.127 0.113 0.760 0.881 0.773 0.884 0.754 0.884 0.490 0.666 0.756 0.859
GOT [wei2024general]0.616 0.402 0.283 0.568 0.309 0.618 0.273 0.544 0.151 0.295 0.255 0.492
PaddleOCR [cui2025paddleocr]0.418 0.325 0.237 0.664 0.232 0.646 0.263 0.700 0.069 0.431 0.236 0.648
TextIn 0.358 0.180 0.362 0.840 0.368 0.869 0.362 0.822 0.098 0.567 0.337 0.751
Ocean-OCR [chen2025ocean]0.145 0.106 0.774 0.885 0.780 0.912 0.782 0.862 0.532 0.736 0.772 0.885
MinerU2.5 [niu2025mineru2]0.238 0.356 0.558 0.619 0.547 0.623 0.574 0.622 0.344 0.489 0.553 0.601
PaddleOCR-VL 0.118 0.034 0.750 0.957 0.748 0.959 0.753 0.957 0.551 0.856 0.787 0.936

Table 7: Comparison of performance on English(EN) and Chinese(ZH) OCR for handwritten recognition on Ocean-OCR-Bench. Results are reported by Ocean-OCR [chen2025ocean] unless MinerU2.5 and Ours.

#### 4.2.2 Table Recognition.

For table recognition, we utilize two benchmarks to validate the effectiveness of PaddleOCR-VL-0.9B based on TEDS [teds] and Edit Distance.

##### OmniDocBench-Table-block:

To evaluate the table recognition performance of PaddleOCR-VL, we crop 512 tables from OmniDocBench v1.5 datasets.

As shown in Table [8](https://arxiv.org/html/2510.14528v4#S4.T8 "Table 8 ‣ OmniDocBench-Table-block: ‣ 4.2.2 Table Recognition. ‣ 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model"), our PaddleOCR-VL leads in the OmniDocBench-Table-block benchmark, surpassing all competitors. It achieves a top overall TEDS of 0.9195, reflecting high accuracy in capturing table structure and content. Its structural TEDS of 0.9543 highlights its ability to parse complex structures, while the lowest Overall Edit Distance of 0.0561 indicates minimal recognition errors. These results confirm PaddleOCR-VL’s superior performance and establish it as the benchmark for accurate table recognition.

Methods Overall TEDS↑\uparrow Structural TEDS↑\uparrow Overall Edit Dist↓\downarrow
MinerU2-VLM [MinerU2]0.9002 0.9369 0.0734
Seed1.6 0.9079 0.9489 0.0652
dots.ocr [dotsocr]0.8194 0.8442 0.1508
MinerU2.5 [niu2025mineru2]0.9005 0.9539 0.0693
PaddleOCR-VL 0.9195 0.9543 0.0561

Table 8: Comparison of OmniDocBench-Table-block Performance

##### In-house-Table:

Our self-built evaluation set contains diverse array of table images with comprehensive annotations and type classifications. It includes 20 different table types such as Chinese, English, mixed Chinese-English, and tables with various characteristics like full, partial, or no borders. The collection also covers tables with formulas, dense data, book/manual formats, lists, academic papers, merged cells, as well as low-quality, watermarked, registration forms, statistical forms, research reports, financial reports, images, invoices, and handwritten tables, among others.

Table [9](https://arxiv.org/html/2510.14528v4#S4.T9 "Table 9 ‣ In-house-Table: ‣ 4.2.2 Table Recognition. ‣ 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") provides a comparison of different methods on the In-house-Table task, highlighting their performance across various metrics. We achieves the highest scores in Overall TEDS (0.8699), Structural TEDS (0.9066), Overall Edit Distance (0.9066) and Structural Edit Distance (0.9339). These results underscore PaddleOCR-VL’s effectiveness and reliability in table recognition tasks.

Methods Overall TEDS↑\uparrow Structural TEDS↑\uparrow Overall Edit Dist↑\uparrow Structural Edit Dist↑\uparrow
MinerU2-VLM [MinerU2]0.8286 0.8730 0.8757 0.9088
MonkeyOCR [li2025monkeyocr]0.7396 0.7824 0.8174 0.8537
Nanonets-OCR-s [Nanonets-OCR-S]0.7824 0.8190 0.8377 0.8692
OCRFlux-3B [OCRFlux2025]0.7741 0.8071 0.8238 0.8617
Qwen2.5-VL-3B [bai2025qwen2]0.7398 0.7765 0.8132 0.8701
Qwen2.5-VL-7B [bai2025qwen2]0.7549 0.7926 0.8251 0.8819
Qwen2.5-VL-72B [bai2025qwen2]0.7762 0.8361 0.843 0.8987
dots.ocr [dotsocr]0.7547 0.7914 0.8047 0.8361
MinerU2.5 [niu2025mineru2]0.8469 0.8955 0.8896 0.9239
PaddleOCR-VL 0.8699 0.9066 0.9066 0.9339

Table 9: Comparison of In-house-Table Performance

#### 4.2.3 Formula Recognition.

For formula recognition, we validate the effectiveness our model based on the Character Detection Matching (CDM) [cdm] metric on OmniDocBench-Formula-block and In-house-Formula datasets.

##### OmniDocBench-Formula-block

Using the formula bounding boxes from OmniDocBench v1.5, 1050 formula sub-images were cropped. This step was taken to minimize the influence of layout detection on formula recognition. As shown in Table [10](https://arxiv.org/html/2510.14528v4#S4.T10 "Table 10 ‣ OmniDocBench-Formula-block ‣ 4.2.3 Formula Recognition. ‣ 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model"), the model achieved state-of-the-art CDM score of 0.9453.

Methods Overall CDM ↑\uparrow EN CDM ↑\uparrow ZH CDM ↑\uparrow
dots.ocr [dotsocr]0.4641 0.4868 0.4414
MinerU2-VLM [MinerU2]0.8286 0.9616 0.6956
MonkeyOCR-pro-1.2B [li2025monkeyocr]0.8531 0.9642 0.7419
MonkeyOCR-3B [li2025monkeyocr]0.8621 0.9718 0.7524
Qwen2.5-VL-72B [bai2025qwen2]0.8747 0.9574 0.7920
MinerU2.5 [niu2025mineru2]0.9187 0.9751 0.8623
PaddleOCR-VL 0.9453 0.9677 0.9228

Table 10: Comparison of OmniDocBench v1.5 Formula-block Performance. Due to dots.ocr [dotsocr] easily recognizing cropped formulas as images, the score is relatively low.

##### In-house-Formula:

The self-constructed formula evaluation set contains 34,816 samples, covering common formula recognition scenarios such as academic papers, mathematics books, and primary and secondary school exam papers. Among them, there are 498 Chinese formulas and 34,318 English formulas. As shown in Table [11](https://arxiv.org/html/2510.14528v4#S4.T11 "Table 11 ‣ In-house-Formula: ‣ 4.2.3 Formula Recognition. ‣ 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model"), our model obtains the best performance of 0.9882 CDM score on the In-house-Formula dataset. These results collectively demonstrate the powerful recognition capability of PaddleOCR-VL in real-world complex formula scenarios.

Methods Overall CDM ↑\uparrow EN CDM ↑\uparrow ZH CDM ↑\uparrow
dots.ocr [dotsocr]0.6737 0.8066 0.5408
MinerU2-VLM [MinerU2]0.9237 0.9764 0.8709
MonkeyOCR-pro-1.2B [li2025monkeyocr]0.9537 0.9656 0.9417
MonkeyOCR-3B [li2025monkeyocr]0.9566 0.9761 0.9371
Qwen2.5-VL-72B [bai2025qwen2]0.9412 0.9519 0.9304
MinerU2.5 [niu2025mineru2]0.9770 0.9832 0.9708
PaddleOCR-VL 0.9882 0.9914 0.9849

Table 11: Comparison of In-house-Formula Performance. Due to dots.ocr [dotsocr] easily recognizing cropped formulas as images, the score is relatively low.

#### 4.2.4 Chart Recognition.

For chart recognition, considering the limitations in dataset size, the imbalanced distribution of chart categories, and the poor annotation quality of publicly available test sets, we only utilize a in-house benchmark to validate the effectiveness of PaddleOCR-VL-0.9B based on the RMS-F1 [deplot] score metric. As shown in Table [12](https://arxiv.org/html/2510.14528v4#S4.T12 "Table 12 ‣ In-house-Chart: ‣ 4.2.4 Chart Recognition. ‣ 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model"), the proposed PaddleOCR-VL not only outperforms expert OCR VLMs but also surpasses some 72B-level multimodal language models.

##### In-house-Chart:

This in-house chart recognition evaluation set comprises 1,801 samples, all of which have underwent a rigorous manual review to ensure annotation correctness. The evaluation set is broadly categorized into 11 chart categories, including bar-line hybrid, pie, 100% stacked bar, area, bar, bubble, histogram, line, scatterplot, stacked area, and stacked bar. Of these samples, 851 are in English and 950 are in Chinese. Prior to evaluation, both the predicted data table and the ground truth data table are normalized to a uniform markdown format to eliminate expression ambiguities.

Methods RMS-F1 ↑\uparrow
Overall EN ZH
TinyChart [zhang2024tinychart]0.2159 0.4726 0.0876
GOT [wei2024general]0.3160 0.1100 0.4190
OneChart [onechart]0.3716 0.1384 0.4882
Qwen2.5-VL-3B [bai2025qwen2]0.5942 0.5619 0.6103
Qwen2.5-VL-7B [bai2025qwen2]0.6821 0.5876 0.7293
Qwen2.5-VL-72B [bai2025qwen2]0.7300 0.6972 0.7464
PP-StructureV3 [cui2025paddleocr]0.8060 0.7963 0.8109
PaddleOCR-VL 0.8440 0.8222 0.8549

Table 12: Comparison of In-house-Chart Performance

### 4.3 Inference Performance

To improve the inference performance of PaddleOCR-VL, we introduce multi-threading asynchronous execution into the inference workflow. The process is divided into three main stages—data loading (e.g., rendering PDF pages as images), layout model processing, and VLM inference—each running in a separate thread. Data is transferred between adjacent stages via queues, enabling concurrent execution for higher efficiency. In particular, for VLM inference, batch processing is only triggered when either the number of items in the queue reaches a predefined threshold or the waiting time for queued data exceeds a specified limit. This design allows blocks across different pages to be aggregated and processed together, thereby maximizing parallelism, especially when handling large volumes of files. We further deploy PaddleOCR-VL-0.9B on high-throughput inference and serving engines [kwon2023efficient, zheng2024sglang, PaddlePaddle_FastDeploy], tuning parameters like max-num-batched-tokens and gpu-memory-utilization to balance inference throughput with GPU memory consumption.

We measured the end-to-end inference speed and GPU usage on the OmniDocBench v1.0 dataset, processing PDF files in batches of 512 on a single NVIDIA A100 GPU. By "end-to-end", we mean that the inference time was measured from providing the PDF file path as input to the complete generation of the Markdown text. For MonkeyOCR, dots.ocr, and MinerU, inference was run with the vLLM backend and the default configuration (including the KV cache settings). The generated Markdown text was tokenized with the "cl100k_base" tokenizer to compute the number of output tokens. For dots.ocr specifically, 200 threads were used for concurrent page processing, and the Base64-encoded image content in the produced Markdown text was replaced with a dummy path (UUID-based, prefixed with "images/" and suffixed with ".png") to ensure a reasonable token count.

Table [13](https://arxiv.org/html/2510.14528v4#S4.T13 "Table 13 ‣ 4.3 Inference Performance ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") provides a comprehensive comparison of inference efficiency across different methods. The proposed PaddleOCR-VL demonstrates clear and consistent advantages in processing speed. When deployed with the FastDeploy 1 1 1 https://github.com/PaddlePaddle/FastDeploy backend, it achieves 53.1% higher page throughput and 50.9% higher token throughput than the leading baseline, MinerU2.5, establishing itself as the most efficient solution overall. These results collectively confirm that PaddleOCR-VL attains state-of-the-art inference efficiency through a balanced optimization of speed, making it highly suitable for real-world, high-throughput document understanding scenarios.

Methods Backend Total Time (s)↓\downarrow Pages/s↑\uparrow Tokens/s↑\uparrow
MonkeyOCR-pro-1.2B [li2025monkeyocr]vLLM (v0.10.2)1456.4 0.6730 1120.3
dots.ocr [dotsocr]vLLM (v0.10.2)2784.6 0.3522 532.9
MinerU2.5 [niu2025mineru2]vLLM (v0.10.2)927.3 1.0574 1647.9
PaddleOCR-VL SGLang (v0.5.2)882.1 1.1115 1707.8
PaddleOCR-VL vLLM (v0.10.2)728.7 1.3453 2067.6
PaddleOCR-VL FastDeploy (v2.3)605.6 1.6184 2486.4

Table 13:  End-to-End Inference Performance Comparison. 

5 Conclusion
------------

This report introduces PaddleOCR-VL, an advanced and efficient model for document parsing that excels at both element-level and page-level recognition. Its core componets, PaddleOCR-VL-0.9B, built with a NaViT-style visual encoder and ERNIE-4.5-0.3B language model, it accurately recognizes complex elements such as text, tables, formulas, and charts in over 100 languages. PaddleOCR-VL achieves fast inference and low resource consumption, making it practical for real-world deployment. It outperforms existing pipeline solutions on many benchmarks and effectively handles challenging content including handwriting and historical documents, as well as converting chart visuals into structured data. Its broad multilingual support and strong performance have the potential to advance the application and development of multimodal document processing technologies, bringing innovation to automated analysis and information retrieval. This will significantly enhance the performance and stability of RAG systems, making information extraction from complex documents more efficient, thereby providing more reliable data support for future AI applications.

Appendix
--------

Appendix A Training Dataset Details
-----------------------------------

This two-stage approach offers unique advantages in terms of data collection, as obtaining isolated element imagesalong with their annotations is more feasible than collecting complete document pages containing different elements. In the following sections, we will elaborate on the construction of multimodal model training data for text, tables, formulas, and charts.

### A.1 Text

We have curated a large-scale dataset comprising 20 Million High-Quality Image-Text Pairs. As shown in Figure [A1](https://arxiv.org/html/2510.14528v4#A1.F1 "Figure A1 ‣ A.1 Text ‣ Appendix A Training Dataset Details ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model"), the dataset generation follows a rigorous multi-stage pipeline which primarily involves:

![Image 8: Refer to caption](https://arxiv.org/html/images/text_dataset.png)

Figure A1:  The construction method and characteristics of the text training data for PaddleOCR-VL-0.9B. 

1.   1.Automatic Data Annotation: We design an automatic annotation pipeline that integrates lightweight document-structure models with large multimodal language models. Specifically, PP-StructureV3 is employed as an expert model to perform layout analysis and text recognition, generating pseudo labels that are converted into prompts for multimodal models such as ERNIE-4.5-VL and Qwen2.5-VL to refine. Finally, the refined labels are aggregated and randomly merged at multiple granularities to produce 20 million high-quality image–text training samples. 
2.   2.High-quality OCR Data Synthesis: During data distillation, low label quality in challenging scenarios like messy handwriting and dense blurry text was addressed by expanding the dataset through synthetic generation. Utilizing diverse CSS styles, over 200 fonts, and various corpora, we rendered a large amount of images, thereby enhancing the model’s capabilities in these difficult scenarios. 

Ultimately, the data is meticulously annotated at three distinct hierarchical levels: text lines, text blocks, and text pages. With extensive language coverage of 109 languages, including major global ones like Chinese, English, French, and Hindi. It includes diverse scenes including Academic Papers, Newspapers, Handwritten texts, Ancient books, Id cards, tickets, seals, etc. Additionally, the dataset addresses compatibility with a variety of writing systems and text styles, covering Printing, Handwriting, Scanned text, Artistic Fonts, etc.

### A.2 Table

As shown in Figure [A2](https://arxiv.org/html/2510.14528v4#A1.F2 "Figure A2 ‣ A.2 Table ‣ Appendix A Training Dataset Details ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model"), we constructed a large-scale dataset of over 5 5 million high-quality image-table pairs. Our dataset construction employs three key strategies: automatic data annotation, potential annotation mining, and high-quality data synthesis. For coding efficiency, we adopt OTSL [lysak2023optimized] as the model’s target format instead of conventional HTML. The main dataset construction process is as follows:

![Image 9: Refer to caption](https://arxiv.org/html/images/table_dataset_1015.png)

Figure A2:  The construction method and characteristics of the table training data for PaddleOCR-VL-0.9B. 

1.   1.Automatic Data Annotation: To enhance the performance of PaddleOCR-VL in table recognition, we built a large-scale, diverse dataset covering various languages, border styles, and table types. Tables are first located using PP-StructureV3 [cui2025paddleocr]. For unlabeled images, we employed a multi-stage annotation pipeline: ERNIE-4.5-VL [ernie2025technicalreport] first generates pseudo-labels, which are then validated by a ERNIE-4.5-VL-28B-A3B [ernie2025technicalreport] as discriminative model. Rejected annotations are refined using DianJin-OCR-R1 [chen2025dianjin] (for tools, we use ERNIE-4.5-VL and PP-StructureV3 [cui2025paddleocr]). Finally, all annotations undergo rigorous rule-based verification, including n-gram analysis and HTML validation, to ensure only high-quality samples are used for training. 
2.   2.Potential Annotation Mining: For public data with potential annotations (e.g., from arXiv), we extract tables and their corresponding official-supported HTML source code. We then employ a mechanism combining regular expression matching with contextual and sequential alignment to construct accurate table-HTML pairs. The extracted HTML subsequently undergoes rule-based filtering, yielding high-quality data samples ready for model training. 
3.   3.High-quality Table Synthesis: To overcome data imbalance and high annotation costs, we introduce an innovative high-quality table synthesis tool which constitutes the cornerstone of our table data collection pipeline. This tool enables both randomized synthesis for comprehensive data supplement and targeted synthesis to enhance recognition of specific table categories. Specifically, we first leverage LLMs to gather a diverse and extensive corpus.Then, our tool generates table training pairs through randomized configurations of structures, fonts, CSS styles, and textual content, while also supporting customized synthesis by specifying particular parameters to accurately simulate specialized table types. With a synthesis speed of 10,000 10,000 samples per hour, our tool has produced over 5,500,000 5,500,000 training instances, substantially enhancing our model’s generalization capability and comprehensive performance in table recognition. 

Through the aforementioned data construction strategies, we build a comprehensive table dataset encompassing diverse table categories and recognition scenarios, thereby providing robust support for training our model in the table recognition task.

### A.3 Formula

As shown in Figure [A3](https://arxiv.org/html/2510.14528v4#A1.F3 "Figure A3 ‣ A.3 Formula ‣ Appendix A Training Dataset Details ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model"), this dataset was developed using a range of strategies, including source code rendering, automatic data annotation, targeted synthesis of long-tail data, and public data collection. It encompasses a variety of formula scenarios, such as educational supplementary materials, test papers for primary and secondary schools, mathematical papers, PowerPoint courseware, university theses, financial research reports, and handwritten mathematical notes. The dataset features four types of formulas: Simple Printed Expressions, Complex Printed Expressions, Screen-Captured Expressions, and Handwritten Expressions, available in both Chinese and English. The main process for constructing the dataset is as follows:

![Image 10: Refer to caption](https://arxiv.org/html/images/formula_dataset.png)

Figure A3:  The construction method and characteristics of the formula training data for PaddleOCR-VL-0.9B. 

1.   1.Source Code Rendering: To enhance the model’s adaptability to a wide variety of unusual formula structures, a large amount of paper source code was scraped from arXiv, and LaTeX code for the formulas was extracted using regular expressions. Then, MinHash was used to remove duplicate and highly similar formula source codes, and KaTeX was employed to normalize the formula source codes, thereby reducing their ambiguity. Finally, the formulas were re-rendered into images using a formula rendering engine. 
2.   2.Automatic Data Annotation: For real-world formula data from exam papers, educational materials, and handwritten notes, the process begins with the use of the layout analysis method PP-StructureV3 [cui2025paddleocr] to identify the bounding boxes for formulas. Based on these bounding boxes, formula regions are cropped from the images. Subsequently, large multimodal language models, such as ERNIE-4.5-VL-28B-A3B [ernie2025technicalreport], are employed to generate the LaTeX source code for these formulas. Given the rarity of Chinese formulas in real-world scenarios—where approximately 1 out of 100 formulas contains Chinese characters—PP-OCRv5 [cui2025paddleocr] is utilized to recognize characters within the cropped regions, enabling targeted optimization when Chinese characters are detected. Due to the complex and diverse nature of real-world formulas, recognition errors may occur with existing large models. To address this, a LaTeX rendering engine is used to filter the formulas generated by these models. Specifically, image-formula pairs that cannot be successfully rendered by xelatex are discarded. For those that render successfully, a more in-depth screening is conducted by comparing metrics such as the aspect ratio between the recognized image and the rendered image. 
3.   3.Targeted Synthesis of Long-tail Data: For certain long-tail formula structures, such as elementary school vertical calculations, formulas with strikethroughs, and handwritten formulas with explanatory arrows, existing multimodal large models struggle to accurately recognize them due to data distribution issues. To address this, LaTeX code is synthetically generated based on rules and inverse rendering is performed using a LaTeX rendering engine, thereby constructing image-formula matching pairs for these long-tail scenarios. 
4.   4.Public Data Collection: In order to enable the model to learn high-quality formula representations, a substantial amount of data has been collected from existing public datasets, including UniMER-1M [unimernet] and MathWriting [MathWriting]. Specifically, UniMER-1M is oriented towards real document scenarios and has gathered 1 million formula data from arXiv, Pix2tex [pix2tex], CROHME [chrome2014, mouchere2016icfhr2016, chrome2019], and HME100K [yuan2022syntax]. On the other hand, MathWriting is currently the largest handwritten mathematical formula dataset, comprising 230,000 real handwritten formula samples and 400,000 synthetic handwritten formula samples. 

### A.4 Chart

We constructed a large-scale, bilingual (Chinese and English) dataset of over 0.8 million high-quality image-chart pairs. Our dataset construction employs four key strategies: public data collection and cleaning, automatic data annotation, data synthesis, and targeted long-tail data augmentation. The dataset covers a wide array of chart types from diverse sources, including academic papers, financial reports, and web pages. The main dataset construction process is as follows:

![Image 11: Refer to caption](https://arxiv.org/html/images/chart_dataset.png)

Figure A4:  The construction method and characteristics of the chart training data for PaddleOCR-VL-0.9B. 

1.   1.Public Data Collection and Cleaning: We collected a large number of samples from public datasets, including ChartQA [chartqa], PlotQA [plotqa], Chart2Text [chart2text], DVQA [dvqa], Unichart [masry2023unichart], Beagle [beagle], ChartINFO [chartinfo], visText [tang2023vistext], and ExcelChart [excelchart]. However, the raw datasets suffered from poor annotation quality and extremely imbalanced data distributions. Inspired by [llavachart], a meticulous data cleaning and filtering pipeline was implemented to remove noisy samples and ensure balanced clustering, resulting in a high-quality dataset of 220k samples. 
2.   2.Automatic Data Annotation: To annotate our large collection of unlabeled public and in-house data, we developed a two-stage annotation pipeline based on the Vision Large Language Model ERNIE-4.5-VL [ernie2025technicalreport]. In the first stage, the model extracts tick labels from the x- and y-axes; in the second, random permutations of these labels are used to query corresponding data points, framing annotation as a data retrieval task. A final consistency check ensures that only verified annotations are included in the training set, guaranteeing high reliability. 
3.   3.Data Synthesis: To capture diverse visual styles and enhance model generalization, we designed a three-stage data synthesis pipeline. It begins with a large collection of base data tables, followed by an LLM Persona [ge2024scaling] strategy using ERNIE-X1 [ernie2025technicalreport], which diversifies table content and generates persona-specific rendering code. This enables control over chart aesthetics such as color, font, and layout. Leveraging a billion distinct personas, the pipeline produces highly varied data structures and visual styles, substantially improving PaddleOCR-VL’s generalization across real-world charts. For rendering, we employ matplotlib and seaborn. 
4.   4.Targeted Long-tail Data Augmentation: To improve generalization on real-world long-tail samples, we designed a data augmentation pipeline based on seed charts. It first selects long-tail samples by their distinctive visual features, then uses ERNIE-4.5-VL [ernie2025technicalreport] to replicate their rendering code. ERNIE-X1 [ernie2025technicalreport], guided by a specific persona [ge2024scaling], further diversifies the code by altering data tables and visual styles. Executing the modified code produces new augmented charts with corresponding data tables. 

Through the four data construction strategies mentioned above, the final chart dataset covers a wide range of application scenarios and a rich variety of chart styles, providing strong support for the training of chart models.

Appendix B Supported Languages
------------------------------

PaddleOCR-VL supports a total of 109 languages. Table [6](https://arxiv.org/html/2510.14528v4#S4.T6 "Table 6 ‣ In-house-OCR: ‣ 4.2.1 Text Recognition ‣ 4.2 Element-level Evaluation ‣ 4 Evaluation ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") in the main text shows the text line recognition accuracy for different languages. Table [A1](https://arxiv.org/html/2510.14528v4#A2.T1 "Table A1 ‣ Appendix B Supported Languages ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") lists the correspondence between each language category and the specific supported languages.

Language Category Specific Languages
Chinese Chinese
English English
Korean Korean
Japanese Japanese
Thai Thai
Greek Greek
Tamil Tamil
Telugu Telugu
Arabic Arabic, Persian, Uyghur, Urdu, Pashto, Kurdish, Sindhi, Balochi
Latin French, German, Afrikaans, Italian, Spanish, Bosnian, Portuguese, Czech, Welsh, Danish, Estonian, Irish, Croatian, Uzbek, Hungarian, Serbian (Latin), Indonesian, Occitan, Icelandic, Lithuanian, Maori, Malay, Dutch, Norwegian, Polish, Slovak, Slovenian, Albanian, Swedish, Swahili, Tagalog, Turkish, Latin, Azerbaijani, Kurdish, Latvian, Maltese, Pali, Romanian, Vietnamese, Finnish, Basque, Galician, Luxembourgish, Romansh, Catalan, Quechua
Cyrillic Russian, Belarusian, Ukrainian, Serbian (Cyrillic), Bulgarian, Mongolian, Abkhazian, Adyghe, Kabardian, Avar, Dargin, Ingush, Chechen, Lak, Lezgin, Tabasaran, Kazakh, Kyrgyz, Tajik, Macedonian, Tatar, Chuvash, Bashkir, Malian, Moldovan, Udmurt, Komi, Ossetian, Buryat, Kalmyk, Tuvan, Sakha, Karakalpak
Devanagari Hindi, Marathi, Nepali, Bihari, Maithili, Angika, Bhojpuri, Magahi, Santali, Newari, Konkani, Sanskrit, Haryanvi

Table A1: Supported Languages

Appendix C Inference Performance on Different Hardware Configurations
---------------------------------------------------------------------

We measured the inference performance of PaddleOCR-VL on different hardware configurations, as summarized in Table [A2](https://arxiv.org/html/2510.14528v4#A3.T2 "Table A2 ‣ Appendix C Inference Performance on Different Hardware Configurations ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model").We used FastDeploy version 2.3.0, vLLM version 0.10.2, and SGLang version 0.5.2. As observed, PaddleOCR-VL demonstrates stable and efficient inference performance across a wide range of hardware and backend configurations, showing that the system can flexibly adapt to diverse computing environments.

Hardware Backend Total Time (s)↓\downarrow Pages/s↑\uparrow Tokens/s↑\uparrow Avg. VRAM Usage (GB)↓\downarrow
H800 FastDeploy 400.4 2.2250 3416.7 63.5
vLLM 596.7 1.6442 2524.9 44.2
SGLang 875.2 1.1198 1720.8 49.5
A100 FastDeploy 605.6 1.6184 2486.4 62.8
vLLM 728.7 1.3453 2067.6 40.1
SGLang 882.1 1.1115 1707.8 49.7
H20 FastDeploy 635.0 1.5432 2371.2 64.3
vLLM 694.7 1.4112 2168.8 74.8
SGLang 759.6 1.2906 1986.6 80.2
L20 FastDeploy 802.0 1.2224 1881.9 40.8
vLLM 904.9 1.0835 1665.1 25.2
SGLang 953.6 1.0280 1579.0 31.3
A10 FastDeploy 1197.2 0.8191 1260.9 21.8
vLLM 1320.5 0.7426 1141.2 15.6
SGLang 1446.2 0.6781 1043.8 19.9
RTX 3060 vLLM 2694.1 0.3641 559.5 11.9
SGLang 2765.2 0.3547 546.1 11.9
RTX 4090D vLLM 851.9 1.1507 1768.1 16.6
SGLang 932.4 1.0514 1618.5 20.9

Table A2: End-to-End Inference Performance

Appendix D Real-world Samples
-----------------------------

This appendix showcases the parsing and recognition capabilities of our proposed algorithm across a variety of challenging scenarios.

Section [D.1](https://arxiv.org/html/2510.14528v4#A4.SS1 "D.1 Comprehensive Document Parsing ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") demonstrates the overall document parsing capability of PaddleOCR-VL. Figures [A5](https://arxiv.org/html/2510.14528v4#A4.F5 "Figure A5 ‣ D.1 Comprehensive Document Parsing ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")-[A8](https://arxiv.org/html/2510.14528v4#A4.F8 "Figure A8 ‣ D.1 Comprehensive Document Parsing ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") are examples of parsing different types of documents in Markdown format.

Figures [A9](https://arxiv.org/html/2510.14528v4#A4.F9 "Figure A9 ‣ D.2 Layout Detection ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")-[A11](https://arxiv.org/html/2510.14528v4#A4.F11 "Figure A11 ‣ D.2 Layout Detection ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") in section [D.2](https://arxiv.org/html/2510.14528v4#A4.SS2 "D.2 Layout Detection ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") illustrate the superior ability of PaddleOCR-VL to process pages featuring intricate or challenging layouts.

Figures [A12](https://arxiv.org/html/2510.14528v4#A4.F12 "Figure A12 ‣ D.3 Reading Order ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") and [A13](https://arxiv.org/html/2510.14528v4#A4.F13 "Figure A13 ‣ D.3 Reading Order ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") in section [D.3](https://arxiv.org/html/2510.14528v4#A4.SS3 "D.3 Reading Order ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") demonstrate that PaddleOCR-VL maintains excellent reading order when faced with complex layouts, such as those found in various reports, textbooks, newspapers, magazines, and even vertical documents.

Section [D.4](https://arxiv.org/html/2510.14528v4#A4.SS4 "D.4 Text Recognition ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") highlights the robust text recognition performance of PaddleOCR-VL in challenging cases, including multilingual text, handwriting text, and vertical text, which are presented in Figures [A14](https://arxiv.org/html/2510.14528v4#A4.F14 "Figure A14 ‣ D.4.1 Multilingual Text Recognition ‣ D.4 Text Recognition ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")-[A22](https://arxiv.org/html/2510.14528v4#A4.F22 "Figure A22 ‣ D.4.3 Vertical Text Recognition ‣ D.4 Text Recognition ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model").

The model’s table recognition abilities are demonstrated in section [D.5](https://arxiv.org/html/2510.14528v4#A4.SS5 "D.5 Table Recognition ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model"). Figures [A23](https://arxiv.org/html/2510.14528v4#A4.F23 "Figure A23 ‣ D.5 Table Recognition ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") and [A24](https://arxiv.org/html/2510.14528v4#A4.F24 "Figure A24 ‣ D.5 Table Recognition ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") showcase its robust handling of a wide array of table formats, including tables from academic papers, tables from financial reports, tables with watermark, tables with image, tables with formulas and photograph of tables.

Figures in section [D.6](https://arxiv.org/html/2510.14528v4#A4.SS6 "D.6 Formula Recognition ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") detail the formula recognition performance. Figure [A25](https://arxiv.org/html/2510.14528v4#A4.F25 "Figure A25 ‣ D.6 Formula Recognition ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") demonstrates the ability to handle various types of english formulas including complex printed expressions, handwritten expressions screen-captured expressions and vertical formula, while Figure [A26](https://arxiv.org/html/2510.14528v4#A4.F26 "Figure A26 ‣ D.6 Formula Recognition ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") focuses on the ability to handle formulas that contain Chinese characters.

In section [D.7](https://arxiv.org/html/2510.14528v4#A4.SS7 "D.7 Chart Recognition ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model"), PaddleOCR-VL demonstrates impressive chart recognition capabilities, a feature currently lacking in many expert OCR VLMs like MinerU2.5 [MinerU2], dots.ocr [dotsocr] or MonkeyOCR [li2025monkeyocr]. Figures [A27](https://arxiv.org/html/2510.14528v4#A4.F27 "Figure A27 ‣ D.7 Chart Recognition ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")-[A29](https://arxiv.org/html/2510.14528v4#A4.F29 "Figure A29 ‣ D.7 Chart Recognition ‣ Appendix D Real-world Samples ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") showcase our ability to parse various chart types, including pie charts, bar charts, line charts, bar-line hybrid charts and heatmap.

### D.1 Comprehensive Document Parsing

![Image 12: Refer to caption](https://arxiv.org/html/images/appendix/overview1.jpg)

Figure A5:  The Layout and Markdown Output for Book, Textbook and Academic Paper. 

![Image 13: Refer to caption](https://arxiv.org/html/images/appendix/overview2.jpg)

Figure A6:  The Layout and Markdown Output for Research Report(with chart recognition enabled), Financial Report, Slides and Exam Paper. 

![Image 14: Refer to caption](https://arxiv.org/html/images/appendix/overview3.jpg)

Figure A7:  The Layout and Markdown Output for Notes, Vertical Book and Ancient Book. 

![Image 15: Refer to caption](https://arxiv.org/html/images/appendix/overview4.jpg)

Figure A8:  The Layout and Markdown Output for Certificate, Newspaper and Magazine. 

### D.2 Layout Detection

![Image 16: Refer to caption](https://arxiv.org/html/images/appendix/layout_01.jpg)

Figure A9:  The Layout Detection results for various types of documents. 

![Image 17: Refer to caption](https://arxiv.org/html/images/appendix/layout_02.jpg)

Figure A10:  The Layout Detection results for various types of documents. 

![Image 18: Refer to caption](https://arxiv.org/html/images/appendix/layout_03.jpg)

Figure A11:  The Layout Detection results for various types of documents. 

### D.3 Reading Order

![Image 19: Refer to caption](https://arxiv.org/html/images/appendix/order_01.jpg)

Figure A12:  The Reading Order results for various types of documents. 

![Image 20: Refer to caption](https://arxiv.org/html/images/appendix/order_02.jpg)

Figure A13:  The Reading Order results for various types of documents. 

### D.4 Text Recognition

#### D.4.1 Multilingual Text Recognition

![Image 21: Refer to caption](https://arxiv.org/html/images/appendix/text_french_hindi.jpg)

Figure A14:  The markdown output for French and Hindi documents. 

![Image 22: Refer to caption](https://arxiv.org/html/images/appendix/text_croatian_spanish.jpg)

Figure A15:  The markdown output for Croatian and Spanish documents. 

![Image 23: Refer to caption](https://arxiv.org/html/images/appendix/text_english_arabic.jpg)

Figure A16:  The markdown output for English and Arabic documents. 

![Image 24: Refer to caption](https://arxiv.org/html/images/appendix/text_german_chinese.jpg)

Figure A17:  The markdown output for German and Chinese documents. 

![Image 25: Refer to caption](https://arxiv.org/html/images/appendix/text_russian_japanese.jpg)

Figure A18:  The markdown output for Russian and Japanese documents. 

![Image 26: Refer to caption](https://arxiv.org/html/images/appendix/text_thai_korean.jpg)

Figure A19:  The markdown output for Thai and Korean documents. 

#### D.4.2 Handwriting Text Recognition

![Image 27: Refer to caption](https://arxiv.org/html/images/appendix/text_handwriting_01.jpg)

Figure A20:  The markdown output for Mixed Printed Handwritten Text and Handwritten Formula documents. 

![Image 28: Refer to caption](https://arxiv.org/html/images/appendix/text_handwriting_02.jpg)

Figure A21:  The markdown output for Handwriting Chinese and Handwriting English documents. 

#### D.4.3 Vertical Text Recognition

![Image 29: Refer to caption](https://arxiv.org/html/images/appendix/text_vertical.jpg)

Figure A22:  The markdown output for various types of vertical documents. 

### D.5 Table Recognition

![Image 30: Refer to caption](https://arxiv.org/html/images/appendix/table_01.jpg)

Figure A23:  The markdown output for various types of Tables. 

![Image 31: Refer to caption](https://arxiv.org/html/images/appendix/table_02.jpg)

Figure A24:  The markdown output for various types of Tables. 

### D.6 Formula Recognition

![Image 32: Refer to caption](https://arxiv.org/html/images/appendix/formula_EN.jpg)

Figure A25:  The markdown output for various types of Formulas. 

![Image 33: Refer to caption](https://arxiv.org/html/images/appendix/formula_ZH.jpg)

Figure A26:  The markdown output for various types of Formulas. 

### D.7 Chart Recognition

![Image 34: Refer to caption](https://arxiv.org/html/images/appendix/chart_01.jpg)

Figure A27:  The markdown output for various types of Charts. 

![Image 35: Refer to caption](https://arxiv.org/html/images/appendix/chart_02.jpg)

Figure A28:  The markdown output for various types of Charts. 

![Image 36: Refer to caption](https://arxiv.org/html/images/appendix/chart_03.jpg)

Figure A29:  The markdown output for various types of Charts. 

Appendix E Compare with Others
------------------------------

PaddleOCR-VL showcases superior performance in scenarios involving PDF pages with complex layout, consistently outperforming existing state-of-the-art (SOTA) models. This is evident from Figures [A30](https://arxiv.org/html/2510.14528v4#A5.F30 "Figure A30 ‣ E.1 Layout Detection ‣ Appendix E Compare with Others ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") and [A31](https://arxiv.org/html/2510.14528v4#A5.F31 "Figure A31 ‣ E.1 Layout Detection ‣ Appendix E Compare with Others ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model"), which highlight its exceptional capability in handling pages with intricate layouts and unique elements, surpassing other solutions.

Moreover, the model demonstrates exceptionally high recognition accuracy in several domains, including Multilingual Text Recognition, Handwriting Text Recognition, and Vertical Text Recognition. Figures [A32](https://arxiv.org/html/2510.14528v4#A5.F32 "Figure A32 ‣ E.2.1 Multilingual Text Recognition ‣ E.2 Text Recognition ‣ Appendix E Compare with Others ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model")- [A37](https://arxiv.org/html/2510.14528v4#A5.F37 "Figure A37 ‣ E.2.3 Vertical Text Recognition ‣ E.2 Text Recognition ‣ Appendix E Compare with Others ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") illustrate how PaddleOCR-VL outperforms competitors such as MinerU2.5 [niu2025mineru2] and MonkeyOCR [li2025monkeyocr], which tend to misidentify languages like Russian and Hindi as English, overlook some handwritten characters, and struggle with vertical text recognition.

In dealing with complex tables, PaddleOCR-VL’s parsing accuracy stands out, as evidenced by Figures [A38](https://arxiv.org/html/2510.14528v4#A5.F38 "Figure A38 ‣ E.3 Table Recognition ‣ Appendix E Compare with Others ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") and [A39](https://arxiv.org/html/2510.14528v4#A5.F39 "Figure A39 ‣ E.3 Table Recognition ‣ Appendix E Compare with Others ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model"). This is a domain where other models frequently encounter difficulties.

Additionally, Figure [A40](https://arxiv.org/html/2510.14528v4#A5.F40 "Figure A40 ‣ E.4 Formula Recognition ‣ Appendix E Compare with Others ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") demonstrates PaddleOCR-VL’s proficiency in accurately parsing complex formulas. In contrast, other SOTA models often produce incorrect or flawed outputs when faced with challenging mathematical notations.

Finally, as depicted in Figures [A41](https://arxiv.org/html/2510.14528v4#A5.F41 "Figure A41 ‣ E.5 Chart Recognition ‣ Appendix E Compare with Others ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model") and [A42](https://arxiv.org/html/2510.14528v4#A5.F42 "Figure A42 ‣ E.5 Chart Recognition ‣ Appendix E Compare with Others ‣ PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model"), PaddleOCR-VL also excels in Chart Recognition. It outperforms multi-modal large language models like Qwen2.5VL-72B [bai2025qwen2] and GPT-4o by accurately reconstructing the structure and content of charts.

### E.1 Layout Detection

![Image 37: Refer to caption](https://arxiv.org/html/images/appendix/compare_with_others/layout_01.jpg)

Figure A30:  Compare with others in Layout Detection. 

![Image 38: Refer to caption](https://arxiv.org/html/images/appendix/compare_with_others/layout_02.jpg)

Figure A31:  Compare with others in Layout Detection. 

### E.2 Text Recognition

#### E.2.1 Multilingual Text Recognition

![Image 39: Refer to caption](https://arxiv.org/html/images/appendix/compare_with_others/text_multi_01.jpg)

Figure A32:  Compare with others in Multilingual Text Recognition. 

![Image 40: Refer to caption](https://arxiv.org/html/images/appendix/compare_with_others/text_multi_02.jpg)

Figure A33:  Compare with others in Multilingual Text Recognition. 

![Image 41: Refer to caption](https://arxiv.org/html/images/appendix/compare_with_others/text_multi_03.jpg)

Figure A34:  Compare with others in Multilingual Text Recognition. 

#### E.2.2 Handwriting Text Recognition

![Image 42: Refer to caption](https://arxiv.org/html/images/appendix/compare_with_others/text_handwrite_01.jpg)

Figure A35:  Compare with others in Handwriting Text Recognition. 

![Image 43: Refer to caption](https://arxiv.org/html/images/appendix/compare_with_others/text_handwrite_02.jpg)

Figure A36:  Compare with others in Handwriting Text Recognition. 

#### E.2.3 Vertical Text Recognition

![Image 44: Refer to caption](https://arxiv.org/html/images/appendix/compare_with_others/text_vertical_01.jpg)

Figure A37:  Compare with others in Vertical Text Recognition. 

### E.3 Table Recognition

![Image 45: Refer to caption](https://arxiv.org/html/images/appendix/compare_with_others/table_01.jpg)

Figure A38:  Compare with others in Table Recognition. 

![Image 46: Refer to caption](https://arxiv.org/html/images/appendix/compare_with_others/table_02.jpg)

Figure A39:  Compare with others in Table Recognition. 

### E.4 Formula Recognition

![Image 47: Refer to caption](https://arxiv.org/html/images/appendix/compare_with_others/formula.jpg)

Figure A40:  Compare with others in Formula Recognition. 

### E.5 Chart Recognition

![Image 48: Refer to caption](https://arxiv.org/html/images/appendix/compare_with_others/chart_01.jpg)

Figure A41:  Compare with others in Chart Recognition. 

![Image 49: Refer to caption](https://arxiv.org/html/images/appendix/compare_with_others/chart_02.jpg)

Figure A42:  Compare with others in Chart Recognition. 

Generated on Mon Nov 24 13:17:11 2025 by [L a T e XML![Image 50: Mascot Sammy](blob:http://localhost/70e087b9e50c3aa663763c3075b0d6c5)](http://dlmf.nist.gov/LaTeXML/)

