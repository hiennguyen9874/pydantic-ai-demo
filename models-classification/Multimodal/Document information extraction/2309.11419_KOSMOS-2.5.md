Title: Kosmos-2.5: A Multimodal Literate Model

URL Source: https://arxiv.org/html/2309.11419

Markdown Content:
Tengchao Lv,Yupan Huang 1 1 footnotemark: 1,Jingye Chen 1 1 footnotemark: 1,Yuzhong Zhao, Yilin Jia, Lei Cui†, 

Shuming Ma,Yaoyao Chang,Shaohan Huang,Wenhui Wang,Li Dong, 

Weiyao Luo,Shaoxiang Wu,Guoxin Wang,Cha Zhang,Furu Wei 2 2 footnotemark: 2

 Microsoft 

\href https://aka.ms/GeneralAIaka.ms/GeneralAI

###### Abstract

The automatic reading of text-intensive images represents a significant advancement toward achieving Artificial General Intelligence (AGI). In this paper we present Kosmos-2.5, a multimodal literate model for machine reading of text-intensive images. Pre-trained on a large-scale corpus of text-intensive images, Kosmos-2.5 excels in two distinct yet complementary transcription tasks: (1) generating spatially-aware text blocks, where each block of text is assigned spatial coordinates within the image, and (2) producing structured text output that captures both style and structure in markdown format. This unified multimodal literate capability is achieved through a shared decoder-only autoregressive Transformer architecture and task-specific prompts. Building on this foundation, we fine-tune Kosmos-2.5 for document understanding tasks, resulting in a document understanding generalist named Kosmos-2.5-chat. Additionally, a large corpus of 357.4 million document pages spanning diverse domains was curated for pre-training. We evaluate Kosmos-2.5 on two newly proposed benchmarks, OCREval and MarkdownEval, for document-level text recognition and image-to-markdown generation, demonstrating impressive literate capabilities comparable to GPT-4o. Kosmos-2.5-chat achieves performance comparable to other state-of-the-art generalists that are five times larger (1.3B vs. 7B) across nine text-rich visual question answering benchmarks. Models and code have been available at [https://aka.ms/kosmos25](https://aka.ms/kosmos25).

![Image 1: Refer to caption](https://arxiv.org/html/2309.11419v2/x1.png)

Figure 1: Kosmos-2.5 is a multimodal document foundation model that takes text images as input and generates spatially-aware texts (i.e., texts with bounding boxes) or markdown-formatted texts (i.e., texts with markdown elements), following different task prompts, respectively. The model possesses the ability to comprehensively perceive textual content, its spatial context, and nuances of formatting and style within a unified framework. Kosmos-2.5-chat is fine-tuned from Kosmos-2.5. It is a visual document understanding generalist that can answer user-provided questions about text-rich images from various domains. 

1 Introduction
--------------

Multimodal large language models (MLLMs) extend the capabilities of large language models (LLMs) to multimodal tasks, enabling them to process and generate responses from both textual and visual inputs(Zhang et al. [2023b](https://arxiv.org/html/2309.11419v2#bib.bib95); Liu et al. [2024b](https://arxiv.org/html/2309.11419v2#bib.bib50); ChatGPT [2022](https://arxiv.org/html/2309.11419v2#bib.bib6); Touvron et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib76)). However, while existing MLLMs have primarily focused on natural images, the challenge of effectively reading and understanding text-intensive images—such as academic papers, receipts, design documents, and web pages—remains underexplored.

Traditional Optical Character Recognition (OCR) methods are primarily designed for generating line-level text content and capturing its spatial positions within an image. Although these methods preserve layout information, they often neglect the document-level reading order and structural integrity that are crucial for accurate document understanding. On the other hand, markdown-formatted text offers significant advantages over plain text by explicitly distinguishing between different structural elements—such as tables, lists, and headings—through specific tokens. Current approaches are either limited to line-level text recognition(Ye et al. [2023a](https://arxiv.org/html/2309.11419v2#bib.bib89); Hu et al. [2024](https://arxiv.org/html/2309.11419v2#bib.bib24); Li et al. [2023c](https://arxiv.org/html/2309.11419v2#bib.bib46)) or focus on structured parsing within a specific document category(Blecher et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib4)), making it difficult to achieve comprehensive document-level reading and understanding capabilities across diverse categories.

Motivated by these observations, we present Kosmos-2.5, a multimodal literate model designed to address the unique challenges of reading and understanding text-intensive documents, including capturing the reading order and structural integrity of the content. As illustrated in Figure[1](https://arxiv.org/html/2309.11419v2#S0.F1 "Figure 1 ‣ Kosmos-2.5: A Multimodal Literate Model"), Kosmos-2.5 is pre-trained on two distinct yet complementary generative tasks: document-level text recognition and image-to-markdown generation. The first task involves generating spatially-aware text blocks, assigning text lines to their corresponding spatial coordinates within the original text-rich image. The second task focuses on producing structured text output that captures both style and structure in markdown format. Both tasks are performed within a unified framework using task-specific prompts, leveraging a shared Transformer architecture that combines a ViT-based vision encoder and a Transformer-based language decoder connected by a resampler module(Dosovitskiy et al. [2021](https://arxiv.org/html/2309.11419v2#bib.bib16); Lee et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib36); Alayrac et al. [2022](https://arxiv.org/html/2309.11419v2#bib.bib1)).

To realize the potential of our pre-trained model and validate its effectiveness in downstream understanding tasks, we further fine-tune Kosmos-2.5 for document understanding tasks, resulting in Kosmos-2.5-chat, which can answer user-provided questions about text-rich images. Despite having only 1.3B parameters, Kosmos-2.5-chat achieves performance comparable to other state-of-the-art generalists with over 7B parameters on various text-rich visual question answering benchmarks.

Given the absence of a comprehensive document reading dataset, we curated a large corpus of 357.4 million document pages, including scanned documents, general documents, academic papers, web pages, design images, handwritten texts, mathematical content, and project documents. Each document is annotated with text lines with bounding boxes or markdown formats. This dataset was constructed using an automatic pipeline for data collection, filtering, and quality control, offering valuable insights for future research.

Existing document reading benchmarks primarily focus on line-level text reading capabilities(Liu et al. [2023b](https://arxiv.org/html/2309.11419v2#bib.bib51)) or are limited to specific domains, such as converting academic papers to markdown format (Blecher et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib4)). To comprehensively evaluate models’ capabilities in document-level text recognition and image-to-markdown generation tasks, we introduce two extensive benchmarks: OCREval and MarkdownEval. Specifically, OCREval contains 2,297 samples, while MarkdownEval includes 5,633 samples. The benchmarks cover a diverse range of document categories, including handwritten texts, design documents, receipts, academic papers, web pages, mathematical content, tables, and more. Experimental results on these benchmarks demonstrate that Kosmos-2.5 exhibits impressive literate capabilities on par with GPT-4o(GPT-4 [2023](https://arxiv.org/html/2309.11419v2#bib.bib19)).

The contributions of this work are summarized as follows:

*   •We propose two distinct yet cooperative document reading tasks for pre-training a foundational document model capable of machine reading and understanding the order and structure of text-intensive documents. The pre-trained Kosmos-2.5 demonstrates impressive multimodal literate capabilities on par with GPT-4o, and the fine-tuned Kosmos-2.5-chat achieves competitive results across nine document understanding benchmarks. 
*   •We curated a large and diverse corpus consisting of 357.4 million text-rich document images, with text lines annotated with bounding boxes or in markdown format. The automated data curation pipeline provides valuable insights for future research. 
*   •We introduce two comprehensive benchmarks, OCREval and MarkdownEval, to provide thorough evaluations of document-level machine reading capabilities. 

![Image 2: Refer to caption](https://arxiv.org/html/2309.11419v2/x2.png)

Figure 2:  The model architecture of Kosmos-2.5 leverages a shared Transformer architecture that combines a ViT-based vision encoder and a Transformer-based language decoder connected by a resampler module. 

Training Stage Task Definition Prompt
Pre-training Document-level Text Recognition Generating spatially-aware text blocks, where each block of text is assigned its spatial coordinates within the image<s><image>Image Embedding</image><ocr>
⋃n=1 N superscript subscript 𝑛 1 𝑁\bigcup_{n=1}^{N}⋃ start_POSTSUBSCRIPT italic_n = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_N end_POSTSUPERSCRIPT (𝐁 n⊕𝐓 n)\mathbf{B}_{n}\oplus\mathbf{T}_{n})bold_B start_POSTSUBSCRIPT italic_n end_POSTSUBSCRIPT ⊕ bold_T start_POSTSUBSCRIPT italic_n end_POSTSUBSCRIPT )</s>
Image-to-Markdown Generation Producing structured text output that captures styles and structures into the markdown format<s><image>Image Embedding</image><md>
[Markdown Text]</s>
Fine-tuning Document Understanding Answering the user-provided text-related questions about text-intensive images<s><image>Image Embedding</image><md>
A chat between a curious user and an artificial intelligence
assistant. The assistant gives helpful, detailed, and polite
answers to the user’s questions. USER: [Question]
ASSISTANT: [Answer]</s>

Table 1: Tasks, prompts, and response sequence formats used to train Kosmos-2.5. Special tokens <s> and </s> denote sequence boundaries, while <image> and </image> indicate the start and end of image embeddings. For document-level text recognition tasks, the operator ⊕direct-sum\oplus⊕ represents the concatenation of the text line 𝐓 n subscript 𝐓 𝑛\mathbf{T}_{n}bold_T start_POSTSUBSCRIPT italic_n end_POSTSUBSCRIPT and its bounding box 𝐁 n subscript 𝐁 𝑛\mathbf{B}_{n}bold_B start_POSTSUBSCRIPT italic_n end_POSTSUBSCRIPT. During pre-training, special tokens <ocr> and <md> denote document-level text recognition and text-to-markdown generation tasks, respectively. For visual document understanding tasks, we use the same format as text-to-markdown generation tasks since these do not require bounding box outputs.

Format and Task Document Category Description Page Num Sampling Ratio
Layout-based(texts+bboxes)Document-level Text Recognition Scanned Document Includes IIT-CDIP(Lewis et al. [2006](https://arxiv.org/html/2309.11419v2#bib.bib37)), a large collection of scanned documents.27.6M 10%
General Document Includes general PDFs and SEC files. General PDFs are crawled from the web, resulting in a diverse open-domain digital PDF corpus. SEC files are sourced from SEC.gov and comprise various companies’ periodic reports, filings, and forms.187.4M 20%
Academic Paper Includes arXiv papers.20.9M 5%
Web Page Self-constructed large-scale dataset of crawled web pages.100.5M 10%
Design Image Includes PowerPoint, posters, and MARIO-10M(Chen et al. [2024](https://arxiv.org/html/2309.11419v2#bib.bib7)) collected from various sources.6.2M 3%
Handwritten Image Includes Synt-handwritten data produced using a wide range of handwritten font files.0.2M 1%
Math Image Includes CROHME(Mouchere et al. [2014](https://arxiv.org/html/2309.11419v2#bib.bib58)) and IM2LATEX-100K(Deng et al. [2017](https://arxiv.org/html/2309.11419v2#bib.bib15)),CROHME contains various handwritten mathematical expressions. IM2LATEX-100K is a large dataset containing mathematical expressions with corresponding LaTeX markup.0.6M 1%
Markup-based(texts+markdown)Image-to-Markdown Generation General Document Includes Docx type files and SEC files sourced from SEC.gov. They are crawled from the web and converted tomarkdown format. Each page corresponds to its markdown information.1.1M 10%
Academic Paper A subset of the entire arXiv papers is used to extract the mapping of PDF pages and its corresponding markdown information converted from the LATEX code.3.7M 15%
Project Document Includes “README.md” files of open-source GitHub projects, primarily in markdown format.2.9M 15%
Web Page Self-constructed large-scale dataset of crawled web pages, and its corresponding markdown information converted from the HTML code.6.3M 10%
Total 357.4M 100%

Table 2: Summary of data used to pre-train Kosmos-2.5, including descriptions of each document category, the number of pages, and their respective sampling ratios in the training data.

Model Text Bbox Size Domain
Donut✓13M Synthetic, Doc
Pix2Struct✓80M Web
QwenVL✓24.8M Synthetic, Doc, Web
UReader 0.1M Doc, Table, Chart, Web, Natural
DocPedia✓0.9M Doc
CogAgent✓✓107M Synthetic, Nature, Doc, Web
DocOwl-1.5✓✓4M Doc, Table, Chart, Web, Natural
Kosmos-2.5✓✓357M Doc, Table, Chart, Web, Natural Handwritten, Design, Math

Table 3: Comparison of pre-training data used by document multimodal models.

2 Kosmos-2.5
------------

### 2.1 Model Architecture

The architecture of Kosmos-2.5 comprises a vision encoder and a language decoder, connected through a resampling module to reduce the sequence length of the image(Alayrac et al. [2022](https://arxiv.org/html/2309.11419v2#bib.bib1)), as illustrated in Figure[2](https://arxiv.org/html/2309.11419v2#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Kosmos-2.5: A Multimodal Literate Model"). The vision encoder is initialized from the Pix2Struct-Large model’s encoder(Lee et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib36)), which is based on the Vision Transformer (ViT)(Dosovitskiy et al. [2021](https://arxiv.org/html/2309.11419v2#bib.bib16)). Consistent with Pix2Struct(Lee et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib36)), we employ a variable resolution strategy and extract the maximum number of fixed-size patches that can fit within a predefined sequence length.

The resampler compresses the image sequence into a shorter, fixed number of tokens:

H 0=f⁢(I)subscript 𝐻 0 𝑓 𝐼\displaystyle H_{0}=f(I)italic_H start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT = italic_f ( italic_I )(1)

H 1=Attention⁢(V,[V;H 0],[V;H 0])subscript 𝐻 1 Attention 𝑉 𝑉 subscript 𝐻 0 𝑉 subscript 𝐻 0\displaystyle H_{1}=\text{Attention}(V,[V;H_{0}],[V;H_{0}])italic_H start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT = Attention ( italic_V , [ italic_V ; italic_H start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT ] , [ italic_V ; italic_H start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT ] )(2)

where I 𝐼 I italic_I is the input image, f 𝑓 f italic_f is the encoder function, V 𝑉 V italic_V represents a set of predefined soft tokens, and [;][;][ ; ] denotes the concatenation operator. The language decoder is based on a Transformer architecture and is designed to condition on both image and text contexts for next-token prediction. Details on the hyperparameters can be found in Appendix[A.1](https://arxiv.org/html/2309.11419v2#A1.SS1 "A.1 Model and Training Hyperparameters ‣ Appendix A Appendix ‣ Kosmos-2.5: A Multimodal Literate Model").

### 2.2 Image and Text Representation

The image representation is derived from the image encoder and resampler as described in Section[2.1](https://arxiv.org/html/2309.11419v2#S2.SS1 "2.1 Model Architecture ‣ 2 Kosmos-2.5 ‣ Kosmos-2.5: A Multimodal Literate Model"). Text representation is obtained through text tokenization and embedding. For markdown text, we directly tokenize it while preserving all special characters and formatting indicators. For text lines with bounding boxes, we convert the coordinates into discrete location tokens, similar to Kosmos-2(Peng et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib61)).

We introduce a set of 2⁢L+2 2 𝐿 2 2L+2 2 italic_L + 2 specialized tokens: <x 0>, <x 1>, …, <x L-1>, <y 0>, …, <y L-1>, <bbox>, and </bbox>, which correspond to the coordinates and the start and end markers of a bounding box. The coordinates are obtained by rounding down the actual positions after resizing the images.

Consider a document T 𝑇 T italic_T with N 𝑁 N italic_N text lines. Each line is represented as 𝐓 n={w 1(n),w 2(n),…,w M n(n)}subscript 𝐓 𝑛 superscript subscript 𝑤 1 𝑛 superscript subscript 𝑤 2 𝑛…superscript subscript 𝑤 subscript 𝑀 𝑛 𝑛\mathbf{T}_{n}=\{w_{1}^{(n)},w_{2}^{(n)},\ldots,w_{M_{n}}^{(n)}\}bold_T start_POSTSUBSCRIPT italic_n end_POSTSUBSCRIPT = { italic_w start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_n ) end_POSTSUPERSCRIPT , italic_w start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_n ) end_POSTSUPERSCRIPT , … , italic_w start_POSTSUBSCRIPT italic_M start_POSTSUBSCRIPT italic_n end_POSTSUBSCRIPT end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_n ) end_POSTSUPERSCRIPT }, where M n subscript 𝑀 𝑛 M_{n}italic_M start_POSTSUBSCRIPT italic_n end_POSTSUBSCRIPT is the number of words in the n 𝑛 n italic_n-th text line. The bounding box for 𝐓 n subscript 𝐓 𝑛\mathbf{T}_{n}bold_T start_POSTSUBSCRIPT italic_n end_POSTSUBSCRIPT is then expressed as 𝐁 n=<bbox><⁢x tl(n)⁢><⁢y tl(n)⁢><⁢x br(n)⁢><⁢y br(n)⁢></bbox>subscript 𝐁 𝑛<bbox><superscript subscript 𝑥 tl 𝑛><superscript subscript 𝑦 tl 𝑛><superscript subscript 𝑥 br 𝑛><superscript subscript 𝑦 br 𝑛></bbox>\mathbf{B}_{n}=\texttt{<bbox><}x_{\text{tl}}^{(n)}\texttt{><}y_{\text{tl}}^{(n% )}\texttt{><}x_{\text{br}}^{(n)}\texttt{><}y_{\text{br}}^{(n)}\texttt{></bbox>}bold_B start_POSTSUBSCRIPT italic_n end_POSTSUBSCRIPT = <bbox>< italic_x start_POSTSUBSCRIPT tl end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_n ) end_POSTSUPERSCRIPT >< italic_y start_POSTSUBSCRIPT tl end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_n ) end_POSTSUPERSCRIPT >< italic_x start_POSTSUBSCRIPT br end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_n ) end_POSTSUPERSCRIPT >< italic_y start_POSTSUBSCRIPT br end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ( italic_n ) end_POSTSUPERSCRIPT ></bbox>, where the coordinates represent the top-left and bottom-right corners of the bounding box.

### 2.3 Pre-training on Document Reading

#### Pre-training Tasks.

Traditional Optical Character Recognition (OCR) tasks primarily focus on generating line-level text content and capturing its spatial positions within an image. While OCR preserves the layout positions of document text, it often overlooks the document-level reading order and structural integrity, both crucial for comprehensive document understanding. In contrast, markdown-formatted text provides an advantage over plain text by explicitly distinguishing various structural elements, such as tables and lists, using specific tokens.

To effectively learn the layout and structure of documents, we propose two complementary generative tasks for pre-training a document foundation model: document-level text recognition and image-to-markdown generation, as detailed in Table[1](https://arxiv.org/html/2309.11419v2#S1.T1 "Table 1 ‣ 1 Introduction ‣ Kosmos-2.5: A Multimodal Literate Model").

#### Training Objective and Formats.

We train the model to predict outputs based on the input image context and task-specific prompts. The training objective is to minimize the cross-entropy loss for next-token prediction, commonly known as autoregressive language modeling(Radford et al. [2018](https://arxiv.org/html/2309.11419v2#bib.bib62)). Table[1](https://arxiv.org/html/2309.11419v2#S1.T1 "Table 1 ‣ 1 Introduction ‣ Kosmos-2.5: A Multimodal Literate Model") illustrates the formats for model training prompts and response sequences.

The prompt is constructed by concatenating the image representation with a task-specific special token. The response corresponds to the text output of the tasks: text lines with bounding boxes for document-level text recognition and markdown text for the image-to-markdown generation task. A qualitative example is provided in Appendix[A.4](https://arxiv.org/html/2309.11419v2#A1.SS4 "A.4 Qualitative Example ‣ Appendix A Appendix ‣ Kosmos-2.5: A Multimodal Literate Model") to illustrate the model’s input and output.

#### Pre-training Data.

Our training data is collected using an automated pipeline from diverse sources, resulting in a large corpus of 357.4 million document images, annotated with text lines using bounding boxes or in markdown format. As shown in Table[2](https://arxiv.org/html/2309.11419v2#S1.T2 "Table 2 ‣ 1 Introduction ‣ Kosmos-2.5: A Multimodal Literate Model"), our pre-training dataset encompasses a wide range of document types, including scanned documents, academic papers, web pages, design images, mathematical content, handwritten text, and more. Compared with the training data used by existing models in Table[3](https://arxiv.org/html/2309.11419v2#S1.T3 "Table 3 ‣ 1 Introduction ‣ Kosmos-2.5: A Multimodal Literate Model"), Kosmos-2.5 leverages the largest and most diverse corpus, which significantly enhances the model’s adaptability and generalization across different domains.

We apply filtering and quality control during data curation. We use fastText for language identification (with a threshold of 0.5) to filter out non-English documents from the entire pre-training dataset. To ensure content diversity within each source, we use MinHash(Broder [1997](https://arxiv.org/html/2309.11419v2#bib.bib5)) to identify and remove redundant pages, applying the same parameters as(Lee et al. [2021](https://arxiv.org/html/2309.11419v2#bib.bib35)), with document pairs having a similarity score of 0.8 or higher marked as duplicates.

For image-to-markdown data sourced from README, DOCX, L a T e X, and HTML files, we encountered discrepancies between the content in text images and their corresponding markdown sequences due to conversion issues. To refine the data, we evaluate token overlap between images and markdown files, requiring a token intersection-to-union ratio greater than 0.95 for inclusion. Details of the processing procedures for each document category are provided in Appendix[A.5](https://arxiv.org/html/2309.11419v2#A1.SS5 "A.5 Pre-training Data Processing ‣ Appendix A Appendix ‣ Kosmos-2.5: A Multimodal Literate Model"), along with sample training data in Appendix[A.8](https://arxiv.org/html/2309.11419v2#A1.SS8 "A.8 Pre-training Data Examples ‣ Appendix A Appendix ‣ Kosmos-2.5: A Multimodal Literate Model"), aiming to offer transparent and reproducible guidelines for future research and applications.

### 2.4 Fine-tuning on Document Understanding

We fine-tune Kosmos-2.5 on document understanding datasets, referring to the fine-tuned model as Kosmos-2.5-chat. Kosmos-2.5-chat is designed to answer diverse user-provided questions about text-intensive images from various domains. To better retain the reading capability of Kosmos-2.5, we freeze the visual encoder of the pre-trained model and fine-tune the resampler and language model using a document understanding task prompt (Line 3 in Table[1](https://arxiv.org/html/2309.11419v2#S1.T1 "Table 1 ‣ 1 Introduction ‣ Kosmos-2.5: A Multimodal Literate Model")), where [Question] and [Answer] represent a question-answer pair from the dataset.

3 Experiments
-------------

### 3.1 Model and Training Configurations

Following Pix2Struct(Lee et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib36)), we employ a short warmup phase of 20k steps to facilitate faster convergence during the pre-training stage. In this phase, the model learns to read text snippets from synthetic images rendered with random colors and fonts. Due to the substantially larger volume of layout-based data compared to markup-based data, we initially trained the model for 100k steps using only the layout-based dataset. We then combined the two datasets for an additional 140k steps of training. The total training involved approximately 260 billion tokens.

Our text tokenization is based on the cl100k_base tiktoken tokenizer 1 1 1[https://github.com/openai/tiktoken](https://github.com/openai/tiktoken), with 8,194 specialized tokens introduced for coordinates and bounding box markers. The newly added word embeddings for location tokens are randomly initialized, with all parameters updated during training. We also incorporate data augmentation techniques from TrOCR(Li et al. [2022b](https://arxiv.org/html/2309.11419v2#bib.bib44)) to enhance the model’s robustness.

Kosmos-2.5 contains a total of 1.3 billion parameters. Further details on model architecture and training hyperparameters are provided in Appendix[A.1](https://arxiv.org/html/2309.11419v2#A1.SS1 "A.1 Model and Training Hyperparameters ‣ Appendix A Appendix ‣ Kosmos-2.5: A Multimodal Literate Model").

Category Data Source Num
Handwritten Synthetic image 200
Design MARIO-LAION(Chen et al. [2024](https://arxiv.org/html/2309.11419v2#bib.bib7))200
MARIO-OpenLibrary(Chen et al. [2024](https://arxiv.org/html/2309.11419v2#bib.bib7))200
MARIO-TMDB(Chen et al. [2024](https://arxiv.org/html/2309.11419v2#bib.bib7))100
MJ&ST(Gupta [2016](https://arxiv.org/html/2309.11419v2#bib.bib21); Jaderberg [2014](https://arxiv.org/html/2309.11419v2#bib.bib30))200
Receipt Receipts crawled from the internet 100
CORD (Park et al. [2019](https://arxiv.org/html/2309.11419v2#bib.bib59))100
SROIE (Huang et al. [2019](https://arxiv.org/html/2309.11419v2#bib.bib28))347
Academic paper Academic papers from ArXiv 200
General Financial statements from SEC 200
General documents from Docx 200
FUNSD(Jaume [2019](https://arxiv.org/html/2309.11419v2#bib.bib31))50
Web Page Self-crawled web pages 200
Total 2,297

Table 4: Summary of document categories, data sources, and the number of samples in the OCREval benchmark.

Category Data Source Num
Math Image CROHME Math 1,000
Ima2LaTeX-100k 922
Academic Paper ArXiv 1,000
Table Table 771
General Document Docx 1,000
Project Document README 1,000
Total 5,693

Table 5: Summary of document categories, data sources, and the number of samples in the MarkdownEval benchmark.

Model Size Handwritten Design Receipt General Academic Web Image Overall Score (Avg)
Tesseract(Smith [2007](https://arxiv.org/html/2309.11419v2#bib.bib67))-42.3 / 58.1 / 62.6 24.3 / 26.2 / 26.4 63.7 / 49.3 / 65.1 92.1 / 56.8 / 86.7 78.1 / 56.9 / 91.7 90.4 / 54.0 / 75.5 65.2 / 50.2 / 68.0
Nougat(Blecher et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib4))350M 37.3 / - / 48.5 2.0 / - / 14.1 55.8 / - / 53.9 75.0 / - / 67.2 58.0 / - / 55.4 16.5 / - / 27.7 40.8 / - / 44.5
Vary(Wei et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib80))7B 28.0 / - / 62.4 43.1 / - / 75.8 31.8 / - / 62.4 55.9 / - / 54.2 45.6 / - / 49.4 10.1 / - / 26.4 35.8 / - / 55.1
Qwen-VL(Bai et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib3))9.6B 53.6 / 70.8 / 74.8 7.6 / 28.3 / 29.2 43.0 / 37.7 / 48.0 76.6 / 78.1 / 74.6 52.2 / 65.5 / 58.4 19.8 / 37.5 / 34.1 42.1 / 53.0 / 53.2
GPT-4o-66.0 / 23.1 / 87.4 74.6 / 15.5 / 82.1 83.6 / 8.6 / 75.4 91.9 / 19.5 / 86.8 69.5 / 22.3 / 75.7 51.1 / 9.4 / 55.9 72.8 / 16.4 / 77.2
KOSMOS-2.5 1.3B 71.6 / 94.1 / 90.6 61.7 / 80.2 / 79.6 89.4 / 80.1 / 83.3 97.6 / 89.8 / 93.9 98.8 / 93.3 / 99.1 57.0 / 72.1 / 69.6 79.4 / 84.9 / 86.0

Table 6: Experimental results for the document-level text recognition task on OCREval. Metrics are reported as F1↑↑\uparrow↑ / IOU↑↑\uparrow↑ / NED↑↑\uparrow↑. As Nougat and Vary produce only textual output without bounding boxes, IOU scores are not available for these models.

Model Docx README Arxiv Tables Math Equation CROHME Math Overall Score (Avg)
MSOCR+T5(Raffel et al. [2019](https://arxiv.org/html/2309.11419v2#bib.bib63))73.1 / 6.7 72.8 / 4.2 55.2 / 4.6 32.4 / 13.0 13.3 / 0.9 30.3 / 5.4 46.2 / 5.8
Nougat(Blecher et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib4))84.8 / 21.9 68.9 / 27.3 88.4 / 44.4 49.0 / 36.1 73.6 / 71.6 10.6 / 14.8 62.6 / 36.0
Vary(Wei et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib80))85.4 / 46.3 72.5 / 35.6 80.6 / 70.2 29.3 / 25.2 30.4 / 44.7 11.5 / 34.2 51.6 / 42.7
GPT-4o 2 2 2[https://openai.com/index/hello-gpt-4o/](https://openai.com/index/hello-gpt-4o/)85.3 / 20.5 83.5 / 49.3 76.7 / 23.0 74.7 / 42.4 56.5 / 78.2 64.7 / 84.2 73.6 / 49.6
KOSMOS-2.5 91.6 / 82.1 95.1 / 91.2 90.8 / 86.4 85.1 / 90.1 88.1 / 95.2 98.5 / 99.7 91.5 / 90.8

Table 7: Experimental results for document-level markdown generation on MDEval. Metrics are reported as NED↑↑\uparrow↑ / NTED↑↑\uparrow↑.

### 3.2 Evaluation on Document Reading

#### Benchmarks.

To comprehensively evaluate models’ capabilities in document-level text recognition and image-to-markdown generation tasks, we collected the OCREval and MarkdownEval benchmarks. The OCREval benchmark consists of 2,297 images from the test sets of 13 datasets, covering categories such as mathematical content, handwritten images, design images, receipts, digitally born documents, and web pages. The MarkdownEval benchmark includes 5,693 images spanning categories such as mathematical equations, academic papers, tables, general documents, and project documentation. The respective categories, data sources, and sample counts are detailed in Table[4](https://arxiv.org/html/2309.11419v2#S3.T4 "Table 4 ‣ 3.1 Model and Training Configurations ‣ 3 Experiments ‣ Kosmos-2.5: A Multimodal Literate Model") and Table[5](https://arxiv.org/html/2309.11419v2#S3.T5 "Table 5 ‣ 3.1 Model and Training Configurations ‣ 3 Experiments ‣ Kosmos-2.5: A Multimodal Literate Model"). More data processing details are provided in Appendix[A.6](https://arxiv.org/html/2309.11419v2#A1.SS6 "A.6 Evaluation Data Processing ‣ Appendix A Appendix ‣ Kosmos-2.5: A Multimodal Literate Model").

#### Metrics.

The metrics for OCREval include word-level F1, IOU, and NED to evaluate document-level OCR performance. The metrics for MarkdownEval include Normalized Edit Distance (NED) and Normalized Tree Edit Distance (NTED) for assessing image-to-markdown generation. NED is a string-based comparison metric, while NTED measures tree edit distance normalized by the number of nodes, capturing structural differences in parse trees. This dual evaluation framework considers both lexical accuracy and the preservation of the original hierarchical structure inherent in the Markdown format. Further details on the evaluation metrics are provided in Appendix[A.2](https://arxiv.org/html/2309.11419v2#A1.SS2 "A.2 OCR Evaluation Metrics ‣ Appendix A Appendix ‣ Kosmos-2.5: A Multimodal Literate Model") and Appendix[A.3](https://arxiv.org/html/2309.11419v2#A1.SS3 "A.3 Image-to-markdown Generation Evaluation Metrics ‣ Appendix A Appendix ‣ Kosmos-2.5: A Multimodal Literate Model").

##### Results.

Kosmos-2.5 is a unified framework that facilitates multitasking with tasks determined by the provided prompts. We compared Kosmos-2.5 against state-of-the-art document reading models on OCREval (Table[6](https://arxiv.org/html/2309.11419v2#S3.T6 "Table 6 ‣ 3.1 Model and Training Configurations ‣ 3 Experiments ‣ Kosmos-2.5: A Multimodal Literate Model")) and MarkdownEval (Table[7](https://arxiv.org/html/2309.11419v2#S3.T7 "Table 7 ‣ 3.1 Model and Training Configurations ‣ 3 Experiments ‣ Kosmos-2.5: A Multimodal Literate Model")). For the document-level text recognition task, Kosmos-2.5 outperforms existing models in reading text-intensive images. For instance, Kosmos-2.5 surpasses Vary Base Base{}_{\text{Base}}start_FLOATSUBSCRIPT Base end_FLOATSUBSCRIPT by a significant margin despite having a smaller model size (1.3B vs. 7B parameters). Kosmos-2.5 also achieved the best performance across all image types on MarkdownEval. Notably, GPT-4o’s omission of markdown symbols affected its NTED scores slightly. For example, while e^ 2 should be represented as e<sup>2</sup> in markdown, GPT-4o outputs e 2 superscript 𝑒 2 e^{2}italic_e start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT directly. For models adhering to markdown standards (e.g., Vary and Nougat), Kosmos-2.5 consistently outperforms them, benefiting from better layout understanding in text recognition.

### 3.3 Evaluation on Document Understanding

##### Settings.

Fine-tuned on downstream datasets, Kosmos-2.5-chat is capable of addressing a wide range of document understanding tasks. We fine-tuned Kosmos-2.5-chat on the standard training sets of ten diverse document understanding datasets. These datasets cover general documents (DocVQA(Mathew, Karatzas, and Jawahar [2020](https://arxiv.org/html/2309.11419v2#bib.bib57)), InfoVQA(Mathew et al. [2022](https://arxiv.org/html/2309.11419v2#bib.bib56)), DeepForm(Svetlichnaya [2020](https://arxiv.org/html/2309.11419v2#bib.bib71)), KLC(Stanislawek et al. [2021](https://arxiv.org/html/2309.11419v2#bib.bib68))), tables (WTQ(Pasupat and Liang [2015](https://arxiv.org/html/2309.11419v2#bib.bib60)), TabFact(Chen et al. [2020](https://arxiv.org/html/2309.11419v2#bib.bib11))), charts (ChartVQA(Masry et al. [2022](https://arxiv.org/html/2309.11419v2#bib.bib55))), natural images (TextVQA(Singh et al. [2019](https://arxiv.org/html/2309.11419v2#bib.bib66)), TextCaps(Sidorov et al. [2020](https://arxiv.org/html/2309.11419v2#bib.bib65))), and webpage screenshots (VisualMRC(Tanaka, Nishida, and Yoshida [2021](https://arxiv.org/html/2309.11419v2#bib.bib72))). Evaluation is performed on the official test sets of nine public document understanding benchmarks. We did not evaluate on TextCaps due to the unavailability of the official evaluation server at this time.

##### Results.

Table[8](https://arxiv.org/html/2309.11419v2#S3.T8 "Table 8 ‣ Results. ‣ 3.3 Evaluation on Document Understanding ‣ 3 Experiments ‣ Kosmos-2.5: A Multimodal Literate Model") presents the experimental results compared to state-of-the-art OCR-free models. Among models with fewer than 2B parameters, Kosmos-2.5-chat outperforms PixStruct LARGE LARGE{}_{\text{LARGE}}start_FLOATSUBSCRIPT LARGE end_FLOATSUBSCRIPT and Donut across various benchmarks without task-specific fine-tuning. Compared to models exceeding 7B parameters, Kosmos-2.5-chat delivers competitive performance on benchmarks covering documents, tables, and charts, including DocVQA, InfoVQA, DeepForm, KLC, WTQ, and ChartVQA. These results highlight the effectiveness of Kosmos-2.5-chat in handling complex document understanding tasks.

Model Size Doc Info Deep KLC WTQ Tab Chart Text Visual
VQA VQA Form Fact QA VQA MRC
DocPeida(Feng et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib17))7.0B 47.1 15.2----46.9 60.2-
DocOwl(Ye et al. [2023a](https://arxiv.org/html/2309.11419v2#bib.bib89))7.1B 62.2 38.2 42.6 30.3 26.9 60.2 57.4 52.6 188.8
QwenVL(Bai et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib3))9.6B 65.1 35.4----65.7 63.8-
UReader(Ye et al. [2023b](https://arxiv.org/html/2309.11419v2#bib.bib90))7.1B 65.4 42.2 49.5 32.8 29.4 67.6 59.3 57.6 221.7
Monkey(Li et al. [2023c](https://arxiv.org/html/2309.11419v2#bib.bib46))9.8B 66.5 36.1 40.6 32.8 25.3--67.6-
HRVDA(Liu et al. [2024a](https://arxiv.org/html/2309.11419v2#bib.bib48))7.1B 72.1 43.5 63.2 37.5 31.2 72.3 67.6 73.3 211.5
DocOwl-1.5(Hu et al. [2024](https://arxiv.org/html/2309.11419v2#bib.bib24))8.1B 81.6 50.4 68.8 37.9 39.8 80.4 70.5 68.8 239.5
CogAgent(Hong et al. [2024](https://arxiv.org/html/2309.11419v2#bib.bib23))17.3B 81.6 44.5----68.4 76.1-
Donut∗(Kim et al. [2021](https://arxiv.org/html/2309.11419v2#bib.bib32))<<<1B 67.5 11.6 61.6 30.0 18.8 54.6 41.8 43.5 93.9
Dessurt∗(Davis et al. [2022](https://arxiv.org/html/2309.11419v2#bib.bib14))<<<1B 63.2--------
Pix2Struct LARGE∗subscript superscript absent LARGE{}^{*}_{\text{LARGE}}start_FLOATSUPERSCRIPT ∗ end_FLOATSUPERSCRIPT start_POSTSUBSCRIPT LARGE end_POSTSUBSCRIPT(Lee et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib36))1.3B 76.6 40.0----58.6--
Vary-toy(Wei et al. [2024](https://arxiv.org/html/2309.11419v2#bib.bib81))1.8B 65.6-----59.1--
MiniCPM-V 2.0(Yao et al. [2024](https://arxiv.org/html/2309.11419v2#bib.bib88))2.8B 71.9-----55.6 74.1-
Kosmos-2.5-chat 1.3B 81.1 41.3 65.8 35.1 32.4 49.9 62.3 40.7 156.0

Table 8: Experimental results on document understanding benchmarks. The models listed above the line have more than 7B parameters, while those below the line are smaller models. The superscript ‘∗*∗’ indicates models fine-tuned separately on each downstream task. Among models with fewer than 7B parameters, the best results are marked in bold.

4 Related Work
--------------

### 4.1 Multimodal Large Language Models

Multimodal large language models (MLLMs) can be broadly categorized into LLM-centric scheduling systems and end-to-end trainable multimodal systems. LLM-centric scheduling systems leverage various vision foundation models, orchestrating them in a language-centric manner(Wu et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib82); Yang et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib87); Liang et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib47); Shen et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib64); Liu et al. [2023c](https://arxiv.org/html/2309.11419v2#bib.bib53); Surís, Menon, and Vondrick [2023](https://arxiv.org/html/2309.11419v2#bib.bib70); Chen et al. [2023b](https://arxiv.org/html/2309.11419v2#bib.bib10)). On the other hand, end-to-end trainable multimodal systems integrate vision and language models into a unified framework(Hao et al. [2022](https://arxiv.org/html/2309.11419v2#bib.bib22); Alayrac et al. [2022](https://arxiv.org/html/2309.11419v2#bib.bib1); Huang et al. [2023a](https://arxiv.org/html/2309.11419v2#bib.bib25); Peng et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib61); Huang et al. [2021](https://arxiv.org/html/2309.11419v2#bib.bib29); Xue et al. [2021](https://arxiv.org/html/2309.11419v2#bib.bib86); Zhu et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib97); Huang et al. [2023b](https://arxiv.org/html/2309.11419v2#bib.bib27); Li et al. [2023b](https://arxiv.org/html/2309.11419v2#bib.bib40); Dai et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib13); Liu et al. [2023a](https://arxiv.org/html/2309.11419v2#bib.bib49); Luo et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib54); Wang et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib78); Su et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib69); Zhang et al. [2023a](https://arxiv.org/html/2309.11419v2#bib.bib93); Gao et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib18); Koh, Salakhutdinov, and Fried [2023](https://arxiv.org/html/2309.11419v2#bib.bib33); Li et al. [2023a](https://arxiv.org/html/2309.11419v2#bib.bib38); Tang et al. [2024b](https://arxiv.org/html/2309.11419v2#bib.bib75), [2023](https://arxiv.org/html/2309.11419v2#bib.bib74); Kondratyuk et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib34); Bai et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib3); Hong et al. [2024](https://arxiv.org/html/2309.11419v2#bib.bib23); Yao et al. [2024](https://arxiv.org/html/2309.11419v2#bib.bib88); Wei et al. [2024](https://arxiv.org/html/2309.11419v2#bib.bib81)).

Our model falls into the latter category, sharing similarities with grounded multimodal models like KOSMOS-2(Peng et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib61)), Shikra(Chen et al. [2023a](https://arxiv.org/html/2309.11419v2#bib.bib9)), and ChatSpot(Zhao et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib96)), which output object locations in natural images. However, Kosmos-2.5 uniquely focuses on text-image reading and understanding capabilities, tackling the challenge of producing high-quality document layouts while maintaining the structural integrity crucial for document understanding.

### 4.2 Document Reading and Understanding

Document reading and understanding leverage AI to automatically read, comprehend, and extract information from documents(Cui et al. [2021](https://arxiv.org/html/2309.11419v2#bib.bib12); Xu et al. [2020](https://arxiv.org/html/2309.11419v2#bib.bib83), [2021b](https://arxiv.org/html/2309.11419v2#bib.bib85), [2021a](https://arxiv.org/html/2309.11419v2#bib.bib84); Huang et al. [2022](https://arxiv.org/html/2309.11419v2#bib.bib26); Kim et al. [2021](https://arxiv.org/html/2309.11419v2#bib.bib32); Chen et al. [2022](https://arxiv.org/html/2309.11419v2#bib.bib8); Li et al. [2021b](https://arxiv.org/html/2309.11419v2#bib.bib41), [2022a](https://arxiv.org/html/2309.11419v2#bib.bib42), [c](https://arxiv.org/html/2309.11419v2#bib.bib45); Appalaraju et al. [2021](https://arxiv.org/html/2309.11419v2#bib.bib2); Wang, Jin, and Ding [2022](https://arxiv.org/html/2309.11419v2#bib.bib77); Gu et al. [2022](https://arxiv.org/html/2309.11419v2#bib.bib20); Li et al. [2021a](https://arxiv.org/html/2309.11419v2#bib.bib39); Chen et al. [2024](https://arxiv.org/html/2309.11419v2#bib.bib7); Yu et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib91); Li et al. [2023c](https://arxiv.org/html/2309.11419v2#bib.bib46); Liu et al. [2024c](https://arxiv.org/html/2309.11419v2#bib.bib52)). Representative document foundation models like LayoutLMv3 integrate text, layout, and image information during pre-training, excelling in tasks like key information extraction and document question answering(Huang et al. [2022](https://arxiv.org/html/2309.11419v2#bib.bib26)). Donut(Kim et al. [2021](https://arxiv.org/html/2309.11419v2#bib.bib32)) introduces an OCR-free document understanding Transformer, directly mapping input document images to desired outputs. Models like Pix2Struct(Lee et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib36)), HRVDA(Liu et al. [2024a](https://arxiv.org/html/2309.11419v2#bib.bib48)), and the mPLUG-DocOwl series(Ye et al. [2023a](https://arxiv.org/html/2309.11419v2#bib.bib89); Hu et al. [2024](https://arxiv.org/html/2309.11419v2#bib.bib24)) pre-train vision encoders on document reading tasks, resulting in impressive document understanding performance. Kosmos-2.5 scales up document pre-training to include up to 357.4 million document pages and more challenging tasks, significantly enhancing the model’s reading and understanding capabilities.

Nougat(Blecher et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib4)) similarly parses documents into markup language, but its focus is limited to scientific documents. In contrast, Kosmos-2.5 excels across a broader range of documents and generalizes well to document understanding tasks. Recent works like DocPedia(Feng et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib17)) enhance MLLMs’ text-rich image understanding by processing visual input in the frequency domain for high-resolution capabilities. Approaches like TextSquare(Tang et al. [2024a](https://arxiv.org/html/2309.11419v2#bib.bib73)), TRINS(Zhang et al. [2024](https://arxiv.org/html/2309.11419v2#bib.bib94)), and LLaVAR(Zhang et al. [2023b](https://arxiv.org/html/2309.11419v2#bib.bib95)) enhance reading abilities by using publicly available OCR tools and closed-source MLLMs to generate instruction-tuning data for text-rich images. LLaVA-read further uses open-source OCR tools to extract text and layout information for language models. UReader(Ye et al. [2023b](https://arxiv.org/html/2309.11419v2#bib.bib90)) introduces a shape-adaptive cropping module to efficiently encode low-resolution sub-images. Meanwhile, Monkey(Li et al. [2023c](https://arxiv.org/html/2309.11419v2#bib.bib46)) boosts training efficiency and resolution, excelling in image captioning and text-rich document processing. However, these methods rely on pre-trained vision encoders without document-specific pre-training, which limits their performance. After extensive pre-training, Kosmos-2.5 achieves strong document understanding performance by fine-tuning on publicly available benchmarks only, without needing complex module designs, OCR tools, or closed-source MLLMs.

### 4.3 Document Reading Benchmarks

Existing OCR evaluation benchmarks like OCRBench(Liu et al. [2023b](https://arxiv.org/html/2309.11419v2#bib.bib51)) or DocLocal4K(Hu et al. [2024](https://arxiv.org/html/2309.11419v2#bib.bib24)) mainly focus on text-line recognition tasks. Textmonkey(Liu et al. [2024c](https://arxiv.org/html/2309.11419v2#bib.bib52)) evaluates the model on natural images only. In contrast, our proposed OCREval is the first benchmark specifically designed to assess document-level text recognition, which demands more advanced recognition capabilities. For markdown evaluation, Nougat(Blecher et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib4)) restricts its performance assessment to academic papers from ArXiv. In contrast, our MarkdownEval offers a more comprehensive assessment by covering a wider range of image domains, providing a more robust assessment of model capabilities.

5 Conclusion and Future Work
----------------------------

In summary, this work advances document-level machine reading by introducing a novel pre-training framework and demonstrating its effectiveness through impressive performance on diverse benchmarks. Our pre-trained model, Kosmos-2.5, excels in document reading, while our fine-tuned model, Kosmos-2.5-chat, achieves competitive results in document understanding benchmarks. The extensive corpus of 357.4 million annotated document images and the development of OCREval and MarkdownEval benchmarks provide comprehensive tools for evaluating and furthering research in document intelligence. Despite these promising results, our current model faces some limitations, offering valuable future research directions. For instance, documents spanning multiple pages pose a challenge as they typically demand holistic processing and comprehension. Meanwhile, it is also feasible that Kosmos-2.5 allows for multiple image pages interleaved with text as input; however, managing long context remains a vital issue we aim to address in future work. In the broader research landscape, a significant direction lies in advancing model scaling capabilities. With an expanding range of tasks and complexities, scaling the model to handle larger data volumes is crucial for multimodal literate models.

References
----------

*   Alayrac et al. (2022) Alayrac, J.-B.; Donahue, J.; Luc, P.; Miech, A.; Barr, I.; Hasson, Y.; Lenc, K.; Mensch, A.; Millican, K.; Reynolds, M.; et al. 2022. Flamingo: a visual language model for few-shot learning. _Advances in Neural Information Processing Systems_, 35: 23716–23736. 
*   Appalaraju et al. (2021) Appalaraju, S.; Jasani, B.; Kota, B.U.; Xie, Y.; and Manmatha, R. 2021. Docformer: End-to-end transformer for document understanding. In _Proceedings of the IEEE/CVF international conference on computer vision_, 993–1003. 
*   Bai et al. (2023) Bai, J.; Bai, S.; Yang, S.; Wang, S.; Tan, S.; Wang, P.; Lin, J.; Zhou, C.; and Zhou, J. 2023. Qwen-vl: A frontier large vision-language model with versatile abilities. _arXiv preprint arXiv:2308.12966_. 
*   Blecher et al. (2023) Blecher, L.; Cucurull, G.; Scialom, T.; and Stojnic, R. 2023. Nougat: Neural Optical Understanding for Academic Documents. arXiv:2308.13418. 
*   Broder (1997) Broder, A.Z. 1997. On the resemblance and containment of documents. In _Proceedings. Compression and Complexity of SEQUENCES 1997 (Cat. No. 97TB100171)_, 21–29. IEEE. 
*   ChatGPT (2022) ChatGPT. 2022. https://openai.com/blog/chatgpt. 
*   Chen et al. (2024) Chen, J.; Huang, Y.; Lv, T.; Cui, L.; Chen, Q.; and Wei, F. 2024. Textdiffuser: Diffusion models as text painters. _Advances in Neural Information Processing Systems_, 36. 
*   Chen et al. (2022) Chen, J.; Lv, T.; Cui, L.; Zhang, C.; and Wei, F. 2022. Xdoc: Unified pre-training for cross-format document understanding. _arXiv preprint arXiv:2210.02849_. 
*   Chen et al. (2023a) Chen, K.; Zhang, Z.; Zeng, W.; Zhang, R.; Zhu, F.; and Zhao, R. 2023a. Shikra: Unleashing multimodal llm’s referential dialogue magic. _arXiv preprint arXiv:2306.15195_. 
*   Chen et al. (2023b) Chen, L.; Li, B.; Shen, S.; Yang, J.; Li, C.; Keutzer, K.; Darrell, T.; and Liu, Z. 2023b. Language Models are Visual Reasoning Coordinators. In _ICLR 2023 Workshop on Mathematical and Empirical Understanding of Foundation Models_. 
*   Chen et al. (2020) Chen, W.; Wang, H.; Chen, J.; Zhang, Y.; Wang, H.; Li, S.; Zhou, X.; and Wang, W.Y. 2020. TabFact : A Large-scale Dataset for Table-based Fact Verification. In _International Conference on Learning Representations (ICLR)_. Addis Ababa, Ethiopia. 
*   Cui et al. (2021) Cui, L.; Xu, Y.; Lv, T.; and Wei, F. 2021. Document AI: Benchmarks, Models and Applications. arXiv:2111.08609. 
*   Dai et al. (2023) Dai, W.; Li, J.; Li, D.; Tiong, A. M.H.; Zhao, J.; Wang, W.; Li, B.; Fung, P.; and Hoi, S. 2023. InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning. arXiv:2305.06500. 
*   Davis et al. (2022) Davis, B.; Morse, B.; Price, B.; Tensmeyer, C.; Wigington, C.; and Morariu, V. 2022. End-to-end document recognition and understanding with dessurt. In _European Conference on Computer Vision_, 280–296. Springer. 
*   Deng et al. (2017) Deng, Y.; Kanervisto, A.; Ling, J.; and Rush, A.M. 2017. Image-to-markup generation with coarse-to-fine attention. In _International Conference on Machine Learning_, 980–989. PMLR. 
*   Dosovitskiy et al. (2021) Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; Dehghani, M.; Minderer, M.; Heigold, G.; Gelly, S.; Uszkoreit, J.; and Houlsby, N. 2021. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In _ICLR_. 
*   Feng et al. (2023) Feng, H.; Liu, Q.; Liu, H.; Zhou, W.; Li, H.; and Huang, C. 2023. Docpedia: Unleashing the power of large multimodal model in the frequency domain for versatile document understanding. _arXiv preprint arXiv:2311.11810_. 
*   Gao et al. (2023) Gao, P.; Han, J.; Zhang, R.; Lin, Z.; Geng, S.; Zhou, A.; Zhang, W.; Lu, P.; He, C.; Yue, X.; et al. 2023. Llama-adapter v2: Parameter-efficient visual instruction model. _arXiv preprint arXiv:2304.15010_. 
*   GPT-4 (2023) GPT-4. 2023. https://openai.com/gpt-4. 
*   Gu et al. (2022) Gu, Z.; Meng, C.; Wang, K.; Lan, J.; Wang, W.; Gu, M.; and Zhang, L. 2022. Xylayoutlm: Towards layout-aware multimodal networks for visually-rich document understanding. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, 4583–4592. 
*   Gupta (2016) Gupta, e.a. 2016. Synthetic data for text localisation in natural images. In _Proceedings of the IEEE conference on computer vision and pattern recognition_, 2315–2324. 
*   Hao et al. (2022) Hao, Y.; Song, H.; Dong, L.; Huang, S.; Chi, Z.; Wang, W.; Ma, S.; and Wei, F. 2022. Language Models are General-Purpose Interfaces. _ArXiv_, abs/2206.06336. 
*   Hong et al. (2024) Hong, W.; Wang, W.; Lv, Q.; Xu, J.; Yu, W.; Ji, J.; Wang, Y.; Wang, Z.; Dong, Y.; Ding, M.; et al. 2024. Cogagent: A visual language model for gui agents. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, 14281–14290. 
*   Hu et al. (2024) Hu, A.; Xu, H.; Ye, J.; Yan, M.; Zhang, L.; Zhang, B.; Li, C.; Zhang, J.; Jin, Q.; Huang, F.; et al. 2024. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. _arXiv preprint arXiv:2403.12895_. 
*   Huang et al. (2023a) Huang, S.; Dong, L.; Wang, W.; Hao, Y.; Singhal, S.; Ma, S.; Lv, T.; Cui, L.; Mohammed, O.K.; Liu, Q.; et al. 2023a. Language is not all you need: Aligning perception with language models. _arXiv preprint arXiv:2302.14045_. 
*   Huang et al. (2022) Huang, Y.; Lv, T.; Cui, L.; Lu, Y.; and Wei, F. 2022. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. In _Proceedings of the 30th ACM International Conference on Multimedia_. 
*   Huang et al. (2023b) Huang, Y.; Meng, Z.; Liu, F.; Su, Y.; Nigel, C.; and Lu, Y. 2023b. Sparkles: Unlocking Chats Across Multiple Images for Multimodal Instruction-Following Models. _arXiv preprint arXiv:2308.16463_. 
*   Huang et al. (2019) Huang, Z.; Chen, K.; He, J.; Bai, X.; Karatzas, D.; Lu, S.; and Jawahar, C. 2019. Icdar2019 competition on scanned receipt ocr and information extraction. In _2019 International Conference on Document Analysis and Recognition (ICDAR)_, 1516–1520. IEEE. 
*   Huang et al. (2021) Huang, Z.; Zeng, Z.; Huang, Y.; Liu, B.; Fu, D.; and Fu, J. 2021. Seeing out of the box: End-to-end pre-training for vision-language representation learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, 12976–12985. 
*   Jaderberg (2014) Jaderberg, e.a. 2014. Synthetic data and artificial neural networks for natural scene text recognition. _arXiv preprint arXiv:1406.2227_. 
*   Jaume (2019) Jaume, e.a. 2019. Funsd: A dataset for form understanding in noisy scanned documents. In _2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)_, volume 2, 1–6. IEEE. 
*   Kim et al. (2021) Kim, G.; Hong, T.; Yim, M.; Park, J.; Yim, J.; Hwang, W.; Yun, S.; Han, D.; and Park, S. 2021. Donut: Document understanding transformer without ocr. _arXiv preprint arXiv:2111.15664_, 7: 15. 
*   Koh, Salakhutdinov, and Fried (2023) Koh, J.Y.; Salakhutdinov, R.; and Fried, D. 2023. Grounding language models to images for multimodal generation. _arXiv preprint arXiv:2301.13823_. 
*   Kondratyuk et al. (2023) Kondratyuk, D.; Yu, L.; Gu, X.; Lezama, J.; Huang, J.; Hornung, R.; Adam, H.; Akbari, H.; Alon, Y.; Birodkar, V.; et al. 2023. Videopoet: A large language model for zero-shot video generation. _arXiv preprint arXiv:2312.14125_. 
*   Lee et al. (2021) Lee, K.; Ippolito, D.; Nystrom, A.; Zhang, C.; Eck, D.; Callison-Burch, C.; and Carlini, N. 2021. Deduplicating training data makes language models better. _arXiv preprint arXiv:2107.06499_. 
*   Lee et al. (2023) Lee, K.; Joshi, M.; Turc, I.R.; Hu, H.; Liu, F.; Eisenschlos, J.M.; Khandelwal, U.; Shaw, P.; Chang, M.-W.; and Toutanova, K. 2023. Pix2struct: Screenshot parsing as pretraining for visual language understanding. In _International Conference on Machine Learning_, 18893–18912. PMLR. 
*   Lewis et al. (2006) Lewis, D.; Agam, G.; Argamon, S.; Frieder, O.; Grossman, D.; and Heard, J. 2006. Building a test collection for complex document information processing. In _Proceedings of the 29th annual international ACM SIGIR conference on Research and development in information retrieval_, 665–666. 
*   Li et al. (2023a) Li, B.; Zhang, Y.; Chen, L.; Wang, J.; Yang, J.; and Liu, Z. 2023a. Otter: A multi-modal model with in-context instruction tuning. _arXiv preprint arXiv:2305.03726_. 
*   Li et al. (2021a) Li, C.; Bi, B.; Yan, M.; Wang, W.; Huang, S.; Huang, F.; and Si, L. 2021a. Structurallm: Structural pre-training for form understanding. _arXiv preprint arXiv:2105.11210_. 
*   Li et al. (2023b) Li, J.; Li, D.; Savarese, S.; and Hoi, S. 2023b. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. _arXiv preprint arXiv:2301.12597_. 
*   Li et al. (2021b) Li, J.; Xu, Y.; Cui, L.; and Wei, F. 2021b. Markuplm: Pre-training of text and markup language for visually-rich document understanding. _arXiv preprint arXiv:2110.08518_. 
*   Li et al. (2022a) Li, J.; Xu, Y.; Lv, T.; Cui, L.; Zhang, C.; and Wei, F. 2022a. Dit: Self-supervised pre-training for document image transformer. In _Proceedings of the 30th ACM International Conference on Multimedia_, 3530–3539. 
*   Li et al. (2020) Li, M.; Cui, L.; Huang, S.; Wei, F.; Zhou, M.; and Li, Z. 2020. TableBank: A Benchmark Dataset for Table Detection and Recognition. arXiv:1903.01949. 
*   Li et al. (2022b) Li, M.; Lv, T.; Chen, J.; Cui, L.; Lu, Y.; Florencio, D.; Zhang, C.; Li, Z.; and Wei, F. 2022b. TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models. arXiv:2109.10282. 
*   Li et al. (2021c) Li, P.; Gu, J.; Kuen, J.; Morariu, V.I.; Zhao, H.; Jain, R.; Manjunatha, V.; and Liu, H. 2021c. Selfdoc: Self-supervised document representation learning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, 5652–5660. 
*   Li et al. (2023c) Li, Z.; Yang, B.; Liu, Q.; Ma, Z.; Zhang, S.; Yang, J.; Sun, Y.; Liu, Y.; and Bai, X. 2023c. Monkey: Image resolution and text label are important things for large multi-modal models. _arXiv preprint arXiv:2311.06607_. 
*   Liang et al. (2023) Liang, Y.; Wu, C.; Song, T.; Wu, W.; Xia, Y.; Liu, Y.; Ou, Y.; Lu, S.; Ji, L.; Mao, S.; et al. 2023. Taskmatrix. ai: Completing tasks by connecting foundation models with millions of apis. _arXiv preprint arXiv:2303.16434_. 
*   Liu et al. (2024a) Liu, C.; Yin, K.; Cao, H.; Jiang, X.; Li, X.; Liu, Y.; Jiang, D.; Sun, X.; and Xu, L. 2024a. Hrvda: High-resolution visual document assistant. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, 15534–15545. 
*   Liu et al. (2023a) Liu, H.; Li, C.; Wu, Q.; and Lee, Y.J. 2023a. Visual instruction tuning. _arXiv preprint arXiv:2304.08485_. 
*   Liu et al. (2024b) Liu, H.; Li, C.; Wu, Q.; and Lee, Y.J. 2024b. Visual instruction tuning. _Advances in neural information processing systems_, 36. 
*   Liu et al. (2023b) Liu, Y.; Li, Z.; Li, H.; Yu, W.; Huang, M.; Peng, D.; Liu, M.; Chen, M.; Li, C.; Jin, L.; and Bai, X. 2023b. On the Hidden Mystery of OCR in Large Multimodal Models. _ArXiv_, abs/2305.07895. 
*   Liu et al. (2024c) Liu, Y.; Yang, B.; Liu, Q.; Li, Z.; Ma, Z.; Zhang, S.; and Bai, X. 2024c. TextMonkey: An OCR-Free Large Multimodal Model for Understanding Document. _arXiv preprint arXiv:2403.04473_. 
*   Liu et al. (2023c) Liu, Z.; He, Y.; Wang, W.; Wang, W.; Wang, Y.; Chen, S.; Zhang, Q.; Yang, Y.; Li, Q.; Yu, J.; et al. 2023c. Internchat: Solving vision-centric tasks by interacting with chatbots beyond language. _arXiv preprint arXiv:2305.05662_. 
*   Luo et al. (2023) Luo, G.; Zhou, Y.; Ren, T.; Chen, S.; Sun, X.; and Ji, R. 2023. Cheap and quick: Efficient vision-language instruction tuning for large language models. _arXiv preprint arXiv:2305.15023_. 
*   Masry et al. (2022) Masry, A.; Long, D.X.; Tan, J.Q.; Joty, S.R.; and Hoque, E. 2022. ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning. In _ACL (Findings)_, 2263–2279. Association for Computational Linguistics. 
*   Mathew et al. (2022) Mathew, M.; Bagal, V.; Tito, R.; Karatzas, D.; Valveny, E.; and Jawahar, C.V. 2022. InfographicVQA. In _WACV_, 2582–2591. IEEE. 
*   Mathew, Karatzas, and Jawahar (2020) Mathew, M.; Karatzas, D.; and Jawahar, C.V. 2020. DocVQA: A Dataset for VQA on Document Images. arXiv:2007.00398. 
*   Mouchere et al. (2014) Mouchere, H.; Viard-Gaudin, C.; Zanibbi, R.; and Garain, U. 2014. ICFHR 2014 competition on recognition of on-line handwritten mathematical expressions (CROHME 2014). In _2014 14th International Conference on Frontiers in Handwriting Recognition_, 791–796. IEEE. 
*   Park et al. (2019) Park, S.; Shin, S.; Lee, B.; Lee, J.; Surh, J.; Seo, M.; and Lee, H. 2019. CORD: A Consolidated Receipt Dataset for Post-OCR Parsing. _Document Intelligence Workshop at Neural Information Processing Systems_. 
*   Pasupat and Liang (2015) Pasupat, P.; and Liang, P. 2015. Compositional Semantic Parsing on Semi-Structured Tables. In _ACL (1)_, 1470–1480. The Association for Computer Linguistics. 
*   Peng et al. (2023) Peng, Z.; Wang, W.; Dong, L.; Hao, Y.; Huang, S.; Ma, S.; and Wei, F. 2023. Kosmos-2: Grounding Multimodal Large Language Models to the World. _arXiv preprint arXiv:2306.14824_. 
*   Radford et al. (2018) Radford, A.; Narasimhan, K.; Salimans, T.; Sutskever, I.; et al. 2018. Improving language understanding by generative pre-training. 
*   Raffel et al. (2019) Raffel, C.; Shazeer, N.M.; Roberts, A.; Lee, K.; Narang, S.; Matena, M.; Zhou, Y.; Li, W.; and Liu, P.J. 2019. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. _J. Mach. Learn. Res._, 21: 140:1–140:67. 
*   Shen et al. (2023) Shen, Y.; Song, K.; Tan, X.; Li, D.; Lu, W.; and Zhuang, Y. 2023. Hugginggpt: Solving ai tasks with chatgpt and its friends in huggingface. _arXiv preprint arXiv:2303.17580_. 
*   Sidorov et al. (2020) Sidorov, O.; Hu, R.; Rohrbach, M.; and Singh, A. 2020. TextCaps: A Dataset for Image Captioning with Reading Comprehension. In _ECCV (2)_, volume 12347 of _Lecture Notes in Computer Science_, 742–758. Springer. 
*   Singh et al. (2019) Singh, A.; Natarajan, V.; Shah, M.; Jiang, Y.; Chen, X.; Batra, D.; Parikh, D.; and Rohrbach, M. 2019. Towards VQA Models That Can Read. _2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_. 
*   Smith (2007) Smith, R. 2007. An Overview of the Tesseract OCR Engine. In _ICDAR ’07: Proceedings of the Ninth International Conference on Document Analysis and Recognition_, 629–633. Washington, DC, USA: IEEE Computer Society. ISBN 0-7695-2822-8. 
*   Stanislawek et al. (2021) Stanislawek, T.; Gralinski, F.; Wróblewska, A.; Lipinski, D.; Kaliska, A.; Rosalska, P.; Topolski, B.; and Biecek, P. 2021. Kleister: Key Information Extraction Datasets Involving Long Documents with Complex Layouts. In _ICDAR (1)_, volume 12821 of _Lecture Notes in Computer Science_, 564–579. Springer. 
*   Su et al. (2023) Su, Y.; Lan, T.; Li, H.; Xu, J.; Wang, Y.; and Cai, D. 2023. Pandagpt: One model to instruction-follow them all. _arXiv preprint arXiv:2305.16355_. 
*   Surís, Menon, and Vondrick (2023) Surís, D.; Menon, S.; and Vondrick, C. 2023. Vipergpt: Visual inference via python execution for reasoning. _arXiv preprint arXiv:2303.08128_. 
*   Svetlichnaya (2020) Svetlichnaya, S. 2020. DeepForm: Understand structured documents at scale. 
*   Tanaka, Nishida, and Yoshida (2021) Tanaka, R.; Nishida, K.; and Yoshida, S. 2021. VisualMRC: Machine Reading Comprehension on Document Images. In _AAAI_, 13878–13888. AAAI Press. 
*   Tang et al. (2024a) Tang, J.; Lin, C.; Zhao, Z.; Wei, S.; Wu, B.; Liu, Q.; Feng, H.; Li, Y.; Wang, S.; Liao, L.; et al. 2024a. TextSquare: Scaling up Text-Centric Visual Instruction Tuning. _arXiv preprint arXiv:2404.12803_. 
*   Tang et al. (2023) Tang, Z.; Yang, Z.; Khademi, M.; Liu, Y.; Zhu, C.; and Bansal, M. 2023. Codi-2: In-context, interleaved, and interactive any-to-any generation. _arXiv preprint arXiv:2311.18775_. 
*   Tang et al. (2024b) Tang, Z.; Yang, Z.; Zhu, C.; Zeng, M.; and Bansal, M. 2024b. Any-to-any generation via composable diffusion. _Advances in Neural Information Processing Systems_, 36. 
*   Touvron et al. (2023) Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux, M.-A.; Lacroix, T.; Rozière, B.; Goyal, N.; Hambro, E.; Azhar, F.; et al. 2023. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_. 
*   Wang, Jin, and Ding (2022) Wang, J.; Jin, L.; and Ding, K. 2022. Lilt: A simple yet effective language-independent layout transformer for structured document understanding. _arXiv preprint arXiv:2202.13669_. 
*   Wang et al. (2023) Wang, W.; Chen, Z.; Chen, X.; Wu, J.; Zhu, X.; Zeng, G.; Luo, P.; Lu, T.; Zhou, J.; Qiao, Y.; et al. 2023. Visionllm: Large language model is also an open-ended decoder for vision-centric tasks. _arXiv preprint arXiv:2305.11175_. 
*   Wang et al. (2021) Wang, Z.; Xu, Y.; Cui, L.; Shang, J.; and Wei, F. 2021. LayoutReader: Pre-training of Text and Layout for Reading Order Detection. In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_, 4735–4744. 
*   Wei et al. (2023) Wei, H.; Kong, L.; Chen, J.; Zhao, L.; Ge, Z.; Yang, J.; Sun, J.; Han, C.; and Zhang, X. 2023. Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models. _ArXiv_, abs/2312.06109. 
*   Wei et al. (2024) Wei, H.; Kong, L.; Chen, J.; Zhao, L.; Ge, Z.; Yu, E.; Sun, J.; Han, C.; and Zhang, X. 2024. Small Language Model Meets with Reinforced Vision Vocabulary. arXiv:2401.12503. 
*   Wu et al. (2023) Wu, C.; Yin, S.; Qi, W.; Wang, X.; Tang, Z.; and Duan, N. 2023. Visual chatgpt: Talking, drawing and editing with visual foundation models. _arXiv preprint arXiv:2303.04671_. 
*   Xu et al. (2020) Xu, Y.; Li, M.; Cui, L.; Huang, S.; Wei, F.; and Zhou, M. 2020. Layoutlm: Pre-training of text and layout for document image understanding. In _Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_, 1192–1200. 
*   Xu et al. (2021a) Xu, Y.; Lv, T.; Cui, L.; Wang, G.; Lu, Y.; Florencio, D.; Zhang, C.; and Wei, F. 2021a. LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding. arXiv:2104.08836. 
*   Xu et al. (2021b) Xu, Y.; Xu, Y.; Lv, T.; Cui, L.; Wei, F.; Wang, G.; Lu, Y.; Florencio, D.; Zhang, C.; Che, W.; Zhang, M.; and Zhou, L. 2021b. LayoutLMv2: Multi-modal Pre-training for Visually-rich Document Understanding. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_, 2579–2591. Online: Association for Computational Linguistics. 
*   Xue et al. (2021) Xue, H.; Huang, Y.; Liu, B.; Peng, H.; Fu, J.; Li, H.; and Luo, J. 2021. Probing inter-modality: Visual parsing with self-attention for vision-and-language pre-training. In _Advances in Neural Information Processing Systems_, volume 34, 4514–4528. 
*   Yang et al. (2023) Yang, Z.; Li, L.; Wang, J.; Lin, K.; Azarnasab, E.; Ahmed, F.; Liu, Z.; Liu, C.; Zeng, M.; and Wang, L. 2023. Mm-react: Prompting chatgpt for multimodal reasoning and action. _arXiv preprint arXiv:2303.11381_. 
*   Yao et al. (2024) Yao, Y.; Yu, T.; Zhang, A.; Wang, C.; Cui, J.; Zhu, H.; Cai, T.; Li, H.; Zhao, W.; He, Z.; Chen, Q.; Zhou, H.; Zou, Z.; Zhang, H.; Hu, S.; Zheng, Z.; Zhou, J.; Cai, J.; Han, X.; Zeng, G.; Li, D.; Liu, Z.; and Sun, M. 2024. MiniCPM-V: A GPT-4V Level MLLM on Your Phone. arXiv:2408.01800. 
*   Ye et al. (2023a) Ye, J.; Hu, A.; Xu, H.; Ye, Q.; Yan, M.; Dan, Y.; Zhao, C.; Xu, G.; Li, C.; Tian, J.; et al. 2023a. mplug-docowl: Modularized multimodal large language model for document understanding. _arXiv preprint arXiv:2307.02499_. 
*   Ye et al. (2023b) Ye, J.; Hu, A.; Xu, H.; Ye, Q.; Yan, M.; Xu, G.; Li, C.; Tian, J.; Qian, Q.; Zhang, J.; Jin, Q.; He, L.; Lin, X.; and Huang, F. 2023b. UReader: Universal OCR-free Visually-situated Language Understanding with Multimodal Large Language Model. In Bouamor, H.; Pino, J.; and Bali, K., eds., _Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023_, 2841–2858. Association for Computational Linguistics. 
*   Yu et al. (2023) Yu, Y.; Li, Y.; Zhang, C.; Zhang, X.; Guo, Z.; Qin, X.; Yao, K.; Han, J.; Ding, E.; and Wang, J. 2023. StrucTexTv2: Masked Visual-Textual Prediction for Document Image Pre-training. _arXiv preprint arXiv:2303.00289_. 
*   Zhang and Shasha (1989) Zhang, K.; and Shasha, D. 1989. Simple fast algorithms for the editing distance between trees and related problems. _SIAM journal on computing_, 18(6): 1245–1262. 
*   Zhang et al. (2023a) Zhang, R.; Han, J.; Zhou, A.; Hu, X.; Yan, S.; Lu, P.; Li, H.; Gao, P.; and Qiao, Y. 2023a. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. _arXiv preprint arXiv:2303.16199_. 
*   Zhang et al. (2024) Zhang, R.; Zhang, Y.; Chen, J.; Zhou, Y.; Gu, J.; Chen, C.; and Sun, T. 2024. TRINS: Towards Multimodal Language Models that Can Read. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, 22584–22594. 
*   Zhang et al. (2023b) Zhang, Y.; Zhang, R.; Gu, J.; Zhou, Y.; Lipka, N.; Yang, D.; and Sun, T. 2023b. LLaVAR: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding. _arXiv preprint arXiv:2306.17107_. 
*   Zhao et al. (2023) Zhao, L.; Yu, E.; Ge, Z.; Yang, J.; Wei, H.; Zhou, H.; Sun, J.; Peng, Y.; Dong, R.; Han, C.; et al. 2023. Chatspot: Bootstrapping multimodal llms via precise referring instruction tuning. _arXiv preprint arXiv:2307.09474_. 
*   Zhu et al. (2023) Zhu, D.; Chen, J.; Shen, X.; Li, X.; and Elhoseiny, M. 2023. Minigpt-4: Enhancing vision-language understanding with advanced large language models. _arXiv preprint arXiv:2304.10592_. 

Appendix A Appendix
-------------------

### A.1 Model and Training Hyperparameters

Model and training hyperparameters are demonstrated in Table[9](https://arxiv.org/html/2309.11419v2#A1.T9 "Table 9 ‣ A.1 Model and Training Hyperparameters ‣ Appendix A Appendix ‣ Kosmos-2.5: A Multimodal Literate Model") and Table[10](https://arxiv.org/html/2309.11419v2#A1.T10 "Table 10 ‣ A.1 Model and Training Hyperparameters ‣ Appendix A Appendix ‣ Kosmos-2.5: A Multimodal Literate Model").

Modules Hyperparameters
Image Encoder Patch size 16
Patch embed hidden size 768
Number of layers 18
Hidden size 1,536
FFN inner hidden size 3,968
Attention heads 24
Activation function GeLU
Max sequence length 4,096
Resampler Number of layers 1
Hidden size 1,536
Output sequence length 2,048
Language Decoder Number of layers 24
Hidden size 1,536
FFN inner hidden size 6,144
Attention heads 16
Activation function GeLU
Vocabulary size 108,481
Max sequence length 4,096

Table 9: Model Hyperparameters of Kosmos-2.5

Hyperparameters Pre-training Fine-tuning
Training steps 260,000 3000
Warmup steps 375 100
Batch size 1,024
Optimizer AdamW
Learning rate 2e-4
Learning rate decay Linear
Adam β 𝛽\beta italic_β(0.9, 0.98)
Weight decay 0.01
Dropout 0.1

Table 10: Training hyperparameters of Kosmos-2.5

### A.2 OCR Evaluation Metrics

##### F1.

The F1 score is a commonly used evaluation metric for measuring the accuracy of models in classification tasks. It is the harmonic mean of Precision and Recall, offering a balanced measure that considers both false positives and false negatives. In the OCR task, the F1 score will be used to assess the effectiveness of OCR models in recognizing words from images. Precision is the ratio of correctly recognized words to the total number of words detected by the model. Recall is the ratio of correctly recognized words to the total number of actual words. The F1 score is the harmonic mean of Precision and Recall, and it is calculated as follows:

Precision=TP TP+FP Precision TP TP FP\text{Precision}=\frac{\text{TP}}{\text{TP}+\text{FP}}Precision = divide start_ARG TP end_ARG start_ARG TP + FP end_ARG

Recall=TP TP+FN Recall TP TP FN\text{Recall}=\frac{\text{TP}}{\text{TP}+\text{FN}}Recall = divide start_ARG TP end_ARG start_ARG TP + FN end_ARG

F⁢1=2⋅Precision⋅Recall Precision+Recall 𝐹 1⋅2⋅Precision Recall Precision Recall F1=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{% Recall}}italic_F 1 = 2 ⋅ divide start_ARG Precision ⋅ Recall end_ARG start_ARG Precision + Recall end_ARG

where TP is the number of correctly recognized words, FP is the number of incorrectly recognized words, and FN is the number of missed words.

##### IoU.

Intersection over Union (IoU) is a critical evaluation metric for assessing the performance of object detection models, including OCR textline detection. IoU measures the overlap between the predicted bounding box and the ground truth bounding box, providing a quantitative measure of how well the model has detected the textlines in an image. The formula for IoU is as follows:

IoU=Area of Intersection Area of Union IoU Area of Intersection Area of Union\text{IoU}=\frac{\text{Area of Intersection}}{\text{Area of Union}}IoU = divide start_ARG Area of Intersection end_ARG start_ARG Area of Union end_ARG

##### NED.

Normalized Edit Distance (NED) is an extension of the Edit Distance (Levenshtein Distance) metric, commonly used to assess the similarity between two text strings. The calculation of NED can be found in Appendix[A.3](https://arxiv.org/html/2309.11419v2#A1.SS3 "A.3 Image-to-markdown Generation Evaluation Metrics ‣ Appendix A Appendix ‣ Kosmos-2.5: A Multimodal Literate Model").

### A.3 Image-to-markdown Generation Evaluation Metrics

In light of the unique nature of the image-to-markdown conversion task, assessing the quality of the generated markdown necessitates specialized metrics. We adopt a two-fold evaluation scheme: Normalized Edit Distance (NED) and Normalized Tree Edit Distance (NTED), considering both the lexical accuracy and the preservation of the original structural elements. The NED is formulated as

NED=1−1 N⁢∑i=1 N D⁢(s i,s^i)/max⁡(len⁢(s i),len⁢(s^i))NED 1 1 𝑁 superscript subscript 𝑖 1 𝑁 𝐷 subscript 𝑠 𝑖 subscript^𝑠 𝑖 len subscript 𝑠 𝑖 len subscript^𝑠 𝑖\textit{NED}=1-\frac{1}{N}\sum_{i=1}^{N}D\left(s_{i},\hat{s}_{i}\right)/\max% \left(\mathrm{len}(s_{i}),\mathrm{len}(\hat{s}_{i}\right))NED = 1 - divide start_ARG 1 end_ARG start_ARG italic_N end_ARG ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_N end_POSTSUPERSCRIPT italic_D ( italic_s start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , over^ start_ARG italic_s end_ARG start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) / roman_max ( roman_len ( italic_s start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) , roman_len ( over^ start_ARG italic_s end_ARG start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) )

where N 𝑁 N italic_N, s 𝑠 s italic_s, and s^^𝑠\hat{s}over^ start_ARG italic_s end_ARG denote the number of samples, prediction, and ground truth, respectively. D⁢(⋅,⋅)𝐷⋅⋅D(\cdot,\cdot)italic_D ( ⋅ , ⋅ ) and len⁢(⋅)len⋅\mathrm{len}(\cdot)roman_len ( ⋅ ) represent the edit distance function and the length of a string. The NED value ranges from 0 to 1, with a higher NED value indicating the prediction is closer to the ground truth.

However, given the hierarchical structure inherent to markdown, relying solely on a string-based comparison metric like NED can be insufficient. Thus, we adopt NTED as an additional evaluation metric for structural differences. NTED is a tree edit distance normalized by the number of nodes in the tree, considering the structural discrepancies between parse trees. Specifically, the predicted markdown sequence is first transformed into an HTML tree. Then, the tree edit distance between the prediction and the ground truth is calculated using the ZSS algorithm (Zhang and Shasha [1989](https://arxiv.org/html/2309.11419v2#bib.bib92)). The NTED is formulated as

NTED=1−1 N⁢∑i=1 N TD⁢(t i,t^i)/max⁡(node⁢(t i),node⁢(t^i))NTED 1 1 𝑁 superscript subscript 𝑖 1 𝑁 TD subscript 𝑡 𝑖 subscript^𝑡 𝑖 node subscript 𝑡 𝑖 node subscript^𝑡 𝑖\textit{NTED}=1-\frac{1}{N}\sum_{i=1}^{N}\mathrm{TD}\left(t_{i},\hat{t}_{i}% \right)/\max\left(\mathrm{node}(t_{i}),\mathrm{node}(\hat{t}_{i}\right))NTED = 1 - divide start_ARG 1 end_ARG start_ARG italic_N end_ARG ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_N end_POSTSUPERSCRIPT roman_TD ( italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , over^ start_ARG italic_t end_ARG start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) / roman_max ( roman_node ( italic_t start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) , roman_node ( over^ start_ARG italic_t end_ARG start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) )

where N 𝑁 N italic_N, t 𝑡 t italic_t, and t^^𝑡\hat{t}over^ start_ARG italic_t end_ARG signify the number of samples, the HTML tree of prediction, and the HTML tree of ground truth, respectively. Besides, TD⁢(⋅,⋅)TD⋅⋅\mathrm{TD}(\cdot,\cdot)roman_TD ( ⋅ , ⋅ ) and node⁢(⋅)node⋅\mathrm{node}(\cdot)roman_node ( ⋅ ) stand for the tree edit distance function and the number of nodes in a tree.

### A.4 Qualitative Example

(a) Input

(b) Using the layout prompt

(c) Using the markup prompt

Figure 3: Kosmos-2.5’s outputs given the same text image and different task prompts.

We illustrate an example in Figure[3](https://arxiv.org/html/2309.11419v2#A1.F3 "Figure 3 ‣ A.4 Qualitative Example ‣ Appendix A Appendix ‣ Kosmos-2.5: A Multimodal Literate Model"), showcasing the model outputs produced by Kosmos-2.5 with various task prompts when presented with the same input text image. For document-level text recognition task, Kosmos-2.5 produces the following text sequence, which includes textual content and corresponding bounding boxes:

1[x_52][y_113][x_756][y_145]:NYC Department of Education School Year Calendar 2023-2024

2[x_52][y_159][x_826][y_181]:This is the 2023-24 school year calendar for all 3 K-12 NYCDOE public schools.If your child attends a private,

3[x_52][y_180][x_820][y_202]:parochial,charter school,NYC Early Education Center(NYCEEC)or Family Childcare Program,please contact

4[x_52][y_201][x_639][y_223]:your child’s school for information about their calendar.Please note the following:

5[x_65][y_223][x_77][y_245]:∙∙\bullet∙

6[x_92][y_223][x_825][y_245]:On days when school buildings are closed due to inclement weather or other emergencies,all students

7...

For image-to-markdown generation task, Kosmos-2.5 generates the text sequence in Markdown format:

1#NYC Department of Education School Year Calendar 2023-2024

2

3 This is the 2023-24 school year calendar for all 3 K-12 NYCDOE public schools.If your child attends a private,parochial,charter school,NYC Early Education Center(NYCEEC)or Family Childcare Program,please contact your child’s school for information about their calendar.Please note the following:

4...

5-On this schedule,**elementary schools**are defined as programs that serve kindergarten(K)through grade 8,including schools with 3-K and Pre-K programs,as well as those that end in grade 5.**Middle schools**are defined as programs that serve grades 6-8,and**high schools**are defined as programs that serve grades 9-12.

6...

The example shows that Kosmos-2.5 precisely identifies text positions and recognizes text content. Moreover, it adeptly captures the styles and structures present within the text image, including elements like titles, bullet points, tables, and bold text. Section[A.7](https://arxiv.org/html/2309.11419v2#A1.SS7 "A.7 Examples of Model Inference ‣ Appendix A Appendix ‣ Kosmos-2.5: A Multimodal Literate Model") provides the full output sequence using different task prompts for this example. Furthermore, Kosmos-2.5 is compatible with more powerful LLMs like GPT-3.5 or GPT-4. The output from our model can serve as contexts for LLMs, enhancing their capabilities through further prompt engineering.

### A.5 Pre-training Data Processing

The pre-training data has a wide coverage, and each type of data requires a different processing workflow, which is introduced as follows:

##### Scanned Document

##### General Document, Academic Paper, Design

We first compile and convert arXiv papers, SEC files, and PowerPoint slides into PDF files. Together with other general PDFs and poster, we employed the PyMuPDF parser 4 4 4[https://github.com/pymupdf/PyMuPDF](https://github.com/pymupdf/PyMuPDF) to extract text and layout information efficiently.

##### Web Page

We also include webpage screenshots in the model pre-training to diversify the layout distribution further. We collect the webpage URLs from the English portion of the mC4 dataset. Playwright 5 5 5[https://github.com/microsoft/playwright-python](https://github.com/microsoft/playwright-python) is used to access a specified URL and open the webpage. The HTML content of the page is extracted and parsed using the lxml library 6 6 6[https://lxml.de/](https://lxml.de/) to obtain a Document Object Model (DOM) tree representation. This DOM tree is traversed, examining the XPath of each element within it. This traversal aims to determine whether each element is visible and retrieve information about its bounding boxes.

##### Mathematical

In CROHME, each piece of data is an individual formula. We randomly select between 5 to 15 formulas and paste them onto a single blank page at random positions for formula recognition in page level.

##### Handwritten

We downloaded 5,427 handwritten fonts from Google Fonts, and for each textline generation, we randomly select one of these fonts.

##### General Document (markdown)

The Microsoft Office WORD files have been extensively used in existing research like TableBank(Li et al. [2020](https://arxiv.org/html/2309.11419v2#bib.bib43)) and ReadingBank(Wang et al. [2021](https://arxiv.org/html/2309.11419v2#bib.bib79)). We collect WORD DOCX files and convert them into texts with markdown. First, we use Pandoc to convert the XML content within the DOCX files into markdown files. As Pandoc keeps the “¡table¿” tags to represent the tabular cells in the generated markdown, we further identify all the tables and use markdownify 7 7 7[https://github.com/matthewwithanm/python-markdownify](https://github.com/matthewwithanm/python-markdownify) to convert them into the markdown formats. Finally, the original DOCX files are converted into PDF files, and each page is aligned to the corresponding span of the markdown content based on a heuristic method.

##### Academic Paper (markdown)

L a T e X documents from arXiv have been used to generate PDF files to obtain texts with bounding boxes. Meanwhile, we also convert the L a T e X content into the markdown texts. Similar to Nougat(Blecher et al. [2023](https://arxiv.org/html/2309.11419v2#bib.bib4)), LaTeXML 8 8 8[https://math.nist.gov/~BMiller/LaTeXML/](https://math.nist.gov/~BMiller/LaTeXML/) is used to convert the L a T e X code into the HTML sequence, which is further transformed into the markdown format. Different from Nougat, we keep all the tables at the beginning of the page as most L a T e X users prefer to position tables with “[t]” or “[h]” instead of “[b]”. Meanwhile, we also convert the table content from the L a T e X format into the markdown format.

##### Project Document (markdown)

In addition to layout-based data, we collect markup-based data for the pre-training. We collect “README.md” files from many GitHub projects and convert these files into HTML using Pandoc 9 9 9[https://pandoc.org/](https://pandoc.org/). Then, wkhtmltopdf 10 10 10[https://wkhtmltopdf.org/](https://wkhtmltopdf.org/) is used to obtain the images from the generated HTML content.

##### Web Page (markdown)

The most straightforward way to obtain markdown resources from HTML webpages is through web scraping. However, webpages are often cluttered with various layouts and styles, resulting from the misuse of HTML tags. Moreover, HTML pages may include extraneous elements, such as advertisements, navigation menus, or formatting elements, making extracting clean and meaningful content challenging. To overcome these obstacles, we employ Playwright, a fast and reliable end-to-end testing framework for the web. The library allows us to navigate the HTML structure, filter out non-essential elements, and extract the relevant text content. We also apply custom rules and regular expressions to further refine the extracted text and format it as markdown, ensuring that the resulting markdown files are coherent and readable.

### A.6 Evaluation Data Processing

##### OCREval.

We constructed the OCREval comprising 2,297 samples, covering data from various domains. The details of each dataset’s construction are provided below:

*   •Design.  We used the Microsoft Read API to obtain OCR results for MARIO-LAION, MARIO-OpenLibrary, and MARIO-TMDB, followed by manual verification to ensure the accuracy of the ground truth. For MJ&ST, MJSynth consists of single-line textlines, which we randomly selected and placed multiple textlines on a single page to create page-level OCR test samples. For SynthText, we used the provided text and bounding box as the OCR ground truth. 
*   •Receipt.  For SROIE and CORD, we use their official annotations, which are carefully annotated by crowd workers and not from third-party OCR results. For Receipts crawled from the internet, we used Bing’s image search engine 11 11 11[https://www.bing.com/images](https://www.bing.com/images) with the keyword ”receipt” to find relevant images. Subsequently, we used the Microsoft Read API to obtain OCR results and manually verified them, filtering out non-English receipts. 
*   •Others.  The processing steps for Handwritten, Academic paper, General, and Web Page are consistent with those used in the pre-training phase, as detailed in Appendix[A.5](https://arxiv.org/html/2309.11419v2#A1.SS5 "A.5 Pre-training Data Processing ‣ Appendix A Appendix ‣ Kosmos-2.5: A Multimodal Literate Model"). 

##### MarkdownEval.

We constructed a dataset called markdownEval, consisting of 5,633 test samples, to evaluate the model’s understanding of image across various domains. The details of each dataset’s construction are provided below:

*   •Math Image.  Both CROHME Math and Ima2LaTeX-100k consist of formulas and their corresponding LaTeX source code. We used Pandoc 13 13 13[https://pandoc.org/](https://pandoc.org/) to convert the LaTeX source code into Markdown format and then randomly selected multiple samples to place on a single page to create test samples. 
*   •Table.  We extracted the LaTeX source code for tables from arXiv sources and then compiled it using pdfLaTeX 14 14 14[https://www.tug.org/texlive/](https://www.tug.org/texlive/) to obtain table images. Subsequently, we used Pandoc to convert the LaTeX source code into Markdown format to create test samples. 
*   •Others.  The processing steps for Project Document, Academic Paper, and General Document are consistent with those used in the pre-training phase, as detailed in Appendix[A.5](https://arxiv.org/html/2309.11419v2#A1.SS5 "A.5 Pre-training Data Processing ‣ Appendix A Appendix ‣ Kosmos-2.5: A Multimodal Literate Model"). 

### A.7 Examples of Model Inference

Listing 1: Model outputs using the layout-based prompt

1[x_52][y_113][x_756][y_145]:NYC Department of Education School Year Calendar 2023-2024

2[x_52][y_159][x_826][y_181]:This is the 2023-24 school year calendar for all 3 K-12 NYCDOE public schools.If your child attends a private,

3[x_52][y_180][x_820][y_202]:parochial,charter school,NYC Early Education Center(NYCEEC)or Family Childcare Program,please contact

4[x_52][y_201][x_639][y_223]:your child’s school for information about their calendar.Please note the following:

5[x_65][y_223][x_77][y_245]:∙∙\bullet∙

6[x_92][y_223][x_825][y_245]:On days when school buildings are closed due to inclement weather or other emergencies,all students

7[x_92][y_244][x_525][y_266]:and families should plan on participating in remote learning.

8[x_65][y_265][x_77][y_287]:∙∙\bullet∙

9[x_92][y_265][x_846][y_287]:Individual schools’Parent-Teacher Conference dates might be different from the dates below.Your child’s

10[x_92][y_286][x_491][y_308]:teacher will work with you to schedule your conference.

11[x_65][y_308][x_77][y_330]:∙∙\bullet∙

12[x_92][y_307][x_845][y_330]:On this schedule,elementary schools are defined as programs that serve kindergarten(K)through grade

13[x_92][y_329][x_826][y_351]:8,including schools with 3-K and Pre-K programs,as well as those that end in grade 5.Middle schools

14[x_92][y_350][x_810][y_372]:are defined as programs that serve grades 6-8,and high schools are defined as programs that serve

15[x_92][y_371][x_186][y_393]:grades 9-12.

16[x_60][y_414][x_106][y_436]:DATE

17[x_318][y_414][x_399][y_436]:WEEKDAY

18[x_605][y_414][x_659][y_436]:EVENT

19[x_60][y_437][x_155][y_459]:September 7

20[x_297][y_437][x_366][y_459]:Thursday

21[x_432][y_437][x_565][y_459]:First day of school

22[x_60][y_470][x_164][y_492]:September 14

23[x_297][y_470][x_366][y_492]:Thursday

24[x_432][y_459][x_804][y_481]:Evening Parent-Teacher Conferences for elementary

25[x_432][y_480][x_622][y_503]:schools and Pre-K Centers

26[x_60][y_514][x_164][y_536]:September 21

27[x_297][y_514][x_366][y_536]:Thursday

28[x_432][y_504][x_832][y_526]:Evening Parent-Teacher Conferences for middle schools

29[x_432][y_525][x_553][y_547]:and D75 schools

30[x_60][y_548][x_164][y_570]:September 25

31[x_297][y_548][x_360][y_570]:Monday

32[x_432][y_548][x_630][y_570]:Yom Kippur,schools closed

33[x_60][y_581][x_164][y_603]:September 28

34[x_297][y_581][x_366][y_603]:Thursday

35[x_432][y_570][x_818][y_593]:Evening Parent-Teacher Conferences for high schools,

36[x_432][y_592][x_601][y_614]:K-12,and 6-12 schools

37[x_60][y_625][x_135][y_647]:October 9

38[x_297][y_625][x_360][y_647]:Monday

39[x_432][y_614][x_786][y_636]:Italian Heritage/Indigenous Peoples’Day,schools

40[x_432][y_636][x_482][y_658]:closed

41[x_60][y_679][x_152][y_701]:November 2

42[x_297][y_679][x_366][y_701]:Thursday

43[x_432][y_658][x_829][y_680]:Afternoon and Evening Parent-Teacher Conferences for

44[x_432][y_679][x_833][y_701]:elementary schools;students in these schools dismissed

45[x_432][y_700][x_556][y_723]:three hours early

46[x_60][y_727][x_152][y_749]:November 7

47[x_297][y_727][x_360][y_749]:Tuesday

48[x_432][y_727][x_745][y_749]:Election Day,students do not attend school

49[x_60][y_775][x_152][y_797]:November 9

50[x_297][y_775][x_366][y_797]:Thursday

51[x_432][y_754][x_829][y_776]:Afternoon and Evening Parent-Teacher Conferences for

52[x_432][y_775][x_793][y_797]:middle schools and D75 schools;students in these

53[x_432][y_796][x_687][y_818]:schools dismissed three hours early

54[x_60][y_829][x_161][y_851]:November 16

55[x_297][y_829][x_366][y_851]:Thursday

56[x_432][y_819][x_818][y_841]:Evening Parent-Teacher Conferences for high schools,

57[x_432][y_840][x_601][y_862]:K-12,and 6-12 schools

58[x_60][y_884][x_161][y_906]:November 17

59[x_297][y_884][x_344][y_906]:Friday

60[x_432][y_863][x_773][y_885]:Afternoon Parent-Teacher Conferences for high

61[x_432][y_884][x_791][y_906]:schools,K-12,and 6-12 schools;students in these

62[x_432][y_905][x_687][y_927]:schools dismissed three hours early

63[x_60][y_928][x_186][y_950]:November 23-24

64[x_297][y_928][x_416][y_950]:Thursday-Friday

65[x_432][y_928][x_692][y_950]:Thanksgiving Recess,schools closed

66[x_60][y_960][x_234][y_983]:December 25-January 1

67[x_297][y_950][x_368][y_972]:Monday-

68[x_297][y_971][x_360][y_994]:Monday

69[x_432][y_960][x_646][y_983]:Winter Recess,schools closed

70[x_60][y_999][x_140][y_1021]:January 15

71[x_297][y_999][x_360][y_1021]:Monday

72[x_432][y_999][x_789][y_1021]:Rev.Dr.Martin Luther King Jr.Day,schools closed

73[x_60][y_1027][x_170][y_1049]:January 23-26

74[x_297][y_1027][x_410][y_1049]:Tuesday-Friday

75[x_432][y_1027][x_603][y_1049]:Regents Administration

76[x_52][y_1099][x_311][y_1118]:NYCDOE School Year Calendar 2023-24

Listing 2: Model outputs using the markup-based prompt

1#NYC Department of Education School Year Calendar 2023-2024

2

3 This is the 2023-24 school year calendar for all 3 K-12 NYCDOE public schools.If your child attends a private,parochial,charter school,NYC Early Education Center(NYCEEC)or Family Childcare Program,please contact your child’s school for information about their calendar.Please note the following:

4

5-On days when school buildings are closed due to inclement weather or other emergencies,all students and families should plan on participating in remote learning.

6

7-Individual schools’Parent-Teacher Conference dates might be different from the dates below.Your child’s teacher will work with you to schedule your conference.

8

9-On this schedule,**elementary schools**are defined as programs that serve kindergarten(K)through grade 8,including schools with 3-K and Pre-K programs,as well as those that end in grade 5.**Middle schools**are defined as programs that serve grades 6-8,and**high schools**are defined as programs that serve grades 9-12.

10

11|DATE|WEEKDAY|EVENT|

12|—|—|—|

13|September 7|Thursday|First day of school|

14|September 14|Thursday|Evening Parent-Teacher Conferences for elementary schools and Pre-K Centers|

15|September 21|Thursday|Evening Parent-Teacher Conferences for middle schools and D75 schools|

16|September 25|Monday|Yom Kippur,schools closed|

17|September 28|Thursday|Evening Parent-Teacher Conferences for high schools,K-12,and 6-12 schools|

18|October 9|Monday|Italian Heritage/Indigenous Peoples’Day,schools closed|

19|November 2|Thursday|Afternoon and Evening Parent-Teacher Conferences for elementary schools;students in these schools dismissed three hours early|

20|November 7|Tuesday|Election Day,students do not attend school|

21|November 9|Thursday|Afternoon and Evening Parent-Teacher Conferences for middle schools and D75 schools;students in these schools dismissed three hours early|

22|November 16|Thursday|Evening Parent-Teacher Conferences for high schools,K-12,and 6-12 schools|

23|November 17|Friday|Afternoon Parent-Teacher Conferences for high schools,K-12,and 6-12 schools;students in these schools dismissed three hours early|

24|November 23-24|Thursday-Friday|Thanksgiving Recess,schools closed|

25|December 25-January 1|Monday-Monday|Winter Recess,schools closed|

26|January 15|Monday|Rev.Dr.Martin Luther King Jr.Day,schools closed|

27|January 23-26|Tuesday-Friday|Regents Administration|

### A.8 Pre-training Data Examples

We demonstrate some of the pre-training data examples used in Kosmos-2.5, which include the input and output from IIT-CDIP, general pdfs, SEC, arXiv papers, web screenshots, PowerPoint slides, poster, mathematical, handwrittens, README, DOCX, L a T e X and HTML.

(a) Input

(b) Rendered output

Figure 4: A training sample for the layout-based task from IIT-CDIP

(a) Input

(b) Rendered output

Figure 5: A training sample for the layout-based task from PDFs

(a) Input

(b) Rendered output

Figure 6: A training sample for the layout-based task from SECs

(a) Input

(b) Rendered output

Figure 7: A training sample for the layout-based task from arXiv papers (single-column)

(a) Input

(b) Rendered output

Figure 8: A training sample for the layout-based task from arXiv papers (two-column)

(a) Input

(b) Rendered output

Figure 9: A training sample for the layout-based task from web screenshots

(a) Input

(b) Rendered output

Figure 10: A training sample for the layout-based task from PowerPoint slides

(a) Input

(b) Rendered output

Figure 11: A training sample for the layout-based task from posters

(a) Input

(b) Rendered output

Figure 12: A training sample for the layout-based task from Mario-10M

(a) Input

(b) Rendered output

Figure 13: A training sample for the layout-based task from mathematical

(a) Input

(b) Rendered output

Figure 14: A training sample for the layout-based task from handwrittens

(a) Input

(b) Rendered output

Figure 15: A training sample for the markup-based task from README

(a) Input

(b) Rendered output

Figure 16: A training sample for the markup-based task from DOCX

(a) Input

(b) Rendered output

Figure 17: A training sample for the markup-based task from SEC

(a) Input

(b) Rendered output

Figure 18: A training sample for the markup-based task from L a T e X(single-column)

(a) Input

(b) Rendered output

Figure 19: A training sample for the markup-based task from L a T e X(two-column)

(a) Input

(b) Rendered output

Figure 20: A training sample for the markup-based task from HTMLs

