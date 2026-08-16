# BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents

Teakgyu Hong<sup>1</sup>, Donghyun Kim<sup>1</sup>, Mingi Ji<sup>2</sup>, Wonseok Hwang<sup>3</sup>, Daehyun Nam<sup>4</sup>, Sungrae Park<sup>4</sup>

<sup>1</sup>NAVER CLOVA, <sup>2</sup>KAIST, <sup>3</sup>LBox, <sup>4</sup>Upstage AI Research, Upstage AI  
teakgyu.hong@navercorp.com, dong.hyun@navercorp.com, qwertgfdcvb@kaist.ac.kr,  
wonseok.hwang@lbox.kr, daehyun.nam@upstage.ai, sungrae.park@upstage.ai

## Abstract

Key information extraction (KIE) from document images requires understanding the contextual and spatial semantics of texts in two-dimensional (2D) space. Many recent studies try to solve the task by developing pre-trained language models focusing on combining visual features from document images with texts and their layout. On the other hand, this paper tackles the problem by going back to the basic: effective combination of text and layout. Specifically, we propose a pre-trained language model, named *BROS (BERT Relying On Spatiality)*, that encodes relative positions of texts in 2D space and learns from unlabeled documents with area-masking strategy. With this optimized training scheme for understanding texts in 2D space, BROS shows comparable or better performance compared to previous methods on four KIE benchmarks (FUNSD, SROIE\*, CORD, and SciTSR) without relying on visual features. This paper also reveals two real-world challenges in KIE tasks—(1) minimizing the error from incorrect text ordering and (2) efficient learning from fewer downstream examples—and demonstrates the superiority of BROS over previous methods. *Code is available at <https://github.com/clovaai/bros>.*

## Introduction

Automatic key information extraction (KIE) from industrial documents is an essential task in robotic process automation (RPA). Extracting an ordered item list from receipts (Park et al. 2019), prices and taxes from invoices (Liu et al. 2019), and paired key-values from form-like documents (Jaume, Ekenel, and Thiran 2019) are representative examples. Since the task requires understanding texts in various layouts, the combination of multiple technical components from both computer vision and natural language processing is required.

Figure 1 describes a schematic illustration of pipeline for the document KIE tasks (Hwang et al. 2019; Denk and Reisswig 2019). First, given a document image, optical character recognition (OCR) detects the texts in the image and recognizes the content to generate a set of text blocks. Next, a serializer identifies a reading order of text blocks distributed in 2D image space and converts them into text sequence in 1D text space to apply NLP technology which is developed for 1D text sequence. The most basic form of serializer is to arrange text blocks in a top-to-bottom and left-to-right way (Clausner, Pletschacher, and Antonacopoulos 2013). Fi-

```

graph LR
    Doc[Document Image] --> OCR[OCR]
    OCR --> Serializer[Serializer]
    Serializer --> Parsing[Parsing Model]
    Parsing --> Info[Date  
9/3/92  
R&D Group  
Licensee  
...]
  
```

Figure 1: Schematic illustrations of document KIE pipeline.

<table border="1">
<thead>
<tr>
<th>Model</th>
<th>Img</th>
<th># Params</th>
<th>(O)</th>
<th>(P)</th>
<th>(F)</th>
</tr>
</thead>
<tbody>
<tr>
<td>LayoutLM<sub>BASE</sub></td>
<td></td>
<td>113M</td>
<td>78.66</td>
<td>33.89</td>
<td>62.50</td>
</tr>
<tr>
<td>LayoutLMv2<sub>BASE</sub></td>
<td>o</td>
<td>200M</td>
<td>82.76</td>
<td>40.77</td>
<td>69.92</td>
</tr>
<tr>
<td><b>BROS<sub>BASE</sub></b></td>
<td></td>
<td><b>110M</b></td>
<td><b>83.05</b></td>
<td><b>76.94</b></td>
<td><b>72.60</b></td>
</tr>
<tr>
<td>LayoutLM<sub>LARGE</sub></td>
<td></td>
<td>343M</td>
<td>78.95</td>
<td>33.11</td>
<td>61.00</td>
</tr>
<tr>
<td>LayoutLMv2<sub>LARGE</sub></td>
<td>o</td>
<td>426M</td>
<td>84.20</td>
<td>62.53</td>
<td>72.12</td>
</tr>
<tr>
<td><b>BROS<sub>LARGE</sub></b></td>
<td></td>
<td><b>340M</b></td>
<td><b>84.52</b></td>
<td><b>79.42</b></td>
<td><b>74.42</b></td>
</tr>
</tbody>
</table>

Table 1: Performance comparison of pre-trained language models on (O)riginal, (P)ermuted, and (F)ew training samples FUNSD KIE tasks. In (F), 10 samples are used.

nally, from the serialized text blocks, key information is extracted via the parsing model.

In the first step, an off-the-shelf OCR tool is often employed as industrial documents consist of relatively clean characters compared to general scene text images. On the other hand, there are no such off-the-shelf tools for the serializer even though it is often non-trivial to determine the proper reading order of text blocks (Li et al. 2020; Wang et al. 2021). Representative examples are documents including multi-columns or multiple tables. This absence of the general-purpose serializer implies the careful design of parsing module is necessary to robustly handle documents with complex layouts where the reading orders can be often incomplete.

In the early studies of KIE, accurate document parsing greatly depends on the order of text blocks. Once the serializer identifies an order, the set of the text blocks are converted into a text sequence and processed via a language model such as BERT (Devlin et al. 2019) to identify key information (Hwang et al. 2019; Denk and Reisswig 2019). The linguistic understanding of the pre-trained language model leads to superior performance than rule-based extractions. However, the conversion of texts in 2D space into a text sequence in 1D space leads to the loss of layoutinformation that is critical in KIE tasks.

To avoid the loss of layout information, a new type of language model, LayoutLM (Xu et al. 2020) expands a 1D positional encoding of BERT to 2D and is trained over a large corpus of industrial documents to understand spatial dependencies between text blocks. Its fine-tuning has shown breakthrough performances on multiple KIE tasks and becomes a strong baseline. After the rise of LayoutLM, several studies try to develop pre-trained language models by combining additional visual features (Xu et al. 2021; Powalski et al. 2021; Li et al. 2021b; Appalaraju et al. 2021; Li et al. 2021c) (e.g. image patches identified by an object detection) and show further performance improvements. However, the extensions using visual features require additional computational costs and they still demand more effective combinations of texts and their spatial information.

In this paper, we introduce a new pre-trained language model, named BROS, by re-focusing on the combinations of texts and their spatial information without relying on visual features. Specifically, we propose an effective spatial encoding method by utilizing relative positions between text blocks, while most of previous works employ absolute 2D positions. Additionally, we introduce a novel self-supervision method, named area-masked language model, that hides texts in an area of a document and supervises the masked texts. With these two approaches for encoding of spatial information, BROS shows superior or comparable performances compared to previous methods using additional visual features.

Aside from improving KIE performances, BROS also addresses two important real-world challenges in KIE tasks: minimizing dependency on the order of text blocks and learning from a few training examples of downstream tasks. The first challenge indicates the robustness on the serialization followed by the OCR process in Figure 1. In real scenario, document images are usually irregular (i.e. rotated or distorted documents) and the serializer might fail to identify a proper order of text blocks. In addition, when serialization fails, the performance of sequence tagging approaches (e.g. BIO tagging), which most previous works employ, drops dramatically. To circumvent the difficulty, we apply SPADE (Hwang et al. 2021) decoder that extracts key text blocks without any order information to the pre-trained models and evaluates them on the new benchmarks where the order of text blocks are permuted. As a result, BROS shows better robustness on the serializers compared to LayoutLM (Xu et al. 2020) and LayoutLMv2 (Xu et al. 2021).

The second challenge is related to the required number of labeled examples to understand the target key contents. Since a single KIE example consists of hundreds of text blocks that should be categorized, the annotation is expensive. Most public benchmarks consist of less than 1,000 samples, even though the target documents contain hundreds of layouts and diverse contexts. In this paper, we analyze KIE performances over the number of training examples and compare the pre-trained models. As a result, BROS performs better on FUNSD KIE tasks, and also BROS only with 20~30% of FUNSD examples achieves better performance than LayoutLM with 100% of them. Summarized results for

these experiments are shown in Table 1.

Our contributions can be summarized as follows:

- • We propose an effective spatial layout encoding method by accounting for relative positions of text blocks.
- • We also propose a novel area-masking self-supervision strategy that reflects 2D natures of text blocks.
- • The proposed model achieves comparable performance to the state-of-the-art without relying on visual features.
- • We compare existing pre-trained models on permuted KIE datasets that lost the orders of text blocks.
- • We compare the fine-tuning efficiency of various pre-trained models under a data-scarce environment.

## Related Work

### Pre-trained Language Models for 2D Text Blocks

Unlike the pre-trained models for conventional NLP tasks, such as BERT (Devlin et al. 2019), LayoutLM (Xu et al. 2020) is first proposed to jointly model interaction between text and layout information for the document KIE task. It encodes the absolute position of text blocks with axis-wise embedding tables and learns a token-level masked language model that hides tokens randomly and estimates the origins. After the publication of LayoutLM, several pre-trained models have been tried to additionally integrate visual features, such as visual feature maps from raw images (Xu et al. 2021; Appalaraju et al. 2021), image patches identified by an object detection module (Li et al. 2021b), and visual representations of text blocks (Powalski et al. 2021; Li et al. 2021c). Although the extensions imposing multi-modalities of visual and textual features provide additional performance gains in KIE tasks, they spend additional computations to process raw document images. Additionally, an effective combination of text and layout is still required as the major component of the multi-modalities.

Aside from incorporating visual features, StructuralLM (Li et al. 2021a) utilizes cell information, a group of ordered text blocks, and shows promising performance improvements. However, the local orders of text blocks might not be available depending on the KIE tasks and the OCR engines. Therefore, this paper focuses on the original granularity of text blocks identified by OCR engines and improves the combination of text and layout by an effective spatial encoding method and an area-based pre-training strategy.

### Parsers for Document Key Information Extraction

BIO tagger, which is a representative parser for entity extraction from the text sequences, extracts key information by identifying spans with the beginning (B) and inside (I) points. Though BIO tagger has been used as a conventional method, it has two limitations for applying to document KIE. One is that the correct order of text blocks is required for extracting key information when post-processing each classified token class (i.e. B- and I- classes). For example, if the text blocks are not ordered properly, such as “recognition, optical, character”, the correct answer can not be made. TheFigure 2: An overview of BROS. The tokens in the document image are masked through token- and area-masking strategy. The position difference between text blocks is encoded directly to the attention mechanism in Transformer. The output token representations are used in both pre-training and fine-tuning.

other is that it cannot solve the tasks that require the relationship between tokens as it performs token-level classification.

To overcome the above two limitations, we adopt a graph-based parser, SPADE (Hwang et al. 2021) decoder, that creates a directed relation graph of tokens to represent key entities and their relationships for KIE tasks. For example, SPADE can determine “optical” as a starting word and “recognition” as the next word. By directly identifying relations between tokens, SPADE enables a description of all key information of KIE tasks regardless of the order of text blocks. In this paper, we apply the SPADE decoder for entity linking tasks of KIE benchmarks and also for all tasks lost perfect order information of text blocks. Specifically, we slightly modify the SPADE decoder for better application with the pre-trained models. The details can be found in the Appendix.

### BERT Relying on Spatiality (BROS)

The main structure of BROS follows LayoutLM (Xu et al. 2020), but there are two critical advances: (1) a use of spatial encoding metric that describes spatial relations between text blocks and (2) a use of 2D pre-training objective designed for text blocks on 2D space. Figure 2 shows a visual description of BROS for document KIE tasks.

### Encoding Spatial Information into BERT

The way to encode spatial information of text blocks decides how text blocks be aware of their spatial relations. LayoutLM (Xu et al. 2020) simply encodes absolute x- and y-axis positions to each text blocks but the specific-point encoding is not robust on the minor position changes of text blocks. Instead, BROS employs relative positions between text blocks to explicitly encode spatial relations. As shown in Figure 3, relative positions provides co-modality of spatial relations between text blocks regardless of their absolute position. This property can make the model better recognize entities which have similar key-value structures.

For formal description, we use  $\mathbf{p} = (x, y)$  to denote a

Figure 3: Comparison between absolute and relative positions. “Project #” and “Total Sample” have their paired values, “72-31” and “285”, respectively. In (a), the paired text blocks have different modalities based their absolute positions. On the other hand, in (b), they can hold co-modality to represent positions of their semantically coupled text blocks.

point on 2D space and a bounding box of a text block consists of four vertices, such as  $\mathbf{p}^{\text{tl}}$ ,  $\mathbf{p}^{\text{tr}}$ ,  $\mathbf{p}^{\text{br}}$ , and  $\mathbf{p}^{\text{bl}}$ , that indicate top-left, top-right, bottom-right, and bottom-left points, respectively. BROS first normalizes all the 2D points of the text blocks using the size of the image. Then, BROS calculates relative positions of the vertices from the same vertices of the other bounding boxes of text blocks and applies sinusoidal functions as  $\bar{\mathbf{p}}_{i,j} = [\mathbf{f}^{\text{sinu}}(x_i - x_j); \mathbf{f}^{\text{sinu}}(y_i - y_j)]$ . Here,  $\mathbf{f}^{\text{sinu}} : \mathbb{R} \rightarrow \mathbb{R}^{D^s}$  indicates a sinusoidal function, which is used in Vaswani et al. (2017),  $D^s$  is the dimensions of sinusoid embedding, and the semicolon (;) indicates concatenation. Through the calculations, the relative positions of  $j^{\text{th}}$  bounding box based on the  $i^{\text{th}}$  bounding box are represented with the four vectors, such as  $\bar{\mathbf{p}}_{i,j}^{\text{tl}}$ ,  $\bar{\mathbf{p}}_{i,j}^{\text{tr}}$ ,  $\bar{\mathbf{p}}_{i,j}^{\text{br}}$ , and  $\bar{\mathbf{p}}_{i,j}^{\text{bl}}$ . Finally, BROS combines the four relative positions by applying a linear transformation,

$$\bar{\mathbf{b}}_{i,j} = \mathbf{W}^{\text{tl}}\bar{\mathbf{p}}_{i,j}^{\text{tl}} + \mathbf{W}^{\text{tr}}\bar{\mathbf{p}}_{i,j}^{\text{tr}} + \mathbf{W}^{\text{br}}\bar{\mathbf{p}}_{i,j}^{\text{br}} + \mathbf{W}^{\text{bl}}\bar{\mathbf{p}}_{i,j}^{\text{bl}}. \quad (1)$$

where  $\mathbf{W}^{\text{tl}}, \mathbf{W}^{\text{tr}}, \mathbf{W}^{\text{br}}, \mathbf{W}^{\text{bl}} \in \mathbb{R}^{(H/A) \times 2D^s}$  are linear transition matrices,  $H$  is a hidden size of BERT, and  $A$  is theYour article of June 8 (Documents Disclose Philip Morris Studied Nicotine's Effect on Body) brings to mind the old adage: If a tree falls in the forest and no one is there to hear it, does it make a sound?

Conversely, if reports of studies on nicotine and smoking have been published in media other than the New York Times, does The Times' own "discovery" of them -- compliments of a "secret source" -- justify front page proclamations of new revelations?

(a) Random *token* selection (red) and *token* masking (gray)

Your article of June 8 (Documents Disclose Philip Morris Studied Nicotine's Effect on Body) brings to mind the old adage: If a tree falls in the forest and no one is there to hear it, does it make a sound?

Conversely, if reports of studies on nicotine and smoking have been published in media other than the New York Times, does The Times' own "discovery" of them -- compliments of a "secret source" -- justify front page proclamations of new revelations?

(b) Random *area* selection (red) and *block* masking (gray)

Figure 4: Illustrations of two masking strategies. The blue boxes represent text blocks including masked tokens. In both figures, 15% of tokens are masked.

number of self-attention heads.

In the process of identifying the relative positional vector,  $\bar{b}b_{i,j}$ , we carefully apply two components: the sinusoidal function,  $f^{\sinus}$ , and the shared embeddings to multiple heads of the attention module. First, the sinusoidal function can encode continuous distances more naturally than using a grid embedding that split a real-valued space into finite number of grids. Second, the multi-head attention modules in Transformer share the same relative positional embeddings to impose the common spatial relationships between text blocks to multiple semantic features identified by the multiple heads.

BROS directly encodes the spatial relations to the contextualization of text blocks. In detail, it calculates an attention logit combining both semantic and spatial features as follows;

$$a_{i,j}^h = (\mathbf{W}_h^q \mathbf{t}_i)^\top (\mathbf{W}_h^k \mathbf{t}_j) + (\mathbf{W}_h^q \mathbf{t}_i)^\top \bar{b}b_{i,j}, \quad (2)$$

where  $\mathbf{t}_i$  and  $\mathbf{t}_j$  are context representations for  $i^{\text{th}}$  and  $j^{\text{th}}$  tokens and both  $\mathbf{W}_h^q$  and  $\mathbf{W}_h^k$  are linear transition matrices for  $h^{\text{th}}$  head. The former is the same as the original attention mechanism in Transformer (Vaswani et al. 2017). The latter, motivated by Dai et al. (2019), considers the relative spatial information of the target text block when the source context and location are given. As we mentioned above, we have shared relative spatial embedding across all of the different attention heads for imposing the common spatial relationships.

Compared to the spatial-aware attention in Xu et al. (2021), which utilizes axis-specific positional difference of text blocks as an attention bias, it has two major differences. First, our method couples the relative embeddings with the semantic information of tokens for better conjugation between texts and their spatial relations. Second, when calculating the relative spatial information between two text blocks, we consider all four vertices of the block. By doing this, our encoding can incorporate not only relative distance but also relative shape and size which play important roles in distinguishing key and value in a document. We compare

our relative encoding method and that of LayoutLMv2's in the ablation study.

## Area-masked Language Model

Pre-training diverse layouts from unlabeled documents is a key factor for document KIE tasks. BROS utilizes two pre-training objectives: one is a token-masked LM (TMLM) used in BERT and the other is a novel area-masked LM (AMLM) introduced in this paper. The area-masked LM, inspired by SpanBERT (Joshi et al. 2020), captures consecutive text blocks based on a 2D area in a document.

TMLM randomly masks tokens while keeping their spatial information, and then the model predicts the masked tokens with the clues of spatial information and the other un-masked tokens. The process is identical to MLM of BERT and Masked Visual-Language Model (MVLM) of LayoutLM. Figure 4 (a) shows how TMLM masks tokens in a document. Since tokens in a text block can be masked partially, their estimation can be conducted by referring to other tokens in the same block or text blocks near the masked token.

AMLM masks all text blocks allocated in a randomly chosen area. It can be interpreted as a span masking for text blocks in 2D space. Specifically, AMLM consists of the following four steps: (1) randomly selects a text block, (2) identifies an area by expanding the region of the text block, (3) determines text blocks allocated in the area, and (4) masks all tokens of the text blocks and predicts them. At the second step, the degree of expansion is identified by sampling a value from an exponential distribution with a hyper-parameter,  $\lambda$ . The rationale behind using exponential distribution is to convert the geometric distribution used in SpanBERT for a discrete domain into a distribution for a continuous domain. Thus, we set  $\lambda = -\ln(1 - p)$  where  $p = 0.2$  used in SpanBERT. Also, we truncated exponential distribution with 1 to prevent an infinity value covering all spaces of the document. It should be noted that the masking area is expanded from a randomly selected text block since the area should be related to the text sizes and locations to represent text spans in 2D space. Figure 4 compares token- and area-masking on text blocks. Because AMLM hides spatially close tokens together, their estimation requires more clues from text blocks far from the estimation targets.

Finally, BROS combines two masked LMs, TMLM and AMLM, to stimulate the model to learn both individual and consolidated token representations. It first masks 15% of tokens for AMLM and then masks 15% of tokens on the left text blocks for TMLM. Similar to BERT (Devlin et al. 2019), the masked tokens are replaced by [MASK] token for 80%, random token for 10%, and original token for the rest 10%.

## Key Information Extraction Tasks

We solve two categories of KIE tasks, entity extraction (EE) and entity linking (EL). The EE task identifies sequences of text blocks that represent desired target texts. Figure 5 (a) is an example of the EE task: identifying header, question, and answer entities in the form-like document. The EL task connects key entities through their hierarchical or semantic relations. Figure 5 (b) is an example(a) An example of FUNSD EE task.

(b) An example of CORD EL task.

Figure 5: Examples of EE and EL tasks. In (a), the colored blocks represent key entities. In (b), the red arrows show the hierarchical relationships between the entities.

<table border="1">
<thead>
<tr>
<th>Dataset</th>
<th>Types</th>
<th>Tasks</th>
<th># Images</th>
</tr>
</thead>
<tbody>
<tr>
<td>FUNSD</td>
<td>Forms</td>
<td>EE, EL</td>
<td>Train 149, Test 50</td>
</tr>
<tr>
<td>SROIE*<sup>†</sup></td>
<td>Receipts</td>
<td>EE</td>
<td>Train 526, Test 100</td>
</tr>
<tr>
<td>CORD</td>
<td>Receipts</td>
<td>EE, EL</td>
<td>Train 800, Val 100, Test 100</td>
</tr>
<tr>
<td>SciTSR</td>
<td>Tables</td>
<td>EL</td>
<td>Train 12,000, Test 3,000</td>
</tr>
</tbody>
</table>

<sup>†</sup> modified version of SROIE. See details in Appendix.

Table 2: Tasks and the number of images for each dataset.

of the EL task: grouping menu entities, such as its name, unit price, amount, and price. Table 2 lists four KIE benchmark datasets: FUNSD (Jaume, Ekenel, and Thiran 2019), SROIE\* (Huang et al. 2019), CORD (Park et al. 2019), and SciTSR (Chi et al. 2019). Details of the benchmarks can be found in Appendix.

Although these four datasets provide testbeds for the EE and EL tasks, they represent the subset of real problems as the order information of text blocks is given. FUNSD provides the orders of text blocks related to target classes in both training and testing examples. In SROIE\*, CORD, and SciTSR, the text blocks are serialized in reading orders. To reflect the real scenario that does not contain perfect order information of text blocks, we remove the order information of KIE benchmarks by randomly permuting the order of text blocks. We denote the permuted datasets as p-FUNSD, p-SROIE\*, p-CORD, and p-SciTSR.

## Experiments

### Experiment Settings

For pre-training, IIT-CDIP Test Collection 1.0<sup>1</sup> (Lewis et al. 2006), which consists of approximately 11M document images, is used but 400K of RVL-CDIP dataset<sup>2</sup> (Harley, Ufkes, and Derpanis 2015) are excluded following LayoutLM. To obtain text blocks from document images,

<sup>1</sup><https://ir.nist.gov/cdip/>

<sup>2</sup><https://www.cs.cmu.edu/aharley/rvl-cdip/>

CLOVA OCR API<sup>3</sup> was applied. We observed no difference in performance depending on the OCR engine; LayoutLM trained in our experimental setting shows comparable performances to the published LayoutLM. The sanity check can be found in Appendix.

The main Transformer structure of BROS is the same as BERT. We set the hidden size, the number of self-attention heads, the feed-forward/filter size, and the number of Transformer layers of BROS<sub>BASE</sub> to 768, 12, 3072, and 12, respectively and those of BROS<sub>LARGE</sub> to 1024, 24, 4096, and 24, respectively. The dimensions of sinusoid embedding  $D^s$  is set to 24 for BROS<sub>BASE</sub> and 32 for BROS<sub>LARGE</sub>.

BROS is trained by using AdamW optimizer (Loshchilov and Hutter 2019) with a learning rate of 5e-5 with linear decay. The batch size is set to 64. During pre-training, the first 10% of the total epochs are used for a warm-up learning rate. We initialized weights of BROS with those of BERT and trained it for 5 epochs on the IIT-CDIP dataset using 8 NVIDIA Tesla V100 32GB GPUs.

During fine-tuning, the learning rate is set to 5e-5. The batch size is set to 16 for all tasks. The number of training epochs or steps is as follows: 100 epochs for FUNSD, 1K steps for SROIE\* and CORD, and 7.5 epochs for SciTSR.

### Experiment Results

To evaluate the performance of the model, we first conduct experiments using the given order of text blocks in the dataset. Then, we verify the robustness of the model against two important challenges in the KIE tasks, which are the dependency about the order of text blocks and learning from a few training examples.

Over our experiments, we report the scores of LayoutLM and LayoutLMv2 using the models published by the authors<sup>4</sup> and denote them as LayoutLM\* and LayoutLMv2\*. We report the mean (and optionally the standard deviation) of the results using the 5 different random seeds.

**With the Order Information of Text Blocks** Table 3 summarizes the results for the FUNSD EE task reported by previous approaches. When comparing models only using text and layout, BROS shows remarkable performance improvements by 2.51 (80.54 → 83.05) for the BASE models and 5.57 (78.95 → 84.52) for the LARGE models from the previous best. Interestingly, BROS provides better or similar performances compared to the multi-modal models incorporating additional visual (*Image\**) or hierarchical (*Cell\**) information. In other words, although BROS does not require extra computations and parameters to process additional features, BROS can achieves better or comparable performances.

Table 4 shows the F1 scores on three EE and EL tasks with the order of text blocks given in the dataset. F, S, C, and Sci refer to FUNSD, SROIE\*, CORD, and SciTSR, respectively. For EE tasks, all models utilize BIO tagger that captures spans of text blocks to represent key entities in documents. For EL tasks, SPADE decoder is used to identify relationships between entities not placed sequentially in a series of

<sup>3</sup><https://clova.ai/ocr>

<sup>4</sup><https://github.com/microsoft/unilm><table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Modality</th>
<th colspan="3">FUNSD EE</th>
<th rowspan="2"># Params</th>
</tr>
<tr>
<th>Precision</th>
<th>Recall</th>
<th>F1</th>
</tr>
</thead>
<tbody>
<tr>
<td>BERT<sub>BASE</sub> (Xu et al. 2020)</td>
<td>Text</td>
<td>54.69</td>
<td>67.10</td>
<td>60.26</td>
<td>110M</td>
</tr>
<tr>
<td>LayoutLM<sub>BASE</sub> (Xu et al. 2020)</td>
<td>Text + Layout</td>
<td>75.97</td>
<td>81.55</td>
<td>78.66</td>
<td>113M</td>
</tr>
<tr>
<td>DocFormer<sub>BASE</sub> (Appalaraju et al. 2021)</td>
<td>Text + Layout</td>
<td>77.63</td>
<td>83.69</td>
<td>80.54</td>
<td>149M</td>
</tr>
<tr>
<td><b>BROS<sub>BASE</sub> (Ours)</b></td>
<td>Text + Layout</td>
<td><b>81.16</b><sub>±0.33</sub></td>
<td><b>85.02</b><sub>±0.32</sub></td>
<td><b>83.05</b><sub>±0.26</sub></td>
<td>110M</td>
</tr>
<tr>
<td>LayoutLM<sub>BASE</sub> (Xu et al. 2020)</td>
<td>Text + Layout + <i>Image</i>*</td>
<td>76.77</td>
<td>81.95</td>
<td>79.27</td>
<td>160M</td>
</tr>
<tr>
<td>LayoutLMv2<sub>BASE</sub> (Xu et al. 2021)</td>
<td>Text + Layout + <i>Image</i>*</td>
<td>80.29</td>
<td>85.39</td>
<td>82.76</td>
<td>200M</td>
</tr>
<tr>
<td>DocFormer<sub>BASE</sub> (Appalaraju et al. 2021)</td>
<td>Text + Layout + <i>Image</i>*</td>
<td>80.76</td>
<td><u>86.09</u></td>
<td>83.34</td>
<td>183M</td>
</tr>
<tr>
<td>SelfDoc (Li et al. 2021b)</td>
<td>Text + Layout + <i>Image</i>*</td>
<td>-</td>
<td>-</td>
<td><u>83.36</u></td>
<td>137M</td>
</tr>
<tr>
<td>StrucTexT (Li et al. 2021c)</td>
<td>Text + Layout + <i>Image</i>*</td>
<td><u>85.68</u></td>
<td>80.97</td>
<td>83.09</td>
<td>107M<sup>†</sup></td>
</tr>
<tr>
<td>BERT<sub>LARGE</sub> (Xu et al. 2020)</td>
<td>Text</td>
<td>61.13</td>
<td>70.85</td>
<td>65.63</td>
<td>340M</td>
</tr>
<tr>
<td>LayoutLM<sub>LARGE</sub> (Xu et al. 2021)</td>
<td>Text + Layout</td>
<td>75.96</td>
<td>82.19</td>
<td>78.95</td>
<td>343M</td>
</tr>
<tr>
<td><b>BROS<sub>LARGE</sub> (Ours)</b></td>
<td>Text + Layout</td>
<td><b>82.81</b><sub>±0.35</sub></td>
<td><b>86.31</b><sub>±0.28</sub></td>
<td><b>84.52</b><sub>±0.30</sub></td>
<td>340M</td>
</tr>
<tr>
<td>LayoutLMv2<sub>LARGE</sub> (Xu et al. 2021)</td>
<td>Text + Layout + <i>Image</i>*</td>
<td>83.24</td>
<td>85.19</td>
<td>84.20</td>
<td>426M</td>
</tr>
<tr>
<td>StructuralLM<sub>LARGE</sub> (Li et al. 2021a)</td>
<td>Text + Layout + <i>Cell</i>*</td>
<td><u>83.52</u></td>
<td><u>86.81</u></td>
<td><u>85.14</u></td>
<td>355M</td>
</tr>
</tbody>
</table>

<sup>†</sup> The number of parameters except for ResNet-FPN processing document images.

Table 3: Performance comparison on the FUNSD EE task. **Bold** indicates the best performance among models using only text and layout, and underline represents the best one. *Image*\* and *Cell*\* denote additional visual and hierarchical information, respectively. Our methods are repeatedly evaluated five times and the values of other methods are the reported scores.

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="3">Entity Extraction</th>
<th colspan="3">Entity Linking</th>
</tr>
<tr>
<th>F</th>
<th>S</th>
<th>C</th>
<th>F</th>
<th>C</th>
<th>Sci</th>
</tr>
</thead>
<tbody>
<tr>
<td>BERT<sub>BASE</sub></td>
<td>60.92</td>
<td>93.67</td>
<td>93.13</td>
<td>27.65</td>
<td>92.83</td>
<td>86.76</td>
</tr>
<tr>
<td>LayoutLM<sub>BASE</sub>*</td>
<td>78.54</td>
<td>95.11</td>
<td>96.26</td>
<td>45.86</td>
<td>95.21</td>
<td>99.05</td>
</tr>
<tr>
<td>LayoutLMv2<sub>BASE</sub>*</td>
<td>81.89</td>
<td>96.09</td>
<td>96.05</td>
<td>42.91</td>
<td>95.59</td>
<td>98.19</td>
</tr>
<tr>
<td><b>BROS<sub>BASE</sub></b></td>
<td><b>83.05</b></td>
<td><b>96.28</b></td>
<td><b>96.50</b></td>
<td><b>71.46</b></td>
<td><b>95.73</b></td>
<td><b>99.45</b></td>
</tr>
<tr>
<td>BERT<sub>LARGE</sub></td>
<td>64.17</td>
<td>94.25</td>
<td>94.74</td>
<td>29.11</td>
<td>94.31</td>
<td>89.23</td>
</tr>
<tr>
<td>LayoutLM<sub>LARGE</sub>*</td>
<td>79.27</td>
<td>95.36</td>
<td>96.12</td>
<td>42.83</td>
<td>95.41</td>
<td>99.33</td>
</tr>
<tr>
<td>LayoutLMv2<sub>LARGE</sub>*</td>
<td>83.59</td>
<td>96.39</td>
<td>97.24</td>
<td>70.57</td>
<td>97.29</td>
<td><b>99.76</b></td>
</tr>
<tr>
<td><b>BROS<sub>LARGE</sub></b></td>
<td><b>84.52</b></td>
<td><b>96.62</b></td>
<td><b>97.28</b></td>
<td><b>77.01</b></td>
<td><b>97.40</b></td>
<td>99.58</td>
</tr>
</tbody>
</table>

Table 4: Performance comparisons on three EE and EL tasks *with* the order information of text blocks.

text blocks. In all cases, BERT performs the worst because those tasks require understanding texts in 2D space, but BERT only encodes 1D sequential information. LayoutLM\* and LayoutLMv2\* show better performance than BERT since they encode layout features as well as text features. And by combining visual features, LayoutLMv2\* performs better than LayoutLM\* in most tasks. BROS shows the best performance in all tasks except SciTSR. It should be noted that the BROS<sub>BASE</sub> show better performance than that of LayoutLM<sub>LARGE</sub>\*, even though it uses three times lower number of parameters (110M vs 343M). These results indicate that BROS effectively encodes the text and layout features.

**Without the Order Information of Text Blocks** As we mentioned in previous section, we introduce the permuted KIE benchmarks lost the orders of text blocks by shuffling the provided orders. To solve EE and EL tasks without the order information, we employ the SPADE decoder for all tasks. Table 5 shows the comparison results. p-F, p-S, p-C, and p-Sci refer to p-FUNSD, p-SROIE\*, p-CORD, and

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="3">Entity Extraction</th>
<th colspan="3">Entity Linking</th>
</tr>
<tr>
<th>p-F</th>
<th>p-S</th>
<th>p-C</th>
<th>p-F</th>
<th>p-C</th>
<th>p-Sci</th>
</tr>
</thead>
<tbody>
<tr>
<td>BERT<sub>BASE</sub></td>
<td>18.85</td>
<td>39.73</td>
<td>59.71</td>
<td>9.59</td>
<td>27.88</td>
<td>1.75</td>
</tr>
<tr>
<td>LayoutLM<sub>BASE</sub>*</td>
<td>33.89</td>
<td>66.05</td>
<td>80.86</td>
<td>22.98</td>
<td>61.51</td>
<td>97.32</td>
</tr>
<tr>
<td>LayoutLMv2<sub>BASE</sub>*</td>
<td>40.77</td>
<td>73.56</td>
<td>80.37</td>
<td>23.25</td>
<td>50.55</td>
<td>95.86</td>
</tr>
<tr>
<td><b>BROS<sub>BASE</sub></b></td>
<td><b>76.94</b></td>
<td><b>82.85</b></td>
<td><b>95.86</b></td>
<td><b>69.61</b></td>
<td><b>87.72</b></td>
<td><b>99.19</b></td>
</tr>
<tr>
<td>BERT<sub>LARGE</sub></td>
<td>18.10</td>
<td>43.19</td>
<td>57.17</td>
<td>10.81</td>
<td>27.12</td>
<td>1.93</td>
</tr>
<tr>
<td>LayoutLM<sub>LARGE</sub>*</td>
<td>33.11</td>
<td>56.84</td>
<td>82.88</td>
<td>20.72</td>
<td>61.98</td>
<td>97.64</td>
</tr>
<tr>
<td>LayoutLMv2<sub>LARGE</sub>*</td>
<td>62.53</td>
<td>84.92</td>
<td>94.43</td>
<td>50.14</td>
<td>85.80</td>
<td><b>99.45</b></td>
</tr>
<tr>
<td><b>BROS<sub>LARGE</sub></b></td>
<td><b>79.42</b></td>
<td><b>85.14</b></td>
<td><b>96.81</b></td>
<td><b>75.61</b></td>
<td><b>90.49</b></td>
<td>99.33</td>
</tr>
</tbody>
</table>

Table 5: Performance comparisons on three EE and EL tasks *without* the order information of text blocks.

<table border="1">
<thead>
<tr>
<th>Model</th>
<th>p-</th>
<th>xy-</th>
<th>yx-</th>
<th>original</th>
</tr>
</thead>
<tbody>
<tr>
<td>LayoutLM<sub>BASE</sub>*</td>
<td>33.89</td>
<td>34.02</td>
<td>55.47</td>
<td>78.47</td>
</tr>
<tr>
<td>LayoutLMv2<sub>BASE</sub>*</td>
<td>40.77</td>
<td>52.08</td>
<td>62.37</td>
<td>78.16</td>
</tr>
<tr>
<td><b>BROS<sub>BASE</sub></b></td>
<td><b>76.94</b></td>
<td><b>77.16</b></td>
<td><b>77.42</b></td>
<td><b>81.61</b></td>
</tr>
<tr>
<td>LayoutLM<sub>LARGE</sub>*</td>
<td>33.11</td>
<td>33.54</td>
<td>41.45</td>
<td>48.30</td>
</tr>
<tr>
<td>LayoutLMv2<sub>LARGE</sub>*</td>
<td>62.53</td>
<td>69.14</td>
<td>75.45</td>
<td>83.00</td>
</tr>
<tr>
<td><b>BROS<sub>LARGE</sub></b></td>
<td><b>79.42</b></td>
<td><b>79.91</b></td>
<td><b>80.02</b></td>
<td><b>83.23</b></td>
</tr>
</tbody>
</table>

Table 6: Comparison of FUNSD EE performances according to sorting methods.

p-SciTSR, respectively. BERT, which does not employ any spatial information of text blocks, shows the worst results on the orderless conditions. By being aware of the spatiality, layout-aware language models show better performances than BERT, and BROS achieves the best except p-SciTSR. More interestingly, BROS shows minor performance drops compared to Table 4, while LayoutLM\* and LayoutLMv2\* suffer from huge performance degradations by losing the or-Figure 6: Performance comparisons according to the amount of fine-tuning data. Each point represents the result of fine-tuning using from 10% to 100% of training data.

<table border="1">
<thead>
<tr>
<th>Dataset</th>
<th># Data</th>
<th>BERT</th>
<th>LayoutLM*</th>
<th>LayoutLMv2*</th>
<th>BROS</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">FUNSD<br/>EE</td>
<td>5</td>
<td>31.51</td>
<td>48.23</td>
<td>64.26</td>
<td><b>68.35</b></td>
</tr>
<tr>
<td>10</td>
<td>40.46</td>
<td>62.50</td>
<td>69.92</td>
<td><b>72.60</b></td>
</tr>
<tr>
<td rowspan="2">FUNSD<br/>EL</td>
<td>5</td>
<td>14.65</td>
<td>21.38</td>
<td>7.32</td>
<td><b>31.11</b></td>
</tr>
<tr>
<td>10</td>
<td>14.88</td>
<td>21.48</td>
<td>13.99</td>
<td><b>39.17</b></td>
</tr>
</tbody>
</table>

Table 7: Results of training with 5 and 10 examples.

der information of text blocks.

To systematically investigate how the order information affects the performance of the models, we construct variants of FUNSD by re-ordering text blocks with two sorting methods based on the top-left points. The text blocks of xy-FUNSD are sorted according to the x-axis with ascending order of y-axis and those of yx-FUNSD are sorted according to y-axis with ascending order of x-axis. Table 6 shows performance on p-FUNSD, xy-FUNSD, yx-FUNSD, and the original FUNSD with the SPADE decoder. In our experiment, LayoutLM<sub>LARGE</sub>\* achieves unstable performance ( $48.30 \pm 12.51$ ), when combined with SPADE decoder. Interestingly, the performances of LayoutLM\* and LayoutLMv2\* are degraded in the order of FUNSD, yx-FUNSD, xy-FUNSD, and p-FUNSD as like the order of the reasonable serialization for text on 2D space. On the other hand, the performance of BROS is relatively consistent. These results show the robustness of BROS on multiple types of serializers.

**Learning from Few Training Examples** One of the advantages of pre-trained models is that it shows effective transfer learning performance even with a few training examples (Devlin et al. 2019). Since collecting fine-tuning data requires a lot of resource, achieving high performance with a small number of training examples is important.

Figure 6 shows the results of the FUNSD KIE tasks by varying the amount of training examples from 10% to 100% during fine-tuning. In all models, performances tend to increase as the ratio of training data increased. In both tasks, BROS shows the best performances regardless of the number of training samples.

To further test extreme cases, we conduct experiments using only 5 and 10 training examples. Table 7 shows the results of the FUNSD KIE tasks. We fine-tune models for 100 epochs with a batch size of 4. In all cases, BROS shows the best performances. The results prove the generalization ability of BROS even with very few training examples.

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="3">Entity Extraction</th>
<th colspan="3">Entity Linking</th>
</tr>
<tr>
<th>F</th>
<th>S</th>
<th>C</th>
<th>F</th>
<th>C</th>
<th>Sci</th>
</tr>
</thead>
<tbody>
<tr>
<td>LayoutLM<sup>†</sup><sub>BASE</sub></td>
<td>76.89</td>
<td>94.99</td>
<td>94.37</td>
<td>44.00</td>
<td>93.60</td>
<td>99.06</td>
</tr>
<tr>
<td>→ <i>pos enc. only</i></td>
<td>78.84</td>
<td>95.45</td>
<td>96.36</td>
<td>59.92</td>
<td>94.83</td>
<td>99.22</td>
</tr>
<tr>
<td>→ <i>objectives only</i></td>
<td>78.44</td>
<td>94.81</td>
<td>95.95</td>
<td>47.22</td>
<td>94.11</td>
<td>99.20</td>
</tr>
<tr>
<td>→ <i>both (= BROS<sub>BASE</sub>)</i></td>
<td><b>80.58</b></td>
<td><b>95.72</b></td>
<td><b>96.64</b></td>
<td><b>65.24</b></td>
<td><b>96.03</b></td>
<td><b>99.28</b></td>
</tr>
</tbody>
</table>

Table 8: Performance improvements on EE and EL tasks through adding components of BROS. At the last line, all components are changed from LayoutLM and the model becomes BROS.

<table border="1">
<thead>
<tr>
<th rowspan="2">SE</th>
<th colspan="3">Entity Extraction</th>
<th colspan="3">Entity Linking</th>
</tr>
<tr>
<th>F</th>
<th>S</th>
<th>C</th>
<th>F</th>
<th>C</th>
<th>Sci</th>
</tr>
</thead>
<tbody>
<tr>
<td>Absolute</td>
<td>78.44</td>
<td>94.81</td>
<td>95.95</td>
<td>47.22</td>
<td>94.11</td>
<td>99.20</td>
</tr>
<tr>
<td>LayoutLMv2’s</td>
<td>78.93</td>
<td>94.71</td>
<td>95.82</td>
<td>53.57</td>
<td>95.27</td>
<td><b>99.28</b></td>
</tr>
<tr>
<td>Ours</td>
<td><b>80.58</b></td>
<td><b>95.72</b></td>
<td><b>96.64</b></td>
<td><b>65.24</b></td>
<td><b>96.03</b></td>
<td><b>99.28</b></td>
</tr>
</tbody>
</table>

Table 9: Spatial encoding methods from BROS’ setting.

## Ablation Study

We conduct ablation studies to investigate which component contributes the performance improvement. For the ablation studies, we utilize LayoutLM<sup>†</sup> that is our own implementation of LayoutLM for fair comparisons under the same experimental settings. All models in these studies are pre-trained for 1 epoch.

Table 8 provides performance changes from adding our proposed components. When applying our proposed positional encoding to LayoutLM, the performances consistently increase with huge margins of 3.62pp on average over all tasks. Independently, our extension on pre-training objectives solely provides 1.14pp of performance improvement on average. By utilizing both, BROS<sub>BASE</sub> provides the best performances with margins of 5.10pp on average. This ablation study proves that each component of BROS solely contributes to performance improvements as well as their combination provides better results.

Table 9 compares three positional encoding methods: absolute position in LayoutLM, relative position in LayoutLMv2, and ours. Relative position methods perform better than absolute one and the performance gap becomes larger in EL tasks. And among them, our method shows the best results.

## Conclusion

We propose a pre-trained language model, BROS, which focuses on modeling text and layout features for effective key information extraction from documents. By encoding texts in 2D space with their relative positions and pre-training the model with the area-masking strategy, BROS shows superior performance without relying on any additional visual features. In addition, under the two real-world settings—imprecise text serialization and small amount of training examples—BROS shows robust performance while other models show significant performance degradation.## Acknowledgments

We thank many colleagues at NAVER CLOVA for their help, in particular Yoonsik Kim, Moonbin Yim, Han-cheol Cho, Bado Lee, Seunghyun Park, and Youngmin Baek for useful discussions.

## References

Appalaraju, S.; Jasani, B.; Kota, B. U.; Xie, Y.; and Manmatha, R. 2021. DocFormer: End-to-End Transformer for Document Understanding. *arXiv preprint arXiv:2106.11539*.

Chi, Z.; Huang, H.; Xu, H.-D.; Yu, H.; Yin, W.; and Mao, X.-L. 2019. Complicated table structure recognition. *arXiv preprint arXiv:1908.04729*.

Clausner, C.; Pletschacher, S.; and Antonacopoulos, A. 2013. The significance of reading order in document recognition and its evaluation. In *2013 12th International Conference on Document Analysis and Recognition (ICDAR)*, 688–692. IEEE.

Dai, Z.; Yang, Z.; Yang, Y.; Carbonell, J. G.; Le, Q.; and Salakhutdinov, R. 2019. Transformer-XL: Attentive Language Models beyond a Fixed-Length Context. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)*.

Denk, T. I.; and Reisswig, C. 2019. BERTgrid: Contextualized Embedding for 2D Document Representation and Understanding. In *Workshop on Document Intelligence at NeurIPS 2019*.

Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), Volume 1 (Long and Short Papers)*, 4171–4186.

Harley, A. W.; Ufkes, A.; and Derpanis, K. G. 2015. Evaluation of deep convolutional nets for document image classification and retrieval. In *Proceedings of the 13th International Conference on Document Analysis and Recognition (ICDAR)*, 991–995.

Huang, Z.; Chen, K.; He, J.; Bai, X.; Karatzas, D.; Lu, S.; and Jawahar, C. 2019. ICDAR2019 competition on scanned receipt ocr and information extraction. In *Proceedings of the 15th International Conference on Document Analysis and Recognition (ICDAR)*, 1516–1520. IEEE.

Hwang, W.; Kim, S.; Seo, M.; Yim, J.; Park, S.; Park, S.; Lee, J.; Lee, B.; and Lee, H. 2019. Post-OCR parsing: building simple and robust parser via BIO tagging. In *Workshop on Document Intelligence at NeurIPS 2019*.

Hwang, W.; Yim, J.; Park, S.; Yang, S.; and Seo, M. 2021. Spatial Dependency Parsing for Semi-Structured Document Information Extraction. In *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*, 330–343.

Jaume, G.; Ekenel, H. K.; and Thiran, J.-P. 2019. FUNSD: A dataset for form understanding in noisy scanned documents. In *2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)*, volume 2, 1–6. IEEE.

Joshi, M.; Chen, D.; Liu, Y.; Weld, D. S.; Zettlemoyer, L.; and Levy, O. 2020. SpanBERT: Improving pre-training by representing and predicting spans. *Transactions of the Association for Computational Linguistics (TACL)*, 8: 64–77.

Lewis, D.; Agam, G.; Argamon, S.; Frieder, O.; Grossman, D.; and Heard, J. 2006. Building a test collection for complex document information processing. In *Proceedings of the 29th annual international ACM SIGIR conference on Research and development in information retrieval (SIGIR)*, 665–666.

Li, C.; Bi, B.; Yan, M.; Wang, W.; Huang, S.; Huang, F.; and Si, L. 2021a. StructuralLM: Structural Pre-training for Form Understanding. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP)*, 6309–6318.

Li, L.; Gao, F.; Bu, J.; Wang, Y.; Yu, Z.; and Zheng, Q. 2020. An End-to-End OCR Text Re-organization Sequence Learning for Rich-text Detail Image Comprehension. In *Proceedings of the 16th European Conference on Computer Vision (ECCV)*.

Li, P.; Gu, J.; Kuen, J.; Morariu, V. I.; Zhao, H.; Jain, R.; Manjunatha, V.; and Liu, H. 2021b. SelfDoc: Self-Supervised Document Representation Learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 5652–5660.

Li, Y.; Qian, Y.; Yu, Y.; Qin, X.; Zhang, C.; Liu, Y.; Yao, K.; Han, J.; Liu, J.; and Ding, E. 2021c. StructTexT: Structured Text Understanding with Multi-Modal Transformers. *arXiv preprint arXiv:2108.02923*.

Liu, X.; Gao, F.; Zhang, Q.; and Zhao, H. 2019. Graph Convolution for Multimodal Information Extraction from Visually Rich Documents. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), Volume 2 (Industry Papers)*, 32–39.

Loshchilov, I.; and Hutter, F. 2019. Decoupled Weight Decay Regularization. In *Proceedings of the 7th International Conference on Learning Representations (ICLR)*.

Park, S.; Shin, S.; Lee, B.; Lee, J.; Surh, J.; Seo, M.; and Lee, H. 2019. CORD: A Consolidated Receipt Dataset for Post-OCR Parsing. In *Workshop on Document Intelligence at NeurIPS 2019*.

Powalski, R.; Borchmann, Ł.; Jurkiewicz, D.; Dwojak, T.; Pietruszka, M.; and Pałka, G. 2021. Going Full-TILT Boogie on Document Understanding with Text-Image-Layout Transformer. *arXiv preprint arXiv:2102.09550*.

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention is all you need. In *Advances in Neural Information Processing Systems 30 (NeurIPS)*, 5998–6008.

Wang, Z.; Xu, Y.; Cui, L.; Shang, J.; and Wei, F. 2021. LayoutReader: Pre-training of Text and Layout for Reading Order Detection. *arXiv:2108.11591*.Xu, Y.; Li, M.; Cui, L.; Huang, S.; Wei, F.; and Zhou, M. 2020. LayoutLM: Pre-training of text and layout for document image understanding. In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD)*, 1192–1200.

Xu, Y.; Xu, Y.; Lv, T.; Cui, L.; Wei, F.; Wang, G.; Lu, Y.; Florencio, D.; Zhang, C.; Che, W.; et al. 2021. LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP)*, 2579–2591.## Two Categories of Document KIE tasks and Parsers for Them

Document KIE tasks can be categorized into two downstream tasks: (1) an entity extraction (EE) task and (2) an entity linking (EL) task. The EE task identifies sequences of text blocks that represent desired target texts (e.g. extracts menu name in a receipt) and the EL task connects key entities through their hierarchical or semantic relations (e.g. connects menu name and its price).

To address the EE and EL tasks, we introduce two parsers. One is a sequence classifier, BIO tagger, that can operate based on prior information about an order of text blocks. It has been utilized as a conventional method for extracting entities from the text sequences. The other is a graph-based parser, SPADE (Hwang et al. 2021) decoder, that does not require any order information of text blocks. Since it directly identifying relations between tokens, it can extract entities whether the order of text blocks is provided or not. It can also perform the task which requires inferring the relationship between text blocks. The following sections briefly introduce the two types of parsers.

### BIO Tagger

BIO tagger, a representative parser that depends on the proper order of text blocks, extracts key information by identifying the beginning (B) and inside (I) points of the ordered text blocks. The proper order of text blocks indicates an order in which all of the key information can be represented in its sub-sequences. The sequence classifier requires this condition because it never succeeds to find the key information with a wrong sequence. For example, if three text blocks, “optical”, “character”, and “recognition”, are ordered as “recognition”, “optical”, and “character”, the sequence classifier cannot find “optical character recognition”.

(a) Recognized text blocks.
(b) Serialized text blocks.
(c) BIO-tagged text sequence.

Figure 7: Visual descriptions of how BIO tagger extracts entities in a document. All recognized tokens are serialized and classified. By combining sub-sequences identified by the BIO taggings, key information can be parsed from the recognized tokens.

Figure 7 shows how the BIO tagger performs the EE task for a given document. First, text blocks are recognized by an OCR engine (Figure 7, a). The recognized text blocks are then serialized by a serializer (Figure 7, b). Finally, for each token, the BIO classes are classified and key information is extracted by combining the classified labels (Figure 7, c).

The BIO tagger cannot solve the EL task since links between text blocks cannot be represented as a sequence unit. In addition, a single text block can hold the same relationships with other multiple text blocks but the sequence-based approach cannot explain the one-to-many relations as well.

### SPADE Decoder

In many practical cases, the proper order of text blocks cannot be available. Most OCR APIs provide the order of text blocks based on rule-based approaches but they cannot guarantee the proper order of text blocks (Clausner, Pletschacher, and Antonacopoulos 2013; Li et al. 2020; Wang et al. 2021).

Here, we utilize SPADE (Hwang et al. 2021) decoder to extract key information without any information about the order. The key idea of SPADE decoder is to extract a directional sub-graph from a fully-connected graph which nodes are text blocks. Due to no limitation on the connections between text blocks, it does not require order information. In our paper, we slightly modified the SPADE decoder for its application to the pre-trained models. The details are described as the followings.

For EE tasks, the SPADE decoder divides the problem into two sub-tasks: initial token classification (Figure 8, a) and subsequent token classification (Figure 8, b). Let  $\tilde{t}_i \in \mathbb{R}^H$  denote the  $i^{\text{th}}$  token representation from the last Transformer layer of the pre-trained model. The initial token classification conducts a token-level tagging to determine whether a token is an initial token of target information as follows,

$$p_{\text{itc}}(\tilde{t}_i) = \text{softmax}(\mathbf{W}^{\text{itc}}\tilde{t}_i), \quad (3)$$

where  $\mathbf{W}^{\text{itc}} \in \mathbb{R}^{(C+1) \times H}$  is a linear transition matrix and  $C$  indicates the number of target classes. Here, the extra +1 dimension is considered to indicate non-initial tokens.Figure 8: Visual descriptions of SPADE decoder downstream tasks. For EE tasks, SPADE decoder combines two sub-tasks such as (a) and (b). SPADE decoder identifies initial tokens and then connects next tokens without any order information of text blocks. For the EL task, SPADE decoder links the first tokens of the entities.

The subsequent token classification is conducted by utilizing pair-wise token representations as follows,

$$p_{\text{stc}}(\tilde{t}_i) = \text{softmax}((\mathbf{W}^{\text{stc-s}}\tilde{t}_i)^\top \mathbf{T}^{\text{stc}})^\top,$$

$$\text{where } \mathbf{T}^{\text{stc}} = [\tilde{t}^{\text{stc}}; \mathbf{W}^{\text{stc-t}}\tilde{t}_1; \dots; \mathbf{W}^{\text{stc-t}}\tilde{t}_N].$$

Here,  $\mathbf{W}^{\text{stc-s}}, \mathbf{W}^{\text{stc-t}} \in \mathbb{R}^{H^{\text{stc}} \times H}$  are linear transition matrices,  $H^{\text{stc}}$  is a hidden feature dimension for the subsequent token classification decoder and  $N$  is the maximum number of tokens. The semicolon (;) indicates concatenation.  $\tilde{t}^{\text{stc}} \in \mathbb{R}^{H^{\text{stc}}}$  is a model parameter to classify tokens which do not have a subsequent token or are not related to any class. It has a similar role with an end-of-sequence token, [EOS], in NLP. By solving these two sub-tasks, the SPADE decoder can identify a sequence of text blocks by finding initial tokens and then connecting subsequent tokens.

For EL tasks, the SPADE decoder conducts a binary classification for all possible pairs of tokens (Figure 8, c) as follows,

$$p_{\text{rel}}(\tilde{t}_i, \tilde{t}_j) = \text{sigmoid}((\mathbf{W}^{\text{rel-s}}\tilde{t}_i)^\top (\mathbf{W}^{\text{rel-t}}\tilde{t}_j)), \quad (4)$$

where  $\mathbf{W}^{\text{rel-s}}, \mathbf{W}^{\text{rel-t}} \in \mathbb{R}^{H^{\text{rel}} \times H}$  are linear transition matrices and  $H^{\text{rel}}$  is a hidden feature dimension. Compared to the subsequent token classification, a single token can hold multiple relations with other tokens to represent hierarchical structures of document layouts.

In our experiments,  $H^{\text{stc}}$  and  $H^{\text{rel}}$  are set to 128 for FUNSD, 64 for SROIE\*, and 256 for CORD and SciTSR.

## KIE Benchmark Datasets

Here, we describe three EE tasks and three EL tasks from four KIE benchmark datasets.

- • Form Understanding in Noisy Scanned Documents (FUNSD) (Jaume, Ekenel, and Thiran 2019) is a set of documents with various forms. The dataset consists of 149 training and 50 testing examples. FUNSD has both EE and EL tasks. In the EE task, there are three semantic entities: Header, Question, and Answer. In the EL task, the semantic hierarchies are represented as relations between text blocks like header-question and question-answer pairs.
- • SROIE\* is a variant of Task 3 of ‘‘Scanned Receipts OCR and Information Extraction’’ (SROIE)<sup>5</sup> that consists of a set of store receipts. In the original SROIE task, semantic contents (Company, Date, Address, and Total price) are generated without explicit connection to the text blocks. To convert SROIE into a EE task, we developed SROIE\* by matching ground truth contents with text blocks. We also split the original training set into 526 training and 100 testing examples because the ground truths are not given in the original test set. SROIE\* will be publicly available.
- • Consolidated Receipt Dataset (CORD) (Park et al. 2019) is a set of store receipts with 800 training, 100 validation, and 100 testing examples. CORD consists of both EE and EL tasks. In the EE task, there are 30 semantic entities including menu name, menu price, and so on. In the EL task, the semantic entities are linked according to their layout structure. For example, menu name entities are linked to menu id, menu count, and menu price.
- • Complicated Table Structure Recognition (SciTSR) (Chi et al. 2019) is an EL task that connects cells in a table to recognize the table structure. There are two types of relations: vertical and horizontal connections between cells. The dataset consists of 12,000 training images and 3,000 test images.

Figure 9 shows the sample images of these benchmark datasets.

<sup>5</sup><https://rrc.cvc.uab.es/?ch=13>Manuscript Review Form

TOBACCO SCIENCE

Registration No. 533 Date March 18, 1968

AUTHORS Andrew G. Kallianos, Richard E. Means, James D. Hold

TITLE "Effect of Nitrates in Tobacco on the Catechol Yield in Cigarette Smoke"

REVIEW COMPLETED 3/29/68 RECOMMENDATION:  APPROVE IN ITS

PRESENT FORM;  NOT APPROVE (Give reasons below);  APPROVE TENTATIVELY,

SUBJECT TO THE FOLLOWING SUGGESTED REVISIONS: (itemize below):

Page 4 - Last line should be Mass spectrometric, instead of Mass spectroscopic.

00070353

NOTE—Execute in triplicate using additional sheets if more space is required. Retain the third copy for your file. Return the original (signed) and the first carbon (unsigned) along with the manuscript to this office. The unsigned copy and the manuscript will be returned to the author for his consideration.

(a) FUNSD

(c) CORD

**tan woon yann**

BOOK TAJIK (TAMAN DAYA) SDN BHD

789-417-W

NO. 55, 57 & 59, JALAN SAGU 18,  
TAMAN DAYA,  
81100 JOHOR BAHRU,  
JOHOR.

Document No : TD01167104

Date : 25/12/2018 8:13:39 PM

Cashier : MANIS

Member :

CASH BILL

<table border="1">
<thead>
<tr>
<th>CODE/DESC</th>
<th>PRICE</th>
<th>Disc</th>
<th>AMOUNT</th>
</tr>
<tr>
<th>QTY</th>
<th>RM</th>
<th></th>
<th>RM</th>
</tr>
</thead>
<tbody>
<tr>
<td>9556939040116 KF MODELLING CLAY KIDDY FISH</td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>1 PC *</td>
<td>9.00</td>
<td>0.00</td>
<td>9.00</td>
</tr>
<tr>
<td colspan="3">Total :</td>
<td>9.00</td>
</tr>
<tr>
<td colspan="3">Rounding Adjustment :</td>
<td>0.00</td>
</tr>
<tr>
<td colspan="3">Rounded Total (RM):</td>
<td>9.00</td>
</tr>
<tr>
<td colspan="3">Cash</td>
<td>10.00</td>
</tr>
<tr>
<td colspan="3">CHANGE</td>
<td>1.00</td>
</tr>
</tbody>
</table>

2.00

GOODS SOLD ARE NOT RETURNABLE OR EXCHANGEABLE

售出物品，恕不退換或更換  
如有下換，請語聲謝，謝謝！

THANK YOU  
PLEASE COME AGAIN !

(b) SROIE

<table border="1">
<thead>
<tr>
<th>Parameter</th>
<th>Code</th>
<th>Value for O<sub>2</sub></th>
<th>Value for Texas</th>
</tr>
</thead>
<tbody>
<tr>
<td>System class</td>
<td>SYSCCLASS</td>
<td>Page server</td>
<td>Centralized</td>
</tr>
<tr>
<td>Network throughput</td>
<td>NETTHRU</td>
<td>+∞</td>
<td>N/A</td>
</tr>
<tr>
<td>Disk page size</td>
<td>PGSIZE</td>
<td>4096 bytes</td>
<td>4096 bytes</td>
</tr>
<tr>
<td>Buffer size</td>
<td>BUFFSIZE</td>
<td>3840 pages</td>
<td>3275 pages</td>
</tr>
<tr>
<td>Buffer page replacement strategy</td>
<td>PGREP</td>
<td>LRU</td>
<td>LRU</td>
</tr>
<tr>
<td>Prefetching policy</td>
<td>PREFETCH</td>
<td>None</td>
<td>None</td>
</tr>
<tr>
<td>Object clustering policy</td>
<td>CLUSTP</td>
<td>None</td>
<td>DSTC</td>
</tr>
<tr>
<td>Objects initial placement</td>
<td>INITPL</td>
<td>Optimized Sequential</td>
<td>Optimized Sequential</td>
</tr>
<tr>
<td>Disk search time</td>
<td>DISKSEA</td>
<td>6.3 ms</td>
<td>7.4 ms</td>
</tr>
<tr>
<td>Disk latency time</td>
<td>DISKLAT</td>
<td>2.99 ms</td>
<td>4.3 ms</td>
</tr>
<tr>
<td>Disk transfer time</td>
<td>DISKTRA</td>
<td>0.7 ms</td>
<td>0.5 ms</td>
</tr>
<tr>
<td>Multiprogramming level</td>
<td>MULTILVL</td>
<td>10</td>
<td>1</td>
</tr>
<tr>
<td>Locks acquisition time</td>
<td>GETLOCK</td>
<td>0.5 ms</td>
<td>0</td>
</tr>
<tr>
<td>Locks release time</td>
<td>RELLOCK</td>
<td>0.5 ms</td>
<td>0</td>
</tr>
<tr>
<td>Number of users</td>
<td>NUSERS</td>
<td>1</td>
<td>1</td>
</tr>
</tbody>
</table>

(d) SciTSR

Figure 9: The sample images of four KIE benchmark datasets.<table border="1">
<thead>
<tr>
<th># Pre-training data</th>
<th># Epochs</th>
<th>Model</th>
<th>Precision</th>
<th>Recall</th>
<th>F1</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">500K</td>
<td rowspan="2">1</td>
<td>LayoutLM<sub>BASE</sub> (Xu et al. 2020)</td>
<td>0.5779</td>
<td>0.6955</td>
<td>0.6313</td>
</tr>
<tr>
<td>LayoutLM<sup>†</sup><sub>BASE</sub></td>
<td>0.5823</td>
<td>0.6935</td>
<td>0.6330</td>
</tr>
<tr>
<td rowspan="2">1M</td>
<td rowspan="2">1</td>
<td>LayoutLM<sub>BASE</sub> (Xu et al. 2020)</td>
<td>0.6156</td>
<td>0.7005</td>
<td>0.6552</td>
</tr>
<tr>
<td>LayoutLM<sup>†</sup><sub>BASE</sub></td>
<td>0.6142</td>
<td>0.7151</td>
<td>0.6608</td>
</tr>
<tr>
<td rowspan="2">2M</td>
<td rowspan="2">1</td>
<td>LayoutLM<sub>BASE</sub> (Xu et al. 2020)</td>
<td>0.6599</td>
<td>0.7355</td>
<td>0.6957</td>
</tr>
<tr>
<td>LayoutLM<sup>†</sup><sub>BASE</sub></td>
<td>0.6562</td>
<td>0.7456</td>
<td>0.6980</td>
</tr>
<tr>
<td rowspan="4">11M</td>
<td rowspan="2">1</td>
<td>LayoutLM<sub>BASE</sub> (Xu et al. 2020)</td>
<td>0.7464</td>
<td>0.7815</td>
<td>0.7636</td>
</tr>
<tr>
<td>LayoutLM<sup>†</sup><sub>BASE</sub></td>
<td>0.7384</td>
<td>0.8022</td>
<td>0.7689</td>
</tr>
<tr>
<td rowspan="2">2</td>
<td>LayoutLM<sub>BASE</sub> (Xu et al. 2020)</td>
<td>0.7597</td>
<td>0.8155</td>
<td>0.7866</td>
</tr>
<tr>
<td>LayoutLM<sup>†</sup><sub>BASE</sub></td>
<td>0.7612</td>
<td>0.8188</td>
<td>0.7889</td>
</tr>
</tbody>
</table>

Table 10: Sanity checking of LayoutLM<sup>†</sup> by comparing its performances on FUNSD EE task from the reported scores in Xu et al. (2020). LayoutLM<sup>†</sup> is trained in our experimental settings including pre-training datasets and hardware devices.

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="3">BIO tagger</th>
<th colspan="3">SPADE decoder</th>
</tr>
<tr>
<th>F</th>
<th>S</th>
<th>C</th>
<th>F</th>
<th>S</th>
<th>C</th>
</tr>
</thead>
<tbody>
<tr>
<td>BERT<sub>BASE</sub></td>
<td>60.92</td>
<td>93.67</td>
<td>93.13</td>
<td>63.38</td>
<td>93.09</td>
<td>95.16</td>
</tr>
<tr>
<td>LayoutLM<sup>*</sup><sub>BASE</sub></td>
<td>78.54</td>
<td>95.11</td>
<td>96.26</td>
<td>78.47</td>
<td>93.33</td>
<td>96.71</td>
</tr>
<tr>
<td>LayoutLMv2<sup>*</sup><sub>BASE</sub></td>
<td>81.89</td>
<td>96.09</td>
<td>96.05</td>
<td>78.16</td>
<td>95.32</td>
<td>96.13</td>
</tr>
<tr>
<td>BROS<sub>BASE</sub></td>
<td><b>83.05</b></td>
<td><b>96.28</b></td>
<td><b>96.50</b></td>
<td><b>81.61</b></td>
<td><b>95.70</b></td>
<td><b>96.73</b></td>
</tr>
<tr>
<td>BERT<sub>LARGE</sub></td>
<td>64.17</td>
<td>94.25</td>
<td>94.74</td>
<td>65.23</td>
<td>93.40</td>
<td>95.30</td>
</tr>
<tr>
<td>LayoutLM<sup>*</sup><sub>LARGE</sub></td>
<td>79.27</td>
<td>95.36</td>
<td>96.12</td>
<td>48.30</td>
<td>93.90</td>
<td>95.63</td>
</tr>
<tr>
<td>LayoutLMv2<sup>*</sup><sub>LARGE</sub></td>
<td>83.59</td>
<td>96.39</td>
<td>97.24</td>
<td>83.00</td>
<td>96.62</td>
<td>97.18</td>
</tr>
<tr>
<td>BROS<sub>LARGE</sub></td>
<td><b>84.52</b></td>
<td><b>96.62</b></td>
<td><b>97.28</b></td>
<td><b>83.23</b></td>
<td><b>96.69</b></td>
<td><b>97.32</b></td>
</tr>
</tbody>
</table>

Table 11: Performance comparisons of BIO tagger and SPADE decoder for the three EE tasks *with* the order information of text blocks.

<table border="1">
<thead>
<tr>
<th>Model</th>
<th>Speed (BASE)</th>
<th>Speed (LARGE)</th>
</tr>
</thead>
<tbody>
<tr>
<td>LayoutLM<sup>*</sup></td>
<td>0.0413s <math>\pm</math> 0.0008</td>
<td>0.1182s <math>\pm</math> 0.0027</td>
</tr>
<tr>
<td>LayoutLMv2<sup>*</sup></td>
<td>0.1438s <math>\pm</math> 0.0009</td>
<td>0.2330s <math>\pm</math> 0.0011</td>
</tr>
<tr>
<td>BROS</td>
<td>0.0721s <math>\pm</math> 0.0007</td>
<td>0.1794s <math>\pm</math> 0.0021</td>
</tr>
</tbody>
</table>

Table 12: Inference speed (Avg. of 50 samples on T4 GPU).

## Compare Published LayoutLM Model and Our Own Implementation

Table 10 compares our implementation of LayoutLM from the reported scores in Xu et al. (2020). As can be seen, multiple experiments are conducted according to the number of pre-training data. Our implementation, referred to LayoutLM<sup>†</sup>, shows comparable performances over all settings.

## Compare BIO Tagger and SPADE Decoder

Table 11 shows the performance comparisons of BIO tagger and SPADE decoder for the three EE tasks when the order of text blocks is given. There is not much difference in performance between the BIO tagger and the SPADE decoder in this case. Interestingly, the performance of BERT on FUNSD and CORD is higher when using SPADE decoder than when using BIO tagger. This seems to be because 2D layout information can be utilized by using the SPADE decoder. In FUNSD EE task, the performance of LayoutLM<sup>\*</sup><sub>LARGE</sub> is unstable to have a standard deviation of 12.51. BROS shows the best performance in all tasks regardless of the parsers.

## Compare the Inference Speed of the Models

Table 12 shows the inference speed of the models. Since BROS considers relative positions for all text block pairs, it is slower than LayoutLM, but faster than LayoutLMv2 using image features. It should be noted that BROS shows the best performance among them.Figure 10: Results of rotated FUNSD EE task.

### Experiments on Rotated Images

To test in another imprecise text serialization setup, we conducted experiment on the rotated images in which the image and locations of all text blocks are rotated and the block order is re-serialized (yx-). Figure 10 shows the results with the same trend as Table 6.

