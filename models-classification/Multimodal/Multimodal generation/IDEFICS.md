# OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents

Hugo Laurençon<sup>\*,1,2</sup> Lucile Saulnier<sup>\*,1</sup> Léo Tronchon<sup>\*,1</sup>  
 Stas Bekman<sup>\*,1</sup> Amanpreet Singh<sup>\*,1</sup> Anton Lozhkov<sup>1</sup>  
 Thomas Wang<sup>1</sup> Siddharth Karamcheti<sup>1,3</sup> Alexander M. Rush<sup>†,1</sup>  
 Douwe Kiela<sup>†,1,3</sup> Matthieu Cord<sup>†,2</sup> Victor Sanh<sup>\*,†,1</sup>

<sup>\*</sup>Equal contributions, <sup>†</sup>Senior contributions

[hugo@huggingface.co](mailto:hugo@huggingface.co)

<sup>1</sup>Hugging Face <sup>2</sup>Sorbonne Université <sup>3</sup>Stanford University

## Abstract

Large multimodal models trained on natural documents, which interleave images and text, outperform models trained on image-text pairs on various multimodal benchmarks. However, the datasets used to train these models have not been released, and the collection process has not been fully specified. We introduce the OBELICS dataset, an open web-scale filtered dataset of interleaved image-text documents comprising 141 million web pages extracted from Common Crawl, 353 million associated images, and 115 billion text tokens. We describe the dataset creation process, present comprehensive filtering rules, and provide an analysis of the dataset’s content. To show the viability of OBELICS, we train vision and language models of 9 and 80 billion parameters named IDEFICS, and obtain competitive performance on different multimodal benchmarks. We release our dataset, models and code.<sup>1</sup>.

## 1 Introduction

Recent systems demonstrate the effectiveness of training large multimodal models such as Flamingo on naturally occurring multimodal documents (Alayrac et al., 2022; Aghajanyan et al., 2022; Huang et al., 2023). A multimodal document is a succession of text paragraphs interleaved by images, such as web pages that contain images. Models trained on these web documents outperform vision and language models trained solely on image-text pairs on various benchmarks (Alayrac et al., 2022). They can also generate long and coherent text about a set of multiple images.

While these results are compelling, they have not been replicable. The datasets used in these works are not publicly available, and relatively little information is known about their creation process and composition. This state motivates the creation of large-scale collections of high-quality multimodal web documents to support the creation of the next generation of models.

We take inspiration from existing large open image-text datasets such as LAION (Schuhmann et al., 2022) and COYO (Byeon et al., 2022), comprised of hundreds of millions of image-text

OBELICS: <https://huggingface.co/datasets/HuggingFaceM4/OBELICS>

<sup>1</sup>OBELICS reproduction code: <https://github.com/huggingface/OBELICS>

IDEFICS models: <https://huggingface.co/HuggingFaceM4/idefics-80b>Figure 1: A comparison of extraction from the same web document. For image-text pairs, the alt-text of images is often short or non-grammatical. For **OBELICS**, the extracted multimodal web document interleaves long-form text with the images on the page.

pairs obtained through web crawling. These datasets have been critical to developing and replicating numerous recent multimodal models (Radford et al., 2021; Wang et al., 2022; Yu et al., 2022; Wang et al., 2022; Liu et al., 2023). While this approach allows for building extremely large and diverse training datasets, we note several limitations to using only image-text pairs. From a language perspective, these datasets rely primarily on alt-text, meaning the text given is brief, captures an approximate snapshot of the image’s content, and often lacks grammatical correctness. From a document perspective, image-text pairs remove an image from its natural context on a page and its relationship with other documents.

In this work, we introduce **OBELICS**<sup>2</sup>, an openly-accessible curated web-scale dataset consisting of 141 million multimodal English web documents which contain 353 million associated images and 115 billion tokens. **OBELICS** collects full multimodal documents interleaving text and images as shown in Figure 1. We describe the dataset creation process, outline the filtering and curation steps and shed light on the dataset’s content and limitations. To demonstrate the viability of **OBELICS**, we train **IDEFICS**, an 80 billion parameter multimodal model and show competitive performance against large-scale multimodal models such as Flamingo (Alayrac et al., 2022).

## 2 Related Works

**Image-text pairs datasets** The largest multimodal datasets, such as LAION (Schuhmann et al., 2021, 2022), Conceptual Captions (Sharma et al., 2018; Changpinyo et al., 2021), ALIGN (Jia et al., 2021), COYO (Byeon et al., 2022), and DataComp (Gadre et al., 2023), contain billions of image-text pairs and are usually obtained through web-crawling and alt-text extraction. A variety of multimodal models have been trained on this type of dataset: multimodal encoder models which use a contrastive objective (Radford et al., 2021; Wang et al., 2022), image generation based on Transformers or diffusion processes (Nichol et al., 2022; Ramesh et al., 2022; Rombach et al., 2021; Saharia et al., 2022). While the scale of these datasets makes them attractive candidates for training, our work focuses on extracting images and the textual context in which they appear instead of extracting the associated alternative text.

**Web document datasets** Insights from scaling language models (Kaplan et al., 2020; Hoffmann et al., 2022) emphasize the need for increasingly bigger datasets. For instance,

<sup>2</sup>Open Bimodal Examples from Large flttered Commoncrawl SnapshotsLLaMA (Touvron et al., 2023) was trained on a dataset of 1.4T tokens created exclusively from openly accessible English web content. The authors noticed that an even bigger dataset would have benefited the model. To address that need, multiple web-scale datasets have been introduced and made available: c4 (Raffel et al., 2019), ROOTS (Laurençon et al., 2022), Pile (Gao et al., 2020), OSCAR (Ortiz Suárez et al., 2020). Although OBELICS falls in the same category of making accessible large collections of curated web documents, the additional extraction of images changes the nature of the resulting dataset. It allows training models with additional vision capabilities.

**Multimodal web document datasets** The recent most performant vision and language models are trained on large sets of multimodal web documents. For instance, Flamingo (Alayrac et al., 2022), an 80 billion multimodal model, was trained on a mix of 2.1 billion image-text pairs, 27 million video-text pairs, and 43 million multimodal web documents. The latter called M3W, includes 185 million images. Similarly, KOSMOS-1 (Huang et al., 2023) was trained on a mixture containing 71 million multimodal web documents. However, in both cases, the dataset is not publicly available, and little information is accessible as to the dataset’s content, the strategies employed to create that dataset (including filtering strategies), and the quality of the resulting web documents, which ultimately hinders further research.

Concurrently to our work, the Multimodal C4 (**mmc4**) dataset (Zhu et al., 2023) was recently made accessible. It consists of 103 million multimodal web documents that include 585 million images. Although there are similarities between our datasets, it is important to highlight particular distinctions. First, our dataset is based on more recent documents from February 2020 to February 2023, whereas **mmc4** uses documents from April 2019. Additionally, our filtering heuristics appear to be more comprehensive: we leverage the HTML DOM trees to filter out undesirable texts and images, whereas **mmc4** uses the HTML to find images in order to merge them with the original C4 dataset by solving a bipartite assignment problem based on a CLIP model similarities. Last, we implement additional deduplication steps at the image, document, and paragraph levels.

### 3 Creation of the Multimodal Web Document Dataset

```

graph LR
    A[Common Crawl data  
41.2B docs] --> B[Collecting a large number of HTML files  
• Selection of English content  
• Early text deduplication  
• Quality classification  
1.1B docs]
    B --> C[Simplifying HTML files  
• DOM tree cleaning strategies  
• Tag unwrapping  
• Node removal  
• Modification of specific nodes  
10x smaller HTML files]
    C --> D[Extracting multimodal web documents  
• Preservation of the original structure of the web pages  
• Image downloading  
1.1B docs  
2B images]
    D --> E[Filtering multimodal web documents  
• Node level image filtering  
• Paragraph-level text filtering  
• Document-level filtering  
365M docs  
1.4B images]
    E --> F[Responsible filtering  
• Exclusion of opted-out images  
• NSFW images removal]
    F --> G[Deduplicating  
• Image deduplication  
• Document deduplication  
• Paragraph deduplication  
141M docs  
353M images]
    G --> H[OBELICS]
  
```

Figure 2: Overview of the steps involved in creating OBELICS.

This section provides an overview of the critical choices of the creation and filtering process. Figure 2 gives a high-level summary of the main steps involved. Many details are omitted from this section, and we invite the reader to refer to the appendix A.1 for completeness.### 3.1 Collecting a Large Number of HTML Files

First, we collect a vast amount of raw web documents by considering the 25 most recent Common Crawl dumps at the time of the creation, spanning from February 2020 to January/February 2023<sup>3</sup>. We extract the main text from the documents while discarding documents with text of insufficient quality. This process results in 41.2 billion documents.

To filter out non-English content, we apply the FastText classifier (Joulin et al., 2017) to the extracted text, which removes 63.6% of the documents. We perform a MinHash (Broder, 1997) deduplication to remove duplicate content. Additionally, we filter out documents with significant proportions of repeated paragraphs and n-grams, following the methodology used in MassiveText (Rae et al., 2022). Previous studies (Lee et al., 2022; Abbas et al., 2023) have demonstrated the prevalence of duplication in crawled data and the benefits of training on deduplicated data.

Similar to Brown et al. (2020), we employ a logistic regression classifier with hashed token frequencies to ensure high-quality text. This classifier, trained using curated datasets like Wikipedia or OpenWebText (Gokaslan and Cohen, 2019) as positive examples and documents sampled from Common Crawl as negative ones, is fast and effective at detecting human-written text. After these steps, we are left with 1.1 billion documents and their HTML sources from the associated Common Crawl WARC files.

### 3.2 Simplifying HTML Files

The original HTML content of a document contains a wealth of valuable information that proves highly beneficial in the process of filtering out undesirable text and images. Therefore, we prioritize pre-processing the raw HTML into simplified HTML, making the subsequent extraction of textual and visual elements more efficient.

To this aim, we devise multiple pre-processing strategies for an HTML DOM tree. By manually inspecting instances of all HTML nodes, we differentiate nodes likely to contain relevant texts or images from those that should be discarded, and we formulate specific rules for each type of node. After these pre-processing steps, the resulting simplified HTML files are more than ten times smaller and have been stripped of a large proportion of generic text (spam, ads, boilerplate template, etc.) and generic images, such as logos, while retaining the relevant content.

### 3.3 Extracting Multimodal Web Documents

In this step, we transform the simplified HTML files previously obtained into a structured web multimodal web document format. This format consists of interleaved texts and images.

We meticulously preserve the original structure of the web pages from the simplified HTML files by extracting the texts and image links while maintaining their rendering defined by the DOM tree. Given that each HTML tag denotes a distinct separation between the preceding and subsequent nodes, we leverage that information to retain line breaks and line feeds on the original page, preserving the formatting and visual rendering of the content.

We obtain 3.6 billion image links and successfully download 55% of them (approximately 2 billion images).

### 3.4 Filtering Multimodal Web Documents

The filtering process comprises two distinct steps operating at different granularity levels. In the first step, filtering occurs at the node level for images and the paragraph level for text. This step guarantees that only high-quality and relevant images and paragraphs are retained. Each paragraph or image is evaluated based on specific criteria and may undergo modifications or be eliminated if necessary. The second step, conducted at the document level, involves deciding whether to retain or discard the output documents obtained from the

---

<sup>3</sup><https://commoncrawl.org/>first step. Most text filters used in both steps are primarily derived from Laurençon et al. (2022).

**Node-level image filtering** We discard images that are too small, excessively large or have disproportionate dimensions. We observe that these images are often indicative of low-quality or irrelevant content. To eliminate some logos and generic images, we remove images whose URLs contain one of the banned sub-strings, like *logo*.

**Paragraph-level text filtering** We apply multiple filters to text paragraphs to remove undesirable content. Specifically, paragraphs that contain an insufficient number of words are discarded. Additionally, we filter out paragraphs with high repetition ratios, excessive ratios of special characters, low ratios of stop words, low punctuation ratios, high proportions of flagged words associated with adult or inappropriate content, or excessively high perplexity scores (as measured by an n-gram language model trained on Wikipedia (Heafield, 2011)). To identify boilerplate sentences or invitations to share articles on social networks, we create a list of frequently used words associated with these paragraphs and remove paragraphs containing an excessive proportion of words from this list. To further identify machine-generated content, we extract words from web-crawled documents to form a list of common words and discard documents with a low ratio of common words.

**Document-level filtering** At the document level, we remove all documents with no or excessively high number of images. For text filters, the same filters used at the paragraph level are applied, with sometimes stricter cutoff values.

After these filtering steps, we are left with 365 million web documents and 1.4 billion images. At this step, images can be duplicated across documents.

### 3.5 Responsible Filtering and Deduplication

We take measures to minimize the amount of inappropriate content in the dataset. In particular, based on manual inspections and tool availability, we implement filters to respect data consent and remove images with pornographic content. Additionally, we also heavily deduplicate content.

**Exclusion of opted-out images** To respect the preferences of content creators, we remove all images for which creators explicitly opted out of AI model training. We used the Spawning API<sup>4</sup> to verify that the images in the dataset respect the original copyright owners' choices.

**Image deduplication based on URL** Some images could be present across different documents. We observe that it is particularly true for browser-specific icons or common advertisements encountered during the crawling process. To address this issue, we remove all images that appear more than ten times across the entire dataset. We intentionally do not perform strict deduplication, as we notice that when an image is duplicated only a few times across different documents, the surrounding text and contextual information tend to be different. We also deduplicate images within the same document.

**NSFW image filtering** To reduce explicit adult content, we use an open-source NSFW classifier to remove entire documents containing pornographically classified images. We also filter out images with URLs containing banned sub-strings.

**Document deduplication based on URL and set of images** We complete the initial deduplication step by forming clusters of documents with the same URLs and retaining the most recent document within each cluster. We repeat this operation by forming clusters of documents containing identical sets of images.

**Paragraph deduplication across documents of the same domain names** To remove generic spam phrases commonly found at the end of documents, we perform paragraph-level

---

<sup>4</sup><https://api.spawning.ai/spawning-api>exact deduplication within documents sharing the same domain name, resulting in the elimination of approximately 15% of the text.

Following these filtering and deduplication steps, the final dataset contains 141 million documents and 353 million images, of which 298 million are unique. We observe that using stricter values for the filtering steps yields fewer multimodal documents, although not of higher quality. As such, we invite users who are interested in manipulating a smaller subset of **OBELICS** to start with a random subset.

## 4 Analysis of **OBELICS**

Figure 1 provides an example showcasing an original webpage alongside the resulting multimodal web document. Extracting and filtering the multimodal document is non-trivial as it requires carefully removing undesirable information on the left, top, and bottom of the page, such as menus and navigation bars. We provide other examples at [https://huggingface.co/spaces/HuggingFaceM4/obelics\\_visualization](https://huggingface.co/spaces/HuggingFaceM4/obelics_visualization) and in Figures 7, 8 and 9.

Given the scale of **OBELICS**, it would be prohibitive to describe its content exhaustively. Instead, we provide high-level statistics and analyses that shed light on the dataset’s properties.

### 4.1 General Statistics

<table border="1">
<thead>
<tr>
<th>Dataset</th>
<th>Images</th>
<th>%<br/>unique<br/>images</th>
<th>Docs</th>
<th>Tokens</th>
<th>Open</th>
</tr>
</thead>
<tbody>
<tr>
<td>KOSMOS-1</td>
<td>-</td>
<td>-</td>
<td>71M</td>
<td>-</td>
<td>✗</td>
</tr>
<tr>
<td>M3W</td>
<td>185M</td>
<td>-</td>
<td>43M</td>
<td>-</td>
<td>✗</td>
</tr>
<tr>
<td>mmc4-ff</td>
<td>385M</td>
<td>60.6%</td>
<td>79M</td>
<td>34B</td>
<td>✓</td>
</tr>
<tr>
<td>mmc4</td>
<td><b>585M</b></td>
<td>-</td>
<td>103M</td>
<td>43B</td>
<td>✓</td>
</tr>
<tr>
<td><b>OBELICS</b></td>
<td>353M</td>
<td><b>84.3%</b></td>
<td><b>141M</b></td>
<td><b>115B</b></td>
<td>✓</td>
</tr>
</tbody>
</table>

Table 1: General statistics of **OBELICS** and the current largest alternatives.

Figure 3: Distribution of images.

Table 1 compares **OBELICS** against the largest existing alternatives. **mmc4-ff** is the **mmc4** dataset with fewer faces. Our dataset has the highest number of unique documents and total tokens while containing a huge number of images.

It is worth mentioning that we have fewer images than **mmc4** (Zhu et al., 2023). This discrepancy can be attributed to two reasons. First, our analysis reveals that **mmc4** contains many duplicated images, with only 60.6% being unique compared to 84.3% for **OBELICS**. We found that images duplicated multiple times often indicate spam or unrelated generic content. Second, **mmc4** does not limit the number of images within a document. As a result, the distribution of images across documents is highly uneven, with a substantial portion of them concentrated in documents with excessive image counts (see Figure 3). The images in these documents are often unrelated to each other and exhibit spam or advertisement content. Moreover, these documents often have little text, making them unsuitable for learning the alignment between text and images (see an example in Figure 10).

Figure 4 shows the joint distribution of a number of tokens and a number of images in **OBELICS**. Although we limit the number of images in a document to 30, we cut the plot at 6 images for clarity. The documents of **OBELICS** contain a median number of images of 1 and a median number of tokens of 677.

**Perplexity analysis** To assess the quality of our text in comparison to reference datasets used for training large language models, we leverage an n-gram language model trained on Wikipedia (Heafield, 2011; Laurençon et al., 2022). This allows us to compute perplexityFigure 4: Heatmap displaying the joint distribution of the number of tokens and the number of images in OBELICS documents, accompanied by their respective marginal distributions.

Figure 5: Kernel density estimations representing the distribution of perplexity scores for OBELICS compared to reference datasets. The lower the perplexity for a document, the more it resembles a Wikipedia article.

scores for 100,000 documents from each dataset. Lower perplexity scores indicate a higher resemblance to Wikipedia documents. Figure 5 displays the distributions of these scores. Our results demonstrate that the texts in OBELICS have a significantly lower average perplexity compared to the texts in c4 (Raffel et al., 2019), mmc4 (Zhu et al., 2023), and OSCAR (Ortiz Suárez et al., 2020). Furthermore, our distribution aligns closely with the one from The Pile (Gao et al., 2020), which was thoughtfully curated from diverse, high-quality sources.

## 4.2 Topic Modeling

Similar to Zhu et al. (2023), we employ a Latent Dirichlet Allocation (LDA) (Blei et al., 2003) to understand the diversity of the dataset. The LDA gives us insights into the distribution of topics in the dataset, along with estimated proportions and frequently associated words. Table 5 and 6 present the results of the LDA with respectively 20 and 200 topics, offering both a high-level and a more granular analysis of the dataset’s content. We observe that the dataset covers topics ranging from Politics to Health by way of Music. Additionally, we compute the most frequent domains and show that news sites are systematically the most represented (Table 4).

## 4.3 Qualitative Assessment of Dataset Samples

We manually inspect 250 documents from OBELICS to verify the dataset’s quality and assess the risks contained in the dataset. We focus on the images’ content in relation to the text since it’s the core addition compared to a language modeling dataset.

80% of documents have photo images, while 29% have graphic images (drawings, cartoons, etc.). 90% of the documents have all images clearly related to the text content. 30% of documents have images containing at least one written word, and 5% of documents have images that are structured text (slides, tables, scanned documents, etc.), which can help models learn OCR capabilities. 7% of documents have content (images or text) that hasn’t been captured by cleaning filters (non-English text, spam or advertisement, etc.). 46% of documents contain images with faces (portraits or group photos). No obvious Personally Identifiable Information (PII) texts were found, except for public personalities and people mentioned in news articles. No NSFW images were found. Only 3% of documents contain images with watermarks, and 2% have images with logos.

## 5 Validating the Viability of OBELICS

To confirm the viability of our dataset, we first show that vision and language models trained on our multimodal web documents outperform the same models trained on image-text pairs on various multimodal benchmarks. Following that, we demonstrate the effectiveness ofOBELICS as an alternative to closed datasets by training models of different sizes on par with closed-source models.

**Model details** We follow the Flamingo (Alayrac et al., 2022) architecture closely: we combine two frozen unimodal backbones - LLaMA (Touvron et al., 2023) for the language model, and OpenClip<sup>5</sup> for the vision encoder - add learnable cross-attention Transformer blocks to connect the language and vision blocks. For multimodal web documents, we feed the model sequences corresponding to the succession of text paragraphs and images. For image-text pairs, we form the training sequences by packing images with their captions. The images are encoded with the vision encoder and vision hidden states are pooled with Transformer Perceiver blocks and then fused into the text sequence through the cross-attention blocks. The training objective is the standard next token prediction. For more details, we refer to the original paper.

Following Alayrac et al. (2022), we evaluate our models on a series of multimodal benchmarks spanning visual question answering (VQAv2 (Antol et al., 2015), OKVQA (Marino et al., 2019), TextVQA (Singh et al., 2019), VizWiz (Gurari et al., 2018)), visual dialogs (VisDial (Das et al., 2017)), hateful speech detection (HatefulMeme (Kiela et al., 2020)), image captioning (COCO (Lin et al., 2014), Flickr30k (Young et al., 2014)), and OCR (IIIT5k (Mishra et al., 2012)).

Additional details about the architecture, the training, the compute and the evaluation are present in Appendix A.4.

Figure 6: Aggregated 4-shot performance through the training using **LAION** only, **OBELICS** only and a mixture of both. The training sequences from multimodal documents and the packed sequences obtained from image-text pairs have different numbers of images but the same number of tokens. Thus, we plot the performance over two log x-axes. The initial uptick of the model trained on image-text pairs is attributed to the fact the performance on VQA tasks starts by increasing and then slowly degrades.

**Training on different mixture of data** Figure 6 shows the result of the first experiment, which consists in training 9B-parameter models on different mixture of data. Training on multimodal web documents allows reaching the same performance using an order of magnitude fewer images than training on image-text pairs, even though the images from the two datasets come from Common Crawl. This underlines the benefit of having longer text contexts for training multimodal models. Moreover, the model trained on multimodal web documents performs better on average. This is particularly striking on visual question-answering benchmarks on which the model trained on image-text pairs slowly degrades through the training. We note, however, that the model trained on image-text pairs has a slight advantage performance-wise in captioning, classification, and OCR tasks (see more details in Appendix A.4.5). We hypothesize that this is due to the nature of image-text pairs: captions can be seen as fuzzy class labels. Last, similarly to Alayrac et al. (2022), we observe that combining the two types of datasets leads to increased performance for a given number of images, tokens, or training compute.

**Models trained on OBELICS achieve competitive performance at different scales** Following these insights, we show that OBELICS is a viable open alternative to other datasets.

<sup>5</sup><https://laion.ai/blog/large-openclip/><table border="1">
<thead>
<tr>
<th></th>
<th>Shot</th>
<th>COCO</th>
<th>Flickr30k</th>
<th>VQAv2</th>
<th>OKVQA</th>
<th>TextVQA</th>
<th>VizWiz</th>
<th>VisDial</th>
<th>HatefulMemes</th>
</tr>
</thead>
<tbody>
<tr>
<td>Flamingo-9B</td>
<td rowspan="3">0</td>
<td>79.4</td>
<td>61.5</td>
<td>51.8</td>
<td>44.7</td>
<td>31.8</td>
<td>22.8</td>
<td>48.0</td>
<td>57.0</td>
</tr>
<tr>
<td>OpenFlamingo-9B</td>
<td>79.5</td>
<td>59.5</td>
<td>52.7</td>
<td>37.8</td>
<td>24.2</td>
<td>27.5</td>
<td>-</td>
<td>51.6</td>
</tr>
<tr>
<td>IDEFICS-9B</td>
<td>46.0</td>
<td>27.3</td>
<td>50.9</td>
<td>38.4</td>
<td>25.9</td>
<td>35.5</td>
<td>48.7</td>
<td>51.8</td>
</tr>
<tr>
<td>Flamingo-9B</td>
<td rowspan="3">4</td>
<td>93.1</td>
<td>72.6</td>
<td>56.3</td>
<td>49.3</td>
<td><b>33.6</b></td>
<td>34.9</td>
<td>50.4</td>
<td>62.7</td>
</tr>
<tr>
<td>OpenFlamingo-9B</td>
<td>89.0</td>
<td>65.8</td>
<td>54.8</td>
<td>40.1</td>
<td>28.2</td>
<td>34.1</td>
<td>-</td>
<td>54.0</td>
</tr>
<tr>
<td>IDEFICS-9B</td>
<td>93.0</td>
<td>59.7</td>
<td>55.4</td>
<td>45.4</td>
<td>27.6</td>
<td>36.9</td>
<td>47.9</td>
<td>50.7</td>
</tr>
<tr>
<td>Flamingo-9B</td>
<td rowspan="3">8</td>
<td>99.0</td>
<td><b>73.4</b></td>
<td>58.0</td>
<td>50.0</td>
<td><b>33.6</b></td>
<td>39.4</td>
<td>51.2</td>
<td>63.9</td>
</tr>
<tr>
<td>OpenFlamingo-9B</td>
<td>96.3</td>
<td>62.9</td>
<td>54.8</td>
<td>41.1</td>
<td>29.1</td>
<td>38.5</td>
<td>-</td>
<td>54.7</td>
</tr>
<tr>
<td>IDEFICS-9B</td>
<td>97.0</td>
<td>61.9</td>
<td>56.4</td>
<td>47.7</td>
<td>27.5</td>
<td>40.4</td>
<td>47.6</td>
<td>51.1</td>
</tr>
<tr>
<td>Flamingo-9B</td>
<td rowspan="3">16</td>
<td>102.2</td>
<td>72.7</td>
<td>59.4</td>
<td>50.8</td>
<td>33.5</td>
<td>43.0</td>
<td><b>51.3</b></td>
<td><b>64.5</b></td>
</tr>
<tr>
<td>OpenFlamingo-9B</td>
<td>98.8</td>
<td>62.8</td>
<td>54.3</td>
<td>42.7</td>
<td>27.3</td>
<td>42.5</td>
<td>-</td>
<td>53.9</td>
</tr>
<tr>
<td>IDEFICS-9B</td>
<td>99.7</td>
<td>64.5</td>
<td>57.0</td>
<td>48.4</td>
<td>27.9</td>
<td>42.6</td>
<td>-</td>
<td>50.1</td>
</tr>
<tr>
<td>Flamingo-9B</td>
<td rowspan="3">32</td>
<td><b>106.3</b></td>
<td>72.8</td>
<td><b>60.4</b></td>
<td><b>51.0</b></td>
<td>32.6</td>
<td><b>44.0</b></td>
<td>50.4</td>
<td>63.5</td>
</tr>
<tr>
<td>OpenFlamingo-9B</td>
<td>99.5</td>
<td>61.3</td>
<td>53.3</td>
<td>42.4</td>
<td>23.8</td>
<td><b>44.0</b></td>
<td>-</td>
<td>53.8</td>
</tr>
<tr>
<td>IDEFICS-9B</td>
<td>98.0</td>
<td>64.3</td>
<td>57.9</td>
<td>49.6</td>
<td>28.3</td>
<td>43.7</td>
<td>-</td>
<td>49.8</td>
</tr>
<tr>
<td>Flamingo</td>
<td rowspan="2">0</td>
<td>84.3</td>
<td>67.2</td>
<td>56.3</td>
<td>50.6</td>
<td>35.0</td>
<td>31.6</td>
<td>52.0</td>
<td>46.4</td>
</tr>
<tr>
<td>IDEFICS</td>
<td>91.8</td>
<td>53.7</td>
<td>60.0</td>
<td>45.2</td>
<td>30.9</td>
<td>36.0</td>
<td>48.9</td>
<td>60.6</td>
</tr>
<tr>
<td>Flamingo</td>
<td rowspan="2">4</td>
<td>103.2</td>
<td>75.1</td>
<td>63.1</td>
<td>57.4</td>
<td>36.5</td>
<td>39.6</td>
<td>55.6</td>
<td>68.6</td>
</tr>
<tr>
<td>IDEFICS</td>
<td>110.3</td>
<td>73.7</td>
<td>63.6</td>
<td>52.4</td>
<td>34.4</td>
<td>40.4</td>
<td>48.4</td>
<td>57.8</td>
</tr>
<tr>
<td>Flamingo</td>
<td rowspan="2">8</td>
<td>108.8</td>
<td>78.2</td>
<td>65.6</td>
<td>57.5</td>
<td>37.3</td>
<td>44.8</td>
<td>56.4</td>
<td><b>70.0</b></td>
</tr>
<tr>
<td>IDEFICS</td>
<td>114.3</td>
<td>76.6</td>
<td>64.8</td>
<td>55.1</td>
<td>35.7</td>
<td>46.1</td>
<td>47.9</td>
<td>58.2</td>
</tr>
<tr>
<td>Flamingo</td>
<td rowspan="2">16</td>
<td>110.5</td>
<td>78.9</td>
<td>66.8</td>
<td><b>57.8</b></td>
<td>37.6</td>
<td>48.4</td>
<td><b>56.8</b></td>
<td><b>70.0</b></td>
</tr>
<tr>
<td>IDEFICS</td>
<td><b>116.6</b></td>
<td>80.1</td>
<td>65.4</td>
<td>56.8</td>
<td>36.3</td>
<td>48.3</td>
<td>-</td>
<td>57.8</td>
</tr>
<tr>
<td>Flamingo</td>
<td rowspan="2">32</td>
<td>113.8</td>
<td>75.4</td>
<td><b>67.6</b></td>
<td><b>57.8</b></td>
<td><b>37.9</b></td>
<td>49.8</td>
<td>55.6</td>
<td><b>70.0</b></td>
</tr>
<tr>
<td>IDEFICS</td>
<td><b>116.6</b></td>
<td><b>81.1</b></td>
<td>65.9</td>
<td><b>57.8</b></td>
<td>36.7</td>
<td><b>50.0</b></td>
<td>-</td>
<td>52.5</td>
</tr>
</tbody>
</table>

Table 2: Performance of IDEFICS against OpenFlamingo and Flamingo. The evaluations were done with random in-context examples, and in an open-ended setting for VQA tasks. (Task, Metric, Query split): (COCO, CIDEr, test), (Flickr30k, CIDEr, test (Karpathy)), (VQAv2, VQA acc., testdev), (OKVQA, VQA acc., val), (TextVQA, VQA acc., val), (VizWiz, VQA acc., testdev), (VisDial, NDCG, val), (HatefulMemes, ROC-AUC, test seen).We train **IDEFICS**, an 80 billion parameters Flamingo-like model on a mixture of image-text pairs from LAION (Schuhmann et al., 2022), openly accessible captioning datasets (Singh et al., 2022), **OBELICS** and multimodal web documents obtained from Wikipedia using a similar extraction strategy. We also train a smaller version of 9 billion parameters, **IDEFICS-9B**. We compare these models against OpenFlamingo v2 (Awadalla et al., 2023) and Flamingo of the same sizes and trained on a similar mixture of multimodal web documents and image-text pairs. We report the results in Table 2.

**IDEFICS** is often on par with Flamingo on various multimodal benchmarks. Out of the 8 evaluation tasks, with 32 in-context examples, it either performs better or obtain the same result as Flamingo on 4 of them. At the 9 billion parameter scale, we are still behind Flamingo-9B. However, it is important to highlight that we outperform OpenFlamingo-9B, which was trained on **mmc4**, in terms of aggregated performance. We achieved a score of 56.5, compared to their score of 55.8, by selecting the best performance across all numbers of in-context examples for each task. This highlights the advantages of **OBELICS** as an open alternative to a multimodal web document dataset.

## 6 Conclusion

With the goal of supporting open-source large multimodal models, we introduce **OBELICS**, an open web-scale collection of filtered interleaved multimodal web documents based on Common Crawl snapshots. We document a collection and filtering process that balances the scale and removal of undesirable texts and images while addressing some of the well-documented ethical concerns of large-scale multimodal datasets, notably data consent and pornographic content. To demonstrate the usefulness of models trained on multimodal documents, we train **IDEFICS** on **OBELICS** and show that it is a viable alternative to closed datasets. Open datasets of multimodal documents with scale, quality, and diversity of sources can help support the ability to train competitive open models.## Acknowledgments and Disclosure of Funding

The authors were granted access to the HPC resources of the Institut du développement et des ressources en informatique scientifique (IDRIS) du Centre national de la recherche scientifique (CNRS) under the allocation 2022-A0121013450 made by Grand équipement national de calcul intensif (GENCI). The initial development of the dataset was done on Jean-Zay cluster of IDRIS, and we thank the IDRIS team for their responsive support throughout the project, in particular Rémi Lacroix. We thank Guillaume Salou for setting up the virtual machines used to download the images of our dataset, and Sebastian Nagel for his valuable assistance in providing insights on Common Crawl. We thank Yacine Jernite and Daniel van Strien for conducting a bias analysis of the models trained on 0BELICS.

## References

Abbas, A., K. Tirumala, D. Simig, S. Ganguli, and A. S. Morcos (2023). Semdedup: Data-efficient learning at web-scale through semantic deduplication.

Aghajanyan, A., B. Huang, C. Ross, V. Karpukhin, H. Xu, N. Goyal, D. Okhonko, M. Joshi, G. Ghosh, M. Lewis, and L. Zettlemoyer (2022). Cm3: A causal masked multimodal model of the internet. *ArXiv abs/2201.07520*.

Alayrac, J.-B., J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, R. Ring, E. Rutherford, S. Cabi, T. Han, Z. Gong, S. Samangooei, M. Monteiro, J. L. Menick, S. Borgeaud, A. Brock, A. Nematzadeh, S. Sharifzadeh, M. a. Bińkowski, R. Barreira, O. Vinyals, A. Zisserman, and K. Simonyan (2022). Flamingo: a visual language model for few-shot learning. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.), *Advances in Neural Information Processing Systems*, Volume 35, pp. 23716–23736. Curran Associates, Inc.

Antol, S., A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, and D. Parikh (2015). VQA: Visual Question Answering. In *International Conference on Computer Vision (ICCV)*.

Awadalla, A., I. Gao, J. Gardner, J. Hessel, Y. Hanafy, W. Zhu, K. Marathe, Y. Bitton, S. Gadre, S. Sagawa, J. Jitsev, S. Kornblith, P. W. Koh, G. Ilharco, M. Wortsman, and L. Schmidt (2023). Openflamingo: An open-source framework for training large autoregressive vision-language models.

Bai, Y., A. Jones, K. Ndousse, A. Askell, A. Chen, N. DasSarma, D. Drain, S. Fort, D. Ganguli, T. Henighan, N. Joseph, S. Kadavath, J. Kernion, T. Conerly, S. El-Showk, N. Elhage, Z. Hatfield-Dodds, D. Hernandez, T. Hume, S. Johnston, S. Kravec, L. Lovitt, N. Nanda, C. Olsson, D. Amodei, T. Brown, J. Clark, S. McCandlish, C. Olah, B. Mann, and J. Kaplan (2022). Training a helpful and harmless assistant with reinforcement learning from human feedback.

Beaumont, R. (2021). img2dataset: Easily turn large sets of image urls to an image dataset. <https://github.com/rom1504/img2dataset>.

Bender, E. M., T. Gebru, A. McMillan-Major, and S. Shmitchell (2021). On the dangers of stochastic parrots: Can language models be too big? *Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency*.

Biderman, S. and W. J. Scheirer (2020, 12 Dec). Pitfalls in machine learning research: Reexamining the development cycle. In J. Zosa Forde, F. Ruiz, M. F. Pradier, and A. Schein (Eds.), *Proceedings on "I Can't Believe It's Not Better!" at NeurIPS Workshops*, Volume 137 of *Proceedings of Machine Learning Research*, pp. 106–117. PMLR.

Biderman, S., H. Schoelkopf, Q. Anthony, H. Bradley, K. O'Brien, E. Hallahan, M. A. Khan, S. Purohit, U. S. Prashanth, E. Raff, A. Skowron, L. Sutawika, and O. van der Wal (2023). Pythia: A suite for analyzing large language models across training and scaling.Birhane, A., V. U. Prabhu, and E. Kahembwe (2021). Multimodal datasets: misogyny, pornography, and malignant stereotypes. *ArXiv abs/2110.01963*.

Blei, D. M., A. Y. Ng, and M. I. Jordan (2003, mar). Latent dirichlet allocation. *J. Mach. Learn. Res.* 3(null), 993–1022.

Broder, A. (1997). On the resemblance and containment of documents. In *Proceedings. Compression and Complexity of SEQUENCES 1997 (Cat. No.97TB100171)*, pp. 21–29.

Brown, T., B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei (2020). Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin (Eds.), *Advances in Neural Information Processing Systems*, Volume 33, pp. 1877–1901. Curran Associates, Inc.

Byeon, M., B. Park, H. Kim, S. Lee, W. Baek, and S. Kim (2022). Coyo-700m: Image-text pair dataset. <https://github.com/kakaobrain/coyo-dataset>.

Caswell, I., T. Breiner, D. van Esch, and A. Bapna (2020). Language id in the wild: Unexpected challenges on the path to a thousand-language web text corpus. *ArXiv abs/2010.14571*.

Changpinyo, S., P. Sharma, N. Ding, and R. Soricut (2021). Conceptual 12M: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In *CVPR*.

Chowdhery, A., S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann, P. Schuh, K. Shi, S. Tsvyashchenko, J. Maynez, A. Rao, P. Barnes, Y. Tay, N. Shazeer, V. Prabhakaran, E. Reif, N. Du, B. Hutchinson, R. Pope, J. Bradbury, J. Austin, M. Isard, G. Gur-Ari, P. Yin, T. Duke, A. Levskaya, S. Ghemawat, S. Dev, H. Michalewski, X. Garcia, V. Misra, K. Robinson, L. Fedus, D. Zhou, D. Ippolito, D. Luan, H. Lim, B. Zoph, A. Spiridonov, R. Sepassi, D. Dohan, S. Agrawal, M. Omernick, A. M. Dai, T. S. Pillai, M. Pellat, A. Lewkowycz, E. Moreira, R. Child, O. Polozov, K. Lee, Z. Zhou, X. Wang, B. Saeta, M. Diaz, O. Firat, M. Catasta, J. Wei, K. Meier-Hellstern, D. Eck, J. Dean, S. Petrov, and N. Fiedel (2022). Palm: Scaling language modeling with pathways.

Das, A., S. Kottur, K. Gupta, A. Singh, D. Yadav, J. M. F. Moura, D. Parikh, and D. Batra (2017, July). Visual dialog. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

Dehghani, M., J. Djolonga, B. Mustafa, P. Padlewski, J. Heek, J. Gilmer, A. Steiner, M. Caron, R. Geirhos, I. Alabdulmohtsin, R. Jenatton, L. Beyer, M. Tschannen, A. Arnab, X. Wang, C. Riquelme, M. Minderer, J. Puigcerver, U. Evci, M. Kumar, S. van Steenkiste, G. F. Elsayed, A. Mahendran, F. Yu, A. Oliver, F. Huot, J. Bastings, M. P. Collier, A. Gritsenko, V. Birodkar, C. Vasconcelos, Y. Tay, T. Mensink, A. Kolesnikov, F. Pavetić, D. Tran, T. Kipf, M. Lučić, X. Zhai, D. Keysers, J. Harmsen, and N. Houlsby (2023). Scaling vision transformers to 22 billion parameters.

Deng, X., P. Shiralkar, C. Lockard, B. Huang, and H. Sun (2022). Dom-lm: Learning generalizable representations for html documents. *ArXiv abs/2201.10608*.

Desai, K., G. Kaul, Z. Aysola, and J. Johnson (2021). Redcaps: Web-curated image-text data created by the people, for the people. In J. Vanschoren and S. Yeung (Eds.), *Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks*, Volume 1. Curran.

Dodge, J., A. Marasović, G. Ilharco, D. Groeneveld, M. Mitchell, and M. Gardner (2021). Documenting large webtext corpora: A case study on the colossal clean crawled corpus. In *Conference on Empirical Methods in Natural Language Processing*.Gadre, S. Y., G. Ilharco, A. Fang, J. Hayase, G. Smyrnis, T. Nguyen, R. Marten, M. Wortsman, D. Ghosh, J. Zhang, E. Orgad, R. Entezari, G. Daras, S. Pratt, V. Ramanujan, Y. Bitton, K. Marathe, S. Mussmann, R. Vencu, M. Cherti, R. Krishna, P. W. Koh, O. Saukh, A. Ratner, S. Song, H. Hajishirzi, A. Farhadi, R. Beaumont, S. Oh, A. Dimakis, J. Jitsev, Y. Carmon, V. Shankar, and L. Schmidt (2023). Datacomp: In search of the next generation of multimodal datasets. *arXiv preprint arXiv:2304.14108*.

Gao, L., S. Biderman, S. Black, L. Golding, T. Hoppe, C. Foster, J. Phang, H. He, A. Thite, N. Nabeshima, S. Presser, and C. Leahy (2020). The Pile: An 800gb dataset of diverse text for language modeling. *arXiv preprint arXiv:2101.00027*.

Gokaslan, A. and V. Cohen (2019). Openwebtext corpus. <http://Skylion007.github.io/OpenWebTextCorpus>.

Gu, J., X. Meng, G. Lu, L. Hou, N. Minzhe, X. Liang, L. Yao, R. Huang, W. Zhang, X. Jiang, C. Xu, and H. Xu (2022). Wukong: A 100 million large-scale chinese cross-modal pre-training benchmark. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.), *Advances in Neural Information Processing Systems*, Volume 35, pp. 26418–26431. Curran Associates, Inc.

Gurari, D., Q. Li, A. J. Stangl, A. Guo, C. Lin, K. Grauman, J. Luo, and J. P. Bigham (2018). Vizwiz grand challenge: Answering visual questions from blind people.

Heafield, K. (2011, July). KenLM: Faster and smaller language model queries. In *Proceedings of the Sixth Workshop on Statistical Machine Translation*, Edinburgh, Scotland, pp. 187–197. Association for Computational Linguistics.

Hoffmann, J., S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas, L. A. Hendricks, J. Welbl, A. Clark, T. Hennigan, E. Noland, K. Millican, G. van den Driessche, B. Damoc, A. Guy, S. Osindero, K. Simonyan, E. Elsen, J. W. Rae, O. Vinyals, and L. Sifre (2022). Training compute-optimal large language models.

Huang, S., L. Dong, W. Wang, Y. Hao, S. Singhal, S. Ma, T. Lv, L. Cui, O. K. Mohammed, B. Patra, Q. Liu, K. Aggarwal, Z. Chi, J. Bjorck, V. Chaudhary, S. Som, X. Song, and F. Wei (2023). Language is not all you need: Aligning perception with language models.

Jaegle, A., F. Gimeno, A. Brock, A. Zisserman, O. Vinyals, and J. Carreira (2021). Perceiver: General perception with iterative attention.

Jia, C., Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. V. Le, Y.-H. Sung, Z. Li, and T. Duerig (2021). Scaling up visual and vision-language representation learning with noisy text supervision. In *International Conference on Machine Learning*.

Jiang, A. Q., S. Welleck, J. P. Zhou, T. Lacroix, J. Liu, W. Li, M. Jannik, G. Lample, and Y. Wu (2023). Draft, sketch, and prove: Guiding formal theorem provers with informal proofs. In *The Eleventh International Conference on Learning Representations*.

Joulin, A., E. Grave, P. Bojanowski, and T. Mikolov (2017, April). Bag of tricks for efficient text classification. In *Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers*, Valencia, Spain, pp. 427–431. Association for Computational Linguistics.

Kaplan, J., S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei (2020). Scaling laws for neural language models.

Kärkkäinen, K. and J. Joo (2021). Fairface: Face attribute dataset for balanced race, gender, and age for bias measurement and mitigation. *2021 IEEE Winter Conference on Applications of Computer Vision (WACV)*, 1547–1557.

Kiela, D., H. Firooz, A. Mohan, V. Goswami, A. Singh, P. Ringshia, and D. Testuggine (2020). The hateful memes challenge: Detecting hate speech in multimodal memes. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin (Eds.), *Advances in Neural Information Processing Systems*, Volume 33, pp. 2611–2624. Curran Associates, Inc.Koh, J. Y., R. Salakhutdinov, and D. Fried (2023). Grounding language models to images for multimodal generation.

Laborde, G. Deep nn for nsfw detection.

Laurençon, H., L. Saulnier, T. Wang, C. Akiki, A. Villanova del Moral, T. Le Scao, L. Von Werra, C. Mou, E. González Ponferrada, H. Nguyen, J. Frohberg, M. Šaško, Q. Lhoest, A. McMillan-Major, G. Dupont, S. Biderman, A. Rogers, L. Ben allal, F. De Toni, G. Pistilli, O. Nguyen, S. Nikpoor, M. Masoud, P. Colombo, J. de la Rosa, P. Villegas, T. Thrush, S. Longpre, S. Nagel, L. Weber, M. Muñoz, J. Zhu, D. Van Strien, Z. Alyafei, K. Almubarak, M. C. Vu, I. Gonzalez-Dios, A. Soroa, K. Lo, M. Dey, P. Ortiz Suarez, A. Gokaslan, S. Bose, D. Adelani, L. Phan, H. Tran, I. Yu, S. Pai, J. Chim, V. Lepercq, S. Ilic, M. Mitchell, S. A. Luccioni, and Y. Jernite (2022). The bigscience roots corpus: A 1.6tb composite multilingual dataset. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.), *Advances in Neural Information Processing Systems*, Volume 35, pp. 31809–31826. Curran Associates, Inc.

Lee, K., D. Ippolito, A. Nystrom, C. Zhang, D. Eck, C. Callison-Burch, and N. Carlini (2022). Deduplicating training data makes language models better. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics.

Li, J., D. Li, S. Savarese, and S. Hoi (2023). Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models.

Li, J., D. Li, C. Xiong, and S. Hoi (2022). Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In *ICML*.

Li, R., L. B. Allal, Y. Zi, N. Muennighoff, D. Kocetkov, C. Mou, M. Marone, C. Akiki, J. Li, J. Chim, Q. Liu, E. Zheltonozhskii, T. Y. Zhuo, T. Wang, O. Dehaene, M. Davaadorj, J. Lamy-Poirier, J. Monteiro, O. Shliazhko, N. Gontier, N. Meade, A. Zebaze, M.-H. Yee, L. K. Umapathi, J. Zhu, B. Lipkin, M. Oblokulov, Z. Wang, R. Murthy, J. Stillerman, S. S. Patel, D. Abulkhanov, M. Zocca, M. Dey, Z. Zhang, N. Fahmy, U. Bhattacharyya, W. Yu, S. Singh, S. Luccioni, P. Villegas, M. Kunakov, F. Zhdanov, M. Romero, T. Lee, N. Timor, J. Ding, C. Schlesinger, H. Schoelkopf, J. Ebert, T. Dao, M. Mishra, A. Gu, J. Robinson, C. J. Anderson, B. Dolan-Gavitt, D. Contractor, S. Reddy, D. Fried, D. Bahdanau, Y. Jernite, C. M. Ferrandis, S. Hughes, T. Wolf, A. Guha, L. von Werra, and H. de Vries (2023). Starcoder: may the source be with you!

Lin, T.-Y., M. Maire, S. Belongie, L. Bourdev, R. Girshick, J. Hays, P. Perona, D. Ramanan, C. L. Zitnick, and P. Dollár (2014). Microsoft coco: Common objects in context. cite arxiv:1405.0312Comment: 1) updated annotation pipeline description and figures; 2) added new section describing datasets splits; 3) updated author list.

Liu, S., L. Fan, E. Johns, Z. Yu, C. Xiao, and A. Anandkumar (2023). Prismer: A vision-language model with an ensemble of experts. *arXiv preprint arXiv:2303.02506*.

Liu, Y., G. Zhu, B. Zhu, Q. Song, G. Ge, H. Chen, G. Qiao, R. Peng, L. Wu, and J. Wang (2022). Taisu: A 166m large-scale high-quality dataset for chinese vision-language pre-training. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.), *Advances in Neural Information Processing Systems*, Volume 35, pp. 16705–16717. Curran Associates, Inc.

Loshchilov, I. and F. Hutter (2017). Fixing weight decay regularization in adam. *CoRR abs/1711.05101*.

Luccioni, A. S., C. Akiki, M. Mitchell, and Y. Jernite (2023). Stable bias: Analyzing societal representations in diffusion models.

Marino, K., M. Rastegari, A. Farhadi, and R. Mottaghi (2019). Ok-vqa: A visual question answering benchmark requiring external knowledge. In *Conference on Computer Vision and Pattern Recognition (CVPR)*.Mishra, A., K. Alahari, and C. V. Jawahar (2012). Scene text recognition using higher order language priors. In *BMVC*.

Nichol, A., P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. McGrew, I. Sutskever, and M. Chen (2022). Glide: Towards photorealistic image generation and editing with text-guided diffusion models.

Ortiz Suárez, P. J., L. Romary, and B. Sagot (2020, July). A monolingual approach to contextualized word embeddings for mid-resource languages. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, Online, pp. 1703–1714. Association for Computational Linguistics.

Ouyang, L., J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. F. Christiano, J. Leike, and R. Lowe (2022). Training language models to follow instructions with human feedback. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.), *Advances in Neural Information Processing Systems*, Volume 35, pp. 27730–27744. Curran Associates, Inc.

Piktus, A., C. Akiki, P. Villegas, H. Laurençon, G. Dupont, A. S. Luccioni, Y. Jernite, and A. Rogers (2023). The roots search tool: Data transparency for llms.

Radenovic, F., A. Dubey, A. Kadian, T. Mihaylov, S. Vandenhende, Y. Patel, Y. Wen, V. Ramanathan, and D. Mahajan (2023). Filtering, distillation, and hard negatives for vision-language pre-training.

Radford, A., J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever (2021). Learning transferable visual models from natural language supervision. In *International Conference on Machine Learning*.

Rae, J. W., S. Borgeaud, T. Cai, K. Millican, J. Hoffmann, F. Song, J. Aslanides, S. Henderson, R. Ring, S. Young, E. Rutherford, T. Hennigan, J. Menick, A. Cassirer, R. Powell, G. van den Driessche, L. A. Hendricks, M. Rauh, P.-S. Huang, A. Glaese, J. Welbl, S. Dathathri, S. Huang, J. Uesato, J. Mellor, I. Higgins, A. Creswell, N. McAleese, A. Wu, E. Elsen, S. Jayakumar, E. Buchatskaya, D. Budden, E. Sutherland, K. Simonyan, M. Paganini, L. Sifre, L. Martens, X. L. Li, A. Kuncoro, A. Nematzadeh, E. Gribovskaya, D. Donato, A. Lazaridou, A. Mensch, J.-B. Lespiau, M. Tsimpoukelli, N. Grigorev, D. Fritz, T. Sottiaux, M. Pajarskas, T. Pohlen, Z. Gong, D. Toyama, C. de Masson d’Autume, Y. Li, T. Terzi, V. Mikulik, I. Babuschkin, A. Clark, D. de Las Casas, A. Guy, C. Jones, J. Bradbury, M. Johnson, B. Hechtman, L. Weidinger, I. Gabriel, W. Isaac, E. Lockhart, S. Osindero, L. Rimell, C. Dyer, O. Vinyals, K. Ayoub, J. Stanway, L. Bennett, D. Hassabis, K. Kavukcuoglu, and G. Irving (2022). Scaling language models: Methods, analysis & insights from training gopher.

Raffel, C., N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu (2019). Exploring the limits of transfer learning with a unified text-to-text transformer. *arXiv e-prints*.

Ramesh, A., P. Dhariwal, A. Nichol, C. Chu, and M. Chen (2022). Hierarchical text-conditional image generation with clip latents.

Rombach, R., A. Blattmann, D. Lorenz, P. Esser, and B. Ommer (2021). High-resolution image synthesis with latent diffusion models.

Saharia, C., W. Chan, S. Saxena, L. Li, J. Whang, E. Denton, S. K. S. Ghasemipour, B. K. Ayan, S. S. Mahdavi, R. G. Lopes, T. Salimans, J. Ho, D. J. Fleet, and M. Norouzi (2022). Photorealistic text-to-image diffusion models with deep language understanding.

Schuhmann, C., R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman, P. Schramowski, S. Kundurthy, K. Crowson, L. Schmidt, R. Kaczmarczyk, and J. Jitsev (2022). Laion-5b: An open large-scale dataset for training next generation image-text models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave,K. Cho, and A. Oh (Eds.), *Advances in Neural Information Processing Systems*, Volume 35, pp. 25278–25294. Curran Associates, Inc.

Schuhmann, C., R. Vencu, R. Beaumont, R. Kaczmarczyk, C. Mullis, A. Katta, T. Coombes, J. Jitsev, and A. Komatsuzaki (2021). Laion-400m: Open dataset of clip-filtered 400 million image-text pairs.

Sharma, P., N. Ding, S. Goodman, and R. Soricut (2018). Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In *Proceedings of ACL*.

Singh, A., R. Hu, V. Goswami, G. Couairon, W. Galuba, M. Rohrbach, and D. Kiela (2022). FLAVA: A foundational language and vision alignment model. In *CVPR*.

Singh, A., V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Parikh, and M. Rohrbach (2019). Towards vqa models that can read. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp. 8317–8326.

Sorscher, B., R. Geirhos, S. Shekhar, S. Ganguli, and A. Morcos (2022). Beyond neural scaling laws: beating power law scaling via data pruning. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.), *Advances in Neural Information Processing Systems*, Volume 35, pp. 19523–19536. Curran Associates, Inc.

Srinivasan, K., K. Raman, J. Chen, M. Bendersky, and M. Najork (2021). Wit: Wikipedia-based image text dataset for multimodal multilingual machine learning. In *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval*, SIGIR '21, New York, NY, USA, pp. 2443–2449. Association for Computing Machinery.

Team, M. N. (2023). Introducing mpt-7b: A new standard for open-source, commercially usable llms.

Touvron, H., T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample (2023). Llama: Open and efficient foundation language models.

Wang, P., A. Yang, R. Men, J. Lin, S. Bai, Z. Li, J. Ma, C. Zhou, J. Zhou, and H. Yang (2022, 17–23 Jul). OFA: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato (Eds.), *Proceedings of the 39th International Conference on Machine Learning*, Volume 162 of *Proceedings of Machine Learning Research*, pp. 23318–23340. PMLR.

Wang, Q., Y. Fang, A. Ravula, F. Feng, X. Quan, and D. Liu (2022). Webformer: The web-page transformer for structure information extraction.

Wang, W., H. Bao, L. Dong, J. Bjorck, Z. Peng, Q. Liu, K. Aggarwal, O. K. Mohammed, S. Singhal, S. Som, and F. Wei (2022). Image as a foreign language: Beit pretraining for all vision and vision-language tasks.

Webster, R., J. Rabin, L. Simon, and F. Jurie (2023). On the de-duplication of laion-2b.

Workshop, B., :, T. L. Scao, A. Fan, C. Akiki, E. Pavlick, S. Ilić, D. Hesslow, R. Castagné, A. S. Luccioni, F. Yvon, M. Gallé, J. Tow, A. M. Rush, S. Biderman, A. Webson, P. S. Ammanamanchi, T. Wang, B. Sagot, N. Muennighoff, A. V. del Moral, O. Ruwase, R. Bawden, S. Bekman, A. McMillan-Major, I. Beltagy, H. Nguyen, L. Saulnier, S. Tan, P. O. Suarez, V. Sanh, H. Laurençon, Y. Jernite, J. Launay, M. Mitchell, C. Raffel, A. Gokaslan, A. Simhi, A. Soroa, A. F. Aji, A. Alfassy, A. Rogers, A. K. Nitzav, C. Xu, C. Mou, C. Emezue, C. Klam, C. Leong, D. van Strien, D. I. Adelani, D. Radev, E. G. Ponferrada, E. Levkovizh, E. Kim, E. B. Natan, F. D. Toni, G. Dupont, G. Kruszewski, G. Pistilli, H. Elsahar, H. Benyamina, H. Tran, I. Yu, I. Abdulmumin, I. Johnson, I. Gonzalez-Dios, J. de la Rosa, J. Chim, J. Dodge, J. Zhu, J. Chang, J. Frohberg, J. Tobing, J. Bhattacharjee, K. Almubarak, K. Chen, K. Lo, L. V. Werra, L. Weber, L. Phan, L. B. allal, L. Tanguy,M. Dey, M. R. Muñoz, M. Masoud, M. Grandury, M. Šaško, M. Huang, M. Coavoux, M. Singh, M. T.-J. Jiang, M. C. Vu, M. A. Jauhar, M. Ghaleb, N. Subramani, N. Kassner, N. Khamis, O. Nguyen, O. Espejel, O. de Gibert, P. Villegas, P. Henderson, P. Colombo, P. Amuok, Q. Lhoest, R. Harliman, R. Bommasani, R. L. López, R. Ribeiro, S. Osei, S. Pyysalo, S. Nagel, S. Bose, S. H. Muhammad, S. Sharma, S. Longpre, S. Nikpoor, S. Silberberg, S. Pai, S. Zink, T. T. Torrent, T. Schick, T. Thrush, V. Danchev, V. Nikoulina, V. Laippala, V. Lepercq, V. Prabhu, Z. Alyafei, Z. Talat, A. Raja, B. Heinzerling, C. Si, D. E. Taşar, E. Salesky, S. J. Mielke, W. Y. Lee, A. Sharma, A. Santilli, A. Chaffin, A. Stiegler, D. Datta, E. Szczechla, G. Chhablani, H. Wang, H. Pandey, H. Strobelt, J. A. Fries, J. Rozen, L. Gao, L. Sutawika, M. S. Bari, M. S. Al-shaibani, M. Manica, N. Nayak, R. Teehan, S. Albanie, S. Shen, S. Ben-David, S. H. Bach, T. Kim, T. Bers, T. Fevry, T. Neeraj, U. Thakker, V. Raunak, X. Tang, Z.-X. Yong, Z. Sun, S. Brody, Y. Uri, H. Tojarieh, A. Roberts, H. W. Chung, J. Tae, J. Phang, O. Press, C. Li, D. Narayanan, H. Bourfoune, J. Casper, J. Rasley, M. Ryabinin, M. Mishra, M. Zhang, M. Shoeybi, M. Peyrounette, N. Patry, N. Tazi, O. Sanseviero, P. von Platen, P. Cornette, P. F. Lavallée, R. Lacroix, S. Rajbhandari, S. Gandhi, S. Smith, S. Requena, S. Patil, T. Dettmers, A. Baruya, A. Singh, A. Cheveleva, A.-L. Ligozat, A. Subramonian, A. Névél, C. Lovering, D. Garrette, D. Tunuguntla, E. Reiter, E. Taktasheva, E. Voloshina, E. Bogdanov, G. I. Winata, H. Schoelkopf, J.-C. Kalo, J. Novikova, J. Z. Forde, J. Clive, J. Kasai, K. Kawamura, L. Hazan, M. Carpuat, M. Clinciu, N. Kim, N. Cheng, O. Serikov, O. Antverg, O. van der Wal, R. Zhang, R. Zhang, S. Gehrman, S. Mirkin, S. Pais, T. Shavrina, T. Scialom, T. Yun, T. Limisiewicz, V. Rieser, V. Protasov, V. Mikhailov, Y. Pruksachatkun, Y. Belinkov, Z. Bamberger, Z. Kasner, A. Rueda, A. Pestana, A. Feizpour, A. Khan, A. Faranak, A. Santos, A. Hevia, A. Unldreaj, A. Aghagol, A. Abdollahi, A. Tammour, A. HajiHosseini, B. Behroozi, B. Ajibade, B. Saxena, C. M. Ferrandis, D. Contractor, D. Lansky, D. David, D. Kiela, D. A. Nguyen, E. Tan, E. Baylor, E. Ozoani, F. Mirza, F. Ononiwu, H. Rezanejad, H. Jones, I. Bhattacharya, I. Solaiman, I. Sedenko, I. Nejadgholi, J. Passmore, J. Seltzer, J. B. Sanz, L. Dutra, M. Samagaio, M. Elbadri, M. Mieskes, M. Gerchick, M. Akinlolu, M. McKenna, M. Qiu, M. Ghauri, M. Burynok, N. Abrar, N. Rajani, N. Elkott, N. Fahmy, O. Samuel, R. An, R. Kromann, R. Hao, S. Alizadeh, S. Shubber, S. Wang, S. Roy, S. Viguier, T. Le, T. Oyebade, T. Le, Y. Yang, Z. Nguyen, A. R. Kashyap, A. Palasciano, A. Callahan, A. Shukla, A. Miranda-Escalada, A. Singh, B. Beilharz, B. Wang, C. Brito, C. Zhou, C. Jain, C. Xu, C. Fourrier, D. L. Periñán, D. Molano, D. Yu, E. Manjavacas, F. Barth, F. Fuhrmann, G. Altay, G. Bayrak, G. Burns, H. U. Vrabec, I. Bello, I. Dash, J. Kang, J. Giorgi, J. Golde, J. D. Posada, K. R. Sivaraman, L. Bulchandani, L. Liu, L. Shinzato, M. H. de Bykhovetz, M. Takeuchi, M. Pàmies, M. A. Castillo, M. Nezhurina, M. Sänger, M. Samwald, M. Cullan, M. Weinberg, M. D. Wolf, M. Mihaljcic, M. Liu, M. Freidank, M. Kang, N. Seelam, N. Dahlberg, N. M. Broad, N. Muellner, P. Fung, P. Haller, R. Chandrasekhar, R. Eisenberg, R. Martin, R. Canalli, R. Su, R. Su, S. Cahyawijaya, S. Garda, S. S. Deshmukh, S. Mishra, S. Kiblawi, S. Ott, S. Sang-aroonsiri, S. Kumar, S. Schweter, S. Bharati, T. Laud, T. Gigant, T. Kainuma, W. Kusa, Y. Labrak, Y. S. Bajaj, Y. Venkatraman, Y. Xu, Y. Xu, Y. Xu, Z. Tan, Z. Xie, Z. Ye, M. Bras, Y. Belkada, and T. Wolf (2023). Bloom: A 176b-parameter open-access multilingual language model.

Xie, S. M., H. Pham, X. Dong, N. Du, H. Liu, Y. Lu, P. Liang, Q. V. Le, T. Ma, and A. W. Yu (2023). Dorem: Optimizing data mixtures speeds up language model pretraining.

Yang, Z., Z. Gan, J. Wang, X. Hu, Y. Lu, Z. Liu, and L. Wang (2022). An empirical study of gpt-3 for few-shot knowledge-based vqa. In *Proceedings of the AAAI Conference on Artificial Intelligence*, Volume 36, pp. 3081–3089.

Young, P., A. Lai, M. Hodosh, and J. Hockenmaier (2014). From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. *Transactions of the Association for Computational Linguistics* 2, 67–78.

Yu, J., Z. Wang, V. Vasudevan, L. Yeung, M. Seyedhosseini, and Y. Wu (2022). Coca: Contrastive captioners are image-text foundation models. *Transactions on Machine Learning Research*.Yuan, S., S. Zhao, J. Leng, Z. Xue, H. Zhao, P. Liu, Z. Gong, W. X. Zhao, J. Li, and J. Tang (2022). Wudaomm: A large-scale multi-modal dataset for pre-training models.

Yuksekgonul, M., F. Bianchi, P. Kalluri, D. Jurafsky, and J. Zou (2023). When and why vision-language models behave like bags-of-words, and what to do about it? In *International Conference on Learning Representations*.

Zhang, B. and R. Sennrich (2019). Root Mean Square Layer Normalization. In *Advances in Neural Information Processing Systems 32*, Vancouver, Canada.

Zhang, J., Y. Zhao, M. Saleh, and P. J. Liu (2019). Pegasus: Pre-training with extracted gap-sentences for abstractive summarization.

Zhang, R., J. Han, A. Zhou, X. Hu, S. Yan, P. Lu, H. Li, P. Gao, and Y. Qiao (2023). Llama-adapter: Efficient fine-tuning of language models with zero-init attention.

Zhang, S., S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab, X. Li, X. V. Lin, T. Mihaylov, M. Ott, S. Shleifer, K. Shuster, D. Simig, P. S. Koura, A. Sridhar, T. Wang, and L. Zettlemoyer (2022). Opt: Open pre-trained transformer language models.

Zhou, Y., Y. Sheng, N. H. Vo, N. Edmonds, and S. Tata (2021). Simplified dom trees for transferable attribute extraction from the web. *ArXiv abs/2101.02415*.

Zhu, W., J. Hessel, A. Awadalla, S. Y. Gadre, J. Dodge, A. Fang, Y. Yu, L. Schmidt, W. Y. Wang, and Y. Choi (2023). Multimodal C4: An open, billion-scale corpus of images interleaved with text. *arXiv preprint arXiv:2304.06939*.## Checklist

1. 1. For all authors...
   1. (a) Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? [\[Yes\]](#)
   2. (b) Did you describe the limitations of your work? [\[Yes\]](#) See Section 4.
   3. (c) Did you discuss any potential negative societal impacts of your work? [\[Yes\]](#)  
      We think that the release of such a dataset strikes a constructive trade-off between the risks associated with datasets built on top of crawled web pages (for instance, the presence of images with faces, the potential of PII in texts, offensive, insulting or threatening, etc.) with the future works that a dataset of such scale, quality and thoughtful filtering can enable. We further discuss these points in A.3.
   4. (d) Have you read the ethics review guidelines and ensured that your paper conforms to them? [\[Yes\]](#) We read the ethics review guidelines and tried our best to match the expectations. Our content is extracted from publicly available websites at the time of the web crawl. Given the size of our dataset, it would be prohibitive to get the explicit consent of the authors of these websites. Instead, we respect the choice of content creators by removing opted-out images. Such a strategy cannot be exhaustive and we remain available for content creators to opt-out of the dataset.
2. 2. If you are including theoretical results...
   1. (a) Did you state the full set of assumptions of all theoretical results? [\[N/A\]](#)
   2. (b) Did you include complete proofs of all theoretical results? [\[N/A\]](#)
3. 3. If you ran experiments (e.g. for benchmarks)...
   1. (a) Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? [\[Yes\]](#)  
      We will release the code used for the creation of the model and its training, along with the model itself.
   2. (b) Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? [\[Yes\]](#) See Appendix A.4.
   3. (c) Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? [\[N/A\]](#)
   4. (d) Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? [\[Yes\]](#) See Appendix A.4.
4. 4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...
   1. (a) If your work uses existing assets, did you cite the creators? [\[Yes\]](#) We mentioned the libraries we used.
   2. (b) Did you mention the license of the assets? [\[Yes\]](#) We only used open-source libraries.
   3. (c) Did you include any new assets either in the supplemental material or as a URL? [\[N/A\]](#)
   4. (d) Did you discuss whether and how consent was obtained from people whose data you’re using/curating? [\[Yes\]](#) See the ethics review guidelines part.
   5. (e) Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? [\[Yes\]](#) The dataset we are releasing is built from publicly accessible websites. As such, there is no content in our dataset that hasn’t been publicly visible on the web at some point. Similarly, the dataset might contain texts or images that can be considered offensive, insulting, or threatening, as such data is prevalent on the web. We took measures to remove pornographic content and low-quality texts as much as possible. We did not take additional intentional measures to remove personal information. A manual inspection of 250 random samples reveals that there isn’t obviouspersonally identifiable information (excluding celebrities and people mentioned in news articles), although it is likely that the dataset contains some.

1. 5. If you used crowdsourcing or conducted research with human subjects...
   1. (a) Did you include the full text of instructions given to participants and screenshots, if applicable? [N/A]
   2. (b) Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? [N/A]
   3. (c) Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [N/A]## A Appendix

### A.1 Creation of the Multimodal Web Document Dataset

#### A.1.1 Collecting of a Large Number of HTML Files

Our data collection process begins by considering the 25 most recent Common Crawl<sup>6</sup> dumps available at the time of dataset creation. It contains webpages spanning from February 2020 to January/February 2023. We use a modified version of **readability-lxml**<sup>7</sup> to extract the main text from the pages, discarding any pages that contain text of excessively high perplexity. This process yields a total of 41.2 billion documents.

**Selection of English content** To identify non-English content, we apply the FastText classifier (Joulin et al., 2017) to the extracted text, effectively filtering out 63.6% of the documents.

**Early text deduplication** Often, a set of URLs is crawled repeatedly across different Common Crawl snapshots. However, the content of these websites may vary as web administrators make changes over time. Hence, at this stage, we refrain from deduplicating documents based on their URLs. Instead, we perform MinHash (Broder, 1997) deduplication with 16 hashes calculated over 5-grams. To further refine the data, we eliminate documents containing substantial proportions of repeated paragraphs and n-grams, employing the methodology described in MassiveText (Rae et al., 2022). (Lee et al., 2022; Abbas et al., 2023) show that crawled data often contains a significant amount of duplication, and training on deduplicated data can improve performance.

**Quality classification** We employ a logistic regression classifier with hashed token frequencies to only retain pages containing human-written text, similar to Brown et al. (2020). The classifier is trained using documents from curated datasets, such as Wikipedia and OpenWebText (Gokaslan and Cohen, 2019), as positive examples, and documents sampled from Common Crawl as negative examples. For simplicity, we use a threshold of 0.5 for the probability that a document comes from a curated corpus, which acts as an indicator that a document is human-written.

Following these steps, we obtain 1.1 billion documents and their HTML sources from the associated Common Crawl WARC files.

#### A.1.2 Simplifying HTML Files

The original HTML content of a document contains a wealth of valuable information that proves highly beneficial in the process of filtering out undesirable text and images. Therefore, we prioritize pre-processing the raw HTML into simplified HTML, making the subsequent extraction of textual and visual elements more efficient. For this purpose, we use the library **selectolax**<sup>8</sup> that facilitates efficient parsing of HTML files and creates corresponding DOM trees.

**DOM Tree cleaning strategies** To simplify the DOM trees, we employ several cleaning strategies. Firstly, we convert tags that indicate line breaks (such as **<br>**) into actual line breaks. Multiple consecutive line breaks and spaces are condensed into a single instance. Additionally, HTML comments are removed from the DOM trees. Furthermore, we implement recursive processes to eliminate empty leaves and unnest nodes. When a parent node lacks attached text and has only one child, the child node replaces the parent node in the DOM hierarchy. We repeat these operations after removing some nodes, and describe this process in the following paragraphs.

---

<sup>6</sup><https://commoncrawl.org/>

<sup>7</sup><https://github.com/buriy/python-readability>

<sup>8</sup><https://github.com/rushter/selectolax>**Tag unwrapping** This operation involves removing unnecessary styling applied to displayed text by unwrapping a predefined set of tags given below. By applying this procedure, tags such as `<i>example</i>` are transformed into `example`, eliminating the associated styling elements.

The following tags are unwrapped during the processing of HTML files: `a`, `abbr`, `acronym`, `b`, `bdi`, `bdo`, `big`, `cite`, `code`, `data`, `dfn`, `em`, `font`, `i`, `ins`, `kbd`, `mark`, `q`, `s`, `samp`, `shadow`, `small`, `span`, `strike`, `strong`, `sub`, `sup`, `time`, `tt`, `u`, `var`, `wbr`.

**Node removal** Following the previous step, we conduct a manual inspection of practical examples encompassing all existing HTML tags. Based on our findings, we establish a curated list that outlines the tags we intend to retain. Any nodes within the HTML DOM tree with tags not included in this list are subsequently removed. We specifically retain tags that define the document structure (e.g., `p` or `h`) and tags associated with media elements (e.g., `img`). However, we opt to remove tags that typically consist of logos, generic content, or spam (e.g., `header`), as well as tags that often contain noisy text related to website navigation (e.g., `li`), or text that poses challenges in terms of linearization (e.g., `table`).

We retain the following tags during the processing of HTML files, as they define the document’s structure: `address`, `article`, `aside`, `blink`, `blockquote`, `body`, `br`, `caption`, `center`, `dd`, `dl`, `dt`, `div`, `figcaption`, `h`, `h1`, `h2`, `h3`, `h4`, `h5`, `h6`, `hgroup`, `html`, `legend`, `main`, `marquee`, `ol`, `p`, `section`, `summary`, `title`, `ul`. Additionally, we also preserve the following tags that define media elements: `audio`, `embed`, `figure`, `iframe`, `img`, `object`, `picture`, `video`. Furthermore, we keep the `source` tag as it may contain an interesting attribute.

**Modification of specific nodes** We then specifically target some `<div>` nodes that contain `footer`, `header`, `navigation`, `nav`, `navbar`, or `menu` as ID or `date` as attribute, as well as CSS rules that possess `footer` or `site-info` as class. These nodes typically contain website navigation content or article dates and are therefore removed. Additionally, we observe that the presence of a CSS rule with the class `more-link` often indicates a distinct shift in topic within the webpage, resembling the start of a new document. To account for this, we replace these nodes with the text `END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED`, which we replace by an end-of-sentence (EOS) token during training.

With these processing steps, we reduce the size of the HTML files by more than 10 on average while preserving the interesting content.

### A.1.3 Extracting Multimodal Web Documents

In this section, we begin with the simplified HTML files obtained from the previous section. Our objective is to transform these files into a structured web document format, which is a sequence of interleaved texts and images.

**Preservation of the original structure of the web pages** During the extraction process, we meticulously preserve the original structure of the web pages from the simplified HTML files. We extract the texts and image links while maintaining their order of appearance in the DOM tree. Each HTML tag denotes a distinct separation between the preceding and subsequent nodes and we retain any line breaks and line feeds that are present in the original page, preserving the formatting and visual rendering of the content.

**Image downloading** To download the images, we use the `img2dataset` (Beaumont, 2021) library. We attempt to download a massive collection of 3.6 billion images, of which 55% (approximately 2 billion images) were successfully downloaded. For that, we employ 20 virtual machines. This distributed approach allow us to complete the operation within a few days.

### A.1.4 Filtering Multimodal Web Documents

The filtering process consists of two steps, targeting different levels of granularity. In the first step, filtering occurs at the node level for images and at the paragraph level (separated by line breaks) for text. We evaluate each paragraph or image and we potentially modify orremove these based on specific criteria. The second step, conducted at the document level, involves deciding whether to retain or discard the output documents from the first step. The majority of the filters for text we use for both steps were adapted from Laurençon et al. (2022).

**Node-level image filtering** We discard images with formats other than `jpg`, `png` or `webp`, with a side length below 150 pixels or exceeding 20,000 pixels, as well as those with an aspect ratio greater than 2 or less than 1/2. These criteria help exclude images that are too small, excessively large, or have disproportionate dimensions, which are often indicative of low-quality or irrelevant content. To eliminate some logos and generic images, as in (Zhu et al., 2023), we remove images whose URL contains one of the sub-strings *logo*, *button*, *icon*, *plugin* or *widget*.

**Paragraph-level text filtering** Regarding text paragraphs, we apply a series of filters to remove undesirable or irrelevant content. We discard paragraphs with fewer than 4 words, as they typically contain insufficient information to be considered meaningful. Additionally, we remove paragraphs with a high repetition ratio, indicating potential spam content, and those with an excessive ratio of special characters, often associated with irrelevant or low-quality text.

Furthermore, we filter out paragraphs with a low ratio of stop words, as it is often indicative of machine-generated or nonsensical content. Similarly, we exclude paragraphs with a low punctuation ratio, as they typically indicate poor-quality texts. We also consider the flagged word ratio, removing paragraphs with a high proportion of flagged words associated with adult or inappropriate content. We also use KenLM (Heafield, 2011) models trained on Wikipedia to filter out paragraphs with excessively high perplexity scores.

To minimize spam, one approach is to identify generic sentences or invitations to share articles on social networks commonly found at the end of documents. We create a list of frequently used words associated with these paragraphs and then filter out paragraphs that contain an excessive proportion of words from this list.

To augment our ability to identify non-human-generated content, we consider a subset of 10 million documents from OSCAR (Ortiz Suárez et al., 2020), a web-crawled corpus. We extract the words from these documents, removed punctuations, converted them to lowercase, and retain only the words occurring at least twice, which we refer to as common words. We filter out paragraphs with a too low common word ratio.

The detail of the cutoff values for all text filters at the paragraph level is present in Table 3.

By applying these node-level and paragraph-level filters, we ensure that only high-quality and relevant images and paragraphs are retained for further processing and analysis.

**Document-level filtering** For document-level filtering, we start by removing all documents with no images or with more than 30 images. We have found that when there are too many images in a document, they are often not related to each other, and are more likely to be considered as spam.

For text filters, we use the same filters as for filtering at paragraph level. Since we are at the document level, the filter metrics are more precise, and we can typically set stricter cutoff values while limiting the number of false positives. The cutoff values used are also present in Table 3.

After these filtering steps, we obtained 365 million web documents and 1.4 billion images (potentially duplicated in different documents at this stage).

#### A.1.5 Additional Filtering and Deduplication Steps

**Exclusion of opted-out images** To respect the preferences of content creators, we remove all images for which creators explicitly opted out of AI model training. We used the Spawning API<sup>9</sup> to verify that the images in the dataset respect the original copyright owners’ choices. This step had a small impact on the overall dataset, by removing only 0.047% of the images.

---

<sup>9</sup><https://api.spawning.ai/spawning-api><table border="1">
<thead>
<tr>
<th>Metric</th>
<th>Cutoff type</th>
<th>Cutoff value (paragraph-level)</th>
<th>Cutoff value (document-level)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Number of words</td>
<td>min</td>
<td>4</td>
<td>10</td>
</tr>
<tr>
<td>Number of words</td>
<td>max</td>
<td>1,000</td>
<td>2,000</td>
</tr>
<tr>
<td>Character repetition ratio</td>
<td>max</td>
<td>0.1</td>
<td>0.1</td>
</tr>
<tr>
<td>Word repetition ratio</td>
<td>max</td>
<td>0.1</td>
<td>0.2</td>
</tr>
<tr>
<td>Special character ratio</td>
<td>max</td>
<td>0.3</td>
<td>0.275</td>
</tr>
<tr>
<td>Stop word ratio</td>
<td>min</td>
<td>0.3</td>
<td>0.35</td>
</tr>
<tr>
<td>Flagged word ratio</td>
<td>max</td>
<td>0.01</td>
<td>0.01</td>
</tr>
<tr>
<td>Punctuation ratio</td>
<td>min</td>
<td>0.001</td>
<td>0.03</td>
</tr>
<tr>
<td>Spam word ratio</td>
<td>max</td>
<td>0.12</td>
<td>0.12</td>
</tr>
<tr>
<td>Common word ratio</td>
<td>min</td>
<td>0.8</td>
<td>0.9</td>
</tr>
<tr>
<td>Language identification prediction score</td>
<td>min</td>
<td>0.8</td>
<td>0.8</td>
</tr>
<tr>
<td>Perplexity score</td>
<td>max</td>
<td>1500</td>
<td>1500</td>
</tr>
</tbody>
</table>

Table 3: Cutoff values for text filters at paragraph and document levels. A 'min' (or 'max') cutoff indicates that any paragraph or document, depending on the level, with a value for the considered metric strictly below (or above) the cutoff value is removed.

**Image deduplication based on URL** Prior to this step, it is possible for the same image to be present in multiple documents under the same URL. However, we observe that the distribution of image occurrences was highly skewed, with the majority of images appearing only once, while a small subset of images appeared hundreds of thousands of times. Upon closer examination, we notice that these frequently occurring images are predominantly comprised of common advertisements encountered during the crawling process, browser-specific icons, and similar elements. To address this issue, we remove all images that appear more than 10 times across the entire dataset. This approach significantly reduces the presence of unwanted images. We intentionally do not perform strict deduplication, as we observe that when an image is duplicated only a few times across different documents, the surrounding text and contextual information tend to vary. These diverse contexts associated with the duplicated image could be beneficial for the training of a model. We also deduplicate images within the same document.

**NSFW image removal** We use an open-source NSFW classifier<sup>10</sup> to reduce the proportion of explicit adult content within our dataset. We carefully choose a cutoff that reduces as much as possible the proportion of false positives. Indeed, if favoring precision to recall may seem to be a good idea to remove as much undesirable content as possible, it hurts diversity. An analysis of false positives shows that in many cases, simple portrait photos of women are classified as pornographic, which is not the case for men. People of color are also more often misclassified. We remove the entire document when a pornographically classified image is found in the document. In addition, we also remove all images whose URLs contain the sub-strings *porn*, *sex* or *xxx*. We remove approximately 1% of the documents with this filter. Note that many pornographic documents have been previously removed by the filter on flagged words.

**Document deduplication based on URL** Since we consider many Common Crawl dumps, it is possible that several documents may be associated with the same URL, despite the initial deduplication efforts. Recognizing the inherent similarity among these documents, we opt to retain only the most recent document for each common URL.

**Document deduplication based on set of images** It is possible that documents with different URLs and domain names are very similar and have not been removed by the first

<sup>10</sup>[https://github.com/GantMan/nsfw\\_model](https://github.com/GantMan/nsfw_model)deduplication, for instance, news articles copied and pasted multiple times across various sources. To mitigate this, we form groups of documents with an identical set of images, and we keep only the most recent document for each group.

**Paragraph deduplication across documents of the same domain names** To eliminate generic spam phrases commonly found at the end of documents, such as "Share on Facebook," "Post a comment," or "Accept the cookies," we implement a paragraph-level deduplication process within documents sharing the same domain name. This approach aims to enhance the quality of the text by removing redundant and repetitive content. For each domain name, we identify paragraphs that appear at least three times in an identical manner across associated documents. These repetitive paragraphs are subsequently removed from the documents, resulting in the elimination of approximately 15% of the text present in the web documents.

After all these steps, the final dataset contains 141 million documents and 353 million images, of which 298 million are unique.

We observe that using stricter values for the filtering steps yields fewer multimodal documents, although not of higher quality. As such, we invite users who are interested in manipulating a smaller subset of **OBELICS** to start with a random subset.## A.2 Analysis of OBELICS

### A.2.1 Examples of Multimodal Web Documents

#### Document

Right now, in Costa Rica, the classic dry season has been evasive. As the sky clouds over just as it did during June, and the rains begin to fall, it almost feels like the whole usual dry season thing has been waived. Cold fronts continue to arrive and subsequently douse the country with Atlantic showers while a "Hina" effect over the Pacific has only added to the wet situation. Despite the umbrella test, there are good things associated with this. High biodiversity is correlated with high rainfall and that makes for more birds. It's one of the main reasons why so many species occur in Costa Rica.

It can be a challenge to find them under varying degrees of precipitation but what's a bird gonna do? It's part of the local binding scene and when the clouds take a lunch break, the birds suddenly come out to play. Get enough of those breaks and you can get into some stellar birding, especially when high rainfall earlier in the year encouraged the trees and bushes to grow lots of bird friendly fruit. Seriously, it's a smorgasbord out there right now, the tanagers, manakins, thrushes, trigons, and toucans are going to feed whether it rains or not.

When the sun eventually does come out, there seem to be certain birds that take advantage of the sudden bloom of warmth and UV rays. Yesterday morning at El Tapio, a client and myself bore witness to what can happen when the rain finally comes to a stop and the sun, unhindered by clouds, punctuates the sky. At first, there was little activity, as if the birds were still numbed by the constant falling of water, still in denial that the rain had stopped. A few wrens and some other birds vocalized, a pair of Mealy Parrots fluttered overhead but pretty quiet otherwise. However, while the birds of the forest slowly came back to life, the Rufous-tailed Hummingbirds were racing around the garden. Judging by their frantic behavior (even for hummingbirds), it seemed like they hadn't eaten quite enough in days. Or maybe they just didn't get their fill of nectar! Whatever the case, they were drinking from the Verbena flowers as if they were participants in some avian Bacchus festivities. Unfortunately, they didn't invite any other hummingbirds to the party and took great efforts to bounce any potentially crashing woodnymph, Snowcap, or Violet-headed.

Dressed for the party, still denied entrance. Name's not down, not coming in.

It took a while but the Rufous-taileds seemed to eventually get their fill (or became too inebriated) and as the sun took over the garden space, a couple other hummingbird species braved the post party scene. One of the most cooperative was a male Black-crested Coquette.

As is typical with coquettes, the male chose to perch on a bare twig for extended periods of time before carefully flying down to drink from the Verbena. Much to our satisfaction, this particular exquisite beauty preferred to feed on a bush right in front of us.

It was interesting to note that as the coquette fed, the Rufous-taileds seemed to be more concerned with chasing a female woodnymph and a Violet-headed Hummingbird. It was as if they didn't notice the coquette as the smaller hummingbird slowly moved in and out of the flowering bushes, pumping its tail up and down the entire time.

As we enjoyed the coquette show, a few raptors eventually took advantage of thermals created by the sun to fly high over the garden.

As it turned out, the Black-crested Coquette was just the headliner for the main act.

The first on stage was an adult Ornate Hawk-Eagle. It called so loudly, I expected to see it floating just over the canopy but no, it was already high above the forest, fooling the eyes into thinking they were seeing something as small as an Accipiter or a dainty kite. The eagle called over and over, it was as if it couldn't help itself, singing because it could finally soar up and reach those heights again after a repressive bout of cool weather and constant rain. Above again! Like there was nothing else in its world, it yelled into the skies above the forest, fluttered its wings and made shallow dives, displaying over a busy roof for all who felt like peering into the high blue sky. Once, I swear it did a barrel roll, vocalizing the entire time.

As the eagle continued with its expression of exuberant defiance, next on the list were a pair of Barred Hawks. These broad-winged, short-tailed raptors gave their gull-like vocalizations as they soared into view. They continued to make circles up above the forest until they reached a point where they also began to display by soaring in tandem, calling the entire time.

One of the Barred Hawks, looks like it found some food that morning.

While this raptor fest was going on, a pair of King Vultures also soared into view, not as close as the hawks but still within eyeshot to appreciate their bold, black and white pattern. They seemed to be displaying as well, one bird almost flying into the other one and then close tandem flight, like the other raptors, taking advantage of a beautiful, new day.

It might rain a lot but it eventually stops. When it does, the sun's coming out something good is going to happen, the time comes for action. Whether you be a Spizetus or a bird, be ready to make your move and catch the lightbridge found in that window of respite.

Figure 7: Example of a document in OBELICS.

From <http://birdingcraft.com/wordpress/2018/01/23/what-happens-with-birding-in-costa-rica-when-the-rain-stops/>## Document

### Can I Expect Compensation For My Injuries?

The word "compensation" can be a touchy issue when discussing personal injuries and settlement. Even when it is the sole objective of a lawsuit or some other legal proceeding, mentioning compensation for my injuries can create false expectations in someone's mind if not addressed in the proper context. A San Diego lawyer who practices personal injury law, for example, says that it is crucial to ensure that a person seeking compensation has the right mindset and expectations whenever such cases are discussed. If mishandled, it can lead to anger and resentment on their part.

After suffering injuries in an accident, whether at the workplace or through some other negligent action, seeking damages is understandably a logical thing to do. Such legal action may entail going to court and making your case known to the judge. If there's a large sum of money involved, one should always prepare for a protracted legal battle.

The truth is that both a trial and an outright settlement can have very different variables and outcomes. Choosing to go to trial might seem like a good option. After all, many culpable parties are usually in a more agreeable frame of mind once the threat of a court case looms, making them more likely to offer a settlement.

Such parties usually settle a case out of self-interest. The strain and financial cost of sustaining an effective legal defense can be ruinous. In many cases, though, insurance companies step in to offer compensation. After all, many employers and other parties like vehicle drivers tend to have insurance coverage for exactly those sorts of situations. After sustaining injuries, an amount of money is offered to the victim to help them with medical bills and any other expenses they may have incurred due to injuries sustained. Many liable parties and insurance companies usually prefer a quick out-of-court settlement because court cases can become an expensive affair.

As a victim, it is always prudent to remember that a court case could be decided against you, thereby leaving you with no compensation at all. While some cases usually result in higher dollar amounts being doled out as a settlement because of successful litigation, many victims do not want to take the risk. Such victims are already drowning in medical bills by the time they think of seeking compensation for their injuries. That's why most prefer a swift settlement if given the option.

### How An Insurance Provider Chooses To Settle A Claim

As mentioned, an insurance provider involved in such cases would rather settle a personal injury case out of court. A jury trial is risky for both the personal injury victim and the insurance provider. The unpredictability of many such cases means that an insurance carrier could find themselves having to fork out significantly higher amounts of money in compensation than if they had chosen a quick, out-of-court settlement.

An insurance provider is always looking to minimize its costs while ensuring less risk. As such, they may opt to compensate a personal injury victim while simultaneously seeking reimbursement from the third party that is responsible for your injuries, usually from such a third party's insurance carrier.

It's crucial to remember that, in some jurisdictions, an insurance provider is entitled to a percentage of your compensation if they already settled your medical bills prior to you receiving the settlement. This amount is commensurate with all your medical expenses.

There now exist online settlement calculators that purport to provide a rough estimate of the compensation a personal injury victim can expect. You put in the various numerical values and factors related to your case, and the site will give you a general idea of what to expect in monetary terms. However, sometimes this information can be misleading and hence you should never rely on it. Even with the best personal injury lawyers handling your case, it is difficult if not impossible to account for all of the numerous variables. Even in cases with admitted liability of a third party, getting a sense of a definitive dollar amount for compensation is still difficult. The extent of the injury suffered, emotional distress and pain, and loss of potential future earnings are things that can prove very tricky to quantify. As such, it is inadvisable to rely on online settlement calculators for such estimates.

Medical costs and other expenses related to economic losses due to the injury are factored into calculating the damages awarded to a personal injury victim. Loss of companionship, deprived enjoyment of life, and emotional distress are some of the issues that determine compensation but may be hard to nail down.

While seemingly straightforward, any compensation awarded to a victim only happens after consideration of all relevant factors. Sometimes, the victim of personal injury is to blame, whether partly or in full. This has the potential to negate any compensation or at least diminish it. An experienced personal injury attorney can help such victims to fully understand all the different scenarios involved in such cases.

### Can A Victim Reject A Settlement Offer?

A personal injury victim is well within his rights to reject compensation. This could arise when the victim feels that the alleged guilty party has not put forward a dollar amount that is representative of the extent of injury and loss incurred. As a victim, you can sit down with your personal injury attorney to get a sense of how such scenarios generally play out. The accused party may be doing this intentionally, hoping that the victim accepts this offer without much consideration. You can express dissatisfaction with such an offer through a personal injury demand letter, outlining your grievances and why you believe you are entitled to more.

In a nutshell, a victim is entitled to compensation when the accused party is found to be responsible for the accident that caused injury to the victim. With many variables in such cases, there is no minimum amount of money set as the standard for compensation. Each case is examined on the merits of its unique factors, ensuring an equitable settlement for all parties.

Figure 8: Example of a document in OBELICS.  
From <https://www.halt.org/can-i-expect-compensation-for-my-injuries/>## Document

The Marvel Cinematic Universe has created some magnificent things over the last decade and a half. This cinematic universe has brought them back from the cusp of bankruptcy and into times of abundance once again. The success of the MCU has now allowed Marvel Studios to bring out the obscure characters from comic pages onto the silver screen. Who would have thought that Kit Harrington would be playing Dane Whitman in the MCU? It is relevant because Dane Whitman will become Black Knight, the greatest swordsman on the planet who fights alongside Avengers.

Who is this Black Knight? Why do we care? And why are we talking about this after a movie about cosmic beings like the Eternals and the Celestials? Does a sword not seem moot in front of infinite cosmic energy? Not when it is this sword. You see, in the after-credits scene of Eternals, Dane Whitman aka the love interest of Sersi unveils a sword. This sword seems to whisper to him and looks like the cursed Ebony Blade from the comics. Dane Whitman in the comics wields this blade and calls himself the Black knight, a superhero who assists the Avengers in various battles.

But there is a catch. The Ebony Blade was supposed to be welded by the pure of heart as explained by Merlin who created the sword. But the secret of the sword is that it can only be wielded by those who are impure of heart. The blade was actually designed by Merlin for Sir Percy (ancestor of Dane Whitman) to make him the greatest swordsman at the time. But the catch is that the blade seeks out evil inside you and amplifies it until there is nothing but a berserker left.

This seems to be true in the MCU too. The Ebony Blade blesses its user with incredible power, but it also comes at an incredible cost. This sword also prolongs its user's life as much as it can. The last Black Knight before Dane Whitman was Nathan Garrett, his uncle who is mentioned in the movie several times. This Black Knight was a villain who was defeated by the Avengers in the comics. But here, he is nowhere to be seen. There is a reason for this and the reason is most likely that Nathan Garrett will work better as a villain against Dane Whitman than the Avengers of the MCU.

This Ebony Blade is a malicious piece of weaponry. It was created by Merlin so that Sir Percy may sully his honor in battle but it also gave him immense power in the series. There is a possibility that we will see a similar story play out with Kit Harrington's character in the MCU. Moreover, there is another question that we must address. Who does the voice at the end of the second after-credits scene belong to? It has been confirmed by Chloe Zhao that it is Mahershala Ali's Blade who has come to recruit Dane.

Blade was the iconic movie that popularised superhero vampire hunters but there is another element to this hero that connects to the Black Knight. The Excaliburs was a team that got together to fight against supernatural foes. One of these foes was Dracula himself who was the one who created a replica of the Ebony Blade. In the comics, it was revealed that the Ebony Blade wielded by Dane was actually the replica created by Dracula.

This made the Blade itself vampiric in some sense and if this storyline is kept intact in the MCU then it won't be surprising to see Dane in Blade. It seems obvious at this point that the Ebony Blade will soon be replaced with Excalibur in the movies. Then plays with the original King Arthur sword in the Domo in Eternals. This is confirmed by sprite. We think that Dane will try to use the Ebony Blade to try to rescue Sersi from Arishem but would be asked by Blade to help him. This would start the Excalibur team-up and lead to the events of Blade where they hunt down Dracula.

After this, Dane might be consumed by the evil within the Ebony Blade and would discard it. To make sure that he can continue to be the hero he needs to be he will be given the Excalibur from The Domo and he will become the true leader of this new team. We think this will be the logical progression of events, taking a note from the current lineup of MCU movies, unless more are announced. Let us know what you think about this in the comments below and keep watching this space for everything Marvel, DC, and Hollywood. Excelsior!!!

Figure 9: Example of a document in OBELICS.

From <https://www.quirkybyte.com/blog/2021/11/how-dane-whitman-will-become-black-knight-kit-harringtons-character-explained/>### A.2.2 Unwanted Document Containing Many Images

Unwanted document containing many images

Figure 10: Undesirable document containing many images. Text is only present in small proportions, and the relation between the images is not always clear.### A.2.3 Top 100 Domains

<table border="1">
<thead>
<tr>
<th>Rank</th>
<th>Domain name</th>
<th>Number of documents</th>
</tr>
</thead>
<tbody>
<tr><td>1</td><td>www.dailymail.co.uk</td><td>434,498</td></tr>
<tr><td>2</td><td>en.wikipedia.org</td><td>155,258</td></tr>
<tr><td>3</td><td>nypost.com</td><td>141,494</td></tr>
<tr><td>4</td><td>www.thestar.com</td><td>138,224</td></tr>
<tr><td>5</td><td>sputniknews.com</td><td>133,695</td></tr>
<tr><td>6</td><td>www.rediff.com</td><td>133,233</td></tr>
<tr><td>7</td><td>www.theepochtimes.com</td><td>132,539</td></tr>
<tr><td>8</td><td>www.fool.com</td><td>125,220</td></tr>
<tr><td>9</td><td>www.businessinsider.com.au</td><td>123,841</td></tr>
<tr><td>10</td><td>www.bustle.com</td><td>122,581</td></tr>
<tr><td>11</td><td>www.dailysabah.com</td><td>120,029</td></tr>
<tr><td>12</td><td>www.firstpost.com</td><td>119,642</td></tr>
<tr><td>13</td><td>www.irishtimes.com</td><td>118,329</td></tr>
<tr><td>14</td><td>theathletic.com</td><td>101,982</td></tr>
<tr><td>15</td><td>www.news.com.au</td><td>98,339</td></tr>
<tr><td>16</td><td>www.indiatimes.com</td><td>98,197</td></tr>
<tr><td>17</td><td>www.theglobeandmail.com</td><td>92,805</td></tr>
<tr><td>18</td><td>tvropes.org</td><td>92,104</td></tr>
<tr><td>19</td><td>www.dailydot.com</td><td>91,034</td></tr>
<tr><td>20</td><td>mashable.com</td><td>88,310</td></tr>
<tr><td>21</td><td>observer.com</td><td>87,336</td></tr>
<tr><td>22</td><td>www.cbsnews.com</td><td>86,759</td></tr>
<tr><td>23</td><td>www.rappler.com</td><td>86,554</td></tr>
<tr><td>24</td><td>www.tnz.com</td><td>84,472</td></tr>
<tr><td>25</td><td>www.salon.com</td><td>84,420</td></tr>
<tr><td>26</td><td>www.modernghana.com</td><td>83,918</td></tr>
<tr><td>27</td><td>www.foxnews.com</td><td>83,002</td></tr>
<tr><td>28</td><td>www.huffpost.com</td><td>81,701</td></tr>
<tr><td>29</td><td>www.ndtv.com</td><td>81,549</td></tr>
<tr><td>30</td><td>www.thisismoney.co.uk</td><td>80,930</td></tr>
<tr><td>31</td><td>www.famousbirthdays.com</td><td>78,931</td></tr>
<tr><td>32</td><td>www.engadget.com</td><td>76,817</td></tr>
<tr><td>33</td><td>www.rnz.co.nz</td><td>76,327</td></tr>
<tr><td>34</td><td>www.metro.us</td><td>75,627</td></tr>
<tr><td>35</td><td>www.patheos.com</td><td>75,003</td></tr>
<tr><td>36</td><td>www.news24.com</td><td>73,883</td></tr>
<tr><td>37</td><td>www.thestar.com.my</td><td>73,265</td></tr>
<tr><td>38</td><td>www.dw.com</td><td>72,774</td></tr>
<tr><td>39</td><td>www.npr.org</td><td>71,939</td></tr>
<tr><td>40</td><td>koreajoongangdaily.joins.com</td><td>71,091</td></tr>
<tr><td>41</td><td>peoplesdaily.pdnews.cn</td><td>71,048</td></tr>
<tr><td>42</td><td>pagesix.com</td><td>70,602</td></tr>
<tr><td>43</td><td>www.thenigerianvoice.com</td><td>70,470</td></tr>
<tr><td>44</td><td>wikimili.com</td><td>69,928</td></tr>
<tr><td>45</td><td>www.indiebound.org</td><td>67,986</td></tr>
<tr><td>46</td><td>www.cricketcountry.com</td><td>66,605</td></tr>
<tr><td>47</td><td>expressdigest.com</td><td>64,250</td></tr>
<tr><td>48</td><td>www.capitalfm.co.ke</td><td>64,163</td></tr>
<tr><td>49</td><td>www.bizpacreview.com</td><td>64,157</td></tr>
<tr><td>50</td><td>www.wionews.com</td><td>63,797</td></tr>
<tr><td>51</td><td>profootballtalk.nbcsports.com</td><td>63,532</td></tr>
<tr><td>52</td><td>jamaica-gleaner.com</td><td>63,137</td></tr>
<tr><td>53</td><td>www.rte.ie</td><td>63,074</td></tr>
</tbody>
</table>

