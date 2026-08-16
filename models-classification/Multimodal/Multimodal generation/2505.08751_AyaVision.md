# Aya Vision: Advancing the Frontier of Multilingual Multimodality

Saurabh Dash<sup>★1</sup>, Yiyang Nan<sup>★1</sup>, John Dang<sup>1</sup>, Arash Ahmadian<sup>1,2</sup>,  
Shivalika Singh<sup>1</sup>, Madeline Smith<sup>1</sup>, Bharat Venkitesh<sup>2</sup>,  
Vlad Shmyhlo<sup>2</sup>, Viraat Aryabumi<sup>2</sup>, Walter Beller-Morales<sup>2</sup>,  
Jeremy Pekmez<sup>2</sup>, Jason Ozuzu<sup>2</sup>, Pierre Richemond<sup>2</sup>,  
Acyr Locatelli<sup>2</sup>, Nick Frosst<sup>2</sup>, Phil Blunsom<sup>2</sup>, Aidan Gomez<sup>2</sup>,  
Ivan Zhang<sup>2</sup>, Marzieh Fadaee<sup>1</sup>, Manoj Govindassamy<sup>2</sup>, Sudip Roy<sup>2</sup>,  
Matthias Gallé<sup>♦1</sup>, Beyza Ermis<sup>♦1</sup>, Ahmet Üstün<sup>♦1</sup>,  
and Sara Hooker<sup>♦1</sup>

<sup>1</sup>Cohere Labs, <sup>2</sup>Cohere

Corresponding authors: {saurabh, olivernan, matthias, beyza, ahmet, sarahooker}@cohere.com

## Abstract

Building multimodal language models is fundamentally challenging: it requires aligning vision and language modalities, curating high-quality instruction data, and avoiding the degradation of existing text-only capabilities once vision is introduced. These difficulties are further magnified in the multilingual setting, where the need for multimodal data in different languages exacerbates existing data scarcity, machine translation often distorts meaning, and catastrophic forgetting is more pronounced. To address the aforementioned challenges, we introduce novel techniques spanning both data and modeling. First, we develop a synthetic annotation framework that curates high-quality, diverse multilingual multimodal instruction data, enabling Aya Vision models to produce natural, human-preferred responses to multimodal inputs across many languages. Complementing this, we propose a cross-modal model merging technique that mitigates catastrophic forgetting, effectively preserving text-only capabilities while simultaneously enhancing multimodal generative performance. Aya-Vision-8B achieves best-in-class performance compared to strong multimodal models such as Qwen-2.5-VL-7B, Pixtral-12B, and even much larger Llama-3.2-90B-Vision. We further scale this approach with Aya-Vision-32B, which outperforms models more than twice its size, such as Molmo-72B and LLaMA-3.2-90B-Vision. Our work advances multilingual progress on the multi-modal frontier, and provides insights into techniques that effectively bend the need for compute while delivering extremely high performance.

**Aya-Vision-8B:** <https://huggingface.co/CohereLabs/aya-vision-8B>

**Aya-Vision-32B:** <https://huggingface.co/CohereLabs/aya-vision-32B>

**AyaVisionBench:** <https://huggingface.co/datasets/CohereLabs/AyaVisionBench>

---

★First authors. ♦Principal senior advisors.Figure 1: **Aya Vision models achieve state-of-the-art multilingual performance across both multimodal and text-only tasks.** We report multimodal and text-only win rates against Pangea-7B [Yue et al., 2024b], averaged over 23 languages. Aya-Vision-8B achieves *best-in-class* multimodal performance without compromising text capabilities, while Aya-Vision-32B outperforms all baselines, including much larger models such as Llama-3.2-90B-Vision [Grattafiori et al., 2024], establishing an optimal balance between efficiency and cross-modal strength.

## 1 Introduction

*We do not describe the world we see, we see the world we can describe.* — **René Descartes**

Although multimodal large language models (MLLMs) [Liu et al., 2023c; 2024; Deitke et al., 2024; Team, 2024b; Laurençon et al., 2024a; Chen et al., 2024; Bai et al., 2025; Team et al., 2025] have demonstrated remarkable success in jointly reasoning over various modalities, their performance remains predominantly confined to English. This linguistic limitation represents a substantial bottleneck in advancing multilingual AI, restricting global accessibility and impact.

Expanding multimodal models across languages exacerbates existing challenges at the frontier of AI. Foremost among these is the scarcity of high-quality multimodal data. While there has been expansion of languages served in language models [Üstün et al., 2024; Aryabumi et al., 2024b; Dang et al., 2024; Cohere et al., 2025], the intersection of both images and languages remains severely underserved. High-quality multimodal instruction-tuning datasets are scarce and primarily composed of short, simplistic, task-oriented image-text pairs [Goyal et al., 2017; Wang et al., 2021; Schwenk et al., 2022]. These datasets, while useful for benchmarking, inadequately prepare models for the rich, conversational scenarios encountered in real-world applications. Existing approaches primarily rely on machine translation to address this disparity [Li et al., 2023b; Maaz et al., 2024; Yue et al., 2024b]. However, translations often introduce linguistic artifacts (“translationese”), biases [Vanmassenhove et al., 2021; Savoldi et al., 2021; Hartung et al., 2023; Muennighoff et al., 2022],---

and fail to capture culture-specific nuances [Singh et al., 2024b; Salazar et al., 2025], contextual subtleties, and image-text alignments [Wang et al., 2022; Pudjiati et al., 2022]. Creating high-quality, culturally and linguistically accurate multimodal instruction data across diverse languages thus remains an essential yet unsolved challenge.

The second significant challenge is the known tension between adding vision capabilities and maintaining robust text-only performance. Integrating vision modalities commonly results in catastrophic forgetting, where models lose previously acquired language skills [Bai et al., 2023; Deitke et al., 2024; Grattafiori et al., 2024; Pozzobon et al., 2023]. This decay is further amplified when expanding coverage across multiple languages.

Equally pressing is the need for robust evaluations to measure progress. Any scientific pursuit requires a reliable metric of success. Existing multimodal benchmarks typically emphasize academic-style, multiple-choice tasks, evaluating models via rigid pattern-matching with predefined answer sets [Changpinyo et al., 2022; Romero et al., 2024; Yue et al., 2024b]. While useful for standardized comparisons, these fall short in capturing the nuanced, open-ended interactions that characterize real-world usage. Moreover, the few benchmarks that support more complex, open-ended interactions [Lu et al., 2024; Agrawal et al., 2024] are currently only available in English—leaving multilingual multimodal evaluation largely unexplored.

In this work, we tackle these challenges collectively. To address data scarcity, we replace naive translation pipelines with a hybrid method that pairs a specialized translation model with a larger LLM to correct and remove systematic translationese artifacts. We term this approach *context-aware rephrasing*, which enables the creation of higher-quality, human-preferred multimodal instruction data. We also systematically explore the benefits of merging to mitigate catastrophic forgetting. We propose a **a novel cross-modal merging strategy** (§ 3) that fuses capabilities across models, allowing for preservation and “on-the-fly” extension of capabilities across modalities. We see this as a powerful new paradigm to create adaptive models efficiently for new tasks. Our merging paradigm improves text-only tasks 50.2% and multimodal tasks 20.5% relative to the unmerged checkpoint, due to the inherent compositionality between the tasks and modalities.

The result of our work is **Aya Vision**, a family of state-of-the-art multilingual multimodal models available in 8B and 32B sizes. In contrast to the many existing MLLMs, Aya Vision models are trained with a strong emphasis on multilingual and multimodal generation, yielding fluent chat performance. Aya-Vision-8B achieves *best-in-class* performance, surpassing Qwen-2.5-VL-7B [Bai et al., 2025], Llama-3.2-11B-Vision [Grattafiori et al., 2024], Pixtral 12B [Agrawal et al., 2024], and Gemini-Flash-1.5-8B [Team, 2024b], with up to 79% win rate across multimodal tasks in 23 languages. Aya-Vision-32B outcompetes models over twice its size, including Llama-3.2-90B-Vision [Grattafiori et al., 2024], Molmo-72B [Deitke et al., 2024], and Qwen-2.5-VL-72B [Bai et al., 2025], with win rates up to 72.4%.

Our primary contributions are as follows:

1. 1. **A family of state-of-the-art multilingual multimodal LLMs:** We introduce Aya-Vision-8B and 32B models, covering 23 languages spoken by half the world’s population. In contrast to the many existing multimodal LLMs, Aya Vision models are trained with a strong emphasis on multilingual, multimodal generation, yielding fluent chat performance preferred by humans.Figure 2: **Aya Vision** establishes a new Pareto frontier in the performance-efficiency trade-off. We show multimodal win rates against Pangea-7B, with respect to the number of parameters for each model.

1. **2. A novel multilingual multimodal synthetic annotation framework:** We develop a novel multilingual multimodal framework that combines synthetic data distillation, automated translation, and context-aware rephrasing to produce high-quality and diverse instruction data across languages, addressing data scarcity challenges. Recaptioning increases the average number of tokens from 27.2 to 140.8 and the measure of lexical diversity from 11.0 to 61.2. Our translation pipeline improves the translation quality by 11.24% over the NLLB-3.3B [Costa-Jussà et al., 2022] translations.
2. **3. Optimizing performance across modalities with cross-modal model merging:** We introduce a novel cross-modal model merging strategy that not only recovers text-only capabilities lost to catastrophic forgetting – boosting text win-rates by up to 50.2% but simultaneously enhances multilingual multimodal performance – improving vision win-rates by up to 20.5%, demonstrating an efficient, training-free path to stronger models across modalities.
3. **4. A comprehensive benchmark suite for real-world multilingual multimodal evaluation:** We introduce *AyaVisionBench*<sup>1</sup>, a benchmark spanning 23 languages and 9 vision-language tasks, specifically designed to evaluate generative, open-ended instruction following. To support multilingual evaluation further, we introduce *m-WildVision*<sup>2</sup>, a high-quality translation of WildVision [Lu et al., 2024]. Together, they offer a meaningful and challenging testbed for multimodal models.

<sup>1</sup><https://huggingface.co/datasets/CohereLabs/AyaVisionBench>

<sup>2</sup><https://huggingface.co/datasets/CohereLabs/m-WildVision>## 2 A Comprehensive Multilingual Multimodal Data Framework

To solve for the scarcity of multilingual multimodal instruction data, prior efforts often depend on direct LLM-based translations of English-centric datasets. Approaches such as Pangea [Yue et al., 2024b] and Palo [Maaz et al., 2024] extend coverage across languages either through large-scale translation or multilingual caption alignment. However, these methods still struggle with limited linguistic diversity, the introduction of “translationese” from overreliance on translation, strict task formulations, and a lack of conversational naturalness.

To address these gaps, we introduce a robust multimodal synthetic re-annotation pipeline for constructing high-quality multilingual multimodal datasets. As illustrated in Figure 3, our pipeline comprises three core stages: 1) *distillation-based recaptioning* (§ 2.2), 2) *dataset filtering* (§ 2.3), and 3) *translation combined with multilingual rephrasing* (§ 2.4). This pipeline significantly enhances the dataset’s quality, diversity, and linguistic coverage, resulting in a rich multilingual instruction dataset spanning 23 languages.

### 2.1 Data Collection

We began dataset construction by curating a diverse English-language multimodal instruction-tuning corpus. We constructed our dataset on well-established open-source resources, including *Cauldron*<sup>3</sup> [Laurençon et al., 2024b], a large-scale collection of 50 vision-language datasets (~30M samples), and *PixMo*<sup>4</sup> [Deitke et al., 2024], a comprehensive dataset spanning seven multimodal tasks (~6M samples). We also drew from other sources such as *SlideVQA* [Tanaka et al., 2023], *PDFVQA* [Ding et al., 2023], and *ScreenQA* [Hsiao et al., 2022]. Our dataset follows Cauldron’s framework and covers a broad range of multimodal tasks: visual question answering (VQA), captioning, OCR and document understanding, chart and figure analysis, table comprehension, logical reasoning, academic or textbook questions, image comparison, and code generation from screenshots. As Cauldron performs upstream filtering to remove duplicates across its aggregated sources,

Table 1: Task-wise distribution in our curated dataset, showing the proportion and the number of samples in the ~2.29M collection.

<table border="1"><thead><tr><th>Task</th><th>VQA</th><th>Capt.</th><th>OCR/<br/>Doc</th><th>Chart/<br/>Fig</th><th>Table<br/>Compr.</th><th>Logic.<br/>Reasoning</th><th>2 Image<br/>Diff.</th><th>Textbook</th><th>SS to<br/>Code</th></tr></thead><tbody><tr><td><b>Total Samples</b></td><td>560K</td><td>220K</td><td>490K</td><td>289K</td><td>222K</td><td>252K</td><td>239K</td><td>20K</td><td>9.5K</td></tr><tr><td><b>Proportion</b></td><td>24.5%</td><td>9.6%</td><td>21.4%</td><td>12.6%</td><td>9.2%</td><td>11.0%</td><td>10.4%</td><td>0.9%</td><td>0.4%</td></tr></tbody></table>

the subset we use does not contain repeated samples. Likewise, the PixMo data we incorporate – primarily within the chart and figure category – consists of synthetically generated content that is distinct from Cauldron and other sources, ensuring no overlap across datasets.

To ensure robust generalization across task types, we regulated the number of samples per category to construct a balanced and representative dataset. The final collection contains ~2.29M samples, with the task-wise sample counts and distribution detailed in Table 1. This English data mixture serves as the basis for our further synthetic re-annotation and translation pipeline, forming multilingual instruction tuning set used to train Aya Vision.

<sup>3</sup>[https://huggingface.co/datasets/HuggingFaceM4/the\\_cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)

<sup>4</sup><https://huggingface.co/collections/allenai/pixmo-674746ea613028006285687b>**Original Data**

**Prompt:**  
What kind of beverage is in the french press?

**Caption:**  
Coffee

Source: OK-VQA [Schwenk et al., 2022]

**Distillation Based Recaptioning**

**Prompt:**  
What kind of beverage is in the french press?

**Recaption:**  
The French press in the image contains a dark liquid, which is likely coffee based on its color and consistency. The presence of a spoon next to the press also suggests that it might be used for stirring a beverage like coffee. Additionally, the overall warm and cozy setting, with a lit candle and autumnal decorations, is often associated with enjoying a hot cup of coffee. Therefore, it is reasonable to conclude that the beverage in the French press is coffee.

**Machine Translation**

**Translated Prompt:**  
法国媒体上有什么样的饮料?

**Translated Recaption:**  
照片中的法式印刷机含有深色液体 根据颜色和<unk>度可能是咖啡 压<unk>机旁边有一个<unk>匙 也表明它可能用于<unk>咖啡等饮料 另外 整体温暖舒适的环境 灯<unk>和秋季装饰 常常与享受一杯热咖啡有关 因此可以合理地得出结论 法国媒体所提到的饮料是咖啡。

**Rephrasing**

**Rephrased Prompt:**  
法国压滤壶中通常冲泡什么饮料?

**Rephrased Recaption:**  
照片中的法式压滤壶装有深色液体，从其颜色和稠度来看，很可能是咖啡。压滤壶旁边有一把勺子，也暗示它可能用于搅拌咖啡等饮料。此外，整体温馨舒适的环境，点燃的蜡烛和秋季装饰，通常与享用一杯热咖啡的氛围相符。因此，可以合理地推断，法式压滤壶中的饮料是咖啡。

Figure 3: **Our synthetic annotation pipeline enables diverse, high quality responses for multimodal instructions.** The pipeline consists of three core stages: (1) distillation-based recaptioning, (2) machine translation, and (3) rephrasing. We highlight common machine translation errors, such as unknown tokens (e.g. consistency, lit candle) or mistranslations, as in the case of ‘French press’ rendered as ‘French media’ due to lexical ambiguity in the word ‘press’. Rephrasing helps to resolve such issues, improving both the fluency and semantic accuracy of translations.

## 2.2 Distillation-based Recaptioning

Our goal with recaptioning is to alter the data space such that it better reflects the data distribution we aim to represent in the real-world. To achieve this, we generate synthetic alternatives to the original completions across the  $\sim 2.3\text{M}$  data points in our English dataset selection.

The original dataset is primarily composed of open-source, academic image captioning corpora, which exhibit limited linguistic diversity and constrained stylistic variation. Much of the data originates from a narrow set of sources such as MS-COCO [Lin et al., 2014], Visual Genome [Krishna et al., 2017], and Open Images [Kuznetsova et al., 2020], leading to repetitive content and reduced variation in captions for similar images. Furthermore, these English datasets are typically short and simplistic (average caption length across datasets is just 14.2), and often lack detailed descriptions or a conversational tone expected from state-of-the-art generative models.

Given these limitations, our goal with synthetic re-annotation is to generate recaptions that are more detailed, natural, and diverse in both tone and content. However, a key constraint in this process is that the recaptioned outputs also must remain anchored to the ground-truth answer.

The effectiveness of the recaptioning depends heavily on the quality of the prompt templates, which play a critical role in shaping the richness and relevance of the generated annotations [Guo et al., 2024; Fang et al., 2024]. To enhance the quality of our synthetic data, we design task-specific prompt templates for the teacher model, which guide the recaptioning process. These prompt strategies are adapted to rewrite captions based on the ground-truth and to meet the requirements of different vision-language tasks. For example, templates for reasoning tasks are more structured to elicit step-by-step explanations; captioning prompts emphasize more detailed and informative descriptions; and VQA prompts are designed to have accurate and image-grounded answers. Table 2 presents examples for two different tasks, illustrating how recaptioning instructions vary by context. For additional task types and full prompt formats, see Appendix D.Table 2: Examples of task-specific recaptioning outputs for different prompt strategies.

<table border="1">
<thead>
<tr>
<th>Task Type</th>
<th>Prompt Instruction (Simplified)</th>
<th>Sample Recaptioned Output</th>
</tr>
</thead>
<tbody>
<tr>
<td><b>Captioning</b></td>
<td>Rewrite the original caption to be more detailed, descriptive, and human-like. Avoid introducing unseen elements.</td>
<td>A man wearing a red helmet rides a mountain bike along a forest trail, surrounded by tall green trees.</td>
</tr>
<tr>
<td><b>Reasoning / Math</b></td>
<td>Solve the visual/mathematical problem with a clear, step-by-step explanation. Ensure logical correctness and clarity. The response should be logical, clear, and easy to follow. Include intermediate reasoning steps.</td>
<td>To find the total, we multiply 4 by 3 because there are 4 rows with 3 items each. <math>4 \times 3 = 12</math>. So, the final answer is 12.</td>
</tr>
</tbody>
</table>

Taken together, recaptioning serves to bridge the gap between limited, narrowly scoped training data and the rich, diverse language found in real-world contexts. To quantify its linguistic impact, we analyze several textual properties –average word count, number of tokens, and lexical diversity – using the Measure of Textual Lexical Diversity (MTLD) [Shen, 2022]. Following recaptioning, the average word count increases **from 14.2 to 100.1**, token count rises **from 27.2 to 140.8**, and MTLD improves **from 11.0 to 61.2**. Higher MTLD scores indicate greater vocabulary variation; a score of 61.2 suggests strong lexical richness comparable to fluent language use [McCarthy & Jarvis, 2010; Ploeger et al., 2024]. These more expressive and natural annotations support better generalization and improved robustness in downstream multimodal tasks. Examples of recaptioned outputs are provided in Appendix E.

## 2.3 Verifying and Filtering Recaptioned Instruction Data

Recaptioning offers a scalable approach to improving the quality of model responses. However, synthetic generations can still introduce errors or hallucinated content that is not grounded in the image [Rohrbach et al., 2018; Liu et al., 2023b; Li et al., 2023c; Gunjal et al., 2023]. Training on such data may amplify a model’s tendency to hallucinate or generate inaccuracies that compromise overall quality. To mitigate these risks and ensure both fluency and correctness, we implement a two-stage filtering pipeline that enhances the overall reliability of the recaptioned dataset. While some methods apply single-pass alignment filtering, e.g CLIP score [Gadre et al., 2023], or train models to avoid hallucinations using reward learning [Ben-Kish et al., 2023; Wang et al., 2024a], our two-stage pipeline adds an extra safeguard against the inclusion of fluent yet hallucinated outputs.

**Stage 1: Keyword-based filtering.** We begin with simple keyword detection to identify recaptioned samples that exhibit common failure modes, such as refusals to respond or repeated phrases from the input prompt. To catch these issues, we compile a list of keywords and phrases that automatically flag such responses. Flagged samples are either sent back to the model for regeneration or discarded if the issue persists.

While keyword-matching can detect basic errors, it still struggles to identify more subtle inaccuracies. This limitation is particularly critical for tasks that require deterministic or subjective answer, such as question answering or mathematical reasoning. In these cases, the teacher model may ignore the provided ground truth or hallucinate details, resulting in flawed or incorrect answers.---

**Stage 2: LLM-based semantic filtering.** To address more nuanced errors, we apply a second-stage filtering using `command-r-plus-08-2024`<sup>5</sup> for semantic verification (see Appendix F for the prompt). In this stage, the original and rephrased captions are presented to the model, which acts as a semantic judge to assess whether the answer to the original caption remains valid given the rephrased version. This ensures that recaption do not alter the underlying meaning or contradict with the ground truth answer. All corrupted samples identified at this stage are discarded. This step reveals an overall error rate of 3.2% (62,370 samples) in the recaptioned data. Task complexity significantly influences error frequency – for example, reasoning tasks exhibit a higher error rate (4.6%) than simpler VQA tasks (2.5%). This trend aligns with findings from prior work [Yue et al., 2024a; Wang et al., 2024c; Song et al., 2025]. By integrating keyword-based filtering with nuanced semantic evaluation capabilities of an LLM, our pipeline generates a recaptioned dataset that is cleaner, more reliable, and better optimized for visual instruction tuning. Examples of filtered samples are provided in Appendix F.

## 2.4 Hybrid Translation Pipeline for Multilingual Instruction Data

Our approach diverges from prior efforts that either rely exclusively on proprietary LLMs for translation [Yue et al., 2024b; Maaz et al., 2024] or highlight disparities in translation quality between high- and low-resource languages without directly addressing how to mitigate them [Hendy et al., 2023]. For example, Hendy et al. [2023] find that GPT models perform competitively on high-resource languages but struggle significantly with low-resource ones. Although machine translation has inherited limitations, it remains essential for broad language coverage, especially in-language human-curated datasets in many languages are scarce and typically reserved for evaluation [Singh et al., 2024b; Romanou et al., 2024; Aakanksha et al., 2024b; Singh et al., 2024a; Salazar et al., 2025]. Prior work has also shown that translating instruction data can significantly improve cross-lingual generalization in language models [Ranaldi & Pucci, 2023; Dang et al., 2024; Ermis et al., 2024; Üstün et al., 2024].

However, while machine translation models offer broad coverage, they often introduce artifacts that compromise fluency and fidelity. These include unnatural phrasing, incorrect lexical choices, or incomplete renderings as documented in prior studies [Bizzoni et al., 2020; Vanmassenhove et al., 2021; Üstün et al., 2024; Singh et al., 2024b]. Large language models may struggle with translation, especially in low-resource language contexts [Zhu et al., 2023]. To balance language coverage with translation quality, we adopt a **hybrid approach**:

- • We begin with machine translation, using the NLLB-3.3B model<sup>6</sup> [Costa-Jussà et al., 2022]. Specifically, we translate our re-annotated English dataset (see §2.2) into the following 22 languages: *Arabic, Chinese, Czech, Dutch, French, German, Greek, Hebrew, Hindi, Indonesian, Italian, Japanese, Korean, Persian, Polish, Portuguese, Romanian, Russian, Spanish, Turkish, Ukrainian, and Vietnamese*.
- • We then apply a post-editing step using a capable multilingual language model, `command-r-plus-08-2024`<sup>5</sup>, to refine the translations. This step uses the initial machine-translated output as an in-context example to guide the model toward generating more fluent and accurate outputs [Zhu et al., 2023; Raunak et al., 2023]. In doing so, we correct common machine translation artifacts while preserving the original semantic content.

---

<sup>5</sup><https://huggingface.co/CohereLabs/c4ai-command-r-plus-08-2024>

<sup>6</sup><https://huggingface.co/facebook/nllb-200-3.3B>---

The prompt used for this rephrasing step and some examples illustrating improvements from rephrasing are in Appendix G.

This two-stage process ensures higher translation quality across languages by combining broad coverage from machine translation with fluency improvements from LLM-based post-editing. To further improve training efficiency and generalization, we do not translate the full English dataset into all 22 languages; instead, we randomly sample subsets of the English pool of examples for each language. This approach improves efficiency and helps avoid overfitting by reducing repeated exposure to identical content across languages. Prior work has shown that partial translation can achieve strong multilingual generalization while significantly reducing data size [Geigle et al., 2023; Shaham et al., 2024], and is commonly used in large-scale multilingual datasets to enhance linguistic diversity without unnecessary duplication [Muennighoff et al., 2022; Nguyen et al., 2024; Üstün et al., 2024; Dang et al., 2024; Aryabumi et al., 2024a].

To evaluate translation quality, we report COMET<sup>7 8</sup> [Rei et al., 2020; 2023], a reference-free machine translation evaluation metric. The translations from NLLB-3.3B achieve an **average score of 0.7455** across the 22 languages. After post-editing, the **average score increases to 0.8293**, indicating the effectiveness of our hybrid strategy. COMET scores typically range from 0 to 1, with higher values indicating better adequacy and fluency. Thus, a gain of over 0.08 reflects a substantial quality improvement. Detailed per-language COMET improvements are reported in Table 7 in Appendix K.

### 3 Balancing Performance across Languages, Modalities and Tasks

For multimodal LLMs, carefully sampling the fine-tuning mixture with high-quality and task-oriented visual instructions is crucial for optimal performance [Liu et al., 2023c; Laurençon et al., 2024b; Tong et al., 2024; Dai et al., 2024]. In multilingual multimodal LLMs, this challenge intensifies as the balancing should be optimized for both multilingual and multimodal dimensions. Previous works [Üstün et al., 2024; Aryabumi et al., 2024b; Dang et al., 2024] have shown that a skewed distribution of languages in the training mixture hampers the model’s ability to learn reliably, leading to measurable drops in accuracy on a subset of languages. Furthermore, a state-of-the-art multimodal LLM should also retain its text-only capabilities, as these models are often deployed in real-world scenarios that encompass both multimodal and text-only use cases.

Retaining the text-only performance of the backbone LLM, while acquiring strong multimodal capabilities through multimodal training is challenging for several reasons. Firstly, choosing the data mixture to strike a balance between multimodal and text datasets is a challenging problem, as finding the right balance is non-trivial and requires a multitude of ablations. For instance, Molmo [Deitke et al., 2024] and Pangea [Yue et al., 2024b] include approximately 10% text-only data in their multimodal SFT mixture to retain text performance. While this might enable minimal degradation on text-only academic benchmarks, we observe in practice that both models suffer a significant drop in open-ended generation performance measured by the preference evaluation as shown in Figure 5.

Secondly, reintroducing previously seen text-only data can potentially lead to overfitting with minimal improvement in text performance and a higher degradation in multimodal performance [Marafi-

---

<sup>7</sup><https://github.com/Unbabel/COMET>

<sup>8</sup><https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xxl><table border="1">
<thead>
<tr>
<th>Task</th>
<th>Orig.</th>
<th>Multi.</th>
<th>Synth.</th>
<th>Total</th>
<th>Per(%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>General VQA</td>
<td>269.0k</td>
<td>311.2k</td>
<td>168.2k</td>
<td>748.4k</td>
<td>27.2</td>
</tr>
<tr>
<td>Captioning</td>
<td>-</td>
<td>74.6k</td>
<td>109.0k</td>
<td>183.6k</td>
<td>6.7</td>
</tr>
<tr>
<td>OCR</td>
<td>231.8k</td>
<td>60.7k</td>
<td>188.8k</td>
<td>481.3k</td>
<td>17.5</td>
</tr>
<tr>
<td>Figures/Charts</td>
<td>290.0k</td>
<td>31.3k</td>
<td>159.6k</td>
<td>480.9k</td>
<td>17.5</td>
</tr>
<tr>
<td>Table Compr.</td>
<td>77.5k</td>
<td>260.7k</td>
<td>56.5k</td>
<td>394.7k</td>
<td>14.4</td>
</tr>
<tr>
<td>Reason./Logic/Math</td>
<td>-</td>
<td>136.4k</td>
<td>60.9k</td>
<td>197.2k</td>
<td>7.2</td>
</tr>
<tr>
<td>Multi Image</td>
<td>39.6k</td>
<td>78.0k</td>
<td>97.3k</td>
<td>214.8k</td>
<td>7.8</td>
</tr>
<tr>
<td>Textbook/Academic</td>
<td>19.1k</td>
<td>-</td>
<td>12.8k</td>
<td>31.9k</td>
<td>1.2</td>
</tr>
<tr>
<td>Screenshot → Code</td>
<td>9.5k</td>
<td>5.2k</td>
<td>-</td>
<td>14.7k</td>
<td>0.5</td>
</tr>
<tr>
<td><b>Total</b></td>
<td><b>936.3k</b></td>
<td><b>958.1k</b></td>
<td><b>853.0k</b></td>
<td><b>2.75M</b></td>
<td><b>100%</b></td>
</tr>
</tbody>
</table>

Figure 4: **Overview of our multilingual multimodal SFT mixture from various task categories.** Left: Number of samples across data sources and tasks categories used in training. Right: Visual breakdown of dataset source distributions.

oti et al., 2025]. We further investigate this pattern through ablations, presented in detail in Section 7.2. Moreover, post-training of state-of-the-art LLMs typically involves several steps of SFT and preference optimization [Dang et al., 2024; Lambert et al., 2024; Cohere et al., 2025], which could cause instability due to the shift in the data distribution in the multimodal fine-tuning step. This highlights the importance of striking a balance during multimodal fine-tuning to maintain model robustness and generalization.

In this work, we explore a variety of mitigation to this solution including (1) systematic weighting of different sources of data to preserve both language balancing and diversity, (2) Cross-modal model merging to seamlessly integrate multimodal and text-only capabilities.

### 3.1 Sampling Visual Instructions from Multiple Sources and Languages

To balance coverage and preserve diversity, we mix and weight three buckets of data:

1. 1. **Synthetically Re-annotated data in English:** This data was generated after the first phase of our data framework (§ 2.2), 2.29M samples in total. Inside this bucket, we upsample datasets with a small number of samples, such as science or textbook questions, to avoid underrepresenting any task categories. Additionally, we also upsample datasets deemed to be of higher quality upon manual inspection leading to a total of 3.5M samples from this bucket being seen by the model.
2. 2. **Multilingual datasets:** This data was generated by using a subset of re-annotated English dataset through our data framework (§ 2.4). We uniformly sample data across 22 languages (except English) and maintain a similar task distribution to the first bucket. While the total data volume in this bucket amounts to a total of 5M samples, we sample 3.4M uniformly distributed across 22 languages (except English) to preserve the balance between tasks.
3. 3. **High-quality original datasets:** In addition to the fully synthetic data, we also use a selection of original datasets, based on their quality. This bucket is required since some downstream VQA evaluations expect syntactically accurate answers that match their trainingdistribution and penalize semantically correct generations (for example 0.5 instead of 1/2). However, we downsample the original corpus to avoid a drop in overall quality, as this data penalizes natural generations and completion length – thereby degrading the model’s free-form conversational abilities. While the total number of samples in this bucket is 6M, we sample 3.7M for training.

In each data bucket, we ensure a diverse set of tasks is represented. To enhance multilingual performance, we experiment with varying proportions of multilingual data – these results are presented in § 7.4. Based on these findings, we use approximately 66% of synthetically re-annotated datasets out of which 35% corresponds to the multilingual datasets; while the remaining 34% are the high-quality original datasets. Figure 4 illustrates the composition of the training data across buckets and tasks, totaling 2.75M sequence-packed final training samples.

### 3.2 Unifying Multimodal Performance with State-of-the-Art Text Capabilities

In Aya Vision, instead of balancing multimodal and text-only abilities in the data space via a sweep over data mixtures, we introduce a novel cross-modal model merging inspired by the recent body of work in model merging [Wortsman et al., 2022; Matena & Raffel, 2022; Yadav et al., 2023; Aakanksha et al., 2024a; Goddard et al., 2024]. Concretely, we posit that since the multimodal model is initialized from the final preference-tuned LLM checkpoint, sharing a part of the optimization trajectory [Izmailov et al., 2018; Frankle et al., 2020; Ilharco et al., 2022] makes the multimodal LLM and the backbone LLM amenable to merging. Cross-modal model merging introduces an efficient, training-free recovery solution for retaining text-only performance by balancing multimodal and text-only capabilities in the weight space *a posteriori*. We conduct systematic study of merging techniques applied to the weights of the original text-only LLM and the LLM backbone of the multimodal model (see § 7.1).

We perform a linear interpolation between the text-only LLM and the backbone LLM of the multimodal model as the merging method, as shown in Equation 1. Since the text-only language model lacks the vision encoder and alignment layer, we simply inherit them from the vision-language model.

$$W_{\text{merged}} = \alpha \cdot W_{\text{mm-LLM}} + (1 - \alpha) \cdot W_{\text{text-LLM}} \quad (1)$$

Figure 5: **Degradation in text-only win-rates after multimodal training.** Each model is compared to their initial LLM on mArenaHard [Dang et al., 2024]. We see that only including a percentage of text-only data in the final multimodal training mix is insufficient to retain open-ended generative performance.---

## 4 Aya Vision’s Architecture and Training Details

### 4.1 Architecture

Aya Vision models follow the common architecture design for vision-language models [Liu et al., 2023c; 2024; Laurençon et al., 2024b; McKinzie et al., 2024; Chen et al., 2024; Deitke et al., 2024] that is based on late-fusion [Team, 2024a] of (1) a vision encoder to compute image patch embeddings which is pre-trained on billions of image-text pairs [Radford et al., 2021; Zhai et al., 2023; Chen et al., 2024; Tschannen et al., 2025], (2) a connector that maps the embeddings from the output space of the vision encoder to the input embedding space of the language model, (3) a large language model.

**Vision Encoder:** We use siglip2-so400m [Tschannen et al., 2025] as the initialization for the vision encoder, which has been pretrained with an auto-regressive decoder-based loss in addition to the original sigmoidal loss [Zhai et al., 2023]. This primes the vision encoder to generate high-quality dense feature representations for generative tasks, making it the perfect candidate for a multilingual vision language model. Specifically, we use siglip2-so400m-patch14-384<sup>9</sup> in Aya-Vision-8B for a reduced activation footprint, making it widely accessible on cheaper hardware. For Aya-Vision-32B, we opt for the higher resolution siglip2-so400m-patch16-512<sup>10</sup> to achieve better performance [Laurençon et al., 2024b].

**Image Processing:** The performance of multimodal LLMs improves with higher input resolution [McKinzie et al., 2024; Laurençon et al., 2024b], however, most vision encoders are pretrained on a fixed resolution. To enable Aya Vision models to process images with arbitrary resolutions, similar to Chen et al. [2024], we map the input images to the nearest supported resolution that minimizes distortion in the aspect ratio. After resizing, we split the image into up to 12 non-overlapping tiles based on the image encoder’s resolution to be processed independently by the vision encoder. In addition to tiles, we include a thumbnail (resized) for a low-resolution overview of the image.

**Vision-Language Connector:** Following the image encoder, the vision-language connector maps features from the vision encoder to the language model’s input embedding space. We use a 2-layer MLP with SwiGLU activation function [Shazeer, 2020]. To reduce the number of image tokens passed to the language model, we perform Pixel Shuffle [Chen et al., 2024], which downsamples the image tokens in the spatial dimensions by stacking  $2 \times 2$  patch embeddings along the embedding dimension before passing through the connector layer. This decreases the number of image tokens by  $4 \times$ , resulting in a maximum of 2,197 and 3,328 image tokens for our 8B and 32B models respectively. When passing image tokens to LLM, we use special delimitation tokens to denote the start and the end of image token sequences. Additionally, we inject 1D-tile tags [Dai et al., 2024] to denote image tiles as a form of explicit positional encoding for the tiles. We use regular text tokens (TILE\_1, ..., TILE\_N and TILE\_GLOBAL for thumbnail) for potential inference-time scaling.

**Language Model:** Although some previous works initialize the language model from a pre-trained base checkpoint [Beyer et al., 2024], we initialize the language model from a multilingually post-trained LLM to inherit strong capabilities in various tasks including chat, instruction-following, and multilingual. For Aya-Vision-8B, we use an LLM based on Command-R7B<sup>11</sup> which is further

---

<sup>9</sup><https://huggingface.co/google/siglip2-so400m-patch14-384>

<sup>10</sup><https://huggingface.co/google/siglip2-so400m-patch16-512>

<sup>11</sup><https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024>---

post-trained with the Aya Expanse recipe [Dang et al., 2024], and for Aya-Vision-32B, we use the Aya-Expanse-32B [Dang et al., 2024].

## 4.2 Multimodal Training

Following previous work that use late-fusion as in our models [Liu et al., 2023c; 2024; Laurençon et al., 2024b; McKinzie et al., 2024; Chen et al., 2024; Deitke et al., 2024], we train Aya Vision models in two steps: (1) Vision-Language Alignment and (2) Supervised Fine-tuning.

**Vision-Language Alignment:** In this step, we only train the vision-language connector by keeping both the vision encoder and the language model frozen. Freezing the language model and vision encoder allows for using a high learning rate to quickly map the image features to the input embedding space. We use a peak learning rate of  $10^{-4}$  and  $10^{-3}$  for Aya-Vision-8B and 32B models respectively. Additionally, we find that the 32B model requires longer training in this step due to the much larger connector size. While Aya-Vision-8B includes a 190M vision-language connector, the parameter size of the connector in 32B model is 428M. Therefore, we train the 8B model for 9.7k steps (1 epoch) and the 32B model for 19k steps (2 epochs). Similar to previous works [Liu et al., 2023c; Yue et al., 2024b] we use LLaVa-Pretrain<sup>12</sup> as the primary source of data in this step. However, since this data is English-only, we add a small fraction of the multilingual data generated by our data framework amounting to 14% of the total data seen during this step. All training details can be found in Table 6 in the appendix.

**Visual Instruction Fine-tuning:** In the instruction fine-tuning step (i.e., supervised fine-tuning with visual instructions), we train both the vision-language connector and the language model but keep the vision encoder frozen. We experiment with both full model fine-tuning and LoRA [Hu et al., 2022]. For both Aya-Vision-8B and Aya-Vision-32B, we use a batch size of 128 and train for 31k iterations with  $\mu$ P enabled on about 10M samples. The peak learning rates are set to  $10^{-4}$  and  $5 \times 10^{-4}$  respectively established via hyperparameter tuning. We utilize sequence packing to pack multiple samples into a single sequence of length 8192 for improved training efficiency. A breakdown of the SFT training data can be found in Figure 4 with detailed discussion presented in § 3.

## 5 Evaluation

### 5.1 Multilingual Multimodal Preference Evaluation

#### 5.1.1 Open-ended Multimodal Evaluation

While recent efforts have explored multilingual evaluation for multimodal LLMs [Changpinyo et al., 2022; Romero et al., 2024; Tang et al., 2024; Yue et al., 2024b], existing benchmarks still fall short of enabling robust, real-world evaluation. Most current suites focus on static, single-turn tasks with predefined answers, failing to capture the nuanced, open-ended, and dynamic nature of real-world user interactions. To address this, we introduce:

**AyaVisionBench**<sup>13</sup>, a benchmark explicitly designed to evaluate not only multimodal understanding and reasoning but also generation quality along human-centric dimensions, such as relevance,

---

<sup>12</sup><https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K>

<sup>13</sup><https://huggingface.co/datasets/CohereLabs/AyaVisionBench><table border="1">
<thead>
<tr>
<th>Dataset</th>
<th>Task</th>
<th>Metric</th>
<th># Languages</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="4"><b>Multimodal Academic Bench.</b></td>
</tr>
<tr>
<td>xMMMU [Yue et al., 2024b]</td>
<td>Multimodal Understanding</td>
<td>Accuracy</td>
<td>7</td>
</tr>
<tr>
<td>MaXM [Changpinyo et al., 2022]</td>
<td>VQA</td>
<td>Accuracy</td>
<td>7</td>
</tr>
<tr>
<td>CVQA [Romero et al., 2024]</td>
<td>VQA</td>
<td>Accuracy</td>
<td>31</td>
</tr>
<tr>
<td>MTVQA [Singh et al., 2019]</td>
<td>VQA</td>
<td>VQA Score</td>
<td>9</td>
</tr>
<tr>
<td>Kaleidoscope [Salazar et al., 2025]</td>
<td>VQA</td>
<td>Accuracy</td>
<td>18</td>
</tr>
<tr>
<td colspan="4"><b>Multimodal Open-Ended Bench.</b></td>
</tr>
<tr>
<td>AyaVisionBench</td>
<td>Multimodal Chat</td>
<td>Win-Rates</td>
<td>23</td>
</tr>
<tr>
<td>m-WildVision [Lu et al., 2024]</td>
<td>Multimodal Chat</td>
<td>Win-Rates</td>
<td>23</td>
</tr>
<tr>
<td>xChat [Yue et al., 2024b]</td>
<td>Multimodal Chat</td>
<td>LLM-Score</td>
<td>7</td>
</tr>
<tr>
<td colspan="4"><b>Text-only Bench.</b></td>
</tr>
<tr>
<td>m-ArenaHard [Dang et al., 2024]</td>
<td>Open-Ended Generations</td>
<td>Win-Rates</td>
<td>23</td>
</tr>
<tr>
<td>MGSM [Shi et al., 2022]</td>
<td>Math. Reasoning</td>
<td>Accuracy</td>
<td>6</td>
</tr>
<tr>
<td>Global MMLU-Lite [Singh et al., 2024a]</td>
<td>Language Understanding</td>
<td>Accuracy</td>
<td>15</td>
</tr>
<tr>
<td>FLORES [Guzmán et al., 2019]</td>
<td>Language Understanding</td>
<td>SpBLEU</td>
<td>23</td>
</tr>
<tr>
<td>IFEval [Zhou et al., 2023]</td>
<td>Instruction Following</td>
<td>Accuracy</td>
<td>1</td>
</tr>
</tbody>
</table>

Table 3: **Multilingual multimodal evaluation suite used in Aya Vision.** Our evaluation suite consists of multilingual multimodal benchmarks, multimodal open-ended benchmarks for preference evaluation, and finally, text-only benchmarks include open-ended, generative, and discriminative evaluation sets.

fluency, and engagement. AyaVisionBench targets the question: *How well can a multimodal model respond to complex, open-ended instructions across languages and modalities?*

AyaVisionBench spans 23 languages and comprises 135 image-question pairs per language, covering 9 diverse task categories: captioning, chart and figure understanding, identifying differences between two images, general visual question answering, OCR, document understanding, text transcription, mathematical or logical reasoning, textbook questions and converting screenshots to code. This multilingual, multi-task design supports comprehensive evaluation of cross-lingual multimodal understanding. Most samples include ground-truth responses for reference. Further construction details are available in Appendix A.1. The benchmark is publicly released for community use and broader evaluation.

**Multilingual WildVision (m-WildVision) and xChatBench** To complement AyaVisionBench, we release **m-WildVision**<sup>14</sup>, a multilingual extension of WildVision-Bench [Lu et al., 2024], featuring translated prompts in 22 languages. WildVision is curated from real-world user interactions and provides practical, context-rich evaluation scenarios. We also incorporate **xChatBench** [Yue et al., 2024b], which supports fine-grained, score-based assessments across 7 languages and various interaction types.

<sup>14</sup><https://huggingface.co/datasets/CohereLabs/m-WildVision>---

To evaluate model performance across all three benchmarks, we follow the VLM-as-a-judge protocol used in prior multilingual studies [Üstün et al., 2024; Dang et al., 2024], conducting pairwise comparisons between Aya Vision and baseline models. For scoring and preference ranking, we use **claude-3-7-sonnet-20250219** [Anthropic, 2025] as the multimodal judge. This choice is based on a comparative study using the translated Multimodal RewardBench [Yasunaga et al., 2025] across 8 languages,<sup>15</sup> where Claude-3-7-Sonnet outperformed GPT-4o [OpenAI, 2024] and Gemini-2.0-Flash [Team et al., 2024] by 6.4% and 25.8% respectively in preference ranking accuracy. Full details on the evaluation prompt are provided in Appendix J.

### 5.1.2 Academic Multilingual Multimodal Benchmarks

In addition to the preference-based open-ended multimodal evaluation, we evaluate Aya Vision on visual question answering and reasoning style benchmarks that require the generations to adhere to a prescribed format, such as multiple-choice style or short-form answers, for easy automated evaluation. Specifically, we use **xMMMU** [Yue et al., 2024b], **MaXM** [Changpinyo et al., 2022], **CVQA** [Romero et al., 2024], **MTVQA** [Tang et al., 2024] and **Kaleidoscope** [Salazar et al., 2025]. These benchmarks, covering a range of languages, measure multimodal understanding, knowledge, and reasoning capabilities of multimodal LLMs. The number of languages in each dataset is shown in Table 3, and details of these benchmarks are given in Appendix A.

## 5.2 Multilingual Text-Only Evaluations

As a final component of our multilingual evaluation suite, we evaluate Aya Vision models and baselines on various text-only benchmarks. This is important to reflect real-world deployment scenarios where models are used with both multimodal and text-only inputs. However, as shown in § 3, many vision-language models experience some degree of degradation in their text-only performance. Therefore, to evaluate models’ performance in various tasks, we include a set of representative text-only evaluations.

**Open-ended evaluation** Similar to AyaVisionBench, we use **m-ArenaHard** [Li et al., 2024; Dang et al., 2024] to evaluate and compare models’ performance in open-ended text generations in 23 languages.<sup>16</sup>

**Task-specific benchmarks** Additionally, we included **MGSM** [Shi et al., 2022], **Global MMLU-Lite** [Singh et al., 2024a], and **FLORES** [Guzmán et al., 2019] covering mathematical reasoning, multilingual language understanding, and machine translation, respectively. Each of these benchmarks includes a different set of languages, as listed in Table 3. For FLORES, we evaluate models’ translation performance from English to the target language ( $\text{En} \rightarrow \text{X}$ ) as translating from English is a harder task and a good indication for multilingual performance. Finally, we also include **IFEval** [Zhou et al., 2023], although it is English-only, as it measures instruction-following capabilities of models, which potentially impacts the performance in other multimodal and text-only benchmarks. Metrics for these benchmarks are given in Table 3, and additional details can be found in Appendix A.

---

<sup>15</sup>English (original), Arabic, Farsi, French, Hindi, Portuguese, Turkish, Vietnamese, Simplified Chinese.

<sup>16</sup>We use **gpt-4o-2024-11-20** [OpenAI, 2024] as the LLM-judge following Dang et al. [2024].Figure 6: **Aya-Vision-8B achieves best-in-class performance on preference evaluation.** Pair-wise win-rates on AyaVisionBench and m-WildVision [Lu et al., 2024] averaged across 23 languages. We compare Aya-Vision-8B with Gemini-Flash-8B, Llama-3.2-11B-Vision, Qwen-2.5-VL-7B, Pixtral-12B and Pangea-7B on AyaVisionBench (left) and m-WildVision (right). Language-specific breakdown for the results can be found in Table 9 & Table 10 in the Appendix.

<table border="1">
<thead>
<tr>
<th>Models / Evaluations</th>
<th>MaxM</th>
<th>xMMMU</th>
<th>CVQA</th>
<th>MTVQA</th>
<th>Kaleidoscope</th>
<th>xChat</th>
<th>avg</th>
</tr>
</thead>
<tbody>
<tr>
<td>Pangea-7B</td>
<td>51.27</td>
<td><u>44.00</u></td>
<td>60.53</td>
<td>18.32</td>
<td>29.46</td>
<td>32.21</td>
<td>39.30</td>
</tr>
<tr>
<td>Molmo-7B-D</td>
<td>44.16</td>
<td>37.87</td>
<td>58.53</td>
<td>16.89</td>
<td>36.42</td>
<td>23.36</td>
<td>36.21</td>
</tr>
<tr>
<td>Llama-3.2-11B-Vision</td>
<td>39.30</td>
<td>42.73</td>
<td>58.92</td>
<td>16.40</td>
<td>36.50</td>
<td>28.59</td>
<td>37.07</td>
</tr>
<tr>
<td>Pixtral-12B</td>
<td>44.43</td>
<td>42.27</td>
<td><u>63.54</u></td>
<td><u>19.81</u></td>
<td>36.08</td>
<td><b>64.50</b></td>
<td>45.11</td>
</tr>
<tr>
<td>Qwen-2.5-VL-7B</td>
<td><u>52.65</u></td>
<td><b>46.77</b></td>
<td><b>73.22</b></td>
<td><b>29.57</b></td>
<td><b>39.64</b></td>
<td>58.14</td>
<td><b>50.00</b></td>
</tr>
<tr>
<td><b>Aya-Vision-8B</b></td>
<td><b>58.21</b></td>
<td>39.94</td>
<td>61.86</td>
<td>19.33</td>
<td><u>38.62</u></td>
<td><u>58.64</u></td>
<td><u>46.16</u></td>
</tr>
<tr>
<td>Molmo-72B</td>
<td>55.62</td>
<td>51.53</td>
<td>72.77</td>
<td>18.66</td>
<td>50.34</td>
<td>45.43</td>
<td>49.06</td>
</tr>
<tr>
<td>Llama-3.2-90B-Vision</td>
<td><b>64.17</b></td>
<td><u>52.40</u></td>
<td><u>81.88</u></td>
<td><u>27.44</u></td>
<td><u>48.41</u></td>
<td>51.12</td>
<td><u>54.24</u></td>
</tr>
<tr>
<td>Qwen-2.5-VL-72B</td>
<td>56.42</td>
<td><b>61.74</b></td>
<td><b>82.10</b></td>
<td><b>31.92</b></td>
<td><b>55.02</b></td>
<td><b>71.13</b></td>
<td><b>59.72</b></td>
</tr>
<tr>
<td><b>Aya-Vision-32B</b></td>
<td><u>62.28</u></td>
<td>45.11</td>
<td>74.06</td>
<td>23.46</td>
<td>41.73</td>
<td><u>70.07</u></td>
<td>52.81</td>
</tr>
</tbody>
</table>

Table 4: **Evaluation on multilingual multimodal benchmarks for Aya-Vision-8B and Aya-Vision-32B together with the baselines.** For each benchmark, we include languages that are in the list of Aya Vision’s 23 languages. The full results on all available languages are given in Appendix K.

### 5.3 Baselines

We compare Aya Vision models against a range of state-of-the-art multimodal LLMs, both open- and closed-weight, to evaluate multilingual, multimodal, and text-only capabilities. We select models based on architecture, model size, base model family, and language coverage. The selected models cover a range of sizes (7B to 90B), base models (Llama-3.2, Qwen-2.5, Molmo), and languageFigure 7: **Aya-Vision-32B outperforms models more than double its size.** Pairwise win-rates on AyaVisionBench and m-WildVision [Lu et al., 2024] averaged across 23 languages. We compare Aya-Vision-32B with Llama-3.2-90B-Vision, Molmo-72B and Qwen-2.5-VL-72B on AyaVisionBench (left) and m-WildVision (right). Language-specific breakdown for the results can be found in Table 12 & Table 13 in the Appendix.

coverage (including both English and multilingual models). Our evaluation includes open-weight models (Pixtral [Agrawal et al., 2024], Molmo [Deitke et al., 2024], Qwen-2.5-VL [Bai et al., 2025] and Pangea [Yue et al., 2024b]) as well as the closed-weight (Gemini-Flash-1.5 [Team, 2024b]). For model families, Qwen, Molmo, and Llama, we report results across multiple sizes ranging from 7B to 90B parameters.

Among the baseline models, Pangea, Qwen, Pixtral, Llama, and Gemini explicitly report multilingual support. We also include Molmo, which does not explicitly claim to support multiple languages, however in practice, they are heavily used by multilingual users relative to some multilingual models like Pangea-7B [Yue et al., 2024b]. Hence, we think it is important to include. Furthermore, we also find that these models achieve considerable performance in many multilingual tasks, as shown in our evaluation.Figure 8: **Aya Vision** models are amongst the best models in text-only preference evaluation compared to models with much larger size. Pairwise win-rates for Aya-Vision-8B (left) and 32B (right) on m-ArenaHard [Li et al., 2024; Dang et al., 2024] averaged across 23 languages. Language-specific breakdown for the results can be found in Table 8 & Table 11 in the Appendix.

<table border="1">
<thead>
<tr>
<th>Models / Evaluations</th>
<th>G-MMLU (Lite)</th>
<th>MGSM</th>
<th>FLORES</th>
<th>IFEval</th>
<th>avg</th>
</tr>
</thead>
<tbody>
<tr>
<td>Pangea-7B</td>
<td>49.35</td>
<td>50.51</td>
<td>28.04</td>
<td>23.99</td>
<td>37.97</td>
</tr>
<tr>
<td>Molmo-7B-D</td>
<td>39.63</td>
<td>49.94</td>
<td>15.74</td>
<td>56.10</td>
<td>40.35</td>
</tr>
<tr>
<td>Llama-3.2-11B-Vision</td>
<td>60.75</td>
<td>72.84</td>
<td><u>31.84</u></td>
<td><b>83.43</b></td>
<td><u>62.22</u></td>
</tr>
<tr>
<td>Pixtral-12B</td>
<td><b>66.09</b></td>
<td><b>77.62</b></td>
<td>29.29</td>
<td>65.59</td>
<td>59.65</td>
</tr>
<tr>
<td>Qwen-2.5-VL-7B</td>
<td><u>64.82</u></td>
<td>60.90</td>
<td>27.98</td>
<td>72.46</td>
<td>56.54</td>
</tr>
<tr>
<td><b>Aya-Vision-8B</b></td>
<td>62.52</td>
<td><u>76.42</u></td>
<td><b>35.90</b></td>
<td><u>82.78</u></td>
<td><b>64.41</b></td>
</tr>
<tr>
<td>Molmo-72B</td>
<td>71.02</td>
<td><u>86.00</u></td>
<td>32.52</td>
<td>78.10</td>
<td>66.91</td>
</tr>
<tr>
<td>Llama-3.2-90B-Vision</td>
<td><u>77.46</u></td>
<td>66.67</td>
<td><b>38.25</b></td>
<td><u>88.14</u></td>
<td><u>67.63</u></td>
</tr>
<tr>
<td>Qwen-2.5-VL-72B</td>
<td><b>81.49</b></td>
<td><b>89.61</b></td>
<td>35.71</td>
<td><b>89.74</b></td>
<td><b>74.14</b></td>
</tr>
<tr>
<td><b>Aya-Vision-32B</b></td>
<td>63.58</td>
<td>79.46</td>
<td><u>37.79</u></td>
<td>78.50</td>
<td>64.83</td>
</tr>
</tbody>
</table>

Table 5: **Evaluation on multilingual text-only academic benchmarks for Aya-Vision-8B and Aya-Vision-32B together with the baselines.** For each benchmark, we include languages that are in the list of Aya Vision’s 23 languages. The full results on all languages are available in Appendix K.---

## 6 Results and Discussion

### 6.1 Multilingual Multimodal Open-Ended Performance

**Aya-Vision-8B achieves best-in-class performance in preference evaluation.** Figure 6 shows pairwise win-rates on AyaVisionBench and m-WildVision, averaged over 23 languages for Aya-Vision-8B against the other state-of-the-art multimodal LLMs. Overall, Aya-Vision-8B achieves the best-in-class performance, outperforming all the models by win-rate, ranging from 49.6% to 80.3%. We find that Aya-Vision-8B achieves slightly higher win-rates on m-WildVision compared to AyaVisionBench – 6% on average, potentially due to the challenging characteristic of AyaVisionBench – higher tie rates also indicate failure cases for both models in the comparison. Aya-Vision-8B outperforms both Qwen-2.5-VL-7B and Pixtral-12B by 54.8% win-rate averaged across the two datasets, even though Pixtral-12B is a larger model. Additionally, Aya-Vision-8B also outperforms strong proprietary models like Gemini-Flash1.5-8B with a win-rate of 60.3% on average. Notably, Aya-Vision-8B outperforms Pangea-7B by a significant margin (71.7% win-rate) even though Pangea includes a large proportion of multilingual data in its training.

Aya-Vision-8B also outperforms Pangea-7B across all 23 languages – ranging from 56% in English to 83.6% in Greek. Given that the “curse of multilinguality” leads to drop in per-language performance as the number of languages covered increases, Aya-Vision-8B is still extremely competitive with Molmo-7B (specifically optimized for English) with a win-rate of 48.3% in English while outperforming it over the other 22 languages with an average win-rate of 80%.

Finally, in addition to AyaVisionBench and m-WildVision, Aya-Vision-8B outperforms all models in the same parameter class on xChatBench as shown in Table 4. Notably, Aya-Vision-8B not only achieves a significant margin against models like Pangea-7B, Molmo-7B-D, and Llama-3.2-11B, but also outperforms much larger models such as Molmo-72B and Llama-3.2-90B by 28.5% and 14.7% relative increase, validating its strong conversational ability.

**Aya Vision outperforms far larger models.** While scaling model size has demonstrated tangible gains in model performance [Kaplan et al., 2020]; complementing this with careful data and model optimization techniques yields significant efficiency gains. Such optimizations improve the underlying scaling dynamics, reducing the parameter count needed for equivalent performance [Hooker, 2024]. Figure 7 shows pairwise win-rates averaged over 23 languages for Aya-Vision-32B on AyaVisionBench and m-WildVision. Across both AyaVisionBench and m-WildVision, Aya-Vision-32B consistently outperforms models over  $2\times$  larger, such as Molmo-72B, Qwen-2.5-VL-72B, and Llama-3.2-90B-Vision by win-rates ranging from 48.5% to 73%. Notably, Aya-Vision-32B outperforms Llama-3.2-90B-Vision on AyaVisionBench and m-WildVision by 65.9% and 73% win-rates, respectively. The closest competitor to Aya-Vision-32B is Qwen-2.5-VL-72B, where Aya-Vision-32B outperforms Qwen-2.5-VL-72B by 50.8% win-rate on average across both datasets. This showcases our critical focus on efficiency by achieving more using less compute. This also enables greater support for the research community, who often have more limited access to compute resources.

### 6.2 Multilingual Multimodal Academic Benchmarks

**Aya Vision models achieve competitive performance in multiple-choice or short-form academic benchmarks.** Aya Vision models are optimized for open-ended real-world usage rather---

than academic benchmarks featuring multiple-choice or short-form answers. These benchmarks, typically designed as visual question answering tasks, tend to prioritize constrained, static evaluation formats and often fail to capture the full generative capabilities of modern MLLMs. As noted in prior work [Muennighoff et al., 2022; Agrawal et al., 2024; Deitke et al., 2024; Üstün et al., 2024], performance on such benchmarks correlates weakly with real-world open-ended tasks. Nonetheless, Aya Vision models demonstrate strong performance across these evaluations. Results are reported in Table 4.

Notably, on MaxM, a short-form VQA benchmark, Aya-Vision-8B outperforms all models in its parameter class, including larger ones like Pixtral-12B and LLaMA-3.2-11B-Vision. Similarly, on Kaleidoscope, it performs competitively with Qwen-2.5-VL-7B and surpasses all other baselines.

Finally, our 32B model Aya Vision model exhibits competitive performance on academic benchmarks against models more than  $2\times$  its size. Aya-Vision-32B outperforms Molmo-72B on all benchmarks except xMMMU, and closely matches Llama-3.2-90B-Vision, despite being nearly  $3\times$  smaller.

### 6.3 Text-Only Performance

**Aya Vision models punch above their size in text-only preference evaluation.** A key concern with multimodal models is that introducing vision can degrade existing text performance. Hence, we evaluate the final overall performance in text performance. Figure 8 shows win-rates for Aya Vision models against the baselines on m-ArenaHard dataset, averaged over 23 languages. At 8B parameter scale, Aya-Vision-8B outperforms all the models except Gemini-Flash1.5-8B, which is a proprietary model. Compared to models that are larger than ours, while Aya-Vision-8B beats Llama-3.2-11B-Vision with a 63.4% win-rate, it is outperformed by Pixtral-12B with 44.0% win-rate. For the larger model comparison, Aya-Vision-32B outperforms Molmo-72B and Qwen-2.5-VL-72B by win-rates of 77.3% and 50.9% respectively. Our 32B model is competitive with Llama-3.2-90B-Vision with a 43.2% win-rate. Considered together with superior multimodal win-rates (Figure 6 & Figure 7), these results show the relative preservation in text performance while adding best-in-class multimodal abilities.

**Aya Vision recovers open-ended text-only performance in a significantly higher degree than the baselines.** As an additional perspective on text-only performance, Figure 5 compares the text-only win-rates on mArenaHard for Aya-Vision-8B, Pangea-7B, Qwen-2.5-7B, and Molmo-7B compared to the LLMs they were initialized from. Here, Aya-Vision-8B with cross-modal merging makes significant strides towards much closer performance to the initial LLM – limiting the degradation to within 5.9%. This degradation, however, is significantly higher in the other models evaluated, 16.4% for Pangea, 22.1% for Qwen-2.5, and 44.1% for Molmo compared to their initial LLMs. These results highlight the benefits of our cross-modal merging framework.

**It is easier to recover text-only performance in academic benchmarks compared to open-ended evaluation.** As we show in § 3, maintaining the base LLM’s text-only performance in academic benchmarks is much easier than preference evaluation due to the nature of these benchmarks. Hence, the performance of similar-sized models is closer in these benchmarks. At 8B parameter scale, Aya-Vision-8B achieves the best average performance across text-only benchmarks of 64.41%, where it outperforms all models in FLORES (En→X, 23 languages) and reaches the second-best performance in both MGSM and IFEval, after Pixtral-12B and Llama-3.2-11B-VisionFigure 9: **Impact of cross-modal merging across various merge ratios.** Multimodal and text win-rates are calculated against Pangea-7B on AyaVisionBench and m-ArenaHard respectively over 7 languages. Multimodal academic benchmark is an average of CVQA and xMMMU; Text-Only academic benchmarks are averaged over IFEval, MGSMLU and MMMLU (subset).

respectively. Notably, both models are much larger than our 8B model. For Aya-Vision-32B, our model achieves second-best performance for FLORES, but falls behind other models on other tasks. We relate this to the original performance of base LLMs in these benchmarks, where recovery is relatively straightforward. It is important to note that the models compared to Aya-Vision-32B are over  $2\times$  its size (72B and 90B models). Overall, we observe that both multimodal and text-only academic benchmark results have poor alignment with their open-ended generation counterparts; as demonstrated in prior works [Muennighoff et al., 2022; Üstün et al., 2024] due to their rigid metrics emphasizing precise format compliance at the expense of semantic correctness and quality of generations.

## 7 Key Ablations and Discussion

To isolate the impact of our design choices, we perform a set of controlled ablations focusing on – (1) cross-modal model merging, (2) comparison with the addition of text-only data, (3) multilingual data percentage during SFT, (4) the vision encoder, and (5) comparison of full model fine-tuning with low-rank adaptation, all at the 8B parameter scale. In each of these ablations, we only vary a single variable of interest, while keeping the rest of the experimental setup fixed. To evaluate each ablation, we use multimodal win-rates on AyaVisionBench and text win-rates on mArena-Hard using a subset of languages<sup>17</sup> against Pangea-7B. In addition, we also report scores on various academic benchmarks based on the ablation.

### 7.1 Model Merging Improves Multilingual Performance Across Tasks and Modalities

To understand the impact of our cross-modal model merging as the merging ratio changes, we ablate the interpolation weight  $\alpha$  in Equation (1) for the multimodal LLM, and evaluate the resulting merged multimodal LLMs. An  $\alpha$  of 0 corresponds to purely the text-only model whereas an  $\alpha$  of 1 corresponds to just the post-multimodal training model. In addition to the win-rates for both multimodal and text-only, we report the average of CVQA and xMMMU for academic vision

<sup>17</sup>English, French, Hindi, Arabic, Turkish, Japanese, Chinesebenchmarks and IFEval, MMMLU (subset), and MGSM for text-only academic benchmarks.

While our original motivation for model merging was retention of performance on text-only multilingual benchmarks, Figure 9 (left) shows that our novel cross-modal merging recipe additionally boosts multilingual vision win-rates as the interpolation weight for text-only model increases. Below 0.6 multimodal interpolation weight, the text-only win-rates keep climbing; however, the vision win-rates saturate. For academic benchmarks, we again observe a similar trend – as the ratio of the text-only model increases, text-only benchmarks rapidly increase until 0.5, following which the gains are minimal. Interestingly, even academic multimodal benchmarks see a minor gain due to model merging. Based on these results, we chose 0.4 as the merging ratio for both our 8B and 32B models.

## 7.2 Model merging is more effective than adding *seen* text data for cross-modal transfer

An alternate approach to recover performance on text-only tasks is to include a certain percentage of text-only data in the training mixture. To understand the role of text-only data on multimodal and text-only win-rates and specifically compare it with our cross-modal merging approach, we train 3 variants with varying proportions of text data – 0%, 10%, and 30%. For the variants with text-data added, we evaluate the final checkpoints without merging, and compare with the model where our merging recipe is applied on the variant with 0% text-data. Figure 10 shows the results of these experiments.

While increasing the amount of text-only data improves the quality of generations for textual prompts as indicated by win-rates going from 50.2% to 74.8%; these gains do not translate to multimodal prompts. In fact, as seen in Figure 10 these win-rates are substantially lower than those obtained by training on purely multimodal data followed by merging with a weight of 0.4. Additionally, increasing the amount of text data added from 10% to 30% leads to a slight decrease in the multimodal win-rates due to increasing share of model capacity being used for text-only modeling. This highlights the simplicity and efficacy of our model merging framework at cross-modal transfer of capabilities.

Figure 10: **Modal merging is an efficient way to enable cross-modal transfer.** Multimodal and text-only win-rates on AyaVisionBench and m-ArenaHard against Pangea-7B. We increase the amount of text-only mixture in SFT and compare to cross-modal merging (dashed line).Figure 11: **A balanced data mixture is essential for multilingual multimodal performance.** Multimodal and text win-rates are calculated against Pangea-7B on AyaVisionBench and m-ArenaHard respectively over 7 languages. Multimodal academic benchmark is an average of CVQA and xMMMU; Text-Only academic benchmarks are averaged over IFEval, MGSM and MMMLU (subset).

### 7.3 Data Improvements has the Highest Impact on the Quality of Generations

Our data generation framework has a strong emphasis on the quality but can we quantify the importance of the data improvement process? To answer this question, we train 2 variants – (1) with only existing open-source data, (2) with the data mixture proposed in § 3 – holding the amount of data and iterations during training fixed; and measure the multimodal win-rates. Please note that no merging is performed here to allow for a cleaner comparison. Figure 12 shows the impact of synthetic annotations on the win-rates. Compared to variant (1) trained purely on original task-specific data, our data improvements lead to the largest jump in win-rates – 17% amongst our various interventions.

Figure 12: **Impact of various interventions.** Step-by-step improvements in Aya Vision 8B’s pairwise win-rates against Pangea-7B.

This underscores the importance of fluent, detailed and diverse completions in the training data mixture towards building a strong conversational multimodal model. Additionally, when paired with cross-modal merging the total improvement increases to nearly 30%.

### 7.4 A Balanced Data Mixture is Essential for Multilingual Multimodal Performance

An important question in building a multilingual multimodal model is – *What is the right ratio of multilingual data in the training mixture?*

To answer this question, we train 3 variants with varying proportions of multilingual multimodal data – 17.5%, 35%, and 67%, which is uniformly distributed across 22 languages (except English). We compare these variants using preference evaluation (win-rates), and a subset of multimodal andtext-only academic benchmarks. Note that we merge each trained checkpoint with the text-only model with the same interpolation factor ( $\alpha$ ) to make it consistent with our final recipe. Figure 11 shows the results.

**Balanced multilingual data leverages cross-lingual transfer from English for best performance across modalities and languages.** We observe that increasing the ratio of multilingual multimodal data from 35% to 67% leads to degradation in the quality of generations – reducing the win-rates from 71.4% to 68.7%, and also hurts multimodal academic benchmarks, emphasizing the importance of the balance between English and multilingual data. Given the scarcity of high-quality multilingual multimodal data, upsampling this bucket requires repeating the data multiple times, limiting its benefit in multilingual multimodal performance. Additionally, a sufficient percentage of the more diverse English data is crucial for cross-lingual transfer. Therefore, we use 35% of multilingual data in our final recipe, leaving 65% for a diverse set of English datasets, which includes selected original datasets (34%), and a high-quality synthetically re-annotated dataset (31%) as presented in Section 3.

## 7.5 Low Rank Finetuning is Comparable to Full Finetuning

Low-rank training (LoRA) is an extremely performant method to reduce the hardware footprint during training for improved efficiency. LoRA drastically reduces the number of trainable parameters and optimizer states to be stored in the accelerator memory [Zadouri et al., 2023]. Furthermore, freezing the LLM and constraining the rank of updates has the potential to prevent catastrophic forgetting on text-only prompts. To understand the impact of the rank of training updates during the SFT stage, we train 2 variants on the same data – (1) trained with LoRA (rank = 256,  $\alpha$  = 512) [Hu et al., 2022] while (2) is trained with full finetuning (all network weights are updated). Once both the models are trained, we merge the multimodal updates to the text-only language model with a weight ( $\alpha$ ) of 0.5. Finally, we evaluate both variants on multimodal and text win-rates; and academic benchmarks like CVQA and xMMMU. Figure 13 shows the results on all the above tasks.

**Figure 13: Impact of training with LoRA vs. Full-Finetuning.** We compare vision win-rates (left) and text-only win-rates (center) against Pangea-7B averaged across 7 languages. We also report the average of CVQA and xMMMU (right).

On academic tasks like CVQA and xMMMU, we observe that both variants perform equally well, 51.2 vs 51.0 average accuracy for LoRA and full model fine-tuning, respectively. On multimodal win-rate evaluations, both models are extremely close – with 68.4% and 67.2% win-rates for the LoRA and fully-finetuned variants respectively. Any improvement exhibited by the LoRA variant on win-rates is well within the noise-margin. On text-only win-rates, the LoRA variant is 3.4% better than full-finetuning which can be attributed to the frozen LLM backbone during training and the amenability of LoRA model to merging due to the shared optimization trajectory.---

## 8 Related Work

**Visual Instruction Tuning** Visual instruction tuning [Liu et al., 2023c; Chen et al., 2023; Liu et al., 2024; Chen et al., 2024; Agrawal et al., 2024; Wang et al., 2024b; Deitke et al., 2024; Bai et al., 2025] combines a pre-trained vision encoder [Radford et al., 2021; Zhai et al., 2023; Chen et al., 2024; Tschannen et al., 2025] with an off-the-shelf large language model via a dedicated vision–language connector. This process extends the LLM’s text capabilities into the visual domain while retaining its desirable attributes— such as in-context learning, reasoning, and instruction following. As a result, visual instruction tuning has emerged as a highly effective method to achieve state-of-the-art performance on a wide range of tasks – even outperforming certain proprietary models.

**Multilingual Multimodal Models** Initial works on multilingual multimodal models [Ni et al., 2021; Jain et al., 2021; Zeng et al., 2023] focused on learning robust, universal representations for retrieval tasks across modalities. However, these models require further downstream training to be used as generative models. On the other hand, [Geigle et al., 2023; Chen et al., 2023; Yue et al., 2024b] perform large-scale multilingual multi-task fine-tuning to enable multilingual understanding and generation. However, they focus only on vision-language academic benchmarks which are reference based – focusing on exact matches rather than free-form holistic evaluations of the generations.

**Multilingual Multimodal Evaluations** Multilingual multimodal evaluation benchmarks have traditionally focused on visual question answering (VQA) tasks, where the model-generated response must exactly match a human-provided reference answer [Changpinyo et al., 2022; Romero et al., 2024; Tang et al., 2024]. This approach often penalizes responses that are semantically correct but differ syntactically from the reference [Agrawal et al., 2024]. To address these limitations, recent work [Yue et al., 2024b; Maaz et al., 2024] has proposed multilingual multimodal chat benchmarks. Instead of relying solely on exact matches, these benchmarks evaluate free-form responses by employing a Vision-Language model as an adjudicator—either by scoring responses against a detailed rubric or by selecting the superior generation from a pair of outputs.

**Multimodal Merging** Recent work by Zhu et al. [2025] introduces REMEDY, a method for merging VLM weights – including the connector layer – after low-rank fine-tuning on various VLM tasks. However, REMEDY does not address the merging of weights that have been trained for different modalities. In a closely related concurrent work, Li et al. [2025] merge a text-only reward model with a vision-language model with the goal to specifically transfer the reward modeling capabilities from the text-based reward model to build a multimodal reward model.

## 9 Conclusion

In this work, we introduced Aya Vision, a family of multilingual vision-language models (8B and 32B) designed to improve multimodal understanding across 23 languages. Addressing key challenges in this space, we propose a scalable synthetic annotation framework to overcome multilingual data scarcity, and a training-free model merging approach to preserve text-only performance during multimodal training. Our models outperform existing open-weight baselines and are supported by AyaVisionBench, a benchmark tailored for evaluating generative multilingual multimodal systems. By releasing our models and evaluation suite, we aim to lower barriers for research in this area and support continued progress toward more inclusive and linguistically diverse multimodal AI.---

## 10 Acknowledgements

Thank you to all our colleagues across Cohere who jumped in to help test Aya Vision in their language: Ivan Zhang, Irem Ergün, Eddie Kim, Hemant Jain, Wei-Yin Ko, Adrian Morisot, Rod Hajjar, Gokce Keskin, Trushant Kalyanpur, Julia Kreutzer, Olivia Lasche, Dennis Aumiller, Felipe Cruz Salinas, Alice Schoenauer Sebag, Dwarak Talupuru, Diana Abagyan, Ammar Khairi, Huey Sun, Varun Kumethi, Viraat Aryabumi, Sungjin Hong, Trent Fowlers, Lidiya Murakhovska, Aidan Peppin, Jay Alammar, Samuel Cahyawijaya, Brittawnya Prince, Daniel D’souza and Vivek Muppalla.

And to the members of our Open Science Community who shared their expertise and insights: Ahmad Anis, Amir Nuriyev, Bronson Bakunga,, Daniel Laurin, Danylo Boiko, David Cairuz, Dina Kliuchareva, Dominik Krzemiński, Erika Watanabe, Fernanda Guerriero Antunes, Gimei Alex, Jie Gao, Joana da Matta, Joseph Pollack, Karthik Kanjula, Kenny Rebelo, Kentaro Kojima, Kian Kyers, Lana Ludmila, Leticia Mie Otani, Louisa Chang, Marek Suppa, Mayuko Koizumi, Mei E., Mei Hirata, Micol Altomare, Nicole Mak, Ning Sun, Rami Rao, Reuben Fernandes, Reza Rob, Selina Tong, Shayekh Bin Islam, Shirley Au, Silvia Fernandez, Sree Harsha Nelaturu, Tai Guratti, Teresa Shiho Waddell, Thiago Correia, Xuelong An Wang, and Yanny Li.

The authors would also like to thank Fraser Greenlee for his contributions during the early stages of this project.

## References

Aakanksha, Arash Ahmadian, Seraphina Goldfarb-Tarrant, Beyza Ermis, Marzieh Fadaee, and Sara Hooker. Mix data or merge models? optimizing for diverse multi-task learning, 2024a. URL <https://arxiv.org/abs/2410.10801>.

Arash Aakanksha, Ahmadian, Beyza Ermis, Seraphina Goldfarb-Tarrant, Julia Kreutzer, Marzieh Fadaee, Sara Hooker, et al. The multilingual alignment prism: Aligning global and local preferences to reduce harm. *arXiv preprint arXiv:2406.18682*, 2024b.

Manoj Acharya, Kushal Kafe, and Christopher Kanan. Tallyqa: Answering complex counting questions. In *AAAI*, 2019.

Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Baptiste Bout, Devendra Chaplot, Jessica Chudnovsky, Diogo Costa, Baudouin De Monicault, Saurabh Garg, Theophile Gervet, et al. Pixtral 12b. *arXiv preprint arXiv:2410.07073*, 2024.

Anthropic. Claude 3.7 sonnet system card. <https://assets.anthropic.com/m/785e231869ea8b3b/original/claude-3-7-sonnet-system-card.pdf>, February 2025. Accessed: 2025-04-17.

Viraat Aryabumi, John Dang, Dwarak Talupuru, Saurabh Dash, David Cairuz, Hangyu Lin, Bharat Venkitesh, Madeline Smith, Jon Ander Campos, Yi Chern Tan, Kelly Marchisio, Max Bartolo, Sebastian Ruder, Acyr Locatelli, Julia Kreutzer, Nick Frosst, Aidan Gomez, Phil Blunsom, Marzieh Fadaee, Ahmet Üstün, and Sara Hooker. Aya 23: Open weight releases to further multilingual progress, 2024a. URL <https://arxiv.org/abs/2405.15032>.

Viraat Aryabumi, John Dang, Dwarak Talupuru, Saurabh Dash, David Cairuz, Hangyu Lin, Bharat---

Venkitesh, Madeline Smith, Jon Ander Campos, Yi Chern Tan, et al. Aya 23: Open weight releases to further multilingual progress. *arXiv preprint arXiv:2405.15032*, 2024b.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. *arXiv preprint arXiv:2308.12966*, 2023.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. URL <https://arxiv.org/abs/2502.13923>.

Assaf Ben-Kish, Moran Yanuka, Morris Alper, Raja Giryes, and Hadar Averbuch-Elor. Mocha: Multi-objective reinforcement mitigating caption hallucinations. *arXiv preprint arXiv:2312.03631*, 2, 2023.

Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohtsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. *arXiv preprint arXiv:2407.07726*, 2024.

Ali Furkan Biten, Rubèn Tito, Andrés Mafla, Lluís Gómez, Marçal Rusiñol, C.V. Jawahar, Ernest Valveny, and Dimosthenis Karatzas. Scene text visual question answering. In *2019 IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 4290–4300, 2019. doi: 10.1109/ICCV.2019.00439.

Yuri Bizzoni, Tom S Juzek, Cristina España-Bonet, Koel Dutta Chowdhury, Josef van Genabith, and Elke Teich. How human is machine translationese? comparing human and machine translations of text and speech. In *Proceedings of the 17th International conference on spoken language translation*, pp. 280–290, 2020.

Soravit Changpinyo, Linting Xue, Michal Yarom, Ashish V Thapliyal, Idan Szpektor, Julien Amelot, Xi Chen, and Radu Soricut. Maxm: Towards multilingual visual question answering. *arXiv preprint arXiv:2209.05401*, 2022.

Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Sebastian Goodman, Xiao Wang, Yi Tay, et al. Pali-x: On scaling up a multilingual vision and language model. *arXiv preprint arXiv:2305.18565*, 2023.

Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. *Science China Information Sciences*, 67 (12):220101, 2024.

Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, et al. Finqa: A dataset of numerical reasoning over financial data. *arXiv preprint arXiv:2109.00122*, 2021.

Team Cohere, Aakanksha, Arash Ahmadian, Marwan Ahmed, Jay Alammar, Yazeed Alnumay, Sophia Althammer, Arkady Arkhangorodsky, Viraat Aryabumi, Dennis Aumiller, Raphaël Avalos, Zahara Aviv, Sammie Bae, Saurabh Baji, Alexandre Barbet, Max Bartolo, Björn Bebensee, Neeral Beladia, Walter Beller-Morales, Alexandre Bérard, Andrew Berneshawi, Anna Bialas,---

Phil Blunsom, Matt Bobkin, Adi Bongale, Sam Braun, Maxime Brunet, Samuel Cahyawijaya, David Cairuz, Jon Ander Campos, Cassie Cao, Kris Cao, Roman Castagné, Julián Cendrero, Leila Chan Currie, Yash Chandak, Diane Chang, Giannis Chatziveroglou, Hongyu Chen, Claire Cheng, Alexis Chevalier, Justin T. Chiu, Eugene Cho, Eugene Choi, Eujeong Choi, Tim Chung, Volkan Cirik, Ana Cismaru, Pierre Clavier, Henry Conklin, Lucas Crawhall-Stein, Devon Crouse, Andres Felipe Cruz-Salinas, Ben Cyrus, Daniel D’souza, Hugo Dalla-Torre, John Dang, William Darling, Omar Darwiche Domingues, Saurabh Dash, Antoine Debugne, Théo Dehaze, Shaan Desai, Joan Devassy, Rishit Dholakia, Kyle Duffy, Ali Edalati, Ace Eldeib, Abdullah Elkady, Sarah Elsharkawy, Irem Ergün, Beyza Ermis, Marzieh Fadaee, Boyu Fan, Lucas Fayoux, Yanis Flet-Berliac, Nick Frosst, Matthias Gallé, Wojciech Galuba, Utsav Garg, Matthieu Geist, Mohammad Gheshlaghi Azar, Seraphina Goldfarb-Tarrant, Tomas Goldsack, Aidan Gomez, Victor Machado Gonzaga, Nithya Govindarajan, Manoj Govindassamy, Nathan Grinsztajn, Nikolas Gritsch, Patrick Gu, Shangmin Guo, Kilian Haefeli, Rod Hajjar, Tim Hawes, Jingyi He, Sebastian Hofstätter, Sungjin Hong, Sara Hooker, Tom Hosking, Stephanie Howe, Eric Hu, Renjie Huang, Hemant Jain, Ritika Jain, Nick Jakobi, Madeline Jenkins, JJ Jordan, Dhruti Joshi, Jason Jung, Trushant Kalyanpur, Siddhartha Rao Kamalakara, Julia Kedrzycki, Gokce Keskin, Edward Kim, Joon Kim, Wei-Yin Ko, Tom Kocmi, Michael Kozakov, Wojciech Kryściński, Arnav Kumar Jain, Komal Kumar Teru, Sander Land, Michael Lasby, Olivia Lasche, Justin Lee, Patrick Lewis, Jeffrey Li, Jonathan Li, Hangyu Lin, Acyr Locatelli, Kevin Luong, Raymond Ma, Lukas Mach, Marina Machado, Joanne Magbitang, Brenda Malacara Lopez, Aryan Mann, Kelly Marchisio, Olivia Markham, Alexandre Matton, Alex McKinney, Dominic McLoughlin, Jozef Mokry, Adrien Morisot, Autumn Moulder, Harry Moynehan, Maximilian Mozes, Vivek Muppalla, Lidiya Murakhovska, Hemangani Nagarajan, Alekhya Nandula, Hisham Nasir, Shauna Nehra, Josh Netto-Rosen, Daniel Ohashi, James Owers-Bardsley, Jason Ozuzu, Dennis Padilla, Gloria Park, Sam Passaglia, Jeremy Pekmez, Laura Penstone, Aleksandra Piktus, Case Ploeg, Andrew Poulton, Youran Qi, Shubha Raghvendra, Miguel Ramos, Ekagra Ranjan, Pierre Richemond, Cécile Robert-Michon, Aurélien Rodriguez, Sudip Roy, Laura Ruis, Louise Rust, Anubhav Sachan, Alejandro Salamanca, Kailash Karthik Saravankumar, Isha Satyakam, Alice Schoenauer Sebag, Priyanka Sen, Sholeh Sepehri, Preethi Seshadri, Ye Shen, Tom Sherborne, Sylvie Chang Shi, Sanal Shivaprasad, Vladyslav Shmyhlo, Anirudh Shrinivason, Inna Shteinbuk, Amir Shukayev, Mathieu Simard, Ella Snyder, Ava Spataru, Victoria Spooner, Trisha Starostina, Florian Strub, Yixuan Su, Jimin Sun, Dwarak Talupuru, Eugene Tarassov, Elena Tommasone, Jennifer Tracey, Billy Trend, Evren Tumer, Ahmet Üstün, Bharat Venkitesh, David Venuto, Pat Verga, Maxime Voisin, Alex Wang, Donglu Wang, Shijian Wang, Edmond Wen, Naomi White, Jesse Willman, Marysia Winkels, Chen Xia, Jessica Xie, Minjie Xu, Bowen Yang, Tan Yi-Chern, Ivan Zhang, Zhenyu Zhao, and Zhoujie Zhao. Command a: An enterprise-ready large language model, 2025. URL <https://arxiv.org/abs/2504.00698>.

Marta R Costa-Jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, et al. No language left behind: Scaling human-centered machine translation. *arXiv preprint arXiv:2207.04672*, 2022.

Wenliang Dai, Nayeon Lee, Boxin Wang, Zhuolin Yang, Zihan Liu, Jon Barker, Tuomas Rintamäki, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nvlm: Open frontier-class multimodal llms. *arXiv preprint arXiv:2409.11402*, 2024.

John Dang, Shivalika Singh, Daniel D’souza, Arash Ahmadian, Alejandro Salamanca, Madeline Smith, Aidan Peppin, Sungjin Hong, Manoj Govindassamy, Terrence Zhao, et al. Aya---

expanse: Combining research breakthroughs for a new multilingual frontier. *arXiv preprint arXiv:2412.04261*, 2024.

Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. *arXiv preprint arXiv:2409.17146*, 2024.

Yihao Ding, Siwen Luo, Hyunsuk Chung, and Soyeon Caren Han. Vqa: A new dataset for real-world vqa on pdf documents. In *Joint European Conference on Machine Learning and Knowledge Discovery in Databases*, pp. 585–601. Springer, 2023.

Beyza Ermis, Luiza Pozzobon, Sara Hooker, and Patrick Lewis. From one to many: Expanding the scope of toxicity mitigation in language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), *Findings of the Association for Computational Linguistics: ACL 2024*, pp. 15041–15058, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.893. URL <https://aclanthology.org/2024.findings-acl.893/>.

Yunhao Fang, Ligeng Zhu, Yao Lu, Yan Wang, Pavlo Molchanov, Jan Kautz, Jang Hyun Cho, Marco Pavone, Song Han, and Hongxu Yin. Vila<sup>2</sup>: Vila augmented vila, 2024. URL <https://arxiv.org/abs/2407.17453>.

Jonathan Frankle, Gintare Karolina Dziugaite, Daniel Roy, and Michael Carbin. Linear mode connectivity and the lottery ticket hypothesis. In *International Conference on Machine Learning*, pp. 3259–3269. PMLR, 2020.

Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruva Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. *Advances in Neural Information Processing Systems*, 36:27092–27112, 2023.

Gregor Geigle, Abhay Jain, Radu Timofte, and Goran Glavaš. mblip: Efficient bootstrapping of multilingual vision-llms. *arXiv preprint arXiv:2307.06930*, 2023.

Charles Goddard, Shamane Siriwardhana, Malikeh Ehghaghi, Luke Meyers, Vladimir Karpukhin, Brian Benedict, Mark McQuade, and Jacob Solawetz. Arcee’s mergekit: A toolkit for merging large language models. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track*, pp. 477–485, 2024.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 6904–6913, 2017.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.

Anisha Gunjal, Jihan Yin, and Erhan Bas. Detecting and preventing hallucinations in large vision language models. *arXiv preprint arXiv:2308.06394*, 2023.---

Jarvis Guo, Tuney Zheng, Yuelin Bai, Bo Li, Yubo Wang, King Zhu, Yizhi Li, Graham Neubig, Wenhui Chen, and Xiang Yue. Mammoth-v1: Eliciting multimodal reasoning with instruction tuning at scale. *arXiv preprint arXiv:2412.05237*, 2024.

Francisco Guzmán, Peng-Jen Chen, Myle Ott, Juan Pino, Guillaume Lample, Philipp Koehn, Vishrav Chaudhary, and Marc’Aurelio Ranzato. The flores evaluation datasets for low-resource machine translation: Nepali-english and sinhala-english. *arXiv preprint arXiv:1902.01382*, 2019.

Kai Hartung, Aaricia Herygers, Shubham Vijay Kurlekar, Khabbab Zakaria, Taylan Volkan, Sören Gröttrup, and Munir Georges. Measuring sentiment bias in machine translation. In *International Conference on Text, Speech, and Dialogue*, pp. 82–93. Springer, 2023.

Amr Hendy, Mohamed Abdelrehim, Amr Sharaf, Vikas Raunak, Mohamed Gabr, Hitokazu Matsushita, Young Jin Kim, Mohamed Afify, and Hany Hassan Awadalla. How good are gpt models at machine translation? a comprehensive evaluation. *arXiv preprint arXiv:2302.09210*, 2023.

Sara Hooker. On the limitations of compute thresholds as a governance strategy, 2024. URL <https://arxiv.org/abs/2407.05694>.

Yu-Chung Hsiao, Fedir Zubach, Gilles Baechler, Victor Carbune, Jason Lin, Maria Wang, Srinivas Sunkara, Yun Zhu, and Jindong Chen. Screenqa: Large-scale question-answer pairs over mobile app screenshots. *arXiv preprint arXiv:2209.08199*, 2022.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. *ICLR*, 1(2):3, 2022.

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. *arXiv preprint arXiv:2212.04089*, 2022.

Pavel Izmailov, Dmitrii Podoprikin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. Averaging weights leads to wider optima and better generalization. *arXiv preprint arXiv:1803.05407*, 2018.

Aashi Jain, Mandy Guo, Krishna Srinivasan, Ting Chen, Sneha Kudugunta, Chao Jia, Yinfei Yang, and Jason Baldridge. Mural: multimodal, multitask retrieval across languages. *arXiv preprint arXiv:2109.05125*, 2021.

Kushal Kafle, Brian Price, Scott Cohen, and Christopher Kanan. Dvqa: Understanding data visualizations via question answering. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 5648–5656, 2018.

Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkinson, Ákos Kádár, Adam Trischler, and Yoshua Bengio. Figureqa: An annotated figure dataset for visual reasoning. *arXiv preprint arXiv:1710.07300*, 2017.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*, 2020.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In *Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14*, pp. 235–251. Springer, 2016.

