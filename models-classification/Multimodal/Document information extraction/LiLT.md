# LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding

Jiapeng Wang<sup>1</sup>      Lianwen Jin<sup>\*1,3,4</sup>      Kai Ding<sup>\*2,3</sup>

<sup>1</sup>South China University of Technology, Guangzhou, China

<sup>2</sup>IntSig Information Co., Ltd, Shanghai, China

<sup>3</sup>INTSIG-SCUT Joint Laboratory of Document Recognition and Understanding, China

<sup>4</sup>Peng Cheng Laboratory, Shenzhen, China

<sup>1</sup>eejpwang@mail.scut.edu.cn, eelwjin@scut.edu.cn

<sup>2</sup>danny\_ding@intsig.net

## Abstract

Structured document understanding has attracted considerable attention and made significant progress recently, owing to its crucial role in intelligent document processing. However, most existing related models can only deal with the document data of specific language(s) (typically English) included in the pre-training collection, which is extremely limited. To address this issue, we propose a simple yet effective **Language-independent Layout Transformer (LiLT)** for structured document understanding. LiLT can be pre-trained on the structured documents of a single language and then directly fine-tuned on other languages with the corresponding off-the-shelf monolingual/multilingual pre-trained textual models. Experimental results on eight languages have shown that LiLT can achieve competitive or even superior performance on diverse widely-used downstream benchmarks, which enables language-independent benefit from the pre-training of document layout structure. Code and model are publicly available at <https://github.com/jpWang/LiLT>.

## 1 Introduction

Structured document understanding (SDU) aims at reading and analyzing the textual and structured information contained in scanned/digital-born documents. With the acceleration of the digitization process, it has been regarded as a crucial part of intelligent document processing and required by many real-world applications in various industries such as finance, medical treatment and insurance.

Recently, inspired by the rapid development of pre-trained language models of plain texts (Devlin et al., 2019; Liu et al., 2019b; Bao et al., 2020; Chi et al., 2021), many researches on structured document pre-training (Xu et al., 2020, 2021a,b; Li et al., 2021a,b,c; Appalaraju et al., 2021) have also

Figure 1: The substitution of language does not appear obviously unnatural when the layout structure remains unchanged, as shown in a (a) form/(b) receipt. The detailed content has been re-synthesized to avoid the sensitive information leak. Best viewed in zoomed-in.

pushed the limit of a variety of SDU tasks. However, almost all of them only focus on pre-training and fine-tuning on the documents in a single language, typically English. This is extremely limited for other languages, especially in the case of lacking pre-training structured document data.

In this regard, we consider how to make the SDU tasks enjoy language-independent benefit from the pre-training of document layout structure. Here, we give an observation as shown in Figure 1. When the layout structure remains unchanged, the substitution of language does not make obvious unnaturalness. It fully motivates us to decouple and reuse the layout invariance among different languages.

Based on this inspiration, in this paper, we propose a simple yet effective **Language-independent Layout Transformer (LiLT)** for structured document understanding. In our framework, the text and layout information are first decoupled and joint-optimized during pre-training, and then re-coupled for fine-tuning. To ensure that the two modalities have sufficient language-independent interaction, we further propose a novel bi-directional attention complementation mechanism (BiACM) to enhance the cross-modality cooperation. Moreover, we present the key point location (KPL) and cross-modal alignment identification (CAI) tasks, which are combined with the widely-used masked visual-

\*Corresponding author.language modeling (MVLM) to serve as our pre-training objectives. During fine-tuning, the layout flow (LiLT) can be separated and combined with the off-the-shelf pre-trained textual models (such as RoBERTa (Liu et al., 2019b), XLM-R (Conneau et al., 2020), InfoXLM (Chi et al., 2021), etc) to deal with the downstream tasks. In this way, our method decouples and learns the layout knowledge from the monolingual structured documents before generalizing it to the multilingual ones.

To the best of our knowledge, the only pre-existing multilingual SDU model is LayoutXLM (Xu et al., 2021b). It scrapes multilingual PDF documents of 53 languages from a web crawler and introduces extra pre-processing steps to clean the collected data, filter the low-quality documents, and classify them into different languages. After this, it utilizes a heuristic distribution to sample 22 million multilingual documents, which are further combined with the 8 million sampled English ones from the IIT-CDIP (Lewis et al., 2006) dataset (11 million English documents), resulting 30 million for pre-training with the LayoutLMv2 (Xu et al., 2021a) framework. However, this process is time-consuming and laborious. On the contrary, LiLT can be pre-trained with only IIT-CDIP and then adapted to other languages. In this respect, LiLT is the first language-independent method for structured document understanding.

Experimental results on eight languages have shown that LiLT can achieve competitive or even superior performance on diverse widely-used downstream benchmarks, which substantially benefits numerous real-world SDU applications. Our main contributions can be summarized as follows:

- • We introduce a simple yet effective language-independent layout Transformer called LiLT for monolingual/multilingual structured document understanding.
- • We propose BiACM to provide language-independent cross-modality interaction, along with an effective asynchronous optimization strategy for textual and non-textual flows in pre-training. Moreover, we present two new pre-training objectives, namely KPL and CAI.
- • LiLT achieves competitive or even superior performance on various widely-used downstream benchmarks of different languages under different settings, which fully demonstrates its effectiveness.

## 2 LiLT

Figure 2 shows the overall illustration of our method. Given an input document image, we first use off-the-shelf OCR engines to get text bounding boxes and contents. Then, the text and layout information are separately embedded and fed into the corresponding Transformer-based architecture to obtain enhanced features. Bi-directional attention complementation mechanism (BiACM) is introduced to accomplish the cross-modality interaction of text and layout clues. Finally, the encoded text and layout features are concatenated and additional heads are added upon them, for the self-supervised pre-training or the downstream fine-tuning.

### 2.1 Model Architecture

The whole framework can be regarded as a parallel dual-stream Transformer. The layout flow shares a similar structure as text flow, except for the reduced hidden size and intermediate size to achieve computational efficiency.

#### 2.1.1 Text Embedding

Following the common practice (Devlin et al., 2019; Xu et al., 2020), in the text flow, all text strings in the OCR results are first tokenized and concatenated as a sequence  $S_t$  by sorting the corresponding text bounding boxes from the top-left to bottom-right. Intuitively, the special tokens [CLS] and [SEP] are also added at the beginning and end of the sequence respectively. After this,  $S_t$  will be truncated or padded with extra [PAD] tokens until its length equals the maximum sequence length  $N$ . Finally, we sum the token embedding  $E_{token}$  of  $S_t$  and the 1D positional embedding  $P_{1D}$  to obtain the text embedding  $E_T \in \mathcal{R}^{N \times d_T}$  as:

$$E_T = \text{LN}(E_{token} + P_{1D}), \quad (1)$$

where  $d_T$  is the number of text feature dimension and LN is the layer normalization (Ba et al., 2016).

#### 2.1.2 Layout Embedding

As for the layout flow, we construct a 2D position sequence  $S_l$  with the same length as the token sequence  $S_t$  using the corresponding text bounding boxes. To be specific, we normalize and discretize all box coordinates to integers in the range  $[0, 1000]$ , and use four embedding layers to generate  $x$ -axis,  $y$ -axis, height, and width features separately. Given the normalized bounding boxes  $B = (x_{min}, x_{max}, y_{min}, y_{max}, width, height)$ , the 2DLegend:  
 ▭ Concatenate  
 + Add  
 ⊠ Detach (only exists in pre-training)  
 ⊠ BiACM

OCR Engines → Document → Text Flow (N\_l layers) and Layout Flow (N\_l layers) → Concatenated Features → Pre-training Objectives and Fine-tuning Tasks

Figure 2: The overall illustration of our framework. Text and layout information are separately embedded and fed into the corresponding flow. BiACM is proposed to accomplish the cross-modality interaction. At the model output, text and layout features are concatenated for the self-supervised pre-training or the downstream fine-tuning.  $N_l$  is the number of Transformer layers. The red  $*_M/*_R$  indicates the randomly masked/replaced item for pre-training.  $t, b$  and  $r$  represent *token*, *box* and *region*, respectively. Best viewed in zoomed-in.

positional embedding  $P_{2D} \in \mathcal{R}^{N \times d_L}$  (where  $d_L$  is the number of layout feature dimension) is constructed as follows:

$$P_{2D} = \text{Linear}(\text{CAT}(E_{x_{min}}, E_{x_{max}}, E_{y_{min}}, E_{y_{max}}, E_{width}, E_{height})). \quad (2)$$

Here, the  $E$ s are embedded vectors. *Linear* is a linear projection layer and *CAT* is the channel-wise concatenation operation. The special tokens [CLS], [SEP] and [PAD] are also attached with (0,0,0,0,0,0), (1000,1000,1000,1000,0,0) and (0,0,0,0,0,0) respectively. It is worth mentioning that, for each token, we directly utilize the bounding box of the text string it belongs to, because the fine-grained token-level information is not always included in the results of some OCR engines.

Since Transformer layers are permutation-invariant, here we introduce the 1D positional embedding again. The resulting layout embedding  $E_L \in \mathcal{R}^{N \times d_L}$  can be formulated as:

$$E_L = \text{LN}(P_{2D} + P_{1D}). \quad (3)$$

### 2.1.3 BiACM

The text embedding  $E_T$  and layout embedding  $E_L$  are fed into their respective sub-models to generate high-level enhanced features. However, it will considerably ignore the cross-modality interaction process if we simply combine the text and layout features at the encoder output only. The network also needs to comprehensively analyse them

at earlier stages. In view of this, we propose a new bi-directional attention complementation mechanism (BiACM) to strengthen the cross-modality interaction across the entire encoding pipeline. Experiments in Section 3.2 will further verify its effectiveness.

The vanilla self-attention mechanism in Transformer layers captures the correlation between query  $x_i$  and key  $x_j$  by projecting the two vectors and calculating the attention score as:

$$\alpha_{ij} = \frac{(x_i W^Q)(x_j W^K)^\top}{\sqrt{d^h}}. \quad (4)$$

Here, the description is for a single head in a single self-attention layer with hidden size of  $d^h$  and projection metrics  $W^Q, W^K$  for simplicity. Given  $\alpha_{ij}^T$  and  $\alpha_{ij}^L$  of the text and layout flows located in the same head of the same layer, BiACM shares them as common knowledge, which is formulated as:

$$\widetilde{\alpha}_{ij}^T = \alpha_{ij}^L + \alpha_{ij}^T, \quad (5)$$

$$\widetilde{\alpha}_{ij}^L = \begin{cases} \alpha_{ij}^L + \text{DETACH}(\alpha_{ij}^T) & \text{if Pre-train,} \\ \alpha_{ij}^L + \alpha_{ij}^T & \text{if Fine-tune.} \end{cases} \quad (6)$$

In order to maintain the ability of LiLT to cooperate with different off-the-shelf text models in fine-tuning as much as possible, we heuristically adopt the detached  $\alpha_{ij}^T$  for  $\widetilde{\alpha}_{ij}^L$ , so that the textual stream will not be affected by the gradient of non-textualone during pre-training, and its overall consistency can be preserved. Finally, the modified attention scores are used to weight the projected value vectors for subsequent modules in both flows.

## 2.2 Pre-training Tasks

We conduct three self-supervised pre-training tasks to guide the model to autonomously learn joint representations with cross-modal cooperation. The details are introduced below.

### 2.2.1 Masked Visual-Language Modeling

This task is originally derived from (Devlin et al., 2019). MVLM randomly masks some of the input tokens and the model is asked to recover them over the whole vocabulary using the output encoded features, driven by a cross-entropy loss. Meanwhile, the non-textual information remains unchanged. MVLM improves model learning on the language side with cross-modality information. The given layout embedding can also help the model better capture both inter- and intra-sentence relationships. We mask 15% text tokens, among which 80% are replaced by the special token [MASK], 10% are replaced by random tokens sampled from the whole vocabulary, and 10% remain the same.

### 2.2.2 Key Point Location

We propose this task to make the model better understand layout information in the structured documents. KPL equally divides the entire layout into several regions (we set  $7 \times 7 = 49$  regions by default) and randomly masks some of the input bounding boxes. The model is required to predict which regions the key points (top-left corner, bottom-right corner, and center point) of each box belong to using separate heads. To deal with it, the model is required to fully understand the text content and know where to put a specific word/sentence when the surrounding ones are given. We mask 15% boxes, among which 80% are replaced by (0,0,0,0,0,0), 10% are replaced by random boxes sampled from the same batch, and 10% remain the same. Cross-entropy loss is adopted.

Since there may exist detection errors in the output of OCR engines, we let the model predict the discretized regions (as mentioned above) instead of the exact location. This strategy can moderately relax the punishment criterion while improving the model performance.

### 2.2.3 Cross-modal Alignment Identification

We collect those encoded features of token-box pairs that are masked and further replaced (misaligned) or kept unchanged (aligned) by MVLM and KPL, and build an additional head upon them to identify whether each pair is aligned. To achieve this, the model is required to learn the cross-modal perception capacity. CAI is a binary classification task, and a cross-entropy loss is applied for it.

## 2.3 Optimization Strategy

Utilizing a unified learning rate for all model parameters to perform the end-to-end training process is the most common optimization strategy. While in our case, it will cause the layout flow to continuously update in the direction of coupling with the evolving text flow in the pre-training stage, which is harmful to the ability of LiLT to cooperate with different off-the-shelf textual models during fine-tuning. Based on this consideration, we explore multiple ratios to greatly slow down the pre-training optimization of the text stream. We also find that an appropriate reduction ratio is better than parameter freezing.

Note that, we adopt a unified learning rate for end-to-end optimization during fine-tuning. The DETACH operation of BiACM is also canceled at this time, as shown in Equation 6.

## 3 Experiments

### 3.1 Pre-training Setting

We pre-train LiLT on the IIT-CDIP Test Collection 1.0 (Lewis et al., 2006), which is a large-scale scanned document image dataset and contains more than 6 million documents with more than 11 million scanned document images. We use TextIn API<sup>1</sup> to obtain the text bounding boxes and strings for this dataset.

In this paper, we initialize the text flow from the existing pre-trained English RoBERTa<sub>BASE</sub> (Liu et al., 2019b) for our document pre-training, and combine LiLT<sub>BASE</sub> with the pre-trained InfoXLM<sub>BASE</sub> (Chi et al., 2021)/a new pre-trained RoBERTa<sub>BASE</sub> for multilingual/monolingual fine-tuning. They have an equal number of self-attention layers, attention heads and maximum sequence length, which ensures that BiACM can work normally. In this BASE setting, LiLT has a 12-layer encoder with 192 hidden size, 768 feed-forward filter size and 12 attention heads, resulting

<sup>1</sup><https://www.textin.com><table border="1">
<thead>
<tr>
<th>#</th>
<th>Inter-modal Operation</th>
<th>Average F1</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>CAT</td>
<td>0.6751</td>
</tr>
<tr>
<td>2</td>
<td>CAT+Co-Attention (Lu et al., 2019)</td>
<td>0.6276</td>
</tr>
<tr>
<td>3</td>
<td>CAT+BiACM</td>
<td><b>0.7963</b></td>
</tr>
<tr>
<td>4</td>
<td>CAT+BiACM–DETACH in pre-training</td>
<td>0.7682</td>
</tr>
<tr>
<td>5</td>
<td>CAT+BiACM+DETACH in fine-tuning</td>
<td>0.7822</td>
</tr>
</tbody>
</table>

<table border="1">
<tbody>
<tr>
<td>6</td>
<td>The text flow alone<br/>(InfoXLM<sub>BASE</sub>, as shown in Table 6)</td>
<td>0.7207</td>
</tr>
</tbody>
</table>

(a) BiACM. CAT is short for concatenation.

<table border="1">
<thead>
<tr>
<th>#</th>
<th>MVLM</th>
<th>KPL</th>
<th>CAI</th>
<th>Average F1</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>✓</td>
<td></td>
<td></td>
<td>0.7616</td>
</tr>
<tr>
<td>2</td>
<td>✓</td>
<td>✓</td>
<td></td>
<td>0.7748</td>
</tr>
<tr>
<td>3</td>
<td>✓</td>
<td></td>
<td>✓</td>
<td>0.7809</td>
</tr>
<tr>
<td>4</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td><b>0.7963</b></td>
</tr>
</tbody>
</table>

(b) Pre-training tasks.

<table border="1">
<thead>
<tr>
<th>#</th>
<th>Slow-down Ratio</th>
<th>Average F1</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>1 (No Slow-down)</td>
<td>0.7840</td>
</tr>
<tr>
<td>2</td>
<td>500</td>
<td>0.7901</td>
</tr>
<tr>
<td>3</td>
<td>800</td>
<td>0.7947</td>
</tr>
<tr>
<td>4</td>
<td>1000</td>
<td><b>0.7963</b></td>
</tr>
<tr>
<td>5</td>
<td>1200</td>
<td>0.7935</td>
</tr>
<tr>
<td>6</td>
<td>+∞ (Parameter Freezing)</td>
<td>0.7893</td>
</tr>
</tbody>
</table>

(c) Slow-down ratios.

Table 1: Ablation study of LiLT<sub>BASE</sub> combined with InfoXLM<sub>BASE</sub> (Chi et al., 2021) on the FUNSD and XFUND datasets (8 languages in total). The average F1 accuracy of language-specific semantic entity recognition (SER) task is given. (a) BiACM. (b) Pre-training tasks. (c) Slow-down ratios of the pre-training optimization for the text flow.

in the number of parameters as 6.1M. The maximum sequence length  $N$  is set as 512.

LiLT<sub>BASE</sub> is pre-trained using Adam optimizer (Kingma and Ba, 2015; Loshchilov and Hutter, 2018), with the learning rate  $2 \times 10^{-5}$ , weight decay  $1 \times 10^{-2}$ , and  $(\beta_1, \beta_2) = (0.9, 0.999)$ . The learning rate is linearly warmed up over the first 10% steps and then linearly decayed. We set the batch size as 96 and train LiLT<sub>BASE</sub> for 5 epochs on the IIT-CDIP dataset using 4 NVIDIA A40 48GB GPUs.

### 3.2 Ablation Study

Considering the complete pre-training takes a long time, we pre-train LiLT<sub>BASE</sub> with 2M documents randomly sampled from IIT-CDIP for 5 epochs to conduct ablation experiments, as shown in Table 1.

We first evaluate the effect of introducing BiACM. In setting (a)#1, the text and layout features are concatenated at the model output without any further interaction. Compared with (a)#6, we find that such a plain design results in a much

<table border="1">
<thead>
<tr>
<th>Model</th>
<th>Precision</th>
<th>Recall</th>
<th>F1</th>
</tr>
</thead>
<tbody>
<tr>
<td>BERT<sub>BASE</sub><sup>1</sup></td>
<td>0.5469</td>
<td>0.6710</td>
<td>0.6026</td>
</tr>
<tr>
<td>RoBERTa<sub>BASE</sub><sup>2</sup></td>
<td>0.6349</td>
<td>0.6975</td>
<td>0.6648</td>
</tr>
<tr>
<td>UniLMv2<sub>BASE</sub><sup>3</sup></td>
<td>0.6349</td>
<td>0.6975</td>
<td>0.6648</td>
</tr>
<tr>
<td>LayoutLM<sub>BASE</sub><sup>4</sup></td>
<td>0.7597</td>
<td>0.8155</td>
<td>0.7866</td>
</tr>
<tr>
<td>BROS<sub>BASE</sub><sup>5</sup></td>
<td>0.8056</td>
<td>0.8188</td>
<td>0.8121</td>
</tr>
<tr>
<td>SelfDoc<sup>6</sup></td>
<td>-</td>
<td>-</td>
<td>0.8336</td>
</tr>
<tr>
<td>LayoutLMv2<sub>BASE</sub><sup>7</sup></td>
<td>0.8029</td>
<td>0.8539</td>
<td>0.8276</td>
</tr>
<tr>
<td>StrucTexT<sub>BASE</sub><sup>8</sup></td>
<td><u>0.8568</u></td>
<td>0.8097</td>
<td>0.8309</td>
</tr>
<tr>
<td>DocFormer<sub>BASE</sub><sup>9</sup></td>
<td>0.8076</td>
<td>0.8609</td>
<td>0.8334</td>
</tr>
<tr>
<td>*LayoutXLM<sub>BASE</sub><sup>10</sup></td>
<td>0.7913</td>
<td>0.8158</td>
<td>0.8034</td>
</tr>
<tr>
<td><b>LiLT[EN-R<sup>2</sup>]<sub>BASE</sub></b></td>
<td><b>0.8721</b></td>
<td><b>0.8965</b></td>
<td><b>0.8841</b></td>
</tr>
<tr>
<td>*LiLT[InfoXLM<sup>11</sup>]<sub>BASE</sub></td>
<td>0.8467</td>
<td><u>0.8709</u></td>
<td><u>0.8586</u></td>
</tr>
</tbody>
</table>

Table 2: Comparison on the semantic entity recognition (SER) task of FUNSD (Jaume et al., 2019) dataset. **Bold** indicates the SOTA and underline indicates the second best. “EN-R” is short for English RoBERTa. \*The multilingual model. [] denotes the off-the-shelf textual model used as the text flow of LiLT. <sup>1</sup>(Devlin et al., 2019);<sup>2</sup>(Liu et al., 2019b);<sup>3</sup>(Bao et al., 2020);<sup>4</sup>(Xu et al., 2020);<sup>5</sup>(Hong et al., 2020);<sup>6</sup>(Li et al., 2021b);<sup>7</sup>(Xu et al., 2021a);<sup>8</sup>(Li et al., 2021c);<sup>9</sup>(Appalaraju et al., 2021);<sup>10</sup>(Xu et al., 2021b);<sup>11</sup>(Chi et al., 2021).

worse performance than using the text flow alone. From (a)#1 to (a)#3, the significant improvement demonstrates that it is the novel BiACM that makes the transfer from “monolingual” to “multilingual” successful. Beside this, we have also tried to replace BiACM with the co-attention mechanism (Lu et al., 2019) which is widely adopted in dual-stream Transformer architecture. It can be seen as a “deeper” cross-modal interaction, since the keys and values from each modality are passed as input to the other modality’s dot-product attention calculation. However, severe drops are observed as shown in (a)#2 vs (a)#1#3. We attribute it to the damage of such a “deeper” interaction to the overall consistency of the text flow in the pre-training optimization. In contrast, BiACM can maintain LiLT’s cross-model cooperation ability on the basis of providing cross-modal information. Moreover, the necessity of DETACH in pre-training is proved in (a)#4 vs (a)#3. Compared (a)#3 to (a)#5, we can also infer that removing DETACH in fine-tuning leads to a better performance.

Then, we compare the proposed KPL and CAI tasks. As shown in Table 1(b), both tasks improve the model performance substantially, and the proposed CAI benefits the model more than KPL. Using both tasks together is more effective than using either one alone.<table border="1">
<thead>
<tr>
<th>Model</th>
<th>Precision</th>
<th>Recall</th>
<th>F1</th>
</tr>
</thead>
<tbody>
<tr>
<td>BERT<sub>BASE</sub></td>
<td>0.8833</td>
<td>0.9107</td>
<td>0.8968</td>
</tr>
<tr>
<td>UniLMv2<sub>BASE</sub></td>
<td>0.8987</td>
<td>0.9198</td>
<td>0.9092</td>
</tr>
<tr>
<td>LayoutLM<sub>BASE</sub></td>
<td>0.9437</td>
<td>0.9508</td>
<td>0.9472</td>
</tr>
<tr>
<td>BROS<sub>BASE</sub></td>
<td>0.9558</td>
<td>0.9514</td>
<td>0.9536</td>
</tr>
<tr>
<td>LAMBERT<sub>BASE</sub><sup>1</sup></td>
<td>-</td>
<td>-</td>
<td>0.9441</td>
</tr>
<tr>
<td>TILT<sub>BASE</sub><sup>2</sup></td>
<td>-</td>
<td>-</td>
<td>0.9511</td>
</tr>
<tr>
<td>LayoutLMv2<sub>BASE</sub></td>
<td>0.9453</td>
<td>0.9539</td>
<td>0.9495</td>
</tr>
<tr>
<td>DocFormer<sub>BASE</sub></td>
<td><b>0.9652</b></td>
<td><b>0.9614</b></td>
<td><b>0.9633</b></td>
</tr>
<tr>
<td>*LayoutXLM<sub>BASE</sub></td>
<td>0.9456</td>
<td>0.9506</td>
<td>0.9481</td>
</tr>
<tr>
<td><b>LiLT[EN-R]<sub>BASE</sub></b></td>
<td><u>0.9598</u></td>
<td><b>0.9616</b></td>
<td><u>0.9607</u></td>
</tr>
<tr>
<td>*<b>LiLT[InfoXLM]<sub>BASE</sub></b></td>
<td>0.9574</td>
<td>0.9581</td>
<td>0.9577</td>
</tr>
</tbody>
</table>

Table 3: Comparison on the semantic entity recognition (SER) task of CORD (Park et al., 2019) dataset. <sup>1</sup>(Garncarek et al., 2021);<sup>2</sup>(Powalski et al., 2021).

<table border="1">
<thead>
<tr>
<th>Model</th>
<th>Precision</th>
<th>Recall</th>
<th>F1</th>
</tr>
</thead>
<tbody>
<tr>
<td>BiLSTM+CRF<sup>1</sup></td>
<td>-</td>
<td>-</td>
<td>0.8910</td>
</tr>
<tr>
<td>GraphIE<sup>2</sup></td>
<td>-</td>
<td>-</td>
<td>0.9026</td>
</tr>
<tr>
<td>GCN-based<sup>3</sup></td>
<td>-</td>
<td>-</td>
<td>0.9255</td>
</tr>
<tr>
<td>TRIE<sup>4</sup></td>
<td>-</td>
<td>-</td>
<td>0.9321</td>
</tr>
<tr>
<td>VIES<sup>5</sup></td>
<td>-</td>
<td>-</td>
<td>0.9523</td>
</tr>
<tr>
<td>MatchVIE<sup>6</sup></td>
<td>-</td>
<td>-</td>
<td>0.9687</td>
</tr>
<tr>
<td>TCPN<sup>7</sup></td>
<td>-</td>
<td>-</td>
<td>0.9759</td>
</tr>
<tr>
<td>RoBERTa<sub>BASE</sub><sup>8</sup></td>
<td>0.9405</td>
<td>0.9640</td>
<td>0.9521</td>
</tr>
<tr>
<td>StrucTexT<sub>BASE</sub></td>
<td>-</td>
<td>-</td>
<td><u>0.9795</u></td>
</tr>
<tr>
<td>*LayoutXLM<sub>BASE</sub></td>
<td><u>0.9699</u></td>
<td><u>0.9820</u></td>
<td>0.9759</td>
</tr>
<tr>
<td><b>LiLT[ZH-R]<sup>8</sup><sub>BASE</sub></b></td>
<td><b>0.9762</b></td>
<td><b>0.9833</b></td>
<td><b>0.9797</b></td>
</tr>
<tr>
<td>*<b>LiLT[InfoXLM]<sub>BASE</sub></b></td>
<td><u>0.9699</u></td>
<td><u>0.9820</u></td>
<td>0.9759</td>
</tr>
</tbody>
</table>

Table 4: Comparison on the semantic entity recognition (SER) task of EPHOIE (Wang et al., 2021a) dataset. “ZH-R” is short for Chinese RoBERTa. <sup>1</sup>(Lample et al., 2016);<sup>2</sup>(Qian et al., 2019);<sup>3</sup>(Liu et al., 2019a);<sup>4</sup>(Zhang et al., 2020);<sup>5</sup>(Wang et al., 2021a);<sup>6</sup>(Tang et al., 2021);<sup>7</sup>(Wang et al., 2021b);<sup>8</sup>(Cui et al., 2020).

Finally, we explore the most suitable slow-down ratio for the pre-training optimization of the text flow. A ratio equal to 1 in (c)#1 means there is no slow-down and a unified learning rate is adopted. It can be found that the F1 scores keep rising with the growth of slow-down ratios and begin to fall when the ratio is greater than 1000. Consequently, we set the slow-down ratio as 1000 by default.

### 3.3 Comparisons with the SOTAs

To demonstrate the performance of LiLT, we conduct experiments on several widely-used monolingual datasets and the multilingual XFUNDBenchmark (Xu et al., 2021b). In addition to the experiments involving typical language-specific fine-tuning, we also follow the two settings designed

<table border="1">
<thead>
<tr>
<th>Model</th>
<th>Accuracy</th>
</tr>
</thead>
<tbody>
<tr>
<td>VGG-16<sup>1</sup></td>
<td>90.97%</td>
</tr>
<tr>
<td>Stacked CNN Single<sup>2</sup></td>
<td>91.11%</td>
</tr>
<tr>
<td>Stacked CNN Ensemble<sup>2</sup></td>
<td>92.21%</td>
</tr>
<tr>
<td>InceptionResNetV2<sup>3</sup></td>
<td>92.63%</td>
</tr>
<tr>
<td>LadderNet<sup>4</sup></td>
<td>92.77%</td>
</tr>
<tr>
<td>Multimodal Single<sup>5</sup></td>
<td>93.03%</td>
</tr>
<tr>
<td>Multimodal Ensemble<sup>5</sup></td>
<td>93.07%</td>
</tr>
<tr>
<td>BERT<sub>BASE</sub></td>
<td>89.81%</td>
</tr>
<tr>
<td>UniLMv2<sub>BASE</sub></td>
<td>90.06%</td>
</tr>
<tr>
<td>LayoutLM<sub>BASE</sub> (w/ image)</td>
<td>94.42%</td>
</tr>
<tr>
<td>BROS<sub>BASE</sub></td>
<td>95.58%</td>
</tr>
<tr>
<td>SelfDoc</td>
<td>93.81%</td>
</tr>
<tr>
<td>TILT<sub>BASE</sub></td>
<td>93.50%</td>
</tr>
<tr>
<td>LayoutLMv2<sub>BASE</sub></td>
<td>95.25%</td>
</tr>
<tr>
<td>DocFormer<sub>BASE</sub></td>
<td><b>96.17%</b></td>
</tr>
<tr>
<td>*LayoutXLM<sub>BASE</sub></td>
<td>95.21%</td>
</tr>
<tr>
<td><b>LiLT[EN-R]<sub>BASE</sub></b></td>
<td><u>95.68%</u></td>
</tr>
<tr>
<td>*<b>LiLT[InfoXLM]<sub>BASE</sub></b></td>
<td>95.62%</td>
</tr>
</tbody>
</table>

Table 5: Comparison on the document classification (DC) task of RVL-CDIP (Harley et al., 2015) dataset. <sup>1</sup>(Afzal et al., 2017);<sup>2</sup>(Das et al., 2018);<sup>3</sup>(Szegedy et al., 2017);<sup>4</sup>(Sarkhel and Nandi, 2019);<sup>5</sup>(Dauphinee et al., 2019).

in (Xu et al., 2021b) to demonstrate the ability to transfer knowledge among different languages, which are zero-shot transfer learning and multitask fine-tuning, for fair comparisons. Specifically, (1) language-specific fine-tuning refers to the typical fine-tuning paradigm of fine-tuning on language X and testing on language X. (2) Zero-shot transfer learning means the models are fine-tuned on English data only and then evaluated on each target language. (3) Multitask fine-tuning requires the model to fine-tune on data in all languages.

#### 3.3.1 Language-specific Fine-tuning

We first evaluate LiLT on four widely-used monolingual datasets - FUNSD (Jaume et al., 2019), CORD (Park et al., 2019), EPHOIE (Wang et al., 2021a) and RVL-CDIP (Lewis et al., 2006), and the results are shown in Table 2, 3, 4 and 5. We have found that (1) LiLT is flexible since it can work with monolingual or multilingual plain text models to deal with downstream tasks. (2) Although LiLT is designed for the transfer from “monolingual” to “multilingual”, it can surprisingly cooperate with monolingual textual models to achieve competitive or even superior performance (especially on the FUNSD dataset with only a few training samples available), compared with existing language-specific SDU models such as LayoutLMv2 and<table border="1">
<thead>
<tr>
<th rowspan="2">Task</th>
<th rowspan="2">Model</th>
<th colspan="2">Pre-training Docs</th>
<th>FUNSD</th>
<th colspan="7">XFUND</th>
<th rowspan="2">Avg.</th>
</tr>
<tr>
<th>Language</th>
<th>Size</th>
<th>EN</th>
<th>ZH</th>
<th>JA</th>
<th>ES</th>
<th>FR</th>
<th>IT</th>
<th>DE</th>
<th>PT</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">SER</td>
<td>XLM-RoBERTa<sub>BASE</sub></td>
<td>-</td>
<td>-</td>
<td>0.6670</td>
<td>0.8774</td>
<td>0.7761</td>
<td>0.6105</td>
<td>0.6743</td>
<td>0.6687</td>
<td>0.6814</td>
<td>0.6818</td>
<td>0.7047</td>
</tr>
<tr>
<td>InfoXLM<sub>BASE</sub></td>
<td>-</td>
<td>-</td>
<td>0.6852</td>
<td>0.8868</td>
<td>0.7865</td>
<td>0.6230</td>
<td>0.7015</td>
<td>0.6751</td>
<td>0.7063</td>
<td>0.7008</td>
<td>0.7207</td>
</tr>
<tr>
<td>LayoutXLM<sub>BASE</sub></td>
<td>Multilingual</td>
<td>30M</td>
<td>0.7940</td>
<td>0.8924</td>
<td>0.7921</td>
<td>0.7550</td>
<td>0.7902</td>
<td>0.8082</td>
<td>0.8222</td>
<td>0.7903</td>
<td>0.8056</td>
</tr>
<tr>
<td><b>LiLT[InfoXLM]<sub>BASE</sub></b></td>
<td><b>English only</b></td>
<td><b>11M</b></td>
<td><b>0.8415</b></td>
<td><b>0.8938</b></td>
<td><b>0.7964</b></td>
<td><b>0.7911</b></td>
<td><b>0.7953</b></td>
<td><b>0.8376</b></td>
<td><b>0.8231</b></td>
<td><b>0.8220</b></td>
<td><b>0.8251</b></td>
</tr>
<tr>
<td rowspan="4">RE</td>
<td>XLM-RoBERTa<sub>BASE</sub></td>
<td>-</td>
<td>-</td>
<td>0.2659</td>
<td>0.5105</td>
<td>0.5800</td>
<td>0.5295</td>
<td>0.4965</td>
<td>0.5305</td>
<td>0.5041</td>
<td>0.3982</td>
<td>0.4769</td>
</tr>
<tr>
<td>InfoXLM<sub>BASE</sub></td>
<td>-</td>
<td>-</td>
<td>0.2920</td>
<td>0.5214</td>
<td>0.6000</td>
<td>0.5516</td>
<td>0.4913</td>
<td>0.5281</td>
<td>0.5262</td>
<td>0.4170</td>
<td>0.4910</td>
</tr>
<tr>
<td>LayoutXLM<sub>BASE</sub></td>
<td>Multilingual</td>
<td>30M</td>
<td>0.5483</td>
<td>0.7073</td>
<td>0.6963</td>
<td>0.6896</td>
<td>0.6353</td>
<td>0.6415</td>
<td>0.6551</td>
<td>0.5718</td>
<td>0.6432</td>
</tr>
<tr>
<td><b>LiLT[InfoXLM]<sub>BASE</sub></b></td>
<td><b>English only</b></td>
<td><b>11M</b></td>
<td><b>0.6276</b></td>
<td><b>0.7297</b></td>
<td><b>0.7037</b></td>
<td><b>0.7195</b></td>
<td><b>0.6965</b></td>
<td><b>0.7043</b></td>
<td><b>0.6558</b></td>
<td><b>0.5874</b></td>
<td><b>0.6781</b></td>
</tr>
</tbody>
</table>

Table 6: Language-specific fine-tuning F1 accuracy on FUNSD and XFUND (fine-tuning on X, testing on X). “SER” denotes the semantic entity recognition and “RE” denotes the relation extraction. [] indicates the off-the-shelf textual model used as the text flow of LiLT.

DocFormer. (3) On these datasets which are widely adopted for monolingual evaluation, LiLT generally performs better than LayoutXLM. This fully demonstrates the effectiveness of our pre-training framework and indicates that the layout and text information can be successfully decoupled in pre-training and re-coupled in fine-tuning.

Then we evaluate LiLT on language-specific fine-tuning tasks of FUNSD and the multilingual XFUND (Xu et al., 2021b), and the results are shown in Table 6. Compared with the plain text models (XLM-R/InfoXLM) or the LayoutXLM model pre-trained with 30M multilingual structured documents, LiLT achieves the highest F1 scores on both the SER and RE tasks of each language while using 11M monolingual data. This significant improvement shows LiLT’s capability to transfer language-independent knowledge from pre-training to downstream tasks.

### 3.3.2 Zero-shot Transfer Learning

The results of cross-lingual zero-shot transfer are presented in Table 7. It can be observed that the LiLT model transfers the most knowledge from English to other languages, and significantly outperforms its competitors. This fully verifies that LiLT can capture the common layout invariance among different languages. Moreover, LiLT has never seen non-English documents before evaluation under this setting, while the LayoutXLM model has been pre-trained with them. This is to say, LiLT faces a stricter cross-lingual zero-shot transfer scenario but achieves better performance.

### 3.3.3 Multi-task Fine-tuning

Table 8 shows the results of multitask learning. In this setting, the pre-trained LiLT model is simultaneously fine-tuned with all eight languages and

evaluated for each specific language. We observe that this setting further improves the model performance compared to the language-specific fine-tuning, which confirms that SDU can benefit from commonalities in the layout of multilingual structured documents. In addition, LiLT once again outperforms its counterparts by a large margin.

## 4 Related Work

During the past decade, deep learning methods became the mainstream for document understanding tasks (Yang et al., 2017; Augusto Borges Oliveira et al., 2017; Siegel et al., 2018). Grid-based methods (Katti et al., 2018; Denk and Reisswig, 2019; Lin et al., 2021) were proposed for 2D document representation where text pixels were encoded using character or word embeddings and classified into specific field types, using a convolutional neural network. GNN-based approaches (Liu et al., 2019a; Yu et al., 2021; Tang et al., 2021) adopted multi-modal features of text segments as nodes to model the document graph, and used graph neural networks to propagate information between neighboring nodes to attain a richer representation.

In recent years, self-supervised pre-training has achieved great success. Inspired by the development of the pre-trained language models in various NLP tasks, recent studies on structured document pre-training (Xu et al., 2020, 2021a,b; Li et al., 2021a,b,c; Appalaraju et al., 2021) have pushed the limits. LayoutLM (Xu et al., 2020) modified the BERT (Devlin et al., 2019) architecture by adding 2D spatial coordinate embeddings. In comparison, our LiLT can be regarded as a more powerful and flexible solution for structured document understanding. LayoutLMv2 (Xu et al., 2021a) improved over LayoutLM by treating the visual fea-<table border="1">
<thead>
<tr>
<th rowspan="2">Task</th>
<th rowspan="2">Model</th>
<th colspan="2">Pre-training Docs</th>
<th>FUNSD</th>
<th colspan="7">XFUND</th>
<th rowspan="2">Avg.</th>
</tr>
<tr>
<th>Language</th>
<th>Size</th>
<th>EN</th>
<th>ZH</th>
<th>JA</th>
<th>ES</th>
<th>FR</th>
<th>IT</th>
<th>DE</th>
<th>PT</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">SER</td>
<td>XLM-RoBERTa<sub>BASE</sub></td>
<td>-</td>
<td>-</td>
<td>0.6670</td>
<td>0.4144</td>
<td>0.3023</td>
<td>0.3055</td>
<td>0.3710</td>
<td>0.2767</td>
<td>0.3286</td>
<td>0.3936</td>
<td>0.3824</td>
</tr>
<tr>
<td>InfoXLM<sub>BASE</sub></td>
<td>-</td>
<td>-</td>
<td>0.6852</td>
<td>0.4408</td>
<td>0.3603</td>
<td>0.3102</td>
<td>0.4021</td>
<td>0.2880</td>
<td>0.3587</td>
<td>0.4502</td>
<td>0.4119</td>
</tr>
<tr>
<td>LayoutXLM<sub>BASE</sub></td>
<td>Multilingual</td>
<td>30M</td>
<td>0.7940</td>
<td>0.6019</td>
<td>0.4715</td>
<td>0.4565</td>
<td>0.5757</td>
<td>0.4846</td>
<td>0.5252</td>
<td>0.5390</td>
<td>0.5561</td>
</tr>
<tr>
<td><b>LiLT[InfoXLM]<sub>BASE</sub>♠</b></td>
<td><b>English only</b></td>
<td><b>11M</b></td>
<td><b>0.8415</b></td>
<td><b>0.6152</b></td>
<td><b>0.5184</b></td>
<td><b>0.5101</b></td>
<td><b>0.5923</b></td>
<td><b>0.5371</b></td>
<td><b>0.6013</b></td>
<td><b>0.6325</b></td>
<td><b>0.6061</b></td>
</tr>
<tr>
<td rowspan="4">RE</td>
<td>XLM-RoBERTa<sub>BASE</sub></td>
<td>-</td>
<td>-</td>
<td>0.2659</td>
<td>0.1601</td>
<td>0.2611</td>
<td>0.2440</td>
<td>0.2240</td>
<td>0.2374</td>
<td>0.2288</td>
<td>0.1996</td>
<td>0.2276</td>
</tr>
<tr>
<td>InfoXLM<sub>BASE</sub></td>
<td>-</td>
<td>-</td>
<td>0.2920</td>
<td>0.2405</td>
<td>0.2851</td>
<td>0.2481</td>
<td>0.2454</td>
<td>0.2193</td>
<td>0.2027</td>
<td>0.2049</td>
<td>0.2423</td>
</tr>
<tr>
<td>LayoutXLM<sub>BASE</sub></td>
<td>Multilingual</td>
<td>30M</td>
<td>0.5483</td>
<td>0.4494</td>
<td>0.4408</td>
<td>0.4708</td>
<td>0.4416</td>
<td>0.4090</td>
<td>0.3820</td>
<td>0.3685</td>
<td>0.4388</td>
</tr>
<tr>
<td><b>LiLT[InfoXLM]<sub>BASE</sub>♠</b></td>
<td><b>English only</b></td>
<td><b>11M</b></td>
<td><b>0.6276</b></td>
<td><b>0.4764</b></td>
<td><b>0.5081</b></td>
<td><b>0.4968</b></td>
<td><b>0.5209</b></td>
<td><b>0.4697</b></td>
<td><b>0.4169</b></td>
<td><b>0.4272</b></td>
<td><b>0.4930</b></td>
</tr>
</tbody>
</table>

Table 7: Cross-lingual zero-shot transfer F1 accuracy on FUNSD and XFUND (fine-tuning on FUNSD, testing on X). ♠ indicates that LiLT faces a stricter zero-shot transfer scenario compared with LayoutXLM, since it has never seen non-English documents before evaluation, even during pre-training.

<table border="1">
<thead>
<tr>
<th rowspan="2">Task</th>
<th rowspan="2">Model</th>
<th colspan="2">Pre-training Docs</th>
<th>FUNSD</th>
<th colspan="7">XFUND</th>
<th rowspan="2">Avg.</th>
</tr>
<tr>
<th>Language</th>
<th>Size</th>
<th>EN</th>
<th>ZH</th>
<th>JA</th>
<th>ES</th>
<th>FR</th>
<th>IT</th>
<th>DE</th>
<th>PT</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">SER</td>
<td>XLM-RoBERTa<sub>BASE</sub></td>
<td>-</td>
<td>-</td>
<td>0.6633</td>
<td>0.8830</td>
<td>0.7786</td>
<td>0.6223</td>
<td>0.7035</td>
<td>0.6814</td>
<td>0.7146</td>
<td>0.6726</td>
<td>0.7149</td>
</tr>
<tr>
<td>InfoXLM<sub>BASE</sub></td>
<td>-</td>
<td>-</td>
<td>0.6538</td>
<td>0.8741</td>
<td>0.7855</td>
<td>0.5979</td>
<td>0.7057</td>
<td>0.6826</td>
<td>0.7055</td>
<td>0.6796</td>
<td>0.7106</td>
</tr>
<tr>
<td>LayoutXLM<sub>BASE</sub></td>
<td>Multilingual</td>
<td>30M</td>
<td>0.7924</td>
<td>0.8973</td>
<td>0.7964</td>
<td>0.7798</td>
<td>0.8173</td>
<td>0.8210</td>
<td>0.8322</td>
<td>0.8241</td>
<td>0.8201</td>
</tr>
<tr>
<td><b>LiLT[InfoXLM]<sub>BASE</sub></b></td>
<td><b>English only</b></td>
<td><b>11M</b></td>
<td><b>0.8574</b></td>
<td><b>0.9047</b></td>
<td><b>0.8088</b></td>
<td><b>0.8340</b></td>
<td><b>0.8577</b></td>
<td><b>0.8792</b></td>
<td><b>0.8769</b></td>
<td><b>0.8493</b></td>
<td><b>0.8585</b></td>
</tr>
<tr>
<td rowspan="4">RE</td>
<td>XLM-RoBERTa<sub>BASE</sub></td>
<td>-</td>
<td>-</td>
<td>0.3638</td>
<td>0.6797</td>
<td>0.6829</td>
<td>0.6828</td>
<td>0.6727</td>
<td>0.6937</td>
<td>0.6887</td>
<td>0.6082</td>
<td>0.6341</td>
</tr>
<tr>
<td>InfoXLM<sub>BASE</sub></td>
<td>-</td>
<td>-</td>
<td>0.3699</td>
<td>0.6493</td>
<td>0.6473</td>
<td>0.6828</td>
<td>0.6831</td>
<td>0.6690</td>
<td>0.6384</td>
<td>0.5763</td>
<td>0.6145</td>
</tr>
<tr>
<td>LayoutXLM<sub>BASE</sub></td>
<td>Multilingual</td>
<td>30M</td>
<td>0.6671</td>
<td>0.8241</td>
<td>0.8142</td>
<td>0.8104</td>
<td>0.8221</td>
<td>0.8310</td>
<td>0.7854</td>
<td>0.7044</td>
<td>0.7823</td>
</tr>
<tr>
<td><b>LiLT[InfoXLM]<sub>BASE</sub></b></td>
<td><b>English only</b></td>
<td><b>11M</b></td>
<td><b>0.7407</b></td>
<td><b>0.8471</b></td>
<td><b>0.8345</b></td>
<td><b>0.8335</b></td>
<td><b>0.8466</b></td>
<td><b>0.8458</b></td>
<td><b>0.7878</b></td>
<td><b>0.7643</b></td>
<td><b>0.8125</b></td>
</tr>
</tbody>
</table>

Table 8: Multitask fine-tuning F1 accuracy on FUNSD and XFUND (fine-tuning on 8 languages all, testing on X).

tures as separate tokens. Furthermore, additional pre-training tasks were explored to improve the utilization of unlabeled document data. SelfDoc (Li et al., 2021b) established the contextualization over a block of content, while StructuralLM (Li et al., 2021a) proposed cell-level 2D position embeddings and the corresponding pre-training objective. Recently, StrucTexT (Li et al., 2021c) introduced a unified solution to efficiently extract semantic features from different levels and modalities to handle the entity labeling and entity linking tasks. DocFormer (Appalaraju et al., 2021) designed a novel multi-modal self-attention layer capable of fusing textual, vision and spatial features.

Nevertheless, the aforementioned SDU approaches mainly focus on a single language - typically English, which is extremely limited with respect to multilingual application scenarios. To the best of our knowledge, LayoutXLM (Xu et al., 2021b) was the only pre-existing multilingual SDU model, which adopted the multilingual textual model InfoXLM (Chi et al., 2021) as the initialization, and adapted the LayoutLMv2 (Xu et al., 2021a) framework to multilingual structured document pre-training. However, it required a heavy process of multilingual data collection, cleaning and pre-training. On the contrary, our LiLT can

deal with the multilingual structured documents by pre-training on the monolingual IIT-CDIP Test Collection 1.0 (Lewis et al., 2006) only.

## 5 Conclusion

In this paper, we present LiLT, a language-independent layout Transformer that can learn the layout knowledge from monolingual structured documents and then generalize it to deal with multilingual ones. Our framework successfully first decouples the text and layout information in pre-training and then re-couples them for fine-tuning. Experimental results on eight languages under three settings (language-specific, cross-lingual zero-shot transfer, and multi-task fine-tuning) have fully illustrated its effectiveness, which substantially bridges the language gap in real-world structured document understanding applications. The public availability of LiLT is also expected to promote the development of document intelligence.

For future research, we will continue to follow the pattern of transferring from “monolingual” to “multilingual” and further unlock the power of LiLT. In addition, we will also explore the generalized rather than language-specific visual information contained in multilingual structured documents.## 6 Acknowledgement

This research is supported in part by NSFC (Grant No.: 61936003) and GD-NSF (No. 2017A030312006).

## References

Muhammad Zeshan Afzal, Andreas Kölsch, Sheraz Ahmed, and Marcus Liwicki. 2017. Cutting the error by half: Investigation of very deep CNN and advanced training strategies for document image classification. In *ICDAR*, volume 1, pages 883–888.

Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota, Yusheng Xie, and R Manmatha. 2021. DocFormer: End-to-end Transformer for document understanding. In *ICCV*.

Dario Augusto Borges Oliveira et al. 2017. Fast CNN-based document layout analysis. In *ICCV Workshop*, pages 1173–1180.

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. 2016. Layer normalization. *arXiv preprint arXiv:1607.06450*.

Hangbo Bao, Li Dong, Furu Wei, Wenhui Wang, Nan Yang, Xiaodong Liu, Yu Wang, Jianfeng Gao, Songhao Piao, Ming Zhou, et al. 2020. UniLMv2: Pseudo-masked language models for unified language model pre-training. In *ICML*, pages 642–652.

Zewen Chi, Li Dong, Furu Wei, Nan Yang, Saksham Singhal, Wenhui Wang, Xia Song, Xian-Ling Mao, He-Yan Huang, and Ming Zhou. 2021. InfoXLM: An information-theoretic framework for cross-lingual language model pre-training. In *NAACL-HLT*, pages 3576–3588.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Édouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In *ACL*, pages 8440–8451.

Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Shijin Wang, and Guoping Hu. 2020. Revisiting pre-trained models for Chinese natural language processing. In *Findings of EMNLP*, pages 657–668.

Arindam Das, Saikat Roy, Ujjwal Bhattacharya, and Swapan K Parui. 2018. Document image classification with intra-domain transfer learning and stacked generalization of deep convolutional neural networks. In *ICPR*, pages 3180–3185.

Tyler Dauphinee, Nikunj Patel, and Mohammad Rashidi. 2019. Modular multimodal architecture for document classification. *arXiv preprint arXiv:1912.04376*.

Timo I Denk and Christian Reisswig. 2019. BERTgrid: Contextualized embedding for 2D document representation and understanding. In *Workshop on Document Intelligence at NeurIPS*.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional Transformers for language understanding. In *NAACL-HLT*, pages 4171–4186.

Łukasz Gąrcarek, Rafał Powalski, Tomasz Stanisławek, Bartosz Topolski, Piotr Halama, and Filip Grialński. 2021. LAMBERT: Layout-aware (language) modeling using BERT for information extraction. In *ICDAR*.

Adam W Harley et al. 2015. Evaluation of deep convolutional nets for document image classification and retrieval. In *ICDAR*, pages 991–995.

Teakgyu Hong, DongHyun Kim, Mingi Ji, Wonseok Hwang, Daehyun Nam, and Sungrae Park. 2020. BROS: A pre-trained language model for understanding texts in document.

Guillaume Jaume et al. 2019. FUNSD: A dataset for form understanding in noisy scanned documents. In *ICDAR*, volume 2, pages 1–6.

Anoop R Katti, Christian Reisswig, Cordula Guder, Sebastian Brarda, Steffen Bickel, Johannes Höhne, and Jean Baptiste Faddoul. 2018. Chargrid: Towards understanding 2D documents. In *EMNLP*, pages 4459–4469.

Diederik P Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. In *ICLR*.

Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, and Chris Dyer. 2016. Neural architectures for named entity recognition. In *NAACL-HLT*, pages 260–270.

David Lewis, Gady Agam, Shlomo Argamon, Ophir Frieder, David Grossman, and Jefferson Heard. 2006. Building a test collection for complex document information processing. In *ACM SIGIR*, pages 665–666.

Chenliang Li, Bin Bi, Ming Yan, Wei Wang, Songfang Huang, Fei Huang, and Luo Si. 2021a. StructuralLM: Structural pre-training for form understanding. In *ACL*.

Peizhao Li, Jiuxiang Gu, Jason Kuen, Vlad I Morariu, Handong Zhao, Rajiv Jain, Varun Manjunatha, and Hongfu Liu. 2021b. SelfDoc: Self-supervised document representation learning. In *CVPR*, pages 5652–5660.

Yulin Li, Yuxi Qian, Yuchen Yu, Xiameng Qin, Chengquan Zhang, Yan Liu, Kun Yao, Junyu Han, Jingtuo Liu, and Errui Ding. 2021c. StructTexT: Structured text understanding with multi-modal Transformers. In *ACM-MM*.Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. 2017. Feature pyramid networks for object detection. In *CVPR*, pages 2117–2125.

Weihong Lin, Qifang Gao, Lei Sun, Zhuoyao Zhong, Kai Hu, Qin Ren, and Qiang Huo. 2021. ViBERT-grid: A jointly trained multi-modal 2D document representation for key information extraction from documents. In *ICDAR*.

Xiaojing Liu, Feiyu Gao, Qiong Zhang, and Huasha Zhao. 2019a. Graph convolution for multimodal information extraction from visually rich documents. In *NAACL-HLT*, pages 32–39.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019b. RoBERTa: A robustly optimized BERT pretraining approach. *arXiv preprint arXiv:1907.11692*.

Ilya Loshchilov and Frank Hutter. 2018. Decoupled weight decay regularization. In *ICLR*.

Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. 2019. ViLBERT: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. *NeurIPS*, 32:13–23.

Seunghyun Park, Seung Shin, Bado Lee, Junyeop Lee, Jaeheung Surh, Minjoon Seo, and Hwalsuk Lee. 2019. CORD: A consolidated receipt dataset for post-OCR parsing. In *Workshop on Document Intelligence at NeurIPS*.

Rafał Powalski, Łukasz Borchmann, Dawid Jurkiewicz, Tomasz Dwojak, Michał Pietruszka, and Gabriela Pałka. 2021. Going full-TILT boogie on document understanding with text-image-layout Transformer. In *ICDAR*.

Yujie Qian, Enrico Santus, Zhijing Jin, Jiang Guo, and Regina Barzilay. 2019. GraphIE: A graph-based framework for information extraction. In *NAACL-HLT*, pages 751–761.

Ritesh Sarkhel and Arnab Nandi. 2019. Deterministic routing between layout abstractions for multi-scale classification of visually rich documents. In *IJCAI*, pages 3360–3366.

Noah Siegel, Nicholas Lourie, Russell Power, and Waleed Ammar. 2018. Extracting scientific figures with distantly supervised neural networks. In *JCDL*, pages 223–232.

Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, and Alexander A Alemi. 2017. Inception-v4, Inception-ResNet and the impact of residual connections on learning. In *AAAI*, pages 4278–4284.

Guozhi Tang, Lele Xie, Lianwen Jin, Jiapeng Wang, Jingdong Chen, Zhen Xu, Qianying Wang, Yaqiang Wu, and Hui Li. 2021. MatchVIE: Exploiting match relevancy between entities for visual information extraction. In *IJCAI*, pages 1039–1045.

Jiapeng Wang, Chongyu Liu, Lianwen Jin, Guozhi Tang, Jiaxin Zhang, Shuaitao Zhang, Qianying Wang, Yaqiang Wu, and Mingxiang Cai. 2021a. Towards robust visual information extraction in real world: New dataset and novel solution. In *AAAI*, volume 35, pages 2738–2745.

Jiapeng Wang, Tianwei Wang, Guozhi Tang, Lianwen Jin, Weihong Ma, Kai Ding, and Yichao Huang. 2021b. Tag, copy or predict: A unified weakly-supervised learning framework for visual information extraction using sequences. In *IJCAI*, pages 1082–1090.

Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. 2017. Aggregated residual transformations for deep neural networks. In *CVPR*, pages 1492–1500.

Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, et al. 2021a. LayoutLMv2: Multi-modal pre-training for visually-rich document understanding. In *ACL*.

Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. 2020. LayoutLM: Pre-training of text and layout for document image understanding. In *ACM-SIGKDD*, pages 1192–1200.

Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, and Furu Wei. 2021b. LayoutXLM: Multimodal pre-training for multilingual visually-rich document understanding. *arXiv preprint arXiv:2104.08836*.

Xiao Yang, Ersin Yumer, Paul Asente, Mike Kraley, Daniel Kifer, and C Lee Giles. 2017. Learning to extract semantic structure from documents using multimodal fully convolutional neural networks. In *CVPR*, pages 5315–5324.

Wenwen Yu, Ning Lu, Xianbiao Qi, Ping Gong, and Rong Xiao. 2021. PICK: Processing key information extraction from documents using improved graph learning-convolutional networks. In *ICPR*, pages 4363–4370.

Peng Zhang, Yunlu Xu, Zhanzhan Cheng, Shiliang Pu, Jing Lu, Liang Qiao, Yi Niu, and Fei Wu. 2020. TRIE: End-to-end text reading and information extraction for document understanding. In *ACM-MM*, pages 1413–1422.## Appendix

### A Dataset Details

**FUNSD** FUNSD (Jaume et al., 2019) is an English dataset for form understanding in noisy scanned documents. It contains 199 real, fully annotated, scanned forms where 9,707 semantic entities are annotated above 31,485 words. The 199 samples are split into 149 for training and 50 for testing. We directly use the official OCR annotations. The semantic entity recognition (SER) task is assigning to each word a semantic entity label from a set of four predefined categories: question, answer, header, or other. The entity-level F1 score is used as the evaluation metric (Table 2).

**CORD** CORD (Park et al., 2019) is an English receipt dataset for key information extraction. Its publicly available subset includes 800 receipts for the training set, 100 for the validation set, and 100 for the test set. A photo and a list of OCR annotations are equipped for each receipt. The dataset defines 30 fields under 4 categories and the task aims to label each word to the right field. The evaluation metric is the entity-level F1 score, as shown in Table 3. We use the official OCR annotations.

**EPHOIE** EPHOIE (Wang et al., 2021a) is collected from actual Chinese examination papers with the diversity of text types and layout distribution. The 1,494 samples are divided into a training set with 1,183 images and a testing set with 311 images, respectively. It defines ten entity categories, and we provide the entity-level F1 score for RoBERTa, LayoutXLM and LiLT in Table 4. The official OCR annotations are adopted.

**RVL-CDIP** RVL-CDIP (Harley et al., 2015) consists of 400,000 gray-scale images of English documents, with 8:1:1 for the training set, validation set, and test set. A multi-class single-label classification task is defined on RVL-CDIP. The images are categorized into 16 classes, with 25,000 images per class. The evaluation metric is the overall classification accuracy as shown in Table 5. Text and layout information are extracted by TextIn API.

**XFUND** XFUND (Xu et al., 2021b) is a multilingual form understanding dataset that contains 1,393 fully annotated forms with seven languages including Chinese (ZH), Japanese (JA), Spanish (ES), French (FR), Italian (IT), German (DE), and Portuguese (PT). Each language includes 199 forms,

where the training set includes 149 forms, and the test set includes 50 forms. We focus on the semantic entity recognition (SER) and relation extraction (RE) tasks defined in the original paper (Xu et al., 2021b). Relation extraction aims to predict the relation between any two given semantic entities, and we mainly focus on the key-value relation extraction. We use the official OCR results, and the same F1 accuracy evaluation metric as in LayoutXLM (Xu et al., 2021b) for Table 6, 7 and 8.

### B Fine-tuning Details

**Fine-tuning for Semantic Entity Recognition** We conduct the semantic entity recognition task on FUNSD, CORD, EPHOIE and XFUND. We build a token-level classification layer above the output representations to predict the BIO tags for each entity field.

**Fine-tuning for Document Classification** This task depends on high-level visual information, thereby we leverage the image features explicitly in the fine-tuning stage, following LayoutLMv2 (Xu et al., 2021a). We pool the visual feature of the ResNeXt101-FPN (Xie et al., 2017; Lin et al., 2017) backbone into a global feature, concatenate it with the [CLS] output feature, and feed them into the final classification layer.

**Fine-tuning for Relation Extraction** We build the additional head for relation extraction on the FUNSD and XFUND datasets following (Xu et al., 2021b) for fair comparison. We first incrementally construct the set of relation candidates by producing all possible pairs of given semantic entities. For every pair, the representation of the head/tail entity is the concatenation of the first token vector in each entity and the entity type embedding obtained with a specific type embedding layer. After respectively projected by two FFN layers, the representations of head and tail are concatenated and then fed into a bi-affine classifier.

