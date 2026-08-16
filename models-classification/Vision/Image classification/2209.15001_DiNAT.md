# Dilated Neighborhood Attention Transformer

Ali Hassani<sup>1</sup>, Humphrey Shi<sup>1,2</sup>

<sup>1</sup>SHI Lab @ U of Oregon & UIUC, <sup>2</sup>Picsart AI Research (PAIR)

**Figure 1.** An illustration of a single pixel’s attention span in Neighborhood Attention (NA) and Dilated Neighborhood Attention (DiNA). NA localizes attention to the pixel’s nearest neighbors. DiNA extends NA’s *local attention* to a less constrained *sparse global attention* without additional computational burden. Transformers comprised of both NA and DiNA are capable of preserving locality, maintaining translational equivariance, expanding the receptive field exponentially, and capturing longer-range inter-dependencies, leading to significant performance boosts in downstream vision tasks.

## Abstract

Transformers are quickly becoming one of the most heavily applied deep learning architectures across modalities, domains, and tasks. In vision, on top of ongoing efforts into plain transformers, hierarchical transformers have also gained significant attention, thanks to their performance and easy integration into existing frameworks. These models typically employ localized attention mechanisms, such as the sliding-window Neighborhood Attention (NA) or Swin Transformer’s Shifted Window Self Attention. While effective at reducing self attention’s quadratic complexity, local attention weakens two of the most desirable properties of self attention: long range inter-dependency modeling, and global receptive field. In this paper, we introduce **Dilated Neighborhood Attention (DiNA)**, a natural, flexible and efficient extension to NA that can capture more global context and expand receptive fields exponentially **at no additional cost**. *NA’s local attention* and *DiNA’s sparse global attention* complement each other, and therefore we introduce **Dilated Neighborhood Attention Transformer (DiNAT)**, a new hierarchical vision transformer built upon both. DiNAT variants enjoy significant improvements over strong baselines such as NAT, Swin, and ConvNeXt. Our

large model is faster and ahead of its Swin counterpart by 1.6% box AP in COCO object detection, 1.4% mask AP in COCO instance segmentation, and 1.4% mIoU in ADE20K semantic segmentation. Paired with new frameworks, our large variant is the new state of the art panoptic segmentation model on COCO (58.5 PQ) and ADE20K (49.4 PQ), and instance segmentation model on Cityscapes (45.1 AP) and ADE20K (35.4 AP) (no extra data). It also matches the state of the art specialized semantic segmentation models on ADE20K (58.1 mIoU), and ranks second on Cityscapes (84.5 mIoU) (no extra data). To support and encourage research in this direction, in vision and beyond, we open-source our project at: <https://github.com/SHI-Labs/Neighborhood-Attention-Transformer>.

## 1. Introduction

Transformers [42] have made a significant contribution to AI research, starting with natural language understanding [12, 34] before being applied to other modalities such as speech [14] and vision [13, 32], thanks to their universal architecture built upon self attention. This success inspired efforts into attention-based models in vision, from backbone networks [35, 41], to more specific applications including**Figure 2. Radar chart comparing Swin-L, ConvNeXt-L, and our DiNAT-L across various visual recognition tasks.** Both ConvNeXt and our DiNAT surpass Swin on all tasks. DiNAT further exhibits noticeable improvements over downstream dense recognition tasks compared to ConvNeXt.

image generation and density modeling [6, 32], object detection [3], image segmentation [20, 43], and more.

Vision Transformer (ViT) [13] was one of the first major demonstrations of transformers as direct alternatives to Convolutional Neural Networks (CNNs) [19, 22, 23], the de facto standard in vision. ViT treats an image as a sequence of patches and uses a plain transformer encoder to encode and classify images. It demonstrated competitive performance to CNNs on large scale image classification, and resulted in a surge in vision research focused on transformer-based architectures as competitors to CNNs [38, 39].

Vision transformers and CNNs are different not only in terms of architecture and building blocks, but also in how they treat data. CNNs typically downsample inputs gradually as they pass through the model and construct hierarchical feature maps. This hierarchical design is crucial for vision, as objects vary in scale, and high-resolution feature maps are important to dense tasks, such as segmentation. On the other hand, transformers are known for their fixed dimensionality throughout the model, and as a result, plain ViTs downsample inputs aggressively from the very beginning to alleviate the quadratic cost of self attention, which in turn hinders the application of plain ViTs as backbones to dense vision tasks.

While research in applying plain ViTs to dense vision tasks continues [17, 24], research into **hierarchical vision transformers** quickly became dominant [29, 44] and contin-

ues to grow [15, 28]. A key advantage of these hierarchical transformer models is their ease of integration with existing hierarchical vision frameworks. Inspired by existing CNNs, hierarchical vision transformers are comprised of multiple (typically 4) levels of transformer encoders, with down-sampling modules in between, and a less aggressive initial downsampling (i.e. 1/4 instead of 1/16). Earlier layers in hierarchical transformers, if using unrestricted self attention, would bear the same quadratically growing complexity and memory usage with respect to input resolution, making them intractable for higher resolution images. Therefore, hierarchical transformers typically employ certain **local attention** mechanisms.

Swin Transformer [29], one of the earliest hierarchical vision transformers, utilizes a Window Self Attention (WSA) module, followed by a pixel-shifted Window Self Attention (SWSA), both of which localize self attention to non-overlapping sub-windows. This reduces the cost of self attention, making its time and space complexity linear with respect to resolution. SWSA is identical to WSA, but with a shift in feature map pixels preceding it, and followed by a reverse shift. This is essential to its performance, as it allows out-of-window interactions, and therefore the expansion of its receptive field. One of the major advantages of Swin is efficiency, as pixel shifts and window partitioning are relatively cheap and easily parallelizable operations. Additionally, it involves little to no changes to the self attention module, making implementation easier. Swin became the state of the art across multiple vision tasks, and followed by Swin-V2 [28] to accommodate large scale pre-training.

Neighborhood Attention Transformer (NAT) [15] was introduced later, with a simple sliding-window based attention, Neighborhood Attention (NA). Unlike Stand Alone Self Attention (SASA) [35], which applies attention in the style of convolutions, NA localizes self attention to the nearest neighbors around each token, which allows it by definition to approach self attention and enjoy a fixed attention span. Such pixel-wise attention operations were assumed to be inefficient and challenging to parallelize [29, 35, 41], until the release of Neighborhood Attention Extension [15]. With this extension, NA can run even faster than Swin’s SWSA in practice. NAT was able to significantly outperform Swin on image classification, and achieved competitive performance on downstream tasks, while also scaling up to be even faster than Swin despite the slightly different architecture.

Despite the efforts into hierarchical vision transformers with local attention, some of self attention’s most important properties, including *global receptive field*, and the ability to *model long-range inter-dependencies*, are weakened as a result of this localization.

This leads to a simple question: ***How does one maintain the tractability that local attention provides in hierarchical vision transformers, while avoiding its shortcomings?***In other words, the optimal scenario is maintaining the linear complexity, while preserving the global receptive field and the ability to model long-range inter-dependencies of self attention. In this paper, we aim to answer this question and improve hierarchical transformers by extending a simple local attention mechanism, Neighborhood Attention, to Dilated Neighborhood Attention (DiNA): a flexible and powerful **sparse global attention**. Dilating neighborhoods in NA into larger sparse regions has multiple advantages: 1. it captures more global context, 2. allows the receptive field to grow exponentially, as opposed to linearly [50], and 3. comes *at no additional computational cost*. To demonstrate the effectiveness of DiNA, we propose Dilated Neighborhood Attention Transformer (DiNAT), which not only improves the existing NAT model in terms of downstream performance, it manages to outperform strong modern CNN baselines, such as ConvNeXt [30], in downstream tasks with a noticeable margin.

Our main contributions can be summarized as follows:

- • Introducing DiNA, a simple and powerful sparse global attention pattern, which allows receptive field to grow exponentially and captures longer-range context without any additional computational burden. DiNA does so while maintaining the symmetry in neighborhoods introduced in NA. It can also adapt to larger resolutions without expanding to larger window sizes.
- • Analyzing theoretical receptive field sizes in models based on convolutions, localized attention, and a DiNA-based model.
- • Introducing DiNAT, a new hierarchical vision transformer made of both dilated and non-dilated variants of NA. DiNAT utilizes a *gradual* dilation change through the model, which extends receptive fields more optimally and helps fine-to-coarse feature learning.
- • Conducting extensive experiments on image classification, object detection, and segmentation with DiNAT, and finding that it exhibits a noticeable improvement in downstream tasks over both attention-based and convolutional baselines. Additionally, we investigate isotropic and hybrid attention variants, scaling experiments with ImageNet-22K pre-training, and the effects of different dilation values. We also achieve state of the art image segmentation performance with advanced segmentation frameworks.
- • Extending *NATTEN*, NA’s CUDA extension for PyTorch, by adding dilation support, and bfloat16 utilization, allowing the research in this direction to be extended to other tasks and applications.

While the initial experiments with DiNAT already exhibit significant improvements in downstream vision tasks,

neither its performance nor applications stop here. NA’s local attention and DiNA’s sparse global attention complement each other: they can preserve locality, model longer-range inter-dependencies, expand the receptive field exponentially, and maintain a linear complexity. Their restriction of self attention can potentially improve convergence by avoiding self attention’s possible redundant interactions, such as those with repetitive, background, or distracting tokens [26, 36]. Combinations of local attention and sparse global attention can potentially empower various vision tasks and beyond. To support research in this direction, we open source our entire project, including our modified *NATTEN*, which can reduce runtime by orders of magnitude compared to naive implementations [15].

## 2. Related Work

We briefly review dot product self attention (DPSA), the Transformer [42], and Vision Transformer [13]. We then move on to localized self attention modules such as SASA [35], SWSA in Swin Transformer [29], and NA in Neighborhood Attention Transformer [15], and discuss their limitations, which are our motivation behind this work. Finally, we discuss previous uses of sparse attention mechanisms in language processing [1, 37] and vision [6, 20].

### 2.1. Self Attention

Vaswani et al. [42] define dot product attention as an operation between a query, and a set of key-value pairs. The dot product of the query and keys is scaled and sent through a softmax activation to produce attention weights. Said attention weights are then applied to the values:

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d}}\right)V, \quad (1)$$

where  $\sqrt{d}$  is the scaling parameter, and  $d$  is the key dimension. Dot product self attention is simply a case of this operation where the queries, keys, and values are all linear projections of the same input. Given an input  $X \in \mathbb{R}^{n \times d}$ , where  $n$  is the number of tokens and  $d$  is the embedding dimension, this operation has a complexity of  $\mathcal{O}(n^2d)$  and a space complexity of  $\mathcal{O}(n^2)$  for the attention weights (space depends on implementation [10]). Vision Transformer (ViT) [13] one of the earliest works applying a pure transformer encoder to vision, showed the power that a large scale self attention based model bears. Follow up works extended the study with minimal changes to training techniques [38], architectural changes [39], and applications to small data regimes [16]. Due to their quadratic time complexity, many works attempt to restrict the attention span in order to reduce compute, specifically when scaling to larger inputs, such as long documents in NLP [1], and large resolutions in vision [29]. Restricting self attention can be done in different patterns, one of which is localization.## 2.2. Local Attention

**Stand-Alone Self Attention (SASA).** SASA [35] is one of the earliest local attention mechanisms that was specifically designed to be used in vision models, years before ViT [13]. It sets the key-value pair to sliding windows over the feature map, therefore localizing attention for each query (pixel) to a window centered around it. Such an operation could easily replace convolutions in existing CNNs, such as ResNets, and theoretically even reduce computational complexity. Despite the promise it showed, the authors found that the resulting model runs slow, due to the inefficient implementation of this module. Works succeeding it therefore switched to alternative methods that could run more efficiently, such as blocked self attention in HaloNet [41], and Window Self Attention in Swin [29].

**Shifted Window Self Attention (SWSA).** Liu et al. [29] proposed Window Self Attention (WSA) and its shifted variant SWSA, and used them in their hierarchical model for vision, Swin Transformer. They pointed out the inefficiency of sliding-window methods such as SASA as one of their motivations behind developing Window Self Attention. The shifted variant (SWSA), as the name suggests, shifts pixels before the attention operation, and reverses the shift afterwards, to create a different window partitioning compared to the previous layer, which allows for out-of-window interactions that are crucial to a growing receptive field (see Fig. 3). Swin initially became the state of the art in object detection and semantic segmentation. It also inspired other works that extended it to different tasks beyond the ones explored in the paper, such as generation [52], restoration [25], masked image modeling [49], video action recognition [31], and more. Additionally, the followup model, Swin-V2 [28], became the new state of the art with their largest model. It is noteworthy that Swin-V2 utilizes much larger window sizes to achieve such performance, which in turn increase time complexity and memory usage.

**Neighborhood Attention (NA).** NA [15] was proposed as a simple sliding-window attention, which localizes self attention for each pixel to its nearest neighbors. NA shares the same time and space complexity and number of parameters to those of Swin’s WSA and SWSA, but instead operates in overlapping sliding windows, and therefore preserves translation equivariance. While NA’s sliding-window pattern is similar to SASA, its formulation of nearest neighbors makes it a direct restriction of self attention, and therefore NA, unlike SASA, approaches SA as its window size grows. A major challenge of sliding-window attention was the lack of efficient implementations, as no existing deep learning or CUDA libraries support such operations directly. Therefore, NA was introduced along with *NATTEN*, an extension with efficient CPU and GPU kernels that allow NA to outperform modules such

**Figure 3. Illustration of attention layers in Swin Transformer and DiNAT.** Swin divides inputs into non-overlapping windows and applies self attention to each window separately, and applies a pixel shift every other layer. Pixel-shifted layers mask attention weights between out-of-order regions, which restricts self attention to shifted subwindows. DiNAT applies Neighborhood Attention, a sliding window attention, and dilates it at every other layer.

as WSA/SWSA in terms of both speed and memory usage. The model Neighborhood Attention Transformer (NAT) is similar in its hierarchical design to Swin Transformer. The key differences, other than the attention modules, is that NAT utilizes overlapping convolutions in downsampling layers, as opposed to the patched ones used in Swin. As a result, to keep variants similar to Swin variants in terms of number of parameters and FLOPs, the models were made slightly deeper, with smaller inverted bottlenecks. NAT achieves superior results in image classification compared to Swin, and performs competitively on downstream tasks.

While local attention based models are able to perform well across different vision tasks due to its preservation of locality and efficiency, they fall short of capturing global context like self attention, which is also crucial to vision. Additionally, localized attention mechanisms utilize a smaller and slowly growing receptive field, similar to that of convolutions, compared to the full-sized receptive field in self attention. Besides self attention, several works also explored global receptive fields in vision, including but not limited to Non-local Neural Networks [45]. However, operations with unrestricted global receptive field usually suffer from high computational complexities compared to restricted ones, which can be local, or sparse.### 2.3. Sparse Attention

Child et al. [6] proposed Sparse Transformers, which in addition to scaling to much deeper variants, utilized a sparse-kernel attention mechanism. Through this, the model was able to train much more efficiently on longer sequences of data. There have been other works in sparse attention, such as Longformer [1], Routing Transformers [37], and CCNet [20], all of which share a common feature: reducing the cost of self attention in cases where longer sequences of tokens are inevitable, but a global context is still necessary. Longformer [1] specifically investigates using a combination of 1-D sliding window attention with and without dilation, along with global attention for specific tokens. This results in a model that is able to process long documents while maintaining the global context. CCNet [20] uses axial attention to improve semantic segmentation heads by introducing global context without the quadratic cost of unrestricted self attention. More recently, MaxViT [40] explored a hybrid model, which uses a combination of MBCnv, Window Attention [29], and sparse grid attention, obtaining high ImageNet accuracy. However, the resulting model yields higher complexity and lower throughput compared to Swin [29].

Even though such non-local and sparse restrictions of self attention have shown to be promising, they are not well-studied in the scope of hierarchical vision transformers. To expand the local receptive fields, and re-introduce global context into hierarchical vision transformers, we introduce Dilated Neighborhood Attention (DiNA), an extension of NA that spans neighborhoods over longer ranges by increasing the step size, while maintaining the overall attention span. DiNA can serve as a sparse and global operation and works most effectively when used in conjunction with NA as a local-only operation. We present an illustration of receptive fields in Fig. 4, where we compare fully connected layers to convolutions and dilated convolutions, and similarly self attention, to NA and DiNA. We provide empirical evidence for this claim with our hierarchical vision transformer, Dilated Neighborhood Attention Transformer (DiNAT).

## 3. Method

In this section, we define DiNA as an extension to NA, analyze its effect on the receptive field, and move on to our model, DiNAT. We also provide brief details on implementation, and integration with the existing *NATTEN* package.

### 3.1. Dilated Neighborhood Attention

For simplicity, we keep notations limited to single-dimensional NA and DiNA. Given input  $X \in \mathbb{R}^{n \times d}$ , whose rows are  $d$ -dimensional token vectors, and query and key

**Figure 4. A single-dimensional illustration of receptive fields in fully connected layers, convolutional layers, and different attention mechanisms.** NA and DiNA restrict self attention through sliding windows, similar to how convolutions and dilated convolutions restrict fully connected layers. These restrictions reduce computational burden, introduce useful inductive biases, and in some cases increase flexibility w.r.t. varying input sizes.

linear projections of  $X$ ,  $Q$  and  $K$ , and relative positional biases between any two tokens  $i$  and  $j$ ,  $B(i, j)$ , we define neighborhood attention weights for the  $i$ -th token with neighborhood size  $k$ ,  $\mathbf{A}_i^k$ , as the matrix multiplication of the  $i$ -th token’s query projection, and its  $k$  nearest neighboring tokens’ key projections:

$$\mathbf{A}_i^k = \begin{bmatrix} Q_i K_{\rho_1(i)}^T + B_{(i, \rho_1(i))} \\ Q_i K_{\rho_2(i)}^T + B_{(i, \rho_2(i))} \\ \vdots \\ Q_i K_{\rho_k(i)}^T + B_{(i, \rho_k(i))} \end{bmatrix}, \quad (2)$$

where  $\rho_j(i)$  denotes  $i$ ’s  $j$ -th nearest neighbor. We similarly define neighboring values,  $\mathbf{V}_i^k$ , as a matrix whose rows are the  $i$ -th token’s  $k$  nearest neighboring value projections:

$$\mathbf{V}_i^k = \left[ V_{\rho_1(i)}^T \quad V_{\rho_2(i)}^T \quad \dots \quad V_{\rho_k(i)}^T \right]^T, \quad (3)$$

where  $V$  is a linear projection of  $X$ . Neighborhood Attention output for the  $i$ -th token with neighborhood size  $k$  is then defined as:

$$\text{NA}_k(i) = \text{softmax} \left( \frac{\mathbf{A}_i^k}{\sqrt{d}} \right) \mathbf{V}_i^k, \quad (4)$$**ViT.** Complexity:  $\mathcal{O}(n^2d)$ , RF =  $n$ .

**Swin.** Complexity:  $\mathcal{O}(ndk)$ , RF =  $\ell k$ .

**ConvNeXt.** Complexity:  $\mathcal{O}(ndk)$ , RF =  $\ell(k-1)+1$ .

**NAT.** Complexity:  $\mathcal{O}(ndk)$ , RF =  $\ell(k-1)+1$ .

**DiNAT.** Complexity:  $\mathcal{O}(ndk)$ , RF  $\in [\ell(k-1)+1, k^\ell]$ .

**Figure 5. Receptive fields in ViT, Swin, ConvNeXt, NAT, and our DiNAT.** We also provide the complexity of each method’s primary operation.  $n$  denotes the number of tokens,  $d$  denotes the embedding dimension, and  $k$  denotes kernel/window size. All receptive fields are bounded by input size,  $n$ . DiNAT’s receptive field is flexible and ranges from linear,  $\ell(k-1)+1$ , to exponential growth,  $k^\ell$ .

where  $\sqrt{d}$  is the scaling parameter, and  $d$  is the embedding dimension. To extend this definition to DiNA, given a dilation value  $\delta$ , we simply define  $\rho_j^\delta(i)$  as token  $i$ ’s  $j$ -th nearest neighbor that satisfies:  $j \bmod \delta = i \bmod \delta$ . We can then define  $\delta$ -**dilated** neighborhood attention weights for the  $i$ -th token with neighborhood size  $k$ ,  $\mathbf{A}_i^{(k,\delta)}$ , as follows:

$$\mathbf{A}_i^{(k,\delta)} = \begin{bmatrix} Q_i K_{\rho_1^\delta(i)}^T + B_{(i,\rho_1^\delta(i))} \\ Q_i K_{\rho_2^\delta(i)}^T + B_{(i,\rho_2^\delta(i))} \\ \vdots \\ Q_i K_{\rho_k^\delta(i)}^T + B_{(i,\rho_k^\delta(i))} \end{bmatrix}. \quad (5)$$

We similarly define  $\delta$ -dilated neighboring values for the  $i$ -th token with neighborhood size  $k$ ,  $\mathbf{V}_i^{(k,\delta)}$ :

$$\mathbf{V}_i^{(k,\delta)} = \begin{bmatrix} V_{\rho_1^\delta(i)}^T & V_{\rho_2^\delta(i)}^T & \cdots & V_{\rho_k^\delta(i)}^T \end{bmatrix}^T. \quad (6)$$

DiNA output for the  $i$ -th token neighborhood size  $k$  is then defined as:

$$\text{DiNA}_k^\delta(i) = \text{softmax} \left( \frac{\mathbf{A}_i^{(k,\delta)}}{\sqrt{d_k}} \right) \mathbf{V}_i^{(k,\delta)}. \quad (7)$$

### 3.2. Choice of Dilation

DiNA introduces a key new architectural hyperparameter: per layer dilation values. We define the upper bound for dilation value to be  $\lfloor \frac{n}{k} \rfloor$ , where  $n$  is the number of tokens, and  $k$  is kernel/neighborhood size. This is simply to ensure exactly  $k$  dilated neighbors exist for each token. The lower bound is always 1, which would be equivalent to vanilla NA. Therefore, dilation value in each layer of the model will be an input-dependent hyperparameter, which can take any

<table border="1">
<thead>
<tr>
<th>Layer structure</th>
<th>Memory usage</th>
<th>FLOPs</th>
<th>Receptive Field</th>
</tr>
</thead>
<tbody>
<tr>
<td>●● DWSCConv-DWSCConv</td>
<td><math>d^2 + dk</math></td>
<td><math>nd^2 + ndk</math></td>
<td><math>\ell(k-1)+1</math></td>
</tr>
<tr>
<td>○○ WSA-WSA</td>
<td><math>3d^2 + nk</math></td>
<td><math>3nd^2 + 2ndk</math></td>
<td><math>k</math></td>
</tr>
<tr>
<td>○○ WSA-SWSA</td>
<td><math>3d^2 + nk</math></td>
<td><math>3nd^2 + 2ndk</math></td>
<td><math>\ell k</math></td>
</tr>
<tr>
<td>○○ NA-NA</td>
<td><math>3d^2 + nk</math></td>
<td><math>3nd^2 + 2ndk</math></td>
<td><math>\ell(k-1)+1</math></td>
</tr>
<tr>
<td>●● NA-DiNA</td>
<td><math>3d^2 + nk</math></td>
<td><math>3nd^2 + 2ndk</math></td>
<td><math>\in [\ell(k-1)+1, k^\ell]</math></td>
</tr>
<tr>
<td>●● SA-SA</td>
<td><math>3d^2 + n^2</math></td>
<td><math>3nd^2 + 2n^2d</math></td>
<td><math>n</math></td>
</tr>
</tbody>
</table>

**Table 1. Memory usage (weights), FLOPs, and receptive field sizes in different models.** Convolutions and NA expand receptive field linearly with model depth. Window Self Attention alone would suffer from a fixed-value receptive field, but the pixel shift in SWSA expands the receptive field linearly. NA and DiNA together can expand receptive fields *exponentially*. Self attention has the maximum receptive field, which comes at the expense of a quadratic computational cost. Note that the denoted receptive fields have an upper bound of  $n$ .

integer  $\delta \in [1, \lfloor \frac{n}{k} \rfloor]$ . Because dilation values are changeable, they provide a flexible receptive field (discussed in Sec. 3.3). It is not feasible to try out all possible combinations, therefore we explored a limited number of choices, which are discussed in Sec. 4.4.

### 3.3. Receptive Fields

We analyze DiNA’s receptive field, as it is important to understanding the power of DiNA, especially in comparison to other models. We present a comparison of receptive field sizes in different attention patterns in Tab. 1, along with FLOPs and memory usage. We also include depth-wise separable convolution (DWSCConv), the key component in ConvNeXt [30], for completeness.

We calculate receptive field size with respect to the number of layers,  $\ell$ , kernel size  $k$ , and number of tokens  $n$ .DiNAT Architecture

DiNAT Block

**Figure 6. An illustration of DiNAT’s architecture.** It downsamples inputs to a quarter of their original spatial resolution initially, and sends them through 4 levels of DiNA Transformer encoders. Feature maps are downsampled to half their spatial size and doubled in channels between levels. DiNAT layers are similar to most Transformers: Attention followed by an MLP with normalization and skip connections in between. It also switches between local NA and sparse global DiNA at every other layer (right).

Both convolutions and NA start out with a receptive field of size  $k$ , and expand by  $k - 1$  per layer (center pixel remains fixed). Swin Transformer’s Window Self Attention [29] on its own maintains a constant receptive field size, as the window partitioning prevents cross-window interactions, hence preventing receptive field expansion. Pixel shifted WSA resolves this issue, and expands receptive fields by exactly one window per layer, which is an expansion of  $k$  per layer.

It is worth noting that while Swin enjoys a slightly larger receptive field compared to NAT and ConvNeXt thanks to its special shifted window design, it breaks an important property: *symmetry*. Since Swin’s feature maps are partitioned into non-overlapping windows, pixels within the same window only attend to each other, regardless of their position (whether at center or corner), leading to some pixels seeing asymmetric context around them.

Unlike the fixed receptive field growth in NAT, Swin, and ConvNeXt, DiNA’s receptive field is flexible and changes with dilation. It can range anywhere from NAT’s original  $\ell(k - 1) + 1$  (all dilation values set to 1), to an exponentially growing receptive field of  $k^\ell$  (gradual dilation increase), which is one of the main reasons behind its power. Regardless of dilation, the first layer always yields a receptive field of size  $k$ . Given large enough dilation values, the preceding DiNA layer will yield a  $k$ -sized receptive field for each of the  $k$  in the DiNA layer, yielding a receptive field of size  $k^2$ . As a result, DiNA and NA combinations with optimal dilation values can potentially increase receptive field *exponentially* to  $k^\ell$ . This comes with no surprise,

<table border="1">
<thead>
<tr>
<th>Variant</th>
<th>Layers per level</th>
<th>Dim x Heads</th>
<th>MLP ratio</th>
<th># of Params</th>
<th>FLOPs</th>
</tr>
</thead>
<tbody>
<tr>
<td>• DiNAT-Mini</td>
<td>3, 4, 6, 5</td>
<td><math>32 \times 2</math></td>
<td>3</td>
<td>20 M</td>
<td>2.7 G</td>
</tr>
<tr>
<td>• DiNAT-Tiny</td>
<td>3, 4, 18, 5</td>
<td><math>32 \times 2</math></td>
<td>3</td>
<td>28 M</td>
<td>4.3 G</td>
</tr>
<tr>
<td>• DiNAT-Small</td>
<td>3, 4, 18, 5</td>
<td><math>32 \times 3</math></td>
<td>2</td>
<td>51 M</td>
<td>7.8 G</td>
</tr>
<tr>
<td>• DiNAT-Base</td>
<td>3, 4, 18, 5</td>
<td><math>32 \times 4</math></td>
<td>2</td>
<td>90 M</td>
<td>13.7 G</td>
</tr>
<tr>
<td>• DiNAT-Large</td>
<td>3, 4, 18, 5</td>
<td><math>32 \times 6</math></td>
<td>2</td>
<td>200 M</td>
<td>30.6 G</td>
</tr>
</tbody>
</table>

**Table 2. DiNAT variants.** In terms of architecture, DiNAT is identical to NAT, which follows Swin closely in overall design. Channels (heads and dim) double after every level. Kernel size is  $7^2$  in all variants.

as dilated convolutions have also been known for having an exponentially-growing receptive field size when using exponentially growing dilation values [50]. An illustration of the increased receptive field size is also presented in Fig. 5.

### 3.4. DiNAT

For a fair evaluation of DiNA’s performance, we design DiNAT to be identical to the original NAT model in terms of architecture and configuration. It uses two  $3 \times 3$  convolutional layers with  $2 \times 2$  strides initially, resulting in feature maps that are a quarter of the input resolution. It also uses a single  $3 \times 3$  convolution with  $2 \times 2$  strides to downsample between levels, which cut spatial resolution in half and double channels. Details are presented in Tab. 2. The key difference in DiNAT is that every other layer uses DiNA instead of NA. Dilation values for DiNA layers are set based on thetask and input resolution. For ImageNet-1k at  $224^2$  resolution, we set dilation values to 8, 4, 2, and 1 in levels one through four respectively. In downstream tasks, because of their larger resolution, we increase dilation values to beyond that. All dilation values and other relevant architecture details are presented in Tab. II.

### 3.5. Implementation

We implemented DiNA on top of the existing Neighborhood Attention Extension (*NATTEN*), allowing ease of use and identical memory usage to NA. The latest public version of the extension includes a more efficient “tiled” implementation of Neighborhood Attention, which is what allows it to compete with methods such as Swin in terms of speed. By adding a dilation element to all the existing CUDA kernels, and re-implementing the “tiled” kernel to support dilated memory format, we managed to implement DiNA without affecting the speed of the existing NA kernels. However, it should be noted that DiNA’s throughput will depend on dilation value, and is expected to be slightly slower than NA in practice. This is simply due to the break in memory access pattern, which would affect throughput overall (see Fig. Ia). We also note that these implementations are still fairly naive and don’t fully utilize newer architecture standards in CUDA, such as Tensor Cores, and are therefore only working as a proof of concept. Despite this limitation, models using NA and DiNA can achieve competitive throughput levels compared to other methods that mostly utilize convolutions, linear projections, and self attention, all of which run through NVIDIA libraries that fully utilize the aforementioned standards. More information on implementation is provided in Appendix A.

## 4. Experiments

We conducted extensive experiments to study the effects of our proposed DiNAT model over existing baselines. Similar to existing methods, we pre-train models on image classification (ImageNet-1K and ImageNet-22K [11]), and then transfer the learned weights to downstream vision tasks. We compare DiNAT to the original NAT model [15], Swin [29], and ConvNeXt [30]. We also pair our model with Mask2Former [5] and perform instance, semantic, and panoptic segmentation experiments.

### 4.1. Image Classification

We used the community standard for ImageNet training in PyTorch, `timm` [46] (Apache License v2), which now serves as the community standard for ImageNet training in PyTorch [33], to train our model on ImageNet-1k [11]. We use the same training configurations, regularization techniques, and augmentations (CutMix [51], Mixup [53], RandAugment [9], and Random Erasing [54]) and training techniques used in NAT [15] and Swin [29]. Models trained

<table border="1">
<thead>
<tr>
<th>Model</th>
<th># of Params</th>
<th>FLOPs</th>
<th>Thru. (img/sec)</th>
<th>Memory (GB)</th>
<th>Top-1 (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6" style="text-align: center;"><i>ImageNet-1K trained models</i></td>
</tr>
<tr>
<td>○ NAT-M</td>
<td>20 M</td>
<td>2.7 G</td>
<td>2132</td>
<td>2.4</td>
<td>81.8</td>
</tr>
<tr>
<td>● DiNAT-M</td>
<td>20 M</td>
<td>2.7 G</td>
<td>2080</td>
<td>2.4</td>
<td>81.8</td>
</tr>
<tr>
<td>○ Swin-T</td>
<td>28 M</td>
<td>4.5 G</td>
<td>1724</td>
<td>4.8</td>
<td>81.3</td>
</tr>
<tr>
<td>● ConvNeXt-T</td>
<td>28 M</td>
<td>4.5 G</td>
<td>2491</td>
<td>3.4</td>
<td>82.1</td>
</tr>
<tr>
<td>○ NAT-T</td>
<td>28 M</td>
<td>4.3 G</td>
<td>1537</td>
<td>2.5</td>
<td><b>83.2</b></td>
</tr>
<tr>
<td>● DiNAT-T</td>
<td>28 M</td>
<td>4.3 G</td>
<td>1500</td>
<td>2.5</td>
<td>82.7</td>
</tr>
<tr>
<td>○ Swin-S</td>
<td>50 M</td>
<td>8.7 G</td>
<td>1056</td>
<td>5.0</td>
<td>83.0</td>
</tr>
<tr>
<td>● ConvNeXt-S</td>
<td>50 M</td>
<td>8.7 G</td>
<td>1549</td>
<td>3.5</td>
<td>83.1</td>
</tr>
<tr>
<td>○ NAT-S</td>
<td>51 M</td>
<td>7.8 G</td>
<td>1049</td>
<td>3.7</td>
<td>83.7</td>
</tr>
<tr>
<td>● DiNAT-S</td>
<td>51 M</td>
<td>7.8 G</td>
<td>1058</td>
<td>3.7</td>
<td><b>83.8</b></td>
</tr>
<tr>
<td>○ Swin-B</td>
<td>88 M</td>
<td>15.4 G</td>
<td>774</td>
<td>6.7</td>
<td>83.5</td>
</tr>
<tr>
<td>● ConvNeXt-B</td>
<td>89 M</td>
<td>15.4 G</td>
<td>1107</td>
<td>4.8</td>
<td>83.8</td>
</tr>
<tr>
<td>○ NAT-B</td>
<td>90 M</td>
<td>13.7 G</td>
<td>781</td>
<td>5.0</td>
<td>84.3</td>
</tr>
<tr>
<td>● DiNAT-B</td>
<td>90 M</td>
<td>13.7 G</td>
<td>764</td>
<td>5.0</td>
<td><b>84.4</b></td>
</tr>
<tr>
<td colspan="6" style="text-align: center;"><i>ImageNet-22K pre-trained models</i></td>
</tr>
<tr>
<td>○ Swin-L</td>
<td>197 M</td>
<td>34.5 G</td>
<td>478</td>
<td>10.4</td>
<td>86.3</td>
</tr>
<tr>
<td>● ConvNeXt-L</td>
<td>198 M</td>
<td>34.4 G</td>
<td>643</td>
<td>7.5</td>
<td><b>86.6</b></td>
</tr>
<tr>
<td>● DiNAT-L</td>
<td>200 M</td>
<td>30.6 G</td>
<td>474</td>
<td>7.8</td>
<td><b>86.6</b></td>
</tr>
</tbody>
</table>

**Table 3. ImageNet-1K image classification performance at  $224^2$  resolution.** Throughput and peak memory usage are measured from forward passes with a batch size of 256 on a single NVIDIA A100 GPU.

<table border="1">
<thead>
<tr>
<th>Model</th>
<th>Win. Size</th>
<th># of Params</th>
<th>FLOPs</th>
<th>Thru. (img/sec)</th>
<th>Memory (GB)</th>
<th>Top-1 (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>○ Swin-L</td>
<td><math>12^2</math></td>
<td>197 M</td>
<td>104.0 G</td>
<td>169</td>
<td>32.7</td>
<td>87.3</td>
</tr>
<tr>
<td>● ConvNeXt-L</td>
<td><math>7^2</math></td>
<td>198 M</td>
<td>101.1 G</td>
<td>221</td>
<td>19.2</td>
<td><b>87.5</b></td>
</tr>
<tr>
<td>● DiNAT-L</td>
<td><math>7^2</math></td>
<td>200 M</td>
<td>89.7 G</td>
<td>161</td>
<td>20.1</td>
<td>87.4</td>
</tr>
<tr>
<td>● DiNAT-L</td>
<td><math>11^2</math></td>
<td>200 M</td>
<td>92.4 G</td>
<td>110</td>
<td>26.9</td>
<td><b>87.5</b></td>
</tr>
</tbody>
</table>

**Table 4. ImageNet-1K image classification performance at  $384^2$  resolution.** Throughput and peak memory usage are measured from forward passes with a batch size of 256 on a single NVIDIA A100 GPU.

on ImageNet-1K directly are trained for 300 epochs with a batch size of 1024, and use an iteration-wise cosine learning rate schedule and a 20 epoch warmup, with a base learning rate of  $1e-3$ , and weight decay rate of 0.05, cooled down for an additional 10 epochs. Larger variants are pre-trained on ImageNet-22K [11] for 90 epochs with a batch size of 4096, but use a linear learning rate schedule and a 5 epoch warmup, with a base learning rate of  $1e-3$ , and weight decay rate of 0.01, again following Swin [29]. We fine-tune models pre-trained on ImageNet-22K to ImageNet-1K for 30 epochs, with a batch size of 512, and a linear learning rate schedule with no warmup, and a base learning rate of  $5e-5$ , and weight decay rate of  $1e-4$ . Final ImageNet-1K validation set accuracy levels, along with number of learnable parameters, FLOPs, throughput, and memory usage are provided in Tabs. 3 and 4. The reason for providing both FLOPs and throughput is to point out the necessity in distinguishing theoretical computational requirements, versus ef-iciency in practice with each method’s available implementation. This is especially important in this case because NA and DiNA are based on from scratch implementations of the algorithms ( $\mathcal{NATTEN}$ ), and are not as well-optimized as ConvNeXt or Swin, which mostly run on native NVIDIA libraries designed for optimal throughput.

**ImageNet-1K.** DiNAT doesn’t show improvement over NAT in smaller variants. Improvement over NAT-Mini is less than 0.1%, and we found that while the Tiny variant converges faster than NAT-Tiny at first, it converges to a lower accuracy of 82.7%. We noticed that despite this, DiNAT consistently outperforms NAT across all four variants on downstream tasks. DiNAT shows a slight improvement of at least 0.1% over NAT on Small and Base variants.

**ImageNet-22K.** We pre-trained our Large variant on ImageNet-22K, and fine-tuned it to ImageNet-1K at both  $224^2$  and  $384^2$  resolutions. We found that our large variant can successfully outperform Swin-Large and match ConvNeXt-Large’s accuracy at  $224^2$  resolution. At  $384^2$ , our large variant exceeds its Swin counterpart’s reported accuracy without increasing its kernel size from  $7^2$  to  $12^2$ . Upon increasing the large variant’s kernel size to  $11^2$  and interpolating positional biases (similar to Swin), we see that our large variant matches ConvNeXt-Large’s accuracy as well. We note that NA/DiNA are in theory limited to odd-sized kernels, which is the reason behind picking  $11^2$  instead of  $12^2$ .

**Isotropic variants.** To further compare NA/DiNA to plain self attention, we also explore *isotropic* variants of NAT and DiNAT, similar to isotropic ConvNeXt [30] variants. These models simply follow ViT in design: a single Transformer encoder operating on feature maps with a fixed spatial size ( $14^2$ ), preceded by a single patch-and-embedding layer; they are not hierarchical transformers. To maintain fairness in comparison to self attention, we trained ViT models with relative positional biases (ViT<sup>+</sup>) to ensure the models are only different in attention patterns. Note that ViT variants with relative positional biases have previously been explored in timm [46], but we run our own to ensure similar training settings. We present a comparison of these models and their performance on ImageNet-1k in Tab. 5. We find that isotropic variants of both NAT and DiNAT exhibit only minor throughput improvements over ViT<sup>+</sup>, which can again be attributed to the lack of fully optimized implementations. Note that these variants reduce FLOPs to almost the same number as isotropic ConvNeXt variants. They also reduce memory usage compared to ViT<sup>+</sup> noticeably. As for performance, we observe that isotropic NAT variants result in a drop in performance compared to ViT<sup>+</sup>, which is to be expected since NAT has half the attention span as ViT<sup>+</sup>. However, we find that isotropic DiNAT variants significantly improve upon NAT’s isotropic variants, without

<table border="1">
<thead>
<tr>
<th>Model</th>
<th># of Params</th>
<th>FLOPs</th>
<th>Thru. (img/sec)</th>
<th>Memory (GB)</th>
<th>Top-1 (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>● ConvNeXt-S (<i>iso.</i>)</td>
<td>22 M</td>
<td>4.3 G</td>
<td>4327</td>
<td>1.2</td>
<td>79.7</td>
</tr>
<tr>
<td>○ NAT-S (<i>iso.</i>)</td>
<td>22 M</td>
<td>4.3 G</td>
<td>3255</td>
<td>1.3</td>
<td>80.0</td>
</tr>
<tr>
<td>● DiNAT-S (<i>iso.</i>)</td>
<td>22 M</td>
<td>4.3 G</td>
<td>3160</td>
<td>1.3</td>
<td>80.8</td>
</tr>
<tr>
<td>● ViT<sup>+</sup>-S</td>
<td>22 M</td>
<td>4.6 G</td>
<td>3086</td>
<td>1.9</td>
<td><b>81.2</b></td>
</tr>
<tr>
<td>● ConvNeXt-B (<i>iso.</i>)</td>
<td>87 M</td>
<td>16.9 G</td>
<td>1661</td>
<td>2.4</td>
<td>82.0</td>
</tr>
<tr>
<td>○ NAT-B (<i>iso.</i>)</td>
<td>86 M</td>
<td>16.9 G</td>
<td>1350</td>
<td>2.7</td>
<td>81.6</td>
</tr>
<tr>
<td>● DiNAT-B (<i>iso.</i>)</td>
<td>86 M</td>
<td>16.9 G</td>
<td>1316</td>
<td>2.7</td>
<td>82.1</td>
</tr>
<tr>
<td>● ViT<sup>+</sup>-B</td>
<td>86 M</td>
<td>17.5 G</td>
<td>1284</td>
<td>3.7</td>
<td><b>82.5</b></td>
</tr>
</tbody>
</table>

**Table 5. ImageNet-1K Top-1 validation accuracy comparison of ConvNeXt, NAT, and DiNAT’s isotropic variants to ViT.** To compare self attention and NA/DiNA fairly, we ran ViT<sup>+</sup>, which uses relative positional biases in attention layers, instead of the one-time absolute positional encoding in the original ViT. Throughput and peak memory usage are measured from forward passes with a batch size of 256 on a single NVIDIA A100 GPU.

<table border="1">
<thead>
<tr>
<th>Model</th>
<th>Layer structure</th>
<th>FLOPs</th>
<th>Top-1 (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>○ ○ NAT-S (<i>iso.</i>)</td>
<td>NA-NA</td>
<td>4.32 G</td>
<td>80.0</td>
</tr>
<tr>
<td>● ●</td>
<td>DiNA-DiNA</td>
<td>4.32 G</td>
<td>77.9</td>
</tr>
<tr>
<td>● ○</td>
<td>DiNA-NA</td>
<td>4.32 G</td>
<td>80.6</td>
</tr>
<tr>
<td>○ ● DiNAT-S (<i>iso.</i>)</td>
<td>NA-DiNA</td>
<td>4.32 G</td>
<td><b>80.8</b></td>
</tr>
<tr>
<td>○ ○</td>
<td>SA-NA</td>
<td>4.45 G</td>
<td>81.0</td>
</tr>
<tr>
<td>● ●</td>
<td>SA-DiNA</td>
<td>4.45 G</td>
<td>81.1</td>
</tr>
<tr>
<td>○ ●</td>
<td>NA-SA</td>
<td>4.45 G</td>
<td><b>81.2</b></td>
</tr>
<tr>
<td>● ●</td>
<td>DiNA-SA</td>
<td>4.45 G</td>
<td>80.9</td>
</tr>
<tr>
<td>● ● ViT<sup>+</sup>-S</td>
<td>SA-SA</td>
<td>4.58 G</td>
<td><b>81.2</b></td>
</tr>
</tbody>
</table>

**Table 6. Comparison of different layer structures in the isotropic variant.** We compare different attention mechanisms in detail by creating hybrid models with both SA and NA/DiNA.

increasing kernel size. This further supports our claim that a combination of NA and DiNA is more effective at producing an alternative to self attention than simply using NA throughout the model. To further study the effects of different attention mechanisms, and investigate whether or not a model fully based on self-attention always yields the best result, we experiment with hybrid isotropic models utilizing both NA/DiNA layers as well as self attention. We present those results in Tab. 6. We found that a small-scale (22M parameter) model with only half the layers performing self attention and the other half neighborhood attention can reach a similar accuracy as a similar model with all 12 layers utilizing self attention. We also found that changing the order of different attention layers can result in an approximately 0.2% change in accuracy.

## 4.2. Object Detection and Instance Segmentation

To explore DiNAT’s effectiveness in object detection and instance segmentation, we used its pre-trained weights as backbones for Mask R-CNN [18] and Cascade Mask R-CNN [2], and trained those models on MS-COCO [27].<table border="1">
<thead>
<tr>
<th>Backbone</th>
<th># of Params</th>
<th>FLOPs</th>
<th>Thru. (FPS)</th>
<th>AP<sup>b</sup></th>
<th>AP<sup>b</sup><sub>50</sub></th>
<th>AP<sup>b</sup><sub>75</sub></th>
<th>AP<sup>m</sup></th>
<th>AP<sup>m</sup><sub>50</sub></th>
<th>AP<sup>m</sup><sub>75</sub></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="10" style="text-align: center;"><i>Mask R-CNN - 3x schedule</i></td>
</tr>
<tr>
<td>○ NAT-M</td>
<td>40 M</td>
<td>225 G</td>
<td>54.1</td>
<td>46.5</td>
<td>68.1</td>
<td>51.3</td>
<td>41.7</td>
<td>65.2</td>
<td>44.7</td>
</tr>
<tr>
<td>● DiNAT-M</td>
<td>40 M</td>
<td>225 G</td>
<td>53.8</td>
<td><b>47.2</b></td>
<td><b>69.1</b></td>
<td><b>51.9</b></td>
<td><b>42.5</b></td>
<td><b>66.0</b></td>
<td><b>45.9</b></td>
</tr>
<tr>
<td>○ Swin-T</td>
<td>48 M</td>
<td>267 G</td>
<td>45.1</td>
<td>46.0</td>
<td>68.1</td>
<td>50.3</td>
<td>41.6</td>
<td>65.1</td>
<td>44.9</td>
</tr>
<tr>
<td>● ConvNeXt-T</td>
<td>48 M</td>
<td>262 G</td>
<td>52.0</td>
<td>46.2</td>
<td>67.0</td>
<td>50.8</td>
<td>41.7</td>
<td>65.0</td>
<td>44.9</td>
</tr>
<tr>
<td>○ NAT-T</td>
<td>48 M</td>
<td>258 G</td>
<td>44.5</td>
<td>47.7</td>
<td>69.0</td>
<td>52.6</td>
<td>42.6</td>
<td>66.1</td>
<td>45.9</td>
</tr>
<tr>
<td>● DiNAT-T</td>
<td>48 M</td>
<td>258 G</td>
<td>43.3</td>
<td><b>48.6</b></td>
<td><b>70.2</b></td>
<td><b>53.4</b></td>
<td><b>43.5</b></td>
<td><b>67.3</b></td>
<td><b>46.8</b></td>
</tr>
<tr>
<td>○ Swin-S</td>
<td>69 M</td>
<td>359 G</td>
<td>31.7</td>
<td>48.5</td>
<td>70.2</td>
<td>53.5</td>
<td>43.3</td>
<td>67.3</td>
<td>46.6</td>
</tr>
<tr>
<td>○ NAT-S</td>
<td>70 M</td>
<td>330 G</td>
<td>34.8</td>
<td>48.4</td>
<td>69.8</td>
<td>53.2</td>
<td>43.2</td>
<td>66.9</td>
<td>46.5</td>
</tr>
<tr>
<td>● DiNAT-S</td>
<td>70 M</td>
<td>330 G</td>
<td>35.3</td>
<td><b>49.3</b></td>
<td><b>70.8</b></td>
<td><b>54.2</b></td>
<td><b>44.0</b></td>
<td><b>68.0</b></td>
<td><b>47.4</b></td>
</tr>
<tr>
<td colspan="10" style="text-align: center;"><i>Cascade Mask R-CNN - 3x schedule</i></td>
</tr>
<tr>
<td>○ NAT-M</td>
<td>77 M</td>
<td>704 G</td>
<td>27.8</td>
<td>50.3</td>
<td>68.9</td>
<td>54.9</td>
<td>43.6</td>
<td>66.4</td>
<td>47.2</td>
</tr>
<tr>
<td>● DiNAT-M</td>
<td>77 M</td>
<td>704 G</td>
<td>27.6</td>
<td><b>51.2</b></td>
<td><b>69.8</b></td>
<td><b>55.7</b></td>
<td><b>44.4</b></td>
<td><b>67.3</b></td>
<td><b>47.8</b></td>
</tr>
<tr>
<td>○ Swin-T</td>
<td>86 M</td>
<td>745 G</td>
<td>25.1</td>
<td>50.4</td>
<td>69.2</td>
<td>54.7</td>
<td>43.7</td>
<td>66.6</td>
<td>47.3</td>
</tr>
<tr>
<td>● ConvNeXt-T</td>
<td>86 M</td>
<td>741 G</td>
<td>27.3</td>
<td>50.4</td>
<td>69.1</td>
<td>54.8</td>
<td>43.7</td>
<td>66.5</td>
<td>47.3</td>
</tr>
<tr>
<td>○ NAT-T</td>
<td>85 M</td>
<td>737 G</td>
<td>24.9</td>
<td>51.4</td>
<td>70.0</td>
<td>55.9</td>
<td>44.5</td>
<td>67.6</td>
<td>47.9</td>
</tr>
<tr>
<td>● DiNAT-T</td>
<td>85 M</td>
<td>737 G</td>
<td>25.0</td>
<td><b>52.2</b></td>
<td><b>71.0</b></td>
<td><b>56.8</b></td>
<td><b>45.1</b></td>
<td><b>68.3</b></td>
<td><b>48.8</b></td>
</tr>
<tr>
<td>○ Swin-S</td>
<td>107 M</td>
<td>838 G</td>
<td>20.3</td>
<td>51.8</td>
<td>70.4</td>
<td>56.3</td>
<td>44.7</td>
<td>67.9</td>
<td>48.5</td>
</tr>
<tr>
<td>● ConvNeXt-S</td>
<td>108 M</td>
<td>827 G</td>
<td>23.0</td>
<td>51.9</td>
<td>70.8</td>
<td>56.5</td>
<td>45.0</td>
<td>68.4</td>
<td>49.1</td>
</tr>
<tr>
<td>○ NAT-S</td>
<td>108 M</td>
<td>809 G</td>
<td>21.7</td>
<td>52.0</td>
<td>70.4</td>
<td>56.3</td>
<td>44.9</td>
<td>68.1</td>
<td>48.6</td>
</tr>
<tr>
<td>● DiNAT-S</td>
<td>108 M</td>
<td>809 G</td>
<td>21.8</td>
<td><b>52.9</b></td>
<td><b>71.8</b></td>
<td><b>57.6</b></td>
<td><b>45.8</b></td>
<td><b>69.3</b></td>
<td><b>49.9</b></td>
</tr>
<tr>
<td>○ Swin-B</td>
<td>145 M</td>
<td>982 G</td>
<td>17.3</td>
<td>51.9</td>
<td>70.9</td>
<td>56.5</td>
<td>45.0</td>
<td>68.4</td>
<td>48.7</td>
</tr>
<tr>
<td>● ConvNeXt-B</td>
<td>146 M</td>
<td>964 G</td>
<td>19.5</td>
<td>52.7</td>
<td>71.3</td>
<td>57.2</td>
<td>45.6</td>
<td>68.9</td>
<td>49.5</td>
</tr>
<tr>
<td>○ NAT-B</td>
<td>147 M</td>
<td>931 G</td>
<td>18.6</td>
<td>52.3</td>
<td>70.9</td>
<td>56.9</td>
<td>45.1</td>
<td>68.3</td>
<td>49.1</td>
</tr>
<tr>
<td>● DiNAT-B</td>
<td>147 M</td>
<td>931 G</td>
<td>18.5</td>
<td><b>53.4</b></td>
<td><b>72.1</b></td>
<td><b>58.2</b></td>
<td><b>46.2</b></td>
<td><b>69.7</b></td>
<td><b>50.2</b></td>
</tr>
<tr>
<td>○ Swin-L*<sup>‡</sup></td>
<td>253 M</td>
<td>1393 G</td>
<td>12.9</td>
<td>53.7</td>
<td>72.2</td>
<td>58.7</td>
<td>46.4</td>
<td>69.9</td>
<td>50.7</td>
</tr>
<tr>
<td>● ConvNeXt-L*<sup>‡</sup></td>
<td>253 M</td>
<td>1354 G</td>
<td>14.8</td>
<td>54.8</td>
<td>73.8</td>
<td>59.8</td>
<td>47.6</td>
<td>71.3</td>
<td>51.7</td>
</tr>
<tr>
<td>● DiNAT-L*<sup>‡</sup></td>
<td>258 M</td>
<td>1276 G</td>
<td>14.0</td>
<td><b>55.3</b></td>
<td><b>74.3</b></td>
<td><b>60.2</b></td>
<td><b>47.8</b></td>
<td><b>71.8</b></td>
<td><b>52.0</b></td>
</tr>
</tbody>
</table>

**Table 7. COCO object detection and instance segmentation performance.** <sup>‡</sup>indicates that the model was pre-trained on ImageNet-22K. \*Swin-L was not reported with Cascade Mask R-CNN, therefore we trained it with their official checkpoint. Throughput is measured on a single NVIDIA A100 GPU.

We followed NAT [15] and Swin [29]’s training settings in `mmdetection` [4] (Apache License v2), and trained with the same accelerated  $3\times$  LR schedule. The results are presented in Tab. 7. We observe that DiNAT consistently shows noticeable improvement over NAT, with little-to-no drop in throughput. There are even instances where DiNAT even surpasses NAT’s throughput, but within the margin of error. Additionally, we observe that this improvement over NAT pushes DiNAT ahead of ConvNeXt [30]. At scale, we see DiNAT continues to outperform both Swin and ConvNeXt with ImageNet-22K pre-training.

### 4.3. Semantic Segmentation

We also trained UPerNet [48] with our DiNAT as the backbone on ADE20K [55], with ImageNet-pre-trained backbones. We followed NAT’s `mmsegmentation` [7] (Apache License v2) configurations, itself following Swin’s configuration for training ADE20K. The results are presented in Tab. 8. We find that DiNAT exhibits a noticeable improvement over the original NAT model. DiNAT also maintains its place ahead of both models at scale with ImageNet-22K pre-training.

<table border="1">
<thead>
<tr>
<th>Backbone</th>
<th>Res.</th>
<th># of Params</th>
<th>FLOPs</th>
<th>Thru. (FPS)</th>
<th>mIoU</th>
</tr>
</thead>
<tbody>
<tr>
<td>○ NAT-M</td>
<td>2048 × 512</td>
<td>50 M</td>
<td>900 G</td>
<td>24.5</td>
<td>46.4</td>
</tr>
<tr>
<td>● DiNAT-M</td>
<td>2048 × 512</td>
<td>50 M</td>
<td>900 G</td>
<td>24.2</td>
<td><b>47.2</b></td>
</tr>
<tr>
<td>○ Swin-T</td>
<td>2048 × 512</td>
<td>60 M</td>
<td>946 G</td>
<td>21.3</td>
<td>45.8</td>
</tr>
<tr>
<td>● ConvNeXt-T</td>
<td>2048 × 512</td>
<td>60 M</td>
<td>939 G</td>
<td>23.3</td>
<td>46.7</td>
</tr>
<tr>
<td>○ NAT-T</td>
<td>2048 × 512</td>
<td>58 M</td>
<td>934 G</td>
<td>21.4</td>
<td>48.4</td>
</tr>
<tr>
<td>● DiNAT-T</td>
<td>2048 × 512</td>
<td>58 M</td>
<td>934 G</td>
<td>21.3</td>
<td><b>48.8</b></td>
</tr>
<tr>
<td>○ Swin-S</td>
<td>2048 × 512</td>
<td>81 M</td>
<td>1040 G</td>
<td>17.0</td>
<td>49.5</td>
</tr>
<tr>
<td>● ConvNeXt-S</td>
<td>2048 × 512</td>
<td>82 M</td>
<td>1027 G</td>
<td>19.1</td>
<td>49.6</td>
</tr>
<tr>
<td>○ NAT-S</td>
<td>2048 × 512</td>
<td>82 M</td>
<td>1010 G</td>
<td>17.9</td>
<td>49.5</td>
</tr>
<tr>
<td>● DiNAT-S</td>
<td>2048 × 512</td>
<td>82 M</td>
<td>1010 G</td>
<td>18.1</td>
<td><b>49.9</b></td>
</tr>
<tr>
<td>○ Swin-B</td>
<td>2048 × 512</td>
<td>121 M</td>
<td>1188 G</td>
<td>14.6</td>
<td>49.7</td>
</tr>
<tr>
<td>● ConvNeXt-B</td>
<td>2048 × 512</td>
<td>122 M</td>
<td>1170 G</td>
<td>16.4</td>
<td>49.9</td>
</tr>
<tr>
<td>○ NAT-B</td>
<td>2048 × 512</td>
<td>123 M</td>
<td>1137 G</td>
<td>15.6</td>
<td>49.7</td>
</tr>
<tr>
<td>● DiNAT-B</td>
<td>2048 × 512</td>
<td>123 M</td>
<td>1137 G</td>
<td>15.4</td>
<td><b>50.4</b></td>
</tr>
<tr>
<td>○ Swin-L<sup>†‡</sup></td>
<td>2560 × 640</td>
<td>234 M</td>
<td>2585 G</td>
<td>8.5</td>
<td>53.5</td>
</tr>
<tr>
<td>● ConvNeXt-L<sup>†‡</sup></td>
<td>2560 × 640</td>
<td>235 M</td>
<td>2458 G</td>
<td>9.6</td>
<td>53.7</td>
</tr>
<tr>
<td>● DiNAT-L<sup>†‡</sup></td>
<td>2560 × 640</td>
<td>238 M</td>
<td>2335 G</td>
<td>9.0</td>
<td><b>54.9</b></td>
</tr>
</tbody>
</table>

**Table 8. ADE20K semantic segmentation performance.** <sup>‡</sup>indicates that the model was pre-trained on ImageNet-22K. <sup>†</sup>indicates increased window size from the default  $7^2$  to  $12^2$ . Throughput is measured on a single NVIDIA A100 GPU.

### 4.4. Ablation study

In this section, we aim to study DiNAT in more depth by analyzing the effects of: dilation values, NA-DiNA order, kernel sizes, and test-time changes in dilation.

**Dilation values.** In Tab. 9, we present models with different dilation values, and their effect on classification, detection, instance segmentation and semantic segmentation performance levels. Note that the increased dilation (16, 8, 4, 2) is applicable to downstream tasks only, because in theory input feature maps should be larger than or equal to the product of kernel size and dilation. As a result, “8, 4, 2, 1” is the maximum applicable dilation to ImageNet at  $224 \times 224$  resolution. Depending on image resolution, even higher dilation values are possible. We explored a “dynamic” dilation value, where DiNA layers apply the maximum possible dilation, which is the floor of resolution divided by kernel size (“Maximum” in Tab. 9). We finally choose settle on “gradual” dilation (see illustration in Fig. 4), in which we gradually increase dilation to the maximum level defined. For instance, if maximum dilation for a specific level to 8, its layers will have dilation values 1, 2, 1, 4, 1, 6, 1, 8 (refer to Appendix B for details).

**NA-DiNA vs. DiNA-NA.** We also experimented with models with DiNA layers before NA layers, as opposed to our final NA before DiNA choice. While the local-global order (NA-DiNA) was our initial choice, we’ve also found it to be the more effective choice. We also tried a model with only DiNA modules, and found that it performs sig-<table border="1">
<thead>
<tr>
<th>Model</th>
<th>Dilation per level</th>
<th>ImageNet Top-1 (%)</th>
<th>MSCOCO AP<sup>b</sup></th>
<th>MSCOCO AP<sup>m</sup></th>
<th>ADE20K mIoU</th>
</tr>
</thead>
<tbody>
<tr>
<td>○ NAT-Tiny</td>
<td>1, 1, 1, 1</td>
<td>83.2</td>
<td>47.7</td>
<td>42.6</td>
<td>48.4</td>
</tr>
<tr>
<td>● DiNAT-Tiny</td>
<td>8, 4, 2, 1</td>
<td>82.7</td>
<td>48.0</td>
<td>42.9</td>
<td>48.5</td>
</tr>
<tr>
<td>● DiNAT-Tiny</td>
<td>16, 8, 4, 2</td>
<td>-</td>
<td>48.3</td>
<td>43.4</td>
<td>48.5</td>
</tr>
<tr>
<td>● DiNAT-Tiny</td>
<td>Maximum</td>
<td>82.7</td>
<td><b>48.6</b></td>
<td><b>43.5</b></td>
<td>48.7</td>
</tr>
<tr>
<td>● DiNAT-Tiny</td>
<td>Gradual</td>
<td>-</td>
<td><b>48.6</b></td>
<td><b>43.5</b></td>
<td><b>48.8</b></td>
</tr>
</tbody>
</table>

**Table 9. Dilation impact on performance.** Dilation values beyond "8, 4, 2, 1" are only applicable to downstream tasks, as their larger resolution allows for it. Maximum dilation indicates it is set to the maximum possible value based on input size. It would be the same as "8, 4, 2, 1" for ImageNet. Gradual dilation indicates that dilation values in DiNA layers increase gradually.

<table border="1">
<thead>
<tr>
<th>Variant</th>
<th>Layer structure</th>
<th>ImageNet Top-1 (%)</th>
<th>MSCOCO AP<sup>b</sup></th>
<th>MSCOCO AP<sup>m</sup></th>
<th>ADE20K mIoU</th>
</tr>
</thead>
<tbody>
<tr>
<td>○○ NAT-Tiny</td>
<td>NA-NA</td>
<td><b>83.2</b></td>
<td>47.7</td>
<td>42.6</td>
<td>48.4</td>
</tr>
<tr>
<td>●● DiNAT-Tiny</td>
<td>NA-DiNA</td>
<td>82.7</td>
<td>48.3</td>
<td>43.4</td>
<td><b>48.5</b></td>
</tr>
<tr>
<td>●○</td>
<td>DiNA-NA</td>
<td>82.6</td>
<td><b>48.5</b></td>
<td><b>43.5</b></td>
<td>47.9</td>
</tr>
<tr>
<td>●●</td>
<td>DiNA-DiNA</td>
<td>82.2</td>
<td>44.9</td>
<td>40.5</td>
<td>45.8</td>
</tr>
</tbody>
</table>

**Table 10. Layer structure impact on performance.** Our final model has the local-global (NA-DiNA) order.

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Win. size</th>
<th colspan="2">ImageNet</th>
<th colspan="3">MSCOCO</th>
<th colspan="2">ADE20K</th>
</tr>
<tr>
<th>Top-1</th>
<th>Thru.</th>
<th>AP<sup>b</sup></th>
<th>AP<sup>m</sup></th>
<th>Thru.</th>
<th>mIoU</th>
<th>Thru.</th>
</tr>
</thead>
<tbody>
<tr>
<td>○ NAT-T</td>
<td>5<sup>2</sup></td>
<td><b>81.6</b></td>
<td>1810 imgs/sec</td>
<td>46.8</td>
<td>42.0</td>
<td>45.5 fps</td>
<td>46.3</td>
<td>22.9 fps</td>
</tr>
<tr>
<td>● DiNAT-T</td>
<td>5<sup>2</sup></td>
<td>81.3</td>
<td>1777 imgs/sec</td>
<td><b>47.6</b></td>
<td><b>42.7</b></td>
<td>45.6 fps</td>
<td><b>46.4</b></td>
<td>22.7 fps</td>
</tr>
<tr>
<td>○ NAT-T</td>
<td>7<sup>2</sup></td>
<td><b>83.2</b></td>
<td>1537 imgs/sec</td>
<td>47.7</td>
<td>42.6</td>
<td>44.5 fps</td>
<td>48.4</td>
<td>21.4 fps</td>
</tr>
<tr>
<td>● DiNAT-T</td>
<td>7<sup>2</sup></td>
<td>82.7</td>
<td>1500 imgs/sec</td>
<td><b>48.3</b></td>
<td><b>43.4</b></td>
<td>43.3 fps</td>
<td><b>48.5</b></td>
<td>21.3 fps</td>
</tr>
<tr>
<td>○ NAT-T</td>
<td>9<sup>2</sup></td>
<td><b>83.1</b></td>
<td>1253 imgs/sec</td>
<td>48.5</td>
<td>43.3</td>
<td>39.4 fps</td>
<td>48.1</td>
<td>20.2 fps</td>
</tr>
<tr>
<td>● DiNAT-T</td>
<td>9<sup>2</sup></td>
<td><b>83.1</b></td>
<td>1235 imgs/sec</td>
<td><b>48.8</b></td>
<td><b>43.5</b></td>
<td>39.2 fps</td>
<td><b>48.4</b></td>
<td>20.0 fps</td>
</tr>
</tbody>
</table>

**Table 11. Kernel size impact on performance.** Note that we set dilation to the maximum values possible in each block based on the default resolutions. Therefore, the variant with kernel size 5 has larger dilation values compared to the one with kernel size 7.

nificantly worse than other combinations. This highlights the importance of having a combination of both local and sparse global attention patterns in the model. The results are summarized in Tab. 10.

**Kernel size.** We study the effect of kernel size on model performance in Tab. 11. We observed that a DiNAT-Tiny sees a significant decay in performance with a smaller kernel size across all three tasks. However, we find increasing kernel size beyond the default 7×7 does not result in a significant increase in return.

**Test-time dilation changes.** We present an analysis of sensitivity to dilation values, in which we attempt different dilation values on already trained models, and evaluate their performance. This can be particularly important to cases with varying resolutions, i.e. multi-scale testing. For DiNAT to be at its best, dilation level needs to be a near-

<table border="1">
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="2">Dilation</th>
<th rowspan="2">ImageNet Top-1 (%)</th>
<th colspan="2">MSCOCO</th>
<th rowspan="2">ADE20K mIoU</th>
</tr>
<tr>
<th>Train</th>
<th>Test</th>
<th>AP<sup>b</sup></th>
<th>AP<sup>m</sup></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">○ NAT-T</td>
<td>1, 1, 1, 1</td>
<td>1, 1, 1, 1</td>
<td>83.2</td>
<td>47.7</td>
<td>42.6</td>
<td>48.4</td>
</tr>
<tr>
<td>1, 1, 1, 1</td>
<td>8, 4, 2, 1</td>
<td>81.0</td>
<td>42.6</td>
<td>39.5</td>
<td>46.3</td>
</tr>
<tr>
<td>1, 1, 1, 1</td>
<td>16, 8, 4, 2</td>
<td>-</td>
<td>36.0</td>
<td>34.4</td>
<td>40.2</td>
</tr>
<tr>
<td>1, 1, 1, 1</td>
<td>Maximum</td>
<td>-</td>
<td>31.7</td>
<td>30.7</td>
<td>38.2</td>
</tr>
<tr>
<td rowspan="4">● DiNAT-T</td>
<td>8, 4, 2, 1</td>
<td>1, 1, 1, 1</td>
<td>78.2</td>
<td>43.0</td>
<td>38.6</td>
<td>41.5</td>
</tr>
<tr>
<td>8, 4, 2, 1</td>
<td>8, 4, 2, 1</td>
<td>82.7</td>
<td>48.0</td>
<td>42.9</td>
<td>48.5</td>
</tr>
<tr>
<td>8, 4, 2, 1</td>
<td>16, 8, 4, 2</td>
<td>-</td>
<td>45.6</td>
<td>41.3</td>
<td>47.1</td>
</tr>
<tr>
<td>8, 4, 2, 1</td>
<td>Maximum</td>
<td>-</td>
<td>40.2</td>
<td>37.3</td>
<td>45.8</td>
</tr>
<tr>
<td rowspan="4">● DiNAT-T</td>
<td>16, 8, 4, 2</td>
<td>1, 1, 1, 1</td>
<td>-</td>
<td>29.0</td>
<td>26.7</td>
<td>26.2</td>
</tr>
<tr>
<td>16, 8, 4, 2</td>
<td>8, 4, 2, 1</td>
<td>-</td>
<td>42.6</td>
<td>38.6</td>
<td>43.3</td>
</tr>
<tr>
<td>16, 8, 4, 2</td>
<td>16, 8, 4, 2</td>
<td>-</td>
<td>48.3</td>
<td>43.4</td>
<td>48.5</td>
</tr>
<tr>
<td>16, 8, 4, 2</td>
<td>Maximum</td>
<td>-</td>
<td>47.4</td>
<td>42.5</td>
<td>48.6</td>
</tr>
</tbody>
</table>

**Table 12. Test time dilation change and its impact on performance.** Dilation values larger than 8, 4, 2, 1 are inapplicable to ImageNet at 224<sup>2</sup>.

maximum number to expand attention to a longer range. The results are presented in Tab. 12.

## 4.5. Image segmentation with Mask2Former

To analyze DiNAT’s segmentation performance further, we conducted experiments with Mask2Former [5]. Mask2Former is an attention-based segmentation architecture, which can be trained on instance segmentation, semantic segmentation, and panoptic segmentation. It set a new state-of-the-art score for panoptic and instance segmentation on MS-COCO, as well as semantic segmentation on ADE20K. Mask2Former additionally used Swin-Large as the backbone, making it the perfect candidate for this experiment. We trained Mask2Former on MS-COCO [27], ADE20K [55], and Cityscapes [8], on all three segmentation objectives (instance, semantic, and panoptic), by simply replacing the Swin-Large backbone in a fork of their original repository. Following their reported environment, we used PyTorch 1.9 with Detectron2 [47]. We present instance segmentation results in Tab. 13, semantic segmentation results in Tab. 14, and panoptic segmentation results in Tab. 15. We note that DiNAT-L is using an 11<sup>2</sup> kernel size, instead of Swin-L’s 12<sup>2</sup>, since even-sized windows break the symmetry in NA and are therefore not defined.

DiNAT-L outperforms Swin-L on all three tasks and datasets. It also sets new state of the art records for image segmentation without using extra data. According to PapersWithCode leaderboards, DiNAT-L with Mask2Former is the SOTA panoptic segmentation on ADE20K and MS-COCO, and instance segmentation on ADE20K and Cityscapes. It also ties with the current SOTA on ADE20K, and ranks second on Cityscapes semantic segmentation (previous SOTA on both is SeMask [21]).<table border="1">
<thead>
<tr>
<th>Backbone</th>
<th>Win. Size</th>
<th># of Params</th>
<th>FLOPs</th>
<th>AP</th>
<th>AP<sup>50</sup></th>
<th>AP<sup>S</sup></th>
<th>AP<sup>M</sup></th>
<th>AP<sup>L</sup></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="9" style="text-align: center;"><i>MS-COCO</i></td>
</tr>
<tr>
<td>○ Swin-L</td>
<td>12 × 12</td>
<td>216 M</td>
<td>641 G</td>
<td>50.1</td>
<td>-</td>
<td>29.9</td>
<td>53.9</td>
<td><b>72.1</b></td>
</tr>
<tr>
<td>● DiNAT-L</td>
<td>11 × 11</td>
<td>220 M</td>
<td>522 G</td>
<td><b>50.8</b></td>
<td><b>75.0</b></td>
<td><b>30.9</b></td>
<td><b>54.7</b></td>
<td><b>72.1</b></td>
</tr>
<tr>
<td colspan="9" style="text-align: center;"><i>ADE20K</i></td>
</tr>
<tr>
<td>○ Swin-L</td>
<td>12 × 12</td>
<td>216 M</td>
<td>654 G</td>
<td>34.9</td>
<td>-</td>
<td><b>16.3</b></td>
<td><b>40.0</b></td>
<td>54.7</td>
</tr>
<tr>
<td>● DiNAT-L</td>
<td>11 × 11</td>
<td>220 M</td>
<td>535 G</td>
<td><b>35.4</b></td>
<td>-</td>
<td><b>16.3</b></td>
<td>39.0</td>
<td><b>55.5</b></td>
</tr>
<tr>
<td colspan="9" style="text-align: center;"><i>Cityscapes</i></td>
</tr>
<tr>
<td>○ Swin-L</td>
<td>12 × 12</td>
<td>216 M</td>
<td>641 G</td>
<td>43.7</td>
<td>71.4</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>● DiNAT-L</td>
<td>11 × 11</td>
<td>220 M</td>
<td>522 G</td>
<td><b>45.1</b></td>
<td><b>72.6</b></td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
</tbody>
</table>

**Table 13. Instance segmentation performance with Mask2Former.** All backbones were pre-trained on ImageNet-22K. FLOPs are reported with respect to resolution 800<sup>2</sup>.

<table border="1">
<thead>
<tr>
<th rowspan="2">Backbone</th>
<th rowspan="2">Win. Size</th>
<th rowspan="2"># of Params</th>
<th rowspan="2">FLOPs</th>
<th colspan="2">mIoU</th>
</tr>
<tr>
<th>single scale</th>
<th>multi scale</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6" style="text-align: center;"><i>ADE20K</i></td>
</tr>
<tr>
<td>○ Swin-L</td>
<td>12 × 12</td>
<td>215 M</td>
<td>636 G</td>
<td>56.1</td>
<td>57.3</td>
</tr>
<tr>
<td>● DiNAT-L</td>
<td>11 × 11</td>
<td>220 M</td>
<td>518 G</td>
<td><b>57.3</b></td>
<td><b>58.1</b></td>
</tr>
<tr>
<td colspan="6" style="text-align: center;"><i>Cityscapes</i></td>
</tr>
<tr>
<td>○ Swin-L</td>
<td>12 × 12</td>
<td>215 M</td>
<td>627 G</td>
<td>83.3</td>
<td>84.3</td>
</tr>
<tr>
<td>● DiNAT-L</td>
<td>11 × 11</td>
<td>220 M</td>
<td>509 G</td>
<td><b>83.9</b></td>
<td><b>84.5</b></td>
</tr>
</tbody>
</table>

**Table 14. Semantic segmentation performance with Mask2Former.** All backbones were pre-trained on ImageNet-22K. FLOPs are reported with respect to resolution 800<sup>2</sup>.

<table border="1">
<thead>
<tr>
<th>Backbone</th>
<th>Win. Size</th>
<th># of Params</th>
<th>FLOPs</th>
<th>PQ</th>
<th>PQ<sup>Th</sup></th>
<th>PQ<sup>St</sup></th>
<th>AP<sup>Th</sup><sub>pan</sub></th>
<th>mIoU<sub>pan</sub></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="9" style="text-align: center;"><i>MS-COCO</i></td>
</tr>
<tr>
<td>○ Swin-L</td>
<td>12 × 12</td>
<td>216 M</td>
<td>658 G</td>
<td>57.8</td>
<td>64.2</td>
<td>48.1</td>
<td>48.6</td>
<td>67.4</td>
</tr>
<tr>
<td>● DiNAT-L</td>
<td>11 × 11</td>
<td>220 M</td>
<td>540 G</td>
<td><b>58.5</b></td>
<td><b>64.9</b></td>
<td><b>48.8</b></td>
<td><b>49.2</b></td>
<td><b>68.3</b></td>
</tr>
<tr>
<td colspan="9" style="text-align: center;"><i>ADE20K</i></td>
</tr>
<tr>
<td>○ Swin-L</td>
<td>12 × 12</td>
<td>216 M</td>
<td>660 G</td>
<td>48.1</td>
<td>-</td>
<td>-</td>
<td>34.2</td>
<td>54.5</td>
</tr>
<tr>
<td>● DiNAT-L</td>
<td>11 × 11</td>
<td>220 M</td>
<td>542 G</td>
<td><b>49.4</b></td>
<td>-</td>
<td>-</td>
<td><b>35.0</b></td>
<td><b>56.3</b></td>
</tr>
<tr>
<td colspan="9" style="text-align: center;"><i>Cityscapes</i></td>
</tr>
<tr>
<td>○ Swin-L</td>
<td>12 × 12</td>
<td>216 M</td>
<td>643 G</td>
<td>66.6</td>
<td>-</td>
<td>-</td>
<td>43.6</td>
<td>82.9</td>
</tr>
<tr>
<td>● DiNAT-L</td>
<td>11 × 11</td>
<td>220 M</td>
<td>525 G</td>
<td><b>67.2</b></td>
<td>-</td>
<td>-</td>
<td><b>44.5</b></td>
<td><b>83.4</b></td>
</tr>
</tbody>
</table>

**Table 15. Panoptic segmentation performance with Mask2Former.** All backbones were pre-trained on ImageNet-22K. FLOPs are reported with respect to resolution 800<sup>2</sup>.

## 5. Conclusion

Local attention modules are effective at reducing complexity, and are crucial when working with a hierarchical model that gradually downsamples inputs. Nevertheless, they cannot capture longer range inter-dependencies as well as global self attention, unless their receptive field size is increased, which defeats their initial purpose of ef-

iciency and tractability. In this paper, we propose DiNA, a natural extension to NA that expands its local attention to sparse global attention at no additional cost. We build DiNAT with combinations of NA and DiNA, and show that it can improve performance significantly, especially in downstream tasks, without introducing any additional computational burden. Paired with new segmentation frameworks, our model achieves state-of-the-art image semantic, instance, and panoptic segmentation performance While our experiments give insight into the power behind such flexible attention modules, neither their performance nor efficiency stop here. We believe that combinations of NA and DiNA will be able to empower various models in vision and beyond, wherever locality and global context matter. We open source our entire project, including our extension to *NATTEN*, and will continue to support it as a toolkit for the community to allow easy experimentation with sparse sliding-window attention.

**Acknowledgments.** We thank Picsart AI Research (PAIR), Meta/Facebook AI, and Intelligence Advanced Research Projects Activity (IARPA) for their generous support that made this work possible.

## References

1. [1] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. *arXiv:2004.05150*, 2020. [3](#), [5](#)
2. [2] Zhaowei Cai and Nuno Vasconcelos. Cascade r-cnn: Delving into high quality object detection. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018. [9](#)
3. [3] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In *European Conference on Computer Vision (ECCV)*, 2020. [2](#)
4. [4] Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, et al. Mmdetection: Open mmlab detection toolbox and benchmark. *arXiv:1906.07155*, 2019. [10](#)
5. [5] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022. [8](#), [11](#)
6. [6] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. *arXiv:1904.10509*, 2019. [2](#), [3](#), [5](#)
7. [7] MMSegmentation Contributors. MMSegmentation: Open-mmlab semantic segmentation toolbox and benchmark. <https://github.com/open-mmlab/mmsegmentation>, 2020. [10](#)
8. [8] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In*IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016. [11](#)

[9] Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. Randaugment: Practical automated data augmentation with a reduced search space. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops*, 2020. [8](#)

[10] Tri Dao, Daniel Y Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2022. [3](#)

[11] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2009. [8](#)

[12] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)*, 2019. [1](#)

[13] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In *International Conference on Learning Representations (ICLR)*, 2020. [1](#), [2](#), [3](#), [4](#)

[14] Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, et al. Conformer: Convolution-augmented transformer for speech recognition. *Interspeech*, 2020. [1](#)

[15] Ali Hassani, Steven Walton, Jiachen Li, Shen Li, and Humphrey Shi. Neighborhood attention transformer. *arXiv:2204.07143*, 2022. [2](#), [3](#), [4](#), [8](#), [10](#), [15](#)

[16] Ali Hassani, Steven Walton, Nikhil Shah, Abulikemu Abuduweili, Jiachen Li, and Humphrey Shi. Escaping the big data paradigm with compact transformers. *arXiv:2104.05704*, 2021. [3](#)

[17] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022. [2](#)

[18] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask r-cnn. In *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2017. [9](#)

[19] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016. [2](#)

[20] Zilong Huang, Xinggang Wang, Yunchao Wei, Lichao Huang, Humphrey Shi, Wenyu Liu, and Thomas S. Huang. Ccnet: Criss-cross attention for semantic segmentation. In *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 2020. [2](#), [3](#), [5](#)

[21] Jitesh Jain, Anukriti Singh, Nikita Orlov, Zilong Huang, Jiachen Li, Steven Walton, and Humphrey Shi. Semask: Semantically masking transformer backbones for effective semantic segmentation. *arXiv:2112.12782*, 2021. [11](#)

[22] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2012. [2](#)

[23] Yann LeCun, Bernhard Boser, John S Denker, Donnie Henderson, Richard E Howard, Wayne Hubbard, and Lawrence D Jackel. Backpropagation applied to handwritten zip code recognition. *Neural computation*, 1989. [2](#)

[24] Yanghao Li, Hanzi Mao, Ross Girshick, and Kaiming He. Exploring plain vision transformer backbones for object detection. In *European Conference on Computer Vision (ECCV)*, 2022. [2](#)

[25] Jingyun Liang, Jiezhong Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: Image restoration using swin transformer. In *IEEE/CVF International Conference on Computer Vision (ICCV) Workshops*, 2021. [4](#)

[26] Youwei Liang, Chongjian Ge, Zhan Tong, Yibing Song, Jue Wang, and Pengtao Xie. Not all patches are what you need: Expediting vision transformers via token reorganizations. In *International Conference on Learning Representations (ICLR)*, 2022. [3](#)

[27] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In *European Conference on Computer Vision (ECCV)*, 2014. [9](#), [11](#)

[28] Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, et al. Swin transformer v2: Scaling up capacity and resolution. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022. [2](#), [4](#)

[29] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021. [2](#), [3](#), [4](#), [5](#), [7](#), [8](#), [10](#), [15](#)

[30] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022. [3](#), [6](#), [8](#), [9](#), [10](#), [15](#)

[31] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022. [4](#)

[32] Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In *International Conference on Machine Learning (ICML)*, 2018. [1](#), [2](#)

[33] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, et al. Pytorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems (NeurIPS)*, 2019. [8](#)

[34] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training, 2018. [1](#)- [35] Prajit Ramachandran, Niki Parmar, Ashish Vaswani, Irwan Bello, Anselm Levskaya, and Jon Shlens. Stand-alone self-attention in vision models. *Advances in Neural Information Processing Systems (NeurIPS)*, 2019. [1](#), [2](#), [3](#), [4](#)
- [36] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit: Efficient vision transformers with dynamic token sparsification. *Advances in Neural Information Processing Systems (NeurIPS)*, 2021. [3](#)
- [37] Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier. Efficient content-based sparse attention with routing transformers. *Transactions of the Association for Computational Linguistics (TACL)*, 9, 2021. [3](#), [5](#)
- [38] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In *International Conference on Machine Learning (ICML)*, 2020. [2](#), [3](#)
- [39] Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, and Hervé Jégou. Going deeper with image transformers. In *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021. [2](#), [3](#)
- [40] Zhengzhong Tu, Hossein Talebi, Han Zhang, Feng Yang, Peyman Milanfar, Alan Bovik, and Yinxiao Li. Maxvit: Multi-axis vision transformer. In *European Conference on Computer Vision (ECCV)*, 2022. [5](#)
- [41] Ashish Vaswani, Prajit Ramachandran, Aravind Srinivas, Niki Parmar, Blake Hechtman, and Jonathon Shlens. Scaling local self-attention for parameter efficient visual backbones. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021. [1](#), [2](#), [4](#)
- [42] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2017. [1](#), [3](#)
- [43] Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. Axial-deeplab: Stand-alone axial-attention for panoptic segmentation. In *European Conference on Computer Vision (ECCV)*, 2020. [2](#)
- [44] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021. [2](#)
- [45] Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. Non-local neural networks. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018. [4](#)
- [46] Ross Wightman. Pytorch image models. <https://github.com/rwightman/pytorch-image-models>, 2019. [8](#), [9](#)
- [47] Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross Girshick. Detectron2. <https://github.com/facebookresearch/detectron2>, 2019. [11](#)
- [48] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In *European Conference on Computer Vision (ECCV)*, 2018. [10](#)
- [49] Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. Simmim: A simple framework for masked image modeling. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022. [4](#)
- [50] Fisher Yu and Vladlen Koltun. Multi-scale context aggregation by dilated convolutions. In *International Conference on Learning Representations (ICLR)*, 2016. [3](#), [7](#)
- [51] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019. [8](#)
- [52] Bowen Zhang, Shuyang Gu, Bo Zhang, Jianmin Bao, Dong Chen, Fang Wen, Yong Wang, and Baining Guo. Styleswin: Transformer-based gan for high-resolution image generation. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022. [4](#)
- [53] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In *International Conference on Learning Representations (ICLR)*, 2018. [8](#)
- [54] Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, and Yi Yang. Random erasing data augmentation. In *AAAI Conference on Artificial Intelligence (AAAI)*, 2020. [8](#)
- [55] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017. [10](#), [11](#)## Appendix

### A. Implementation notes

As discussed in Sec. 3.5, we extend the existing *NATTEN* package to support dilated neighborhoods. *NATTEN* has a two-stage attention computation, similar to many other implementations: QK, and AV. The former computes the dot product of queries and keys, and produces attention weights, and the latter applies attention weights to the values. Scaling, softmax, and dropout are not included, as to prevent re-implementation. One of the advantages of this two-stage structure over manual implementations is that, like implementations of convolutions, sliding windows are taken directly from the source tensor, and not cached into an intermediary tensor, thus using significantly less memory. We refer readers to *NATTEN* documentation, and NAT [15] for further details.

**Dilation support.** Adding dilation to *NATTEN*’s naive kernels is mostly simple: instead of incrementing neighbors across each axis by 1, we simply instruct the kernels to increment by a variable  $d$ . NA however has a special way to handle edge/corner pixels, which requires additional changes to support dilation. The greater challenge in adding dilation to *NATTEN* was adding it to the “tiled” kernels that utilize shared memory. Tiled NA kernels are a more recent addition to *NATTEN*, and boost NA’s throughput significantly. Tiled implementations of matrix multiplication and convolutions are essential in parallelizing these operations efficiently, while minimizing DRAM accesses. As the name suggests, tiled implementations divide the operation into tiles and cache tiles of inputs from the global memory into the shared memory within each threadblock. Accessing values from shared memory is typically much faster compared to directly accessing global memory, but also comes with challenges such as bank conflicts. Tiled implementations also operate with the assumption that access patterns are not broken. Introducing dilation values would break those access patterns and require a re-implementation that ensures dilated neighbors are cached instead of local neighbors. We present a layer-wise relative speed and memory usage comparison between NAT and DiNAT with respect to Swin in Fig. I.

**Scaling and brain float support.** In order to train our larger models and avoid overflowing activation values in later layers of the model, we’ve had to switch from automatic mixed-precision training with the default half precision data type, float16, which has 5 exponent bits and 10 mantissa bits, to bfloat16, which has the advantage of having 8 exponent bits while having only 7 mantissa bits. Utilizing bfloat16 has often been recommended for cases which lead to large activations, which includes ours as we scale our model. However, switching to bfloat16 required

<table border="1">
<thead>
<tr>
<th>Variant</th>
<th>Downsampling</th>
<th>Layers per level</th>
<th>Dim × Heads</th>
<th>MLP ratio</th>
<th># of Params</th>
<th>FLOPs</th>
</tr>
</thead>
<tbody>
<tr>
<td>• DiNAT<sub>s</sub>-T</td>
<td>Patched</td>
<td>2, 2, 6, 2</td>
<td>32 × 3</td>
<td>4</td>
<td>28 M</td>
<td>4.5 G</td>
</tr>
<tr>
<td>• DiNAT<sub>s</sub>-S</td>
<td>Patched</td>
<td>2, 2, 18, 2</td>
<td>32 × 3</td>
<td>4</td>
<td>50 M</td>
<td>8.7 G</td>
</tr>
<tr>
<td>• DiNAT<sub>s</sub>-B</td>
<td>Patched</td>
<td>2, 2, 18, 2</td>
<td>32 × 4</td>
<td>4</td>
<td>88 M</td>
<td>15.4 G</td>
</tr>
<tr>
<td>• DiNAT<sub>s</sub>-L</td>
<td>Patched</td>
<td>2, 2, 18, 2</td>
<td>32 × 6</td>
<td>4</td>
<td>197 M</td>
<td>34.5 G</td>
</tr>
<tr>
<td>• DiNAT-M</td>
<td>Conv</td>
<td>3, 4, 6, 5</td>
<td>32 × 2</td>
<td>3</td>
<td>20 M</td>
<td>2.7 G</td>
</tr>
<tr>
<td>• DiNAT-T</td>
<td>Conv</td>
<td>3, 4, 18, 5</td>
<td>32 × 2</td>
<td>3</td>
<td>28 M</td>
<td>4.3 G</td>
</tr>
<tr>
<td>• DiNAT-S</td>
<td>Conv</td>
<td>3, 4, 18, 5</td>
<td>32 × 3</td>
<td>2</td>
<td>51 M</td>
<td>7.8 G</td>
</tr>
<tr>
<td>• DiNAT-B</td>
<td>Conv</td>
<td>3, 4, 18, 5</td>
<td>32 × 4</td>
<td>2</td>
<td>90 M</td>
<td>13.7 G</td>
</tr>
<tr>
<td>• DiNAT-L</td>
<td>Conv</td>
<td>3, 4, 18, 5</td>
<td>32 × 6</td>
<td>2</td>
<td>200 M</td>
<td>30.6 G</td>
</tr>
</tbody>
</table>

**Table I. Summary of DiNAT and DiNAT<sub>s</sub> configurations.** Channels (heads and dim) double after every level until the final one. Default dilation values for the four levels are 8, 4, 2, and 1. Kernel size is 7×7 in all variants.

<table border="1">
<thead>
<tr>
<th>Variant</th>
<th>Resolution</th>
<th>Level 1</th>
<th>Level 2</th>
<th>Level 3</th>
<th>Level 4</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6" style="text-align: center;"><i>ImageNet classification.</i></td>
</tr>
<tr>
<td>• DiNAT<sub>s</sub>-T</td>
<td>224<sup>2</sup></td>
<td>1, 8</td>
<td>1, 4</td>
<td>1, 2, 1, 2, 1, 2</td>
<td>1, 1</td>
</tr>
<tr>
<td>• DiNAT<sub>s</sub>-S/B/L</td>
<td>224<sup>2</sup></td>
<td>1, 8</td>
<td>1, 4</td>
<td>1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2</td>
<td>1, 1</td>
</tr>
<tr>
<td>• DiNAT<sub>s</sub>-L</td>
<td>384<sup>2</sup></td>
<td>1, 13</td>
<td>1, 6</td>
<td>1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3</td>
<td>1, 1</td>
</tr>
<tr>
<td>• DiNAT-M</td>
<td>224<sup>2</sup></td>
<td>1, 8, 1</td>
<td>1, 4, 1, 4</td>
<td>1, 2, 1, 2, 1, 2</td>
<td>1, 1, 1, 1, 1</td>
</tr>
<tr>
<td>• DiNAT-T/S/B/L</td>
<td>224<sup>2</sup></td>
<td>1, 8, 1</td>
<td>1, 4, 1, 4</td>
<td>1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2</td>
<td>1, 1, 1, 1, 1</td>
</tr>
<tr>
<td>• DiNAT-L</td>
<td>384<sup>2</sup></td>
<td>1, 13, 1</td>
<td>1, 6, 1, 6</td>
<td>1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3</td>
<td>1, 1, 1, 1, 1</td>
</tr>
<tr>
<td colspan="6" style="text-align: center;"><i>MS-COCO detection and instance segmentation.</i></td>
</tr>
<tr>
<td>• DiNAT<sub>s</sub>-T</td>
<td>800<sup>2</sup></td>
<td>1, 28</td>
<td>1, 14</td>
<td>1, 3, 1, 5, 1, 7</td>
<td>1, 3</td>
</tr>
<tr>
<td>• DiNAT<sub>s</sub>-S/B/L</td>
<td>800<sup>2</sup></td>
<td>1, 28</td>
<td>1, 14</td>
<td>1, 3, 1, 5, 1, 7, 1, 3, 1, 5, 1, 7, 1, 3, 1, 5, 1, 7</td>
<td>1, 3</td>
</tr>
<tr>
<td>• DiNAT-M</td>
<td>800<sup>2</sup></td>
<td>1, 28, 1</td>
<td>1, 7, 1, 14</td>
<td>1, 3, 1, 5, 1, 7</td>
<td>1, 3, 1, 3, 1</td>
</tr>
<tr>
<td>• DiNAT-T/S/B/L</td>
<td>800<sup>2</sup></td>
<td>1, 28, 1</td>
<td>1, 7, 1, 14</td>
<td>1, 3, 1, 5, 1, 7, 1, 3, 1, 5, 1, 7, 1, 3, 1, 5, 1, 7</td>
<td>1, 3, 1, 3, 1</td>
</tr>
<tr>
<td colspan="6" style="text-align: center;"><i>ADE20K semantic segmentation.</i></td>
</tr>
<tr>
<td>• DiNAT<sub>s</sub>-T</td>
<td>512<sup>2</sup></td>
<td>1, 16</td>
<td>1, 8</td>
<td>1, 2, 1, 3, 1, 4</td>
<td>1, 2</td>
</tr>
<tr>
<td>• DiNAT<sub>s</sub>-S/B</td>
<td>512<sup>2</sup></td>
<td>1, 16</td>
<td>1, 8</td>
<td>1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4</td>
<td>1, 2</td>
</tr>
<tr>
<td>• DiNAT<sub>s</sub>-L</td>
<td>640<sup>2</sup></td>
<td>1, 20</td>
<td>1, 10</td>
<td>1, 2, 1, 3, 1, 4, 1, 5, 1, 2, 1, 3, 1, 4, 1, 5, 1, 5</td>
<td>1, 2</td>
</tr>
<tr>
<td>• DiNAT-M</td>
<td>512<sup>2</sup></td>
<td>1, 16, 1</td>
<td>1, 4, 1, 8</td>
<td>1, 2, 1, 3, 1, 4</td>
<td>1, 2, 1, 2, 1</td>
</tr>
<tr>
<td>• DiNAT-T/S/B</td>
<td>512<sup>2</sup></td>
<td>1, 16, 1</td>
<td>1, 4, 1, 8</td>
<td>1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4</td>
<td>1, 2, 1, 2, 1</td>
</tr>
<tr>
<td>• DiNAT-L</td>
<td>640<sup>2</sup></td>
<td>1, 20, 1</td>
<td>1, 5, 1, 10</td>
<td>1, 2, 1, 3, 1, 4, 1, 5, 1, 2, 1, 3, 1, 4, 1, 5, 1, 5</td>
<td>1, 2, 1, 2, 1</td>
</tr>
</tbody>
</table>

**Table II. Dilation values.** Due to ImageNet’s relatively small input resolution, level 4 layers cannot go beyond a dilation value of 1, which is equivalent to NA. Also note that at 224×224 resolution, level 4 inputs will be exactly 7×7, therefore NA will be equivalent to self attention. This is not true in downstream tasks where resolutions are noticeably higher where levels 2 and 3 have *gradually* increasing dilation values, which are repeated in deeper models. This corresponds to the highlighted rows in Tab. 9 labeled “Gradual”. These configurations apply to all downstream experiments (excluding those in Sec. 4.4).

a re-implementation of *NATTEN*’s half precision kernels to support and utilize bfloat16 correctly.

### B. Training settings

We provide additional details on training DiNAT in Tab. I. We also provide details on DiNAT<sub>s</sub>, which utilizes non-overlapping patch embedding and downsampling, similar to Swin [29] and ConvNeXt [30]. DiNAT<sub>s</sub> serves as an alternative DiNA-based model, which has an architecture identical to Swin. DiNAT<sub>s</sub> can also serve as an ablation model, since it is identical to Swin in architecture, with WSA replaced with NA, and SWSA replaced with DiNA.

One of the most important architecture-related hyperparameters in DiNA-based models is dilation values. Both**Figure I. Layer-wise relative speed and memory comparison between NAT and DiNAT, with respect to Swin.** NAT layers, which are only two consecutive NA layers with kernel size  $7^2$ , are already up to 40% faster than Swin layers with the same kernel size. DiNAT layers, comprised of an NA layer followed by a DiNA layer, are slightly slower in practice due to the break in memory access pattern, but are still faster than Swin layers.

DiNAT and DiNAT<sub>s</sub> use a combination of NA and DiNA layers. We typically set dilation values in DiNA layers to be the maximum possible value with respect to input resolutions, if known. For example, ImageNet classification at 224×224 is downsampled to a quarter of the original size initially, therefore Level 1 layers take feature maps of resolution 56×56 as input. With a kernel size of 7×7, the maximum possible dilation value is  $\lfloor 56/7 \rfloor = 8$ . Level 2 will take feature maps of resolution 28×28 as input, leading to a maximum possible dilation value of 4. Because of this, we change dilation values depending on the task and resolution. We present the final dilation values we used in classification, detection, and segmentation in Tab. II. Note that we only change dilation values for DiNA layers, since we found that fine-tuning NA layers to DiNA layers may result in a slight decrease in initial performance (see Sec. 4.4, Tab. 12).

### C. Experiments with alternative architecture

We conducted all primary experiments with both our main model, DiNAT, as well as DiNAT<sub>s</sub>. We found that DiNAT<sub>s</sub> could serve as alternatives in certain cases, as they still provide noticeable improvements over Swin in terms of speed, accuracy, and memory usage. Classification results are provided in Tab. III, object detection and instance segmentation results are provided in Tab. IV, and semantic segmentation results are provided in Tab. VI.

In Sec. 4.4 we experimented with architecture-related hyperparameters that are introduced by DiNA: dilation values, and the ordering of NA and DiNA layers. We also complete those dilation experiments by adding DiNAT<sub>s</sub> and Swin, and present the results in Tabs. V and VII.

<table border="1">
<thead>
<tr>
<th>Model</th>
<th>Res.</th>
<th># of Params</th>
<th>FLOPs</th>
<th>Thru. (img/sec)</th>
<th>Memory (GB)</th>
<th>Top-1 (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="7" style="text-align: center;"><i>ImageNet-1K trained models</i></td>
</tr>
<tr>
<td>○ NAT-M</td>
<td>224<sup>2</sup></td>
<td>20 M</td>
<td>2.7 G</td>
<td>2132</td>
<td>2.4</td>
<td><b>81.8</b></td>
</tr>
<tr>
<td>● DiNAT-M</td>
<td>224<sup>2</sup></td>
<td>20 M</td>
<td>2.7 G</td>
<td>2080</td>
<td>2.4</td>
<td><b>81.8</b></td>
</tr>
<tr>
<td>○ Swin-T</td>
<td>224<sup>2</sup></td>
<td>28 M</td>
<td>4.5 G</td>
<td>1724</td>
<td>4.8</td>
<td>81.3</td>
</tr>
<tr>
<td>● DiNAT<sub>s</sub>-T</td>
<td>224<sup>2</sup></td>
<td>28 M</td>
<td>4.5 G</td>
<td>1954</td>
<td>4.0</td>
<td>81.8</td>
</tr>
<tr>
<td>● ConvNeXt-T</td>
<td>224<sup>2</sup></td>
<td>28 M</td>
<td>4.5 G</td>
<td>2491</td>
<td>3.4</td>
<td>82.1</td>
</tr>
<tr>
<td>○ NAT-T</td>
<td>224<sup>2</sup></td>
<td>28 M</td>
<td>4.3 G</td>
<td>1537</td>
<td>2.5</td>
<td><b>83.2</b></td>
</tr>
<tr>
<td>● DiNAT-T</td>
<td>224<sup>2</sup></td>
<td>28 M</td>
<td>4.3 G</td>
<td>1500</td>
<td>2.5</td>
<td>82.7</td>
</tr>
<tr>
<td>○ Swin-S</td>
<td>224<sup>2</sup></td>
<td>50 M</td>
<td>8.7 G</td>
<td>1056</td>
<td>5.0</td>
<td>83.0</td>
</tr>
<tr>
<td>● DiNAT<sub>s</sub>-S</td>
<td>224<sup>2</sup></td>
<td>50 M</td>
<td>8.7 G</td>
<td>1203</td>
<td>4.1</td>
<td>83.5</td>
</tr>
<tr>
<td>● ConvNeXt-S</td>
<td>224<sup>2</sup></td>
<td>50 M</td>
<td>8.7 G</td>
<td>1549</td>
<td>3.5</td>
<td>83.1</td>
</tr>
<tr>
<td>○ NAT-S</td>
<td>224<sup>2</sup></td>
<td>51 M</td>
<td>7.8 G</td>
<td>1049</td>
<td>3.7</td>
<td>83.7</td>
</tr>
<tr>
<td>● DiNAT-S</td>
<td>224<sup>2</sup></td>
<td>51 M</td>
<td>7.8 G</td>
<td>1058</td>
<td>3.7</td>
<td><b>83.8</b></td>
</tr>
<tr>
<td>○ Swin-B</td>
<td>224<sup>2</sup></td>
<td>88 M</td>
<td>15.4 G</td>
<td>774</td>
<td>6.7</td>
<td>83.5</td>
</tr>
<tr>
<td>● DiNAT<sub>s</sub>-B</td>
<td>224<sup>2</sup></td>
<td>88 M</td>
<td>15.4 G</td>
<td>877</td>
<td>5.5</td>
<td>83.8</td>
</tr>
<tr>
<td>● ConvNeXt-B</td>
<td>224<sup>2</sup></td>
<td>89 M</td>
<td>15.4 G</td>
<td>1107</td>
<td>4.8</td>
<td>83.8</td>
</tr>
<tr>
<td>○ NAT-B</td>
<td>224<sup>2</sup></td>
<td>90 M</td>
<td>13.7 G</td>
<td>781</td>
<td>5.0</td>
<td>84.3</td>
</tr>
<tr>
<td>● DiNAT-B</td>
<td>224<sup>2</sup></td>
<td>90 M</td>
<td>13.7 G</td>
<td>764</td>
<td>5.0</td>
<td><b>84.4</b></td>
</tr>
<tr>
<td colspan="7" style="text-align: center;"><i>ImageNet-22K pre-trained models</i></td>
</tr>
<tr>
<td>○ Swin-L</td>
<td>224<sup>2</sup></td>
<td>197 M</td>
<td>34.5 G</td>
<td>478</td>
<td>10.4</td>
<td>86.3</td>
</tr>
<tr>
<td>● DiNAT<sub>s</sub>-L</td>
<td>224<sup>2</sup></td>
<td>197 M</td>
<td>34.5 G</td>
<td>528</td>
<td>8.6</td>
<td>86.5</td>
</tr>
<tr>
<td>● ConvNeXt-L</td>
<td>224<sup>2</sup></td>
<td>198 M</td>
<td>34.4 G</td>
<td>643</td>
<td>7.5</td>
<td><b>86.6</b></td>
</tr>
<tr>
<td>● DiNAT-L</td>
<td>224<sup>2</sup></td>
<td>200 M</td>
<td>30.6 G</td>
<td>474</td>
<td>7.8</td>
<td><b>86.6</b></td>
</tr>
<tr>
<td>○ Swin-L<sup>†</sup></td>
<td>384<sup>2</sup></td>
<td>197 M</td>
<td>104.0 G</td>
<td>169</td>
<td>32.7</td>
<td>87.3</td>
</tr>
<tr>
<td>● DiNAT<sub>s</sub>-L</td>
<td>384<sup>2</sup></td>
<td>197 M</td>
<td>101.5 G</td>
<td>181</td>
<td>22.6</td>
<td>87.4</td>
</tr>
<tr>
<td>● ConvNeXt-L</td>
<td>384<sup>2</sup></td>
<td>198 M</td>
<td>101.1 G</td>
<td>221</td>
<td>19.2</td>
<td><b>87.5</b></td>
</tr>
<tr>
<td>● DiNAT-L</td>
<td>384<sup>2</sup></td>
<td>200 M</td>
<td>89.7 G</td>
<td>161</td>
<td>20.1</td>
<td>87.4</td>
</tr>
<tr>
<td>● DiNAT-L<sup>†</sup></td>
<td>384<sup>2</sup></td>
<td>200 M</td>
<td>92.4 G</td>
<td>110</td>
<td>26.9</td>
<td><b>87.5</b></td>
</tr>
</tbody>
</table>

**Table III. ImageNet-1K image classification performance.**

<sup>†</sup>indicates increased window size from  $7^2$  to  $11^2$  (DiNAT) and  $12^2$  (Swin). Throughput and peak memory usage are measured from forward passes with a batch size of 256 on a single A100 GPU. Note that DiNAT<sub>s</sub> is identical in architecture to Swin, and only different in attention modules (WSA/SWSA replaced with NA/DiNA).<table border="1">
<thead>
<tr>
<th>Backbone</th>
<th># of Params</th>
<th>FLOPs</th>
<th>Thru. (FPS)</th>
<th>AP<sup>b</sup></th>
<th>AP<sup>b</sup><sub>50</sub></th>
<th>AP<sup>b</sup><sub>75</sub></th>
<th>AP<sup>m</sup></th>
<th>AP<sup>m</sup><sub>50</sub></th>
<th>AP<sup>m</sup><sub>75</sub></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="10" style="text-align: center;"><i>Mask R-CNN - 3x schedule</i></td>
</tr>
<tr>
<td>○ NAT-M</td>
<td>40 M</td>
<td>225 G</td>
<td>54.1</td>
<td>46.5</td>
<td>68.1</td>
<td>51.3</td>
<td>41.7</td>
<td>65.2</td>
<td>44.7</td>
</tr>
<tr>
<td>● DiNAT-M</td>
<td>40 M</td>
<td>225 G</td>
<td>53.8</td>
<td><b>47.2</b></td>
<td><b>69.1</b></td>
<td><b>51.9</b></td>
<td><b>42.5</b></td>
<td><b>66.0</b></td>
<td><b>45.9</b></td>
</tr>
<tr>
<td>○ Swin-T</td>
<td>48 M</td>
<td>267 G</td>
<td>45.1</td>
<td>46.0</td>
<td>68.1</td>
<td>50.3</td>
<td>41.6</td>
<td>65.1</td>
<td>44.9</td>
</tr>
<tr>
<td>● DiNAT<sub>s</sub>-T</td>
<td>48 M</td>
<td>263 G</td>
<td>52.5</td>
<td>46.6</td>
<td>68.8</td>
<td>51.3</td>
<td>42.1</td>
<td>65.7</td>
<td>45.4</td>
</tr>
<tr>
<td>● ConvNeXt-T</td>
<td>48 M</td>
<td>262 G</td>
<td>52.0</td>
<td>46.2</td>
<td>67.0</td>
<td>50.8</td>
<td>41.7</td>
<td>65.0</td>
<td>44.9</td>
</tr>
<tr>
<td>○ NAT-T</td>
<td>48 M</td>
<td>258 G</td>
<td>44.5</td>
<td>47.7</td>
<td>69.0</td>
<td>52.6</td>
<td>42.6</td>
<td>66.1</td>
<td>45.9</td>
</tr>
<tr>
<td>● DiNAT-T</td>
<td>48 M</td>
<td>258 G</td>
<td>43.3</td>
<td><b>48.6</b></td>
<td><b>70.2</b></td>
<td><b>53.4</b></td>
<td><b>43.5</b></td>
<td><b>67.3</b></td>
<td><b>46.8</b></td>
</tr>
<tr>
<td>○ Swin-S</td>
<td>69 M</td>
<td>359 G</td>
<td>31.7</td>
<td>48.5</td>
<td>70.2</td>
<td>53.5</td>
<td>43.3</td>
<td>67.3</td>
<td>46.6</td>
</tr>
<tr>
<td>● DiNAT<sub>s</sub>-S</td>
<td>69 M</td>
<td>350 G</td>
<td>38.7</td>
<td>48.6</td>
<td>70.4</td>
<td>53.2</td>
<td>43.5</td>
<td>67.6</td>
<td>46.9</td>
</tr>
<tr>
<td>○ NAT-S</td>
<td>70 M</td>
<td>330 G</td>
<td>34.8</td>
<td>48.4</td>
<td>69.8</td>
<td>53.2</td>
<td>43.2</td>
<td>66.9</td>
<td>46.5</td>
</tr>
<tr>
<td>● DiNAT-S</td>
<td>70 M</td>
<td>330 G</td>
<td>35.3</td>
<td><b>49.3</b></td>
<td><b>70.8</b></td>
<td><b>54.2</b></td>
<td><b>44.0</b></td>
<td><b>68.0</b></td>
<td><b>47.4</b></td>
</tr>
<tr>
<td colspan="10" style="text-align: center;"><i>Cascade Mask R-CNN - 3x schedule</i></td>
</tr>
<tr>
<td>○ NAT-M</td>
<td>77 M</td>
<td>704 G</td>
<td>27.8</td>
<td>50.3</td>
<td>68.9</td>
<td>54.9</td>
<td>43.6</td>
<td>66.4</td>
<td>47.2</td>
</tr>
<tr>
<td>● DiNAT-M</td>
<td>77 M</td>
<td>704 G</td>
<td>27.6</td>
<td><b>51.2</b></td>
<td><b>69.8</b></td>
<td><b>55.7</b></td>
<td><b>44.4</b></td>
<td><b>67.3</b></td>
<td><b>47.8</b></td>
</tr>
<tr>
<td>○ Swin-T</td>
<td>86 M</td>
<td>745 G</td>
<td>25.1</td>
<td>50.4</td>
<td>69.2</td>
<td>54.7</td>
<td>43.7</td>
<td>66.6</td>
<td>47.3</td>
</tr>
<tr>
<td>● DiNAT<sub>s</sub>-T</td>
<td>86 M</td>
<td>742 G</td>
<td>27.4</td>
<td>51.0</td>
<td>69.9</td>
<td>55.4</td>
<td>44.1</td>
<td>67.3</td>
<td>47.6</td>
</tr>
<tr>
<td>● ConvNeXt-T</td>
<td>86 M</td>
<td>741 G</td>
<td>27.3</td>
<td>50.4</td>
<td>69.1</td>
<td>54.8</td>
<td>43.7</td>
<td>66.5</td>
<td>47.3</td>
</tr>
<tr>
<td>○ NAT-T</td>
<td>85 M</td>
<td>737 G</td>
<td>24.9</td>
<td>51.4</td>
<td>70.0</td>
<td>55.9</td>
<td>44.5</td>
<td>67.6</td>
<td>47.9</td>
</tr>
<tr>
<td>● DiNAT-T</td>
<td>85 M</td>
<td>737 G</td>
<td>25.0</td>
<td><b>52.2</b></td>
<td><b>71.0</b></td>
<td><b>56.8</b></td>
<td><b>45.1</b></td>
<td><b>68.3</b></td>
<td><b>48.8</b></td>
</tr>
<tr>
<td>○ Swin-S</td>
<td>107 M</td>
<td>838 G</td>
<td>20.3</td>
<td>51.8</td>
<td>70.4</td>
<td>56.3</td>
<td>44.7</td>
<td>67.9</td>
<td>48.5</td>
</tr>
<tr>
<td>● DiNAT<sub>s</sub>-S</td>
<td>107 M</td>
<td>829 G</td>
<td>23.1</td>
<td>52.3</td>
<td>71.2</td>
<td>56.7</td>
<td>45.2</td>
<td>68.6</td>
<td>49.1</td>
</tr>
<tr>
<td>● ConvNeXt-S</td>
<td>108 M</td>
<td>827 G</td>
<td>23.0</td>
<td>51.9</td>
<td>70.8</td>
<td>56.5</td>
<td>45.0</td>
<td>68.4</td>
<td>49.1</td>
</tr>
<tr>
<td>○ NAT-S</td>
<td>108 M</td>
<td>809 G</td>
<td>21.7</td>
<td>52.0</td>
<td>70.4</td>
<td>56.3</td>
<td>44.9</td>
<td>68.1</td>
<td>48.6</td>
</tr>
<tr>
<td>● DiNAT-S</td>
<td>108 M</td>
<td>809 G</td>
<td>21.8</td>
<td><b>52.9</b></td>
<td><b>71.8</b></td>
<td><b>57.6</b></td>
<td><b>45.8</b></td>
<td><b>69.3</b></td>
<td><b>49.9</b></td>
</tr>
<tr>
<td>○ Swin-B</td>
<td>145 M</td>
<td>982 G</td>
<td>17.3</td>
<td>51.9</td>
<td>70.9</td>
<td>56.5</td>
<td>45.0</td>
<td>68.4</td>
<td>48.7</td>
</tr>
<tr>
<td>● DiNAT<sub>s</sub>-B</td>
<td>145 M</td>
<td>966 G</td>
<td>19.7</td>
<td>52.6</td>
<td>71.5</td>
<td>57.2</td>
<td>45.3</td>
<td>68.8</td>
<td>49.1</td>
</tr>
<tr>
<td>● ConvNeXt-B</td>
<td>146 M</td>
<td>964 G</td>
<td>19.5</td>
<td>52.7</td>
<td>71.3</td>
<td>57.2</td>
<td>45.6</td>
<td>68.9</td>
<td>49.5</td>
</tr>
<tr>
<td>○ NAT-B</td>
<td>147 M</td>
<td>931 G</td>
<td>18.6</td>
<td>52.3</td>
<td>70.9</td>
<td>56.9</td>
<td>45.1</td>
<td>68.3</td>
<td>49.1</td>
</tr>
<tr>
<td>● DiNAT-B</td>
<td>147 M</td>
<td>931 G</td>
<td>18.5</td>
<td><b>53.4</b></td>
<td><b>72.1</b></td>
<td><b>58.2</b></td>
<td><b>46.2</b></td>
<td><b>69.7</b></td>
<td><b>50.2</b></td>
</tr>
<tr>
<td>○ Swin-L<sup>*,‡</sup></td>
<td>253 M</td>
<td>1393 G</td>
<td>12.9</td>
<td>53.7</td>
<td>72.2</td>
<td>58.7</td>
<td>46.4</td>
<td>69.9</td>
<td>50.7</td>
</tr>
<tr>
<td>● DiNAT<sub>s</sub>-L<sup>‡</sup></td>
<td>253 M</td>
<td>1357 G</td>
<td>15.0</td>
<td>54.8</td>
<td>74.2</td>
<td>59.8</td>
<td>47.2</td>
<td>71.3</td>
<td>51.2</td>
</tr>
<tr>
<td>● ConvNeXt-L<sup>‡</sup></td>
<td>253 M</td>
<td>1354 G</td>
<td>14.8</td>
<td>54.8</td>
<td>73.8</td>
<td>59.8</td>
<td>47.6</td>
<td>71.3</td>
<td>51.7</td>
</tr>
<tr>
<td>● DiNAT-L<sup>‡</sup></td>
<td>258 M</td>
<td>1276 G</td>
<td>14.0</td>
<td><b>55.3</b></td>
<td><b>74.3</b></td>
<td><b>60.2</b></td>
<td><b>47.8</b></td>
<td><b>71.8</b></td>
<td><b>52.0</b></td>
</tr>
</tbody>
</table>

**Table IV. COCO object detection and instance segmentation performance.** <sup>‡</sup>indicates that the model was pre-trained on ImageNet-22K. \*Swin-L was not reported with Cascade Mask R-CNN, therefore we trained it with their official checkpoint. Throughput is measured on a single A100 GPU. Note that DiNAT<sub>s</sub> is identical in architecture to Swin, and only different in attention modules (WSA/SWSA replaced with NA/DiNA).

<table border="1">
<thead>
<tr>
<th>Model</th>
<th>Dilation per level</th>
<th>ImageNet Top-1 (%)</th>
<th>MSCOCO AP<sup>b</sup></th>
<th>MSCOCO AP<sup>m</sup></th>
<th>ADE20K mIoU</th>
</tr>
</thead>
<tbody>
<tr>
<td>○ Swin-Tiny</td>
<td>Not Applicable</td>
<td>81.3</td>
<td>46.0</td>
<td>41.6</td>
<td>45.8</td>
</tr>
<tr>
<td>○ NAT<sub>s</sub>-Tiny</td>
<td>1, 1, 1, 1</td>
<td>81.8</td>
<td>46.1</td>
<td>41.5</td>
<td>46.2</td>
</tr>
<tr>
<td>● DiNAT<sub>s</sub>-Tiny</td>
<td>8, 4, 2, 1</td>
<td>81.8</td>
<td>46.3</td>
<td>41.6</td>
<td>46.7</td>
</tr>
<tr>
<td>● DiNAT<sub>s</sub>-Tiny</td>
<td>16, 8, 4, 2</td>
<td>-</td>
<td><b>46.4</b></td>
<td><b>41.8</b></td>
<td>47.1</td>
</tr>
<tr>
<td>● DiNAT<sub>s</sub>-Tiny</td>
<td>Maximum</td>
<td>81.8</td>
<td><b>46.4</b></td>
<td><b>41.9</b></td>
<td>47.0</td>
</tr>
<tr>
<td>● DiNAT<sub>s</sub>-Tiny</td>
<td>Gradual</td>
<td>-</td>
<td><b>46.6</b></td>
<td><b>42.1</b></td>
<td><b>47.4</b></td>
</tr>
<tr>
<td>○ NAT-Tiny</td>
<td>1, 1, 1, 1</td>
<td>83.2</td>
<td>47.7</td>
<td>42.6</td>
<td>48.4</td>
</tr>
<tr>
<td>● DiNAT-Tiny</td>
<td>8, 4, 2, 1</td>
<td>82.7</td>
<td>48.0</td>
<td>42.9</td>
<td>48.5</td>
</tr>
<tr>
<td>● DiNAT-Tiny</td>
<td>16, 8, 4, 2</td>
<td>-</td>
<td>48.3</td>
<td>43.4</td>
<td>48.5</td>
</tr>
<tr>
<td>● DiNAT-Tiny</td>
<td>Maximum</td>
<td>82.7</td>
<td><b>48.6</b></td>
<td><b>43.5</b></td>
<td>48.7</td>
</tr>
<tr>
<td>● DiNAT-Tiny</td>
<td>Gradual</td>
<td>-</td>
<td><b>48.6</b></td>
<td><b>43.5</b></td>
<td><b>48.8</b></td>
</tr>
</tbody>
</table>

**Table V. Dilation impact on performance.** Models listed within the same section have identical architectures and are different only in attention patterns (NAT<sub>s</sub> is identical to Swin with both WSA and SWSA replaced with NA, DiNAT<sub>s</sub> replaces SWSA with DiNA).

<table border="1">
<thead>
<tr>
<th>Backbone</th>
<th>Res.</th>
<th># of Params</th>
<th>FLOPs</th>
<th>Thru. (FPS)</th>
<th colspan="2">mIoU</th>
</tr>
<tr>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>single scale</th>
<th>multi scale</th>
</tr>
</thead>
<tbody>
<tr>
<td>○ NAT-M</td>
<td>2048 × 512</td>
<td>50 M</td>
<td>900 G</td>
<td>24.5</td>
<td>45.1</td>
<td>46.4</td>
</tr>
<tr>
<td>● DiNAT-M</td>
<td>2048 × 512</td>
<td>50 M</td>
<td>900 G</td>
<td>24.2</td>
<td><b>45.8</b></td>
<td><b>47.2</b></td>
</tr>
<tr>
<td>○ Swin-T</td>
<td>2048 × 512</td>
<td>60 M</td>
<td>946 G</td>
<td>21.3</td>
<td>44.5</td>
<td>45.8</td>
</tr>
<tr>
<td>● DiNAT<sub>s</sub>-T</td>
<td>2048 × 512</td>
<td>60 M</td>
<td>941 G</td>
<td>23.5</td>
<td>46.0</td>
<td>47.4</td>
</tr>
<tr>
<td>● ConvNeXt-T</td>
<td>2048 × 512</td>
<td>60 M</td>
<td>939 G</td>
<td>23.3</td>
<td>46.0</td>
<td>46.7</td>
</tr>
<tr>
<td>○ NAT-T</td>
<td>2048 × 512</td>
<td>58 M</td>
<td>934 G</td>
<td>21.4</td>
<td>47.1</td>
<td>48.4</td>
</tr>
<tr>
<td>● DiNAT-T</td>
<td>2048 × 512</td>
<td>58 M</td>
<td>934 G</td>
<td>21.3</td>
<td><b>47.8</b></td>
<td><b>48.8</b></td>
</tr>
<tr>
<td>○ Swin-S</td>
<td>2048 × 512</td>
<td>81 M</td>
<td>1040 G</td>
<td>17.0</td>
<td>47.6</td>
<td>49.5</td>
</tr>
<tr>
<td>● DiNAT<sub>s</sub>-S</td>
<td>2048 × 512</td>
<td>81 M</td>
<td>1030 G</td>
<td>19.1</td>
<td>48.6</td>
<td><b>49.9</b></td>
</tr>
<tr>
<td>● ConvNeXt-S</td>
<td>2048 × 512</td>
<td>82 M</td>
<td>1027 G</td>
<td>19.1</td>
<td>48.7</td>
<td>49.6</td>
</tr>
<tr>
<td>○ NAT-S</td>
<td>2048 × 512</td>
<td>82 M</td>
<td>1010 G</td>
<td>17.9</td>
<td>48.0</td>
<td>49.5</td>
</tr>
<tr>
<td>● DiNAT-S</td>
<td>2048 × 512</td>
<td>82 M</td>
<td>1010 G</td>
<td>18.1</td>
<td><b>48.9</b></td>
<td><b>49.9</b></td>
</tr>
<tr>
<td>○ Swin-B</td>
<td>2048 × 512</td>
<td>121 M</td>
<td>1188 G</td>
<td>14.6</td>
<td>48.1</td>
<td>49.7</td>
</tr>
<tr>
<td>● DiNAT<sub>s</sub>-B</td>
<td>2048 × 512</td>
<td>121 M</td>
<td>1173 G</td>
<td>16.5</td>
<td>49.4</td>
<td>50.2</td>
</tr>
<tr>
<td>● ConvNeXt-B</td>
<td>2048 × 512</td>
<td>122 M</td>
<td>1170 G</td>
<td>16.4</td>
<td>49.1</td>
<td>49.9</td>
</tr>
<tr>
<td>○ NAT-B</td>
<td>2048 × 512</td>
<td>123 M</td>
<td>1137 G</td>
<td>15.6</td>
<td>48.5</td>
<td>49.7</td>
</tr>
<tr>
<td>● DiNAT-B</td>
<td>2048 × 512</td>
<td>123 M</td>
<td>1137 G</td>
<td>15.4</td>
<td><b>49.6</b></td>
<td><b>50.4</b></td>
</tr>
<tr>
<td>○ Swin-L<sup>†,‡</sup></td>
<td>2560 × 640</td>
<td>234 M</td>
<td>2585 G</td>
<td>8.5</td>
<td>-</td>
<td>53.5</td>
</tr>
<tr>
<td>● DiNAT<sub>s</sub>-L<sup>‡</sup></td>
<td>2560 × 640</td>
<td>234 M</td>
<td>2466 G</td>
<td>9.7</td>
<td>53.4</td>
<td>54.6</td>
</tr>
<tr>
<td>● ConvNeXt-L<sup>‡</sup></td>
<td>2560 × 640</td>
<td>235 M</td>
<td>2458 G</td>
<td>9.6</td>
<td>53.2</td>
<td>53.7</td>
</tr>
<tr>
<td>● DiNAT-L<sup>‡</sup></td>
<td>2560 × 640</td>
<td>238 M</td>
<td>2335 G</td>
<td>9.0</td>
<td><b>54.0</b></td>
<td><b>54.9</b></td>
</tr>
</tbody>
</table>

**Table VI. ADE20K semantic segmentation performance.** <sup>‡</sup>indicates that the model was pre-trained on ImageNet-22K. <sup>†</sup>indicates increased window size from 7<sup>2</sup> to 12<sup>2</sup>. Throughput is measured on a single A100 GPU. Note that DiNAT<sub>s</sub> is identical in architecture to Swin, and only different in attention modules (WSA/SWSA replaced with NA/DiNA).

<table border="1">
<thead>
<tr>
<th>Variant</th>
<th>Layer structure</th>
<th>ImageNet Top-1 (%)</th>
<th>MSCOCO AP<sup>b</sup></th>
<th>MSCOCO AP<sup>m</sup></th>
<th>ADE20K mIoU</th>
</tr>
</thead>
<tbody>
<tr>
<td>○○ Swin-Tiny</td>
<td>WSA-SWSA</td>
<td>81.3</td>
<td>46.0</td>
<td>41.6</td>
<td>45.8</td>
</tr>
<tr>
<td>○○ NAT<sub>s</sub>-Tiny</td>
<td>NA-NA</td>
<td><b>81.8</b></td>
<td>46.1</td>
<td>41.5</td>
<td>46.2</td>
</tr>
<tr>
<td>○○ DiNAT<sub>s</sub>-Tiny</td>
<td>NA-DiNA</td>
<td><b>81.8</b></td>
<td>46.4</td>
<td><b>41.8</b></td>
<td><b>47.1</b></td>
</tr>
<tr>
<td>● ○</td>
<td>DiNA-NA</td>
<td>81.5</td>
<td><b>46.5</b></td>
<td><b>41.8</b></td>
<td>46.9</td>
</tr>
<tr>
<td>● ●</td>
<td>DiNA-DiNA</td>
<td>79.7</td>
<td>39.8</td>
<td>36.8</td>
<td>40.7</td>
</tr>
<tr>
<td>○○ NAT-Tiny</td>
<td>NA-NA</td>
<td><b>83.2</b></td>
<td>47.7</td>
<td>42.6</td>
<td>48.4</td>
</tr>
<tr>
<td>○○ DiNAT-Tiny</td>
<td>NA-DiNA</td>
<td>82.7</td>
<td>48.3</td>
<td>43.4</td>
<td><b>48.5</b></td>
</tr>
<tr>
<td>● ○</td>
<td>DiNA-NA</td>
<td>82.6</td>
<td><b>48.5</b></td>
<td><b>43.5</b></td>
<td>47.9</td>
</tr>
<tr>
<td>● ●</td>
<td>DiNA-DiNA</td>
<td>82.2</td>
<td>44.9</td>
<td>40.5</td>
<td>45.8</td>
</tr>
</tbody>
</table>

**Table VII. Layer structure impact on performance.**

