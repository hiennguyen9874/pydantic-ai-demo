Title: LAuReL: Learned Augmented Residual Layer

URL Source: https://arxiv.org/html/2411.07501

Markdown Content:
###### Abstract

One of the core pillars of efficient deep learning methods are architectural improvements, such as residual/skip connections, which have led to significantly better model convergence and quality. Since their introduction, residual connections have become ubiquitous not only in convolutional neural networks but also in transformer-based architectures, the backbone of LLMs.

In this paper, we introduce the _Learned Augmented Residual Layer_ (LAuReL)—a novel generalization of the canonical residual connection—designed to serve as an in-situ replacement while outperforming it in both model quality and footprint metrics. Our experiments show that LAuReL can enhance quality for both vision and language models while adding fewer parameters and incurring less latency and memory overhead than naively increasing parameter count.

For example, on the ImageNet-1K task, LAuReL achieves the same model quality improvements as naively adding an extra layer while using 2.6×2.6\times 2.6 × fewer parameters. Similarly, when pre-training 1B and 4B parameter LLMs, LAuReL improves performance on a variety of challenging downstream evaluation tasks by 2.54% to 20.05%, while adding only 0.012% and 0.1% additional parameters, respectively.

Machine Learning, ICML

1 Introduction
--------------

Model efficiency is of critical importance in the age of extremely large language and vision models. Even if a given model has impressive quality, its footprint metrics such as train-time compute, inference latency, resident memory size, peak memory consumption, etc. dictate if it can be experimented with and/or deployed in real-world settings. A large and slow model can be impractical to train and use, making it unsuitable for applications that require fast responses, no matter how well it performs on benchmarks.

LLMs such as Gemini 1.5 Flash (Gemini-Team et al., [2024](https://arxiv.org/html/2411.07501v4#bib.bib12)), DeepSeek V3 (DeepSeek-A.I. et al., [2024](https://arxiv.org/html/2411.07501v4#bib.bib8)) have been explicitly designed with these efficiencies in mind and consistently outperform much larger models that preceded them. Consequently, improving the Pareto-frontier of model quality and model footprint, via efficient learning methods has been an area of active research in the past few years. Areas of interests span from algorithmic techniques (Menghani, [2023](https://arxiv.org/html/2411.07501v4#bib.bib26)) and efficient hardware (Sze et al., [2017](https://arxiv.org/html/2411.07501v4#bib.bib33)) to best practices around model efficiency (Dehghani et al., [2022](https://arxiv.org/html/2411.07501v4#bib.bib9)).

One of the core pillars of efficient deep learning methods are architectural improvements such as the residual/skip connection, which had led to significantly better model convergence and quality ([He et al.,](https://arxiv.org/html/2411.07501v4#bib.bib13)). The residual connection has become ubiquitous not only in convolutional neural networks but also in transformer-based architectures (Vaswani et al., [2017](https://arxiv.org/html/2411.07501v4#bib.bib34)), which are the backbone of today’s LLMs.

In this paper we introduce _learned augmented residual layer_, LAuReL, which generalizes the canonical residual connection. Recall that deep-learning models with residual connections have a ‘block’ structure, with many blocks chained together between the input and final output; these could be convolution/identity blocks within a ResNet, a transformer block in a transformer encoder/decoder, etc. Within a block, a typical residual connection is given by:

x i+1=f⁢(x i)+x i.subscript 𝑥 𝑖 1 𝑓 subscript 𝑥 𝑖 subscript 𝑥 𝑖 x_{i+1}=f(x_{i})+x_{i}.italic_x start_POSTSUBSCRIPT italic_i + 1 end_POSTSUBSCRIPT = italic_f ( italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) + italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT .(1)

Here, f⁢(⋅)𝑓⋅f(\cdot)italic_f ( ⋅ ) can be any non-linear function such as attention, MLP, multiple non-linear layers, etc., x i subscript 𝑥 𝑖 x_{i}italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT is the input to the said non-linear function, and x i+1 subscript 𝑥 𝑖 1 x_{i+1}italic_x start_POSTSUBSCRIPT italic_i + 1 end_POSTSUBSCRIPT is the combined output of the non-linear function and the residual component (Figure[1](https://arxiv.org/html/2411.07501v4#S1.F1 "Figure 1 ‣ 1 Introduction ‣ LAuReL: Learned Augmented Residual Layer")). For simplicity, we ignore pre-processing functions such as layer norm, which can be folded into f⁢(⋅)𝑓⋅f(\cdot)italic_f ( ⋅ ).

![Image 1: Refer to caption](https://arxiv.org/html/2411.07501v4/extracted/6565902/images/Residual-Connection.jpg)

Figure 1: A standard residual connection. We assume the model to be divided into logical ‘blocks’, which is true for most modern architectures including transformers. The residual connection combines the output of a non-linear function f 𝑓 f italic_f and the input to the said non-linear function. Here, f 𝑓 f italic_f can be attention, MLP, or any other combination of non-linear layers.

2 Learned Augmented Residual Layer
----------------------------------

In this section we describe the main idea behind LAuReL. In its most general form, we reformulate the residual connection to be the following:

x i+1=α⋅f⁢(x i)+g⁢(x i,x i−1,…,x 0).subscript 𝑥 𝑖 1⋅𝛼 𝑓 subscript 𝑥 𝑖 𝑔 subscript 𝑥 𝑖 subscript 𝑥 𝑖 1…subscript 𝑥 0 x_{i+1}=\alpha\cdot f(x_{i})+g(x_{i},x_{i-1},\ldots,x_{0}).italic_x start_POSTSUBSCRIPT italic_i + 1 end_POSTSUBSCRIPT = italic_α ⋅ italic_f ( italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) + italic_g ( italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_x start_POSTSUBSCRIPT italic_i - 1 end_POSTSUBSCRIPT , … , italic_x start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT ) .(2)

Here, α 𝛼\alpha italic_α is a learned scalar parameter, and g⁢(⋅)𝑔⋅g(\cdot)italic_g ( ⋅ ) is a learned linear function with x i,x i−1,…,x 0 subscript 𝑥 𝑖 subscript 𝑥 𝑖 1…subscript 𝑥 0 x_{i},x_{i-1},\ldots,x_{0}italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_x start_POSTSUBSCRIPT italic_i - 1 end_POSTSUBSCRIPT , … , italic_x start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT as inputs, where x j subscript 𝑥 𝑗 x_{j}italic_x start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT is the output of the j 𝑗 j italic_j-th residual connection.

The main intuition is that one can learn a richer set of (linear) functions than just using x i subscript 𝑥 𝑖 x_{i}italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT as the residual component. One motivation behind seeking these richer linear functions is the concept of a “residual stream” (Elhage et al., [2021](https://arxiv.org/html/2411.07501v4#bib.bib11)), where the residual connection is considered to be part of a stream of information that passes through each layer without being exposed to any non-linearities. This allows the learning process to focus on the non-linear components better.

Each layer/operation can read from, and subsequently write to this residual stream. Since the residual connection has been shown to be important for model quality and convergence, we designed LAuReL to operate on this residual stream in a learned fashion, while being light-weight in terms of the model size and latency changes.

![Image 2: Refer to caption](https://arxiv.org/html/2411.07501v4/extracted/6565902/images/Laurel.jpg)

Figure 2: An illustration of the LAuReL framework; see ([2](https://arxiv.org/html/2411.07501v4#S2.E2 "Equation 2 ‣ 2 Learned Augmented Residual Layer ‣ LAuReL: Learned Augmented Residual Layer")). LAuReL can be used to replace the regular residual connection in Figure [1](https://arxiv.org/html/2411.07501v4#S1.F1 "Figure 1 ‣ 1 Introduction ‣ LAuReL: Learned Augmented Residual Layer"). Again, f 𝑓 f italic_f can be any non-linear function such as attention, MLPs, and groups of multiple non-linear layers.

In this paper we study three specific versions of the LAuReL framework; although as described in ([2](https://arxiv.org/html/2411.07501v4#S2.E2 "Equation 2 ‣ 2 Learned Augmented Residual Layer ‣ LAuReL: Learned Augmented Residual Layer")), the framework can be generalized beyond these versions.

### 2.1 Residual Weights Version (LAuReL-RW)

In this version, we keep α 𝛼\alpha italic_α learnable and set g⁢(x i,…,x 0)=β⁢x i 𝑔 subscript 𝑥 𝑖…subscript 𝑥 0 𝛽 subscript 𝑥 𝑖 g(x_{i},\ldots,x_{0})=\beta x_{i}italic_g ( italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , … , italic_x start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT ) = italic_β italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT. Therefore, ([2](https://arxiv.org/html/2411.07501v4#S2.E2 "Equation 2 ‣ 2 Learned Augmented Residual Layer ‣ LAuReL: Learned Augmented Residual Layer")) can be rewritten as:

x i+1=α⁢f⁢(x i)+β⁢x i.subscript 𝑥 𝑖 1 𝛼 𝑓 subscript 𝑥 𝑖 𝛽 subscript 𝑥 𝑖 x_{i+1}=\alpha f(x_{i})+\beta x_{i}.italic_x start_POSTSUBSCRIPT italic_i + 1 end_POSTSUBSCRIPT = italic_α italic_f ( italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) + italic_β italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT .

Notice that this version assigns learnable weights to the f⁢(x i)𝑓 subscript 𝑥 𝑖 f(x_{i})italic_f ( italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) and x i subscript 𝑥 𝑖 x_{i}italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT from ([1](https://arxiv.org/html/2411.07501v4#S1.E1 "Equation 1 ‣ 1 Introduction ‣ LAuReL: Learned Augmented Residual Layer")). In practice, we found that we cannot let α 𝛼\alpha italic_α and β 𝛽\beta italic_β grow unbounded, and using a normalization function such as softmax or sigmoid helps. Clearly, this version will add only two new parameters per LAuReL layer. If necessary, we can replace these two parameters by a single learnable parameter and use a function such as sigmoid to define α,β 𝛼 𝛽\alpha,\beta italic_α , italic_β in terms of this single parameter.

This variant can be useful for learning the relative importance of the non-linear component (f⁢(x i)𝑓 subscript 𝑥 𝑖 f(x_{i})italic_f ( italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT )) and the residual input (x i subscript 𝑥 𝑖 x_{i}italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT). In the earlier layers the former might be more important, while in the later layers, the latter could be useful for mitigating problems such as vanishing gradients. This variant can help by adaptively learning these weights.

### 2.2 Low-Rank Version (LAuReL-LR)

Here, we fix α=1 𝛼 1\alpha=1 italic_α = 1, and g⁢(x i)=W⁢x i 𝑔 subscript 𝑥 𝑖 𝑊 subscript 𝑥 𝑖 g(x_{i})=Wx_{i}italic_g ( italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) = italic_W italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT in ([2](https://arxiv.org/html/2411.07501v4#S2.E2 "Equation 2 ‣ 2 Learned Augmented Residual Layer ‣ LAuReL: Learned Augmented Residual Layer")) to obtain:

x i+1=f⁢(x i)+W⁢x i.subscript 𝑥 𝑖 1 𝑓 subscript 𝑥 𝑖 𝑊 subscript 𝑥 𝑖 x_{i+1}=f(x_{i})+Wx_{i}.italic_x start_POSTSUBSCRIPT italic_i + 1 end_POSTSUBSCRIPT = italic_f ( italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) + italic_W italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT .

As written, W 𝑊 W italic_W is a learnable D×D 𝐷 𝐷 D\times D italic_D × italic_D matrix, where D 𝐷 D italic_D is the model dimension for transformer-based models, or more generally it is the last dimension of x i subscript 𝑥 𝑖 x_{i}italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT. Hence, the W 𝑊 W italic_W matrix will add D 2 superscript 𝐷 2 D^{2}italic_D start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT new parameters (per LAuReL layer).

In practice, to reduce the number of new parameters added to the model and to help with convergence, we consider a low rank version of W 𝑊 W italic_W. In particular, let W=A×B+I 𝑊 𝐴 𝐵 𝐼 W=A\times B+I italic_W = italic_A × italic_B + italic_I, where A 𝐴 A italic_A and B T superscript 𝐵 𝑇 B^{T}italic_B start_POSTSUPERSCRIPT italic_T end_POSTSUPERSCRIPT are D×r 𝐷 𝑟 D\times r italic_D × italic_r matrices and r≪D much-less-than 𝑟 𝐷 r\ll D italic_r ≪ italic_D. Thus, we can rewrite ([2](https://arxiv.org/html/2411.07501v4#S2.E2 "Equation 2 ‣ 2 Learned Augmented Residual Layer ‣ LAuReL: Learned Augmented Residual Layer")) as:

x i+1=f⁢(x i)+B⁢A⁢x i+x i.subscript 𝑥 𝑖 1 𝑓 subscript 𝑥 𝑖 𝐵 𝐴 subscript 𝑥 𝑖 subscript 𝑥 𝑖 x_{i+1}=f(x_{i})+BAx_{i}+x_{i}.italic_x start_POSTSUBSCRIPT italic_i + 1 end_POSTSUBSCRIPT = italic_f ( italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) + italic_B italic_A italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT + italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT .(3)

Here, both A 𝐴 A italic_A and B 𝐵 B italic_B matrices are learnable. The number of new parameters is 2⁢r⁢D 2 𝑟 𝐷 2rD 2 italic_r italic_D, per LAuReL layer.

This variant helps with allocating learning capacity for the linear part of the network (the residual input, i.e., x i subscript 𝑥 𝑖 x_{i}italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT in Figure 2), such that the main network can use its capacity towards learning better nonlinear functions (f⁢(x i)𝑓 subscript 𝑥 𝑖 f(x_{i})italic_f ( italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT )), while LAuReL contributes the linear components (x i+B⁢A⁢x i subscript 𝑥 𝑖 𝐵 𝐴 subscript 𝑥 𝑖 x_{i}+BAx_{i}italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT + italic_B italic_A italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT) to the residual stream.

### 2.3 Previous Activations Version (LAuReL-PA)

This is similar to LAuReL-LR, except that we use k 𝑘 k italic_k activations from the previous blocks. In particular, we set

g⁢(x i,…,x 0)=x i+∑j=0 k−1 γ i,j⋅h i⁢(x i−j),𝑔 subscript 𝑥 𝑖…subscript 𝑥 0 subscript 𝑥 𝑖 superscript subscript 𝑗 0 𝑘 1⋅subscript 𝛾 𝑖 𝑗 subscript ℎ 𝑖 subscript 𝑥 𝑖 𝑗 g(x_{i},\ldots,x_{0})=x_{i}+\sum_{j=0}^{k-1}\gamma_{i,j}\cdot h_{i}(x_{i-j}),italic_g ( italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , … , italic_x start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT ) = italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT + ∑ start_POSTSUBSCRIPT italic_j = 0 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k - 1 end_POSTSUPERSCRIPT italic_γ start_POSTSUBSCRIPT italic_i , italic_j end_POSTSUBSCRIPT ⋅ italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( italic_x start_POSTSUBSCRIPT italic_i - italic_j end_POSTSUBSCRIPT ) ,

where γ i,0,…,γ i,k−1 subscript 𝛾 𝑖 0…subscript 𝛾 𝑖 𝑘 1\gamma_{i,0},\ldots,\gamma_{i,k-1}italic_γ start_POSTSUBSCRIPT italic_i , 0 end_POSTSUBSCRIPT , … , italic_γ start_POSTSUBSCRIPT italic_i , italic_k - 1 end_POSTSUBSCRIPT are learned scalar parameters and h i subscript ℎ 𝑖 h_{i}italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT is another linear function.1 1 1 For simplicity, we fix α=1 𝛼 1\alpha=1 italic_α = 1. This allows us to rewrite ([2](https://arxiv.org/html/2411.07501v4#S2.E2 "Equation 2 ‣ 2 Learned Augmented Residual Layer ‣ LAuReL: Learned Augmented Residual Layer")) as:

x i+1=f⁢(x i)+x i+∑j=0 k−1 γ i,j⋅h i⁢(x i−j).subscript 𝑥 𝑖 1 𝑓 subscript 𝑥 𝑖 subscript 𝑥 𝑖 superscript subscript 𝑗 0 𝑘 1⋅subscript 𝛾 𝑖 𝑗 subscript ℎ 𝑖 subscript 𝑥 𝑖 𝑗 x_{i+1}=f(x_{i})+x_{i}+\sum_{j=0}^{k-1}\gamma_{i,j}\cdot h_{i}(x_{i-j}).italic_x start_POSTSUBSCRIPT italic_i + 1 end_POSTSUBSCRIPT = italic_f ( italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) + italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT + ∑ start_POSTSUBSCRIPT italic_j = 0 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k - 1 end_POSTSUPERSCRIPT italic_γ start_POSTSUBSCRIPT italic_i , italic_j end_POSTSUBSCRIPT ⋅ italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( italic_x start_POSTSUBSCRIPT italic_i - italic_j end_POSTSUBSCRIPT ) .(4)

In practice, we replace h i subscript ℎ 𝑖 h_{i}italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT by a low-rank product similar to the LAuReL-LR version, but using the identity function is also an option. When using a rank r 𝑟 r italic_r product for h i subscript ℎ 𝑖 h_{i}italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT, the number of new parameters per LAuReL layer is 2⁢r⁢D+k 2 𝑟 𝐷 𝑘 2rD+k 2 italic_r italic_D + italic_k, where k 𝑘 k italic_k is the number of previous activations used.

We can consider this variant to be a hybrid of the LAuReL-RW and LAuReL-LR variants, where multiple previous activations are used in a weighted manner that is learned. This allows layers accelerated access to previous activations, along with learning their relative importance.

### 2.4 Other Derived Variants

All three proposed LAuReL versions are a combination of scalar and/or low-rank products on top of the vanilla residual connection in ([1](https://arxiv.org/html/2411.07501v4#S1.E1 "Equation 1 ‣ 1 Introduction ‣ LAuReL: Learned Augmented Residual Layer")). This makes it especially light-weight in terms of its impact on model size and latency. We discuss the efficiency of the variants in more detail in Section[4](https://arxiv.org/html/2411.07501v4#S4 "4 Efficiency ‣ LAuReL: Learned Augmented Residual Layer").

That being said, LAuReL is generic enough to allow combinations of the above variants, as well as new variants. For instance, one straight-forward combination is LAuReL-RW+LR, where the residual weights from LAuReL-RW can be applied along with the LAuReL-LR variant to rewrite ([3](https://arxiv.org/html/2411.07501v4#S2.E3 "Equation 3 ‣ 2.2 Low-Rank Version (LAuReL-LR) ‣ 2 Learned Augmented Residual Layer ‣ LAuReL: Learned Augmented Residual Layer")) as follows:

x i+1=α⁢f⁢(x i)+β⁢(B⁢A⁢x i+x i).subscript 𝑥 𝑖 1 𝛼 𝑓 subscript 𝑥 𝑖 𝛽 𝐵 𝐴 subscript 𝑥 𝑖 subscript 𝑥 𝑖 x_{i+1}=\alpha f(x_{i})+\beta(BAx_{i}+x_{i}).italic_x start_POSTSUBSCRIPT italic_i + 1 end_POSTSUBSCRIPT = italic_α italic_f ( italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) + italic_β ( italic_B italic_A italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT + italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) .(5)

Similarly, LAuReL-PA as defined in ([4](https://arxiv.org/html/2411.07501v4#S2.E4 "Equation 4 ‣ 2.3 Previous Activations Version (LAuReL-PA) ‣ 2 Learned Augmented Residual Layer ‣ LAuReL: Learned Augmented Residual Layer")) can be combined with LAuReL-RW as follows:

x i+1=α⁢f⁢(x i)+β⁢(x i+∑j=0 k−1 γ i,j⋅h i⁢(x i−j)).subscript 𝑥 𝑖 1 𝛼 𝑓 subscript 𝑥 𝑖 𝛽 subscript 𝑥 𝑖 superscript subscript 𝑗 0 𝑘 1⋅subscript 𝛾 𝑖 𝑗 subscript ℎ 𝑖 subscript 𝑥 𝑖 𝑗 x_{i+1}=\alpha f(x_{i})+\beta\left(x_{i}+\sum_{j=0}^{k-1}\gamma_{i,j}\cdot h_{% i}(x_{i-j})\right).italic_x start_POSTSUBSCRIPT italic_i + 1 end_POSTSUBSCRIPT = italic_α italic_f ( italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) + italic_β ( italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT + ∑ start_POSTSUBSCRIPT italic_j = 0 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k - 1 end_POSTSUPERSCRIPT italic_γ start_POSTSUBSCRIPT italic_i , italic_j end_POSTSUBSCRIPT ⋅ italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( italic_x start_POSTSUBSCRIPT italic_i - italic_j end_POSTSUBSCRIPT ) ) .

When using a low-rank product for h i subscript ℎ 𝑖 h_{i}italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT, we can create the LAuReL-RW+LR+PA variant as follows:

x i+1=α⁢f⁢(x i)+β⁢(x i+∑j=0 k−1 γ i,j⋅A i,j⁢B i,j⁢(x i−j)).subscript 𝑥 𝑖 1 𝛼 𝑓 subscript 𝑥 𝑖 𝛽 subscript 𝑥 𝑖 superscript subscript 𝑗 0 𝑘 1⋅subscript 𝛾 𝑖 𝑗 subscript 𝐴 𝑖 𝑗 subscript 𝐵 𝑖 𝑗 subscript 𝑥 𝑖 𝑗 x_{i+1}=\alpha f(x_{i})+\beta\left(x_{i}+\sum_{j=0}^{k-1}\gamma_{i,j}\cdot A_{% i,j}B_{i,j}(x_{i-j})\right).italic_x start_POSTSUBSCRIPT italic_i + 1 end_POSTSUBSCRIPT = italic_α italic_f ( italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) + italic_β ( italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT + ∑ start_POSTSUBSCRIPT italic_j = 0 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k - 1 end_POSTSUPERSCRIPT italic_γ start_POSTSUBSCRIPT italic_i , italic_j end_POSTSUBSCRIPT ⋅ italic_A start_POSTSUBSCRIPT italic_i , italic_j end_POSTSUBSCRIPT italic_B start_POSTSUBSCRIPT italic_i , italic_j end_POSTSUBSCRIPT ( italic_x start_POSTSUBSCRIPT italic_i - italic_j end_POSTSUBSCRIPT ) ) .(6)

Yet another variant while slightly relaxing our above formulations would be to treat α 𝛼\alpha italic_α and β 𝛽\beta italic_β as vectors of length D 𝐷 D italic_D. Here, we would learn fine-grained per-dimension weights when mixing x i subscript 𝑥 𝑖 x_{i}italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT and f⁢(x i)𝑓 subscript 𝑥 𝑖 f(x_{i})italic_f ( italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) at the cost of 2⁢D 2 𝐷 2D 2 italic_D parameters per LAuReL-LR layer, instead of two per parameters per layer.

To summarize, LAuReL is inherently flexible and provides many possible cheap learnable augmentations on top of the vanilla residual connection. In the following section we demonstrate that these combinations are efficient and effective at improving the model quality of common vision and language models, while having a minimal impact on the number of parameters and latency.

3 Experiments
-------------

We experiment with LAuReL in two domains, namely, vision and language. For the first case, our goal is to improve the image classification accuracy of the ResNet-50 model on the ImageNet-1K dataset (Deng et al., [2009](https://arxiv.org/html/2411.07501v4#bib.bib10)). For the second case our goal is to improve the performance of two different large language models (LLMs) of size 1B and 4B parameters respectively, evaluated after the pre-training stage, on common benchmarks.

The underlying motivation behind these experiments is not necessarily to improve on the SOTA results, but to show how LAuReL can be easily integrated on top of common model architectures with residual/skip connections in order to achieve a better model quality and footprint trade off.

### 3.1 ResNet-50 on ImageNet-1K

In this setup we train a standard ResNet-50 model on the ImageNet 1K dataset (Deng et al., [2009](https://arxiv.org/html/2411.07501v4#bib.bib10)) using 16 Google Cloud TPUv5e chips over one epoch with data-augmentation turned on. In order to obtain a strong baseline, we fine-tuned the model learning rate schedule and picked a schedule that maximized the average of the best accuracy@1 values over 5 trials (which we simply refer to as accuracy in this subsection). The baseline model that we obtained achieves an accuracy of 74.95±0.016%plus-or-minus 74.95 percent 0.016 74.95\pm 0.016\%74.95 ± 0.016 %.

In addition, we also find that if we simply add another layer to the ResNet-50 model (i.e., naive scaling), we can increase the model’s accuracy by 0.25%percent 0.25 0.25\%0.25 % to reach 75.20%percent 75.20 75.20\%75.20 %, while adding 4.37%percent 4.37 4.37\%4.37 % new parameters. With that in context, applying LAuReL on the model improves it (Table [1](https://arxiv.org/html/2411.07501v4#S3.T1 "Table 1 ‣ 3.1 ResNet-50 on ImageNet-1K ‣ 3 Experiments ‣ LAuReL: Learned Augmented Residual Layer")).

If we only use the LAuReL-RW version, we get an improvement of 0.15%percent 0.15 0.15\%0.15 % on average with only 0.003%percent 0.003 0.003\%0.003 % extra parameters, which is essentially negligible. When we try the LAuReL-RW+LR version from ([5](https://arxiv.org/html/2411.07501v4#S2.E5 "Equation 5 ‣ 2.4 Other Derived Variants ‣ 2 Learned Augmented Residual Layer ‣ LAuReL: Learned Augmented Residual Layer")) with r=16 𝑟 16 r=16 italic_r = 16, we achieve 75.20%percent 75.20 75.20\%75.20 % accuracy while adding only 1.68%percent 1.68 1.68\%1.68 % new parameters; this matches the performance of the baseline with an extra layer, while using 2.6×2.6\times 2.6 × fewer extra parameters. Additionally, when we use the combined LAuReL-RW+LR+PA version from ([6](https://arxiv.org/html/2411.07501v4#S2.E6 "Equation 6 ‣ 2.4 Other Derived Variants ‣ 2 Learned Augmented Residual Layer ‣ LAuReL: Learned Augmented Residual Layer")), we improve the accuracy to 75.25%percent 75.25 75.25\%75.25 % while still using 1.82×1.82\times 1.82 × fewer extra parameters than the baseline with one extra layer, demonstrating that LAuReL is superior to naively scaling the model. Notably, despite substantial changes to the residual connection, we did not find any training instabilities with LAuReL.

Table 1: Applying LAuReL on a ResNet-50 trained on the ImageNet 1K classification dataset. LAuReL-RW provides a significant boost with negligible extra parameters. LAuReL-RW+LR and LAuReL-RW+LR+PA meet and beat the naive scaling baseline while using 2.6×2.6\times 2.6 × and 1.82×1.82\times 1.82 × fewer parameters. Results that provide statistically significant boost over the baseline are highlighted in green and bold.

Model Avg. Best Params Added
Accuracy@1 vs Baseline
(%), 5 trials(%)
Baseline 74.95±0.01 plus-or-minus 74.95 0.01 74.95\pm 0.01 74.95 ± 0.01 0.00 0.00 0.00 0.00
Baseline + 1 Layer 75.20±0.12 plus-or-minus 75.20 0.12 75.20\pm 0.12 75.20 ± 0.12 4.37 4.37 4.37 4.37
(Naive Scaling)
LAuReL-RW 75.10±0.10 plus-or-minus 75.10 0.10\bf 75.10\pm 0.10 bold_75.10 ± bold_0.10 0.003 0.003\bf 0.003 bold_0.003
LAuReL-RW+LR 75.20±0.07 plus-or-minus 75.20 0.07\bf 75.20\pm 0.07 bold_75.20 ± bold_0.07 1.68 1.68\bf 1.68 bold_1.68
LAuReL-RW+LR+PA 75.25±0.09 plus-or-minus 75.25 0.09\bf{75.25\pm 0.09}bold_75.25 ± bold_0.09 2.40 2.40\bf 2.40 bold_2.40

### 3.2 Large-Scale LLM Pre-training

In this setup, our goal was to test the performance of LAuReL when applied on top of strong LLMs. During the course of our work we evaluated LAuReL on two separate LLMs, which we pre-trained from scratch. The first LLM (LLM-1B) is a 1B parameter model pre-trained with a data-mixture consisting of only text tokenss. The second LLM (LLM-4B) is a 4B parameter model that was pre-trained with a multi-modal and multi-lingual data mixture. Both LLMs were trained with ∼0.5 similar-to absent 0.5\sim 0.5∼ 0.5 T tokens.

What we varied across the two LLMs was to allow for different budgets for increasing model footprint (parameters, latency, memory, etc.) when applying LAuReL. For LLM-1B we allow a very small increase in these metrics (∼similar-to\sim∼0.01% extra parameters, and nearly no latency increase). For LLM-4B we allow a lenient, yet modest increase in parameters (∼similar-to\sim∼0.1% extra parameters), and latency (1% increase).

Given the scale of LLMs today, both the budgets would be considered negligible. For instance, a 0.1%percent 0.1 0.1\%0.1 % increase in parameters for a 4B model will only correspond to 4M more parameters. As demonstrated in the ResNet-50/ImageNet experiments (Section [3.1](https://arxiv.org/html/2411.07501v4#S3.SS1 "3.1 ResNet-50 on ImageNet-1K ‣ 3 Experiments ‣ LAuReL: Learned Augmented Residual Layer")), LAuReL outperforms naive scaling; see Section [4.5](https://arxiv.org/html/2411.07501v4#S4.SS5 "4.5 Comparison with Naive Model Scaling ‣ 4 Efficiency ‣ LAuReL: Learned Augmented Residual Layer") for a more detailed comparison.

Our objective behind testing LAuReL in these conditions was to demonstrate its efficacy and ensure that it scales well across different LLM setups in the wild.

#### 3.2.1 LLM-1B: Very low additional footprint

For our first baseline, we chose a 1B parameter decoder-only transformer-based model. We pre-trained both the baseline, and our experiment with LAuReL, from scratch; we use the LAuReL-RW and LAuReL-LR versions (with r=4 𝑟 4 r=4 italic_r = 4). Both the models were trained using 256 Google Cloud TPU v5e chips for approximately two weeks each, using a pre-training mixture consisting of only text data that included webpages, books, code, and translations.

It is worth noting that the combined LAuReL-RW+LR variant adds only 0.012% more parameters as compared to the baseline model. Since we chose r=4 𝑟 4 r=4 italic_r = 4, the number of parameters added by LAuReL-LR is 8⁢N⁢D 8 𝑁 𝐷 8ND 8 italic_N italic_D and the number of parameters added by LAuReL-RW is 2⁢N 2 𝑁 2N 2 italic_N, for a total of 2⁢N⁢(4⁢D+1)2 𝑁 4 𝐷 1 2N(4D+1)2 italic_N ( 4 italic_D + 1 ) additional parameters. Typically N∈[10,100]𝑁 10 100 N\in[10,100]italic_N ∈ [ 10 , 100 ] and D∈[500,5000]𝐷 500 5000 D\in[500,5000]italic_D ∈ [ 500 , 5000 ]. For the sake of illustration, assuming N=20 𝑁 20 N=20 italic_N = 20 and D=1000 𝐷 1000 D=1000 italic_D = 1000 and using LAuReL-RW+LR leads to 160,040 160 040 160,040 160 , 040 extra parameters in a 1B parameter model. Thus, the number of new parameters is dwarfed by that of the original model. Furthermore, the additional latency introduced by LAuReL-RW+LR was within the noise range.

We evaluated both the pre-trained baseline and LAuReL models on a host of common LLM tasks such as Q&A, NLU, Math, Code, etc; see Table [2](https://arxiv.org/html/2411.07501v4#S3.T2 "Table 2 ‣ 3.2.1 LLM-1B: Very low additional footprint ‣ 3.2 Large-Scale LLM Pre-training ‣ 3 Experiments ‣ LAuReL: Learned Augmented Residual Layer") for the results. The task type and individual tasks are listed in the first and second columns respectively, and a higher score is better for all the tasks. LAuReL outperforms the baselines on all tasks except on the MBPP dataset where it was neutral. To reiterate, these improvements were achieved with only 0.012% extra parameters and nearly no increase in latency.

Table 2: Evaluation results on LLM-1B as described in Section [3.2.1](https://arxiv.org/html/2411.07501v4#S3.SS2.SSS1 "3.2.1 LLM-1B: Very low additional footprint ‣ 3.2 Large-Scale LLM Pre-training ‣ 3 Experiments ‣ LAuReL: Learned Augmented Residual Layer"); a 1B parameter decoder-only LLM pre-trained from scratch with (a) once with the baseline model architecture, and (b) once using LAuReL on top. We evaluated both the models on a number of common evaluation benchmarks (higher is better for all task type and task combinations listed below). LAuReL variant outperforms the baseline on all but one dataset while adding only 0.012% extra parameters. Results that provide ≥2%absent percent 2\geq 2\%≥ 2 % relative improvement over the baseline are highlighted in green and bold.

Task Type Task Baseline LAuReL
Math Math 3.54 3.70
(Hendrycks et al., [2021b](https://arxiv.org/html/2411.07501v4#bib.bib16))(+4.51%)
GSM8K-CoT 8.34 8.79
(Cobbe et al., [2021](https://arxiv.org/html/2411.07501v4#bib.bib7))(+5.39%)
General MMLU 25.72 25.89
Reasoning(Hendrycks et al., [2021a](https://arxiv.org/html/2411.07501v4#bib.bib15))(+0.06%)
Q&A BoolQ 58.07 65.66
(Clark et al., [2019](https://arxiv.org/html/2411.07501v4#bib.bib5))(+13.07%)
TyDi QA 67.98 72.58
(GoldP)(+6.76%)
(Clark et al., [2020](https://arxiv.org/html/2411.07501v4#bib.bib6))
Sentence Hellaswag 64.84 65.06
Completion(Zellers et al., [2019](https://arxiv.org/html/2411.07501v4#bib.bib36))(+0.03%)
Code HumanEval 18.29 18.90
(Chen et al., [2021](https://arxiv.org/html/2411.07501v4#bib.bib4))(3.33%)
MBPP 27.00 27.00
(Austin et al., [2021](https://arxiv.org/html/2411.07501v4#bib.bib1))
GSM8K-PAL 10.31 11.37
(Cobbe et al., [2021](https://arxiv.org/html/2411.07501v4#bib.bib7))(10.28%)

Table 3: Evaluation results on LLM-4B as described in Section [3.2.2](https://arxiv.org/html/2411.07501v4#S3.SS2.SSS2 "3.2.2 LLM-4B: Low additional footprint ‣ 3.2 Large-Scale LLM Pre-training ‣ 3 Experiments ‣ LAuReL: Learned Augmented Residual Layer"); a 4B parameter decoder-only LLM pre-trained from scratch with (a) once with the baseline model architecture, and (b) once using LAuReL on top. We evaluated both the models on a number of common evaluation benchmarks (higher is better for all task type and task combinations listed below). LAuReL variant outperforms the baseline on all but two tasks, while adding only ∼similar-to\sim∼ 0.1% extra parameters. Results that provide statistically significant improvement over the baseline are highlighted in green and bold.

Task Type Task Baseline LAuReL
Math Math 14.70 15.30
(Hendrycks et al., [2021b](https://arxiv.org/html/2411.07501v4#bib.bib16))(+4.08%)
MGSM 20.0 23.09
(Shi et al., [2023](https://arxiv.org/html/2411.07501v4#bib.bib30))(+15.45%)
General MMLU 49.85 51.12
Reasoning(Hendrycks et al., [2021a](https://arxiv.org/html/2411.07501v4#bib.bib15))(2.54%)
Reading Belebele 58.40 63.23
Compre-(Bandarkar et al., [2024](https://arxiv.org/html/2411.07501v4#bib.bib2))(+8.27%)
hension BookQA 50.36 60.46
(Mihaylov et al., [2018](https://arxiv.org/html/2411.07501v4#bib.bib27))(+20.05%)
Translation WMT23 68.32 68.24
(Kocmi et al., [2023](https://arxiv.org/html/2411.07501v4#bib.bib22))(-0.11%)
Multimodal MMMU 32.22 36.33
(Yue et al., [2024](https://arxiv.org/html/2411.07501v4#bib.bib35))(+12.75%)
Coco-Caption 95.69 99.15
(Lin et al., [2014](https://arxiv.org/html/2411.07501v4#bib.bib24))(+3.61%)
DocVQA 68.28 68.34
(Mathew et al., [2021](https://arxiv.org/html/2411.07501v4#bib.bib25))(+0.08%)
TextVQA 60.07 62.64
(Singh et al., [2019](https://arxiv.org/html/2411.07501v4#bib.bib31))(+4.27%)

#### 3.2.2 LLM-4B: Low additional footprint

In this second setting, we experimented with a 4B parameter decoder-only model with a similar token budget, but trained on a multimodal and multilingual corpus of tokens.

To compensate for a 4×4\times 4 × larger model, and also the fact that the dataset and the evaluation tasks are harder, we allowing a bigger footprint budget for LAuReL; ∼0.1%similar-to absent percent 0.1\sim 0.1\%∼ 0.1 % extra parameters and ∼1%similar-to absent percent 1\sim 1\%∼ 1 % extra latency. Note that this is still a negligible increase in model parameters and latency.

To match these budgets, we set r=64 𝑟 64 r=64 italic_r = 64, which provides a favorable trade-off of providing more capacity to the low-rank matrices as specified in the LAuReL-RW+LR formulation in ([6](https://arxiv.org/html/2411.07501v4#S2.E6 "Equation 6 ‣ 2.4 Other Derived Variants ‣ 2 Learned Augmented Residual Layer ‣ LAuReL: Learned Augmented Residual Layer")), while also meeting the above-mentioned budgets.

In this setting, both the baseline and the LAuReL experiment were trained using 1024 Google Cloud TPU v4 chips for slightly more than two days each. See Table[3](https://arxiv.org/html/2411.07501v4#S3.T3 "Table 3 ‣ 3.2.1 LLM-1B: Very low additional footprint ‣ 3.2 Large-Scale LLM Pre-training ‣ 3 Experiments ‣ LAuReL: Learned Augmented Residual Layer") for the evaluation results of both the baseline model and the model with LAuReL. The task type and individual tasks are listed in the first and second columns respectively, and a higher score is better for all the tasks. LLM-4B had a sophisticated suite of evaluation tasks, including math, general reasoning, reading comprehension, translation, and multimodal tasks. All of the listed evaluation tasks are used by leading LLMs to evaluate model quality, further solidifying our confidence in the model evaluation setup.

To start off, two of the LLM-4B evaluation tasks, Math, and MMLU were common with LLM-1B (refer to Table[2](https://arxiv.org/html/2411.07501v4#S3.T2 "Table 2 ‣ 3.2.1 LLM-1B: Very low additional footprint ‣ 3.2 Large-Scale LLM Pre-training ‣ 3 Experiments ‣ LAuReL: Learned Augmented Residual Layer") for LLM-1B results). It can be seen that LLM-4B is much stronger than LLM-1B on both the common evals; 4.16×4.16\times 4.16 × better on the Math task, and 1.93×1.93\times 1.93 × better on the MMLU task. We attribute this to LLM-4B being 4×4\times 4 × larger, and having a more sophisticated pre-training mixture. LAuReL using the LAuReL-RW+LR variant improves on both Math and MMLU, which is pleasantly surprising given how powerful is LLM-4B. For other tasks LAuReL improves significantly over the baseline as well, except for WMT23 and DocVQA, where it was neutral.

In terms of costs, the model adds ∼0.1%similar-to absent percent 0.1\sim 0.1\%∼ 0.1 % more parameters. This is still a very reasonable since it means additional 4M parameters on top of a 4B parameter model. In terms of latency, we tested on both server-side (Google CloudTPU) for cloud serving, and a leading smartphone for on-device inference. In both the benchmarks, we measure nearly 1–2% increase in latency for prefill and generation. There was no human perceptible difference in terms of time-to-first-token.

We did not try the LAuReL-PA version for the above LLM experiments, as the LLM training was expensive. However, we expect the LAuReL-PA results from the ResNet experiments to also hold in this case as well.

### 3.3 LAuReL-LR: Rank vs Accuracy

We note that for the LAuReL-LR version on the ResNet-50/ImageNet combination, there is a pattern in terms of the best accuracy achieved with different values of r 𝑟 r italic_r. In the combined LAuReL-RW+LR version, we experimented with different values of r 𝑟 r italic_r, and computed the average of the best accuracy@1 achieved over 5 trials; see Figure [3](https://arxiv.org/html/2411.07501v4#S3.F3 "Figure 3 ‣ 3.3 LAuReL-LR: Rank vs Accuracy ‣ 3 Experiments ‣ LAuReL: Learned Augmented Residual Layer"). From Table [1](https://arxiv.org/html/2411.07501v4#S3.T1 "Table 1 ‣ 3.1 ResNet-50 on ImageNet-1K ‣ 3 Experiments ‣ LAuReL: Learned Augmented Residual Layer"), with the LAuReL-RW version alone we already achieve an average best accuracy@1 of 75.10%, therefore for the combined LAuReL-RW+LR version we would like to see the accuracy exceeding that.

We observe that when r 𝑟 r italic_r is small (r∈{4,8}𝑟 4 8 r\in\{4,8\}italic_r ∈ { 4 , 8 }), there is not a significant improvement over the baseline LAuReL-RW experiment. This could be because a very small r 𝑟 r italic_r acts as an information bottleneck in the low-rank product in ([3](https://arxiv.org/html/2411.07501v4#S2.E3 "Equation 3 ‣ 2.2 Low-Rank Version (LAuReL-LR) ‣ 2 Learned Augmented Residual Layer ‣ LAuReL: Learned Augmented Residual Layer")). As r 𝑟 r italic_r increases, the accuracy reaches the maximum for r∈{16,32}𝑟 16 32 r\in\{16,32\}italic_r ∈ { 16 , 32 }; beyond this, the accuracy seems to drop though still higher than the LAuReL-RW baseline.

We believe this unimodal phenomenon could be due to the number of parameters added to the model (which increases linearly in r 𝑟 r italic_r), since this would also require appropriate tuning of hyperparameters such as the learning rate as well as the regularization penalty.

Another possible cause could be how the low-rank matrices A 𝐴 A italic_A and B 𝐵 B italic_B are initialized. For A∈ℝ D×r 𝐴 superscript ℝ 𝐷 𝑟 A\in\mathbb{R}^{D\times r}italic_A ∈ blackboard_R start_POSTSUPERSCRIPT italic_D × italic_r end_POSTSUPERSCRIPT, we used the Xavier initialization for the ResNet experiment and a column orthogonal initialization for the LLM experiments 2 2 2 We use A i,j=1/r⁢D subscript 𝐴 𝑖 𝑗 1 𝑟 𝐷 A_{i,j}=1/\sqrt{rD}italic_A start_POSTSUBSCRIPT italic_i , italic_j end_POSTSUBSCRIPT = 1 / square-root start_ARG italic_r italic_D end_ARG if i⁢mod⁢r=j 𝑖 mod 𝑟 𝑗 i~{}\mathrm{mod}~{}r=j italic_i roman_mod italic_r = italic_j and 0 0 otherwise. while B 𝐵 B italic_B was always initialized to zero; this is similar to the scheme used in Hu et al. ([2022](https://arxiv.org/html/2411.07501v4#bib.bib18)). We found that initialization made a significant difference in the performance of the LAuReL-LR variant, and we posit that further work studying and improving the initialization scheme of the low-rank matrices could lead to better performance of the variant.

In terms of the tuning required, since r≪D much-less-than 𝑟 𝐷 r\ll D italic_r ≪ italic_D, and if D=512,768,1024,…𝐷 512 768 1024…D=512,768,1024,\dots italic_D = 512 , 768 , 1024 , … as in typical LLMs, this leaves a small range of discrete values for r 𝑟 r italic_r (unlike real-valued hyperparameters such as learning rate, weight decay, etc). In our experience r∈{32,48,64}𝑟 32 48 64 r\in\{32,48,64\}italic_r ∈ { 32 , 48 , 64 } work well for LLMs.

![Image 3: Refer to caption](https://arxiv.org/html/2411.07501v4/extracted/6565902/images/Laurel-LR-Rank-vs-Accuracy.png)

Figure 3: Average best accuracy@1 vs the rank (r 𝑟 r italic_r) in the LAuReL-RW+LR variant.

4 Efficiency
------------

Large models (LLMs and beyond) have been scaled both in the number of parameters as well as the number of tokens to achieve better model quality that scaling laws (Hoffmann et al., [2022](https://arxiv.org/html/2411.07501v4#bib.bib17)) promise. However, this is directly in conflict with keeping training and inference costs reasonably low (we describe these costs shortly). These competing forces have resulted in the emergence of models such as Gemini Flash and Nano (Gemini-Team et al., [2024](https://arxiv.org/html/2411.07501v4#bib.bib12)), DeepSeek (DeepSeek-A.I. et al., [2024](https://arxiv.org/html/2411.07501v4#bib.bib8)), etc., which have attractive cost versus quality trade-offs.

To achieve these favorable trade-offs, the above-mentioned models incorporate training techniques and architectural changes that can help improve quality while keeping training and inference costs low. Therefore, any proposed model efficiency technique including LAuReL should demonstrate not just improvement in quality, but also training and inference efficiency.

### 4.1 Efficiency Metrics

We now define some of the metrics on which we can measure model training and inference efficiency, before describing how LAuReL scores on them.

#### 4.1.1 Number of Parameters

This is the most common metric designed to capture the cost of training and serving the model. A larger number of parameters implies larger forward and backward pass costs.

#### 4.1.2 Latency (Training and Inference)

For training, a key metric to consider would be the number of steps taken per second. Similarly, another key metric would be the inference latency when deploying the model. For LLMs this can be broken down into the latency metrics in the warm-up stage (e.g., time-to-first-token, which can be further refined to prefill latency), and the generation stage (output tokens/second). For LLMs and other interactive models, both the time-to-first token and output tokens/second are important for a good user experience.

#### 4.1.3 Peak Memory

Peak memory used during training is another key metric that is tracked to ensure that accelerators can accommodate the model graph and have enough free memory available for forward and backward passes. Inefficiencies can be compensated by strategies like rematerialization (Kumar et al., [2019](https://arxiv.org/html/2411.07501v4#bib.bib23)), which can help reduce peak memory usage but need recomputing some activations. Similarly, inference-time peak memory usage is also a key metric to track.

### 4.2 Efficiency Analysis

LAuReL variants are designed with the above efficiency metrics in mind, and in this section we study the performance of the variants on these metrics.

Table 4: Analysis of extra parameters, memory, and latency incurred for each LAuReL variant, per instantiation, except the LAuReL-PA case where at any time no more than k 𝑘 k italic_k previous activations are in-memory, incurring a total Θ⁢(k⁢D)Θ 𝑘 𝐷\Theta(kD)roman_Θ ( italic_k italic_D ) extra memory cost. Also note that the latency cost of LAuReL-LR can be tighter than O⁢(r⁢D 2)𝑂 𝑟 superscript 𝐷 2 O(rD^{2})italic_O ( italic_r italic_D start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ), depending upon which matrix multiplication algorithm used. For simplicity, the latency bounds drop the batch and sequence dimensions.

LAuReL Params Memory Latency
variant
LAuReL-RW 1⁢or⁢ 2 1 or 2 1\ \textrm{or}\ 2 1 or 2 Θ⁢(1)Θ 1\Theta(1)roman_Θ ( 1 )O⁢(1)𝑂 1 O(1)italic_O ( 1 )
LAuReL-LR 2⁢r⁢D 2 𝑟 𝐷 2rD 2 italic_r italic_D Θ⁢(2⁢r⁢D)Θ 2 𝑟 𝐷\Theta(2rD)roman_Θ ( 2 italic_r italic_D )O⁢(r⁢D 2)𝑂 𝑟 superscript 𝐷 2 O(rD^{2})italic_O ( italic_r italic_D start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT )
LAuReL-PA k 𝑘 k italic_k Θ⁢(k⁢D)Θ 𝑘 𝐷\Theta(kD)roman_Θ ( italic_k italic_D )O⁢(k⁢D)𝑂 𝑘 𝐷 O(kD)italic_O ( italic_k italic_D )
LAuReL-LR+PA 2⁢r⁢k⁢D+k 2 𝑟 𝑘 𝐷 𝑘 2rkD+k 2 italic_r italic_k italic_D + italic_k Θ⁢(2⁢r⁢k⁢D+k)Θ 2 𝑟 𝑘 𝐷 𝑘\Theta(2rkD+k)roman_Θ ( 2 italic_r italic_k italic_D + italic_k )O⁢(k⁢r⁢D 2+k⁢D)𝑂 𝑘 𝑟 superscript 𝐷 2 𝑘 𝐷 O(krD^{2}+kD)italic_O ( italic_k italic_r italic_D start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT + italic_k italic_D )

Table[4](https://arxiv.org/html/2411.07501v4#S4.T4 "Table 4 ‣ 4.2 Efficiency Analysis ‣ 4 Efficiency ‣ LAuReL: Learned Augmented Residual Layer") lists the costs associated for each LAuReL layer. We assume that we are working with a vector of size D 𝐷 D italic_D (the last dimension of the input). Note that the listed costs are per-LAuReL instantiation. Hence, if there are many transformer layers and if a particular LAuReL variant is used in each layer, then the relevant costs accumulate across the layers. We now examine the costs of the variants.

LAuReL-RW is the cheapest, with one or two scalars as parameters, a constant memory cost (the constant depending on the data type: fp32, fp16 / bf16, unit8 / int8, etc.), and a constant latency cost since the scalar multiplication might be optimized by the compiler.

The LAuReL-LR variant has 2⁢r⁢D 2 𝑟 𝐷 2rD 2 italic_r italic_D parameters per instantiation (r⁢D 𝑟 𝐷 rD italic_r italic_D for each of the two rank-r 𝑟 r italic_r matrices, A 𝐴 A italic_A and B 𝐵 B italic_B), and uses Θ⁢(r⁢D)Θ 𝑟 𝐷\Theta(rD)roman_Θ ( italic_r italic_D ) memory. The latency is upper bounded by O⁢(r⁢D 2)𝑂 𝑟 superscript 𝐷 2 O(rD^{2})italic_O ( italic_r italic_D start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ), since the input is a vector of dimension D 𝐷 D italic_D (ignoring the batch and sequence dimensions), and there are two matrix multiplication steps.

Finally, the LAuReL-PA variant uses k 𝑘 k italic_k previous activations. Assuming h i subscript ℎ 𝑖 h_{i}italic_h start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT is the identity function, it uses k 𝑘 k italic_k extra parameters, and uses Θ⁢(k⁢D)Θ 𝑘 𝐷\Theta(kD)roman_Θ ( italic_k italic_D ) extra memory for the previous activations. Since it requires k 𝑘 k italic_k extra vector additions, this introduces an additional latency of O⁢(k⁢D)𝑂 𝑘 𝐷 O(kD)italic_O ( italic_k italic_D ).

### 4.3 Ablation Study of LAuReL Variants

To supplement the efficiency analysis in Section [4.2](https://arxiv.org/html/2411.07501v4#S4.SS2 "4.2 Efficiency Analysis ‣ 4 Efficiency ‣ LAuReL: Learned Augmented Residual Layer"), we provide an ablation study of LAuReL variants on a small LLM pre-training baseline. We used the C4 corpus (Raffel et al., [2020](https://arxiv.org/html/2411.07501v4#bib.bib28)) with ∼similar-to\sim∼ 10B tokens, and a 4×4 4 4 4\times 4 4 × 4 Google Cloud TPU v6e (Trillium) topology 3 3 3 We expect similar results with a comparable GPU setup. for compute.

In order to simplify the comparison across many ablations (and also to avoid the noise in downstream evaluations at the 10B tokens scale), we report model performance using the test loss, a reasonable proxy for downstream model quality. For each model we report the number of parameters, test loss, peak memory reported by profiling tools, and the average step time. The last two metrics are proxies for the theoretical memory and latency bounds respectively, as mentioned in Section [4.2](https://arxiv.org/html/2411.07501v4#S4.SS2 "4.2 Efficiency Analysis ‣ 4 Efficiency ‣ LAuReL: Learned Augmented Residual Layer"). Lower is better for all metrics.

We trained our main baseline with 24 layers (Baseline-24) and 157.2M parameters, along with a larger baseline with 28 layers (Baseline-28) and 179.2M parameters. We ran all the LAuReL variants and two combinations (RW+LR, RW+LR+PA) on top of Baseline-24. For the LAuReL-LR variants and its combinations, we picked r=32 𝑟 32 r=32 italic_r = 32. Similarly for LAuReL-PA variant and its combinations, we chose k=3 𝑘 3 k=3 italic_k = 3. Table [5](https://arxiv.org/html/2411.07501v4#S4.T5 "Table 5 ‣ 4.3 Ablation Study of LAuReL Variants ‣ 4 Efficiency ‣ LAuReL: Learned Augmented Residual Layer") shows the results.

Table 5: Performance of LAuReL variants on a small LLM pretraining task. LAuReL experiments are based on top of Baseline-24. LAuReL variants achieve a better test loss than Baseline-28, while having fewer parameters, requiring lesser memory, and being faster in average step time.

Variant Params Test Peak Avg. Step
(M)Loss Memory Time
(GB)(sec)
Baseline-24 157.20 3.0159 11.65 0.095
Baseline-28 179.23 2.9963 13.23 0.105
LAuReL variants (24 layers)
LAuReL-RW 157.20 2.9557 11.93 0.095
LAuReL-LR 158.40 2.9624 12.29 0.098
LAuReL-PA 157.22 2.9512 12.55 0.100
LAuReL-RW+LR 158.40 2.9531 12.57 0.099
LAuReL-RW+LR+PA 160.83 2.9499 12.90 0.104

As seen in the results, all LAuReL variants have a lower test loss than the Baseline-24, while having a negligible impact on additional parameters, peak memory, or average step time. In fact, LAuReL variants perform better than even Baseline-28 in terms of the test loss while using much fewer parameters, lower peak memory, and lower average step time.

### 4.4 Practical Recommendations

Given the experiments and the analysis of individual variants and their combinations, LAuReL-RW is clearly the first candidate to try. LAuReL-RW+LR offers further improvements to model quality, and seems to provide the best trade-off in terms of quality improvements and the additional overhead. We validated this in the ResNet (Section [3.1](https://arxiv.org/html/2411.07501v4#S3.SS1 "3.1 ResNet-50 on ImageNet-1K ‣ 3 Experiments ‣ LAuReL: Learned Augmented Residual Layer")) and LLM experiments (Sections [3.2.1](https://arxiv.org/html/2411.07501v4#S3.SS2.SSS1 "3.2.1 LLM-1B: Very low additional footprint ‣ 3.2 Large-Scale LLM Pre-training ‣ 3 Experiments ‣ LAuReL: Learned Augmented Residual Layer"), [3.2.2](https://arxiv.org/html/2411.07501v4#S3.SS2.SSS2 "3.2.2 LLM-4B: Low additional footprint ‣ 3.2 Large-Scale LLM Pre-training ‣ 3 Experiments ‣ LAuReL: Learned Augmented Residual Layer"), [4.3](https://arxiv.org/html/2411.07501v4#S4.SS3 "4.3 Ablation Study of LAuReL Variants ‣ 4 Efficiency ‣ LAuReL: Learned Augmented Residual Layer")).

The LAuReL-RW+LR+PA variant also leads to improvements over LAuReL-RW+LR. The main caveat is the additional Θ⁢(k⁢D)Θ 𝑘 𝐷\Theta(kD)roman_Θ ( italic_k italic_D ) memory, which might need to be monitored. If it is a concern, it can be mitigated to some extent by choosing a small value of k 𝑘 k italic_k or by changing the model sharding/re-materialization schemes.

Given the above tradeoffs in loss, memory, step time, etc. we recommend trying the LAuReL variants in the following order: RW →→\rightarrow→ LR →→\rightarrow→ RW+LR / PA →→\rightarrow→ RW+LR+PA.

### 4.5 Comparison with Naive Model Scaling

One of the advantages of LAuReL we would like to highlight is that it is competitive against naive scaling. Concretely, given a modest additional budget for parameters, latency, and memory, allocating that budget to LAuReL variants is likely to produce larger gains than using that budget in naive scaling methods such as additional layers. We first observed this in the ResNet experiments (Section 3.1), where LAuReL variants match the performance of naive scaling while adding 2.6×\times× fewer parameters, thus showing the Pareto-efficiency of LAuReL (versus naive scaling).

In the LLM experiments in Section 3.2, the gains achieved using LAuReL variants are at the cost of ∼similar-to\sim∼ 0.01–0.1% more parameters. A naive way to use this ‘additional’ budget would be to increase the model dimension (D 𝐷 D italic_D) such that we match the extra parameter budget. However, this is may not be feasible due to hardware limitations and memory alignment issues. Another way to use this ‘additional’ budget would be to increase the vocabulary size. However, the gains would be limited since this will include more tokens only from the tail of the distribution and may not contribute much to the model quality. Interestingly, Karpathy ([2023](https://arxiv.org/html/2411.07501v4#bib.bib21)) reported that for the NanoGPT model (Karpathy, [2022](https://arxiv.org/html/2411.07501v4#bib.bib20)) pre-training was sped up by ∼similar-to\sim∼ 25% when the vocabulary size was made divisible by 64. Therefore naive scaling might also hurt metrics like latency and memory by making the model setup suboptimal on the hardware.

Additionally, we also conducted a detailed study comparing LAuReL with naive scaling on the same LLM pre-training task as LLM-4B as mentioned in Section [3.2.2](https://arxiv.org/html/2411.07501v4#S3.SS2.SSS2 "3.2.2 LLM-4B: Low additional footprint ‣ 3.2 Large-Scale LLM Pre-training ‣ 3 Experiments ‣ LAuReL: Learned Augmented Residual Layer"). The baseline model and the the LAuReL experiments are similar to their counterparts in Section [3.2.2](https://arxiv.org/html/2411.07501v4#S3.SS2.SSS2 "3.2.2 LLM-4B: Low additional footprint ‣ 3.2 Large-Scale LLM Pre-training ‣ 3 Experiments ‣ LAuReL: Learned Augmented Residual Layer"), except a configuration change, which allows adding a layer. Both have 40 layers each, and are referred to as Baseline-40 and LAuReL-40 respectively.

The naive scaling baseline had 41 layers, and is referred to as Baseline-41 henceforth. In Table [6](https://arxiv.org/html/2411.07501v4#S4.T6 "Table 6 ‣ 4.5 Comparison with Naive Model Scaling ‣ 4 Efficiency ‣ LAuReL: Learned Augmented Residual Layer"), we present the respective number of parameters and average step times of the three models when training (forward + backward pass). A lower average step time is better. For Baseline-41 and LAuReL, we report the delta in number of parameters and average step time when compared to the original baseline.

Table 6: Comparison between Baseline-40 (40 layers), Baseline-41 (41 layers), and LAuReL in terms of parameters and average step time (forward and backward pass).

Model Params Avg. Step Time
(B)(sec)
Baseline-40 4.400 1.65
Baseline-41 4.560 (+3.63%)1.68 (+1.81%)
LAuReL-40 4.404 (+0.1%)1.69 (+2.42%)

Note LAuReL-40 adds only +0.1% parameters and incurs a step time penalty of 2.42%. In terms of extra parameters added, LAuReL-40 adds ∼36×\sim 36\times∼ 36 × fewer parameters than the extra parameters added by Baseline-41. The latency of LAuReL-40 is slightly higher than Baseline-41 since each layer incurs minor additional overhead.

Table [7](https://arxiv.org/html/2411.07501v4#S4.T7 "Table 7 ‣ 4.5 Comparison with Naive Model Scaling ‣ 4 Efficiency ‣ LAuReL: Learned Augmented Residual Layer") shows the downstream quality of the baselines and LAuReL-40 on 10 tasks across Math, General Reasoning, Reading Comprehension, Translation, and Mutimodal domains. LAuReL-40 wins on all tasks, except WMT23 and DocVQA where it matches the baselines. It outperforms not just Baseline-40, but also Baseline-41, achieving between 2%–21% improvement over Baseline-40’s performance.

Table 7: LAuReL-40’s comparison with naive scaling. Results that provide statistically significant improvement over the Baseline-40 are highlighted in green and bold. The percentages in the last row of the LAuReL-40 section indicate the relative improvement over the baseline.

Math General Reasoning Reading Comprehension Translation Multimodal
Method Math MGSM MMLU Belebele BookQA WMT23 MMMU Coco-Cap DocVQA TextVQA
Baseline-40 14.20 20.29 48.83 57.92 47.11 67.72 33.77 97.29 66.87 60.86
(4.40B Params)±plus-or-minus\pm± 0.88±plus-or-minus\pm± 3.16±plus-or-minus\pm± 0.81±plus-or-minus\pm± 3.42±plus-or-minus\pm± 4.06±plus-or-minus\pm± 0.20±plus-or-minus\pm± 3.11±plus-or-minus\pm± 4.41±plus-or-minus\pm± 2.67±plus-or-minus\pm± 2.86
Baseline-41 14.50 20.29 49.10 59.30 42.77 67.74 35.33 98.50 66.18 60.23
(4.56B Params)±plus-or-minus\pm± 0.9±plus-or-minus\pm± 3.15±plus-or-minus\pm± 0.82±plus-or-minus\pm± 3.34±plus-or-minus\pm± 4.15±plus-or-minus\pm± 0.21±plus-or-minus\pm± 3.12±plus-or-minus\pm± 3.53±plus-or-minus\pm± 2.76±plus-or-minus\pm± 2.87
LAuReL-40 15.11 23.12 50.32 62.65 57.22 67.71 37.57 99.27 66.92 63.15
(4.404B Params)±plus-or-minus\pm± 1.01±plus-or-minus\pm± 3.51±plus-or-minus\pm± 0.82±plus-or-minus\pm± 3.15±plus-or-minus\pm± 3.81±plus-or-minus\pm± 0.19±plus-or-minus\pm± 3.10±plus-or-minus\pm± 5.03±plus-or-minus\pm± 2.65±plus-or-minus\pm± 2.82
(+6.48%)(+13.94%)(+3.05%)(+8.16%)(+21.46%)(-0.02%)(+11.25%)(+2.03%)(+0.07%)(+3.76%)

We posit that naive scaling by adding an extra layer or two in a very deep network does not always lead to a corresponding improvement in performance out of the box. For such networks, we might require hyperparameter tuning (learning rate, weight decay) to counter overfitting, or training on more tokens to realize the expected theoretical model performance predicted by the scaling laws.

It is also well known that with deeper networks, the residual stream starts to play a crucial role in model convergence and quality. We suggest that this is the primary reason why LAuReL performs well. It augments the residual stream with learned components, allocating capacity for the network to learn the linear components of the input better.

5 Related Work
--------------

Residual Stream. DenseNet (Huang et al., [2017](https://arxiv.org/html/2411.07501v4#bib.bib19)) connects every pair of layers in the network and hence in the basic variant of DenseNet, all the activations need to be in memory. This is prohibitively expensive for deep LLMs and other modern transformers. When introducing dense-blocks, all previous activations within the block need to be visible to any given layer within the block; this requires refactoring the model architecture into dense blocks.

On the other hand, LAuReL requires minimal changes. In fact, in LAuReL-PA, which is the most similar to DenseNet, we make three design choices to achieve memory efficiency and performance. First, each layer only looks at the k 𝑘 k italic_k past activations (k=3 𝑘 3 k=3 italic_k = 3 seems sufficient in our experiments). Second, we use low-rank linear functions to further reduce memory usage due to activations. Third, we use learned scalars (γ i,γ i−1,…subscript 𝛾 𝑖 subscript 𝛾 𝑖 1…\gamma_{i},\gamma_{i-1},\dots italic_γ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT , italic_γ start_POSTSUBSCRIPT italic_i - 1 end_POSTSUBSCRIPT , …) to weigh the previous activations (which we found to be crucial in practice), whereas DenseNet assumes a simple sum of the previous activations.

He et al. ([2016](https://arxiv.org/html/2411.07501v4#bib.bib14)) introduce variants of residual connections with different types of ‘gating’, which look similar to the LAuReL-RW variant, except that they use a much larger number of parameters (O⁢(D 2)𝑂 superscript 𝐷 2 O(D^{2})italic_O ( italic_D start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT ) per layer), where LAuReL-RW uses one or two extra parameters per layer. Highway Nets (Srivastava et al., [2015](https://arxiv.org/html/2411.07501v4#bib.bib32)) is similar to LAuReL-RW but they also use D 2+D superscript 𝐷 2 𝐷 D^{2}+D italic_D start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT + italic_D parameters; furthermore, they incur additional latency due a full-rank matrix multiplication. Residual Gates (Savarese, [2016](https://arxiv.org/html/2411.07501v4#bib.bib29)) also is similar to LAuReL-RW, except they use ReLU as the gating function. However, LAuReL is a more general formulation.

Architectural Changes. Our work is inspired by recent model architecture improvements such as LoRA (Hu et al., [2022](https://arxiv.org/html/2411.07501v4#bib.bib18)) and AltUp (Baykal et al., [2023](https://arxiv.org/html/2411.07501v4#bib.bib3)) amongst others. LoRA is designed to efficiently fine-tune large pre-trained models and it works directly on the model weight matrices level by introducing low-rank ‘adapter’ weights that are learned during the fine-tuning stage, while other model weights are held constant. In contrast, LAuReL works at the residual connection level, which likely spans multiple weight matrices involved in the function f 𝑓 f italic_f; furthermore, it is applied during the pre-training stage.

AltUp (Baykal et al., [2023](https://arxiv.org/html/2411.07501v4#bib.bib3)) is designed to replicate the quality improvements of a model with a large model dimension, without having to pay the additional cost. It operates at the transformer-block level, constructing parallel ‘lightweight’ transformer blocks to approximate the model dimension scaling effect. In contrast, LAuReL does not aim to replicate the dimension scaling effect.

Interestingly, LAuReL can be applied in conjunction with both LoRA (during fine-tuning) and AltUp (during pre-training and fine-tuning). LAuReL can also be enabled at the same time as parameter-sharing techniques.

6 Conclusion
------------

In this paper we introduce the LAuReL framework, which is a novel architectural change and a generalization of the residual/skip connection aimed at improving the model quality without significantly increasing the model size or latency. We study three versions (LAuReL-RW, LAuReL-LR, LAuReL-PA) that can be mixed-and-matched together.

Through experiments, we demonstrate the efficacy of replacing the conventional residual connection with LAuReL on both vision and language tasks, while also providing evidence for its advantages over naive model scaling methods. For future work, we aim to further improve LAuReL by developing new variants with better trade-offs between quality and model footprint.

Acknowledgments
---------------

We thank Cenk Baykal, Erik Vee, Fotis Iliopoulos, Khoa Trinh, and Sushant Sachdeva (in alphabetical order) for many helpful discussions. We would also like to thank Andrew Tomkins for suggesting the name for the work.

Impact Statement
----------------

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

References
----------

*   Austin et al. (2021) Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., and Sutton, C. Program Synthesis with Large Language Models. _arXiv_, 2108.07732, 2021. 
*   Bandarkar et al. (2024) Bandarkar, L., Liang, D., Muller, B., Artetxe, M., Shukla, S.N., Husa, D., Goyal, N., Krishnan, A., Zettlemoyer, L., and Khabsa, M. The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants. _ACL Anthology_, pp. 749–775, 2024. 
*   Baykal et al. (2023) Baykal, C., Cutler, D., Dikkala, N., Ghosh, N., Panigrahy, R., and Wang, X. Alternating updates for efficient transformers. In _NeurIPS_, pp. 76718–76736, 2023. 
*   Chen et al. (2021) Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d.O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., et al. Evaluating Large Language Models Trained on Code. _arXiv_, 2107.03374, 2021. 
*   Clark et al. (2019) Clark, C., Lee, K., Chang, M.-W., Kwiatkowski, T., Collins, M., and Toutanova, K. BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions. In _NAACL-HLT_, pp. 2924–2936, 2019. 
*   Clark et al. (2020) Clark, J.H., Choi, E., Collins, M., Garrette, D., Kwiatkowski, T., Nikolaev, V., and Palomaki, J. TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages. _TACL_, 8:454–470, 2020. 
*   Cobbe et al. (2021) Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. Training Verifiers to Solve Math Word Problems. _arXiv_, 2110.14168, 2021. 
*   DeepSeek-A.I. et al. (2024) DeepSeek-A.I., Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Guo, D., Yang, D., Chen, D., Ji, D., Li, E., et al. DeepSeek-V3 Technical Report. _arXiv_, 2412.19437, 2024. 
*   Dehghani et al. (2022) Dehghani, M., Tay, Y., Arnab, A., Beyer, L., and Vaswani, A. The efficiency misnomer. In _ICLR_, 2022. 
*   Deng et al. (2009) Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In _CVPR_, pp. 248–255, 2009. 
*   Elhage et al. (2021) Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., DasSarma, N., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., and Olah, C. A mathematical framework for transformer circuits. _Transformer Circuits Thread_, 2021. https://transformer-circuits.pub/2021/framework/index.html. 
*   Gemini-Team et al. (2024) Gemini-Team, G., Georgiev, P., Lei, V.I., Burnell, R., Bai, L., Gulati, A., Tanzer, G., Vincent, D., Pan, Z., Wang, S., Mariooryad, S., Ding, Y., Geng, X., Alcober, F., and et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. _arXiv_, 2403.05530, 2024. 
*   (13) He, K., Zhang, X., Ren, S., and Sun, J. Deep Residual Learning for Image Recognition. In _CVPR_, pp. 27–30. 
*   He et al. (2016) He, K., Zhang, X., Ren, S., and Sun, J. Identity mappings in deep residual networks. In _ECCV_, pp. 630–645, 2016. 
*   Hendrycks et al. (2021a) Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring Massive Multitask Language Understanding. In _ICLR_, 2021a. 
*   Hendrycks et al. (2021b) Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the MATH dataset. In _NeurIPS_, 2021b. 
*   Hoffmann et al. (2022) Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L.A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Vinyals, O., Rae, J.W., and Sifre, L. Training compute-optimal large language models. In _NeurIPS_, 2022. 
*   Hu et al. (2022) Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. LoRA: Low-rank adaptation of large language models. In _ICLR_, 2022. 
*   Huang et al. (2017) Huang, G., Liu, Z., van der Maaten, L., and Weinberger, K.Q. Densely connected convolutional networks. In _CVPR_, 2017. 
*   Karpathy (2022) Karpathy, A. nanoGPT, January 2022. URL [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT). [Online; accessed 27. Jan. 2025]. 
*   Karpathy (2023) Karpathy, A. Andrej Karpathy on X: ”The most dramatic optimization to nanoGPT so far (~25% speedup) is to simply increase vocab size from 50257 to 50304 (nearest multiple of 64). This calculates added useless dimensions but goes down a different kernel path with much higher occupancy. Careful with your Powers of 2.” / X, January 2023. URL [https://x.com/karpathy/status/1621578354024677377](https://x.com/karpathy/status/1621578354024677377). [Online; accessed 27. Jan. 2025]. 
*   Kocmi et al. (2023) Kocmi, T., Avramidis, E., Bawden, R., Bojar, O., Dvorkovich, A., Federmann, C., Fishel, M., Freitag, M., Gowda, T., Grundkiewicz, R., Haddow, B., Koehn, P., Marie, B., Monz, C., Morishita, M., Murray, K., Nagata, M., Nakazawa, T., Popel, M., Popović, M., and Shmatova, M. LLMs are here but not quite there yet. In _WMT, Findings_, pp. 1–42, 2023. 
*   Kumar et al. (2019) Kumar, R., Purohit, M., Svitkina, Z., Vee, E., and Wang, J. Efficient rematerialization for deep networks. In _NIPS_, 2019. 
*   Lin et al. (2014) Lin, T.-Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J., Perona, P., Ramanan, D., Zitnick, C.L., and Dollár, P. Microsoft COCO: Common Objects in Context. In _ECCV_, pp. 740–755, 2014. 
*   Mathew et al. (2021) Mathew, M., Karatzas, D., and Jawahar, C.V. DocVQA: A dataset for VQA on document images. In _WACV_, pp. 2199–2208, 2021. 
*   Menghani (2023) Menghani, G. Efficient deep learning: A survey on making deep learning models smaller, faster, and better. _ACM Computing Surveys_, 2023. 
*   Mihaylov et al. (2018) Mihaylov, T., Clark, P., Khot, T., and Sabharwal, A. Can a suit of armor conduct electricity? a new dataset for open book question answering. In _EMNLP_, pp. 2381–2391, 2018. 
*   Raffel et al. (2020) Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P.J. Exploring the limits of transfer learning with a unified text-to-text transformer. _J. Mach. Learn. Res._, 21(1), January 2020. ISSN 1532-4435. 
*   Savarese (2016) Savarese, P. H.P. Learning identity mappings with residual gates. _arXiv_, 1611.01260, 2016. 
*   Shi et al. (2023) Shi, F., Suzgun, M., Freitag, M., Wang, X., Srivats, S., Vosoughi, S., Chung, H.W., Tay, Y., Ruder, S., Zhou, D., Das, D., and Wei, J. Language Models are Multilingual Chain-of-Thought Reasoners. In _ICLR_, 2023. 
*   Singh et al. (2019) Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., and Rohrbach, M. Towards VQA models that can read. In _CVPR_, pp. 8309–8318, 2019. 
*   Srivastava et al. (2015) Srivastava, R.K., Greff, K., and Schmidhuber, J. Training very deep networks. In _NIPS_, 2015. 
*   Sze et al. (2017) Sze, V., Chen, Y., Yang, T., and Emer, J.S. Efficient processing of deep neural networks: A tutorial and survey. _Proc. IEEE_, 105(12):2295–2329, 2017. 
*   Vaswani et al. (2017) Vaswani, A., Shazeer, N.M., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., and Polosukhin, I. Attention is all you need. _NIPS_, 2017. 
*   Yue et al. (2024) Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., Wei, C., Yu, B., Yuan, R., Sun, R., Yin, M., Zheng, B., Yang, Z., Liu, Y., Huang, W., Sun, H., Su, Y., and Chen, W. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In _CVPR_, 2024. 
*   Zellers et al. (2019) Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. HellaSwag: Can a machine really finish your sentence? In _ACL_, pp. 4791–4800, 2019.

