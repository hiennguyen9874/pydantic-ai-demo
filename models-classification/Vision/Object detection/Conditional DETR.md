Title: Conditional DETR for Fast Training Convergence

URL Source: https://arxiv.org/html/2108.06152

Markdown Content:
Depu Meng 1 1{}^{1}start_FLOATSUPERSCRIPT 1 end_FLOATSUPERSCRIPT Xiaokang Chen 2 2{}^{2}start_FLOATSUPERSCRIPT 2 end_FLOATSUPERSCRIPT††footnotemark:  Zejia Fan 2 2{}^{2}start_FLOATSUPERSCRIPT 2 end_FLOATSUPERSCRIPT Gang Zeng 2 2{}^{2}start_FLOATSUPERSCRIPT 2 end_FLOATSUPERSCRIPT

Houqiang Li 1 1{}^{1}start_FLOATSUPERSCRIPT 1 end_FLOATSUPERSCRIPT Yuhui Yuan 3 3{}^{3}start_FLOATSUPERSCRIPT 3 end_FLOATSUPERSCRIPT Lei Sun 3 3{}^{3}start_FLOATSUPERSCRIPT 3 end_FLOATSUPERSCRIPT Jingdong Wang 3 3{}^{3}start_FLOATSUPERSCRIPT 3 end_FLOATSUPERSCRIPT

1 1{}^{1}start_FLOATSUPERSCRIPT 1 end_FLOATSUPERSCRIPT University of Science and Technology of China 2 2{}^{2}start_FLOATSUPERSCRIPT 2 end_FLOATSUPERSCRIPT Peking University 3 3{}^{3}start_FLOATSUPERSCRIPT 3 end_FLOATSUPERSCRIPT Microsoft Research Asia The two authors share first authorship, and the order was determined by rolling dice. This work was done when D. Meng, X. Chen, and Z. Fan were interns at Microsoft Research, Beijing, P.R. ChinaCorresponding author.

###### Abstract

The recently-developed DETR approach applies the transformer encoder and decoder architecture to object detection and achieves promising performance. In this paper, we handle the critical issue, slow training convergence, and present a conditional cross-attention mechanism for fast DETR training. Our approach is motivated by that the cross-attention in DETR relies highly on the content embeddings for localizing the four extremities and predicting the box, which increases the need for high-quality content embeddings and thus the training difficulty.

Our approach, named conditional DETR, learns a conditional spatial query from the decoder embedding for decoder multi-head cross-attention. The benefit is that through the conditional spatial query, each cross-attention head is able to attend to a band containing a distinct region, e.g., one object extremity or a region inside the object box. This narrows down the spatial range for localizing the distinct regions for object classification and box regression, thus relaxing the dependence on the content embeddings and easing the training. Empirical results show that conditional DETR converges 6.7×6.7\times 6.7 × faster for the backbones R 50 50 50 50 and R 101 101 101 101 and 10×10\times 10 × faster for stronger backbones DC 5 5 5 5-R 50 50 50 50 and DC 5 5 5 5-R 101 101 101 101. Code is available at[https://github.com/Atten4Vis/ConditionalDETR](https://github.com/Atten4Vis/ConditionalDETR).

1 Introduction
--------------

The DEtection TRansformer (DETR) method[[3](https://arxiv.org/html/2108.06152#bib.bib3)] applies the transformer encoder and decoder architecture to object detection and achieves good performance. It effectively eliminates the need for many hand-crafted components, including non-maximum suppression and anchor generation.

![Image 1: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/teaser/dog/ours_0_image_000000071226object_3.png)

![Image 2: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/teaser/dog/ours_1_image_000000071226object_3.png)

![Image 3: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/teaser/dog/ours_2_image_000000071226object_3.png)

![Image 4: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/teaser/dog/ours_3_image_000000071226object_3.png)

![Image 5: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/teaser/dog/detr50_0_image_000000071226object_2.png)

![Image 6: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/teaser/dog/detr50_1_image_000000071226object_2.png)

![Image 7: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/teaser/dog/detr50_2_image_000000071226object_2.png)

![Image 8: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/teaser/dog/detr50_3_image_000000071226object_2.png)

![Image 9: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/teaser/dog/detr500_0_image_000000071226object_5.png)

![Image 10: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/teaser/dog/detr500_1_image_000000071226object_5.png)

![Image 11: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/teaser/dog/detr500_2_image_000000071226object_5.png)

![Image 12: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/teaser/dog/detr500_3_image_000000071226object_5.png)

Figure 1: Comparison of spatial attention weight maps for our conditional DETR-R 50 50 50 50 with 50 50 50 50 training epochs (the first row), the original DETR-R 50 50 50 50 with 50 50 50 50 training epochs (the second row), and the original DETR-R 50 50 50 50 with 500 500 500 500 training epochs (the third row). The maps for our conditional DETR and DETR trained with 500 500 500 500 epochs are able to highlight the four extremity regions satisfactorily. In contrast, the spatial attention weight maps responsible for the left and right edges (the third and fourth images in the second row) from DETR trained with 50 50 50 50 epochs cannot highlight the extremities satisfactorily. The green box is the ground-truth box. 

The DETR approach suffers from slow convergence on training, and needs 500 500 500 500 training epochs to get good performance. The very recent work, deformable DETR[[53](https://arxiv.org/html/2108.06152#bib.bib53)], handles this issue by replacing the global dense attention (self-attention and cross-attention) with deformable attention that attends to a small set of key sampling points and using the high-resolution and multi-scale encoder. Instead, we still use the global dense attention and propose an improved decoder cross-attention mechanism for accelerating the training process.

Our approach is motivated by high dependence on content embeddings and minor contributions made by the spatial embeddings in cross-attention. The empirical results in DETR[[3](https://arxiv.org/html/2108.06152#bib.bib3)] show that if removing the positional embeddings in keys and the object queries from the second decoder layer and only using the content embeddings in keys and queries, the detection AP drops slightly 1 1 1 The minor AP drop 1.4 1.4 1.4 1.4 is reported on R 50 50 50 50 with 300 300 300 300 epochs in Table 3 from[[3](https://arxiv.org/html/2108.06152#bib.bib3)]. We empirically got the consistent observation: the AP drops to 34.0 34.0 34.0 34.0 from 34.9 34.9 34.9 34.9 for 50 50 50 50 training epochs. .

Figure[1](https://arxiv.org/html/2108.06152#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Conditional DETR for Fast Training Convergence") (the second row) shows that the spatial attention weight maps from the cross-attention in DETR trained with 50 50 50 50 epochs. One can see that two among the four maps do not correctly highlight the bands for the corresponding extremities, thus weak at shrinking the spatial range for the content queries to precisely localize the extremities. The reasons are that (i) the spatial queries, i.e., object queries, only give the general attention weight map without exploiting the specific image information; and that (ii) due to short training the content queries are not strong enough to match the spatial keys well as they are also used to match the content keys. This increases the dependence on high-quality content embeddings, thus increasing the training difficulty.

![Image 13: Refer to caption](https://arxiv.org/html/x1.png)

Figure 2: Convergence curves for conditional DETR-DC5-R50 and DETR-DC5-R50 on COCO 2017 val. The conditional DETR is trained for 50 50 50 50, 75 75 75 75, 108 108 108 108 epochs. Conditional DETR training is converged much faster than DETR.

We present a conditional DETR approach, which learns a conditional spatial embedding for each query from the corresponding previous decoder output embedding, to form a so-called conditional spatial query for decoder multi-head cross-attention. The conditional spatial query is predicted by mapping the information for regressing the object box to the embedding space, the same to the space that the 2 2 2 2 D coordinates of the keys are also mapped to.

We empirically observe that using the spatial queries and keys, each cross-attention head spatially attends to a band containing the object extremity or a region inside the object box (Figure[1](https://arxiv.org/html/2108.06152#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Conditional DETR for Fast Training Convergence"), the first row). This shrinks the spatial range for the content queries to localize the effective regions for class and box prediction. As a result, the dependence on the content embeddings is relaxed and the training is easier. The experiments show that conditional DETR converges 6.7×6.7\times 6.7 × faster for the backbones R 50 50 50 50 and R 101 101 101 101 and 10×10\times 10 × faster for stronger backbones DC 5 5 5 5-R 50 50 50 50 and DC 5 5 5 5-R 101 101 101 101. Figure[2](https://arxiv.org/html/2108.06152#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Conditional DETR for Fast Training Convergence") gives the convergence curves for conditional DETR and the original DETR[[3](https://arxiv.org/html/2108.06152#bib.bib3)].

2 Related Work
--------------

Anchor-based and anchor-free detection. Most existing object detection approaches make predictions from initial guesses that are carefully designed. There are two main initial guesses: anchor boxes or object centers. The anchor box-based methods inherit the ideas from the proposal-based method, Fast R-CNN. Example methods include Faster R-CNN[[9](https://arxiv.org/html/2108.06152#bib.bib9)], SSD[[26](https://arxiv.org/html/2108.06152#bib.bib26)], YOLOv2[[31](https://arxiv.org/html/2108.06152#bib.bib31)], YOLOv3[[32](https://arxiv.org/html/2108.06152#bib.bib32)], YOLOv4[[1](https://arxiv.org/html/2108.06152#bib.bib1)], RetinaNet[[24](https://arxiv.org/html/2108.06152#bib.bib24)], Cascade R-CNN[[2](https://arxiv.org/html/2108.06152#bib.bib2)], Libra R-CNN[[29](https://arxiv.org/html/2108.06152#bib.bib29)], TSD[[35](https://arxiv.org/html/2108.06152#bib.bib35)] and so on.

The anchor-free detectors predict the boxes at points near the object centers. Typical methods include YOLOv1[[30](https://arxiv.org/html/2108.06152#bib.bib30)], CornerNet[[21](https://arxiv.org/html/2108.06152#bib.bib21)], ExtremeNet[[50](https://arxiv.org/html/2108.06152#bib.bib50)], CenterNet[[49](https://arxiv.org/html/2108.06152#bib.bib49), [6](https://arxiv.org/html/2108.06152#bib.bib6)], FCOS[[39](https://arxiv.org/html/2108.06152#bib.bib39)] and others[[23](https://arxiv.org/html/2108.06152#bib.bib23), [28](https://arxiv.org/html/2108.06152#bib.bib28), [52](https://arxiv.org/html/2108.06152#bib.bib52), [19](https://arxiv.org/html/2108.06152#bib.bib19), [51](https://arxiv.org/html/2108.06152#bib.bib51), [22](https://arxiv.org/html/2108.06152#bib.bib22), [15](https://arxiv.org/html/2108.06152#bib.bib15), [46](https://arxiv.org/html/2108.06152#bib.bib46), [47](https://arxiv.org/html/2108.06152#bib.bib47)].

DETR and its variants. DETR successfully applies transformers to object detection, effectively removing the need for many hand-designed components like non-maximum suppression or initial guess generation. The high computation complexity issue, caused by the global encoder self-attention, is handled in adaptive clustering transformer[[48](https://arxiv.org/html/2108.06152#bib.bib48)] and by sparse attentions in deformable DETR[[53](https://arxiv.org/html/2108.06152#bib.bib53)].

The other critical issue, slow training convergence, has been attracting a lot of recent research attention. The TSP (transformer-based set prediction) approach[[37](https://arxiv.org/html/2108.06152#bib.bib37)] eliminates the cross-attention modules and combines the FCOS and R-CNN-like detection heads. Deformable DETR[[53](https://arxiv.org/html/2108.06152#bib.bib53)] adopts deformable attention, which attends to sparse positions learned from the content embedding, to replace decoder cross-attention.

The spatially modulated co-attention (SMCA) approach[[7](https://arxiv.org/html/2108.06152#bib.bib7)], which is concurrent to our approach, is very close to our approach. It modulates the DETR multi-head global cross-attentions with Gaussian maps around a few (shifted) centers that are learned from the decoder embeddings, to focus more on a few regions inside the estimated box. In contrast, the proposed conditional DETR approach learns the conditional spatial queries from the decoder content embeddings, and predicts the spatial attention weight maps without human-crafting the attention attenuation, which highlight four extremities for box regression, and distinct regions inside the object for classification.

Conditional and dynamic convolution. The proposed conditional spatial query scheme is related to conditional convolutional kernel generation. Dynamic filter network[[16](https://arxiv.org/html/2108.06152#bib.bib16)] learns the convolutional kernels from the input, which is applied to instance segmentation in CondInst[[38](https://arxiv.org/html/2108.06152#bib.bib38)] and SOLOv2[[42](https://arxiv.org/html/2108.06152#bib.bib42)] for learning instance-dependent convolutional kernels. CondConv[[44](https://arxiv.org/html/2108.06152#bib.bib44)] and dynamic convolution[[4](https://arxiv.org/html/2108.06152#bib.bib4)] mix convolutional kernels with the weights learned from the input. SENet[[14](https://arxiv.org/html/2108.06152#bib.bib14)], GENet[[13](https://arxiv.org/html/2108.06152#bib.bib13)] abd Lite-HRNet[[45](https://arxiv.org/html/2108.06152#bib.bib45)] learn from the input the channel-wise weights.

These methods learn from the input the convolutional kernel weights and then apply the convolutions to the input. In contrast, the linear projection in our approach is learned from the decoder embeddings for representing the displacement and scaling information.

Transformers. The transformer[[40](https://arxiv.org/html/2108.06152#bib.bib40)] relies on the attention mechanism, self-attention and cross-attention, to draw global dependencies between the input and the output. There are several works closely related to our approach. Gaussian transformer[[11](https://arxiv.org/html/2108.06152#bib.bib11)] and T-GSA (Transformer with Gaussian-weighted self-attention)[[18](https://arxiv.org/html/2108.06152#bib.bib18)], followed by SMCA[[7](https://arxiv.org/html/2108.06152#bib.bib7)], attenuate the attention weights according to the distance between target and context symbols with learned or human-crafted Gaussian variance. Similar to ours, TUPE[[17](https://arxiv.org/html/2108.06152#bib.bib17)] computes the attention weight also from the spatial attention weight and the content attention weight. Instead, our approach mainly focuses on the attention attenuation mechanism in a learnable form other than a Gaussian function, and potentially benefits speech enhancement[[18](https://arxiv.org/html/2108.06152#bib.bib18)] and natural language inference[[11](https://arxiv.org/html/2108.06152#bib.bib11)].

3 Conditional DETR
------------------

### 3.1 Overview

Pipeline. The proposed approach follows detection transformer (DETR), an end-to-end object detector, and predicts all the objects at once without the need for NMS or anchor generation. The architecture consists of a CNN backbone, a transformer encoder, a transformer decoder, and object class and box position predictors. The transformer encoder aims to improve the content embeddings output from the CNN backbone. It is a stack of multiple encoder layers, where each layer mainly consists of a self-attention layer and a feed-forward layer.

The transformer decoder is a stack of decoder layers. Each decoder layer, illustrated in Figure[3](https://arxiv.org/html/2108.06152#S3.F3 "Figure 3 ‣ 3.1 Overview ‣ 3 Conditional DETR ‣ Conditional DETR for Fast Training Convergence"), is composed of three main layers: (1) a self-attention layer for removing duplication prediction, which performs interactions between the embeddings, outputted from the previous decoder layer and used for class and box prediction, (2) a cross-attention layer, which aggregates the embeddings output from the encoder to refine the decoder embeddings for improving class and box prediction, and (3) a feed-forward layer.

Box regression. A candidate box is predicted from each decoder embedding as follows,

𝐛=sigmoid⁡(FFN⁡(𝐟)+[𝐬⊤⁢0⁢0]⊤).𝐛 sigmoid FFN 𝐟 superscript delimited-[]superscript 𝐬 top 0 0 top\displaystyle\mathbf{b}=\operatorname{sigmoid}(\operatorname{FFN}(\mathbf{f})+% [{\mathbf{s}}^{\top}~{}0~{}0]^{\top}).bold_b = roman_sigmoid ( roman_FFN ( bold_f ) + [ bold_s start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT 0 0 ] start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT ) .(1)

Here, 𝐟 𝐟\mathbf{f}bold_f is the decoder embedding. 𝐛 𝐛\mathbf{b}bold_b is a four-dimensional vector [b c⁢x⁢b c⁢y⁢b w⁢b h]⊤superscript delimited-[]subscript 𝑏 𝑐 𝑥 subscript 𝑏 𝑐 𝑦 subscript 𝑏 𝑤 subscript 𝑏 ℎ top[b_{cx}~{}b_{cy}~{}b_{w}~{}b_{h}]^{\top}[ italic_b start_POSTSUBSCRIPT italic_c italic_x end_POSTSUBSCRIPT italic_b start_POSTSUBSCRIPT italic_c italic_y end_POSTSUBSCRIPT italic_b start_POSTSUBSCRIPT italic_w end_POSTSUBSCRIPT italic_b start_POSTSUBSCRIPT italic_h end_POSTSUBSCRIPT ] start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT, consisting of the box center, the box width and the box height. sigmoid⁢()sigmoid\operatorname{sigmoid}()roman_sigmoid ( ) is used to normalize the prediction 𝐛 𝐛\mathbf{b}bold_b to the range [0,1]0 1[0,1][ 0 , 1 ]. FFN⁢()FFN\operatorname{FFN}()roman_FFN ( ) aims to predictthe unnormalized box. 𝐬 𝐬{\mathbf{s}}bold_s is the unnormalized 2 2 2 2 D coordinate of the reference point, and is (0,0)0 0(0,0)( 0 , 0 ) in the original DETR. In our approach, we consider two choices: learn the reference point 𝐬 𝐬\mathbf{s}bold_s as a parameter for each candidate box prediction, or generate it from the corresponding object query.

Category prediction. The classification score for each candidate box is also predicted from the decoder embedding through an FNN, 𝐞=FFN⁡(𝐟)𝐞 FFN 𝐟\mathbf{e}=\operatorname{FFN}(\mathbf{f})bold_e = roman_FFN ( bold_f ).

![Image 14: Refer to caption](https://arxiv.org/html/x2.png)

Figure 3: Illustrating one decoder layer in conditional DETR. The main difference from the original DETR[[3](https://arxiv.org/html/2108.06152#bib.bib3)] lies in the input queries and the input keys for cross-attention. The conditional spatial query is predicted from learnable 2 2 2 2 D coordinates 𝐬 𝐬\mathbf{s}bold_s and the embeddings output from the previous decoder layer, through the operations depicted in the gray-shaded box. The 2 2 2 2 D coordinate 𝐬 𝐬\mathbf{s}bold_s can be predicted from the object query (the dashed box), or simply learned as model parameters The spatial query (key) and the content query (key) are concatenated as the query (key). The resulting cross-attention is called conditional cross-attention. Same as DETR[[3](https://arxiv.org/html/2108.06152#bib.bib3)], the decoder layer is repeated 6 6 6 6 times. 

Main work. The cross-attention mechanism aims to _localize the distinct regions, four extremities for box detection and regions inside the box for object classification, and aggregates the corresponding embeddings_. We propose a conditional cross-attention mechanism with introducing conditional spatial queries for improving the localization capability and accelerating the training process.

### 3.2 DETR Decoder Cross-Attention

The DETR decoder cross-attention mechanism takes three inputs: queries, keys and values. Each key is formed by adding a content key 𝐜 k subscript 𝐜 𝑘\mathbf{c}_{k}bold_c start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT (the content embedding output from the encoder) and a spatial key 𝐩 k subscript 𝐩 𝑘\mathbf{p}_{k}bold_p start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT (the positional embedding of the corresponding normalized 2 2 2 2 D coordinate). The value is formed from the content embedding, same with the content key, output from the encoder.

In the original DETR approach, each query is formed by adding a content query 𝐜 q subscript 𝐜 𝑞\mathbf{c}_{q}bold_c start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT (the embedding output from the decoder self-attention), and a spatial query 𝐩 q subscript 𝐩 𝑞\mathbf{p}_{q}bold_p start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT (i.e., the object query 𝐨 q subscript 𝐨 𝑞\mathbf{o}_{q}bold_o start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT). In our implementation, there are N=300 𝑁 300 N=300 italic_N = 300 object queries, and accordingly there are N 𝑁 N italic_N queries 2 2 2 For description simplicity and clearness, we drop the query, key, and value indices., each query outputting a candidate detect result in one decoder layer.

The attention weight is based on the dot-product between the query and the key, used for attention weight computation,

(𝐜 q+𝐩 q)⊤⁢(𝐜 k+𝐩 k)superscript subscript 𝐜 𝑞 subscript 𝐩 𝑞 top subscript 𝐜 𝑘 subscript 𝐩 𝑘\displaystyle(\mathbf{c}_{q}+\mathbf{p}_{q})^{\top}(\mathbf{c}_{k}+\mathbf{p}_% {k})( bold_c start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT + bold_p start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT ) start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT ( bold_c start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT + bold_p start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT )
=\displaystyle=~{}=𝐜 q⊤⁢𝐜 k+𝐜 q⊤⁢𝐩 k+𝐩 q⊤⁢𝐜 k+𝐩 q⊤⁢𝐩 k superscript subscript 𝐜 𝑞 top subscript 𝐜 𝑘 superscript subscript 𝐜 𝑞 top subscript 𝐩 𝑘 superscript subscript 𝐩 𝑞 top subscript 𝐜 𝑘 superscript subscript 𝐩 𝑞 top subscript 𝐩 𝑘\displaystyle\mathbf{c}_{q}^{\top}\mathbf{c}_{k}+\mathbf{c}_{q}^{\top}\mathbf{% p}_{k}+\mathbf{p}_{q}^{\top}\mathbf{c}_{k}+\mathbf{p}_{q}^{\top}\mathbf{p}_{k}bold_c start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT bold_c start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT + bold_c start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT bold_p start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT + bold_p start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT bold_c start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT + bold_p start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT bold_p start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT
=\displaystyle=~{}=𝐜 q⊤⁢𝐜 k+𝐜 q⊤⁢𝐩 k+𝐨 q⊤⁢𝐜 k+𝐨 q⊤⁢𝐩 k.superscript subscript 𝐜 𝑞 top subscript 𝐜 𝑘 superscript subscript 𝐜 𝑞 top subscript 𝐩 𝑘 superscript subscript 𝐨 𝑞 top subscript 𝐜 𝑘 superscript subscript 𝐨 𝑞 top subscript 𝐩 𝑘\displaystyle\mathbf{c}_{q}^{\top}\mathbf{c}_{k}+\mathbf{c}_{q}^{\top}\mathbf{% p}_{k}+\mathbf{o}_{q}^{\top}\mathbf{c}_{k}+\mathbf{o}_{q}^{\top}\mathbf{p}_{k}.bold_c start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT bold_c start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT + bold_c start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT bold_p start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT + bold_o start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT bold_c start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT + bold_o start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT bold_p start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT .(2)

### 3.3 Conditional Cross-Attention

The proposed conditional cross-attention mechanism forms the query by concatenating the content query 𝐜 q subscript 𝐜 𝑞\mathbf{c}_{q}bold_c start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT, outputting from decoder self-attention, and the spatial query 𝐩 q subscript 𝐩 𝑞\mathbf{p}_{q}bold_p start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT. Accordingly, the key is formed as the concatenation of the content key 𝐜 k subscript 𝐜 𝑘\mathbf{c}_{k}bold_c start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT and the spatial key 𝐩 k subscript 𝐩 𝑘\mathbf{p}_{k}bold_p start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT.

The cross-attention weights consist of two components, content attention weight and spatial attention weight. The two weights are from two dot-products, content and spatial dot-products,

𝐜 q⊤⁢𝐜 k+𝐩 q⊤⁢𝐩 k.superscript subscript 𝐜 𝑞 top subscript 𝐜 𝑘 superscript subscript 𝐩 𝑞 top subscript 𝐩 𝑘\displaystyle\mathbf{c}_{q}^{\top}\mathbf{c}_{k}+\mathbf{p}_{q}^{\top}\mathbf{% p}_{k}.bold_c start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT bold_c start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT + bold_p start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT bold_p start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT .(3)

Different from the original DETR cross-attention, our mechanism separates the roles of content and spatial queries so that spatial and content queries focus on the spatial and content attention weights, respectively.

An additional important task is to compute the spatial query 𝐩 q subscript 𝐩 𝑞\mathbf{p}_{q}bold_p start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT from the embedding 𝐟 𝐟\mathbf{f}bold_f of the previous decoder layer. We first identify that the spatial information of the distinct regions are determined by the two factors together, decoder embedding and reference point. We then show how to map them to the embedding space, forming the query 𝐩 q subscript 𝐩 𝑞\mathbf{p}_{q}bold_p start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT, so that the spatial query lies in the same space the 2 2 2 2 D coordinates of the keys are mapped to.

The decoder embedding contains the displacements of the distinct regions with respect to the reference point. The box prediction process in Equation[1](https://arxiv.org/html/2108.06152#S3.E1 "1 ‣ 3.1 Overview ‣ 3 Conditional DETR ‣ Conditional DETR for Fast Training Convergence") consists of two steps: (1) predicting the box with respect to the reference point in the unnormalized space, and (2) normalizing the predicted box to the range [0,1]0 1[0,1][ 0 , 1 ]3 3 3 The origin (0,0)0 0(0,0)( 0 , 0 ) in the unnormalized space for the original DETR method is mapped to (0.5,0.5)0.5 0.5(0.5,0.5)( 0.5 , 0.5 ) (the center in the image space) in the normalized space through the sigmoid sigmoid\operatorname{sigmoid}roman_sigmoid function..

Step (1) means that the decoder embedding 𝐟 𝐟\mathbf{f}bold_f contains the displacements of the four extremities (forming the box) with respect to the reference point 𝐬 𝐬\mathbf{s}bold_s in the unnormalized space. This implies that both the embedding 𝐟 𝐟\mathbf{f}bold_f and the reference point 𝐬 𝐬\mathbf{s}bold_s are necessary to determine the spatial information of the distinct regions, the four extremities as well as the region for predicting the classification score.

Conditional spatial query prediction. We predict the conditional spatial query from the embedding 𝐟 𝐟\mathbf{f}bold_f and the reference point 𝐬 𝐬\mathbf{s}bold_s,

(𝐬,𝐟)→𝐩 q,→𝐬 𝐟 subscript 𝐩 𝑞\displaystyle(\mathbf{s},\mathbf{f})\rightarrow\mathbf{p}_{q},( bold_s , bold_f ) → bold_p start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT ,(4)

so that it is aligned with the positional space which the normalized 2 2 2 2 D coordinates of the keys are mapped to. The process is illustrated in the gray-shaded box area of Figure[3](https://arxiv.org/html/2108.06152#S3.F3 "Figure 3 ‣ 3.1 Overview ‣ 3 Conditional DETR ‣ Conditional DETR for Fast Training Convergence").

We normalize the reference point 𝐬 𝐬\mathbf{s}bold_s and then map it to a 256 256 256 256-dimensional sinusoidal positional embedding in the same way as the positional embedding for keys:

𝐩 s=sinusoidal⁡(sigmoid⁡(𝐬)).subscript 𝐩 𝑠 sinusoidal sigmoid 𝐬\displaystyle\mathbf{p}_{s}=\operatorname{sinusoidal}(\operatorname{sigmoid}(% \mathbf{s})).bold_p start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT = roman_sinusoidal ( roman_sigmoid ( bold_s ) ) .(5)

We then map the displacement information contained in the decoder embedding 𝐟 𝐟\mathbf{f}bold_f to a linear projection in the same space through an FFN consisting of learnable linear projection + ReLU + learnable linear projection: 𝐓=FFN⁡(𝐟)𝐓 FFN 𝐟\mathbf{T}=\operatorname{FFN}(\mathbf{f})bold_T = roman_FFN ( bold_f ).

The conditional spatial query is computed by transforming the reference point in the embedding space: 𝐩 q=𝐓𝐩 s subscript 𝐩 𝑞 subscript 𝐓𝐩 𝑠\mathbf{p}_{q}=\mathbf{T}\mathbf{p}_{s}bold_p start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT = bold_Tp start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT. We choose the simple and computationally-efficient projection matrix, a diagonal matrix. The 256 256 256 256 diagonal elements are denoted as a vector 𝛌 q subscript 𝛌 𝑞\boldsymbol{\uplambda}_{q}bold_λ start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT. The conditional spatial query is computed by the element-wise multiplication:

𝐩 q=𝐓𝐩 s=𝛌 q⊙𝐩 s.subscript 𝐩 𝑞 subscript 𝐓𝐩 𝑠 direct-product subscript 𝛌 𝑞 subscript 𝐩 𝑠\displaystyle\mathbf{p}_{q}=\mathbf{T}{\mathbf{p}}_{s}=\boldsymbol{\uplambda}_% {q}\odot{\mathbf{p}}_{s}.bold_p start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT = bold_Tp start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT = bold_λ start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT ⊙ bold_p start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT .(6)

Multi-head cross-attention. Following DETR[[3](https://arxiv.org/html/2108.06152#bib.bib3)], we adopt the standard multi-head cross-attention mechanism. Object detection usually needs to implicitly or explicitly localize the four object extremities for accurate box regression and localize the object region for accurate object classification. The multi-head mechanism is beneficial to disentangle the localization tasks.

We perform multi-head parallel attentions by projecting the queries, the keys, and the values M=8 𝑀 8 M=8 italic_M = 8 times with learned linear projections to low dimensions. The spatial and content queries (keys) are separately projected to each head with different linear projections. The projections for values are the same as the original DETR and are only for the contents.

![Image 15: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/fig4/bicycle_blue/pos_left.png)

![Image 16: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/fig4/bicycle_blue/pos_top.png)

![Image 17: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/fig4/bicycle_blue/pos_right.png)

![Image 18: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/fig4/bicycle_blue/pos_bottom.png)

![Image 19: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/fig4/bicycle_blue/pos_object.png)

![Image 20: Refer to caption](https://arxiv.org/html/2108.06152)

![Image 21: Refer to caption](https://arxiv.org/html/2108.06152)

![Image 22: Refer to caption](https://arxiv.org/html/2108.06152)

![Image 23: Refer to caption](https://arxiv.org/html/2108.06152)

![Image 24: Refer to caption](https://arxiv.org/html/2108.06152)

![Image 25: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/fig4/bicycle_blue/content_left.png)

![Image 26: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/fig4/bicycle_blue/content_top.png)

![Image 27: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/fig4/bicycle_blue/content_right.png)

![Image 28: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/fig4/bicycle_blue/content_bottom.png)

![Image 29: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/fig4/bicycle_blue/content_object.png)

![Image 30: Refer to caption](https://arxiv.org/html/2108.06152)

![Image 31: Refer to caption](https://arxiv.org/html/2108.06152)

![Image 32: Refer to caption](https://arxiv.org/html/2108.06152)

![Image 33: Refer to caption](https://arxiv.org/html/2108.06152)

![Image 34: Refer to caption](https://arxiv.org/html/2108.06152)

![Image 35: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/fig4/bicycle_blue/overall_left.png)

![Image 36: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/fig4/bicycle_blue/overall_top.png)

![Image 37: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/fig4/bicycle_blue/overall_right.png)

![Image 38: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/fig4/bicycle_blue/overall_bottom.png)

![Image 39: Refer to caption](https://arxiv.org/html/extracted/5142480/figs/fig4/bicycle_blue/overall_object.png)

![Image 40: Refer to caption](https://arxiv.org/html/2108.06152)

![Image 41: Refer to caption](https://arxiv.org/html/2108.06152)

![Image 42: Refer to caption](https://arxiv.org/html/2108.06152)

![Image 43: Refer to caption](https://arxiv.org/html/2108.06152)

![Image 44: Refer to caption](https://arxiv.org/html/2108.06152)

Figure 4:  Illustrating the spatial attention weight maps (the first row), the content attention weight maps (the second row), and the combined attention weight maps (the third row) computed from our conditional DETR. The attention weight maps are from 5 5 5 5 heads out of the 8 8 8 8 heads and are responsible for the four extremities and a region inside the object box. The content attention weight maps for the four extremities highlight scattered regions inside the box (bicycle) or similar regions in two object instances (cow), and the corresponding combined attention weight maps highlight the extremity regions with the help of the spatial attention weight maps. The combined attention weight map for the region inside the object box mainly depends on the spatial attention weight map, which implies that the representation of a region inside the object might encode enough class information. The maps are from conditional DETR-R 50 50 50 50 trained with 50 50 50 50 epochs. 

### 3.4 Visualization and Analysis

Visualization. Figure[4](https://arxiv.org/html/2108.06152#S3.F4 "Figure 4 ‣ 3.3 Conditional Cross-Attention ‣ 3 Conditional DETR ‣ Conditional DETR for Fast Training Convergence") visualizes the attention weight maps for each head: the spatial attention weight maps, the content attention weight maps, and the combined attention weight maps. The maps are soft-max normalized over the spatial dot-products 𝐩 q⊤⁢𝐩 k superscript subscript 𝐩 𝑞 top subscript 𝐩 𝑘\mathbf{p}_{q}^{\top}\mathbf{p}_{k}bold_p start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT bold_p start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT, the content dot-products 𝐜 q⊤⁢𝐜 k superscript subscript 𝐜 𝑞 top subscript 𝐜 𝑘\mathbf{c}_{q}^{\top}\mathbf{c}_{k}bold_c start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT bold_c start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT, and the combined dot-products 𝐜 q⊤⁢𝐜 k+𝐩 q⊤⁢𝐩 k superscript subscript 𝐜 𝑞 top subscript 𝐜 𝑘 superscript subscript 𝐩 𝑞 top subscript 𝐩 𝑘\mathbf{c}_{q}^{\top}\mathbf{c}_{k}+\mathbf{p}_{q}^{\top}\mathbf{p}_{k}bold_c start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT bold_c start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT + bold_p start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT bold_p start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT. We show 5 5 5 5 out of the 8 8 8 8 maps, and other three are the duplicates, corresponding to bottom and top extremities, and a small region inside the object box 4 4 4 The duplicates might be different for models trained several times, but the detection performance is almost the same..

We can see that the spatial attention weight map at each head is able to localize a distinct region, a region containing one extremity or a region inside the object box. It is interesting that each spatial attention weight map corresponding to an extremity highlights a spatial band that overlaps with the corresponding edge of the object box. The other spatial attention map for the region inside the object box merely highlights a small region whose representations might already encode enough information for object classification.

The content attention weight maps of the four heads corresponding to the four extremities highlight scattered regions in addition to the extremities. The combination of the spatial and content maps filters out other highlights and keeps extremity highlights for accurate box regression.

Comparison to DETR. Figure[1](https://arxiv.org/html/2108.06152#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Conditional DETR for Fast Training Convergence") shows the spatial attention weight maps of our conditional DETR (the first row) and the original DETR trained with 50 50 50 50 epochs (the second row). The maps of our approach are computed by soft-max normalizing the dot-products between spatial keys and queries, 𝐩 q⊤⁢𝐩 𝐤 superscript subscript 𝐩 𝑞 top subscript 𝐩 𝐤\mathbf{p}_{q}^{\top}\mathbf{p_{k}}bold_p start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT bold_p start_POSTSUBSCRIPT bold_k end_POSTSUBSCRIPT. The maps for DETR are computed by soft-max normalizing the dot-products with the spatial keys, (𝐨 q+𝐜 q)⊤⁢𝐩 k superscript subscript 𝐨 𝑞 subscript 𝐜 𝑞 top subscript 𝐩 𝑘(\mathbf{o}_{q}+\mathbf{c}_{q})^{\top}\mathbf{p}_{k}( bold_o start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT + bold_c start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT ) start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT bold_p start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT.

It can be seen that our spatial attention weight maps accurately localize the distinct regions, four extremities. In contrast, the maps from the original DETR with 50 50 50 50 epochs can not accurately localize two extremities, and 500 500 500 500 training epochs (the third row) make the content queries stronger, leading to accurate localization. This implies that it is really hard to learn the content query 𝐜 q subscript 𝐜 𝑞\mathbf{c}_{q}bold_c start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT to serve as two roles 5 5 5 Strictly speaking, the embedding output from decoder self-attention for more training epochs contains both spatial and content information. For discussion convenience, we still call it content query.: match the content key and the spatial key simultaneously, and thus more training epochs are needed.

Analysis. The spatial attention weight maps shown in Figure[4](https://arxiv.org/html/2108.06152#S3.F4 "Figure 4 ‣ 3.3 Conditional Cross-Attention ‣ 3 Conditional DETR ‣ Conditional DETR for Fast Training Convergence") imply that the conditional spatial query, used to form the spatial query, have at least two effects. (i) Translate the highlight positions to the four extremities and the position inside the object box: interestingly the highlighted positions are spatially similarly distributed in the object box. (ii) Scale the spatial spread for the extremity highlights: large spread for large objects and small spread for small objects.

The two effects are realized in the spatial embedding space through applying the transformation 𝐓 𝐓\mathbf{T}bold_T over 𝐩 s subscript 𝐩 𝑠\mathbf{p}_{s}bold_p start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT (further disentangled through image-independent linear projections contained in cross-attention and distributed to each head). This indicates that the transformation 𝐓 𝐓\mathbf{T}bold_T not only contains the displacements as discussed before, but also the object scale.

Table 1: Comparison of conditional DETR with DETR on COCO 2017 val. Our conditional DETR approach for high-resolution backbones DC 5 5 5 5-R 50 50 50 50 and DC 5 5 5 5-R 101 101 101 101 is 10×10\times 10 × faster than the original DETR, and for low-resolution backbones R 50 50 50 50 and R 101 101 101 101 6.67×6.67\times 6.67 × faster. Conditional DETR is empirically superior to other two single-scale DETR variants. *{}^{*}start_FLOATSUPERSCRIPT * end_FLOATSUPERSCRIPT The results of deformable DETR are from the GitHub repository provided by the authors of deformable DETR[[53](https://arxiv.org/html/2108.06152#bib.bib53)]. 

### 3.5 Implementation Details

Architecture. Our architecture is almost the same with the DETR architecture[[3](https://arxiv.org/html/2108.06152#bib.bib3)] and contains the CNN backbone, transformer encoder, transformer decoder, prediction feed-forward networks (FFNs) following each decoder layer (the last decoder layer and the 5 5 5 5 internal decoder layers) with parameters shared among the 6 6 6 6 prediction FFNs. The hyper-parameters are the same as DETR.

The main architecture difference is that we introduce the conditional spatial embeddings as the spatial queries for conditional multi-head cross-attention and that the spatial query (key) and the content query (key) are combined through concatenation other than addition. In the first cross-attention layer there are no decoder content embeddings, we make simple changes based on the DETR implementation[[3](https://arxiv.org/html/2108.06152#bib.bib3)]: concatenate the positional embedding predicted from the object query (the positional embedding) into the original query (key).

Reference points. In the original DETR approach, 𝐬=[0⁢0]⊤𝐬 superscript delimited-[]0 0 top{\mathbf{s}}=[0~{}0]^{\top}bold_s = [ 0 0 ] start_POSTSUPERSCRIPT ⊤ end_POSTSUPERSCRIPT is the same for all the decoder embeddings. We study two ways forming the reference points: regard the unnormalized 2 2 2 2 D coordinates as learnable parameters, and the unnormalized 2 2 2 2 D coordinate predicted from the object query 𝐨 q subscript 𝐨 𝑞\mathbf{o}_{q}bold_o start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT. In the latter way that is similar to deformable DETR[[53](https://arxiv.org/html/2108.06152#bib.bib53)], the prediction unit is an FFN and consists of learnable linear projection + ReLU + learnable linear projection: 𝐬=FFN⁡(𝐨 q)𝐬 FFN subscript 𝐨 𝑞\mathbf{s}=\operatorname{FFN}(\mathbf{o}_{q})bold_s = roman_FFN ( bold_o start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT ). When used for forming the conditional spatial query, the 2 2 2 2 D coordinates are normalized by the sigmoid function.

Loss function. We follow DETR[[3](https://arxiv.org/html/2108.06152#bib.bib3)] to find an optimal bipartite matching[[20](https://arxiv.org/html/2108.06152#bib.bib20)] between the predicted and ground-truth objects using the Hungarian algorithm, and then form the loss function for computing and back-propagate the gradients. We use the same way with deformable DETR[[53](https://arxiv.org/html/2108.06152#bib.bib53)] to formulate the loss: the same matching cost function, the same loss function with 300 300 300 300 object queries, and the same trade-off parameters; The classification loss function is focal loss[[24](https://arxiv.org/html/2108.06152#bib.bib24)], and the box regression loss (including L1 and GIoU[[34](https://arxiv.org/html/2108.06152#bib.bib34)] loss) is the same as DETR[[3](https://arxiv.org/html/2108.06152#bib.bib3)].

4 Experiments
-------------

### 4.1 Setting

Dataset. We perform the experiments on the COCO 2017 2017 2017 2017[[25](https://arxiv.org/html/2108.06152#bib.bib25)] detection dataset. The dataset contains about 118 118 118 118 K training images and 5 5 5 5 K validation (val) images.

Table 2: Results for multi-scale and higher-resolution DETR variants. We do not expect that our approach performs on par as our approach (single-scale, 16×16\times 16 × resolution) does not use a strong multi-scale or 8×8\times 8 × resolution encoder. Surprisingly, the AP scores of our approach with DC 5 5 5 5-R 50 50 50 50 and DC 5 5 5 5-R 101 101 101 101 are close to the two multi-scale and higher-resolution DETR variants. 

Model#epochs GFLOPs#params (M)AP AP 50 50{}_{50}start_FLOATSUBSCRIPT 50 end_FLOATSUBSCRIPT AP 75 75{}_{75}start_FLOATSUBSCRIPT 75 end_FLOATSUBSCRIPT AP S 𝑆{}_{S}start_FLOATSUBSCRIPT italic_S end_FLOATSUBSCRIPT AP M 𝑀{}_{M}start_FLOATSUBSCRIPT italic_M end_FLOATSUBSCRIPT AP L 𝐿{}_{L}start_FLOATSUBSCRIPT italic_L end_FLOATSUBSCRIPT
Faster RCNN-FPN-R 50 50 50 50[[33](https://arxiv.org/html/2108.06152#bib.bib33)]36 36 36 36 180 180 180 180 42 42 42 42 40.2 40.2 40.2 40.2 61.0 61.0 61.0 61.0 43.8 43.8 43.8 43.8 24.2 24.2 24.2 24.2 43.5 43.5 43.5 43.5 52.0 52.0 52.0 52.0
Faster RCNN-FPN-R 50 50 50 50[[33](https://arxiv.org/html/2108.06152#bib.bib33)]108 108 108 108 180 180 180 180 42 42 42 42 42.0 42.0 42.0 42.0 62.1 62.1 62.1 62.1 45.5 45.5 45.5 45.5 26.6 26.6 26.6 26.6 45.5 45.5 45.5 45.5 53.4 53.4 53.4 53.4
Deformable DETR-R 50 50 50 50[[53](https://arxiv.org/html/2108.06152#bib.bib53)]50 50 50 50 173 173 173 173 40 40 40 40 43.8 43.8 43.8 43.8 62.6 62.6 62.6 62.6 47.7 47.7 47.7 47.7 26.4 26.4 26.4 26.4 47.1 47.1 47.1 47.1 58.0 58.0 58.0 58.0
TSP-FCOS-R 50 50 50 50[[37](https://arxiv.org/html/2108.06152#bib.bib37)]36 36 36 36 189 189 189 189−--43.1 43.1 43.1 43.1 62.3 62.3 62.3 62.3 47.0 47.0 47.0 47.0 26.6 26.6 26.6 26.6 46.8 46.8 46.8 46.8 55.9 55.9 55.9 55.9
TSP-RCNN-R 50 50 50 50[[37](https://arxiv.org/html/2108.06152#bib.bib37)]36 36 36 36 188 188 188 188−--43.8 43.8 43.8 43.8 63.3 63.3 63.3 63.3 48.3 48.3 48.3 48.3 28.6 28.6 28.6 28.6 46.9 46.9 46.9 46.9 55.7 55.7 55.7 55.7
TSP-RCNN-R 50 50 50 50[[37](https://arxiv.org/html/2108.06152#bib.bib37)]96 96 96 96 188 188 188 188−--45.0 45.0 45.0 45.0 64.5 64.5 64.5 64.5 49.6 49.6 49.6 49.6 29.7 29.7 29.7 29.7 47.7 47.7 47.7 47.7 58.0 58.0 58.0 58.0
Conditional DETR-DC5-R50 50 50 50 50 195 195 195 195 44 44 44 44 43.8 43.8 43.8 43.8 64.4 64.4 64.4 64.4 46.7 46.7 46.7 46.7 24.0 24.0 24.0 24.0 47.6 47.6 47.6 47.6 60.7 60.7 60.7 60.7
Conditional DETR-DC5-R50 108 108 108 108 195 195 195 195 44 44 44 44 45.1 45.1 45.1 45.1 65.4 65.4 65.4 65.4 48.5 48.5 48.5 48.5 25.3 25.3 25.3 25.3 49.0 49.0 49.0 49.0 62.2 62.2 62.2 62.2
Faster RCNN-FPN-R 101 101 101 101[[33](https://arxiv.org/html/2108.06152#bib.bib33)]36 36 36 36 246 246 246 246 60 60 60 60 42.0 42.0 42.0 42.0 62.5 62.5 62.5 62.5 45.9 45.9 45.9 45.9 25.2 25.2 25.2 25.2 45.6 45.6 45.6 45.6 54.6 54.6 54.6 54.6
Faster RCNN-FPN-R 101 101 101 101[[33](https://arxiv.org/html/2108.06152#bib.bib33)]108 108 108 108 246 246 246 246 60 60 60 60 44.0 44.0 44.0 44.0 63.9 63.9 63.9 63.9 47.8 47.8 47.8 47.8 27.2 27.2 27.2 27.2 48.1 48.1 48.1 48.1 56.0 56.0 56.0 56.0
TSP-FCOS-R 101 101 101 101[[37](https://arxiv.org/html/2108.06152#bib.bib37)]36 36 36 36 255 255 255 255−--44.4 44.4 44.4 44.4 63.8 63.8 63.8 63.8 48.2 48.2 48.2 48.2 27.7 27.7 27.7 27.7 48.6 48.6 48.6 48.6 57.3 57.3 57.3 57.3
TSP-RCNN-R 101 101 101 101[[37](https://arxiv.org/html/2108.06152#bib.bib37)]36 36 36 36 254 254 254 254−--44.8 44.8 44.8 44.8 63.8 63.8 63.8 63.8 49.2 49.2 49.2 49.2 29.0 29.0 29.0 29.0 47.9 47.9 47.9 47.9 57.1 57.1 57.1 57.1
TSP-RCNN-R 101 101 101 101[[37](https://arxiv.org/html/2108.06152#bib.bib37)]96 96 96 96 254 254 254 254−--46.5 46.5 46.5 46.5 66.0 66.0 66.0 66.0 51.2 51.2 51.2 51.2 29.9 29.9 29.9 29.9 49.7 49.7 49.7 49.7 59.2 59.2 59.2 59.2
Conditional DETR-DC5-R101 50 50 50 50 262 262 262 262 63 63 63 63 45.0 45.0 45.0 45.0 65.5 65.5 65.5 65.5 48.4 48.4 48.4 48.4 26.1 26.1 26.1 26.1 48.9 48.9 48.9 48.9 62.8 62.8 62.8 62.8
Conditional DETR-DC5-R101 108 108 108 108 262 262 262 262 63 63 63 63 45.9 45.9 45.9 45.9 66.8 66.8 66.8 66.8 49.5 49.5 49.5 49.5 27.2 27.2 27.2 27.2 50.3 50.3 50.3 50.3 63.3 63.3 63.3 63.3

Training. We follow the DETR training protocol[[3](https://arxiv.org/html/2108.06152#bib.bib3)]. The backbone is the ImageNet-pretrained model from TORCHVISION with batchnorm layers fixed, and the transformer parameters are initialized using the Xavier initialization scheme[[10](https://arxiv.org/html/2108.06152#bib.bib10)]. The weight decay is set to be 10−4 superscript 10 4 10^{-4}10 start_POSTSUPERSCRIPT - 4 end_POSTSUPERSCRIPT. The AdamW[[27](https://arxiv.org/html/2108.06152#bib.bib27)] optimizer is used. The learning rates for the backbone and the transformer are initially set to be 10−5 superscript 10 5 10^{-5}10 start_POSTSUPERSCRIPT - 5 end_POSTSUPERSCRIPT and 10−4 superscript 10 4 10^{-4}10 start_POSTSUPERSCRIPT - 4 end_POSTSUPERSCRIPT, respectively. The dropout rate in transformer is 0.1 0.1 0.1 0.1. The learning rate is dropped by a factor of 10 10 10 10 after 40 40 40 40 epochs for 50 50 50 50 training epochs, after 60 60 60 60 epochs for 75 75 75 75 training epochs, and after 80 80 80 80 epochs for 108 108 108 108 training epochs.

We use the augmentation scheme same as DETR[[3](https://arxiv.org/html/2108.06152#bib.bib3)]: resize the input image such that the short side is at least 480 480 480 480 and at most 800 800 800 800 pixels and the long size is at most 1333 1333 1333 1333 pixels; randomly crop the image such that a training image is cropped with probability 0.5 0.5 0.5 0.5 to a random rectangular patch.

Evaluation. We use the standard COCO evaluation. We report the average precision (AP), and the AP scores at 0.50 0.50 0.50 0.50, 0.75 0.75 0.75 0.75 and for the small, medium, and large objects.

### 4.2 Results

Comparison to DETR. We compare the proposed conditional DETR to the original DETR[[3](https://arxiv.org/html/2108.06152#bib.bib3)]. We follow[[3](https://arxiv.org/html/2108.06152#bib.bib3)] and report the results over four backbones: ResNet-50 50 50 50[[12](https://arxiv.org/html/2108.06152#bib.bib12)], ResNet-101 101 101 101, and their 16×16\times 16 ×-resolution extensions DC 5 5 5 5-ResNet-50 50 50 50 and DC 5 5 5 5-ResNet-101 101 101 101.

The corresponding DETR models are named as DETR-R 50 50 50 50, DETR-R 101 101 101 101, DETR-DC5-R 50 50 50 50, and DETR-DC5-R 101 101 101 101, respectively. Our models are named as conditional DETR-R 50 50 50 50, conditional DETR-R 101 101 101 101, conditional DETR-DC5-R 50 50 50 50, and conditional DETR-DC5-R 101 101 101 101, respectively.

Table[1](https://arxiv.org/html/2108.06152#S3.T1 "Table 1 ‣ 3.4 Visualization and Analysis ‣ 3 Conditional DETR ‣ Conditional DETR for Fast Training Convergence") presents the results from DETR and conditional DETR. DETR with 50 50 50 50 training epochs performs much worse than 500 500 500 500 training epochs. Conditional DETR with 50 50 50 50 training epochs for R 50 50 50 50 and R 101 101 101 101 as the backbones performs slightly worse than DETR with 500 500 500 500 training epochs. Conditional DETR with 50 50 50 50 training epochs for DC 5 5 5 5-R 50 50 50 50 and DC 5 5 5 5-R 101 101 101 101 performs similarly as DETR with 500 500 500 500 training epochs. Conditional DETR for the four backbones with 75/108 75 108 75/108 75 / 108 training epochs performs better than DETR with 500 500 500 500 training epochs. In summary, conditional DETR for high-resolution backbones DC 5 5 5 5-R 50 50 50 50 and DC 5 5 5 5-R 101 101 101 101 is 10×10\times 10 × faster than the original DETR, and for low-resolution backbones R 50 50 50 50 and R 101 101 101 101 6.67×6.67\times 6.67 × faster. In other words, conditional DETR performs better for stronger backbones with better performance.

In addition, we report the results of single-scale DETR extensions: deformable DETR-SS[[53](https://arxiv.org/html/2108.06152#bib.bib53)] and UP-DETR[[5](https://arxiv.org/html/2108.06152#bib.bib5)] in Table[1](https://arxiv.org/html/2108.06152#S3.T1 "Table 1 ‣ 3.4 Visualization and Analysis ‣ 3 Conditional DETR ‣ Conditional DETR for Fast Training Convergence"). Our results over R 50 50 50 50 and DC 5 5 5 5-R 50 50 50 50 are better than deformable DETR-SS: 40.9 40.9 40.9 40.9 vs. 39.4 39.4 39.4 39.4 and 43.8 43.8 43.8 43.8 vs. 41.5 41.5 41.5 41.5. The comparison might not be fully fair as for example parameter and computation complexities are different, but it implies that the conditional cross-attention mechanism is beneficial. Compared to UP-DETR-R 50 50 50 50, our results with fewer training epochs are obviously better.

Comparison to multi-scale and higher-resolution DETR variants. We focus on accelerating the DETR training, without addressing the issue of high computational complexity in the encoder. We do not expect that our approach achieves on par with DETR variants w/ multi-scale attention and 8×8\times 8 ×-resolution encoders, e.g., TSP-FCOS and TSP-RCNN[[37](https://arxiv.org/html/2108.06152#bib.bib37)] and deformable DETR[[53](https://arxiv.org/html/2108.06152#bib.bib53)], which are able to reduce the encoder computational complexity and improve the performance due to multi-scale and higher-resolution.

The comparisons in Table[2](https://arxiv.org/html/2108.06152#S4.T2 "Table 2 ‣ 4.1 Setting ‣ 4 Experiments ‣ Conditional DETR for Fast Training Convergence") surprisingly show that our approach on DC 5 5 5 5-R 50 50 50 50 (16×16\times 16 ×) performs same as deformable DETR-R 50 50 50 50 (multi-scale, 8×8\times 8 ×). Considering that the AP of the single-scale deformable DETR-DC 5 5 5 5-R 50 50 50 50-SS is 41.5 41.5 41.5 41.5 (lower than ours 43.8 43.8 43.8 43.8) (Table[1](https://arxiv.org/html/2108.06152#S3.T1 "Table 1 ‣ 3.4 Visualization and Analysis ‣ 3 Conditional DETR ‣ Conditional DETR for Fast Training Convergence")), one can see that deformable DETR benefits a lot from the multi-scale and higher-resolution encoder that potentially benefit our approach, which is currently not our focus and left as our future work.

The performance of our approach is also on par with TSP-FCOS and TSP-RCNN. The two methods contain a transformer encoder over a small number of selected positions/regions (feature of interest in TSP-FCOS and region proposals in TSP-RCNN) without using the transformer decoder, are extensions of FCOS[[39](https://arxiv.org/html/2108.06152#bib.bib39)] and Faster RCNN[[33](https://arxiv.org/html/2108.06152#bib.bib33)]. It should be noted that position/region selection removes unnecessary computation in self-attention and reduces computation complexity dramatically.

### 4.3 Ablations

Reference points. We compare three ways of forming reference points 𝐬 𝐬\mathbf{s}bold_s: (i) 𝐬=(0,0)𝐬 0 0\mathbf{s}=(0,0)bold_s = ( 0 , 0 ), same to the original DETR, (ii) learn 𝐬 𝐬\mathbf{s}bold_s as model parameters and each prediction is associated with different reference points, and (iii) predict each reference point 𝐬 𝐬\mathbf{s}bold_s from the corresponding object query. We conducted the experiments with ResNet-50 50 50 50 as the backbone. The AP scores are 36.8 36.8 36.8 36.8, 40.7 40.7 40.7 40.7, and 40.9 40.9 40.9 40.9, suggesting that (ii) and (iii) perform on par and better than (i).

Table 3: Ablation study for the ways forming the conditional spatial query. CSQ = our proposed conditional spatial query scheme. Please see the first two paragraphs in Section 5.3 for the meanings of CSQ variants. Our proposed CSQ manner performs better. The backbone ResNet-50 50 50 50 is adopted.

The effect of the way forming the conditional spatial query. We empirically study how the transformation 𝛌 q subscript 𝛌 𝑞\boldsymbol{\uplambda}_{q}bold_λ start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT and the positional embedding 𝐩 s subscript 𝐩 𝑠\mathbf{p}_{s}bold_p start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT of the reference point, used to form the conditional spatial query 𝐩 q=𝛌 q⊙𝐩 s subscript 𝐩 𝑞 direct-product subscript 𝛌 𝑞 subscript 𝐩 𝑠\mathbf{p}_{q}=\boldsymbol{\uplambda}_{q}\odot\mathbf{p}_{s}bold_p start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT = bold_λ start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT ⊙ bold_p start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT, make contributions to the detection performance.

We report the results of our conditional DETR, and the other ways forming the spatial query with: (i) CSQ-P - only the positional embedding 𝐩 s subscript 𝐩 𝑠\mathbf{p}_{s}bold_p start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT, (ii) CSQ-T - only the transformation 𝛌 q subscript 𝛌 𝑞\boldsymbol{\uplambda}_{q}bold_λ start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT, (iii) CSQ-C - the decoder content embedding 𝐟 𝐟\mathbf{f}bold_f, and (iv) CSQ-I - the element-wise product of the transformation predicted from the decoder self-attention output 𝐜 q subscript 𝐜 𝑞\mathbf{c}_{q}bold_c start_POSTSUBSCRIPT italic_q end_POSTSUBSCRIPT and the positional embedding 𝐩 s subscript 𝐩 𝑠\mathbf{p}_{s}bold_p start_POSTSUBSCRIPT italic_s end_POSTSUBSCRIPT. The studies in Table[3](https://arxiv.org/html/2108.06152#S4.T3 "Table 3 ‣ 4.3 Ablations ‣ 4 Experiments ‣ Conditional DETR for Fast Training Convergence") imply that our proposed way (CSQ) performs overall the best, validating our analysis about the transformation predicted from the decoder embedding and the positional embedding of the reference point in Section[3.3](https://arxiv.org/html/2108.06152#S3.SS3 "3.3 Conditional Cross-Attention ‣ 3 Conditional DETR ‣ Conditional DETR for Fast Training Convergence").

Focal loss and offset regression with respect to learned reference point. Our approach follows deformable DETR[[53](https://arxiv.org/html/2108.06152#bib.bib53)]: use the focal loss with 300 300 300 300 object queries to form the classification loss and predict the box center by regressing the offset with respect to the reference point. We report how the two schemes affect the DETR performance in Table[4](https://arxiv.org/html/2108.06152#S4.T4 "Table 4 ‣ 4.3 Ablations ‣ 4 Experiments ‣ Conditional DETR for Fast Training Convergence"). One can see that separately using the focal loss or center offset regression without learning referecence points leads to a slight AP gain and combining them together leads to a larger AP gain. Conditional cross-attention in our approach built on the basis of focal loss and offset regression brings a major gain 4.0 4.0 4.0 4.0.

The effect of linear projections 𝐓 𝐓\mathbf{T}bold_T forming the transformation. Predicting the conditional spatial query needs to learn the linear projection 𝐓 𝐓\mathbf{T}bold_T from the decoder embedding (see Equation[6](https://arxiv.org/html/2108.06152#S3.E6 "6 ‣ 3.3 Conditional Cross-Attention ‣ 3 Conditional DETR ‣ Conditional DETR for Fast Training Convergence")). We empirically study how the linear projection forms affect the performance. The linear projection forms include: an _identity_ matrix that means not to learn the linear projection, a _single_ scalar, a _block_ diagonal matrix meaning that each head has a learned 32×32 32 32 32\times 32 32 × 32 linear projection matrix, a _full_ matrix without constraints, and a _diagonal_ matrix. Figure[5](https://arxiv.org/html/2108.06152#S4.F5 "Figure 5 ‣ 4.3 Ablations ‣ 4 Experiments ‣ Conditional DETR for Fast Training Convergence") presents the results. It is interesting that a single-scalar helps improve the performance, maybe due to narrowing down the spatial range to the object area. Other three forms, _block_ diagonal, _full_, and _diagonal_ (ours), perform on par.

Table 4: The empirical results about the focal loss (FL), offset regression (OR) for box center prediction, and our conditional spatial query (CSQ). The backbone ResNet-50 is adopted.

{tikzpicture}
[baseline] {axis}[ footnotesize, scale only axis, ybar, enlargelimits=0.15, symbolic x coords= Identity, Single, Block, Full, Diagonal, x post scale=1.3, y post scale=.8, ytick distance=1, ymin=37, xtick=data, nodes near coords, nodes near coords align=vertical, ylabel=AP, bar width=16pt, ] \addplot[fill=mayablue, draw=mediumelectricblue, fill opacity=0.66, draw opacity=0.8, text opacity=1] coordinates (Identity, 37.8) (Single, 39.0) (Block, 40.8) (Full, 40.7) (Diagonal, 40.9) ;

Figure 5: The empirical results for different forms of linear projections that are used to compute the spatial queries for conditional multi-head cross-attention. Diagonal (ours), Full, and Block perform on par. The backbone ResNet-50 50 50 50 is adopted. 

5 Conclusion
------------

We present a simple conditional cross-attention mechanism. The key is to learn a spatial query from the corresponding reference point and decoder embedding. The spatial query contains the spatial information mined for the class and box prediction in the previous decoder layer, and leads to spatial attention weight maps highlighting the bands containing extremities and small regions inside the object box. This shrinks the spatial range for the content query to localize the distinct regions, thus relaxing the dependence on the content query and reducing the training difficulty. In the future, we will study the proposed conditional cross-attention mechanism for human pose estimation[[8](https://arxiv.org/html/2108.06152#bib.bib8), [41](https://arxiv.org/html/2108.06152#bib.bib41), [36](https://arxiv.org/html/2108.06152#bib.bib36)] and line segment detection[[43](https://arxiv.org/html/2108.06152#bib.bib43)].

Acknowledgments. We thank the anonymous reviewers for their insightful comments and suggestions on our manuscript.

References
----------

*   [1] Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao. Yolov4: Optimal speed and accuracy of object detection. CoRR, abs/2004.10934, 2020. 
*   [2] Zhaowei Cai and Nuno Vasconcelos. Cascade R-CNN: delving into high quality object detection. In CVPR, 2018. 
*   [3] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In ECCV, 2020. 
*   [4] Yinpeng Chen, Xiyang Dai, Mengchen Liu, Dongdong Chen, Lu Yuan, and Zicheng Liu. Dynamic convolution: Attention over convolution kernels. In CVPR, 2020. 
*   [5] Zhigang Dai, Bolun Cai, Yugeng Lin, and Junying Chen. UP-DETR: unsupervised pre-training for object detection with transformers. CoRR, abs/2011.09094, 2020. 
*   [6] Kaiwen Duan, Song Bai, Lingxi Xie, Honggang Qi, Qingming Huang, and Qi Tian. Centernet: Keypoint triplets for object detection. In ICCV, 2019. 
*   [7] Peng Gao, Minghang Zheng, Xiaogang Wang, Jifeng Dai, and Hongsheng Li. Fast convergence of DETR with spatially modulated co-attention. CoRR, abs/2101.07448, 2021. 
*   [8] Zigang Geng, Ke Sun, Bin Xiao, Zhaoxiang Zhang, and Jingdong Wang. Bottom-up human pose estimation via disentangled keypoint regression. In CVPR, pages 14676–14686, June 2021. 
*   [9] Ross B. Girshick. Fast R-CNN. In ICCV, 2015. 
*   [10] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In AISTATS, 2010. 
*   [11] Maosheng Guo, Yu Zhang, and Ting Liu. Gaussian transformer: A lightweight approach for natural language inference. In AAAI, 2019. 
*   [12] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016. 
*   [13] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, and Andrea Vedaldi. Gather-excite: Exploiting feature context in convolutional neural networks. In NeurIPS, 2018. 
*   [14] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In CVPR, 2018. 
*   [15] Lichao Huang, Yi Yang, Yafeng Deng, and Yinan Yu. Densebox: Unifying landmark localization with end to end object detection. CoRR, abs/1509.04874, 2015. 
*   [16] Xu Jia, Bert De Brabandere, Tinne Tuytelaars, and Luc Van Gool. Dynamic filter networks. In NeurIPS, 2016. 
*   [17] Guolin Ke, Di He, and Tie-Yan Liu. Rethinking positional encoding in language pre-training. CoRR, abs/2006.15595, 2020. 
*   [18] Jaeyoung Kim, Mostafa El-Khamy, and Jungwon Lee. T-GSA: transformer with gaussian-weighted self-attention for speech enhancement. In ICASSP, 2020. 
*   [19] Tao Kong, Fuchun Sun, Huaping Liu, Yuning Jiang, and Jianbo Shi. Foveabox: Beyond anchor-based object detector. CoRR, abs/1904.03797, 2019. 
*   [20] Harold W. Kuhn. The hungarian method for the assignment problem. Naval Research Logistics Quarterly, 1995. 
*   [21] Hei Law and Jia Deng. Cornernet: Detecting objects as paired keypoints. In ECCV, 2018. 
*   [22] Hei Law, Yun Teng, Olga Russakovsky, and Jia Deng. Cornernet-lite: Efficient keypoint based object detection. In BMVC. BMVA Press, 2020. 
*   [23] Yanghao Li, Yuntao Chen, Naiyan Wang, and Zhaoxiang Zhang. Scale-aware trident networks for object detection. In ICCV, pages 6054–6063, 2019. 
*   [24] Tsung-Yi Lin, Priya Goyal, Ross B. Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. TPAMI, 2020. 
*   [25] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C.Lawrence Zitnick. Microsoft COCO: common objects in context. In ECCV, 2014. 
*   [26] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott E. Reed, Cheng-Yang Fu, and Alexander C. Berg. SSD: single shot multibox detector. In ECCV, 2016. 
*   [27] Ilya Loshchilov and Frank Hutter. Fixing weight decay regularization in adam. In ICLR, 2017. 
*   [28] Xin Lu, Buyu Li, Yuxin Yue, Quanquan Li, and Junjie Yan. Grid R-CNN. In CVPR, 2019. 
*   [29] Jiangmiao Pang, Kai Chen, Jianping Shi, Huajun Feng, Wanli Ouyang, and Dahua Lin. Libra R-CNN: towards balanced learning for object detection. In CVPR, 2019. 
*   [30] Joseph Redmon, Santosh Kumar Divvala, Ross B. Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In CVPR, 2016. 
*   [31] Joseph Redmon and Ali Farhadi. YOLO9000: better, faster, stronger. In CVPR, 2017. 
*   [32] Joseph Redmon and Ali Farhadi. Yolov3: An incremental improvement. CoRR, abs/1804.02767, 2018. 
*   [33] Shaoqing Ren, Kaiming He, Ross B. Girshick, and Jian Sun. Faster R-CNN: towards real-time object detection with region proposal networks. TPAMI, 2017. 
*   [34] Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian D. Reid, and Silvio Savarese. Generalized intersection over union: A metric and a loss for bounding box regression. In CVPR, 2019. 
*   [35] Guanglu Song, Yu Liu, and Xiaogang Wang. Revisiting the sibling head in object detector. In CVPR, 2020. 
*   [36] Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. Deep high-resolution representation learning for human pose estimation. In CVPR, pages 5693–5703, 2019. 
*   [37] Zhiqing Sun, Shengcao Cao, Yiming Yang, and Kris Kitani. Rethinking transformer-based set prediction for object detection. CoRR, abs/2011.10881, 2020. 
*   [38] Zhi Tian, Chunhua Shen, and Hao Chen. Conditional convolutions for instance segmentation. In ECCV, 2020. 
*   [39] Zhi Tian, Chunhua Shen, Hao Chen, and Tong He. FCOS: fully convolutional one-stage object detection. In ICCV, 2019. 
*   [40] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 
*   [41] Jingdong Wang, Ke Sun, Tianheng Cheng, Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, Wenyu Liu, and Bin Xiao. Deep high-resolution representation learning for visual recognition. TPAMI, 2019. 
*   [42] Xinlong Wang, Rufeng Zhang, Tao Kong, Lei Li, and Chunhua Shen. Solov2: Dynamic and fast instance segmentation. In NeurIPS, 2020. 
*   [43] Yifan Xu, Weijian Xu, David Cheung, and Zhuowen Tu. Line segment detection using transformers without edges. In CVPR, pages 4257–4266, June 2021. 
*   [44] Brandon Yang, Gabriel Bender, Quoc V. Le, and Jiquan Ngiam. Condconv: Conditionally parameterized convolutions for efficient inference. In NeurIPS, 2019. 
*   [45] Changqian Yu, Bin Xiao, Changxin Gao, Lu Yuan, Lei Zhang, Nong Sang, and Jingdong Wang. Lite-hrnet: A lightweight high-resolution network. In CVPR, pages 10440–10450, June 2021. 
*   [46] Jiahui Yu, Yuning Jiang, Zhangyang Wang, Zhimin Cao, and Thomas S. Huang. Unitbox: An advanced object detection network. In MM, 2016. 
*   [47] Shifeng Zhang, Cheng Chi, Yongqiang Yao, Zhen Lei, and Stan Z. Li. Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection. In CVPR, 2020. 
*   [48] Minghang Zheng, Peng Gao, Xiaogang Wang, Hongsheng Li, and Hao Dong. End-to-end object detection with adaptive clustering transformer. CoRR, abs/2011.09315, 2020. 
*   [49] Xingyi Zhou, Dequan Wang, and Philipp Krähenbühl. Objects as points. CoRR, abs/1904.07850, 2019. 
*   [50] Xingyi Zhou, Jiacheng Zhuo, and Philipp Krähenbühl. Bottom-up object detection by grouping extreme and center points. In CVPR, 2019. 
*   [51] Chenchen Zhu, Fangyi Chen, Zhiqiang Shen, and Marios Savvides. Soft anchor-point object detection. In ECCV, 2020. 
*   [52] Chenchen Zhu, Yihui He, and Marios Savvides. Feature selective anchor-free module for single-shot object detection. In CVPR, 2019. 
*   [53] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable DETR: deformable transformers for end-to-end object detection. CoRR, abs/2010.04159, 2020.

