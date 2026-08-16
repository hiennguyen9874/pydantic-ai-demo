# Big Transfer (BiT): General Visual Representation Learning

Alexander Kolesnikov,\* Lucas Beyer,\* Xiaohua Zhai,\*  
Joan Puigcerver, Jessica Yung, Sylvain Gelly, and Neil Houlsby

Google Research, Brain Team  
Zürich, Switzerland  
{akolesnikov,lbeyer,xzhai}@google.com  
{jpuigcerver,jessicayung,sylvaingelly,neilhoulby}@google.com

**Abstract.** Transfer of pre-trained representations improves sample efficiency and simplifies hyperparameter tuning when training deep neural networks for vision. We revisit the paradigm of pre-training on large supervised datasets and fine-tuning the model on a target task. We scale up pre-training, and propose a simple recipe that we call Big Transfer (BiT). By combining a few carefully selected components, and transferring using a simple heuristic, we achieve strong performance on over 20 datasets. BiT performs well across a surprisingly wide range of data regimes — from 1 example per class to 1 M total examples. BiT achieves 87.5% top-1 accuracy on ILSVRC-2012, 99.4% on CIFAR-10, and 76.3% on the 19 task Visual Task Adaptation Benchmark (VTAB). On small datasets, BiT attains 76.8% on ILSVRC-2012 with 10 examples per class, and 97.0% on CIFAR-10 with 10 examples per class. We conduct detailed analysis of the main components that lead to high transfer performance.

## 1 Introduction

Strong performance using deep learning usually requires a large amount of task-specific data and compute. These per-task requirements can make new tasks prohibitively expensive. Transfer learning offers a solution: task-specific data and compute are replaced with a pre-training phase. A network is trained once on a large, generic dataset, and its weights are then used to initialize subsequent tasks which can be solved with fewer data points, and less compute [40,44,14].

We revisit a simple paradigm: pre-train on a large supervised source dataset, and fine-tune the weights on the target task. Numerous improvements to deep network training have recently been introduced, e.g. [55,62,26,35,22,1,64,67,54,60]. We aim not to introduce a new component or complexity, but to provide a recipe that uses the minimal number of tricks yet attains excellent performance on many tasks. We call this recipe “Big Transfer” (BiT).

We train networks on three different scales of datasets. The largest, BiT-L is trained on the JFT-300M dataset [51], which contains 300 M noisily labelled

---

\* Equal contributionFig. 1: Transfer performance of our pre-trained model, BiT-L, the previous state-of-the-art (SOTA), and a ResNet-50 baseline pre-trained on ILSVRC-2012 to downstream tasks. Here we consider only methods that are pre-trained independently of the final task (generalist representations), like BiT. The bars show the accuracy when fine-tuning on the full downstream dataset. The curve on the left-hand side of each plot shows that BiT-L performs well even when transferred using only few images (1 to 100) per class.

images. We transfer BiT to many diverse tasks; with training set sizes ranging from 1 example per class to 1M total examples. These tasks include ImageNet’s ILSVRC-2012 [10], CIFAR-10/100 [27], Oxford-IIIT Pet [41], Oxford Flowers-102 [39] (including few-shot variants), and the 1000-sample VTAB-1k benchmark [66], which consists of 19 diverse datasets. BiT-L attains state-of-the-art performance on many of these tasks, and is surprisingly effective when very little downstream data is available (Figure 1). We also train BiT-M on the public ImageNet-21k dataset, and attain marked improvements over the popular ILSVRC-2012 pre-training.

Importantly, BiT only needs to be pre-trained once and subsequent fine-tuning to downstream tasks is cheap. By contrast, other state-of-the-art methods require extensive training on support data conditioned on the task at hand [38,61,63]. Not only does BiT require a short fine-tuning protocol for each new task, but BiT also does not require extensive hyperparameter tuning on new tasks. Instead, we present a heuristic for setting the hyperparameters for transfer, which works well on our diverse evaluation suite.

We highlight the most important components that make Big Transfer effective, and provide insight into the interplay between scale, architecture, and training hyperparameters. For practitioners, we will release the performant BiT-M model trained on ImageNet-21k.

## 2 Big Transfer

We review the components that we found necessary to build an effective network for transfer. *Upstream* components are those used during pre-training, and *downstream* are those used during fine-tuning to a new task.## 2.1 Upstream Pre-Training

The first component is scale. It is well-known in deep learning that larger networks perform better on their respective tasks [15,48]. Further, it is recognized that larger datasets require larger architectures to realize benefits, and vice versa [25,45]. We study the effectiveness of scale (during pre-training) in the context of transfer learning, including transfer to tasks with very few datapoints. We investigate the interplay between computational budget (training time), architecture size, and dataset size. For this, we train three BiT models on three large datasets: ILSVRC-2012 [46] which contains 1.3M images (BiT-S), ImageNet-21k [10] which contains 14M images (BiT-M), and JFT [51] which contains 300M images (BiT-L).

The second component is Group Normalization (GN) [60] and Weight Standardization (WS) [34]. Batch Normalization (BN) [21] is used in most state-of-the-art vision models to stabilize training. However, we find that BN is detrimental to Big Transfer for two reasons. First, when training large models with small per-device batches, BN performs poorly or incurs inter-device synchronization cost. Second, due to the requirement to update running statistics, BN is detrimental for transfer. GN, when combined with WS, has been shown to improve performance on small-batch training for ImageNet and COCO [34]. Here, we show that the combination of GN and WS is useful for training with large batch sizes, and has a significant impact on transfer learning.

## 2.2 Transfer to Downstream Tasks

We propose a cheap fine-tuning protocol that applies to many diverse downstream tasks. Importantly, we avoid expensive hyperparameter search for every new task and dataset size; we try only one hyperparameter per task. We use a heuristic rule—which we call BiT-HyperRule—to select the most important hyperparameters for tuning as a simple function of the task’s intrinsic image resolution and number of datapoints. We found it important to set the following hyperparameters per-task: training schedule length, resolution, and whether to use MixUp regularization [67]. We use BiT-HyperRule for over 20 tasks in this paper, with training sets ranging from 1 example per class to over 1M total examples. The exact settings for BiT-HyperRule are presented in Section 3.3.

During fine-tuning, we use the following standard data pre-processing: we resize the image to a square, crop out a smaller random square, and randomly horizontally flip the image at training time. At test time, we only resize the image to a fixed size. In some tasks horizontal flipping or cropping destroys the label semantics, making the task impossible. An example is if the label requires predicting object orientation or coordinates in pixel space. In these cases we omit flipping or cropping when appropriate.

Recent work has shown that existing augmentation methods introduce inconsistency between training and test resolutions for CNNs [57]. Therefore, it is common to scale up the resolution by a small factor at test time. As an alternative, one can add a step at which the trained model is fine-tuned to thetest resolution [57]. The latter is well-suited for transfer learning; we include the resolution change during our fine-tuning step.

We found that MixUp [67] is not useful for pre-training BiT, likely due to the abundance of data. However, it is sometimes useful for transfer. Interestingly, it is most useful for mid-sized datasets, and not for few-shot transfer, see Section 3.3 for where we apply MixUp.

Surprisingly, we do not use any of the following forms of regularization during downstream tuning: weight decay to zero, weight decay to initial parameters [31], or dropout. Despite the fact that the network is very large—BiT has 928 million parameters—the performance is surprisingly good without these techniques and their respective hyperparameters, even when transferring to very small datasets. We find that setting an appropriate schedule length, i.e. training longer for larger datasets, provides sufficient regularization.

### 3 Experiments

We train three upstream models using three datasets at different scales: BiT-S, BiT-M, and BiT-L. We evaluate these models on many downstream tasks and attain very strong performance on high and low data regimes.

#### 3.1 Data for Upstream Training

BiT-S is trained on the ILSVRC-2012 variant of ImageNet, which contains 1.28 million images and 1000 classes. Each image has a single label. BiT-M is trained on the full ImageNet-21k dataset [10], a public dataset containing 14.2 million images and 21k classes organized by the WordNet hierarchy. Images may contain multiple labels. BiT-L is trained on the JFT-300M dataset [51,38,61]. This dataset is a newer version of that used in [18,8]. JFT-300M consists of around 300 million images with 1.26 labels per image on average. The labels are organized into a hierarchy of 18 291 classes. Annotation is performed using an automatic pipeline, and are therefore imperfect; approximately 20% of the labels are noisy. We remove all images present in downstream test sets from JFT-300M. See Supplementary Material section C for details. Note: the “-S/M/L” suffix refers to the pre-training datasets size and schedule, not architecture. We train BiT with several architecture sizes, the default (largest) being ResNet152x4.

#### 3.2 Downstream Tasks

We evaluate BiT on long-standing benchmarks: ILSVRC-2012 [10], CIFAR-10/100 [27], Oxford-IIIT Pet [41] and Oxford Flowers-102 [39]. These datasets differ in the total number of images, input resolution and nature of their categories, from general object categories in ImageNet and CIFAR to fine-grained ones in Pets and Flowers. We fine-tune BiT on the official training split and report results on the official *test* split if publicly available. Otherwise, we use the *val* split.Table 1: Top-1 accuracy for BiT-L on many datasets using a single model and single hyperparameter setting per task (BiT-HyperRule). The entries show median  $\pm$  standard deviation across 3 fine-tuning runs. Specialist models are those that condition pre-training on each task, while generalist models, including BiT, perform task-independent pre-training. (\*Concurrent work.)

<table border="1">
<thead>
<tr>
<th></th>
<th>BiT-L</th>
<th>Generalist SOTA</th>
<th>Specialist SOTA</th>
</tr>
</thead>
<tbody>
<tr>
<td>ILSVRC-2012</td>
<td><b>87.54 <math>\pm</math> 0.02</b></td>
<td>86.4 [57]</td>
<td>88.4 [61]*</td>
</tr>
<tr>
<td>CIFAR-10</td>
<td><b>99.37 <math>\pm</math> 0.06</b></td>
<td>99.0 [19]</td>
<td>-</td>
</tr>
<tr>
<td>CIFAR-100</td>
<td><b>93.51 <math>\pm</math> 0.08</b></td>
<td>91.7 [55]</td>
<td>-</td>
</tr>
<tr>
<td>Pets</td>
<td><b>96.62 <math>\pm</math> 0.23</b></td>
<td>95.9 [19]</td>
<td>97.1 [38]</td>
</tr>
<tr>
<td>Flowers</td>
<td><b>99.63 <math>\pm</math> 0.03</b></td>
<td>98.8 [55]</td>
<td>97.7 [38]</td>
</tr>
<tr>
<td>VTAB (19 tasks)</td>
<td><b>76.29 <math>\pm</math> 1.70</b></td>
<td>70.5 [58]</td>
<td>-</td>
</tr>
</tbody>
</table>

To further assess the generality of representations learned by BiT, we evaluate on the Visual Task Adaptation Benchmark (VTAB) [66]. VTAB consists of 19 diverse visual tasks, each of which has 1000 training samples (VTAB-1k variant). The tasks are organized into three groups: *natural*, *specialized* and *structured*. The VTAB-1k score is top-1 recognition performance averaged over these 19 tasks. The *natural* group of tasks contains classical datasets of natural images captured using standard cameras. The *specialized* group also contains images captured in the real world, but through specialist equipment, such as satellite or medical images. Finally, the *structured* tasks assess understanding of the the structure of a scene, and are mostly generated from synthetic environments. Example structured tasks include object counting and 3D depth estimation.

### 3.3 Hyperparameter Details

**Upstream Pre-Training** All of our BiT models use a vanilla ResNet-v2 architecture [16], except that we replace all Batch Normalization [21] layers with Group Normalization [60] and use Weight Standardization [43] in all convolutional layers. See Section 4.3 for analysis. We train ResNet-152 architectures in all datasets, with every hidden layer widened by a factor of four (ResNet152x4). We study different model sizes and the coupling with dataset size in Section 4.1.

We train all of our models upstream using SGD with momentum. We use an initial learning rate of 0.03, and momentum 0.9. During image preprocessing stage we use image cropping technique from [53] and random horizontal mirroring followed by  $224 \times 224$  image resize. We train both BiT-S and BiT-M for 90 epochs and decay the learning rate by a factor of 10 at 30, 60 and 80 epochs. For BiT-L, we train for 40 epochs and decay the learning rate after 10, 23, 30 and 37 epochs. We use a global batch size of 4096 and train on a Cloud TPUv3-512 [24], resulting in 8 images per chip. We use linear learning rate warm-up for 5000 optimization steps and multiply the learning rate by  $\frac{\text{batch\_size}}{256}$  following [11]. During pre-training we use a weight decay of 0.0001, but as discussed in Section 2, we do not use any weight decay during transfer.Table 2: Improvement in accuracy when pre-training on the public ImageNet-21k dataset over the “standard” ILSVRC-2012. Both models are ResNet152x4.

<table border="1">
<thead>
<tr>
<th></th>
<th>ILSVRC-2012</th>
<th>CIFAR-10</th>
<th>CIFAR-100</th>
<th>Pets</th>
<th>Flowers</th>
<th>VTAB-1k (19 tasks)</th>
</tr>
</thead>
<tbody>
<tr>
<td>BiT-S (ILSVRC-2012)</td>
<td>81.30</td>
<td>97.51</td>
<td>86.21</td>
<td>93.97</td>
<td>89.89</td>
<td>66.87</td>
</tr>
<tr>
<td>BiT-M (ImageNet-21k)</td>
<td>85.39</td>
<td>98.91</td>
<td>92.17</td>
<td>94.46</td>
<td>99.30</td>
<td>70.64</td>
</tr>
<tr>
<td>Improvement</td>
<td>+4.09</td>
<td>+1.40</td>
<td>+5.96</td>
<td>+0.49</td>
<td>+9.41</td>
<td>+3.77</td>
</tr>
</tbody>
</table>

**Downstream Fine-Tuning** To attain a low per-task adaptation cost, we do not perform any hyperparameter sweeps downstream. Instead, we present BiT-HyperRule, a heuristic to determine all hyperparameters for fine-tuning. Most hyperparameters are fixed across all datasets, but schedule, resolution, and usage of MixUp depend on the tasks image resolution and training set size.

For all tasks, we use SGD with an initial learning rate of 0.003, momentum 0.9, and batch size 512. We resize input images with area smaller than  $96 \times 96$  pixels to  $160 \times 160$  pixels, and then take a random crop of  $128 \times 128$  pixels. We resize larger images to  $448 \times 448$  and take a  $384 \times 384$ -sized crop.<sup>1</sup> We apply random crops and horizontal flips for all tasks, except those for which cropping or flipping destroys the label semantics, see Supplementary section F for details.

For schedule length, we define three scale regimes based on the number of examples: we call *small* tasks those with fewer than 20 k labeled examples, *medium* those with fewer than 500 k, and any larger dataset is a *large* task. We fine-tune BiT for 500 steps on small tasks, for 10k steps on medium tasks, and for 20k steps on large tasks. During fine-tuning, we decay the learning rate by a factor of 10 at 30%, 60% and 90% of the training steps. Finally, we use MixUp [67], with  $\alpha = 0.1$ , for medium and large tasks. See Supplementary section A for details.

### 3.4 Standard Computer Vision Benchmarks

We evaluate BiT-L on standard benchmarks and compare its performance to the current state-of-the-art results (Table 1). We separate models that perform task-independent pre-training (“general” representations), from those that perform task-dependent auxiliary training (“specialist” representations). The specialist methods condition on a particular task, for example ILSVRC-2012, then train using a large support dataset, such as JFT-300M [38] or Instagram-1B [63]. See discussion in Section 5. Specialist representations are highly effective, but require a large training cost *per task*. By contrast, generalized representations require large-scale training *only once*, followed by a cheap adaptation phase.

BiT-L outperforms previously reported generalist SOTA models as well as, in many cases, the SOTA specialist models. Inspired by strong results of BiT-L trained on JFT-300M, we also train models on the public ImageNet-21k dataset.

<sup>1</sup> For our largest R152x4, we increase resolution to  $512 \times 512$  and crop to  $480 \times 480$ .Fig. 2: Experiments in the low data regime. **Left:** Transfer performance of BiT-L. Each point represents the result after training on a balanced random subsample of the dataset (5 subsamples per dataset). The median across runs is highlighted by the curves. The variance across data samples is usually low, with the exception of 1-shot CIFAR-10, which contains only 10 images. **Right:** We summarize the state-of-the-art in semi-supervised learning as reference points. Note that a direct comparison is not meaningful; unlike BiT, semi-supervised methods have access to extra unlabelled data from the training distribution, but they do not make use of out-of-distribution labeled data.

This dataset is more than 10 times bigger than ILSVRC-2012, but it is mostly overlooked by the research community. In Table 2 we demonstrate that BiT-M trained on ImageNet-21k leads to substantially improved visual representations compared to the same model trained on ILSVRC-2012 (BiT-S), as measured by all our benchmarks. In Section 4.2, we discuss pitfalls that may have hindered wide adoption of ImageNet-21k as a dataset model for pre-training and highlight crucial components of BiT that enabled success on this large dataset.

For completeness, we also report top-5 accuracy on ILSVRC-2012 with median  $\pm$  standard deviation format across 3 runs:  $98.46\% \pm 0.02\%$  for BiT-L,  $97.69\% \pm 0.02\%$  for BiT-M and  $95.65\% \pm 0.03\%$  for BiT-S.

### 3.5 Tasks with Few Datapoints

We study the number of *downstream* labeled samples required to transfer BiT-L successfully. We transfer BiT-L using subsets of ILSVRC-2012, CIFAR-10, and CIFAR-100, down to 1 example per class. We also evaluate on a broader suite of 19 VTAB-1k tasks, each of which has 1000 training examples.

Figure 2 (left half) shows BiT-L using few-shots on ILSVRC-2012, CIFAR-10, and CIFAR-100. We run multiple random subsamples, and plot every trial. Surprisingly, even with very few samples per class, BiT-L demonstrates strong performance and quickly approaches performance of the full-data regime. In particular, with just 5 labeled samples per class it achieves top-1 accuracy of 72.0% on ILSVRC-2012 and with 100 samples the top-1 accuracy goes to 84.1%. On CIFAR-100, we achieve 82.6% with just 10 samples per class.Fig. 3: Results on VTAB (19 tasks) with 1000 examples/task, and the current SOTA. It compares methods that sweep few hyperparameters per task: either four hyperparameters in previous work (“4 HPs”) or the single BiT-HyperRule.

Semi-supervised learning also tackles learning with few labels. However, such approaches are not directly comparable to BiT. BiT uses extra labelled out-of-domain data, whereas semi-supervised learning uses extra unlabelled in-domain data. Nevertheless, it is interesting to observe the relative benefits of transfer from out-of-domain labelled data versus in-domain semi-supervised data. In Figure 2 we show state-of-the-art results from the semi-supervised learning.

Figure 3 shows the performance of BiT-L on the 19 VTAB-1k tasks. BiT-L with BiT-HyperRule substantially outperforms the previously reported state-of-the-art. When looking into performance of VTAB-1k task subsets, BiT is the best on *natural*, *specialized* and *structured* tasks. The recently-proposed VIVI-Ex-100% [58] model that employs video data during upstream pre-training shows very similar performance on the *structured* tasks.

We investigate heavy per-task hyperparameter tuning in Supplementary Material section A and conclude that this further improves performance.

### 3.6 ObjectNet: Recognition on a “Real-World” Test Set

We evaluate BiT on the new test-only ObjectNet dataset [2]. Importantly, this dataset closely resembles real-life scenarios, where object categories may appear in non-canonical context, viewpoint, rotation, etc. There are 313 object classes in total, with 113 overlapping with ILSVRC-2012. We follow the literature [2,6] and evaluate our models on those 113 classes.

Figure 4 shows that larger architectures and pre-training on more data results in higher accuracies. Crucially, our results highlight that scaling both is essential for achieving unprecedented top-5 accuracy of 80.0%, an almost 25% absolute improvement over the previous state-of-the-art. We provide numeric results and additional results when classifying individual object bounding boxes [6] in the Supplementary Material section B.

Fig. 4: Accuracy of BiT models along with baselines on ObjectNet. R is short for ResNet in x-axis.### 3.7 Object Detection

Finally, we evaluate BiT on object detection. We use the COCO-2017 dataset [34] and train a top-performing object detector, RetinaNet [33], using pre-trained BiT models as backbones. Due to memory constraints, we use the ResNet-101x3 architecture for all of our BiT models. We fine-tune the detection models on the COCO-2017 train split and report results on the validation split using the standard metric [34] in Table 3. Here, we do not use BiT-HyperRule, but stick to the standard RetinaNet training protocol, see the Supplementary Material section E for details. Table 3 demonstrates that BiT models outperform standard ImageNet pre-trained models. We can see clear benefits of pre-training on large data beyond ILSVRC-2012: pre-training on ImageNet-21k results in a 1.5 point improvement in Average Precision (AP), while pre-training on JFT-300M further improves performance by 0.6 points.

<table border="1">
<thead>
<tr>
<th>Model</th>
<th>Upstream data</th>
<th>AP</th>
</tr>
</thead>
<tbody>
<tr>
<td>RetinaNet [33]</td>
<td>ILSVRC-2012</td>
<td>40.8</td>
</tr>
<tr>
<td>RetinaNet (BiT-S)</td>
<td>ILSVRC-2012</td>
<td>41.7</td>
</tr>
<tr>
<td>RetinaNet (BiT-M)</td>
<td>ImageNet-21k</td>
<td>43.2</td>
</tr>
<tr>
<td>RetinaNet (BiT-L)</td>
<td>JFT-300M</td>
<td><b>43.8</b></td>
</tr>
</tbody>
</table>

Table 3: Object detection performance on COCO-2017 [34] validation data of RetinaNet models with pre-trained BiT backbones and the literature baseline.

## 4 Analysis

We analyse various components of BiT: we demonstrate the importance of model capacity, discuss practical optimization caveats and choice of normalization layer.

### 4.1 Scaling Models and Datasets

The general consensus is that larger neural networks result in better performance. We investigate the interplay between model capacity and upstream dataset size on downstream performance. We evaluate the BiT models of different sizes (ResNet-50x1, ResNet-50x3, ResNet-101x1, ResNet-101x3, and ResNet-152x4) trained on ILSVRC-2012, ImageNet-21k, and JFT-300M on various downstream benchmarks. These results are summarized in Figure 5.

Fig. 5: Effect of upstream data (shown on the x-axis) and model size on downstream performance. Note that exclusively using more data or larger models may hurt performance; instead, both need to be increased in tandem.Fig. 6: Performance of BiT models in the low-data regime. The x-axis corresponds to the architecture, where R is short for ResNet. We pre-train on the three upstream datasets and evaluate on two downstream datasets: ILSVRC-2012 (left) and CIFAR-10 (right) with 1 or 5 examples per class. For each scenario, we train 5 models on random data subsets, represented by the lighter dots. The line connects the medians of these five runs.

When pre-training on ILSVRC-2012, the benefit from larger models diminishes. However, the benefits of larger models are more pronounced on the larger two datasets. A similar effect is observed when training on Instagram hashtags [36] and in language modelling [25].

Not only is there limited benefit of training a large model size on a small dataset, but there is also limited (or even negative) benefit from training a small model on a larger dataset. Perhaps surprisingly, the ResNet-50x1 model trained on the JFT-300M dataset can even perform worse than the same architecture trained on the smaller ImageNet-21k. Thus, if one uses only a ResNet50x1, one may conclude that scaling up the dataset does not bring any additional benefits. However, with larger architectures, models pre-trained on JFT-300M significantly outperform those pre-trained on ILSVRC-2012 or ImageNet-21k.

Figure 2 shows that BiT-L attains strong results even on tiny downstream datasets. Figure 6 ablates few-shot performance across different pre-training datasets and architectures. In the extreme case of one example per class, larger architectures outperform smaller ones when pre-trained on large upstream data. Interestingly, on ILSVRC-2012 with few shots, BiT-L trained on JFT-300M outperforms the models trained on the entire ILSVRC-2012 dataset itself. Note that for comparability, the classifier head is re-trained from scratch during fine-tuning, even when transferring ILSVRC-2012 full to ILSVRC-2012 few shot.

## 4.2 Optimization on Large Datasets

For standard computer vision datasets such as ILSVRC-2012, there are well-known training procedures that are robust and lead to good performance. Progress in high-performance computing has made it feasible to learn from much larger datasets, such as ImageNet-21k, which has 14.2M images compared to ILSVRC-2012’s 1.28M. However, there are no established procedures for training from such large datasets. In this section we provide some guidelines.Fig. 7: **Left:** Applying the “standard” computational budget of ILSVRC-2012 to the larger ImageNet-21k seems detrimental. Only when we train longer (3x and 10x) do we see the benefits of training on the larger dataset. **Middle:** The learning progress of a ResNet-101x3 on JFT-300M seems to be flat even after 8 GPU-weeks, but after 8 GPU-months progress is clear. If one decays the learning rate too early (dashed curve), final performance is significantly worse. **Right:** Faster initial convergence with lower weight decay may trick the practitioner into selecting a sub-optimal value. Higher weight decay converges more slowly, but results in a better final model.

Sufficient computational budget is crucial for training performant models on large datasets. The standard ILSVRC-2012 training schedule processes roughly 100 million images ( $1.28\text{M images} \times 90\text{ epochs}$ ). However, if the same computational budget is applied to ImageNet-21k, the resulting model performs worse on ILSVRC-2012, see Figure 7, left. Nevertheless, as shown in the same figure, by increasing the computational budget, we not only recover ILSVRC-2012 performance, but significantly outperforms it. On JFT-300M the validation error may not improve over a long time—Figure 7 middle plot, “8 GPU weeks” zoom-in—although the model is still improving as evidenced by the longer time window.

Another important aspect of pre-training with large datasets is the weight decay. Lower weight decay can result in an apparent acceleration of convergence, Figure 7 rightmost plot. However, this setting eventually results in an under-performing final model. This counter-intuitive behavior stems from the interaction of weight decay and normalization layers [29,32]. Low weight decay results in growing weight norms, which in turn results in a diminishing effective learning rate. Initially this effect creates an impression of faster convergence, but it eventually prevents further progress. A sufficiently large weight decay is required to avoid this effect, and throughout we use  $10^{-4}$ .

Finally, we note that in all of our experiments we use stochastic gradient descent with momentum without any modifications. In our preliminary experiments we did not observe benefits from more involved adaptive gradient methods.Table 4: Top-1 accuracy of ResNet-50 trained from scratch on ILSVRC-2012 with a batch-size of 4096.

<table border="1">
<thead>
<tr>
<th></th>
<th>Plain Conv</th>
<th>Weight Std.</th>
</tr>
</thead>
<tbody>
<tr>
<td>Batch Norm.</td>
<td>75.6</td>
<td>75.8</td>
</tr>
<tr>
<td>Group Norm.</td>
<td>70.2</td>
<td><b>76.0</b></td>
</tr>
</tbody>
</table>

Table 5: Transfer performance of the corresponding models from Table 4 fine-tuned to the 19 VTAB-1k tasks.

<table border="1">
<thead>
<tr>
<th></th>
<th>Plain Conv</th>
<th>Weight Std.</th>
</tr>
</thead>
<tbody>
<tr>
<td>Batch Norm.</td>
<td>67.72</td>
<td>66.78</td>
</tr>
<tr>
<td>Group Norm.</td>
<td>68.77</td>
<td><b>70.39</b></td>
</tr>
</tbody>
</table>

### 4.3 Large Batches, Group Normalization, Weight Standardization

Currently, training on large datasets is only feasible using many hardware accelerators. Data parallelism is the most popular distribution strategy, and this naturally entails large batch sizes. Many known algorithms for training with large batch sizes use Batch Normalization (BN) [21] as a component [12] or even highlight it as the key instrument required for large batch training [9].

Our larger models have a high memory requirement for any single accelerator chip, which necessitates small per-device batch sizes. However, BN performs worse when the number of images on each accelerator is too low [20]. An alternative strategy is to accumulate BN statistics across all of the accelerators. However, this has two major drawbacks. First, computing BN statistics across large batches has been shown to harm generalization [9]. Second, using global BN requires many aggregations across accelerators which incurs significant latency.

We investigated Group Normalization (GN) [60] and Weight Standardization (WS) [43] as alternatives to BN. We tested large batch training using 128 accelerator chips and a batch size of 4096. We found that GN alone does not scale to large batches; we observe a performance drop of 5.4% on ILSVRC-2012 top-1 accuracy compared to using BN in a ResNet-50x1. The addition of WS enables GN to scale to such large batches, even outperforming BN, see Table 4.

We are not only interested in upstream performance, but also how models trained with GN and WS transfer. We thus transferred models with different combinations of BN, GN, and WS pre-trained on ILSVRC-2012 to the 19 tasks defined by VTAB. The results in Table 5 indicate that the GN/WS combination transfers better than BN, so we use GN/WS in all BiT models.

## 5 Related Work

**Large-scale Weakly Supervised Learning of Representations** A number of prior works use large supervised datasets for pre-training visual representations [23,51,30,36]. In [23,30] the authors use a dataset containing 100M Flickr images [56]. This dataset appears to transfer less well than JFT-300M. While studying the effect of dataset size, [51] show good transfer performance when training on JFT-300M, despite reporting a large degree of noise (20% precision errors) in the labels. An even larger, noisily labelled dataset of 3.5B Instagram images is used in [36]. This increase in dataset size and an improved model architecture [62] lead to better results when transferring to ILSVRC-2012. Weshow that we can attain even better performance with ResNet using JFT-300M with appropriate adjustments presented in Section 2. The aforementioned papers focus on transfer to ImageNet classification, and COCO or VOC detection and segmentation. We show that transfer is also highly effective in the low data regime, and works well on the broader set of 19 tasks in VTAB [66].

**Specialized Representations** Rather than pre-train generic representations, recent works have shown strong performance by training task-specific representations [63,38,61]. These papers condition on a particular task when training on a large support dataset. [63,61] train student networks on a large unlabelled support dataset using the predictions of a teacher network trained on the target task. [38] compute importance weights on the a labelled support dataset by conditioning on the target dataset. They then train the representations on the re-weighted source data. Even though these approaches may lead to superior results, they require knowing the downstream dataset in advance and substantial computational resources for each downstream dataset.

**Unsupervised and Semi-Supervised Representation learning** Self-supervised methods have shown the ability to leverage unsupervised datasets to transfer to labelled tasks. For example, [13] show that unsupervised representations trained on 1B unlabelled Instagram images transfer comparably or better than supervised ILSVRC-2012 features. Semi-supervised learning exploits unsupervised data drawn from the same domain as the labelled data. [5,50] used semi-supervised learning to attain strong performance on CIFAR-10 and SVHN using only 40 or 250 labels. Recent works combine self-supervised and semi-supervised learning to attain good performance with fewer labels on ImageNet [65,17]. [66] study many representation learning algorithms (unsupervised, semi-supervised, and supervised) and evaluate their representation’s ability to generalize to novel tasks, concluding that a combination of supervised and self-supervised signals works best. However, all models were trained on ILSVRC-2012. We show that supervised pre-training on larger datasets continues to be an effective strategy.

**Few-shot Learning** Many strategies have been proposed to attain good performance when faced with novel classes and only a few examples per class. Meta-learning or metric-learning techniques have been proposed to learn with few or no labels [59,49,52]. However, recent work has shown that a simple linear classifier on top of pre-trained representations or fine-tuning can attain similar or better performance [7,37]. The upstream pre-training and downstream few-shot learning are usually performed on the same domain, with disjoint class labels. In contrast, our goal is to find a generalist representation which works well when transferring to many downstream tasks.Fig. 8: Cases where BiT-L’s predictions (top word) do not match the ground-truth labels (bottom word), and hence are counted as top-1 errors. **Left:** All mistakes on CIFAR-10, colored by whether five human raters agreed with BiT-L’s prediction (green), with the ground-truth label (red) or were unsure or disagreed with both (yellow). **Right:** Selected representative mistakes of BiT-L on ILSVRC-2012. Top group: The model’s prediction is more representative of the primary object than the label. Middle group: According to top-1 accuracy the model is incorrect, but according to top-5 it is correct. Bottom group: The model’s top-10 predictions are incorrect.

## 6 Discussion

We revisit classical transfer learning, where a large pre-trained generalist model is fine-tuned to downstream tasks of interest. We provide a simple recipe which exploits large scale pre-training to yield good performance on all of these tasks. BiT uses a clean training and fine-tuning setup, with a small number of carefully selected components, to balance complexity and performance.

In Figure 8 and the Supplementary Material section D, we take a closer look at the remaining mistakes that BiT-L makes. In many cases, we see that these label/prediction mismatches are not true ‘mistakes’: the model’s classification is valid, but it does not match the label. For example, the model may identify another prominent object when there are multiple objects in the image, or may provide an valid classification when the main entity has multiple attributes. There are also cases of label noise, where the model’s prediction is a better fitthan the ground-truth label. In a quantitative study, we found that around half of the model’s mistakes on CIFAR-10 are due to ambiguity or label noise (see Figure 8, left), and in only 19.21% of the ILSVRC-2012 mistakes do human raters clearly agree with the label over the prediction. Overall, by inspecting these mistakes, we observe that performance on the standard vision benchmarks seems to approach a saturation point.

We therefore explore the effectiveness of transfer to two classes of more challenging tasks: classical image recognition tasks, but with very few labelled examples to adapt to the new domain, and VTAB, which contains more diverse tasks, such as spatial localization, tasks from simulated environments, and medical and satellite imaging tasks. These benchmarks are much further from saturation; while BiT-L performs well on them, there is still substantial room for further progress.

## 7 Acknowledgements

We thank the whole Google Brain team in Zürich and its collaborators for many fruitful discussions and engineering support. In particular, we thank Andrei Giurgiu for finding a bug in our data input pipeline, Marcin Michalski for the naming idea and general helpful advice, and Damien Vincent and Daniel Keysers for detailed feedback on the initial draft of this paper.

## References

1. 1. Athiwaratkun, B., Finzi, M., Izmailov, P., Wilson, A.G.: There are many consistent explanations of unlabeled data: Why you should average. In: ICLR (2019)
2. 2. Barbu, A., Mayo, D., Alverio, J., Luo, W., Wang, C., Gutfreund, D., Tenenbaum, J., Katz, B.: Objectnet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. In: Advances in Neural Information Processing Systems (2019)
3. 3. Barz, B., Denzler, J.: Do we train on test data? purging CIFAR of near-duplicates. CoRR (2019), <http://arxiv.org/abs/1902.00423>
4. 4. Beery, S., Horn, G.V., Perona, P.: Recognition in terra incognita. CoRR [abs/1807.04975](https://arxiv.org/abs/1807.04975) (2018), <http://arxiv.org/abs/1807.04975>
5. 5. Berthelot, D., Carlini, N., Cubuk, E.D., Kurakin, A., Sohn, K., Zhang, H., Rafel, C.: ReMixMatch: Semi-supervised learning with distribution alignment and augmentation anchoring. arXiv preprint arXiv:1911.09785 (2019)
6. 6. Borji, A.: Objectnet dataset: Reanalysis and correction. In: arXiv 2004.02042 (2020)
7. 7. Chen, W., Liu, Y., Kira, Z., Wang, Y.F., Huang, J.: A closer look at few-shot classification. In: ICLR (2019)
8. 8. Chollet, F.: Xception: Deep learning with depthwise separable convolutions. In: CVPR (2017)
9. 9. De, S., Smith, S.L.: Batch normalization has multiple benefits: An empirical study on residual networks (2020), <https://openreview.net/forum?id=BJeVklHtPr>
10. 10. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A large-scale hierarchical image database. In: CVPR (2009)1. 11. Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., He, K.: Accurate, large minibatch sgd: training imagenet in 1 hour. *arXiv preprint arXiv:1706.02677* (2017)
2. 12. Goyal, P., Dollár, P., Girshick, R.B., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., He, K.: Accurate, large minibatch sgd: Training imagenet in 1 hour. *ArXiv abs/1706.02677* (2017)
3. 13. He, K., Fan, H., Wu, Y., Xie, S., Girshick, R.: Momentum contrast for unsupervised visual representation learning. *arXiv preprint arXiv:1911.05722* (2019)
4. 14. He, K., Girshick, R., Dollár, P.: Rethinking imagenet pre-training. In: *ICCV* (2019)
5. 15. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: *CVPR* (2016)
6. 16. He, K., Zhang, X., Ren, S., Sun, J.: Identity mappings in deep residual networks. In: *ECCV* (2016)
7. 17. Hénaff, O.J., Razavi, A., Doersch, C., Eslami, S., Oord, A.v.d.: Data-efficient image recognition with contrastive predictive coding. *arXiv preprint arXiv:1905.09272* (2019)
8. 18. Hinton, G., Vinyals, O., Dean, J.: Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531* (2015)
9. 19. Huang, Y., Cheng, Y., Chen, D., Lee, H., Ngiam, J., Le, Q.V., Chen, Z.: GPipe: Efficient training of giant neural networks using pipeline parallelism. *arXiv preprint arXiv:1811.06965* (2018)
10. 20. Ioffe, S.: Batch renormalization: Towards reducing minibatch dependence in batch-normalized models. In: *NIPS* (2017)
11. 21. Ioffe, S., Szegedy, C.: Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML* (2015)
12. 22. Izmailov, P., Podoprikin, D., Garipov, T., Vetrov, D., Wilson, A.G.: Averaging weights leads to wider optima and better generalization. *arXiv preprint arXiv:1803.05407* (2018)
13. 23. Joulin, A., van der Maaten, L., Jabri, A., Vasilache, N.: Learning visual features from large weakly supervised data. In: *ECCV* (2016)
14. 24. Jouppi, N.P., Young, C., Patil, N., Patterson, D., Agrawal, G., Bajwa, R., Bates, S., Bhatia, S., Boden, N., Borchers, A., et al.: In-datacenter performance analysis of a tensor processing unit. In: *International Symposium on Computer Architecture (ISCA)* (2017)
15. 25. Kaplan, J., McCandlish, S., Henighan, T., Brown, T.B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., Amodei, D.: Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361* (2020)
16. 26. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980* (2014)
17. 27. Krizhevsky, A.: Learning multiple layers of features from tiny images. Tech. rep. (2009)
18. 28. Kuznetsova, A., Rom, H., Alldrin, N., Uijlings, J., Krasin, I., Pont-Tuset, J., Kamali, S., Popov, S., Malloci, M., Duerig, T., Ferrari, V.: The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. *arXiv:1811.00982* (2018)
19. 29. van Laarhoven, T.: L2 regularization versus batch and weight normalization. *CoRR* (2017)
20. 30. Li, A., Jabri, A., Joulin, A., van der Maaten, L.: Learning visual n-grams from web data. In: *ICCV* (2017)
21. 31. Li, X., Grandvalet, Y., Davoine, F.: Explicit inductive bias for transfer learning with convolutional networks. In: *ICML* (2018)1. 32. Li, Z., Arora, S.: An exponential learning rate schedule for deep learning. arXiv preprint arXiv:1910.07454 (2019)
2. 33. Lin, T.Y., Goyal, P., Girshick, R., He, K., Dollár, P.: Focal loss for dense object detection. In: ICCV (2017)
3. 34. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft COCO: Common objects in context. In: ECCV (2014)
4. 35. Loshchilov, I., Hutter, F.: Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983 (2016)
5. 36. Mahajan, D., Girshick, R., Ramanathan, V., He, K., Paluri, M., Li, Y., Bharambe, A., van der Maaten, L.: Exploring the limits of weakly supervised pretraining. In: ECCV (2018)
6. 37. Nakamura, A., Harada, T.: Revisiting fine-tuning for few-shot learning. arXiv preprint arXiv:1910.00216 (2019)
7. 38. Ngiam, J., Peng, D., Vasudevan, V., Kornblith, S., Le, Q.V., Pang, R.: Domain adaptive transfer learning with specialist models. arXiv:1811.07056 (2018)
8. 39. Nilsback, M.E., Zisserman, A.: Automated flower classification over a large number of classes. In: Indian Conference on Computer Vision, Graphics and Image Processing (2008)
9. 40. Pan, S.J., Yang, Q.: A survey on transfer learning. IEEE Transactions on knowledge and data engineering (2009)
10. 41. Parkhi, O.M., Vedaldi, A., Zisserman, A., Jawahar, C.V.: Cats and dogs. In: CVPR (2012)
11. 42. Peyre, J., Laptev, I., Schmid, C., Sivic, J.: Weakly-supervised learning of visual relations. CoRR **abs/1707.09472** (2017), <http://arxiv.org/abs/1707.09472>
12. 43. Qiao, S., Wang, H., Liu, C., Shen, W., Yuille, A.: Weight standardization. arXiv preprint arXiv:1903.10520 (2019)
13. 44. Raghu, M., Zhang, C., Kleinberg, J., Bengio, S.: Transfusion: Understanding transfer learning with applications to medical imaging. arXiv:1902.07208 (2019)
14. 45. Rosenfeld, J.S., Rosenfeld, A., Belinkov, Y., Shavit, N.: A constructive prediction of the generalization error across scales. In: ICLR (2020)
15. 46. Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A.C., Fei-Fei, L.: ImageNet Large Scale Visual Recognition Challenge. IJCV (2015)
16. 47. Shetty, R., Schiele, B., Fritz, M.: Not using the car to see the sidewalk: Quantifying and controlling the effects of context in classification and segmentation. CoRR **abs/1812.06707** (2018), <http://arxiv.org/abs/1812.06707>
17. 48. Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556 (2014)
18. 49. Snell, J., Swersky, K., Zemel, R.: Prototypical networks for few-shot learning. In: NIPS (2017)
19. 50. Sohn, K., Berthelot, D., Li, C.L., Zhang, Z., Carlini, N., Cubuk, E.D., Kurakin, A., Zhang, H., Raffel, C.: Fixmatch: Simplifying semi-supervised learning with consistency and confidence. arXiv preprint arXiv:2001.07685 (2020)
20. 51. Sun, C., Shrivastava, A., Singh, S., Gupta, A.: Revisiting unreasonable effectiveness of data in deep learning era. In: ICCV (2017)
21. 52. Sung, F., Yang, Y., Zhang, L., Xiang, T., Torr, P.H., Hospedales, T.M.: Learning to compare: Relation network for few-shot learning. In: CVPR (2018)
22. 53. Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., Rabinovich, A.: Going deeper with convolutions. In: CVPR (2015)
23. 54. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z.: Rethinking the inception architecture for computer vision. In: CVPR (2016)1. 55. Tan, M., Le, Q.: Efficientnet: Rethinking model scaling for convolutional neural networks. In: ICML (2019)
2. 56. Thomee, B., Shamma, D.A., Friedland, G., Elizalde, B., Ni, K., Poland, D., Borth, D., Li, L.J.: Yfcc100m: The new data in multimedia research. arXiv preprint arXiv:1503.01817 (2015)
3. 57. Touvron, H., Vedaldi, A., Douze, M., Jégou, H.: Fixing the train-test resolution discrepancy. In: NeurIPS (2019)
4. 58. Tschannen, M., Djolonga, J., Ritter, M., Mahendran, A., Houlsby, N., Gelly, S., Lucic, M.: Self-supervised learning of video-induced visual invariances (2019)
5. 59. Vinyals, O., Blundell, C., Lillicrap, T., Wierstra, D., et al.: Matching networks for one shot learning. In: NIPS (2016)
6. 60. Wu, Y., He, K.: Group normalization. In: ECCV (2018)
7. 61. Xie, Q., Hovy, E., Luong, M.T., Le, Q.V.: Self-training with noisy student improves imagenet classification. arXiv preprint arXiv:1911.04252 (2019)
8. 62. Xie, S., Girshick, R., Dollár, P., Tu, Z., He, K.: Aggregated residual transformations for deep neural networks. In: CVPR (2017)
9. 63. Yalniz, I.Z., Jégou, H., Chen, K., Paluri, M., Mahajan, D.: Billion-scale semi-supervised learning for image classification. arXiv preprint arXiv:1905.00546 (2019)
10. 64. Yun, S., Han, D., Oh, S.J., Chun, S., Choe, J., Yoo, Y.: Cutmix: Regularization strategy to train strong classifiers with localizable features. arXiv preprint arXiv:1905.04899 (2019)
11. 65. Zhai, X., Oliver, A., Kolesnikov, A., Beyer, L.: S<sup>4</sup>L: Self-Supervised Semi-Supervised Learning. In: ICCV (2019)
12. 66. Zhai, X., Puigcerver, J., Kolesnikov, A., Ruyssen, P., Riquelme, C., Lucic, M., Djolonga, J., Pinto, A.S., Neumann, M., Dosovitskiy, A., et al.: A large-scale study of representation learning with the visual task adaptation benchmark. arXiv preprint arXiv:1910.04867 (2019)
13. 67. Zhang, H., Cisse, M., Dauphin, Y.N., Lopez-Paz, D.: mixup: Beyond empirical risk minimization. In: ICLR (2017)Fig. 9: Blue curves display VTAB-1k score (mean accuracy across tasks) depending on the total number of random hyperparameters tested. Reported VTAB-1k scores are averaged over 100 random hyperparameter orderings, the shaded blue area indicates the standard error. Dashed gray line displays the performance on the small hold-out validation split with 200 examples.

## A Tuning hyperparameters for transfer

Throughout the paper we evaluate BiT using BiT-HyperRule. Here, we investigate whether BiT-L would benefit from additional computational budget for selecting fine-tuning hyperparameters.

For this investigation we use VTAB-1k as it contains a diverse set of 19 tasks. For each task we fine-tune BiT-L 40 times using 800 training images. Each trial uses randomly sampled hyperparameters as described below. We select the best model for each dataset using the validation set with 200 images. The results are shown in Figure 9. Overall, we observe that VTAB-1k score saturates roughly after 20 trials and that further tuning results in overfitting on the validation split. This indicates that practitioners do not need to do very heavy tuning in order to find optimal parameters for their task.

After re-training BiT-L model with selected hyper-parameters using all union of training and validation splits (1000 images) we obtain the VTAB-1k score of 78.72%, an absolute improvement of 2.43% over 76.29% score obtained with BiT-HyperRule.

Our random search includes following hyperparameters with the following ranges and sampling strategies:

- – Initial learning rate is sampled log-uniformly from the range  $[10^{-1}, 10^{-4}]$ .
- – Total number of updates is sampled from the set  $\{500, 1000, 2000, 4000, 8000, 16000\}$ .
- – Dropout rate for the penultimate layer is uniformly sampled from the range  $[0.0, 0.7]$ .
- – Weight decay to the initial weight values is sampled log-uniformly from the range  $[10^{-1}, 10^{-6}]$ .
- – MixUp  $\alpha$  parameter is sampled from the set  $\{\text{None}, 0.05, 0.1, 0.2, 0.4\}$ .
- – Input image resolution is sampled from the set  $\{64, 128, 192, 256, 320, 384\}$ .## B Full ObjectNet results

Figure 10 shows more results on the ObjectNet test set, with top-5 accuracy reported on the left and top-1 accuracy on the right. In Figure 10 (a), we first resize the shorter side of the image to 512 pixels and then take a  $480 \times 480$  pixel sized central crop, similar to BiT-HyperRule.

ObjectNet is a dataset collected in the real world, where multiple objects are present most of the time. A recent analysis shows that cropping out a single object from the cluttered scene could significantly improve performance [6]. In Figure 10 (b), we follow the same setup and report our models’ performance on the cropped ObjectNet with a single object in each crop. We observe a solid improvement in performance in this setting.

Overall, the trend of our improvements is consistent with the results on the original ObjectNet test set. We provide our full numeric results in Table 6.

(a) Results on the original ObjectNet test set with resize and central crop.

(b) Results on the cropped ObjectNet, where individual objects are cropped for evaluation. The bounding boxes are provided by [6].

Fig. 10: Results on ObjectNet; left: top-5 accuracy, right: top-1 accuracy.Table 6: Results (%) on the ObjectNet test set. We report numbers for both the standard setting, as well as for the setting where the ground-truth bounding box is used.

<table border="1">
<thead>
<tr>
<th rowspan="3">BiT-</th>
<th colspan="6">Top-1 accuracy</th>
<th colspan="6">Top-5 accuracy</th>
</tr>
<tr>
<th colspan="3">Resize &amp; Crop</th>
<th colspan="3">Bounding Box</th>
<th colspan="3">Resize &amp; Crop</th>
<th colspan="3">Bounding Box</th>
</tr>
<tr>
<th>S</th>
<th>M</th>
<th>L</th>
<th>S</th>
<th>M</th>
<th>L</th>
<th>S</th>
<th>M</th>
<th>L</th>
<th>S</th>
<th>M</th>
<th>L</th>
</tr>
</thead>
<tbody>
<tr>
<td>R50x1</td>
<td>30.8</td>
<td>35.0</td>
<td>37.6</td>
<td>35.1</td>
<td>41.6</td>
<td>42.5</td>
<td>51.8</td>
<td>56.4</td>
<td>59.5</td>
<td>58.7</td>
<td>64.9</td>
<td>66.0</td>
</tr>
<tr>
<td>R101x1</td>
<td>32.2</td>
<td>39.2</td>
<td>54.6</td>
<td>37.4</td>
<td>46.1</td>
<td>49.1</td>
<td>54.2</td>
<td>61.3</td>
<td>75.6</td>
<td>61.1</td>
<td>69.4</td>
<td>72.4</td>
</tr>
<tr>
<td>R50x3</td>
<td>33.7</td>
<td>40.3</td>
<td>49.1</td>
<td>38.4</td>
<td>46.2</td>
<td>54.7</td>
<td>54.7</td>
<td>62.4</td>
<td>71.1</td>
<td>61.5</td>
<td>70.1</td>
<td>77.5</td>
</tr>
<tr>
<td>R101x3</td>
<td>34.6</td>
<td>44.3</td>
<td>54.6</td>
<td>40.2</td>
<td>50.5</td>
<td>60.4</td>
<td>56.4</td>
<td>66.4</td>
<td>75.6</td>
<td>63.4</td>
<td>73.6</td>
<td>82.5</td>
</tr>
<tr>
<td>R152x4</td>
<td>36.0</td>
<td>47.0</td>
<td><b>58.7</b></td>
<td>41.6</td>
<td>52.8</td>
<td><b>63.8</b></td>
<td>57.0</td>
<td>69.0</td>
<td><b>80.0</b></td>
<td>64.4</td>
<td>76.0</td>
<td><b>85.1</b></td>
</tr>
</tbody>
</table>Table 7: Performance of BiT-L on the original (“Full”) and deduplicated (“Dedup”) test data. The “Dups” column shows the total number of near-duplicates found.

<table border="1">
<thead>
<tr>
<th rowspan="2"></th>
<th colspan="3">From JFT</th>
<th colspan="3">From ImageNet21k</th>
<th colspan="3">From ILSVRC-2012</th>
</tr>
<tr>
<th>Full</th>
<th>Dedup</th>
<th>Dups</th>
<th>Full</th>
<th>Dedup</th>
<th>Dups</th>
<th>Full</th>
<th>Dedup</th>
<th>Dups</th>
</tr>
</thead>
<tbody>
<tr>
<td>ILSVRC-2012</td>
<td>87.8</td>
<td>87.9</td>
<td>6470</td>
<td>84.5</td>
<td>85.3</td>
<td>3834</td>
<td>80.3</td>
<td>81.3</td>
<td>879</td>
</tr>
<tr>
<td>CIFAR-10</td>
<td>99.4</td>
<td>99.3</td>
<td>435</td>
<td>98.5</td>
<td>98.4</td>
<td>687</td>
<td>97.2</td>
<td>97.2</td>
<td>82</td>
</tr>
<tr>
<td>CIFAR-100</td>
<td>93.6</td>
<td>93.4</td>
<td>491</td>
<td>91.2</td>
<td>90.7</td>
<td>890</td>
<td>85.3</td>
<td>85.2</td>
<td>136</td>
</tr>
<tr>
<td>Pets</td>
<td>96.8</td>
<td>96.4</td>
<td>600</td>
<td>94.6</td>
<td>94.5</td>
<td>80</td>
<td>93.7</td>
<td>93.6</td>
<td>58</td>
</tr>
<tr>
<td>Flowers</td>
<td>99.7</td>
<td>99.7</td>
<td>412</td>
<td>99.5</td>
<td>99.5</td>
<td>335</td>
<td>91.0</td>
<td>91.0</td>
<td>0</td>
</tr>
</tbody>
</table>

## C Duplicates and near-duplicates

In order to make sure that our results are not inflated due to overlap between upstream training and downstream test data, we run extensive de-duplication experiments. For training our flagship model, BiT-L, we remove all images from JFT-300M dataset that are duplicates and near-duplicates of test images of all our downstream datasets. In total, we removed less than 50k images from the JFT-300M dataset. Interestingly, we did not observe any drastic difference by doing de-duplication, evidenced by comparing the first column of Table 1 (de-duplicated upstream) and the first column of Table 7 (full upstream).

In another realistic setting, eventual downstream tasks are not known in advance. To better understand this setting, we also investigate how duplicates affect performance by removing them from the downstream test data after the upstream model has already been trained. The results of this experiment are shown in Table 7: “Full” is the accuracy on the original test set that contains near-duplicates, “Dedup” is the accuracy on the test set cleaned of near-duplicates, and “Dups” is the number of near-duplicates that have been removed from said test set. We observe that near-duplicates barely affect the results in all of our experiments. Note that near-duplicates between training and test sets have previously been reported by [51] for ILSVRC-2012, and by [3] for CIFAR.

In Figure 11, we present a few duplicates found between the ILSVRC-2012 training set and test splits of four standard downstream datasets.Fig. 11: Detected duplicates between the ILSVRC-2012 training set and test splits of various downstream datasets. Note that Flowers is not listed because there are no duplicates. Green borders mark true positives and red borders mark (rare) false positives.## D All of BiT-L’s Mistakes

Here we take a closer look at the mistakes made by BiT-L<sup>2</sup>. Figure 8, we show *all* mistakes on CIFAR-10, as well as a representative selection of mistakes on ILSVRC-2012. Figures 12 and 13 again show *all* mistakes on the Pets and Flowers datasets, respectively. The first word always represents the model’s prediction, while the second word represents the ground-truth label. The larger panels are best viewed on screen, where they can be magnified.

Fig. 12: All of BiT-L’s mistakes on Oxford-IIIT-Pet.

<sup>2</sup> To be precise, the figures are obtained by an earlier version of our BiT-L model but which reaches almost the same accuracy. We did not re-run the figures and human evaluation with the latest model as they serve for illustration purposes and the models perform essentially the same, modulo a few flips.Fig. 13: All of BiT-L’s mistakes on Oxford-Flowers102.

## E Object detection experiments

As discussed in the main text, for object detection evaluation we use the RetinaNet model [33]. Our implementation is based on publicly available code<sup>3</sup> and uses standard hyper-parameters for training all detection models. We repeat training 5 times and report median performance.

Specifically, we train all of our models for 30 epochs using a batch size of 256 with stochastic gradient descent, 0.08 initial learning rate, 0.9 momentum and  $10^{-4}$  weight decay. We decrease the initial learning rate by a factor of 10 at epochs number 16 and 22. We did try training for longer (60 epochs) and did not observe performance improvements. The input image resolution is  $1024 \times 1024$ . During training we use a data augmentation scheme as in [34]: random horizontal image flips and scale jittering. We set the classification loss parameters  $\alpha$  to 0.25 and  $\gamma$  to 2.0, see [33] for the explanation of these parameters.

## F Horizontal flipping and cropping for VTAB-1k tasks

When fine-tuning BiT models, we apply random horizontal flipping and cropping as image augmentations. However, these operations are not reasonable for certain VTAB tasks, where the semantic label (e.g. angle, location or object count) is not invariant to these operations.

Thus, we disable random horizontal flipping as preprocessing for dSprites-orientation, SmallNORB-azimuth and dSprites-location tasks. Random cropping preprocessing is disabled for Clevr-count, Clevr-distance, DMLab, KITTI-distance and dSprites-location tasks.

<sup>3</sup> <https://github.com/tensorflow/tpu/tree/master/models/official/retinanet>Fig. 14: **Left:** Top 5 predictions produced by an ILSVRC-2012 model (IN-R50) and BiT-L on an example out-of-context object. Bar lengths indicate predictive probability on a log scale. **Right:** Top-1 accuracy on the ILSVRC-2012 validation set plotted against top-1 accuracy on objects out-of-context. The legend indicates the pre-training data. All models are subsequently fine-tuned on ILSVRC-2012 with BiT-HyperRule. Larger markers size indicates larger architectures, as in Fig. 5.

## G Robustness: Objects out-of-context

It has been shown that CNNs can lack robustness when classifying objects out-of-context [4, 42, 47]. We investigate whether BiT not only improves classification accuracy, but also out-of-context robustness. For this, we create a dataset of foreground objects corresponding to ILSVRC-2012 classes pasted onto miscellaneous backgrounds (Fig. 14 left). We obtain images of foreground objects using OpenImages-v5 [28] segmentation masks. Figure 14 shows an example, and more are given in Figure 15. Sometime foreground objects are partially occluded, resulting in an additional challenge.

We transfer BiT models pre-trained on various datasets to ILSVRC-2012 and see how they perform on this out-of-context dataset. In Figure 14 we can see that the performance of models pre-trained on ILSVRC-2012 saturates on the out-of-context dataset, whereas by using more data during pre-training of larger models, better performance on ILSVRC-2012 *does* translate to better out-of-context performance.

More qualitatively, when we look at the predictions of the models on out-of-context data, we observe a tendency for BiT-L to confidently classify the foreground object regardless of the context, while ILSVRC-2012 models also predict objects absent from the image, but that could plausibly appear with the background. An example of this is shown in Figure 14 left.

### G.1 Out of context dataset details

We generate this dataset by combining foreground objects extracted from OpenImages V5 [28] with backgrounds, licensed for reuse with modification, mined from search engine results.**Foreground objects** In this study, we evaluate models that output predictions over ILSVRC-2012 classes. We therefore fine-tune BiT models on ILSVRC-2012 using BiT-HyperRule. We choose 20 classes from OpenImages that correspond to one such class or a subset thereof. These 20 classes cover a spectrum of different object types. We then extract foreground objects that belong to these classes from images in OpenImages using the provided segmentation masks. Note that this leads to some objects being partially occluded; however, humans can still easily recognize the objects, and we would like the same from our models.

**Backgrounds** We define a list of 41 backgrounds that cover a range of contexts such that (1) we have reasonable diversity, and (2) the objects we choose would not likely be seen in some of these backgrounds. We then collect a few examples of each background using a search engine, limiting to results licensed for reuse with modification. We take the largest square crop of the background from the top left corner.

We paste the extracted foreground objects onto the backgrounds. This results in a total of 3321 images in our dataset (81 foreground objects  $\times$  41 backgrounds). We fix the size of the objects such that the longest side corresponds to 80% of the width of the background; thus, the object is prominent in the image.

Figure 15 shows more examples of out-of-context images from our dataset, contrasting the predictions given by a standard ResNet50 trained on ILSVRC-2012 from scratch and the predictions of BiT-L fine-tuned on ILSVRC-2012.

## G.2 Image Attributions

In this section we provide attributions for images used to generate the examples from the out-of-context dataset.

All images are licensed CC-BY-2.0 unless noted otherwise.

### Foreground objects:

- – Traffic light: [U turn to Tophane](#) by *Istanbul Photo Guide*.
- – Sofa: [Welcome](#) by *woot*.
- – Zebra: [i like his tail in this one](#) by *meg and rahul*.
- – Starfish: [Starfish](#) by *Summer Skyes 11*.
- – Limousine: [Hummer limousine stopping at the door](#) [nb: title translated] by *duncan\_su*.

### Backgrounds:

- – Grass: [Photo](#) by *zoosnow*  
  (Pexels license; Free to use, no attribution required).
- – Wood: [Tree Bark Texture 04](#) by *Jacob Gube, SixRevisions*.
- – Street at night: [City street calm buildings](#) by *csr\_ch*  
  (Pixabay license; Free for commercial use, no attribution required).
- – Underwater: [Photo](#) by *MaxX42*  
  (Pixabay license; Free for commercial use, no attribution required).
- – Kitchen: [Interior of a modern modular home](#) by *Riverview Homes, Inc.*  
  (CC-BY-SA-3.0 Unported license).Fig. 15: Top 5 predictions produced by an ILSVRC-2012 model (INet-R50) and BiT-L on examples of out-of-context objects. Bar lengths indicate predicted probability on a log scale. We choose images that highlight the qualitative differences between INet-R50 and BiT-L predictions when the INet-R50 model makes mistakes.

