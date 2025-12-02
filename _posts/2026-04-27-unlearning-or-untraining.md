---
layout: distill
title: Is your algorithm Unlearning or Untraining?
description: >-
  Machine unlearning aims to post-process a trained model in order to remove the influence of specific training examples or higher-level knowledge. We argue that the term unlearning is overloaded, with different use cases belonging to distinct problem formulations. This issue causes confusion in the community: it is often unclear what the goals of different proposed methods are, when they are expected to work, how they should be evaluated, and what baselines they should be compared against. To address this, we establish a fundamental distinction between two notions that we identify as Unlearning and Untraining, aiming to guide the field towards disambiguating technical definitions, to unlock more progress in clarifying goals, designing evaluation metrics for each, and ultimately better algorithms.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

# Anonymize when submitting
authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2026-04-27-unlearning-or-untraining.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Training and learning
  - name: Memorization
  - name: 'Machine "unlearning": the classic definition'
  - name: Distinguishing Un-training from Un-learning
  - name: "'Unlearning' of Definition 2 is actually Untraining"
  - name: Defining Unlearning
  - name: Mapping Untraining and Unlearning to the literature
  - name: Limitations
  - name: Conclusion and outlook

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

$$
\newcommand{\alg}{\mathcal{A}}
\newcommand{\unlearn}{\mathcal{U}}
\newcommand{\loss}{\ell}
\newcommand{\dataset}{\mathcal{D}}
\newcommand{\forgetset}{\mathcal{S}}
\newcommand{\retainset}{\mathcal{D} \setminus \mathcal{S}}
$$

<!-- Note: please use the table of contents as defined in the front matter rather than the traditional markdown styling. -->


## Introduction
"Unlearning" was first coined by <d-cite key="cao2015towards"></d-cite>, who envisioned systems that are "capable of forgetting certain data and their lineages, completely and quickly". Since then, there has been an explosion of work on the topic. Most early work on "unlearning" was motivated by the need to enable users to revoke access to their personal data that may have already been used to train machine learning models <d-cite key="neel2021descent,sekhari2021remember,golatkar2020eternal,bourtoule2021machine,golatkar2020forgetting,thudi2022unrolling"></d-cite>. However, more recently, several works propose methods that use "unlearning" for a wider range of use cases, including removing dangerous "knowledge" that could e.g. aid a malicious actor to develop biological, cyber, and chemical weapons <d-cite key="li2024wmdp"></d-cite>, removing harmful "capabilities" or "concepts" to make models safer <d-cite key="liu2024towards,yao2024large,lucki2024adversarial,barez2025open,zhang2024forget,lynch2024eight,fan2023salun"></d-cite>, erasing backdoors <d-cite key="liu2022backdoor"></d-cite>, eliminating poisoning attacks <d-cite key="schoepf2024potion,schoepf2025redirection,pawelczyk2024machine"></d-cite>, or unlearning copyrighted content like the "Harry Potter" books <d-cite key="eldan2023s,shi2024muse"></d-cite> or specific artistic styles <d-cite key="zhang2024unlearncanvas,fan2023salun"></d-cite>.

In the following sections, we will review important background and define key notions of training, learning, and memorization. We will then make a novel distinction between two notions that we refer to as *Un*&thinsp;**learning** and *Un*&thinsp;**training**, establishing appropriate vocabulary for distinguishing the problem settings that correspond to different use cases studied in the literature. We will provide a technical definition for each of *Un*&thinsp;**learning** and *Un*&thinsp;**training**, drawing from previously-proposed notions from the literature for the latter. We will discuss how the solutions to these two problem settings differ through illustrative examples, and we will close by discussing important research questions that we hope future work pursues. 

## Training and learning

To set the scene for discussing the distinction between *Un*&thinsp;**training** and *Un*&thinsp;**learning**, lets first revisit the definitions of training and learning, and related concepts of generalization and memorization.

**Training**, from a statistical learning theory perspective <d-cite key="vapnik2013nature"></d-cite>, can be defined as the process of obtaining a function that minimizes empirical risk on a finite dataset $\mathcal{D}$, which generally involves solving an optimization problem. For parametric functions like neural networks, the goal of training algorithm $\mathcal{A}$ is to find a set of parameters $\theta$ that minimizes the error on a specific finite set of data $\mathcal{D}$.

**Learning**, on the other hand, involves reducing expected risk on the underlying data distribution via training i.e., reducing the empirical risk on a finite dataset $\mathcal{D}$. In this sense, learning is synonymous with generalization -- the ability of making correct predictions on the underlying distribution of data, including unseen data, rather than just $\mathcal{D}$.

**Training does not always lead to Learning.**
In fact, there is a vast literature exploring the design of training algorithms that optimally result in learning, e.g., optimization techniques like sharpness aware minimization <d-cite key="foret2021sharpnessaware"></d-cite> that use the loss landscape geometry to reach a flatter minima, techniques like MixUp <d-cite key="zhang2017mixup"></d-cite> that improve generalization by augmenting the training data with convex combinations of samples, and regularization techniques like spectral normalization of parameters <d-cite key="miyato2018spectral"></d-cite> that improves Lipschitz continuity of the learned function.

Recent work, has demonstrated that training can even lead to risk minimization only on the training data $\mathcal{D}$ without learning <d-cite key="power2022grokking,liu2023omnigrok,humayun2024grok"></d-cite>. This behavior occurs alongside a training dynamics phenomenon termed *grokking*, where deep networks tend to **memorize** the labels for the training data but perform poorly for a held out dataset from the underlying data distribution---until a large number of training iterations. The discussion from above points to one important finding---a finite training dataset $\mathcal{D}$ may or may not influence the behavior of the obtained function for samples that are not in $\mathcal{D}$.
Furthermore, as we discuss in the next section, different examples influence the model differently during training and have different consequences for learning.

## Memorization
The way in which different examples influence the function obtained via training, remains an active area of research <d-cite key="jaeckel1972infinitesimal,koh2017understanding,pruthi2020estimating"></d-cite>. A phenomenon that has been studied extensively is the fact that training can lead to **memorization** of certain examples from $\mathcal{D}$. <d-cite key="feldman2020does,zhang2023counterfactual"></d-cite> define a notion of counterfactual memorization, where the memorization score for a training example is given as follows.

> **Definition 1 (Memorization score).** <d-cite key="feldman2020does"></d-cite>
> <a name="defn:mem"></a>
> The *memorization score* for an example $i \in \dataset$, with respect to a training dataset $\dataset$ and training algorithm $\alg$ is
>
> $$
> \text{mem}(\alg, \dataset, i) = \Pr_{f \sim \alg(\dataset)}[f(x_i) = y_i]  \ - \Pr_{f \sim \alg(\dataset \setminus i)}[f(x_i) = y_i]
> $$
>
> where $x_i$ and $y_i$ are the feature and label, respectively, of example $i$.

The first term in the above equation considers models trained on all of $\dataset$ whereas the second term considers models trained on $\dataset$ excluding example $i$. Intuitively, the memorization score for an example $i$ is high if including it in training results in a different distribution of predictions on that example compared to excluding it from training. <d-footnote>While the above definition assumes a classification problem, a more general notion of <em>counterfactual memorization</em> has been proposed in <d-cite key="zhang2023counterfactual"></d-cite> that can be applied to Large Language Models (LLMs) or other generative models. All arguments made in this blog post hold for any type of architecture and training algorithm.</d-footnote>

Recent works <d-cite key="feldman2020does,feldman2020neural,jiang2020characterizing"></d-cite> find that outliers or noisy, e.g. mislabeled, data points are more highly memorized, since these are examples that would not have been predicted correctly unless they were part of the training dataset. For example, in a classification task, imagine a training example that is an image of a cat being labeled as a chair. During training, the model can fit this strange data point and predict its assigned label of chair. However, had this example been excluded from the training set, the model would not predict that this cat image is a chair. This discrepancy, between the prediction on this data point of models that included it in training, compared to models that excluded it from training, leads to this data point being highly memorized according to Definition 1.

Generally, the interaction between memorization and learning is an important active area of research, with theory works advocating that some memorization is even necessary for learning <d-cite key="feldman2020does,attias2024information"></d-cite>.
However, while memorization is sometimes needed, it is also sometimes unwanted, as it may cause vulnerability to membership inference attacks or data extraction attacks <d-cite key="shokri2017membership,carlini2022membership,carlini2021extracting"></d-cite>.

## Machine "unlearning": the classic definition
Before we introduce the distinction between *Un*&thinsp;**learning** and *Un*&thinsp;**training**, let us first describe the classic problem formulation of "unlearning", which was motivated by the need to remove memorized examples or to enable users to request their data to be deleted from machine learning models.

Let $\alg(\dataset)$ denote the weights of a model obtained by applying learning algorithm $\alg$ on dataset $\dataset$; we refer to this as the "original model". Informally, according to the classic definition of machine unlearning, the goal is to remove the influence of a forget set $\forgetset \subset \dataset$ from the weights of the original model.

A straightforward solution for this problem is to simply retrain a model from scratch on an adjusted training set that excludes $\forgetset$, referred to as the "retain set". The ideal solution is therefore $\alg(\retainset)$.
However, retraining from scratch is inefficient, especially for larger models. To address this, the goal of unlearning is to avoid throwing away the original model and instead devise an efficient algorithm $\unlearn$ that can post-process it to produce an unlearned model  $\unlearn(\alg(\dataset), \forgetset, \dataset)$ that approximates the ideal solution of having trained from scratch.

Variations of technical definitions have been proposed that formalize this intuition <d-cite key="sekhari2021remember,gupta2021adaptive,neel2021descent"></d-cite>, drawing inspiration from differential privacy <d-cite key="dwork2006differential"></d-cite>.

> **Definition 2 (($\varepsilon$, $\delta$)-unlearning).** <d-cite key="neel2021descent"></d-cite>
> <a name="defn:classic_unlearning"></a>
> For a fixed randomized learning algorithm $\alg$, an unlearning algorithm $\unlearn$ is $(\varepsilon,\delta)$-unlearning with respect to $\alg$ if for any dataset $\dataset$, forget set $\forgetset \subset \dataset$, it holds that for all $R \subseteq \mathcal{R}$ we have:
>
> $$
> \begin{aligned}
> \Pr[\alg(\retainset) \in R] &\le e^\varepsilon \Pr[\unlearn(\alg(\dataset), \forgetset, \dataset) \in R] + \delta,    \quad \mathrm{and} \\
> \Pr[\unlearn(\alg(\dataset), \forgetset, \dataset) \in R]  &\le e^\varepsilon \Pr[\alg(\retainset) \in R] + \delta.
> \end{aligned}
> $$
>
> where $\mathcal{R}$ denotes the output space, in this case, the space of model parameters.

Measuring success of unlearning according to this definition requires estimating how close two distributions are to one another: the distribution of $\unlearn(\alg(\dataset), \forgetset, \dataset)$ and that of $\alg(\retainset)$.  We refer to distributions here since running $\alg$ and $\unlearn$ with different random seeds that control, for instance, the initialization and order of mini-batches, will yield slightly different model weights each time. <d-cite key="triantafillou2024we,hayes2024inexact,kurmanji2024towards,pawelczyk2023context"></d-cite> discuss this issue of evaluation in more depth and propose rigorous evaluation procedures for this definition.

We will later argue that this classic definition of unlearning is better described as *Un*&thinsp;**training**. We will contrast it below with a different definition of the problem, which we will refer to as *Un*&thinsp;**learning**.


## Distinguishing *Un*&thinsp;**training** from *Un*&thinsp;**learning**

In this section, we present our main contribution: disentangling two problem formulations that are currently both referred to as "unlearning" in the literature.

* As **training** is about minimizing empirical risk on a finite dataset $\dataset$, *Un*&thinsp;**training** on $\forgetset \subset \dataset$ is the process of reversing the empirical risk minimization on $\forgetset$. In other words, the ideal solution to *Un*&thinsp;**training** is to find the model parameters minimizing empirical risk on only $\retainset$.

* On the other hand, **learning** and *Un*&thinsp;**learning** are both about generalization (inducing it, and removing it, respectively). *Un*&thinsp;**learning** a pattern, "concept" or "behaviour" from a given representative forget set $\forgetset$ is to generalize the removal beyond the specific examples in $\forgetset$, to approximate a model that was never trained on *any* instance of the pattern.

Notice that *Un*&thinsp;**training** a forget set $\forgetset$ does not mean that the model is unable to predict the examples of $\forgetset$ correctly. A model trained purely on $\retainset$ may still be able to predict the examples of $\forgetset$ correctly due to generalization. On the other hand, *Un*&thinsp;**learning** a pattern, concept or "behaviour" from a forget set $\forgetset$ would make the model unable to predict the examples of $\forgetset$ any better than a model that never trained on *any* instances of the concept (i.e. a model that never learned the concept).

We illustrate the difference between *Un*&thinsp;**training** versus *Un*&thinsp;**learning**, for a given forget set $\forgetset$ in Figure 1.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-unlearning-or-untraining/figure1.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <strong>Figure 1. Illustration of the difference between <em>Un</em>&thinsp;<strong>learning</strong> and <em>Un</em>&thinsp;<strong>training</strong>, using a given forget set $\forgetset$.</strong>
    The different shapes correspond to data points in a dataset $\dataset$. A shape is coloured white if the model predicts it correctly and black otherwise.
    In the middle of the figure, we depict the predictions of the original model that was trained on all of $\dataset$, before any <em>Un</em>&thinsp;<strong>learning</strong> or <em>Un</em>&thinsp;<strong>training</strong> is carried out. This model predicts correctly on all of its training set.
    On the left, we show the predictions of the model obtained by <em>Un</em>&thinsp;<strong>training</strong> $\forgetset$, which matches the behaviour of a model trained on $\dataset \setminus \forgetset$  -- i.e. a model trained on all of the data points except for the star and the two circles that belong to the forget set $\forgetset$.
    Notice that the untrained model still predicts the circles correctly. This is because there are several other circles in the remaining dataset from where the model trained on $\dataset \setminus \forgetset$ can learn about circles. On the other hand, the star that was in the forget set is no longer predicted correctly, as there are fewer other stars in the dataset.
    On the right, we show the predictions of the model obtained by <em>Un</em>&thinsp;<strong>learning</strong> the "behaviour" underlying $\forgetset$, where the "behaviour" in this case is "being a circle or a star". The unlearned model makes incorrect predictions on <em>all</em> examples of that behaviour.
</div>

## "Unlearning" of Definition 2 is actually *Un*&thinsp;**training**
Let's take a closer look at Definition 2. According to this definition, the ideal unlearning algorithm produces a model that is indistinguishable (in distribution) from one retrained from scratch on the retain set $\mathcal{D} \setminus \mathcal{S}$. Notably, this means that it is not the case that the ideal "unlearned" model is unable to predict correctly on the forget set (nor on examples that are similar to the forget set).

In other words, "unlearning", according to this definition, is not about removing all knowledge about the forget set (and related examples); it's about removing only the *additional* knowledge about the forget set that existed only *due to having trained on the forget set*.

Let's take as an example the case where $\mathcal{S}$ consists of non-memorized examples (more precisely, examples with low "memorization scores" according to Definition 1). We assume the original model $\mathcal{A}(\mathcal{D})$ predicts the examples of $\mathcal{S}$ correctly; a reasonable assumption given $\mathcal{S} \subset \mathcal{D}$. Now, the fact that the examples in $\mathcal{S}$ are not memorized means that the retrained-from-scratch model $\mathcal{A}(\mathcal{D \setminus S})$ also predicts the examples of $\mathcal{S}$ correctly. This means that the classic definition of "unlearning" (where retrain-from-scratch is the ideal "unlearning" algorithm), wants the "unlearned" model to still predict the examples of $\mathcal{S}$ correctly, effectively necessitating no change over the original model. <d-footnote>This observation has been previously made by <d-cite key="zhao2024makes"></d-cite>, who also studies how existing unlearning algorithms perform on forget sets of different degrees of memorization.</d-footnote><d-footnote>Note a nuance about a mismatch between the unlearning definition referring to model weights whereas the memorization definition referring to model outputs. This is to comply with prior work that defines these notions; we leave it to future work to address this minor inconsistency.</d-footnote> An example of this phenomenon is given by the circles that are present in $\forgetset$ in Figure 1: the ideal *Un*&thinsp;**training** solution is still able to predict circles correctly.
In the next section, we will discuss a different problem formulation, that we will refer to as *Un*&thinsp;**learning**, where this will no longer be the case.

## Defining *Un*&thinsp;**learning**

We now define our notion of *Un*&thinsp;**learning** a "concept" or "behaviour" from a trained model. Unlike *Un*&thinsp;**training** where the goal is simply to remove the influence that the specific forget set had on the model, here we aim to generalize the removal beyond the given forget set, towards entirely removing the concept or behaviour that the forget set represents.
We sketch a definition of this below.

> **Definition 3 (*Un*&thinsp;**learning**).**
> <a name="defn:actual_unlearning"></a>
> For a dataset $\dataset$, let $\forgetset^{full} \subset \dataset$ be the complete set of examples of $\dataset$ that capture a behaviour $\mathcal{B}$. Let $\forgetset \subseteq \forgetset^{full}$ denote the forget set that contains some examples of $\mathcal{B}$.
> Then, for a fixed randomized learning algorithm $\alg$, an *Un*&thinsp;**learning** algorithm $\unlearn$ is one such that
> $\unlearn(\alg(\dataset), \forgetset, \dataset)$ is indistinguishable (in distribution) from  $\alg(\dataset \setminus \forgetset^{full})$.

We make the following remarks about this definition.

1. Indistinguishability can be defined in different ways, e.g. through the Hockey-stick divergence, similar to Definition 2, but other divergences are also possible. We purposefully keep this abstract in this blog post.
2. This is a conceptual definition to illustrate the notion of *Un*&thinsp;**learning** (contrasting it to the notion of *Un*&thinsp;**training**), but in practice we may not be able to specify the set $\forgetset^{full}$ containing every instantiation of a behaviour in a given dataset.
3. When $\forgetset$ is very small, relative to $\forgetset^{full}$, this may be referred to as "few-shot" *Un*&thinsp;**learning** <d-cite key="yoon2024few,de2024unlearning"></d-cite>.
4. When $\forgetset$ includes all training instances of $\mathcal{B}$, the *Un*&thinsp;**training** $\forgetset$ and *Un*&thinsp;**learning** $\mathcal{B}$ are one and the same.

Notice that the definition of *Un*&thinsp;**learning** has an element of generalization: from the specific forget set $\forgetset$ showcasing a "behaviour" $\mathcal{B}$, *Un*&thinsp;**learning** *generalizes* to removing *all* knowledge of $\mathcal{B}$, beyond $\forgetset$; analogously to how learning from a specific set of examples also leads to generalization in the sense of acquiring broader knowledge beyond the given set of examples. This is the key aspect that distinguishes *Un*&thinsp;**learning** from *Un*&thinsp;**training**. And this generalization of knowledge removal is crucial for removing unwanted behaviours from models, as in practice it's not possible to specify every possible instantiation of a behaviour $\mathcal{B}$ in $\mathcal{S}$.

Let's now discuss how, for the same forget set $\mathcal{S}$, the solutions of *Un*&thinsp;**training** and *Un*&thinsp;**learning** will (in general) be different.
Let's revisit our scenario from before where $\forgetset$ contains only non-memorized examples of the target "behaviour" $\mathcal{B}$. In that case, the solution to *Un*&thinsp;**training** would be to not take any action, leaving the model as is.
On the other hand, the solution to *Un*&thinsp;**learning** would be quite different, as a large modification would be required in this case to approximate a model $\alg(\forgetset^{full})$ that never trained on *any* example of the behaviour that $\forgetset$ represents.

## Mapping *Un*&thinsp;**training** and *Un*&thinsp;**learning** to the literature
**Examples of *Un*&thinsp;**training**.**
The most prominent example, which was a key motivation behind several "unlearning" (more accurately, *Un*&thinsp;**training**) works, is that of protecting user privacy, by honoring users' requests to delete their data from models <d-cite key="neel2021descent,sekhari2021remember,golatkar2020eternal,bourtoule2021machine,golatkar2020forgetting,thudi2022unrolling"></d-cite>.
Another example is removing data points that are mislabeled, outdated, or noisy <d-cite key="kurmanji2024towards,goel2022towards"></d-cite>.
Finally, "unlearning" (more accurately *Un*&thinsp;**training**) specific data points from LLMs <d-cite key="jang2023knowledge,barbulescu2024each"></d-cite> or from diffusion models <d-cite key="alberti2025data"></d-cite> are also applications that fall into this category, that may be useful to address either privacy or copyright issues.

**Examples of *Un*&thinsp;**learning**.**
There are several recent examples of *Un*&thinsp;**learning** in the literature, including *Un*&thinsp;**learning** dangerous knowledge or capabilities <d-cite key="li2024wmdp,liu2024towards,lynch2024eight"></d-cite> <d-footnote>Recent research in LLMs is conducted on top of large pretrained models, where we don't have control of the training set and we don't have knowledge of which training examples gave rise to different "behaviours". It's possible that the forget sets used aren't even part of the training set. We can accordingly also broaden [Definition 3](#defn:actual_unlearning) to consider $\forgetset$ that isn't necessarily part of $\mathcal{D}$.</d-footnote>, erasing backdoors <d-cite key="liu2022backdoor"></d-cite>, or concepts like "not safe for work" <d-cite key="fan2023salun,zhang2024forget"></d-cite>.
The notion of "corrective unlearning" <d-cite key="goel2024corrective"></d-cite> is also a type of *Un*&thinsp;**learning**: the goal is to remove a "corruption" or poison from partial discovery of the training data that cause the corruption or poison. This topic is enjoying increasing attention recently <d-cite key="schoepf2024potion,schoepf2025redirection"></d-cite>. Similarly, *Un*&thinsp;**learning** an "artistic style" from some examples of that style is falls into this category <d-cite key="fan2023salun,zhang2024unlearncanvas"></d-cite>. Finally, *Un*&thinsp;**learning** of "classes" fits into this category too, e.g. <d-cite key="golatkar2020eternal,kurmanji2024towards,shah2023unlearning"></d-cite>. <d-footnote>While most works do class unlearning via *Un*&thinsp;**training**, since on small benchmark datasets the set $\forgetset^{full}$ of examples belonging to a target class is known. But a compelling alternative formulation that is perhaps more realistic would be *Un*&thinsp;**learning** of a class via a strict subset of its examples.</d-footnote>

These are non-exhaustive lists to illustrate how use cases studied in the literature fit within our framework. We invite researchers to reflect on whether their work falls under *Un*&thinsp;**learning** or *Un*&thinsp;**training**.

## Limitations
Our goal in this blog post is not a complete taxonomy of unlearning algorithms; we refer the reader to existing surveys for this <d-cite key="nguyen2022survey"></d-cite>. Similarly, we don't attempt to discuss potential failure modes of unlearning methodology for different use cases <d-cite key="cooper2024machine,shumailov2024ununlearning"></d-cite>. Instead, our contribution is to establish one important, yet previously overlooked, axis that differentiates unlearning problems from one other: the fundamental distinction between *Un*&thinsp;**learning** and *Un*&thinsp;**training**. Other dimensions discussed in prior work, such as differentiating "deleting" knowledge from "suppressing" knowledge <d-cite key="hu2024unlearning,deeb2024unlearning,siddiqui2025dormant,che2025model"></d-cite>, are orthogonal to our definitions.

We also note that not all possible "concepts" can be unlearned using our definition of *Un*&thinsp;**learning**. The concepts or behaviours that are in scope are those that can be specified through a set of examples $\forgetset$. Other concepts that don't have that property, such as, for example, the concept of edge detection, necessitate different definitions and are out of scope of this work.


## Conclusion and outlook
We have argued that the term "unlearning" has been overloaded, with work falling under that umbrella spanning two distinct problem formulations, that we identify as *Un*&thinsp;**learning** and *Un*&thinsp;**training**. We establish the fundamental distinction between *Un*&thinsp;**learning** and *Un*&thinsp;**training**, aiming to initiate a discussion on technical formulations of "unlearning" for different use cases, clarify their goals, and interpret the expectations and failure modes associated with existing "unlearning" algorithms.
We hope the field now *Un*&thinsp;**learns** the previous terminology and adopts our proposed conceptual framework for definitions that go beyond mere *Un*&thinsp;**training**.

This conceptual distinction gives rise to several important research questions that we hope future work pursues, including (i) Which existing "unlearning" algorithms are better suited for *Un*&thinsp;**training** compared to *Un*&thinsp;**learning**? (ii) How large does $\forgetset$ need to be relative to $\mathcal{S}^{full}$ for unlearning to succeed? In what ways does the answer to the previous question depend on the *Un*&thinsp;**learning** algorithm, and the behaviour $\mathcal{B}$? (iii) Having clarified the goals for *Un*&thinsp;**learning**, can we devise novel algorithms with appropriate inductive biases for *Un*&thinsp;**learning** a concept from few examples, perhaps drawing inspiration from algorithms developed for *learning* from few examples?
