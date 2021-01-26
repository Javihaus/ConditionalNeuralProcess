# ConditionalNeuralProcesses
Conditional Neural Process

In this repo we have applied Conditional Neural Processes to time series prediction. This work is based on 


# The Functional Neural Process (FNP)

Based on the work from Amsterdam Machine Learning lab.

A neural network (NN) is a parameterised function that can be tuned via gradient descent to approximate a labelled collection of data with high precision. A Gaussian process (GP), on the other hand, is a non-parametric probabilistic model that defines a distribution over possible functions, and is updated with new data via the rules of bayesian inference. GPs are probabilistic, data-efficient and flexible, however they are also computationally in-tensive and thus limited in their applicability.

Neural Processes (NP) are a class of neural latent variable models that:

- defines distributions over functions,
- capable of rapid adaptation to new observations, and
- can estimate the uncertainty in their predictions (just as GP do).

## Differences betwen NP, ANP and FNP

Functional Neural Processes (FNP) don't require explicit global latent variables in their construction as NP, but they rather operate by building a graph of dependencies among local latent variables, reminiscing more of autoencoder type of latent variable models. Neural Processes (NPs) define distributions over global latent variables in terms of subsets of the data, while Attentive Neural Processes (ANP) are extend NPs with a deterministic path that has across-attention mechanism among the datapoints. In a sense, FNPs can be seen as a variant where we discard the global latent variables and instead incorporate cross-attention in the form of a dependency graph among local latent variables.
Attention Mechanism

An Attention Mechanism (AM) enables a neural network to focus only on relevant parts of input data instead of trying to deal with all data when doing a prediction task. AM learns in witch part of the dataset in a similar way than Neural Turing Machines (NTM), it is focusing everywhere, just to different extents.

Neural Turing Machines combine a RNN with an external memory store in order to Neural Network write and read from this memory everywhere, just to different extents at each step. Instead of specifying a single location, the RNN outputs an 'attention distribution' that describes how we spread out the amount we care about different memory positions. With the same logic an 'attention distribution' describes how much we write at every location. We do this by having the new value of a position in memory be a convex combination of the old memory content and the write value, with the position between the two decided by the attention weight. To decide which positions in memory to focus their attention on, an NTM use a combination of two different methods:

    Content-based attention. Search through their memory and focus on places that match what they’re looking for.
    Location-based attention. Allows relative movement in memory, enabling the NTM to loop.

## Functional Neural Process Model

imagen.png

FNP assumes a distribution over functions $h∈H$, from $x$ to $y$, by first selecting a 'reference' set of points from $X$,and then basing the probability distribution over $h$ around those points. Instead of following the process of NP (encode, aggregate and decode), in FNP there are two steps:

    - Embedding the inputs $(x_i,y_i) ∈ D$ and $x_i ∈ X$ to a latent space $U$. We compute $p(x|u)$
    - Constructing a graph of dependencies in the embedding space. We compute $p(G,A|U)$
    - Parametrizing the predictive distribution that we will use to to perform predictions for unseen points $x^*$. We compute $p(Z|A,G,R,y_R)$ and $p(y|Z)$.

## FNP and FNP+ models

In FNP, prediction for a given value of $y_i$depends on the input covariates $x_i$ only indirectly via the graphs $G,A$ which are a function of $u_i$. Intuitively, it encodes the inductive bias that predictions on points that are distant, i.e. have very small probability of being connected to the reference set via $A$, will default to an uninformative standard normal prior over $z_i$ hence a constant prediction for $y_i$. In the case of FNP+, we want to add extrapolation so we condition the prediction also on $u_i$, so we are adding a direct path to $U$ instead of working only with directed graphs of dependencies.
References

    - 'The Functional Neural Process'. Christos Louizos, Xiahan Shi, Klamer Schutte, Max Welling, arXiv:1906.08324, 2019. https://arxiv.org/abs/1906.08324
    - 'Cross Attention Network for Few-shot Classification'. Ruibing Hou, Hong Chang, Bingpeng Ma, Shiguang Shan, Xilin Chen , arXiv:1910.07677, 2019 (https://arxiv.org/abs/1910.07677)
    - 'Attention and Augmented Recurrent Neural Networks', Olah and Carter, Distill, 2016. (http://distill.pub/2016/augmented-rnns)

