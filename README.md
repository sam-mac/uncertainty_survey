# UQ PhD - Uncertainty Survey
Survey of a few methods for inferring posterior, which is required for ascertaining uncertainty (about model/knowledge, as well as data). Datasets will include standard benchmarks for classification, regression, multi-model learning, and domain adaption. If time permits, wother benchmarks might be added (relating to graph learning and counterfactual inference), but let's see how things progress.

## Tasks

#### TODO
- validate weiner
- Implement MAP solution (HMC architecture) on cifar10 
- Conda yaml to base of repo
- 1. ensure cifar_bdl and cifar_10_1 align w.r.t. image shape and labels
- 2. ensure cifar10 test isnt validation...
- Implement basic report output
- Implement Baseline LL solution on CIFAR-10
- Implement Laplace inference on CIFAR-10
- Implement SG-MCMC inference on CIFAR-10

#### DONE
- validate laptop 
- bash script for obtaining CIFAR-10.1 data
- basic py script for uploading CIFAR-10.1 data
- basic py script for uploading hmcbaseline CIFAR-10.1 data

## Computational environment
-   pytorch (abstracted with lightning), 
-   W&B (tracking experiments)
-   docker (todo)
-   Weiner (todo)
-   website output
	-   visualisations
	-   results output

# Structure of Survey

## Motivation and Problem Definition
We aspire (in the PhD) to create an interpretable model for multi-modal omics problems in order to make statements about safety and scientific explanations (i.e. need *transparency*) 

The problem is that technical and biological confounders cause distribution shifts in multi-omic inference. Further, the task we model has a lot of (heteroscedastic) noise.

Hence, the problem is specificaly:
- multi-omics data are high-dimensional which makes deep learning (DL) attractive, but
- data hosts (heteroscedastic) noise and is distributionally inconsistent
- distribution shifts in lieu of (principled) adaptive inference is unreliable, unsafe, which hinders DL adoption (in practice)
- Unreliability is w.r.t. accuracy and uncertainty (miscalibration), which renders notions of significance and causality unreliable too...

## Solution Strategy
Practically driven/directed (i.e. in omics), theoretically focused - Relevant Rigour Research. Must be important, accessible, and suitable. 

**Importance** guaranteed by everyones input. Focuses on key issue (ref problem definition). Real data hence real-world. Very timely since growth in research interest.

**Accessibility** facilitated by making research understandable via making survey open-source code. Writing papers (and perhaps blogs). Communications may focus on results, rather than proces, without diminishing necessity of process.

**Suitable** - when are we complete? Communicate guidance and/or direction. Give concrete directions.

## Solution Design

The predictive marginal distribution is given as

$$p(y*|x*, \mathcal{D}, \mathcal{A}) = \int_{\Theta \in \Omega}{p(y*|x*, \mathcal{D}, \mathcal{A}, \Theta)p(\Theta|\mathcal{D}, \mathcal{A})} d\Theta,$$
which requies evaluating the parametric posterior:
$$p(\Theta|\mathcal{D}, \mathcal{A}) = \frac{p(\mathcal{D}|\Theta, \mathcal{A})p(\Theta|\mathcal{A})}{p(D)}.$$

where 

- $\mathcal{D}$ is evidence/data
- $y* = x*$ if we are in a generative setting
- $\mathcal{A}$ represents model architecture, hyperparameters, and assumptions (e.g. kernel)
    - defines whether we are in **a parametric**, or **non-parametric** setting
- $\Theta$ is some sufficient statistic: either (1) the parameters which are ???conditioned??? by evidence, or some latent  variable sufficient statistic

Hence, we must be aware of

- **Evidence** $p(D)$
- **Likelihood** $p(y*|x*, \mathcal{D}, \mathcal{A}, \Theta)$
- **Predictive Prior / Parametric Posterior** $p(\Theta|\mathcal{D}, \mathcal{A})=\frac{p(\mathcal{D}|\Theta, \mathcal{A})p(\Theta|\mathcal{D}, \mathcal{A})}{p(\mathcal{D})}$,

This posterior must be estimated via '*approximate* posterior inference' since $p(\mathcal{D})=\int p(\mathcal{D}, \Theta) d\Theta$ is intractable.

## Focused Aim
This survey will focus on $p(\Theta|\mathcal{D}, \mathcal{A})$ as the central object of interest (to ramp the Bayesian crank). 

This is the central focus??? i.e. our exploration space. There are many reasons why (ref report).

Thus, we constrain other factor's in the predictive distribution, to not only control changes within the predictive prior, but to focus the survey - to facilitate progress.

Importantly, we aim to be relevant, and applicable in our survey.

## Methodology 
- Review of postierior inference techniques
- first with basic benchmarks and baselines
- keep non-posterior factors fixed, but relevant: i.e. datasets, likelihoods, model prior
- establish best inference technique w.r.t. complexity, fidelity (to true posterior), and robustness (to shifts) 
- then we extend to multi-omic dataset 

Basically, the following set of combinations will be implemented; that defines the exploration space.

#### Evidence / Dataset
-   {consider} Classification:
	-   CIFAR-10 [[Approximate Inference in BDL (comp)]]
	-   CIFAR-10.1-v6 from https://github.com/modestyachts/CIFAR-10.1
-   {later} Regression
	-   RNA target
-   {later} Regression (unknown likelihood)
	-   UCI-GAP [[Shifts (comp)]]
-   {later} NLP (transfer learning)
	-   [[Shifts (comp)]]

#### Priors

**Inductive Priors:**
- {consider} weight init (Gaussian or sparsity promoting, e.g. Laplace prior)
- {consider} data augmentation (dual to architecture?)
- {consider} weight decay,
- {consider} dropout
- {later} Architectural priors - CONSTANT WITHIN DATASET - changes with dataset/task
- {later} learning prior using the Marginal Likelihood ??? needs to be in function space, not weight space (e.g. with kernels..)

**Distributional Priors:**
- {consider} weight init (Gaussian or sparsity promoting, e.g. Laplace prior)
- {later} learning prior using the Marginal Likelihood ??? needs to be in function space, not weight space (e.g. with kernels..)
- {later} meta-learning? 

#### Likelihoods

- {consider} **categorical** -> CIFAR
- {consider? / later if time permits} **dirichlet** -> CIFAR
- {later} zero-inflated negative binomial -> some RNA?
- {later} zero-inflated poisson -> some RNA?
- {later} gaussian -> weather?
- {later} tempered likelihood for cold/hot posterior sect Likelihood for BNNs

#### Approximate Posterior (i.e. Predictive Prior)
- {consider} pointwise MAP solution 
- {consider} Laplace
	- [simple eg](https://arxiv.org/abs/2106.14806)
- {consider} (SG) MCMC 
	- [SGLD](https://github.com/alisiahkoohi/Langevin-dynamics)
	- [cSGHMC](https://arxiv.org/abs/1902.03932v2)
- {consider} VI
	- normalising flow
	- BBB
	- GNVI
- {consider} Deep Kernel Learning
	- inducing points
	- random fourier approximation
- {consider} Dropout
	- naive 
	- extended
- {consider} Ensembles
	- deep ensembles (original)
	- batched 
- {later} Gradient based method
	- SWAG

## (out-of-scope) 
In the future focus on $\mathcal{A}$.
- **graph neural networks**
- deep GP
- deep kernel learning
- meta learning is for distribution shift
- active learning is distribution shift
- causal model's (law of independent mechanisms)