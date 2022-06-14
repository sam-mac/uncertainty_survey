# UQ PhD - Uncertainty Survey
Survey of a few methods for inferring uncertainty. Datasets will include standard benchmarks for classification, regression, multi-model learning, and domain adaption. If time permits, wother benchmarks might be added (relating to graph learning and counterfactual inference), but let's see how things progress.

## Tasks

#### TODO
- Implement anything on CIFAR-10 
- 1. ensure cifar_bdl and cifar_10_1 align w.r.t. image shape and labels
- 2. ensure cifar10 test isnt validation...
- validate laptop 
- validate weiner
- Implement basic report output
- Implement Baseline LL solution on CIFAR-10
- Implement Laplace inference on CIFAR-10 
- Implement SG-MCMC inference on CIFAR-10

#### DONE
- bash script for obtaining CIFAR-10.1 data
- basic py script for uploading CIFAR-10.1 data
- basic py script for uploading hmcbaseline CIFAR-10.1 data


## Computational environment
-   Github with collab first
-   pytorch lightning, W&B, pyro
-   Colab 
-   docker (todo)
-   Weiner (todo)
-   website output
	-   visualisations
	-   results output

## Datasets and likelihoods
-   Classification (softmax likelihood):
	-   CIFAR-10 [[Approximate Inference in BDL (comp)]]
	-   CIFAR-10.1-v6 from https://github.com/modestyachts/CIFAR-10.1
-   Regression (unknown likelihood)
	-   UCI-GAP [[Shifts (comp)]]
-   Regression (zero inflated Poisson)
	-   RNA target???
-   NLP (transfer learning)
	-   [[Shifts (comp)]]

## Motivation and Problem Definition
We aspire (in the PhD) to create an interpretable model for multi-modal omics problems in order to make statements about safety and scientific explanations (i.e. need *transparency*) 

The problem is that technical and biological confounders cause distribution shifts in multi-omic inference. Further, the task we model has a lot of (heteroscedastic) noise.

Hence, the problem is specificaly:
- distribution shifts in lieu of (principled) adaptive inference is unreliable
- uncertainties are overconfident or miscalibrated
- high-dimensionality of data requires deep learning models which are non-interpretable (i.e. black-box)

## Solution Design

The predictive marginal distribution is given as
$$p(y*|x*, \mathcal{D}, \mathcal{A}) = \int_{\Theta \in \Omega}{p(y*|x*,\Theta)p(\Theta|\mathcal{D}, \mathcal{A})} d\Theta,$$
which requies evaluating the parametric posterior:
$$p(\Theta|\mathcal{D}, \mathcal{A}) = \frac{p(\mathcal{D}|\Theta, \mathcal{A})p(\Theta|\mathcal{A})}{p(D|\mathcal{A})}.$$
### 1. Advancing Uncertainty Quantification 
This focuses on the (approximate) posterior $p(\Theta|\mathcal{D})$.

- require review of postierior inference techniques (to ramp the Bayesian crank)
- first with basic benchmarks and baselines (for sanity check)
- establish best inference technique w.r.t. complexity, fidelity (to true posterior), and robustness (to shifts)
- then we extend to multi-omic dataset 

### 2. (out-of-scope) Priors
- architectural priors
- distributional priors

### 3. (out-of-scope) Meta-learning
In the future focus on $\mathcal{A}$.
