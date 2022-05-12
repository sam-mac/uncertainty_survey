# UQ PhD - Uncertainty Survey
Survey of a few methods for inferring uncertainty. Datasets will include standard benchmarks for classification, regression, multi-model learning, and domain adaption. If time permits, wother benchmarks might be added (relating to graph learning and counterfactual inference), but let's see how things progress.

## Computational environment
-   Github with collab first
-   pytorch lightning, W&B, pyro
-   Colab 
-   docker
-   GCP 
-   website output
	-   visualisations
	-   results output

## Tasks/datasets
-   Classification:
	-   CIFAR-10 [[Approximate Inference in BDL (comp)]]
-   Regression
	-   UCI-GAP [[Approximate Inference in BDL (comp)]]

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

### 2. (out-of-scope) Meta-learning
In the future focus on $\mathcal{A}$.
