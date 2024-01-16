# fast_feature_selection


HYBparsimony explained in:

[Automatic Hyperparameter Optimization and Feature Selection with HYBparsimony Package](https://medium.com/@jpison/automatic-hyperparameter-optimization-and-feature-selection-with-hybparsimony-package-fb69d0c800bc)

[Searching Parsimonious Models in Small and High-Dimensional Datasets with HYBparsimony Python Package](https://jpison.medium.com/search-for-parsimonious-models-in-small-and-high-dimensional-data-sets-with-hybparsimony-package-962803e96bf7)


vs


methods explained in:

[Efficient Feature Selection via CMA-ES (Covariance Matrix Adaptation Evolution Strategy)](https://towardsdatascience.com/efficient-feature-selection-via-cma-es-covariance-matrix-adaptation-evolution-strategy-ee312bc7b173)

[Efficient Feature Selection via Genetic Algorithms](https://towardsdatascience.com/efficient-feature-selection-via-genetic-algorithms-d6d3c9aff274)


Code extracted from: https://github.com/FlorinAndrei/fast_feature_selection, and adapted to include the following code:

```python

from hybparsimony import HYBparsimony
import random

def fitness_custom(cromosoma, **kwargs):
    X_train = kwargs["X"]
    y_train = kwargs["y"]
        
    # Extract features from the original DB plus response (last column)
    X_fs_selec = X_train.loc[: , cromosoma.columns]
    predictor = sm.OLS(y_train, X_fs_selec, hasconst=True).fit()
    fitness_val = -predictor.bic
    return np.array([fitness_val, np.sum(cromosoma.columns)]), predictor

random.seed(0)
num_indiv_hyb = 20
HYBparsimony_model = HYBparsimony(fitness=fitness_custom,
                                features=X.columns,
                                rerank_error=1.0, #Diff between bics to promote parsimonious solution
                                seed_ini=0,
                                npart=num_indiv_hyb, #Population 1000 individuals
                                maxiter=10000,
                                early_stop=500,
                                verbose=0,
                                n_jobs=1)

HYBparsimony_model.fit(X, y)
print(HYBparsimony_model.best_complexity, HYBparsimony_model.best_score, HYBparsimony_model.minutes_total)

```


The model used for regression is `statsmodels.api.OLS()`. The objective function used to select the best features is BIC, or the Bayesian Information Criterion - less is better.

Four feature selection techniques are explored:

- Sequential Feature Search (SFS) implemented via the [mlxtend](https://github.com/rasbt/mlxtend) library
- Genetic Algorithms (GA) implemented via the [deap](https://github.com/DEAP/deap) library
- Covariance Matrix Adaptation Evolution Strategy (CMA-ES) implemented via the [cmaes](https://github.com/CyberAgentAILab/cmaes) library
- HYBparsimony is a Python package that simultaneously performs automatic: feature selection (FS), model hyperparameter optimization (HO), and parsimonious model selection (PMS) with GA and PSO [HYBparsimony](https://github.com/jodivaso/hybparsimony/)

SFS and GA used a multiprocessing pool with 24 workers to run the objective function. CMA-ES used a single process for everything.


Test system:

- AMD Ryzen Threadripper 3960X 24-Core
- Ubuntu 22.04
- Python 3.10.13


Results:

Run time (less is better):

```
SFS:    79.762 sec
GA:     240.776 sec
CMA-ES: 70.152 sec
HYB-PARSIMONY: 101.067 sec
```

Number of the selected features:

```
baseline:     214
SFS:          36
GA:           33
CMA-ES:       35
HYB-PARSIMONY: 32
```


Number of times the objective function was invoked (less is better):

```
SFS:    22791
GA:     600525
CMA-ES: 20000
HYB-PARSIMONY: 33520
```

Objective function best value found (less is better):

```
baseline BIC: 34570.1662
SFS:          33708.9860
GA:           33706.2129
CMA-ES:       33712.1037
HYB-PARSIMONY: 33710.6326
```
