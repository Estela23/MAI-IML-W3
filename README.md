# MAI-IML-W3

## How to execute:

In the source of the project, there is one file called `evaluation.py`.

You can use this file to run several experiments in a simple way. 
The different parameters are:

* __data_to_use__: Selection of which dataset we want to choose. String.
* __file_name_to_export__: Name of the file where the results will be exported. String.
* __k__: Number of neighbours for KNN. List of integers.
* __distance_functions__: Functions to perform distance calculation. List of functions.
* __voting_functions__: Functions to perform voting. List of functions.
* __weighting_functions__: Functions to calculate weights. List of functions.
* __reduction_techniques__: Functions to perform data reduction. List of functions.

So, if for example we want to run KNN with the following parameters:
* K=3, camberra_metric, majority_class, info_gain and no reduction technique

We need to establish `evaluation.py` as follows:
```python
data_to_use = "datasets/hypothyroid"
file_name_to_export = "knn_results/test.txt"

# Options: 1, 3, 5, 7,
k = [3]
# Options: manhattan_metric, euclidean_metric, camberra_metric
distance_functions = [camberra_metric]
# Options: majority_class, inverse_distance_weighted, sheppards_work
voting_functions = [majority_class]
# Options: equal_weight, info_gain, reliefF
weighting_functions = [info_gain]

reduction_techniques = [None]  # new_FCNN_rule, ENN, ib2

run_experiment(data_to_use, file_name_to_export, distance_functions, k, voting_functions, weighting_functions,
               reduction_techniques)
```
Note that if no reduction technique is required, a `None` is needed in the list.
If you want to select multiple options, you just need to put them in the correspondent lists.
Then, all combinations will be performed.

## Project Structure

This project is divided in 4 main Python packages:
* data_cleaning: Here we have several scripts to read and parse datasets.
* datasets: Datasets used in this project.
* KNN: Implementation of KNN, with its algorithms and reduction techniques in subpackages.
* knn_results: Folder where results of KNN are saved. We also provide a few scripts for analysis.