
# Predicting mixture etching rates with Graph Neural Networks (GNNs)
#### Enhancing plasma etching efficiency, repeatability, and environmental footprint via AI-based modeling and optimization (plasmAI)

### TL;DR
This branch contains the codebase for running experiments on mixtures of $Ar$ and $O_2$, including training and evaluating an MLP and various types of GNNs.

## Project structure:
* **`/data`** : Different datasets and data scalers
* **`config.yaml`** : Hyperparameter configurations required for the experiments
* **`pyproject.toml`**, **`poetry.lock`** : Project's poetry configuration
* **`/scripts`** : Models, training and evaluation code and utilities
	* `data_loader.py` : Dataloaders as required by *torch*
	* `loss.py` : Custom loss function implementation
	* `model.py` : MLP and GNN implementations
	* `train.py` : training pipeline
	* `test.py` : evaluation pipeline
	* `utils.py` : utility code
* **`experiments_MLP.py`** : Training and evaluation script for the mixture MLP experiments
* **`experiments_GNN.py`** : Training and evaluation script for the mixture GNN experiments

## Experiments execution pipeline:
  1. Download the requirements from `pyproject.toml` [cf. [poetry installation guide]].
  2. Prepare the configurations by adjusting the`config.yaml`:
> (2.1.)  In *`mixture > test_type`* choose the type of test set split from (i) *percentage*, (ii) *power* [cf. Report]

> (2.2.) *If you wish to execute the GNN experiments*, then in *`GNN > arch`* choose the GNN architecture from (i) *SAGEConv*, (ii) *GCNConv*, and (iii) *GATConv* [again, cf. Report]
  4. Choose the experiment to run and execute it through poetry. For example, if you wish to run the GNN experiments, execute :

		    poetry run python experiments_GNN.py
5. If you want to run the experiments that utilize only specific layers of the pre-trained single-element MLP, unfortunately, you have to comment as required the `forward()` method within the `Model` class in `scripts/model.py`. It is pretty straightforward, since there is no mix-up with the dimensions.



[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

[poetry installation guide]: <https://python-poetry.org/docs/basic-usage/#initialising-a-pre-existing-project>
