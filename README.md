This repository contains the source code associated with the paper "Cross-Institutional Transfer Learning for Educational Models: Implications for Model Performance, Fairness, and Equity".

If you use this code, please cite our paper.

# Installation

To run the code, first install anaconda on your machine (instructions [here](https://docs.anaconda.com/free/anaconda/install/)).

Then create and activate the virtual environment:

```
conda env create -f environment.yml
conda activate transfer
```

All of the commands below assume the environment has been activated.

# Using Your Own Institutional Data

We aren't able to provide the datasets used in the paper. However, we hope that you use the code to run experiments on your own instutitutional datasets. 

Before starting, give each institution a unique string identifier using the `INSTITUTION_CODES` variable in `keys.py`.

## Data validation

The first step to running multi-institutional training experiments is to ensure that the datasets at each institution are aligned. The script `validate_data.py` in `scripts` can be used at each institution to verify that their preprocessed data conforms to the expected schema.

More details about the expected schema are in the paper, and in the `validate_data()` function in `transfer/validation.py`.

You may also find it useful to modify the schema, and the downstream code should work fine when the schema is changed; however, we cannot provide support in adapting the schema to fit your data.

## Running transfer experiments

**Training Local Models**

Once the data has been prepared and validated, each institution can train their local models. These are the foundation of the later experiments.

Local models can be trained with `train.py`:

```
python scripts/train.py --src_institution my_institution --lambda_sweep
```

This will train all variants of all local models (tuned MLP, LightGBM, and logistic regression, plus sweeps of 10 values of the regularization parameter for each model, for a total of 33 local models).

**Training Ensembles and Evaluating Models**

Once each institution has trained their local models, these need to be shared with the other institutions. (This is the only model sharing step.) Place the models in a `models` directory in this repository.

Once all institutions' local models have been shared, each institution runs the models against their own data.

Evaluate direct transfer models via

```
python scripts/evaluate.py --target_institution my_institution --lambda_sweep
```

and train/evaluate the ensemble models via

```
python scripts/train_and_evaluate_ensembles.py --target_institution my_institution
```

The results will be placed in a folder `metrics`.
