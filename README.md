# QF4199 Honours Year Project: Construct Individualized Portfolios for Robo-advisors with Goal-based Portfolio Optimization and Conditional Quantile Estimation on Target Returns

## Introduction
This repository contains the implementations and tests for my thesis.

## Step-by-Step Guide
There are several ways to use this repository:

1. View results that have been previously generated
2. Run the algorithm again on a completely new dataset

### View results that have been previously generated

1. Go to the bottom of `evaluation.py` and choose one of the following:

    - Uncomment lines 191 to 219 for `evaluation_1()`
    
        `evaluation_1()` was done with a specific value of `tau = 0.05` on a slightly larger sample to observe how the
        conditional quantile estimates vary with the covariates. 
    
    - Uncomment lines 224 to 236 for `evaluation_2()`:
    
        `evaluation_2()` was done with various values of `tau = 0.01, 0.05, 0.1, 0.25, 0.5` on a slightly smaller sample, 
        to observe how portfolios vary with `tau`.
    
    - Uncomment lines 241 to 257 to get a portfolio for a new investor.
    
2. Re-run on a completely new dataset: 
    
    1. Use the `cross_validation` method in `nonparametric_conditional_quantile_estimator.py` to optimize the bandwidth parameters.
    2. Use the `estimate_quantiles` method in `evaluation.py` to estimate the conditional quantiles.
    3. Use `plot_estimated_quantiles` method in `evaluation.py` if you wish to visualize how the conditional quantile estimates 
vary with the covariates. 

    Refer to examples in `evaluation_1()` and `evaluation_2()` on how this can be done in a similar fashion. 
