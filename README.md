# Overview

Various exercises and python notebook samples to showcase the end to end MLOps lifecycle. 

The various folders cover the following:

1. 01_Model_Train
* Format: step by step guide to running the exercises, code provided / UI provided
* How to train a model using Vertex AI AutoML, BQML, and custom training jobs. 
* Includes example of how to train a custom model using custom container, with example Dockerfile and code to run. 
* How to test the prediction (batch and online) for registered models (including model monitoring on batch prediction and generation of local feature explanations)

2. 02_Experiments
* Format: separate ipynb for exercises and solutions 
* How to ues Vertex AI experiments for experiment tracking
* How to use experiments to compare pipeline runs 
* How to set up and use a managed Tensorboard instance

3. 03_Pipelines
* References the following repo as a FIRST introduction to Kubeflow pipelines 101: https://github.com/eliasecchig/mlops_workshop
* Then also contains exercises and solutions for running pipelines using other examples

4. 04_Model_monitoring
* Format: separate ipynb for exercises and solutions
* How to set up automated model monitoring jobs on a batch prediction job

5. 05_XAI
* Format: separate ipynb for exercises and solutions
* How to configure and use explainable AI for feature attributions

6. 06_Feature Store
* Format: separate ipynb for exercises and solutions
* Exercises to set up, configure, import data into a managed Feature Store instance 
