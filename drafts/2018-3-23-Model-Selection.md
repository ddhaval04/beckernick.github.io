---
title: How to evaluate your models?
tags: [machine learning]
header:
  overlay_image: /assets/images/model-selection-header-2.jpg
  caption: "Photo credit: [**Unsplash**](https://www.unsplash.com)"
excerpt: "This blog outlines some of the strategies I use for model selection and evaluation ..."
---

## Intorduction

This post aims to answer one of the basic question that every machine learning practitioner has; "How to evaluate a model's performance?" Once we have a well-defined problem statement for our dataset, we train different possible machine learning algorithms that tries to model the target function accurately. Now, whether we are applying machine learning techniques to solve some business problems or aiming to win any Kaggle competitions, I believe we got one thing in common: We want the model that makes "good" predictions! However, training a model to learn the target function is one thing but how do we know which model generalizes well to the unseen data? That is, how do we know that the learning algorithm simply doesn't memorize the data that we fed in?
In this article I will pen some of these techniques and learn how they fit in a typical machine learning pipeline.

## Assumptions

Model evaluation is a complicated subject. Before we dive into using any of the techniques described below, I would like to mention certain assumptions regarding the data. We assume that our sample has been drawn from the same probability distribution and are statistically independent from each other. This assumption holds true for most of the data that you will be working with (except time-series or temporal data).

## Dataset

[**Iris**](/assets/images/iris.png)

Throughout this article we will be working with the [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris). We will focus on classification, and aim to assign category target labels to each sample accurately. Now let us answer a few questions:

-	How do we define accuracy for our dataset? 
	We will target the prediction accuracy which is defined as the number of correct predictions divided by the number of samples 'n'.

-	We need to define a loss function which our learning algorithm tries to optimize.
	The loss function we define here is a **0-1 loss** where we assign each correctly assigned datapoint a 0 loss and every misclassified sample a 1. Our objective is to learn a model that maximizes the prediction accuracy.






