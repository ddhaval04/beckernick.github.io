---
title: Logistic Regression from scratch
tags: [machine learning]
header:
  overlay_image: /assets/images/model-selection-header-2.jpg
  caption: "Photo credit: [**Unsplash**](https://www.unsplash.com)"
excerpt: "This blog outlines some of the strategies I use for model selection and evaluation ..."
---

## Introduction

In this article, I am going to implement logistic regression from scratch. It is primarily used for classification tasks. *Does this email belong to the spam folder or the inbox? Should we increase the credit limit of a customer?* When we are interested in such similar tasks where we have to assign each data point to a particular category or estimating probability that a category applies, we define such tasks 'classification'.

One of the basic type of classification problem is binary classification, i.e. when there are only two categories for each data point to be classified from.

## Linear Classifiers

Let us call our two categories as 'positive' and 'negative'. In order to not deviate from the core topic at hand, let us confine ourselves to linear models. 