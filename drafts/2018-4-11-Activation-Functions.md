---
title: Activation Functions
tags: [Deep Learning]
header:
  overlay_image: /assets/images/gradient-header.jpg
  caption: "Photo credit: [**Unsplash**](https://www.unsplash.com)"
excerpt: "This blog outlines my understanding of different activation functions used in deep neural networks..."
---

### Sigmoid:

The sigmoid activation function has the mathematical form sig(z) = 1/ (1 + e^-z). As we can see, it basically takes a real valued number as the input and squashes it between 0 and 1. It is often termed as a squashing function as well. It aims to introduce non-linearity in the input space. The non-linearity is where we get the wiggle and the network learns to capture complicated relationships. As we can see from the above mathematical representation, a large negative number passed through the sigmoid function becomes 0 and a large positive number becomes 1. Due to this property, sigmoid function often has a really nice interpretation associated with it as the firing rate of the neuron; from not firing at all (0) to fully-saturated firing at an assumed maximum frequency (1). However, sigmoid activation functions have become less popular over the period of time due to the following two major drawbacks:

- *Killing gradients*: 
Sigmoid neurons get saturated on the boundaries and hence the local gradients at these regions is almost zero. To give you a more intuitive example to understand this, consider the inputs to the sigmoid function to be +15 and -15. The derivative of sigmoid function is sig(z) * (1 - sig(z)). As mentioned above, the large positive values are squashed near 1 and large negative values are squashed near 0. Hence, effectively making the local gradient to near 0. As a result, during backpropagation, this gradient gets multiplied to the gradient of this neurons' output for the final objective function, hence it will effectively "kill" the gradient and no signal will flow through the neuron to its weights. Also, we have to pay attention to initializing the weights of sigmoid neurons to avoid saturation, because, if the initial weights are too large, then most neurons will get saturated and hence the network will hardly learn.

- *Non zero-centered outputs*: 
The output is always between 0 and 1, that means that the output after applying sigmoid is always positive hence, during gradient-descent, the gradient on the weights during backpropagation will always be either positive or negative depending on the output of the neuron. As a result, the gradient updates go too far in different directions which makes optimization harder.


### Tanh:

The tanh or hyperbolic tangent activation function has the mathematical form tanh(z) = (e^z - e^-z) / (e^z + e^-z). It is basically a shifted sigmoid neuron. It basically takes a real valued number and squashes it between -1 and +1. Similar to sigmoid neuron, it saturates at large positive and negative values. However, its output is always zero-centered which helps since the neurons in the later layers of the network would be receiving inputs that are zero-centered. Hence, in practice, tanh activation functions are preffered in hidden layers over sigmoid.


### ReLU:

The ReLU or Rectified Linear Unit is represented as ReLU(z) = max(0, z). It basically thresholds the inputs at zero, i.e. all negative values in the input to the ReLU neuron are set to zero. Fairly recently, it has become popular as it was found that it greatly accelerates the convergence of stochastic gradient descent as compared to Sigmoid or Tanh activation functions. Just to give an intuition, the gradient is either 0 or 1 depending on the sign of the input. However, it has one following drawback:

- *Dead Neurons*:
ReLU units can be fragile during training and can "die". That is, if the units are not activated initially, then during backpropagation zero gradients flow through them. However, there are concepts such as Leaky ReLU that can be used to overcome this problem. Also, having a proper setting of the learning rate can prevent causing the neurons to be dead.