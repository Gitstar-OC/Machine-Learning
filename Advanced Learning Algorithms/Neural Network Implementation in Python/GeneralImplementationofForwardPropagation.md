# General Implementation of Forward Propagation

In the last part, you saw how to implement forward prop in Python, but by hard coding lines of code for every single neuron. Let's now take a look at the more general implementation of forward prop in Python. Similar to the previous part, my goal in this part is to show you the code so that when you see it again in their practice lab, in the optional labs, you know how to interpret it. As we walk through this example, don't worry about taking notes on every single line of code. If you can read through the code and understand it, that's definitely enough.

What you can do is write a function to implement a dense layer, that is a single layer of a neural network. I'm going to define the dense function, which takes as input the activation from the previous layer, as well as the parameters w and b for the neurons in a given layer.

![GI 1](./../../Assets/Algorithms/NNIP/GI%20(1).png)

Using the example from the previous part, if layer 1 has three neurons, and if w_1 and w_2 and w_3 are these,

![GI 2](./../../Assets/Algorithms/NNIP/GI%20(2).png)

then what we'll do is stack all of these wave vectors into a matrix.

This is going to be a two by three matrix, where the first column is the parameter w_1,1 the second column is the parameter w_1, 2, and the third column is the parameter w_1,3.

![GI 3](./../../Assets/Algorithms/NNIP/GI%20(3).png)

Then in a similar way, if you have parameters be, b_1,1 equals negative one, b_1,2 equals one, and so on, then we're going to stack these three numbers into a 1D array b as follows,

![GI 4](./../../Assets/Algorithms/NNIP/GI%20(4).png)

negative one, one, two. What the dense function will do is take as inputs the activation from the previous layer, and a here could be a_0,

![GI 4](./../../Assets/Algorithms/NNIP/GI%20(4).png)

which is equal to x, or the activation from a later layer, as well as the w parameters stacked in columns, like shown on the right, as well as the b parameters also stacked into a 1D array, like shown to the left over there.

![GI 4](./../../Assets/Algorithms/NNIP/GI%20(4).png)

What this function would do is input a to activation from the previous layer and will output the activations from the current layer.

Let's step through the code for doing this. Here's the code.

![GI 5](./../../Assets/Algorithms/NNIP/GI%20(5).png)

First, units equals W.shape,1. W here is a two-by-three matrix, and so the number of columns is three. That's equal to the number of units in this layer. Here, units would be equal to three.

![GI 6](./../../Assets/Algorithms/NNIP/GI%20(6).png)

Looking at the shape of w, is just a way of pulling out the number of hidden units or the number of units in this layer.

Next, we set a to be an array of zeros with as many elements as there are units. In this example, we need to output three activation values, so this just initializes a to be zero, zero, zero, an array of three zeros.

![GI 6](./../../Assets/Algorithms/NNIP/GI%20(6).png)

Next, we go through a for loop to compute the first, second, and third elements of a. For j in range units, so j goes from zero to units minus one. It goes from 0, 1, 2 indexing from zero and Python as usual.

![GI 6](./../../Assets/Algorithms/NNIP/GI%20(6).png)

This command w equals W colon comma j, this is how you pull out the jth column of a matrix in Python. The first time through this loop, this will pull the first column of w, and so will pull out w_1,1. The second time through this loop, when you're computing the activation of the second unit, will pull out the second column corresponding to w_1, 2, and so on for the third time through this loop.

Then you compute z using the usual formula, is a dot product between that parameter w and the activation that you have received, plus b, j.

And then you compute the activation a, j, equals g sigmoid function applied to z. Three times through this loop and you compute it, the values for all three values of this vector of activation is a.

![GI 7](./../../Assets/Algorithms/NNIP/GI%20(7).png)

Then finally you return a. What the dense function does is it inputs the activations from the previous layer, and given the parameters for the current layer, it returns the activations for the next layer.

Given the dense function, here's how you can string together a few dense layers sequentially, in order to implement forward prop in the neural network.

![GI 8](./../../Assets/Algorithms/NNIP/GI%20(8).png)

Given the input features x, you can then compute the activations a_1 to be a_1 equals dense of x, w_1, b_1, where here w_1, b_1 are the parameters, sometimes also called the weights of the first hidden layer.

![GI 9](./../../Assets/Algorithms/NNIP/GI%20(9).png)

Then you can compute a_2 as dense of now a_1, which you just computed above. W_2, b-2 which are the parameters or weights of this second hidden layer.

![GI 10](./../../Assets/Algorithms/NNIP/GI%20(10).png)

Then compute a_3 and a_4. If this is a neural network with four layers, then define the output f of x is just equal to a_4, and so you return f of x.

Notice that here I'm using W, because under the notational conventions from linear algebra is to use uppercase or a capital alphabet is when it's referring to a matrix and lowercase refer to vectors and scalars.

![GI 11](./../../Assets/Algorithms/NNIP/GI%20(11).png)

So because it's a matrix, this is W. That's it. You now know how to implement forward prop yourself from scratch.

You get to see all this code and run it and practice it yourself in the practice lab coming off to this as well. I think that even when you're using powerful libraries like TensorFlow, it's helpful to know how it works under the hood. Because in case something goes wrong, in case something runs really slowly, or you have a strange result, or it looks like there's a bug, your ability to understand what's actually going on will make you much more effective when debugging your code. When I run machine learning algorithms a lot of the time, frankly, it doesn't work. Softly, not the first time. I find that my ability to debug my code to be a TensorFlow code or something else, is really important to being an effective machine learning engineer. Even when you're using TensorFlow or some other framework, I hope that you find this deeper understanding useful for your own applications and for debugging your own machine learning algorithms as well. That's it. That's the last required part of this week with code in it.

In the next part, I'd like to dive into what I think is a fun and fascinating topic, which is, `What is the relationship between neural networks and AI or AGI, artificial general intelligence?` This is a controversial topic, but because it's been so widely discussed, I want to share with you some thoughts on this. When you are asked, are neural networks at all on the path to human level intelligence? You have a framework for thinking about that question. Let's go take a look at that fun topic, I think, in the next part.

​
