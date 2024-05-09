# Regularized Logistic Regression

In this parts, you see how to implement `regularized logistic regression`. Just as the gradient update for logistic regression has seemed surprisingly similar to the gradient update for linear regression, you find that the gradient descent update for regularized logistic regression will also look similar to the update for regularized linear regression. Let's take a look.

Here is the idea. We saw earlier that logistic regression can be prone to overfitting if you fit it with very high order polynomial features like this. Here, z is a high order polynomial that gets passed into the sigmoid function like so to compute f.

![RLOR (1)](./../../Assets/Supervised/PO/RLOR%20(1).png)

In particular, you can end up with a decision boundary that is overly complex and overfits as training set. More generally, when you train logistic regression with a lot of features, whether polynomial features or some other features, there could be a higher risk of overfitting.

![RLOR (2)](./../../Assets/Supervised/PO/RLOR%20(2).png)

This was the cost function for logistic regression. If you want to modify it to use regularization, all you need to do is add to it the following term.

Let's add lambda to regularization parameter over 2m times the sum from j equals 1 through n, where n is the number of features as usual of wj squared.

![RLOR (3)](./../../Assets/Supervised/PO/RLOR%20(3).png)

When you minimize this cost function as a function of w and b, it has the effect of penalizing parameters w_1, w_2 through w_n, and preventing them from being too large. If you do this, then even though you're fitting a high order polynomial with a lot of parameters, you still get a decision boundary that looks like this.

![RLOR (4)](./../../Assets/Supervised/PO/RLOR%20(4).png)

Something that looks more reasonable for separating positive and negative examples while also generalizing hopefully to new examples not in the training set.

When using regularization, even when you have a lot of features. How can you actually implement this? How can you actually minimize this cost function j of wb that includes the regularization term? Well, let's use gradient descent as before. Here's a cost function that you want to minimize.

![RLOR (5)](./../../Assets/Supervised/PO/RLOR%20(5).png)

To implement gradient descent, as before, we'll carry out the following simultaneous updates over wj and b. These are the usual update rules for gradient descent.

![RLOR (6)](./../../Assets/Supervised/PO/RLOR%20(6).png)

Just like regularized linear regression, when you compute where there are these derivative terms, the only thing that changes now is that the derivative respect to wj gets this additional term, lambda over m times wj added here at the end. 

![RLOR (7)](./../../Assets/Supervised/PO/RLOR%20(7).png)

Again, it looks a lot like the update for `regularized linear regression`. In fact is the exact same equation, except for the fact that the definition of f is now no longer the linear function, it is the logistic function applied to z. 

![RLOR (8)](./../../Assets/Supervised/PO/RLOR%20(8).png)

Similar to linear regression, `we will regularize only the parameters w, j, but not the parameter b, which is why there's no change the update you will make for b`.

In the final notebook of this week, you revisit overfitting. In the interactive plot in the notebook, you can now choose to regularize your models, both regression and classification, by enabling regularization during gradient descent by selecting a value for lambda. Please take a look at the code for implementing regularized logistic regression in particular, because you'll implement this in practice lab yourself at the end of this week. Now you know how to implement regularized logistic regression. When I walk around Silicon Valley, there are many engineers using machine learning to create a ton of value, sometimes making a lot of money for the companies. I know you've only been studying this stuff for a few weeks but if you understand and can apply linear regression and logistic regression, that's actually all you need to create some very valuable applications. While the specific learning outcomes you use are important, knowing things like when and how to reduce overfitting turns out to be one of the very valuable skills in the real world as well. 

I want to say congratulations on how far you've come and I want to say great job for getting through all the way to the end of this parts I hope you also work through the practice labs. Having said that, there are still many more exciting things to learn. In the next section of this specialization, you'll learn about neural networks, also called deep learning algorithms. Neural networks are responsible for many of the latest breakthroughs in the eye today, from practical speech recognition to computers accurately recognizing objects and images, to self-driving cars. The way neural network gets built actually uses a lot of what you've already learned, like cost functions, and gradient descent, and sigmoid functions. 

Again, congratulations on reaching the end of this supervised machine learning . I hope you have fun in the noteboks and I will see you in Advanced Algorithms.


