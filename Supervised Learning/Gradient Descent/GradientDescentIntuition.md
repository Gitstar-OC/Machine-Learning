# Gradient Descent Intuition 

Now let's dive more deeply in gradient descent to gain better intuition about what it's doing and why it might make sense. Here's the gradient descent algorithm that you saw in the previous part.

![GDI1](./../../Assets/Supervised/GradientDescent/GDI1.png)

As a reminder, this variable, this Greek symbol `Alpha`, `is the learning rate`. `The learning rate controls how big of a step you take when updating the model's parameters, w and b`. `This term here, this d over dw, this is a derivative term`. By convention in math, this d is written with this funny font here. 

![GDI2](./../../Assets/Supervised/GradientDescent/GDI2.png)

<u> In case anyone watching this has PhD in math or is an expert in multivariate calculus, they may be wondering, that's not the derivative, that's the partial derivative. Yes, they be right. But for the purposes of implementing a machine learning algorithm, I'm just going to call it derivative. Don't worry about these little distinctions </u>. 

What we're going to focus on now is get more intuition about what this learning rate and what this derivative are doing and why when multiplied together like this, it results in updates to parameters w and b. That makes sense. In order to do this let's use a slightly simpler example where we work on minimizing just one parameter. 


Let's say that you have a cost function J of just one parameter w with w is a number. `This means the gradient descent now looks like this. W is updated to w minus the learning rate Alpha times d over dw of J of w`. `You're trying to minimize the cost by adjusting the parameter w`. 

![GDI3](./../../Assets/Supervised/GradientDescent/GDI3.png)

<u>This is like our previous example where we had temporarily set b equal to 0 with one parameter w instead of two, you can look at two-dimensional graphs of the cost function j, instead of three dimensional graphs </u>. 

Let's look at what gradient descent does on just function J of w. 

![GDI4](./../../Assets/Supervised/GradientDescent/GDI4.png)

Here on the `horizontal axis is parameter w`, and on the `vertical axis is the cost j of w`. Now less initialized gradient descent with some starting value for w. Let's initialize it at this location. Imagine that you start off at this point right here on the `function J`, `what gradient descent will do is it will update w to be w minus learning rate Alpha times d over dw of J of w`. Let's look at what this derivative term here means. `A way to think about the derivative at this point on the line is to draw a tangent line, which is a straight line that touches this curve at that point`. Enough, `the slope of this line is the derivative of the function j at this point`. To get the slope, you can draw a little triangle like this. 

![GDI5](./../../Assets/Supervised/GradientDescent/GDI5.png)

`If you compute the height divided by the width of this triangle, that is the slope`. <u> For example, this slope might be 2 over 1 </u> , `for instance and when the tangent line is pointing up and to the right, the slope is positive`, `which means that this derivative is a positive number`, `so is greater than 0`. `The updated w is going to be w minus the learning rate times some positive number`. `The learning rate is always a positive number`. 

If you take w minus a positive number, you end up with a new value for w, that's smaller. On the graph, you're moving to the left, you're decreasing the value of w. You may notice that this is the right thing to do if your goal is to decrease the cost J, because when we move towards the left on this curve, the cost j decreases, and you're getting closer to the minimum for J, which is over here. So far, gradient descent, seems to be doing the right thing.

Now, let's look at another example. Let's take the same function j of w as above, and now let's say that you initialized gradient descent at a different location. Say by choosing a starting value for w that's over here on the left. 

![GDI6](./../../Assets/Supervised/GradientDescent/GDI6.png)

That's this point of the function j. Now, the derivative term, remember is `d over dw of J of w`, and `when we look at the tangent line at this point over here, the slope of this line is a derivative of J at this point`. `But this tangent line is sloping down into the right`. `So this line has a negative slope`. `In other words, the derivative of J at this point is a negative number`. 

For instance, if you draw a triangle, then the height like this is negative 2 and the width is 1, the slope is negative 2 divided by 1, which is negative 2, which is a negative number. 

![GDI7](./../../Assets/Supervised/GradientDescent/GDI7.png)

`When you update w, you get w minus the learning rate times a negative number`. `This means you subtract from w, a negative number. But subtracting a negative number means adding a positive number`, <u> and `so you end up increasing w` </u>. Because subtracting a negative number is the same as adding a positive number to w. 

This step of gradient descent causes w to increase, which means you're moving to the right of the graph and your cost J has decrease down to here.

![GDI8](./../../Assets/Supervised/GradientDescent/GDI8.png)

Again, it looks like gradient descent is doing something reasonable, is getting you closer to the minimum. Hopefully, these last two examples show some of the intuition behind what a derivative term is doing and why this host gradient descent change w to get you closer to the minimum. I hope this part gave you some sense for why the derivative term in gradient descent makes sense. One other key quantity in the gradient descent algorithm is the learning rate `Alpha`. How do you choose it? What happens if it's too small or what happens when it's too big? In the next part, let's take a deeper look at the parameter `Alpha` to help build intuitions about what it does, as well as how to make a good choice for a good value of Alpha for your implementation of gradient descent.

