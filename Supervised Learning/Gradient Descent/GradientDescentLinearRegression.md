# Gradient Descent with Linear Regression 

Previously, you took a look at the linear regression model and then the cost function, and then the gradient descent algorithm. In this part, `we're going to pull out together and use the squared error cost function for the linear regression model with gradient descent`. `This will allow us to train the linear regression model to fit a straight line to achieve the training data`. 

Let's get to it. Here's the linear regression model. To the right is the squared error cost function. Below is the gradient descent algorithm. 

![GDLR1](./../../Assets/Supervised/GradientDescent/GDLR1.png)

It turns out if you calculate these derivatives, these are the terms you would get. 

`The derivative with respect to W is this 1 over m, sum of i equals 1 through m`. `Then the error term, that is the difference between the predicted and the actual values times the input feature xi`. 

![GDLR2](./../../Assets/Supervised/GradientDescent/GDLR2.png)

`The derivative with respect to b is this formula over here, which looks the same as the equation above, except that it doesn't have that xi term at the end`. `If you use these formulas to compute these two derivatives and implements gradient descent this way, it will work`. 

> Now, you may be wondering, where did I get these formulas from? They're derived using calculus. If you want to see the full derivation, I'll quickly run through the derivation on the next slide. But if you don't remember or aren't interested in the calculus, don't worry about it. You can skip the materials on the next slide entirely and still be able to implement gradient descent and finish this part and everything will work just fine. In this slide, which is one of the most mathematical slide of the entire specialization, and again is completely optional, we'll show you how to calculate the derivative terms. 

>Let's start with the first term. The derivative of the cost function J with respect to w. `We'll start by plugging in the definition of the cost function J. J of WP is this`. 
![GDLR3](./../../Assets/Supervised/GradientDescent/GDLR3.png) 
`1 over 2m times this sum of the squared error terms`. Now remember also that `f of wb of X^i is equal to this term over here`, which is `WX^i plus b`. What we would like to do is compute the derivative, also called the partial derivative with respect to w of this equation right here on the right. 
![GDLR4](./../../Assets/Supervised/GradientDescent/GDLR4.png)`If you taken a calculus class before, and again is totally fine if you haven't, you may know that by the rules of calculus, the derivative is equal to this term over here. Which is why the two here and two here cancel out, leaving us with this equation that you saw on the previous slide`.
This is why we had to find the cost function with the 1.5 earlier this week is because it makes the partial derivative neater. It cancels out the two that appears from computing the derivative. 

>For the other derivative with respect to b, this is quite similar. I can write it out like this, and once again, plugging the definition of `f of X^i`, giving this equation. 
![GDLR5](./../../Assets/Supervised/GradientDescent/GDLR5.png) `By the rules of calculus, this is equal to this where there's no X^i anymore at the end. The 2's cancel one small and you end up with this expression for the derivative with respect to b`. `Now you have these two expressions for the derivatives. You can plug them into the gradient descent algorithm`. 

Here's the gradient descent algorithm for linear regression. You repeatedly carry out these updates to w and b until convergence. 

![GDLR6](./../../Assets/Supervised/GradientDescent/GDLR6.png)

Remember that this `f of x is a linear regression model, so as equal to w times x plus b`. This expression here is the derivative of the cost function with respect to w. This expression is the derivative of the cost function with respect to b.

![GDLR7](./../../Assets/Supervised/GradientDescent/GDLR7.png)

 `Just as a reminder, you want to update w and b simultaneously on each step`.

Now, let's get familiar with how gradient descent works. `One issue we saw with gradient descent is that it can lead to a local minimum instead of a global minimum`. Whether global minimum means the point that has the lowest possible value for the cost function J of all possible points. You may recall this surface plot that looks like an outdoor park with a few hills with the process and the birds as a relaxing Hobo Hill.

![GDLR8](./../../Assets/Supervised/GradientDescent/GDLR8.png)

This function has more than one local minimum. Remember, depending on where you initialize the parameters w and b, you can end up at different local minima. You can end up on any of the minimum. `But it turns out when you're using a squared error cost function with linear regression, the cost function does not and will never have multiple local minima`. `It has a single global minimum because of this bowl-shape`.

![GDLR8](./../../Assets/Supervised/GradientDescent/GDLR9.png)

The technical term for this is that this cost function is a `convex function`. `Informally, a convex function is of bowl-shaped function and it cannot have any local minima other than the single global minimum`. `When you implement gradient descent on a convex function, one nice property is that as long as you're learning rate is chosen appropriately, it will always converge to the global minimum`. 

Congratulations, you now know how to implement gradient descent for linear regression. We have just one last past of this section. In that part, you'll see this algorithm in action.


