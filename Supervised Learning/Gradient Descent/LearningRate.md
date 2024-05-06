# Learning Rate

The choice of the learning rate, alpha will have a huge impact on the efficiency of your implementation of gradient descent. And if alpha, the learning rate is chosen poorly rate of descent may not even work at all. In this part, let's take a deeper look at the learning rate. This will also help you choose better learning rates for your implementations of gradient descent. 

![LR1](./../../Assets/Supervised/GradientDescent/LR1.png)

So here again, is the gradient descent rule. `W is updated to be W minus the learning rate, alpha times the derivative term`. To learn more about what the learning rate alpha is doing. Let's see what could happen if the learning rate alpha is either too small or if it is too large. 

For the case where the `learning rate is too small`. Here's a graph where the horizontal axis is `W` and the vertical axis is the `cost J`. And here's the graph of the function J of W. 

![LR2](./../../Assets/Supervised/GradientDescent/LR2.png)

Let's start grading descent at this point here, if the learning rate is too small. Then what happens is that you multiply your derivative term by some really, really small number. So you're going to be multiplying by number alpha. That's really small, like 0.0000001. And so you end up taking a very small baby step like that. Then from this point you're going to take another tiny tiny little baby step. But because the learning rate is so small, the second step is also just minuscule. `The outcome of this process is that you do end up decreasing the cost J but incredibly slowly`. So, here's another step and another step, another tiny step until you finally approach the minimum. But as you may notice you're going to need a lot of steps to get to the minimum. `So to summarize if the learning rate is too small, then gradient descents will work, but it will be slow. It will take a very long time because it's going to take these tiny tiny baby steps. And it's going to need a lot of steps before it gets anywhere close to the minimum`.

Now, let's look at a different case. What happens if the `learning rate is too large?` Here's another graph of the cost function. 

![LR3](./../../Assets/Supervised/GradientDescent/LR3.png)

And let's say we start grating descent with W at this value here. So it's actually already pretty close to the minimum. So the decorative points to the right. `But if the learning rate is too large then you update W very giant step to be all the way over here`. And that's this point here on the function J. `So you move from this point on the left, all the way to this point on the right. And now the cost has actually gotten worse`. It has increased because it started out at this value here and after one step, it actually increased to this value here. Now the derivative at this new point says to decrease W but when the learning rate is too big. Then you may take a huge step going from here all the way out here. So now you've gotten to this point here and again, if the learning rate is too big. Then you take another huge step with an acceleration and way overshoot the minimum again. So now you're at this point on the right and one more time you do another update. And end up all the way here and so you're now at this point here. `So as you may notice you're actually getting further and further away from the minimum`. `So if the learning rate is too large, then creating the sense may overshoot and may never reach the minimum`. 

And another way to say that is that great intersect may fail to converge and may even diverge. So, here's another question, you may be wondering one of your parameter W is already at this point here. 

![LR4](./../../Assets/Supervised/GradientDescent/LR4.png)

So that your cost J is already at a local minimum. `What do you think? One step of gradient descent will do if you've already reached a minimum?` So this is a tricky one. When I was first learning this stuff, it actually took me a long time to figure it out. But let's work through this together. 

Let's suppose you have some cost function J. And the one you see here isn't a square error cost function and this cost function has two local minima corresponding to the two valleys that you see here. Now let's suppose that after some number of steps of gradient descent, your parameter W is over here, say equal to five. And so this is the current value of W. This means that you're at this point on the cost function J. And that happens to be a local minimum, turns out if you draw attention to the function at this point. `The slope of this line is zero and thus the derivative term`. `Here is equal to zero for the current value of W. And so you're grading descent update becomes W is updated to W minus the learning rate times zero`. We're here that's because the derivative term is equal to zero. And this is the same as saying let's set W to be equal to W. `So this means that if you're already at a local minimum, gradient descent leaves W unchanged. Because it just updates the new value of W to be the exact same old value of W`. So concretely, let's say if the current value of W is five. And alpha is 0.1 after one iteration, you update W as W minus alpha times zero and it is still equal to five. `So if your parameters have already brought you to a local minimum, then further gradient descent steps to absolutely nothing`. It doesn't change the parameters which is what you want because it keeps the solution at that local minimum. 

This also explains why gradient descent can reach a local minimum, even with a fixed learning rate alpha. Here's what I mean, to illustrate this, let's look at another example. Here's the cost function J of W that we want to minimize. 

![LR5](./../../Assets/Supervised/GradientDescent/LR5.png)

Let's initialize gradient descent up here at this point. If we take one update step, maybe it will take us to that point. And because this derivative is pretty large, grading, descent takes a relatively big step right. 

![LR6](./../../Assets/Supervised/GradientDescent/LR6.png)

Now, we're at this second point where we take another step. And you may notice that the slope is not as steep as it was at the first point. So the derivative isn't as large. And so the next update step will not be as large as that first step. Now, read this third point here and the derivative is smaller than it was at the previous step. 

![LR7](./../../Assets/Supervised/GradientDescent/LR7.png)

And will take an even smaller step as we approach the minimum. The decorative gets closer and closer to zero. So as we run gradient descent, eventually we're taking very small steps until you finally reach a local minimum. `So just to recap, as we get nearer a local minimum gradient descent will automatically take smaller steps. And that's because as we approach the local minimum, the derivative automatically gets smaller. And that means the update steps also automatically gets smaller. Even if the learning rate alpha is kept at some fixed value`.

![LR8](./../../Assets/Supervised/GradientDescent/LR8.png)

 So that's the `gradient descent algorithm`, you can use it to try to minimize any cost function J. Not just the mean squared error cost function that we're using for the new regression. In the next part, you're going to take the function J and set that back to be exactly the linear regression models cost function. The mean squared error cost function that we come up with earlier. And putting together great in dissent with this cost function that will give you your first learning algorithm, the linear regression algorithm.

