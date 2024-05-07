# Feature Scaling Part 1
So welcome back. Let's take a look at some techniques that make great inter sense work much better. In this part you see a technique called `feature scaling` that will enable gradient descent to run much faster. 


As a concrete example, let's predict the price of a house using two features x1 the size of the house and x2 the number of bedrooms. 

![FSP1.1](./../../Assets/Supervised/GDP/FSP1.1.png)

Let's say that x1 typically ranges from 300 to 2000 square feet. And x2 in the data set ranges from 0 to 5 bedrooms. So for this example, x1 takes on a relatively large range of values and x2 takes on a relatively small range of values. 

Now let's take an example of a house that has a size of 2000 square feet has five bedrooms and a price of 500k or $500,000. For this one training example, what do you think are reasonable values for the size of parameters w1 and w2? Well, let's look at one possible set of parameters. Say w1 is 50 and w2 is 0.1 and b is 50 for the purposes of discussion.

![FSP1.2](./../../Assets/Supervised/GDP/FSP1.2.png)

So in this case the estimated price in thousands of dollars is 100,000k here plus 0.5 k plus 50 k. Which is slightly over 100 million dollars. So that's clearly very far from the actual price of $500,000. And so this is not a very good set of parameter choices for w1 and w2. 

Now let's take a look at another possibility. Say w1 and w2 were the other way around. W1 is 0.1 and w2 is 50 and b is still also 50. In this choice of w1 and w2, w1 is relatively small and w2 is relatively large, 50 is much bigger than 0.1. 

![FSP1.3](./../../Assets/Supervised/GDP/FSP1.3.png)

So here the predicted price is 0.1 times 2000 plus 50 times five plus 50. The first term becomes 200k, the second term becomes 250k, and the plus 50. So this version of the model predicts a price of $500,000 which is a much more reasonable estimate and happens to be the same price as the true price of the house.

`So hopefully you might notice that when a possible range of values of a feature is large, like the size and square feet which goes all the way up to 2000. It's more likely that a good model will learn to choose a relatively small parameter value, like 0.1`. `Likewise, when the possible values of the feature are small, like the number of bedrooms, then a reasonable value for its parameters will be relatively large like 50`. 

So how does this relate to grading descent? Well, let's take a look at the scatter plot of the features where the size square feet is the horizontal axis x1 and the number of bedrooms exudes is on the vertical axis. 

![FSP1.4](./../../Assets/Supervised/GDP/FSP1.4.png)

If you plot the training data, you notice that the horizontal axis is on a much larger scale or much larger range of values compared to the vertical axis.

Next let's look at how the cost function might look in a contour plot. 

![FSP1.5](./../../Assets/Supervised/GDP/FSP1.5.png)

You might see a contour plot where the horizontal axis has a much narrower range, say between zero and one, whereas the vertical axis takes on much larger values, say between 10 and 100. So the contours form ovals or ellipses and they're short on one side and longer on the other. And this is because a very small change to w1 can have a very large impact on the estimated price and that's a very large impact on the cost J. 

![FSP1.6](./../../Assets/Supervised/GDP/FSP1.6.png)

Because w1 tends to be multiplied by a very large number, the size and square feet. In contrast, it takes a much larger change in w2 in order to change the predictions much. And thus small changes to w2, don't change the cost function nearly as much. 

So where does this leave us? This is what might end up happening if you were to run great in descent, if you were to use your training data as is. Because the contours are so tall and skinny gradient descent may end up bouncing back and forth for a long time before it can finally find its way to the global minimum. 

![FSP1.7](./../../Assets/Supervised/GDP/FSP1.7.png)

In situations like this, a useful thing to do is to scale the features. This means performing some transformation of your training data so that x1 say might now range from 0 to 1 and x2 might also range from 0 to 1. 
​
![FSP1.8](./../../Assets/Supervised/GDP/FSP1.8.png)

So the data points now look more like this and you might notice that the scale of the plot on the bottom is now quite different than the one on top.

The key point is that the re scale x1 and x2 are both now taking comparable ranges of values to each other. And if you run gradient descent on a cost function to find on this, re scaled x1 and x2 using this transformed data, then the contours will look more like this more like circles and less tall and skinny. 

![FSP1.9](./../../Assets/Supervised/GDP/FSP1.9.png)

And gradient descent can find a much more direct path to the global minimum.

`So to recap, when you have different features that take on very different ranges of values, it can cause gradient descent to run slowly but re scaling the different features so they all take on comparable range of values. because speed, upgrade and dissent significantly`. How do you actually do this? Let's take a look at that in the next part.

