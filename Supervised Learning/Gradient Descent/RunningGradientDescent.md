# Running Gradient Descent 

Let's see what happens when you run gradient descent for linear regression. Let's go see the algorithm in action. 

Here's a plot of the model and data on the upper left and a contour plot of the cost function on the upper right and at the bottom is the surface plot of the same cost function. 

![RGD1](./../../Assets/Supervised/GradientDescent/RGD1.png)

Often w and b will both be initialized to 0, but for this demonstration, lets initialized w = -0.1 and b = 900. So this corresponds to f(x) = -0.1x + 900.

![RGD7](./../../Assets/Supervised/GradientDescent/RGD7.png)

Now, if we take one step using gradient descent, we ended up going from this point of the cost function out here to this point just down and to the right and notice that the straight line fit is also changed a bit.

![RGD2](./../../Assets/Supervised/GradientDescent/RGD2.png)

Let's take another step. The cost function has now moved to this third and again the function f(x) has also changed a bit. As you take more of these steps, the cost is decreasing at each update. So the parameters w and b are following this trajectory.

![RGD3](./../../Assets/Supervised/GradientDescent/RGD3.png)

And if you look on the left, you get this corresponding straight line fit that fits the data better and better until we've reached the global minimum. The global minimum corresponds to this straight line fit, which is a relatively good fit to the data. I mean, isn't that cool. And so that's gradient descent and we're going to use this to fit a model to the holding data.

And you can now use this f(x) model to predict the price of your clients house or anyone else's house. For instance, if your friend's house size is 1250 square feet, you can now read off the value and predict that maybe they could get, I don't know, $250,000 for the house. 

![RGD4](./../../Assets/Supervised/GradientDescent/RGD4.png)

To be more precise, this gradient descent process is called `batch gradient descent`. `The term batch gradient descent refers to the fact that on every step of gradient descent, we're looking at all of the training examples, instead of just a subset of the training data`.

![RGD5](./../../Assets/Supervised/GradientDescent/RGD5.png)

`So in computing grading descent, when computing derivatives, when computing the sum from i =1 to m. And bash gradient descent is looking at the entire batch of training examples at each update. I know that bash grading percent may not be the most intuitive name, but this is what people in the machine learning community call it`. <u>If you've heard of the newsletter The Batch, that's published by DeepLearning.AI. The newsletter The batch was also named for this concept in machine learning </u>.

![RGD6](./../../Assets/Supervised/GradientDescent/RGD6.png)

And then it turns out that there are other versions of gradient descent that do not look at the entire training set, but instead looks at smaller subsets of the training data at each update step. But we'll use batch gradient descent for linear regression.

So that's it for linear regression. Congratulations on getting through your first machine learning model. I hope you go and celebrate or I don't know maybe take a nap in your hammock. `In the notebook that follows this part You'll see a review of the gradient descent algorithm as was how to implement it in code. You'll also see a plot that shows how the cost decreases as you continue training more iterations. And you'll also see a contour plot, seeing how the cost gets closer to the global minimum as gradient descent finds better and better values for the parameters w and b`. So remember that to do the notebook. You just need to read and run this code. You will not need to write any code yourself and I hope you take a few moments to do that. And also become familiar with the gradient descent code because this will help you to implement this and similar algorithms in the future yourself.

You now know how to implement linear regression with one variable and that brings us to the close of this section. `In the next section, you'll learn to make linear regression much more powerful instead of one feature like size of a house, you learn how to get it to work with lots of features`. `You'll also learn how to get it to fit nonlinear curves. These improvements will make the algorithm much more useful and valuable`. Lastly, you'll also go over some practical tips that will really hope for getting linear regression to work on practical applications.

