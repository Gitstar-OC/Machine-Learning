# Cost Function Formula
In order to implement linear regression the first key step is first to define something called a `cost function`. This is something we'll build and learn in this part, and the `cost function` **will tell us how well the model is doing so that we can try to get it to do better**. Let's look at what this means.


Recall that you have a training set that contains input features x and output targets y. The model you're going to use to fit this training set is this linear function f_w, b of x equals to w times x plus b. To introduce a little bit more terminology the w and b are called the parameters of the model. In machine learning parameters of the model are the variables you can adjust during training in order to improve the model.

![CFF1](./../../Assets/Supervised/RegressionModel/CFF1.png)

Sometimes you also hear the parameters w and b referred to as coefficients or as weights. Now let's take a look at what these parameters w and b do. Depending on the values you've chosen for w and b you get a different function f of x, which generates a different line on the graph. Remember that we can write f of x as a shorthand for f_w, b of x. We're going to take a look at some plots of f of x on a chart. 

Maybe you're already familiar with drawing lines on charts, but even if this is a review for you, I hope this will help you build intuition on how w and b the parameters determine f. When w is equal to 0 and b is equal to 1.5, then f looks like this horizontal line. 

![CFF2](./../../Assets/Supervised/RegressionModel/CFF2.png)

<u> In this case, the function f of x is 0 times x plus 1.5 so f is always a constant value </u>. It always predicts 1.5 for the estimated value of y. Y hat is always equal to b and here b is also called the y intercept because that's where it crosses the vertical axis or the y axis on this graph. 

As a second example, if w is 0.5 and b is equal 0, then f of x is 0.5 times x. When x is 0, the prediction is also 0, and when x is 2, then the prediction is 0.5 times 2, which is 1. You get a line that looks like this

![CFF3](./../../Assets/Supervised/RegressionModel/CFF3.png)

notice that the slope is 0.5 divided by 1. The value of w gives you the slope of the line, which is 0.5. 

Finally, if w equals 0.5 and b equals 1, then f of x is 0.5 times x plus 1 and when x is 0, then f of x equals b, which is 1 so the line intersects the vertical axis at b, the y intercept. 

![CFF4](./../../Assets/Supervised/RegressionModel/CFF4.png)

Also when x is 2, then f of x is 2, so the line looks like this. Again, this slope is 0.5 divided by 1 so the value of w gives you the slope which is 0.5. 

Recall that you have a training set like the one shown here. 

![CFF5](./../../Assets/Supervised/RegressionModel/CFF5.png)

With linear regression, what you want to do is to choose values for the parameters w and b so that the straight line you get from the function f somehow fits the data well. Like maybe this line shown here. When I see that the line fits the data visually, you can think of this to mean that the line defined by f is roughly passing through or somewhere close to the training examples as compared to other possible lines that are not as close to these points. 

Just to remind you of some notation, a training example like this point here is defined by x superscript i, y superscript i where y is the target. 

![CFF6](./../../Assets/Supervised/RegressionModel/CFF6.png)

For a given input x^i, the function f also makes a predictive value for y and a value that it predicts to y is y hat i shown here. For our choice of a model f of x^i is w times x^i plus b. Stated differently, the prediction y hat i is f of wb of x^i where for the model we're using f of x^i is equal to wx^i plus b. 

![CFF7](./../../Assets/Supervised/RegressionModel/CFF7.png)

Now the question is <u>how do you find values for w and b so that the prediction y hat i is close to the true target y^i for many or maybe all training examples x^i, y^i ?</u>. To answer that question, let's first take a look at how to measure how well a line fits the training data. To do that, we're going to construct a `cost function`. The `cost function takes the prediction y hat and compares it to the target y by taking y hat minus y`. This <u>difference is called the `error`</u>, we're measuring how far off to prediction is from the target. 

![CFF8](./../../Assets/Supervised/RegressionModel/CFF8.png)

Next, let's computes the square of this error. Also, we're going to want to compute this term for different training examples i in the training set. When measuring the error, for example i, we'll compute this squared error term. 

Finally, we want to measure the error across the entire training set. In particular, let's sum up the squared errors like this. We'll sum from i equals 1,2, 3 all the way up to m and remember that m is the number of training examples, which is 47 for this dataset. 

![CFF9](./../../Assets/Supervised/RegressionModel/CFF9.png)

Notice that if we have more training examples m is larger and your cost function will calculate a bigger number. This is summing over more examples. To build a cost function that doesn't automatically get bigger as the training set size gets larger by convention, we will compute the average squared error instead of the total squared error and we do that by dividing by m like this.

![CFF10](./../../Assets/Supervised/RegressionModel/CFF10.png)

We're nearly there. Just one last thing. By convention, the cost function that machine learning people use actually divides by 2 times m. The extra division by 2 is just meant to make some of our later calculations look neater, but the cost function still works whether you include this division by 2 or not. 

This expression right here is the cost function and we're going to write `J of wb` to refer to the cost function. This is also called the `squared error cost function`, and it's called this because you're taking the square of these error terms. In machine learning different people will use different cost functions for different applications, but the `squared error cost function` is by far the most commonly used one for linear regression and for that matter, for all regression problems where it seems to give good results for many applications. 

Just as a reminder, the prediction y hat is equal to the outputs of the model f at x. We can rewrite the cost function `J of wb` as `1 over 2m times the sum from i equals 1 to m of f of x^i minus y^i the quantity squared`.

![CFF11](./../../Assets/Supervised/RegressionModel/CFF11.png)

**Question:**
About the cost function that you just learned, Which of these are the parameters of the model that can be adjusted in the cost function that is used to train the model?
- w,b
- f(X)
- ŷ
> w,b as they are constant

Eventually we're going to want to find values of w and b that make the cost function small. But before going there, let's first gain more intuition about what J of wb is really computing. At this point you might be thinking we've done a whole lot of math to define the cost function. But what exactly is it doing? Let's go on to the next part where you'll step through one example of what the cost function is really computing that I hope will help you build intuition about what it means if J of wb is large versus if the cost j is small. 


