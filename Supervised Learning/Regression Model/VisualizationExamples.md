# Visualization Examples
Let's look at some more visualizations of `w and b`.
Here's one example.

![VE1](./../../Assets/Supervised/RegressionModel/VE1.png)

Over here, you have a particular point on the graph j. For this point, w equals about negative 0.15 and b equals about 800. This point corresponds to one pair of values for w and b that use a particular cost j. In fact, this booklet pair of values for w and b corresponds to this function f of x, which is this line you can see on the left.

![VE2](./../../Assets/Supervised/RegressionModel/VE2.png)

This line intersects the vertical axis at 800 because b equals 800 and the slope of the line is negative 0.15, because w equals negative 0.15.

Now, if you look at the data points in the training set, you may notice that this line is not a good fit to the data. For this function f of x, with these values of w and b, many of the predictions for the value of y are quite far from the actual target value of y that is in the training data.

![VE3](./../../Assets/Supervised/RegressionModel/VE3.png)

Because this line is not a good fit, if you look at the graph of j, the cost of this line is out here, which is pretty far from the minimum. There's a pretty high cost because this choice of w and b is just not that good a fit to the training set.


Now, let's look at another example with a different choice of w and b.

![VE4](./../../Assets/Supervised/RegressionModel/VE4.png)

Now, here's another function that is still not a great fit for the data, but maybe slightly less bad. This points here represents the cost for this booklet pair of w and b that creates that line. The value of w is equal to 0 and the value b is about 360. This pair of parameters corresponds to this function, which is a flat line, because f of x equals 0 times x plus 360. I hope that makes sense.

Let's look at yet another example.

![VE5](./../../Assets/Supervised/RegressionModel/VE5.png)

Here's one more choice for w and b, and with these values, you end up with this line f of x. Again, not a great fit to the data, is actually further away from the minimum compared to the previous example. `Remember that the minimum is at the center of that smallest ellipse`.

Last example,

![VE6](./../../Assets/Supervised/RegressionModel/VE6.png)

if you look at f of x on the left, this looks like a pretty good fit to the training set. You can see on the right, this point representing the cost is very close to the center of the smaller ellipse, it's not quite exactly the minimum, but it's pretty close. For this value of w and b, you get to this line, f of x. You can see that if you measure the vertical distances between the data points and the predicted values on the straight line, you'd get the error for each data point.

![VE7](./../../Assets/Supervised/RegressionModel/VE7.png)

The sum of squared errors for all of these data points is pretty close to the minimum possible sum of squared errors among all possible straight line fits. I hope that by looking at these figures, you can get a better sense of how different choices of the parameters affect the line f of x and how this corresponds to different values for the cost j, and hopefully you can see how the better fit lines correspond to points on the graph of j that are closer to the minimum possible cost for this cost function j of w and b.

In the notebook that follows, you'll get to `run some codes` and `remember all the code is given`, `so you just need to hit Shift Enter to run it` and `take a look at it` and the notebook will show you how the cost function is implemented in code. 

Given a small training set and different choices for the parameters, you'll be able to see how the cost varies depending on how well the model fits the data. In the notebook, you also can play with in interactive console plot. [Check this notebook](Jupyter%20Notebooks/CostFunctionVisualization.ipynb) You can use your mouse cursor to click anywhere on the contour plot and you will see the straight line defined by the values you chose for the parameters w and b. 

You'll see a dot up here also on the 3D surface plot showing the cost. 

Finally, the notebook also has a 3D surface plot that you can manually rotate and spin around using your mouse cursor to take a better look at what the cost function looks like. I hope you'll enjoy playing with the notebooks. 

Now in `linear regression`, `rather than having to manually try to read a contour plot for the best value for w and b, which isn't really a good procedure and also won't work once we get to more complex machine learning models`. `What you really want is an efficient algorithm that you can write in code for automatically finding the values of parameters w and b they give you the best fit line`. `That minimizes the cost function j`. <u> `There is an algorithm for doing this called "gradient descent"`</u>. `This algorithm is one of the most important algorithms in machine learning`. `Gradient descent and variations on gradient descent are used to train, not just linear regression, but some of the biggest and most complex models in all of AI`.
