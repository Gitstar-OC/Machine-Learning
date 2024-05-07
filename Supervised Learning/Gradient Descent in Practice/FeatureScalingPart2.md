# Feature Scaling Part 2

Let's look at how you can implement feature scaling, to take features that take on very different ranges of values and skill them to have comparable ranges of values to each other. 

How do you actually scale features? Well, if x_1 ranges from 3-2,000, one way to get a scale version of x_1 is to `take each original x1_ value and divide by` 2,000, `the maximum of the range`.

![FSP2.1](./../../Assets/Supervised/GDP/FSP2.1.png)

The scale x_1 will range from 0.15 up to one. 

Similarly, since x_2 ranges from 0-5, you can calculate a scale version of x_2 by taking each original x_2 and dividing by five, which is again the maximum.

![FSP2.2](./../../Assets/Supervised/GDP/FSP2.2.png)

So the scale is x_2 will now range from 0-1. If you plot the scale to x_1 and x_2 on a graph, it might look like this. 

![FSP2.3](./../../Assets/Supervised/GDP/FSP2.3.png)

In addition to dividing by the maximum, you can also do what's called `mean normalization`. 

![FSP2.4](./../../Assets/Supervised/GDP/FSP2.4.png)

`What this looks like is, you start with the original features and then you re-scale them so that both of them are centered around zero`.

![FSP2.5](./../../Assets/Supervised/GDP/FSP2.5.png)

`Whereas before they only had values greater than zero, now they have both negative and positive values that may be usually between negative one and plus one`.

`To calculate the mean normalization` of x_1, `first find the average`, also called the mean of x_1 on your training set, and let's call this `mean M_1`, with this being the `Greek alphabets Mu`. For example, you may find that the average of feature 1, Mu_1 is 600 square feet. 

![FSP2.6](./../../Assets/Supervised/GDP/FSP2.6.png)

Let's `take each x_1, subtract the mean Mu_1`, and `then` let's `divide by the difference 2,000 minus 300, where 2,000 is the maximum and 300 the minimum`, and if you do this, you get the normalized x_1 to range from negative 0.18-0.82. 

Similarly, to mean normalized x_2, you can calculate the average of feature 2. For instance, Mu_2 may be 2.3. Then you can take each x_2, subtract Mu_2 and divide by 5 minus 0.

![FSP2.7](./../../Assets/Supervised/GDP/FSP2.7.png)

Again, the max 5 minus the mean, which is 0. The mean normalized x_2 now ranges from `negative 0.46-0.54`. If you plot the training data using the mean normalized x_1 and x_2, it might look like the rescaled graph.

There's one last common re-scaling method call `Z-score normalization`. `To implement Z-score normalization, you need to calculate something called the standard deviation of each feature`.

![FSP2.8](./../../Assets/Supervised/GDP/FSP2.8.png)

>If you don't know what the standard deviation is, don't worry about it, you won't need to know it for this course. Or if you've heard of the normal distribution or the bell-shaped curve, sometimes also called the Gaussian distribution, 
![FSP2.9](./../../Assets/Supervised/GDP/FSP2.9.png)this is what the standard deviation for the normal distribution looks like. But if you haven't heard of this, you don't need to worry about that either. But if you do know what is the standard deviation, then to implement a Z-score normalization, you first calculate the mean `Mu`, as well as the standard deviation, which is often denoted by the `lowercase Greek alphabet Sigma` of each feature. 

For instance, maybe feature 1 has a standard deviation of 450 and mean 600, then to Z-score normalize x_1, take each x_1, subtract Mu_1, and then divide by the standard deviation, which I'm going to denote as Sigma 1. What you may find is that the Z-score normalized x_1 now ranges from negative 0.67-3.1.


![FSP2.10](./../../Assets/Supervised/GDP/FSP2.10.png)

Similarly, if you calculate the second features standard deviation to be 1.4 and mean to be 2.3, then you can compute x_2 minus Mu_2 divided by Sigma_2, and in this case, the Z-score normalized by x_2 might now range from `negative 1.6-1.9`.

![FSP2.11](./../../Assets/Supervised/GDP/FSP2.11.png)

If you plot the training data on the normalized x_1 and x_2 on a graph, it might look like this. 

![FSP2.12](./../../Assets/Supervised/GDP/FSP2.12.png)

`As a rule of thumb, when performing feature scaling, you might want to aim for getting the features to range from maybe anywhere around negative one to somewhere around plus one for each feature x`. 

But these values, negative one and plus one can be a little bit loose. If the features range from `negative three to plus three or negative 0.3 to plus 0.3`, all of these are `completely okay`. If you have a feature x_1 that winds up being `between zero and three`, that's `not a problem`. You can re-scale it if you want, but if you don't re-scale it, it should work okay too. Or if you have a different feature, x_2, whose values are between `negative 2 and plus 0.5`, again, `that's okay`, no harm re-scaling it, but it might be okay if you leave it alone as well. 

![FSP2.13](./../../Assets/Supervised/GDP/FSP2.13.png)

But if another feature, like x_3 here, ranges from `negative 100 to plus 100`, `then this takes on a very different range of values, say something from around negative one to plus one. You're probably better off re-scaling this feature x_3 so that it ranges from something closer to negative one to plus one`. Similarly, if you have a feature x_4 that takes on really small values, say between `negative 0.001 and plus 0.001`, `then these values are so small`. That means `you may need to re-scale it as well`. Finally, what if your feature x_5, such as measurements of a hospital patients by the temperature ranges from `98.6-105` degrees Fahrenheit? In this case, these values are around 100, which is actually pretty large compared to other scale features, and `this will actually cause gradient descent to run more slowly`. In this case, `feature re-scaling will likely help`. 

![FSP2.14](./../../Assets/Supervised/GDP/FSP2.14.png)

There's almost never any harm to carrying out `feature re-scaling`. When in doubt, I encourage you to just carry it out. That's it for `feature scaling`. With this little technique, you'll often be able to get gradient descent to run much faster. That's features scaling. With or without feature scaling, when you run gradient descent, how can you know, how can you check if gradient descent is really working? If it is finding you the global minimum or something close to it. In the next part, let's take a look at how to recognize if `gradient descent is converging`, and then in the later part after that, this will lead to discussion of how to choose a good learning rate for gradient descent.
