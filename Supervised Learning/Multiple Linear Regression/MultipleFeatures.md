# Multiple Features 
Welcome back. In this part, you'll learn to make linear regression much faster and much more powerful. 

Let's start by looking at the version of linear regression that look at not just one feature, but a lot of different features. Let's take a look. 

![MF1](./../../Assets/Supervised/MLR/MF1.png)

`In the original version of linear regression, you had a single feature x, the size of the house and you're able to predict y, the price of the house`. `The model was fwb of x equals wx plus b`. But now, what if you did not only have the size of the house as a feature with which to try to predict the price, but if you also knew the number of bedrooms, the number of floors and the age of the home in years. It seems like this would give you a lot more information with which to predict the price. 

![MF2](./../../Assets/Supervised/MLR/MF2.png)

To introduce a little bit of new notation, `we're going to use the variables X_1, X_2, X_3 and X_4, to denote the four features`. For simplicity, let's introduce a little bit more notation. 

We'll write X subscript j or sometimes I'll just say for short, `X sub j`, to represent the list of features. Here, j will go from one to four, because we have four features. I'm going to use `lowercase n to denote the total number of features`, so in this example, n is equal to 4. 

![MF3](./../../Assets/Supervised/MLR/MF3.png)

As before, we'll use `X superscript i to denote the ith training example`. Here `X superscript i is actually going to be a list of four numbers`, `or` sometimes we'll call this a `vector that includes all the features of the ith training example`. As a concrete example, `X superscript in parentheses 2, will be a vector of the features for the second training example`, so it will equal to this 1416, 3, 2 and 40 and technically, 

![MF5](./../../Assets/Supervised/MLR/MF5.png)

`I'm writing these numbers in a row, so sometimes this is called a row vector rather than a column vector`. But if you don't know what the difference is, don't worry about it, it's not that important for this purpose. `To refer to a specific feature in the ith training example, I will write X superscript i, subscript j, so for example, X superscript 2 subscript 3 will be the value of the third feature, that is the number of floors in the second training example and so that's going to be equal to 2`. `Sometimes in order to emphasize that this X^2 is not a number but is actually a list of numbers that is a vector`, we'll draw an arrow on top of that just to visually show that is a vector and over here as well, but you don't have to draw this arrow in your notation. You can think of the arrow as an optional signifier. They're sometimes used just to emphasize that this is a vector and not a number. 

Now that we have multiple features, let's take a look at what a model would look like. Previously, this is how we defined the model, where X was a single feature, so a single number. 

![MF6](./../../Assets/Supervised/MLR/MF6.png)

`But now with multiple features, we're going to define it differently. Instead, the model will be, fwb of X equals w1x1 plus w2x2 plus w3x3 plus w4x4 plus b`.

>Concretely for housing price prediction, one possible model may be that we estimate the price of the house as 0.1 times X_1, the size of the house, plus four times X_2, the number of bedrooms, plus ten times X_3, the number of floors, minus 2 times X_4, the age of the house in years plus 80. 
![MF7](./../../Assets/Supervised/MLR/MF7.png)
Let's think a bit about how you might interpret these parameters. If the model is trying to predict the price of the house in thousands of dollars, you can think of this b equals 80 as saying that the base price of a house starts off at maybe $80,000, assuming it has no size, no bedrooms, no floor and no age. You can think of this 0.1 as saying that maybe for every additional square foot, the price will increase by 0.1 $1,000 or by $100, because we're saying that for each square foot, the price increases by 0.1, times $1,000, which is $100. Maybe for each additional bedroom, the price increases by $4,000 and for each additional floor the price may increase by $10,000 and for each additional year of the house's age, the price may decrease by $2,000, because the parameter is negative 2. 

In general, if you have n features, then the model will look like this. Here again is the definition of the model with n features. What we're going to do next is introduce a little bit of notation to rewrite this expression in a simpler but equivalent way. Let's define W as a list of numbers that list the parameters W_1, W_2, W_3, all the way through W_n. 

![MF8](./../../Assets/Supervised/MLR/MF8.png)

`In mathematics, this is called a vector and sometimes to designate that this is a vector, which just means a list of numbers, I'm going to draw a little arrow on top`. You don't always have to draw this arrow and you can do so or not in your own notation, so you can think of this little arrow as just an optional signifier to remind us that this is a vector. If you've taken the linear algebra class before, you might recognize that this is a row vector as opposed to a column vector. But if you don't know what those terms means, you don't need to worry about it. Next, same as before, b is a single number and not a vector and so this vector W together with this number b are the parameters of the model. 

Let me also write X as a list or a vector, again a row vector that lists all of the features X_1, X_2, X_3 up to X_n, this is again a vector, so I'm going to add a little arrow up on top to signify. In the notation up on top, we can also add little arrows here and here to signify that that W and that X are actually these lists of numbers, that they're actually these vectors. 

![MF9](./../../Assets/Supervised/MLR/MF9.png)

With this notation, the model can now be rewritten more succinctly as `f of x equals, the vector w dot and this dot refers to a dot product from linear algebra of X the vector, plus the number b`. 

What is this dot product thing? Well, the dot products of two vectors of two lists of numbers W and X, is computed by checking the corresponding pairs of numbers, W_1 and X_1 multiplying that, W_2 X_2 multiplying that, W_3 X_3 multiplying that, all the way up to W_n and X_n multiplying that and then summing up all of these products. 

![MF10](./../../Assets/Supervised/MLR/MF10.png)

Writing that out, this means that the dot products is equal to W_1X_1 plus W_2X_2 plus W_3X_3 plus all the way up to W_nX_n. Then finally we add back in the b on top. You notice that this gives us exactly the same expression as we had on top. `The dot traffic notation lets you write the model in a more compact form with fewer characters`. The name for this type of `linear regression model with multiple input features` is `multiple linear regression`. This is in contrast to univariate regression, which has just one feature. `By the way, you might think this algorithm is called multivariate regression, but that term actually refers to something else that we won't be using here`. I'm going to refer to this model as multiple linear regression. That's it for linear regression with multiple features, which is also called multiple linear regression. In order to implement this, there's a really neat trick called vectorization, which will make it much simpler to implement this and many other learning algorithms.

<!-- 
![MF4](./../../Assets/Supervised/MLR/MF4.png) -->
