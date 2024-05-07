# Vectorization Part 2

I remember when I first learned about vectorization, I spent many hours on my computer taking an unvectorized version of an algorithm running it, see how long it run, and then running a vectorized version of the code and seeing how much faster that run, and I just spent hours playing with that. And it frankly blew my mind that the same algorithm vectorized would run so much faster. It felt almost like a magic trick to me. In this part, let's figure out how this magic trick really works. 

Let's take a deeper look at how a vectorized implementation may work on your computer behind the scenes. Let's look at this for loop. 

![VD1](./../../Assets/Supervised/MLR/VD1.png)

The for loop like this runs `without vectorization`. If j ranges from 0 to say 15, this piece of code performs operations one after another. On the first timestamp which I'm going to write as t0. It first operates on the values at index 0. At the next time-step, it calculates values corresponding to index 1 and so on until the 15th step, where it computes that. `In other words, it calculates these computations one step at a time, one step after another`. 

`In contrast, this function in NumPy is implemented in the computer hardware with vectorization`. The computer can get all values of the vectors w and x, and in a single-step, it multiplies each pair of w and x with each other all at the same time in parallel. Then after that, the computer takes these 16 numbers and uses specialized hardware to add them altogether very efficiently, rather than needing to carry out distinct additions one after another to add up these 16 numbers.

![VD2](./../../Assets/Supervised/MLR/VD2.png)

`This means that codes with vectorization can perform calculations in much less time than codes without vectorization`. This matters more when you're running algorithms on large data sets or trying to train large models, which is often the case with machine learning. That's why being able to vectorize implementations of learning algorithms, has been a key step to getting learning algorithms to run efficiently, and therefore scale well to large datasets that many modern machine learning algorithms now have to operate on. 

Now, let's take a look at a concrete example of how this helps with implementing multiple linear regression and this linear regression with multiple input features. Say you have a problem with 16 features and 16 parameters, w1 through w16, in addition to the parameter b. 

![VD3](./../../Assets/Supervised/MLR/VD3.png)

You calculate it 16 derivative terms for these 16 weights and codes, maybe you store the values of w and d in two np.arrays, with d storing the values of the derivatives. For this example, I'm just going to ignore the parameter b. Now, you want to compute an update for each of these 16 parameters. W_j is updated to w_j minus the learning rate, say 0.1, times d_j, for j from 1 through 16. 

Encodes without vectorization, you would be doing something like this. 

![VD4](./../../Assets/Supervised/MLR/VD4.png)

Update w1 to be w1 minus the learning rate 0.1 times d1, next, update w2 similarly, and so on through w16, updated as w16 minus 0.1 times d16. Encodes without vectorization, you can use a for loop like this for j in range 016, that again goes from 0-15, said w_j equals w_j minus 0.1 times d_j. 

In contrast, with `factorization`, you can imagine the computer's parallel processing hardware like this. It takes all 16 values in the vector w and subtracts in parallel, 0.1 times all 16 values in the vector d, and assign all 16 calculations back to w all at the same time and all in one step.

In code, you can implement this as follows
​
![VD5](./../../Assets/Supervised/MLR/VD5.png)

w is assigned to `w minus 0.1 times d`. Behind the scenes, the computer takes these NumPy arrays, w and d, and uses parallel processing hardware to carry out all 16 computations efficiently. Using a vectorized implementation, you should get a much more efficient implementation of linear regression. Maybe the speed difference won't be huge if you have 16 features, but if you have thousands of features and perhaps very large training sets, this type of vectorized implementation will make a huge difference in the running time of your learning algorithm. It could be the difference between codes finishing in one or two minutes, versus taking many hours to do the same thing. 

In the notebook that follows this part, you see an introduction to one of the most used Python libraries and Machine Learning, which we've already touched on in this part called `NumPy`. You see how they create vectors encode and these vectors or lists of numbers are called `NumPy arrays`, and you also see how to take the dot product of two vectors using a NumPy function called dot. You also get to see how vectorized code such as using the dot function, can run much faster than a for-loop. In fact, you'd get to time this code yourself, and hopefully see it run much faster. This notebook introduces a fair amount of new NumPy syntax, so when you read through the notebook, please still feel like you have to understand all the code right away, but you can save this notebook and use it as a reference to look at when you're working with data stored in NumPy arrays. 

Congrats on finishing this video on vectorization. You've learned one of the most important and useful techniques in implementing machine learning algorithms. In the next part, we'll put the math of multiple linear regression together with vectorization, so that you will influence gradient descent for multiple linear regression with vectorization.