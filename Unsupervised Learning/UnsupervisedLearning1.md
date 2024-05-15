# Unsupervised Learning Part 1
After `supervised learning`, the most widely used form of machine learning is `unsupervised learning`. Let's take a look at what that means,  Unsupervised learning is I think just as super as supervised learning. When you're studying supervised learning in the supervised learning section recall, it looks something like this in the case of a classification problem. Each example, was associated with an output label y such as benign or malignant, designated by the poles and crosses.
 
![USL1](./../Assets/Unsupervised/USL1.1.png)

In `unsupervised learning We're given data that isn't associated with any output labels y`, say you're given data on patients and their tumor size and the patient's age. But not whether the tumor was benign or malignant, so the dataset looks like this on the right. `We're not asked to diagnose whether the tumor is benign or malignant, because we're not given any labels. But in the dataset, instead, our job is to find some structure or some pattern or just find something interesting in the data.` <u>This is unsupervised learning</u>, we call it `unsupervised because we're not trying to supervise the algorithm`. 

![USL2](./../Assets/Unsupervised/USL1.2.png)

To give some quote right answer for every input, instead, we asked the our room to figure out all by yourself what's interesting. Or what patterns or structures that might be in this data, with this particular data set. An unsupervised learning algorithm, might decide that the data can be assigned to two different groups or two different clusters. And so it might decide, that there's one cluster what group over here, and there's another cluster or group over here. 

![USL3](./../Assets/Unsupervised/USL1.3.png)

This is a particular type of unsupervised learning, called a `clustering algorithm`. Because it places the unlabeled data, into different clusters and this turns out to be used in many applications. 

For example, clustering is used in google news, what google news does is every day it goes. And looks at hundreds of thousands of news articles on the internet, and groups related stories together. For example, 

![USL4](./../Assets/Unsupervised/USL1.4.png)

here is a sample from Google News, where the headline of the top article, is giant panda gives birth to rear twin cubs at Japan's oldest zoo. This article has actually caught my eye, and looking at this, you might notice that below this are other related articles. 

![USL5](./../Assets/Unsupervised/USL1.5.png)

Maybe from the headlines alone, you can start to guess what clustering might be doing. Notice that the word panda appears in all five articles and notice that the word twin also appears in all five articles. And the word Zoo also appears in all of these articles, so the clustering algorithm is finding articles. 



All of all the hundreds of thousands of news articles on the internet that day, finding the articles that mention similar words and grouping them into clusters. Now, what's cool is that this clustering algorithm figures out on his own which words suggest, that certain articles are in the same group. What I mean is there isn't an employee at google news who's telling the algorithm to find articles that the word panda. And twins and zoo to put them into the same cluster, the news topics change every day. And there are so many news stories, it just isn't feasible to people doing this every single day for all the topics that use covers. Instead the algorithm has to figure out on his own without supervision, what are the clusters of news articles today. So that's why this clustering algorithm, is a type of unsupervised learning algorithm. 

![USL6](./../Assets/Unsupervised/USL1.6.png)

Let's look at the second example of unsupervised learning applied to clustering genetic or DNA data. This image shows a picture of DNA micro array data, these look like tiny grids of a spreadsheet. And each tiny column represents the genetic or DNA activity of one person, So for example, this entire Column here is from one person's DNA. And this other column is of another person, each row represents a particular gene. So just as an example, perhaps this role here might represent a gene that affects eye color, or this role here is a gene that affects how tall someone is.


![USL7](./../Assets/Unsupervised/USL1.7.png)

Researchers have even found a genetic link to whether someone dislikes certain vegetables, such as broccoli, or brussels sprouts, or asparagus. So next time someone asks you why didn't you finish your salad, you can tell them, maybe it's genetic for DNA micro race. The idea is to measure how much certain genes, are expressed for each individual person. So these colors red, green, gray, and so on, show the degree to which different individuals do, or do not have a specific gene active. And what you can do is then run a clustering algorithm to group individuals into different categories. 

![USL8](./../Assets/Unsupervised/USL1.8.png)

Or different types of people like maybe these individuals that group together, and let's just call this type one. And these people are grouped into type two, and these people are groups as type three. This is unsupervised learning, because we're not telling the algorithm in advance, that there is a type one person with certain characteristics. Or a type two person with certain characteristics, instead what we're saying is here's a bunch of data. I don't know what the different types of people are but can you automatically find structure into data. And automatically figure out whether the major types of individuals, since we're not giving the algorithm the right answer for the examples in advance. This is unsupervised learning. 

![USL9](./../Assets/Unsupervised/USL1.9.png)

Here's the third example, many companies have huge databases of customer information given this data. Can you automatically group your customers, into different market segments so that you can more efficiently serve your customers. Concretely the [Deep Learning AI](https://www.deeplearning.ai/) team did some research to better understand the deep learningAI community. And why different individuals take these classes, subscribed to the batch weekly newsletter, or attend our AI events. 

Let's visualize the deep learning dot AI community, as this collection of people running clustering. That is market segmentation found a few distinct groups of individuals, one group's primary motivation is seeking knowledge to grow their skills. Perhaps this is you, and so that's great, a second group's primary motivation is looking for a way to develop their career. 

![USL10](./../Assets/Unsupervised/USL1.10.png)

Maybe you want to get a promotion or a new job, or make some career progression if this describes you, that's great too. And yet another group wants to stay updated on how AI impacts their field of work, perhaps this is you, that's great too. This is a clustering that the team used to try to better serve our community as we're trying to figure out. Whether the major categories of learners in the deeper and community, So if any of these is your top motivation for learning, that's great. And I hope this will help  you on your journey, or in case this is you,

![USL11](./../Assets/Unsupervised/USL1.11.png)

and you want something totally different than the other three categories. That's fine too, so to summarize a `clustering algorithm` <u> Which is a type of unsupervised learning algorithm</u>, `takes data without labels and tries to automatically group them into clusters.` And so maybe the next time you see or think of a panda, maybe you think of clustering as well. And besides clustering, there are other types of unsupervised learning as well.







