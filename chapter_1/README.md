# Hands-On Machine Learning with Scikit-Learn and PyTorch

In 2006, Geoffrey Hinton et al. published a paper⁠⁠ showing how to train a deep neural network capable of recognizing handwritten digits with state-of-the-art precision (>98%). They branded this technique “deep learning”. A deep neural network is a (very) simplified model of our cerebral cortex, composed of a stack of layers of artificial neurons. Training a deep neural net was widely considered impossible at the time,⁠and most researchers had abandoned the idea in the late 1990s. This paper revived the interest of the scientific community, and before long many new papers demonstrated that deep learning was not only possible, but capable of mind-blowing achievements that no other machine learning (ML) technique could hope to match (with the help of tremendous computing power and great amounts of data). This enthusiasm soon extended to many other areas of machine learning.

A decade later, machine learning had already conquered many industries, ranking web results, recommending videos to watch and products to buy, sorting items on production lines, sometimes even driving cars. Machine learning often made the headlines, for example when DeepMind’s AlphaFold machine learning system solved a long-standing protein-folding problem that had stomped researchers for decades. But most of the time, machine learning was just working discretely in the background. However, another decade later came the rise of AI assistants: from ChatGPT in 2022, Gemini, Claude, and Grok in 2023, and many others since then. AI has now truly taken off and it is rapidly transforming every single industry: what used to be sci-fi is now very real.⁠

## QUESTIONS

- What is a neural network?
- What event made people enthusiastic again about Machine Learning in 2006

## Tools

- Google Colab: free service that allows you to run any Jupyter notebook directly online without having to install anything on your machine.

## Install the required libraries and tools (or the Docker image)

[Install Required Libraries](https://github.com/ageron/handson-mlp/blob/main/INSTALL.md)

[Run the notebooks in a Docker container](https://github.com/ageron/handson-mlp/blob/main/docker/README.md)

[Machine Learning Notebooks](https://github.com/ageron/handson-mlp/blob/main/index.ipynb)

## Caution

Don’t jump into deep waters too hastily: deep learning is no doubt one of the most exciting areas in machine learning, but you should master the fundamentals first. Moreover, many problems can be solved quite well using simpler techniques such as random forests and ensemble methods. Deep learning is best suited for complex problems such as image recognition, speech recognition, or natural language processing, and it often requires a lot of data, computing power, and patience (unless you can leverage a pretrained neural network)

# Chapter 1. The Machine Learning Landscape

ML has actually been around for decades in some specialized applications, such as optical character recognition (OCR). The first ML application that really became mainstream, improving the lives of hundreds of millions of people, discretely took over the world back in the 1990s: the spam filter. It’s not exactly a self-aware robot, but it does technically qualify as machine learning: it has actually learned so well that you seldom need to flag an email as spam anymore. Then thanks to big data, hardware improvements, and a few algorithmic innovations, hundreds of ML applications followed and now quietly power hundreds of products and features that you use regularly: voice prompts, automatic translation, image search, product recommendations, and many more. And finally came ChatGPT, Gemini (formerly Bard), Claude, Perplexity, and many other chatbots: AI is no longer just powering services in the background, it is the service itself.

## What is Machine Learning

* Machine learning is the science (and art) of programming computers so they can learn from data.

* [Machine learning is the] field of study that gives computers the ability to learn without being explicitly programmed.

* A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.

Your spam filter is a machine learning program that, given examples of spam emails (flagged by users) and examples of regular emails (nonspam, also called “ham”), can learn to flag spam. The examples that the system uses to learn are called the training set. Each training example is called a training instance (or sample). The part of a machine learning system that learns and makes predictions is called a model. Neural networks and random forests are examples of models.

In this case, the task T is to flag spam for new emails, the experience E is the training data, and the performance measure P needs to be defined; for example, you can use the ratio of correctly classified emails. This particular performance measure is called accuracy, and it is often used in classification tasks.

If you just download a copy of all Wikipedia articles, your computer has a lot more data, but it is not suddenly better at any task. This is not machine learning.

## Why Use Machine Learning?

Consider how you would write a spam filter using traditional programming techniques.

1. First you would examine what spam typically looks like. You might notice that some words or phrases (such as “4U”, “credit card”, “free”, and “amazing”) tend to come up a lot in the subject line. Perhaps you would also notice a few other patterns in the sender’s name, the email’s body, and other parts of the email.

2. You would write a detection algorithm for each of the patterns that you noticed, and your program would flag emails as spam if a number of these patterns were detected.

3. You would test your program and repeat steps 1 and 2 until it was good enough to launch.

![The traditional approach](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9798341607972/files/assets/hmls_0101.png)

Since the problem is difficult, your program will likely become a long list of complex rules—pretty hard to maintain.

In contrast, a spam filter based on machine learning techniques automatically learns which words and phrases are good predictors of spam by detecting unusually frequent patterns of words in the spam examples compared to the ham examples. The program is much shorter, easier to maintain, and most likely more accurate.

![The machine learning approach](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9798341607972/files/assets/hmls_0102.png)

What if spammers notice that all their emails containing “4U” are blocked? They might start writing “For U” instead. A spam filter using traditional programming techniques would need to be updated to flag “For U” emails. If spammers keep working around your spam filter, you will need to keep writing new rules forever.

In contrast, a spam filter based on machine learning techniques automatically notices that “For U” has become unusually frequent in spam flagged by users, and it starts flagging them without your intervention.

![Automatically adapting to change](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9798341607972/files/assets/hmls_0103.png)


Another area where machine learning shines is for problems that either are too complex for traditional approaches or have no known algorithm. For example, consider speech recognition. Say you want to start simple and write a program capable of distinguishing the words “one” and “two”. You might notice that the word “two” starts with a high-pitch sound (“T”), so you could hardcode an algorithm that measures high-pitch sound intensity and use that to distinguish ones and twos⁠—but obviously this technique will not scale to thousands of words spoken by millions of very different people in noisy environments and in dozens of languages. The best solution (at least today) is to write an algorithm that learns by itself, given many example recordings for each word.

Finally, machine learning can help humans learn (Figure 1-4). ML models can be inspected to see what they have learned (although for some models this can be tricky). For instance, once a spam filter has been trained on enough spam, it can easily be inspected to reveal the list of words and combinations of words that it believes are the best predictors of spam. Sometimes this will reveal unsuspected correlations or new trends, and thereby lead to a better understanding of the problem. Digging into large amounts of data to discover hidden patterns is called **data mining**, and machine learning excels at it.

![Machine Learning can help humans learn](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9798341607972/files/assets/hmls_0104.png)

To summarize, machine learning is great for:

* Problems for which existing solutions require a lot of work and maintenance, such as long lists of rules (a machine learning model can often simplify code and perform better than the traditional approach)

* Complex problems for which using a traditional approach yields no good solution (the best machine learning techniques can perhaps find a solution)

* Fluctuating environments (a machine learning system can easily be retrained on new data, always keeping it up to date)

* Getting insights about complex problems and large amounts of data

## Examples of Applications


Let’s look at some concrete examples of machine learning tasks, along with the techniques that can tackle them:

*Analyzing images of products on a production line to automatically classify them*
> This is image classification, typically performed using convolutional neural networks  or vision transformers (see Chapter 16).

---

*Detecting tumors in brain scans*
> This is semantic image segmentation, where each pixel in the image is classified (as we want to determine the exact location and shape of tumors), typically using CNNs or vision transformers.

---

*Automatically classifying news articles*

> This is natural language processing (NLP), and more specifically text classification, which can be tackled using recurrent neural networks (RNNs) and CNNs, but transformers work even better.

---

*Automatically flagging offensive comments on discussion forums*
> This is also text classification, using the same NLP tools.

---

*Summarizing long documents automatically*
> This is a branch of NLP called text summarization, again using the same tools.

---

*Estimating a person’s genetic risk for a given disease by analyzing a very long DNA sequence*

> Such a task requires discovering spread out patterns across very long sequences, which is where state space models (SSMs) particularly shine (see “State-Space Models (SSMs)” at [https://homl.info](https://ageron.github.io/)).

---

*Creating a chatbot or a personal assistant*

> This involves many NLP components, including natural language understanding (NLU) and question-answering modules.

---

*Forecasting your company’s revenue next year, based on many performance metrics*

> This is a regression task (i.e., predicting values) that may be tackled using any regression model, such as a linear regression or polynomial regression model, a regression support vector machine, a regression random forest, or an artificial neural network. If you want to take into account sequences of past performance metrics, you may want to use RNNs, CNNs, or transformers.

---

*Making your app react to voice commands*

> This is speech recognition, which requires processing audio samples. Since they are long and complex sequences, they are typically processed using RNNs, CNNs, or transformers.

---

*Detecting credit card fraud*

> This is anomaly detection, which can be tackled using isolation forests, Gaussian mixture models, or autoencoders.

---

*Segmenting clients based on their purchases so that you can design a different marketing strategy for each segment*

> This is clustering, which can be achieved using k-means, DBSCAN, and more.

---

*Representing a complex, high-dimensional dataset in a clear and insightful diagram*

> This is data visualization, often involving dimensionality reduction techniques.

---

*Recommending a product that a client may be interested in, based on past purchases*

> This is a recommender system. One approach is to feed past purchases (and other information about the client) to an artificial neural network, and get it to output the most likely next purchase. This neural net would typically be trained on past sequences of purchases across all clients.

*Building an intelligent bot for a game*

> This is often tackled using reinforcement learning, which is a branch of machine learning that trains agents (such as bots) to pick the actions that will maximize their rewards over time (e.g., a bot may get a reward every time the player loses some life points), within a given environment (such as the game). The famous AlphaGo program that beat the world champion at the game of Go was built using RL.

## Types of Machine Learning Systems

There are so many different types of machine learning systems that it is useful to classify them in broad categories, based on the following criteria:

* How they are guided during training (supervised, unsupervised, semi-supervised, self-supervised, and others)

* Whether or not they can learn incrementally on the fly (online versus batch learning)

* Whether they work by simply comparing new data points to known data points, or instead by detecting patterns in the training data and building a predictive model, much like scientists do (instance-based versus model-based learning)

These criteria are not exclusive; you can combine them in any way you like. For example, a state-of-the-art spam filter may learn on the fly using a deep neural network model trained using human-provided examples of spam and ham; this makes it an online, model-based, supervised learning system.

Let’s look at each of these criteria a bit more closely.

### Training Supervision

ML systems can be classified according to the amount and type of supervision they get during training. There are many categories, but we’ll discuss the main ones: supervised learning, unsupervised learning, self-supervised learning, semi-supervised learning, and reinforcement learning.

---

#### Supervised learning

In supervised learning, the training set you feed to the algorithm includes the desired solutions, called labels.

![A labeled training set for spam classification (an example of supervised learning)](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9798341607972/files/assets/hmls_0105.png)

A typical supervised learning task is classification. The spam filter is a good example of this: it is trained with many example emails along with their class (spam or ham), and it must learn how to classify new emails.

Another typical task is to predict a target numeric value, such as the price of a car, given a set of features (mileage, age, brand, etc.). This sort of task is called regression.⁠1 To train the system, you need to give it many examples of cars, including both their features and their targets (i.e., their prices).

Note that some regression models can be used for classification as well, and vice versa. For example, logistic regression is commonly used for classification, as it can output a value that corresponds to the probability of belonging to a given class (e.g., 20% chance of being spam).

![A regression problem: predict a value, given an input feature (there are usually multiple input features, and sometimes multiple output values)](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9798341607972/files/assets/hmls_0106.png)

> *Note:* The words *target* and *label* are generally treated as synonyms in supervised learning, but *target* is more common in regression tasks and *label* is more common in classification tasks. Moreover, *features* are sometimes called *predictors* or *attributes*. These terms may refer to individual samples (e.g., “this car’s mileage feature is equal to 15,000”) or to all samples (e.g., “the mileage feature is strongly correlated with price”).

---

#### Unsupervised learning

In *unsupervised learning*, as you might guess, the training data is unlabeled. The system tries to learn without a teacher.

For example, say you have a lot of data about your blog’s visitors. You may want to run a *clustering* algorithm to try to detect groups of similar visitors. The features may include the user’s age group, their region, their interests, the duration of their sessions, and so on. At no point do you tell the algorithm which group a visitor belongs to: it finds those connections without your help. For example, it might notice that 40% of your visitors are teenagers who love comic books and generally read your blog after school, while 20% are adults who enjoy sci-fi and who visit during the weekends. If you use a *hierarchical clustering* algorithm, it may also subdivide each group into smaller groups. This may help you target your posts for each group.

![Clustering](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9798341607972/files/assets/hmls_0107.png)

*Visualization* algorithms are also good examples of unsupervised learning: you feed them a lot of complex and unlabeled data, and they output a 2D or 3D representation of your data that can easily be plotted (Figure 1-8). These algorithms try to preserve as much structure as they can (e.g., trying to keep separate clusters in the input space from overlapping in the visualization) so that you can understand how the data is organized and perhaps identify unsuspected patterns.

A related task is *dimensionality reduction*, in which the goal is to simplify the data without losing too much information. One way to do this is to merge several correlated features into one. For example, a car’s mileage may be strongly correlated with its age, so the dimensionality reduction algorithm will merge them into one feature that represents the car’s wear and tear. This is called *feature extraction*.

> **TIP**: It is often a good idea to try to reduce the number of dimensions in your training data using a dimensionality reduction algorithm before you feed it to another machine learning algorithm (such as a supervised learning algorithm). It will run much faster, the data will take up less disk and memory space, and in some cases it may also perform better.

![Example of a t-SNE visualization highlighting semantic clusters⁠](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9798341607972/files/assets/hmls_0108.png)

Notice how animals are rather well separated from vehicles and how horses are close to deer but far from birds.

Yet another important unsupervised task is anomaly detection—for example, detecting unusual credit card transactions to prevent fraud, catching manufacturing defects, or automatically removing outliers from a dataset before feeding it to another learning algorithm. The system is shown mostly normal instances during training, so it learns to recognize them; then, when it sees a new instance, it can tell whether it looks like a normal one or whether it is likely an anomaly.

The features may include distance from home, time of day, day of the week, amount withdrawn, merchant category, transaction frequency, etc. A very similar task is **novelty detection**: it aims to detect new instances that look different from all instances in the training set. This requires having a very “clean” training set, devoid of any instance that you would like the algorithm to detect. For example, if you have thousands of pictures of dogs, and 1% of these pictures represent Chihuahuas, then a **novelty detection** algorithm should not treat new pictures of Chihuahuas as novelties. On the other hand, anomaly detection algorithms may consider these dogs as so rare and so different from other dogs that they would likely classify them as anomalies (no offense to Chihuahuas).

![Anomaly detection](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9798341607972/files/assets/hmls_0109.png)

Finally, another common unsupervised task is **association rule learning**, in which the goal is to dig into large amounts of data and discover interesting relations between attributes. For example, suppose you own a supermarket. Running an association rule on your sales logs may reveal that people who purchase barbecue sauce and potato chips also tend to buy steak. Thus, you may want to place these items close to one another.

---

#### Semi-supervised learning

Since labeling data is usually time-consuming and costly, you will often have plenty of unlabeled instances, and few labeled instances. Some algorithms can deal with data that’s partially labeled. This is called **semi-supervised learning**.

![Semi-supervised learning with two classes (triangles and squares): the unlabeled examples (circles) help classify a new instance (the cross) into the triangle class rather than the square class, even though it is closer to the labeled squares](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9798341607972/files/assets/hmls_0110.png)


Some photo-hosting services, such as Google Photos, are good examples of this. Once you upload all your family photos to the service, it automatically recognizes that the same person A shows up in photos 1, 5, and 11, while another person B shows up in photos 2, 5, and 7. This is the unsupervised part of the algorithm (clustering). Now all the system needs is for you to tell it who these people are. Just add one label per person⁠ and it is able to name everyone in every photo, which is useful for searching photos.


That’s when the system works perfectly. In practice it often creates a few clusters per person, and sometimes mixes up two people who look alike, so you may need to provide a few labels per person and manually clean up some clusters.

Most semi-supervised learning algorithms are combinations of unsupervised and supervised algorithms. For example, a clustering algorithm may be used to group similar instances together, and then every unlabeled instance can be labeled with the most common label in its cluster. Once the whole dataset is labeled, it is possible to use any supervised learning algorithm.

---

#### Self-supervised learning

Another approach to machine learning involves actually generating a fully labeled dataset from a fully unlabeled one. Again, once the whole dataset is labeled, any supervised learning algorithm can be used. This approach is called **self-supervised learning**.

For example, if you have a large dataset of unlabeled images, you can randomly mask a small part of each image and then train a model to recover the original image. During training, the masked images are used as the inputs to the model, and the original images are used as the labels.

![Self-supervised learning example: input (left) and target (right)](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9798341607972/files/assets/hmls_0111.png)


The resulting model may be quite useful in itself—for example, to repair damaged images or to erase unwanted objects from pictures. But more often than not, a model trained using self-supervised learning is not the final goal. You’ll usually want to tweak and fine-tune the model for a slightly different task—one that you actually care about.

For example, suppose that what you really want is to have a pet classification model: given a picture of any pet, it will tell you what species it belongs to. If you have a large dataset of unlabeled photos of pets, you can start by training an image-repairing model using self-supervised learning. Once it’s performing well, it should be able to distinguish different pet species: when it repairs an image of a cat whose face is masked, it must know not to add a dog’s face. Assuming your model’s architecture allows it (and most neural network architectures do), it is then possible to tweak the model so that it predicts pet species instead of repairing images. The final step consists of fine-tuning the model on a labeled dataset: the model already knows what cats, dogs, and other pet species look like, so this step is only needed so the model can learn the mapping between the species it already knows and the labels we expect from it.

> **NOTE:** Transferring knowledge from one task to another is called ***transfer learning***, and it’s one of the most important techniques in machine learning today, especially when using deep neural networks (i.e., neural networks composed of many layers of neurons). 

LLMs are trained in a very similar way, by masking random words in a huge text corpus and training the model to predict the missing words. This large pretrained model can then be fine-tuned for various applications, from sentiment analysis to chatbots.

Some people consider self-supervised learning to be a part of unsupervised learning, since it deals with fully unlabeled datasets. But self-supervised learning uses (generated) labels during training, so in that regard it’s closer to supervised learning. And the term “unsupervised learning” is generally used when dealing with tasks like clustering, dimensionality reduction, or anomaly detection, whereas self-supervised learning focuses on the same tasks as supervised learning: mainly classification and regression. In short, it’s best to treat self-supervised learning as its own category.

---

#### Reinforcement learning

Reinforcement learning is a very different beast. The learning system, called an agent in this context, can observe the environment, select and perform actions, and get rewards in return (or ***penalties*** in the form of negative rewards). It must then learn by itself what is the best strategy, called a ***policy***, to get the most reward over time. A ***policy*** defines what action the agent should choose when it is in a given situation.


![Reinforcement learning](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9798341607972/files/assets/hmls_0112.png)

For example, many robots implement reinforcement learning algorithms to learn how to walk. DeepMind’s AlphaGo program is also a good example of reinforcement learning: it made the headlines in May 2017 when it beat Ke Jie, the number one ranked player in the world at the time, at the game of Go. It learned its winning policy by analyzing millions of games, and then playing many games against itself. Note that learning was turned off during the games against the champion; AlphaGo was just applying the policy it had learned. This is called ***offline learning***.

---

## Batch Versus Online Learning

Another criterion used to classify machine learning systems is whether the system can learn incrementally from a stream of incoming data. For example, random forests can only be trained from scratch on the full dataset—this is called batch learning—while other models can be trained one batch of data at a time, for example, using ***gradient descent*** —this is called online learning.

---

### Batch learning

In **batch learning**, the system must be trained using all the available data. This will generally take a lot of time and computing resources, so it is typically done offline. First the system is trained, and then it is launched into production and runs without learning anymore; it just applies what it has learned. This is called **offline learning**.

Unfortunately, a model’s performance tends to decay slowly over time, simply because the world continues to evolve while the model remains unchanged. This phenomenon is often called **data drift** (or **model rot**). The solution is to regularly retrain the model on up-to-date data. How often you need to do that depends on the use case: if the model classifies pictures of cats and dogs, its performance will decay very slowly, but if the model deals with fast-evolving systems, for example making predictions on the financial market, then it is likely to decay quite fast.

> WARNING: Even a model trained to classify pictures of cats and dogs may need to be retrained regularly, not because cats and dogs will mutate overnight, but because cameras keep changing, along with image formats, sharpness, brightness, and size ratios. Moreover, people may love different breeds next year, or they may decide to dress their pets with tiny hats—who knows?

If you want a batch learning system to know about new data (such as a new type of spam), you need to train a new version of the system from scratch on the full dataset (not just the new data, but also the old data), then replace the old model with the new one. Fortunately, the whole process of training, evaluating, and launching a machine learning system can be automated, so even a batch learning system can adapt to change. Simply update the data and train a new version of the system from scratch as often as needed.

This solution is simple and often works fine, but training using the full set of data can take many hours, so you would typically train a new system only every 24 hours or even just weekly. If your system needs to adapt to rapidly changing data (e.g., to predict stock prices), then you need a more reactive solution.

Also, training on the full set of data requires a lot of computing resources (CPU, memory space, disk space, disk I/O, network I/O, etc.). If you have a lot of data and you automate your system to train from scratch every day, it will end up costing you a lot of money. If the amount of data is huge and your system must always be up to date, it may even be impossible to use batch learning.

Finally, if your system needs to be able to learn autonomously and it has limited resources (e.g., a smartphone application or a rover on Mars), then carrying around large amounts of training data and taking up a lot of resources to train for hours every day is a showstopper.

A better option in all these scenarios is to use algorithms that are capable of learning incrementally.

### Online learning

In **online learning**, you train the system incrementally by feeding it data instances sequentially, either individually or in small groups called **mini-batches**. Each learning step is fast and cheap, so the system can learn about new data on the fly, as it arrives . The most common online algorithm by far is gradient descent, but there are a few others.

![In online learning, a model is trained and launched into production, and then it keeps learning as new data comes in](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9798341607972/files/assets/hmls_0113.png)

Online learning is useful for systems that need to adapt to change extremely rapidly (e.g., to detect new patterns in the stock market). It is also a good option if you have limited computing resources; for example, if the model is trained on a mobile device.

Most importantly, online learning algorithms can be used to train models on huge datasets that cannot fit in one machine’s memory (this is called **out-of-core** learning). The algorithm loads part of the data, runs a training step on that data, and repeats the process until it has run on all of the data.

![Using online learning to handle huge datasets](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9798341607972/files/assets/hmls_0114.png)

One important parameter of online learning systems is how fast they should adapt to changing data: this is called the learning rate. If you set a high **learning rate**, then your system will rapidly adapt to new data, but it will also tend to quickly forget the old data: this is called catastrophic forgetting (or **catastrophic interference**). You don’t want a spam filter to flag only the latest kinds of spam it was shown! Conversely, if you set a low learning rate, the system will have more inertia; that is, it will learn more slowly, but it will also be less sensitive to noise in the new data or to sequences of nonrepresentative data points (outliers).

> **WARNING**: Out-of-core learning is usually done offline (i.e., not on the live system), so **online learning** can be a confusing name. Think of it as incremental learning. Moreover, mini-batches are often just called “batches”, so **batch learning** is also a confusing name. Think of it as learning from scratch on the full dataset.

A big challenge with online learning is that if bad data is fed to the system, the system’s performance will decline, possibly quickly (depending on the data quality and learning rate). If it’s a live system, your clients will notice. For example, bad data could come from a bug (e.g., a malfunctioning sensor on a robot), or it could come from someone trying to game the system (e.g., spamming a search engine to try to rank high in search results). To reduce this risk, you need to monitor your system closely and promptly switch learning off (and possibly revert to a previously working state) if you detect a drop in performance. You may also want to monitor the input data and react to abnormal data; for example, using an anomaly detection algorithm.

## Instance-Based Versus Model-Based Learning

One more way to categorize machine learning systems is by how they *generalize*. Most machine learning tasks are about making predictions. This means that given a number of training examples, the system needs to be able to make good predictions for (generalize to) examples it has never seen before. Having a good performance measure on the training data is good, but insufficient; the true goal is to perform well on new instances.

There are two main approaches to generalization: instance-based learning and model-based learning.

### Instance-based learning

Possibly the most trivial form of learning is simply to learn by heart. If you were to create a spam filter this way, it would just flag all emails that are identical to emails that have already been flagged by users—not the worst solution, but certainly not the best.

Instead of just flagging emails that are identical to known spam emails, your spam filter could be programmed to also flag emails that are very similar to known spam emails. This requires a *measure of similarity* between two emails. A (very basic) similarity measure between two emails could be to count the number of words they have in common. The system would flag an email as spam if it has many words in common with a known spam email.

This is called *instance-based learning*: the system learns the examples by heart, then generalizes to new cases by using a similarity measure to compare them to the learned examples (or a subset of them). For example, in Figure 1-15 the new instance would be classified as a triangle because the majority of the most similar instances belong to that class.

Instance-based learning often shines with small datasets, especially if the data keeps changing, but it does not scale very well: it requires deploying a whole copy of the training set to production; making predictions requires searching for similar instances, which can be quite slow; and it doesn’t work well with high-dimensional data such as images.

![Instance-based learning: in this example we consider the class of the three nearest neighbors in the training set
](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9798341607972/files/assets/hmls_0115.png)

### Model-based learning and a typical machine learning workflow

Another way to generalize from a set of examples is to build a model of these examples and then use that model to make **predictions**. This is called **model-based learning**.

![Model-based learning](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9798341607972/files/assets/hmls_0116.png)

For example, suppose you want to know if money makes people happy, so you download the Better Life Index data from the [OECD’s website](https://www.oecdbetterlifeindex.org/), and [World Bank stats](https://ourworldindata.org/) about gross domestic product (GDP) per capita. Then you join the tables and sort by GDP per capita. 

Let’s plot the data for these countries

![Do you see a trend here?](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9798341607972/files/assets/hmls_0117.png)

There does seem to be a trend here! Although the data is **noisy** (i.e., partly random), it looks like life satisfaction goes up more or less linearly as the country’s GDP per capita increases. So you decide to model life satisfaction as a linear function of GDP per capita (you assume that any deviation from that line is just random noise). This step is called **model selection**: you selected a **linear model** of life satisfaction with just one attribute, GDP per capita

#### Equation. A simple linear model

$$
\text{life\_satisfaction} = \theta_0 + \theta_1 \times \text{GDP\_per\_capita}
$$

This model has two *model parameters*, $\theta_0$ and $\theta_1$. By tweaking these parameters, you can make your model represent any linear
function.

![ A few possible linear models](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9798341607972/files/assets/hmls_0118.png)

Before you can use your model, you need to define the parameter values
$\theta_0$ and $\theta_1$. How can you know which values will make your model
perform best? To answer this question, you need to specify a performance
measure. You can either define a *utility function* (or *fitness function*)
that measures how good your model is, or you can define a *cost function*
(a.k.a., *loss function*) that measures how bad it is. For linear regression
problems, people typically use a cost function that measures the distance
between the linear model’s predictions and the training examples; the
objective is to minimize this distance.

This is where the linear regression algorithm comes in: you feed it your
training examples, and it finds the parameters that make the linear model fit
best to your data. This is called *training* the model. In our case, the
algorithm finds that the optimal parameter values are
$\theta_0 = 3.75$ and $\theta_1 = 6.78 \times 10^{-5}$.

> ⚠️ **WARNING**
>
> Confusingly, the word “model” can refer to a *type of model*
> (e.g., linear regression), to a *fully specified model architecture*
> (e.g., linear regression with one input and one output), or to the
> *final trained model* ready to be used for predictions (e.g., linear
> regression with one input and one output, using
> $\theta_0 = 3.75$ and $\theta_1 = 6.78 \times 10^{-5}$).
> Model selection consists in choosing the type of model and fully
> specifying its architecture. Training a model means running an
> algorithm to find the model parameters that will make it best fit the
> training data, and hopefully make good predictions on new data.

Now the model fits the training data as closely as possible (for a linear model)

![The linear model that fits the training data best](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9798341607972/files/assets/hmls_0119.png)

