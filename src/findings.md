How to improve your AirBnb review score?
================
Rodrigo Valdes Ortiz
5/15/2018

Exploratory analysis
--------------------

There are three main conclusions of exploratory analysis:

1.  50% of the listings have perfect scores.
2.  94% of the listings have good scores (above 9/10)
3.  Only 6% are below 9/10.

Characteristics of good and bad reviews
---------------------------------------

What are the characteristics of good and bad reviews?

To explore this visually, I create two word clouds.

### Words in positive reviews

![](../images/positive_words.png?raw=true)

### Words in negative reviews

![](../images/negative_words.png?raw=true)

**Positive reviews: the emphasis is in New York**, the host, the location, and the place.

**Negative reviews: the most important issues are related to the room** and the apartment.

Conclusion: in **positive** reviews customers remember the **city**. However, they talk about their **AirBnb spot in bad reviews**.

Classification model
--------------------

To analyze this claim analytically, I trained a classification model based on text. The accuracy of this model is 0.8 in the development set, and 0.78 in the development test set. Then, this model increases the accuracy more than 50% against random.

Now, I present the most interesting weights from this model.

### Words associated with positive reviews

| weight |     word     |
|--------|:------------:|
| 4.23   | 'definitely' |
| 3.71   |   'perfect'  |
| 3.53   |    'super'   |
| 3.40   |     'We'     |
| 2.83   |  'wonderful' |
| 2.82   |     'New'    |
| 2.43   |    'great'   |
| 2.38   |  'Manhattan' |
| 2.38   |    'clean'   |
| 2.33   |  'recommend' |
| 2.30   |    'loved'   |
| 2.17   |    'here'    |
| 2.17   |   'subway'   |
| 2.06   |    'room'    |
| 2.05   | 'everything' |
| 2.01   |    'home'    |

### Words associated with negative reviews

| weight |    word    |
|--------|:----------:|
| 8.26   |    'not'   |
| 4.60   |    'no'    |
| 3.87   |    'we'    |
| 3.28   |   'good'   |
| 2.56   |    'host   |
| 2.09   |   'dirty'  |
| 1.90   |   'quite'  |
| 1.71   |   'floor'  |
| 1.66   |    'pas'   |
| 1.60   |   'wasnt'  |
| 1.58   |   'night'  |
| 1.42   |   'didnt'  |
| 1.41   |    'but'   |
| 1.40   |  'friends' |
| 1.40   |  'people'  |
| 1.17   |   'sleep'  |
| 1.16   | 'bathroom' |
| 1.12   |   'beds'   |

### Conclusions from the model

According to the analysis of the weights, and reading the actual text of the reviews, there are some patterns in bad comments from the guests:

**I. Find a place that does not look as it was depicted.**

-   For instance, "huge room with spectacular view," but the guest describe the room as "small and dark without a view to the street."

**II. The bathroom.**

-   Cleanliness
-   Not working

**III. Cleanliness.**

-   Dirty common spaces
-   Guest are asked to clean what they did not use

**IV. Difficulties to sleep.**

-   Noise
-   Habits of the hosts
