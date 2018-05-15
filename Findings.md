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

![](../images/positive_words.png)

### Words in negative reviews

![](../images/negative_words.png)

**Positive reviews: the emphasis is in New York**, the host, the location, and the place.

**Negative reviews: the most important issues are related to the room** and the apartment.

Conclusion: in **positive** reviews customers remember the **city**. However, they talk about their **AirBnb spot in bad reviews**.

Classification model
--------------------

To analyze this claim analytically, I trained a classification model based on text. The accuracy of this model is 0.8 in the development set, and 0.78 in the development test set. Then, this model increases the accuracy more than 50% against random.

Now, I present the most interesting weights from this model.

### Words associated with positive reviews

| weight |      word     |
|--------|:-------------:|
| 2.78   |  'definitely' |
| 2.21   |    'great'    |
| 2.11   |   'perfect'   |
| 2.06   |    'super'    |
| 2.03   |  'everything' |
| 2.01   |  'recommend'  |
| 1.95   |    'subway'   |
| 1.71   | 'comfortable' |
| 1.65   |    'hosts'    |
| 1.62   |  'fantastic'  |
| 1.57   |  'beautiful'  |
| 1.51   |     'walk'    |
| 1.44   |  'Manhattan'  |
| 1.43   |    'clean'    |
| 1.32   |     'view'    |

### Words associated with negative reviews

| weight |      word     |
|--------|:-------------:|
| 5.55   |     'not'     |
| 3.06   |      'no'     |
| 2.63   |     'good'    |
| 2.39   |     'but'     |
| 1.93   |   'bathroom'  |
| 1.55   |     'Good'    |
| 1.29   |    'didnt'    |
| 1.21   |    'sleep'    |
| 1.15   |    'people'   |
| 1.06   | 'reservation' |
| 1.03   |     'East'    |
| 0.99   |    'dirty'    |
| 0.99   |    'small'    |
| 0.95   |     'old'     |
| 0.94   |    'living'   |
| 0.92   |     'what'    |
| 0.92   |    'where'    |
| 0.87   |     'pour'    |

### Conclusions from the model

According to the analysis of the weights, and reading the actual text of the reviews, there are some patterns in bad comments from the guests:

**I. Find a place that does not look as it was depicted. **

-   For instance, "huge room with spectacular view," but the guest describe the room as "small and dark without a view to the street."

**II. The bathroom. **

-   Cleanliness
-   Not working

**III. Cleanliness. **

-   Dirty common spaces
-   Guest are asked to clean what they did not use

**IV. Difficulties to sleep.**

-   Noise
-   Habits of the hosts
