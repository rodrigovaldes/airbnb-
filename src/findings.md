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

To analyze this claim analytically, I trained a classification model based on text. The accuracy of this model is 0.82 in the development set, and 0.82 in the development test set. Then, the accuracy of the model is 64% bigger than random.

Please find below the most interesting weights of the model.

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

According to the analysis of the weights, and reading the actual text of the reviews, there are patterns in the reviews of low score spots:

**I. Guests find a place that does not look as it was depicted.**

-   For instance, "huge room with spectacular view," but the guest describe the room as "small and dark without a view to the street"
-   Some of the most important weights for negative reviews are contrasting words, such as no, not, pas (no in French), and but

**II. The bathroom.**

-   Cleanliness
-   Not working
-   Weights of this claim: bathroom, dirty, no, and not

**III. Cleanliness.**

-   Dirty common spaces
-   Guest are asked to clean what they did not use
-   Associated weights: beds, floor, dirty, no, and not

**IV. Difficulties to sleep.**

-   Noise
-   Habits of the hosts
-   For instance, guests complain about parties of the host, people in the apartment, and noise during the night.
-   Some of the weights are host, night, friend, and people

Recommendations for hosts trying to increase its score
------------------------------------------------------

1.  Describe your spot nicely but accurately.
2.  When you have guest, clean the common areas and bathrooms. Your guest might have different cleaning standards that you.
3.  Avoid hassles for your guest, such as organizing parties when they want to sleep. Also, you might want to clarify quiet hours –or the lack of those– in your posting.

Final remarks
-------------

An interesting finding of the exploratory analysis and the model is that **the apartment in less important than the city in positive reviews**. In positive reviews, people talk about the city, hosts, location, and public transportation. However, in negative reviews, the pitfalls in the apartment opaque the city, host, or location.

This model helps to understand the reasons behind positive and negative reviews. However, **further research is needed to understand the nuances in this data**. For instance, are there different patterns of reviews according to the nationality of the guest? Which are the specific necessities of leisure and business travellers?

In addition, the data only provides the aggregated score by spot – the average of guests' reviews. Then, **data which contains scores per each guest review will clarify reasons behind positive and negative reviews.**

------------------------------------------------------------------------

Return to the main repo [here](https://github.com/rodrigovaldes/airbnb-).
