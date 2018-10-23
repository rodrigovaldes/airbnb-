# Analysis of Airbnb reviews

## For a quick view of the general results, please check:

* visual_exp.md
* findings.md


## The analysis was made in two phases:

1. Exploratory analysis
  + Analysis done in src/exploratory.py
  + Visualizations in src/visual_exp.Rmd

2. NLP classification model to identify the words related to positive and negative reviews
  + Code in nlp_1.py, nlp_2.py, text_an.py, and nlp_main.py
  + To execute this phase, just run nlp_nain.py
  + Visualizations in src/findings.Rmd

## Data
The data was grabbed from http://insideairbnb.com/ at the beginning of May 2018. I used listings.csv and reviews.csv for NYC.
The raw data can't be uploaded to GitHub because it's too big. However, I am happy to share it, please send me a message.
