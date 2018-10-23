import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
from nlp_1 import *
from nlp_2 import *
from text_an import *
seed_ = 82848392

listings = pd.read_csv("data/listings.csv")
listings.dropna(subset = ['review_scores_rating'], inplace = True)

reviews = pd.read_csv("data/reviews.csv")

'''
------------------------------------------------------------------------------
# Explore the data
------------------------------------------------------------------------------
'''

# Keep those with 5 - 100 reviews
listings_relevant = define_subset(listings, 5, 100)

# When reviews are in the neighbourhood 5 -100
plot_histogram(listings_relevant['number_of_reviews'], "num_reviews_subset")

# Relationship of the number of reviews with the rating
keep_cols = ['review_scores_rating', 'number_of_reviews']
rating_reviews = listings[keep_cols]

mask_more_90 = rating_reviews['review_scores_rating'] > 90
rating_reviews["more_90"] = mask_more_90
compilation_more90 = rating_reviews.groupby("more_90").mean()

# Relationship among the different scores

cols_ratings =  ['review_scores_rating',
                  'review_scores_accuracy', 'review_scores_cleanliness',
                   'review_scores_checkin', 'review_scores_communication',
                   'review_scores_location', 'review_scores_value']

# for item in cols_ratings:
#     plot_histogram(listings_relevant[item], item)

# The general rating and cleanliness are the more "fat" histograms. Location
# a little bit.

# Analyze this with a simple linear regression
# Which characteristics explain the changes in general score?
scores_df = listings_relevant[cols_ratings]
scores_df_log = np.log(scores_df)

# Define the X
x_df = scores_df_log.drop(
    columns = ["review_scores_value", 'review_scores_rating'])
x_train = np.array(x_df)
x_train = sm.add_constant(x_train)

# Define the y
y_train = np.array(scores_df_log['review_scores_value'])

# Train the model
model = sm.OLS(y_train, x_train).fit()
model.summary2()

# Notice that a great part of the general score is explained by accuracy (0.35%)
# & cleanliness (0.20%), checkin, communication, and location is about (0.12%)

'''
------------------------------------------------------------------------------
# Create Data For Host
------------------------------------------------------------------------------
'''
# Only text DF

# Host vector
text_features = ['summary',
       'space', 'description', 'neighborhood_overview',
       'notes', 'transit', 'access', 'interaction', 'house_rules',
       'host_about']

text_df = listings_relevant[text_features]

all_text = pd.DataFrame(text_df.fillna('').sum(axis=1)).rename(
    columns = {0: "description"})
all_text['id'] = listings_relevant['id']
all_text['review_scores_value'] = listings_relevant['review_scores_value']
all_text['review_scores_cleanliness'] = listings_relevant[
    'review_scores_cleanliness']
all_text.dropna(inplace=True)


'''
------------------------------------------------------------------------------
# Create Data For Reviews
------------------------------------------------------------------------------
'''

# Cultural data: 7K have reviewed 3 or more.
nr_per_person = reviews.groupby("reviewer_id").count().sort_values(
    "id", ascending = False)
above_3 = sum(nr_per_person["id"] > 3)

reviews_relevant = reviews[['listing_id', 'comments']]
reviews_relevant.dropna(inplace=True)
compilation_reviews = pd.DataFrame(reviews_relevant.groupby(["listing_id"]
    ).apply(lambda x: " ".join(x["comments"]))).rename(columns = {0: 'reviews'})
compilation_reviews['id'] = compilation_reviews.index
compilation_reviews.reset_index(drop=True, inplace=True)

'''
------------------------------------------------------------------------------
# Compile data and save files
------------------------------------------------------------------------------
'''

data = pd.merge(all_text, compilation_reviews,
    on="id")[['id', 'description', 'reviews', 'review_scores_value']]

data = data.replace(["\n", "\r", "\'"], [" ", " ", ""], regex=True)

# Next step: train model to identify bad reviews
# 7 & 8 are bad reviews
data["indicator"] = data["review_scores_value"] > 8.9
data["indicator"].replace([True, False], [1, 0], inplace = True)

data_to_save = data[["reviews", "indicator"]]
all_negative = data_to_save[data_to_save["indicator"] == 0]
all_positive = data_to_save[data_to_save["indicator"] == 1].sample(
    n=len(all_negative), random_state=seed_)
all_save = pd.concat([all_negative, all_positive]).sample(
    frac=1).reset_index(drop=True)

all_dev = all_save[:400]
all_dev_test = all_save[400:800]
all_train = all_save[800:]

all_train.to_csv("data/train_airbnb.txt", sep="\t", header=None)
all_dev.to_csv("data/dev_airbnb.txt", sep="\t", header=None)
all_dev_test.to_csv("data/dev_test_airbnb.txt", sep="\t", header=None)

all_positive["reviews"].to_csv("data/positive.txt", sep="\t", header=None,
    index=None)
all_negative["reviews"].to_csv("data/negative.txt", sep="\t", header=None,
    index=None)

'''
------------------------------------------------------------------------------
# Cloud of words
------------------------------------------------------------------------------
'''

prefix = "data"
prefix_images = "images"

for file in ["data/positive.txt", "data/negative.txt"]:
    try:
        file_object = open(file, "r")
        name_images = prefix_images + file[len(prefix):-4]
        create_images(file_object, name_images)
    except:
        print("File: ", file, " skipped.")


'''
------------------------------------------------------------------------------
# Model to classify
------------------------------------------------------------------------------
'''

# Controls
file_train = "data/train_airbnb.txt"
file_dev = "data/dev_airbnb.txt"
file_dev_test = "data/dev_test_airbnb.txt"

file_results_accuracy = "results/accuracy_findings.txt"
file_weights_analysis_base = "results/words_base.txt"
file_weights_analysis_hinge = "results/words_hinge.txt"
file_weights_all_special = "results/words_all_sf.txt"
file_wrong = "results/sample_wrong.csv"

list_files_remove = [file_results_accuracy, file_weights_analysis_base,
    file_weights_analysis_hinge, file_wrong]

for file in list_files_remove:
    remove_file_if_exists(file)


num_categories = 2
eta = 0.01
n_epochs = 5

# Base
hinge, special, names_sf = False, False, [None, None, None]
compilation_base, lc_df_base, key_b = run_cases(file_train, file_dev,
    file_dev_test, num_categories, n_epochs, eta, hinge, special, names_sf)

dev, dev_test = print_accuracy(compilation_base, file_results_accuracy,
    model = "base")

weights = compilation_base["unique"]["weights"]
weigths_analysis(weights, num_categories, file_weights_analysis_base,
    total_ex=100)

# Hinge
hinge, special, names_sf = True, False, [None, None, None]
compilation_hinge, lc_df_hinge, key_h = run_cases(file_train, file_dev,
    file_dev_test, num_categories, n_epochs, eta, hinge, special, names_sf)

dev, dev_test = print_accuracy(compilation_hinge, file_results_accuracy,
    model = "hinge")

weights_h = compilation_hinge["unique"]["weights"]
weigths_analysis(weights_h, num_categories, file_weights_analysis_hinge,
    total_ex=100)









#
