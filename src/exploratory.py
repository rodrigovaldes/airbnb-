import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

listings = pd.read_csv("data/listings.csv")
listings.dropna(subset = ['review_scores_rating'], inplace = True)

# What makes a review positive?

features_preserve = ['host_response_time', 'host_response_rate',
               'host_is_superhost',
               'host_listings_count',
               'host_total_listings_count', 'host_verifications',
               'host_has_profile_pic',
               'accommodates',
               'bathrooms', 'bedrooms', 'beds', 'amenities',
               'price', 'guests_included', 'extra_people', 'minimum_nights',
               'maximum_nights', 'number_of_reviews',
               'instant_bookable',
               'is_business_travel_ready', 'cancellation_policy',
               'require_guest_profile_picture',
               'require_guest_phone_verification',
               'calculated_host_listings_count', 'reviews_per_month']


independent_preserve = ['review_scores_rating']

listings = listings[features_preserve + independent_preserve]

# Trasform to count number of elements in a list: host_verifications, amenities
listings["host_verifications"] = listings["host_verifications"].apply(
        lambda x: len(x.split(",")))
listings["amenities"] = listings["amenities"].apply(
        lambda x: len(x.split(",")))
listings['host_response_rate'] = listings['host_response_rate'].apply(
        lambda x: int(x.strip("%")) if isinstance(x, str) else None)
listings["price"] = listings["price"].apply(
        lambda x: int(x.split(".")[0].strip("$").replace(",","")))
listings["extra_people"] = listings["extra_people"].apply(
        lambda x: int(x.split(".")[0].strip("$").replace(",","")))

cols = ["host_response_time", "host_is_superhost", "host_has_profile_pic"]
for col in cols:
    listings[col] = listings[col].fillna(listings[col].mode()[0])

listings = listings.fillna(listings.mean())
listings = listings.replace("t", 1)
listings = listings.replace("f", 0)

# Save data to graph in R
listings["cancellation_policy"] = listings["cancellation_policy"].replace(
    to_replace=["super_strict_30", "super_strict_60"],
    value=["+strict", "++strict"])
listings["host_response_time"] = listings["host_response_time"].replace(
    to_replace=["a few days or more", "within a day", "within a few hours",
        "within an hour"],
    value=["days", "1 day", "hours", "< 1 hour "])
listings.to_csv("results/listings_graph.csv", index = None)

# Analysis
# To keep things simple, I will use random forest. Only with
# some predefined parameters

listings_x = listings[features_preserve]
listings_y = listings[independent_preserve]

listings_x["cancellation_policy"] = listings_x["cancellation_policy"].replace(
    to_replace=['++strict', '+strict', 'flexible', 'moderate', 'strict'],
    value=[1,2,5,4,3])
listings_x["host_response_time"] = listings_x["host_response_time"].replace(
    to_replace=["days", "1 day", "hours", "< 1 hour "],
    value=[1,2,3,4])

clf = RandomForestClassifier(max_depth=5, random_state=0)
clf.fit(listings_x, listings_y)
predicted = clf.predict(listings_x)

cols = list(listings_x.columns)
importance = list(clf.feature_importances_)
importance_df = pd.DataFrame({'feature': cols,'importance': importance}
    ).sort_values("importance", ascending=False)

importance_df_save = importance_df[:7].sort_values("importance")

importance_df_save.to_csv("results/feature_importance.csv", index = None)
