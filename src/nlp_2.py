import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def plot_histogram(vector, title = None):
    '''
    Plots an histogram

    Input:
        vector = array, series, list
    '''

    vector = vector.dropna()
    mu = np.mean(vector)
    sigma = np.std(vector)

    n, bins, patches = plt.hist(vector, normed=1, facecolor='green', alpha=0.75)
    n, bins, patches = plt.hist(vector, facecolor='green', alpha=0.75)

    # add a 'best fit' line
    y = mlab.normpdf( bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=1)

    plt.xlabel('Value')
    plt.ylabel('Probability')
    if title:
        plt.title(title)
    plt.grid(True)

    # plt.show()
    plt.savefig('images/g_'+title)
    plt.close()


def define_subset(data, min_reviews, max_reviews = None,
        name_col = 'number_of_reviews'):
    '''
    Creates a subset of a dataframe

    Inputs:
        data = dataframe
        min_reviews = integer
        name_col = string
    Outpus:
        data = dataframe (filtered dataframe)
        gb = groupby object
    '''
    print("The len of the old data is: ", len(data))

    mask = data[name_col] >= min_reviews

    data = data[mask]

    if max_reviews:
        mask2 = data[name_col] <= max_reviews
        data = data[mask2]

    print("The len of the new data is: ", len(data))

    return data


def describe_data(df):
    '''
    Auxiliary function to undertand the general characteristics of
    the data

    Imput:
        df = pandas dataframe
    Output:
        None
    '''

    print(" ")
    print("*********************************")
    print("Description of the data")
    print("*********************************")
    print(df.describe())
    print(" ")

    print("*********************************")
    print("General Information")
    print("*********************************")
    print(df.info())
    print(" ")

    print("*********************************")
    print("About Misisng Values")
    print("*********************************")
    print(df.isnull().sum())

    print(" ")
    return "Done"


# To explore later
control_features = ['property_type', 'room_type', 'number_of_reviews',
       'review_scores_rating',
       'review_scores_accuracy', 'review_scores_cleanliness',
       'review_scores_checkin', 'review_scores_communication',
       'review_scores_location',
       'calculated_host_listings_count', 'reviews_per_month']

# 'review_scores_value'

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
               'require_guest_profile_picture', 'require_guest_phone_verification',
               'calculated_host_listings_count', 'reviews_per_month']
