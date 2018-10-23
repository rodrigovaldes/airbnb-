
import re, wordcloud
import numpy as np
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

'''
This file creates word clouds for papers.

Based on:
A. Mueller: https://github.com/amueller/word_cloud
Ling Fei Wu: https://lingfeiwu1.gitbooks.io/data-mining-in-social-science/content/text_visualization/
'''

def words(text):
    '''
    Returns the words
    '''
    return re.findall('[a-z]+', text.lower())


def file_generator(num, prefix=None):
    if prefix:
        base = str(prefix) + "/{}.txt"
    else:
        base = "{}.txt"
    list_files = []
    for i in range(num):
        list_files.append(base.format(i + 1))
    return list_files


def create_images(file_object, name_images):
    '''
    Creates and saves word clouds
    '''

    text_aux = file_object.read()
    w = words(text_aux)
    stopwords_rvo = stopwords.words('english') + []
    filtered_words = [word for word in w if word not in stopwords_rvo]
    text = ' '.join(filtered_words)

    # Generate a word cloud image
    wordcloud = WordCloud(background_color="white").generate(text)

    # Display the generated image:
    # the matplotlib way:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig("{}_words.png".format(name_images))

    # lower max_font_size
    # wordcloud = WordCloud(background_color="white", max_font_size=40).generate(text)
    # plt.figure()
    # plt.imshow(wordcloud, interpolation="bilinear")
    # plt.axis("off")
    # plt.savefig("{}_two.png".format(name_images))
    # plt.show()

def create_images_big(all_files_big, name_images):

    w = []
    for j, file in enumerate(all_files_big):
        try:
            print("In file: ", j, "The len of w is: ", len(w))
            file_object = open(file, "r")
            text_aux = file_object.read()
            w += words(text_aux)
            print(len(w))
        except:
            print("File: ", file, " skipped.")

    print("Arrive to stop words.")
    stopwords_rvo = stopwords.words('english')
    print("I got filtered words.")
    filtered_words = [word for word in w if word not in stopwords_rvo]
    print("About to join text.")
    print("The head is: ", filtered_words[:5])
    print("The tail is: ", filtered_words[-5:])
    text = ' '.join(filtered_words)


    print("About to generate image.")
    # Generate a word cloud image
    wordcloud = WordCloud(background_color="white").generate(text)

    print("About to graph.")
    # Display the generated image:
    # the matplotlib way:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig("{}_words.png".format(name_images))
