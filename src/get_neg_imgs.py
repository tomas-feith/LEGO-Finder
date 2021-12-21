# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 06:37:42 2021

"""

import os
import requests
from bs4 import BeautifulSoup

# define the url to search the images
google_image = "https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&"

# some definitions for the browser
user_agent = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"
}

# define the folder where to save the extracted images
saved_folder = '../data/Negative_Samples'

def main(queries, n_images):
    # if the output folder doesn't exist, create it
    if not os.path.exists(saved_folder):
        os.mkdir(saved_folder)

    # if n_images isn't a number, we just assume the user wants all the
    # images possible
    # otherwise we divide the images requested equally by the classes
    if type(n_images) != str:
        n_img_class = n_images // len(queries)

    # get the images for each one of the queries
    for query in queries:
        # add the query to the general url
        search_url = google_image + 'q=' + query.replace(' ', '+')
        print(search_url)

        # get the response from Google
        response = requests.get(search_url, headers=user_agent)

        # get the html and parse it
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')

        # the results are all the images in the html
        results = soup.findAll('img', {'class': 'rg_i Q4LuWd'})

        # now we get the links, either until we run through all of them or
        # until we get to the number of images requested
        count = 1
        links = []
        for result in results:
            try:
                link = result['data-src']
                links.append(link)
                count += 1
                if type(n_images) != str and count > n_img_class:
                    break
            # if we get some error just go to the next image
            except KeyError:
                continue

        print(f"Downloading {len(links)} images...")

        # now we download all images and save them to the folder
        for i, link in enumerate(links):
            response = requests.get(link)
            image_name = saved_folder + '/' + query.replace(' ', '_') + str(i+1) + '.jpg'

            with open(image_name, 'wb') as fh:
                fh.write(response.content)


if __name__ == "__main__":
    # these are the queries we considered adequate for our purposes
    # but they can be edited
    # our main idea was that LEGOs will mainly be used on surfaces like tables
    # or the floor, and we also want to keep the model from just learning easy
    # geometric shapes like squares, triangles and blocks.
    queries = ['wooden texture',
               'marble texture',
               'stainless stell texture',
               'pile of bricks',
               'table corner',
               'table window',
               'carpet texture',
               'tile texture',
               'ceramic texture',
               'background color squares',
               'color circles random',
               'color triangles random',
               'metal rods texture',
               'sticks texture']
    main(queries, '')
