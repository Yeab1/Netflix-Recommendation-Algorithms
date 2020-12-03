#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as math
import time
def cleaning():
    df = pd.read_csv("data/netflix_titles.csv")

    # splitter
    def splitter(inp):
        if (inp == 'nan'):
            return []
        lst = inp.split(", ")
        return lst

    directors = []
    casts = []
    genres = []
    countries = []
    year = []
    for i in range(6234):
        stringDirector = str(df.director[i])
        stringCast = str(df.cast[i])
        stringGenre = str(df.listed_in[i])
        stringCountries = str(df.country[i])
        stringYear = str(df.release_year[i])

        directors.append(splitter(stringDirector))
        casts.append(splitter(stringCast))
        genres.append(splitter(stringGenre))
        countries.append(splitter(stringCountries))
        year.append(splitter(stringYear))

    df['directors'] = directors
    df['casts'] = casts
    df['genres'] = genres
    df['countries'] = countries
    df['year'] = year
    return df

def modeling(df, genres):

    plt.style.use('seaborn')
    plt.rcParams['figure.figsize'] = [14,14]

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    from sklearn.cluster import MiniBatchKMeans

    # Build the tfidf matrix with the descriptions
    start_time = time.time()
    text_content = df['description']
    vector = TfidfVectorizer(max_df=0.4,         # drop words that occur in more than X percent of documents
                                 min_df=1,      # only use words that appear at least X times
                                 stop_words='english', # remove stop words
                                 lowercase=True, # Convert everything to lower case
                                 use_idf=True,   # Use idf
                                 norm=u'l2',     # Normalization
                                 smooth_idf=True # Prevents divide-by-zero errors
                                )
    tfidf = vector.fit_transform(text_content)
    '''
    # Clustering  Kmeans
    k = [50, 100, 150, 200, 250, 300]
    for i in k:
        print("k = ", i)
        
        kmeans = MiniBatchKMeans(n_clusters = i)
        kmeans.fit(tfidf)
        centers = kmeans.cluster_centers_.argsort()[:,::-1]
        terms = vector.get_feature_names()
    
        request_transform = vector.transform(df['description'])
        # new column cluster based on the description
        df['cluster'] = kmeans.predict(request_transform) 
    
        print(df['cluster'].value_counts().head())
    
    '''

    G = nx.Graph(label="MOVIE")
    start_time = time.time()
    for i, rowi in df.iterrows():
        if (i%1000==0):
            print(" iter {} -- {} seconds --".format(i,time.time() - start_time))
        G.add_node(rowi['title'],key=rowi['show_id'],label="MOVIE",mtype=rowi['type'],rating=rowi['rating'])
    #    G.add_node(rowi['cluster'],label="CLUSTER")
    #    G.add_edge(rowi['title'], rowi['cluster'], label="DESCRIPTION")
        for element in rowi['casts']:
            G.add_node(element,label="PERSON")
            G.add_edge(rowi['title'], element, label="ACTED_IN")
        for element in rowi['genres']:
            G.add_node(element,label="GEN")
            G.add_edge(rowi['title'], element, label="GEN_IN")
        for element in rowi['directors']:
            G.add_node(element,label="PERSON")
            G.add_edge(rowi['title'], element, label="DIRECTED")
        for element in rowi['countries']:
            G.add_node(element,label="COU")
            G.add_edge(rowi['title'], element, label="COU_IN")
        for element in rowi['year']:
            G.add_node(element,label="YR")
            G.add_edge(rowi['title'], element, label="YEAR_IN")
        #indices = find_similar(tfidf, i, top_n = 5)
        #snode="Sim("+rowi['title'][:15].strip()+")"
        #G.add_node(snode,label="SIMILAR")
        #G.add_edge(rowi['title'], snode, label="SIMILARITY")
       # for element in indices:
          #  G.add_edge(snode, df['title'].loc[element], label="SIMILARITY")
    print(" finish -- {} seconds --".format(time.time() - start_time))

    def get_all_adj_nodes(list_in):
        sub_graph=set()
        for m in list_in:
            sub_graph.add(m)
            for e in G.neighbors(m):
                    sub_graph.add(e)
        return list(sub_graph)
    def draw_sub_graph(sub_graph):
        subgraph = G.subgraph(sub_graph)
        colors=[]
        for e in subgraph.nodes():
            if G.nodes[e]['label']=="MOVIE":
                colors.append('blue')
            elif G.nodes[e]['label']=="PERSON":
                colors.append('red')
            elif G.nodes[e]['label']=="GEN":
                colors.append('green')
            elif G.nodes[e]['label']=="COU":
                colors.append('yellow')
            elif G.nodes[e]['label']=="SIMILAR":
                colors.append('orange')
            elif G.nodes[e]['label']=="CLUSTER":
                colors.append('orange')
            elif G.nodes[e]['label']=="YR":
                colors.append("pink")
        nx.draw(subgraph, with_labels=True, font_weight='bold',node_color=colors)
        plt.show()

    list_in=["Ocean's Twelve","Ocean's Thirteen"]
    sub_graph = get_all_adj_nodes(list_in)
    #draw_sub_graph(sub_graph)

    def get_recommendation(root):
        commons_dict = {}
        for e in G.neighbors(root):
            for e2 in G.neighbors(e):
                if e2==root:
                    continue
                if G.nodes[e2]['label']=="MOVIE":
                    commons = commons_dict.get(e2)
                    if commons==None:
                        commons_dict.update({e2 : [e]})
                    else:
                        commons.append(e)
                        commons_dict.update({e2 : commons})
        movies=[]
        weight=[]
        for key, values in commons_dict.items():
            w=0.0
            for e in values:
                w=w+1/math.log(G.degree(e))
            movies.append(key)
            weight.append(w)

        result = pd.Series(data=np.array(weight),index=movies)
        result.sort_values(inplace=True,ascending=False)
        return result;
    '''
    result = get_recommendation("Ocean's Twelve")
    result2 = get_recommendation("Ocean's Thirteen")
    result3 = get_recommendation("The Devil Inside")
    result4 = get_recommendation("Stranger Things")
    print("*"*40+"\n Recommendation for 'Ocean's Twelve'\n"+"*"*40)
    print(result.head())
    print("*"*40+"\n Recommendation for 'Ocean's Thirteen'\n"+"*"*40)
    print(result2.head())
    print("*"*40+"\n Recommendation for 'Belmonte'\n"+"*"*40)
    print(result3.head())
    print("*"*40+"\n Recommendation for 'Stranger Things'\n"+"*"*40)
    print(result4.head())
    
    
    reco=list(result.index[:4].values)
    reco.extend(["Ocean's Twelve"])
    sub_graph = get_all_adj_nodes(reco)
    draw_sub_graph(sub_graph)
    '''

from flask import Flask, render_template, redirect, url_for, request
app = Flask(__name__)

posts = [
    {
        'title': 'Movie 1'
    },
    {
        'title': 'Movie 2'
    },
{
        'title': 'Movie 3'
    },
    {
        'title': 'Movie 4'
    },
{
        'title': 'Movie 5'
    },
    {
        'title': 'Movie 6'
    },
{
        'title': 'Movie 7'
    },
    {
        'title': 'Movie 8'
    },
{
        'title': 'Movie 9'
    },
    {
        'title': 'Movie 10'
    }
]

def MovieChoice(genres):
    df = cleaning()
    possibleMovies = []
    for i in range(len(df.genres)):
        if genres in df.genres[i]:
            possibleMovies.append(df.title[i])
    return possibleMovies

import random
def pickRandomPossibilities(titles, code):
    lst = []
    if code == "demo":
        lst = ["Ocean's Twelve", "Hunt to Kill","My Schoolmate, the Barbarian","Rocky","Rowdy Rathore","The Bare-Footed Kid","Love on Delivery","Snowpiercer","OfficerDowne", "Dhakal"]
    elif code == "notDemo":
        for i in range(10):
            n = random.randint(0, len(titles)-1)
            lst.append(titles[n])
    return lst


randomP = []
genre = ""


@app.route("/")
@app.route("/genre")
def Genre():
    return render_template('genre.html')

@app.route("/movies", methods=["POST", "GET"])
def movies():
    if request.method == "POST":

        if request.form.get("Comedy"):
            genres = "Comedies"
        elif request.form.get("Action"):
            genres = "Action & Adventure"
        elif request.form.get("Adventure"):
            genres = "Adventure"
        elif request.form.get("Kids TV"):
            genres = "Kids TV"
        elif request.form.get("TV Action & Adventure"):
            genres = "TV Action & Adventure"
        elif request.form.get("Drama"):
            genres = "Drama"
        elif request.form.get("TV Drama"):
            genres = "TV Drama"
        elif request.form.get("Anime"):
            genres = "Anime"
        elif request.form.get("family"):
            genres = "family"
        elif request.form.get("Horror"):
            genres = "Horror"
        elif request.form.get("International"):
            genres = "International"
        elif request.form.get("Romance"):
            genres = "Romance"
        elif request.form.get("Sci-Fi & fantasy"):
            genres = "Sci-Fi & fantasy"
        elif request.form.get("Classics"):
            genres = "Classics"

        print(genres)
        possibleMovies = MovieChoice(genres)
        randomPossibilities = pickRandomPossibilities(possibleMovies, "demo")
        print(randomPossibilities)

    elif request.method == "GET":
        print("the second one")

    randomP = [
        {
            'name': randomPossibilities[0]
        },
        {
            'name': randomPossibilities[1]
        },
        {
            'name': randomPossibilities[2]
        },
        {
            'name': randomPossibilities[3]
        },
        {
            'name': randomPossibilities[4]
        },
        {
            'name': randomPossibilities[5]
        },
        {
            'name': randomPossibilities[6]
        },
        {
            'name': randomPossibilities[7]
        },
        {
            'name': randomPossibilities[8]
        },
        {
            'name': randomPossibilities[9]
        }
    ]

    return render_template('movies.html', posts=randomP)

@app.route("/rating", methods=["POST", "GET"])
def rating():

    randomP = [
        {
            'name': pickRandomPossibilities("Action", "demo")[0]
        },
        {
            'name': pickRandomPossibilities("Action", "demo")[1]
        },
        {
            'name': pickRandomPossibilities("Action", "demo")[2]
        },
        {
            'name': pickRandomPossibilities("Action", "demo")[3]
        },
        {
            'name': pickRandomPossibilities("Action", "demo")[4]
        },
        {
            'name': pickRandomPossibilities("Action", "demo")[5]
        },
        {
            'name': pickRandomPossibilities("Action", "demo")[6]
        },
        {
            'name': pickRandomPossibilities("Action", "demo")[7]
        },
        {
            'name': pickRandomPossibilities("Action", "demo")[8]
        },
        {
            'name': pickRandomPossibilities("Action", "demo")[9]
        }
    ]
    if request.method == "POST":

        if request.form.get(randomP[0]['name']):
            #######would not go in here!!!!!!!!
            print("got here!")
            movieChoice = "chosen the first one!"
        '''elif request.form.get("Action"):
            genres = "Action & Adventure"
        elif request.form.get("Adventure"):
            genres = "Adventure"
        elif request.form.get("Kids TV"):
            genres = "Kids TV"
        elif request.form.get("TV Action & Adventure"):
            genres = "TV Action & Adventure"
        elif request.form.get("Drama"):
            genres = "Drama"
        elif request.form.get("TV Drama"):
            genres = "TV Drama"
        elif request.form.get("Anime"):
            genres = "Anime"
        elif request.form.get("family"):
            genres = "family"
        elif request.form.get("Horror"):
            genres = "Horror"
        elif request.form.get("International"):
            genres = "International"
        elif request.form.get("Romance"):
            genres = "Romance"
        elif request.form.get("Sci-Fi & fantasy"):
            genres = "Sci-Fi & fantasy"
        elif request.form.get("Classics"):
            genres = "Classics"'''
        print("here")
        print(randomP[0]['name'])
        #print(movieChoice)

    elif request.method == "GET":
        print("the second one")

    return render_template('rating.html')
if __name__ == '__main__':
    app.run(debug=True)