

# ## Import data set and get important libraries

# In[3]:


# Import sklearn and other modules used to create a clustering model.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans

# Import Networkx and other essential plotting libraries.
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as math
import time 
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [14,14]

# Import data
df = pd.read_csv("data/netflix_titles.csv")


# ## Data Exploration

# In[4]:


df.head()


# In[7]:


# How old are the movies?

old = df.sort_values("release_year", ascending = True)
old = old[old['duration'] != ""]
old = old[old['type'] !="Movie"]
old[['title', "release_year","country","duration"]][:5]


# In[10]:


# What is the ratio of movies to series (TV shows)?
df.type.value_counts(normalize=True)


# In[11]:


# Count of records by country
df['country'].value_counts()


# ## Data Cleaning

# In[12]:


# Since the "listed_in", "cast", and "country" fields are multivalued fields. Let's break them up.
# A function that changes the comma separated values into lists for more efficient access later.
def splitter(inp):
    if(inp == 'nan'):
        return []
    lst = inp.split(", ")
    return lst


# In[14]:


# use the splitter function to break up the multivalued fields into accessable lists.
directors = []
casts = []
genres = []
countries = []
year = []
for i in range(6234):
    # NaN's have a type of float, for easier manipulation, change every record to a string data type.
    stringDirector = str(df.director[i])
    stringCast = str(df.cast[i])
    stringGenre = str(df.listed_in[i])
    stringCountries = str(df.country[i])
    stringYear = str(df.release_year[i])
    
    # Use the splitter function to split the comma separated strings into lists.
    directors.append(splitter(stringDirector))
    casts.append(splitter(stringCast))
    genres.append(splitter(stringGenre))
    countries.append(splitter(stringCountries))
    year.append(splitter(stringYear))

# Add the new lists to the dataframe.
df['directors'] = directors
df['casts'] = casts
df['genres'] = genres
df['countries'] = countries
df['year'] = year


# In[16]:


# remove the ID column
df=df.drop('show_id',axis=1)


# In[19]:


# remove "country", "director", "cast", "release_year" and "listed_in" because they have been modified 
#and saved as new fields
df=df.drop('country',axis=1)
df=df.drop('director',axis=1)
df=df.drop('cast',axis=1)
df=df.drop('release_year',axis=1)
df=df.drop('listed_in',axis=1)

df.head()


# ## Modeling

# In[21]:


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


# In[24]:


# Let's try a clustering model.
# Clustering  Kmeans
kmeans = MiniBatchKMeans(n_clusters = 200)
kmeans.fit(tfidf)
centers = kmeans.cluster_centers_.argsort()[:,::-1]
terms = vector.get_feature_names()

request_transform = vector.transform(df['description'])
# new column cluster based on the description
df['cluster'] = kmeans.predict(request_transform) 

print(df['cluster'].value_counts().head())


# ###### The model is clustering almost all records together. Could be a problem with the k value chosen, let's try it with multiple K values.

# In[22]:



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


# ###### some clusters are a lot bigger than others. Seems like it is clustering the records into the same cluster.

# ### Let's try a new approach.

# In[25]:


G = nx.Graph(label="MOVIE")
for i, rows in df.iterrows():
    G.add_node(rows['title'],label="MOVIE",mtype=rows['type'],rating=rows['rating'])

    for element in rows['casts']:
        G.add_node(element,label="PERSON")
        G.add_edge(rows['title'], element, label="ACTED_IN")
    for element in rows['genres']:
        G.add_node(element,label="GEN")
        G.add_edge(rows['title'], element, label="GEN_IN")
    for element in rows['directors']:
        G.add_node(element,label="PERSON")
        G.add_edge(rows['title'], element, label="DIRECTED")
    for element in rows['countries']:
        G.add_node(element,label="COU")
        G.add_edge(rows['title'], element, label="COU_IN")
    for element in rows['year']:
        G.add_node(element,label="YR")
        G.add_edge(rows['title'], element, label="YEAR_IN")
    


# In[26]:


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


# In[31]:


list_in=["Jeff Dunham: Relative Disaster","Jeff Dunham: Beside Himself"]
sub_graph = get_all_adj_nodes(list_in)
draw_sub_graph(sub_graph)


# In[32]:


list_in=["XXx","Jeff Dunham: Beside Himself"]
sub_graph = get_all_adj_nodes(list_in)
draw_sub_graph(sub_graph)


# In[33]:


list_in=["XXx","XXX: State of the Union"]
sub_graph = get_all_adj_nodes(list_in)
draw_sub_graph(sub_graph)


# In[35]:


list_in=["Ocean's Thirteen","Ocean's Twelve"]
sub_graph = get_all_adj_nodes(list_in)
draw_sub_graph(sub_graph)


# In[37]:


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


# In[38]:


result = get_recommendation("Ocean's Twelve")
result2 = get_recommendation("XXx")
result3 = get_recommendation("Jeff Dunham: Relative Disaster")
print("Recommendations for 'Ocean's Twelve'\n")
print(result.head())
print("Recommendations for 'Ocean's Thirteen'\n")
print(result2.head())
print("Recommendations for 'Belmonte'\n")
print(result3.head())


# In[39]:


reco=list(result.index[:4].values)
reco.extend(["Ocean's Twelve"])
sub_graph = get_all_adj_nodes(reco)
draw_sub_graph(sub_graph)

