
# coding: utf-8

# In[3]:


import itertools as it
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import nltk
import re
import unicodedata
import string
from nltk.corpus import stopwords
import pandas as pd
import inflect
import pickle
import math
from nltk.stem.snowball import ItalianStemmer
import heapq
import re
from collections import OrderedDict








# ---------- SECTION 0: CLEANING DATA ----------

#This function checks for Nones or other non castable to numeric elements and
# replace the rows containing them with an empty row (that will be dropped)
def to_numeric_2(x):
    for el in x:
        if el is None:
            x = pd.Series()
            break
            
    try:
        pd.to_numeric(x)
    except:
        x = pd.Series()
    
    
    return x









# ---------- SECTION 1 : SCRAPING ----------



# the function will create the data and return it, just give the path to save the file

def create_data(
path = r'D:\\Data Science\\ADM(Algorithmic Methods of Data Mining)\\hw\\hw4\\'):# path to save file
    # Create a dataframe to store all data

    # find max page number
    max_page_no = int(BeautifulSoup(requests.get("https://www.immobiliare.it/vendita-case/roma/?criterio=rilevanza&pag=1").text,
                                    "lxml").find_all('span',attrs={"class": "pagination__label"})[-1:][0].text)

    # save the file after each iteration with pickle 
    fileObject = open(path + "adm-hw4-data-dene",'wb')

    for page in range(1,max_page_no+1): # go through for all pages

        try:
            url = "https://www.immobiliare.it/vendita-case/roma/?criterio=rilevanza&pag=" + str(page)

            print("page:",page)

            content = requests.get(url) # create a content for specified url
            soup  = BeautifulSoup(content.text, "lxml") # turn it as lxml format

            all_adv = soup.find_all('p',attrs={"class": "titolo text-primary"})

            # store the data for only 1 page in a dataframe
            data = pd.DataFrame(columns=["url","price", "locali", "superficie", "bagni", "piano", "description"])

            for link_i in range(len(all_adv)):
                link = all_adv[link_i].find('a').get('href') # get all link that are advertised on the page that we specified above

                if 'https://' in link:
                    linkURL = link
                else: # some of link do not have http, http://, www.immobiliare.it or both parts. Thus, add it at the beginning of link
                    if 'www.immobiliare.it' in link:
                        if link[:2] == '//':
                            linkURL = 'https:' + link
                        elif link[0] == '/':
                            linkURL = 'https:/' + link
                    else:
                        if link[0] != '/':
                            linkURL = 'https://www.immobiliare.it/' + link
                        else:
                            linkURL = 'https://www.immobiliare.it' + link

                print(linkURL)

                link_content = requests.get(linkURL)
                link_soup = BeautifulSoup(link_content.text, "lxml") # convert the content into lxml
                ul = link_soup.find_all("ul", attrs={"class": "list-inline list-piped features__list"}) # this list includes all features except price

                # check which features having and store it if there is present or not
                features = [] # at the end, it'll show that locali, superficie, bagni and piano exist in the link or not
                all_features = list(map(lambda x: x.text, ul[0].find_all('div',attrs={"class": "features__label"}))) # which featers link have
                features_check_list =['locali', 'superficie', 'bagni', 'piano']

                for i in range(len(features_check_list)):
                    if features_check_list[i] in all_features:
                        features.append(1) # 1 means that feature is present
                    else:
                        features.append(0) # 0 means that feature is not present


                feature_values = [] # all features will be on that list   


                # first add linkURL
                feature_values.append(linkURL)


                # add avg. price to feature_values
                price_block = link_soup.find_all('ul',attrs={"class": "list-inline features__price-block"})[0].find_all('li',attrs={"class": "features__price"})[0]
                price = []

                if not(str(str(price_block).find("features__price--double")).isdigit()) and not(str(str(price_block).find("features__price-old")).isdigit()):
                    for s in price_block.text.split():
                        if s.isdigit() or s.replace('.','').isdigit(): # check whether it is int or float
                            s = s.replace('.','') # web site uses dot instead of comma. So first destroy dots
                            s = s.replace(',','.') # then replace comma with dot because in python decimal numbers indicates w/ dot
                            price.append(float(s))
                elif str(str(price_block).find("features__price--double")).isdigit():
                    # for the price feature, sometimes a range is given. In that case, we'll take average of min and max value of price
                    for s in price_block.text.split():
                        if s.isdigit() or s.replace('.','').isdigit(): # check whether it is int or float
                            s = s.replace('.','')
                            s = s.replace(',','.')
                            price.append(float(s))
                elif str(str(price_block).find("features__price-old")).isdigit():
                    start_idx = str(price_block).find('<li class="features__price"><span>') + len('<li class="features__price"><span>')
                    end_idx = str(price_block).find("</span>") 
                    for s in str(price_block)[start_idx:end_idx].split():
                        if s.isdigit() or s.replace('.','').isdigit(): # check whether it is int or float
                            s = s.replace('.','')
                            s = s.replace(',','.')
                            price.append(float(s))

                feature_values.append(np.mean(price))


                # fill the features; locali, superficie, bagni and piano (price is already added.)
                loc_sficie_bag = list(map(lambda x: x.text, ul[0].find_all('span',attrs={"class": "text-bold"})))

                j = 0
                for i in range(3): # we'll fill locali, superficie and bagni
                    if features[i] == 0: # we are checking absence of the feature
                        feature_values.append(None) # if it is absent, put it None 
                    else:
                        if i == 0:
                            # this part is only for locali. If there is range for locali, take it average ot it
                            loc = []
                            for e in loc_sficie_bag[j]:
                                for s in e.split():
                                    if s.isdigit() or s.replace('.','',1).isdigit(): # check whether it is int or float
                                        loc.append(float(s))
                            feature_values.append(np.mean(loc)) # take it average and add the value to feature_values
                            j += 1
                        else:
                            feature_values.append(int(re.search(r'\d+', loc_sficie_bag[j]).group())); j += 1


                # adding piano; it can be integer or string
                piano = ul[0].find_all('abbr',attrs={"class": "text-bold im-abbr"})

                if piano != []: # check whether piano feature does not exist in the link or not
                    feature_values.append(piano[0].text.split("\xa0")[0]) # if it exists, add the value to feature_values 
                else:
                    feature_values.append(None) # if it does not exists, add None to feature_values


                # adding description
                desc = link_soup.find_all('div',attrs={"id": "description"})[0].find_all('div',attrs={"class": "col-xs-12 description-text text-compressed"})[0].text
                feature_values.append(desc)

                data.loc[data.shape[0]+1]= feature_values # add all features as new row

                time.sleep(0.5)

            pickle.dump(data, fileObject) # save the dataframe that we got for just 1 page
            time.sleep(0.5) # this helps to prevent the website block
        except:
            pass


    fileObject.close()
    
    # read the data part by part
    ADM_HW4_data = pd.DataFrame(columns=["url","price", "locali", "superficie", "bagni", "piano", "description"]) # rename columns

    fileObject = open(path + "adm-hw4-data-dene",'rb') # open the file to read

    # to not force memory, we send the data for each page to pickle. Now, we are collecting them. Since we use try and except,
    # some of pages are lost (we have 1729 at the beginning) but around 41000 rows are quite enough also.
    for i in range(1,1678+1):
        ADM_HW4_data = ADM_HW4_data.append(pickle.load(fileObject))

    fileObject.close() # close to file

    ADM_HW4_data.reset_index(drop=True, inplace=True) # drop indexes

    # since we create data from too many pickle files, I will save it as one piece
    ADM_HW4_data.to_pickle(path + 'hw4_data')

    # read the data
    hw4_data = pd.read_pickle(path + 'hw4_data')
    
    return hw4_data










# ---------- SECTION 2 : ANNOUNCEMENT PREPROCESSING ----------
    
# F1 : This function removes stop words from list of tokenized words

def remove_stopwords(wrd):
    new_wrd = [] #List of updated words
    
    for word in wrd:
        if word not in stopwords.words("italian"): # If the current word is not a stopword (ckeck using nltk)
            new_wrd.append(word)                   #appends it to the list
  
    return new_wrd




# F2 : This function removes punctuation from list of tokenized words

def remove_punctuation(wrd):
    new_wrds = []  #List of updated words
    
    for word in wrd:
        new_wrd = re.sub(r'[^\w\s]', '', word) # Replaces all punctuation word with "" using RegEx
        if new_wrd != '':
            new_wrds.append(new_wrd)           #And then appends all words different from "" to the list 
    
    return new_wrds



# F3 : This function stems words in a list of tokenized words

def stem_words(wrd):
    stemmer = ItalianStemmer() # Selects the stemmmer from nltk
    stems = [] # List of updated words
    
    for word in wrd:
        stem = stemmer.stem(word) # Stems the word
        stems.append(stem)        # and appends it to the list
        
    return stems




# F4 : This functions removes non ascii chars from a list of tokenized words

def remove_non_ascii(wrd):
    new_wrds = [] # List of updated words
    
    for word in wrd:
        new_wrd = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') # Filters non ascii chars
        new_wrds.append(new_wrd) # Appends the word to the list
    
    return new_wrds



# F5 : This function converts all characters to lowercase from a list of tokenized words

def to_lowercase(wrd):
    new_wrds = [] # List of updated words
    
    for word in wrd:
        new_wrd = word.lower()   # Converts the current word to lower case
        new_wrds.append(new_wrd) # And append it to the list
        
    return new_wrds


# The following function takes a record of a dataFrame containg our docs and preprocesses it's title and description 
# with all the previous functions

def preProcessing (x):
    
    
    xt = nltk.word_tokenize(x) # Tokenizes title using nltk
        
     
    # Uses previous functions
    xt = remove_stopwords(xt)
    xt = remove_punctuation(xt)
    xt = stem_words(xt)
    xt = remove_non_ascii(xt)
    xt = to_lowercase(xt)
   
    
    return xt # Returns the preprocessed doc









# ----------SECTION 3: INVERTED INDICES & Co.----------

def create_vocabulary(data):
    vocabulary = {} # The vocabulary is a dictionary of the form "Word : word_id"
    wid = 0 # word_id
       
    for line in data: # For every word in title +  description
        for word in line:
            if not word in vocabulary.keys() : # if the word is not in the dic
                vocabulary[word] = wid # adds it
                wid += 1 # Update word_id
                    
    with open("vocabulary", "wb") as f :
            pickle.dump(vocabulary, f) # Saves the vocabulary as a pickle
            
    return vocabulary # Returns the vocabulary





# This function create the first inverted index we need in the form "word (key) : [list of docs that contain word] (value)".
# It takes the number of (preprocessed) docs  and the path where they are saved and returns the reverted index as a dictionary.

def create_inverted_index(data):
    inverted_index = {} # Initializes the inverted index, in our case a dic
    
    for (idx,line) in enumerate(data):
        for word in line:
            if word in inverted_index.keys(): # if the word is in the inverted index
                    inverted_index[word] = inverted_index[word] + ["row_"+str(idx)] # adds the current doc to the list of docs that contain the word
            else :
                    inverted_index[word] = ["row_"+str(idx)] # else creates a record in the dic for the current word and doc

    with open("inverted_index", "wb") as f :
            pickle.dump(inverted_index, f) # Saves the inverted index as a pickle
            
    return inverted_index # returns the inverted index




# This function takes a term, a riverted index and the total number of docs in the corpus to compute the IDF of the term
        
def IDFi(term, inverted_index, number_of_rows):
    return math.log10(number_of_rows/len(inverted_index[term]))




# This function create the second inverted index we need in the form "word (key) : [(doc that contain the word, TFID of the term in the doc),....]"
# It takes the number of (preprocessed) docs, the path where they are saved, the vocabulary and a list containig all the idfs and returns the reverted index as a dictionary.

def create_inverted_index_with_TFIDF(data ,vocabulary, idfi):
    inverted_index2 = {} # Initializes the inverted index, in our case a dic
    
    for (idx,line) in enumerate(data):
        for word in line:
                if word in inverted_index2.keys() : # if the word is in the inverted index
                    # adds to the index line of the current word a tuple that contains the current doc and its TFID for the current word. It uses the vocabulary to get the index of the word
                    # in the IDF list.
                    inverted_index2[word] = inverted_index2[word] + [("row_"+str(idx),(line.count(word)/len(line))*idfi[vocabulary[word]])] # Just applying the def
                else :
                    # Makes the same initializing the index line of the current word
                    inverted_index2[word] = [("row_"+str(idx),(line.count(word)/len(line))*idfi[vocabulary[word]])]

    with open("inverted_index2", "wb") as f : # Saves the inverted index as a pickle
            pickle.dump(inverted_index2, f)
    
    return inverted_index2





#This function buils the second requested matrix.
def second_mat(data,inverted_index2,inverted_index,vocabulary):
    mat = np.zeros((len(data),len(vocabulary))) # Initializes the matrix
    count= 0
    
    #This loops search for every announcment the tfid of the 
    # words contained in the descripton and adds it in the right place
    # in the matrix using the id of the word in the vocabulary
    # (the columns are in the same order of the vocabulary)
    for (idx,line) in enumerate(data):
        for word in line:
            if "row_"+str(idx) in inverted_index[word]:
                ind_row = inverted_index[word].index("row_"+str(idx))
                mat[idx,vocabulary[word]]=inverted_index2[word][ind_row][1]
    
   
    with open("tfidf_matrix1", "wb") as f :
            pickle.dump(mat, f, ) # Saves the mat as a pickle
    return mat






# ----------SECTION 4: CLUSTERING----------

# This function simply  computes the k-means++ clustering for the selected range of ks
# and plots the elbow curve. If the param big_df is setted to 1 the specified range is 
# plotted using a step = 10
def elbow_method(X, k_max, big_df = 0 ):
    distortions = []
    if big_df == 1:
        K = range(1,k_max,10)
    else:
        K = range(1,k_max)
    for k in K:
        kmeanModel = KMeans(n_clusters=k,precompute_distances=True, n_init = 2, max_iter = 50 ).fit(X)
        kmeanModel.fit(X)
        distortions.append((sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])) #compute the distortion (the values calculated with the cost function)
    plt.plot(K, distortions, linewidth = 2.5, color = "orange")
    plt.xlabel('k')
    plt.ylabel('Cost')
    plt.title('The Elbow Method')
    plt.show()
    

    

# This function plots the clusters projected on a bidimensional space
# through a scatterplot, not so useful for high dimensional featuers
def clusters_plot(kmeans_object, data):
   
    fig, ax = plt.subplots()
    ax.scatter(x=data.iloc[:, 2], y = data.iloc[:, 4], c=kmeans_object.labels_.astype(np.float), s=200, label=kmeans_object.labels_.astype(np.float),
               alpha=0.3, edgecolors='k')

    plt.ylim (0,300)
    ax.grid(True)

    plt.show()
    
    
    
    

# This function takes a sklearn object or labels of a clustering process
# (the already_labels allows this double possibility) and returns a dic
# indexed by clusters id and containing the announcment belonging to that cluster
def clusters_dic(kmeans_object, already_labels = 0):
    if already_labels == 0:
        lab=kmeans_object.labels_
    else:
        lab = kmeans_object 
    C={}
    idx=0
    for i in lab:
        if not (i in C):
            C[i] = [idx]
        else:
            C[i].append(idx)
        idx+=1
    return C




# Just jaccard similiraty by def
def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))




# This function takes the two dics containing the clusters of the two matrices
# and the number k of largest similarity it has to return and computes the jaccard
# similaritiy of all the possible pair of clusters.
# It uses a max heap to compute the largest ks elements.
def cluster_jaccard(clust_1, clust_2, k):
    per = it.permutations(clust_1.keys(),2)
    sim = []
    for c in per:
        sim.append((jaccard_similarity(clust_1[c[0]],clust_2[c[1]]), ["mat 1 -> cluster: ",c[0],"mat 2 -> cluster: ",c[1]]))
    heapq._heapify_max(sim) # Creates a max heap based on the scores in "sim"
    res =  heapq.nlargest(k, sim, key=lambda x: x[0]) # Get the first k highest score elements of "sim"
    
    return res




# This function takes the complete matrix (check the notebook) and a list of sets that are the insetersections of the most similiar clusters and computes, for each of them, a huge string containing the descriptions
# of the announcements present in the intersections.
def wordcloud(list_of_int,data):
    
    text = []
    for el in list_of_int:
        string_ = ""
        for ann in el:
            string_ += data.iloc[ann] 

        text.append(string)
    
    return(text)










# ----------BONUS SECTION----------

# This calss implements k-means from scratch.
# The constructor takes a df in which each row is a point, the number of
# clusters k and the maximum number of iterations.
class kmeans:
    
    def __init__(self, Xt, k, max_steps):
        
        # The following  string build the matrix X of points (not a df)
        # that will be used to cluster
        Xt1 = Xt.reset_index(inplace = False) 
        Xt1 = Xt1.iloc[:,1:]
        self.X = Xt1.values
        
        # Initializer randomly the centroids
        self.centroids = self.X[np.random.choice(self.X.shape[0],k, replace = False),:]
        
        # Initializes a dic that will be indexed by the clusters ID and
        # contain the announcements of each cluster
        self.clusters = dict()
        
        # Here starts theclustering (check the methods)
        self.assign_points(self.X)
        self.run(self.X, max_steps)
          
        # Now that we have the clusters, we need to get the labels,
        # so assign at each announcement, in the same order they're in the matrix,
        # the corresponding cluster. To do this, we implemented an inverted index
        # indexeded by the values of the announcements and containing their indexes.
        # Some announcements have the same values, we append the indexes.
        #(ie, key <- [1,2,22000,114] : index_in_the_mat <- 5
        inverted = {}
        for idx,ann in Xt1.iterrows():
            ass = str(ann.tolist())
            if  ass in inverted.keys():
                inverted[ass].append(idx)
            else:
                inverted[ass] = [idx]
        
        # In this way we can use an ordered dict indexed by the indexes of the announcements in the mat
        # and containig the cluster ids corresponding to them.
        # Everytime it picks an index, that index is poped from the 
        # inverted index above.
        lab = OrderedDict()
        for num,cluster in enumerate(self.clusters.values()):
            for lang in cluster :
                    ass1 = str(lang.tolist()) 
                    lab[reverted[ass1][0]] = num
                    reverted[ass1].pop(0)
        od = OrderedDict(sorted(lab.items())) # This step is fundamental to have the labels in the same order of the announcements
        self.labels = od.values() # This is, finally, the vector containing, for each announcement, and in the same order of the mat,
                                  # the numebr of the cluster that announcement belongs to.
    
    # The following method  reassigns the centroids by computing the mean of
    # the points belonging to the clusters
    def recompute_centroids(self):
        self.centroids = [np.array(self.clusters[el]).mean(axis = 0) for el in self.clusters.keys()]
    
    
    # This method it's the core one. It recomputes the centroids 
    # and reassings the points at every iteration
    def run(self,X,max_steps):
        for iterat in range(max_steps):
            self.recompute_centroids()
            self.assign_points(X)
    
    
    
    # This method assign points to clusters by computing the distance matrix
    # of the point (rows) from the centroids (columns) and taking the argmin of
    # every row (so the number of the column is the cluster at which the point
    # of the current row has to be assigned.
    def assign_points(self,X):
            dis_mat = distance_matrix(X, self.centroids)
            tmp_clust = dict()
            for (idx,row) in enumerate(dis_mat):
                if np.argmin(row) in tmp_clust.keys():
                    tmp_clust[np.argmin(row)].append(self.X[idx,:])
                else:
                    tmp_clust[np.argmin(row)] = [self.X[idx,:]]
            self.clusters = tmp_clust
            
            
            
            
            
            
            
            
            
            
# ----------SECTION 5: FIND DUPLICATES!----------

# This is the first presented hash function
def first_hash(file_name):
     # Initializes the two lists that will allow to find duplicates
    unique = set()
    duplicate = []
    SIZE =  1787178291199 # Hash table size
    
    # Opens the file 
    with open(file_name,"r") as f:
        for line in f: # For every password..
            line = line[:-1] # Cleans the \\n chars
            sums = 0 # Initializes the number we associate to the string
            for c in line: # For every char in the password...
                sums = (sums  ^ ord(c)*(37**17)); # XORs to the current value with the ascii value of the current char and times a wide number
            sums = sums%SIZE # Maps that number to the choosen size of the hash table
    
            if (sums in unique): # if the number has been already computed at least one time
                duplicate.append(line) # Appends the password to the list of duplicates
            else:
                unique.add(sums) #Otherwise traces it's first appearance
    return duplicate # returns the duplicate



# This function implements a first logic to find false positives.
#The function is similiar to the previous one. The different parts are commented.(for the rest of the lib,too)
def first_hash_e(file_name):
    unique = set()
    duplicate = []
    table = dict() # The hash table (not properly by def,of course, but something similiar :D )
    SIZE =  1787178291199
    with open(file_name,"r") as f:
        for line in f:
            line = line[:-1]
            sums = 0;
            for c in line:
                sums = (sums  ^ ord(c)*(37**17)); # adds to the number the ascii value of the current char
            sums = sums%(SIZE)
        
            fp = 0
            if sums in table.keys(): # If the value is already in the table
                if sorted(table[sums]) != sorted(line): #if the current password contains at least one differnt chars from the password 
                                                        #stored in the table
                    fp += 1 # counts a false positive
            else:
                table[sums] = line # Otherwise adds the current password to the table
            
            if sums in unique:
                duplicate.append(line)
            else:
                unique.add(sums)
    return fp, duplicate




#The two following functions implements a more accurated way to check for false positives.
#The first one is exactly the previous but has an improvment: it returns also a dict containing tha hash values and the corresponding passwords
def first_hash_true(file_name):
    unique = set()
    duplicate = []
    table = dict()
    SIZE =  1787178291199
    with open(file_name,"r") as f:
        for line in f:
           
            line = line[:-1]
            sums = 0;
            for c in line:
                sums = (sums  ^ ord(c)*(37**17)); # adds to the number the ascii value of the current char
            sums = sums%(SIZE)
        
            if sums in table.keys():
                table[sums].append(line)
            else:
                table[sums] = [line]
            
            if sums in unique:
                duplicate.append(line)
            else:
                unique.add(sums)
    return duplicate,table





#Cycling on the table, this function returns FPs.
def find_fps(table):    
    fp = 0
    for el in table: 
        # Takes the values of the table that are associated to more than one password and for each of them
        #Counts a false positive only if  there is an element that is not a permutation of the others
        #(if there are permutations of three different set of chars maybe it's the case to redesign the function rather than 
        # check for false positives,as in this case :D ...)
        if len(table[el])>1:
            lst=list(map(sorted, table[el]))
            if lst[1:] != lst[:-1]:
                fp += 1
    return fp
    




#This is the first hash function that takes in account the order of the characters.
def o_hash_1(file_name):
    unique = set()
    duplicate = []
    SIZE =  1787178291199
    with open(file_name,"r") as f:
        for line in f:
            line = line[:-1]
 
            # The following loop builds iteratively the polynomial function explained in the notes 
            # (easy to see)
            h = 0
            for c in line:
                h = h*37 + ord(c);
            h = h%SIZE
 
            if (h in unique):
                duplicate.append(line)
            else:
                unique.add(h)
    return duplicate




# The second hash function that takes in account the order of the characters.
def o_hash_2 (file_name):
    A = 719 # all primes
    B = 137
    SIZE =   1787178291199
    unique = set()
    duplicate = []
    with open(file_name,"r") as f:
        for line in f:
            line = line[:-1]
            f = 37 # f(0)
            
            # Just implements the explained funtion
            for c in line:
                f = (f * A) ^ (ord(c) * B)
            h = f%SIZE
            
            
            if (h in unique):
                duplicate.append(line)
            else:
                unique.add(h)
    return duplicate




# The logic of false-positives detection is the same as previous, just returning the table and not only the duplicates. Then is sufficient using "find_fps"
def o_hash_1_fp(file_name):
    unique = set()
    duplicate = []
    table = dict()
    SIZE =  1787178291199
    with open(file_name,"r") as f:
        for line in f:
            line = line[:-1]
           
            h = 0
            for c in line:
                h = h*37 + ord(c);
            h = h%SIZE
            
            if h in table.keys():
                table[h].append(line)
            else:
                table[h] = [line]
 
            if (h in unique):
                duplicate.append(line)
            else:
                unique.add(h)
                
    fp = 0
    for el in table: 
        if len(table[el])>1:
            lst=list(map(sorted, table[el]))
            if lst[1:] != lst[:-1]:
                fp += 1
    return duplicate,fp




def o_hash_2_fp (file_name):
    A = 719 # all primes
    B = 137
    SIZE =  1787178291199
    unique = set()
    duplicate = []
    table = dict()
    with open(file_name,"r") as f:
        for line in f:
            line = line[:-1]
            h = 37
            for c in line:
                h = (h * A) ^ (ord(c) * B)
            h = h%SIZE
            
            if h in table.keys():
                table[h].append(line)
            else:
                table[h] = [line]
            
            if (h in unique):
                duplicate.append(line)
            else:
                unique.add(h)
                
    fp = 0
    for el in table: 
        if len(table[el])>1:
            lst=list(map(sorted, table[el]))
            if lst[1:] != lst[:-1]:
                fp += 1
    return duplicate,fp
    

