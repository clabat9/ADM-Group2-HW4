# ADM-Group26-HW4
#  Part 1) Does basic house information reflect house's description?


![alt text](https://www.socialandtech.net/wp-content/uploads/2018/03/Immobiliare.png?style=centerme)

In this assignment we will perform a clustering analysis of house announcements in Rome from Immobiliare.it. 

## Get data to analyze 

The data in usage are scraped from [*immobiliare.it*](https://www.immobiliare.it/).

# Part 2) Find the duplicates!


![alt text](https://ds055uzetaobb.cloudfront.net/image_optimizer/8fe3d7d3de4f4c83c255dbe78471d2625aa03640.png)


We are given ***passwords2.txt*** file as input. Each row corresponds to a string of 20 characters. We define three hash functions that associate a value to each string. In this case, the goal is to check whether there are some duplicate strings.

The first function doesn't take in account the order of the characters so, i.e., "AAB" and "ABA" are considered duplicates.

The second and third functions take in account the order of the characters so, i.e., "AAB" amd "ABA" are not considered duplicates-

**NB** : we suggest to **download the notebook file** because html doesn't work well with the rendering of some latex formulas and pandas DataFrames. 

## Script and Other files descriptions

1. __`hw4_lib.py`__: 
	This script contains all the useful functions to get the proposed requests, deeply commented to have a clear view of the logic behind.
  
## `IPython Notebook: Homework4_final.ipynb `
The goal of the `Notebook` is to provide a storytelling friendly format that shows how to use the implemented code and to carry out the reesults. It provides explanations and examples.
We tried to organize it in a way the reader can follow our logic and conlcusions.
Obviously, it is splitted in two parts,one for each main topic of the assignment.
