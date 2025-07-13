import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


df = pd.read_json("banking_complaints_dataset1k.json")


#basic dataset profiling

print(f"dataset shape : {df.shape}")
print(f"Missing Values : {df.isnull().sum}")
print(f"Duplicated rows : {df.duplicated().sum}")


df['input_length']=df['input'].str.len()
df['output_length']=df['output'].str.len()

print(f"Input text length stats:\n{df['input_length'].describe()}")
print(f"Output text length stats:\n{df['output_length'].describe()}")

# anomalies and outliers 
vectorizer = TfidfVectorizer(max_features=1000,stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['input'])

clustering = DBSCAN(eps)