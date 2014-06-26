import nltk
from nltk.corpus import stopwords
import os
import re

# NIPS dataset can be downloaded from: http://www.cs.nyu.edu/~roweis/data/nips12raw_str602.tgz
# Location of the NIPS dataset
parent_folder = '/Users/test/Downloads/nipstxt'
# Find the folders that contain all the papers from each year from 1987 to 1999
nips_folders = [os.path.join(parent_folder, i) for i in os.listdir(parent_folder) if i.startswith('nips')]

# Location to save the resulting file
destination_file = '/Users/test/Documents/nips.txt'

pattern = re.compile('^[A-Za-z]+$')
wnl = nltk.stem.WordNetLemmatizer()

papers = list()
incidence_dict = dict()

for folder in nips_folders:
	files = os.listdir(folder)
	txt_files = [f for f in files if f.endswith('.txt')]
	for f in txt_files:
		path = os.path.join(folder, f)
		reader = open(path, 'rU')
		raw = reader.read()
		# Tokenize the document
		tokens = nltk.word_tokenize(raw)
		text = nltk.Text(tokens)
		# Extract the words that have only alphabetical characters, remove stopwords among them, and lemmatize the resulting set of words
		filtered_text = [wnl.lemmatize(i.lower()) for i in text if pattern.match(i) and i.lower() not in stopwords.words('english')]
		papers.append(filtered_text)
		# Build incidence dictionary which holds each word in the vocabulary and number of times it is found in the documents
		for word in filtered_text:
			if word in incidence_dict:
				incidence_dict[word] += 1
			else:
				incidence_dict[word] = 1

# Extract the frequent words
frequent_words = set([word for word in incidence_dict.keys() if incidence_dict[word] > 50])

# Dicretize the words
word_dict = dict()
index = 0
for word in frequent_words:
	word_dict[word] = index
	index += 1

f = open(destination_file, 'w')

# Write the resulting documents to a single file
for n, paper in enumerate(papers):
	filtered_paper = [word_dict[word] for word in paper if word in frequent_words]
	line = str(n) + ','
	for n,w in enumerate(filtered_paper):
		if n != 0:
			line += ' '
		line += str(w)
	line += '\n'
	f.write(line)

f.close()