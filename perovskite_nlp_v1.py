#@Author: otrejo
#@Date:   2020-04-09T18:50:06-04:00
# @Last modified by:   otrejo
# @Last modified time: 2020-04-09T21:05:36-04:00

# Import libraries
import docx
import pandas as pd
import numpy as np
import spacy
import scispacy
nlp = spacy.load("en_core_sci_lg")
from spacy import displacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span, Token
from summa import summarizer
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA


# Make list of papers
files = os.listdir('papers/')[1:]
print(files)

# Make dictionary of papers
doc_names = ['doc_{}'.format(i) for i in range(len(files))]
doc_dict = {}

# Make doc2vec matrix
for i in range(len(files)):
    doc = docx.Document("papers/"+files[i])
    doc_dict[doc_names[i]] = [p.text for p in doc.paragraphs if len(p.text) > 150]

doc_i = []
par_i = []
for i in range(len(doc_names)):
    for p in doc_dict[doc_names[i]]:
        p_nlp = nlp(p)
        doc_i.append(i)
        par_i.append(p_nlp.vector)

X_i = np.stack(par_i)

pca = PCA(3)  # project to 2 dimensions
project_i = pca.fit_transform(X_i)
print(X_i.shape)
print(project_i.shape)

plt.scatter(project_i[:, 0], project_i[:, 1],
            c=doc_i, edgecolor='none', alpha=1,
            cmap=plt.cm.get_cmap('plasma', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();
plt.show()


# Plot initialisation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pnt3d = ax.scatter(project_i[:, 0], project_i[:, 1], project_i[:, 2], c=doc_i,
                    cmap=plt.cm.get_cmap('Dark2', 10), s=60)
cbar = plt.colorbar(pnt3d)
cbar.set_label("Paper No.")
# make simple, bare axis lines through space:
xAxisLine = ((min(project_i[:, 0]), max(project_i[:, 0])), (0, 0), (0,0))
ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
yAxisLine = ((0, 0), (min(project_i[:, 1]), max(project_i[:, 1])), (0,0))
ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
zAxisLine = ((0, 0), (0,0), (min(project_i[:, 2]), max(project_i[:, 2])))
ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

# label the axes
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA on 10 perovskite papers")
ax.view_init(0, 90)
plt.show()

def find_indices(arr):
    indices = [0]
    for i in range(1, len(arr)):
        if arr[i-1] != arr[i]:
            indices.append(i)
    indices.append(len(arr)-1)
    return indices

indices = find_indices(doc_i)

doc_vecs = []
for i in range(1, len(indices)):
    doc_vecs.append(sum(X_i[indices[i-1]:indices[i]]))

X_ii = np.stack(doc_vecs)
y_ii = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

pca = PCA(3)  # project to 2 dimensions
project_i = pca.fit_transform(X_ii)
print(X_ii.shape)
print(project_i.shape)

# Plot initialisation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pnt3d = ax.scatter(project_i[:, 0], project_i[:, 1], project_i[:, 2],
                    c=y_ii, cmap=plt.cm.get_cmap('Dark2', 12), s=120)
cbar = plt.colorbar(pnt3d)
cbar.set_label("Paper No.")

# make simple, bare axis lines through space:
xAxisLine = ((min(project_i[:, 0]), max(project_i[:, 0])), (0, 0), (0,0))
ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
yAxisLine = ((0, 0), (min(project_i[:, 1]), max(project_i[:, 1])), (0,0))
ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
zAxisLine = ((0, 0), (0,0), (min(project_i[:, 2]), max(project_i[:, 2])))
ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

# label the axes
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA on 10 perovskite papers")
#ax.view_init(0, 90)
plt.show()
