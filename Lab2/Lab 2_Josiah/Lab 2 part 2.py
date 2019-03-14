from scipy import stats
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns; sns.set()
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import pandas

colnames = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width',
            'class']
data = pandas.read_csv('iris.csv', names=colnames,skiprows = (0,1))
data.drop(columns='class')
#This line removes all rows of the data that have an element that is more than
# 3 standard deviations away from the mean of the coloumn


X = data['sepal_length']
Y = data['sepal_width']
#displays the pre cleaned data
plt.show(sns.scatterplot(X,Y))
#Exptracts the two columns of interest for cleaning
xy = data[['sepal_length','sepal_width']]
#cleans the data
#This line removes all rows of the data that have an element that is more than
# 3 standard deviations away from the mean of the coloumn
xy_cleaned = xy[(np.abs(stats.zscore(xy)) < 3).all(axis =1)]

#extracts data from cleaned data set
x1 = xy_cleaned['sepal_length']
x2 = xy_cleaned['sepal_width']


plt.plot()
plt.title('Cleaned Dataset')
plt.show(sns.scatterplot(x1,x2))

# create new plot and data
plt.plot()
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']

# k means determine k
distortion = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortion.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortion, 'bx-')
plt.xlabel('k')
plt.ylabel('Error Distortion')
plt.title('Elbow Method')
plt.show()



