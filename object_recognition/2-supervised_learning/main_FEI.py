# -*- coding: utf-8 -*-

"""
	Author: Pietro Gori
	Creation Date: 06/02/2018
"""
# %%
import numpy as np
from time import time
import sklearn
import matplotlib

import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import paired_distances
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print('The numpy version is {}.'.format(np.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The matplotlib version is {}.'.format(matplotlib.__version__))

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


# Code from scikit-learn
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="red")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# %%
# Parameters
dim = 2  # dimension
Working_directory = "./"
with np.load(Working_directory + 'Data_FEI.npz') as data:
    Images = data['Images_FEI']  # list of images
    X = data['Landmarks_FEI']  # original landmarks
    XGPA = data['Landmarks_FEI_GPA']  # landmarks after GPA
    Y = data['Emotions_FEI']  # class, 0 for neutral and 1 for happy
    Names = data['Names_FEI']
N, M = X.shape  # number subjects and landmarks respectively
M = int(M / 2)
class_names = ["neutral", "happy"]

# %%
# plot first 6 images 
for i in range(0, 6):
    image = Images[i, :, :]
    plt.figure()
    plt.imshow(image, cmap='gray', origin='upper')
    landmark = X[i, :]
    x = landmark[::2]
    y = landmark[1::2]
    plt.plot(x, y, 'o', label = str(Y[i]))
    plt.legend(loc = 'best')
    plt.show()

# %%
# Shuffle data randomly
indeces = np.arange(N)

np.random.shuffle(indeces)
XpGPA = XGPA[indeces]
Xp = X[indeces]
Yp = Y[indeces]
Imagesp = Images[indeces]
Namesp = [''] * N
for i in range(0, N):
    Namesp[i] = Names[indeces[i]]

# %%
# Plot all landmarks after GPA
Xmean = np.mean(XpGPA, axis=0)  # Compute average
plt.figure()
for i in range(0, N):
    landmark = XpGPA[i]
    x = landmark[::2]
    y = landmark[1::2]
    if Yp[i].astype(int) == 0:
        neutral = plt.scatter(x, y, c='b')
    else:
        happy = plt.scatter(x, y, c='r')
average = plt.scatter(Xmean[::2], Xmean[1::2], color='k')
plt.legend((neutral, happy, average), ('neutral', 'happy', 'average'))
plt.gca().invert_yaxis()

# %%
################################################################################
################################################################################
################################################################################

# Compute distances from the average configuration (features)
dist_average = np.zeros((N, M))
average = np.reshape(Xmean, (M, 2))
for i in range(N):
    landmark = np.reshape(XpGPA[i], (M, 2))
    dist_average[i] = paired_distances(landmark, average)

# Scale data (each feature will have average equal to 0 and unit variance)
scaler = StandardScaler()
scaler.fit(dist_average)
dist_average_scale = scaler.transform(dist_average)

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(dist_average_scale, np.ravel(Yp), test_size=0.33, random_state=42)

# Fitting LDA
print("Fitting LDA to training set")
t0 = time()
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
print(classification_report(y_test, y_pred))

# Question 6 : we use directly the positions
# # Scale data (each feature will have average equal to 0 and unit variance)
# scaler = StandardScaler()
# scaler.fit(XpGPA)
# positions_scale = scaler.transform(XpGPA)
#
# # Create training and test set
# X_train, X_test, y_train, y_test = train_test_split(positions_scale, np.ravel(Yp), test_size=0.33, random_state=42)
#
# # Fitting LDA
# print("Fitting LDA to training set")
# t0 = time()
# lda = LinearDiscriminantAnalysis()
# lda.fit(X_train, y_train)
# y_pred = lda.predict(X_test)
# print(classification_report(y_test, y_pred))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()

# %%
## Cross-validation

# Fitting LDA
print("Fitting LDA")
t0 = time()
lda = LinearDiscriminantAnalysis()
lda_score = cross_val_score(lda, X=dist_average_scale, y=np.ravel(Yp), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(lda_score.mean(), lda_score.std()))

# Fitting QDA
print("Fitting QDA")
t0 = time()
qda = QuadraticDiscriminantAnalysis()
qda_score = cross_val_score(qda, X=dist_average_scale, y=np.ravel(Yp), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(qda_score.mean(), qda_score.std()))

# Fitting Logistic Regression
print("Fitting Logistic Regression")
t0 = time()
log = LogisticRegression(solver='lbfgs')
log_score = cross_val_score(log, X=dist_average_scale, y=np.ravel(Yp), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(log_score.mean(), log_score.std()))

# Fitting K-nearest neighbour
print("Fitting K-nearest neighbour")
t0 = time()
neigh = KNeighborsClassifier(n_neighbors=3)
neigh_score = cross_val_score(neigh, X=dist_average_scale, y=np.ravel(Yp), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(neigh_score.mean(), neigh_score.std()))

# %%
################################################################################
################################################################################
################################################################################

# Use distances between all combinations of landmarks. Each subject has M*(M-1)/2 features
dist_combination = np.zeros((N, int((M * (M - 1) / 2))))
for s in range(N):
    temp = []
    landmarks = np.reshape(XpGPA[s], (M, 2))
    for i in range(M - 1):
        a = landmarks[i, :]
        for j in range(i + 1, M):
            b = landmarks[j, :]
            dist_2 = np.sqrt(np.dot(a, a) - 2 * np.dot(a, b) + np.dot(b, b))
            temp.append(dist_2)
    dist_combination[s] = np.array(temp)

# Scale data (each feature will have average equal to 0 and unit variance)
scaler.fit(dist_combination)
dist_combination_scale = scaler.transform(dist_combination)

# Fitting LDA
print("Fitting LDA")
t0 = time()
lda_score = cross_val_score(lda, X=dist_combination_scale, y=np.ravel(Yp), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(lda_score.mean(), lda_score.std()))

# Fitting QDA
print("Fitting QDA")
t0 = time()
qda_score = cross_val_score(qda, X=dist_combination_scale, y=np.ravel(Yp), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(qda_score.mean(), qda_score.std()))

# Fitting Logistic Regression
print("Fitting Logistic Regression")
t0 = time()
log = LogisticRegression(solver='lbfgs')
log_score = cross_val_score(log, X=dist_combination_scale, y=np.ravel(Yp), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(log_score.mean(), log_score.std()))

# Fitting K-nearest neighbour
print("Fitting K-nearest neighbour")
t0 = time()
neigh = KNeighborsClassifier(n_neighbors=3)
neigh_score = cross_val_score(neigh, X=dist_combination_scale, y=np.ravel(Yp), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(neigh_score.mean(), neigh_score.std()))

# %%
################################################################################
################################################################################
############################### Data Reduction #################################
# Use PCA
##############################################

pca = decomposition.PCA(n_components=0.9)  # use number of components take explain 90% of variability
pca.fit(dist_combination_scale)
dist_combination_pca = pca.transform(dist_combination_scale)

# Fitting LDA
print("Fitting LDA")

t0 = time()
lda_score = cross_val_score(lda, X=dist_combination_pca, y=np.ravel(Yp), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(lda_score.mean(), lda_score.std()))

# Fitting QDA
print("Fitting QDA")

t0 = time()
qda_score = cross_val_score(qda, X=dist_combination_pca, y=np.ravel(Yp), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(qda_score.mean(), qda_score.std()))

# Fitting Logistic Regression
print("Fitting Logistic Regression")

t0 = time()
log = LogisticRegression(solver='lbfgs')
log_score = cross_val_score(log, X=dist_combination_pca, y=np.ravel(Yp), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(log_score.mean(), log_score.std()))

# Fitting K-nearest neighbour
print("Fitting K-nearest neighbour")

t0 = time()
neigh = KNeighborsClassifier(n_neighbors=3)
neigh_score = cross_val_score(neigh, X=dist_combination_pca, y=np.ravel(Yp), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(neigh_score.mean(), neigh_score.std()))

# %%
#############################################################################
######################## Selection of few landmarks #########################
#############################################################################

# Select lateral landmarks mouth
#select_land = [0, 55, 61]
# Select landmarks with maximum variance
Xstd = np.std(XpGPA, axis=0)
Xstd2 = Xstd[::2] + Xstd[1::2]
Xstd_indexed = np.vstack((Xstd2, np.arange(Xstd2.shape[0])))
Xstd_sorted = Xstd_indexed[:, Xstd_indexed[0,:].argsort()]
select_land = [Xstd_sorted[1, -1], Xstd_sorted[1, -2], Xstd_sorted[1, -3]]

indeces_central = []
for k in range(0, len(select_land)):
    indeces_central.append(select_land[k] * 2 - 2)
    indeces_central.append(select_land[k] * 2 - 1)

indeces_central = np.array(indeces_central, dtype=int)
Ms = int(len(indeces_central) / 2)
Xps = np.zeros((N, Ms * dim))
XpsGPA = np.zeros((N, Ms * dim))
for i in range(0, N):
    XpsGPA[i, :] = XpGPA[i, indeces_central]
    Xps[i, :] = Xp[i, indeces_central]

Yps = Yp

# plot two test images
for i in range(0, 2):
    image = Imagesp[i, :, :]
    plt.figure()
    plt.imshow(image, cmap='gray', origin='upper')
    landmark = Xps[i, :]
    x = landmark[::2]
    y = landmark[1::2]
    # x,y=np.split(landmark,2)
    plt.plot(x, y, 'o')
    plt.show()

# %%
# Plot landmarks
plt.figure()
for i in range(0, N):
    landmark = XpsGPA[i]
    x = landmark[::2]
    y = landmark[1::2]
    if Yps[i].astype(int) == 0:
        neutral = plt.scatter(x, y, c='b')
    else:
        happy = plt.scatter(x, y, c='r')

plt.legend((neutral, happy), ('neutral', 'happy'))
plt.gca().invert_yaxis()
plt.show()

# %%
# Fitting LDA
print("Fitting LDA")

# t0 = time()
lda = LinearDiscriminantAnalysis()
lda_validate = cross_validate(lda, X=XpsGPA, y=np.ravel(Yps), cv=5, n_jobs=-1, return_train_score=True,
                              return_estimator=True)
print(" Average and std train score : {0} +- {1}".format(lda_validate['train_score'].mean(),
                                                         lda_validate['train_score'].std()))
print(" Average and std test score : {0} +- {1}".format(lda_validate['test_score'].mean(),
                                                        lda_validate['test_score'].std()))
best_estimator = lda_validate['estimator'][np.argmax(lda_validate['test_score'])]
C = best_estimator.predict(XpsGPA)
error = np.ravel(np.array(np.where(np.abs(C - np.ravel(Yps)))))
if len(error) > 5:
    kk = 5
else:
    kk = len(error)

# plot error images
for i in range(0, kk):
    image = Imagesp[error[i], :, :]
    plt.figure()
    plt.imshow(image, cmap='gray', origin='upper')
    landmarkALL = Xp[error[i], :]
    landmark = Xps[error[i], :]
    xALL = landmarkALL[::2]
    yALL = landmarkALL[1::2]
    x = landmark[::2]
    y = landmark[1::2]
    plt.plot(xALL, yALL, 'ob')
    plt.plot(x, y, 'or')
    if C[error[i]] == 0:
        plt.title('Image ' + Namesp[error[i]] + ' predicted as neutral')
    elif C[error[i]] == 1:
        plt.title('Image ' + Namesp[error[i]] + ' predicted as happy')
    plt.show()

# %%
# Fitting QDA
print("Fitting QDA")

t0 = time()
qda = QuadraticDiscriminantAnalysis()
qda_score = cross_val_score(qda, X=XpsGPA, y=np.ravel(Yps), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(qda_score.mean(), qda_score.std()))

# Fitting Logistic Regression
print("Fitting Logistic Regression")
t0 = time()
log = LogisticRegression(solver='lbfgs')
log_score = cross_val_score(log, X=XpsGPA, y=np.ravel(Yp), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(log_score.mean(), log_score.std()))

# Fitting K-nearest neighbour
print("Fitting K-nearest neighbour")
t0 = time()
neigh = KNeighborsClassifier(n_neighbors=3)
neigh_score = cross_val_score(neigh, X=XpsGPA, y=np.ravel(Yp), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(neigh_score.mean(), neigh_score.std()))
