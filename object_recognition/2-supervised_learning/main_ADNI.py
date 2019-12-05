import numpy as np
from nilearn import plotting
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from time import time
import matplotlib.pyplot as plt
import itertools
from sklearn import decomposition


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

dim = 2
working_dir = "./"
with np.load(working_dir + "Data_ADNI.npz") as data:
    X = data["X"]  # ROIs
    y = data["y"]  # classes
    labels = data["labels"]

N, M = X.shape  # number subjects and ROIs
class_names = ["control", "alzheimer"]  # y = 0, y = 1

plotting.plot_roi("atlas.nii", title="Atlas")
plotting.show()

# %%
# Shuffle data randomly
# indeces = np.arange(N)
#
# np.random.shuffle(indeces)
# XpGPA = X[indeces]
# Xp = X[indeces]
# y = y[indeces]
# labels = labels[indeces]

# %%

# Scale data (each feature will have average equal to 0 and unit variance)
scaler = StandardScaler()
scaler.fit(X)
X_scale = scaler.transform(X)

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X_scale, np.ravel(y), test_size=0.33, random_state=42)

# Fitting LDA
print("Fitting LDA to training set")
t0 = time()
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
print(classification_report(y_test, y_pred))

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
lda_score = cross_val_score(lda, X=X_scale, y=np.ravel(y), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(lda_score.mean(), lda_score.std()))

# Fitting QDA
print("Fitting QDA")
t0 = time()
qda = QuadraticDiscriminantAnalysis()
qda_score = cross_val_score(qda, X=X_scale, y=np.ravel(y), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(qda_score.mean(), qda_score.std()))

# Fitting Logistic Regression
print("Fitting Logistic Regression")
t0 = time()
log = LogisticRegression(solver='lbfgs')
log_score = cross_val_score(log, X=X_scale, y=np.ravel(y), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(log_score.mean(), log_score.std()))

# Fitting K-nearest neighbour
print("Fitting K-nearest neighbour")
t0 = time()
neigh = KNeighborsClassifier(n_neighbors=3)
neigh_score = cross_val_score(neigh, X=X_scale, y=np.ravel(y), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(neigh_score.mean(), neigh_score.std()))

# %%
################################################################################
################################################################################
############################### Data Reduction #################################
# Use PCA
##############################################

pca = decomposition.PCA(n_components=0.9)  # use number of components take explain 90% of variability
pca.fit(X_scale)
X_pca = pca.transform(X_scale)

print("\n We use PCA ---------------------------------------------------------- \n")

# Fitting LDA
print("Fitting LDA")
t0 = time()
lda = LinearDiscriminantAnalysis()
lda_score = cross_val_score(lda, X=X_pca, y=np.ravel(y), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(lda_score.mean(), lda_score.std()))

# Fitting QDA
print("Fitting QDA")
t0 = time()
qda = QuadraticDiscriminantAnalysis()
qda_score = cross_val_score(qda, X=X_pca, y=np.ravel(y), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(qda_score.mean(), qda_score.std()))

# Fitting Logistic Regression
print("Fitting Logistic Regression")
t0 = time()
log = LogisticRegression(solver='lbfgs')
log_score = cross_val_score(log, X=X_pca, y=np.ravel(y), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(log_score.mean(), log_score.std()))

# Fitting K-nearest neighbour
print("Fitting K-nearest neighbour")
t0 = time()
neigh = KNeighborsClassifier(n_neighbors=3)
neigh_score = cross_val_score(neigh, X=X_pca, y=np.ravel(y), cv=5)
print("done in %0.3fs" % (time() - t0))
print(" Average and std CV score : {0} +- {1}".format(neigh_score.mean(), neigh_score.std()))

# %%

for i in range(len(labels)):
    tmp = ""
    for char in labels[i,:]:
        tmp += char
    print(tmp)