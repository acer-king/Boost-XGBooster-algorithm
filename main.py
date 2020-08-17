from sklearn.feature_selection import VarianceThreshold

X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
selector = VarianceThreshold(threshold=0.1)
X = selector.fit_transform(X)
print(selector.get_params())