import numpy as np
import math

class Node():
    def __init__(self, node_indxs, leaf=False):
        self.node_indxs = node_indxs
        if not leaf:
            self.is_leaf = leaf
            self.left = None
            self.right = None
            self.split_threshold = None
            self.impurity_before = None
            self.feature_index = None
        else:
            self.is_leaf = leaf
            self.predicted_value = None
            self.leaf_impurity = None
            self.class_counts = None

class DecisionTreeClassifier():
    def __init__(self, max_depth=7):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        self.n_classes = np.max(y) + 1
        
        if len(X.shape) == 2:
            self.n_features = X.shape[1]
        else:
            self.n_features = 1
            X = X[:, None]


        full_indexes = np.arange(X.shape[0])
        self.root = Node(full_indexes)
        self.root.impurity_before = self.calc_gini_(y)

        self.grow_tree_(X, y, self.root, 0)

    def predict(self, X):
        if len(X.shape) == 1:
            X = X[None, :]
        preds = np.fromiter((self.predict_one_(x) for x in X), dtype=float)
        return preds

    def predict_proba(self, X):
        if len(X.shape) == 1:
            X = X[None, :]
        probs = np.vstack([self.predict_proba_one_(x) for x in X])
        return probs

    def find_leaf_(self, x):
        c_node = self.root
        while not c_node.is_leaf:
            c_thresh = c_node.split_threshold
            c_feature_index = c_node.feature_index
            if x[c_feature_index] <= c_thresh:
                c_node = c_node.left
            else:
                c_node = c_node.right
        return c_node

    def predict_proba_one_(self, x):
        leaf = self.find_leaf_(x)
        N = np.sum(leaf.class_counts)
        probs = leaf.class_counts / N
        return probs

    def predict_one_(self, x):
        leaf = self.find_leaf_(x)
        return leaf.predicted_value


    def grow_tree_(self, X, y, root: Node, depth):
        if depth >= self.max_depth or math.isclose(root.impurity_before, 0.0):
            self.change_to_leaf_(root, y)
            return
        
        gain, feature_index, tresh = self.find_best_split_(root, X, y)

        if gain > 1e-6:
            root.feature_index = feature_index
            root.split_threshold = tresh
            self.create_children_(feature_index, tresh, X, y, root, root.node_indxs)
        else:
            self.change_to_leaf_(root, y)
            return

        self.grow_tree_(X, y, root.left, depth + 1)
        self.grow_tree_(X, y, root.right, depth + 1)
        


    def make_split_one_feature_(self, X, y):
        if self.n_classes is None:
            self.n_classes = np.max(y) + 1
        
        N = X.shape[0]

        order = np.argsort(X, kind='mergesort')
        X_sorted = X[order]
        y_sorted = y[order]

        total = np.bincount(y, minlength=self.n_classes)
        left = np.zeros(self.n_classes, dtype=int)
        right = total.copy()

        s_left = 0
        s_right = (right ** 2).sum()

        G_parent = 1 - s_right / (N ** 2)

        best_gain = -np.inf
        best_thresh = None
        best_imp = None

        for i in range(N - 1):
            c = y_sorted[i]
            s_left += 2 * left[c] + 1
            s_right -= 2 * right[c] - 1

            left[c] += 1
            right[c] -= 1

            if X_sorted[i] == X_sorted[i + 1]:
                continue

            NL = i + 1
            NR = N - NL

            G_left = 1 - s_left / (NL ** 2) if NL > 0 else 0
            G_right = 1 - s_right / (NR ** 2) if NR > 0 else 0
            G_split = (NL * G_left + NR * G_right) / N

            gain = G_parent - G_split
            if gain > best_gain:
                best_gain = gain
                best_imp = G_split
                best_thresh = 0.5 * (X_sorted[i] + X_sorted[i+1])
        
        return best_thresh, best_imp, best_gain


    def find_best_split_(self, node: Node, X, y):
        
        cur_indxs = node.node_indxs
        cur_data = X[cur_indxs]
        cur_labels = y[cur_indxs]

        best_gain = -np.inf
        best_tresh = None
        best_feature_index = None

        for i in range(self.n_features):
            feature_values = cur_data[:, i]
            tresh, imp, gain = self.make_split_one_feature_(feature_values, cur_labels)

            if gain == -np.inf:
                continue
            
            if gain > best_gain:
                best_gain = gain
                best_tresh = tresh
                best_feature_index = i

        return best_gain, best_feature_index, best_tresh


    def change_to_leaf_(self, node: Node, y):
        node.is_leaf = True

        class_counts = np.zeros(self.n_classes, dtype=int)
        unique, counts = np.unique(y[node.node_indxs], return_counts=True)
        for i, unique_v in enumerate(unique):
            class_counts[unique_v] = counts[i]
        node.class_counts = class_counts

        node_lables = y[node.node_indxs]
        node.predicted_value = self.mode_(node_lables)
        node.leaf_impurity = node.impurity_before

    def create_children_(self, feature_index, tresh, X, y, parent:Node, parent_indxs):
        left_mask = X[parent_indxs][:, feature_index] <= tresh

        left_indxs = parent_indxs[left_mask]
        right_indxs = parent_indxs[~left_mask]
        
        left_node = Node(left_indxs)
        left_node.impurity_before = self.calc_gini_(y[left_indxs])

        right_node = Node(right_indxs)
        right_node.impurity_before = self.calc_gini_(y[right_indxs])

        parent.left = left_node
        parent.right = right_node
        
        
    def calc_gini_(self, y):
        N = len(y)
        total = np.bincount(y, minlength=self.n_classes)
        return 1 - np.sum(total ** 2) / N ** 2
    
    def mode_(self, y):
        return np.bincount(y).argmax()


# if __name__ == "__main__":
#     X = np.random.randint(0, 50, size=(20))
#     y = np.random.randint(0, 2, size=(20))

#     model = DecisionTreeClassifier(max_depth=2)

#     model.fit(X, y)

#     probs = model.predict_proba(X)
#     preds = model.predict(X)
