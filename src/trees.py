import numpy as np
from scipy import stats
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
            self.samples_count = None

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
        self.root.impurity_before = self.calc_gini(y)

        self.grow_tree(X, y, self.root, 0)


    def grow_tree(self, X, y, root: Node, depth):
        self.find_best_split(root, X, y)

        if root.is_leaf:
            return

        if depth >= self.max_depth - 1 or math.isclose(root.impurity_before, 0.0):
            self.create_leafs(root.feature_index, 
                              root.split_threshold, 
                              X, y, 
                              root, root.node_indxs, 
                              )
            return

        self.grow_tree(X, y, root.left, depth + 1)
        self.grow_tree(X, y, root.right, depth + 1)
        


    def make_slit_one_feature(self, X, y):
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


    def find_best_split(self, node: Node, X, y):
        
        cur_indxs = node.node_indxs
        cur_data = X[cur_indxs]
        cur_labels = y[cur_indxs]

        best_gain = -np.inf
        best_tresh = None
        best_imp = None
        best_feature_index = None

        for i in range(self.n_features):
            feature_values = cur_data[:, i]
            tresh, imp, gain = self.make_slit_one_feature(feature_values, cur_labels)

            if gain == -np.inf:
                continue
            
            if gain > best_gain:
                best_gain = gain
                best_imp = imp
                best_tresh = tresh
                best_feature_index = i

        if best_gain > 0:
            node.feature_index = best_feature_index
            node.split_threshold = best_tresh
            self.create_children(best_feature_index, best_tresh, X, y, node, cur_indxs)
        else:
            self.change_to_leaf(node, y)


    def change_to_leaf(self, node: Node, y):
        node.is_leaf = True
        node.samples_count = len(node.node_indxs)

        node_lables = y[node.node_indxs]
        node.predicted_value = stats.mode(node_lables, keepdims=False)[0]
        node.leaf_impurity = node.impurity_before

    def create_leafs(self, feature_index, tresh, X, y, parent:Node, parent_indxs):
        left_mask = X[parent_indxs][:, feature_index] <= tresh

        left_indxs = parent_indxs[left_mask]
        right_indxs = parent_indxs[~left_mask]

        y_left = y[left_indxs]
        y_right = y[right_indxs]

        left_leaf = Node(left_indxs, leaf=True)
        left_leaf.leaf_impurity = self.calc_gini(y_left)
        left_leaf.samples_count = len(y_left)
        left_leaf.predicted_value = stats.mode(y_left, keepdims=False)[0]
        
        right_leaf = Node(right_indxs, leaf=True)
        right_leaf.leaf_impurity = self.calc_gini(y_right)
        right_leaf.samples_count = len(y_right)
        right_leaf.predicted_value = stats.mode(y_right, keepdims=False)[0]

        parent.left = left_leaf
        parent.right = right_leaf


    def create_children(self, feature_index, tresh, X, y, parent:Node, parent_indxs):
        left_mask = X[parent_indxs][:, feature_index] <= tresh

        left_indxs = parent_indxs[left_mask]
        right_indxs = parent_indxs[~left_mask]
        
        left_node = Node(left_indxs)
        left_node.impurity_before = self.calc_gini(y[left_indxs])

        right_node = Node(right_indxs)
        right_node.impurity_before = self.calc_gini(y[right_indxs])

        parent.left = left_node
        parent.right = right_node
        
        
    def calc_gini(self, y):
        N = len(y)
        total = np.bincount(y, minlength=self.n_classes)
        return 1 - np.sum(total ** 2) / N ** 2

X = np.random.randint(0, 50, size=(20, 4))
y = np.random.randint(0, 2, size=(20))

model = DecisionTreeClassifier(max_depth=2)

model.fit(X, y)

print('Hi')