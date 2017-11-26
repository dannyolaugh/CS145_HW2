from sklearn import cross_validation
import numpy as np
import math
import matplotlib.pyplot as plt

def load_dataset():
    x = np.zeros((150,4))
    y = np.zeros(150, dtype=int)
    instance_index = 0
    with open("iris.data", 'r') as f:
        for line in f:
            data = line.strip().split(',')
            x[instance_index] = data[0:4]
            if data[4] == 'Iris-setosa':
                y[instance_index] = 0
            elif data[4] == 'Iris-versicolor':
                y[instance_index] = 1
            else:
                y[instance_index] = 2
            instance_index += 1
    
    perm = np.random.permutation(np.arange(x.shape[0]))
    x = x[perm]
    y = y[perm]
    return x, y
    
# The function calculates euclidean distance between two vectors (data points)
# x1 and x2 are two data points
# Returns the euclidean distance value
def euclidean_distance(x1, x2):
    distance = 0.0
    ########## Please Fill Missing Lines Here ##########
    for i in range(0, len(x1)):
        distance += ((x2[i] - x1[i]) ** 2)

    distance = math.sqrt(distance)

    return distance

# The function finds the class labels of K nearest neighbours for the given test data point 'j'
# Uses function euclidean_distance
# train_x and train_y are the training data points and their class labels
# test_x_j is the 'jth' test point whose neighbours need to be found
# Returns a 1-d numpy array with class labels of nearest neighbours
def find_K_neighbours(train_x, train_y, test_x_j, K):
    neighbours_y = np.full(K,-1, dtype=int)
    ########## Please Fill Missing Lines Here ##########
    #my_array = np.array([10, 6, 3, 9, 100])
    #neighbours_y = train_x.argsort()[:3]
    #print neighbours_y
    distance_array = np.full(len(train_x), -1, dtype = float)

    my_range = len(train_x)

    for i in range(0, my_range):
        distance_array[i] = euclidean_distance(train_x[i], test_x_j)  #now you'll have all the distances

    #my_array = np.array([10, 6, 3, 9, 100])
    #neighbours_y = my_array.argsort()[:3]
    my_array = distance_array.argsort()[:K] #gets the indices of the K smallest values in distance_array

    for j in range(0, len(my_array)):
        neighbours_y[j] = train_y[my_array[j]]

    print neighbours_y


    return neighbours_y

# The function classifies a data point given the labels of its nearest neighbours
# Returns the label for the data point
def classify(neighbours_y):
    label = -1
    ########## Please Fill Missing Lines Here ##########
    count_0 = 0
    count_1 = 0
    count_2 = 0

    for i in range(0, len(neighbours_y)):
        if neighbours_y[i] == 0:
            count_0 += 1
        elif neighbours_y[i] == 1:
            count_1 += 1
        elif neighbours_y[i] == 2:
            count_2 += 1

    count_array = [count_0, count_1, count_2]

    max_count = max(count_array)

    if count_0 == max_count:
        label = 0
    elif count_1 == max_count:
        label = 1
    elif count_2 == max_count:
        label = 2

    return label
    
if __name__ == '__main__':
    x, y = load_dataset()
    cv = cross_validation.KFold(len(x), n_folds = 5)
    
    average_accuracies = np.zeros(119)
    
    for K in range(1, 120):
        
        fold_accuracies = []
        
        for traincv, testcv in cv:
            train_x = x[traincv]
            train_y = y[traincv]
            test_x = x[testcv]
            test_y = y[testcv]

            predicted_labels = np.full(test_x.shape[0], -1, dtype=int)
            
            for j in range(test_x.shape[0]):
                neighbours_y = find_K_neighbours(train_x, train_y, test_x[j], K)
                predicted_labels[j] = classify(neighbours_y)
            
            fold_accuracies.append(np.mean(predicted_labels == test_y))
            
        average_accuracies[K-1] = np.mean(fold_accuracies)
        
    print("Average accuracies with 5-fold cross validation for K varying from 1 to 119:")
    print(average_accuracies)
    
    print("Best value of K: ")
    print(np.argmax(average_accuracies)+1)


    ########################################


    ########## Please Fill Missing Lines Here ##########
    # Plot K values vs. average accuracies
    plt.plot(range(0, len(average_accuracies)), average_accuracies)
    plt.show()
    
    