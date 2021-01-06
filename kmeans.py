import math
import random
import copy
from time import time


# calculates euclidean distance between two points
def euclidean_distance(p1, p2):
    x_axis = abs(p1[0] - p2[0]) ** 2
    y_axis = abs(p1[1] - p2[1]) ** 2
    return math.sqrt(x_axis + y_axis)


# calculates the index of the closest cluster
def closest_cluster(distance_array):
    min_value = float("inf")
    min_index = -1
    for i, value in enumerate(distance_array):
        if value < min_value:
            min_index = i
            min_value = value
    # randomly choosing if value is the same
    same_value_index = []
    for i, value in enumerate(distance_array):
        if min_value is value:
            same_value_index.append(i)
    min_index = random.choice(same_value_index)
    return min_index


class KMeans:
    def __init__(self, points, number_of_clusters, tolerance=0.001):
        # array of points with index
        self.points = points
        # total number of clusters as an integer
        self.number_of_clusters = number_of_clusters
        # array of which points are assigned to which cluster
        self.assigned_cluster = [-1] * len(self.points)
        # array of current centroid coordinates of index
        self.current_centroid = [[-1, -1]] * self.number_of_clusters
        # array of past centroid coordinates for calculating tolerance
        self.previous_centroid = [[-1, -1]] * self.number_of_clusters
        self.tolerance = tolerance

    def initialize_centroid(self):
        for i in range(self.number_of_clusters):
            while True:
                # max value is 5.x, so 6 is multiplied to space out the values
                x = random.random() * 6
                y = random.random() * 6
                # make sure there are no duplicate centroids
                if [x, y] not in self.current_centroid:
                    self.current_centroid[i] = [x, y]
                    break

    def assign_points_to_cluster(self):
        for i, point in enumerate(self.points):
            distance_array = []
            for j in range(self.number_of_clusters):
                distance = euclidean_distance(point, self.current_centroid[j])
                # index is the cluster index, value is the distance from the point to cluster
                distance_array.append(distance)
            # assign point to closest cluster
            self.assigned_cluster[i] = closest_cluster(distance_array)

    def update_centroid(self):
        # initialize sum for x and y values and num of elements
        x_value_sum = [0] * self.number_of_clusters
        y_value_sum = [0] * self.number_of_clusters
        num_of_elements = [0] * self.number_of_clusters
        # for each point, get the sum of x values and y values
        for i, point in enumerate(self.points):
            centroid_num = self.assigned_cluster[i]
            num_of_elements[centroid_num] += 1
            x_value_sum[centroid_num] += point[0]
            y_value_sum[centroid_num] += point[1]
        # for each cluster, set the previous centroid as the current centroid
        for i in range(self.number_of_clusters):
            self.previous_centroid[i] = copy.deepcopy(self.current_centroid[i])
        # get the value of the current centroid
        for i, num in enumerate(num_of_elements):
            if num != 0:
                for j in range(self.number_of_clusters):
                    self.current_centroid[i][0] = x_value_sum[i] / num_of_elements[i]
                    self.current_centroid[i][1] = y_value_sum[i] / num_of_elements[i]
            else:
                self.current_centroid[i][0] = x_value_sum[i]
                self.current_centroid[i][1] = y_value_sum[i]

    def calculate_tolerance(self):
        total = 0
        # calculate tolerance for each centroid
        for i in range(self.number_of_clusters):
            if self.previous_centroid[i][0] != 0 and self.previous_centroid[i][1] != 0:
                x_axis = abs(self.current_centroid[i][0] - self.previous_centroid[i][0]) \
                         / abs(self.previous_centroid[i][0])
                y_axis = abs(self.current_centroid[i][1] - self.previous_centroid[i][1]) \
                         / abs(self.previous_centroid[i][1])
                total += (x_axis + y_axis)
        return total


def main():
    # open the data file
    file = open("./data.txt", "r")
    output = open("./result_test.txt", "w")
    points = []
    for line in file:
        line = [float(x) for x in line.strip().split(",")]
        line[0] = int(line[0])
        points.insert(line[0], [line[1], line[2]])

    start_time = time()
    k_means = KMeans(points, 5)
    k_means.initialize_centroid()
    init_tolerance = k_means.calculate_tolerance()
    while init_tolerance > k_means.tolerance:
        k_means.assign_points_to_cluster()
        k_means.update_centroid()
        init_tolerance = k_means.calculate_tolerance()
    end_time = time()

    print("Detected centroid:")
    for i in range(5):
        x_point = round(k_means.current_centroid[i][0], 4)
        y_point = round(k_means.current_centroid[i][1], 4)
        print("Centroid {}: {}, {}".format(i, x_point, y_point))
    print("Total Elapsed time: {} sec".format(round(end_time - start_time, 3)))

    for i, cluster in enumerate(k_means.assigned_cluster):
        output.write("{}, {}\n".format(i, cluster))


if __name__ == "__main__":
    main()
