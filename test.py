from random import sample
def find_dist(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def k_means(k, points):
    centroids = sample(points, k)
    data = [[] for _ in range(k)]
    data_calc = [[0, 0, 0] for _ in range(k)]
    for point in points:
        distances = []
        for centroid in centroids:
            distances.append(find_dist(point, centroid))
        data[distances.index(min(distances))].append(point)
        data_calc[distances.index(min(distances))][0] += point[0]
        data_calc[distances.index(min(distances))][1] += point[1]
        data_calc[distances.index(min(distances))][2] += 1
    print(data, data_calc)

k_means(2, [[1, 1], [1, 2], [0, 0], [1, 0], [-1, 0], [0, 1], [0, 1], [11,0], [12, 1], [11, 1], [11, -1], [10, -1], [12, -1], [10, 0]])