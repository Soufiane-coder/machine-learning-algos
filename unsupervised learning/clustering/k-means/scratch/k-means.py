import numpy as np
import random as rd
import matplotlib.pyplot as plt

################################################################
from utils import draw_points, draw_points_assigned

################################################################

# Question 1


def generer_random_data(n):
    donnees = []
    for _ in range(n):
        objet = []
        for _ in range(2):
            objet.append(rd.random())
        donnees.append(objet)
    return donnees


dataset = generer_random_data(20)

# print(dataset)
# draw_points(dataset)

# Question 2


def distance2(d1, d2):
    d = 0
    for i in range(len(d1)):
        d += (d2[i]-d1[i])**2
    resu = np.sqrt(d)
    return resu


def distance_all(donnees, z):
    distances = []
    for donnee in donnees:
        distances.append([distance2(donnee, z)])
    return distances


# print(distance_all([[1, 3, 1], [3, 2.5, 1], [5, 6, 0]], [3, 3, 1]))

# Question3

def moyenne_arith(dataset: list):
    n = len(dataset)  # nombre de lignes
    m = len(dataset[0])  # nombre de colonnes
    moyenne_v = [0] * m
    for i in range(m):
        total = 0
        for j in range(n):
            total += dataset[j][i]
        moyenne_v[i] = total / n
    return moyenne_v


# matrice = np.array([[1, 2], [3, 2], [2, 8]])
# moyenne = moyenne_arith(matrice)

# resultat = np.mean(matrice, axis=0)
# print(resultat)

# Question 4


def indice_plus_petit(list_value: list):
    Imin = 0
    for i in range(len(list_value)):
        if list_value[i] < list_value[Imin]:
            Imin = i
    return Imin


# L = np.array([3, 1, 0, 2, 6, 10])
# Imin = indice_plus_petit(L)
# print(np.argmin(L))
# print(Imin)


def random_centroid(dataset: list, k: int):
    centroides = []
    n = len(dataset)
    for i in range(k):
        position_alea = rd.randint(0, n-1)
        centroide = dataset[position_alea]
        centroides.append(centroide)
    return centroides


centroids = random_centroid(dataset, 3)
# draw_points(dataset, centroids)


def assign_cluster(dataset: list, centroides: list):
    assignments = []
    for data_point in dataset:
        dist_point_clust = []
        for centroid in centroides:
            d_clust = distance2(data_point, centroid)
            dist_point_clust.append(d_clust)
        assignment = indice_plus_petit(dist_point_clust)
        assignments.append(assignment)
    return assignments


# draw_points_assigned(dataset, assign_cluster(dataset, centroids), centroids)


def new_centroids(dataset: list, centroids: list, assignments: list, k: int):
    new_centroids = []
    for i in range(k):
        pt_cluster = []
        for j in range(len(dataset)):
            if assignments[j] == i:
                pt_cluster.append(dataset[j])
        mean_c = moyenne_arith(pt_cluster)
        new_centroids.append(mean_c)
    return new_centroids


# centroids = new_centroids(
#     dataset, centroids, assign_cluster(dataset, centroids), 3)
# draw_points_assigned(dataset, assign_cluster(dataset, centroids), centroids)


def kmeans(dataset, k):
    n = len(dataset)
    centroides = random_centroid(dataset, k)
    assign = [0] * n
    while True:
        old_assign = assign.copy()
        assign = assign_cluster(dataset, centroides)
        centroides = new_centroids(dataset, centroides, assign, k)
        if assign == old_assign:
            break
    return assign, centroides


# assign, centroids = kmeans(dataset, 3)

# draw_points_assigned(dataset, assign, centroids)
# print(kmeans(dataset, 3))


def w(dataset, kmax):
    x_axis = []
    y_axis = []
    for k in range(1, kmax+1):
        assignments, clusters = kmeans(dataset, k)
        w = 0
        for i in range(k):
            for j in range(len(dataset)):
                if assignments[j] == i:
                    w += np.linalg.norm(np.array(dataset[j]) -
                                        np.array(assignments[i])) ** 2
        x_axis.append(k)
        y_axis.append(w)
    plt.figure()
    plt.plot(x_axis, y_axis)
    plt.xlabel('Nombre de clusters (k)')
    plt.ylabel('Somme des carrés intra-cluster (W)')
    plt.title('Elbow Method')
    plt.show()


w(dataset, 4)
