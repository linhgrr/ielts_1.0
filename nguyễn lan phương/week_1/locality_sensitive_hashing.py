import numpy as np

def get_side_of_plane(planes_list, point):
    a = np.dot(planes_list, point.T)
    return np.sign(a)

def locality_sensitive_hashing(data):
    value = 0
    for i in range(len(data)):
        value += 2**i if data[i] > 0 else 0
    return value


num_dimensions = 2
num_planes = 3

random_planes = np.random.normal(size = (num_planes, num_dimensions))

print(random_planes)

v = np.array([1, 2])

print(locality_sensitive_hashing(get_side_of_plane(random_planes, v)))