import numpy as np
def cal_distance(point2):
    def cal_instance(point1):
        return np.sqrt((point1[0]-point2[0]) ** 2 + (point1[1]-point2[1]) ** 2 + (point1[2]-point2[2]) ** 2)
    return cal_instance

def distance_based_sort(pointcloud, original_point=(0,0,0)):
    return np.asarray(sorted(pointcloud, key=cal_distance(original_point)))
    
# Example: pointcloud = distance_based_sort(pointcloud)[:n_sample]
