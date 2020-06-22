from scipy.spatial.distance import cdist
import numpy as np

class DBScan():
    """This is a simple implementation of clustering using the 
    DBScan algorithm. 
    My algorithm will return labels for clusters, -1 indicating
    an outlier"""
    def __init__(self, data, max_dist, min_pts):
        self.data = data
        self.max_dist = max_dist
        self.min_pts = min_pts
        self.labels = [0] * len(data)
    """The following list will hold the final label assignments
    I'll initialize it with 0's, but cluster assignment will start
    at 1"""
    

    """Next for each point P in data, I'll run a function to 
    determine if a point is a valid seed, then fill out the cluster
    with reachable points"""
    def fit(self):
        cluster = 0
        for P in range(0,len(self.data)):
            reachable = self.find_reachable(P)
            """If a point isn't a valid seed, it's an outlier, this is the only
            condition when a label is set to outlier it may still be claimed
            as a boundary point for a cluster later"""
            if len(reachable)<self.min_pts:
                self.labels[P] = -1
            elif self.labels[P]==0:
                cluster+=1
                self.create_cluster(P, cluster, reachable)
        return self.labels
    
    def predict(self, P):
        """Given a new point of data, P assign it a cluster label"""
        for i in range(0, len, self.data):
            if cdist(np.reshape(self.data[P],(-1,2)), np.reshape(self.data[i],(-1,2))) < self.max_dist:
                return self.labels[i]

    def create_cluster(self, P, cluster, reachable):
        """Given a valid seed point, create the cluster with 
        every point that belongs according to distance threshold"""
        self.labels[P] = cluster
        """Run a while loop, to step through each point in our seed's
        list of reachable, checking for each of their neighbors, adding them
        to this cluster, if they aren't already in another cluster"""
        i=0
        while i < len(reachable):
            next_point = reachable[i]
            #If the label was previously noise, it's not a valid branch
            #So we'll just add it to the cluster and move on
            if self.labels[next_point] == -1:
                self.labels[next_point] = cluster
            #If the point was unclaimed, let's claim it, and grow from there
            elif self.labels[next_point] == 0:
                self.labels[next_point] = cluster
                
                next_point_reachable = self.find_reachable(next_point)
                #If this point is a valid branch, let's get it's neighbors in here too
                if len(next_point_reachable)>self.min_pts:
                    reachable = reachable + next_point_reachable
            i+=1



    
    def find_reachable(self, P):
        """The following function will take a point in data
        and find reachable points from it"""
        reachable = []
        for i in range(0, len(self.data)):
            if cdist(np.reshape(self.data[P],(-1,2)), np.reshape(self.data[i],(-1,2)))<self.max_dist:
                """If the distance between the point and a given point in data
                let's add it to a list of reachable points"""
                reachable.append(i)
        return reachable