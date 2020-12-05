import numpy as np
from scipy.spatial import distance as dist


centroids_current=np.array([ [0,1], [1,2], [3,5], [6,8 ]])
centroids_after=np.array( [[1,3],  [1,5]])

#C1R1= C2R2 
#C2R1= C1R2
#C3R1= C3R2

D=dist.cdist(centroids_current,centroids_after  )
print(D)

rows= D.min(axis=1).argsort()
print(rows)
cols= D.argmin(axis=1)[rows]
print(cols)
