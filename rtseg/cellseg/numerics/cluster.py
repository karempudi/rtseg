
import torch


from sklearn.neighbors import NearestNeighbors # type: ignore


def cluster(points, semantic, fast = True, eps = 2.2, min_samples = 15, snap_noise = True):
    """
    Returns the instance segmentation given the the semantic segmentation and 
    the integrated euclidean semantic points. 

    Args:
        points (torch.Tensor): Of shape (D, N).
        semantic (torch.Tensor): The semantic segmentation of shape (1, H, W).
        eps: EPS value to use for DBSCAN. This is the neighborhood 'radius' from 
            a point we want to consider. 
        min_samples: ...

    Returns:
        torch.Tensor: The instance segmentation of shape (1, H, W).

    """
    if fast:
        from dbscan import sklDBSCAN as DBSCAN # type: ignore
    else:
        from sklearn.cluster import DBSCAN # type: ignore
    

    points = points.T

    if points.shape[0] == 0:
        return semantic

    if fast:
        clustering = DBSCAN(eps = eps, min_samples = min_samples,).fit(points)
    else:
        clustering = DBSCAN(eps = eps, min_samples = min_samples, n_jobs=-1).fit(points)

    clusters = torch.Tensor(clustering.labels_)

    if snap_noise:
        outliers_idx = clusters == -1
        if torch.any(outliers_idx):
            non_outliers_idx = ~outliers_idx

            # If there are ONLY outliers, then nothing was 
            # clustered and return the semantic segmentation. 
            if not torch.any(non_outliers_idx):
                return semantic

            outliers     = points[outliers_idx]
            non_outliers = points[non_outliers_idx]

            NN = NearestNeighbors(metric = "euclidean")#, radius = eps)
            NN.fit(non_outliers)

            _, nn_idx = NN.kneighbors(outliers)
            nn_idx = torch.Tensor(nn_idx).long()

            nearest_n = clusters[non_outliers_idx][nn_idx]
            values = torch.mode(nearest_n, dim = 1)[0]

            # Give the outliers the instance value of their nearest 
            # clustered neighbor.
            clusters[outliers_idx] = values

    # It sets first cluster to 0, be careful as this would
    # blend it into the background.
    instance_segmentation = semantic.clone().float()
    instance_segmentation[semantic] = clusters + 1

    return instance_segmentation
