from sklearn.neighbors import NearestNeighbors

def get_nearest_neighbours(feature_vectors, nearest_neighbours_count):
    nn = NearestNeighbors(n_neighbors=nearest_neighbours_count, algorithm='ball_tree').fit(feature_vectors)
    _, nearest_neighbours = nn.kneighbors(feature_vectors)
    return nearest_neighbours
