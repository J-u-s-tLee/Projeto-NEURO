import umap
import matplotlib.pyplot as plt

def dimensionality_reduce(data, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state
    )

    embedding = reducer.fit_transform(data)
    return embedding

def plot_embedding(embedding, labels=None, figsize=(8, 6), point_size=10, cmap='plasma', save_path=None):

    plt.figure(figsize=figsize)

    if labels is not None:
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=cmap, s=point_size)
        plt.colorbar(scatter, label='Label')
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], color='gray', s=point_size)
    
    plt.title("UMAP Projection")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close()

