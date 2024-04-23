# run_iNMF.py
## BMEN4480 Poisson-Driven Integrated Non-Negative Matrix Factorization for Multimodal Analysis of Microbial and Metabolite Abundance Data**

# Alexa Schlotter (aps2211) and Shahd ElNaggar (se2481)
# Import packages necessary for analysis

from iNMF_functions import *
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import itertools as it
import csv
import time
import os
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

# import networkx as nx
# import matplotlib.cm as cm
# import community as community_louvain
# import networkx as nx
# from node2vec import Node2Vec
# from sklearn.cluster import KMeans

########### DATA IMPORT AND PRE-PROCESSING ###########

# The following script runs the workflow on the Yachida dataset, but the other datasets were analyzed the same way. 

base_path = "/Users/shahdelnaggar/Desktop/machine-learning/final-project/"

mtb_df = load_data(os.path.join(base_path, "data/YACHIDA_CRC_2019/mtb.tsv"))
genera_df = load_data(os.path.join(base_path, "data/YACHIDA_CRC_2019/genera.tsv"))
metadata_df = load_data(os.path.join(base_path, "data/YACHIDA_CRC_2019/metadata.tsv"))

# Ensure that the metabolite and microbiome datasets share the same indices (the same samples)
assert mtb_df.index.equals(genera_df.index), "Indices (samples) of mtb_data and genera_data do not match."

# Filter out any taxa or metabolites that are present in less than 10% of samples
threshold_genera = 0.1 * genera_df.shape[0]
genera_filtered = genera_df.loc[:, (genera_df > 0).sum(axis = 0) > threshold_genera]
threshold_mtb = 0.1 * mtb_df.shape[0]
mtb_filtered = mtb_df.loc[:, (mtb_df > 0).sum(axis = 0) > threshold_mtb]

# Combine the two views into a list of numpy arrays
X_microbiome = genera_filtered
X_metabolite = mtb_filtered.div(mtb_filtered.sum(axis=1), axis=0)
X_combined = [X_microbiome, X_metabolite]

if isinstance(X_combined, list) and all(isinstance(x, pd.DataFrame) for x in X_combined):
    X_combined = [x.values for x in X_combined]

if isinstance(X_combined, pd.DataFrame):
    X_combined = [X_combined.values]

for idx, X in enumerate(X_combined):
    print(f"Matrix {idx} shape: {X.shape}")

k = 3
iterations = 10000
reps = 1
lambda1 = 0.1
lambda2 = 0.0001
tol = 1e-2

best_W, best_Hs, best_Vs, best_loss = run_inmf(X_combined, k, iterations, reps, lambda1, lambda2, tol)
output_path = os.path.join(base_path, "output/test_k_3_normalized_1.pk1")
data = {
    'best_W': best_W,
    'best_Hs': best_Hs,
    'best_Vs': best_Vs,
    'best_loss' : best_loss
}
with open(output_path, 'wb') as file:
    pickle.dump(data, file)
print(f"Data successfully saved to {output_path}")

# Code to load the data

# Load data
with open(output_path, 'rb') as file:
    data_loaded = pickle.load(file)

# Put data into variables
best_W = data_loaded['best_W']
best_Hs = data_loaded['best_Hs']
best_Vs = data_loaded['best_Vs']
best_loss = data_loaded['best_loss']
print("Data successfully loaded")

"""Visualization

Let's first visualize the W matrix, which is the shared feature matrix

"""
plot_heatmap(best_W)

plt.figure(figsize=(16,50))
sns.clustermap(best_W, yticklabels = mtb_df.index)
plt.show()

plot_tsne_by_component(best_W)

disease_state_column = 'Study.Group'

plot_tsne(best_W, metadata_df, disease_state_column)

"""Now, we can visualize the H matrices"""

plot_heatmap_H(best_Hs[0].T)
# plot_tsne_by_component(best_Hs[0].T)

# Metabolite
plot_heatmap_H(best_Hs[1].T)
# plot_tsne_by_component(best_Hs[1].T)

def plot_orig_recon(original_data, reconstructed_data, titles=['Original Data', 'Reconstructed Data']):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(50, 50))

    # Plot original data
    sns.heatmap(original_data, ax=axs[0], cmap='viridis')
    axs[0].set_title(titles[0])

    # Plot reconstructed data
    sns.heatmap(reconstructed_data, ax=axs[1], cmap='viridis')
    axs[1].set_title(titles[1])

    plt.show()

# Assuming X and V are your original and reconstructed datasets respectively
plot_orig_recon(X_microbiome, best_Vs[0])

plot_orig_recon(X_metabolite, best_Vs[1])

microbiome_metrics = evaluate_reconstruction_quality(X_microbiome, best_Vs[0])
print("Reconstruction of Microbiome data metrics")
for metric, value in microbiome_metrics.items():
    print(f"{metric}: {value:.4f}")

metabolite_metrics = evaluate_reconstruction_quality(X_metabolite, best_Vs[1])
print("Reconstruction of Metabolite data metrics")
for metric, value in metabolite_metrics.items():
    print(f"{metric}: {value:.4f}")

top_microbes = top_features_by_component(best_Hs[0])
top_metabolites = top_features_by_component(best_Hs[1])


# concatenate microbes and metabolites and plot top loadings
combined_features = genera_filtered.columns.to_list() + mtb_filtered.columns.to_list() 
H_concatenated = np.concatenate([best_Hs[0], best_Hs[1]], axis = 1)
top_all = top_features_by_component(H_concatenated, combined_features, 20, True, "Microbes and Metabolites")


# def build_cross_view_network(top_microbes, top_metabolites, H_microbiome, H_metabolite):
#     G = nx.Graph()
#     # Consider each component from both views
#     for comp_microbe in top_microbes:
#         for comp_metabolite in top_metabolites:
#             # For simplicity, we are considering all combinations, you may want to refine this logic
#             microbes = top_microbes[comp_microbe]
#             metabolites = top_metabolites[comp_metabolite]
#             for microbe in microbes:
#                 for metabolite in metabolites:
#                     # Connect microbes to metabolites possibly using correlation or simple linkage
#                     # Adding an edge based on similarity between component vectors can be one way
#                     correlation = np.corrcoef(H_microbiome[comp_microbe], H_metabolite[comp_metabolite])[0, 1]
#                     if np.abs(correlation) > 0.5:  # Threshold can be adjusted
#                         G.add_edge(f"Microbe_{microbe}", f"Metabolite_{metabolite}", weight=correlation)
#     return G

# cross_view_network = build_cross_view_network(top_microbes, top_metabolites, best_Hs[0].T, best_Hs[1].T)
# nx.draw(cross_view_network, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=8)
# plt.title('Cross-View Network of Microbes and Metabolites')
# plt.show()

# def cluster_and_visualize_network(G, dimensions=64, walk_length=30, num_walks=200, num_clusters=3):
#     # Generate embeddings using Node2Vec
#     node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=4)
#     model = node2vec.fit(window=10, min_count=1, batch_words=4)

#     # Apply K-means clustering to the embeddings
#     kmeans = KMeans(n_clusters=num_clusters)
#     kmeans.fit(model.wv.vectors)
#     labels = kmeans.labels_

#     # Map nodes to their clusters
#     cluster_map = {i: [] for i in range(num_clusters)}
#     for i, node in enumerate(G.nodes()):
#         G.nodes[node]['cluster'] = labels[i]
#         cluster_map[labels[i]].append(node)

#     # Draw the network with nodes colored by their cluster assignment
#     pos = nx.spring_layout(G)  # Node positions
#     cmap = cm.get_cmap('viridis', max(labels) + 1)  # Colormap for different clusters
#     ec = nx.draw_networkx_edges(G, pos, alpha=0.2)
#     nc = nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color=labels,node_size=100, cmap=cmap, alpha=0.9)
#     plt.colorbar(nc)
#     plt.axis('off')
#     plt.show()

#     # Print the nodes in each cluster
#     for cluster_id, nodes in cluster_map.items():
#         print(f"Cluster {cluster_id} contains {len(nodes)} nodes:")
#         for node in nodes:
#             print(f"  Node {node} ")

#     return cluster_map


# G = cross_view_network
# cluster_assignments = cluster_and_visualize_network(G, dimensions=64, walk_length=30, num_walks=200, num_clusters=4)


# r = 0.6
# k = 3
# iterations = 10000
# reps = 1
# lambda1 = 0.01
# lambda2 = 0.0001
# tol = 1e-2

# best_W_gp, best_Hs_gp, best_Vs_gp, best_loss_gp = run_inmf_gp(X_combined, k, iterations, reps, lambda1, lambda2, tol)