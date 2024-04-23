# iNMF_functions.py
# last updated april 23rd, 2024 
# BMEN4480 Poisson-Driven Integrated Non-Negative Matrix Factorization for Multimodal Analysis of Microbial and Metabolite Abundance Data**
# written by Alexa Schlotter and Shahd ElNaggar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import itertools as it
import csv
import time
import os
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm

# load in data (as formatted by the curated datasets)
def load_data(file_path):
  return pd.read_csv(file_path, sep = '\t', index_col = 0)

"""# iNMF Poisson + Multiplicative Optimization"""

# Purpose of this function is to calculate the Poisson loss used to measure the difference betweent the observed data X and the model's predictions W.
def poisson_loss(X, WH):
    WH = np.clip(WH, 1e-10, None) # Prevents taking the logarithm of zero by setting a lower bound
    return np.sum(WH - X * np.log(WH)) # Calculates the Poisson loss between the actual data V and the estimated data WH

# Purpose of this function is to update the matrix W based on the current values of W, H, and the dataset X
# Incorporates L2 regularization
def update_W(Xs, W, Hs, lambda2):
    W_new = np.zeros_like(W)  # Initializes a new W matrix with the same shape as W filled with zeros
    for X, H in zip(Xs, Hs): # Iterates over each pair of V and H matrices
        WH = W @ H # Calculates the product of W and H
        numerator = X / np.clip(WH, 1e-10, None) @ H.T
        denominator = np.ones_like(X) @ H.T + lambda2 * W
        W_new += W * numerator / denominator
    W_new /= len(Xs)  # Average the updates from all views
    return W_new

# Updated H based on the values of W, H, and the sample data X
# Incorporates L1 and L2 regularization
def update_H(X, W, H, lambda1, lambda2):
    WH = W @ H # Calculates the product of W and H
    numerator = W.T @ (X / np.clip(WH, 1e-10, None))
    denominator = W.T @ np.ones_like(X) + lambda2 * H + lambda1
    return H * numerator / denominator

# Runs iNMF with multiplicative updates according to a Poisson distribution
def run_inmf(Xs, k, iterations, reps, lambda1, lambda2, tol=1e-5):
    best_loss = float('inf') # Initializes the best loss to infinity
    best_W = None
    best_Hs = None
    best_Vs = None

    m = max(X.shape[0] for X in Xs)  # Determines the maximum number of samples across all datasets
    n = Xs[0].shape[1]  # Assumes a consistent feature count across views
    W = np.random.rand(m, k)  # Initializes W randomnly

    total_updates = reps * iterations
    progress = tqdm(total=total_updates, desc='Overall Progress', leave=False)

    for rep in range(reps): # Loop over the number of repetitions
        Hs = [np.random.rand(k, X.shape[1]) for X in Xs]
        Vs = [np.zeros_like(X) for X in Xs]
        loss_curve = []
        last_loss = float('inf')

        for i in range(iterations): # Loop over the number of iterations
            W = update_W(Xs, W, Hs, lambda2)
            reconstructions = []
            for idx, X in enumerate(Xs):
                Hs[idx] = update_H(X, W, Hs[idx], lambda1, lambda2)
                reconstructions.append(W @ Hs[idx])

            current_loss = sum(poisson_loss(X, W @ H) for X, H in zip(Xs, Hs))
            loss_curve.append(current_loss)
            if abs(last_loss - current_loss) < tol:
                print(f"\nConvergence reached in Rep {rep + 1} at Iteration {i + 1} with Loss: {current_loss}")
                break # Stops if the improvement in loss is below the tolerance
            last_loss = current_loss

            # Update progress bar
            progress.update(1)

        if current_loss < best_loss:
            best_loss = current_loss
            best_W = W.copy()
            best_Hs = [H.copy() for H in Hs]
            best_Vs = [V.copy() for V in Vs]
            best_Vs = reconstructions

        plt.figure(figsize=(10, 6))
        plt.plot(loss_curve, label='Loss per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'Loss Curve for Repetition {rep + 1}')
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"\nRep: {rep}, Best loss: {best_loss}, loss: {current_loss}")

    progress.close()
    return best_W, best_Hs, best_Vs, best_loss

"""# Gamma Poisson + Multiplicative Update"""

# Computes the Gamma-Poisson Loss (negative log likelihood) for overdispersed count data.
# r is the dispersion parameter (shape parameter of the Gamma distribution)
def gamma_poisson_loss(X, WH, r):
    return np.sum(r * np.log(WH) - (r + X) * np.log(r + WH))
  
# Purpose of this function is to update the matrix W based on the current values of W, H, and the dataset X
# Incorporates L2 regularization
def update_W_gp(Xs, W, Hs, lambda2, r):
    W_new = np.zeros_like(W)
    for X, H in zip(Xs, Hs):
        WH = W @ H
        numerator = (X + r) / (WH + r) @ H.T
        denominator = np.ones_like(X) @ H.T + lambda2 * W
        W_new += W * numerator / denominator
    W_new /= len(Xs)  # Average the updates from all views
    return W_new

# Updated H based on the values of W, H, and the sample data X
# Incorporates L1 and L2 regularization
def update_H_gp(X, W, H, lambda1, lambda2, r):
    WH = W @ H
    numerator = W.T @ ((X + r) / (WH + r))
    denominator = W.T @ np.ones_like(X) + lambda2 * H + lambda1
    return H * numerator / denominator

# Runs iNMF with multiplicative updates according to a Poisson distribution
def run_inmf_gp(Xs, k, iterations, reps, lambda1, lambda2, r, tol=1e-5):
    best_loss = float('inf')
    best_W = None
    best_Hs = None
    best_loss_curve = []
    best_Vs = None

    m = max(X.shape[0] for X in Xs)  # Largest number of samples across views
    n = Xs[0].shape[1]  # consistent feature count across views
    W = np.random.rand(m, k)  # Initialize W to have the same number of features as the largest V

    total_updates = reps * iterations
    progress = tqdm(total=total_updates, desc='Overall Progress', leave=False)

    for rep in range(reps):
        Hs = [np.random.rand(k, X.shape[1]) for X in Xs]
        loss_curve = []
        last_loss = float('inf')

        for i in range(iterations):
            W = update_W_gp(Xs, W, Hs, lambda2, r)
            reconstructions = []
            for idx, X in enumerate(Xs):
                Hs[idx] = update_H_gp(X, W, Hs[idx], lambda1, lambda2, r)
                reconstructions.append(W @ Hs[idx])

            current_loss = sum(gamma_poisson_loss(X, W @ H, r) for X, H in zip(Xs, Hs))
            loss_curve.append(current_loss)
            if abs(last_loss - current_loss) < tol:
                print(f"\nConvergence reached in Rep {rep + 1} at Iteration {i + 1} with Loss: {current_loss}")
                break
            last_loss = current_loss

            # Update progress bar
            progress.update(1)

        if current_loss < best_loss:
            best_loss = current_loss
            best_W = W.copy()
            best_Hs = [H.copy() for H in Hs]
            best_loss_curve = loss_curve  # Capture the best loss curve
            best_Vs = reconstructions

    progress.close()

    # Plot the best loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(best_loss_curve, label="Loss per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Best Repetition Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_W, best_Hs, best_Vs, best_loss


# original iNMF function implementation
# https://github.com/yangzi4/iNMF/blob/master/iNMF%20base
def NMF_obj_eval_yang(Xs, W, Hs, L, Sp, Vs=0):
    "Evaluates NMF objective function (sparsity, Vs optional)."
    if Vs == 0: Vs = [zeros((1, shape(Hs[i])[0])) for i in range(len(Xs))]

    obj = np.sum([np.linalg.norm(Xs[i] - (W+Vs[i]).dot(Hs[i]))**2
        for i in range(len(Xs))])
    if type(L) != list: L = [L]*len(Xs)
    pen = np.sum([L[i]*np.linalg.norm(Vs[i].dot(Hs[i]))**2 for i in range(len(Xs))])
    spars = 0
    if Sp != 0:
      spars = Sp*np.sum([np.sum(abs(Hs[i])) for i in range(len(Xs))])
    return obj+pen + spars

def iNMF_run_yang(Xs, D, L, Sp=0, nrep=200, steps=100):
    "Integrated NMF (sparsity optional)."
    K = len(Xs) # number of matrices
    N = shape(Xs[0])[0] #number of samples 
    Ms = [shape(Xs[i])[1] for i in range(K)] # number of taxa and metabolites 
    W_f = zeros((N, D)) # initialize factorized matrix
    Hs_f = [zeros((D, Ms[i])) for i in range(K)] # initialize factorized matrix
    Vs_f = [zeros((N, D)) for i in range(K)] # initialize factorized matrix

    obj_vals = []
    n_iter = []
    for j in range(nrep):
        loss_curve = []
        W = random.uniform(0, 1, (N, D)) # changed from between 0 and 2 to 0 and 1
        Hs = [random.uniform(0, 1, (D, Ms[i])) for i in range(K)]
        Vs = [random.uniform(0, 1, (N, D)) for i in range(K)]
        start_eval = 0
        old_eval = inf
        new_eval = 0
        count = 0
        while (abs(old_eval - new_eval) > (start_eval - new_eval)*1e-2
            ) and (count <= 2e3):  ## 1.5e3
            num = sum([Xs[i].dot(Hs[i].T) for i in range(K)], 0)
            den = sum([(W+Vs[i]).dot(Hs[i]).dot(Hs[i].T) for i in range(K)], 0)

            W *= num/den
            for i in range(K):
                WV = W+Vs[i]
                den = (WV.T.dot(WV) + L*Vs[i].T.dot(Vs[i])).dot(Hs[i]) + Sp
                Hs[i] *= (WV.T.dot(Xs[i]))/den
            for i in range(K):
                den = (W+(1+L)*Vs[i]).dot(Hs[i]).dot(Hs[i].T)
                Vs[i] *= Xs[i].dot(Hs[i].T)/den
            if count == 0:
                new_eval = NMF_obj_eval_yang(Xs, W, Hs, L, Sp, Vs)
                start_eval = new_eval
            if count != 0 and count % steps == 0:  ## count > thres, != 0
                old_eval = new_eval
                new_eval = NMF_obj_eval_yang(Xs, W, Hs, L, Sp, Vs)
            count += 1
        loss_curve.append(new_eval)
        obj_vals.append(new_eval)
        n_iter.append(count)
        #print j, new_eval, count
        if obj_vals[-1] == min(obj_vals):
            W_f = W
            Hs_f = Hs
            Vs_f = Vs
    #print "done"
        print(f"Replication {j+1}/{nrep} completed, Objective: {new_eval}")
    return W_f, Hs_f, Vs_f, obj_vals, n_iter, loss_curve

# VISUALIZATION AND ANALYSIS

def plot_heatmap(best_W, title="Shared component Matrix W"):
    plt.figure(figsize=(10, 50))
    ax = sns.heatmap(best_W, cmap="viridis", linewidths=0.2)
    plt.title(title)
    plt.xlabel("Components")
    plt.ylabel("Features")
    plt.show()

def plot_heatmap_H(H, title):
    plt.figure(figsize=(10, 50))
    ax = sns.heatmap(H, cmap="viridis")
    plt.title(title)
    plt.xlabel("Components")
    plt.ylabel("Features")
    plt.show()

def plot_grouped_stacked_bar(W, metadata, disease_state_column, title="Sample Contributions by Components"):
    # Transpose W for proper orientation
    df_W = pd.DataFrame(W, columns=[f'Component {i+1}' for i in range(W.T.shape[0])])

    # Add disease state from metadata to the W DataFrame
    # Make sure the order of the disease state aligns with the rows in W
    df_W['Disease State'] = metadata[disease_state_column].values

    # Group by disease and sum the contributions
    grouped_W = df_W.groupby('Disease State').sum().T

    # Plotting
    ax = grouped_W.plot(kind='bar', stacked=True, figsize=(12, 6), cmap='viridis')

    plt.title(title)
    plt.xlabel("Components")
    plt.ylabel("Contribution")
    plt.xticks(rotation=0)
    plt.legend(title='Disease State', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_tsne_by_component(W):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(W)  # Perform t-SNE on transposed W

    num_components = W.T.shape[0]
    fig, axes = plt.subplots(nrows=1, ncols=num_components, figsize=(15, 5))

    # For each component, create a t-SNE plot
    for i in range(num_components):
        tsne_df = pd.DataFrame(data=tsne_results, columns=['Dim 1', 'Dim 2'])
        tsne_df[f'Component {i}'] = W.T[i, :] 

        # Create a scatter plot colored by component value
        sns.scatterplot(
            x="Dim 1", y="Dim 2",
            hue=f'Component {i}',
            data=tsne_df,
            palette="viridis",
            alpha=0.8,
            ax=axes[i]
        )
        axes[i].set_title(f"t-SNE Colored by Component {i}")

    plt.tight_layout()
    plt.show()

def plot_tsne(W, metadata, disease_state_column):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(W)

    labels = metadata[disease_state_column].values

    if len(tsne_results) != len(labels):
        raise ValueError("The number of labels does not match the number of samples in W.T.")

    tsne_df = pd.DataFrame(data=tsne_results, columns=['Dim 1', 'Dim 2'])
    tsne_df['Disease State'] = labels

    unique_disease_states = sorted(tsne_df['Disease State'].unique())

    plt.figure(figsize=(5, 8))
    sns.scatterplot(x="Dim 1", y="Dim 2", hue="Disease State", data=tsne_df, palette="viridis", alpha=0.8, hue_order=unique_disease_states)
    plt.title("t-SNE of Components")
    plt.tight_layout()
    plt.show()

def top_features_by_component(H, column_names, top_n=20, plot=True, name = 'Microbe'):
    top_features = {}
    for i in range(H.shape[0]):
        top_indices = np.argsort(-np.abs(H[i]))[:top_n]
        top_features[i] = [column_names[idx] for idx in top_indices]
        if plot:
            plt.figure(figsize=(8, 5))
            plt.barh(range(len(top_features[i])), [np.abs(H[i, idx]) for idx in top_indices], color='skyblue')
            plt.yticks(range(len(top_features[i])), [column_names[idx] for idx in top_indices])
            plt.ylabel(name)
            plt.xlabel('Loading')
            plt.title(f"Top {top_n} {name} for Component {i}")
            plt.show()
    return top_features

def evaluate_reconstruction_quality(original_data, reconstructed_data):
    metrics = {}
    # Calculate Mean Squared Error
    metrics['MSE'] = mean_squared_error(original_data, reconstructed_data)
    # Calculate Root Mean Squared Error
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    # Calculate Mean Absolute Error
    metrics['MAE'] = mean_absolute_error(original_data, reconstructed_data)
    # Calculate R-squared
    metrics['R2'] = r2_score(original_data, reconstructed_data)
    # Calculate Explained Variance Score
    metrics['Explained Variance'] = explained_variance_score(original_data, reconstructed_data)

    return metrics
