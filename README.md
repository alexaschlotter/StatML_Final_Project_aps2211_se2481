# StatML Final Project aps2211 se2481
# Detection of Stable Microbe-Metabolite Modules Using Integrative Nonnegative Matrix Factorization  

By Alexa Schlotter and Shahd ElNaggar 


**data files:**  
~/final-project/data/YACHIDA_CRC_2019:  Yachida-CRC-2019 (patients with colonoscopy findings from normal to stage 4 colorectal cancer, n = 347)  
~/final-project/data/SINHA_CRC_2016: Sinha-CRC-2016 (CRC and controls, n = 131)  
~/final-project/data/KIM_ADENOMAS_2020: Kim-Aden-Omas-2020 (patients with advanced colorectal adenomas, CRC, controls, n = 240)  
All data in this directory was already preprocessed and downloaded directly from the Borenstein (https://github.com/borenstein-lab/microbiome-metabolome-curated-data)  

**code files:** 
~/final-project/code/iNMF_functions.py : the model implementation and additional visualization functions are all stored here
functions:  
- load data(file_path): Loads data from a given file path containing the taxa or metabolite counts from the given datasets, expecting a tab-separated CSV file, and returns it as a pandas DataFrame with the first column set as the index.
- poisson_loss(X, WH): defines the objective function for iNMF under a Poisson distribution assumption. Calculates the Poisson loss between the observed data X and the model's predictions WH to measure the difference, adjusting WH to prevent zero in logarithmic calculations.
- update_W(Xs, W, Hs, lambda2): updates the current W matrix using the gradient of the objective function with respect to W, with the addition of regularization and normalization terms
- update_H(X, W, H, lambda1, lambda2): updates current H matrices using gradient of objective function with respect to H, with the addition of regularization and normalization terms
- run_inmf(Xs, k, iterations, reps, lambda1, lambda2, tol=1e-5): function to run iNMF under a Poisson distribution assumption using iterative multiplicative updates. Function initializes random W and H matrices, and then iteratively calls update W and H functions to update the respective matrices. After each iteration, the loss is recorded, and convergence is checked using a threshold of difference between iterations. 
- gamma_poisson_loss(X, WH, r): defines objective function for iNMF under a Gamma-Poisson distribution assumption 
- update_W_gp(X, W, H, lambda1, lambda2, r): updates the current W matrix using the gradient of the objective function with respect to W, with the addition of regularization and normalization terms
- update_H_gp(X, W, H, lambda1, lambda2, r): updates current H matrices using gradient of objective function with respect to H, with the addition of regularization and normalization terms
- run_inmf_gp(Xs, k, iterations, reps, lambda1, lambda2, r, tol=1e-5): function to run iNMF under a Gamma-Poisson distribution assumption using iterative multiplicative updates. Function initializes random W and H matrices, and then iteratively calls update W and H functions to update the respective matrices. After each iteration, the loss is recorded, and convergence is checked using a threshold of difference between iterations.
- NMF_obj_eval_yang(Xs, W, Hs, L, Sp, Vs=0): original iNMF objective function code using Frobenius norm (Yang et. Al)
- iNMF_run_yang(Xs, D, L, Sp=0, nrep=200, steps=100): original iNMF run code, (Yang. Et. Al)
- plot_heatmap(best_W, title="Shared component Matrix W"): Displays a heatmap of the matrix best_W, labeling axes for components and features, to visually represent the shared component matrix.
- plot_heatmap_H(H, title): Generates a heatmap for the matrix H with specified title, useful for visualizing the component-feature relationship in H.
- plot_grouped_stacked_bar(W, metadata, disease_state_column, title="Sample Contributions by Components"): Creates a stacked bar chart that groups and visualizes the contributions of samples to components, grouped by disease state from the provided metadata
- plot_tsne_by_component(W): Constructs a series of t-SNE plots, each colored by the values of a different component of matrix W, to visualize the data distribution influenced by each component.
- plot_tsne(W, metadata, disease_state_column): Visualizes a t-SNE plot of matrix W, annotated with disease states from metadata, providing insights into the clustering pattern relative to the disease conditions
- top_features_by_component(H, column_names, top_n=20, plot=True, name = 'Microbe'): Identifies and optionally visualizes the top features for each component in H, showing the features with the highest influence on the components
- evaluate_reconstruction_quality(original_data, reconstructed_data): Computes various statistical metrics to assess the quality of reconstruction between original and reconstructed datasets, including MSE, RMSE, MAE, R-squared, and explained variance.  
	
~/final-project/code/iNMF_run.py : this is our code for running our model on the three datasets and producing visualizations 

**output files: (all pickle files storing output variables)**  

These pickle files contain the model output when run on the three datasets. Each file contains the W, H, V matrices and the best loss curve.  

~/final-project/output/yachida-output.pk1 (generated via poisson iNMF)  
~/final-project/output/yachida-output-gp.pk1 (generated via gamma-poisson iNMF)  
~/final-project/output/yachida-output-yang.pka (generated via original iNMF)  
~/final-project/output/sinha-output.pk1 (generated via poisson iNMF)  
~/final-project/output/sinha-output-gp.pk1 (generated via gamma-poisson iNMF)  
~/final-project/output/sinha-output-yang.pka (generated via original iNMF)  
~/final-project/output/kim-output.pk1 (generated via poisson iNMF)  
~/final-project/output/kim-output-gp.pk1 (generated via gamma-poisson iNMF)  
~/final-project/output/kim-output-yang.pka (generated via original iNMF)  
