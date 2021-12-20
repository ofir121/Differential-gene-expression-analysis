import ast
import multiprocessing
import os
import pickle
import shutil
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

from scipy import stats
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from bioinfokit import visuz
import scipy.stats as st
import mapply


################################# Part 1 ##########################################

def load_expression_dataset(counts_file: str, sample_annotation_file: str) -> DataFrame:
    """
    Loads the 3 files and merge them into 1 file
    :param counts_file: File with read count for each gene for each sample
    :param sample_annotation_file: Metadata for each sample
    :param gene_annotation_file: Information about each gene, such as name
    :return: A merged dataframe
    """
    print("Loading dataset")
    counts_df = pd.read_csv(counts_file, sep="\t", index_col=0)
    sample_annotation_df = pd.read_csv(sample_annotation_file, sep="\t")

    df = counts_df
    df = df.transpose()  # Make the genes the columns
    df = pd.merge(df, sample_annotation_df, left_index=True, right_on='sample_id').set_index(['sample_id', 'type'])
    return df


def annotate_genes(df: DataFrame, gene_annotation_file: str):
    """
    Replace ensembel id with gene names for each column
    :param df: A dataframe with columns as genes
    :param gene_annotation_file: Gene annotation file
    :return: A dataframe with gene symbols instead of ensembl ids
    """
    gene_annotation_df = pd.read_csv(gene_annotation_file, sep="\t")
    # Make a dictionary of gene annotation
    gene_annotation_dict = dict(zip(gene_annotation_df.ENSEMBL, gene_annotation_df.SYMBOL))
    # Annotate genes
    result_df = df.rename(columns=lambda x: gene_annotation_dict[x] if x in gene_annotation_dict else x)
    return result_df


def normalize_cpm(counts_df: DataFrame, to_print=True) -> DataFrame:
    """
    Normalize read count by CPM method
    :param counts_df: A dataframe with read count for each gene
    :param to_print: print step name
    :return: A CPM normalized dataframe
    """
    if to_print:
        print("Normalizing by CPM")
    res = counts_df.div(counts_df.sum(axis=1), axis=0).mul(pow(10, 6))
    return res


def filter_lowly_expressed_genes(cpm_counts_df: DataFrame, X: float, Y: float, Z: int, to_print=True) -> DataFrame:
    """
    Filter the count data for lowly-expressed genes, for example, only keep genes with a
    CPM >= X in at least Y% samples, in at least Z of the groups (X, Y, Z to be defined by
    you).
    :param cpm_counts_df:
    :param X: The minimum number of CPM for at least Y% samples
    :param Y: The percentage of number of samples that need to pass threshold in each group
    :param Z: The minimum number of groups that need to pass the filter for each gene
    :param to_print: Print step name
    :return: A filtered dataframe without lowly-expressed genes
    """
    to_keep_genes = []
    count_passed_filter_genes_dict = defaultdict(lambda: 0)
    if to_print:
        print("Filtering lowly expressed genes")
    grouped = cpm_counts_df.groupby('type')
    for name, group in grouped:
        num_of_samples = len(group)
        group_pass_filter_count = group[group > X].count()
        min_num_of_samples = num_of_samples * (Y / 100)
        for g, passed_sample_count in group_pass_filter_count.items():
            if passed_sample_count >= min_num_of_samples:
                count_passed_filter_genes_dict[g] += 1
    for k, v in count_passed_filter_genes_dict.items():
        if v >= Z:
            to_keep_genes.append(k)
    filtered_df = cpm_counts_df[to_keep_genes]
    return filtered_df


def log_cpm(cpm_counts_df: DataFrame) -> DataFrame:
    """
    Log2 of cpm of each gene
    :param cpm_counts_df: Dataframe with CPM counts
    :return: A dataframe that each cell is the log2(value) of the cell
    """

    result_df = pd.DataFrame(np.log2(cpm_counts_df, out=np.zeros_like(cpm_counts_df), where=(cpm_counts_df != 0)),
                             index=cpm_counts_df.index, columns=cpm_counts_df.columns)

    return result_df


def save_df_to_file(df: DataFrame, output_file: str):
    """
    Save a dataframe to a file
    :param df: Dataframe
    :param output_file: File to save the dataframe to
    """
    with open(output_file, 'wb') as f:
        pickle.dump(df, f)


def calculate_pca(df: DataFrame, target_column_name, normalize, n_dim: int) -> DataFrame:
    """
    Calculate the prinipal componenets of the PCA
    :param df: Dataframe
    :param target_column_name: label column in (Exist in index)
    :param normalize: Normalize the data
    :param n_dim: Number of dimentions for PCA
    :return: Dataframe with principal components
    """
    # Separating out the features
    x = df.loc[:, df.columns].values
    # Separating out the target
    y = pd.Series(df.index.get_level_values(target_column_name))
    # Standardizing the features
    if normalize:
        x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=n_dim)
    principal_components = pca.fit_transform(x)
    df_principal = pd.DataFrame(data=principal_components,
                                columns=[f'principal component {i}' for i in range(1, n_dim + 1)])

    result_df = pd.concat([df_principal, y], axis=1)
    return result_df


def plot_pca(df: DataFrame, output_dir, suffix, normalize=True, target_column_name='type'):
    """
    Plots PCA for the dataframe
    :param df: A dataframe with features, index will contain the label in column 'type'
    :param output_dir: Output directory for plot
    :param suffix: Suffix for file and title
    :param normalize: Do standard scalar normaliztion, should be used if data is not normalized
    :param target_column_name: The name of the column containing the label (located in the index)
    """
    print(f"Plotting PCA for {suffix}")
    output_file = os.path.join(*[output_dir, 'PCA-' + suffix + '.png'])

    final_df = calculate_pca(df, target_column_name, normalize, n_dim=2)

    # Plot PCA
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA ' + suffix, fontsize=15)
    targets = list(final_df[target_column_name].unique())
    colors = ['r', 'g', 'b']
    for target, color in zip(targets, colors):
        indicesToKeep = final_df[target_column_name] == target
        ax.scatter(final_df.loc[indicesToKeep, 'principal component 1']
                   , final_df.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    plt.savefig(output_file)
    plt.clf()
    print(f"PCA {suffix} was saved to {output_file}")


def plot_gene_count_density(df: DataFrame, output_dir):
    """
    Plots the density of the gene count for all samples and features
    :param df: Gene expression count dataframe
    :param output_dir: Output directory for plot
    """
    print("Plotting density")
    output_file = os.path.join(*[output_dir, 'Gene-count-density.png'])
    gene_count = df.stack()
    gene_count.name = 'value'
    gene_count = gene_count.reset_index()
    plot = sns.kdeplot(np.array(gene_count["value"]), bw=0.5)
    plt.title("Density of Log CPM expression for expressed genes")
    plt.xlabel("Log CPM")
    plt.ylabel("Percentage of data points for expressed genes")
    plt.savefig(output_file)
    plot.get_figure().savefig(output_file)
    plt.clf()
    print(f"Gene count density plot was saved to: {output_file}")


def plot_library_sizes_histogram(df: DataFrame, output_dir, line_cutoff=3.725 * 1e7, bins=30):
    """
    Plots the library size of each sample - The total count of the features
    :param df: A dataframe of gene expression counts
    :param output_dir: Output directory for plot
    :param line_cutoff: Suggested cutoff for filteration, will be printed on plot
    :param bins: Number of bins for histogram
    :return:
    """
    print("Plotting library sizes")
    output_file = os.path.join(*[output_dir, 'Library-sizes-histogram.png'])
    library_sizes = df.sum(axis=1)

    plt.hist(list(library_sizes), bins=bins)
    plt.title("Histogram of library sizes")
    plt.xlabel("Library size")
    plt.ylabel("Number of samples")
    plt.axvline(x=line_cutoff, color='r', linestyle='--')  # Suggested cutoff for cleaning

    plt.savefig(output_file)
    plt.clf()
    print(f"Library sizes histogram was saved to: {output_file}")


def fdr(p_vals: List):
    """
    Calculated FDR of p-values
    :param p_vals: A list of p-values
    :return: A list of adjusted p-values by fdr
    """

    from scipy.stats import rankdata
    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr


def plot_non_zero_feature_count(df: DataFrame, output_dir: str, line_cutoff=31000, bins=30):
    """
    Plot the number of non zero features for each sample in a histogram
    :param df: A dataframe of gene counts
    :param output_dir: Output directory
    :param line_cutoff: A suggested cutoff for cleaning
    :param bins: Number of bins for histogram
    """
    print("Plotting feature counts per sample")

    output_file = os.path.join(*[output_dir, 'Expressed-feature-counts-histogram.png'])
    feature_counts = df.astype(bool).sum(axis=1)
    plt.hist(list(feature_counts), bins=bins)
    plt.title("Histogram of non zero feature count")
    plt.xlabel("Number of non zero features")
    plt.ylabel("Number of samples")
    df.astype(bool).sum(axis=0)
    plt.axvline(x=line_cutoff, color='r', linestyle='--')  # Suggested cutoff for cleaning
    plt.savefig(output_file)
    plt.clf()
    print(f"None zero features count was saved to: {output_file}")


def explore_main_properties(df_raw, df_norm: DataFrame, output_dir: str, min_library_size: int,
                            min_non_zero_features: int):
    """
    Explore the main properties of the data
    :param df_raw: Raw dataframe of gene expression counts
    :param df_norm: Df normalized and logged by cpm
    :param output_dir: Output directory, inside Plots/ will be created to output the plots
    :param min_library_size: Minimum library size suggested cutoff, will be written on plot
    :param min_non_zero_features: Minimum non zero features suggested cutoff, will be written on plot
    """
    print("Exploring main properties")
    plots_output_dir = os.path.join(*[output_dir, "Plots"])

    # Make directory for the plots if doesn't exist
    Path(plots_output_dir).mkdir(exist_ok=True)

    # Plot different statistics on the data
    plot_library_sizes_histogram(df_raw, plots_output_dir, min_library_size)
    plot_gene_count_density(df_norm, plots_output_dir)
    plot_non_zero_feature_count(df_raw, plots_output_dir, min_non_zero_features)
    plot_pca(df_norm, plots_output_dir, 'expressed-cpm-normalized-genes', normalize=False)
    plot_pca(df_raw, plots_output_dir, 'all-genes', normalize=False)


def clean_df(df_raw: DataFrame, output_dir, X, Y, Z, min_library_size, min_non_zero_features,
             plot_pca_after_cleaning=True) -> DataFrame:
    """
    Clean the dataframe by the cutoffs found in statistics exploration
    Also plots a PCA after cleaning
    :param df: Dataframe
    :return: A cleaned dataframe
    """

    # Clean samples with many non-zero features
    result_df = df_raw[df_raw.sum(axis=1) > min_library_size]
    result_df = result_df[result_df.astype(bool).sum(axis=1) > min_non_zero_features]

    df_norm_cleaned = log_cpm(
        filter_lowly_expressed_genes(normalize_cpm(result_df, to_print=False), X, Y, Z, to_print=False))

    # Plot PCA after cleaning
    if plot_pca_after_cleaning:
        plots_output_dir = os.path.join(*[output_dir, "Plots"])
        plot_pca(df_norm_cleaned, plots_output_dir, 'expressed-cpm-cleaned-normalized-genes', normalize=False)

    return df_norm_cleaned


def differential_analysis(df: DataFrame) -> DataFrame:
    """
    Runs statistical test to check the difference between the samples
    :param df: A dataframe of logged cpm normalized expression data, label will be in index in 'type' column
    :return: A dictionary with genes and their respected p-value
    """
    # create an empty dictionary
    stats_test_results = {}
    labels = pd.Series(df.index.get_level_values('type'))
    group_normal = df[list(labels == 'normal')]
    group_lesional = df[list(labels == 'lesional')]
    # loop over column_list and execute code explained above
    for column in df.columns:
        group_normal_column = group_normal[column]
        group_lesional_column = group_lesional[column]
        # add the output to the dictionary
        stats_test_results[column] = stats.ttest_ind(group_normal_column, group_lesional_column)
    results_df = pd.DataFrame.from_dict(stats_test_results, orient='Index')
    results_df.columns = ['statistic', 'pvalue']
    results_df['fdr'] = fdr(results_df['pvalue'])
    # Adjust p-values with FDR because many tests were preformed
    return results_df


def select_top_k_genes(df_genes_statistics: DataFrame, k=100, decision_method='fdr') -> List[str]:
    """
    Select lowest fdr
    :param df_genes_statistics: Dataframe with the the p-value for each gene and fdr
    :param decision_method: Decision method for smallest: pvalue or fdr
    :param k The number of genes to return
    :return: A list of genes that got the best result
    """
    best_genes = list(df_genes_statistics.nsmallest(k, decision_method).index)
    return best_genes


def filter_df_by_genes(df_genes: DataFrame, top_genes: List[str]) -> DataFrame:
    df_filtered = df_genes.filter(items=top_genes)
    return df_filtered


def plot_gene_expression_heatmap(df: DataFrame, output_dir: str):
    df.sort_index(level='type', inplace=True)
    labels = pd.Series(df.index.get_level_values('type'))
    y_labels = df.columns
    x_labels = pd.Series(df.index.get_level_values('sample_id'))
    df = df.transpose()
    output_file = os.path.join(*[output_dir, "Plots", "top-100-genes-heatmap.svg"])
    lut = dict(zip(labels.unique(), "rbg"))
    col_colors = labels.map(lut)
    sns.set(font_scale=0.35)
    ax = sns.clustermap(df.reset_index(drop=True), xticklabels=x_labels, yticklabels=y_labels,
                        col_colors=col_colors.values)
    ax.ax_heatmap.set(xlabel="Gene", ylabel="Sample id", )
    # Add legend
    handles = [Patch(facecolor=lut[name]) for name in lut]
    plt.legend(handles, lut, title='Type',
               bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right', prop={'size': 6})
    # Save figure
    ax.savefig(output_file)
    sns.set(font_scale=1)
    plt.clf()
    print(f"Gene expression heatmap was saved to: {output_file}")


def plot_volcano(log_cpm_counts_df: DataFrame, statistics_df: DataFrame, output_dir: str):
    """
    Plots a volcano plot of the log cpm counts
    :param log_cpm_counts_df: A dataframe with log cpm counts on each gene for all samples
    :param statistics_df: A dataframe with p-value and fdr on the difference between the 2 groups
    :param output_dir: Output directory
    """
    output_file = os.path.join(*[output_dir, "Plots", "volcano.png"])
    labels = pd.Series(log_cpm_counts_df.index.get_level_values('type'))
    df_normal = log_cpm_counts_df[list(labels == 'normal')]
    df_lesional = log_cpm_counts_df[list(labels == 'lesional')]

    df_normal = df_normal.transpose()
    df_lesional = df_lesional.transpose()
    df_normal = df_normal.sum(axis=1)
    df_lesional = df_lesional.sum(axis=1)
    df = pd.DataFrame(df_lesional - df_normal, columns=['log2FC'])  # Same as Log(A/B) (Log(A/B)= LogA - LogB)
    df = pd.merge(df, statistics_df[['fdr']], left_index=True, right_index=True)
    df = df.rename(columns={'fdr': 'adjusted p-value (FDR)'})

    # Get p-value threshold for top 100 genes
    best_genes = select_top_k_genes(statistics_df)
    top_10_genes = select_top_k_genes(statistics_df, 10)  # For the annotation on the plot

    best_genes_pvalues = statistics_df[statistics_df.index.isin(set(best_genes))]['fdr']
    max_pvalue_best_genes = best_genes_pvalues.max()
    min_pvalue_best_genes = best_genes_pvalues.min()
    df.reset_index(inplace=True)

    visuz.GeneExpression.volcano(df=df, lfc='log2FC', pv='adjusted p-value (FDR)', geneid='index',
                                 pv_thr=(max_pvalue_best_genes, max_pvalue_best_genes), ar=0, plotlegend=True,
                                 gfont=5)
    shutil.move("volcano.png",
                output_file)  # File is created automatically in current directory, this moves the file to designated location
    print(f"Volcano plot was saved to {output_file}")


def run_part_1():
    """
    Runs the first part of the analysis
    """
    # Define script variables

    # Variables
    X, Y, Z = 1, 20, 1
    min_library_size = int(3.7275 * 1e7)
    min_non_zero_features = 31000

    # Output files
    output_dir = 'Output'
    output_norm_data_file = os.path.join(*[output_dir, "counts_normalized_log_filtered.pkl"])
    # Input files
    counts_file = 'Data/counts.txt'
    gene_annotation_file = 'Data/gene-annotation.txt'
    sample_annotation_file = 'Data/sample-annotation.txt'

    print("Running part 1 - Differential gene expression analysis")
    df = load_expression_dataset(counts_file, sample_annotation_file)
    df_norm = normalize_cpm(df)
    df_norm_filtered = filter_lowly_expressed_genes(df_norm, X, Y, Z)
    df_norm_filtered_log = log_cpm(df_norm_filtered)
    save_df_to_file(df_norm_filtered_log, output_norm_data_file)
    # explore_main_properties(df, df_norm_filtered_log, output_dir, min_library_size, min_non_zero_features)
    df_norm_cleaned = clean_df(df, output_dir, X, Y, Z, min_library_size, min_non_zero_features)
    differential_statistics = differential_analysis(df_norm_cleaned)
    top_genes = select_top_k_genes(differential_statistics)
    df_top_genes = filter_df_by_genes(df_norm_cleaned, top_genes)
    top_genes_annotated = annotate_genes(df_top_genes, gene_annotation_file)
    plot_gene_expression_heatmap(top_genes_annotated, output_dir)
    plot_volcano(df_norm_filtered_log, differential_statistics, output_dir)


########################################### Part 2 ###############################################

def load_features_and_labels(features_file: str, labels_file: str) -> DataFrame:
    """
    Load the features and the labels
    :param features_file: The file with all the features
    :param labels_file: The file with all the labels
    :return: A merged dataframe while the label is in the index of the dataframe
    """
    X = pd.read_csv(features_file)
    X.columns.values[0] = 'Sample id'
    y = pd.read_csv(labels_file)
    y.columns.values[0] = 'Sample id'
    df = pd.merge(X, y, on="Sample id")
    df.set_index(["Sample id", 'target'], inplace=True)

    return df


def get_na_columns(df: DataFrame) -> List[str]:
    """
    Get a list of all the columns with NA in the dataframe
    :param df: Dataframe
    :return: List of column names that contain NA values
    """
    return df.columns[df.isna().any()].tolist()


def export_basic_statistics(df: DataFrame, output_dir: str):
    """
    Explore basic statistics of the Dataframe for classification
    :param df: Dataframe
    :param output_dir: Output directory for statistics
    """
    output_dir_plots = os.path.join(*[output_dir, "Classification_Plots"])
    Path(output_dir_plots).mkdir(exist_ok=True)

    # Plot number of samples in each group
    output_file = os.path.join(*[output_dir_plots, 'Targets-count.png'])
    ax = sns.countplot(df.index.get_level_values('target'), label="target")
    ax.set(xlabel='Target class', ylabel='Number of samples')
    plt.savefig(output_file)
    plt.clf()
    print(f"Number of samples in each group plot was saved to: {output_file}")

    # Evaluate general statistics on each feature
    output_file = os.path.join(*[output_dir, "features-statistics.csv"])
    df.describe().to_csv(output_file)
    print(f"General statistics of features was saved to: {output_file}")
    output_file = os.path.join(*[output_dir, 'features-info.txt'])
    with open(output_file, "w") as f:
        df.info(verbose=True, buf=f)
    print(f"Additional features information was saved to: {output_file}")

    # Print columns with NA
    print(f"Columns with NA values in features (Will be removed): {get_na_columns(df)}")


def clean_classification_data(df: DataFrame) -> DataFrame:
    """
    Clean the classification data from negative values and fill NAs with mean of the feature
    :param df: A dataframe of features
    :return: Cleaned dataframe
    """
    # Remove negative value features (Some features have noisy negative values)
    df_cleaned = df.transpose()[(df.transpose() > 0).all(1) | df.isna().any()].transpose()

    # Replace NAN values with mean of feature in group
    for col in get_na_columns(df_cleaned):
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned.groupby(level='target')[col].transform('mean'))

    return df_cleaned


def get_best_distribution(data: Series, print_statistics=False):
    """
    Find the best distribution for the data and it's parameters
    :param data:
    :param print_statistics: Print the statistics to stdout
    :return: The best distribution name, p-value that the data does NOT belong to the distribution, the distribution parameters
    """
    dist_names = ["norm", "genextreme", "weibull_max", "weibull_min"]  # "exponweib", #"pareto"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        if print_statistics:
            print("p value for " + dist_name + " = " + str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    if print_statistics:
        print("Best fitting distribution: " + str(best_dist))
        print("Best p value: " + str(best_p))
        print("Parameters for the best fit: " + str(params[best_dist]))

    return {"Feature name": data.name, "Distribution": best_dist, "Reject pvalue": best_p, "Params": params[best_dist]}


def explore_pca(df: DataFrame, output_dir_plots: str):
    """
    Explore the PCA components distribution
    :param df: Dataframe with features and labels (in index target column)
    :param output_dir_plots: Output directory for plots
    """
    # Plot pca after cleaning
    plot_pca(df, output_dir_plots, suffix='All-features-PCA', normalize=True,
             target_column_name='target')
    # Explore components distribution
    output_file = os.path.join(*[output_dir_plots, 'PCA-pc1-histogram.png'])
    df_pca = calculate_pca(df, 'target', normalize=True, n_dim=2)
    sns.kdeplot(data=df_pca, x='principal component 1', hue='target', fill=True, common_norm=False, alpha=0.4)
    plt.savefig(output_file)
    print(f"PCA principal component 1 distribution of the 2 groups was saved to {output_file}")
    plt.clf()
    output_file = os.path.join(*[output_dir_plots, 'PCA-pc2-histogram.png'])
    sns.kdeplot(data=df_pca, x='principal component 2', hue='target', fill=True, common_norm=False, alpha=0.4)
    plt.savefig(output_file)
    print(f"PCA principal component 2 distribution of the 2 groups was saved to {output_file}")
    plt.clf()


def create_feature_from_distribution(dist_row: DataFrame, num_of_samples: int):
    """
    Create k samples from a feature from a given distribution and its parameters
    :param dist_row: Row that contains Distribution name, its params and the feature name.
    :param num_of_samples: Number of samples from the feature to create
    :return: k random features from the given distribution
    """
    np.random.seed(seed=233423)  # In order to reproduce the result
    dist_params = list(dist_row['Params'])
    dist_name = dist_row['Distribution']
    feature_name = dist_row['Feature name']
    dist = getattr(st, dist_name)
    if dist_name == 'norm':
        result = dist.rvs(dist_params[-2], dist_params[-1], size=num_of_samples)  # Norm requires only 2 params
    else:
        result = dist.rvs(*dist_params, size=num_of_samples)
    return pd.Series(result, name=feature_name)


def create_samples_from_features_distribution(df_features_distribution: DataFrame, num_of_samples: int) -> DataFrame:
    """
    Create samples using known distributions and their parameters
    :param df_features_distribution: A dataframe with distribution names, feature names, target labels and the distribution params
    :param num_of_samples: Number of samples to create for each group
    :return: A dataframe with samples created from the distribution
    """
    print("Creating meta samples from features distribution")
    meta_samples = pd.DataFrame()
    mapply.init(n_workers=-1)  # Run apply parallel for fast computation
    for name, group in df_features_distribution.groupby('target'):
        feature_samples = group.mapply(create_feature_from_distribution, args=(num_of_samples,), axis=1,
                                       result_type=None)
        df_meta_samples_group = feature_samples.transpose()
        df_meta_samples_group.columns = list(group['Feature name'])
        df_meta_samples_group['target'] = name
        meta_samples = meta_samples.append(df_meta_samples_group).reset_index(drop=True)
    # Redefine index with target
    meta_samples = meta_samples.reset_index().set_index(['index', 'target'])
    return meta_samples


def get_features_distribution(df: DataFrame, output_dir: str, compute_distribution: bool = True) -> DataFrame:
    """
    Get distribution of all features
    :param df: Dataframe
    :param output_dir: Output directory for features distribution
    :param compute_distribution: Compute the distribution or load it from file (To avoid heavy computation)
    :return: A dataframe with the distribution of the different features for each group and its parameters
    """
    print("Get the matched distribution for all features and their parameters")
    output_file = os.path.join(*[output_dir, "features-distribution.csv"])
    if compute_distribution and os.path.isfile(output_file):  # If file exist avoid heavy computation
        mapply.init(n_workers=max(multiprocessing.cpu_count() - 1, 1))  # Run apply parallel for fast computation
        result = pd.DataFrame()
        for name, group in df.groupby(level="target"):
            print(f"Computing group {name}")
            group_result = group.mapply(get_best_distribution, args=(False,), axis=0)
            group_result = pd.DataFrame.from_records(group_result)
            group_result['target'] = name
            result = result.append(group_result, ignore_index=True)
        result.to_csv(output_file, index=False)
    else:
        result = pd.read_csv(output_file, converters={"Params": ast.literal_eval})
    return result


def get_metrics(clf, x, y) -> Dict[str, float]:
    """
    Return metrics for a sklearn classifier
    :param clf: Train classifier
    :param x: Features
    :param y: Labels
    :return: Dictionary of precision, recall, f1 score and AUC score
    """
    results = {}
    # Calculate metrics for fold
    y_pred = clf.predict(x)
    results['precision'] = precision_score(y, y_pred)
    results['recall'] = recall_score(y, y_pred)
    results['f1'] = f1_score(y, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred)
    results['roc auc score'] = metrics.auc(fpr, tpr)
    tn, fp, fn, tp = metrics.confusion_matrix(y, y_pred).ravel()
    results['tn'] = tn
    results['fp'] = fp
    results['fn'] = fn
    results['tp'] = tp
    return results


def seperate_features_and_labels_from_df(df: DataFrame):
    """
    Separate the dataframe features and labels.
    Labels should be in index in target column
    :param df: Dataframe with features and labels
    :return: Tuple for features and labels
    """
    X, y = df.loc[:, df.columns].values, pd.Series(df.index.get_level_values('target'))
    y.replace({'normal': 0, 'lesional': 1}, inplace=True)  # Replace labels with 0,1 for metrics calculation

    return X, y


def create_classifier_by_name(clf_name: str):
    """
    Classifier factory by name
    :param clf_name: Name of classifier (From the list below)
    :return:
    """
    if clf_name == 'Random Forest':
        return RandomForestClassifier(random_state=0)
    elif clf_name == 'AdaBoost':
        return AdaBoostClassifier(random_state=0)
    elif clf_name == 'Multilayer Perceptron':
        return MLPClassifier(random_state=0, alpha=0.2)
    else:
        print("Incorrect chosen classifier")
        exit(1)


def evaluate_classifier(clf, x_train, x_test, y_train, y_test) -> Dict:
    """
    Evaluate a specific classifier on a defined train and test
    :param clf: Sklearn Classifier
    :param x_train: Features of the training set
    :param x_test: Features of the testing set
    :param y_train: Labels of the training set
    :param y_test: Labels of the testing set
    :return: Classifier metrics result on the testing set
    """
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    clf.fit(x_train, y_train)
    metrics = get_metrics(clf, x_test, y_test)
    return metrics


def cross_evaluate_classifier(df: DataFrame, classifier_name: str, n_splits: int = 5) -> Dict:
    """
    Run cross validation on a specific classifier
    :param df: Dataframe of features and labels
    :param classifier_name: Classifier name
    :param n_splits: Number of splits for cross validation, default 80% train :20% test cross validation
    :return: Mean metrics of cross validation
    """
    # Separate features and labels
    X, y = seperate_features_and_labels_from_df(df)

    clf = create_classifier_by_name(classifier_name)
    # Run k-fold stratified classification
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    results = defaultdict(list)
    # enumerate the splits and summarize the distributions
    for train_ix, test_ix in kfold.split(X, y):
        x_train_fold, x_test_fold = X[train_ix], X[test_ix]
        y_train_fold, y_test_fold = y[train_ix], y[test_ix]

        results_fold = evaluate_classifier(clf, x_train_fold, x_test_fold, y_train_fold, y_test_fold)

        # Combine results
        for key, value in results_fold.items():
            results[key].append(value)

    # Mean results for all folds
    mean_results = {key: np.mean(values) for key, values in results.items()}
    mean_results['classifier'] = classifier_name
    return mean_results


def cross_evaluate_classifiers(df: DataFrame, classifiers_names: List, output_dir: str):
    """
    Cross evaluate the different classifiers in list
    :param df: Dataframe
    :param classifiers_names: List of classifier names
    :return: Dataframe with mean result of different classifier metrics on the test
    """
    print("Cross evaluation of classifiers:")
    output_file = os.path.join(*[output_dir, "cross-validation-classification-results.csv"])
    results = []
    for clf_name in classifiers_names:
        results.append(cross_evaluate_classifier(df, clf_name))
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results for classification with cross-validation 80-20 was saved to {output_file}")

    print(results_df)


def evaluate_classifiers(df_train: DataFrame, df_test: DataFrame, classifiers_names: List, output_dir: str):
    """
    Evaluate different classifiers for a specific train-test set
    :param df_train: Train dataframe with features and labels
    :param df_test: Test dataframe with features and labels
    :param classifiers_names: Classifiers names to test
    :return: Dataframe with different statistics
    """
    print("Classifiers evaluation of meta-samples:")
    output_file = os.path.join(*[output_dir, "meta-samples-classification-results.csv"])
    results = []
    for clf_name in classifiers_names:
        x_train, y_train = seperate_features_and_labels_from_df(df_train)
        x_test, y_test = seperate_features_and_labels_from_df(df_test)
        clf_results = evaluate_classifier(create_classifier_by_name(clf_name), x_train, x_test, y_train, y_test)
        clf_results['classifier'] = clf_name
        results.append(clf_results)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results for classification using meta-samples was saved to {output_file}")
    print(results_df)


def run_part_2():
    """
    Run the second part of the analysis
    """
    # Define variables
    num_of_meta_samples = 5
    features_file = os.path.join(*["Data", 'features.txt'])
    labels_file = os.path.join(*["Data", "labels.txt"])
    output_dir = "Output"
    output_dir_plots = os.path.join(*[output_dir, "Classification_Plots"])
    compute_distribution = False  # Set to false to avoid heavy computation

    print("Running part 2 - classification problem solving")
    df = load_features_and_labels(features_file, labels_file)
    export_basic_statistics(df, output_dir)
    df = clean_classification_data(df)
    explore_pca(df, output_dir_plots)
    classifier_names = ['Random Forest', 'AdaBoost', 'Multilayer Perceptron']
    cross_evaluate_classifiers(df, classifier_names, output_dir)
    df_features_distribution = get_features_distribution(df, output_dir, compute_distribution=compute_distribution)
    df_meta_samples = create_samples_from_features_distribution(df_features_distribution,
                                                                num_of_samples=num_of_meta_samples)
    # Train by meta-samples and predict on real data
    evaluate_classifiers(df_train=df_meta_samples, df_test=df, classifiers_names=classifier_names, output_dir=output_dir)


def main() -> int:
    pd.set_option('display.max_columns', None)  # Display all columns of dataframes
    run_part_1()
    run_part_2()
    return 0


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.exit(main())
