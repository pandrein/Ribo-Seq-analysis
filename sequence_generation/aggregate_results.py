import multiprocessing
import os
import sys
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from Comparison_digital_profiles import compare_pair
from Signal_digitalisation import MatricesExtractor
from utils import InputFileManager
from utils import create_dir_if_not_exist

np.random.seed(10)

np.set_printoptions(threshold=sys.maxsize)

load_intermediate_results_from_csv = False
add_small_random_value_to_real_scores = False


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')


num_cores = multiprocessing.cpu_count()
num_task = num_cores - 1
plot_data = True
num_bins = 15

# I/O directories
input_dir = os.path.join(os.getcwd(), "check_reduced/")  # get the path to the data input directory
match_scores_output_dir = os.path.join(os.getcwd(), "matrix_python/match_scores/")  # Sets the directory where all the saved outputs will be stored
reproducible_sequence_output_dir = os.path.join(os.getcwd(), "matrix_python/reproducible_sequence/")  # Sets the directory where all the saved outputs will be stored
genes_lengths_path = os.path.join(os.getcwd(), "gene_lengths.csv")  # path to upload the file containing each gene's ID and the correspondent gene length
histogram_plot_path = os.path.join(os.getcwd(), "genes_histograms/")  # path to upload the file containing each gene's ID and the correspondent gene length
intermediate_results = os.path.join(os.getcwd(), "intermediate_results/")
plots_folder = os.path.join(os.getcwd(), "plots/")
match_scores_hist_plot_folder = os.path.join(plots_folder, "match_scores_hist/")
match_coverage_hist_plot_folder = os.path.join(plots_folder, "coverage_hist/")
path_match_score_csv = os.path.join(os.getcwd(), "path_match_score_csv/")
create_dir_if_not_exist([input_dir, match_scores_output_dir, histogram_plot_path, reproducible_sequence_output_dir, intermediate_results, plots_folder, match_scores_hist_plot_folder,path_match_score_csv])

FDR = 0.01


def signal_digitalisation(genes, bed_files_dicts, areReadsRandomized, add_small_random_value):
    matrix_01_list = []
    for bed_files_dict in bed_files_dicts:
        bed_file = bed_files_dict["bed_file"]
        bed_file_name = bed_files_dict["bed_file_name"]
        me = MatricesExtractor(bed_file, genes)
        # extract the matrices
        pd_matrix_coverage, matrix_01 = me.extract_matrices(areReadsRandomized=areReadsRandomized, add_small_random_value=add_small_random_value)
        if plot_data:
            for gene, coverage in pd_matrix_coverage.iterrows():
                coverage = coverage[~np.isnan(coverage)]

                match_scores_hist_pair_plot_folder = os.path.join(match_coverage_hist_plot_folder, bed_file_name)
                create_dir_if_not_exist([match_scores_hist_pair_plot_folder])
                # print_full(coverage)
                x = range(0, len(coverage))
                plot = sns.lineplot(x, coverage, color='black')
                plot.fill_between(x, coverage, color='black')

                plot.set(xticks=((x[0::int(len(coverage) * 0.08)])))
                plot.get_figure().savefig(os.path.join(match_scores_hist_pair_plot_folder, "gene:" + gene))
                plot.get_figure().clf()

        matrix_01_list.append({'matrix': matrix_01, 'file_name': bed_file_name})

    return matrix_01_list


def compute_real_match_scores(genes, bed_files_dicts, save_results=True):
    print("start real matrix digitalization...")
    matrix_01_list = signal_digitalisation(genes, bed_files_dicts, areReadsRandomized=False, add_small_random_value=add_small_random_value_to_real_scores)
    print("digitalization complete...")
    # first = matrix_01_list[0]["matrix"]
    # second = matrix_01_list[1]["matrix"]

    # gets the gene list for each matrix
    gene_lists = [pd.DataFrame(f['matrix'].index, columns={"GeneID"}) for f in matrix_01_list]
    # gets the genes in common for each matrix
    gene_list = gene_lists[0] 
    if len(gene_lists) > 1:
        for i in range(1, len(gene_lists)):
            l = gene_lists[i]
            gene_list = gene_list.merge(l, on="GeneID")

    # gene_list['GeneID'] = gene_list.index
    gene_list = gene_list.to_numpy().squeeze()

    # lists of pair of matrices
    pairs = [list(f) for f in combinations(matrix_01_list, 2)]

    print("start pair comparison...")
    # saves match scores
    match_scores = []
    pair_names_list = []
    for pair in pairs:
        # print ("compare " + pair[0]['file_name'] + " and " + pair[1]['file_name'])
        match_score, pair_names = compare_pair(pair, genes.set_index('GeneID'), gene_list)
        pair_names = Path(pair_names[0]).stem + ":" + Path(pair_names[1]).stem
        match_scores.append({"match_score": match_score, "pair_name": pair_names})
        pair_names_list.append(pair_names)
        if save_results:
            pair_names = pair_names + ".csv"
            match_score.to_csv(os.path.join(match_scores_output_dir, pair_names), index=True, header=True, decimal='.', sep=',', float_format='%.6f')
    print("real comparison complete")

    return gene_list, match_scores, pair_names_list, matrix_01_list


def calc_reproducible_sequences(match_scores_list, gene_list, pair_names_list, match_scores_real, matrix_01_list):
    # compute the match score histograms for the random comparisons
    match_scores_hist = {}
    for fake_match_scores in match_scores_list:
        for fake_match_score in fake_match_scores:
            # fake_match_score contains the scores of one pair
            pair_name = fake_match_score['pair_name']
            match_scores_fake = fake_match_score['match_score']
            gene_hist = {}
            for gene, match_score in match_scores_fake.items():
                gene_hist[gene] = [match_score]

            if pair_name in match_scores_hist:
                for gene, match_score in match_scores_fake.items():
                    match_scores_hist[pair_name][gene].append(match_score)  # = [match_score]
            else:
                match_scores_hist[pair_name] = gene_hist

    p_value_matrix = pd.DataFrame(index=gene_list, columns=pair_names_list)

    plot_num = 0

    # extract pvalues for each gene and dataset pair
    for pair_name in match_scores_hist:
        for gene in match_scores_hist[pair_name]:
            gene_hist = pd.Series(match_scores_hist[pair_name][gene])
            hist_mean = np.mean(gene_hist)
            hist_std = np.std(gene_hist)

            if plot_data:
                match_scores_hist_pair_plot_folder = os.path.join(match_scores_hist_plot_folder, pair_name)
                create_dir_if_not_exist([match_scores_hist_pair_plot_folder])
                sns.set_style('darkgrid')
                plot = sns.distplot(gene_hist, bins=num_bins).set_title("hist_mean: " + str('%.5f' % hist_mean) + "   hist_std: " + str('%.5f' % hist_std))
                plot.get_figure().savefig(os.path.join(match_scores_hist_pair_plot_folder, "gene:" + gene))
                plot.get_figure().clf()

            for match_score_real in match_scores_real:
                pair_name_real = match_score_real["pair_name"]
                if pair_name_real == pair_name:
                    real_score = match_score_real["match_score"][gene]
                    z_score = (real_score - hist_mean) / hist_std
                    pvalue = st.norm.sf(abs(z_score))
                    p_value_matrix[pair_name][gene] = pvalue

            plot_num += 1

    reproducible_genes = []
    for gene, pvalue_row in p_value_matrix.iterrows():
        pvalue_row = pvalue_row.to_numpy()

        y = multipletests(pvals=pvalue_row, alpha=FDR, method="fdr_bh")
        number_of_significative_values_python = len(y[1][np.where(y[1] < FDR)])

        pvalue_row = np.sort(pvalue_row)
        critical_values = ((np.nonzero(pvalue_row >= 0)[0] + 1) / len(pair_names_list)) * FDR
        bh_candidates = pvalue_row[pvalue_row <= critical_values]

        if len(bh_candidates) > 0:
            idx_of_max_value = np.argwhere(bh_candidates == np.amax(bh_candidates)).flatten().tolist()[-1] + 1
            bh_selected = pvalue_row[np.array(range(0, idx_of_max_value))]
            if len(bh_selected) == len(pair_names_list):
                reproducible_genes.append(gene)

    reproducible_sequence_mask, first_matrix_01_with_only_reproducible_genes = extract_reproducible_sequences(reproducible_genes, matrix_01_list)
    # take the first matrix 01 with only reproducible genes and put to zero the non reproducible parts
    first_matrix_01_with_only_reproducible_genes[~reproducible_sequence_mask] = 0
    reproducible_sequence = pd.DataFrame(first_matrix_01_with_only_reproducible_genes, index=reproducible_genes)
    reproducible_sequence.to_csv(os.path.join(reproducible_sequence_output_dir, "reproducible_sequence.csv"), index=True, header=True, decimal='.', sep=',', float_format='%.6f')


def extract_reproducible_sequences(reproducible_genes, matrix_01_list):
    # for each matrix_01 select only the reproducible genes
    reproducible_genes_tables = [matrix_01_struct['matrix'][matrix_01_struct['matrix'].index.isin(reproducible_genes)] for matrix_01_struct in matrix_01_list]
    # select the elements that are one for all the sequences
    sizes = [f.shape for f in reproducible_genes_tables]
    max_size = max(sizes, key=lambda x:x[1])[1]

    matrix_with_same_col_dim = []
    for m in reproducible_genes_tables:
        size = m.shape[1]
        fill_range = np.arange(size, max_size)
        fill_range = [str(item) for item in fill_range]
        matrix_with_same_col_dim.append(m.reindex(list(m) + fill_range, axis=1))

    sequences_ones_mask = [(f == 1).to_numpy() for f in matrix_with_same_col_dim]
    sequences_ones_mask = np.stack(sequences_ones_mask)
    all_ones = np.all(sequences_ones_mask, axis=0)
    # select the elements that are minus one for all the sequences
    sequences_minus_ones_mask = [(f == -1).to_numpy() for f in matrix_with_same_col_dim]
    sequences_minus_ones_mask = np.stack(sequences_minus_ones_mask)
    all_minus_ones = np.all(sequences_minus_ones_mask, axis=0)
    # get a mask with all the elements that are one and minus one for all the sequences
    reproducible_sequence_mask = np.stack([all_ones, all_minus_ones])
    reproducible_sequence_mask = np.any(reproducible_sequence_mask, axis=0)
    return reproducible_sequence_mask, matrix_with_same_col_dim[0].to_numpy()


def main():
    print("num of core available: " + str(num_cores) + " used: " + str(num_task))

    ifm = InputFileManager(genes_lengths_path, input_dir)
    genes = ifm.get_genes()
    bed_files_dicts = ifm.get_bed_files()

    match_scores_list = []
    if load_intermediate_results_from_csv:
        intermediate_results_path_list = [os.path.abspath(os.path.join(path_match_score_csv, f)) for f in os.listdir(path_match_score_csv) if f.endswith(tuple([".csv"]))]
        for i in intermediate_results_path_list:
            with open(i, 'r') as f:
                match_scores = []
                gene_list = []
                ir = pd.read_csv(f)
                for (columnName, columnData) in ir.iteritems():
                    if columnName == "Unnamed: 0":
                        gene_list.append(columnData.to_numpy())
                    else:
                        match_scores.append(columnData.to_numpy())

                for match_score in match_scores:
                    s = pd.Series(match_score, index=gene_list[0]).rename_axis("GeneID")
                    match_scores_list.append({"pair_name": str(Path(os.path.basename(i)).with_suffix('')), "match_score": s})
        match_scores_list = [match_scores_list]
    else:
        intermediate_results_path_list = [os.path.abspath(os.path.join(intermediate_results, f)) for f in os.listdir(intermediate_results) if f.endswith(tuple([".npy"]))]
        for i in intermediate_results_path_list:
            with open(i, 'rb') as f:
                ir = np.load(f, allow_pickle=True)
                match_scores_list.extend(ir)

    gene_list, match_scores_real, pair_names_list, matrix_01_list = compute_real_match_scores(genes, bed_files_dicts)

    calc_reproducible_sequences(match_scores_list, gene_list, pair_names_list, match_scores_real, matrix_01_list)


if __name__ == '__main__':
    main()
