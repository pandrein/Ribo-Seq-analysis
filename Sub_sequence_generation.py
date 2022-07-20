import os, shutil, math
import numpy as np
import pandas as pd
from itertools import product


#######################################################################################################################
### FUNCTIONS #########################################################################################################
#######################################################################################################################
def splitList(targs_nucleotides, val, len_min, len_max):
    accepted_lengths = range(len_min, len_max + 1)
    w = np.where(targs_nucleotides == val)
    boolean_vector = -1 * np.ones_like(targs_nucleotides)
    boolean_vector[w] = w
    from itertools import groupby
    lista = [list(group) for k, group in groupby(boolean_vector, lambda x: x >= 0) if k]
    return [i for i in lista if len(i) in accepted_lengths]


#######################################################################################################################
### SCRIPT OPTIONS ####################################################################################################
#######################################################################################################################

### INSTRUCTION
# relax			: used for codon-based examples
# 					> If True, a codon label is based on the major internal label: [1,1,1]->1 , [0,-1,-,1] = -1
# 					> iF False, a codon label has to be homogeneous wrt internal labels: [1,1,1]->1 , [0,-1,-,1] = 0
# len_min 		: minimun length for homogeneous sequence (bases or codons) to be selected for becoming an example
# len_max 		: maximum length for homogeneous sequence (bases or codons) to be selected for becoming an example
# len_example 	: length of the nucleotide-based example (to build the train set)
# save_path		: If '' examples will not be saved, otherwise they will be saved in directory <save_path>
# merge_all		: If True, ALL the following dataset will be merged to create a singole, huge dataset. If False, pass.

### CHANGEBLE PARAMETERS
relax: bool = True
nlen_example: int = 36
nlen_min: int = 1
nlen_max: int = 18
save_path: str = 'Datasets/'
num_test_seqs = 4 #DATASET SPLIT

#######################################################################################################################
#### SCRIPT ###########################################################################################################
#######################################################################################################################

### Loading the DataFrame from Excel file
df = pd.read_excel('E_coli_geni_riproducibili.xlsx', sheet_name=None, header=None, names=['nL', 't'], dtype={'nL': str, 't': int})

### NUCLEOTIDES G:1,0,0,0  A:0,1,0,0  C:0,0,1,0  T:0,0,0,1
nucleotides = dict(zip('GACT', np.eye(4, dtype=int)))

# a = 0
# for key, value in df.items():
# 	if(a==29 or a==30 or a==31 or a==42):
# 		print (a)
# 		print (key, value)
# 	a=a+1

### Matrices: List of matrices encoded-nucleotide-based with dimension (len(nucleotide_sequence), [4+1]) [1hot + target]
matrices = list()
for i in df:
    temp = np.zeros((len(df[i]), 5), dtype=int)
    temp[:, :-1] = [nucleotides[j] for j in df[i]['nL']]
    temp[:, -1] = df[i]['t']
    matrices.append(temp)

### CODONS
### LABEL OF A GENERIC CODON: [1hot n1, 1hot n2, 1hot n3], n=nucleotide. EXAMPLE: ['AAG']->[0,1,0,0,0,1,0,0,1,0,0,0]
# list of 49 elements, each of which is a matrix of dimension (number of codons,3,4)
clen_example = nlen_example // 3 + max(1, nlen_example % 3)

# list of 49 elements, each of which of dimension (number of codons,3) ->[last columns of i for i in Elements]
nlabels = [i[:, :-1] for i in matrices]
ntargets = [i[:, -1] for i in matrices]

Codons = [np.array(tuple([i[j:j + 3, :] for j in range(0, i.shape[0], 3)])) for i in matrices]
Codons_nucleotides_targs = [np.array([i[:, -1] for i in j]) for j in Codons]
clabels = [np.array([i[:, :-1].flatten() for i in j]) for j in Codons]
ctargets = [np.array([v[np.argmax(c)] if max(c) >= (2 * relax + 3 * (1 - relax)) else 0
                      for v, c in [np.unique(j, return_counts=True) for j in i]]) for i in Codons_nucleotides_targs]

### SEQUENCES AS UNIQUE MATRICES
nSeqs = [np.concatenate([nlabels[i], np.array(ntargets[i], ndmin=2).transpose()], axis=1) for i in range(len(nlabels))]
cSeqs = [np.concatenate([clabels[i], np.array(ctargets[i], ndmin=2).transpose()], axis=1) for i in range(len(clabels))]

### FEATURE DIMENSION
nlen_label = nSeqs[0].shape[1]
clen_label = cSeqs[0].shape[1]

### NEGATIVE AND POSITIVE SUB-SEQUENCES
nSeqP = [splitList(i, 1, len_min=nlen_min, len_max=nlen_max) for i in ntargets]
nSeqN = [splitList(i, -1, len_min=nlen_min, len_max=nlen_max) for i in ntargets]

## CONDON BASED
cSeqP = [[list(set(list(np.array(i) // 3))) for i in seq] for seq in nSeqP]
cSeqN = [[list(set(list(np.array(i) // 3))) for i in seq] for seq in nSeqN]


def extract_different_alignments(Seqs, SeqsP, SeqsN, len_example, targets):
    # feature dimension
    len_label = Seqs[0].shape[1]
    ### TRIMMERED 0 PADDING - ST: zeros AFTER sub-sequence, as Pietro did; END: zeros BEFORE sub-sequence
    Trimmed_0_Pos_ST = [np.array([np.concatenate([Seqs[i][j], np.zeros((nlen_example - len(j), len_label))], axis=0) for j in elem], dtype=int) for i, elem in enumerate(SeqsP)]
    Trimmed_0_Neg_ST = [np.array([np.concatenate([Seqs[i][j], np.zeros((nlen_example - len(j), len_label))], axis=0) for j in elem], dtype=int) for i, elem in enumerate(SeqsN)]

    Trimmed_0_Pos_END = [np.array([np.concatenate([np.zeros((nlen_example - len(j), len_label)), Seqs[i][j]], axis=0) for j in elem], dtype=int) for i, elem in enumerate(SeqsP)]
    Trimmed_0_Neg_END = [np.array([np.concatenate([np.zeros((nlen_example - len(j), len_label)), Seqs[i][j]], axis=0) for j in elem], dtype=int) for i, elem in enumerate(SeqsN)]
    ### TRIMMERED E PADDING	- ST: zeros AFTER sub-sequence; END: zeros BEFORE sub-sequence
    Trimmed_E_idxPos_ST = [[range(j[0], j[0] + nlen_example) for j in elem if j[0] + nlen_example <= len(targets[i])] for i, elem in enumerate(SeqsP)]
    Trimmed_E_idxN_ST = [[range(j[0], j[0] + nlen_example) for j in elem if j[0] + nlen_example <= len(targets[i])] for i, elem in enumerate(SeqsN)]
    Trimmed_E_idxPos_END = [[range(j[-1] - nlen_example + 1, j[-1] + 1) for j in i if j[-1] - nlen_example + 1 >= 0] for i in SeqsP]
    Trimmed_E_idxN_END = [[range(j[-1] - nlen_example + 1, j[-1] + 1) for j in i if j[-1] - nlen_example + 1 >= 0] for i in SeqsN]
    Trimmed_E_Pos_ST = [np.array([Seqs[i][j] for j in elem], dtype=int) for i, elem in enumerate(Trimmed_E_idxPos_ST)]
    Trimmed_E_Neg_ST = [np.array([Seqs[i][j] for j in elem], dtype=int) for i, elem in enumerate(Trimmed_E_idxN_ST)]
    Trimmed_E_Pos_END = [np.array([Seqs[i][j] for j in elem], dtype=int) for i, elem in enumerate(Trimmed_E_idxPos_END)]
    Trimmed_E_Neg_END = [np.array([Seqs[i][j] for j in elem], dtype=int) for i, elem in enumerate(Trimmed_E_idxN_END)]
    ### CENTERED 0 PADDING
    Centered_0_Pos_R = [np.array([np.concatenate([np.zeros((math.floor((nlen_example - len(j)) / 2), len_label)), Seqs[i][j], np.zeros((math.ceil((nlen_example - len(j)) / 2), len_label))], axis=0) for j in elem], dtype=int) for i, elem in enumerate(SeqsP)]
    Centered_0_Neg_R = [np.array([np.concatenate([np.zeros((math.floor((nlen_example - len(j)) / 2), len_label)), Seqs[i][j], np.zeros((math.ceil((nlen_example - len(j)) / 2), len_label))], axis=0) for j in elem], dtype=int) for i, elem in enumerate(SeqsN)]
    ### CENTERED E PADDING
    Centered_E_idxP_R = [[range(j[0] - (nlen_example - len(j)) // 2, j[0] - (nlen_example - len(j)) // 2 + nlen_example) for j in elem if j[0] - (nlen_example - len(j)) // 2 >= 0 and j[0] - (nlen_example - len(j)) // 2 + nlen_example <= len(targets[i])] for i, elem in enumerate(SeqsP)]
    Centered_E_idxN_R = [[range(j[0] - (nlen_example - len(j)) // 2, j[0] - (nlen_example - len(j)) // 2 + nlen_example) for j in elem if j[0] - (nlen_example - len(j)) // 2 >= 0 and j[0] - (nlen_example - len(j)) // 2 + nlen_example <= len(targets[i])] for i, elem in enumerate(SeqsN)]
    Centered_E_Pos_R = [np.array([Seqs[i][j] for j in elem], dtype=int) for i, elem in enumerate(Centered_E_idxP_R)]
    Centered_E_Neg_R = [np.array([Seqs[i][j] for j in elem], dtype=int) for i, elem in enumerate(Centered_E_idxN_R)]
    ### RIGHT ALIGNED
    Right_E_idxP = [
        [range(j[-1] - nlen_example // 2 + (nlen_example + 1) % 2, j[-1] - nlen_example // 2 + (nlen_example + 1) % 2 + nlen_example) for j in elem if j[-1] - nlen_example // 2 + (nlen_example + 1) % 2 >= 0 and j[-1] - nlen_example // 2 + (nlen_example + 1) % 2 + nlen_example <= len(targets[i])]
        for i, elem in enumerate(SeqsP)]
    Right_E_idxN = [
        [range(j[-1] - nlen_example // 2 + (nlen_example + 1) % 2, j[-1] - nlen_example // 2 + (nlen_example + 1) % 2 + nlen_example) for j in elem if j[-1] - nlen_example // 2 + (nlen_example + 1) % 2 >= 0 and j[-1] - nlen_example // 2 + (nlen_example + 1) % 2 + nlen_example <= len(targets[i])]
        for i, elem in enumerate(SeqsN)]
    Right_E_Pos = [np.array([Seqs[i][j] for j in elem], dtype=int) for i, elem in enumerate(Right_E_idxP)]
    Right_E_Neg = [np.array([Seqs[i][j] for j in elem], dtype=int) for i, elem in enumerate(Right_E_idxN)]
    ### LEFT ALIGNED
    Left_E_idxP = [[range(j[0] - len_example // 2, j[0] - len_example // 2 + len_example) for j in elem if j[0] - len_example // 2 >= 0 and j[0] - len_example // 2 + len_example <= len(targets[i])] for i, elem in enumerate(SeqsP)]
    Left_E_idxN = [[range(j[0] - len_example // 2, j[0] - len_example // 2 + len_example) for j in elem if j[0] - len_example // 2 >= 0 and j[0] - len_example // 2 + len_example <= len(targets[i])] for i, elem in enumerate(SeqsN)]
    Left_E_Pos = [np.array([Seqs[i][j] for j in elem], dtype=int) for i, elem in enumerate(Left_E_idxP)]
    Left_E_Neg = [np.array([Seqs[i][j] for j in elem], dtype=int) for i, elem in enumerate(Left_E_idxN)]
    return [Trimmed_0_Pos_ST, Trimmed_0_Neg_ST, Trimmed_0_Pos_END, Trimmed_0_Neg_END, Trimmed_E_Pos_ST, Trimmed_E_Neg_ST, Trimmed_E_Pos_END, Trimmed_E_Neg_END, Centered_0_Pos_R, Centered_0_Neg_R, Centered_E_Pos_R, Centered_E_Neg_R, Right_E_Pos, Right_E_Neg, Left_E_Pos, Left_E_Neg]

def concatenate_elements(alligmnents):
    all_concats = []
    for allignment in alligmnents:
        all_concats.append([j for i in allignment for j in i])
    return all_concats

### Nucleotides
nAlignements = extract_different_alignments(nSeqs, nSeqP, nSeqN, nlen_example, ntargets)
nAlignements_test = [a[:num_test_seqs] for a in nAlignements]
nAlignements_test = concatenate_elements(nAlignements_test)
nTrimmed_0_Pos_ST_test, nTrimmed_0_Neg_ST_test, nTrimmed_0_Pos_END_test, nTrimmed_0_Neg_END_test, nTrimmed_E_Pos_ST_test, nTrimmed_E_Neg_ST_test, nTrimmed_E_Pos_END_test, nTrimmed_E_Neg_END_test, nCentered_0_Pos_R_test, nCentered_0_Neg_R_test, nCentered_E_Pos_R_test, nCentered_E_Neg_R_test, nRight_E_Pos_test, nRight_E_Neg_test, nLeft_E_Pos_test, nLeft_E_Neg_test = nAlignements_test

nAlignements_training = [a[num_test_seqs:] for a in nAlignements]
nAlignements_training = concatenate_elements(nAlignements_training)
nTrimmed_0_Pos_ST_training, nTrimmed_0_Neg_ST_training, nTrimmed_0_Pos_END_training, nTrimmed_0_Neg_END_training, nTrimmed_E_Pos_ST_training, nTrimmed_E_Neg_ST_training, nTrimmed_E_Pos_END_training, nTrimmed_E_Neg_END_training, nCentered_0_Pos_R_training, nCentered_0_Neg_R_training, nCentered_E_Pos_R_training, nCentered_E_Neg_R_training, nRight_E_Pos_training, nRight_E_Neg_training, nLeft_E_Pos_training, nLeft_E_Neg_training = nAlignements_training

### Codons
cAlignements = extract_different_alignments(cSeqs, cSeqP, cSeqN, clen_example, ctargets)
cAlignements_test = [a[:num_test_seqs] for a in cAlignements]
cAlignements_test = concatenate_elements(cAlignements_test)
cTrimmed_0_Pos_ST_test, cTrimmed_0_Neg_ST_test, cTrimmed_0_Pos_END_test, cTrimmed_0_Neg_END_test, cTrimmed_E_Pos_ST_test, cTrimmed_E_Neg_ST_test, cTrimmed_E_Pos_END_test, cTrimmed_E_Neg_END_test, cCentered_0_Pos_R_test, cCentered_0_Neg_R_test, cCentered_E_Pos_R_test, cCentered_E_Neg_R_test, cRight_E_Pos_test, cRight_E_Neg_test, cLeft_E_Pos_test, cLeft_E_Neg_test = cAlignements_test

cAlignements_training = [a[num_test_seqs:] for a in cAlignements]
cAlignements_training = concatenate_elements(cAlignements_training)
cTrimmed_0_Pos_ST_training, cTrimmed_0_Neg_ST_training, cTrimmed_0_Pos_END_training, cTrimmed_0_Neg_END_training, cTrimmed_E_Pos_ST_training, cTrimmed_E_Neg_ST_training, cTrimmed_E_Pos_END_training, cTrimmed_E_Neg_END_training, cCentered_0_Pos_R_training, cCentered_0_Neg_R_training, cCentered_E_Pos_R_training, cCentered_E_Neg_R_training, cRight_E_Pos_training, cRight_E_Neg_training, cLeft_E_Pos_training, cLeft_E_Neg_training = cAlignements_training

### STORE DATASET
# Nucleotides
nRES = {'Trimmed_0_ST': {'P_Train': nTrimmed_0_Pos_ST_training, 'N_Train': nTrimmed_0_Neg_ST_training, 'P_Test': nTrimmed_0_Pos_ST_test, 'N_Test': nTrimmed_0_Neg_ST_test},
        'Trimmed_0_END': {'P_Train': nTrimmed_0_Pos_END_training, 'N_Train': nTrimmed_0_Neg_END_training, 'P_Test': nTrimmed_0_Pos_END_test, 'N_Test': nTrimmed_0_Neg_END_test},
        'Trimmed_E_ST': {'P_Train': nTrimmed_E_Pos_ST_training, 'N_Train': nTrimmed_E_Neg_ST_training, 'P_Test': nTrimmed_E_Pos_ST_test, 'N_Test': nTrimmed_E_Neg_ST_test},
        'Trimmed_E_END': {'P_Train': nTrimmed_E_Pos_END_training, 'N_Train': nTrimmed_E_Neg_END_training, 'P_Test': nTrimmed_E_Pos_END_test, 'N_Test': nTrimmed_E_Neg_END_test},
        'Centered_0_R': {'P_Train': nCentered_0_Pos_R_training, 'N_Train': nCentered_0_Neg_R_training, 'P_Test': nCentered_0_Pos_R_test, 'N_Test': nCentered_0_Neg_R_test},
        'Centered_E_R': {'P_Train': nCentered_E_Pos_R_training, 'N_Train': nCentered_E_Neg_R_training, 'P_Test': nCentered_E_Pos_R_test, 'N_Test': nCentered_E_Neg_R_test},
        'Left': {'P_Train': nLeft_E_Pos_training, 'N_Train': nLeft_E_Neg_training, 'P_Test': nLeft_E_Pos_test, 'N_Test': nLeft_E_Neg_test},
        'Right': {'P_Train': nRight_E_Pos_training, 'N_Train': nRight_E_Neg_training, 'P_Test': nRight_E_Pos_test, 'N_Test': nRight_E_Neg_test}}
# Codons
cRES = {'Trimmed_0_ST': {'P_Train': cTrimmed_0_Pos_ST_training, 'N_Train': cTrimmed_0_Neg_ST_training, 'P_Test': cTrimmed_0_Pos_ST_test, 'N_Test': cTrimmed_0_Neg_ST_test},
        'Trimmed_0_END': {'P_Train': cTrimmed_0_Pos_END_training, 'N_Train': cTrimmed_0_Neg_END_training, 'P_Test': cTrimmed_0_Pos_END_test, 'N_Test': cTrimmed_0_Neg_END_test},
        'Trimmed_E_ST': {'P_Train': cTrimmed_E_Pos_ST_training, 'N_Train': cTrimmed_E_Neg_ST_training, 'P_Test': cTrimmed_E_Pos_ST_test, 'N_Test': cTrimmed_E_Neg_ST_test},
        'Trimmed_E_END': {'P_Train': cTrimmed_E_Pos_END_training, 'N_Train': cTrimmed_E_Neg_END_training, 'P_Test': cTrimmed_E_Pos_END_test, 'N_Test': cTrimmed_E_Neg_END_test},
        'Centered_0_R': {'P_Train': cCentered_0_Pos_R_training, 'N_Train': cCentered_0_Neg_R_training, 'P_Test': cCentered_0_Pos_R_test, 'N_Test': cCentered_0_Neg_R_test},
        'Centered_E_R': {'P_Train': cCentered_E_Pos_R_training, 'N_Train': cCentered_E_Neg_R_training, 'P_Test': cCentered_E_Pos_R_test, 'N_Test': cCentered_E_Neg_R_test},
        'Left': {'P_Train': cLeft_E_Pos_training, 'N_Train': cLeft_E_Neg_training, 'P_Test': cLeft_E_Pos_test, 'N_Test': cLeft_E_Neg_test},
        'Right': {'P_Train': cRight_E_Pos_training, 'N_Train': cRight_E_Neg_training, 'P_Test': cRight_E_Pos_test, 'N_Test': cRight_E_Neg_test}}

### SAVE DATASETs
if save_path:
    # create folders
    save_path += 'RibProf_Seqs/N' if save_path[-1] == '/' else '/RibProf_Seqs/N'
    save_path += 'L{0}m{1}M{2}'.format(nlen_example, nlen_min, nlen_max)
    save_path += '_CL{0}/'.format(clen_example)
    if os.path.exists(save_path): shutil.rmtree(save_path)
    for key in nRES:
        os.makedirs(save_path + 'Nucleotide/' + key + '/')
        for inKey in nRES[key]:
            np.save(save_path + 'Nucleotide/' + key + '/' + inKey + '.npy', nRES[key][inKey], allow_pickle=False)
    for key in cRES:
        codon_type = 'CodonRelaxed/' if relax else 'Codon/'
        os.makedirs(save_path + codon_type + key + '/')
        for inKey in cRES[key]:
            np.save(save_path + codon_type + key + '/' + inKey + '.npy', cRES[key][inKey], allow_pickle=False)
