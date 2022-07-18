import os, shutil, math, time
import tensorflow as tf
import numpy as np
import pandas as pd
from itertools import product
pd.options.display.max_rows = 10



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
relax 		:bool = False
nlen_example:int  = 30
nlen_min	:int  = 3
nlen_max	:int  = 15
save_path	:str  = 'Datasets'
c_emb_type  :str  = 'N'



#######################################################################################################################
### FUNCTIONS #########################################################################################################
#######################################################################################################################
def splitList(targs_nucleotides, val, len_min, len_max):
	accepted_lengths = range(len_min,len_max+1)
	w = np.where(targs_nucleotides==val)
	boolean_vector = -1*np.ones_like(targs_nucleotides)
	boolean_vector[w] = w
	from itertools import groupby
	lista = [list(group) for k, group in groupby(boolean_vector, lambda x: x >= 0) if k]
	return [i for i in lista if len(i) in accepted_lengths]

# ---------------------------------------------------------------------------------------------------------------------
def getCodonLabel(codons, embedding_type):
	if embedding_type.upper() not in ['A','N','C']: raise ValueError('Unknown >Codon Label> Embedding Type')
	clabels = [np.array([i[:, :-1].flatten() for i in j]) for j in codons]
	if embedding_type == 'C':
		# one hot 64 (codons) and 20 (aminoacids)
		keys64C = np.array([np.concatenate(i) for i in product(np.eye(4, dtype=int), repeat=3)], dtype=int)
		vals64C = np.eye(64, dtype=int)
		clabels = [np.concatenate([vals64C[np.all(keys64C == j, axis=1)] for j in i], axis=0) for i in clabels]
	if embedding_type == 'A': pass
	return clabels

# ---------------------------------------------------------------------------------------------------------------------
def getLeft(Seqs, SeqsP, SeqsN, len_example, targets):
	# indices - Nucleotides
	idxP = [[range(j[0] - len_example // 2, j[0] - len_example // 2 + len_example) for j in elem if j[0] - len_example // 2 >= 0 and j[0] - len_example // 2 + len_example <= len(targets[i])] for i, elem in enumerate(SeqsP)]
	idxN = [[range(j[0] - len_example // 2, j[0] - len_example // 2 + len_example) for j in elem if j[0] - len_example // 2 >= 0 and j[0] - len_example // 2 + len_example <= len(targets[i])] for i, elem in enumerate(SeqsN)]
	E_Pos = [np.array([Seqs[i][j] for j in elem], dtype=int) for i, elem in enumerate(idxP)]
	E_Neg = [np.array([Seqs[i][j] for j in elem], dtype=int) for i, elem in enumerate(idxN)]
	return E_Pos, E_Neg

# ---------------------------------------------------------------------------------------------------------------------
def getLeft_zeroPadding(Seqs, SeqsP, SeqsN, len_example):
	len_label = nSeqs[0].shape[1]
	Left_0_Pos = [np.array([np.concatenate([np.zeros((len_example // 2, len_label)), Seqs[i][j], np.zeros((len_example // 2 - len(j), len_label))], axis=0) for j in elem], dtype=int) for i, elem in enumerate(SeqsP)]
	Left_0_Neg = [np.array([np.concatenate([np.zeros((len_example // 2, len_label)), Seqs[i][j], np.zeros((len_example // 2 - len(j), len_label))], axis=0) for j in elem], dtype=int) for i, elem in enumerate(SeqsN)]
	return Left_0_Pos, Left_0_Neg

# ---------------------------------------------------------------------------------------------------------------------
def augment_data(Seqs, SeqsP, SeqsN, len_example, targets):
	def filter(args):
		return_list = [i for i in args if i.shape[0]!=0]
		if return_list: return return_list
		else: return [np.array([])]
	# feature dimension
	len_label = nSeqs[0].shape[1]
	### TRIMMERED 0 PADDING - ST: zeros AFTER sub-sequence, as Pietro did; END: zeros BEFORE sub-sequence
	Trimmed_0_Pos_ST  = [np.array([np.concatenate([Seqs[i][j], np.zeros((len_example - len(j), len_label))], axis=0) for j in elem], dtype=int) for i, elem in enumerate(SeqsP)]
	Trimmed_0_Neg_ST  = [np.array([np.concatenate([Seqs[i][j], np.zeros((len_example - len(j), len_label))], axis=0) for j in elem], dtype=int) for i, elem in enumerate(SeqsN)]

	Trimmed_0_Pos_END = [np.array([np.concatenate([np.zeros((len_example - len(j), len_label)), Seqs[i][j]], axis=0) for j in elem], dtype=int) for i, elem in enumerate(SeqsP)]
	Trimmed_0_Neg_END = [np.array([np.concatenate([np.zeros((len_example - len(j), len_label)), Seqs[i][j]], axis=0) for j in elem], dtype=int) for i, elem in enumerate(SeqsN)]
	### TRIMMERED E PADDING	- ST: zeros AFTER sub-sequence; END: zeros BEFORE sub-sequence
	Trimmed_E_idxP_ST  = [[range(j[0], j[0] + len_example) for j in elem if j[0] + len_example <= len(targets[i])] for i, elem in enumerate(SeqsP)]
	Trimmed_E_idxN_ST  = [[range(j[0], j[0] + len_example) for j in elem if j[0] + len_example <= len(targets[i])] for i, elem in enumerate(SeqsN)]
	Trimmed_E_idxP_END = [[range(j[-1] - len_example + 1, j[-1] + 1) for j in i if j[-1] - len_example + 1 >= 0] for i in SeqsP]
	Trimmed_E_idxN_END = [[range(j[-1] - len_example + 1, j[-1] + 1) for j in i if j[-1] - len_example + 1 >= 0] for i in SeqsN]
	Trimmed_E_Pos_ST   = [np.array([Seqs[i][j] for j in elem], dtype=int) for i, elem in enumerate(Trimmed_E_idxP_ST)]
	Trimmed_E_Neg_ST   = [np.array([Seqs[i][j] for j in elem], dtype=int) for i, elem in enumerate(Trimmed_E_idxN_ST)]
	Trimmed_E_Pos_END  = [np.array([Seqs[i][j] for j in elem], dtype=int) for i, elem in enumerate(Trimmed_E_idxP_END)]
	Trimmed_E_Neg_END  = [np.array([Seqs[i][j] for j in elem], dtype=int) for i, elem in enumerate(Trimmed_E_idxN_END)]
	### CENTERED 0 PADDING
	Centered_0_Pos_R = [np.array([np.concatenate([np.zeros((math.floor((len_example - len(j)) / 2), len_label)), Seqs[i][j], np.zeros((math.ceil((len_example - len(j)) / 2), len_label))], axis=0) for j in elem], dtype=int) for i, elem in enumerate(SeqsP)]
	Centered_0_Neg_R = [np.array([np.concatenate([np.zeros((math.floor((len_example - len(j)) / 2), len_label)), Seqs[i][j], np.zeros((math.ceil((len_example - len(j)) / 2), len_label))], axis=0) for j in elem], dtype=int) for i, elem in enumerate(SeqsN)]
	### CENTERED E PADDING
	Centered_E_idxP_R = [[range(j[0] - (len_example - len(j)) // 2, j[0] - (len_example - len(j)) // 2 + len_example) for j in elem if j[0] - (len_example - len(j)) // 2 >= 0 and j[0] - (len_example - len(j)) // 2 + len_example <= len(targets[i])] for i, elem in enumerate(SeqsP)]
	Centered_E_idxN_R = [[range(j[0] - (len_example - len(j)) // 2, j[0] - (len_example - len(j)) // 2 + len_example) for j in elem if j[0] - (len_example - len(j)) // 2 >= 0 and j[0] - (len_example - len(j)) // 2 + len_example <= len(targets[i])] for i, elem in enumerate(SeqsN)]
	Centered_E_Pos_R  = [np.array([Seqs[i][j] for j in elem], dtype=int) for i, elem in enumerate(Centered_E_idxP_R)]
	Centered_E_Neg_R  = [np.array([Seqs[i][j] for j in elem], dtype=int) for i, elem in enumerate(Centered_E_idxN_R)]
	### RIGHT ALIGNED
	Right_E_idxP = [[range(j[-1] - len_example // 2 + (len_example + 1) % 2, j[-1] - len_example // 2 + (len_example + 1) % 2 + len_example) for j in elem if j[-1] - len_example // 2 + (len_example + 1) % 2 >= 0 and j[-1] - len_example // 2 + (len_example + 1) % 2 + len_example <= len(targets[i])] for i, elem in enumerate(SeqsP)]
	Right_E_idxN = [[range(j[-1] - len_example // 2 + (len_example + 1) % 2, j[-1] - len_example // 2 + (len_example + 1) % 2 + len_example) for j in elem if j[-1] - len_example // 2 + (len_example + 1) % 2 >= 0 and j[-1] - len_example // 2 + (len_example + 1) % 2 + len_example <= len(targets[i])] for i, elem in enumerate(SeqsN)]
	Right_E_Pos  = [np.array([Seqs[i][j] for j in elem], dtype=int) for i, elem in enumerate(Right_E_idxP)]
	Right_E_Neg  = [np.array([Seqs[i][j] for j in elem], dtype=int) for i, elem in enumerate(Right_E_idxN)]
	### MERGE ALL THE AUGMENTED DATASET
	merged_Pos = [np.concatenate(filter([Trimmed_0_Pos_ST[i], Trimmed_0_Pos_END[i], Trimmed_E_Pos_ST[i], Trimmed_E_Pos_END[i], Centered_0_Pos_R[i], Centered_E_Pos_R[i], Right_E_Pos[i]]), axis=0) for i in range(len(Seqs))]
	merged_Neg = [np.concatenate(filter([Trimmed_0_Neg_ST[i], Trimmed_0_Neg_END[i], Trimmed_E_Neg_ST[i], Trimmed_E_Neg_END[i], Centered_0_Neg_R[i], Centered_E_Neg_R[i], Right_E_Neg[i]]), axis=0) for i in range(len(Seqs))]
	merged_Pos = [np.unique(i, axis=0) if i.shape[0]!=0 else i for i in merged_Pos]
	merged_Neg = [np.unique(i, axis=0) if i.shape[0]!=0 else i for i in merged_Neg]
	return merged_Pos, merged_Neg



#######################################################################################################################
#### SCRIPT ###########################################################################################################
#######################################################################################################################

### Loading the DataFrame from Excel file
df = pd.read_excel('AllSequences49.xlsx', sheet_name=None, header=None, names=['nL', 't'], dtype={'nL':str, 't':int})

### NUCLEOTIDES G:1,0,0,0  A:0,1,0,0  C:0,0,1,0  T:0,0,0,1
nucleotides = dict(zip('GACT',np.eye(4,dtype=int)))

### Matrices: List of matrices encoded-nucleotide-based with dimension (len(nucleotide_sequence), [4+1]) [1hot + target]
matrices = list()
for i in df:
	temp = np.zeros((len(df[i]),5),dtype=int)
	temp[:,:-1] = [nucleotides[j] for j in df[i]['nL']]
	temp[:,-1] = df[i]['t']
	matrices.append(temp)

# list of 49 elements, each of which of dimension (number of codons,3) ->[last columns of i for i in Elements]
nlabels = [i[:,:-1] for i in matrices]
ntargets = [i[:,-1] for i in matrices]

### SEQUENCES AS UNIQUE MATRICES
nSeqs = [np.concatenate([nlabels[i],np.array(ntargets[i],ndmin=2).transpose()],axis=1) for i in range(len(nlabels))]

### NEGATIVE AND POSITIVE SUB-SEQUENCES
nSeqP = [splitList(i, 1, len_min=nlen_min, len_max=nlen_max) for i in ntargets]
nSeqN = [splitList(i,-1, len_min=nlen_min, len_max=nlen_max) for i in ntargets]

########### DATASETS EXTRACTION
Left_E_Pos, Left_E_Neg = getLeft(nSeqs, nSeqP, nSeqN, nlen_example, ntargets)
#Aug_Pos, Aug_Neg = augment_data(nSeqs, nSeqP, nSeqN, nlen_example, ntargets)
Left_0_Pos, Left_0_Neg = getLeft_zeroPadding(nSeqs, nSeqP, nSeqN, nlen_example)