import pandas as pd
import random 
import os 
import matplotlib.pyplot as plt
from statistics import *

path_data="Dati/Sequences/"

sequence_paths = os.listdir(path_data)


listafreqA=list()
listafreqT=list()
listafreqG=list()
listafreqC=list()
pA=0
pT=0
pG=0
pC=0

for i in range(10000):
    nucl1=list()
    for sp in sequence_paths:
        
        originale=pd.read_csv(path_data+sp, names=['nucleotidi1', 'label1'])
        nucleotidi=list()
        label=list()
        for row in originale.itertuples(index='nucleotidi1',name='label1'): 
                    nucleotidi.append(row[1])
                    label.append(row[2])
        
        label_blocchi=list() 
        
        stato='a' 
        for i in range(len(label)):   
            if stato=='a' and label[i]!=1: # use -1 for the fast subsequences
                label_blocchi.append(label[i])
                
            elif stato=='a' and label[i]==1:
                        lista = list()
                        lista.append(label[i])
                        stato='b'    
            elif stato=='b' and label[i-1]==label[i]:
                        lista.append(label[i])
                        
            elif stato=='b' and label[i-1]!=label[i]:
                        stato='a'
                        label_blocchi.append(lista)
                        label_blocchi.append(label[i])
        if stato=='b':
            label_blocchi.append(lista)	

        
        random.shuffle(label_blocchi)

        l_shuffle=list() 
        
        for i in range(len(label_blocchi)):
            if type(label_blocchi[i]) is list:
                for j in range(len(label_blocchi[i])):
                    l_shuffle.append(label_blocchi[i][j])
            else:   
                l_shuffle.append(label_blocchi[i])
                
        df=pd.DataFrame({'nucleotidi': nucleotidi, 'label': l_shuffle}) 
        
        for row in df.itertuples(index='nucleotidi',name='label'): 
                    if row[2]==1:
                        nucl1.append(row[1])

    freqA=nucl1.count('A')/len(nucl1) 
    freqT=nucl1.count('T')/len(nucl1)
    freqG=nucl1.count('G')/len(nucl1)
    freqC=nucl1.count('C')/len(nucl1)
    
    
    listafreqA.append(freqA) 
    listafreqT.append(freqT)
    listafreqG.append(freqG)
    listafreqC.append(freqC)

a=mean(listafreqA)
t=mean(listafreqT)
c=mean(listafreqC)
g=mean(listafreqG)


sub_sequences_list=list()

w=0

for sp in sequence_paths:
    w=w+1
    sequence= pd.read_csv(path_data+sp, names=['nucleotidi', 'label']) 
    stato='a' 

    for ind in range(len(sequence)):
        if stato=='a' and list(sequence.values[ind])[1]==1:
            lista = list()
            lista.append(list(sequence.values[ind])[0])
            stato='b'    
        elif  stato=='b' and list(sequence.values[ind-1])[1]==list(sequence.values[ind])[1]:
            lista.append(list(sequence.values[ind])[0])
                
        elif stato=='b' and list(sequence.values[ind-1])[1]!=list(sequence.values[ind])[1]:
            stato='a'
            sub_sequences_list.append(lista)
                
        sub_sequences_list.append(lista)
        
l=0
        
for sub_sequence in sub_sequences_list:
    numA=0
    numT=0
    numC=0
    numG=0
    if len(sub_sequence)>4:
        l=l+1
        for i in sub_sequence:
            if i=='A':
                numA=numA+1
            if i=='T':
                numT=numT+1
            if i=='C':
                numC=numC+1
            if i=='G':
                numG=numG+1
        freqsottseqA= numA/len(sub_sequence)
        freqsottseqT=numT/len(sub_sequence)
        freqsottseqC=numC/len(sub_sequence)
        freqsottseqG=numG/len(sub_sequence)
                
    
        if freqsottseqA>a:
            for freq in listafreqA:
                if freq>freqsottseqA:
                    pA+=1
        else:
            for freq in listafreqA:
                if freq<freqsottseqA:
                    pA+=1
                
                
                            
        if freqsottseqT>t:
            for freq in listafreqT:
                if freq>freqsottseqT:
                    pT+=1
        else:
            for freq in listafreqT:
                if freq<freqsottseqT:
                    pT+=1
                
                            
                            
        if freqsottseqC>c:
            for freq in listafreqC:
                if freq>freqsottseqC:
                    pC+=1
        else:
            for freq in listafreqC:
                if freq<freqsottseqC:
                    pC+=1
                
                
                            
        if freqsottseqG>g:
            for freq in listafreqG:
                if freq>freqsottseqG:
                    pG+=1
        else:
            for freq in listafreqG:
                if freq<freqsottseqG:
                    pG+=1
                
        print('p value A:', pA/10000, sub_sequence, l)
        print('p value T:', pT/10000, sub_sequence, l)
        print('p value G:', pG/10000, sub_sequence, l)
        print('p value C:', pC/10000, sub_sequence, l)



plt.hist(listafreqA)
plt.xlabel('Frequence of A, slow sequence');
plt.show()
   
plt.hist(listafreqT)
plt.xlabel('Frequence of T, slow sequence');
plt.show()

plt.hist(listafreqG)
plt.xlabel('Frequence of G, slow sequence');
plt.show()

plt.hist(listafreqC)
plt.xlabel('Frequence of C, slow sequence');
plt.show()







