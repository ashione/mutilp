#!/usr/bin/python
import networkx as nx
from tree import  GraphGenerator
from tree import *
import os,sys
import numpy as np

def calData(filename='./pretrain/data.txt'):
    raw_data = np.loadtxt(filename)
    raw_shape = raw_data.shape
    zero_one_index = raw_data!=0

    zero_one_matrix = np.zeros(raw_shape)
    zero_one_matrix[zero_one_index] = 1
    firm_matrix = np.matrix(zero_one_matrix) * np.matrix(zero_one_matrix.transpose())
    bank_matrix = np.matrix(zero_one_matrix.transpose()) * np.matrix(zero_one_matrix)
    firm_dia_index = np.diag_indices(firm_matrix.shape[0])
    firm_matrix[firm_dia_index] = 0
    bank_dia_index = np.diag_indices(bank_matrix.shape[0])
    bank_matrix[bank_dia_index] = 0
    return firm_matrix,bank_matrix
    #bank_matrix = np.zeros((raw_shape[1],raw_shape[1]))
    #firm_matrix = np.zeros((raw_shape[0],raw_shape[0]))
def coff():
    firm_matrix,bank_matrix = calData()
    G = nx.from_numpy_matrix(firm_matrix)
    #print np.loadtxt(args.df)
    pr = nx.pagerank(G,alpha=0.85)
    #print pr
    print sorted(pr.iteritems(),key=lambda x : x[1] ,reverse = True)
    #closeness = nx.closeness_centrality(G)
    #print closeness
    #print sorted(pr.iteritems(),key= lambda x : x[1],reverse = True)
    #betweenness = nx.betweenness_centrality(G)
    #print sorted(betweenness.iteritems(),key= lambda x : x[1],reverse = True)

if __name__ == '__main__':
    coff()
