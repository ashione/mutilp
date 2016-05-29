#!/usr/bin/python
# -*- coding: utf-8 -*-
import networkx as nx
import os,sys
import numpy as np
import matplotlib.pyplot as plt

firmNameList = np.loadtxt('firm.list',dtype=np.str)
bankNameList = np.loadtxt('bank.list',dtype=np.str)

def calData(filename='./pretrain/data.txt'):
    raw_data = np.loadtxt(filename)
    return raw_data

def calMatrix(raw_data,remove_list=None):
    firm_matrix = cal_remove_firm(raw_data,remove_list)
    bank_matrix = cal_remove_bank(raw_data,remove_list)
    return firm_matrix,bank_matrix
    #bank_matrix = np.zeros((raw_shape[1],raw_shape[1]))
    #firm_matrix = np.zeros((raw_shape[0],raw_shape[0]))

def cal_remove_firm(raw_data,remove_list=None):
    raw_shape = raw_data.shape
    zero_one_index = raw_data!=0

    zero_one_matrix = np.zeros(raw_shape)
    zero_one_matrix[zero_one_index] = 1

    if remove_list is not None :
        mask = np.ones(zero_one_matrix.shape[0],dtype=bool)
        mask[remove_list] = False
        zero_one_matrix = zero_one_matrix[mask,:]

    firm_matrix = np.matrix(zero_one_matrix) * np.matrix(zero_one_matrix.transpose())
    firm_dia_index = np.diag_indices(firm_matrix.shape[0])
    firm_matrix[firm_dia_index] = 0
    return firm_matrix

def cal_remove_bank(raw_data,remove_list=None):
    raw_shape = raw_data.shape
    zero_one_index = raw_data!=0

    zero_one_matrix = np.zeros(raw_shape)
    zero_one_matrix[zero_one_index] = 1
    if remove_list is not None :
        mask = np.ones(zero_one_matrix.shape[1],dtype=bool)
        mask[remove_list] = False
        zero_one_matrix = zero_one_matrix[:,mask]

    bank_matrix = np.matrix(zero_one_matrix.transpose()) * np.matrix(zero_one_matrix)
    bank_dia_index = np.diag_indices(bank_matrix.shape[0])
    bank_matrix[bank_dia_index] = 0
    return bank_matrix

def remove_nodes(G,nodes):
    G_new =G.copy()
    for node in nodes:
        G_new.remove_node(node)
    return G_new

def max_strongly_connected_components(G):
    return max(nx.strongly_connected_components(G),key=len)

def floy_washall_shortest_distance(G):
    return nx.floyd_warshall_numpy(G)

def influence(raw_data,sorted_pr,func=cal_remove_firm,save_img=None,savepr=None):
    to_removed_list = []
    largest_components_arr = []
    E_arr = []
    print save_img

    x_len = len(sorted_pr)-1
    #x_len = 2
    random_pr = np.copy(sorted_pr)

    for i in range(x_len):
        to_removed_list.append(sorted_pr[i][0])
        fnc_matrix = func(raw_data,to_removed_list)
        G_new = nx.from_numpy_matrix(fnc_matrix)
        #G_new = #remove_nodes(G,to_removed_list)
        largest_components = max_strongly_connected_components(G_new)
        distance_sum = np.power(np.sum(floy_washall_shortest_distance(G_new)),-1.0)
        n_node = len(G_new.nodes())
        pct_components = len(largest_components)*1.0/raw_data.shape[0]
        E = distance_sum/(n_node*(n_node-1))
        print 'ith :',i,len(largest_components)*1.0/raw_data.shape[0],distance_sum/(n_node*(n_node-1))
        largest_components_arr.append(pct_components)
        E_arr.append(E)
    x = np.array(range(x_len))
    figure = plt.figure()
    figure.suptitle(save_img)
    plt.subplot(121)
    plt.xlabel('removed nodes')
    plt.ylabel('H')
    plt.plot(x,largest_components_arr,'bo')
    plt.subplot(122)
    plt.xlabel('removed nodes')
    plt.ylabel('E')
    plt.plot(x,E_arr,'ro')
    #plt.show()
    figure.savefig('inf_dir/'+save_img+'_'+func.func_name+'.png')

    # if call function is belonged to firm list, then useing firm list map to sorted results
    if func == cal_remove_firm :
        mapNameList = firmNameList
    else:
        mapNameList = bankNameList

    np.savetxt('{dirc}/{savetxt}.txt'.format(dirc='inf_dir',savetxt=save_img+'_'+func.func_name),np.transpose([largest_components_arr,E_arr]))
    #np.savetxt('{dirc}/{savetxt}.txt'.format(dirc='inf_dir',savetxt='sorted_'+save_img+'_'+func.func_name),np.hstack((order_name,sorted_pr)),fmt='%s %d %.10f')
    save_f = open('{dirc}/{savetxt}.csv'.format(dirc='inf_dir',savetxt='sorted_'+save_img+'_'+func.func_name),'w')

    order_name = np.empty_like(mapNameList)
    for i in range(0,len(sorted_pr)):
        order_name[i] = mapNameList[sorted_pr[i][0]]
        save_f.write('%s,%d,%.10f\n' %(order_name[i],sorted_pr[i][0],sorted_pr[i][1]))
    save_f.close()
    order_name.shape = (order_name.shape[0],1)
    #print np.hstack((order_name,sorted_pr))


def coff():
    raw_data = calData()
    firm_matrix,bank_matrix = calMatrix(raw_data)
    G = nx.from_numpy_matrix(firm_matrix)
    #pagerank
    pr = nx.pagerank(G,alpha=0.85)
    sorted_pr = sorted(pr.iteritems(),key=lambda x : x[1] ,reverse = True)
    influence(raw_data,sorted_pr,cal_remove_firm,'pagerank')

    closeness = nx.closeness_centrality(G)
    sorted_closeness = sorted(closeness.iteritems(),key= lambda x : x[1],reverse = True)
    #print sorted_closeness
    influence(raw_data,sorted_closeness,cal_remove_firm,'closeness_centrality')

    betweenness = nx.betweenness_centrality(G)
    sorted_betweenness = sorted(betweenness.iteritems(),key= lambda x : x[1],reverse = True)
    #print sorted_betweenness
    influence(raw_data,sorted_betweenness,cal_remove_firm,'betweenness_centrality')

    G = nx.from_numpy_matrix(bank_matrix)
    #pagerank
    pr = nx.pagerank(G,alpha=0.85)
    sorted_pr = sorted(pr.iteritems(),key=lambda x : x[1] ,reverse = True)
    influence(raw_data,sorted_pr,cal_remove_bank,'pagerank')

    closeness = nx.closeness_centrality(G)
    sorted_closeness = sorted(closeness.iteritems(),key= lambda x : x[1],reverse = True)
    #print sorted_closeness
    influence(raw_data,sorted_closeness,cal_remove_bank,'closeness_centrality')

    betweenness = nx.betweenness_centrality(G)
    sorted_betweenness = sorted(betweenness.iteritems(),key= lambda x : x[1],reverse = True)
    #print sorted_betweenness
    influence(raw_data,sorted_betweenness,cal_remove_bank,'betweenness_centrality')
if __name__ == '__main__':
    coff()
