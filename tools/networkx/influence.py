#!/usr/bin/python
# -*- coding: utf-8 -*-
import networkx as nx
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.patches as mpatches

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
    #bank_dia_index = np.diag_indices(bank_matrix.shape[0])
    #bank_matrix[bank_dia_index] = 0
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

def get_cc1_cc2(G):
    nodes_n = G.number_of_nodes()
    return np.sum(nx.clustering(G).values())/nodes_n,np.sum(nx.square_clustering(G).values())/nodes_n

def influence(raw_data,sorted_pr,func=cal_remove_firm,save_img=None,savepr=None):
    to_removed_list = []
    largest_components_arr = []
    E_arr = []
#    cc1s = []
#    cc2s = []
    print save_img

    x_len = len(sorted_pr)-1
    #x_len = 2
    random_pr = np.copy(sorted_pr)
    random_to_removed_list = []
    random_largest_components_arr = []
    random_E_arr = []
#    random_cc1s = []
#    random_cc2s = []


    for i in range(x_len):
        # remove by sorted
        to_removed_list.append(sorted_pr[i][0])
        fnc_matrix = func(raw_data,to_removed_list)
        G_new = nx.from_numpy_matrix(fnc_matrix)
        #G_new = #remove_nodes(G,to_removed_list)
        largest_components = max_strongly_connected_components(G_new)
        distance_all = floy_washall_shortest_distance(G_new)
        distance_sum = np.sum(np.power(distance_all[distance_all!=0],-1.0))
        n_node = len(G_new.nodes())
        pct_components = len(largest_components)*1.0/raw_data.shape[0]
        #E = distance_sum/(n_node*(n_node-1))
        E = distance_sum/(x_len*(x_len+1))
        print 'ith :',sorted_pr[i][0],len(largest_components)*1.0/raw_data.shape[0],distance_sum/(n_node*(n_node-1)),E
        largest_components_arr.append(pct_components)
        E_arr.append(E)
#        cc1,cc2 = get_cc1_cc2(G_new)
#        print 'cc1cc2',cc1,cc2
#        cc1s.append(cc1)
#        cc2s.append(cc2)

        # remove random
        random_index = random.randint(0,len(random_pr)-1)
        print random_index
        random_to_removed_list.append(random_pr[random_index][0])
        random_pr = np.delete(random_pr,[random_index],axis=0)
        print 'random_pr len',len(random_pr)
        random_fnc_matrix = func(raw_data,random_to_removed_list)
        random_G_new = nx.from_numpy_matrix(random_fnc_matrix)
        #G_new = #remove_nodes(G,to_removed_list)
        random_largest_components = max_strongly_connected_components(random_G_new)
        random_distance_all = floy_washall_shortest_distance(random_G_new)
        random_distance_sum = np.sum(np.power(random_distance_all[random_distance_all!=0],-1.0))
        n_node = len(random_G_new.nodes())
        random_pct_components = len(random_largest_components)*1.0/raw_data.shape[0]
        #E = random_distance_sum/(n_node*(n_node-1))
        E = random_distance_sum/(x_len * (x_len+1))
        print 'random ith :',random_index,len(random_largest_components)*1.0/raw_data.shape[0],random_distance_sum/(n_node*(n_node-1)),E
        random_largest_components_arr.append(random_pct_components)
        random_E_arr.append(E)

#        random_cc1,random_cc2 = get_cc1_cc2(random_G_new)
#        print 'random cc1cc2',random_cc1,random_cc2
#        random_cc1s.append(random_cc1)
#        random_cc2s.append(random_cc2)
#
    x = np.array(range(x_len),dtype=np.float)/(x_len)
    figure = plt.figure()
    figure.suptitle(save_img)
    plt.subplot(121)
    plt.xlabel('removed nodes')
    plt.ylabel('H')
    lc1 = plt.plot(x,largest_components_arr,'b*',label='line1')
    lc2 = plt.plot(x,random_largest_components_arr,'ro',label='line2')
    red_patch = mpatches.Patch(color='red',label = 'deliberatly')
    blue_patch = mpatches.Patch(color='blue',label = 'randomly')
    #plt.legend([lc1,lc2],[blue_patch,red_patch])
    plt.subplots_adjust(wspace=.5)
    plt.subplot(122)
    plt.xlabel('removed nodes')
    plt.ylabel('E')
    ec1 = plt.plot(x,E_arr,'b*')
    ec2 = plt.plot(x,random_E_arr,'ro')
    #plt.legend([ec1,ec2],[blue_patch,red_patch])
    #plt.show()

    #plt.subplot(223)
    #plt.xlabel('removed nodes')
    #plt.ylabel('CC1')
    #clc1 = plt.plot(x,cc1s,'b*',label='cc1')
    #clc2 = plt.plot(x,random_cc1s,'ro',label='random_cc1')
    ##red_patch = mpatches.Patch(color='red',label = 'deliberatly')
    ##blue_patch = mpatches.Patch(color='blue',label = 'randomly')
    ##plt.legend([lc1,lc2],[blue_patch,red_patch])
    #plt.subplot(224)
    #plt.xlabel('removed nodes')
    #plt.ylabel('CC2')
    #scc1 = plt.plot(x,cc2s,'b*')
    #scc2 = plt.plot(x,random_cc2s,'ro')

    figure.savefig('inf_dir/'+save_img+'_'+func.func_name+'.png')

    # if call function is belonged to firm list, then useing firm list map to sorted results
    if func == cal_remove_firm :
        mapNameList = firmNameList
    else:
        mapNameList = bankNameList

    np.savetxt('{dirc}/{savetxt}.txt'.format(dirc='inf_dir',savetxt=save_img+'_'+func.func_name),np.transpose([largest_components_arr,E_arr]))
    #np.savetxt('{dirc}/{savetxt}.txt'.format(dirc='inf_dir',savetxt='cc1_cc1_'+save_img+'_'+func.func_name),np.column_stack((cc1s,random_cc1s,cc2s,random_cc2s)),fmt='%.6f %.6f %.6f %.6f')
    #np.savetxt('{dirc}/{savetxt}.txt'.format(dirc='inf_dir',savetxt='sorted_'+save_img+'_'+func.func_name),np.hstack((order_name,sorted_pr)),fmt='%s %d %.10f')
    save_f = open('{dirc}/{savetxt}.csv'.format(dirc='inf_dir',savetxt='sorted_'+save_img+'_'+func.func_name),'w')

    order_name = np.empty_like(mapNameList)
    for i in range(0,len(sorted_pr)):
        order_name[i] = mapNameList[sorted_pr[i][0]]
        save_f.write('%s,%d,%.10f\n' %(order_name[i],sorted_pr[i][0]+1,sorted_pr[i][1]))
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

    closeness = nx.closeness_centrality(G,normalized=True)
    sorted_closeness = sorted(closeness.iteritems(),key= lambda x : x[1],reverse = True)
    #print sorted_closeness
    influence(raw_data,sorted_closeness,cal_remove_firm,'closeness_centrality')

    betweenness = nx.betweenness_centrality(G,normalized = True)
    sorted_betweenness = sorted(betweenness.iteritems(),key= lambda x : x[1],reverse = True)
    #print sorted_betweenness
    influence(raw_data,sorted_betweenness,cal_remove_firm,'betweenness_centrality')

    G = nx.from_numpy_matrix(bank_matrix)
    #pagerank
    pr = nx.pagerank(G,alpha=0.85)
    sorted_pr = sorted(pr.iteritems(),key=lambda x : x[1] ,reverse = True)
    influence(raw_data,sorted_pr,cal_remove_bank,'pagerank')

    closeness = nx.closeness_centrality(G,normalized = True)
    sorted_closeness = sorted(closeness.iteritems(),key= lambda x : x[1],reverse = True)
    #print sorted_closeness
    influence(raw_data,sorted_closeness,cal_remove_bank,'closeness_centrality')

    betweenness = nx.betweenness_centrality(G, normalized = True)
    sorted_betweenness = sorted(betweenness.iteritems(),key= lambda x : x[1],reverse = True)
    #print sorted_betweenness
    influence(raw_data,sorted_betweenness,cal_remove_bank,'betweenness_centrality')

def mst(alpha=0.00):
    raw_data = calData()
    firm_matrix,bank_matrix = calMatrix(raw_data)

    firm_distance = np.loadtxt('./pretrain/distfirm_%.2f.txt' %(alpha))
    G = nx.from_numpy_matrix(firm_distance)
    G_mst = nx.kruskal_mst(G)
    sorted_degree = sorted(G_mst.degree_iter(),key = lambda x : x[1], reverse = True)

    G_matrix = nx.from_numpy_matrix(firm_matrix)

    influence(raw_data,sorted_degree,cal_remove_firm,'mst_%.2f' %(alpha))


    bank_distance = np.loadtxt('./pretrain/distbank_%.2f.txt' %(alpha))
    G = nx.from_numpy_matrix(bank_distance)
    G_mst = nx.kruskal_mst(G)
    sorted_degree = sorted(G_mst.degree_iter(),key = lambda x : x[1], reverse = True)

    G_matrix = nx.from_numpy_matrix(bank_matrix)

    influence(raw_data,sorted_degree,cal_remove_bank,'mst_%.2f' %(alpha))

def pariticipation(data,fn):
    G = nx.from_numpy_matrix(data)
    s = np.array([0.0]*data.shape[0])
    Y = np.array([0.0]*data.shape[0])
    for i in range(0,data.shape[0]):
        for item in G.edge[i].keys():
            s[i]+=G.edge[i][item]['weight']
        for item in G.edge[i].keys():
            Y[i]+=(G.edge[i][item]['weight']/s[i]) ** 2.0

    np.savetxt(fn,np.column_stack((s,Y)),fmt='%d %.5f')


def neigh():
    raw_data = calData()
    firm_matrix,bank_matrix = calMatrix(raw_data)
    pariticipation(bank_matrix,'./inf_dir/bank_weight.txt')
    pariticipation(firm_matrix,'./inf_dir/firm_weight.txt')

if __name__ == '__main__':
    #coff()
    #for alpha in [0.00,0.50,1.0]:
    #    mst(alpha)
    neigh()
