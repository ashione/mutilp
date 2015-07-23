# author : Nelson Zuo
# date   : 2015-7-24
import networkx as nx
import numpy as np
#import os
import matplotlib.pyplot as plt
import glog
import xlrd
import random
import matplotlib.path as mpath
import matplotlib.patches as mpatches
#coding=utf-8

def ConvertMatrixToGraph(filename):
    return nx.from_numpy_matrix(np.loadtxt(filename))

def ConvertTripleArrayToGraph(filename):
    data = np.asarray(np.loadtxt(filename))
    G = nx.Graph()
    map(lambda x : G.add_edge(np.int(x[0]),np.int(x[1]),weight=x[2]),data)
    return G

def ConvertXlsxToArray(filename):
    glog.info('Now reading xlsx file')
    compydata = xlrd.open_workbook(filename)
    glog.info(u'xlsx file , %d sheets , and their names are %s' %(compydata.nsheets,compydata.sheet_names()))
    first_sheet = compydata.sheet_by_index(0)
    nodesSeq  =  np.asarray(map(lambda x : x.value,first_sheet.col_slice(start_rowx=2,end_rowx=first_sheet.nrows,colx=0)),dtype=np.int)
    nodesType =  map(lambda x : x.value,first_sheet.col_slice(start_rowx=2,end_rowx=first_sheet.nrows,colx=2))
    glog.info('Reading nodes label orders and types : %d' %(nodesSeq.size))
    return { s : t for s,t in zip(nodesSeq,nodesType)}


def curvePosPoly(pos,color):
    Path = mpath.Path
    patch_data = []
    for k,ipos in pos.items():
        patch_data.append((Path.CURVE4,ipos))
    patch_data[0] = ((Path.MOVETO,patch_data[0][1]))
    codes,verts = zip(*patch_data)
    #path = mpath.Path(verts,codes)
    #print 'path : ',path.to_polygons(),verts
    #patch= mpatches.PathPatch(path,facecolor=color,alpha=0.5)
    patch= mpatches.Polygon(verts,facecolor=color,alpha=0.5)
    #patch= mpatches.Ellipse((xindex,yindex),height=height,width=width,facecolor=color,alpha=0.3,linestyle='dashed')
    return patch
def curvePos(pos,color):
    Path = mpath.Path
    patch_data = []
    for k,ipos in pos.items():
        patch_data.append((Path.CURVE4,ipos))
    patch_data[0] = ((Path.MOVETO,patch_data[0][1]))
    #patch_data[-1] = ((Path.CLOSEPOLY,patch_data[-1][1]))
    #patch_data.append((Path.CLOSEPOLY,patch_data[0][1]))
    #patch_data.append((Path.CLOSEPOLY,ipos))
    codes,verts = zip(*patch_data)
    #print verts
    path = mpath.Path(verts,codes)
    print 'path : ',path.to_polygons(),verts
    #patch= mpatches.PathPatch(path,facecolor=color,alpha=0.5)
    #patch= mpatches.Polygon(verts,facecolor=color,alpha=0.5)
    verts = np.asarray(verts)
    fcenter = lambda x : (np.max(x)+np.min(x))/2
    xindex,yindex = fcenter(verts[:,0]),fcenter(verts[:,1])
    width = np.max(verts[:,0]) - np.min(verts[:,0])+ np.median(verts[:,0])*0.1
    height = np.max(verts[:,1]) - np.min(verts[:,1]) + np.median(verts[:,1])*0.1
    patch= mpatches.Ellipse((xindex,yindex),height=height,width=width,facecolor=color,alpha=0.3,linestyle='dashed')
    return patch

def drawByTripleArry(G,nodesDict,colorDict):
    G = nx.kruskal_mst(G)
    pos = nx.spring_layout(G)

    #pos = nx.circular_layout(G)
    #pos = nx.random_layout(G)
    #pos = nx.spectral_layout(G)

    #print pos
    np.savetxt('pos.txt',pos.values())
    #print len(G.nodes()),len(nodesDict.keys())

    fig,ax = plt.subplots()
    print type(ax)
    #patch = curvePos(pos)
    #ax.add_patch(patch)

    # different nodes cover different color
    for node in G.nodes():
        G.node[node]['category'] = colorDict[nodesDict[node+1]]
    submap = {}

    # counting each color subgraph map set
    for k,v in G.node.items():
        if not submap.has_key(v['category']) :
            submap[v['category']] = set()
        else :
            submap[v['category']].add(k)

    # add connected_component_subgraphs in each subgraph
    for color in colorDict.values():
        spos = { x : pos[x] for x in submap[color] }
        #print spos
        if len(spos.keys()) <=2 :
            continue
        subG =  G.subgraph(spos.keys())
        for connt in nx.connected_component_subgraphs(subG):
            sgpos = { x : pos[x] for x in connt.nodes() }
            if len(sgpos.keys())>2:
                patch = curvePos(sgpos,color)
                ax.add_patch(patch)

    nx.draw_networkx(G,pos=pos,node_color= [ G.node[node]['category'] for node in G ],
                     alpha=0.6,node_size=200,labels=None,font_size=8)
    plt.xlim(-0.02,1.02)
    plt.ylim(-0.02,1.02)
    plt.show()

if __name__ == '__main__':
    #gdata = np.loadtxt('./firmmst.txt')
    #drawByTripleArry(ConvertTripleArrayToGraph('./firmmst.txt'))
    xlsxDict =  ConvertXlsxToArray('./companyPty.xlsx')
    colorTypes = set(xlsxDict.values())
    colorDict = { x : ('#%06X' % random.randint(0,0xFFFFFF)) for x  in colorTypes }
    #drawByTripleArry(ConvertTripleArrayToGraph('./firmmst.txt'),xlsxDict,colorDict)
    G = ConvertMatrixToGraph('./firmMstMatrix.txt')

    #for x,y in G.edges():
        #print G.edge[x][y]
        #G.edge[x][y]['weight'] = 0.5
        #G.edge[y][x]['weight'] = 0.5
    drawByTripleArry(G,xlsxDict,colorDict)
    print nx.algorithms.topological_sort(G)
    #print gdata
