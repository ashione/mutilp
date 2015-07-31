# author : Nelson Zuo
# date   : 2015-7-24
# 7-25 adding spring layout in graphviz
# 7-26 adding HierarchicalTree
import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt
import glog
import xlrd
import random
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as sch
import argparse
#coding=utf-8
class GraphGenerator(object):
    @classmethod
    def ConvertMatrixToGraph(self,filename):
        #mtx = np.loadtxt(filename)
        #print mtx
        #G = nx.Graph()
        #for i in range(0,mtx.shape[0]) :
        #    for j in range(0,mtx.shape[1]):
        #        G.add_edge(i,j,weight=mtx[i,j])
        return nx.from_numpy_matrix(np.loadtxt(filename))
        #return G

    @classmethod
    def ConvertTripleArrayToGraph(filename):
        data = np.asarray(np.loadtxt(filename))
        G = nx.Graph()
        map(lambda x : G.add_edge(np.int(x[0]),np.int(x[1]),weight=x[2]),data)
        return G

    @classmethod
    def ConvertXlsxToArray(self,filename):
        glog.info('Now reading xlsx file')
        compydata = xlrd.open_workbook(filename)
        glog.info(u'xlsx file , %d sheets , and their names are %s' %(compydata.nsheets,compydata.sheet_names()))
        first_sheet = compydata.sheet_by_index(0)
        nodesSeq  =  np.asarray(map(lambda x : x.value,first_sheet.col_slice(start_rowx=2,end_rowx=first_sheet.nrows,colx=0)),dtype=np.int)
        nodesType =  map(lambda x : x.value,first_sheet.col_slice(start_rowx=2,end_rowx=first_sheet.nrows,colx=1))
        glog.info('Reading nodes label orders and types : %d' %(nodesSeq.size))
        return { s : t for s,t in zip(nodesSeq,nodesType)}


    @classmethod
    def curvePosPoly(self,pos,color):
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

    @classmethod
    def PolyArea2D(self,pts):
        print pts
        lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
        area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
        return area

    @classmethod
    def curvePosHull(self,pos,color):
        Path = mpath.Path
        patch_data = []
        for k,ipos in pos.items():
            patch_data.append((Path.CURVE4,ipos))
        patch_data[0] = ((Path.MOVETO,patch_data[0][1]))
        codes,verts = zip(*patch_data)
        verts = np.asarray(verts)
        hull = ConvexHull(verts)
        hull_border = hull.points[hull.vertices]
        glog.info( 'ConvexHull Area : %f' %self.PolyArea2D(hull_border))

        patch= mpatches.Polygon(hull_border,facecolor=color,alpha=0.3,linestyle='dashed')
        return patch,hull_border

    @classmethod
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
        #path = mpath.Path(verts,codes)
        #print 'path : ',path.to_polygons(),verts
        #patch= mpatches.PathPatch(path,facecolor=color,alpha=0.5)
        #patch= mpatches.Polygon(verts,facecolor=color,alpha=0.5)
        verts = np.asarray(verts)
        fcenter = lambda x : (np.max(x)+np.min(x))/2
        xindex,yindex = fcenter(verts[:,0]),fcenter(verts[:,1])
        width = np.max(verts[:,0]) - np.min(verts[:,0])+ np.median(verts[:,0])*0.1
        height = np.max(verts[:,1]) - np.min(verts[:,1]) + np.median(verts[:,1])*0.1
        patch= mpatches.Ellipse((xindex,yindex),height=height,width=width,facecolor=color,alpha=0.3,linestyle='dashed')
        return patch

    @classmethod
    def autofited(self,G,pos):
        glog.info('autofited debug information starting')
        atnode1 = G.adj[80].keys()+[80]
        atnode2 = G.adj[43].keys()+[43]
        #print atnode1,atnode2

        s =set()
        for i in atnode1:
        #    print pos[i]
            if i in s or i ==43 :
                continue
            s.add(i)
            pos[i] = (pos[i][0]-50,pos[i][1]-100)
        #    print pos[i]

        for i in atnode2:
            if i in s or i == 80:
                continue
            s.add(i)
            pos[i] = (pos[i][0],pos[i][1]-100)
        pos[119] =( pos[119][0],pos[119][1]+50)
        pos[225] =( pos[225][0],pos[225][1]+50)
        return G,pos

    @classmethod
    def drawByTripleArry(self,G,nodesDict,colorDict,showAllPoly=False):
        G = nx.kruskal_mst(G)
        #pos = nx.spring_layout(G)
        #pos = nx.drawing.graphviz_layout(G,'fdp')
        pos = nx.drawing.graphviz_layout(G,'sfdp')
        #print pos[80],pos[43]
        #G,pos = autofited(G,pos)
        #print pos[80],pos[43]
        #pos = nx.circular_layout(G)
        #pos = nx.random_layout(G)
        #pos = nx.spectral_layout(G)

        #print pos
        #np.savetxt('pos.txt',pos.values())
        #print len(G.nodes()),len(nodesDict.keys())

        fig,ax = plt.subplots()
        print type(ax)
        #patch = curvePos(pos)
        #ax.add_patch(patch)

        # different nodes cover different color
        for node in G.nodes():
            G.node[node]['category'] = colorDict[nodesDict[node+1]]
            G.node[node]['node_size'] = np.log(np.e+G.degree([node])[node])*100
        # differen edges width
        #for edge in G.edges():
        #    edge = edge[0]
        #    if G.degree([edge]).has_key(edge):
        #        edge_size = np.log(np.e+G.degree([edge])[edge])
        #    else :
        #        edge_size = 1.
        #    G.edge[edge]['edge_size'] = edge_size
        submap = {}

        # counting each color subgraph map set
        for k,v in G.node.items():
            if not submap.has_key(v['category']) :
                submap[v['category']] = set()
            else :
                submap[v['category']].add(k)

        # add connected_component_subgraphs in each subgraph
        patchList = []
        for color in colorDict.values():
            spos = { x : pos[x] for x in submap[color] }
            #print spos
            if len(spos.keys()) <=2 :
                continue
            subG =  G.subgraph(spos.keys())
            for connt in nx.connected_component_subgraphs(subG):
                sgpos = { x : pos[x] for x in connt.nodes() }
                if len(sgpos.keys())>2:
                    #patch = curvePos(sgpos,color)
                    patchList.append(self.curvePosHull(sgpos,color))
                    #patchList.append(patch)

        if not showAllPoly :
            patchList = self.check_border(patchList)
        map(lambda patch:ax.add_patch(patch),patchList)
        #nx.draw_networkx(G,pos=pos,node_color= [ G.node[node]['category'] for node in G ], alpha=0.6,node_size=300,font_size=10,)
        #nx.draw_networkx_nodes(G,pos=pos,node_color= [ G.node[node]['category'] for node in G ],
        #                 alpha=0.6,node_size=100,font_size=10,)
        #                 #labels=None)
        nx.draw_networkx_nodes(G,pos=pos,node_color= [ G.node[node]['category'] for node in G ],
                         alpha=0.6, node_size=[G.node[node]['node_size'] for node in G])
                         #labels=None)
        #nx.draw_networkx_edges(G,pos=pos)
        for edge in G.edges():
            #edge_size=  G.edge[edge[0]][ 'edge_size' ]
            edge_size = np.log(1.0+G.degree([edge[0]])[edge[0]])
            nx.draw_networkx_edges(G,{edge[0] : pos[edge[0]],edge[1] : pos[edge[1]]},
                                   edgelist=[edge], width=edge_size,alpha=0.4)
        arrpos = np.asarray(pos.values())
        xmin,xmax = arrpos[:,0].min()-arrpos[:,0].mean()*0.05,arrpos[:,0].max()+arrpos[:,0].mean()*0.05
        ymin,ymax = arrpos[:,1].min()-arrpos[:,1].mean()*0.05,arrpos[:,1].max()+arrpos[:,1].mean()*0.05
        glog.info('x range[%f,%f], y range[%f,%f]' %(xmin,xmax,ymin,ymax))
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        #plt.xlim(-0.02*1000,1.02*1500)
        #plt.ylim(-0.02*1000,1.02*1500)
        #plt.show()

    @classmethod
    def distMaxInPath(self,G):
        dist_shape = (len(G.nodes()),len(G.nodes()))
        dist_matrix = np.zeros(dist_shape)
        p = nx.shortest_path(G)
        #print G.edge
        for i in range(dist_shape[0]):
            for j in range(dist_shape[1]):
                #if len(p[i][j]):
                maxdist = 0
                for k in range(len(p[i][j])-1):
                    if maxdist == 0 or G.edge[p[i][j][k]][p[i][j][k+1]]['weight']>maxdist:
                        maxdist = G.edge[p[i][j][k]][p[i][j][k+1]]['weight']
                dist_matrix[i,j] = maxdist

        #np.savetxt('dist.txt',dist_matrix)
        return dist_matrix

    @classmethod
    def distToTree(self,linkage_matrix):
        return sch.to_tree(linkage_matrix)
        #self.travelTree(link_tree)
        #print self.snode
        #return link_tree

    @classmethod
    def travelTree(self,node,G):
        #print node.id,node.count
        if node.is_leaf():
            #print node.id,node.count
            return {node.id:G.node[node.id]['category']}
        snodedict = {}
        leftdict = self.travelTree(node.left,G)
        snodedict.update(leftdict)
        rightdict = self.travelTree(node.right,G)
        snodedict.update(rightdict)
        if snodedict[node.left.id]!= snodedict[node.right.id]:
            snodedict[node.id] = 'black'
        else :
            snodedict[node.id] = snodedict[node.left.id]
        return snodedict

    @classmethod
    def drawHierarchicalTree(self,G,nodesDict,colorDict):
        G = nx.kruskal_mst(G)
        for node in G.nodes():
            G.node[node]['category'] = colorDict[nodesDict[node+1]]
        dist_matrix = self.distMaxInPath(G)
        glog.info('shortest_path shape : {0}'.format( dist_matrix.shape))
        #triu_dist_matrix =  dist_matrix[np.triu_indices_from(dist_matrix)]
        #dist_matrix[dist_matrix == 0 ] = np.inf
        #print 'dist max',dist_matrix.max()
        linkage_matrix = linkage(dist_matrix, 'single',metric='euclidean')
        #linkage_matrix = linkage(dist_matrix, 'complete',metric='euclidean')
        #np.savetxt('link.txt',linkage_matrix,fmt='%d %d %f %d')
        plt.plot()
        plt.title("ascending")

        degrm = dendrogram(linkage_matrix,
                   #p=10000,
                   #color_threshold=1,
                   #truncate_mode='lastp',
                   truncate_mode='mlab',
                   #truncate_mode='level',
                   #truncate_mode='mtica',
                   get_leaves = True,
                   #labels=np.array([str(x) for x in range(smxt.shape[0])]),
                   distance_sort='ascending',
                   show_leaf_counts=True,
                   orientation='left',
                   show_contracted = True,
                   #link_color_func = lambda x : G.node[x%smxt.shape[0]]['category'],
                   )
        #glog.info('{0},{1},{2}'.format( len( degrm['leaves'] ),degrm['leaves'],degrm))
        glog.info('{0},\nlen = {1}'.format( degrm['ivl'],len(degrm['ivl'])))
        print len(set(degrm['ivl']))
        link_tree = self.distToTree(linkage_matrix)

        treecolordict = self.travelTree(link_tree,G)
        count = 0
        for i in range(dist_matrix.shape[0]-1,(dist_matrix.shape[0]-1)*2):
            if treecolordict[i] is not 'black':
                count += 1

        glog.info('There are %d clusters'%count)

        #print type(link_tree),link_tree.is_leaf()

    @classmethod
    def check_border(self,patchList):
        converPatches = []
        import poly
        for i in range(len(patchList)):
            flag = True
            for j in range(0,len(patchList)):
                if i == j :
                    continue
                if poly.check_polygon_cross_polyon(patchList[i][1],patchList[j][1]):
                    flag = False
            if flag :
                converPatches.append(patchList[i][0])
        return converPatches

if __name__ == '__main__':
    #gdata = np.loadtxt('./firmmst.txt')
    #drawByTripleArry(ConvertTripleArrayToGraph('./firmmst.txt'))
    parser =argparse.ArgumentParser(description="draw graph script.")
    parser.add_argument('-xls',type=str,default='./companyPty.xlsx',help='excel file with color map')
    parser.add_argument('-df',type=str,default='./firmMstMatrix.txt',help='data set sources filename')
    args = parser.parse_args()
    generator = GraphGenerator()
    xlsxDict =  generator.ConvertXlsxToArray(args.xls)
    colorTypes = set(xlsxDict.values())
    colorDict = { x : ('#%06X' % random.randint(0,0xFFFFFF)) for x  in colorTypes }
    #drawByTripleArry(ConvertTripleArrayToGraph('./firmmst.txt'),xlsxDict,colorDict)
    G = generator.ConvertMatrixToGraph(args.df)
    #print 'MAXTRIX'
    #print nx.to_numpy_matrix(G)
    #for x,y in G.edges():
    #    print G.edge[x][y]
    #    G.edge[x][y]['weight'] = 1
    #    G.edge[y][x]['weight'] = 1
    #generator.drawByTripleArry(G,xlsxDict,colorDict)
    generator.drawHierarchicalTree(G,xlsxDict,colorDict)
    #generator.distMaxInPath(G)
    plt.show()
    #print G.edge[0][173]['weight']
    #print nx.algorithms.topological_sort(G)
    #print gdata
