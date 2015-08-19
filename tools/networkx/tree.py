# author : Nelson Zuo
# date   : 2015-7-24
# 7-25 adding spring layout in graphviz
# 7-26 adding HierarchicalTree
# 7-30 fix spring layout distance
# 8-7  generate data and picture
# 8-9  generate tree dist and dpi about picture
# 8-16 addition
# 8-18 update
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
import poly
from multiprocessing import Pool
#from IPython.Debugger import Tracer
#coding=utf-8
class GraphGenerator(object):
    def ConvertMatrixToGraph(self,filename):
        #mtx = np.loadtxt(filename)
        #print mtx
        #G = nx.Graph()
        #for i in range(0,mtx.shape[0]) :
        #    for j in range(0,mtx.shape[1]):
        #        G.add_edge(i,j,weight=mtx[i,j])
        return nx.from_numpy_matrix(np.loadtxt(filename))
        #return G

    def ConvertTripleArrayToGraph(filename):
        data = np.asarray(np.loadtxt(filename))
        G = nx.Graph()
        map(lambda x : G.add_edge(np.int(x[0]),np.int(x[1]),weight=x[2]),data)
        return G

    def ConvertOriginalXlsx(self,filename):
        glog.info('now reading xlsx : company name')
        compyorgdata = xlrd.open_workbook(filename)
        first_sheet = compyorgdata.sheet_by_index(0)
        compyname  =  np.asarray(map(lambda x : x.value,first_sheet.col_slice(start_rowx=2,end_rowx=first_sheet.nrows,colx=1)))
        #glog.info('company name : {0}'.format(compyname))
        self.compyname = compyname
        #print compyname[0],compyname[300]
        return self.compyname

    def ConvertXlsxToArray(self,filename):
        glog.info('Now reading xlsx file')
        compydata = xlrd.open_workbook(filename)
        glog.info(u'xlsx file , %d sheets , and their names are %s' %(compydata.nsheets,compydata.sheet_names()))
        first_sheet = compydata.sheet_by_index(0)
        nodesSeq  =  np.asarray(map(lambda x : x.value,first_sheet.col_slice(start_rowx=2,end_rowx=first_sheet.nrows,colx=0)),dtype=np.int)
        nodesType =  map(lambda x : x.value,first_sheet.col_slice(start_rowx=2,end_rowx=first_sheet.nrows,colx=1))
        glog.info('Reading nodes label orders and types : %d' %(nodesSeq.size))
        self.xlxsDict = { s : t for s,t in zip(nodesSeq,nodesType)}
        return self.xlxsDict


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

    def PolyArea2D(self,pts):
        #print pts
        lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
        area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
        return area

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
        #glog.info( 'ConvexHull Area : %f' %self.PolyArea2D(hull_border))

        patch= mpatches.Polygon(hull_border,facecolor=color,alpha=0.6,linestyle='dashed')
        return patch,hull_border

    @classmethod
    def curvePos(self,pos,color):
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
        #print 'pos',pos
        #print 'verts:',verts
        fcenter = lambda x : (np.max(x)+np.min(x))/2
        xindex,yindex = fcenter(verts[:,0]),fcenter(verts[:,1])
        dist_r = np.sum(np.power(verts[0,:]-verts[1,:],2.0))**0.5/2.0
        dist_a = dist_r/0.618
        dist_b = dist_r
        #width = ( np.max(verts[:,0]) - np.min(verts[:,0])+ np.median(verts[:,0])*0.00 )/2.0
        #height = ( np.max(verts[:,1]) - np.min(verts[:,1]) + np.median(verts[:,1])*0.00 )/2.0
        angle = np.arctan(poly.get_slope(verts[0,:],verts[1,:]))
        #angle = poly.get_slope(verts[0,:],verts[1,:])
        if angle < 0 :
            angle += 1.0*np.pi
        #print angle,verts,xindex,yindex
        #patch= mpatches.Ellipse((xindex,yindex),height=height,width=width,angle=angle,facecolor=color,alpha=0.3,linestyle='dashed')
        patch= mpatches.Ellipse((xindex,yindex),height=2*dist_a,width=2*dist_b,facecolor=color,alpha=0.6,linestyle='dashed')
        return patch,verts

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

    def writeClusterMst(self,G):
        G = nx.kruskal_mst(G)
        f = open('data/alpha_%.2f_cluster_mst.txt'%self.alpha,'w')
        #glog.info('{0},{1}'.format(len(G.edges()),G.edge))
        self.Lntl = 0.0
        #if not hasattr(self,'direct_dist_matrix'):
        self.direct_dist_matrix = self.distShortestInPath(G)

        triu_dist_matrix =  self.direct_dist_matrix[np.triu_indices_from(self.direct_dist_matrix)]
        node_shape = (len(G.nodes()),len(G.nodes()))
        self.lmsm = np.sum(triu_dist_matrix)*2.0/(node_shape[0]*(node_shape[1]-1))
        for i in range(0,node_shape[0]):
            for j in range(0,node_shape[1]):
                try :
                    f.write('%.5f '%G.edge[i][j]['weight'])
                    self.Lntl+= G.edge[i][j]['weight']
                except Exception,e:
                    #glog.error(e)
                    f.write('%.5f '%0.0)
            f.write('\n')
        self.Lntl = self.Lntl / (node_shape[0]-1)
        f.close()
        glog.info('write mst cluster done!')

    def writeMstDegree(self,filename='data/alpha_0.50_mstDegree.txt'):
        self.gdegree = sorted(self.gdegree,key = lambda x:(x[1],x[0]),reverse=True)
        with open(filename,'w') as fp :
            map(lambda x:fp.write('{0} : ({1}, {2})\n'\
                .format(self.compyname[x[0]].encode('utf8'),\
                x[0]+1,x[1])),self.gdegree)
            fp.close()
        glog.info('write degree done!')

    def computeMOL(self,G):
        if not hasattr(self,'gdegree') and \
            not hasattr(self,'direct_dist_matrix')  and \
            not hasattr(self,'lengthinPath'):

            glog.error('NotImplementedError')
            return

        maxdegreepos = filter(lambda x: x[1] == self.gdegree[0][1],self.gdegree)
        glog.info('maxdegreepos : {0}'.format(maxdegreepos))
        maxdegreeinMindist = np.zeros(len(maxdegreepos),dtype=np.float)
        for i,pos in enumerate(maxdegreepos):
            for v in G.edge[pos[0]].values():
                maxdegreeinMindist[i]+=v['weight']

        argcenter = np.argmin(maxdegreeinMindist)
        glog.info(maxdegreeinMindist.shape)
        glog.info(argcenter)
        centerp =  self.gdegree[argcenter][0]
        self.lev = np.zeros((len(G.nodes()),))
        for v in range(len(G.nodes())):
            if v is not centerp:
                self.lev[v] = sum(map(lambda x: self.lengthinPath[centerp][x],G.edge[v].keys()))

        self.Lmol = np.mean(self.lev)

    def drawByTripleArry(self,G,nodesDict,colorDict,showAllPoly=False):

        G = nx.kruskal_mst(G)
        self.writeClusterMst(G)
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
        #print type(ax)
        #patch = curvePos(pos)
        #ax.add_patch(patch)

        # different nodes cover different color
        #number of non-leaf nodes
        self.nln = 0
        self.gdegree = [(0,0)]*len(G.nodes())
        for node in G.nodes():
            G.node[node]['category'] = colorDict[nodesDict[node+1]]
            G.node[node]['node_size'] = np.log(np.e+G.degree([node])[node])*30
            self.gdegree[node] = (node,G.degree([node])[node])
            if self.gdegree[node][1] > 1:
                self.nln+=1

        self.writeMstDegree('data/alpha_%.2f_MstDegree.txt'%self.alpha)
        self.computeMOL(G)
        self.computeCopheneticCoreelationCoef(G)
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
                submap[v['category']] = set([k])
            else :
                submap[v['category']].add(k)

        #glog.info('submap :{0}'.format(submap))
        # add connected_component_subgraphs in each subgraph
        patchList = []
        clusterNodesDt = []
        for color in colorDict.values():
            spos = { x : pos[x] for x in submap[color] }
            #print spos
            if len(spos.keys()) <2 :
                continue
            subG =  G.subgraph(spos.keys())
            #glog.info('sub nodes: {0}'.format(subG.edges()))
            for connt in nx.connected_component_subgraphs(subG):
            #for connt in nx.weakly_connected_component_subgraphs(subG):
                sgpos = { x : pos[x] for x in connt.nodes() }
                #if len(filter(lambda x : x == 25, sgpos.keys())):
                #    print sgpos
                #    print spos
                #    glog.info('connt : {0}'.format(connt.nodes()))
                #    print 'subG',subG
                #    dnodes = G.edge[25].keys()
                #    for x in dnodes :
                #        print x,G.node[x]['category']
                #if 24 in sgpos.keys():
                #    glog.info('connt : {0}'.format(connt.nodes()))
                if len(sgpos.keys()) == 1:
                    continue
                clusterNodesDt.append((self.xlxsDict[connt.nodes()[0]+1],sgpos.keys()))
                if len(sgpos.keys())>2:
                    #patch = curvePos(sgpos,color)
                    patchList.append(self.curvePosHull(sgpos,color))
                    #patchList.append(patch)
                elif len(sgpos.keys())==2:
                    patchList.append(self.curvePos(sgpos,color))
        self.writeTreeCluster(G,clusterNodesDt,'data/alpha_%.2f_clusterfn.txt'%self.alpha)
        if not showAllPoly :
            patchList = self.check_border(patchList)
        map(lambda patch:ax.add_patch(patch),patchList)
        #nx.draw_networkx(G,pos=pos,node_color= [ G.node[node]['category'] for node in G ], alpha=0.6,node_size=300,font_size=10,)
        #nx.draw_networkx_nodes(G,pos=pos,node_color= [ G.node[node]['category'] for node in G ],
        #                 alpha=0.6,node_size=100,font_size=10,)
        #                 #labels=None)
        nx.draw_networkx_nodes(G,pos=pos,node_color= [ G.node[node]['category'] for node in G ],
                         alpha=0.6, node_size=[G.node[node]['node_size'] for node in G])
        # draw lables
        nx.draw_networkx_labels(G,pos,font_size=4)#,labels=[ node for node in G.nodes() ])
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
        plt.savefig('data/alpha_%.2f_clusterfig.pdf'%self.alpha,format='pdf',dpi=20)
        #plt.xlim(-0.02*1000,1.02*1500)
        #plt.ylim(-0.02*1000,1.02*1500)
        #plt.show()

    def normalizationLenCluster(self,G,spos):
        #if not hasattr(self,'direct_dist_matrix'):
        #self.direct_dist_matrix = self.distShortestInPath(G)
        nlc = 0.0
        for i in range(0,len(spos)):
            for j in range(i+1,len(spos)):
                #nlc+=self.direct_dist_matrix[spos[i],spos[j]]
                try :
                    nlc+=G.edge[spos[i]][spos[j]]['weight']
                except Exception,e:
                    glog.error('no edege includeing in {node}'.format(node=e))

        return nlc/len(spos)

    def writeTreeCluster(self,G,clusterNodesDt,clusterfn='clusterfn.txt'):
        if os.path.exists(clusterfn):
            glog.info('exists clusterfn,now removing it ,then writing')
            os.remove(clusterfn)
        #print clusterNodesDt
        self.nlc = map(lambda x : self.normalizationLenCluster(G,x[1]),clusterNodesDt)
        self.cpl = np.sum(np.sum(self.lengthinPath))/(self.lengthinPath.shape[0]*(self.lengthinPath.shape[1]-1))
        with open(clusterfn,'w') as fp :
            fp.write('Lmsm : {0} \nLntl : {1} \nLcpl : {2} \nnln : {3}\nLmol : {4}\nccc : {5}\n'\
                     .format(self.lmsm,self.Lntl,self.cpl,self.nln,self.Lmol,self.ccc))

            map(lambda line:fp.write('{0:>12} : Lc(t) : {1:.5f} {2} \n'\
                .format(str( line[0][0].encode('utf8') ),\
                line[1],\
                ' '.join(map(lambda x: str(self.compyname[x].encode('utf8'))+' ('+str(x+1)+') ',line[0][1])))),\
                zip(clusterNodesDt,self.nlc))

            fp.close()
            glog.info('write done!')

    def distMaxInPath(self,G):
        #if hasattr(self,'dist_matrix'):
        #    return self.dist_matrix

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
        self.dist_matrix = dist_matrix
        return dist_matrix

    def distShortestInPath(self,G):
        #if hasattr(self,'direct_dist_matrix'):
        #    return self.direct_dist_matrix

        dist_shape = (len(G.nodes()),len(G.nodes()))
        dist_matrix = np.zeros(dist_shape)
        self.lengthinPath = np.zeros(dist_shape)
        p = nx.shortest_path(G)
        for i in range(dist_shape[0]):
            for j in range(dist_shape[1]):
                self.lengthinPath[i,j] = len(p[i][j])-1
                for k in range(len(p[i][j])-1):
                    dist_matrix[i,j]+= G.edge[p[i][j][k]][p[i][j][k+1]]['weight']
        return dist_matrix

    def distToTree(self,linkage_matrix):
        return sch.to_tree(linkage_matrix)
        #self.travelTree(link_tree)
        #print self.snode
        #return link_tree

    def travelTree(self,node,G):
        #print node.id,node.count
        if node.is_leaf():
            #print node.id,node.count
            #print glog.info(node.dist)
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

    def treePreOrder(self,node):
        if node.is_leaf():
    #        Tracer()
            #print glog.info(node.dist)
            #return {'id' : [node.id],'dist' : node.dist}
            return [node.id]
        tempNodes = []
        tempNodes.extend(self.treePreOrder(node.left))
        tempNodes.extend(self.treePreOrder(node.right))
        #return {'id' : tempNodes , 'dist' : node.dist}
        return  tempNodes

    def clusterNodes(self,node,treecolordict):
        if node.is_leaf():
            return {node.id : (node.id,node.dist)}

        if treecolordict[node.id] is not 'black':
            #print node.id,node.count,node.left.id,node.right.id
            #print node.pre_order()
            return {node.id : (self.treePreOrder(node),node.dist)}

        rootorder = {}
        leftorder = self.clusterNodes(node.left,treecolordict)
        rightorder = self.clusterNodes(node.right,treecolordict)
        rootorder.update(leftorder)
        rootorder.update(rightorder)
        return rootorder

    def writeTreeRecords(self,treecluster,tclusterf='./data/alpha_0.50_clustertree.txt'):
        if os.path.exists(tclusterf):
            os.remove(tclusterf)
        #treecluster
        with open(tclusterf,'w') as fp :
            map(lambda x: fp.write('{0},{1}, dist : {2} , {3}\n'.format(\
                self.xlxsDict[x[0][0]+1].encode('utf8'),x[0],x[1],' '.join(map(lambda y : self.compyname[y].encode('utf8'),x[0])))) \
                if type(x[0]) is list and len(x[0]) >1 else 0\
                ,treecluster.values())
            glog.info('write cluster tree done!')

    def drawHierarchicalTree(self,G,nodesDict,colorDict):
        G = nx.kruskal_mst(G)
        for node in G.nodes():
            G.node[node]['category'] = colorDict[nodesDict[node+1]]
        dist_matrix = self.distMaxInPath(G)
        np.savetxt('data/alpha_%.2f_clusterdist.txt'%self.alpha,dist_matrix,fmt='%.5f')
        glog.info('shortest_path shape : {0}'.format( dist_matrix.shape))
        #triu_dist_matrix =  dist_matrix[np.triu_indices_from(dist_matrix)]
        #dist_matrix[dist_matrix == 0 ] = np.inf
        #print 'dist max',dist_matrix.max()
        linkage_matrix = linkage(dist_matrix, 'single',metric='euclidean')
        #linkage_matrix = linkage(dist_matrix, 'centroid',metric='euclidean')
        #linkage_matrix = linkage(dist_matrix, 'complete',metric='euclidean')
        np.savetxt('link.txt',linkage_matrix,fmt='%d %d %f %d')
        np.savetxt('dist.txt',dist_matrix,fmt='%f')
        plt.plot()
        plt.title("ascending")

        degrm = dendrogram(linkage_matrix,
                   #p=10000,
                   #color_threshold=1,
                   truncate_mode='lastp',
                   #truncate_mode='mlab',
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
        #glog.info('{0},\nlen = {1}'.format( degrm['ivl'],len(degrm['ivl'])))
        #print len(set(degrm['ivl']))
        link_tree = self.distToTree(linkage_matrix)
        #glog.info('{0}'.format(link_tree.dist))
        treecolordict = self.travelTree(link_tree,G)
        treeclusters = self.clusterNodes(link_tree,treecolordict)
        self.writeTreeRecords(treeclusters,'data/alpha_%.2f_clustertree.txt'%self.alpha)
        #glog.info('clusters : {0},\n{1}'.format(treeclusters,len(treeclusters.keys())))
        plt.savefig('data/alpha_%.2f_treefig.png'%self.alpha,pad_inches=0.05)
        #plt.show()
        #count = 0
        #for i in range(dist_matrix.shape[0]-1,(dist_matrix.shape[0]-1)*2):
        #    if treecolordict[i] is not 'black':
        #        count += 1

        #glog.info('There are %d clusters'%count)

        #print type(link_tree),link_tree.is_leaf()

    def check_border(self,patchList):
        converPatches = []
        #converPatches.append(filter(lambda x: type(x) is not tuple,patchList)
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

    def setalpha(self,alpha):
        self.alpha = alpha

    def computeCopheneticCoreelationCoef(self,G):
        direct_dist_matrix = self.distShortestInPath(G)
        dist_matrix = self.distMaxInPath(G)
        mean_d = np.mean(direct_dist_matrix)
        mean_c = np.mean(dist_matrix)
        glog.info('mean_d : {0} / mean_c {1}'.format(mean_d,mean_c))
        dup = 0.0
        ddown1,ddown2 = 0.0,0.0
        for i in range(dist_matrix.shape[0]):
            for j in range(i+1,dist_matrix.shape[1]):
                dup+=(direct_dist_matrix[i,j] - mean_d)*(dist_matrix[i,j] - mean_c)
                #print direct_dist_matrix[i,j]
                ddown1+=((direct_dist_matrix[i,j] - mean_d)**2)
                ddown2+=( (dist_matrix[i,j] - mean_c)**2 )
        #glog.info(dup)
        self.ccc = dup/( ( ddown1*ddown2 ) ** 0.5)

def diffalpha(alpha):
    #gdata = np.loadtxt('./firmmst.txt')
    #drawByTripleArry(ConvertTripleArrayToGraph('./firmmst.txt'))
    parser =argparse.ArgumentParser(description="draw graph script.")
    parser.add_argument('-xls',type=str,default='./companyPty.xlsx',help='excel file with color map')
    parser.add_argument('-tls',type=str,default='./shouxin.xlsx',help='excel file with node name')
    parser.add_argument('-df',type=str,default='./pretrain/distfirm_%.2f.txt'%alpha,help='data set sources filename')
    args = parser.parse_args()
    generator = GraphGenerator()
    generator.setalpha(alpha)
    compyname =  generator.ConvertOriginalXlsx(args.tls)
    xlsxDict =  generator.ConvertXlsxToArray(args.xls)
    #print xlsxDict[1]
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
    generator.drawByTripleArry(G,xlsxDict,colorDict)
    generator.drawHierarchicalTree(G,xlsxDict,colorDict)
    #generator.distMaxInPath(G)
    #plt.show()
    #print G.edge[0][173]['weight']
    #print nx.algorithms.topological_sort(G)
    #print gdata
    del generator
def mapSolved(alpha):
        pwd = os.getcwd()
        os.chdir('./pretrain')
        if not os.path.exists('distfirm_%.2f.txt'%alpha):
            os.system('python MatrixTransform.py %f > /dev/null' %alpha)

        os.chdir(pwd)
        diffalpha(alpha)

if __name__ == '__main__':
    alphalist  = [x*1.0/100 for x in  range(0,105,5)]
    pooling = Pool(len(alphalist))
    print pooling.map(mapSolved,alphalist)

    #diffalpha(alpha=0.50)
