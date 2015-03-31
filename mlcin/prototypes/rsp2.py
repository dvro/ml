import numpy as np
import random
import sklearn
import math
from scipy.spatial.distance import pdist, squareform
from .base import InstanceReductionMixIn
import matplotlib.pyplot as plt


class ReductionBySpacePartitioning2(InstanceReductionMixIn):

    '''
    This class is based on the prototype generation method developed by Sanchez 2012.
    This class was created by Joao Paulo and Rene. 
    '''

    def __init__(self, b=1,plotStep = False):
        if b:
            self.b = b
        else:
            self.b = 1
        self.plotStep = plotStep
            
    def isHomogeneous(self,y,lista):
        """ Recebe duas listas, uma com os indices e uma com as classes de toda a base como paremetro e verifica se o subconjunto da lista eh homogeneo """
        ret = True
        if (len(lista) > 0):
            a = y[lista[0]]
            for i in lista:
                if (y[i] != a):
                    ret = False
                    break
    
        return ret

    def diametroConj(self,dist,conj,d):
        """The diameter of a set is defined as the distance between the two farthest points."""
        diametro = -1
        p1 = -1
        p2 = -1
        for a in conj:
            for b in conj:
                if (dist[a][b] > diametro):
                    diametro = dist[a][b]
                    p1 = a
                    p2 = b  
                                        
        return diametro,p1,p2
    
    def overlappingDegree(self,dist,conj,y):
        """This function calculates the overlapping degree of a set. 
           Dist is the pairwise distance matrix. Conj is the list of elements in the set.y is the class list
         This measure is defined as the ratio of the average distance between instances belonging to different classes, D,
         and the average distance between instances that are from the same class, D=."""
         
        Ddif = 0
        countDif = 0
        Dsame = 0
        countSame =0
        classes = list(set(y)) 
        for i in conj:
            for j in conj:
                if (y[i] == y[j]): #elements same class
                    Dsame += dist[i][j]
                    countSame += 1.0
                else:
                    Ddif += dist[i][j]
                    countDif += 1.0
                    
        if (countDif > 0):
            Ddif = Ddif/countDif
            
        if (countSame > 0):
            Dsame = Dsame/countSame
            
        if (Dsame > 0):
            return Ddif/Dsame
        else:
            return 9999999 #A very large number    

    def centroidsConjuntos(self, X,y,conjuntos):
        """ Retorna o centroid de cada classe em cada conjunto dentro de conjuntos"""
        centroidsX =[]
        centroidsY =[]
    
        for B in conjuntos:
            y_B = [y[i] for i in B]
            set_y = list(set(y_B))
            centroidsY += set_y
            for c in set_y: #para cada classe
                x_classe = [X[i] for i in B if y[i] == c] 
                centroidsX += [list(np.mean(x_classe[:],axis=0))]
    
        return centroidsX,centroidsY
    
    def row_col_from_condensed_index(self,d,i):
        b = 1 -2*d 
        x = math.floor((-b - math.sqrt(b**2 - 8*i))/2)
        y = i + x*(b + x + 2)/2 + 1
        return (x,y) 
            
    def getFatherstPoints(self,sfDist,conj):
        p1 = -1
        p2 = -1
        max = 0
        for i in conj:
            for j in conj:
                if (sfDist[i][j] > max):
                    max = sfDist[i][j]
                    p1 = i
                    p2 = j
        
        return p1,p2

    def RSP2(self):
        X = self.X
        y = self.y
        b = self.b
        dist = pdist(X) #Calcula a distancia par a par e retorna em uma lista condensada
        sfDist = squareform(dist)
        d = len(X) 
        conjuntos = []
        conjuntoAtual = range(0,d)
        
        p1,p2 = self.row_col_from_condensed_index(d, dist.argmax()) 
        
    
        for z in range(0,b):      
            conjp1 = []
            conjp2 = []
            for i in conjuntoAtual:
                if (sfDist[p1][i] > sfDist[p2][i]):
                    conjp2 += [i]
                else:
                    conjp1 += [i]
            
            conjuntos += [conjp1,conjp2] #atribui os dois conjuntos a lista de conjuntos
            
            if (self.plotStep):
                title = "RSP 2 - b = " + str(b) + " step: " + str(z)
                plt.title(title)
                plt.xlim(-10,10)
                plt.ylim(-10,10)
                plt.xlabel('feature 1')
                plt.ylabel('feature 2')
                plt.plot(X[p1,0],X[p1,1],"+")
                plt.plot(X[p2,0],X[p2,1],".")
                

                for c1 in conjp1:
                    plt.plot(X[c1,0],X[c1,1],"ro")
                
                for c2 in conjp2:
                    plt.plot(X[c2,0],X[c2,1],"yo")
                    
                markers=["bs", "1", "k^", "d","h","4","8","s","p"]
                p = 0
                for cj in conjuntos:
                    for t in cj: 
                        a = (t in conjp1) or (t in conjp2) or (t == p1) or (t == p2)
                        if a == False:
                            plt.plot(X[t,0],X[t,1],markers[p])
                    p += 1 
                #plt.show()
                plt.savefig("RSP2Step/" + title + ".png")
                plt.clf() 
    
            #Seleciona proxima divisao dentro de conjuntos
            if (z < b-1):
                maxOD = -1
                proxConj = -1 #guarda o indice do proximo conjunto
                listaEhHomogeneo = [self.isHomogeneous(y,c) for c in conjuntos]
    
                aux = False in listaEhHomogeneo
                for c in range(0,len(conjuntos)):
                    if ((aux and listaEhHomogeneo[c] == False) or (aux == False)):
                        overlappingDegree = self.overlappingDegree(sfDist,conjuntos[c],y)
                        if (overlappingDegree > maxOD):
                            maxOD = overlappingDegree
                            proxConj = c
    
                #atualiza p1 e p2
                conjuntoAtual = conjuntos[proxConj]
                p1,p2 = self.getFatherstPoints(sfDist, conjuntoAtual)
                conjuntos = conjuntos[:proxConj] + conjuntos[proxConj +1:]
                
        return self.centroidsConjuntos(X,y,conjuntos) 


    def reduce_data(self):
        prototypes, prototypes_labels = self.RSP2()
        self.prototypes = np.asarray(prototypes)
        self.prototypes_labels = np.asarray(prototypes_labels)
        self.reduction_ratio = 1 - float(len(self.prototypes_labels)) / len(self.y)
        return self.prototypes, self.prototypes_labels
