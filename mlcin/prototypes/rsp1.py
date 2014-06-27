'''
Created on Jun 26, 2014

@author: jpaborges
'''
import numpy as np
import random
import sklearn
import math
from scipy.spatial.distance import pdist, squareform
from .base import InstanceReductionMixIn


class ReductionBySpacePartitioning1(InstanceReductionMixIn):

    '''
    This class is based on the prototype generation method developed by Sanchez 2012.
    This class was created by Joao Paulo and Rene. 
    '''

    def __init__(self, b=1):
        if b:
            self.b = b
        else:
            self.b = 1
            
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
            

    def RSP1(self):
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
    
            #Seleciona proxima divisao dentro de conjuntos
            if (z < b-1):
                maxDiam = -1
                proxConj = -1 #guarda o indice do proximo conjunto
                listaEhHomogeneo = [self.isHomogeneous(y,c) for c in conjuntos]
    
                aux = False in listaEhHomogeneo
                for c in range(0,len(conjuntos)):
                    if ((aux and listaEhHomogeneo[c] == False) or (aux == False)):
                        diametro,aux1,aux2 = self.diametroConj(sfDist,conjuntos[c],d)
                        if (diametro > maxDiam):
                            maxDiam = diametro
                            p1 = aux1
                            p2 = aux2
                            proxConj = c
    
                #atualiza p1 e p2
                conjuntoAtual = conjuntos[proxConj]
                conjuntos = conjuntos[:proxConj] + conjuntos[proxConj +1:]
                
        return self.centroidsConjuntos(X,y,conjuntos) 


    def reduce_data(self):
        prototypes, prototypes_labels = self.RSP1()
        self.prototypes = np.asarray(prototypes)
        self.prototypes_labels = np.asarray(prototypes_labels)
        self.reduction_ratio = 1 - float(len(self.prototypes_labels)) / len(self.y)
        print "REDUCTION: ", self.reduction_ratio
        return self.prototypes, self.prototypes_labels
