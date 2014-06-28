# -*- coding: utf-8 -*-
"""
Federal University of Pernambuco
Center of Informatic
Authors: Tiago José & Tiago Neves
Algorithm: ENPC
"""

import numpy  as np
import random as rd
from sklearn.neighbors import NearestNeighbors


class ENPC:
    
    #Flags
    _zeroInClasse = False
    
    #Dados de entrada
    setInstancias       = None          # Conjunto de instancias
    setClassesIntancias = None          # Classes de cada instancia

    #Dados internos
    V  = None                           # Guarda o Vij
    R  = None                           # Dados sobre o prototipo    
    Rclasses = None                     # Guarda a classe dos prototipos    
    quality  = None                     # Guarda o quality de cada prototipo
    
    #Informacoes de quantidade
    qtdClasses    = None                # Guarda a quantidade de classes
    qtdAtributos  = None                # Guarda a quantidade de atributos
    qtdInstancias = None                # Guarda o numero de instancias que o conjunto de treinamento tem
    qtdInitialPrototypes = None         # Quantidade de prototipos que o algoritmo comecou
    
    #Parametros
    maxIteracoes = 10                   # Numero maximo que o algoritmo iterara
    K = 3                               # KNN


    """
    " Construtor
    """
    def __init__( self, X, y, K, maxIteracoes ):
        
        if( 0 in y ):
            self._zeroInClasse = True
            y = y + 1

        y_sort = np.sort(y)
        self.qtdClasses = y_sort[-1]
        
        self.setInstancias = X
        self.setClassesIntancias = y        
        
        self.qtdInstancias = np.size( self.setInstancias, 0 )
        self.qtdAtributos  = np.size( self.setInstancias, 1 )

        self.maxIteracoes = maxIteracoes
        self.K = K


    """
    " Funções auxiliares
    """
    def printInformation( self ):
        
        print "Valor de V: "
        print self.V
        print "Valor de R: "
        print self.R
        print "Valor de Classes de R: "
        print self.Rclasses
        print "Qualidades de R: "
        print self.quality
        print "\n\n\n"


    def preencherMatrizComListaVazia( self, matriz ):
        
        m, n = np.size( matriz, 0 ), np.size( matriz, 1 )        
        i, j = 0, 0        
        
        while( i < m ):
            j = 0
            while( j < n ):
                matriz[i][j] = []            
                j = j + 1            
            i = i + 1


    def prune( self ):
        
        prot, n = 0, np.size( self.R, 0 )
        while( prot < n ):
            
            if( self.sumRi( prot ) == 0 ):
                self.V = np.delete( self.V, prot, 0 )
                self.R = np.delete( self.R, prot, 0 )
                self.Rclasses = np.delete( self.Rclasses, prot, 0 )
                self.quality  = np.delete( self.quality, prot, 0 )
                n = n - 1
                prot = prot - 1
            
            prot = prot + 1


    def sumRi( self, i ):
        
        sum, j = 0, 0   
        while( j < self.qtdClasses ):
            sum = sum + np.size( self.V[i][j] )
            j = j + 1
            
        return sum


    def getRandomPrototypes( self ):
        
        maxQtdInitialPrototypes = 20
        self.qtdInitialPrototypes = rd.randrange( 2, min(self.qtdInstancias, maxQtdInitialPrototypes+1) )
        
        listPrototipesAdd = []        
        
        prot = 1
        while(  prot <  self.qtdInitialPrototypes ):
            
            i = rd.randrange(0, self.qtdInstancias-1)

            if( i not in listPrototipesAdd ):
                
                listPrototipesAdd = listPrototipesAdd + [i]
                
                self.R        = np.concatenate( ( self.R       , [np.copy( self.setInstancias[i] )]       ), 0 )
                self.Rclasses = np.concatenate( ( self.Rclasses, [np.copy( self.setClassesIntancias[i] )] ), 0 )
            
            prot = prot + 1


    def getQuality( self ):
            
        prot, n = 0, np.size( self.R, 0 )
        while( prot < n ):
            self.quality[prot] = self.Quality( prot, self.Rclasses[prot] )
            prot = prot + 1


    """
    " Funções do artigo
    """
    def distancia( self, v1, v2 ):
        
        v = np.subtract( v1, v2 )
        v = np.multiply( v, v )
        v = np.sum( v )
        v = np.sqrt( v )
        
        return v


    def centroide( self, Vij ):
        
        v = np.zeros_like( self.setInstancias[Vij[0]] )        
        nearsLista = np.copy( Vij )
        
        for indice in nearsLista:
            v = np.add( v, self.setInstancias[indice] )
            
        v = np.divide( v, np.size( Vij ) )

        return v


    def Expectation( self, jclasse ):
        
        regions = 0
        Sj = 0
        
        numPrototipos = np.size( self.R, 0 )
        
        i = 0
        while( i < numPrototipos ):
            if( self.Rclasses[i] == jclasse ):
                regions = regions + 1
            i = i + 1
        
        i = 0
        while( i < numPrototipos ):
            Sj = Sj + np.size( self.V[i][jclasse-1] )
            i = i + 1
        
        return float(Sj)/float(regions)


    def Apportation( self, iPrototipo, jclasse ):
        
        sumVij = np.size( self.V[iPrototipo][jclasse-1] )
        
        return float(2*sumVij)/float(self.Expectation(jclasse))


    def Accuracy( self, iPrototipo, jclasse ):
        
        sumVij = np.size( self.V[iPrototipo][jclasse-1] )
        sumRi  = 0
        
        classe = 0
        while( classe < self.qtdClasses ):
            sumRi = sumRi + np.size( self.V[iPrototipo][classe] )
            classe = classe + 1

        return float(sumVij)/float(sumRi)


    def Quality( self, iPrototipo, jClasse ):
        
        return float( min( 1, self.Apportation( iPrototipo, jClasse ) * self.Accuracy( iPrototipo, jClasse ) ) )

    def getInformation( self ):

        i, j = None, None
        inst, prot = 0, 0
        dist, dMin = 0, 0

        n, m = np.size( self.R, 0 ), self.qtdInstancias

        self.quality = np.empty( n )
        self.V = np.empty([n, self.qtdClasses], list)
        self.preencherMatrizComListaVazia( self.V )

        while( inst < m ):
            
            dMin = float("inf")
            prot = 0
            
            while( prot < n ):          
            
                dist = self.distancia( self.R[prot], self.setInstancias[inst] )
                if( dist < dMin ):
                    
                    dMin = dist
                    i = prot
            
                prot = prot + 1
            
            j = self.setClassesIntancias[inst] - 1
            
            self.V[i][j] = self.V[i][j] + [inst]
            
            inst = inst + 1
            
        self.prune( )
        self.getQuality( )


    def mutation( self ):
        
        i, j, qtd, qtdMax = 0, 0, 0, 0
        
        classe = -1
        
        qtdPrototipos = np.size( self.R, 0 )
        
        while( i < qtdPrototipos ):
            
            j, qtdMax = 0, -float("inf")
            while( j < self.qtdClasses ):
                
                qtd = np.size( self.V[i][j] )
                
                if( qtd > qtdMax ):
                    classe = j+1
                    qtdMax = qtd
                    
                j = j + 1
            
            if( qtdMax != np.size( self.V[i][self.Rclasses[i]-1], 0 ) ):
                self.Rclasses[i] = classe
                
            i = i + 1


    def reprodution( self ):
        
        i, j, soma, sizeRoleta, classe = 0, 0, 0, 0, None
        qtdPrototipos = np.size( self.R, 0 )
        
        while( i < qtdPrototipos ):
            
            sizeRoleta, j = 0, 0
            while( j < self.qtdClasses ):
                sizeRoleta = sizeRoleta + np.size( self.V[i][j] )            
                j = j + 1
                
            roleta = rd.randrange(0, sizeRoleta)
            
            j, soma = 0, 0
            while( j < self.qtdClasses ):
                soma = soma + np.size( self.V[i][j] )
                if( roleta < soma ):
                    classe = j + 1
                    break
                j = j + 1
            
            j = classe - 1
            
            if( classe != self.Rclasses[i] ):
                newPrototipo = self.centroide( self.V[i][j] )
                self.R = np.concatenate( (self.R, [newPrototipo]), 0 )
                self.Rclasses = np.concatenate( ( self.Rclasses, [classe] ), 0 )
            
            i = i + 1


    def fight( self ):
        
        if( np.size( self.R, 0 ) > 1 ):

            qMax, iDesafiador, jPrototipo, jdesafiador = 0, None, None, None

            neigh = NearestNeighbors( )
            neigh.fit( self.R )
            neighbors = neigh.kneighbors( self.R, self.K, False )
            
            neighbors = np.delete( neighbors, 0, 1 )
            
            for iPrototipo, prototipo in enumerate( self.R ):
                
                qMax, iDesafiador = -float("inf"), None
                for iNeighbor in neighbors[iPrototipo]:
                    
                    difQuality = abs( self.quality[iPrototipo] - self.quality[iNeighbor] )
                    
                    if( difQuality  > qMax ):
                        
                        qMax = difQuality
                        iDesafiador = iNeighbor
                
                randon_num = rd.random()
                
                if( randon_num < qMax ):
                    
                    jPrototipo, jdesafiador = self.Rclasses[iPrototipo] - 1, self.Rclasses[iDesafiador] - 1
                    
                    if( self.Rclasses[iDesafiador] != self.Rclasses[iPrototipo] ):
                                                
                        self.V[iPrototipo][jPrototipo] = self.V[iPrototipo][jPrototipo] + self.V[iDesafiador][jPrototipo]
                        self.V[iDesafiador][jPrototipo] = []
                        
                    else:
                        
                        sizeRoleta = int((self.quality[iPrototipo] + self.quality[iDesafiador])*100)
                        randon_num = rd.randrange(0, sizeRoleta)
                        
                        if( randon_num < int(self.quality[iPrototipo]*100) ):
                            
                            self.V[iPrototipo][jPrototipo]  = self.V[iPrototipo][jPrototipo] + self.V[iDesafiador][jPrototipo]
                            self.V[iDesafiador][jPrototipo] = []
                        else:
                            self.V[iDesafiador][jdesafiador] = self.V[iDesafiador][jdesafiador] + self.V[iPrototipo][jdesafiador]
                            self.V[iPrototipo][jdesafiador]  = []


    def move( self ):
        
        iPrototipo, jClasse, kAtributo = 0, 0, 0
        qtdPrototipos = np.size( self.R, 0 )
        
        while( iPrototipo < qtdPrototipos ):
            
            jClasse = self.Rclasses[iPrototipo]
            
            if( np.size( self.V[iPrototipo][jClasse-1] ) > 0 ):                
                self.R[iPrototipo] = self.centroide( self.V[iPrototipo][jClasse-1] )
            
            iPrototipo = iPrototipo + 1


    def die( self ):
        
        qtdPrototipos, pDie = np.size( self.R, 0 ), 0

        prot = 0
        while( prot < qtdPrototipos ):
            
            if( self.quality[prot] > 0.5 ):
                pDie = 0.0
            else:
                pDie = 1 - 2*self.quality[prot]
            
                randon_num = rd.random()
            
                if( randon_num < pDie ):
                    self.R = np.delete( self.R, prot, 0 )
                    self.Rclasses = np.delete( self.Rclasses, prot, 0 )
                    qtdPrototipos = qtdPrototipos - 1

            prot = prot + 1


    def run_ENPC( self ):
        
        rd.seed(None)
        
        i = rd.randrange(0, self.qtdInstancias-1)

        self.R = np.empty( [1, np.size(self.setInstancias[0])] )
        self.Rclasses = np.empty( 1 )
        self.R[0] = np.copy( self.setInstancias[i] )
        self.Rclasses[0] = np.copy( self.setClassesIntancias[i] )
        
        self.getRandomPrototypes()

        iteracoes = 1

        while( iteracoes <= self.maxIteracoes ):
        
            ## Get Information
            #print "Get Information: \n"
            if( np.size( self.R, 0 ) == 0 ):
                self.getRandomPrototypes()
            self.getInformation()
            #self.printInformation()

            ## Mutation
            #print "Mutation: \n"
            self.mutation()
            #self.printInformation()
            
            ## Reproduction
            #print "Reproduction: \n"
            self.reprodution()
            self.getInformation()
            #self.printInformation()
            
            ## Fight
            #print "Fight: \n"
            self.fight()
            self.prune()
            #self.printInformation()
            
            ## Move
            #print "Move: \n"
            self.move()
            self.getQuality()
            #self.printInformation()
            
            ## Die
            #print "Die: \n"
            self.die()
            #self.printInformation()
        
            iteracoes = iteracoes + 1
            
            
        if( self._zeroInClasse ):
            self.Rclasses = self.Rclasses - 1


    """
    " Obtenção do resultado
    """
    def getResult( self ):
        
        #print "Terminou"
        print "Quantidade inicial de prototipos: ", self.qtdInitialPrototypes
        print "Quantidade final de prototipos: ", np.size( self.R, 0 ), "\n"
        return self.R, self.Rclasses