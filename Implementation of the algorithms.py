import numpy as np
import time
import heapq
import random
from collections import defaultdict
import itertools


#traitement des données afin de les rendre utilisables


def scenario(k):
    K = []
    with open("scenario" + str(k)+".txt") as f:
        u = f.readline()
        v = f.readline()
        for line in f:
            K.append([line.strip()])
        for j in range(len(K)):
            K[j] = K[j][0].split()
        for L in K:
            for i in range(len(L)):
                L[i] = int(eval(L[i]))
    return(int(eval(u)),int(eval(v)),np.array(K))


def aff(scenar):
    #affiche proprement la matrice d'adjacence pour un scénario donné
    M = scenar[2]
    n = len(M[0])
    forme = ""
    print("\n")
    for i in range(n):
        forme += "{" + str(i) + ":^3}"
    for elem in M:
        print(forme.format(*elem))
    print("\n")


def cree_scenarios():
    global s1
    global s2
    global s3
    global s4
    global s5
    global s6
    s1 = scenario(1)
    s2 = scenario(2)
    s3 = scenario(3)
    s4 = scenario(4)
    s5 = scenario(5)
    s6 = scenario(6)


#question 2---------------------------------------------------------------------
#K est l'ensemble des k premiers sommets du graphe
#K = {0,...,k-1}


def ret_non_term(scenar):
    t1 = time.time()
    k,n,M = scenar                  #M matrice d'adjacence pondérée du graphe G à n sommets
    ind_to_delete = []
    for i in range(k,n):            #pour chaque sommet non-terminal
        nb = non_nuls(M[i])[0]
        if nb == 1:
            ind_to_delete.append(i) #s'il est de degré 1, on l'ajoute aux sommets à supprimer
    for j in reversed(ind_to_delete):
        M = np.delete(M,j,0)        #on parcourt les indices dans l'ordre décroissant, donc il n'y a pas de décalagé d'indices au fur et à mesure qu'on supprime des lignes et des colonnes
        M = np.delete(M,j,1)
    t2 = time.time()
    #print(t2-t1)
    return k,n-len(ind_to_delete),M



def non_nuls(L):
    #renvoie le nombre d'éléments non nuls d'une liste
    #et les indices des 2 éléments non nuls s'il y en a 2
    c = 0
    tmp = ()
    first_found = False
    tmp2 = ()
    for ind, elem in enumerate(L):
        if elem != 0:
            if not first_found:
                tmp2 = ind
                first_found = True
            c+=1
            tmp = ind
    if c == 2:
        return (c, (tmp2, tmp))
    else:
        return (c, () )



#question 4---------------------------------------------------------------------


def non_term_deg2(scenar):
    t1 = time.time()
    k,n,M = scenar                  #M matrice d'adjacence pondérée du graphe G à n sommets
    ind_to_delete = []
    for h in range(k,n):            #pour chaque sommet non-terminal
        nb,u = non_nuls(M[h])
        if nb == 2:
            i = u[0]
            j = u[1]
            if M[i][j] > 0:         #s'il y a une arête entre les deux sommets i et j, on applique les conditions de la question
                if M[h][i] + M[h][j] >= M[i][j]:
                    ind_to_delete.append(h)
                else:
                    M[i][j] = 0
    for j in reversed(ind_to_delete):
        M = np.delete(M,j,0)
        M = np.delete(M,j,1)
    t2 = time.time()
    #print(t2-t1)
    return (k,n-len(ind_to_delete),M)


#question 5---------------------------------------------------------------------
#on a d'abord besoin d'une fonction, dist, qui calcule l'ensemble des distances entre chaque sommet ;
#on calcule cette matrice des distances grâce à l'algorithme de Floyd-Warshall

def dist(M):
    n = M.shape[0]
    D = np.asarray(M,dtype = float)
    D[D == 0] = np.inf
    for l in range(n):
        D[l,l] = 0.0

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if D[i,j] > D[i,k] + D[k,j]:
                    D[i,j] = D[i,k] + D[k,j]
    D[D == np.inf] = float(1e8)
    return np.asarray(D,dtype=int)


#puis, on applique l'algorithme dont le pseudo-code est détaillé question 5


def long_edges(scenar):
    k,n,M = scenar
    D = dist(M)
    for i in range(n):
        for j in range(i,n):
            if M[i][j] > 0 and M[i][j] > D[i][j]:
                M[i][j] = 0
                M[j][i] = 0
    return k,n,M


#---------------------------------------------------------------------


# ------------------ MISE À JOUR DES SCÉNARIOS -----------

# met à jour les scénatios en appliquant successivement les trois algorithmes des 'Pre-proccesing steps', jusqu'à ce qu'on ne détecte plus aucun changement

def update(scenar):
    k,n,M = scenar
    u = ret_non_term(scenar)        #question 1
    u = non_term_deg2(u)            #question 3
    u = long_edges(u)               #question 5
    while u != scenar:
        scenar = u
        u = ret_non_term(scenar)
        u = non_term_deg2(u)
        u = long_edges(u)
    return u

# ---------------------------------------------------------------------

#Fonctions auxiliaires utiles pour les questions suivantes


def graphe_induit(M, T): #renvoie le graphe induit par l'ensemble de sommets T
    n = M.shape[0]
    nouv_M = np.copy(M)
    for i in range(n):
        for j in range(n):
            if i not in T or j not in T:
                nouv_M[i][j] = 0
    return nouv_M


def floyd_warshall_chemins(M):
    n = M.shape[0]
    D = np.asarray(M,dtype = float)
    D[D == 0] = np.inf
    P = [[[] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                P[i][j] = [i, j]
    # Algorithme de Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if D[i,j] > D[i,k] + D[k,j]:
                    D[i,j] = D[i,k] + D[k,j]
                    P[i][j] = P[i][k] + P[k][j][1:]
    for i in range(n):
        D[i][i] = 0
        P[i][i] = [i]
    return (D, P) #P[i][j] contient la liste des sommets du plus court chemin de i à j


def kruskal(M): #renvoie les arêtes sélectionnées
    n = len(M)
    D = np.asarray(M,dtype = float)
    D[D == 0] = np.inf
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if D[i][j] != float('inf'):
                edges.append((D[i][j], i, j))
    edges.sort()
    parents = list(range(n))
    def find(i):
        if parents[i] != i:
            parents[i] = find(parents[i])
        return parents[i]
    def union(i, j):
        pi, pj = find(i), find(j)
        parents[pi] = pj
    mst = []
    for w, i, j in edges:
        pi, pj = find(i), find(j)
        if pi != pj:
            union(i, j)
            mst.append((i, j))
    return mst


def arbre_couvrant_minimal(M): #utilise Kruskal pour renvoyer un arbre couvrant minimal
    n = M.shape[0]
    MST = np.zeros((n,n))
    mst = kruskal(M)
    for (i,j) in mst:
        MST[i][j] = M[i][j]
        MST[j][i] = M[i][j]
    return(MST)


def poids(M):
    #Renvoie le poids d'un graphe (en particulier pour un Steiner tree)
    n = M.shape[0]
    poids = 0
    for i in range(n):
        for j in range(i+1, n):
            poids += M[i][j]
    return(poids)


#Question 7---------------------------------------------------------------------


def all_possible_branching_points(n, k):
    #Renvoie toutes les combinaisons de branching points de taille k-2
    return set(itertools.combinations(range(k, n), k - 2))


def Enumeration(scenar):
    k, n, M = update(scenar)
    D, P = floyd_warshall_chemins(M) #Plus courts chemins
    poids_minimal = np.inf
    for S in all_possible_branching_points(D.shape[0], k):
        T = set(S).union(set(range(k)))
        arbre_possible = arbre_couvrant_minimal(graphe_induit(D, T))
        weight = poids(arbre_possible)
        if weight < poids_minimal:
            steiner_tree_min = arbre_possible
            poids_minimal = weight
    def remonter(M, ST):
        #Retrouve l'arbre de Steiner sur G à partir de celui sur le distance network D
        n = M.shape[0]
        steiner_tree = np.zeros((n,n))
        for i in range(len(ST)):
            for j in range(i+1, len(ST)):
                if ST[i][j] != 0 :
                    for l in range(len(P[i][j])-1):
                        steiner_tree[P[i][j][l],P[i][j][l+1]] = M[P[i][j][l],P[i][j][l+1]]
                        steiner_tree[P[i][j][l+1],P[i][j][l]] = M[P[i][j][l],P[i][j][l+1]]
        return(steiner_tree)
    steiner_tree = remonter(M, steiner_tree_min)
    return k, n, steiner_tree


#question 10---------------------------------------------------------------------


#list(itertools.combinations(E,k)) renvoie une liste contenant l'ensemble des sous-ensembles à k éléments de E


def subsets(my_set):
    #renvoie l'ensemble des sous-ensembles d'une liste.d'un ensemble... sous forme d'une liste
    result = [()]
    for x in my_set:
        result = result + [y + (x,) for y in result]
    return result


def DW(scenar):
    t1 = time.time()
    k,n,M = scenar
    D = dist(M)
    C = floyd_warshall_chemins(M)[1]
    L = []
    for i in range(1,k+1):
        L.append(list(itertools.combinations(range(k),i)))
        #L[i] contient l'ensemble des sous-ensembles de K de cardinal i+1
    dico = {}
    for v in range(n):
        for elem in L[0]:
            dico[(elem,v)] = [ctu(C[min(elem)][v]),D[min(elem)][v]]
            #min(elem) permet d'accéder au seul élément de elem qui est un ensemble à un élément. On n'utilise pas de listes car elles ne peuvent pas être utilisées comme clés pourun dictionnaire.
    for j in range(1,k):        #on itère sur X par taille croissante de X
        for X in L[j]:
            for v in range(n):
                val = np.inf
                X_0, w_0 = "nul","nul"   #recherche du minimum
                for w in range(n):
                    d = D[v,w]
                    for Xp in subsets(X):
                        if Xp != () and Xp != X:
                            valeur_test = d + dico[(Xp,w)][1] + dico[(comp(X,Xp),w)][1]
                            if valeur_test < val:
                                val = valeur_test
                                w_0 = w
                                X_0 = Xp
                dico[(X,v)] = [[C[w_0][v]] + dico[(X_0,w_0)][0] + dico[(comp(X,X_0),w_0)][0],val]
    t2 = time.time()
    print(t2-t1)
    return dico[(tuple(range(0,k)),0)]


def ctu(L):
    #convertit [e1,e2,e3,e4...] en [[e1,e2],[e2,e3],[e3,e4],...]
    R = []
    for i in range(len(L)-1):
        R.append([L[i],L[i+1]])
    return R


def comp(t1,t2):
    #renvoie t1\t2 en supposant t2 inclus dans t1
    t = ()
    for elem in t1:
        if elem not in t2:
            t += (elem,)
    return t


#question 12--------------------------------------------------------------------


def restriction(M, h):
    #Renvoie la sous-matrice constituée des h premières lignes et colonnes de M
    return(M[:h].transpose()[:h].transpose())


def distance_network_heuristic(scenar):
    t1 = time.time()
    h, n, M = update(scenar)
    D, P = floyd_warshall_chemins(M) #Distance network et Plus courts chemins
    DK = restriction(D, h) #On s'intéresse au graphe induit par K
    TD = kruskal(DK)
    M_nouv = np.zeros((n,n))
    for (i,j) in TD: #Ajout de toutes les chemins les plus courts au graphe
        for l in range(len(P[i][j])-1):
            M_nouv[P[i][j][l],P[i][j][l+1]] = M[P[i][j][l],P[i][j][l+1]]
            M_nouv[P[i][j][l+1],P[i][j][l]] = M[P[i][j][l],P[i][j][l+1]]
    T = kruskal(M_nouv)
    Steiner_tree = np.zeros((n,n))
    for (i,j) in T:
        Steiner_tree[i][j] = M[i][j]
        Steiner_tree[j][i] = M[i][j]
    t2 = time.time()
    print("dnh :" + str(t2-t1))
    return(ret_non_term((h, n, Steiner_tree)))


#Question 13--------------------------------------------------------------------


def shortest_path_distance(M, start, end):
    #Algo de Djikstra qui renvoie la plus courte distance et le chemin correspondant
    if start == end:
        return(0., [start])
    n = len(M)
    D = np.asarray(M,dtype = float)
    D[D == 0] = np.inf
    dist = [float('inf') for _ in range(n)]
    dist[start] = 0
    heap = [(0, start)]
    previous = defaultdict(lambda: -1)
    while heap:
        (curr_dist, curr) = heapq.heappop(heap)
        if curr == end:
            break
        for i in range(n):
            if D[curr][i] == float('inf'):
                continue
            if dist[i] > curr_dist + D[curr][i]:
                dist[i] = curr_dist + D[curr][i]
                heapq.heappush(heap, (dist[i], i))
                previous[i] = curr
    path = []
    if previous[end] == -1:
        return float('inf'), path
    curr = end
    while curr != start:
        path.append(curr)
        curr = previous[curr]
    path.append(start)
    return dist[end], path[::-1]


def shortest_path_heuristic(scenar):
    t1 = time.time()
    h, n, M = update(scenar)
    T = set() #Liste des terminaux actuels
    T.add(0)
    while not set(range(h)).issubset(T) : #Tant que tous les terminaux n'ont pas été ajoutés
        distance_plus_proche = float("inf")
        for terminal in range(h):
            if terminal not in T:
                for t in T:
                    distance, chemin = shortest_path_distance(M, terminal, t)
                    if distance < distance_plus_proche:
                        distance_plus_proche = distance
                        chemin_a_ajouter = chemin
        T = T.union(chemin_a_ajouter)
        subgraph = graphe_induit(M, T)
        MST = arbre_couvrant_minimal(subgraph)
        MST = ret_non_term_local(MST,T,h)
    t2 = time.time()
    print("sph :"  + str(t2-t1))
    return h, n, MST


def ret_non_term_local(MST,T,h):
    #On la réécrit car l'ensemble des terminaux "évolue" (ensemble T)
    n = MST.shape[0]
    ind_to_delete = []
    for i in range(h, n):
        nb = non_nuls(MST[i])[0]
        if nb == 1:
            ind_to_delete.append(i)         #s'il est de degré 1, on l'ajoute aux sommets à supprimer
            T.remove(i)     #... et on le supprime de T
    for j in reversed(ind_to_delete):
        MST = np.delete(MST,j,0)        #on parcourt les indices dans l'ordre décroissant, donc il n'y a pas de décalage d'indices au fur et à mesure qu'on supprime des lignes et des colonnes
        MST = np.delete(MST,j,1)
    return MST


#question_15---------------------------------------------------------------------


def eucl(p1,p2):
    #renvoie la distance euclidienne entre deux points
    x1,y1 = p1
    x2,y2 = p2
    return ((x1-x2)**2+(y1-y2)**2)**(1/2)


def scen_a(d,n):
    #génère un graphe aléatoire de n noeuds,
    #selon les modalités de la question 15
    #renvoie la matrice d'adjacence du graphe en question
    L = [] #L[i] va contenir les coordonnées x_i,y_i du i-ème point
    for _ in range(n):
        x = random.random()
        y = random.random()
        L.append((x,y))
    M = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            if eucl(L[i],L[j]) < d:
                M[i][j] = 1
                M[j][i] = 1
            elif eucl(L[i],L[j]) < 2*d:
                M[i][j] = 2
                M[j][i] = 2
            elif eucl(L[i],L[j]) < 3*d:
                M[i][j] = 3
                M[j][i] = 3
    return(M)

def do(n):
    M = scen_a(0.025*500/n,n)
    return shortest_path_heuristic((n//10,n,M)),distance_network_heuristic((n//10,n,M))

def statistix():
    for _ in range(8):
        sph,dnh = do(100)
        print(poids(sph[2]),poids(dnh[2]))
        print("\n")
    for _ in range(8):
        sph,dnh = do(200)
        print(poids(sph[2]),poids(dnh[2]))
        print("\n")








