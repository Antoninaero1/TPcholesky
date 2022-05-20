# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import time

def Cholesky(A):
    
    """
    IN: Prend en argument la matrice A symétrique et inversible
    OUT: Retourne la matrice L de la décomposition LLt de
    Cholesky
    """
    
    n = len(A)
    L = np.zeros((n,n))
    L[0][0] = (A[0][0])**(1/2)
    for j in range(1,n):
        L[j][0] = A[0][j]/L[0][0]
        for i in range(1,n):
            s1 = 0
            s2 = 0
            for k in range(0,i):
                s1 += ((L[i][k])**2)
                s2 += L[i][k]*L[j][k]
            L[i][i] = (A[i][i]-s1)**(1/2)
            if j>i:
                L[j][i] = (A[i][j]-s2)/L[i][i]  
    return L

def ResolCholesky(A,b):
    """
    IN: Prend en arguments la matrice A et le vecteur b
    OUT: Retourne le vecteur X de l'équation AX = b, en trouvant dans un 
    premier temps le vecteur Y de LY=b, puis en résolvant le système LtX=Y
    """
    n = len(A)
    L = Cholesky(A)
    L_ = np.transpose(L)
    Y = np.zeros((n,1))
    X = np.zeros((n,1))
    
    Y[0] = b[0]/L[0][0]
    for i in range(1,n):
        s = 0
        for k in range(0,i):
            s += L[i][k]*Y[k]
        Y[i] = (b[i]-s)/L[i][i]
    
    X[-1] = Y[-1]/L_[-1][-1]
    for i in range(n-2,-1,-1):
        s = 0
        for k in range(i+1, n):
            s += L_[i][k]*X[k]
        X[i] = (Y[i] - s)/L_[i][i]

    return X

def CholeskyAlternative(A):
    
    """
    IN: Prend en argument la matrice A
    OUT: Retourne les matrices L et D de la décomposition de Cholesky
    alternative de la matrice A qui s'écrit A = LDLt
    """
    
    n = len(A)
    L = np.identity(n) 
    D = np.zeros((n,n))
    
    autorise = True
    
    if A[0][0] == 0:
        autorise = False
    else:
        for j  in range(0,n):
            s = 0      
            for k in range(0,j):
                s += (L[j, k] ** 2) *D[k,k] 
            D[j, j] = A[j,j] - s
            
            if D[j, j]== 0:
                autorise = False
                
            for i in range(j + 1, n):
                if i>j:
                    s = 0
                    for k in range(0, j):
                        s += L[i, k] * L[j, k]*D[k,k] 
                    L[i, j] = (A[i, j] - s) / D[j, j]
    
    if autorise == True:
        return L, D
    else:
        return "La matrice ne peut pas être décomposée sous forme LDLt"


def ResolCholeskyAlternative(A,b):
    
    """
    IN: Prend en arguments la matrice A et le vecteur b
    OUT: Retiurne le vecteur X provenant du système AX = b, en resolvant dans
    un premier temps le système LY = b, puis le système DL*X = Y
    """

    n = len(A)
    L = CholeskyAlternative(A)[0]
    L_ = np.transpose(L)
    D = CholeskyAlternative(A)[1]
    Y = np.zeros((n,1))
    X = np.zeros((n,1))
    
    Y[0] = b[0]/L[0][0]
    for i in range(1,n):
        s = 0
        for k in range(0,i):
            s += L[i][k]*Y[k]
        Y[i] = (b[i]-s)/L[i][i]
        
    U = np.dot(D,L_)
    X[-1] = Y[-1]/L_[-1][-1]
    for i in range(n-2,-1,-1):
        s = 0
        for k in range(i+1, n):
            s += U[i][k]*X[k]
        X[i] = (Y[i] - s)/U[i][i]
    
    return X


def vérification_cholesky(A):
    
    """
    IN: Prend en argument la matrice A
    OUT: Confirme si la matrice A est bien égale à la matrice que l'on obtient
    si on fait la multiplication LLt
    """
    
    L = Cholesky(A)
    L_ = L.transpose()
    R = np.dot(L,L_) #B = L*Lt
    if np.allclose(A, R, rtol=1e-05, atol=1e-08): #vérification que A = L*Lt
        return "A est bien égale à L*Lt"
    else:
        return "A n'est pas égale à L*Lt"
    
    
    
    
    

def vérification_ResolCholesky(A,b):
    
    """
    IN: Prend en arguments la matrice A et le vecteur b
    OUT: Confirme si la multiplication AX = b est égale à ce qu'on trouve avec 
    la fonction ResolCholesky, en utilisant la fonction de la librairie numpy 
    "linalg.solve()"
    """
    
    x = ResolCholesky(A,b)
    x_theorique = np.linalg.solve(A,b)
    if np.allclose(x_theorique, x, rtol=1e-05, atol=1e-08):
        return "x est bien la solution de l'équation Ax=b"
    else:
        return "x n'est pas la solution à Ax=b"
    
    
    
    
    
    
def vérification_CholeskyAlternative(A):
    
    """
    IN: Prend en argument la matrice A
    OUT: Confirme si la forme LDLt que nous avons trouver grâce à la fonction
    CholeskyAlternative est égale à la matrice A
    """
    
    C = CholeskyAlternative(A)
    B = np.dot(C[0],C[1])
    B = np.dot(B,np.transpose(C[0]))
    if np.allclose(A, B, rtol=1e-05, atol=1e-08):
        return "A est bien égale à LDLt"
    else:
        return "A n'est pas égale à LDLt"
    
    
    
    
    
    
def vérification_ResolCholeskyAlternative(A,b):
    
    """
    IN: Prend en argument la matrice A et le vecteur b
    OUT: Confirme si la multiplication AX = b est égale à ce qu'on trouve avec 
    la fonction ResolCholeskyAlternative, en utilisant la fonction de la 
    librairie numpy "linalg.solve()"
    """
    
    x = ResolCholeskyAlternative(A,b)
    x_theorique = np.linalg.solve(A,b)
    if np.allclose(x_theorique, x, rtol=1e-05, atol=1e-08): 
        return "x est bien la solution de l'équation Ax=b"
    else:
        return "x n'est pas la solution à Ax=b"
    
    



def temps(A,b):
    
    """
    IN: Prend en arguments la matrice A et le vecteur b
    OUT: Retourne le temps que prend l'ordinateur pour réaliser la résolution
    du système AX = b par notre fonction ResolCholesky et la fonction numpy
    "linalg.solve()"
    """
    
    start = time.process_time_ns()
    x = ResolCholesky(A,b)
    end = time.process_time_ns()
    start1 = time.process_time_ns()
    x_theorique = np.linalg.solve(A,b)
    end1 = time.process_time_ns()
    start2 = time.process_time_ns()
    x2 = ResolCholeskyAlternative(A,b)
    end2 = time.process_time_ns()
    return "cholesky: ", end-start,"cholesky alternatif",end2-start2,"fonction:", end1-start1

    
def matrice_def_pos_syme (n):
    A=abs(np.random.random(size=(n,n)))
    At=np.transpose(A)
    M=np.dot(A,At)
    b=abs(np.random.random(size=(n,)))
    return M,b


import sys
print(sys.float_info.epsilon)

A,B=matrice_def_pos_syme(10)




print(vérification_cholesky(A))
print(vérification_ResolCholesky(A, B))
print(vérification_CholeskyAlternative(A))
print(vérification_ResolCholeskyAlternative(A, B))
print(temps(A,B))