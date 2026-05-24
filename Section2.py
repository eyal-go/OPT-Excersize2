import numpy as np

def BwdSub(U: np.array, b: np.array) -> np.array:
#Assume U is square upper triangular with no 0s on the diagonal

    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        row_sum = 0
        for j in range(i+1, n, 1):
            row_sum += U[i,j]*x[j]
        x[i] = (b[i] - row_sum)/U[i, i]
    
    return x


def FwdSub(L: np.array, b: np.array) -> np.array:
#Assume L is square lower triangular with no 0s on the diagonal

    n = L.shape[0]
    x = np.zeros(n)
    for i in range(0, n, 1):
        row_sum = 0
        for j in range(0, i, 1):
            row_sum += L[i,j]*x[j]
        x[i] = (b[i] - row_sum)/L[i,i]

    return x


A = np.array([[2, 1, 2], [1, -2, 1], [1, 2, 3], [1, 1, 1]])
b = np.array([6,1,5,2])
print(A)
AT = np.transpose(A)
ATA = np.linalg.matmul(AT, A)
L = np.linalg.cholesky(ATA)

########## b ##########
#We want to solve ATAx = ATb.
#We have ATA = LLT.
#Essentially, we will treat ATb as the new vector to solve.

c_b = np.matmul(AT, b)

#Now we have LLTx = c. first we'll solve Ly = b using FwdSub:
y_b = FwdSub(L = L, b = c_b)

#Next we will solve Ux = y, with U = LT with BwdSub to get the LS solution:
x_b = BwdSub(U = np.transpose(L), b = y_b)

print(x_b)

########## c ##########
#Using QR factorization, we saw in class that the LS solution is given by x = R^-1QTb.

Q, R = np.linalg.qr(A)

#First, we'll calculate QTb:

QTb = np.matmul(np.transpose(Q), b)

#Using the LS formula with the QR factorization, we can get Rx = QTb. We will use BwdSub to get x:

x_qr = BwdSub(R, QTb)

print(x_qr)