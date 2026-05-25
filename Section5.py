import numpy as np

A = np.array([[5, 4, 0], [4, 5, 0], [0, 0, 2], [0, 0, 1]])
B = np.array([[15, 14, 0], [14, 15, 0], [0, 0, 2], [0, 0, 1]])

Ua, Sa, Va = np.linalg.svd(A)
Ub, Sb, Vb = np.linalg.svd(B)

zeros_row = np.zeros(Sa.shape[0])
Sa[1:]=0
Sa = np.diag(Sa)
Sa = np.vstack((Sa, zeros_row))
Sb[1:]=0   
Sb = np.diag(Sb)
Sb = np.vstack((Sb, zeros_row))

print(Ua, '\n', Sa, '\n', Va)

A1 = np.matmul(Ua, np.matmul(Sa, Va))
B1 = np.matmul(Ub, np.matmul(Sb, Vb))

print("A1:", A1)
print("B1:", B1)
print("A approximation error: ", ((np.linalg.norm(A - A1, ord='fro'))))
print("B approximation error: ", ((np.linalg.norm(B - B1, ord='fro'))))

print("A approximation error normalized: ", ((np.linalg.norm(A - A1, ord='fro'))/(np.linalg.norm(A, ord='fro'))))
print("B approximation error normalized: ", ((np.linalg.norm(B - B1, ord='fro'))/(np.linalg.norm(B, ord='fro'))))