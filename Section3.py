import numpy as np


#### a ####
def GMQR(A: np.array) -> tuple[np.array, np.array]:
    a1 = A[:, 0]
    R = np.zeros((A.shape[1], A.shape[1]))
    R[0, 0] = np.linalg.norm(a1)
    q1 = a1/R[0, 0]
    Q_list = [q1]
    for i in range(1, A.shape[1], 1):
        qi = A[:, i]
        for j in range(0, i, 1):
            R[j, i] = np.dot(Q_list[j], A[:, i])
            qi = qi - R[j, i]*Q_list[j]
        R[i, i] = np.linalg.norm(qi)
        qi = qi/R[i, i]
        Q_list.append(qi)

    Q = np.column_stack(Q_list)
    return Q, R    

def MGMQR(A: np.array) -> tuple[np.array, np.array]:
    a1 = A[:, 0]
    R = np.zeros((A.shape[1], A.shape[1]))
    R[0, 0] = np.linalg.norm(a1)
    q1 = a1/R[0, 0]
    Q_list = [q1]
    for i in range(1, A.shape[1], 1):
        qi = A[:, i]
        for j in range(0, i, 1):
            R[j, i] = np.dot(Q_list[j], qi)
            qi = qi - R[j, i]*Q_list[j]
        R[i, i] = np.linalg.norm(qi)
        qi = qi/R[i, i]
        Q_list.append(qi)

    Q = np.column_stack(Q_list)
    return Q, R    

#### b ####
DTYPE = np.float32
n = 50
t = 2.0 ** np.arange(-8, 10, 2.0)
t = t.astype(DTYPE)
m = len(t)
A = ( np.random.rand(n, m).astype(DTYPE)
@ np.diag(t) @ np.random.rand(m, m).astype(DTYPE))

Q1c, R1c = GMQR(A)
Q2c, R2c = MGMQR(A)

#### c ####

print("GMQR: ", np.linalg.norm(Q1c.T @ Q1c - np.eye(m), ord='fro'))
print("MGMQR: ", np.linalg.norm(Q2c.T @ Q2c - np.eye(m), ord='fro'))

#It is clear that the MGS is much more numerically stable even for a very ill-conditioned matrix.



#### d ####

DTYPE = np.float64
n = 50
t = 2.0 ** np.arange(-8, 10, 2.0)
t = t.astype(DTYPE)
m = len(t)
A = ( np.random.rand(n, m).astype(DTYPE)
@ np.diag(t) @ np.random.rand(m, m).astype(DTYPE))

Q1d, R1d = GMQR(A)
Q2d, R2d = MGMQR(A)

print("GMQR: ", np.linalg.norm(Q1d.T @ Q1d - np.eye(m), ord='fro'))
print("MGMQR: ", np.linalg.norm(Q2d.T @ Q2d - np.eye(m), ord='fro'))

