import numpy as np

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