import func_basic_mpc_unconstrained as fMPC
import numpy as np

# khai bao cac dac diem cua he con lac nguoc 
M = 0.5
m = 0.2
b = 0.1
I = 0.006
g = 9.8
l = 0.3

p = I*(M+m)+M*m*l*l

# Tinh toán các ma trận A, B, C, D cho phương trình trạng thái
# x_dot = Ax + Bu
# y = Cx + D

A = np.array([[0, 1, 0, 0],
              [0, -(I+m*l*l)*b/p, (m*m*g*l*l)/p, 0],
              [0, 0, 0, 1],
              [0, -(m*l*b)/p , m*g*l*(M+m)/p ,0]])
B = np.array([[0],
              [(I+m*l*l)/p],
              [0],
              [m*l/p]])
C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])
D = np.array([[0],[0]])

# Roi rac hoa phuong trình trạng thái theo phương pháp ZOH.
A = np.array([[0, 1, 0, 0],
              [0, 2, 3, 0],
              [0, 0, 0, 1],
              [0, 2 ,3 ,0]])
B = np.array([[0],
              [1],
              [0],
              [1]])
Ts = 0.01
Ad = np.identity(A.shape[1]) + Ts * A
Bd = Ts*B
Cd = C
Dd = D

fmpc = fMPC.MPCfunction(A, B, Cd, 4, 3, 10.0, 0.3)