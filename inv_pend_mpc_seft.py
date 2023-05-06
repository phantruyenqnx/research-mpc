import numpy as np
import matplotlib 
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib.patches import Rectangle

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

print(A)
print(B)
print(C)
print(D)

# Roi rac hoa phuong trình trạng thái theo phương pháp ZOH.
Ts = 0.01
Ad = np.identity(A.shape[1]) + Ts * A
Bd = Ts*B
Cd = C
Dd = D

# Cd = np.array([[1, 2, 3, 4],
#                [5, 6, 7, 8]])
# Ad = np.array([[1, 2, 3, 4],
#                [5, 6, 7, 8],
#                [9, 10, 11, 12],
#                [13, 14, 15, 16]])
# Bd = np.array([[10],
#                [11],
#                [12],
#                [13]])

# print(Cd.dot(Bd + Ad.dot(Bd)))


# print(Ad)
# print(Bd)
# print(Cd)
# print(Dd)

## Calculate prediction of z(k+1..k+Hp) constants ************************************************
# Prediction of state variable of the system
# z(k+1..k+Hp) = (CPSI)*x(k) + (COMEGA)*u(k-1) + (CTHETA)*dU(k..k+Hu-1)         ...{MPC_1}
# CPSI   = [CA C(A^2) ... C(A^Hp)]'                                       : (Hp*N)xN
Hu = 2
Hp = Hu
CPSI = np.array([Cd.dot(Ad), Cd.dot(np.linalg.matrix_power(Ad, 2))])

# COMEGA = [CB C(B+A*B) ... C*Sigma(i=0->Hp-1)A^i*B]'                     : (Hp*N)xM
COMEGA = np.array([Cd.dot(Bd), Cd.dot(Bd + Ad.dot(Bd))])

#  *          CTHETA = [         CB                0  ....           0              ]
#  *                   [       C(B+A*B)           CB   .             0              ]
#  *                   [           .               .    .           CB              ] : (Hp*N)x(Hu*M)
#  *                   [           .               .     .           .              ]
#  *                   [C*Sigma(i=0->Hp-1)(A^i*B)  .  ....  C*Sigma(i=0->Hp-Hu)A^i*B]
CTHETA = np.array([[Cd.dot(Bd), np.zeros(Cd.dot(Bd).shape)],
                   [Cd.dot(Bd + Ad.dot(Bd)), Cd.dot(Bd)]])
                  
# print(CTHETA.shape)

x0 = np.array([[-1],
               [0],
               [0],
               [0]])
u = np.array([[0]])

SP = np.array([[[0], [0]],
               [[0], [0]]])

Q = np.array([[10, 0],
              [0, 10]])
R = np.array([[10, 0],
              [0, 10]])

E_k = SP - CPSI.dot(x0) - COMEGA.dot(u)

G = (2 * CTHETA).T 
G = G.dot(Q)
G = G.dot(E_k)
print(CTHETA.shape)

H = CTHETA.T
H = H.dot(Q).dot(CTHETA) + R
print(H.shape)

dU_k_optimal = np.linalg.inv(H).dot(G)
dU_k_optimal = dU_k_optimal * 1/2
print(dU_k_optimal.shape)


figure = plt.figure()
ax = figure.add_subplot(111, 
                        autoscale_on=False,
                        xlim=(-1.5, 1.5),
                        ylim=(-0.5, 2))
# ax.set_aspect('equal')

ax.grid()
patch = ax.add_patch(Rectangle((0, 0), 0, 0, linewidth=1, edgecolor='k', facecolor='g'))

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

cart_width = 0.3
cart_height = 0.2

def init():
    line.set_data([], [])
    time_text.set_text('')
    patch.set_xy((-cart_width/2, -cart_height/2))
    patch.set_width(cart_width)
    patch.set_height(cart_height)
    return line, time_text, patch


def animate(i):
    thisx = [i, i]
    thisy = [0, 0.7]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*0.01))
    patch.set_x(i - cart_width/2)
    return line, time_text, patch

ani = ani.FuncAnimation(figure, animate, np.arange(0, 2),
                              interval=25, blit=True, init_func=init)

plt.show()