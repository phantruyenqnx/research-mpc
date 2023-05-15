import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

class mpc_custom:
    ''' The following functions interact with the main file''' 

    def mpc(self, Ad, Bd, Cd, hz, Q_val, S_val, R_val, U_shape):
        '''This function creates the compact matrices for Model Predictive Control'''
        # db - double bar
        # dbt - double bar transpose
        # dc - double circumflex
        A_aug=np.concatenate((Ad,Bd),axis=1)
        temp1=np.zeros((np.size(Bd,1),np.size(Ad,1)))
        temp2=np.identity(np.size(Bd,1))
        temp=np.concatenate((temp1,temp2),axis=1)

        A_aug=np.concatenate((A_aug,temp),axis=0)
        B_aug=np.concatenate((Bd,np.identity(np.size(Bd,1))),axis=0)
        C_aug=np.concatenate((Cd,np.zeros((np.size(Cd,0),np.size(Bd,1)))),axis=1)

        Q = Q_val*np.ones((C_aug.shape))
        Q = np.diag(np.diag(Q))
        S = S_val*np.ones(C_aug.shape)
        S = np.diag(np.diag(S))
        R = R_val*np.ones((U_shape, U_shape))
        R = np.diag(np.diag(R))

        # print(S)
        # print(R)
        # Q=self.constants[7]
        # S=self.constants[8]
        # R=self.constants[9]

        CQC=np.matmul(np.transpose(C_aug),Q)
        CQC=np.matmul(CQC,C_aug)

        CSC=np.matmul(np.transpose(C_aug),S)
        CSC=np.matmul(CSC,C_aug)

        QC=np.matmul(Q,C_aug)
        SC=np.matmul(S,C_aug)


        Qdb=np.zeros((np.size(CQC,0)*hz,np.size(CQC,1)*hz))
        Tdb=np.zeros((np.size(QC,0)*hz,np.size(QC,1)*hz))
        Rdb=np.zeros((np.size(R,0)*hz,np.size(R,1)*hz))
        Cdb=np.zeros((np.size(B_aug,0)*hz,np.size(B_aug,1)*hz))
        Adc=np.zeros((np.size(A_aug,0)*hz,np.size(A_aug,1)))

        for i in range(0,hz):
            if i == hz-1:
                Qdb[np.size(CSC,0)*i:np.size(CSC,0)*i+CSC.shape[0],np.size(CSC,1)*i:np.size(CSC,1)*i+CSC.shape[1]]=CSC
                Tdb[np.size(SC,0)*i:np.size(SC,0)*i+SC.shape[0],np.size(SC,1)*i:np.size(SC,1)*i+SC.shape[1]]=SC
            else:
                Qdb[np.size(CQC,0)*i:np.size(CQC,0)*i+CQC.shape[0],np.size(CQC,1)*i:np.size(CQC,1)*i+CQC.shape[1]]=CQC
                Tdb[np.size(QC,0)*i:np.size(QC,0)*i+QC.shape[0],np.size(QC,1)*i:np.size(QC,1)*i+QC.shape[1]]=QC

            Rdb[np.size(R,0)*i:np.size(R,0)*i+R.shape[0],np.size(R,1)*i:np.size(R,1)*i+R.shape[1]]=R

            for j in range(0,hz):
                if j<=i:
                    Cdb[np.size(B_aug,0)*i:np.size(B_aug,0)*i+B_aug.shape[0],np.size(B_aug,1)*j:np.size(B_aug,1)*j+B_aug.shape[1]]=np.matmul(np.linalg.matrix_power(A_aug,((i+1)-(j+1))),B_aug)

            Adc[np.size(A_aug,0)*i:np.size(A_aug,0)*i+A_aug.shape[0],0:0+A_aug.shape[1]]=np.linalg.matrix_power(A_aug,i+1)

        Hdb=np.matmul(np.transpose(Cdb),Qdb)
        
        Hdb=np.matmul(Hdb,Cdb)+Rdb

        temp=np.matmul(np.transpose(Adc),Qdb)
        temp=np.matmul(temp,Cdb)

        temp2=np.matmul(-Tdb,Cdb)

        Fdbt=np.concatenate((temp,temp2),axis=0)

        return Hdb,Fdbt,Cdb,Adc
    
    def model_system(self, Vk,thetak,dt):
        A = np.zeros((3,3))
        A[0][0] = 1.0
        A[1][1] = 1.0
        A[2][2] = 1.0
        A[0][2] = -Vk*dt*np.sin(thetak)
        A[1][2] = Vk*dt*np.cos(thetak)

        B = np.zeros((3, 2))
        B[0][0] = dt*np.cos(thetak)
        B[0][1] = -0.5*dt*dt*Vk*np.sin(thetak)
        B[1][0] = dt*np.sin(thetak)
        B[1][1] = 0.5*dt*dt*Vk*np.cos(thetak)
        B[2][1] = dt

        C = np.diag(np.diag(np.ones((3,3))))

        return A, B, C
    
    def trajectory_generator(self, t) :
        radius = 30
        period = 100
        x = np.empty_like(t)
        y = np.empty_like(t)
        print(t)
        for i in range(0, len(t)):
            x[i] = radius * np.sin(2 * np.pi * i / period)
            y[i] = -radius * np.cos(2 * np.pi * i / period)
        dx = x[1:len(t)] - x[0:len(t)-1]
        dy = y[1:len(t)] - y[0:len(t)-1]
        # psi = np.zeros((1, len(x)))
        psi = np.empty_like(t)*0.0
        psiInt = psi

        psi[0] = np.arctan2(dy[0],dx[0])
        psi[1:len(x)] = np.arctan2(dy[0:len(dy)],dx[0:len(dx)])

        dpsi   = psi[1:len(x)] - psi[0:len(x)-1]
        psiInt[0] = psi[0]

        for i in range(1, len(x)):
            if dpsi[i - 1 : i]< -np.pi :
                psiInt[i] = psiInt[i - 1] + (dpsi[i - 1] + 2 * np.pi)
            elif dpsi[i - 1] > np.pi :
                psiInt[i] = psiInt[i - 1] + (dpsi[i - 1] - 2 * np.pi)
            else:
                psiInt[i] = psiInt[i - 1] + dpsi[i - 1] 
        x_ref   = x.T
        y_ref   = y.T
        psi_ref = psiInt.T
        return x_ref, y_ref, psi_ref
    
    def trajectory_generator_1(self, t) :  
        x=20*t/20+1
        y=20*t/20-2
        dx = x[1:len(t)] - x[0:len(t)-1]
        dy = y[1:len(t)] - y[0:len(t)-1]
        dx=np.append(np.array(dx[0]),dx)
        dy=np.append(np.array(dy[0]),dy)
        # Define the reference yaw angles
        psi=np.zeros(len(x))
        psiInt=psi
        psi[0]=np.arctan2(y[0],x[0])+np.pi/2
        psi[1:len(psi)]=np.arctan2(dy[1:len(dy)],dx[1:len(dx)])

        # We want the yaw angle to keep track the amount of rotations
        dpsi=psi[1:len(psi)]-psi[0:len(psi)-1]
        psiInt[0]=psi[0]
        for i in range(1,len(psiInt)):
            if dpsi[i-1]<-np.pi:
                psiInt[i]=psiInt[i-1]+(dpsi[i-1]+2*np.pi)
            elif dpsi[i-1]>np.pi:
                psiInt[i]=psiInt[i-1]+(dpsi[i-1]-2*np.pi)
            else:
                psiInt[i]=psiInt[i-1]+dpsi[i-1]

        x_ref   = x.T
        y_ref   = y.T
        psi_ref = psiInt.T
        return x_ref, y_ref, psi_ref 

def nonlinear_lateral_car_model(z, t, u):
    dxdt = u[0]*np.cos(z[2])
    dydt = u[0]*np.sin(z[2])
    dthetadt = u[1]

    dzdt = np.array([dxdt, dydt, dthetadt]).flatten()

    return dzdt 

mpc = mpc_custom()

''' Setup and Parameters '''
dt = 0.1
nx = 3 # number of states
ny = 3 # number of ovservation
nu = 2 # number of input

S = np.diag(np.diag(np.ones((3,3))))
Q = np.diag(np.diag(np.ones((3,3))))
R = np.diag(np.diag(np.ones((2,2)).dot(0.1)))
N = 10

Vx = 4.5
thetak = 0.0

# Load the trajectory
t=np.arange(0, 10+dt, dt)

x_ref, y_ref, psi_ref = mpc.trajectory_generator(t);    

sim_length = len(t) # Number of control loop iterations

refSignals = np.zeros((len(x_ref) * ny, 1)).flatten()
ref_index = 0
for i in range(0, len(refSignals), ny):
    refSignals[i] = x_ref[ref_index]
    refSignals[i+1] = y_ref[ref_index]
    refSignals[i+2] = psi_ref[ref_index]
    ref_index = ref_index + 1


# % initial state
x0 = np.zeros((nx, 1))
x0[0][0] = x_ref[0] + 15
x0[1][0] = y_ref[0] + 15
x0[2][0] = psi_ref[0]
# x0 = [x_ref(1, 1); y_ref(1, 1); psi_ref(1, 1)]; % Initial state of mobile robot

Ad, Bd, Cd = mpc.model_system(Vx, thetak, dt)

A_aug=np.concatenate((Ad,Bd),axis=1)
temp1=np.zeros((np.size(Bd,1),np.size(Ad,1)))
temp2=np.identity(np.size(Bd,1))
temp=np.concatenate((temp1,temp2),axis=1)

A_aug=np.concatenate((A_aug,temp),axis=0)
B_aug=np.concatenate((Bd,np.identity(np.size(Bd,1))),axis=0)
C_aug=np.concatenate((Cd,np.zeros((np.size(Cd,0),np.size(Bd,1)))),axis=1)


xTrue = np.zeros((3, 1))
# xTrue = np.hstack((xTrue, np.ones((3,1))))
# xTrue = np.hstack((xTrue, np.ones((3,1))))
xTrue =  x0
uk = np.zeros((2, 1))
du = np.zeros((2, 1))
current_step = 0
ref_sig_num = 0

for i in range(0, sim_length - 15):
    current_step = current_step + 1

    Ad, Bd, Cd = mpc.model_system(Vx, thetak, dt)

    xTrue_aug = np.vstack((xTrue[:, current_step - 1 : current_step], uk[:, current_step - 1 : current_step]))

    ref_sig_num = ref_sig_num + ny
    if ref_sig_num + ny * N  <= len(refSignals) :
        ref = refSignals[ref_sig_num:ref_sig_num+ny*N]
    else:
        ref = refSignals[ref_sig_num:len(refSignals)]
        N = N - 1

    Hdb,Fdbt,Cdb,Adc = mpc.mpc(Ad, Bd, Cd, N, 1.0, 1.0, 0.1, 2)

    ft=np.matmul(np.concatenate((np.transpose(xTrue_aug)[0][0:len(xTrue_aug)], ref),axis=0), Fdbt)
    _du=-np.matmul(np.linalg.inv(Hdb),np.transpose([ft]))

    du = np.hstack((du, np.ones((2,1))))

    du[:, current_step : current_step + 1] = _du[0 : nu, : ]
#     % add du input
#     uk(:, current_step) = uk(:, current_step - 1) + du(:, current_step)
    uk = np.hstack((uk, np.ones((2,1))))
    uk[:, current_step : current_step + 1] = uk[:, current_step - 1 : current_step] + du[:, current_step : current_step + 1]
#     % update state
    # T = dt*i:dt:dt*i+dt
#     T = np.array([dt*i, dt*i+dt])
#     z = xTrue[:, current_step - 1 : current_step].T.flatten()
#     z = odeint(nonlinear_lateral_car_model, z,  T, args= (uk[:, current_step - 1 : current_step], ) )
# #     [T, x] = ode45(@(t,x) nonlinear_lateral_car_model(t, x, uk(:, current_step)), T, xTrue(:, current_step - 1));
# #     xTrue(:, current_step) = x(end,:);
# #     X = xTrue(:, current_step);
# #     thetak = X(3);
#     _temp = np.zeros((3, 1))
#     _temp[0] = z[1][0]
#     _temp[1] = z[1][1]
#     _temp[2] = z[1][2]
    xTrue = np.hstack((xTrue, np.ones((3,1))))
    xTrue[:, current_step : current_step + 1] = Ad.dot(xTrue[:, current_step - 1 : current_step]) + Bd.dot(uk[:, current_step : current_step + 1])
    thetak = xTrue[:, current_step : current_step + 1][2]
    # uk = np.hstack((uk, np.ones((2,1)) * i))
x_0 = xTrue[1, 0 : xTrue.shape[1]]
# print(xTrue)
plt.plot(xTrue[0, 0 : xTrue.shape[1]], xTrue[1, 0 : xTrue.shape[1]])
plt.plot(x_ref, y_ref)
plt.show()
