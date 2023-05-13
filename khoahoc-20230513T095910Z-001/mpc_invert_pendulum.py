import numpy as np
import matplotlib.pyplot as plt



class mpc_custom:
    ''' The following functions interact with the main file''' 

    def discrete_matrix(self, M, m, b, I, g, l, Ts):
        p = I*(M+m)+M*m*l*l

        A = np.zeros((4, 4))
        B = np.zeros((4, 1))
        C = np.zeros((2, 4))

        A[0][1] = 1
        A[1][1] = -(I+m*l*l)*b/p
        A[1][2] = (m*m*g*l*l)/p
        A[2][3] = 1
        A[3][1] = -(m*l*b)/p  
        A[3][2] = m*g*l*(M+m)/p

        B[1][0] =(I+m*l*l)/p
        B[3][0] = m*l/p

        C[0][0] = 1
        C[1][2] = 1
        # Discretize the system (Forward Euler)
        Ad=np.identity(np.size(A,1))+Ts*A
        Bd=Ts*B
        Cd=C
        return Ad, Bd, Cd

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
        # print(Q)
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
        # print(Rdb.shape)
        Hdb=np.matmul(Hdb,Cdb)+Rdb

        temp=np.matmul(np.transpose(Adc),Qdb)
        temp=np.matmul(temp,Cdb)

        temp2=np.matmul(-Tdb,Cdb)

        Fdbt=np.concatenate((temp,temp2),axis=0)

        return Hdb,Fdbt,Cdb,Adc

M = 0.5
m = 0.2
b = 0.1
I = 0.006
g = 9.8
l = 0.3
mpc = mpc_custom()
Ad, Bd, Cd = mpc.discrete_matrix(M, m, b, I, g, l, 0.1)

# Load the initial state vector
xt = 0
x_dot_t = 0
thetat = 0
theta_dot_t=0
states = np.zeros((4,1))
states[0] = xt
states[1] = x_dot_t
states[2] = thetat
states[3] = theta_dot_t
# states=np.array([xt, x_dot_t, thetat, theta_dot_t])

U1 = 0
UTotal=np.array([[U1]]) # 1 inputs

xt_ref = 1

# Generate the refence signals
t=np.arange(0,10+0.1*4,0.1*4) # time from 0 to 100 seconds, sample time (Ts=0.4 second)
plotl=len(t) # Number of outer control loop iterations

controlled_states = 2

for i_global in range(0,plotl-1):
    # references
    Xt_ref=np.transpose([xt_ref*np.ones(4+1)])
    # Create a reference vector
    refSignals=np.zeros(len(Xt_ref)*controlled_states)
    k=0
    for i in range(0,len(refSignals),controlled_states):
        refSignals[i]=Xt_ref[k]
        refSignals[i+1]=np.pi/2
        k=k+1
    # Initiate the controller - simulation loops
    k=0 # for reading reference signals
    hz = 4

    for i in range(0,4):

        x_aug_t=np.transpose([np.concatenate((np.array([xt, x_dot_t, thetat, theta_dot_t]).flatten(),[U1]),axis=0)])

        k = k + controlled_states
        # print(len(refSignals))
        if k+controlled_states*hz<=len(refSignals):
            r=refSignals[k:k+controlled_states*hz]
        else:
            r=refSignals[k:len(refSignals)]
            hz=hz-1

        Hdb,Fdbt,Cdb,Adc = mpc.mpc(Ad, Bd, Cd, hz, 10, 20, 10, 1)
        ft=np.matmul(np.concatenate((np.transpose(x_aug_t)[0][0:len(x_aug_t)],r),axis=0), Fdbt)

        du=-np.matmul(np.linalg.inv(Hdb),np.transpose([ft]))
        # Update the real inputs
        U1=U1+du[0][0]

        # update states

        states = Ad.dot(states) + Bd * U1
        xt =  states[0]
        x_dot_t =  states[1]
        thetat =  states[2]
        theta_dot_t =  states[3]

        print(xt)

