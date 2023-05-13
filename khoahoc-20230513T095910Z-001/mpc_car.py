import numpy as np
import matplotlib.pyplot as plt



class mpc_custom:
    ''' The following functions interact with the main file''' 

    def discrete_matrix(self, V, Ts):
        Ad = np.zeros((2,2))
        Bd = np.zeros((2, 1))
        Cd = np.zeros((1,2))
        Ad[0][0] = 1
        Ad[1][0] = V*Ts
        Ad[1][1] = 1
        Bd[0][0] = Ts
        Bd[1][0] = 0.5*V*Ts*Ts
        Cd[0][1] = 1
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
    
    def f_yt(x,y):
        return x+y

    def open_loop_new_states(self, at, yt):
        x = yt
        # Runge-Kutta method
        x_or=x
        Ts_pos=2
        for j in range(0,4):
            x_dot=pos_vel_fixed[0]
            # Save the slopes:
            if j == 0:
                x_dot_k1=x_dot
            elif j == 1:
                x_dot_k2=x_dot

            elif j == 2:
                x_dot_k3=x_dot
                Ts_pos=1
            else:
                x_dot_k4=x_dot
            if j<3:
                # New states using k_x
                x=x_or+x_dot*Ts/Ts_pos
            else:
                # New states using average k_x
                x=x_or+1/6*(x_dot_k1+2*x_dot_k2+2*x_dot_k3+x_dot_k4)*Ts
        # End of Runge-Kutta method

        # Take the last states
        new_states[6]=x
        return new_states

V = 0.5
Ts = 0.1
mpc = mpc_custom()
Ad, Bd, Cd = mpc.discrete_matrix(V, Ts)

# Load the initial state vector
at = 0.0
yt = 1.0

U1 = 0.0
xt_ref = 1.5

# # Generate the refence signals
t=np.arange(0,10+Ts*4,Ts*4) # time from 0 to 100 seconds, sample time (Ts=0.4 second)
plotl=len(t) # Number of outer control loop iterations

controlled_states = 1

Xt_ref=np.transpose([xt_ref*np.ones(4+1)])
# Create a reference vector
refSignals=np.zeros(len(Xt_ref)*controlled_states)
k=0
for i in range(0,len(refSignals),controlled_states):
    refSignals[i]=Xt_ref[k]
    k=k+1
# Initiate the controller - simulation loops
k=0 # for reading reference signals
hz = 4
for j in range(0,7):
    k=0
    hz = 4
   
    for i in range(0,4):

        # x_aug_t= np.transpose([np.concatenate((np.array([at, yt]).flatten(),[U1]),axis=0)])
        x_aug_t=np.transpose([np.concatenate(([at,yt],[U1]),axis=0)])
        # print(type(x_aug_t))
        # x_aug_t = np.matrix([[at], [yt], [U1]])
        # print(x_aug_t.shape)
        k = k + controlled_states
        # print(len(refSignals))
        if k+controlled_states*hz<=len(refSignals):
            r=refSignals[k:k+controlled_states*hz]
        else:
            r=refSignals[k:len(refSignals)]
            hz=hz-1
        # print(r)
        Hdb,Fdbt,Cdb,Adc = mpc.mpc(Ad, Bd, Cd, hz, 0.10, 0.20, 0.10, 1)
        # print(np.transpose(x_aug_t)[0][0:len(x_aug_t)].shape)
        ft=np.matmul(np.concatenate((np.transpose(x_aug_t)[0][0:len(x_aug_t)],r),axis=0), Fdbt)

        du=-np.matmul(np.linalg.inv(Hdb),np.transpose([ft]))
        
        # Update the real inputs
        U1=U1+du[0][0]

        # update states

        # states = Ad.dot(states) + Bd * U1
        at =  at + U1*Ts
        yt =  yt + at*V*Ts + U1*0.5*Ts*Ts
        # x_dot_t =  states[1]
        # thetat =  states[2]
        # theta_dot_t =  states[3]

        print(U1)


# for i_global in range(0,plotl-1):
#     # references

#     Xt_ref=np.transpose([xt_ref*np.ones(4+1)])
#     # Create a reference vector
#     refSignals=np.zeros(len(Xt_ref)*controlled_states)
#     k=0
#     for i in range(0,len(refSignals),controlled_states):
#         refSignals[i]=Xt_ref[k]
#         k=k+1
#     # Initiate the controller - simulation loops
#     k=0 # for reading reference signals
#     hz = 4

#     # print(refSignals)

#     for i in range(0,4):

#         # x_aug_t= np.transpose([np.concatenate((np.array([at, yt]).flatten(),[U1]),axis=0)])
#         x_aug_t=np.transpose([np.concatenate(([at,yt],[U1]),axis=0)])
#         # print(type(x_aug_t))
#         # x_aug_t = np.matrix([[at], [yt], [U1]])
#         # print(x_aug_t.shape)
#         k = k + controlled_states
#         # print(len(refSignals))
#         if k+controlled_states*hz<=len(refSignals):
#             r=refSignals[k:k+controlled_states*hz]
#         else:
#             r=refSignals[k:len(refSignals)]
#             hz=hz-1

#         Hdb,Fdbt,Cdb,Adc = mpc.mpc(Ad, Bd, Cd, hz, 0.10, 0.20, 0.10, 1)
#         # print(np.transpose(x_aug_t)[0][0:len(x_aug_t)].shape)
#         ft=np.matmul(np.concatenate((np.transpose(x_aug_t)[0][0:len(x_aug_t)],r),axis=0), Fdbt)

#         du=-np.matmul(np.linalg.inv(Hdb),np.transpose([ft]))
#         # Update the real inputs
#         U1=U1+du[0][0]

#         # update states

#         # states = Ad.dot(states) + Bd * U1
#         at =  at + U1*Ts
#         yt =  yt + at*V*Ts + U1*0.5*Ts*Ts
#         # x_dot_t =  states[1]
#         # thetat =  states[2]
#         # theta_dot_t =  states[3]

#         print(yt)

