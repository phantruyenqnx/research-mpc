import numpy as np

class MPCfunction:
    ''' The following functions interact with the main file'''

    def __init__(self, A, B, C, MPC_HP_LEN, MPC_HU_LEN, Q_Val, R_Val):
        ''' Load the constants that do not change'''
        # Matrix CPSI     {(MPC_HP_LEN*SS_Z_LEN), SS_X_LEN};
        # Matrix COMEGA   {(MPC_HP_LEN*SS_Z_LEN), SS_U_LEN};
        # Matrix CTHETA   {(MPC_HP_LEN*SS_Z_LEN), (MPC_HU_LEN*SS_U_LEN)};

        # Matrix DU       {(MPC_HU_LEN*SS_U_LEN), 1};

        # Matrix A        {SS_X_LEN, SS_X_LEN};
        # Matrix B        {SS_X_LEN, SS_U_LEN};
        # Matrix C        {SS_Z_LEN, SS_X_LEN}; 

        # Matrix Q        {(MPC_HP_LEN*SS_Z_LEN), (MPC_HP_LEN*SS_Z_LEN)};
        # Matrix R        {(MPC_HU_LEN*SS_U_LEN), (MPC_HU_LEN*SS_U_LEN)};
        SS_X_LEN = A.shape[0]
        SS_U_LEN = B.shape[1]
        SS_Z_LEN = C.shape[0]

        CPSI = np.zeros(((MPC_HP_LEN*SS_Z_LEN), SS_X_LEN))
        COMEGA = np.zeros(((MPC_HP_LEN*SS_Z_LEN), SS_U_LEN))

        for i in range(1, MPC_HP_LEN+1):
            # print(C.dot(np.linalg.matrix_power(A,i)))
            CPSI[:][(i-1)*SS_Z_LEN: (i-1)*SS_Z_LEN + 2] = C.dot(np.linalg.matrix_power(A,i))
            tempCOMEGA = B
            for j in range(1, i):
                tempCOMEGA += np.linalg.matrix_power(A, j).dot(B)
            COMEGA[:][(i-1)*SS_Z_LEN: (i-1)*SS_Z_LEN + SS_Z_LEN] = C.dot(tempCOMEGA)
            # print(C.dot(np.linalg.matrix_power(A,i)))
            # np.append(CPSI, C.dot(np.linalg.matrix_power(A,i)))
        
        # print((SS_X_LEN, SS_U_LEN, SS_Z_LEN))

        CTHETA = np.zeros(((MPC_HP_LEN*SS_Z_LEN), (MPC_HU_LEN*SS_U_LEN)))
        zerosCTHETA = np.zeros(C.dot(B).shape)
        ''' khi row = 1 : column = 1 : CTHETA_cột 1 = từng phần tử của COMEGA'''

        for row in range (0, MPC_HP_LEN):
            for col in range(0, MPC_HU_LEN):
                mrow = row*SS_Z_LEN
                mcol = col*SS_U_LEN
                # CTHETA[mrow : mrow + SS_Z_LEN , mcol : mcol + SS_U_LEN] 
                for i in range(row, MPC_HP_LEN):
                    CTHETA[i, mcol] = 

        return None

# def init():
#     # dinh nghia kich thuoc cac ma tran can dung

#     A = []
#     B = []

import numpy as np
import matplotlib.pyplot as plt

class SupportFilesDrone:
    ''' The following functions interact with the main file'''

    def __init__(self):
        ''' Load the constants that do not change'''

        # Constants (Astec Hummingbird)
        Ix = 0.0034 # kg*m^2
        Iy = 0.0034 # kg*m^2
        Iz  = 0.006 # kg*m^2
        m  = 0.698 # kg
        g  = 9.81 # m/s^2
        Jtp=1.302*10**(-6) # N*m*s^2=kg*m^2
        Ts=0.1 # s

        # Matrix weights for the cost function (They must be diagonal)
        Q=np.matrix('10 0 0;0 10 0;0 0 10') # weights for outputs (all samples, except the last one)
        S=np.matrix('20 0 0;0 20 0;0 0 20') # weights for the final horizon period outputs
        R=np.matrix('10 0 0;0 10 0;0 0 10') # weights for inputs

        ct = 7.6184*10**(-8)*(60/(2*np.pi))**2 # N*s^2
        cq = 2.6839*10**(-9)*(60/(2*np.pi))**2 # N*m*s^2
        l = 0.171 # m

        controlled_states=3 # Number of attitude outputs: Phi, Theta, Psi
        hz = 4 # horizon period

        innerDyn_length=4 # Number of inner control loop iterations

        # The poles
        px=np.array([-1,-2])
        py=np.array([-1,-2])
        pz=np.array([-1,-2])

        # # Complex poles
        # px=np.array([-0.1+0.3j,-0.1-0.3j])
        # py=np.array([-0.1+0.3j,-0.1-0.3j])
        # pz=np.array([-1+1.3j,-1-1.3j])

        r=2
        f=0.025
        height_i=5
        height_f=25

        pos_x_y=0 # Default: 0. Make positive x and y longer for visual purposes (1-Yes, 0-No). It does not affect the dynamics of the UAV.
        sub_loop=5 # for animation purposes
        sim_version=2 # Can only be 1 or 2 - depending on that, it will show you different graphs in the animation

        # Drag force:
        drag_switch=0 # Must be either 0 or 1 (0 - drag force OFF, 1 - drag force ON)

        # Drag force coefficients [-]:
        C_D_u=1.5
        C_D_v=1.5
        C_D_w=2.0

        # Drag force cross-section area [m^2]
        A_u=2*l*0.01+0.05**2
        A_v=2*l*0.01+0.05**2
        A_w=2*2*l*0.01+0.05**2

        # Air density
        rho=1.225 # [kg/m^3]
        trajectory=7 # Choose the trajectory: only from 1-9
        no_plots=0 # 0-you will see the plots; 1-you will skip the plots (only animation)

        self.constants=[Ix, Iy, Iz, m, g, Jtp, Ts, Q, S, R, ct, cq, l, controlled_states, hz, innerDyn_length, px, py, pz, r, f, height_i, height_f,pos_x_y, sub_loop, sim_version, drag_switch, C_D_u, C_D_v, C_D_w, A_u, A_v, A_w, rho, trajectory, no_plots]

        return None

    def trajectory_generator(self,t):
        '''This method creates the trajectory for a drone to follow'''

        Ts=self.constants[6]
        innerDyn_length=self.constants[15]
        r=self.constants[19]
        f=self.constants[20]
        height_i=self.constants[21]
        height_f=self.constants[22]
        trajectory=self.constants[34]
        d_height=height_f-height_i

        # Define the x, y, z dimensions for the drone trajectories
        alpha=2*np.pi*f*t

        if trajectory==1 or trajectory==2 or trajectory==3 or trajectory==4:
            # Trajectory 1
            x=r*np.cos(alpha)
            y=r*np.sin(alpha)
            z=height_i+d_height/(t[-1])*t

            x_dot=-r*np.sin(alpha)*2*np.pi*f
            y_dot=r*np.cos(alpha)*2*np.pi*f
            z_dot=d_height/(t[-1])*np.ones(len(t))

            x_dot_dot=-r*np.cos(alpha)*(2*np.pi*f)**2
            y_dot_dot=-r*np.sin(alpha)*(2*np.pi*f)**2
            z_dot_dot=0*np.ones(len(t))

            if trajectory==2:
                # Trajectory 2
                # Make sure you comment everything except Trajectory 1 and this bonus trajectory
                x[101:len(x)]=2*(t[101:len(t)]-t[100])/20+x[100]
                y[101:len(y)]=2*(t[101:len(t)]-t[100])/20+y[100]
                z[101:len(z)]=z[100]+d_height/t[-1]*(t[101:len(t)]-t[100])

                x_dot[101:len(x_dot)]=1/10*np.ones(len(t[101:len(t)]))
                y_dot[101:len(y_dot)]=1/10*np.ones(len(t[101:len(t)]))
                z_dot[101:len(z_dot*(t/20))]=d_height/(t[-1])*np.ones(len(t[101:len(t)]))

                x_dot_dot[101:len(x_dot_dot)]=0*np.ones(len(t[101:len(t)]))
                y_dot_dot[101:len(y_dot_dot)]=0*np.ones(len(t[101:len(t)]))
                z_dot_dot[101:len(z_dot_dot)]=0*np.ones(len(t[101:len(t)]))

            elif trajectory==3:
                # Trajectory 3
                # Make sure you comment everything except Trajectory 1 and this bonus trajectory
                x[101:len(x)]=2*(t[101:len(t)]-t[100])/20+x[100]
                y[101:len(y)]=2*(t[101:len(t)]-t[100])/20+y[100]
                z[101:len(z)]=z[100]+d_height/t[-1]*(t[101:len(t)]-t[100])**2

                x_dot[101:len(x_dot)]=1/10*np.ones(len(t[101:len(t)]))
                y_dot[101:len(y_dot)]=1/10*np.ones(len(t[101:len(t)]))
                z_dot[101:len(z_dot)]=2*d_height/(t[-1])*(t[101:len(t)]-t[100])

                x_dot_dot[101:len(x_dot_dot)]=0*np.ones(len(t[101:len(t)]))
                y_dot_dot[101:len(y_dot_dot)]=0*np.ones(len(t[101:len(t)]))
                z_dot_dot[101:len(z_dot_dot)]=2*d_height/t[-1]*np.ones(len(t[101:len(t)]))

            elif trajectory==4:
                # Trajectory 4
                # Make sure you comment everything except Trajectory 1 and this bonus trajectory
                x[101:len(x)]=2*(t[101:len(t)]-t[100])/20+x[100]
                y[101:len(y)]=2*(t[101:len(t)]-t[100])/20+y[100]
                z[101:len(z)]=z[100]+50*d_height/t[-1]*np.sin(0.1*(t[101:len(t)]-t[100]))

                x_dot[101:len(x_dot)]=1/10*np.ones(len(t[101:len(t)]))
                y_dot[101:len(y_dot)]=1/10*np.ones(len(t[101:len(t)]))
                z_dot[101:len(z_dot)]=5*d_height/t[-1]*np.cos(0.1*(t[101:len(t)]-t[100]))

                x_dot_dot[101:len(x_dot_dot)]=0*np.ones(len(t[101:len(t)]))
                y_dot_dot[101:len(y_dot_dot)]=0*np.ones(len(t[101:len(t)]))
                z_dot_dot[101:len(z_dot_dot)]=-0.5*d_height/t[-1]*np.sin(0.1*(t[101:len(t)]-t[100]))

        elif trajectory==5 or trajectory==6:
            if trajectory==5:
                power=1
            else:
                power=2

            if power == 1:
                # Trajectory 5
                r_2=r/15
                x=(r_2*t**power+r)*np.cos(alpha)
                y=(r_2*t**power+r)*np.sin(alpha)
                z=height_i+d_height/t[-1]*t

                x_dot=r_2*np.cos(alpha)-(r_2*t+r)*np.sin(alpha)*2*np.pi*f
                y_dot=r_2*np.sin(alpha)+(r_2*t+r)*np.cos(alpha)*2*np.pi*f
                z_dot=d_height/(t[-1])*np.ones(len(t))

                x_dot_dot=-r_2*np.sin(alpha)*4*np.pi*f-(r_2*t+r)*np.cos(alpha)*(2*np.pi*f)**2
                y_dot_dot=r_2*np.cos(alpha)*4*np.pi*f-(r_2*t+r)*np.sin(alpha)*(2*np.pi*f)**2
                z_dot_dot=0*np.ones(len(t))
            else:
                # Trajectory 6
                r_2=r/500
                x=(r_2*t**power+r)*np.cos(alpha)
                y=(r_2*t**power+r)*np.sin(alpha)
                z=height_i+d_height/t[-1]*t

                x_dot=power*r_2*t**(power-1)*np.cos(alpha)-(r_2*t**(power)+r)*np.sin(alpha)*2*np.pi*f
                y_dot=power*r_2*t**(power-1)*np.sin(alpha)+(r_2*t**(power)+r)*np.cos(alpha)*2*np.pi*f
                z_dot=d_height/(t[-1])*np.ones(len(t))

                x_dot_dot=(power*(power-1)*r_2*t**(power-2)*np.cos(alpha)-power*r_2*t**(power-1)*np.sin(alpha)*2*np.pi*f)-(power*r_2*t**(power-1)*np.sin(alpha)*2*np.pi*f+(r_2*t**power+r_2)*np.cos(alpha)*(2*np.pi*f)**2)
                y_dot_dot=(power*(power-1)*r_2*t**(power-2)*np.sin(alpha)+power*r_2*t**(power-1)*np.cos(alpha)*2*np.pi*f)+(power*r_2*t**(power-1)*np.cos(alpha)*2*np.pi*f-(r_2*t**power+r_2)*np.sin(alpha)*(2*np.pi*f)**2)
                z_dot_dot=0*np.ones(len(t))

        elif trajectory==7:
            # Trajectory 7
            x=2*t/20+1
            y=2*t/20-2
            z=height_i+d_height/t[-1]*t

            x_dot=1/10*np.ones(len(t))
            y_dot=1/10*np.ones(len(t))
            z_dot=d_height/(t[-1])*np.ones(len(t))

            x_dot_dot=0*np.ones(len(t))
            y_dot_dot=0*np.ones(len(t))
            z_dot_dot=0*np.ones(len(t))

        elif trajectory==8:
            # Trajectory 8
            x=r/5*np.sin(alpha)+t/100
            y=t/100-1
            z=height_i+d_height/t[-1]*t

            x_dot=r/5*np.cos(alpha)*2*np.pi*f+1/100
            y_dot=1/100*np.ones(len(t))
            z_dot=d_height/(t[-1])*np.ones(len(t))

            x_dot_dot=-r/5*np.sin(alpha)*(2*np.pi*f)**2
            y_dot_dot=0*np.ones(len(t))
            z_dot_dot=0*np.ones(len(t))

        elif trajectory==9:
            # Trajectory 9
            wave_w=1
            x=r*np.cos(alpha)
            y=r*np.sin(alpha)
            z=height_i+7*d_height/t[-1]*np.sin(wave_w*t)

            x_dot=-r*np.sin(alpha)*2*np.pi*f
            y_dot=r*np.cos(alpha)*2*np.pi*f
            z_dot=7*d_height/(t[-1])*np.cos(wave_w*t)*wave_w

            x_dot_dot=-r*np.cos(alpha)*(2*np.pi*f)**2
            y_dot_dot=-r*np.sin(alpha)*(2*np.pi*f)**2
            z_dot_dot=-7*d_height/(t[-1])*np.sin(wave_w*t)*wave_w**2

        else:
            print("You only have 9 trajectories. Choose a number from 1 to 9")
            exit()

        # Vector of x and y changes per sample time
        dx=x[1:len(x)]-x[0:len(x)-1]
        dy=y[1:len(y)]-y[0:len(y)-1]
        dz=z[1:len(z)]-z[0:len(z)-1]

        dx=np.append(np.array(dx[0]),dx)
        dy=np.append(np.array(dy[0]),dy)
        dz=np.append(np.array(dz[0]),dz)


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

        return x, x_dot, x_dot_dot, y, y_dot, y_dot_dot, z, z_dot, z_dot_dot, psiInt
