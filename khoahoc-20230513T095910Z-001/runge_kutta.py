# Python program to implement Runge Kutta method
# A sample differential equation "dy / dx = (x - y)/2"
def dydx(x, y):
	return ((x - y)/2)

# Finds value of y for a given x using step size h
# and initial value y0 at x0.
def rungeKutta(x0, y0, x, h):
	# Count number of iterations using step size or
	# step height h
	n = (int)((x - x0)/h)
	# Iterate for number of iterations
	y = y0
	for i in range(1, n + 1):
		"Apply Runge Kutta Formulas to find next value of y"
		k1 = h * dydx(x0, y)
		k2 = h * dydx(x0 + 0.5 * h, y + 0.5 * k1)
		k3 = h * dydx(x0 + 0.5 * h, y + 0.5 * k2)
		k4 = h * dydx(x0 + h, y + k3)

		# Update next value of y
		y = y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)

		# Update next value of x
		x0 = x0 + h
	return y

# Driver method
x0 = 0
y = 1
x = 2
h = 0.2
print ('The value of y at x is:', rungeKutta(x0, y, x, h))

# This code is contributed by Prateek Bhindwar

def model_vehicle(z,t,u):
    a = z[0]
    y = z[1]
    dadt = u
    dydt = V*a
    dzdt = [dadt,dydt]
    return dzdt

# function that returns dz/dt
def model(z,t,u):
    x = z[0]
    y = z[1]
    dxdt = (-x + u)/2.0
    dydt = (-y + x)/5.0
    dzdt = [dxdt,dydt]
    return dzdt

# initial condition
z0 = [0,0]

# number of time points
n = 1

# time points
t = np.linspace(0,4,n)

# step input
u = np.zeros(n)
# change to 2.0 at time = 5.0
u[50:] = 2.0

# store solution
x = np.empty_like(t)
y = np.empty_like(t)
# record initial conditions
x[0] = z0[0]
y[0] = z0[1]

# solve ODE
for i in range(1,n):
    # span for next time step
    tspan = [t[i-1],t[i]]
    # solve for next step
    z = odeint(model_vehicle,z0,tspan,args=(u[i],))
    # store solution for plotting
    x[i] = z[1][0]
    y[i] = z[1][1]
    # next initial condition
    z0 = z[1]

# plot results
plt.plot(t,u,'g:',label='u(t)')
plt.plot(t,x,'b-',label='x(t)')
plt.plot(t,y,'r--',label='y(t)')
plt.ylabel('values')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()