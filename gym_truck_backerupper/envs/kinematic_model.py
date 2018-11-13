import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from drawnow import drawnow
import pdb
import time

def kinematic_model(t, x, u):
    n = len(x)
    xd = np.zeros((n, 1))
    xd[0] = (v / L1) * np.tan(u)

    theta = x[0] - x[1]

    xd[1] = (v / L2) * np.sin(theta) - (h / L2) * xd[0] * np.cos(theta)
    
    vt = v * np.cos(theta) + h * xd[0] * np.sin(theta)

    xd[2] = v * np.cos(x[0])
    xd[3] = v * np.sin(x[0])

    xd[4] = vt * np.cos(x[1])
    xd[5] = vt * np.sin(x[1])

    return xd

def init():
    ''' '''
    tractor.set_data([], [])
    return tractor,

def animate(i):
    ''' '''
    H_c = L2 / 3.0
    x_trac = [x1[i]+L1, x1[i], x1[i], x1[i]+L1, x1[i]+L1]
    y_trac = [y1[i]+H_c/2, y1[i]+H_c/2, y1[i]-H_c/2, y1[i]-H_c/2, y1[i]+H_c/2]
    #tractor.set_data(x_trac, y_trac)
    ax.clear()
    ax.plot(x_trac, y_trac)
    ax.set_xlim(x2[i]-25, x2[i]+25)
    ax.set_ylim(y2[i]-25, y2[i]+25)
    return tractor,

def manual_ani():
    H_c = L2 / 3.0
    x_trac = [x1[i]+L1, x1[i], x1[i], x1[i]+L1, x1[i]+L1]
    y_trac = [y1[i]+H_c/2, y1[i]+H_c/2, y1[i]-H_c/2, y1[i]-H_c/2, y1[i]+H_c/2]
    ax.clear()
    ax.plot(x_trac, y_trac)
    ax.set_xlim(x2[i]-25, x2[i]+25)
    ax.set_ylim(y2[i]-25, y2[i]+25)
    plt.pause(np.finfo(np.float32).eps)

def make_fig():
    H_c = L2 / 3.0
    x_trac = [x1[i]+L1, x1[i], x1[i], x1[i]+L1, x1[i]+L1]
    y_trac = [y1[i]+H_c/2, y1[i]+H_c/2, y1[i]-H_c/2, y1[i]-H_c/2, y1[i]+H_c/2]
    plt.plot(x_trac, y_trac)
    plt.axis([x2[i]-25, x2[i]+25, y2[i]-25, y2[i]+25])

def gen():
    i = 0
    while i <= num_steps:
        i += 1
        yield i-1

if __name__ == '__main__':
    t0 = 0.0
    t_final = 80.0
    dt = .001
    num_steps = int((t_final - t0)/dt) + 1
    
    L1 = 5.7336
    L2 = 12.192
    h = -0.2286
    v = 25.0
    u = 0
    # psi_1, psi_2, x1, y1, x2, y2
    x0 = [np.radians(0), np.radians(0), h+L2, 0, 0, 0]
    
    solver = spi.ode(kinematic_model).set_integrator('dopri5')
    solver.set_initial_value(x0, t0)
    solver.set_f_params(u)

    t = np.zeros((num_steps, 1))
    psi_1 = np.zeros((num_steps, 1))
    psi_2 = np.zeros((num_steps, 1))
    x1 = np.zeros((num_steps, 1))
    y1 = np.zeros((num_steps, 1))
    x2 = np.zeros((num_steps, 1))
    y2 = np.zeros((num_steps, 1))

    t[0] = t0
    psi_1[0] = x0[0]
    psi_2[0] = x0[1]
    x1[0] = x0[2]
    y1[0] = x0[3]
    x2[0] = x0[4]
    y2[0] = x0[5]

    i = 1
    while solver.successful() and i < num_steps:
        if solver.t > 40.0:
            u = .0167
        solver.set_f_params(u)
        solver.integrate(solver.t + dt)
        t[i] = solver.t
        psi_1[i] = solver.y[0]
        psi_2[i] = solver.y[1]
        x1[i] = solver.y[2]
        y1[i] = solver.y[3]
        x2[i] = solver.y[4]
        y2[i] = solver.y[5]
        i += 1
    '''
    ax1 = plt.subplot(311)
    ax1.plot(t, np.degrees(psi_1))
    ax1.set_ylabel(r'$\psi_{1} [\degree]$')

    ax2 = plt.subplot(312
    ax2.plot(t, np.degrees(psi_2))
    ax2.set_ylabel(r'$\psi_{2} [\degree]$')

    ax3 = plt.subplot(313)
    ax3.plot(t, y2)
    ax3.set_ylabel('y [m]')
    ax3.set_xlabel('time [s]')

    fig1 = plt.figure()
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    plt.show()'''

    ## Funcanimation
    #fig = plt.figure()
    #ax = plt.axes(xlim=(-25, 25), ylim=(-25, 25))
    #tractor, = ax.plot([], [], lw=2)
    
    ## Funcanimation
    fig, ax = plt.subplots(1, 1)
    tractor, = ax.plot([], [], lw=2)

    anim = animation.FuncAnimation(fig, animate, init_func=init, 
                                   frames=range(num_steps)[::25], interval=1, 
                                   blit=False, repeat=False)
    ## Drawnow
    #fig = plt.figure()

    ## manual ani or Drawnow
    #fig, ax = plt.subplots(1, 1)
    
    for i in range(num_steps):
        #startTime = time.time()
        #drawnow(make_fig)
        #manual_ani()
        plt.pause(np.finfo(np.float32).eps)
        #renderTime = time.time() - startTime
        #print('{} FPS'.format(1 / renderTime))
