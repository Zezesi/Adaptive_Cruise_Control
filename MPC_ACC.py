import casadi as ca
import numpy as np
import math
from casadi import *
import matplotlib.pyplot as plt
from scipy.io import loadmat

Ts = 0.1  # sampling time [s]
tau = Ts  # time delay [s]
Lf = 1.2  # front wheel center to c.o.g. [m]
Lr = 1.6  # rear wheel center to c.o.g. [m]
L = Lf + Lr  # wheelbase [m]
Wt = 1.365  # wheel track [m]
# for drawing
Vw = 1.580  # vehicle width [m]
Lh = 2  # c.o.g. to the head [m]
Lt = 2.4  # c.o.g. to the tail [m]
Tr = 0.2888  # tire radius 185/60 R 14
Tw = 0.185  # tire width
Length = Lh + Lt  # vehicle length


def acc_transition(x, u, no, x_position):
    u=u  # u is the desired acceleration
    ap=no # no is the acceleration of the front vehicle
    x_next = np.zeros(5)
    x_next[0] = x[0] + Ts * (x[1]-x[2]+1/2*ap[0]*Ts-1/2*x[3]*Ts)
    x_next[1] = x[1] + Ts * ap[0]
    x_next[2] = x[2] + Ts * x[3]
    x_next[3] = x[3] - Ts * (x[3]/tau-u/tau)
    x_next[4] = u/tau- x[3]/tau

    # traveling distance of the two vehicles
    x_position_next = np.zeros(2)
    x_position_next[0] = x_position[0]+x[1]*Ts+1/2*ap[0]*Ts**2
    x_position_next[1] = x_position[1] + x[2]*Ts + 1 / 2 * x[3] * Ts**2
    return x_next, x_position_next


def draw_vehicle(x, ax, colort='black', colorb='red'):
    vehicle_outline = np.array(
        [[-Lt, Lh, Lh, -Lt, -Lt],
         [Vw / 2, Vw / 2, -Vw / 2, -Vw / 2, Vw / 2]])

    wheel = np.array([[-Tr, Tr, Tr, -Tr, -Tr],
                      [Tw / 2, Tw / 2, -Tw / 2, -Tw / 2,
                       Tw / 2]])

    rr_wheel = wheel.copy()  # rear right
    rl_wheel = wheel.copy()  # rear left
    fr_wheel = wheel.copy()  # front right
    fl_wheel = wheel.copy()  # front left


    fr_wheel += np.array([[Lf], [-Wt / 2]])
    fl_wheel += np.array([[Lf], [Wt / 2]])
    rr_wheel += np.array([[-Lr], [-Wt / 2]])
    rl_wheel += np.array([[-Lr], [Wt / 2]])

    fr_wheel[0, :] += x


    fl_wheel[0, :] += x


    rr_wheel[0, :] += x


    rl_wheel[0, :] += x


    vehicle_outline[0, :] += x


    ax.plot(fr_wheel[0, :], fr_wheel[1, :], colort)
    ax.plot(rr_wheel[0, :], rr_wheel[1, :], colort)
    ax.plot(fl_wheel[0, :], fl_wheel[1, :], colort)
    ax.plot(rl_wheel[0, :], rl_wheel[1, :], colort)

    ax.plot(vehicle_outline[0, :], vehicle_outline[1, :], colorb)


def read_acceleration_profile():
    # acceleration profile of front vehicle
    mat_file_path = 'wltp3.mat'
    acceleration_profile = loadmat(mat_file_path)
    acceleration_profile = acceleration_profile['accl']
    return acceleration_profile[0:5000]


if __name__ == "__main__":
    Ts = 0.1  # sampling time [s]
    N=15 # predict horizon length
    tau = Ts  # time delay [s]
    v_max = 40.0  # max longitudinal velocity [m/s]
    v_min = 0.0  # min longitudinal velocity, replaced with soft constraints in case that infeasible solutions caused by the noises
    u_max = 4.0  # max acceleration
    u_min = -5.0  # max deceleration
    j_max = 8.0 # max jerk to ensure smooth acceleration transition
    j_min = -8.0
    rdx_min = 5.0 # min relative distance [m], replace with soft constraints in case that infeasible solutions caused by the noises
    rdx_max=math.inf
    h=1.5 # [s] headway can be adaptive based on the speed
    # h_cof=0.01
    accel_profile=read_acceleration_profile()


    # visual settings
    show_animation = True
    skip_frame = 2


    opti = ca.Opti()


    opt_controls = opti.variable(N, 1)
    u_opt = opt_controls[:, 0]  # action: desired acceleration


    # acc states
    acc_states = opti.variable(N + 1, 5)
    rdx = acc_states[:, 0]
    Vp = acc_states[:, 1]
    V = acc_states[:, 2]
    a = acc_states[:, 3]
    j = acc_states[:, 4]

    # acc states need to be optimized
    opt_states = opti.variable(N + 1, 4)
    Dise = opt_states[:, 0]
    Ve= opt_states[:, 1]
    ae = opt_states[:, 2]
    je=opt_states[:,3]

    # parameters, the actual initial acc states
    init_acc_states = opti.parameter(1, 5)

    # acceleration of the front vehicle
    front_accl_signal = opti.parameter(1, 1)

    # acc model
    f = lambda x, u, ap: ca.vertcat(*[x[1]-x[2]+1/2*ap[0]*Ts-1/2*x[3]*Ts,
                                  ap[0],
                                  x[3],
                                  -x[3]/tau+u[0]/tau,
                                  -x[3]/tau/Ts+u[0]/tau/Ts])

    # for assigning opti.variable, must use opti.subject_to; for assigning opti.parameter, use opti.set_value,but not during the optimization process; for others use =
    opti.subject_to(acc_states[0, :] == init_acc_states)
    opti.subject_to(opt_states[0, 0] == acc_states[0,0] - h*acc_states[0,2]-rdx_min)
    opti.subject_to(opt_states[0, 1] == acc_states[0,1]-acc_states[0,2])
    opti.subject_to(opt_states[0, 2:] == acc_states[0, 3:])
    for i in range(N):
        next_acc_states = acc_states[i, :] + f(acc_states[i, :], opt_controls[i],front_accl_signal).T * Ts
        opti.subject_to(acc_states[i + 1, :] == next_acc_states)
        opti.subject_to(
            opt_states[i + 1, 0] == acc_states[i + 1, 0] - h*acc_states[i + 1,2] - rdx_min)
        opti.subject_to(opt_states[i + 1, 1] == acc_states[i + 1, 1] - acc_states[i + 1, 2])
        opti.subject_to(opt_states[i + 1, 2:] == acc_states[i + 1, 3:])

    # weight matrix
    Q = np.diag([1e2, 1e2, 1e1, 1e1])
    P = np.diag([1e2, 1e2, 1e1, 1e1])



    # cost function
    obj = 0
    for i in range(N):
        state_error = opt_states[i, :]
        obj += ca.mtimes([state_error, Q, state_error.T])
    state_error_final = opt_states[N, :]
    obj += ca.mtimes([state_error_final, P, state_error_final.T])

    for i in range(N+1):
        soft_obj_rdx=1/(ca.exp(acc_states[i, 0]-rdx_min)**2)
        soft_obj_v=1/(ca.exp(10*(acc_states[i, 2]-v_min))**2)
        obj += soft_obj_rdx+soft_obj_v

    opti.minimize(obj)

    # state and action constraints
    opti.subject_to(V<=v_max)
    opti.subject_to(opti.bounded(u_min, u_opt, u_max))
    opti.subject_to(opti.bounded(u_min, a, u_max))
    opti.subject_to(opti.bounded(j_min, j, j_max))

    # remember the cost function and the hard constraints can be replaced by soft functions

    opts_setting = {'ipopt.max_iter': 20000,
                    'ipopt.print_level': 0,
                    'print_time': 0,
                    'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-6}

    opti.solver('ipopt', opts_setting)

    current_state = np.array([rdx_min,0.0,0.0,0.0,0.0]) # initial acc state
    init_x_position=np.array([rdx_min,0.0]) # initial position
    opt_controls0 = np.zeros((N, 1))  # initial optimized actions guess
    init_states = np.tile(current_state, N + 1).reshape(N + 1, -1) # set the initial acc states
    init_error=np.array([current_state[0]-h*current_state[2]-rdx_min,current_state[1]-current_state[2],current_state[3],current_state[4]])
    init_errors = np.tile(init_error, N + 1).reshape(N + 1, -1)
    no = accel_profile[0]  # initial acceleration of the front vehicle

    # contains the history of the states and the actions
    rdx_h=[]
    xp_h = []
    x_h=[]
    vp_h = []
    v_h = []
    a_h = []
    j_h = []
    u_h = []
    no_h=[]
    rdx_h.append(current_state[0])
    vp_h.append(current_state[1])
    v_h.append(current_state[2])
    a_h.append(current_state[3])
    j_h.append(current_state[4])
    xp_h.append(init_x_position[0])
    x_h.append(init_x_position[1])
    u_h.append(opt_controls0[0, 0])
    no_h.append(no)
    i=1
    # start MPC loop
    while True:
        #print(i)
        if show_animation and i % skip_frame == 0:
            plt.cla()
            current_accel = a_h[-1]
            current_velocity=v_h[-1]
            draw_vehicle(init_x_position[1], plt.gca())
            draw_vehicle(init_x_position[0]+Length, plt.gca())
            plt.grid(True)
            plt.axis('equal')
            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')
            plt.title(
                f'MPC Adaptive Cruise Control | Accel:{current_accel:.3f}m/s^2| Velocity:{current_velocity:.3f}m/s')
            plt.pause(0.001)

        # set parameters which are the local reference trajectories, opti.set_value only works for an opti.parameter not an opti.variable
        opti.set_value(front_accl_signal, no)
        opti.set_value(init_acc_states, current_state)

        # provide the initial guess of the optimization targets
        opti.set_initial(opt_controls, opt_controls0.reshape(N, 1))
        opti.set_initial(acc_states, init_states.reshape(N + 1, 5))
        opti.set_initial(opt_states, init_errors.reshape(N + 1, 4))

        # solve the problem once again
        sol = opti.solve()
        # obtain the control input
        u_res = sol.value(opt_controls)
        # print(u_res)
        u_h.append(u_res[1])
        next_state,x_position = acc_transition(current_state, u_res[1],no,init_x_position)
        no = accel_profile[i]  # initial acceleration of the front vehicle
        i = i + 1
        init_x_position=x_position
        current_state = next_state
        init_error = np.array(
            [current_state[0] - h * current_state[2] - rdx_min, current_state[1] - current_state[2], current_state[3],
             current_state[4]])
        init_errors = np.tile(init_error, N + 1).reshape(N + 1, -1)
        init_states = np.tile(current_state, N + 1).reshape(N + 1, -1)
        rdx_h.append(current_state[0])
        vp_h.append(current_state[1])
        v_h.append(current_state[2])
        a_h.append(current_state[3])
        j_h.append(current_state[4])
        xp_h.append(x_position[0])
        x_h.append(x_position[1])
        no_h.append(no)
        if i == len(accel_profile) -1:
            break

    time_axis = np.arange(len(rdx_h)) * Ts

    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, rdx_h, 'b-')
    plt.grid(True)
    plt.title('Relative Distance During Cruising')
    plt.ylabel('Relative Distance [m]')


    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, v_h, 'b-',label='Ego-vehicle Velocity')
    plt.plot(time_axis, vp_h, 'k--', label='Front Vehicle Velocity')
    plt.grid(True)
    plt.title('Velocity During Cruising')
    plt.ylabel('Velocity [m/s]')
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, a_h, 'g-', label='Ego-vehicle Acceleration')
    plt.plot(time_axis, no_h, 'k--',  label='Front vehicle Acceleration')
    plt.grid(True)
    plt.title('Acceleration During Cruising')
    plt.ylabel('Acceleration [m/s^2]')
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, j_h, 'b-')
    plt.grid(True)
    plt.title('Ego-vehicle Jerk During Cruising')
    plt.ylabel('Jerk [m/s^3]')

    plt.show()