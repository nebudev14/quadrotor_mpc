import opengen as og
import casadi as cs
import numpy as np

horizon = 5
dt = 0.005

mass = 1.3
gravity = 9.81
drag_coeff = 0.00

Ixx = 3.04e-3
Iyy = 4.55e-3
Izz = 2.82e-3
Ixy = 0.0
Ixz = 0.0
Iyz = 0.0

I = cs.SX(3,3)
I[0,0] = Ixx;  I[0,1] = Ixy;  I[0,2] = Ixz
I[1,0] = Ixy;  I[1,1] = Iyy;  I[1,2] = Iyz
I[2,0] = Ixz;  I[2,1] = Iyz;  I[2,2] = Izz

I_inv = cs.inv(I)

nx = 13
nu = 4

u = cs.SX.sym("u", nu*horizon)

x0 = cs.SX.sym("x0", nx)
xref = cs.SX.sym("xref", nx)
p = cs.vertcat(x0, xref)

def normalize_quaternion(q):
    norm_q = cs.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2 + 1e-12)
    return q / norm_q

def quat_multiply(q1, q2):
    """
    Hamilton product: q1 * q2
    Both are [qw, qx, qy, qz].
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return cs.vertcat(w, x, y, z)

def from_scaled_axis(ang_vel_dt):

    angle = cs.sqrt(ang_vel_dt[0]**2 + ang_vel_dt[1]**2 + ang_vel_dt[2]**2)
    eps = 1e-12
    def quat_from_axis(a, v):
        axis_x = v[0]/a
        axis_y = v[1]/a
        axis_z = v[2]/a
        half = a * 0.5
        c = cs.cos(half)
        s = cs.sin(half)
        return cs.vertcat(c, axis_x*s, axis_y*s, axis_z*s)

    rot_quat = cs.if_else(
        angle < eps,
        cs.vertcat(1.0, 0.0, 0.0, 0.0),
        quat_from_axis(angle, ang_vel_dt)
    )
    return rot_quat

def quadrotor_dynamics_euler(x, u):

    pos    = x[0:3]
    vel    = x[3:6]
    quat   = x[6:10]
    omega  = x[10:13]

    thrust = u[0]
    torque = u[1:4]

    gravity_force = cs.vertcat(0.0, 0.0, -mass*gravity)

    speed = cs.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2 + 1e-12)
    drag_force = -drag_coeff * speed * vel

    quat_n = normalize_quaternion(quat)
    qw, qx, qy, qz = quat_n[0], quat_n[1], quat_n[2], quat_n[3]
    R = cs.SX(3,3)
    R[0,0] = 1 - 2*(qy**2 + qz**2)
    R[0,1] = 2*(qx*qy - qw*qz)
    R[0,2] = 2*(qx*qz + qw*qy)
    R[1,0] = 2*(qx*qy + qw*qz)
    R[1,1] = 1 - 2*(qx**2 + qz**2)
    R[1,2] = 2*(qy*qz - qw*qx)
    R[2,0] = 2*(qx*qz - qw*qy)
    R[2,1] = 2*(qy*qz + qw*qx)
    R[2,2] = 1 - 2*(qx**2 + qy**2)

    thrust_body = cs.vertcat(0, 0, thrust)
    thrust_world = R @ thrust_body

    total_force = thrust_world + gravity_force + drag_force
    acceleration = total_force / mass

    vel_next = vel + acceleration * dt
    pos_next = pos + vel_next * dt


    Iomega = I @ omega
    gyro_torque = cs.cross(omega, Iomega)
    net_torque = torque - gyro_torque
    ang_acc = I_inv @ net_torque

    omega_next = omega + ang_acc * dt

    delta_quat = from_scaled_axis(omega_next * dt)
    quat_next = quat_multiply(quat, delta_quat)
    quat_next = normalize_quaternion(quat_next)

    x_next = cs.vertcat(pos_next, vel_next, quat_next, omega_next)
    return x_next

def quaternion_error(q_current, q_desired):

    nc = cs.sqrt(cs.sum1(q_current**2)); qc = q_current/nc
    nd = cs.sqrt(cs.sum1(q_desired**2)); qd = q_desired/nd

  
    qw_inv =  qc[0]
    qx_inv = -qc[1]
    qy_inv = -qc[2]
    qz_inv = -qc[3]

    qdw, qdx, qdy, qdz = qd[0], qd[1], qd[2], qd[3]
    diff_w = qdw*qw_inv - qdx*qx_inv - qdy*qy_inv - qdz*qz_inv
    diff_x = qdw*qx_inv + qdx*qw_inv + qdy*qz_inv - qdz*qy_inv
    diff_y = qdw*qy_inv - qdx*qz_inv + qdy*qw_inv + qdz*qx_inv
    diff_z = qdw*qz_inv + qdx*qy_inv - qdy*qx_inv + qdz*qw_inv

    sinr_cosp = 2*(diff_w*diff_x + diff_y*diff_z)
    cosr_cosp = 1 - 2*(diff_x*diff_x + diff_y*diff_y)
    roll = cs.atan2(sinr_cosp, cosr_cosp)

    sinp = 2*(diff_w*diff_y - diff_z*diff_x)
    pitch = cs.if_else(
        cs.fabs(sinp) >= 1,
        cs.pi/2*cs.sign(sinp),
        cs.asin(sinp)
    )

    siny_cosp = 2*(diff_w*diff_z + diff_x*diff_y)
    cosy_cosp = 1 - 2*(diff_y*diff_y + diff_z*diff_z)
    yaw = cs.atan2(siny_cosp, cosy_cosp)

    return cs.vertcat(roll, pitch, yaw)


Q_pos = cs.diag(cs.SX([80.0, 80.0, 80.0]))
Q_vel = cs.diag(cs.SX([10.0, 10.0, 10.0]))
Q_rpy = cs.diag(cs.SX([30.0, 30.0, 30.0]))
Q_omega = cs.diag(cs.SX([25.0, 25.0, 25.0]))

R_mat = cs.diag(cs.SX([0.0001, 0.0001, 0.0001, 0.0001]))

Q_pos_t   = 2.0 * Q_pos
Q_vel_t   = 2.0 * Q_vel
Q_rpy_t   = 2.0 * Q_rpy
Q_omega_t = 2.0 * Q_omega

cost = 0
u_mat = cs.reshape(u, nu, horizon)

x_k = x0

for k in range(horizon):
    pos_k  = x_k[0:3]
    vel_k  = x_k[3:6]
    quat_k = x_k[6:10]
    omg_k  = x_k[10:13]

    pos_ref  = xref[0:3]
    vel_ref  = xref[3:6]
    quat_ref = xref[6:10]
    omg_ref  = xref[10:13]

    pos_err  = pos_k  - pos_ref
    vel_err  = vel_k  - vel_ref
    rpy_err  = quaternion_error(quat_k, quat_ref)
    omg_err  = omg_k  - omg_ref

    stage_cost = 0
    stage_cost += pos_err.T @ Q_pos @ pos_err
    stage_cost += vel_err.T @ Q_vel @ vel_err
    stage_cost += rpy_err.T @ Q_rpy @ rpy_err
    stage_cost += omg_err.T @ Q_omega @ omg_err

    u_k = u_mat[:, k]
    stage_cost += u_k.T @ R_mat @ u_k

    cost += stage_cost

    x_next = quadrotor_dynamics_euler(x_k, u_k)
    x_k = x_next

pos_err_f  = x_k[0:3]  - xref[0:3]
vel_err_f  = x_k[3:6]  - xref[3:6]
quat_err_f = quaternion_error(x_k[6:10], xref[6:10])
omg_err_f  = x_k[10:13] - xref[10:13]

cost += (
    pos_err_f.T @ Q_pos_t   @ pos_err_f
  + vel_err_f.T @ Q_vel_t   @ vel_err_f
  + quat_err_f.T @ Q_rpy_t  @ quat_err_f
  + omg_err_f.T @ Q_omega_t @ omg_err_f
)


u_min = [0.0, -5.0, -5.0, -5.0] * horizon
u_max = [60.0,  5.0,  5.0,  5.0] * horizon
bounds = og.constraints.Rectangle(u_min, u_max)


pos_min = [-4.0, -2.0, 0.0]
pos_max = [4.0, 2.0, 2.0]

pos_constraint_weight = 2000.0

x_k_constraint = x0

for k in range(horizon):
    u_k = u_mat[:, k]
    
    pos_k = x_k_constraint[0:3]
    
    for i in range(3):
        lower_violation = cs.fmax(0, pos_min[i] - pos_k[i])
        upper_violation = cs.fmax(0, pos_k[i] - pos_max[i])
        
        cost += pos_constraint_weight * (lower_violation**2 + upper_violation**2)
    
    x_k_constraint = quadrotor_dynamics_euler(x_k_constraint, u_k)

pos_terminal = x_k_constraint[0:3]
for i in range(3):
    lower_violation = cs.fmax(0, pos_min[i] - pos_terminal[i])
    upper_violation = cs.fmax(0, pos_terminal[i] - pos_max[i])
    cost += pos_constraint_weight * (lower_violation**2 + upper_violation**2)


problem = og.builder.Problem(u, p, cost).with_constraints(bounds)

build_config = og.config.BuildConfiguration() \
    .with_build_directory("extra_quadrotor_mpc_build") \
    .with_tcp_interface_config()

meta = og.config.OptimizerMeta() \
    .with_optimizer_name("quadrotor_mpc")

solver_config = og.config.SolverConfiguration() \
    .with_tolerance(1e-5) \
    .with_max_inner_iterations(500)

builder = og.builder.OpEnOptimizerBuilder(problem, meta, build_config, solver_config)
builder.build()
