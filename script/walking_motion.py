# Copyright 2018 CNRS

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import time
import numpy as np
from math import atan2, cos, sin
from pinocchio import centerOfMass, forwardKinematics
from cop_des import CoPDes
from com_trajectory import ComTrajectory
from inverse_kinematics import InverseKinematics
from tools import Constant, Piecewise

import sys

# Computes the trajectory of a swing foot.
#
# Input data are
#  - initial and final time of the trajectory,
#  - initial and final pose of the foot,
#  - maximal height of the foot,
#
# The trajectory is polynomial with zero velocities at start and end.
# The orientation of the foot is kept as in intial pose.
class SwingFootTrajectory(object):
    def __init__(self, t_init, t_end, init, end, height):
        
        self.t_init = t_init
        self.t_end = t_end
        self.height = height
        # Write your code here

        x0 = init[0]
        x1 = end[0]
        y0 = init[1]
        y1 = end[1]
        z0 = init[2]
        z1 = end[2]
        T = (t_end-t_init)

        self.ax = -2* (x1-x0)/((T)**3)
        self.bx = 3* (x1-x0)/((T)**2)
        self.cx = 0
        self.dx = x0

        self.ay = -2* (y1-y0)/((T)**3)
        self.by = 3* (y1-y0)/((T)**2)
        self.cy = 0
        self.dy = y0

        self.az = (height-z0)*16/(T**4)
        self.bz = -2*self.az*T
        self.cz = self.az*T**2
        self.dz = 0
        self.ez = z0

    def __call__(self, t):
        # write your code here

        if t < self.t_init:
            t = self.t_init
        elif t > self.t_end:
            t = self.t_end
        
        x = self.ax*(t-self.t_init)**3+self.bx*(t-self.t_init)**2+self.dx
        y = self.ay*(t-self.t_init)**3+self.by*(t-self.t_init)**2+self.dy
        z = self.az*(t-self.t_init)**4+self.bz*(t-self.t_init)**3+self.cz*(t-self.t_init)**2+self.dz*(t-self.t_init)+self.ez
        
        return np.array([x,y,z])

# Computes a walking whole-body motion
#
# Input data are
#  - an initial configuration of the robot,
#  - a sequence of step positions (x,y,theta) on the ground,
#  - a mapping from time to R corresponding to the desired orientation of the
#    waist. If not provided, keep constant orientation.
#
class WalkingMotion(object):
    step_height = 0.05
    single_support_time = .5
    double_support_time = .1

    def __init__(self, robot):
        self.robot = robot

    def compute(self, q0, steps, waistOrientation = None):
        # Test input data
        if len(steps) < 4:
            raise RuntimeError("sequence of step should be of length at least 4 instead of " +
                               f"{len(steps)}")
        # Copy steps in order to avoid modifying the input list.
        steps_ = steps[:]
        # Compute offset between waist and center of mass since we control the center of mass
        # indirectly by controlling the waist.
        data = self.robot.model.createData()
        forwardKinematics(self.robot.model, data, q0)
        com = centerOfMass(self.robot.model, data, q0)
        waist_pose = data.oMi[self.robot.waistJointId]
        com_offset = waist_pose.translation - com
        # Trajectory of left and right feet
        self.lf_traj = Piecewise()
        self.rf_traj = Piecewise()
        # write your code here

        initial_left_foot_position = np.array([0, .1, 0.])
        initial_right_foot_position = np.array([0, -.1, 0.])

        steps_l = steps_[1::2]
        steps_r = steps_[0::2]

        n_r = 0
        n_l = 0

        t_init = 0
        t_end = 0
        l_init = initial_left_foot_position
        r_init = initial_right_foot_position

        sst = self.single_support_time
        dst = self.double_support_time

        for i in range(len(steps_)-1):
            print(f"adding trajectory of step {i} / {len(steps_)-1}")
            t_init = sst * i + dst * (i)
            t_end = t_init + dst
            self.lf_traj.segments.append(Constant(t_init,t_end,l_init))
            self.rf_traj.segments.append(Constant(t_init,t_end,r_init))
                                         
            t_init = t_end
            t_end = t_init + sst


            l_end = steps_l[n_l]
            r_end = steps_r[n_r]

            if (i % 2 == 0):
                n_r += 1

                self.lf_traj.segments.append(SwingFootTrajectory(t_init,t_end,l_init,l_end,self.step_height))
                self.rf_traj.segments.append(Constant(t_init,t_end,r_init))
            else:
                n_l += 1

                self.rf_traj.segments.append(SwingFootTrajectory(t_init,t_end,r_init,r_end,self.step_height))
                self.lf_traj.segments.append(Constant(t_init,t_end,l_init))

            l_init = l_end
            r_init = r_end

        
        last_step_right = steps_r[-1]
        last_step_left = steps_l[-1]

        self.lf_traj.segments.append(Constant(t_end,np.inf,last_step_left))
        self.rf_traj.segments.append(Constant(t_end,np.inf,last_step_right))

        z_com = com_offset[2]

        com_final = np.array([
            (last_step_left[0]+last_step_right[0])/2,
            (last_step_left[1]+last_step_right[1])/2,
        ])

        steps_xy = [step[0:2] for step in steps_]

        self.COM_trajectory = ComTrajectory(com_offset[0:2],steps_xy,com_final,z_com)
        X = self.COM_trajectory.compute()
        timestep = self.COM_trajectory.delta_t
        times = timestep*np.arange(self.COM_trajectory.N+1)
        com_positions = np.array(list(map(self.COM_trajectory, times)))

        for com in com_positions:
            np.append(com,z_com)

        left_foot_positions = np.array(list(map(self.lf_traj, times))) 
        right_foot_positions = np.array(list(map(self.rf_traj, times))) 

        ik = InverseKinematics(self.robot)
        configurations = [q0]

        total_configurations = len(com_positions)

        for i, (com, left_foot, right_foot) in enumerate(zip(com_positions, left_foot_positions, right_foot_positions)):
            ik.rightFootRefPose.translation = left_foot
            ik.leftFootRefPose.translation = right_foot
            ik.waistRefPose.translation = com
            q_last = configurations[i-1]

            print(f"Solving for configuration {i} / {total_configurations}")

            q_new = ik.solve(q_last)
            configurations.append(q_new)

        return configurations




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from talos import Robot
    from pinocchio import neutral
    import numpy as np
    from inverse_kinematics import InverseKinematics
    import eigenpy

    robot = Robot ()
    ik = InverseKinematics (robot)
    ik.rightFootRefPose.translation = np.array ([0, -0.1, 0.1])
    ik.leftFootRefPose.translation = np.array ([0, 0.1, 0.1])
    ik.waistRefPose.translation = np.array ([0, 0, 0.95])

    q0 = neutral (robot.model)
    q0 [robot.name_to_config_index["leg_right_4_joint"]] = .2
    q0 [robot.name_to_config_index["leg_left_4_joint"]] = .2
    q0 [robot.name_to_config_index["arm_left_2_joint"]] = .2
    q0 [robot.name_to_config_index["arm_right_2_joint"]] = -.2
    q = ik.solve (q0)
    robot.display(q)
    wm = WalkingMotion(robot)
    # First two values correspond to initial position of feet
    # Last two values correspond to final position of feet
    steps = [np.array([0, -.1, 0.]), np.array([0.4, .1, 0.]),
             np.array([.8, -.1, 0.]), np.array([1.2, .1, 0.]),
             np.array([1.6, -.1, 0.]), np.array([1.6, .1, 0.])]
    configs = wm.compute(q, steps)
    for q in configs:
        time.sleep(1e-2)
        robot.display(q)
    delta_t = wm.COM_trajectory.delta_t
    times = delta_t*np.arange(wm.COM_trajectory.N+1)
    lf = np.array(list(map(wm.lf_traj, times)))
    rf = np.array(list(map(wm.rf_traj, times)))
    cop_des = np.array(list(map(wm.COM_trajectory.cop_des, times)))
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.plot(times, lf[:,0], label="x left foot")
    ax1.plot(times, rf[:,0], label="x right foot")
    ax1.plot(times, cop_des[:,0], label="x CoPdes")
    ax1.legend()
    ax2.plot(times, lf[:,1], label="y left foot")
    ax2.plot(times, rf[:,1], label="y right foot")
    ax2.plot(times, cop_des[:,1], label="y CoPdes")
    ax2.legend()
    ax3.plot(times, lf[:,2], label="z left foot")
    ax3.plot(times, rf[:,2], label="z right foot")
    ax3.legend()
    plt.show()
