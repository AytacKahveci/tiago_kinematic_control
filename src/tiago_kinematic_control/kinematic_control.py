#!/usr/bin/env python3
import math
import numpy as np
from math import sin, cos
from copy import deepcopy

import rospy
import tf.transformations as tft
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float64
from control_msgs.msg import FollowJointTrajectoryFeedback
import PyKDL as kdl


T1_0 = kdl.Frame(kdl.Rotation.RPY(0.0, 0.0, -np.pi/2.0),          kdl.Vector(0.0, 0.0, 0.0))
T2_1 = kdl.Frame(kdl.Rotation.RPY(np.pi/2.0, 0.0, 0.0),          kdl.Vector(0.125,0.019, -0.031))
T3_2 = kdl.Frame(kdl.Rotation.RPY(-np.pi/2.0, 0.0, np.pi/2.0),     kdl.Vector(0.089, 0.0002, -0.002))
T4_3 = kdl.Frame(kdl.Rotation.RPY(-np.pi/2.0, -np.pi/2.0, 0.0),  kdl.Vector(-0.02, -0.027, -0.222))

T5_4 = kdl.Frame(kdl.Rotation.RPY(np.pi/2.0, -np.pi/2.0, -np.pi/2.0),       kdl.Vector(-0.162, 0.020, 0.027))
T6_5 = kdl.Frame(kdl.Rotation.RPY(0.0, -np.pi/2.0, -np.pi/2.0),     kdl.Vector(0.001, -0.0001, 0.150))
T7_6 = kdl.Frame(kdl.Rotation.RPY(np.pi/2.0, 0.0, np.pi/2.0),     kdl.Vector(0.0, 0.0, 0.0))


lx = 0.093
ly = 0.014

def fk(angles):
  T7_0 = T1_0 * kdl.Frame(kdl.Rotation.RPY(0.0, 0.0, angles[0])) * \
         T2_1 * kdl.Frame(kdl.Rotation.RPY(0.0, 0.0, angles[1])) * \
         T3_2 * kdl.Frame(kdl.Rotation.RPY(0.0, 0.0, angles[2])) * \
         T4_3 * kdl.Frame(kdl.Rotation.RPY(0.0, 0.0, angles[3])) * \
         T5_4 * kdl.Frame(kdl.Rotation.RPY(0.0, 0.0, angles[4])) * \
         T6_5 * kdl.Frame(kdl.Rotation.RPY(0.0, 0.0, angles[5])) * \
         T7_6 * kdl.Frame(kdl.Rotation.RPY(0.0, 0.0, angles[6]))
  return T7_0

def fk_(angles):
	T1_0_ = kdl.Frame(kdl.Rotation().EulerZYX(-np.pi/2.0 + 0.07 + angles[0], 0.0, 0.0),          kdl.Vector(0.093, 0.014, 0.639))
	T2_1_ = kdl.Frame(kdl.Rotation().EulerZYX(0.0 + angles[1], 0.0, np.pi/2.0),          kdl.Vector(0.125, 0.019, -0.031))
	T3_2_ = kdl.Frame(kdl.Rotation().EulerZYX(np.pi/2.0 + angles[2], 0.0, -np.pi/2.0),     kdl.Vector(0.089, 0.0, -0.002))
	T4_3_ = kdl.Frame(kdl.Rotation().EulerZYX(-np.pi + angles[3], -np.pi/2.0, np.pi/2.0),  kdl.Vector(-0.020, -0.027, -0.222))
	T5_4_ = kdl.Frame(kdl.Rotation().EulerZYX(0.0 + angles[4], -np.pi/2.0, 0.0),       kdl.Vector(-0.162, 0.020, 0.027))
	T6_5_ = kdl.Frame(kdl.Rotation().EulerZYX(0.0 + angles[5], -np.pi/2.0, -np.pi/2.0),     kdl.Vector(0.0, 0.0, 0.150))
	T7_6_ = kdl.Frame(kdl.Rotation().EulerZYX(np.pi/2.0 + angles[6], 0.0, np.pi/2.0),     kdl.Vector(0.0, 0.0, 0.0))

	T7_0 = T1_0_ * T2_1_ * T3_2_ * T4_3_ * T5_4_ * T6_5_ * T7_6_
	return T7_0


def t_ag(platform_state):
	"""Transform from frame A to global frame G
	"""
	TP_G = kdl.Frame(kdl.Rotation.RPY(0.0, 0.0, platform_state[2]), kdl.Vector(platform_state[0], platform_state[1], 0))
	TA_P = kdl.Frame(kdl.Vector(0.093, 0.014, 0.639))
	n = TP_G * TA_P
	t = np.zeros((4,4))
	for i in range(3):
		for j in range(3):
			t[i,j] = n.M[i,j]
	t[3,0] = n.p[0]
	t[3,1] = n.p[1]
	t[3,2] = n.p[2]
	t[3,3] = 1.0
	return t


def t_ag_(platform_state):
	"""Transform from frame A to global frame G
	"""
	TP_G = kdl.Frame(kdl.Rotation.RPY(0.0, 0.0, platform_state[2]), kdl.Vector(platform_state[0], platform_state[1], 0))
	TA_P = kdl.Frame(kdl.Vector(0.093, 0.014, 0.639))
	return TP_G * TA_P


def j(angles):
	angles = deepcopy(angles)
	T1_0p = T1_0 * kdl.Frame(kdl.Rotation.RPY(0.0, 0.0, angles[0]))

	T2_0 = T1_0p * \
				T2_1 * kdl.Frame(kdl.Rotation.RPY(0.0, 0.0, angles[1]))
	
	T3_0 = T2_0 * \
				T3_2 * kdl.Frame(kdl.Rotation.RPY(0.0, 0.0, angles[2]))
				
	T4_0 = T3_0 * \
				T4_3 * kdl.Frame(kdl.Rotation.RPY(0.0, 0.0, angles[3]))

	T5_0 = T4_0 * \
				T5_4 * kdl.Frame(kdl.Rotation.RPY(0.0, 0.0, angles[4]))

	T6_0 = T5_0 * \
				T6_5 * kdl.Frame(kdl.Rotation.RPY(0.0, 0.0, angles[5]))

	T7_0 = T6_0 * \
				T7_6 * kdl.Frame(kdl.Rotation.RPY(0.0, 0.0, angles[6]))

	Z1_0 = kdl.Vector(T1_0p.M[0,2], T1_0p.M[1,2], T1_0p.M[2,2])
	Z2_0 = kdl.Vector(T2_0.M[0,2], T2_0.M[1,2], T2_0.M[2,2])
	Z3_0 = kdl.Vector(T3_0.M[0,2], T3_0.M[1,2], T3_0.M[2,2])
	Z4_0 = kdl.Vector(T4_0.M[0,2], T4_0.M[1,2], T4_0.M[2,2])
	Z5_0 = kdl.Vector(T5_0.M[0,2], T5_0.M[1,2], T5_0.M[2,2])
	Z6_0 = kdl.Vector(T6_0.M[0,2], T6_0.M[1,2], T6_0.M[2,2])
	Z7_0 = kdl.Vector(T7_0.M[0,2], T7_0.M[1,2], T7_0.M[2,2])
	print("Z1: ", Z1_0)
	print("Z2: ", Z2_0)
	print("Z3: ", Z3_0)
	print("Z4: ", Z4_0)
	print("Z5: ", Z5_0)
	print("Z6: ", Z6_0)
	print("Z7: ", Z6_0)



	O1_0 = T1_0p.p
	O2_0 = T2_0.p
	O3_0 = T3_0.p
	O4_0 = T4_0.p
	O5_0 = T5_0.p
	O6_0 = T6_0.p
	O7_0 = T7_0.p
	print("O1 ", O1_0)
	print("O2 ", O2_0)
	print("O3 ", O3_0)
	print("O4 ", O4_0)
	print("O5 ", O5_0)
	print("O6 ", O6_0)
	print("O7 ", O7_0)

	J = np.zeros((6, 7), dtype=float)
	temp = (Z1_0 * (O7_0 - O1_0))
	J[0, 0] = deepcopy(temp)[0]
	J[1, 0] = deepcopy(temp)[1]
	J[2, 0] = deepcopy(temp)[2]
	J[3, 0] = Z1_0[0]
	J[4, 0] = Z1_0[1]
	J[5, 0] = Z1_0[2]

	temp = (Z2_0 * (O7_0 - O2_0))
	J[0, 1] = deepcopy(temp)[0]
	J[1, 1] = deepcopy(temp)[1]
	J[2, 1] = deepcopy(temp)[2]
	J[3, 1] = Z2_0[0]
	J[4, 1] = Z2_0[1]
	J[5, 1] = Z2_0[2]

	temp = (Z3_0 * (O7_0 - O3_0))
	J[0, 2] = deepcopy(temp)[0]
	J[1, 2] = deepcopy(temp)[1]
	J[2, 2] = deepcopy(temp)[2]
	J[3, 2] = Z3_0[0]
	J[4, 2] = Z3_0[1]
	J[5, 2] = Z3_0[2]

	temp = (Z4_0 * (O7_0 - O4_0))
	J[0, 3] = deepcopy(temp)[0]
	J[1, 3] = deepcopy(temp)[1]
	J[2, 3] = deepcopy(temp)[2]
	J[3, 3] = Z4_0[0]
	J[4, 3] = Z4_0[1]
	J[5, 3] = Z4_0[2]

	temp = (Z5_0 * (O7_0 - O5_0))
	J[0, 4] = deepcopy(temp)[0]
	J[1, 4] = deepcopy(temp)[1]
	J[2, 4] = deepcopy(temp)[2]
	J[3, 4] = Z5_0[0]
	J[4, 4] = Z5_0[1]
	J[5, 4] = Z5_0[2]

	temp = (Z6_0 * (O7_0 - O6_0))
	J[0, 5] = deepcopy(temp)[0]
	J[1, 5] = deepcopy(temp)[1]
	J[2, 5] = deepcopy(temp)[2]
	J[3, 5] = Z6_0[0]
	J[4, 5] = Z6_0[1]
	J[5, 5] = Z6_0[2]

	temp = (Z7_0 * (O7_0 - O7_0))
	J[0, 6] = deepcopy(temp)[0]
	J[1, 6] = deepcopy(temp)[1]
	J[2, 6] = deepcopy(temp)[2]
	J[3, 6] = Z7_0[0]
	J[4, 6] = Z7_0[1]
	J[5, 6] = Z7_0[2]

	return J


def j_gap(th):
	"""Jacobian Platform of A frame wrt G global frame

	Args:
		th (float): Mobile platform angle wrt G global reference frame

	Returns:
		np.ndarray: Jacobian
	"""
	global lx, ly
	J = np.zeros((6, 2))
	J[0,0] = cos(th)
	J[1,0] = sin(th)
	J[0,1] = -(lx*sin(th) + ly*cos(th))
	J[1,1] = lx*cos(th) - ly*sin(th)
	J[5,1] = 1.0
	return J 


def j_gep(Pex, Pey, th):
	"""Jacobian Platform of E end effector frame wrt G global frame

	Args:
		Pex (flaot): End effector x position wrt A reference frame
		Pey (flaot): End effector x position wrt A reference frame
		th (float): Mobile platform angle wrt G global reference frame

	Returns:
		np.ndarray: Jacobian
	"""
	J = np.eye(6)
	J[0,5] = -(Pex*sin(th) + Pey*cos(th))
	J[1,5] = Pex*cos(th) - Pey*sin(th)
	return J


class KinematicControl:
	def __init__(self):
		self.joint_state_sub = rospy.Subscriber("/joint_states", JointState, self.stateCb, queue_size=1)
		self.odom_sub = rospy.Subscriber("/mobile_base_controller/odom", Odometry, self.odomCb, queue_size=1)
		self.arm1_pub = rospy.Publisher("/arm1_controller/command", Float64, queue_size=1)
		self.arm2_pub = rospy.Publisher("/arm2_controller/command", Float64, queue_size=1)
		self.arm3_pub = rospy.Publisher("/arm3_controller/command", Float64, queue_size=1)
		self.arm4_pub = rospy.Publisher("/arm4_controller/command", Float64, queue_size=1)
		self.arm5_pub = rospy.Publisher("/arm5_controller/command", Float64, queue_size=1)
		self.arm6_pub = rospy.Publisher("/arm6_controller/command", Float64, queue_size=1)
		self.arm7_pub = rospy.Publisher("/arm7_controller/command", Float64, queue_size=1)
		self.platform_pub = rospy.Publisher("/nav_vel", Twist, queue_size=1)
		self.feedback_pub = rospy.Publisher("/feedback", FollowJointTrajectoryFeedback, queue_size=1)
		self.desired_pub = rospy.Publisher("/desired", Path, queue_size=1)
		self.actual_pub = rospy.Publisher("/actual", Path, queue_size=1)


		self.mp_state = np.zeros((9, 1))
		self.arm_joints = np.zeros((7, 1))
		self.platform_state = np.zeros((3, 1))
		self.arm_msg_received = False
		self.platform_msg_received = False
		self.desired = Path()
		self.desired.header.frame_id = "odom"
		self.actual = Path()
		self.actual.header.frame_id = "odom"

	def stateCb(self, msg):
		for i, name in enumerate(msg.name):
			if name == "arm_1_joint":
				ind = 0
			elif name == "arm_2_joint":
				ind = 1
			elif name == "arm_3_joint":
				ind = 2
			elif name == "arm_4_joint":
				ind = 3
			elif name == "arm_5_joint":
				ind = 4
			elif name == "arm_6_joint":
				ind = 5
			elif name == "arm_7_joint":
				ind = 6
			else:
				continue
			self.arm_joints[ind] = msg.position[i]
			self.arm_msg_received = True
		rospy.loginfo("A State: [1]: {} [2]: {} [3]: {} [4]: {} [5]: {} [6]: {} [7]: {}".format(
				self.arm_joints[0], self.arm_joints[1], self.arm_joints[2], self.arm_joints[3], self.arm_joints[4], self.arm_joints[5], self.arm_joints[6]))

	def odomCb(self, msg):
		self.platform_msg_received = True
		self.platform_state[0] = msg.pose.pose.position.x
		self.platform_state[1] = msg.pose.pose.position.y
		self.platform_state[2] = tft.euler_from_quaternion([msg.pose.pose.orientation.x, 
																											  msg.pose.pose.orientation.y,
																												msg.pose.pose.orientation.z,
																												msg.pose.pose.orientation.w])[2]
		rospy.loginfo("P State: X: {} Y: {} Th: {}".format(self.platform_state[0], self.platform_state[1], self.platform_state[2]))


	def command_to_joints(self, q_dot, dt):
		self.arm1_pub.publish(self.arm_joints[0] + q_dot[0]*dt)
		self.arm2_pub.publish(self.arm_joints[1] + q_dot[1]*dt)
		self.arm3_pub.publish(self.arm_joints[2] + q_dot[2]*dt)
		self.arm4_pub.publish(self.arm_joints[3] + q_dot[3]*dt)
		self.arm5_pub.publish(self.arm_joints[4] + q_dot[4]*dt)
		self.arm6_pub.publish(self.arm_joints[5] + q_dot[5]*dt)
		self.arm7_pub.publish(self.arm_joints[6] + q_dot[6]*dt)

		cmd_vel = Twist()
		cmd_vel.linear.x = q_dot[7]
		cmd_vel.angular.z = q_dot[8]
		self.platform_pub.publish(cmd_vel)


	def control(self):
		Kp = 2.5
		Kd = 0.5
		
		self.feedback = FollowJointTrajectoryFeedback()
		self.feedback.joint_names = ["X", "Y", "Z", "R", "P", "Y"]
		self.feedback.desired.positions = [0.0]*6
		self.feedback.desired.velocities = [0.0]*6
		self.feedback.desired.accelerations = [0.0]*6
		self.feedback.actual.positions = [0.0]*6
		self.feedback.actual.velocities = [0.0]*6
		self.feedback.actual.accelerations = [0.0]*6
		self.feedback.error.positions = [0.0]*6
		self.feedback.error.velocities = [0.0]*6
		self.feedback.error.accelerations = [0.0]*6

		loop_rate = rospy.Rate(10)
		self.start = rospy.Time.now()
		self.elapsed = 0.0
		first_state = False
		XE_d_prev = np.zeros((6,1))
		while not rospy.is_shutdown():
			if not self.arm_msg_received or not self.platform_msg_received:
				rospy.logwarn_throttle(1, "Waiting for joint states")
				loop_rate.sleep()	
				continue
			else:
				if not first_state:
					first_state = True
					TE_A = fk(self.arm_joints)
					TA_G = t_ag_(self.platform_state)
					TE = TA_G * TE_A
					r, p, y = TE.M.GetRPY()
					XE_init = np.array([TE.p[0], TE.p[1], TE.p[2], r, p, y]).reshape((6, 1))
					XE_prev = deepcopy(XE_init)
					self.start = rospy.Time.now()
			
			dt = (rospy.Time.now() - self.start).to_sec()
			self.start = rospy.Time.now()
			XE_desired = np.array([1 * 0.1 *self.elapsed, 1.0*sin(0.05*self.elapsed), 0.3*sin(0.1*self.elapsed), 0.0, 0.0, 0.0]).reshape((6,1)) + XE_init
			self.elapsed += dt

			TE_A = fk(self.arm_joints)
			Pex = TE_A.p.x()
			Pey = TE_A.p.y()

			TA_G = t_ag(self.platform_state)
			RA_G = TA_G[:3,:3]
			JA_Ea = j(self.arm_joints)
			T = np.zeros((6,6))
			T[:3,:3] = RA_G
			T[3:,3:] = RA_G
			JG_Ea = np.matmul(T, JA_Ea) 

			JG_Ep = j_gep(Pex, Pey, self.platform_state[2])
			JG_Ap = j_gap(self.platform_state[2])
			JG_Ep_ = np.matmul(JG_Ep, JG_Ap)

			JG_E = np.hstack([JG_Ea, JG_Ep_])
			JG_E_inv = np.linalg.pinv(JG_E)

			TA_G = t_ag_(self.platform_state)
			TE = TA_G * TE_A
			r, p, y = TE.M.GetRPY()
			XE = np.array([TE.p[0], TE.p[1], TE.p[2], r, p, y]).reshape((6, 1))
			Kp = np.array([0.7, 0.7, 3, 1, 1.0, 1.0]).reshape(6,1)
			Kd = np.array([0.4, 0.4, 1.0, 1.0, 1.0, 1.0]).reshape(6,1)
			XE_dot = (XE_desired - XE) * Kp - (XE - XE_prev) / dt * Kd
			XE_d_prev = XE_dot
			XE_prev = XE
			q_dot = np.matmul(JG_E_inv, XE_dot)
			self.command_to_joints(q_dot, dt)

			self.feedback.desired.positions = [XE_desired[0], XE_desired[1], XE_desired[2], XE_desired[3], XE_desired[4], XE_desired[5]]
			self.feedback.desired.velocities = [0.0]*6
			self.feedback.desired.accelerations = [0.0]*6
			self.feedback.actual.positions = [XE[0], XE[1], XE[2], XE[3], XE[4], XE[5]]
			self.feedback.actual.velocities = [0.0]*6
			self.feedback.actual.accelerations = [0.0]*6
			p = PoseStamped()
			p.header.stamp=rospy.Time.now()
			p.pose.position.x = XE_desired[0]
			p.pose.position.y = XE_desired[1]
			p.pose.position.z = XE_desired[2]
			self.desired.header.stamp = rospy.Time.now()
			self.desired.poses.append(deepcopy(p))
			p.pose.position.x = XE[0]
			p.pose.position.y = XE[1]
			p.pose.position.z = XE[2]
			self.actual.header.stamp = rospy.Time.now()
			self.actual.poses.append(deepcopy(p))
			self.actual_pub.publish(self.actual)
			self.desired_pub.publish(self.desired)
			T = (XE_desired - XE)
			self.feedback.error.positions = [T[0], T[1], T[2], T[3], T[4], T[5]]
			self.feedback.error.velocities = [0.0]*6
			self.feedback.error.accelerations = [0.0]*6
			self.feedback.header.stamp = rospy.Time.now()
			self.feedback_pub.publish(self.feedback)
			loop_rate.sleep()


if __name__ == "__main__":
	rospy.init_node("test")
	
	kc = KinematicControl()
	kc.control()