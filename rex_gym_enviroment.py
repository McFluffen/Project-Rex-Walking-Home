import pybullet as p
import pybullet_data,gym,time
import numpy as np

class RexLeg:
    def __init__(self, robot_id, leg_id, joints):
        self.robot_id = robot_id
        self.leg_id = leg_id
        self.hip_joint, self.upper_joint, self.lower_joint = joints
        self.joints = [self.hip_joint, self.upper_joint, self.lower_joint]

    def set_pd_control(self, target_positions, kp=35, kd=1.0, max_force=45): #PD controller for some smoothing of motors
        for joint, target_pos in zip(self.joints, target_positions):
            pos, vel, _, _ = p.getJointState(self.robot_id, joint)
            torque = kp * (target_pos - pos) - kd * vel
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint,
                controlMode=p.TORQUE_CONTROL,
                force=float(np.clip(torque, -max_force, max_force))
            )
class Rex:
    def __init__(self, urdf_path, start_position):
        self.robot_id = p.loadURDF(urdf_path, start_position)
        self.num_joints = p.getNumJoints(self.robot_id)
        self.legs = self.init_legs()
        # Disable default motors set by urdf file 
        for j in range(self.num_joints):
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=j,
                controlMode=p.VELOCITY_CONTROL,
                force=0
            )

    def init_legs(self):
        leg_joint_map = {
            0: [1, 2, 3],    # FR leg
            1: [5, 6, 7],    # RL leg
            2: [9, 10, 11],  # RL leg
            3: [13, 14, 15]  # RL leg
        }
        return [RexLeg(self.robot_id, leg_id, indices)
                for leg_id, indices in leg_joint_map.items()]

    def get_observation(self):
        obs = []
        for leg in self.legs:
            for joint in leg.joints:
                pos, vel, _, _ = p.getJointState(self.robot_id, joint) # robots joints
                obs.extend([pos, vel])

        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id) # base position (x,y,z)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot_id)       # base velocity (linear,angular (m/s))
        obs.extend(base_pos)
        obs.extend(base_lin_vel)
        obs.extend(base_ang_vel)
        return np.array(obs, dtype=np.float32)


class QuadrupedEnv(gym.Env):
    def __init__(self, render=True):
        super().__init__()

        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT) #set render false to disable gui

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.time_step = 1 / 240
        p.setTimeStep(self.time_step)

        self.plane_id = p.loadURDF("plane.urdf")
        self.rex = Rex("aliengo/aliengo.urdf", [0, 0, 0.45])
        self.counter = 0

        obs_dim = len(self.rex.get_observation())
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # actionspace is 12 joints 3 per leg (4th is fixed)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

    def step(self, action):
        self.counter += 1
        base_pose = [0.0, 0.8, -1.5]

        # split action into 4 legs × 3 joints
        action = np.clip(action, -1.0, 1.0).reshape(4, 3)
        for leg, act in zip(self.rex.legs, action):
            target_positions = [bp + 0.3 * a for bp, a in zip(base_pose, act)]
            leg.set_pd_control(target_positions, kp=35, kd=1.0, max_force=45)

        p.stepSimulation()
        time.sleep(self.time_step)

        obs = self.rex.get_observation() # observartion
        base_pos, base_orn = p.getBasePositionAndOrientation(self.rex.robot_id) #reward

        # Reward = height stability + orientation uprightness
        height = base_pos[2]
        up_vector = p.getMatrixFromQuaternion(base_orn)[6]  # z-vector
        reward = 1.0 * height + 2.0 * up_vector

        done = height < 0.2  # did not walk
        info = {}
        return obs, reward, done, info

    def reset(self):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.time_step)
        self.plane_id = p.loadURDF("plane.urdf")
        self.rex = Rex("aliengo/aliengo.urdf", [0, 0, 0.45])
        self.counter = 0
        return self.rex.get_observation()

    def close(self):
        p.disconnect(self.physics_client) # stop rex forever :(

if __name__ == "__main__":
    env = QuadrupedEnv(render=True)
    obs = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()  # random action for testing
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
    env.close()
