import pybullet as p
import pybullet_data,gymnasium ,time,math
import numpy as np
import os

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure, Logger

def calculate_distance(Currposition, endPosition): # [x,y,z]
    distance_to_target = math.sqrt(math.pow(endPosition[0]-Currposition[0],2)+math.pow(endPosition[1]-Currposition[1],2)+math.pow(endPosition[2]-Currposition[2],2))
    return distance_to_target

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
    def __init__(self, urdf_path, start_position,end_position):
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
        self.start_pos  = start_position    # rex start position        (x,y,z)
        self.end_pos    = end_position      # rexs desired end position (x,y,z)  

    def init_legs(self):
        leg_joint_map = {
        0: [1, 2, 3],    # Front Right
        1: [5, 6, 7],    # Front Left
        2: [9, 10, 11],  # Rear Right
        3: [13, 14, 15]  # Rear Left
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


class QuadrupedEnv(gymnasium.Env):
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

        self.max_episode_steps = 4000

        self.plane_id = p.loadURDF("plane.urdf")
        self.rex = Rex("aliengo/aliengo.urdf", [0, 0, 0.45],[5,5,0.45])
        self.counter = 0

        obs_dim = len(self.rex.get_observation())
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # actionspace is 12 joints 3 per leg (4th is fixed)
        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)


    def step(self, action):
        done = False
        self.counter += 1
        base_pose = [0.0, 0.4, -0.6] # default pose for each leg [hip, upper, lower]

        # split action into 4 legs × 3 joints
        action = np.clip(action, -1.0, 1.0).reshape(4, 3)
        scales = [0.30, 0.60, 0.90]
        for leg, act in zip(self.rex.legs, action):
            target_positions = [bp + s * a for bp, s, a in zip(base_pose, scales, act)]
            leg.set_pd_control(target_positions, kp=35, kd=1.0, max_force=50)

        p.stepSimulation()

        obs = self.rex.get_observation() # observartion
        base_pos, base_orn = p.getBasePositionAndOrientation(self.rex.robot_id) #reward
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.rex.robot_id)       # base velocity (linear,angular (m/s))
        forward_vel = base_lin_vel[0]
        distance_to_target = calculate_distance(base_pos,self.rex.end_pos)

        # Reward = height stability + orientation uprightness
        r_height = base_pos[2]
        roll, pitch, yaw = p.getEulerFromQuaternion(base_orn) # roll = sideways, ptich = forward/backward, yaw = left/right
        r_up_vector = p.getMatrixFromQuaternion(base_orn)[8]  # z-vector

        v = max(0.0, forward_vel)
        r_forward = np.tanh(v / 7.0)

        progress = self.prev_distance - distance_to_target
        progress_cap = 0.03
        r_progress = np.clip(progress / progress_cap, -1.0, 1.0)

        r_upright = max(0, r_up_vector)
        r_height_bonus = np.clip((r_height - 0.25) / 0.25, 0.0, 1.0)
        r_pitch = 1 - abs(pitch)
        r_roll = 1 -abs(roll)
        r_stable = (r_pitch + r_roll) / 2

        # r_survival = self.counter / self.max_episode_steps
        # r_movement = r_forward * r_survival
        # r_survival_speed = (self.counter / self.max_episode_steps) * r_forward

        reward = (
            # rewards
            1.0 * r_forward
            + 0.8 * r_upright
            + 0.8 * r_height_bonus
            + 0.4 * r_stable
            + 2.0 * r_forward
            # penalties
            # small penality for many actions, to make it use less hopefully
            # -0.002 * np.sum(np.square(action))
        )

        info = {}
        #done = height < 0.2 or pitch < -0.7 or pitch > 0.7 # did not walk or just felly fell
        if r_height < 0.15 or abs(pitch) > 0.8 or abs(roll) > 0.8:
            done = True
            reward -= 5.0
        if distance_to_target < 0.1:
            done = True
            reward+=100.0

        self.prev_distance = distance_to_target
        terminated = done
        truncated = self.counter > self.max_episode_steps

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.time_step)
        self.plane_id = p.loadURDF("plane.urdf")
        self.rex = Rex("aliengo/aliengo.urdf", [0, 0, 0.6],[5,5,0.45])
        self.prev_distance = calculate_distance(self.rex.start_pos, self.rex.end_pos)
        self.counter = 0

        goal_vis = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.1,
            rgbaColor=[1, 0, 0, 1]
        )
        goal_marker = p.createMultiBody(
            baseVisualShapeIndex=goal_vis,
            basePosition=self.rex.end_pos
        )

        info = {}
        return self.rex.get_observation(), info

    def close(self):
        p.disconnect(self.physics_client) # stop rex forever :( NO RIP REX MY GUY MY G SAD RIP AMEN REST IN PEPERINO ((((

def make_env(rank=0, seed=0):
    def _init():
        env = QuadrupedEnv(render=False)
        env = Monitor(env)
        return env
    return _init

run_model = True
model_name = "SAC_model"
model_xx = "./logs/best_model.zip"
model_best_xx = "./logs/best_model.zip"
seed = 66

model_to_run = model_xx

if __name__ == "__main__":
    log_dir = "./logs/"
    checkpoint_dir ="./checkpoints/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if run_model:
        env = QuadrupedEnv(render=True)
        model = SAC.load(model_to_run)
        obs, _ = env.reset()
        env.max_episode_steps = np.inf
        done = False
        for ep in range(10):
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                time.sleep(1/120)
            obs, _ = env.reset()
            done = False
        env.close()
    else:
        num_enviroments = 48 # code for making multiple enviroments 
        if (num_enviroments > 1 ):
            env = DummyVecEnv([make_env(i, seed) for i in range(num_enviroments)])
        else:   
            env = QuadrupedEnv(render=True)



        eval_env = DummyVecEnv([make_env(888, seed)])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=500_000 // num_enviroments,
            deterministic=True,
            render=False
        )

        logger = configure(log_dir, ["stdout", "tensorboard"])

        save_freq = 500_000 // num_enviroments
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path="./checkpoints/",
            name_prefix="SAC_rex"
            )

        # sac_model = SAC(
        #     "MlpPolicy",
        #     env,
        #     verbose=0,
        #     tensorboard_log=log_dir,
        #     learning_rate=1e-4)

        sac_model = SAC.load("base_model.zip", env=env)
        
        sac_model.set_logger(logger)
        callback = CallbackList([checkpoint_callback, eval_callback])
        sac_model.learn(total_timesteps=10_000_000, progress_bar=True, callback=callback)
        sac_model.save(model_name)

        del sac_model