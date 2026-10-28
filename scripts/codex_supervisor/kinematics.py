"""Read-only robot kinematics, never scene simulation or an action rollout."""

import numpy as np
from scipy.spatial.transform import Rotation


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class RobotFK:
    def __init__(self, env):
        self.agent = env.unwrapped.agent
        self.robot = self.agent.robot
        self.articulation = self.robot._objs[0]  # noqa: SLF001 - SAPIEN articulation needed for independent FK
        self.model = self.articulation.create_pinocchio_model()
        self.joint_names = [joint.name for joint in self.robot.get_active_joints()]
        self.arm_names = list(self.agent.arm_joint_names)
        self.arm_indices = [self.joint_names.index(name) for name in self.arm_names]
        # Actions are passed to the active pd_joint_pos controller in this exact order.
        controller = self.agent.controller
        if hasattr(controller, "controllers"):
            controller = controller.controllers["arm"]
        if list(controller.config.joint_names) != self.arm_names:
            raise ValueError("FK and controller joint order differ")
        if controller.config.normalize_action or controller.config.use_delta:
            raise ValueError("absolute unnormalized joint position control is required")
        self.tcp_name = self.agent.ee_link_name
        self.link_index = [link.name for link in self.robot.get_links()].index(self.tcp_name)
        self.metadata = {
            "robot_urdf": str(self.agent.urdf_path),
            "arm_joint_order": self.arm_names,
            "tcp_link": self.tcp_name,
            "frame": "world",
            "position_units": "metres",
            "orientation": "extrinsic xyz RPY radians",
            "method": "independent SAPIEN Pinocchio model; base pose composed with robot-local TCP",
        }
        self.verify_current()

    def _poses(self, actions):
        q = self.articulation.get_qpos().copy()
        base = self.articulation.get_pose()
        result = []
        for action in actions:
            q[self.arm_indices] = action[:7]
            self.model.compute_forward_kinematics(q)
            pose = base * self.model.get_link_pose(self.link_index)
            quat = np.asarray(pose.q)
            rpy = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler("xyz")
            result.append(np.concatenate([pose.p, rpy]).tolist())
        return result

    def verify_current(self):
        before_q = self.articulation.get_qpos().copy()
        before_v = self.articulation.get_qvel().copy()
        measured_p = array(self.agent.tcp.pose.p).reshape(-1, 3)[0]
        measured_q = array(self.agent.tcp.pose.q).reshape(-1, 4)[0]
        predicted = self._poses([before_q[self.arm_indices]])[0]
        position_error = float(np.linalg.norm(np.asarray(predicted[:3]) - measured_p))
        actual_rotation = Rotation.from_quat(measured_q[[1, 2, 3, 0]])
        orientation_error = float((Rotation.from_euler("xyz", predicted[3:]) * actual_rotation.inv()).magnitude())
        if position_error > 1e-5 or orientation_error > 1e-4:
            raise ValueError("robot FK does not agree with measured TCP")
        if not np.array_equal(before_q, self.articulation.get_qpos()) or not np.array_equal(
            before_v, self.articulation.get_qvel()
        ):
            raise ValueError("FK changed robot state")
        self.metadata["current_check"] = {
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
        }
        return self.metadata["current_check"]

    def describe(self, actions, current):
        self.verify_current()
        before_q = self.articulation.get_qpos().copy()
        before_v = self.articulation.get_qvel().copy()
        poses = self._poses(actions)
        if not np.array_equal(before_q, self.articulation.get_qpos()) or not np.array_equal(
            before_v, self.articulation.get_qvel()
        ):
            raise ValueError("FK changed robot state")
        trajectory = [
            {
                "step": i + 1,
                "target_xyz_rpy": pose,
                "delta_xyz_from_measured": (np.asarray(pose[:3]) - current[:3]).tolist(),
                "gripper_command": float(action[7]),
                "gripper_command_change": None if i == 0 else float(action[7] - actions[i - 1][7]),
            }
            for i, (pose, action) in enumerate(zip(poses, actions, strict=True))
        ]
        return {
            "conventions": self.metadata,
            "measured_current_xyz_rpy": list(current),
            "summary": {"first": trajectory[0], "last": trajectory[-1], "steps": len(trajectory)},
            "trajectory": trajectory,
            "limitation": "Kinematic targets only; no prediction of contact, collision, object motion or grasp success. "
            "Gripper commands (-1 close, +1 open) are not measured finger positions; stick tasks ignore them.",
        }
