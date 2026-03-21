# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from gymnasium import spaces

from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import RayCaster
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .go2_isaaclabws2_env_cfg import Go2Isaaclabws2EnvCfg, threshhold, obstacle_phase, path_planning_threshold, WAYPOINTS, WAYPOINT_RADII
import inspect
from .go2scenecfg import merge_spheres_for_lidar_from_states, MAX_OBS
from pxr import UsdGeom, Vt, Gf
import omni.usd
import omni.kit.app
import sys
sys.path.append("/home/helix-intern/isaac-go2-ros2")  # path to their repo
import omni.usd
from pxr import UsdGeom, Usd

def create_full_warehouse_env():
    try:
        import isaacsim.storage.native as nucleus_utils
    except ModuleNotFoundError:
        import isaacsim.core.utils.nucleus as nucleus_utils
    from isaacsim.core.utils.prims import define_prim, get_prim_at_path

    assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Warehouse")
    prim = define_prim("/World/Warehouse", "Xform")
    asset_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
    prim.GetReferences().AddReference(asset_path)

def merge_warehouse_for_lidar(output_path: str = "/World/merged_warehouse"):
    from pxr import Usd, UsdGeom, Vt, Gf
    import omni.usd
    
    stage = omni.usd.get_context().get_stage()
    
    all_points = []
    all_indices = []
    all_counts = []
    offset = 0
    
    warehouse_prim = stage.GetPrimAtPath("/World/Warehouse")
    for prim in Usd.PrimRange(warehouse_prim):
        if prim.GetTypeName() != "Mesh":
            continue
            
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        
        if not points or not face_indices or not face_counts:
            continue
        
        # Get world transform
        xform = UsdGeom.Xformable(prim)
        world_transform = xform.ComputeLocalToWorldTransform(0)
        
        # Transform points to world space
        for pt in points:
            world_pt = world_transform.Transform(Gf.Vec3d(pt))
            all_points.append(Gf.Vec3f(world_pt))
        
        all_indices.extend([idx + offset for idx in face_indices])
        all_counts.extend(face_counts)
        offset += len(points)
    
    print(f"[MERGE] Merged {offset} points from warehouse", flush=True)
    
    merged = UsdGeom.Mesh.Define(stage, output_path)
    merged.GetPointsAttr().Set(Vt.Vec3fArray(all_points))
    merged.GetFaceVertexIndicesAttr().Set(Vt.IntArray(all_indices))
    merged.GetFaceVertexCountsAttr().Set(Vt.IntArray(all_counts))
    
    return output_path





def create_placeholder_mesh(output_path: str = "/World/merged_spheres"):
    stage = omni.usd.get_context().get_stage()
    mesh = UsdGeom.Mesh.Define(stage, output_path)
    # Dummy triangle so the mesh is not empty
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([
        Gf.Vec3f(0.0, 0.0, 0.0),
        Gf.Vec3f(0.1, 0.0, 0.0),
        Gf.Vec3f(0.0, 0.1, 0.0),
    ]))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2]))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3]))



class Go2Isaaclabws2Env(DirectRLEnv):
    cfg: Go2Isaaclabws2EnvCfg

    def __init__(self, **kwargs):
        create_placeholder_mesh("/World/merged_spheres")

        super().__init__(**kwargs)  # _setup_scene runs here, warehouse loads

        for _ in range(20):
            self.sim.step()

        # Delete placeholder prim so it can't interfere with lidar
        stage = omni.usd.get_context().get_stage()
        placeholder_prim = stage.GetPrimAtPath("/World/merged_spheres")
        if placeholder_prim.IsValid():
            stage.RemovePrim("/World/merged_spheres")
            print("[INIT] Placeholder mesh deleted", flush=True)

        # Merge warehouse geometry into single mesh for lidar
        try:
            merge_warehouse_for_lidar("/World/merged_warehouse")
            
            # Reinitialize lidar with warehouse mesh
            self.scene.sensors["lidar"].cfg.mesh_prim_paths = ["/World/merged_warehouse"]
            self.scene.sensors["lidar"]._initialize_warp_meshes()

            # Remove any stale meshes from dict
            lidar_sensor = self.scene.sensors["lidar"]
            stale = [k for k in lidar_sensor.meshes if k != "/World/merged_warehouse"]
            for k in stale:
                del lidar_sensor.meshes[k]
                print(f"[INIT] Removed stale mesh: {k}", flush=True)

            print(f"[INIT] Final meshes: {list(lidar_sensor.meshes.keys())}", flush=True)
            print(f"[INIT] Warehouse mesh points: {lidar_sensor.meshes['/World/merged_warehouse'].points.shape}", flush=True)

        except Exception as e:
            print(f"[INIT] Lidar reinit failed: {e}", flush=True)

        # Buffers
        self.sim_time = torch.zeros(self.num_envs, device=self.device)
        self.cmd = torch.zeros((self.num_envs, 2), device=self.device)
        self.joint_targets = None
        self.actions = None
        self.prev_actions = None
        self._dbg = 0
        self.leg_joint_ids = None
        self.yaw_ref = torch.zeros(self.num_envs, device=self.device)
        self.pos_ref = torch.zeros((self.num_envs, 2), device=self.device)
        self.yaw_cmd_int = torch.zeros(self.num_envs, device=self.device)
        self.dead_leg_counter = torch.zeros(self.num_envs, device=self.device)
        self.calf_motion_ema = torch.zeros(self.num_envs, 4, device=self.device)
        self.prev_front_dist = torch.zeros(self.num_envs, device=self.device)
        self.prev_front = torch.zeros(self.num_envs, device=self.device)
        self.stuck_counter = torch.zeros(self.num_envs, device=self.device)
        self.collision_counter = torch.zeros(self.num_envs, device=self.device)
        self.train_step_counter = 0
        self.leg_stuck_counter = torch.zeros((self.num_envs, 4), device=self.device)
        self.last_yaw = torch.zeros(self.num_envs, device=self.device)
        self.yaw_accumulator = torch.zeros(self.num_envs, device=self.device)
        self.last_pos = torch.zeros((self.num_envs, 2), device=self.device)
        self.pos_update_counter = torch.zeros(self.num_envs, device=self.device)
        self.prev_pos_waypoint = torch.zeros((self.num_envs, 2), device=self.device)

        print(f"env_origins: {self.scene.env_origins[0].cpu().numpy().round(2)}", flush=True)

        # Waypoint navigation
        self.waypoints = torch.tensor(WAYPOINTS, device=self.device)
        self.current_waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_dist_to_waypoint = torch.zeros(self.num_envs, device=self.device)
        self.heading_ref = torch.zeros(self.num_envs, device=self.device)
        self.waypoint_radii = torch.tensor(WAYPOINT_RADII, device=self.device)

        # Initialize heading toward first waypoint
        first_waypoint = self.waypoints[0]
        to_first = first_waypoint.unsqueeze(0) - torch.zeros((self.num_envs, 2), device=self.device)
        self.heading_ref = torch.atan2(to_first[:, 1], to_first[:, 0])
        self.sharp_turn_counter = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------
    # Scene setup
    # ------------------------------------------------
    def _setup_scene(self):
        super()._setup_scene()
        self.scene.clone_environments(copy_from_source=False)

        self.robot = self.scene["robot"]
        self.lidar = self.scene["lidar"]
        self.feet_sensor = self.scene["contact_sensor"]
        self.spheres = self.scene["spheres"]
        self.front_camera = self.scene["front_camera"]
        self.depth_camera = self.scene["depth_camera"]

        # Load warehouse using their function
        create_full_warehouse_env()

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/Light", light_cfg)


    # ------------------------------------------------
    # RL hooks
    # ------------------------------------------------
    def _init_leg_joint_ids(self):
        if hasattr(self.robot, "joint_names"):
            names = list(self.robot.joint_names)
        elif hasattr(self.robot, "data") and hasattr(self.robot.data, "joint_names"):
            names = list(self.robot.data.joint_names)
        else:
            raise RuntimeError("Can't access joint names from robot/articulation.")

        def idx(n: str) -> int:
            if n not in names:
                raise RuntimeError(f"Joint name '{n}' not found. Available: {names}")
            return names.index(n)

        self.leg_joint_ids = {
            "FL": torch.tensor([idx("FL_hip_joint"), idx("FL_thigh_joint"), idx("FL_calf_joint")], device=self.device),
            "FR": torch.tensor([idx("FR_hip_joint"), idx("FR_thigh_joint"), idx("FR_calf_joint")], device=self.device),
            "RL": torch.tensor([idx("RL_hip_joint"), idx("RL_thigh_joint"), idx("RL_calf_joint")], device=self.device),
            "RR": torch.tensor([idx("RR_hip_joint"), idx("RR_thigh_joint"), idx("RR_calf_joint")], device=self.device),
        }
        print("Leg joint ids:", {k: v.tolist() for k, v in self.leg_joint_ids.items()}, flush=True)

    def _ensure_action_buffers(self):
        # already initialized
        if self.actions is not None and self.prev_actions is not None:
            return

        # safest dimension source (works even when num_joints is flaky):
        # robot.data.joint_pos is (N, num_joints) once buffers exist
        if hasattr(self.robot, "data") and hasattr(self.robot.data, "joint_pos"):
            action_dim = int(self.robot.data.joint_pos.shape[1])
        else:
            # fallback
            action_dim = int(self.robot.num_joints)

        self.actions = torch.zeros((self.num_envs, action_dim), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._ensure_action_buffers()

        if self.leg_joint_ids is None:
            try:
                self._init_leg_joint_ids()
            except Exception as e:
                print("Joint-id init not ready yet:", e, flush=True)

        self.actions[:] = actions
        self.yaw_cmd_int += self.cmd[:, 1] * self.cfg.sim.dt
        self.sim_time += self.cfg.sim.dt
        self.train_step_counter += 1

        # UNCOMMENT THIS:
        if hasattr(self, 'lidar_front'):
            # Always point toward waypoint
            current_pos = self.robot.data.root_link_pos_w[:, :2]
            target = self.waypoints[self.current_waypoint_idx]
            to_target = target - current_pos
            dist_to_target = torch.norm(to_target, dim=-1)

            # Per-waypoint acceptance radius
            current_radius = self.waypoint_radii[self.current_waypoint_idx]
            
            # Advance waypoint if within radius
            reached_here = dist_to_target < current_radius
            if reached_here.any():
                new_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
                self.current_waypoint_idx = torch.where(reached_here, new_idx, self.current_waypoint_idx)
                target = self.waypoints[self.current_waypoint_idx]
                to_target = target - current_pos
                dist_to_target = torch.norm(to_target, dim=-1)
                if reached_here[0].item():
                    print(f"[WAYPOINT REACHED in pre_physics] idx={self.current_waypoint_idx[0].item()}", flush=True)

            waypoint_heading = torch.atan2(to_target[:, 1], to_target[:, 0])
            self.heading_ref = waypoint_heading

            current_yaw = self._yaw_from_quat(self.robot.data.root_quat_w)
            heading_err = torch.atan2(
                torch.sin(self.heading_ref - current_yaw),
                torch.cos(self.heading_ref - current_yaw)
            )

            # Lateral correction
            dx = to_target[:, 0]
            lateral_correction = torch.clamp(dx / 3.0, -1.0, 1.0) * 0.3

            # Corridor detection
            in_corridor = (self.lidar_left < 2.0) & (self.lidar_right < 2.0) & (self.lidar_front > 2.0)

            # Sharp turn detection
            sharp_turn_needed = heading_err.abs() > 0.75

            # In corridor: ignore side walls, only use heading
            left_score = torch.where(
                in_corridor,
                heading_err,
                (self.lidar_left - self.lidar_right) + 2.0 * heading_err - lateral_correction
            )

            noise = torch.randn(self.num_envs, device=self.device) * 0.05

            # Don't trigger blocked from side walls in corridor or during sharp turn
            blocked = (self.lidar_front < 2.0) & ~in_corridor & ~sharp_turn_needed

            turn_left = blocked & (left_score + noise > 0)
            turn_right = blocked & (left_score + noise <= 0)
            go_forward = ~blocked

            # Speed scaling
            speed_scale = torch.clamp((self.lidar_front - 0.5) / 1.5, 0.1, 1.0)
            waypoint_speed_scale = torch.clamp(dist_to_target / 3.0, 0.2, 1.0)

            # Reverse if too close to wall during sharp turn
            too_close_while_turning = sharp_turn_needed & (self.lidar_front < 0.6)

            #Slow down instead of stopping for sharp turns
            SHARP_TURN_SPEED_FACTOR = 0.55

            sharp_turn_speed = torch.where(
                too_close_while_turning,
                torch.full_like(waypoint_speed_scale, -0.3),  # reverse
                torch.where(
                    sharp_turn_needed,
                    waypoint_speed_scale * SHARP_TURN_SPEED_FACTOR,  # half speed
                    waypoint_speed_scale  # normal
                )
            )

            # Full speed in corridor, scaled elsewhere
            corridor_speed = torch.where(in_corridor, torch.ones_like(speed_scale), speed_scale)

            # Heading dead zone
            significant_err = heading_err.abs() > 0.1

            # Stronger yaw correction in corridor
            corridor_yaw_gain = torch.where(
                in_corridor,
                torch.full_like(heading_err, 2.5),
                torch.full_like(heading_err, 1.5)
            )

            yaw_correction = torch.where(
                significant_err,
                torch.clamp(heading_err * corridor_yaw_gain, -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate),
                torch.zeros_like(heading_err)
            )

            # Wall centering in corridor
            wall_center_err = (self.lidar_left - self.lidar_right) * 0.15
            corridor_correction = torch.where(
                in_corridor,
                yaw_correction + wall_center_err,
                yaw_correction
            )

            # Dynamic yaw rate - higher for sharp turns
            SHARP_TURN_YAW = 2.1
            sharp_turn_yaw_rate = torch.where(
                sharp_turn_needed,
                torch.full_like(heading_err, SHARP_TURN_YAW),
                torch.full_like(heading_err, self.cfg.max_yaw_rate)
            )

            # Apply commands
            self.cmd[go_forward, 0] = self.cfg.max_lin_vel * corridor_speed[go_forward] * sharp_turn_speed[go_forward]

            self.cmd[go_forward, 1] = torch.where(
                sharp_turn_needed[go_forward],
                torch.sign(heading_err[go_forward]) * sharp_turn_yaw_rate[go_forward],
                corridor_correction[go_forward]
            )

            self.cmd[turn_left, 0] = self.cfg.max_lin_vel * 0.2
            self.cmd[turn_left, 1] = self.cfg.max_yaw_rate

            self.cmd[turn_right, 0] = self.cfg.max_lin_vel * 0.2
            self.cmd[turn_right, 1] = -self.cfg.max_yaw_rate

            if self.train_step_counter % 50 == 0:
                current_pos_p = self.robot.data.root_link_pos_w[0, :2]
                target_p = self.waypoints[self.current_waypoint_idx[0]]
                dx_p = (target_p[0] - current_pos_p[0]).item()
                dy_p = (target_p[1] - current_pos_p[1]).item()
                dist_p = torch.norm(current_pos_p - target_p).item()
                print(f"[NAV] pos=({current_pos_p[0]:.2f}, {current_pos_p[1]:.2f}) "
                    f"target=({target_p[0]:.2f}, {target_p[1]:.2f}) "
                    f"dx={dx_p:.2f} dy={dy_p:.2f} dist={dist_p:.2f}m "
                    f"wp={self.current_waypoint_idx[0].item()} "
                    f"sharp={sharp_turn_needed[0].item()} "
                    f"heading_err={heading_err[0]:.2f} "
                    f"radius={current_radius[0]:.1f}", flush=True)
            if sharp_turn_needed[0].item():
                print(f"[SHARP] cmd_wz={self.cmd[0,1]:.2f} should be ±2.5")
                actual_wz = self.robot.data.root_ang_vel_b[0, 2].item()
                print(f"[SHARP] cmd_wz={self.cmd[0,1]:.2f} actual_wz={actual_wz:.2f} heading_err={heading_err[0]:.2f}")
        



    def _setup_action_space(self):
        num_joints = self.robot.num_joints
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_joints,), dtype=np.float32)

    def _setup_observation_space(self):
        num_joints = self.robot.num_joints
        obs_dim = 46 # g + v + w + qrel + qd + cmd + contacts + airtime
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _dbg_left_right_work(self, env_id: int = 0, every: int = 50):
        if int(self.episode_length_buf[env_id]) % every != 0:
            return
        if self.actions is None or self.joint_targets is None:
            return

        q  = self.robot.data.joint_pos[env_id]
        q0 = self.robot.data.default_joint_pos[env_id]
        qd = self.robot.data.joint_vel[env_id]
        jt = self.joint_targets[env_id]
        a  = self.actions[env_id]

        # indices
        L = torch.tensor([0,4,8, 2,6,10], device=self.device)  # FL + RL
        R = torch.tensor([1,5,9, 3,7,11], device=self.device)  # FR + RR

        def stats(idx):
            a_mag  = torch.mean(torch.abs(a[idx])).item()
            cmd_mag = torch.mean(torch.abs((jt - q0)[idx])).item()     # commanded delta pos
            qrel_mag = torch.mean(torch.abs((q - q0)[idx])).item()     # actual delta pos
            qd_mag = torch.mean(torch.abs(qd[idx])).item()             # motion
            track = torch.mean(torch.abs((jt - q)[idx])).item()        # tracking error
            return a_mag, cmd_mag, qrel_mag, qd_mag, track

        sL = stats(L); sR = stats(R)

        print(
            f"[LR] step={int(self.episode_length_buf[env_id])} "
            f"L a={sL[0]:.3f} cmd={sL[1]:.3f} qrel={sL[2]:.3f} qd={sL[3]:.3f} err={sL[4]:.3f} | "
            f"R a={sR[0]:.3f} cmd={sR[1]:.3f} qrel={sR[2]:.3f} qd={sR[3]:.3f} err={sR[4]:.3f}",
            flush=True
        )

    def _dbg_worst_joint(self, env_id: int = 0, every: int = 50):
        if self.joint_targets is None:
            return
        if int(self.episode_length_buf[env_id]) % every != 0:
            return

        q  = self.robot.data.joint_pos[env_id]
        jt = self.joint_targets[env_id]
        e  = (jt - q).abs()

        j = int(torch.argmax(e).item())

        # joint name lookup
        if hasattr(self.robot, "joint_names"):
            names = list(self.robot.joint_names)
        else:
            names = list(self.robot.data.joint_names)

        print(
            f"[worst] step={int(self.episode_length_buf[env_id])} "
            f"j={j} name={names[j]} jt={jt[j]:+.3f} q={q[j]:+.3f} |err|={e[j]:.3f}",
            flush=True
        )

    
    
    
    
    
    def _build_action_sign(self):
        sign = torch.ones(12, device=self.device)

        # hips (leave as-is for now)
        sign[0] = -1.0   # FL hip
        sign[1] = -1.0   # FR hip
        sign[2] = -1.0   # RL hip
        sign[3] = -1.0   # RR hip

        # thighs
        sign[4:8] = 1.0

        # calves
        sign[8]  =  1.0   # FL calf
        sign[9]  =  1.0   # FR calf
        sign[10] = -1.0   # RL calf
        sign[11] = -1.0   # RR calf

        self.action_sign = sign.unsqueeze(0)


    def _apply_action(self) -> None:
        default = self.robot.data.default_joint_pos

        if self.joint_targets is None:
            self.joint_targets = default.clone()

        # clamp raw actions
        a = torch.clamp(self.actions, -1.0, 1.0)

        # apply action sign (your best-performing version)
        if not hasattr(self, "action_sign"):
            self._build_action_sign()
        a = a * self.action_sign

        # joint scaling
        scale = torch.tensor(
            [
                0.05, 0.05, 0.05, 0.05,   # hips
                0.55, 0.55, 0.55, 0.55,   # thighs
                0.70, 0.70, 0.70, 0.70,   # calves
            ],
            device=self.device,
        ).unsqueeze(0)

        # desired joint positions
        desired = default + scale * a
        desired[:, 0:4] = default[:, 0:4]

        # ------------------------------------------------
        # OPTION A: prevent hind thigh kneeling (NO calves)
        # ------------------------------------------------
        min_raise = 0.12  # radians above default (tune 0.06–0.20)

        # If this makes them kneel MORE, flip torch.maximum → torch.minimum
        desired[:, 6] = torch.maximum(desired[:, 6], default[:, 6] + min_raise)  # RL thigh
        desired[:, 7] = torch.maximum(desired[:, 7], default[:, 7] + min_raise)  # RR thigh

        # ------------------------------------------------
        # smooth joint target update (prevents twitching)
        # ------------------------------------------------
        max_delta = 0.10
        delta = torch.clamp(desired - self.joint_targets, -max_delta, max_delta)
        self.joint_targets = self.joint_targets + delta
        
        self.robot.set_joint_position_target(self.joint_targets)

        # write to sim 
        """
        self.robot.set_joint_position_target(self.joint_targets)
        self._dbg_left_right_work(env_id=0, every=50)
        self._dbg_worst_joint(env_id=0, every=50)"""




    def _lin_vel_body(self) -> torch.Tensor:
        """Return base linear velocity in body frame. Shape (N,3)."""
        v_w = self.robot.data.root_lin_vel_w
        q = self.robot.data.root_quat_w  # (w,x,y,z)

        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # R_wb (body->world), so v_b = R_wb^T v_w
        r00 = 1 - 2 * (y * y + z * z)
        r01 = 2 * (x * y - w * z)
        r02 = 2 * (x * z + w * y)

        r10 = 2 * (x * y + w * z)
        r11 = 1 - 2 * (x * x + z * z)
        r12 = 2 * (y * z - w * x)

        r20 = 2 * (x * z - w * y)
        r21 = 2 * (y * z + w * x)
        r22 = 1 - 2 * (x * x + y * y)

        vx_b = r00 * v_w[:, 0] + r10 * v_w[:, 1] + r20 * v_w[:, 2]
        vy_b = r01 * v_w[:, 0] + r11 * v_w[:, 1] + r21 * v_w[:, 2]
        vz_b = r02 * v_w[:, 0] + r12 * v_w[:, 1] + r22 * v_w[:, 2]

        return torch.stack([vx_b, vy_b, vz_b], dim=1)
    def _world_vecs_to_body(self, v_w: torch.Tensor) -> torch.Tensor:
        """
        Rotate world-frame vectors into body frame.
        v_w: (N, R, 3) -> (N, R, 3)
        """
        q = self.robot.data.root_quat_w  # (N,4) (w,x,y,z)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        # R_wb (body->world), so v_b = R_wb^T v_w
        r00 = 1 - 2 * (y * y + z * z)
        r01 = 2 * (x * y - w * z)
        r02 = 2 * (x * z + w * y)

        r10 = 2 * (x * y + w * z)
        r11 = 1 - 2 * (x * x + z * z)
        r12 = 2 * (y * z - w * x)

        r20 = 2 * (x * z - w * y)
        r21 = 2 * (y * z + w * x)
        r22 = 1 - 2 * (x * x + y * y)

        # broadcast (N,) -> (N,1)
        r00 = r00[:, None]; r01 = r01[:, None]; r02 = r02[:, None]
        r10 = r10[:, None]; r11 = r11[:, None]; r12 = r12[:, None]
        r20 = r20[:, None]; r21 = r21[:, None]; r22 = r22[:, None]

        vx = r00 * v_w[..., 0] + r10 * v_w[..., 1] + r20 * v_w[..., 2]
        vy = r01 * v_w[..., 0] + r11 * v_w[..., 1] + r21 * v_w[..., 2]
        vz = r02 * v_w[..., 0] + r12 * v_w[..., 1] + r22 * v_w[..., 2]

        return torch.stack([vx, vy, vz], dim=-1)

    def _yaw_from_quat(self, q):
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _get_observations(self):
        hits = self.lidar.data.ray_hits_w
        if hits.dim() == 4:
            hits = hits[:, 0]

        max_range = 50.0
        finite = torch.isfinite(hits).all(dim=-1)  # (N,R)

        base = self.robot.data.root_link_pos_w[:, None, :3]  # (N,1,3)
        rel_w = hits - base                                   # (N,R,3)

        # Invalid rays: set to far forward
        far = torch.tensor([max_range, 0.0, 0.0], device=self.device)
        rel_w = torch.where(finite[..., None], rel_w, far)

        # Full quaternion rotation
        rel_b = self._world_vecs_to_body(rel_w)  # (N,R,3)
        dist = torch.linalg.norm(rel_b, dim=-1).clamp(max=max_range)  # (N,R)

        # Sector masks in body frame: x forward, y left
        x = rel_b[..., 0]
        y = rel_b[..., 1]
        z = rel_b[..., 2]

        # Height filter - ignore ground and ceiling hits
        height_filter = (z > -0.1) & (z < 1.5)

        near_x = (x > 0.0) & (x < 5.0)
        front_mask = near_x & (y.abs() < 0.6) & height_filter
        left_mask  = near_x & (y > 0.3) & height_filter
        right_mask = near_x & (y < -0.3) & height_filter

        forward_cone = near_x & (y.abs() < 0.5) & height_filter
        blocked_rays = (~finite | (dist < 1.0)) & forward_cone
        self.forward_blocked_fraction = blocked_rays.float().sum(dim=1) / forward_cone.float().sum(dim=1).clamp(min=1)

        def masked_min(d, m):
            d2 = torch.where(m, d, torch.full_like(d, max_range))
            return d2.min(dim=1).values

        d_front = masked_min(dist, front_mask)
        d_left  = masked_min(dist, left_mask)
        d_right = masked_min(dist, right_mask)

        # Cache for rewards/dones
        self.lidar_front = d_front
        self.lidar_left = d_left
        self.lidar_right = d_right
        self.lidar_dmin = d_front

        if self.train_step_counter % 50 == 0:
            print(f"d_front={d_front[0]:.2f} d_left={d_left[0]:.2f} d_right={d_right[0]:.2f}", flush=True)

        # Normalized lidar features
        lidar_obs = torch.stack([d_front, d_left, d_right], dim=1) / max_range  # (N,3)

        # Existing observations
        g_b = self.robot.data.projected_gravity_b
        v_b = self._lin_vel_body()
        w_b = self.robot.data.root_ang_vel_b

        q = self.robot.data.joint_pos
        q0 = self.robot.data.default_joint_pos
        q_rel = q - q0
        qd = self.robot.data.joint_vel

        data = self.feet_sensor.data
        Fmag = torch.linalg.norm(data.net_forces_w, dim=-1)
        thr = self.feet_sensor.cfg.force_threshold
        contacts = (Fmag > thr).float()
        air_time = data.current_air_time

        obs = torch.cat(
            [g_b, v_b, w_b, q_rel, qd, self.cmd, contacts, air_time, lidar_obs],
            dim=1
        )
        return {"policy": obs}

    def _get_rewards(self):
        # -----------------------------
        # Body-frame velocities
        # -----------------------------
        v_b = self._lin_vel_body()
        vx, vy, vz = v_b[:, 0], v_b[:, 1], v_b[:, 2]

        w_b = self.robot.data.root_ang_vel_b
        wy, wz = w_b[:, 1], w_b[:, 2]

        vx_cmd = self.cmd[:, 0]
        wz_cmd = self.cmd[:, 1]

        # -----------------------------
        # Upright / tilt
        # -----------------------------
        g_b = self.robot.data.projected_gravity_b
        g_x, g_y, g_z = g_b[:, 0], g_b[:, 1], g_b[:, 2]

        upright = torch.clamp((-g_z - 0.7) / 0.3, 0.0, 1.0)
        tilt_pen = g_x**2 + g_y**2
        pitch_pen = g_x**2

        # -----------------------------
        # Height: prevent belly-crawl
        # -----------------------------
        h = self.robot.data.root_link_pos_w[:, 2]
        h_target = 0.28

        low = torch.relu(h_target - h)
        low_pen = low**2

        r_height = torch.exp(-((h - h_target) / 0.05) ** 2)
        height_ok = (h > (h_target - 0.03)).float()

        # -----------------------------
        # Command tracking
        # -----------------------------
        r_vx = torch.exp(-((vx - vx_cmd) / 0.15) ** 2)
        r_vx = r_vx * (0.15 + 0.85 * height_ok)

        r_lat = torch.exp(-(vy / 0.20) ** 2)
        r_yaw = torch.exp(-((wz - wz_cmd) / 0.8) ** 2)

        underspeed = torch.relu(vx_cmd - vx)
        backward = torch.relu(-vx)

        # -----------------------------
        # Anti-hop / smooth base motion
        # -----------------------------
        vz_pen = vz**2
        pitch_rate_pen = wy**2

        # -----------------------------
        # Joint posture anti-collapse (front legs)
        # -----------------------------
        q = self.robot.data.joint_pos
        q0 = self.robot.data.default_joint_pos
        q_rel = q - q0

        front_thigh = torch.tensor([4, 5], device=self.device)
        front_calf = torch.tensor([8, 9], device=self.device)

        front_collapse = (q_rel[:, front_thigh] ** 2).sum(dim=1) + (q_rel[:, front_calf] ** 2).sum(dim=1)
        posture_cost = (q_rel ** 2).sum(dim=1)

        # -----------------------------
        # ContactSensor gait constraints
        # -----------------------------
        data = self.feet_sensor.data
        Fmag = torch.linalg.norm(data.net_forces_w, dim=-1)
        thr = self.feet_sensor.cfg.force_threshold
        contacts = (Fmag > thr).float()
        n_contact = contacts.sum(dim=1)

        sigma_c = 0.35
        r_three = torch.exp(-((n_contact - 3.0) / sigma_c) ** 2)
        pen_two = torch.exp(-((n_contact - 2.0) / sigma_c) ** 2)
        pen_air = torch.exp(-((n_contact - 0.0) / sigma_c) ** 2)

        air_time = data.current_air_time
        t_swing = 0.12
        num_swing = (air_time > t_swing).float().sum(dim=1)

        r_one_swing = torch.exp(-((num_swing - 1.0) / 0.25) ** 2)
        progress = torch.clamp(vx / (vx_cmd + 1e-6), 0.0, 1.0)
        r_one_swing = r_one_swing * progress * height_ok

        # -----------------------------
        # Action regularization
        # -----------------------------
        act_cost = torch.sum(self.actions ** 2, dim=1)
        actrate_cost = torch.sum((self.actions - self.prev_actions) ** 2, dim=1)
        self.prev_actions[:] = self.actions

        # -----------------------------
        # Leg participation + coordination
        # -----------------------------
        qd = self.robot.data.joint_vel

        leg_groups = torch.tensor(
            [
                [0, 4, 8],   # FL
                [1, 5, 9],   # FR
                [2, 6, 10],  # RL
                [3, 7, 11],  # RR
            ],
            device=self.device,
        )

        leg_motion = torch.sum(torch.abs(qd[:, leg_groups]), dim=2)
        mean_motion = leg_motion.mean(dim=1, keepdim=True)
        rel_motion = leg_motion / (mean_motion + 1e-4)

        dead_leg_pen = torch.relu(0.6 - rel_motion).sum(dim=1)

        m_FL, m_FR, m_RL, m_RR = leg_motion[:, 0], leg_motion[:, 1], leg_motion[:, 2], leg_motion[:, 3]
        motion_scale = (m_FL + m_FR + m_RL + m_RR) / 4.0 + 1e-4

        hind_imbalance = torch.abs(m_RL - m_RR) / motion_scale
        front_imbalance = torch.abs(m_FL - m_FR) / motion_scale
        hind_air_imbalance = torch.abs(air_time[:, 2] - air_time[:, 3])

        height_ok2 = (h > 0.25).float()
        upright_ok = ((-g_b[:, 2]) > 0.85).float()
        warm_ok = (self.episode_length_buf > 40).float()
        gate = height_ok2 * upright_ok * warm_ok


        yaw = self._yaw_from_quat(self.robot.data.root_quat_w)
        yaw_err = torch.atan2(torch.sin(yaw - self.yaw_cmd_int), torch.cos(yaw - self.yaw_cmd_int))
        yaw_track_pen = yaw_err**2
        # -----------------------------
        # LIDAR shaping (use directional sectors)
        # -----------------------------
        max_range = 50.0  # keep consistent with observations
        d_front = getattr(self, "lidar_front", self.lidar_dmin)
        d_left  = getattr(self, "lidar_left",  self.lidar_dmin)
        d_right = getattr(self, "lidar_right", self.lidar_dmin)

        d_safe = 0.60
        near = torch.relu(d_safe - d_front) / d_safe     # 0..1
        near_pen = near ** 2

        # discourage pushing forward when close
        slow_when_close = torch.relu(vx) * near_pen

        # prefer turning toward the more open side when close
        # If left is closer than right, want to turn right (sign may need flip)
        turn_dir = torch.sign(d_right - d_left)          # + means "turn left" if wz positive is left
        turn_align = torch.relu(turn_dir * wz)           # reward turning the correct direction
        turn_reward = turn_align * near_pen

        # collision indicator (terminate will handle too)
        collision_pen = torch.clamp((0.5 - d_front) / 0.5, 0.0, 1.0)

        # ---------------------------------
        # Escape / unstuck shaping
        # ---------------------------------

        # Smooth "front blocked" (0..1)
        d_trigger = 0.6
        front_blocked = torch.clamp((d_trigger - d_front) / d_trigger, 0.0, 1.0)

        # Smooth "not moving forward"
        slow = torch.exp(-(vx / 0.08) ** 2)

        # Only escape when upright and high enough
        stable = upright * height_ok

        # Escape activation (0..1)
        escape = front_blocked * slow * stable

        # Reverse reward
        backward_vel = torch.relu(-vx)
        reverse_reward = backward_vel * escape

        # Escape turning reward (turn toward open side)
        turn_dir = torch.sign(d_right - d_left)
        escape_turn = torch.relu(turn_dir * wz)
        escape_turn_reward = escape_turn * escape

        # ---------------------------------
        # Persistent escape shaping
        # ---------------------------------

        d_trigger = 0.7
        d_escape = 1.2

        front_blocked = torch.clamp((d_trigger - d_front) / d_trigger, 0.0, 1.0)

        # Escape goal: reach safe distance
        escape_progress = torch.clamp(d_front / d_escape, 0.0, 1.0)

        # Reward increases as it approaches safe clearance
        persistent_escape_reward = front_blocked * escape_progress

        # ---------------------------------
        # Obstacle clearing reward
        # ---------------------------------

        d_clear = 4.5   # distance considered fully clear

        # detect transition from blocked to clear
        prev_blocked = self.prev_front < 0.7
        now_clear = d_front > d_clear

        cleared_obstacle = prev_blocked & now_clear

        clear_reward = cleared_obstacle.float()

        # encourage forward motion when clear
        forward_after_clear = clear_reward * torch.relu(vx)

        self.prev_front[:] = d_front

        obstacle_weight = 1.0 if obstacle_phase else 0.0
        underspeed_weight = 2.0 if obstacle_phase else 1.0

        blocked = (d_front < 0.6).float()

        sharp_turn = torch.relu(turn_dir * wz) * blocked

        path_clear = (d_front > 0.6).float()
        forward_progress = torch.relu(vx) * path_clear

        path_blocked = (d_front < 0.6).float()
        reckless_forward = torch.relu(vx) * path_blocked

        max_clearance = torch.maximum(d_front, torch.maximum(d_left, d_right))

        # Reward moving toward whichever direction is most open
        open_dir_forward = (d_front == max_clearance).float()
        open_dir_left = (d_left == max_clearance).float()  
        open_dir_right = (d_right == max_clearance).float()

        # Reward aligning velocity with open direction
        moving_to_open = (
            open_dir_forward * torch.relu(vx) +
            open_dir_left * torch.relu(wz) +   # turning left
            open_dir_right * torch.relu(-wz)   # turning right
        )

        self.leg_stuck_counter = torch.where(
            leg_motion < 0.05,
            self.leg_stuck_counter + 1,
            torch.zeros_like(self.leg_stuck_counter)
        )

        # Leg stuck for more than 0.5 seconds
        stuck_limit = 0.5 / (self.cfg.sim.dt * self.cfg.decimation)
        stuck_leg_pen = (self.leg_stuck_counter > stuck_limit).any(dim=1).float()

        self.last_pos = torch.zeros((self.num_envs, 2), device=self.device)
        self.pos_update_counter = torch.zeros(self.num_envs, device=self.device)

        # In _get_rewards:
        current_pos = self.robot.data.root_link_pos_w[:, :2]
        displacement = torch.norm(current_pos - self.last_pos, dim=-1)

        # Displacement since last checkpoint
        displacement = torch.norm(current_pos - self.last_pos, dim=-1)

        # Reward forward displacement
        displacement_reward = torch.clamp(displacement / 2.0, 0.0, 1.0)

        # Penalize near-zero displacement (stuck or circling)
        circle_pen = torch.exp(-displacement / 0.5)  # 1.0 when no movement, 0 when moved 2m+

        # Update checkpoint every 50 steps
        update_mask = (self.episode_length_buf % 50 == 0)
        self.last_pos = torch.where(update_mask[:, None], current_pos, self.last_pos)

        current_yaw = self._yaw_from_quat(self.robot.data.root_quat_w)

        # Accumulate yaw change
        yaw_delta = torch.abs(torch.atan2(
            torch.sin(current_yaw - self.last_yaw),
            torch.cos(current_yaw - self.last_yaw)
        ))
        self.yaw_accumulator += yaw_delta
        self.last_yaw = current_yaw

        # Reset accumulator every 50 steps
        update_mask = (self.episode_length_buf % 50 == 0)
        yaw_turned = self.yaw_accumulator.clone()
        self.yaw_accumulator = torch.where(update_mask, torch.zeros_like(self.yaw_accumulator), self.yaw_accumulator)

        # Penalize excessive yaw when moving forward
        # Turning is ok when blocked, wasteful when path is clear
        path_clear = (self.lidar_front > 1.5).float()
        unnecessary_yaw = yaw_turned * path_clear * (vx > 0.1).float()

        target = self.waypoints[self.current_waypoint_idx]

        # Separate x and y distances
        dx = target[:, 0] - current_pos[:, 0]
        dy = target[:, 1] - current_pos[:, 1]

        dist_to_waypoint = torch.sqrt(dx**2 + dy**2)

        # Progress in each axis separately
        prev_dx = self.waypoints[self.current_waypoint_idx][:, 0] - self.prev_pos_waypoint[:, 0]
        prev_dy = self.waypoints[self.current_waypoint_idx][:, 1] - self.prev_pos_waypoint[:, 1]

        x_progress = prev_dx.abs() - dx.abs()  # positive when getting closer in x
        y_progress = prev_dy.abs() - dy.abs()  # positive when getting closer in y

        # Reward each axis independently
        x_reward = torch.clamp(x_progress, -1.0, 1.0)
        y_reward = torch.clamp(y_progress, -1.0, 1.0)

        self.prev_pos_waypoint = current_pos.clone()

        # Bonus for reaching waypoint
        current_radius = self.waypoint_radii[self.current_waypoint_idx]  # (N,)
        reached = (dist_to_waypoint < current_radius).float()
        waypoint_bonus = reached * 20.0

        # Advance waypoint
        self.current_waypoint_idx = torch.where(
            dist_to_waypoint < 1.0,
            (self.current_waypoint_idx + 1) % len(self.waypoints),
            self.current_waypoint_idx
        )

        current_pos = self.robot.data.root_link_pos_w[:, :2]
        target = self.waypoints[self.current_waypoint_idx]
        to_target = target - current_pos  # (N, 2)
        to_target_norm = to_target / (torch.norm(to_target, dim=-1, keepdim=True) + 1e-4)

        # Project velocity onto direction to waypoint
        # Positive = moving toward waypoint, negative = moving away
        v_world = self.robot.data.root_lin_vel_w[:, :2]  # world frame velocity
        toward_waypoint = (v_world * to_target_norm).sum(dim=-1)

        # Reward moving toward waypoint, penalize moving away
        direction_reward = torch.clamp(toward_waypoint / 0.4, -1.0, 1.0)

        # -----------------------------
        # Final reward
        # -----------------------------
        reward = (
            0.05
            + 2.0 * upright
            + 3.0 * r_height
            - 6.0 * tilt_pen
            - 3.0 * pitch_pen
            - 12.0 * low_pen
            + 3.0 * r_vx * path_clear
            + 3.0 * r_lat * path_clear
            + 3.0 * r_yaw
            - 3.0 * underspeed * underspeed_weight * (1.0 - blocked)
            #- 1.0 * backward
            + 1.0 * r_three
            + 0.5 * r_one_swing
            - 1.0 * pen_two
            - 1.0 * pen_air
            - 3.0 * front_collapse
            - 0.05 * posture_cost
            - 3.0 * vz_pen
            - 0.5 * pitch_rate_pen
            - 1e-5 * act_cost
            - 5e-5 * actrate_cost
            # shaping (gated)
            - 0.08 * dead_leg_pen * gate
            - 0.12 * hind_imbalance * gate
            - 0.05 * front_imbalance * gate
            - 0.08 * hind_air_imbalance * gate
            #- 0.5 * yaw_track_pen * (1.0 - blocked)


            - 3.0 * collision_pen * obstacle_weight
            + 3.0 * sharp_turn * obstacle_weight
            - 3.0 * stuck_leg_pen * obstacle_weight

            + 3.0 * displacement_reward * obstacle_weight
            - 2.0 * circle_pen * obstacle_weight
            - 2.0 * unnecessary_yaw * obstacle_weight

            + 5.0 * x_reward * obstacle_weight
            + 5.0 * y_reward * obstacle_weight
            + waypoint_bonus * obstacle_weight
            + 4.0 * direction_reward * obstacle_weight

            


            
            #- 0.8 * near_pen * obstacle_weight
            #- 0.8 * slow_when_close * obstacle_weight
            #+ 2.0 * turn_reward * obstacle_weight
            #- 3.0 * collision_pen * obstacle_weight

            #+ 4.0 * reverse_reward * obstacle_weight
            #+ 1.6 * escape_turn_reward * obstacle_weight
            #+ 4.0 * persistent_escape_reward * obstacle_weight

            #+ 5.0 * clear_reward * obstacle_weight
            #+ 2.0 * forward_after_clear * obstacle_weight
            #+ 3.0 * sharp_turn * obstacle_weight

            #+ 3.0 * forward_progress * obstacle_weight
            #- 4.0 * reckless_forward * obstacle_weight
            #+ 5.0 * moving_to_open * obstacle_weight'''
        )

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        gravity_z = self.robot.data.projected_gravity_b[:, 2]
        tipped_over = gravity_z > -0.2

        base_height = self.robot.data.root_link_pos_w[:, 2]
        too_low = base_height < 0.16

        collided = (self.lidar_front < 0.35) | (self.lidar_left < 0.15) | (self.lidar_right < 0.15)


        terminated = tipped_over | too_low #| collided
        return terminated, collided #, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        self.pos_ref[env_ids] = self.robot.data.root_link_pos_w[env_ids, 0:2]

        robot = self.scene["robot"]

        yaw = self._yaw_from_quat(self.robot.data.root_quat_w)
        self.yaw_ref[env_ids] = yaw[env_ids]

        if self.joint_targets is None:
            self.joint_targets = self.robot.data.default_joint_pos.clone()
        else:
            self.joint_targets[env_ids] = self.robot.data.default_joint_pos[env_ids]



        root_state = robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.25
        root_state[:, 7:13] = 0.0
        spawn_yaw = math.pi / 2
        root_state[:, 3] = math.cos(spawn_yaw / 2)
        root_state[:, 4] = 0.0
        root_state[:, 5] = 0.0
        root_state[:, 6] = math.sin(spawn_yaw / 2)
        robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        joint_pos = robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        self.episode_length_buf[env_ids] = 0

        yaw = self._yaw_from_quat(self.robot.data.root_quat_w)
        self.yaw_cmd_int[env_ids] = yaw[env_ids]

        # Push spheres far away
        num_objects = MAX_OBS
        object_ids = torch.arange(num_objects, device=self.device)
        N = env_ids.numel()
        M = object_ids.numel()

        sphere_radius = 0.15
        z_center = sphere_radius + 0.01

        pos_xy = torch.full((N, M, 2), 9999.0, device=self.device)
        z = torch.full((N, M), z_center, device=self.device)
        pos = torch.cat([pos_xy, z.unsqueeze(-1)], dim=-1)
        quat = torch.zeros((N, M, 4), device=self.device)
        quat[..., 0] = 1.0
        object_pose = torch.cat([pos, quat], dim=-1)

        self.spheres.write_object_pose_to_sim(
            object_pose,
            env_ids=env_ids,
            object_ids=object_ids,
        )

        self.scene.write_data_to_sim()

        # Reset counters
        self.collision_counter[env_ids] = 0
        self.stuck_counter[env_ids] = 0
        self.leg_stuck_counter[env_ids] = 0
        self.last_yaw[env_ids] = yaw[env_ids]
        self.yaw_accumulator[env_ids] = 0.0
        self.last_pos[env_ids] = self.robot.data.root_link_pos_w[env_ids, :2]
        self.pos_update_counter[env_ids] = 0.0

        # Reset waypoint navigation
        self.current_waypoint_idx[env_ids] = 0
        robot_pos = self.robot.data.root_link_pos_w[env_ids, :2]
        self.prev_dist_to_waypoint[env_ids] = torch.norm(
            robot_pos - self.waypoints[0], dim=-1
        )

        # Initialize heading toward first waypoint
        to_first = self.waypoints[0] - robot_pos
        self.heading_ref[env_ids] = torch.atan2(to_first[:, 1], to_first[:, 0])
        self.prev_pos_waypoint[env_ids] = self.robot.data.root_link_pos_w[env_ids, :2]

        if self.prev_actions is not None:
            self.prev_actions[env_ids] = 0.0
        
