import omni.kit.commands

import omni.kit.app
import omni.usd
manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate("isaacsim.util.merge_mesh", True)

from isaacsim.util.merge_mesh import MeshMerger

from pxr import UsdGeom, Vt, Gf, Usd

import numpy as np

def sphere_to_mesh_points(sphere_prim, world_transform, subdivisions=8):
    """Generate mesh points for a USD Sphere prim."""
    sphere = UsdGeom.Sphere(sphere_prim)
    radius = sphere.GetRadiusAttr().Get()

    # Generate UV sphere points
    points = []
    indices = []
    counts = []

    for i in range(subdivisions + 1):
        lat = np.pi * (-0.5 + i / subdivisions)
        for j in range(subdivisions + 1):
            lon = 2 * np.pi * j / subdivisions
            x = radius * np.cos(lat) * np.cos(lon)
            y = radius * np.cos(lat) * np.sin(lon)
            z = radius * np.sin(lat)
            pt = world_transform.Transform(Gf.Vec3d(x, y, z))
            points.append(Gf.Vec3f(pt))

    # Generate quad faces
    for i in range(subdivisions):
        for j in range(subdivisions):
            a = i * (subdivisions + 1) + j
            b = a + 1
            c = a + (subdivisions + 1)
            d = c + 1
            indices.extend([a, b, d, c])
            counts.append(4)

    return points, indices, counts

def sphere_to_mesh_points_at(center, radius: float, subdivisions: int = 8):
    import numpy as np
    points = []
    indices = []
    counts = []

    for i in range(subdivisions + 1):
        lat = np.pi * (-0.5 + i / subdivisions)
        for j in range(subdivisions + 1):
            lon = 2 * np.pi * j / subdivisions
            x = float(center[0]) + radius * np.cos(lat) * np.cos(lon)
            y = float(center[1]) + radius * np.cos(lat) * np.sin(lon)
            z = float(center[2]) + radius * np.sin(lat)
            points.append(Gf.Vec3f(x, y, z))

    for i in range(subdivisions):
        for j in range(subdivisions):
            a = i * (subdivisions + 1) + j
            b = a + 1
            c = a + (subdivisions + 1)
            d = c + 1
            indices.extend([a, b, d, c])
            counts.append(4)

    return points, indices, counts

def merge_spheres_for_lidar_from_states(env, num_spheres: int, output_path: str = "/World/merged_spheres"):
    stage = omni.usd.get_context().get_stage()
    sphere_pos = env.scene["spheres"].data.object_pos_w  # (num_envs, M, 3)
    radius = 0.15
    subdivisions = 4  # reduced for performance

    all_points = []
    all_indices = []
    all_counts = []
    offset = 0

    num_envs = sphere_pos.shape[0]
    for env_idx in range(num_envs):
        for i in range(num_spheres):
            center = sphere_pos[env_idx, i]
            pts, idxs, cnts = sphere_to_mesh_points_at(center, radius, subdivisions)
            all_points.extend(pts)
            all_indices.extend([idx + offset for idx in idxs])
            all_counts.extend(cnts)
            offset += len(pts)

    merged = UsdGeom.Mesh.Define(stage, output_path)
    merged.GetPointsAttr().Set(Vt.Vec3fArray(all_points))
    merged.GetFaceVertexIndicesAttr().Set(Vt.IntArray(all_indices))
    merged.GetFaceVertexCountsAttr().Set(Vt.IntArray(all_counts))
    return output_path



from isaaclab.utils import configclass
import gymnasium as gym
import numpy as np
from isaaclab.assets import ArticulationCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.sensors import RayCasterCfg, patterns, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
import gymnasium as gym
import numpy as np
from isaaclab.sensors import CameraCfg, ContactSensorCfg
import torch

from isaaclab.assets import RigidObjectCollectionCfg, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import UsdFileCfg


MAX_OBS = 1

@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    """
    ground = AssetBaseCfg(
        prim_path="/World/environment",
        spawn=UsdFileCfg(
            usd_path="/Isaac/Environments/Simple_Warehouse/warehouse.usd"
        )
    )"""

    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.02,
        attach_yaw_only=False,
        max_distance=50.0,
        pattern_cfg=patterns.BpearlPatternCfg(
            horizontal_fov=360.0,
            horizontal_res=0.5,   # degrees per step (NOT 1024)
            vertical_ray_angles=[
                float(x) for x in [
                    -15.0, -14.032258, -13.064516, -12.096774, -11.129032, -10.161290, -9.193548, -8.225806,
                    -7.258065, -6.290323, -5.322581, -4.354839, -3.387097, -2.419355, -1.451613, -0.483871,
                    0.483871,  1.451613,  2.419355,  3.387097,  4.354839,  5.322581,  6.290323,  7.258065,
                    8.225806,  9.193548, 10.161290, 11.129032, 12.096774, 13.064516, 14.032258, 15.0
                ]
            ],
        ),
        mesh_prim_paths=["/World/merged_spheres"],
        debug_vis=False,
    )

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",  # regex for all feet
        update_period=0.0,                         # every sim step
        history_length=1,
        track_air_time=True,                       # VERY useful for walking
        force_threshold=5.0,                       # tune: 3–10 N typical
        debug_vis=False,
    )

    front_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        )

    depth_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/depth_camera",
        update_period=0.1,
        height=64,
        width=64,
        data_types=["distance_to_camera"],  # depth data
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0)  # 10m max depth range
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 0.05),
            rot=(0.707, 0.0, 0.707, 0.0),
            convention="ros"
        ),
    )


    
    spheres: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            f"sphere_{i}": RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/sphere_{i}",
                spawn=sim_utils.SphereCfg(
                    radius=0.15,
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True
                    ),
                ),
            )
            for i in range(MAX_OBS)
        },
    )



    env_spacing = 0
    replicate_physics = True