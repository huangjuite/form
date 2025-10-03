from pathlib import Path
from typing import Optional, cast
from evalio import datasets
from evalio.cli.parser import PipelineBuilder
from evalio.types import Point, SE3

from itertools import chain
from tqdm import tqdm
import numpy as np
import pyvista as pv
import seaborn as sns

pv.global_theme.allow_empty_mesh = True


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--length", type=int, default=500)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/animation.gif")
    )
    return parser.parse_args()


def set_camera_position(
    pl: pv.Plotter,
    focal_point: tuple[float, float, float],
    distance: float,
    azimuth: float,
    elevation: float,
):
    """Set the camera position given spherical coordinates."""
    pl.camera.focal_point = focal_point
    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)
    x = focal_point[0] + distance * np.cos(elevation) * np.cos(azimuth)
    y = focal_point[1] + distance * np.cos(elevation) * np.sin(azimuth)
    z = focal_point[2] + distance * np.sin(elevation)
    pl.camera.position = (x, y, z)


def to_np(input: list[Point]) -> np.ndarray:
    return np.asarray([[p.x, p.y, p.z] for p in input])


def plot_scan(
    frame: SE3,
    points: dict[str, list[Point]],
    pl: pv.Plotter,
    actor: Optional[pv.Actor] = None,
) -> pv.Actor:
    # remove old actor
    if actor is not None:
        pl.remove_actor(actor, render=False)  # type: ignore

    # Add in new actor
    all_points = to_np(list(chain.from_iterable(points.values())))

    # transform to global frame
    pose = frame.toMat()
    R = pose[:3, :3]
    t = pose[:3, 3]
    all_points = all_points @ R.T + t

    return pl.add_points(
        pv.PointSet(all_points),
        color="black",
        point_size=6,
        render_points_as_spheres=True,
        lighting=True,
        render=False,
    )


def plot_map(
    points: dict[str, list[Point]],
    pl: pv.Plotter,
    actor: Optional[pv.Actor] = None,
) -> pv.Actor:
    if actor is not None:
        pl.remove_actor(actor, render=False)  # type: ignore

    use_glyph = False

    # make sphere glyph
    if use_glyph:
        sphere = pv.Sphere(radius=0.07, theta_resolution=8, phi_resolution=8)
        num = sphere.n_points
    else:
        num = 1

    # Sort into scan groups
    c = sns.color_palette("colorblind")
    all_points = list(chain.from_iterable(points.values()))
    all_colors = np.vstack([np.tile(c[p.col % len(c)], (num, 1)) for p in all_points])
    all_points = np.array([[p.x, p.y, p.z] for p in all_points])

    all_pv = pv.PolyData(all_points)
    if use_glyph:
        all_pv = all_pv.glyph(geom=sphere, scale=False, orient=False)  # type: ignore
        all_pv = cast(pv.PolyData, all_pv)

    return pl.add_points(
        all_pv,
        scalars=all_colors,
        rgb=True,
        smooth_shading=True,
        render_points_as_spheres=True,
        point_size=1.5,
        render=False,
    )


# ------------------------- Setup pipeline & data ------------------------- #
args = parse_args()

dataset = datasets.OxfordSpires.observatory_quarter_01
lidar_iter = iter(dataset.lidar())
pipe = PipelineBuilder.parse("form")[0].build(dataset)


# ------------------------- Setup pyvista ------------------------- #
pv.set_plot_theme("dark")
pl = pv.Plotter(
    off_screen=not args.show,
    window_size=[1280, 720],
    lighting="three lights",
)
scan_actor = None
map_actor = None

if args.show:
    pl.show(interactive_update=True, auto_close=False)
else:
    pl.open_gif(str(args.output), fps=args.fps, loop=False)

# ------------------------- Run! ------------------------- #
for i in tqdm(range(args.start), leave=False):
    next(lidar_iter)

for i in tqdm(range(args.length), leave=True):
    # run through pipeline
    features = pipe.add_lidar(next(lidar_iter))
    pose = pipe.pose()
    global_map = pipe.map()

    # update visualization
    # scan_actor = plot_scan(pose * dataset.imu_T_lidar(), features, pl, scan_actor)
    map_actor = plot_map(global_map, pl, map_actor)

    focal_point = (pose.trans[0], pose.trans[1], pose.trans[2])
    set_camera_position(
        pl,
        focal_point=focal_point,
        distance=90.0,
        azimuth=45.0,
        elevation=25.0,
    )

    if args.show:
        pl.update()
    else:
        pl.write_frame()

pl.close()
