from pathlib import Path
from typing import cast
from evalio import datasets as ds, types as ty
from evalio.rerun import convert
import sys
import numpy as np
import rerun as rr

data = Path(sys.argv[1])
every = 10
if len(sys.argv) > 2:
    every = int(sys.argv[2])

# load trajectory
trajectory = ty.Trajectory.from_file(data)
if not isinstance(trajectory, ty.Trajectory):
    print(f"Failed to parse trajectory metadata: {trajectory}")
    quit()

# load dataset
trajectory = cast(ty.Trajectory[ty.Experiment], trajectory)
sequence = trajectory.metadata.sequence
dataset = ds.get_sequence(sequence)
dataset = cast(ds.Dataset, dataset)
lidar_params = dataset.lidar_params()
min2 = lidar_params.min_range**2
max2 = lidar_params.max_range**2

# create map
index = 0
all_points = []
for lidar in dataset.lidar():
    while trajectory.stamps[index] < lidar.stamp:
        index += 1
        if index >= len(trajectory.stamps):
            print("Reached end of trajectory stamps")
            quit()

    assert trajectory.stamps[index] == lidar.stamp

    if index % every == 0:
        rot = cast(np.ndarray, trajectory.poses[index].rot.toMat())
        pos = cast(np.ndarray, trajectory.poses[index].trans)

        local_points = np.asarray(lidar.to_vec_positions())
        dist2 = (
            local_points[:, 0] ** 2 + local_points[:, 1] ** 2 + local_points[:, 2] ** 2
        )
        local_points = local_points[np.logical_and(dist2 > min2, dist2 < max2)]

        points = local_points @ rot.T + pos
        all_points.append(points)

    print(index)
    if index > 1000:
        break

all_points = np.concatenate(all_points, axis=0)

rr.init("map_points")
rr.connect_grpc()
rr.log("points", convert(all_points, color="z"))
