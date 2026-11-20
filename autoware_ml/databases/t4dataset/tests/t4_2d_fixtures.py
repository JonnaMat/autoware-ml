# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Builder of a small camera-only T4 scene annotated with 2D boxes, for the tests."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path

import cv2
import numpy as np


def write_scene(
    database_root: Path,
    dataset_name: str,
    scene_id: str,
    num_frames: int = 3,
    categories: Sequence[str] = ("green", "crosswalk_red", "billboard"),
    image_size: tuple[int, int] = (48, 64),
) -> Path:
    """Write the annotation tables and the images of one scene.

    The scene has one camera with ``num_frames`` key frames and one frame that is no key
    frame. Key frame ``i`` carries ``i`` boxes, cycling through the categories, so the first
    one has none.

    Args:
        database_root: Root directory of the database.
        dataset_name: Name of the dataset the scene belongs to.
        scene_id: ID of the scene.
        num_frames: Number of key frames.
        categories: Category names of the scene.
        image_size: ``(height, width)`` of the images.

    Returns:
        Root directory of the scene.
    """
    scene_root = database_root / dataset_name / scene_id / "0"
    annotation_dir = scene_root / "annotation"
    image_dir = scene_root / "data" / "CAM_TRAFFIC_LIGHT_NEAR"
    annotation_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)

    sample_data = []
    object_ann = []
    ego_pose = []
    for frame in range(num_frames + 1):
        is_key_frame = frame < num_frames
        token = f"{scene_id}_frame_{frame}"
        cv2.imwrite(
            str(image_dir / f"{frame}.jpg"), np.full((*image_size, 3), 40 * frame, dtype=np.uint8)
        )
        ego_pose.append(
            {
                "token": f"pose_{frame}",
                "translation": [float(frame), 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "timestamp": 1_000_000 * (frame + 1),
            }
        )
        sample_data.append(
            {
                "token": token,
                "sample_token": f"sample_{frame}",
                "ego_pose_token": f"pose_{frame}",
                "calibrated_sensor_token": "calibrated_camera",
                "filename": f"data/CAM_TRAFFIC_LIGHT_NEAR/{frame}.jpg",
                "fileformat": "jpg",
                "width": image_size[1],
                "height": image_size[0],
                "timestamp": 1_000_000 * (frame + 1),
                "is_key_frame": is_key_frame,
            }
        )
        if not is_key_frame:
            continue
        for box in range(frame):
            category = box % len(categories)
            object_ann.append(
                {
                    "token": f"{token}_box_{box}",
                    "sample_data_token": token,
                    "instance_token": f"instance_{box}",
                    "category_token": f"category_{category}",
                    "attribute_tokens": ["attribute_0"],
                    "bbox": [4.0 + 10 * box, 6.0, 12.0 + 10 * box, 20.0 + frame],
                }
            )

    tables = {
        "sensor": [{"token": "camera", "channel": "CAM_TRAFFIC_LIGHT_NEAR", "modality": "camera"}],
        "calibrated_sensor": [
            {
                "token": "calibrated_camera",
                "sensor_token": "camera",
                "translation": [1.0, 0.0, 2.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "camera_intrinsic": [[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]],
                "camera_distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        ],
        "ego_pose": ego_pose,
        "category": [
            {"token": f"category_{index}", "name": name} for index, name in enumerate(categories)
        ],
        "attribute": [{"token": "attribute_0", "name": "occlusion_state.none"}],
        "scene": [{"token": "scene", "name": f"recording_of_{scene_id}"}],
        "sample_data": sample_data,
        "object_ann": object_ann,
    }
    for name, table in tables.items():
        (annotation_dir / f"{name}.json").write_text(json.dumps(table))
    return scene_root


def write_scenario_list(
    scenario_root: Path, dataset_name: str, splits: dict[str, Sequence[str]]
) -> None:
    """Write the scenario list of a dataset, the scenes given as ``<scene>   <version>``.

    Args:
        scenario_root: Directory of the scenario lists.
        dataset_name: Name of the dataset, the file name of its list.
        splits: Scene IDs of every split.
    """
    scenario_root.mkdir(parents=True, exist_ok=True)
    lines = ["version: 1"]
    for split in ("train", "val", "test"):
        scenes = splits.get(split, [])
        lines.append(f"{split}:" if scenes else f"{split}: []")
        lines.extend(f"- {scene}   0" for scene in scenes)
    (scenario_root / f"{dataset_name}.yaml").write_text("\n".join(lines) + "\n")
