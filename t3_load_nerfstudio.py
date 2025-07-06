# extractor_final_fixed.py

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import tyro
from typing import List, Optional, Tuple

import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from PIL import Image

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.rich_utils import CONSOLE

from ultralytics import YOLOWorld, ASSETS, FastSAM


# project_points のインポートは不要なため削除しました

# --- Helper Functions (変更なし) ---
def create_camera_to_world(
        radius: float, elevation_deg: float, azimuth_deg: float
) -> torch.Tensor:
    elevation_rad = np.radians(elevation_deg)
    azimuth_rad = np.radians(azimuth_deg)
    camera_origin = np.array([
        radius * np.cos(elevation_rad) * np.sin(azimuth_rad),
        radius * np.sin(elevation_rad),
        radius * np.cos(elevation_rad) * np.cos(azimuth_rad),
    ])
    forward = -camera_origin / np.linalg.norm(camera_origin)
    temp_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, temp_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = camera_origin
    return torch.from_numpy(c2w).float()


def save_points_as_ply(filepath: Path, points: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(str(filepath), pcd)
    CONSOLE.print(f"[bold green]✅ Saved {len(points)} points to {filepath}[/bold green]")


def visualize_projection_with_bbox(
        image: np.ndarray, bbox: List[int], projected_points: np.ndarray, output_path: Path
):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    x_min, y_min, x_max, y_max = bbox
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='red',
                             facecolor='none')
    ax.add_patch(rect)
    if projected_points.shape[0] > 0:
        ax.scatter(projected_points[:, 0], projected_points[:, 1], s=1, c='lime', alpha=0.7)
    ax.set_title(f"Projected 3D Points inside BBox ({len(projected_points)} points)")
    ax.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    CONSOLE.print(f"[bold green]✅ Saved BBox visualization to {output_path}[/bold green]")


def visualize_projection_with_mask(
        image: np.ndarray, mask: np.ndarray, projected_points: np.ndarray, output_path: Path
):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    overlay = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
    overlay[mask] = [1.0, 0.0, 0.0, 0.4]
    ax.imshow(overlay)
    if projected_points.shape[0] > 0:
        ax.scatter(projected_points[:, 0], projected_points[:, 1], s=1, c='lime', alpha=0.7)
    ax.set_title(f"Projected 3D Points inside Mask ({len(projected_points)} points)")
    ax.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    CONSOLE.print(f"[bold green]✅ Saved mask visualization to {output_path}[/bold green]")


class Nerf3DExtractor:
    def __init__(self, load_config: Path, device: str = "cuda:0"):
        self.device = device
        CONSOLE.print(f"Loading model from {load_config}...")
        _, self.pipeline, _, _ = eval_setup(load_config)
        self.pipeline.model.to(self.device)
        self.pipeline.model.eval()
        self.xyz = self.pipeline.model.gauss_params.means.to(self.device)
        self.camera: Optional[Cameras] = None
        self.rendered_image_np: Optional[np.ndarray] = None

        # ✨ --- 修正点 1: OpenGLからOpenCVへの座標系変換行列を定義 ---
        self.opengl_to_opencv_transform = torch.tensor([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32, device=self.device)

    # ✨ --- 修正点 1: 投影計算を自前で実装するプライベートメソッドを追加 ---
    def _project_points_manual(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        インスタンスの3D点群(self.xyz)を現在のカメラ(self.camera)の
        2D画像平面に手動で投影し、深度チェック用のマスクも返す。
        """
        # ワールド座標からカメラ座標へ変換するための行列(w2c)を取得
        # c2w = self.camera.camera_to_worlds
        c2w = self.camera.camera_to_worlds @ self.opengl_to_opencv_transform
        w2c = torch.linalg.inv(c2w)

        # 3D点を同次座標に変換 (N, 3) -> (N, 4)
        points_homogeneous = torch.cat(
            [self.xyz, torch.ones_like(self.xyz[..., :1])], dim=-1
        )

        # カメラ座標に変換 (N, 4)
        points_in_camera_frame = (w2c @ points_homogeneous.T).squeeze(0).T
        points_in_camera_xyz = points_in_camera_frame[..., :3]

        # 深度（カメラからの距離、Z軸）を取得し、深度チェック用のマスクを作成
        depth = points_in_camera_xyz[..., 2]
        depth_mask = depth > 0

        # ゼロ除算を避けるために、深度が非常に小さい値をクリップ
        safe_depth = torch.clamp(depth, min=1e-8)

        # 透視投影計算
        fx = self.camera.fx.squeeze()
        fy = self.camera.fy.squeeze()
        cx = self.camera.cx.squeeze()
        cy = self.camera.cy.squeeze()

        u = fx * (points_in_camera_xyz[..., 0] / safe_depth) + cx
        v = fy * (points_in_camera_xyz[..., 1] / safe_depth) + cy

        projected_points_2d = torch.stack([u, v], dim=-1)

        return projected_points_2d, depth_mask

    def define_camera(self, radius: float, elevation_deg: float, azimuth_deg: float, image_height: int = 720,
                      image_width: int = 1280):
        CONSOLE.print(f"Defining new camera view: radius={radius}, elevation={elevation_deg}, azimuth={azimuth_deg}")
        c2w = create_camera_to_world(radius, elevation_deg, azimuth_deg)
        # c2w_opengl = create_camera_to_world(radius, elevation_deg, azimuth_deg)

        # ✨ --- 修正点 2: 生成したカメラ行列をOpenCV形式に変換 ---
        # c2w_opencv = (c2w_opengl @ self.opengl_to_opencv_transform).unsqueeze(0)

        try:
            intrinsics = self.pipeline.datamanager.train_dataset.cameras
            idx = 130
            fx, fy, cx, cy = intrinsics.fx[idx, 0], intrinsics.fy[idx, 0], intrinsics.cx[idx, 0], intrinsics.cy[idx, 0]
        except Exception:
            CONSOLE.print("[yellow]Warning: Could not load intrinsics from dataset. Using default values.[/yellow]")
            fx, fy = image_width * 1.2, image_width * 1.2
            cx, cy = image_width / 2.0, image_height / 2.0
        self.camera = Cameras(
            fx=fx, fy=fy, cx=cx, cy=cy, height=image_height, width=image_width,
            camera_to_worlds=c2w.to(self.device), camera_type=CameraType.PERSPECTIVE
        )
        self.rendered_image_np = None

    # ✨ --- 修正点 2: JSONファイルからカメラを定義する新メソッド ---
    def define_camera_from_json(self, camera_path: Path, keyframe_index: int = 0):
        """
        NerfstudioのカメラパスJSONファイルからカメラの視点を定義します。

        Args:
            camera_path (Path): カメラパスJSONファイルへのパス。
            keyframe_index (int): 使用するキーフレームのインデックス（0から始まる）。
        """
        CONSOLE.print(f"Defining camera from '{camera_path}' using keyframe {keyframe_index}...")

        try:
            with open(camera_path, "r") as f:
                data = json.load(f)

            keyframe = data["keyframes"][keyframe_index]
            c2w_list = keyframe["matrix"]
            fov_deg = keyframe["fov"]
            height = data["render_height"]
            width = data["render_width"]
        except (KeyError, IndexError, FileNotFoundError) as e:
            CONSOLE.print(
                f"[bold red]Error: Failed to parse camera JSON file. Check path, format, and keyframe index.[/bold red]")
            CONSOLE.print(f"Details: {e}")
            return

        # c2w行列をTensorに変換
        c2w = torch.tensor(c2w_list, dtype=torch.float32).unsqueeze(0).reshape([1, 4, 4])
        # c2w_opengl = torch.tensor(c2w_list, dtype=torch.float32).unsqueeze(0).reshape([1, 4, 4])
        # c2w = c2w_opengl @ self.opengl_to_opencv_transform.cpu()

        # FOV (視野角) から焦点距離を計算 (垂直FOVを想定)
        fov_rad = torch.tensor(fov_deg * torch.pi / 180.0)
        fy = height / (2 * torch.tan(fov_rad * 0.5))
        # 多くのカメラではfxとfyはほぼ同じ。より正確にはアスペクト比で計算も可能。
        fx = fy

        # 主点を計算
        cx = width / 2.0
        cy = height / 2.0

        # Camerasオブジェクトを作成してインスタンス変数に格納
        self.camera = Cameras(
            fx=torch.tensor([[fx]], device=self.device),
            fy=torch.tensor([[fy]], device=self.device),
            cx=torch.tensor([[cx]], device=self.device),
            cy=torch.tensor([[cy]], device=self.device),
            height=torch.tensor([[int(height)]], device=self.device),
            width=torch.tensor([[int(width)]], device=self.device),
            camera_to_worlds=c2w.to(self.device),
            camera_type=CameraType.PERSPECTIVE,
        )

        self.rendered_image_np = None  # 以前のレンダリング結果をクリア
        CONSOLE.print("✅ Camera successfully defined from JSON.")

    def render_image(self) -> Optional[np.ndarray]:
        if self.camera is None:
            CONSOLE.print("[bold red]Error: Camera is not defined. Call define_camera() first.[/bold red]")
            return None
        if self.rendered_image_np is not None:
            CONSOLE.print("Returning cached image.")
            return self.rendered_image_np
        CONSOLE.print("Rendering new view...")
        with torch.no_grad():
            outputs = self.pipeline.model.get_outputs_for_camera(self.camera)
        rgb_tensor = outputs['rgb']
        self.rendered_image_np = (rgb_tensor.cpu().numpy() * 255).astype(np.uint8)
        return self.rendered_image_np

    def extract_and_save_from_bbox(self, bbox: List[int], output_ply_path: Path,
                                   visualization_path: Optional[Path] = None):
        if self.camera is None: return

        # ✨ --- 修正点 2: 新しいプライベートメソッドを呼び出す ---
        projected_points, depth_mask = self._project_points_manual()

        x_min, y_min, x_max, y_max = bbox
        bbox_mask = (projected_points[:, 0] >= x_min) & (projected_points[:, 0] < x_max) & \
                    (projected_points[:, 1] >= y_min) & (projected_points[:, 1] < y_max)
        final_mask = bbox_mask & depth_mask

        if not torch.any(final_mask):
            CONSOLE.print("[bold yellow]Warning: No 3D points found in the BBox.[/bold yellow]")
            return

        selected_3d_points = self.xyz[final_mask]
        save_points_as_ply(output_ply_path, selected_3d_points.detach().cpu().numpy())

        if visualization_path:
            rendered_image = self.render_image()
            if rendered_image is not None:
                selected_2d_points = projected_points[final_mask].detach().cpu().numpy()
                visualize_projection_with_bbox(rendered_image, bbox, selected_2d_points, visualization_path)

    def extract_and_save_from_mask(
            self,
            segmentation_mask: np.ndarray,
            output_ply_path: Path,
            visualization_path: Optional[Path] = None
    ):
        if self.camera is None:
            CONSOLE.print("[bold red]Error: Camera is not defined. Call define_camera() first.[/bold red]")
            return

        H, W = segmentation_mask.shape
        if H != self.camera.height or W != self.camera.width:
            CONSOLE.print(
                f"[bold red]Error: Mask shape ({H}, {W}) does not match camera render shape ({self.camera.height}, {self.camera.width}).[/bold red]")
            return

        CONSOLE.print(f"Filtering 3D points using segmentation mask...")

        # ✨ --- 修正点 3: 新しいプライベートメソッドを呼び出す ---
        projected_points, depth_mask = self._project_points_manual()

        coords = torch.round(projected_points).long()
        u, v = coords[:, 0], coords[:, 1]

        bounds_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        mask_tensor = torch.from_numpy(segmentation_mask).to(self.device)

        segment_mask_on_points = torch.zeros_like(depth_mask)
        valid_indices = torch.where(bounds_mask)[0]
        segment_mask_on_points[valid_indices] = mask_tensor[v[valid_indices], u[valid_indices]]

        final_mask = segment_mask_on_points & depth_mask

        if not torch.any(final_mask):
            CONSOLE.print("[bold yellow]Warning: No 3D points found within the segmentation mask.[/bold yellow]")
            return

        selected_3d_points = self.xyz[final_mask]
        save_points_as_ply(output_ply_path, selected_3d_points.detach().cpu().numpy())

        if visualization_path:
            rendered_image = self.render_image()
            if rendered_image is not None:
                selected_2d_points = projected_points[final_mask].detach().cpu().numpy()
                visualize_projection_with_mask(rendered_image, segmentation_mask, selected_2d_points,
                                               visualization_path)


# --- Example Usage (Updated) ---
def main(
        load_config: Path = Path("path/to/your/experiment/config.yaml"),
        camera_json_path: Optional[Path] = None,
        output_dir: Path = Path("output_extractions/"),
):
    if str(load_config) == "path/to/your/experiment/config.yaml":
        CONSOLE.print("[bold red]Please specify the path to your config.yaml using --load-config[/bold red]")
        return

    # Initialize a YOLO-World model
    model = YOLOWorld("yolov8x-worldv2.pt")  # or select yolov8m/l-world.pt for different sizes
    # Define custom classes
    model.set_classes(["plant"])

    # Profile FastSAM-s
    fast_sam = FastSAM("FastSAM-s.pt")
    fast_sam.info()
    fast_sam(ASSETS)

    extractor = Nerf3DExtractor(load_config)

    # --- ✨ 修正点 3: 新しいメソッドの使用例を追加 ---
    if camera_json_path is not None and camera_json_path.exists():
        CONSOLE.print("\n[bold cyan]--- Scenario: Defining camera from JSON and extracting with a mask ---[/bold cyan]")
        extractor.define_camera_from_json(camera_json_path, keyframe_index=0)

        image_np = extractor.render_image()
        image = Image.fromarray(image_np)
        if image is not None:
            results = model.predict(image,
                        conf=0.05,
                        iou=0.1)

            car_bbox = results[0].boxes.xyxy[0].detach().cpu().numpy()

            # make segmentation mask
            dummy_mask = fast_sam.predict(image, bboxes=car_bbox, imgsz=1920)
            mask_array = dummy_mask[0].masks.data[0, 4:1084, :].detach().cpu().numpy().astype(np.bool_)

            extractor.extract_and_save_from_mask(
                segmentation_mask=mask_array,
                output_ply_path=output_dir / "json_cam_mask_points.ply",
                visualization_path=output_dir / "json_cam_mask_visualization.png"
            )
    else:
        CONSOLE.print(
            "\n[bold yellow]No camera_json_path provided or file not found. Running default scenarios.[/bold yellow]")
        # --- 従来の方法のシナリオ ---
        CONSOLE.print("\n[bold cyan]--- Scenario: Defining camera with angles and extracting from BBox ---[/bold cyan]")
        extractor.define_camera(radius=3.5, elevation_deg=15, azimuth_deg=30)
        _ = extractor.render_image()
        car_bbox = [550, 450, 850, 650]
        extractor.extract_and_save_from_bbox(
            bbox=car_bbox,
            output_ply_path=output_dir / "bbox_car_points.ply",
            visualization_path=output_dir / "bbox_car_visualization.png"
        )


def main2(
        load_config: Path = Path("path/to/your/experiment/config.yaml"),
        camera_json_path: Optional[Path] = None,
        output_dir: Path = Path("output_extractions/"),
):
    if str(load_config) == "path/to/your/experiment/config.yaml":
        CONSOLE.print("[bold red]Please specify the path to your config.yaml using --load-config[/bold red]")
        return

    # Initialize a YOLO-World model
    model = YOLOWorld("yolov8x-worldv2.pt")  # or select yolov8m/l-world.pt for different sizes
    # Define custom classes
    model.set_classes(["plant"])

    extractor = Nerf3DExtractor(load_config)

    # --- ✨ 修正点 3: 新しいメソッドの使用例を追加 ---
    if camera_json_path is not None and camera_json_path.exists():
        CONSOLE.print("\n[bold cyan]--- Scenario: Defining camera from JSON and extracting with a mask ---[/bold cyan]")
        extractor.define_camera_from_json(camera_json_path, keyframe_index=0)

        image = Image.fromarray(extractor.render_image())
        if image is not None:
            CONSOLE.print(
                "\n[bold yellow]No camera_json_path provided or file not found. Running default scenarios.[/bold yellow]")
            # --- 従来の方法のシナリオ ---
            CONSOLE.print(
                "\n[bold cyan]--- Scenario: Defining camera with angles and extracting from BBox ---[/bold cyan]")

            # detect plant from image
            # Execute inference with the YOLOv8s-world model on the specified image
            results = model.predict(image,
                        conf=0.05,
                        iou=0.1)

            car_bbox = results[0].boxes.xyxy[0]
            extractor.extract_and_save_from_bbox(
                bbox=car_bbox,
                output_ply_path=output_dir / "bbox_car_points.ply",
                visualization_path=output_dir / "bbox_car_visualization.png"
            )
    else:
        CONSOLE.print(
            "\n[bold yellow]No camera_json_path provided or file not found. Running default scenarios.[/bold yellow]")
        # --- 従来の方法のシナリオ ---
        CONSOLE.print("\n[bold cyan]--- Scenario: Defining camera with angles and extracting from BBox ---[/bold cyan]")
        extractor.define_camera(radius=3.5, elevation_deg=15, azimuth_deg=30)
        _ = extractor.render_image()
        car_bbox = [550, 450, 850, 650]
        extractor.extract_and_save_from_bbox(
            bbox=car_bbox,
            output_ply_path=output_dir / "bbox_car_points.ply",
            visualization_path=output_dir / "bbox_car_visualization.png"
        )


model = YOLOWorld("yolov8s-worldv2.pt")

if __name__ == "__main__":
    config_yaml = Path("outputs/plant_pot/splatfacto/2025-05-11_111819/config.yml")
    out_dir = Path("analysis/exports/plant_pot2/bbox_transition")
    camera_json_path = Path("analysis/plant_pot2/camera_paths/2025-07-05-16-27-26.json")

    main(
        load_config= config_yaml,
        output_dir=out_dir,
        camera_json_path=camera_json_path
    )

    # main2(
    #     load_config= config_yaml,
    #     output_dir=out_dir,
    #     camera_json_path=camera_json_path
    # )


