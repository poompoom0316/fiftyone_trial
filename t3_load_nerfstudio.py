# extractor.py

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import tyro
from typing import List, Optional, Tuple

import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.rich_utils import CONSOLE


# --- Helper Functions (No changes from previous script) ---
def create_camera_to_world(
        radius: float, elevation_deg: float, azimuth_deg: float
) -> torch.Tensor:
    """球面座標からカメラのc2w（camera-to-world）行列を生成します。"""
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
    """3D点群をPLYファイルとして保存する"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(str(filepath), pcd)
    CONSOLE.print(f"[bold green]✅ Saved {len(points)} points to {filepath}[/bold green]")


def visualize_projection(
        image: np.ndarray,
        bbox: List[int],
        projected_points: np.ndarray,
        output_path: Path,
):
    """画像、BBox、投影点を可視化して保存する"""
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    x_min, y_min, x_max, y_max = bbox
    rect = patches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)
    if projected_points.shape[0] > 0:
        ax.scatter(projected_points[:, 0], projected_points[:, 1], s=1, c='lime', alpha=0.7)
    ax.set_title(f"Projected 3D Points inside BBox ({len(projected_points)} points)")
    ax.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    CONSOLE.print(f"[bold green]✅ Saved visualization to {output_path}[/bold green]")


# --- Main Class ---

class Nerf3DExtractor:
    """
    Nerfstudioモデルから特定の視点の画像をレンダリングし、
    その画像上のBBoxに対応する3D点群を抽出するクラス。
    """

    def __init__(self, load_config: Path, device: str = "cuda:0"):
        """
        クラスを初期化し、訓練済みモデルをロードします。

        Args:
            load_config (Path): 訓練済みモデルのconfig.yamlへのパス。
            device (str): 計算に使用するデバイス。
        """
        self.device = device
        CONSOLE.print(f"Loading model from {load_config}...")
        _, self.pipeline, _, _ = eval_setup(load_config)
        self.pipeline.model.to(self.device)
        self.pipeline.model.eval()

        self.xyz = self.pipeline.model.gaussians.get_xyz.to(self.device)

        # 現在のカメラとレンダリング結果を保持する属性
        self.camera: Optional[Cameras] = None
        self.rendered_image_np: Optional[np.ndarray] = None

    def define_camera(
            self,
            radius: float,
            elevation_deg: float,
            azimuth_deg: float,
            image_height: int = 720,
            image_width: int = 1280,
    ):
        """
        レンダリングと投影に使用するカメラの視点を定義します。

        Args:
            radius: カメラの原点からの距離。
            elevation_deg: カメラの仰角（度）。
            azimuth_deg: カメラの方位角（度）。
            image_height: レンダリング画像の高さ。
            image_width: レンダリング画像の幅。
        """
        CONSOLE.print(f"Defining new camera view: radius={radius}, elevation={elevation_deg}, azimuth={azimuth_deg}")
        c2w = create_camera_to_world(radius, elevation_deg, azimuth_deg).unsqueeze(0)

        try:
            intrinsics = self.pipeline.datamanager.train_dataset.cameras
            fx, fy, cx, cy = intrinsics.fx[0, 0], intrinsics.fy[0, 0], intrinsics.cx[0, 0], intrinsics.cy[0, 0]
        except Exception:
            CONSOLE.print("[yellow]Warning: Could not load intrinsics from dataset. Using default values.[/yellow]")
            fx, fy = image_width * 1.2, image_width * 1.2
            cx, cy = image_width / 2.0, image_height / 2.0

        self.camera = Cameras(
            fx=fx, fy=fy, cx=cx, cy=cy,
            height=image_height, width=image_width,
            camera_to_worlds=c2w.to(self.device),
            camera_type=CameraType.PERSPECTIVE,
        )
        # カメラが変更されたので、以前のレンダリング結果をクリア
        self.rendered_image_np = None

    def render_image(self) -> Optional[np.ndarray]:
        """
        定義されたカメラ視点から画像をレンダリングし、Numpy配列として返します。
        2回目以降の呼び出しではキャッシュされた画像を返します。

        Returns:
            Optional[np.ndarray]: レンダリングされたRGB画像 (H, W, 3) uint8。
        """
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

    def extract_and_save(
            self,
            bbox: List[int],
            output_ply_path: Path,
            visualization_path: Optional[Path] = None
    ):
        """
        BBoxに対応する3D点を抽出し、結果をファイルに保存します。

        Args:
            bbox (List[int]): 抽出対象のバウンディングボックス [x_min, y_min, x_max, y_max]。
            output_ply_path (Path): 出力する3D点群ファイル（.ply）のパス。
            visualization_path (Optional[Path]): 投影結果を可視化する画像の保存パス。
        """
        if self.camera is None:
            CONSOLE.print("[bold red]Error: Camera is not defined. Call define_camera() first.[/bold red]")
            return

        # 1. 3D点群の取得とフィルタリングの準備
        CONSOLE.print(f"Filtering 3D points within BBox: {bbox}...")

        # 2. 3D点をレンダリングした視点から2Dへ投影
        projected_points = self.camera.project(self.xyz, project_to_screen=True).squeeze(0)

        # 3. 深度チェック（カメラ前方）
        w2c = torch.linalg.inv(self.camera.camera_to_worlds)
        xyz_h = torch.cat([self.xyz, torch.ones_like(self.xyz[..., :1])], dim=-1)
        points_cam = (w2c.unsqueeze(1) @ xyz_h.unsqueeze(-1)).squeeze()
        depth_mask = points_cam[:, 2] > 0

        # 4. BBox内の点をフィルタリング
        x_min, y_min, x_max, y_max = bbox
        bbox_mask = (projected_points[:, 0] >= x_min) & (projected_points[:, 0] < x_max) & \
                    (projected_points[:, 1] >= y_min) & (projected_points[:, 1] < y_max)

        final_mask = bbox_mask & depth_mask

        if not torch.any(final_mask):
            CONSOLE.print("[bold yellow]Warning: No 3D points found in the BBox.[/bold yellow]")
            return

        selected_3d_points = self.xyz[final_mask]

        # 5. 結果の保存
        output_ply_path.parent.mkdir(parents=True, exist_ok=True)
        save_points_as_ply(output_ply_path, selected_3d_points.cpu().numpy())

        if visualization_path:
            # 可視化のためにレンダリング画像が必要
            rendered_image = self.render_image()
            if rendered_image is not None:
                selected_2d_points = projected_points[final_mask].cpu().numpy()
                visualization_path.parent.mkdir(parents=True, exist_ok=True)
                visualize_projection(rendered_image, bbox, selected_2d_points, visualization_path)


# --- Example Usage ---
def main(
        load_config: Path = Path("path/to/your/experiment/config.yaml"),
        ply_output_dir: Path = Path("output_plys/"),
        viz_output_dir: Path = Path("output_visualizations/"),
):
    """
    Nerf3DExtractorクラスの使用例を示すメイン関数。
    """
    if str(load_config) == "path/to/your/experiment/config.yaml":
        CONSOLE.print("[bold red]Please specify the path to your config.yaml using --load-config[/bold red]")
        return

    # 1. Extractorを初期化
    extractor = Nerf3DExtractor(load_config)

    # --- シナリオ1: 車を抽出 ---
    CONSOLE.print("\n[bold cyan]--- Scenario 1: Extracting a 'car' ---[/bold cyan]")

    # 2. 視点を定義
    extractor.define_camera(radius=3.5, elevation_deg=15, azimuth_deg=30)

    # 3. 画像をレンダリング（物体検出モデルにかけることを想定）
    image_np = extractor.render_image()
    # (ここで物体検出モデルを呼び出す)
    # 例: bboxes = object_detection_model(image_np)
    # 今回はBBoxをハードコーディングします
    car_bbox = [550, 450, 850, 650]

    # 4. BBoxから3D点を抽出して保存
    extractor.extract_and_save(
        bbox=car_bbox,
        output_ply_path=ply_output_dir / "car_points.ply",
        visualization_path=viz_output_dir / "car_visualization.png"
    )

    # --- シナリオ2: 別のオブジェクトを別の角度から抽出 ---
    CONSOLE.print("\n[bold cyan]--- Scenario 2: Extracting a 'sign' from another angle ---[/bold cyan]")

    # 2. 新しい視点を定義
    extractor.define_camera(radius=2.0, elevation_deg=25, azimuth_deg=-20)

    # 3. 新しい画像をレンダリング
    _ = extractor.render_image()  # レンダリングを確実に実行
    sign_bbox = [800, 300, 950, 480]

    # 4. 新しいBBoxから3D点を抽出
    extractor.extract_and_save(
        bbox=sign_bbox,
        output_ply_path=ply_output_dir / "sign_points.ply",
        visualization_path=viz_output_dir / "sign_visualization.png"
    )


if __name__ == "__main__":
    tyro.cli(main)
