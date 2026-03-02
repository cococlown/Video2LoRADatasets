"""
视频处理模块：从视频中随机抽取帧并生成人物数据集
"""

import os
import random
import threading
from pathlib import Path
from typing import Callable, Optional, Tuple
from PIL import Image
import torch
import cv2
import numpy as np

# 尝试导入 YOLO
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# 尝试导入人脸聚类相关依赖
try:
    from sklearn.cluster import DBSCAN
    from insightface.app import FaceAnalysis
    FACE_CLUSTER_AVAILABLE = True
except ImportError:
    FACE_CLUSTER_AVAILABLE = False


def get_smart_crop_box(img_w, img_h, bbox, padding_ratio=0.15, min_side_len=800):
    """
    计算智能裁切框 (任意比例模式)：
    1. 以人物为中心，增加基础安全边距。
    2. 检查裁切框尺寸，如果宽或高不足 min_side_len (800)，则向外扩展直至满足要求 (或达到原图边界)。
    3. 不再强制匹配原图长宽比，允许生成竖构图或方构图，减少横向图片中的无用背景。
    """
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1

    # --- 1. 基础扩展 (Padding) ---
    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio

    center_x = x1 + box_w / 2
    center_y = y1 + box_h / 2

    current_w = box_w + (pad_x * 2)
    current_h = box_h + (pad_y * 2)

    # --- 2. 最小边长保障 ---
    long_side = max(current_w, current_h)
    target_w = max(current_w, min(long_side / 2, img_w))
    target_h = max(current_h, min(long_side / 2, img_h))

    # --- 3. 坐标计算与平移 ---
    final_x1 = center_x - target_w / 2
    final_y1 = center_y - target_h / 2
    final_x2 = center_x + target_w / 2
    final_y2 = center_y + target_h / 2

    if final_x1 < 0:
        offset = -final_x1
        final_x1 += offset
        final_x2 += offset
    if final_x2 > img_w:
        offset = final_x2 - img_w
        final_x1 -= offset
        final_x2 -= offset

    if final_y1 < 0:
        offset = -final_y1
        final_y1 += offset
        final_y2 += offset
    if final_y2 > img_h:
        offset = final_y2 - img_h
        final_y1 -= offset
        final_y2 -= offset

    final_x1 = max(0, final_x1)
    final_y1 = max(0, final_y1)
    final_x2 = min(img_w, final_x2)
    final_y2 = min(img_h, final_y2)

    return int(final_x1), int(final_y1), int(final_x2), int(final_y2)


class FaceCluster:
    """人脸聚类器（从 dataset_select.py 复用）"""

    def __init__(self):
        if not FACE_CLUSTER_AVAILABLE:
            print("[FaceCluster] 缺少依赖（sklearn/insightface），禁用人脸聚类功能。")
            self.enabled = False
            return

        print("正在加载 InsightFace (buffalo_l)...")
        try:
            self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.enabled = True
        except Exception as e:
            print(f"[FaceCluster] 初始化失败，禁用人脸聚类: {e}")
            self.enabled = False
            return

        self.embeddings = []
        self.filenames = []

    def extract_feature(self, img_pil, filename):
        """从 PIL 图片中提取人脸特征"""
        if not getattr(self, 'enabled', False):
            return False

        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        faces = self.app.get(img_cv2)

        if len(faces) == 0:
            return False

        best_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        embedding = best_face.embedding
        embedding = embedding / np.linalg.norm(embedding)

        self.embeddings.append(embedding)
        self.filenames.append(filename)
        return True

    def run_clustering_and_move(self, dest_dir, eps=0.95, min_samples=3):
        """执行聚类并按人物分文件夹"""
        if not getattr(self, 'enabled', False):
            print("[FaceCluster] 功能被禁用，跳过聚类。")
            return

        if not self.embeddings:
            print("没有提取到任何人脸特征。")
            return

        print(f"正在对 {len(self.embeddings)} 张人脸进行聚类分析...")
        X = np.array(self.embeddings)

        clt = DBSCAN(metric="euclidean", n_jobs=-1, eps=eps, min_samples=min_samples)
        clt.fit(X)

        labels = clt.labels_
        unique_labels = set(labels)
        print(f"聚类完成，共发现 {len(unique_labels) - (1 if -1 in labels else 0)} 个不同的人物。")

        dest_path = Path(dest_dir)

        for file_name, label in zip(self.filenames, labels):
            src_file = dest_path / file_name
            if label == -1:
                target_folder = dest_path / "Uncategorized"
            else:
                target_folder = dest_path / f"Person_{label:03d}"

            target_folder.mkdir(exist_ok=True)
            try:
                import shutil
                shutil.move(str(src_file), str(target_folder / file_name))
            except Exception as e:
                print(f"移动文件失败: {e}")


class VideoProcessor:
    """视频处理器：抽取帧并进行人物检测与裁切"""

    def __init__(self, model_name: str = 'yolov8x.pt'):
        self.model_name = model_name
        self.model = None
        self.device = None

    def _init_model(self):
        """延迟初始化模型"""
        if self.model is not None:
            return

        if YOLO is None:
            raise ImportError("ultralytics 未安装，请运行: pip install ultralytics")

        print(f"正在加载 {self.model_name} 模型...")
        self.model = YOLO(self.model_name)

        if torch.cuda.is_available():
            self.device = 'cuda'
            self.model.to(self.device)
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU 加速已开启: {gpu_name}")
        else:
            self.device = 'cpu'
            print("⚠️ 警告: 未检测到 GPU，正在使用 CPU (速度较慢)。")

    def get_video_info(self, video_path: str) -> Tuple[int, float, int, int]:
        """
        获取视频信息
        返回: (总帧数, 帧率FPS, 宽度, 高度)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return total_frames, fps, width, height

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[Image.Image]:
        """
        处理单帧：检测人物并智能裁切
        返回: 处理后的 PIL Image 或 None
        """
        self._init_model()

        # BGR -> RGB -> PIL
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        img_w, img_h = img.size

        # 人物检测
        results = self.model.predict(img, classes=[0], conf=0.5, verbose=False)

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            # 选择面积最大的人
            best_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

            # 智能裁切
            crop_box = get_smart_crop_box(img_w, img_h, best_box, padding_ratio=0.15, min_side_len=800)
            img = img.crop(crop_box)
        else:
            return None  # 未检测到人物

        # 尺寸调整（最长边不超过1280）
        cur_w, cur_h = img.size
        max_side = max(cur_w, cur_h)
        if max_side > 1280:
            scale = 1280 / max_side
            new_w = int(cur_w * scale)
            new_h = int(cur_h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return img

    def process_video(
        self,
        video_path: str,
        dest_dir: str,
        num_frames: int,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None
    ) -> Tuple[int, int]:
        """
        处理视频：随机抽取帧并进行人物检测与裁切

        参数:
            video_path: 视频文件路径
            dest_dir: 输出目录
            num_frames: 要抽取的帧数
            progress_callback: 进度回调函数 (current, total, message)
            cancel_event: 取消事件，用于中断处理

        返回:
            (成功处理的帧数, 总共检测到人物的帧数)
        """
        self._init_model()

        dest_path = Path(dest_dir)
        dest_path.mkdir(parents=True, exist_ok=True)

        # 获取视频信息
        total_frames, fps, width, height = self.get_video_info(video_path)
        print(f"视频信息: {total_frames} 帧, {fps:.1f} FPS, {width}x{height}")

        if num_frames > total_frames:
            print(f"提示: 请求数量({num_frames}) 大于总帧数({total_frames})，将处理所有帧。")
            num_frames = total_frames

        # 随机选择帧索引
        selected_indices = sorted(random.sample(range(total_frames), num_frames))

        # 初始化人脸聚类器
        face_cluster = FaceCluster()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        success_count = 0
        detected_count = 0
        saved_idx = 0

        try:
            for i, frame_idx in enumerate(selected_indices):
                # 检查取消信号
                if cancel_event and cancel_event.is_set():
                    print("用户取消操作")
                    break

                # 跳转到目标帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    msg = f"警告: 无法读取第 {frame_idx} 帧"
                    print(msg)
                    if progress_callback:
                        progress_callback(i + 1, num_frames, msg)
                    continue

                # 处理帧
                processed_img = self.process_frame(frame)

                if processed_img is not None:
                    detected_count += 1
                    # 保存
                    saved_idx += 1
                    new_filename = f"{saved_idx:06d}.png"
                    save_path = dest_path / new_filename
                    processed_img.save(save_path, format='PNG', optimize=True)
                    success_count += 1

                    # 提取人脸特征
                    has_face = face_cluster.extract_feature(processed_img, new_filename)
                    face_status = "已提取人脸特征" if has_face else "未检测到正脸"
                    msg = f"[{i + 1}/{num_frames}] 帧 {frame_idx}: 检测到人物 -> {new_filename} ({face_status})"
                else:
                    msg = f"[{i + 1}/{num_frames}] 帧 {frame_idx}: 未检测到人物，跳过"

                print(msg)
                if progress_callback:
                    progress_callback(i + 1, num_frames, msg)

        finally:
            cap.release()

        # 执行人脸聚类
        try:
            face_cluster.run_clustering_and_move(dest_dir)
        except Exception as e:
            print(f"人脸聚类出错: {e}")

        return success_count, detected_count


def process_video_threaded(
    video_path: str,
    dest_dir: str,
    num_frames: int,
    on_complete: Optional[Callable[[int, int], None]] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    on_error: Optional[Callable[[str], None]] = None
) -> threading.Thread:
    """
    在新线程中处理视频（用于 GUI）

    参数:
        video_path: 视频文件路径
        dest_dir: 输出目录
        num_frames: 要抽取的帧数
        on_complete: 完成回调 (success_count, detected_count)
        on_progress: 进度回调 (current, total, message)
        on_error: 错误回调 (error_message)

    返回:
        Thread 对象
    """
    cancel_event = threading.Event()

    def worker():
        try:
            processor = VideoProcessor()
            success, detected = processor.process_video(
                video_path, dest_dir, num_frames,
                progress_callback=on_progress,
                cancel_event=cancel_event
            )
            if on_complete:
                on_complete(success, detected)
        except Exception as e:
            if on_error:
                on_error(str(e))

    thread = threading.Thread(target=worker, daemon=True)
    return thread, cancel_event