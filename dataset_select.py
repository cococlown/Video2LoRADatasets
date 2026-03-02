import os
import random
import sys
from pathlib import Path
from PIL import Image
import torch
import shutil

# 尝试导入 YOLO，如果不存在则在 main 中捕获
try:
    from ultralytics import YOLO
    import numpy as np
except ImportError:
    pass

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
    # 保证人物周围有留白，不贴边
    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio
    
    # 计算初始裁切框的中心点
    center_x = x1 + box_w / 2
    center_y = y1 + box_h / 2

    # 应用 Padding 后的初始宽高
    current_w = box_w + (pad_x * 2)
    current_h = box_h + (pad_y * 2)

    # --- 2. 最小边长保障 (Ensure Min Side Length) ---
    # 逻辑：如果当前长宽比小于1：2，就强制扩展到保留短边为长边的一半（前提是原图有这么大）
    
    long_side = max(current_w, current_h)

    target_w = max(current_w, min(long_side/2, img_w))
    target_h = max(current_h, min(long_side/2, img_h))

    # --- 3. 坐标计算与平移 (Shift & Clamp) ---
    # 根据新的宽高和中心点计算坐标
    final_x1 = center_x - target_w / 2
    final_y1 = center_y - target_h / 2
    final_x2 = center_x + target_w / 2
    final_y2 = center_y + target_h / 2

    # 检查边界并平移 (Shift)
    # 核心逻辑：如果扩展后左边超出了，往右挪；右边超出了，往左挪
    # 这样可以保证在满足尺寸要求的同时，尽可能保留人物在框内
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

    # 最后一道防线：硬性截断 (Clamp)
    # 防止因平移计算误差导致微量越界
    final_x1 = max(0, final_x1)
    final_y1 = max(0, final_y1)
    final_x2 = min(img_w, final_x2)
    final_y2 = min(img_h, final_y2)

    return int(final_x1), int(final_y1), int(final_x2), int(final_y2)


class FaceCluster:
    def __init__(self):
        # 延迟导入第三方依赖，若缺失则禁用人脸聚类功能
        try:
            import numpy as np
            import cv2
            from sklearn.cluster import DBSCAN
            from insightface.app import FaceAnalysis
        except Exception as e:
            print(f"[FaceCluster] 无法加载依赖，禁用人脸聚类: {e}")
            self.enabled = False
            return

        print("正在加载 InsightFace (buffalo_l)...")
        try:
            # 创建并准备模型（在 GPU 上）
            self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.enabled = True
        except Exception as e:
            print(f"[FaceCluster] 初始化 FaceAnalysis 失败，禁用人脸聚类: {e}")
            self.enabled = False
            return

        self.np = np
        self.cv2 = cv2
        self.DBSCAN = DBSCAN
        # 存储特征和对应的文件名
        self.embeddings = []
        self.filenames = []

    def extract_feature(self, img_pil, filename):
        """
        从 PIL 图片中提取人脸特征。返回 True 表示成功提取并保存特征。
        """
        if not getattr(self, 'enabled', False):
            return False

        # PIL 转 CV2 (RGB -> BGR)
        img_cv2 = self.cv2.cvtColor(self.np.array(img_pil), self.cv2.COLOR_RGB2BGR)

        # 检测人脸
        faces = self.app.get(img_cv2)

        if len(faces) == 0:
            return False

        # 选择最大人脸
        best_face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))

        embedding = best_face.embedding
        embedding = embedding / self.np.linalg.norm(embedding)

        self.embeddings.append(embedding)
        self.filenames.append(filename)
        return True

    def run_clustering_and_move(self, dest_dir, eps=0.95, min_samples=3):
        if not getattr(self, 'enabled', False):
            print("[FaceCluster] 功能被禁用，跳过聚类。")
            return

        if not self.embeddings:
            print("没有提取到任何人脸特征。")
            return

        print(f"正在对 {len(self.embeddings)} 张人脸进行聚类分析...")
        X = self.np.array(self.embeddings)

        clt = self.DBSCAN(metric="euclidean", n_jobs=-1, eps=eps, min_samples=min_samples)
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
                shutil.move(str(src_file), str(target_folder / file_name))
            except Exception as e:
                print(f"移动文件失败: {e}")

def process_images(source_dir, dest_dir, num_images, recursive=True):
    """
    核心处理逻辑
    """
    # --- 1. 初始化模型与 GPU 设置 ---
    model_name = 'yolov8x.pt' 
    print(f"正在加载 {model_name} 模型...")
    
    model = YOLO(model_name)

    # 强制将模型移动到 GPU
    if torch.cuda.is_available():
        device = 'cuda'
        model.to(device)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU 加速已开启: {gpu_name}")
    else:
        print("⚠️ 警告: 未检测到 GPU，正在使用 CPU (速度较慢)。")

    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    # 创建目标目录
    dest_path.mkdir(parents=True, exist_ok=True)
    # 初始化人脸聚类器（若依赖缺失则为禁用状态）
    face_cluster = FaceCluster()

    # 2. 扫描图片
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    all_files = []
    if recursive:
        for f in source_path.rglob('*'):
            if f.is_file() and f.suffix.lower() in valid_exts:
                all_files.append(f)
    else:
        all_files = [f for f in source_path.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]

    if not all_files:
        raise ValueError("源目录中未找到支持的图片格式")

    if num_images > len(all_files):
        print(f"提示: 请求数量({num_images}) 大于图片总数({len(all_files)})，将处理所有图片。")
        num_images = len(all_files)

    # 3. 随机抽取
    selected_files = random.sample(all_files, num_images)
    print(f"已选中 {len(selected_files)} 张图片，开始处理...")

    success_count = 0

    for idx, file_path in enumerate(selected_files):
        try:
            # 打开图片
            with Image.open(file_path) as img:
                # 统一转换为 RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_w, img_h = img.size

                # --- A. 人物检测 ---
                results = model.predict(img, classes=[0], conf=0.5, verbose=False)
                
                crop_box = None
                
                if results and len(results[0].boxes) > 0:
                    # 获取所有检测到的框
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    
                    # 策略：选择面积最大的人
                    best_box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
                    
                    # --- B. 计算智能裁切框 (修改点) ---
                    # 这里设定 min_side_len=800，确保裁切出来的区域足够大
                    crop_box = get_smart_crop_box(img_w, img_h, best_box, padding_ratio=0.15, min_side_len=800)
                    
                    # 执行裁切
                    img = img.crop(crop_box)
                else:
                    print(f"[{idx+1}/{num_images}] 警告: {file_path.name} 未检测到人物，保留原图。")

                # --- C. 尺寸调整 (Resize) ---
                # 此时 img 已经是裁切好的了（且最短边尽量保证了>800）
                # 下面执行“最长边不超过1280”的逻辑
                cur_w, cur_h = img.size
                max_side = max(cur_w, cur_h)
                
                if max_side > 1280:
                    scale = 1280 / max_side
                    new_w = int(cur_w * scale)
                    new_h = int(cur_h * scale)
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                # --- D. 保存结果 ---
                new_filename = f"{idx + 1:06d}.png"
                save_path = dest_path / new_filename
                
                img.save(save_path, format='PNG', optimize=True)
                success_count += 1
                # 保存后尝试提取人脸特征（如果 FaceCluster 可用）
                has_face = face_cluster.extract_feature(img, new_filename)
                if has_face:
                    print(f"[{idx+1}/{num_images}] 处理完成: {new_filename} (尺寸: {img.size})  -> 已提取人脸特征")
                else:
                    print(f"[{idx+1}/{num_images}] 处理完成: {new_filename} (尺寸: {img.size})  -> 未检测到正脸 (可能是背影)")

        except Exception as e:
            print(f"[{idx+1}/{num_images}] 错误: 处理 {file_path.name} 失败 - {e}")

    # 循环结束后执行人脸聚类并移动文件（如果可用）
    try:
        face_cluster.run_clustering_and_move(dest_dir)
    except Exception as e:
        print(f"人脸聚类或移动过程中发生错误: {e}")

    return success_count

def main():
    # 设置参数
    print("--- 智能人物裁切脚本 (任意比例版) ---")
    source_directory = input("请输入源图片目录路径: ").strip()
    destination_directory = input("请输入目标目录路径: ").strip()
    
    while True:
        try:
            num_input = input("请输入要选择的图片数量: ").strip()
            num_to_select = int(num_input)
            if num_to_select <= 0:
                print("请输入大于 0 的数字。")
                continue
            break
        except ValueError:
            print("输入无效，请输入一个整数。")

    if not os.path.isdir(source_directory):
        print(f"错误: 源目录 '{source_directory}' 不存在")
        return
    
    try:
        process_images(source_dir=source_directory, dest_dir=destination_directory, num_images=num_to_select)
        print(f"\n全部任务结束! 图片已保存至 '{destination_directory}'")
    except ValueError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    try:
        import ultralytics
        import PIL
        import numpy
        import cv2 
    except ImportError as e:
        print("缺少必要的依赖库，请安装:")
        print("pip install ultralytics pillow numpy opencv-python-headless")
        sys.exit(1)
    
    main()