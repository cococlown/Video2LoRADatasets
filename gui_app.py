"""
视频人物数据集生成器 - 图形界面
支持视频处理和图片预览两大功能模块
使用 ttkbootstrap 实现现代化 UI
"""

import os
import sys
import ctypes
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

# ============================================================
# Windows DPI 支持（必须在导入 tkinter 之前）
# ============================================================
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# 导入 ttkbootstrap
try:
    import ttkbootstrap as ttk
    from ttkbootstrap.constants import *
    BOOTSTRAP_AVAILABLE = True
except ImportError:
    import tkinter.ttk as ttk
    BOOTSTRAP_AVAILABLE = False
    print("提示: 安装 ttkbootstrap 可获得更美观的界面: pip install ttkbootstrap")

from video_processor import VideoProcessor, process_video_threaded


# ============================================================
# 工具函数
# ============================================================

def get_output_base_dir() -> Path:
    """获取 output 基础目录"""
    return Path(__file__).parent / "output"


def get_available_output_dirs() -> list:
    """获取所有可用的输出目录列表"""
    output_base = get_output_base_dir()
    if not output_base.exists():
        return []

    dirs = []
    for d in output_base.iterdir():
        if d.is_dir():
            has_images = any(
                f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
                for f in d.rglob('*') if f.is_file()
            )
            if has_images:
                dirs.append(d.name)

    return sorted(dirs, reverse=True)


# ============================================================
# 缩略图网格组件（动态调整版本）
# ============================================================

class ThumbnailGrid(ttk.Frame):
    """可滚动的缩略图网格 - 支持动态调整大小"""

    MIN_THUMB_SIZE = 80
    MAX_THUMB_SIZE = 200
    DEFAULT_THUMB_SIZE = 120
    RESIZE_THRESHOLD = 30  # 重绘阈值，减少频繁重绘

    def __init__(self, master, on_select_callback: Optional[Callable] = None, **kwargs):
        super().__init__(master, **kwargs)
        self.on_select = on_select_callback
        self.image_files = []
        self.thumbnails = {}  # 缓存原始缩略图
        self.card_widgets = {}
        self.selected_index = -1
        self.current_thumb_size = self.DEFAULT_THUMB_SIZE
        self.current_cols = 5
        self._resize_timer = None
        self._last_render_width = 0

        self._setup_ui()

    def _setup_ui(self):
        """构建 UI"""
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(container, highlightthickness=0, bg="#f5f5f5")
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

        self.inner_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.bind("<MouseWheel>", self._on_mousewheel)

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """画布大小变化时，动态调整缩略图"""
        self.canvas.itemconfig(self.canvas_window, width=event.width)
        self._schedule_resize(event.width)

    def _schedule_resize(self, width: int):
        """延迟调整大小（避免频繁重绘）"""
        if self._resize_timer:
            self.after_cancel(self._resize_timer)
        self._resize_timer = self.after(300, lambda: self._recalculate_layout(width))

    def _recalculate_layout(self, available_width: int):
        """根据可用宽度重新计算布局"""
        if not self.image_files or available_width < 100:
            return

        # 只有宽度变化超过阈值才重绘
        if abs(available_width - self._last_render_width) < self.RESIZE_THRESHOLD:
            return

        self._last_render_width = available_width

        # 计算最佳缩略图大小
        target_cols = max(4, min(8, available_width // 150))
        new_thumb_size = max(self.MIN_THUMB_SIZE, min(self.MAX_THUMB_SIZE, (available_width - 20) // target_cols - 10))
        new_cols = max(4, available_width // (new_thumb_size + 10))

        if abs(new_thumb_size - self.current_thumb_size) > self.RESIZE_THRESHOLD or new_cols != self.current_cols:
            self.current_thumb_size = new_thumb_size
            self.current_cols = new_cols
            self._refresh_all_thumbnails()

    def _refresh_all_thumbnails(self):
        """刷新所有缩略图"""
        cell_width = self.current_thumb_size + 10
        cell_height = self.current_thumb_size + 45  # 给文件名预留更多空间

        # 清空现有卡片
        for widget in self.inner_frame.winfo_children():
            widget.destroy()
        self.card_widgets.clear()

        # 重新创建卡片
        for idx, img_path in enumerate(self.image_files):
            self._create_thumbnail_card(idx, cell_width, cell_height)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def load_directory(self, directory: str):
        """加载目录"""
        self.clear()
        dir_path = Path(directory)

        if not dir_path.exists():
            return

        valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}

        for f in sorted(dir_path.rglob('*')):
            if f.is_file() and f.suffix.lower() in valid_exts:
                self.image_files.append(str(f))

        # 初始渲染
        cell_width = self.current_thumb_size + 10
        cell_height = self.current_thumb_size + 45
        self._render_thumbnails_async(cell_width, cell_height)

    def clear(self):
        """清空"""
        for widget in self.inner_frame.winfo_children():
            widget.destroy()
        self.image_files.clear()
        self.thumbnails.clear()
        self.card_widgets.clear()
        self.selected_index = -1
        self._last_render_width = 0

    def _render_thumbnails_async(self, cell_width: int, cell_height: int):
        """异步渲染缩略图"""
        total = len(self.image_files)

        def render_batch(start_idx=0, batch_size=50):
            end_idx = min(start_idx + batch_size, total)

            for idx in range(start_idx, end_idx):
                self._create_thumbnail_card(idx, cell_width, cell_height)

            if end_idx < total:
                self.after(5, lambda: render_batch(end_idx, batch_size))

        render_batch()

    def _create_thumbnail_card(self, idx: int, cell_width: int, cell_height: int):
        """创建单个缩略图卡片"""
        if idx in self.card_widgets:
            return

        img_path = self.image_files[idx]
        thumb_size = self.current_thumb_size

        # 创建卡片
        card = ttk.Frame(self.inner_frame, width=cell_width, height=cell_height)
        card.grid(row=idx // self.current_cols, column=idx % self.current_cols, padx=2, pady=2)
        card.grid_propagate(False)

        # 图片标签
        img_label = ttk.Label(card, text="...", font=("Arial", 8), anchor="center")
        img_label.place(x=5, y=2, width=thumb_size, height=thumb_size)

        # 文件名 - 放在底部，预留足够空间
        name = Path(img_path).name
        if len(name) > 15:
            name = name[:12] + "..."
        name_label = ttk.Label(card, text=name, font=("Arial", 8), anchor="center", wraplength=cell_width - 4)
        name_label.place(x=2, y=thumb_size + 8, width=cell_width - 4, height=25)

        # 缓存
        self.card_widgets[idx] = {'card': card, 'img_label': img_label}

        # 绑定点击
        card.bind("<Button-1>", lambda e, i=idx: self._on_thumbnail_click(i))
        img_label.bind("<Button-1>", lambda e, i=idx: self._on_thumbnail_click(i))

        # 异步加载图片
        self._load_thumbnail_async(idx, img_path, img_label, thumb_size)

    def _load_thumbnail_async(self, idx: int, img_path: str, label: ttk.Label, thumb_size: int):
        """异步加载缩略图"""
        def load():
            try:
                with Image.open(img_path) as img:
                    img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                    thumb = ImageTk.PhotoImage(img)
                    self.thumbnails[img_path] = {'photo': thumb, 'path': img_path}
                    self.after(0, lambda: self._update_thumbnail_label(label, thumb))
            except Exception:
                pass

        threading.Thread(target=load, daemon=True).start()

    def _update_thumbnail_label(self, label: ttk.Label, thumb: ImageTk.PhotoImage):
        """更新缩略图标签"""
        try:
            label.configure(image=thumb, text="")
            label.image = thumb
        except Exception:
            pass

    def _on_thumbnail_click(self, index: int):
        """点击缩略图"""
        if self.selected_index >= 0 and self.selected_index in self.card_widgets:
            old_card = self.card_widgets[self.selected_index]['card']
            old_card.configure(style="TFrame")

        self.selected_index = index

        if index in self.card_widgets:
            new_card = self.card_widgets[index]['card']
            if BOOTSTRAP_AVAILABLE:
                new_card.configure(style="info.TFrame")

        if self.on_select and 0 <= index < len(self.image_files):
            self.on_select(self.image_files[index])

    def get_selected_index(self) -> int:
        return self.selected_index

    def get_image_count(self) -> int:
        return len(self.image_files)

    def remove_image(self, index: int):
        """移除图片并重新渲染"""
        if 0 <= index < len(self.image_files):
            img_path = self.image_files[index]
            del self.image_files[index]

            if img_path in self.thumbnails:
                del self.thumbnails[img_path]

            # 重新渲染整个网格（确保正确补位）
            self._refresh_all_thumbnails()
            self.selected_index = min(index, len(self.image_files) - 1)

            # 自动选中下一个
            if self.image_files and self.selected_index >= 0:
                self._on_thumbnail_click(self.selected_index)


# ============================================================
# 图片预览面板（优化版 - 固定布局）
# ============================================================

class ImagePreviewPanel(ttk.Frame):
    """大图预览面板 - 固定布局防止抖动"""

    def __init__(self, master, on_delete_callback: Optional[Callable] = None,
                 on_navigate_callback: Optional[Callable] = None, **kwargs):
        super().__init__(master, **kwargs)
        self.on_delete = on_delete_callback
        self.on_navigate = on_navigate_callback
        self.current_image_path = None
        self.current_photo = None

        self._setup_ui()

    def _setup_ui(self):
        """构建 UI"""
        # 固定高度的图片显示区域
        self.image_frame = ttk.Frame(self)
        self.image_frame.pack(fill="both", expand=True)

        # 使用 Canvas 显示图片（固定尺寸）
        self.canvas = tk.Canvas(self.image_frame, highlightthickness=0, bg="#2b2b2b")
        self.canvas.pack(fill="both", expand=True)

        # 图片信息
        self.info_label = ttk.Label(self, text="", font=("Arial", 9), anchor="center")
        self.info_label.pack(pady=5)

        # 操作按钮
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", pady=5)

        self.prev_btn = ttk.Button(btn_frame, text="◀ 上一张", command=lambda: self._navigate("prev"), width=12)
        self.prev_btn.pack(side="left", padx=10)

        self.delete_btn = ttk.Button(btn_frame, text="🗑 删除", command=self._delete_image, width=10)
        self.delete_btn.pack(side="left", padx=10, expand=True)

        self.next_btn = ttk.Button(btn_frame, text="下一张 ▶", command=lambda: self._navigate("next"), width=12)
        self.next_btn.pack(side="right", padx=10)

    def load_image(self, image_path: Optional[str]):
        """加载并显示图片"""
        # 清空画布
        self.canvas.delete("all")

        if not image_path or not os.path.exists(image_path):
            self.canvas.create_text(
                self.canvas.winfo_width() // 2 or 200,
                self.canvas.winfo_height() // 2 or 150,
                text="请选择一张图片", fill="white", font=("Arial", 12)
            )
            self.current_image_path = None
            self.info_label.configure(text="")
            return

        self.current_image_path = image_path

        # 在线程中加载图片
        def load():
            try:
                with Image.open(image_path) as img:
                    img_w, img_h = img.size
                    file_size = os.path.getsize(image_path) / 1024

                    # 更新信息
                    file_name = Path(image_path).name
                    self.after(0, lambda: self.info_label.configure(
                        text=f"📄 {file_name}  |  📐 {img_w}×{img_h}  |  💾 {file_size:.1f} KB"
                    ))

                    # 计算缩放
                    self.canvas.update_idletasks()
                    max_w = max(self.canvas.winfo_width(), 200)
                    max_h = max(self.canvas.winfo_height(), 200)

                    scale = min(max_w / img_w, max_h / img_h, 1.0)

                    if scale < 1.0:
                        new_w = int(img_w * scale)
                        new_h = int(img_h * scale)
                        display_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    else:
                        display_img = img.copy()
                        new_w, new_h = img_w, img_h

                    # 转换并显示
                    photo = ImageTk.PhotoImage(display_img)
                    self.current_photo = photo

                    # 在主线程更新画布
                    self.after(0, lambda: self._display_photo(photo, new_w, new_h, max_w, max_h))

            except Exception as e:
                self.after(0, lambda: self._show_error(str(e)))

        threading.Thread(target=load, daemon=True).start()

    def _display_photo(self, photo: ImageTk.PhotoImage, img_w: int, img_h: int, canvas_w: int, canvas_h: int):
        """在画布上显示图片"""
        self.canvas.delete("all")

        # 居中显示
        x = canvas_w // 2
        y = canvas_h // 2

        self.canvas.create_image(x, y, image=photo, anchor="center")
        self.current_photo = photo

    def _show_error(self, error: str):
        """显示错误"""
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas.winfo_width() // 2 or 200,
            self.canvas.winfo_height() // 2 or 150,
            text=f"加载失败: {error}", fill="red", font=("Arial", 10)
        )

    def _navigate(self, direction: str):
        if self.on_navigate:
            self.on_navigate(direction)

    def _delete_image(self):
        """删除当前图片（无确认）"""
        if not self.current_image_path:
            return

        try:
            file_name = Path(self.current_image_path).name
            os.remove(self.current_image_path)
            # 删除成功后直接回调，不弹窗
            if self.on_delete:
                self.on_delete(self.current_image_path)
        except Exception as e:
            messagebox.showerror("删除失败", f"删除文件时出错:\n{e}")

    def update_navigation(self, has_prev: bool, has_next: bool):
        """更新导航按钮状态"""
        self.prev_btn.configure(state="normal" if has_prev else "disabled")
        self.next_btn.configure(state="normal" if has_next else "disabled")


# ============================================================
# 预览标签页
# ============================================================

class PreviewTab(ttk.Frame):
    """图片预览标签页"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.current_directory = None

        self._setup_ui()
        self._refresh_dir_list()

    def _setup_ui(self):
        """构建 UI"""
        # 顶部工具栏
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=10, pady=10)

        ttk.Label(toolbar, text="快速选择:", font=("Arial", 10, "bold")).pack(side="left")

        self.dir_combo = ttk.Combobox(toolbar, width=35, state="readonly")
        self.dir_combo.pack(side="left", padx=10)
        self.dir_combo.bind("<<ComboboxSelected>>", self._on_combo_select)

        ttk.Button(toolbar, text="🔄 刷新", command=self._refresh_dir_list, width=10).pack(side="left", padx=5)
        ttk.Button(toolbar, text="📂 浏览", command=self._select_directory, width=8).pack(side="left", padx=5)
        ttk.Button(toolbar, text="📁 打开目录", command=self._open_folder, width=10).pack(side="left", padx=5)

        # 状态标签
        self.status_label = ttk.Label(self, text="请选择包含图片的目录", font=("Arial", 10))
        self.status_label.pack(pady=5)

        # 主内容区域 - 使用 Frame 固定布局
        content_frame = ttk.Frame(self)
        content_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # 左侧：缩略图网格（固定宽度）
        left_container = ttk.Frame(content_frame, width=650)
        left_container.pack(side="left", fill="both", expand=True)
        left_container.pack_propagate(False)

        left_label = ttk.Label(left_container, text="缩略图列表", font=("Arial", 10, "bold"))
        left_label.pack(anchor="w", padx=5, pady=5)

        self.thumbnail_grid = ThumbnailGrid(left_container, on_select_callback=self._on_thumbnail_select)
        self.thumbnail_grid.pack(fill="both", expand=True, padx=5, pady=5)

        # 右侧：大图预览
        right_container = ttk.Frame(content_frame, width=450)
        right_container.pack(side="right", fill="both", expand=True, padx=(10, 0))

        right_label = ttk.Label(right_container, text="图片预览", font=("Arial", 10, "bold"))
        right_label.pack(anchor="w", padx=5, pady=5)

        self.preview_panel = ImagePreviewPanel(
            right_container,
            on_delete_callback=self._on_image_delete,
            on_navigate_callback=self._on_navigate
        )
        self.preview_panel.pack(fill="both", expand=True, padx=5, pady=5)

    def _refresh_dir_list(self):
        """刷新目录列表"""
        dirs = get_available_output_dirs()
        self.dir_combo['values'] = dirs

        if dirs:
            self.dir_combo.set(dirs[0])
            self._on_combo_select(None)
        else:
            self.dir_combo.set("")
            self.status_label.configure(text="output 目录下暂无图片文件夹")

    def _on_combo_select(self, event):
        """下拉框选择"""
        selected = self.dir_combo.get()
        if selected:
            output_base = get_output_base_dir()
            full_path = str(output_base / selected)
            self.load_directory(full_path)

    def _select_directory(self):
        """选择目录"""
        path = filedialog.askdirectory(title="选择图片目录")
        if path:
            self.load_directory(path)

    def load_directory(self, directory: str):
        """加载目录"""
        self.current_directory = directory
        self.thumbnail_grid.load_directory(directory)
        count = self.thumbnail_grid.get_image_count()
        self.status_label.configure(text=f"共加载 {count} 张图片")

        if count > 0:
            self.after(100, lambda: self.thumbnail_grid._on_thumbnail_click(0))

    def _open_folder(self):
        """打开文件夹"""
        if self.current_directory and os.path.exists(self.current_directory):
            os.startfile(self.current_directory)

    def _on_thumbnail_select(self, image_path: str):
        """缩略图选中"""
        self.preview_panel.load_image(image_path)
        index = self.thumbnail_grid.get_selected_index()
        count = self.thumbnail_grid.get_image_count()
        self.preview_panel.update_navigation(index > 0, index < count - 1)
        self.status_label.configure(text=f"共 {count} 张图片 | 当前第 {index + 1} 张")

    def _on_image_delete(self, image_path: str):
        """图片删除"""
        index = self.thumbnail_grid.get_selected_index()
        self.thumbnail_grid.remove_image(index)
        count = self.thumbnail_grid.get_image_count()
        self.status_label.configure(text=f"共 {count} 张图片")

        if count > 0:
            new_index = min(index, count - 1)
            self.thumbnail_grid._on_thumbnail_click(new_index)
        else:
            self.preview_panel.load_image(None)
            self.status_label.configure(text="无图片")

    def _on_navigate(self, direction: str):
        """导航"""
        current = self.thumbnail_grid.get_selected_index()
        count = self.thumbnail_grid.get_image_count()

        if direction == "prev" and current > 0:
            self.thumbnail_grid._on_thumbnail_click(current - 1)
        elif direction == "next" and current < count - 1:
            self.thumbnail_grid._on_thumbnail_click(current + 1)


# ============================================================
# 处理标签页
# ============================================================

class ProcessingTab(ttk.Frame):
    """视频处理标签页"""

    def __init__(self, master, on_complete_callback: Optional[Callable] = None, **kwargs):
        super().__init__(master, **kwargs)
        self.on_complete = on_complete_callback
        self.video_path = ""
        self.total_frames = 0
        self.fps = 0.0
        self.video_width = 0
        self.video_height = 0
        self.cancel_event = None
        self.processing = False
        self.output_dir = ""

        self._setup_ui()

    def _setup_ui(self):
        """构建 UI"""
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # 视频文件
        video_frame = ttk.Frame(main_frame)
        video_frame.pack(fill="x", pady=5)

        ttk.Label(video_frame, text="视频文件:", width=12).pack(side="left")
        self.video_entry = ttk.Entry(video_frame, width=60)
        self.video_entry.pack(side="left", padx=5, fill="x", expand=True)
        ttk.Button(video_frame, text="浏览...", command=self._browse_video, width=10).pack(side="right")

        # 角色名称
        role_frame = ttk.Frame(main_frame)
        role_frame.pack(fill="x", pady=5)

        ttk.Label(role_frame, text="角色名称:", width=12).pack(side="left")
        self.role_entry = ttk.Entry(role_frame, width=30)
        self.role_entry.pack(side="left", padx=5)
        ttk.Label(role_frame, text="(可选，用于命名输出目录)", foreground="gray").pack(side="left", padx=5)

        # 抽取帧数
        frames_frame = ttk.Frame(main_frame)
        frames_frame.pack(fill="x", pady=5)

        ttk.Label(frames_frame, text="抽取帧数:", width=12).pack(side="left")
        self.frames_entry = ttk.Entry(frames_frame, width=15)
        self.frames_entry.pack(side="left", padx=5)
        self.frames_entry.insert(0, "100")

        self.video_info_label = ttk.Label(frames_frame, text="", foreground="gray")
        self.video_info_label.pack(side="left", padx=15)

        # 输出目录
        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill="x", pady=5)

        ttk.Label(output_frame, text="输出目录:", width=12).pack(side="left")
        self.output_entry = ttk.Entry(output_frame, width=60, state="readonly")
        self.output_entry.pack(side="left", padx=5, fill="x", expand=True)

        # 进度条
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill="x", pady=15)

        self.progress_var = ttk.Progressbar(progress_frame, mode="determinate", length=500)
        self.progress_var.pack(fill="x", pady=5)

        self.progress_label = ttk.Label(progress_frame, text="等待开始...", font=("Arial", 10))
        self.progress_label.pack()

        # 操作按钮
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x", pady=10)

        self.start_btn = ttk.Button(btn_frame, text="▶ 开始处理", command=self._start_processing, width=15)
        self.start_btn.pack(side="left", padx=10)

        self.cancel_btn = ttk.Button(btn_frame, text="⏹ 取消", command=self._cancel_processing, width=15, state="disabled")
        self.cancel_btn.pack(side="left", padx=10)

        # 日志区域
        log_label = ttk.Label(main_frame, text="处理日志:", font=("Arial", 10, "bold"))
        log_label.pack(anchor="w", pady=(10, 5))

        log_container = ttk.Frame(main_frame)
        log_container.pack(fill="both", expand=True)

        self.log_text = tk.Text(log_container, height=10, state="disabled", font=("Consolas", 9),
                                bg="#1e1e1e", fg="#d4d4d4", insertbackground="white")
        log_scroll = ttk.Scrollbar(log_container, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)

        log_scroll.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)

    def _get_output_dir(self) -> str:
        """生成输出目录名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        role_name = self.role_entry.get().strip()

        output_base = Path(__file__).parent / "output"

        if role_name:
            safe_name = "".join(c for c in role_name if c.isalnum() or c in ('_', '-', ' '))
            safe_name = safe_name.strip()
            dir_name = f"{timestamp}_{safe_name}"
        else:
            dir_name = timestamp

        return str(output_base / dir_name)

    def _browse_video(self):
        """选择视频文件"""
        filetypes = [
            ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
            ("所有文件", "*.*")
        ]
        path = filedialog.askopenfilename(title="选择视频文件", filetypes=filetypes)
        if path:
            self.video_path = path
            self.video_entry.delete(0, "end")
            self.video_entry.insert(0, path)

            self.output_dir = self._get_output_dir()
            self.output_entry.configure(state="normal")
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, self.output_dir)
            self.output_entry.configure(state="readonly")

            self._load_video_info()

    def _load_video_info(self):
        """加载视频信息"""
        try:
            processor = VideoProcessor()
            self.total_frames, self.fps, self.video_width, self.video_height = processor.get_video_info(self.video_path)
            info_text = f"共 {self.total_frames} 帧 | {self.fps:.1f} FPS | {self.video_width}×{self.video_height}"
            self.video_info_label.configure(text=info_text)
        except Exception as e:
            self.video_info_label.configure(text=f"无法读取视频: {e}")
            self.total_frames = 0

    def _log(self, message: str):
        """添加日志"""
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _start_processing(self):
        """开始处理"""
        video_path = self.video_entry.get().strip()

        if not video_path:
            messagebox.showerror("错误", "请选择视频文件")
            return

        if not os.path.isfile(video_path):
            messagebox.showerror("错误", "视频文件不存在")
            return

        try:
            num_frames = int(self.frames_entry.get().strip())
            if num_frames <= 0:
                raise ValueError("帧数必须大于 0")
        except ValueError as e:
            messagebox.showerror("错误", f"无效的帧数: {e}")
            return

        self.output_dir = self._get_output_dir()
        self.output_entry.configure(state="normal")
        self.output_entry.delete(0, "end")
        self.output_entry.insert(0, self.output_dir)
        self.output_entry.configure(state="readonly")

        if self.total_frames == 0:
            try:
                self._load_video_info()
            except Exception as e:
                messagebox.showerror("错误", f"无法读取视频: {e}")
                return

        if num_frames > self.total_frames:
            if not messagebox.askyesno("提示", f"请求数量({num_frames})大于总帧数({self.total_frames})，将处理所有帧。\n是否继续？"):
                return

        self.processing = True
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress_var["value"] = 0
        self.progress_label.configure(text="正在初始化...")

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

        self._log(f"🎬 开始处理视频: {Path(video_path).name}")
        self._log(f"📁 输出目录: {self.output_dir}")
        self._log(f"📊 计划抽取帧数: {num_frames}")
        self._log("─" * 50)

        thread, cancel_event = process_video_threaded(
            video_path=video_path,
            dest_dir=self.output_dir,
            num_frames=num_frames,
            on_progress=self._on_progress,
            on_complete=self._on_complete,
            on_error=self._on_error
        )
        self.cancel_event = cancel_event
        thread.start()

    def _cancel_processing(self):
        if self.cancel_event:
            self.cancel_event.set()
            self._log("⏹ 正在取消...")
            self.cancel_btn.configure(state="disabled")

    def _on_progress(self, current: int, total: int, message: str):
        def update():
            progress_percent = (current / total) * 100 if total > 0 else 0
            self.progress_var["value"] = progress_percent
            self.progress_label.configure(text=f"{current}/{total} ({progress_percent:.1f}%)")
            self._log(message)
        self.after(0, update)

    def _on_complete(self, success_count: int, detected_count: int):
        def update():
            self.processing = False
            self.start_btn.configure(state="normal")
            self.cancel_btn.configure(state="disabled")
            self.progress_var["value"] = 100
            self.progress_label.configure(text="✅ 处理完成!")
            self._log("─" * 50)
            self._log(f"✅ 处理完成! 成功保存 {success_count} 张图片，检测到人物 {detected_count} 帧。")

            result = messagebox.askyesno("完成",
                f"处理完成!\n\n成功保存: {success_count} 张图片\n检测到人物: {detected_count} 帧\n\n是否打开预览？")

            if result and self.on_complete:
                self.on_complete(self.output_dir)

        self.after(0, update)

    def _on_error(self, error_message: str):
        def update():
            self.processing = False
            self.start_btn.configure(state="normal")
            self.cancel_btn.configure(state="disabled")
            self._log(f"❌ 错误: {error_message}")
            messagebox.showerror("错误", f"处理失败:\n{error_message}")
        self.after(0, update)


# ============================================================
# 主窗口
# ============================================================

class VideoDatasetApp:
    """视频人物数据集生成器主窗口"""

    def __init__(self, root):
        self.root = root
        self.root.title("视频人物数据集生成器")
        self.root.geometry("1100x750")
        self.root.minsize(1100, 700)

        self._setup_ui()
        self._bind_keys()

    def _setup_ui(self):
        """构建 UI"""
        if BOOTSTRAP_AVAILABLE:
            self.notebook = ttk.Notebook(self.root, bootstyle="primary")
        else:
            self.notebook = ttk.Notebook(self.root)

        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.processing_tab = ProcessingTab(self.notebook, on_complete_callback=self._on_processing_complete)
        self.notebook.add(self.processing_tab, text="  🎬 视频处理  ")

        self.preview_tab = PreviewTab(self.notebook)
        self.notebook.add(self.preview_tab, text="  🖼 图片预览  ")

    def _bind_keys(self):
        """绑定快捷键"""
        self.root.bind("<Left>", lambda e: self._navigate_preview("prev"))
        self.root.bind("<Right>", lambda e: self._navigate_preview("next"))
        self.root.bind("<Delete>", lambda e: self._delete_current_image())

    def _navigate_preview(self, direction: str):
        try:
            if self.notebook.index(self.notebook.select()) == 1:
                self.preview_tab._on_navigate(direction)
        except Exception:
            pass

    def _delete_current_image(self):
        try:
            if self.notebook.index(self.notebook.select()) == 1:
                self.preview_tab.preview_panel._delete_image()
        except Exception:
            pass

    def _on_processing_complete(self, output_dir: str):
        self.preview_tab._refresh_dir_list()
        self.preview_tab.load_directory(output_dir)
        self.notebook.select(1)


def main():
    """主函数"""
    if BOOTSTRAP_AVAILABLE:
        root = ttk.Window(themename="cosmo")
    else:
        root = tk.Tk()

    app = VideoDatasetApp(root)

    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    main()