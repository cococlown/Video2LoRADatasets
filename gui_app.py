"""
视频人物数据集生成器 - NiceGUI 版本
支持视频处理和图片预览两大功能模块
使用 NiceGUI 实现现代化 Web UI 桌面应用
"""

import os
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from nicegui import ui, app
from nicegui import run
from PIL import Image

from video_processor import VideoProcessor


# ============================================================
# 全局状态管理
# ============================================================

class AppState:
    """应用状态管理"""
    def __init__(self):
        self.selected_index = -1
        self.image_list: List[str] = []
        self.current_directory = ""
        self.processing = False
        self.cancel_event: Optional[threading.Event] = None

    def reset(self):
        self.selected_index = -1
        self.image_list = []
        self.current_directory = ""


state = AppState()


# ============================================================
# 工具函数
# ============================================================

def get_output_base_dir() -> Path:
    """获取 output 基础目录"""
    return Path(__file__).parent / "output"


def get_available_output_dirs() -> List[str]:
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


def scan_images_in_directory(directory: str) -> List[str]:
    """扫描目录中的所有图片"""
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    images = []

    for f in sorted(dir_path.rglob('*')):
        if f.is_file() and f.suffix.lower() in valid_exts:
            # 返回相对路径，用于 URL
            rel_path = f.relative_to(dir_path)
            images.append(str(rel_path))

    return images


def get_image_info(image_path: str) -> tuple:
    """获取图片尺寸和文件大小"""
    try:
        with Image.open(image_path) as img:
            w, h = img.size
        file_size = os.path.getsize(image_path) / 1024
        return w, h, file_size
    except Exception:
        return 0, 0, 0


# ============================================================
# 文件选择对话框（使用 tkinter）
# ============================================================

async def select_video_file() -> Optional[str]:
    """选择视频文件"""
    import tkinter.filedialog as filedialog

    def _select():
        root = __import__('tkinter').Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm"),
                ("所有文件", "*.*")
            ]
        )
        root.destroy()
        return path

    return await run.io_bound(_select)


async def select_directory() -> Optional[str]:
    """选择目录"""
    import tkinter.filedialog as filedialog

    def _select():
        root = __import__('tkinter').Tk()
        root.withdraw()
        path = filedialog.askdirectory(title="选择图片目录")
        root.destroy()
        return path

    return await run.io_bound(_select)


# ============================================================
# 自定义样式
# ============================================================

CUSTOM_CSS = '''
/* 渐变色 Header */
.gradient-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* 卡片悬停效果 */
.card-hover {
    transition: all 0.3s ease;
}
.card-hover:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

/* 缩略图选中效果 */
.thumbnail-selected {
    ring: 2px;
    ring-color: #3b82f6;
    transform: scale(1.02);
}

/* 按钮渐变 */
.btn-gradient {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    color: white;
}

/* 进度条动画 */
.progress-animated {
    animation: progress-pulse 2s infinite;
}

@keyframes progress-pulse {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
}

/* 日志区域样式 */
.log-container {
    background: #1a1a2e;
    border-radius: 12px;
    font-family: 'Consolas', 'Monaco', monospace;
}

/* 图片预览区域 */
.preview-area {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
}
'''


# ============================================================
# 视频处理页
# ============================================================

class ProcessingPage:
    """视频处理页面"""

    def __init__(self):
        self.video_path = ""
        self.total_frames = 0
        self.fps = 0.0
        self.video_width = 0
        self.video_height = 0
        self.output_dir = ""
        self.processor = VideoProcessor()

        # UI 组件引用
        self.video_input = None
        self.role_input = None
        self.frames_input = None
        self.video_info_label = None
        self.output_label = None
        self.progress_bar = None
        self.progress_label = None
        self.log_component = None
        self.start_btn = None
        self.cancel_btn = None

    def build(self):
        """构建 UI"""
        with ui.column().classes('w-full gap-4'):
            # 标题卡片
            with ui.card().classes('w-full p-0 overflow-hidden'):
                with ui.element('div').classes('gradient-header p-4'):
                    ui.label('🎬 视频处理').classes('text-2xl font-bold text-white')
                    ui.label('从视频中智能抽取人物图片').classes('text-white/70 text-sm')

            # 表单区域
            with ui.card().classes('w-full p-6 card-hover'):
                with ui.column().classes('w-full gap-4'):
                    # 视频文件选择
                    with ui.row().classes('w-full items-center gap-3'):
                        with ui.icon('videocam', color='primary').classes('text-2xl'):
                            pass
                        ui.label('视频文件').classes('w-20 font-semibold text-gray-700')
                        self.video_input = ui.input(
                            placeholder='请选择视频文件或输入路径...',
                            on_change=self._on_video_change
                        ).classes('flex-1').props('outlined dense clearable')
                        ui.button('浏览', icon='folder_open', on_click=self._browse_video).props('outline')

                    # 角色名称
                    with ui.row().classes('w-full items-center gap-3'):
                        with ui.icon('person', color='secondary').classes('text-2xl'):
                            pass
                        ui.label('角色名称').classes('w-20 font-semibold text-gray-700')
                        self.role_input = ui.input(placeholder='可选，用于命名输出目录').classes('w-80').props('outlined dense')
                        ui.badge('可选', color='grey').classes('text-xs')

                    # 抽取帧数
                    with ui.row().classes('w-full items-center gap-3'):
                        with ui.icon('images', color='orange').classes('text-2xl'):
                            pass
                        ui.label('抽取帧数').classes('w-20 font-semibold text-gray-700')
                        self.frames_input = ui.number(value=100, min=1, max=100000).classes('w-32').props('outlined dense')
                        self.video_info_label = ui.label('').classes('text-gray-500 text-sm ml-4 flex-1')

                    # 输出目录
                    with ui.row().classes('w-full items-center gap-3'):
                        with ui.icon('folder', color='green').classes('text-2xl'):
                            pass
                        ui.label('输出目录').classes('w-20 font-semibold text-gray-700')
                        self.output_label = ui.label('自动生成').classes('text-gray-600 bg-gray-100 px-3 py-1 rounded')

            # 进度区域
            with ui.card().classes('w-full p-6 card-hover'):
                with ui.column().classes('w-full gap-3'):
                    with ui.row().classes('w-full items-center justify-between'):
                        ui.label('📊 处理进度').classes('font-semibold text-gray-700')
                        self.progress_label = ui.label('等待开始...').classes('text-gray-500')

                    self.progress_bar = ui.linear_progress(value=0).classes('w-full h-2')

                    # 操作按钮
                    with ui.row().classes('w-full justify-center gap-4 mt-2'):
                        self.start_btn = ui.button(
                            '🚀 开始处理',
                            on_click=self._start_processing
                        ).props('size=lg color=primary icon=play_arrow')

                        self.cancel_btn = ui.button(
                            '⏹ 取消',
                            on_click=self._cancel_processing
                        ).props('size=lg color=negative icon=stop outline disable')
                        self.cancel_btn.disable()

            # 日志区域
            with ui.card().classes('w-full p-6 card-hover'):
                ui.label('📋 处理日志').classes('font-semibold text-gray-700 mb-3')
                self.log_component = ui.log(max_lines=200).classes(
                    'w-full h-56 log-container text-green-400'
                )

    async def _browse_video(self):
        """浏览选择视频文件"""
        path = await select_video_file()
        if path:
            self.video_input.value = path
            await self._on_video_change()

    async def _on_video_change(self):
        """视频路径变化时更新信息"""
        video_path = self.video_input.value
        if not video_path or not os.path.exists(video_path):
            self.video_info_label.text = ''
            return

        try:
            def get_info():
                return self.processor.get_video_info(video_path)

            total, fps, w, h = await run.io_bound(get_info)
            self.total_frames = total
            self.fps = fps
            self.video_width = w
            self.video_height = h

            duration = total / fps if fps > 0 else 0
            self.video_info_label.text = f'📹 {total:,} 帧 | ⏱ {fps:.1f} FPS | 📐 {w}×{h} | ⏳ {duration:.1f}s'

            if self.frames_input.value > total:
                self.frames_input.value = total

        except Exception as e:
            self.video_info_label.text = f'❌ 读取失败: {e}'

    def _generate_output_dir(self) -> str:
        """生成输出目录路径"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        role = self.role_input.value.strip()
        if role:
            dir_name = f"{timestamp}_{role}"
        else:
            dir_name = timestamp

        output_base = get_output_base_dir()
        return str(output_base / dir_name)

    async def _start_processing(self):
        """开始处理视频"""
        video_path = self.video_input.value
        if not video_path or not os.path.exists(video_path):
            ui.notify('请选择有效的视频文件', type='warning', position='top')
            return

        num_frames = int(self.frames_input.value)
        self.output_dir = self._generate_output_dir()

        self.start_btn.disable()
        self.cancel_btn.enable()
        self.progress_bar.value = 0
        self.progress_label.text = '正在初始化...'
        self.output_label.text = self.output_dir
        state.processing = True

        state.cancel_event = threading.Event()

        self.log_component.clear()
        self.log_component.push(f'🎬 开始处理: {video_path}')
        self.log_component.push(f'📁 输出目录: {self.output_dir}')
        self.log_component.push(f'📊 抽取帧数: {num_frames}')
        self.log_component.push('─' * 50)

        def process_thread():
            try:
                success, detected = self.processor.process_video(
                    video_path,
                    self.output_dir,
                    num_frames,
                    progress_callback=self._progress_callback,
                    cancel_event=state.cancel_event
                )
                return success, detected, None
            except Exception as e:
                return 0, 0, str(e)

        result = await run.io_bound(process_thread)
        success, detected, error = result

        state.processing = False
        self.start_btn.enable()
        self.cancel_btn.disable()

        if error:
            self.log_component.push(f'❌ 处理出错: {error}')
            ui.notify(f'处理失败: {error}', type='negative', position='top')
        elif state.cancel_event and state.cancel_event.is_set():
            self.log_component.push('⏹ 处理已取消')
            ui.notify('处理已取消', type='warning', position='top')
        else:
            self.log_component.push('─' * 50)
            self.log_component.push(f'✅ 处理完成!')
            self.log_component.push(f'📈 成功: {success} 张 | 👤 检测到人物: {detected} 张')
            self.progress_label.text = f'✅ 完成! 共生成 {success} 张图片'
            ui.notify(f'处理完成! 共生成 {success} 张图片', type='positive', position='top')

    def _progress_callback(self, current: int, total: int, message: str):
        """进度回调"""
        progress = current / total if total > 0 else 0
        ui.run_javascript(f'''
            document.querySelector('.q-linear-progress')?.__vue__?.setProgress({progress});
        ''')
        self.progress_label.text = f'{current}/{total} - {message}'
        self.log_component.push(message)

    async def _cancel_processing(self):
        """取消处理"""
        if state.cancel_event:
            state.cancel_event.set()
            self.log_component.push('⏳ 正在取消...')
            ui.notify('正在取消处理...', type='warning', position='top')


# ============================================================
# 图片预览页
# ============================================================

class PreviewPage:
    """图片预览页面"""

    def __init__(self):
        self.dir_select = None
        self.status_label = None
        self.thumbnail_container = None
        self.preview_image = None
        self.prev_btn = None
        self.next_btn = None
        
        # 图片信息绑定属性
        self._img_size = '-'
        self._img_filesize = '-'
        self._img_filename = '-'

    def build(self):
        """构建 UI"""
        with ui.column().classes('w-full gap-4'):
            # 标题卡片
            with ui.card().classes('w-full p-0 overflow-hidden'):
                with ui.element('div').classes('gradient-header p-4'):
                    ui.label('🖼 图片预览').classes('text-2xl font-bold text-white')
                    ui.label('浏览和管理生成的数据集').classes('text-white/70 text-sm')

            # 工具栏
            with ui.card().classes('w-full p-4 card-hover'):
                with ui.row().classes('w-full items-center gap-3'):
                    ui.icon('folder_open', color='primary').classes('text-xl')
                    ui.label('选择目录:').classes('font-semibold text-gray-700')

                    self.dir_select = ui.select(
                        options=[],
                        label='选择输出目录',
                        on_change=self._on_dir_select
                    ).classes('w-64').props('outlined dense')

                    ui.button('刷新', icon='refresh', on_click=self._refresh_dirs).props('outline dense')
                    ui.button('浏览', icon='explore', on_click=self._browse_dir).props('outline dense')
                    ui.button('打开', icon='launch', on_click=self._open_folder).props('outline dense')

                self.status_label = ui.label('请选择包含图片的目录').classes('text-gray-500 mt-2')

        # 主内容区域
        with ui.splitter(value=55).classes('w-full h-[580px]') as splitter:
            with splitter.before:
                # 左侧缩略图网格
                with ui.scroll_area().classes('h-full pr-2'):
                    with ui.row().classes('items-center gap-2 mb-3'):
                        ui.icon('grid_view', color='primary').classes('text-lg')
                        ui.label('📷 缩略图列表').classes('text-lg font-semibold text-gray-700')

                    self.thumbnail_container = ui.element('div').classes(
                        'grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-3'
                    )

            with splitter.after:
                # 右侧大图预览
                with ui.column().classes('h-full pl-4 gap-3'):
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('image', color='primary').classes('text-lg')
                        ui.label('🖼 大图预览').classes('text-lg font-semibold text-gray-700')

                    # 图片显示区域
                    with ui.card().classes('w-full preview-area items-center justify-center p-4').style('min-height: 400px;'):
                        self.preview_image = ui.image('').classes('max-w-full max-h-[380px] object-contain rounded-lg')

                    # 图片详细信息
                    with ui.card().classes('w-full p-3 bg-gray-50'):
                        with ui.row().classes('w-full items-center justify-around gap-4'):
                            with ui.row().classes('items-center gap-1'):
                                ui.label('📐 尺寸:').classes('text-gray-500')
                                ui.label('-').classes('font-semibold text-gray-800').bind_text_from(self, '_img_size')
                            
                            ui.label('|').classes('text-gray-300')
                            
                            with ui.row().classes('items-center gap-1'):
                                ui.label('💾 大小:').classes('text-gray-500')
                                ui.label('-').classes('font-semibold text-gray-800').bind_text_from(self, '_img_filesize')
                            
                            ui.label('|').classes('text-gray-300')
                            
                            with ui.row().classes('items-center gap-1'):
                                ui.label('📄 文件:').classes('text-gray-500')
                                ui.label('-').classes('font-semibold text-gray-800 truncate max-w-[150px]').bind_text_from(self, '_img_filename')

                    # 操作按钮
                    with ui.row().classes('w-full justify-center gap-3'):
                        self.prev_btn = ui.button(
                            '◀ 上一张',
                            on_click=lambda: self._navigate('prev')
                        ).props('outline icon=navigate_before')

                        ui.button(
                            '🗑 删除',
                            on_click=self._delete_image
                        ).props('color=negative outline icon=delete')

                        self.next_btn = ui.button(
                            '下一张 ▶',
                            on_click=lambda: self._navigate('next')
                        ).props('outline icon=navigate_next')

        self._refresh_dirs()

    async def _refresh_dirs(self):
        """刷新目录列表"""
        dirs = get_available_output_dirs()
        self.dir_select.options = dirs

        if dirs:
            self.dir_select.value = dirs[0]
            await self._on_dir_select()
        else:
            self.status_label.text = 'output 目录下暂无图片文件夹'

    async def _browse_dir(self):
        """浏览选择目录"""
        path = await select_directory()
        if path:
            state.current_directory = path
            await self._load_images()

    async def _on_dir_select(self):
        """选择目录"""
        selected = self.dir_select.value
        if selected:
            output_base = get_output_base_dir()
            state.current_directory = str(output_base / selected)
            await self._load_images()

    def _open_folder(self):
        """打开当前目录"""
        if state.current_directory and os.path.exists(state.current_directory):
            os.startfile(state.current_directory)

    async def _load_images(self):
        """加载图片列表"""
        if not state.current_directory:
            return

        state.image_list = await run.io_bound(
            scan_images_in_directory,
            state.current_directory
        )

        count = len(state.image_list)
        self.status_label.text = f'📊 共加载 {count} 张图片'

        await self._render_thumbnails()

        if count > 0:
            state.selected_index = 0
            await self._show_image(0)

    async def _render_thumbnails(self):
        """渲染缩略图网格"""
        self.thumbnail_container.clear()

        with self.thumbnail_container:
            for idx, img_path in enumerate(state.image_list):
                is_selected = idx == state.selected_index
                selected_class = "ring-2 ring-blue-500 ring-offset-2" if is_selected else ""

                with ui.card().classes(
                    f'p-1 cursor-pointer transition-all duration-200 hover:shadow-lg {selected_class}'
                ).on('click', lambda i=idx: self._select_image(i)):
                    dir_name = Path(state.current_directory).name
                    img_url = f'/output/{dir_name}/{img_path}'

                    ui.image(img_url).classes('w-full h-16 object-cover rounded')

                    name = Path(img_path).name
                    if len(name) > 10:
                        name = name[:8] + '...'
                    ui.label(name).classes('text-xs text-center text-gray-500 truncate w-full mt-1')

    async def _select_image(self, index: int):
        """选择图片"""
        state.selected_index = index
        await self._show_image(index)
        await self._render_thumbnails()

    async def _show_image(self, index: int):
        """显示图片"""
        if not state.image_list or index < 0 or index >= len(state.image_list):
            return

        img_path = state.image_list[index]
        dir_name = Path(state.current_directory).name
        img_url = f'/output/{dir_name}/{img_path}'

        self.preview_image.set_source(img_url)

        # 获取图片详细信息
        full_path = Path(state.current_directory) / img_path
        if full_path.exists():
            w, h, file_size = await run.io_bound(get_image_info, str(full_path))

            # 更新绑定属性
            self._img_size = f'{w} × {h}'
            self._img_filesize = f'{file_size:.1f} KB'
            self._img_filename = Path(img_path).name

        self.prev_btn.props(f'disable={index == 0}')
        self.next_btn.props(f'disable={index >= len(state.image_list) - 1}')

        self.status_label.text = f'📊 共 {len(state.image_list)} 张图片 | 当前第 {index + 1} 张'

    async def _navigate(self, direction: str):
        """导航"""
        if direction == 'prev' and state.selected_index > 0:
            state.selected_index -= 1
        elif direction == 'next' and state.selected_index < len(state.image_list) - 1:
            state.selected_index += 1

        await self._show_image(state.selected_index)
        await self._render_thumbnails()

    async def _delete_image(self):
        """删除当前图片"""
        if state.selected_index < 0 or state.selected_index >= len(state.image_list):
            return

        img_path = state.image_list[state.selected_index]
        full_path = Path(state.current_directory) / img_path

        try:
            if full_path.exists():
                os.remove(full_path)
                ui.notify(f'已删除: {Path(img_path).name}', type='info', position='top')

            state.image_list.pop(state.selected_index)

            if state.image_list:
                state.selected_index = min(state.selected_index, len(state.image_list) - 1)
                await self._show_image(state.selected_index)
            else:
                state.selected_index = -1
                self.preview_image.set_source('')
                self._img_size = '-'
                self._img_filesize = '-'
                self._img_filename = '-'
                self.status_label.text = '无图片'

            await self._render_thumbnails()

        except Exception as e:
            ui.notify(f'删除失败: {e}', type='negative', position='top')


# ============================================================
# 主应用
# ============================================================

def create_app():
    """创建应用"""
    # 添加自定义样式
    ui.add_head_html(f'<style>{CUSTOM_CSS}</style>')

    # 挂载静态文件目录
    output_base = get_output_base_dir()
    if output_base.exists():
        app.add_static_files('/output', str(output_base))

    # 创建页面组件实例
    processing_page = ProcessingPage()
    preview_page = PreviewPage()

    # Header
    with ui.header().classes('gradient-header px-6 py-3 items-center'):
        with ui.row().classes('items-center gap-3'):
            ui.icon('movie_creation', size='md').classes('text-white')
            ui.label('🎬 视频人物数据集生成器').classes('text-xl font-bold text-white')
        ui.space()
        with ui.row().classes('items-center gap-2'):
            ui.badge('VideoPersonDataset', color='white').classes('text-white/80 text-xs')

    # 主标签页
    with ui.tabs().classes('w-full bg-white shadow-sm') as tabs:
        with ui.row().classes('w-full justify-center gap-8 py-2'):
            tab1 = ui.tab('🎬 视频处理', icon='videocam').props('no-caps')
            tab2 = ui.tab('🖼 图片预览', icon='image').props('no-caps')

    with ui.tab_panels(tabs, value=tab1).classes('w-full p-4 bg-gray-50'):
        with ui.tab_panel(tab1):
            processing_page.build()

        with ui.tab_panel(tab2):
            preview_page.build()


# ============================================================
# 启动入口
# ============================================================

if __name__ in {'__main__', '__mp_main__'}:
    create_app()
    ui.run(
        native=True,
        window_size=(1280, 900),
        title='视频人物数据集生成器',
        favicon='🎬',
        dark=False,
        reload=False
    )