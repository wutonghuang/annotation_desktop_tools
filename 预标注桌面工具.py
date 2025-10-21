#!/usr/bin/python3
# encoding: utf-8
"""
@author: pengjie.jiang
@email: jp_jie@163.com
@file: 预标注桌面工具.py
annotation desktop tools
"""
import json
import os
import tkinter as tk
import xml.etree.ElementTree as ET
from tkinter import filedialog, messagebox, ttk
from xml.dom import minidom

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO


class ImageAnnotator(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        # 变量初始化
        self.state = False
        self.image_path = ""  # 当前图像路径
        self.original_image = None  # 原始图像
        self.display_image = None  # 显示图像
        self.tk_image = None  # 用于显示的PIL图像
        self.canvas = None
        self.annotations = []
        self.current_annotation = None
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.scale_factor = 1.0  # 添加一个变量来跟踪缩放比例
        self.selected_bbox_id = None  # 添加一个变量来跟踪选中的标注框
        self.dragging = False  # 添加一个变量来跟踪是否正在拖动标注
        self.drag_offset_x = 0  # 添加一个变量来跟踪是否修改了标注
        self.drag_offset_y = 0  # 添加一个变量来跟踪是否修改了标注
        self.modified = False  # 添加一个变量来跟踪是否修改了标注
        self.image_files = []  # 存储文件夹中的所有图像文件
        self.current_image_index = 0  # 当前图像的索引
        self.mode = "select"  # 添加一个变量来跟踪当前模式
        self.updating_selection = False  # 添加一个变量来跟踪是否正在更新选择
        # 调整大小相关变量
        self.resizing = False
        self.resize_mode = None  # 'nw', 'n', 'ne', 'w', 'e', 'sw', 's', 'se'
        self.resize_handle_size = 8  # 调整手柄大小

        # 标签管理
        try:
            with open('./predefined_classes.txt', 'r') as f:
                self.labels = f.read().splitlines()
        except FileNotFoundError:
            self.labels = []
        if self.labels:
            self.current_label = tk.StringVar(value=self.labels[0])
        else:
            self.current_label = tk.StringVar()

        # 保存格式选择
        self.save_format = tk.StringVar(value="JSON")

        self.setup_ui()

    def toggle_state(self):
        self.state = not self.state
        if self.state:
            self.radio_button1.config(text="默认选中标签")
        else:
            self.radio_button1.config(text="自定义选标签")

    def setup_ui(self):
        """设置界面"""
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 顶部控制面板
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.BOTH, pady=(0, 5))

        # 文件操作按钮
        file_frame = ttk.LabelFrame(control_frame, text="文件操作", padding=5)
        file_frame.pack(side=tk.LEFT, padx=5)

        ttk.Button(file_frame, text="打开图像", command=self.open_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="打开文件夹", command=self.open_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="上一张", command=self.prev_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="下一张", command=self.next_image).pack(side=tk.LEFT, padx=2)

        # 标注操作
        annotate_frame = ttk.LabelFrame(control_frame, text="标注操作", padding=5)
        annotate_frame.pack(side=tk.LEFT, padx=5)

        ttk.Button(annotate_frame, text="新建矩形", command=self.start_rect_mode).pack(side=tk.LEFT, padx=2)
        ttk.Button(annotate_frame, text="删除选中", command=self.delete_selected_annotation).pack(side=tk.LEFT, padx=2)
        ttk.Button(annotate_frame, text="清除所有", command=self.clear_annotations).pack(side=tk.LEFT, padx=2)

        # 缩放控制
        zoom_frame = ttk.LabelFrame(control_frame, text="缩放控制", padding=5)
        zoom_frame.pack(side=tk.LEFT, padx=5)

        ttk.Button(zoom_frame, text="放大", command=lambda: self.zoom_image(0.1)).pack(side=tk.TOP, padx=2)
        ttk.Button(zoom_frame, text="缩小", command=lambda: self.zoom_image(-0.1)).pack(side=tk.BOTTOM, padx=2)
        ttk.Button(zoom_frame, text="适应窗口", command=self.fit_to_window).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="实际尺寸", command=self.actual_size).pack(side=tk.RIGHT, padx=2)

        # 保存操作
        save_frame = ttk.LabelFrame(control_frame, text="保存选项", padding=5)
        save_frame.pack(side=tk.LEFT, padx=5)

        format_frame = tk.Frame(save_frame)
        format_frame.pack(pady=2)

        ttk.Radiobutton(format_frame, text="JSON", variable=self.save_format, value="JSON").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(format_frame, text="PASCAL VOC", variable=self.save_format, value="VOC").pack(side=tk.LEFT,
                                                                                                      padx=2)
        ttk.Radiobutton(format_frame, text="YOLO", variable=self.save_format, value="YOLO").pack(side=tk.LEFT, padx=2)

        ttk.Button(save_frame, text="保存标注", command=self.save_annotations_direct).pack(side=tk.LEFT, pady=2)
        ttk.Button(save_frame, text="确认标注", command=self.save_confirm_annotations_direct).pack(side=tk.RIGHT,
                                                                                                   pady=2)

        # 主内容区域
        content_frame = tk.Frame(main_frame)
        content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # content_frame.pack_propagate(False)  # 禁止内容区域自动调整大小

        # 左侧画布区域
        canvas_frame = ttk.LabelFrame(content_frame, text="图像显示", padding=5)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # 画布和滚动条
        # self.canvas = tk.Canvas(canvas_frame, bg="white", cursor="crosshair")
        self.canvas = tk.Canvas(canvas_frame, bg="lightgray", cursor="crosshair")
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)

        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 绑定画布事件
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<MouseWheel>", self.on_canvas_scroll)
        self.canvas.bind("<Button-4>", self.on_canvas_scroll)
        self.canvas.bind("<Button-5>", self.on_canvas_scroll)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())
        self.canvas.bind("<Left>", lambda event: self.prev_image())
        self.canvas.bind("<Right>", lambda event: self.next_image())
        self.canvas.bind("<KeyPress-A>", lambda event: self.prev_image())
        self.canvas.bind("<KeyPress-D>", lambda event: self.next_image())
        self.canvas.bind("<KeyPress-a>", lambda event: self.prev_image())
        self.canvas.bind("<KeyPress-d>", lambda event: self.next_image())
        self.canvas.bind("<space>", lambda event: self.save_annotations_direct())
        self.canvas.bind("<KeyPress-W>", lambda event: self.start_rect_mode())
        self.canvas.bind("<KeyPress-w>", lambda event: self.start_rect_mode())

        # 中间控制面板
        control_panel = ttk.LabelFrame(content_frame, text="标注控制", padding=5, width=200)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0))
        control_panel.pack_propagate(False)
        # 标签选择
        label_frame = ttk.LabelFrame(control_panel, text="标签选择", padding=5)
        label_frame.pack(fill=tk.X, pady=5)

        self.label_combo = ttk.Combobox(label_frame, textvariable=self.current_label, values=self.labels,
                                        state="readonly")
        self.label_combo.pack(fill=tk.X, padx=2, pady=2)
        self.label_combo.bind("<<ComboboxSelected>>", self.on_label_select)
        self.radio_button1 = ttk.Button(label_frame, text="自定义选标签", command=self.toggle_state)
        self.radio_button1.pack()

        # 标注列表
        list_frame = ttk.LabelFrame(control_panel, text="标注列表", padding=5)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.annotation_listbox = tk.Listbox(list_frame, exportselection=False)  #
        v_control_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.annotation_listbox.yview)
        h_control_scrollbar = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL, command=self.annotation_listbox.xview)
        v_control_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_control_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.annotation_listbox.config(yscrollcommand=v_control_scrollbar.set, xscrollcommand=h_control_scrollbar.set)
        self.annotation_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.annotation_listbox.bind("<<ListboxSelect>>", self.on_annotation_select)

        # 标注详情
        detail_frame = ttk.LabelFrame(control_panel, text="标注详情", padding=5)
        detail_frame.pack(fill=tk.X, pady=5)

        self.detail_text = tk.Text(detail_frame, height=10, wrap=tk.WORD)
        detail_scrollbar = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=self.detail_text.yview)
        self.detail_text.config(yscrollcommand=detail_scrollbar.set)
        self.detail_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        detail_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 右侧文件列表
        files_panel = ttk.LabelFrame(content_frame, text="文件列表", padding=5, width=250)
        files_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        files_panel.pack_propagate(False)
        self.files_annotation_listbox = tk.Listbox(files_panel, exportselection=False)
        v_files_scrollbar = ttk.Scrollbar(files_panel, orient=tk.VERTICAL, command=self.files_annotation_listbox.yview)
        h_files_scrollbar = ttk.Scrollbar(files_panel, orient=tk.HORIZONTAL,
                                          command=self.files_annotation_listbox.xview)
        v_files_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_files_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.files_annotation_listbox.config(yscrollcommand=v_files_scrollbar.set, xscrollcommand=h_files_scrollbar.set)
        self.files_annotation_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.files_annotation_listbox.bind("<<ListboxSelect>>", self.files_on_annotation_select)
        # 状态栏
        canvas_frame_bottom = ttk.LabelFrame(self, text="状态栏", relief=tk.SUNKEN, height=5)
        canvas_frame_bottom.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.status_var = tk.StringVar(value="就绪 - 欢迎使用图像标注工具")
        status_bar = ttk.Label(canvas_frame_bottom, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.status_var_right = tk.StringVar(value="")
        status_bar_right = ttk.Label(canvas_frame_bottom, textvariable=self.status_var_right, relief=tk.SUNKEN,
                                     anchor=tk.W)
        status_bar_right.pack(side=tk.RIGHT, fill=tk.BOTH)

        # 初始化模式
        self.mode = "select"
        # 初始化画布图像引用
        self.canvas_img = None
        self.image_offset_x = 0
        self.image_offset_y = 0

    def files_listbox(self, index=0, judge=True, confirm_judge=False):

        if judge:
            if self.files_annotation_listbox.get(0, tk.END):  # 如果列表不为空
                self.files_annotation_listbox.delete(0, tk.END)  # 清空列表
            for image_index, i in enumerate(self.image_files):
                self.files_annotation_listbox.insert(tk.END, os.path.basename(i))
                self.files_annotation_listbox.itemconfig(image_index, {'bg': "#FF9999"})
        if confirm_judge:
            confirm_path = os.path.join(os.path.dirname(self.image_path), "confirm_example.txt")
            confirm_judge_file = True
            if os.path.exists(confirm_path):
                with open(confirm_path, 'r', encoding="utf-8") as f:
                    files_name = f.readlines()
                    if len(self.image_files) == len(files_name):
                        confirm_judge_file = False
                    for line in files_name:
                        line_split = line.strip().split(" ")
                        f_name = line_split[0]
                        file_num = line_split[-1]
                        if file_num == "1":
                            self.files_annotation_listbox.itemconfig(self.image_files.index(f_name), {'bg': "#99FF99"})

            if confirm_judge_file:
                with open(confirm_path, 'w', encoding="utf-8") as f:
                    for i in self.image_files:
                        f.write(i + " 0" + '\n')

        self.files_annotation_listbox.selection_clear(0, tk.END)  # 清除所有选中项
        self.files_annotation_listbox.selection_set(index)  # 选择当前项
        self.files_annotation_listbox.see(index)  # 滚动到选中项

    def open_image(self):
        """打开单个图像文件"""
        if self.modified and not self.prompt_save():
            return

        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.image_files.clear()
            self.image_files.append(file_path.replace("\\", "/"))
            self.load_image(file_path)
            self.files_listbox()

    def open_folder(self):
        """打开图像文件夹"""
        if self.modified and self.prompt_save():
            return

        folder_path = filedialog.askdirectory(title="选择图像文件夹")
        if folder_path:
            self.image_folder = folder_path
            self.image_files.clear()
            # 获取所有支持的图像文件
            self.image_files = [
                os.path.join(folder_path, f).replace("\\", "/") for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'))
            ]
            if self.image_files:
                self.image_files.sort()  # 按文件名排序
                self.current_image_index = 0
                self.load_image(self.image_files[0])
                self.files_listbox(confirm_judge=True)
                self.status_var.set(f"已加载文件夹: {len(self.image_files)} 张图像")
            else:
                messagebox.showwarning("警告", "文件夹中没有找到支持的图像文件")

    def load_image(self, image_path):
        """加载图像"""
        try:
            self.image_path = image_path
            self.original_image = Image.open(image_path)

            self.annotations = []
            self.current_annotation = None
            self.scale_factor = 1.0
            self.selected_bbox_id = None
            self.modified = False
            # 加载已有的标注文件
            self.load_annotations()
            # 显示图像（适应界面）
            self.display_image_func()

            self.status_var_right.set(
                f"已加载: {os.path.basename(image_path)}，素材进行{self.image_files.index(self.image_path) + 1} / {len(self.image_files)}")
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图像: {str(e)}")

    def display_image_func(self):
        """在画布上显示图像 - 默认自动适应界面"""
        if not self.original_image:
            return

        # 清除画布
        self.canvas.delete("all")
        # 获取画布尺寸
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        # 如果画布尺寸小于等于1，则等待下一次调用
        if canvas_width <= 1 or canvas_height <= 1:
            self.after(100, self.display_image_func)
            return
        # 计算缩放后的尺寸（保持原始比例）
        original_width, original_height = self.original_image.size
        scale_x = canvas_width / original_width
        scale_y = canvas_height / original_height
        if self.scale_factor == 1:
            self.scale_factor = min(scale_x, scale_y) * 0.95  # 留出边距
            # self.scale_factor = min(scale_x, scale_y)  # 留出边距
        new_width = int(original_width * self.scale_factor)  # 缩放后的宽度
        new_height = int(original_height * self.scale_factor)  # 缩放后的高度
        # print(new_width, new_height, self.scale_factor)
        # 调整图像大小
        self.display_image = self.original_image.resize((new_width, new_height), Image.LANCZOS)

        # 转换为Tkinter图像
        self.tk_image = ImageTk.PhotoImage(self.display_image)

        # # 在画布上显示图像（从左上角开始，不居中）
        # self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # 居中显示图像
        self.image_offset_x = (canvas_width - new_width) // 2
        self.image_offset_y = (canvas_height - new_height) // 2
        # self.image_offset_x = abs(image_offset_x)
        # self.image_offset_y = abs(image_offset_y)
        self.canvas_img = self.canvas.create_image(self.image_offset_x, self.image_offset_y, anchor=tk.NW,
                                                   image=self.tk_image)

        # 更新画布滚动区域以适应图像尺寸
        # self.canvas.config(scrollregion=(0, 0, new_width, new_height))
        # self.canvas.config(scrollregion=(0, 0, max(canvas_width, new_width + self.image_offset_x * 2),
        #                                  max(canvas_height, new_height + self.image_offset_y * 2)
        #                                  ))
        self.canvas.config(scrollregion=(min(0, self.image_offset_x), min(0, self.image_offset_y),
                                         max(canvas_width, new_width + self.image_offset_x),
                                         max(canvas_height, new_height + self.image_offset_y)
                                         ))

        # 重绘标注
        self.redraw_annotations()
        self.status_var.set(f"显示尺寸: {new_width} x {new_height} (缩放: {int(self.scale_factor * 100)}%)")

    def zoom_image(self, factor):
        """缩放图像"""
        if not self.original_image:
            return

        # 限制缩放范围
        # new_scale = self.scale_factor * factor
        new_scale = self.scale_factor + factor
        if 0.1 <= new_scale <= 5.0:
            self.scale_factor = new_scale
            self.display_image_func()

    def fit_to_window(self):
        """适应窗口大小"""
        if not self.original_image:
            return

        self.scale_factor = 1.0
        self.display_image_func()
        self.status_var.set("图像已适应窗口")

    def actual_size(self):
        """显示实际尺寸"""
        self.scale_factor = 1.0001
        self.display_image_func()
        self.status_var.set("实际尺寸: 100%")

    def on_canvas_scroll(self, event):
        """处理画布滚动事件 - 缩放图像"""
        if event.delta > 0:
            # self.zoom_image(1.1)
            self.zoom_image(0.1)
        elif event.delta < 0:
            # self.zoom_image(0.9)
            self.zoom_image(-0.1)

    def start_rect_mode(self):
        """进入矩形绘制模式"""
        self.mode = "rect"
        self.canvas.config(cursor="crosshair")
        self.status_var.set("模式: 绘制矩形 - 点击并拖动以创建标注框")

    def get_resize_mode(self, x, y, bbox_coords):
        """确定调整大小的模式"""
        x1, y1, x2, y2 = bbox_coords
        handle_size = self.resize_handle_size

        # 检查是否在角点或边缘附近
        if abs(x - x1) <= handle_size and abs(y - y1) <= handle_size:
            return "nw"
        elif abs(x - x2) <= handle_size and abs(y - y1) <= handle_size:
            return "ne"
        elif abs(x - x1) <= handle_size and abs(y - y2) <= handle_size:
            return "sw"
        elif abs(x - x2) <= handle_size and abs(y - y2) <= handle_size:
            return "se"
        elif abs(x - (x1 + x2) / 2) <= handle_size and abs(y - y1) <= handle_size:
            return "n"
        elif abs(x - (x1 + x2) / 2) <= handle_size and abs(y - y2) <= handle_size:
            return "s"
        elif abs(x - x1) <= handle_size and abs(y - (y1 + y2) / 2) <= handle_size:
            return "w"
        elif abs(x - x2) <= handle_size and abs(y - (y1 + y2) / 2) <= handle_size:
            return "e"
        return None

    def get_resize_cursor(self):
        """获取调整大小光标"""
        if self.resize_mode == "nw":
            return "size_nw_se"
        elif self.resize_mode == "ne":
            return "size_ne_sw"
        elif self.resize_mode == "sw":
            return "size_ne_sw"
        elif self.resize_mode == "se":
            return "size_nw_se"
        elif self.resize_mode == "n":
            return "size_ns"
        elif self.resize_mode == "s":
            return "size_ns"
        elif self.resize_mode == "w":
            return "size_we"
        elif self.resize_mode == "e":
            return "size_we"
        return "arrow"

    def on_canvas_click(self, event):
        """处理画布点击事件"""

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        # 转换为图像坐标
        img_x = x - self.image_offset_x
        img_y = y - self.image_offset_y

        # 检查是否在图像范围内
        if not self.original_image:
            return
        img_width = self.original_image.width * self.scale_factor
        img_height = self.original_image.height * self.scale_factor

        if img_x < 0 or img_y < 0 or img_x > img_width or img_y > img_height:
            return  # 点击在图像外部
        if self.mode == "rect":
            self.start_x = img_x
            self.start_y = img_y
            self.rect = self.canvas.create_rectangle(x, y, x, y, outline="red", width=1, dash=(4, 4))
        else:
            # 选择模式：检查是否点击了标注框
            self.selected_bbox_id = None
            overlapping_items = self.canvas.find_overlapping(x - 3, y - 3, x + 3, y + 3)

            for item in overlapping_items:
                tags = self.canvas.gettags(item)
                if "bbox" in tags:
                    self.selected_bbox_id = item
                    bbox_index = int(tags[1]) if len(tags) > 1 else -1
                    if 0 <= bbox_index < len(self.annotations):
                        self.current_annotation = self.annotations[bbox_index]
                        self.highlight_annotation(bbox_index)
                        # 检查是否是调整大小操作
                        bbox_coords = self.canvas.coords(item)
                        self.resize_mode = self.get_resize_mode(x, y, bbox_coords)
                        if self.resize_mode:
                            # 进入调整大小模式
                            self.mode = "resize"
                            self.resizing = True
                            self.canvas.config(cursor=self.get_resize_cursor())
                            self.status_var.set(f"模式: 调整标注框大小 ({self.resize_mode})")
                        else:
                            # 进入移动模式
                            self.mode = "move"
                            self.dragging = True
                            bbox_coords = self.canvas.coords(item)
                            self.drag_offset_x = x - bbox_coords[0]
                            self.drag_offset_y = y - bbox_coords[1]
                            self.canvas.config(cursor="fleur")
                            self.status_var.set("模式: 移动标注框")
                    break

    def on_canvas_drag(self, event):
        """处理画布拖动事件"""
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)

        if self.mode == "rect" and self.rect:
            self.canvas.coords(self.rect, self.start_x + self.image_offset_x,
                               self.start_y + self.image_offset_y, cur_x, cur_y)
        elif self.mode == "move" and self.dragging and self.selected_bbox_id:
            # 移动标注框
            new_x1 = cur_x - self.drag_offset_x
            new_y1 = cur_y - self.drag_offset_y
            bbox_coords = self.canvas.coords(self.selected_bbox_id)
            width = bbox_coords[2] - bbox_coords[0]
            height = bbox_coords[3] - bbox_coords[1]

            new_x2 = new_x1 + width
            new_y2 = new_y1 + height

            self.canvas.coords(self.selected_bbox_id, new_x1, new_y1, new_x2, new_y2)

            # 更新关联的文本标签
            tags = self.canvas.gettags(self.selected_bbox_id)
            if len(tags) > 2:
                text_id = int(tags[2])
                self.canvas.coords(text_id, new_x1, new_y1 - 10)
        elif self.mode == "resize" and self.resizing and self.selected_bbox_id:
            # 调整标注框大小
            bbox_coords = self.canvas.coords(self.selected_bbox_id)
            x1, y1, x2, y2 = bbox_coords
            if self.resize_mode == "nw":
                x1, y1 = cur_x, cur_y
            elif self.resize_mode == "ne":
                x2, y1 = cur_x, cur_y
            elif self.resize_mode == "sw":
                x1, y2 = cur_x, cur_y
            elif self.resize_mode == "se":
                x2, y2 = cur_x, cur_y
            elif self.resize_mode == "n":
                y1 = cur_y
            elif self.resize_mode == "s":
                y2 = cur_y
            elif self.resize_mode == "w":
                x1 = cur_x
            elif self.resize_mode == "e":
                x2 = cur_x
            # 确保坐标有效
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            self.canvas.coords(self.selected_bbox_id, x1, y1, x2, y2)

            # 更新关联的文本位置
            tags = self.canvas.gettags(self.selected_bbox_id)
            if len(tags) > 2:
                text_id = int(tags[2])
                self.canvas.coords(text_id, x1, y1 - 10)

    def on_canvas_release(self, event):
        """处理画布释放事件"""
        if self.mode == "rect" and self.rect:
            end_x = self.canvas.canvasx(event.x) - self.image_offset_x
            end_y = self.canvas.canvasy(event.y) - self.image_offset_y

            # 转换为原始图像坐标
            if abs(end_x - self.start_x) > 5 and abs(end_y - self.start_y) > 5:  # 忽略过小的矩形
                x1 = int(min(self.start_x, end_x) / self.scale_factor)
                y1 = int(min(self.start_y, end_y) / self.scale_factor)
                x2 = int(max(self.start_x, end_x) / self.scale_factor)
                y2 = int(max(self.start_y, end_y) / self.scale_factor)
                self.prompt_for_label(x1, y1, x2, y2)
            else:
                self.canvas.delete(self.rect)

            self.mode = "select"
            self.canvas.config(cursor="arrow")

        elif self.mode == "move" and self.dragging:
            if self.selected_bbox_id and self.current_annotation:
                bbox_coords = self.canvas.coords(self.selected_bbox_id)
                # 转换为原始图像坐标
                x1 = int((bbox_coords[0] - self.image_offset_x) / self.scale_factor)
                y1 = int((bbox_coords[1] - self.image_offset_y) / self.scale_factor)
                x2 = int((bbox_coords[2] - self.image_offset_x) / self.scale_factor)
                y2 = int((bbox_coords[3] - self.image_offset_y) / self.scale_factor)

                self.current_annotation["bbox"] = [x1, y1, x2, y2]
                self.update_annotation_list()  # 更新标注列表
                self.show_annotation_details(self.current_annotation)
                self.modified = True

            self.dragging = False
            self.mode = "select"
            self.canvas.config(cursor="arrow")
            self.highlight_annotation(-1)
            self.status_var.set("移动完毕")

        elif self.mode == "resize" and self.resizing:
            if self.selected_bbox_id and self.current_annotation:
                bbox_coords = self.canvas.coords(self.selected_bbox_id)
                # 转换为原始图像坐标
                x1 = int((bbox_coords[0] - self.image_offset_x) / self.scale_factor)
                y1 = int((bbox_coords[1] - self.image_offset_y) / self.scale_factor)
                x2 = int((bbox_coords[2] - self.image_offset_x) / self.scale_factor)
                y2 = int((bbox_coords[3] - self.image_offset_y) / self.scale_factor)

                self.current_annotation["bbox"] = [x1, y1, x2, y2]
                self.update_annotation_list()
                self.show_annotation_details(self.current_annotation)
                self.modified = True

            self.resizing = False
            self.resize_mode = None
            self.mode = "select"
            self.canvas.config(cursor="arrow")
            self.status_var.set("修改框完毕")
        self.canvas.config(bg='lightgray')  # 更改背景色为灰色

    def on_canvas_motion(self, event):
        """处理鼠标移动事件"""
        if self.mode == "select":
            x = self.canvas.canvasx(event.x)  # 获取相对于画布的x坐标
            y = self.canvas.canvasy(event.y)  # 获取相对于画布的y坐标
            self.status_var.set(
                f"界面位置: {x}, {y}，图片当前位置: {int((event.x - self.image_offset_x) / self.scale_factor)}, {int((event.y - self.image_offset_y) / self.scale_factor)}")
            overlapping_items = self.canvas.find_overlapping(x - 3, y - 3, x + 3, y + 3)  # 找到所有与鼠标位置重叠的项
            on_bbox = False
            resize_mode = None

            for item in overlapping_items:
                tags = self.canvas.gettags(item)
                if "bbox" in tags:
                    on_bbox = True
                    bbox_coords = self.canvas.coords(item)
                    resize_mode = self.get_resize_mode(x, y, bbox_coords)
                    break
            if resize_mode:
                # 根据调整模式设置光标形状
                self.canvas.config(cursor=self.get_resize_cursor())
            elif on_bbox:
                self.canvas.config(cursor="hand2")
            else:
                self.canvas.config(cursor="arrow")

    def prompt_for_label(self, x1, y1, x2, y2):
        """为新建的标注弹出标签选择对话框"""
        if self.state:
            label = self.current_label.get()
            if label:
                annotation = {
                    "label": label,
                    "bbox": [x1, y1, x2, y2],
                    "id": len(self.annotations)
                }
                self.annotations.append(annotation)  # 添加到标注列表
                self.draw_annotation(annotation)  # 绘制标注
                self.update_annotation_list()  # 更新标注列表
                self.modified = True
            if self.rect:
                self.canvas.delete(self.rect)
        else:
            dialog = tk.Toplevel(self)
            dialog.title("选择标签")
            dialog.geometry("300x150")
            dialog.transient(self)  # 使对话框成为主窗口的子窗口
            dialog.grab_set()

            # 居中对话框
            dialog.update_idletasks()
            root_x = self.winfo_x()
            root_y = self.winfo_y()
            root_width = self.winfo_width()
            root_height = self.winfo_height()
            dialog_width = dialog.winfo_width()
            dialog_height = dialog.winfo_height()
            x = root_x + (root_width - dialog_width) // 2
            y = root_y + (root_height - dialog_height) // 2
            dialog.geometry(f"+{x}+{y}")

            tk.Label(dialog, text="选择或输入标签:").pack(pady=10)

            label_var = tk.StringVar(value=self.current_label.get())
            label_combo = ttk.Combobox(dialog, textvariable=label_var, values=self.labels)
            label_combo.pack(pady=5, padx=20, fill=tk.X)

            button_frame = tk.Frame(dialog)
            button_frame.pack(pady=10)

            def on_ok():
                label_name = label_var.get()
                if label_name:
                    annotation_data = {
                        "label": label_name,
                        "bbox": [x1, y1, x2, y2],
                        "id": len(self.annotations)
                    }
                    self.annotations.append(annotation_data)  # 添加到标注列表
                    self.draw_annotation(annotation_data)  # 绘制标注
                    self.update_annotation_list()  # 更新标注列表
                    self.modified = True

                    if label_name not in self.labels:
                        self.labels.append(label_name)
                        self.label_combo.config(values=self.labels)
                if self.rect:
                    self.canvas.delete(self.rect)
                dialog.destroy()

            def on_cancel():
                if self.rect:
                    self.canvas.delete(self.rect)
                dialog.destroy()

            ttk.Button(button_frame, text="确定", command=on_ok).pack(side=tk.LEFT, padx=10)
            ttk.Button(button_frame, text="取消", command=on_cancel).pack(side=tk.LEFT, padx=10)
            dialog.protocol("WM_DELETE_WINDOW", on_cancel)
            label_combo.focus_set()
            label_combo.select_range(0, tk.END)

    def draw_annotation(self, annotation):
        """在画布上绘制标注"""
        x1, y1, x2, y2 = annotation["bbox"]

        # 转换为显示坐标
        x1_disp = int(x1 * self.scale_factor) + self.image_offset_x
        y1_disp = int(y1 * self.scale_factor) + self.image_offset_y
        x2_disp = int(x2 * self.scale_factor) + self.image_offset_x
        y2_disp = int(y2 * self.scale_factor) + self.image_offset_y
        # print(x1_disp, y1_disp, x2_disp, y2_disp, self.scale_factor, self.image_offset_x, self.image_offset_y)

        # 绘制矩形框
        rect_id = self.canvas.create_rectangle(
            x1_disp, y1_disp, x2_disp, y2_disp,
            outline="red", width=1, tags=("bbox", str(annotation["id"]))
        )

        # 绘制标签文本
        text_id = self.canvas.create_text(x1_disp, y1_disp - 2, text=annotation["label"], fill="red", anchor=tk.SW,
                                          font=("Arial", 5, "bold"), tags=("label", str(annotation["id"])))

        annotation["canvas_ids"] = [rect_id, text_id]
        self.canvas.itemconfig(rect_id, tags=("bbox", str(annotation["id"]), str(text_id)))  # 关联矩形框和文本

    def redraw_annotations(self):
        """重绘所有标注"""
        for annotation in self.annotations:
            self.draw_annotation(annotation)

    def highlight_annotation(self, index):
        """高亮显示选中的标注"""
        if self.updating_selection:
            return
        self.updating_selection = True
        # 重置所有标注的颜色
        for ann in self.annotations:
            if "canvas_ids" in ann:
                for canvas_id in ann["canvas_ids"]:
                    if self.canvas.type(canvas_id) == "rectangle":
                        self.canvas.itemconfig(canvas_id, outline="red", width=1)
                    elif self.canvas.type(canvas_id) == "text":
                        self.canvas.itemconfig(canvas_id, fill="red")

        # 高亮选中的标注
        if index >= 0 and index < len(self.annotations):
            annotation = self.annotations[index]
            if "canvas_ids" in annotation:
                for canvas_id in annotation["canvas_ids"]:
                    if self.canvas.type(canvas_id) == "rectangle":
                        self.canvas.itemconfig(canvas_id, outline="blue", width=2)
                    elif self.canvas.type(canvas_id) == "text":
                        self.canvas.itemconfig(canvas_id, fill="blue")
            # 更新标注列表选中状态
            self.annotation_listbox.selection_clear(0, tk.END)  # 清除所有选中项
            self.annotation_listbox.selection_set(index)  # 选择当前项
            self.annotation_listbox.see(index)  # 滚动到选中项

        self.updating_selection = False

    def update_annotation_list(self):
        """更新标注列表显示"""
        self.annotation_listbox.delete(0, tk.END)
        for i, annotation in enumerate(self.annotations):
            label = annotation["label"]
            x1, y1, x2, y2 = annotation["bbox"]
            self.annotation_listbox.insert(tk.END, f"{i + 1}: {label} ({x1},{y1},{x2},{y2})")

    def on_annotation_select(self, event):
        """处理标注列表选择事件"""
        selection = self.annotation_listbox.curselection()  # 获取当前选中项的索引
        if selection:
            index = selection[0]
            if index < len(self.annotations):
                self.current_annotation = self.annotations[index]
                self.highlight_annotation(index)
                self.show_annotation_details(self.current_annotation)

    def files_on_annotation_select(self, event):
        """处理文件夹列表选择事件"""
        if self.updating_selection:
            return
        selection = self.files_annotation_listbox.curselection()  # 获取当前选中项的索引
        if selection:
            index = selection[0]
            if index < len(self.image_files):
                self.current_annotation = self.image_files[index]
                self.load_image(self.current_annotation)
                self.files_listbox(index, judge=False)

    def show_annotation_details(self, annotation):
        """显示标注详情"""
        self.detail_text.delete(1.0, tk.END)

        label = annotation["label"]
        id = annotation["id"] + 1
        x1, y1, x2, y2 = annotation["bbox"]
        width = x2 - x1
        height = y2 - y1

        details = f"标签: #{id}-{label}\n"
        details += f"位置: ({x1}, {y1}) - ({x2}, {y2})\n"
        details += f"尺寸: {width} x {height}\n"
        details += f"面积: {width * height} 像素"

        self.detail_text.insert(1.0, details)

    def delete_selected_annotation(self):
        """删除选中的标注"""
        if self.current_annotation:
            if "canvas_ids" in self.current_annotation:
                for canvas_id in self.current_annotation["canvas_ids"]:
                    self.canvas.delete(canvas_id)

            if self.current_annotation in self.annotations:
                self.annotations.remove(self.current_annotation)
                self.modified = True

            self.update_annotation_list()
            self.detail_text.delete(1.0, tk.END)
            self.current_annotation = None
            self.highlight_annotation(-1)
            self.status_var.set("已删除选中标注")
        else:
            messagebox.showwarning("警告", "请先选择一个标注")

    def clear_annotations(self):
        """清除所有标注"""
        if self.annotations and messagebox.askyesno("确认", "确定要清除所有标注吗？"):
            for annotation in self.annotations:
                if "canvas_ids" in annotation:
                    for canvas_id in annotation["canvas_ids"]:
                        self.canvas.delete(canvas_id)

            self.annotations = []
            self.modified = True
            self.update_annotation_list()
            self.detail_text.delete(1.0, tk.END)
            self.current_annotation = None
            self.highlight_annotation(-1)

    def prompt_save(self):
        """提示用户保存未保存的修改"""
        if self.modified:
            response = messagebox.askyesnocancel("未保存的修改", "当前图像有未保存的标注，是否保存？")
            if response is None:
                return False
            elif response:
                self.save_annotations_direct()
        return True

    def prev_image(self):
        """切换到上一张图像"""
        if not self.image_files:
            messagebox.showerror("错误", "没有图像切换")
            return

        if not self.prompt_save():
            return
        self.detail_text.delete(1.0, tk.END)  # 清除详情文本框内容
        self.status_var.set("已切换到上一张图像")
        if self.original_image:
            self.original_image.close()
        if self.files_annotation_listbox.curselection():
            self.current_image_index = self.files_annotation_listbox.curselection()[0]  # 获取当前选中项的索引
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.files_listbox(self.current_image_index, judge=False)
            self.canvas.config(bg='lightgray')  # 更改背景色为灰色
            self.load_image(self.image_files[self.current_image_index])
        else:
            messagebox.showerror("错误", "已经是第一张图像")

    def next_image(self):
        """切换到下一张图像"""
        if not self.image_files:
            messagebox.showerror("错误", "没有图像切换")
            return

        if not self.prompt_save():
            return
        self.detail_text.delete(1.0, tk.END)  # 清除详情文本框内容
        self.status_var.set("已切换到下一张图像")
        if self.original_image:
            self.original_image.close()
        if self.files_annotation_listbox.curselection():
            self.current_image_index = self.files_annotation_listbox.curselection()[0]  # 获取当前选中项的索引
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.files_listbox(self.current_image_index, judge=False)
            self.canvas.config(bg='lightgray')  # 更改背景色为灰色
            self.load_image(self.image_files[self.current_image_index])
        else:
            messagebox.showerror("错误", "已经是最后一张图像")

    def check_annotations_bounds(self):
        """检查标注框是否超出图像边界"""
        if not self.original_image or not self.annotations:
            return True

        width, height = self.original_image.size

        for i, annotation in enumerate(self.annotations):
            x1, y1, x2, y2 = annotation["bbox"]

            if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 >= x2 or y1 >= y2:
                messagebox.showerror(
                    "错误",
                    f"标注框 #{i + 1} ({annotation['label']}) 超出图像边界或无效！\n"
                    f"图像尺寸: {width} × {height}\n"
                    f"标注框位置: ({x1},{y1})-({x2},{y2})"
                )
                return False

        return True

    def save_annotations_direct(self):
        """直接保存标注"""
        if not self.image_path or not self.annotations:
            messagebox.showwarning("警告", "没有可保存的标注")
            return

        # 检查标注框边界
        bounds_check = self.check_annotations_bounds()
        if not bounds_check:
            return

        format_type = self.save_format.get()

        if format_type == "JSON":
            self.save_annotations()
        elif format_type == "VOC":
            self.export_pascal_voc()
        elif format_type == "YOLO":
            self.export_yolo()
        # 更改背景色为绿色
        self.canvas.config(bg='#00FF00')

    def save_confirm_annotations_direct(self):
        """确认保存标注"""
        confirm_path = os.path.join(os.path.dirname(self.image_path), "confirm_example.txt")
        if os.path.exists(confirm_path):
            with open(confirm_path, 'r', encoding="utf-8") as file:
                lines = file.readlines()
            for i, line in enumerate(lines):
                if str(self.image_path) in line:
                    lines[i] = str(self.image_path) + " 1" + '\n'
                    self.files_annotation_listbox.itemconfig(i, {'bg': "#99FF99"})
                    break
            with open(confirm_path, 'w', encoding="utf-8") as file:
                file.writelines(lines)
        return

    def save_annotations(self):
        """保存标注到JSON文件"""
        annotation_data = {
            "image_path": self.image_path,
            "image_size": self.original_image.size,
            "scale_factor": self.scale_factor,
            "annotations": self.annotations
        }

        json_path = os.path.splitext(self.image_path)[0] + "_annotations.json"
        if  not self.annotations:
            self.remove_annotations_file(json_path)
            messagebox.showwarning("警告", "没有可导出的标注")
            return
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)
            self.modified = False
            self.status_var.set(f"JSON标注已保存: {os.path.basename(json_path)}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")

    def load_annotations(self):
        """从JSON文件加载标注"""
        if not self.image_path:
            return
        format_type = self.save_format.get()

        if format_type == "JSON":
            json_path = os.path.splitext(self.image_path)[0] + "_annotations.json"
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        annotation_data = json.load(f)

                    self.annotations = annotation_data.get("annotations", [])
                    # if "scale_factor" in annotation_data:
                    #     self.scale_factor = annotation_data["scale_factor"]
                    self.update_annotation_list()  # 更新标注列表
                    self.status_var_right.set(f"已加载标注: {len(self.annotations)} 个")
                except Exception as e:
                    messagebox.showwarning("警告", f"加载标注文件失败: {str(e)}")
            else:
                self.annotations = []
                self.update_annotation_list()
                self.status_var.set("未找到标注文件")
        elif format_type == "VOC":
            xml_file = os.path.splitext(self.image_path)[0] + ".xml"
            all_objects = []
            if os.path.exists(xml_file):
                # 解析XML文件
                tree = ET.parse(xml_file)
                xml_root = tree.getroot()
                # 提取所有person元素
                img_width, img_height = xml_root.findall('.//size')[0].find('width').text, xml_root.findall('.//size')[
                    0].find('height').text
                # 提取xml_object元素
                xml_objects = xml_root.findall('.//object')
                for o_index, ob in enumerate(xml_objects):
                    cdata = {"label": ob.find('name').text,
                             "bbox": [int(ob.find('bndbox/xmin').text), int(ob.find('bndbox/ymin').text),
                                      int(ob.find('bndbox/xmax').text), int(ob.find('bndbox/ymax').text)],
                             "id": o_index}
                    all_objects.append(cdata)

            self.annotations = all_objects
            self.update_annotation_list()  # 更新标注列表
            self.status_var_right.set(f"已加载标注: {len(self.annotations)} 个")

        elif format_type == "YOLO":
            img_width, img_height = self.original_image.size
            text_to_path = os.path.join(os.path.dirname(self.image_path), "classes.txt")
            txt_file = os.path.splitext(self.image_path)[0] + ".txt"
            lines = None
            if os.path.exists(text_to_path):
                with open(text_to_path, 'r', encoding="utf-8") as file:
                    lines = file.read().splitlines()
            if self.labels:
                lines = self.labels
            elif os.path.exists(text_to_path):
                with open(text_to_path, 'r', encoding="utf-8") as file:
                    lines = file.readlines()
            else:
                messagebox.showerror("错误", "未找到标签文件")
            all_objects = []
            if os.path.exists(txt_file) and lines:
                with open(txt_file, 'r', encoding="utf-8") as f:
                    objects = f.read().splitlines()
                    for o_index, object in enumerate(objects):
                        object_info = object.strip().split(' ')
                        x_center = float(object_info[1]) * img_width
                        y_center = float(object_info[2]) * img_height
                        xminVal = int(x_center - 0.5 * float(object_info[3]) * img_width)  # object_info列表中的元素都是字符串类型
                        yminVal = int(y_center - 0.5 * float(object_info[4]) * img_height)
                        xmaxVal = int(x_center + 0.5 * float(object_info[3]) * img_width)
                        ymaxVal = int(y_center + 0.5 * float(object_info[4]) * img_height)
                        cdata = {"label": lines[int(object_info[0])].strip(), "bbox": [xminVal, yminVal, xmaxVal, ymaxVal],
                                 "id": o_index}
                        all_objects.append(cdata)
            self.annotations = all_objects
            self.update_annotation_list()  # 更新标注列表
            self.status_var_right.set(f"已加载标注: {len(self.annotations)} 个")
    
    def remove_annotations_file(self, file_path):
        """
        移除所有标注删除文件
        """
        if os.path.exists(file_path):
            os.remove(file_path)

    def on_label_select(self, event):
        """处理标签选择事件"""
        self.current_label.set(self.label_combo.get())

    def export_pascal_voc(self):
        """导出为PASCAL VOC格式的XML文件"""

        def parse_xml(file_path):
            """
            解析XML文件并返回结构化数据
            参数:
                file_path (str): XML文件路径
            返回:
                dict: 包含解析结果的字典
            """
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return {"error": f"文件 '{file_path}' 不存在"}

            try:
                # 解析XML文件
                tree = ET.parse(file_path)
                root = tree.getroot()
                # 存储解析结果的字典
                result = {
                    "root_tag": root.tag,
                    "root_attributes": root.attrib,
                    "elements": []
                }

                # 递归遍历所有子元素
                def traverse(element, parent_path=""):
                    # 构建当前元素的完整路径
                    current_path = f"{parent_path}/{element.tag}" if parent_path else element.tag
                    # 创建元素字典
                    element_data = {
                        "tag": element.tag,
                        "attributes": element.attrib,
                        "text": element.text.strip() if element.text else None,
                        "children": []
                    }
                    # 递归处理子元素
                    for child in element:
                        element_data["children"].append(traverse(child, current_path))
                    return element_data

                result["elements"].append(traverse(root))
                return result
            except ET.ParseError as e:
                return {"error": f"XML解析错误: {e}"}
            except Exception as e:
                return {"error": f"处理过程中发生错误: {e}"}
        xml_path = os.path.splitext(self.image_path)[0] + ".xml"
        if not self.annotations:
            self.remove_annotations_file(xml_path)
            messagebox.showwarning("警告", "没有可导出的标注")
            return

        try:
            annotation = ET.Element("annotation")
            folder = ET.SubElement(annotation, "folder")
            # 获取文件所在目录的父目录路径
            parent_directory = os.path.dirname(self.image_path)
            # 从父目录路径中提取最后一个文件夹名
            folder.text = os.path.basename(parent_directory)

            filename = ET.SubElement(annotation, "filename")
            filename.text = os.path.basename(self.image_path)

            source = ET.SubElement(annotation, "source")
            database = ET.SubElement(source, "database")
            database.text = "Custom software"
            annotation_source = ET.SubElement(source, "annotation")
            annotation_source.text = "PASCAL VOC2025"
            image = ET.SubElement(source, "path")
            image.text = str(self.image_path)

            owner = ET.SubElement(annotation, "owner")
            name = ET.SubElement(owner, "name")
            name.text = "jpj"

            size = ET.SubElement(annotation, "size")
            width = ET.SubElement(size, "width")
            width.text = str(self.original_image.width)
            height = ET.SubElement(size, "height")
            height.text = str(self.original_image.height)
            depth = ET.SubElement(size, "depth")
            depth.text = "3"

            segmented = ET.SubElement(annotation, "segmented")
            segmented.text = "0"

            for i, ann in enumerate(self.annotations):
                obj = ET.SubElement(annotation, "object")

                name = ET.SubElement(obj, "name")
                name.text = ann["label"]

                pose = ET.SubElement(obj, "pose")
                pose.text = "Unspecified"

                truncated = ET.SubElement(obj, "truncated")
                truncated.text = "0"

                difficult = ET.SubElement(obj, "difficult")
                difficult.text = "0"

                bndbox = ET.SubElement(obj, "bndbox")
                xmin = ET.SubElement(bndbox, "xmin")
                xmin.text = str(ann["bbox"][0])
                ymin = ET.SubElement(bndbox, "ymin")
                ymin.text = str(ann["bbox"][1])
                xmax = ET.SubElement(bndbox, "xmax")
                xmax.text = str(ann["bbox"][2])
                ymax = ET.SubElement(bndbox, "ymax")
                ymax.text = str(ann["bbox"][3])

            rough_string = ET.tostring(annotation, encoding='utf-8')
            reparsed = minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent="  ")
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write(pretty_xml)

            self.modified = False
            self.status_var.set(f"PASCAL VOC格式已导出: {os.path.basename(xml_path)}")
        except Exception as e:
            messagebox.showerror("错误", f"导出PASCAL VOC格式失败: {str(e)}")

    def export_yolo(self):
        """导出为YOLO格式 - 新增方法"""
        txt_path = os.path.splitext(self.image_path)[0] + ".txt"
        if not self.annotations:
            self.remove_annotations_file(txt_path)
            messagebox.showwarning("警告", "没有可导出的标注")
            return

        try:
            # 获取图像尺寸
            img_width, img_height = self.original_image.size

            # 创建类别映射
            unique_labels = list(set(ann["label"] for ann in self.annotations))
            label_to_id = {label: self.labels.index(label) for label in unique_labels}

            # 生成YOLO格式内容
            yolo_lines = []
            for ann in self.annotations:
                x1, y1, x2, y2 = ann["bbox"]

                # 计算归一化坐标 [6](@ref)
                x_center = (x1 + x2) / 2.0 / img_width
                y_center = (y1 + y2) / 2.0 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                class_id = label_to_id[ann["label"]]
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # 写入YOLO格式文件
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))

            # 同时生成数据集配置文件（可选）
            self.generate_yolo_dataset_config()

            self.modified = False
            self.status_var.set(f"YOLO格式已导出: {os.path.basename(txt_path)}")

        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {str(e)}")

    def generate_yolo_dataset_config(self):
        """生成YOLO数据集配置文件（可选功能）"""
        text_to_path = os.path.join(os.path.dirname(self.image_path), "classes.txt")
        labels_judge = True
        if os.path.isfile(text_to_path):
            with (open(text_to_path, "r", encoding="utf-8") as f):
                labels = f.read().splitlines()
                if labels == self.labels:
                    labels_judge = False
        if labels_judge:
            try:
                with open(text_to_path, "w", encoding="utf-8") as f:
                    for i in self.labels:
                        f.write(i + "\n")

                # 创建数据集配置文件 [6](@ref)
                dataset_config = {
                    "path": os.path.dirname(self.image_path),
                    "train": "images/train",
                    "val": "images/val",
                    "test": "images/test",
                    "nc": len(self.labels),
                    "names": {i: label for i, label in enumerate(self.labels)}
                }

                yaml_path = os.path.join(os.path.dirname(self.image_path), "dataset.yaml")
                with open(yaml_path, "w", encoding="utf-8") as f:
                    f.write(f"path: {dataset_config['path']}\n")
                    f.write(f"train: {dataset_config['train']}\n")
                    f.write(f"val: {dataset_config['val']}\n")
                    f.write(f"test: {dataset_config['test']}\n")
                    f.write(f"nc: {dataset_config['nc']}\n")
                    f.write("names:\n")
                    for i, label in enumerate(self.labels):
                        f.write(f"  {i}: {label}\n")

                self.status_var.set("YOLO数据集配置文件已生成")
            except Exception as e:
                # 配置文件生成失败不影响主功能
                print(f"生成YOLO配置文件失败: {str(e)}")


class HomePage(ttk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.gif_path = "./welcome_animation.gif"
        self.is_playing = True
        self.playback_speed = 1.0  # 播放速度倍数

        self.setup_ui()
        self.load_gif()

    def setup_ui(self):
        """设置用户界面"""
        main_frame = ttk.Frame(self, padding=10, relief="solid")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        # 欢迎文字
        welcome_label = ttk.Label(main_frame, text="欢迎使用多功能应用系统", font=("微软雅黑", 24, "bold"),
                                  foreground='darkblue')
        welcome_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # ttk.Button(main_frame, text="访问推理", command=lambda: self.controller.show_frame("PredictPage")).pack(
        #     side=tk.LEFT, pady=10)
        # GIF显示区域
        self.gif_label = tk.Label(main_frame, bg='white')
        self.gif_label.pack(side=tk.TOP, fill=tk.X, expand=True)

        # 控制按钮框架
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, expand=True, padx=50)

        # 控制按钮
        self.play_button = ttk.Button(control_frame, text="暂停", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        ttk.Button(control_frame, text="减速", command=self.slow_down).pack(side=tk.LEFT, expand=True, fill=tk.X,
                                                                            padx=5)
        ttk.Button(control_frame, text="加速", command=self.speed_up).pack(side=tk.LEFT, expand=True, fill=tk.X,
                                                                           padx=5)

    def load_gif(self):
        """加载GIF文件"""
        try:
            # gif_data = base64.b64decode(self.gif_path)
            # gif_buffer = io.BytesIO(gif_data)
            # 获取画布尺寸
            # new_width = self.winfo_width()
            # new_height = self.winfo_height()
            # print(new_width, new_height)
            self.gif_image = Image.open(self.gif_path, 'r')
            # self.gif_image = gif_image.resize((new_width, new_height), Image.LANCZOS)
            self.frames = []
            self.delays = []

            # 提取所有帧
            try:
                frame_idx = 0
                while True:
                    frame = self.gif_image.copy().convert('RGBA')
                    photo = ImageTk.PhotoImage(frame)
                    self.frames.append(photo)

                    # 获取延迟时间
                    delay = self.gif_image.info.get('duration', 100)
                    self.delays.append(delay)

                    frame_idx += 1
                    self.gif_image.seek(frame_idx)
            except EOFError:
                pass
            self.current_frame_index = 0
            self.animate()

        except Exception as e:
            error_label = tk.Label(self, text=f"加载动画失败: {e}")
            error_label.pack()

    def animate(self):
        """动画播放控制"""
        if not hasattr(self, 'frames') or not self.frames:
            return

        if self.is_playing:
            # 显示当前帧
            self.gif_label.configure(image=self.frames[self.current_frame_index])
            # 更新帧索引
            self.current_frame_index = (self.current_frame_index + 1) % len(self.frames)

        # 计算下一帧延迟（考虑播放速度）
        delay = self.delays[self.current_frame_index] if self.delays else 100
        adjusted_delay = int(delay / self.playback_speed)

        self.after(adjusted_delay, self.animate)

    def toggle_play(self):
        """切换播放/暂停状态"""
        self.is_playing = not self.is_playing
        self.play_button.config(text="播放" if not self.is_playing else "暂停")

    def speed_up(self):
        """加快播放速度"""
        self.playback_speed = min(3.0, self.playback_speed + 0.5)

    def slow_down(self):
        """减慢播放速度"""
        self.playback_speed = max(0.5, self.playback_speed - 0.5)


class BasePage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.configure(padding=10)

        # 配置页面的行列权重
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def reset(self):
        """基础重置方法，子类可以覆盖"""
        pass


class AboutPage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        # 添加关于信息
        ttk.Label(self, text="关于程序", font=("Arial", 18)).pack(pady=20)
        ttk.Label(self, text="版本: 1.0.0", font=("Arial", 12)).pack(pady=5)
        ttk.Label(self, text="作者: jpj", font=("Arial", 12)).pack(pady=5)
        ttk.Label(self, text="© 2025 版权所有", font=("Arial", 10)).pack(pady=20)


class PredictPage(BasePage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.current_frame_name = ""
        # 创建主框架
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧单选框区域
        radio_frame = ttk.LabelFrame(main_frame, text="选择配置", padding="10 5")
        radio_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        self.current_option = {
            "文件处理": FileProcessingFrame,
        }
        self.selector_var = tk.StringVar()
        ttk.Label(radio_frame, text="请选择：").pack(side=tk.LEFT, fill=tk.X)
        selector_combobox = ttk.Combobox(radio_frame, textvariable=self.selector_var,
                                         values=list(self.current_option.keys()), state="readonly")
        selector_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        selector_combobox.set(list(self.current_option.keys())[0])
        selector_combobox.bind("<<ComboboxSelected>>", self.on_select)

        # 右侧内容显示区域
        self.content_frame = ttk.LabelFrame(main_frame, text="文件配置详情", padding="15")
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.show_frame(list(self.current_option.keys())[0])

    def create_frames(self):
        """创建各个界面容器"""
        # 存储所有页面的字典
        self.frames = {}
        for widget in self.content_frame.winfo_children():
            widget.destroy()  # 清除所有子组件

        # 创建所有页面
        for frame_name, frame_class in self.current_option.items():
            frame = frame_class(self.content_frame)
            self.frames[frame_name] = frame
            frame.pack()

    def show_frame(self, frame_name):
        """显示指定界面"""
        self.create_frames()
        for frame in self.frames.values():
            frame.pack_forget()  # 隐藏页面
        self.frames[frame_name].pack(fill=tk.BOTH, expand=True)
        self.frames[frame_name].update()  # 更新页面内容

    def on_select(self, event):
        if self.current_frame_name == event.widget.get():
            return
        self.create_frames()
        frame_name = event.widget.get()
        if frame_name not in self.frames:
            frame_name = list(self.current_option.items())[0][0]
        for frame in self.frames.values():
            frame.pack_forget()  # 隐藏页面
        name = str(event.widget.get()).split('处理')[0]
        self.content_frame.config(text=f"{name}配置详情")
        self.current_frame_name = frame_name
        self.frames[frame_name].tkraise()  # 将指定页面置于顶层
        # self.frames[frame_name].lift()  # 将指定页面置于顶层
        self.frames[frame_name].pack(fill="both", expand=True)
        self.frames[frame_name].update()  # 更新页面内容


class FileProcessingFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()
        self.labels = []

    def create_widgets(self):
        def validate_input(new_value):
            """验证输入是否为0-1范围内的两位小数"""
            if new_value == "":
                return True
            try:
                num = float(new_value)
                if 0 <= num <= 1:
                    if len(new_value) > 1 and new_value[1] != '.':
                        return False
                    else:
                        # 确保小数位不超过两位
                        if '.' in new_value:
                            parts = new_value.split('.')
                            if len(parts[1]) > 2:
                                return False
                    return True
            except ValueError:
                return False
            return False

        frame1 = ttk.Frame(self)
        frame1.pack(fill=tk.X, pady=5)
        ttk.Label(frame1, text="模型文件地址:", width=15).pack(side=tk.LEFT, padx=5)
        self.file1_var = tk.StringVar()
        ttk.Entry(frame1, textvariable=self.file1_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(frame1, text="浏览", command=lambda: self.select_file(self.file1_var)).pack(side=tk.RIGHT, padx=5)

        # 文件选择框2的容器（独占一行）
        frame2 = ttk.Frame(self)
        frame2.pack(fill=tk.X, pady=5)
        ttk.Label(frame2, text="素材文件夹地址:", width=15).pack(side=tk.LEFT, padx=5)
        self.file2_var = tk.StringVar()
        ttk.Entry(frame2, textvariable=self.file2_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(frame2, text="浏览文件", command=lambda: self.select_file(self.file2_var)).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame2, text="浏览文件夹", command=lambda: self.select_folder(self.file2_var)).pack(side=tk.RIGHT,
                                                                                                       padx=5)

        frame3 = ttk.Frame(self)
        frame3.pack(fill=tk.X, pady=5)
        # 创建带验证的Entry
        self.conf_vcmd = (self.register(validate_input), '%P')
        ttk.Label(frame3, text="conf:", width=15).pack(side=tk.LEFT, padx=5)
        self.conf_entry = ttk.Entry(frame3, validate="key", validatecommand=self.conf_vcmd)
        self.conf_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.conf_entry.insert(0, "0.50")
        # 创建带验证的Entry
        frame4 = ttk.Frame(self)
        frame4.pack(fill=tk.X, pady=5)
        self.iou_vcmd = (self.register(validate_input), '%P')
        ttk.Label(frame4, text="iou:", width=15).pack(side=tk.LEFT, padx=5)
        self.iou_entry = ttk.Entry(frame4, validate="key", validatecommand=self.iou_vcmd)
        self.iou_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.iou_entry.insert(0, "0.50")
        # 保存格式
        frame5 = ttk.Frame(self)
        frame5.pack(fill=tk.X, pady=5)
        ttk.Label(frame5, text="保存格式:", width=15).pack(side=tk.LEFT, padx=5)
        self.method_var = tk.StringVar()
        method_combobox = ttk.Combobox(frame5, textvariable=self.method_var, values=["VOC", "TXT"])
        method_combobox.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        method_combobox.current(0)
        # 提交按钮和进度条容器
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, pady=10)
        self.submit_btn = ttk.Button(btn_frame, text="提交", command=self.start_progress)
        self.submit_btn.pack(fill=tk.X)

        canvas_frame_bottom = ttk.LabelFrame(self, text="状态栏", relief=tk.SUNKEN, height=5)
        canvas_frame_bottom.pack(side=tk.BOTTOM, fill=tk.X)
        # 进度条容器（包含进度条和数字标签）
        progress_frame = ttk.Frame(canvas_frame_bottom)
        progress_frame.pack(fill=tk.X, pady=5)
        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.pack(side=tk.RIGHT)

        status_frame = ttk.Frame(canvas_frame_bottom)
        status_frame.pack(fill=tk.X, pady=5)
        self.status_var_data = tk.StringVar(value="就绪")
        status_bar = ttk.Label(status_frame, textvariable=self.status_var_data, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, expand=True)

    def select_file(self, var):
        file_path = filedialog.askopenfilename()
        if file_path:
            var.set(file_path)

    def select_folder(self, var):
        folder_path = filedialog.askdirectory(title="选择文件夹")
        if folder_path:
            var.set(folder_path)

    def start_progress(self):
        self.file1_var_data = self.file1_var.get()
        self.file2_var_data = self.file2_var.get()
        self.conf_vcmd_data = self.conf_entry.get()
        self.iou_vcmd_data = self.iou_entry.get()
        self.method_var_data = self.method_var.get()
        self.progress["value"] = 0
        self.progress_label.config(text="0%")
        if self.file1_var_data and self.file2_var_data and self.conf_vcmd_data and self.iou_vcmd_data:
            self.status_var_data.set("开始处理文件...")
            if not os.path.exists(self.file2_var_data):
                messagebox.showwarning("警告", "素材文件夹不存在！")
                return  # 目标文件夹不存在，直接返回
            if os.path.isfile(self.file2_var_data):
                self.image_paths = [self.file2_var_data]  # 如果是文件，则将其作为单个图像路径
            else:
                extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
                self.image_paths = sorted(
                    [os.path.join(self.file2_var_data, str(ext)) for ext in os.listdir(self.file2_var_data) if
                     str(os.path.splitext(ext)[1]).lower() in extensions])  # 素材列表
            # 模拟文件处理过程
            self.progress['maximum'] = len(self.image_paths)  # 设置最大值
            self.submit_btn.config(state=tk.DISABLED)  # 禁用提交按钮
            self.update_progress()
        else:
            messagebox.showwarning("警告", "请填写所有必填项！")
            self.status_var_data.set("就绪")
            return

    def save_boxes_to_txt(self, results, txt_path):
        """
        直接从boxes对象保存检测结果为TXT文件（更直接的方法）
        """
        if results.boxes is not None and len(results.boxes) > 0:
            with open(txt_path, 'w') as f:
                # 获取归一化坐标
                boxes_xywhn = results.boxes.xywhn.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy().astype(int)
                for i in range(len(results.boxes)):
                    class_id = class_ids[i]
                    x_center, y_center, width, height = boxes_xywhn[i]
                    # 写入YOLO标准格式
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def save_boxes_to_xml(self, xml_path, img_name, img_path, results, class_names, orig_img_size):
        """
        创建PASCAL VOC格式的XML标注文件 [1,6](@ref)

        Args:
            xml_path: XML文件保存路径
            img_name: 图像文件名
            img_path: 图像完整路径
            results: YOLO预测结果
            class_names: 类别名称列表
            orig_img_size: 原始图像尺寸
        """

        def prettify_xml(elem):
            """
            美化XML输出格式
            Args:
                elem: XML元素
            Returns:
                格式化后的XML字符串
            """
            rough_string = ET.tostring(elem, 'utf-8')
            reparsed = minidom.parseString(rough_string)
            return reparsed.toprettyxml(indent="  ", encoding="utf-8").decode('utf-8')

        if results.boxes is not None and hasattr(results, 'boxes'):
            # 创建根节点
            annotation = ET.Element('annotation')

            # 添加文件夹信息
            folder = ET.SubElement(annotation, 'folder')
            folder.text = os.path.dirname(img_path) or 'images'

            # 添加文件名
            filename = ET.SubElement(annotation, 'filename')
            filename.text = img_name

            # 添加路径
            path = ET.SubElement(annotation, 'path')
            path.text = img_path

            # 添加源信息
            source = ET.SubElement(annotation, 'source')
            database = ET.SubElement(source, 'database')
            database.text = 'Unknown'

            # 添加图像尺寸信息
            size = ET.SubElement(annotation, 'size')
            width_elem = ET.SubElement(size, 'width')
            width_elem.text = str(orig_img_size[1])
            height_elem = ET.SubElement(size, 'height')
            height_elem.text = str(orig_img_size[0])
            depth_elem = ET.SubElement(size, 'depth')
            depth_elem.text = str(orig_img_size[2])  # 假设为RGB图像
            # 添加分割信息
            segmented = ET.SubElement(annotation, 'segmented')
            segmented.text = '0'

            # 处理检测结果 [2,3](@ref)
            boxes_data = results.boxes.data.cpu().numpy()
            for i in range(len(boxes_data)):
                # 解析边界框数据 [x1, y1, x2, y2, conf, class_id]
                if len(boxes_data[i]) >= 6:
                    x1, y1, x2, y2, conf, class_id = boxes_data[i][:6]
                    # 转换为整数
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    class_id = int(class_id)
                    # 获取类别名称
                    class_name = class_names[class_id]
                    # 只保留置信度大于0的检测结果
                    if conf > 0:
                        # 创建对象标注
                        obj = ET.SubElement(annotation, 'object')
                        # 添加类别信息
                        name = ET.SubElement(obj, 'name')
                        name.text = class_name
                        # 添加姿态信息
                        pose = ET.SubElement(obj, 'pose')
                        pose.text = 'Unspecified'
                        # 添加截断信息
                        truncated = ET.SubElement(obj, 'truncated')
                        truncated.text = '0'
                        # 添加难度信息
                        difficult = ET.SubElement(obj, 'difficult')
                        difficult.text = '0'
                        # 添加置信度
                        confidence = ET.SubElement(obj, 'confidence')
                        confidence.text = f'{conf:.4f}'
                        # 添加边界框
                        bndbox = ET.SubElement(obj, 'bndbox')
                        xmin = ET.SubElement(bndbox, 'xmin')
                        xmin.text = str(x1)
                        ymin = ET.SubElement(bndbox, 'ymin')
                        ymin.text = str(y1)
                        xmax = ET.SubElement(bndbox, 'xmax')
                        xmax.text = str(x2)
                        ymax = ET.SubElement(bndbox, 'ymax')
                        ymax.text = str(y2)

            # 美化XML格式并保存
            xml_str = prettify_xml(annotation)
            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(xml_str)

    def predict_mode(self, source_path, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        使用YOLO模型进行预测，并手动保存TXT格式的检测结果

        参数:
            source_path: 输入源路径（可以是图片、视频、目录等）
            model_path: YOLO模型路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
        """
        # 加载YOLO模型
        model = YOLO(model_path)
        if self.method_var_data == "TXT":
            class_id_name = list(model.names.values())
            if os.path.isdir(source_path):
                text_to_path = os.path.join(source_path, "classes.txt")
            else:
                text_to_path = os.path.join(os.path.dirname(source_path), "classes.txt")
            if os.path.isfile(text_to_path) and not self.labels:
                with (open(text_to_path, "r", encoding="utf-8") as f):
                    self.labels = f.read().splitlines()
            if self.labels != class_id_name:
                with (open(text_to_path, "w", encoding="utf-8") as f):
                    for label in class_id_name:
                        f.write(label + "\n")
        # 执行预测（不自动保存TXT，我们手动处理）
        results = model.predict(
            source=source_path,
            conf=conf_threshold,
            iou=iou_threshold,
            save_txt=False,  # 重要：不自动保存TXT，我们手动处理
            save_conf=False  # 在手动保存时包含置信度
        )
        try:
            # 处理每个检测结果
            for i, result in enumerate(results):
                original_path = result.path
                filename = os.path.basename(original_path)
                img_name_without_ext = os.path.splitext(filename)[0]
                labels_dir = os.path.dirname(original_path)
                # 获取图像尺寸
                if hasattr(result, 'orig_img'):
                    orig_img_size = result.orig_img.shape
                else:
                    # 如果无法直接从结果获取，则读取图像
                    img = cv2.imdecode(np.fromfile(original_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                    orig_img_size = img.shape
                # 构建TXT文件路径

                if self.method_var_data == "TXT":
                    txt_path = os.path.join(labels_dir, f"{img_name_without_ext}.txt")
                    # 获取检测结果的DataFrame格式
                    self.save_boxes_to_txt(result, txt_path)
                elif self.method_var_data == "VOC":
                    xml_path = os.path.join(labels_dir, f"{img_name_without_ext}.xml")
                    # 获取检测结果的DataFrame格式
                    self.save_boxes_to_xml(xml_path, filename, original_path, result, result.names, orig_img_size)
                self.progress["value"] += 1
                self.progress_label.config(text=f"{int((self.progress['value'] / self.progress['maximum']) * 100)}%")
                self.update()
            return results
        except Exception as e:
            messagebox.showerror("错误", f"处理模型时发生错误: {e}")
            return None

    def update_progress(self):
        """
        更新进度条
        """
        if self.image_paths:
            for i_path in self.image_paths:
                predict_result = self.predict_mode(i_path, model_path=str(self.file1_var_data),
                                                   conf_threshold=float(self.conf_vcmd_data),
                                                   iou_threshold=float(self.iou_vcmd_data))
                if not predict_result:
                    self.status_var_data.set("处理失败")
                    break
            self.status_var_data.set("处理完成")
        else:
            self.status_var_data.set("选择文件没有素材")
        self.submit_btn.config(state=tk.NORMAL)  # 处理完成后启用提交按钮)


class Application(object):
    def __init__(self, root):
        self.frames = None
        self.root = root
        self.root.title("多功能工具箱")
        self.root.geometry("1400x900")
        # 跟踪全屏状态
        self.is_fullscreen = False
        # 创建主框架容器
        self.container = ttk.Frame(self.root)
        self.container.pack(fill=tk.BOTH, expand=True)
        self.current_frame_name = ""
        # 创建所有页面
        self.frame_classes = {
            "HomePage": [HomePage, ""],
            "PredictPage": [PredictPage, "模型推理界面"],
            "LabelImages": [ImageAnnotator, "素材标注界面"],
            "AboutPage": [AboutPage, ""]
        }
        self.create_menu()
        # 默认显示主页
        self.show_frame(list(self.frame_classes.keys())[0])

    def create_menu(self):
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)
        # 创建"功能"菜单
        self.main_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="导航", menu=self.main_menu)

        self.main_menu.add_command(label="主页", command=lambda: self.show_frame("HomePage"))
        self.main_menu.add_command(label="推理", command=lambda: self.show_frame("PredictPage"))
        self.main_menu.add_command(label="标注", command=lambda: self.show_frame("LabelImages"))
        self.main_menu.add_separator()
        self.main_menu.add_command(label="退出", command=self.root.quit)
        self.root.protocol("WM_DELETE_WINDOW", self.root.quit)
        # 创建"视图"菜单
        self.view_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="视图", menu=self.view_menu)
        self.view_menu.add_command(label="刷新布局", command=self.refresh_layout)
        self.view_menu.add_command(label="全屏模式", command=self.toggle_fullscreen)
        # 创建"帮助"菜单
        self.help_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="帮助", menu=self.help_menu)
        self.help_menu.add_command(label="关于", command=lambda: self.show_frame("AboutPage"))
        self.help_menu.add_command(label="帮助文档", command=self.show_help)

    def create_frames(self):
        """创建各个界面容器"""
        # 存储所有页面的字典
        self.frames = {}
        for widget in self.container.winfo_children():
            widget.destroy()  # 清除所有子组件

        for frame_name, frame_class in self.frame_classes.items():
            frame = frame_class[0](parent=self.container, controller=self)
            self.frames[frame_name] = frame
            frame.pack()

    def switch_title(self, new_title):
        """切换窗口标题并更新内容区域"""
        # 更新窗口标题
        self.root.title(f"{new_title}")

    def show_frame(self, frame_name):
        # 避免重复切换同一界面
        if self.current_frame_name == frame_name:
            return
        self.create_frames()
        if frame_name not in self.frames:
            frame_name = list(self.frame_classes.items())[0][0]

        if self.frame_classes.get(frame_name)[1]:
            self.switch_title(self.frame_classes.get(frame_name)[1])
        else:
            self.switch_title("多功能工具箱")
        # 隐藏所有页面
        for frame in self.frames.values():
            frame.pack_forget()  # 隐藏页面

        # 显示指定页面
        self.frames[frame_name].tkraise()  # 将指定页面置于顶层
        # self.frames[frame_name].lift()  # 将指定页面置于顶层
        self.frames[frame_name].pack(fill="both", expand=True)
        self.frames[frame_name].update()  # 更新页面内容
        # 更新当前界面名称
        self.current_frame_name = frame_name

    def toggle_fullscreen(self):
        """切换全屏模式"""
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)

        # 更新菜单项文本
        if self.is_fullscreen:
            self.view_menu.entryconfig(1, label="取消全屏模式")
        else:
            self.view_menu.entryconfig(1, label="全屏模式")

    def refresh_layout(self):
        """刷新布局，强制重新计算组件大小"""
        self.root.update_idletasks()
        self.root.event_generate("<Configure>", width=self.root.winfo_width(), height=self.root.winfo_height())

    def show_help(self):
        """显示帮助文档"""
        help_text = "帮助文档内容\n" \
                    "1. 使用功能菜单切换不同界面\n" \
                    "2. 使用视图菜单调整界面布局\n" \
                    "3. 使用帮助菜单查看软件信息"
        messagebox.showinfo("帮助", help_text)


def main():
    root = tk.Tk()
    Application(root)
    root.mainloop()


if __name__ == "__main__":
    main()
###PATH:预标注桌面工具.py
