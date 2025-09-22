#!/usr/bin/env python3
"""
视频标注应用后端服务
提供视频文件访问、标注数据管理等功能
"""

import os
import csv
import json
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path

app = Flask(__name__)

# 配置
import os
VIDEO_DIR = os.path.abspath('../huggingface_datasets/mhi/videos')
ANNOTATION_FILE = 'annotation_data.csv'
STATIC_DIR = 'static'
TEMPLATE_DIR = 'templates'

def init_annotation_file():
    """初始化标注文件"""
    # 获取所有视频文件
    videos = get_video_files()
    
    # 如果CSV文件不存在，创建并初始化
    if not os.path.exists(ANNOTATION_FILE):
        with open(ANNOTATION_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['video_path', 'annotation', 'status', 'annotated_time'])
        
        # 添加所有视频路径，标记为未标注
        if videos:
            with open(ANNOTATION_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for video in videos:
                    writer.writerow([video, '', 'pending', ''])
    else:
        # 如果CSV文件已存在，检查是否有新的视频文件需要添加
        existing_videos = set()
        
        # 读取现有数据
        with open(ANNOTATION_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if row and len(row) > 0:
                    existing_videos.add(row[0])
        
        # 添加新视频文件
        new_videos = set(videos) - existing_videos
        if new_videos:
            with open(ANNOTATION_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for video in new_videos:
                    writer.writerow([video, '', 'pending', ''])

def get_video_files():
    """获取所有视频文件"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
    videos = []
    
    if os.path.exists(VIDEO_DIR):
        for root, dirs, files in os.walk(VIDEO_DIR):
            for file in files:
                if Path(file).suffix.lower() in video_extensions:
                    # 获取相对于VIDEO_DIR的路径
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, VIDEO_DIR)
                    videos.append(relative_path)
    
    return sorted(videos)

def load_annotations():
    """加载所有标注数据"""
    annotations = {}
    if os.path.exists(ANNOTATION_FILE):
        try:
            with open(ANNOTATION_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    annotations[row['video_path']] = row
        except Exception as e:
            print(f"读取标注文件出错: {e}")
    return annotations

def save_annotation(video_path, annotation):
    """保存标注数据"""
    # 读取现有数据
    annotations = load_annotations()
    
    # 更新或添加新标注
    annotations[video_path] = {
        'video_path': video_path,
        'annotation': annotation,
        'status': 'completed',
        'annotated_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 写入文件
    try:
        with open(ANNOTATION_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['video_path', 'annotation', 'status', 'annotated_time'])
            writer.writeheader()
            for row in annotations.values():
                writer.writerow(row)
        return True
    except Exception as e:
        print(f"保存标注数据出错: {e}")
        return False

@app.route('/')
def index():
    """主页"""
    return render_template('annotation.html')

@app.route('/api/videos')
def list_videos():
    """获取视频列表"""
    videos = get_video_files()
    annotations = load_annotations()
    
    # 添加日志：查看后端数据
    print(f"总视频数: {len(videos)}")
    print(f"已标注视频数: {len(annotations)}")
    
    # 组合视频和标注信息
    video_data = []
    for video in videos:
        # 检查视频是否已标注完成（status为completed）
        is_annotated = video in annotations and annotations[video].get('status') == 'completed'
        annotation_text = annotations.get(video, {}).get('annotation', '') if is_annotated else ''
        
        video_info = {
            'path': video,
            'annotated': is_annotated,
            'annotation': annotation_text
        }
        video_data.append(video_info)
    
    return jsonify(video_data)

@app.route('/api/annotation', methods=['GET', 'POST'])
def handle_annotation():
    """处理标注数据"""
    if request.method == 'POST':
        data = request.get_json()
        video_path = data.get('video_path')
        annotation = data.get('annotation')
        
        if not video_path or not annotation:
            return jsonify({'error': '缺少必要参数'}), 400
            
        if save_annotation(video_path, annotation):
            return jsonify({'success': True})
        else:
            return jsonify({'error': '保存失败'}), 500
    
    elif request.method == 'GET':
        annotations = load_annotations()
        return jsonify(annotations)

@app.route('/videos/<path:filename>')
def serve_video(filename):
    """提供视频文件访问"""
    # 处理URL编码的文件名
    import urllib.parse
    decoded_filename = urllib.parse.unquote(filename)
    
    # 处理相对路径
    full_video_path = os.path.join(VIDEO_DIR, decoded_filename)
    directory = os.path.dirname(full_video_path)
    file_name = os.path.basename(full_video_path)
    return send_from_directory(directory, file_name)

@app.route('/api/next_unannotated')
def next_unannotated():
    """获取下一个未标注的视频"""
    videos = get_video_files()
    annotations = load_annotations()
    
    # 查找第一个未标注的视频（status不为completed的视频）
    for video in videos:
        # 检查视频是否未标注完成
        if video not in annotations or annotations[video].get('status') != 'completed':
            return jsonify({'video_path': video})
    
    # 如果所有视频都已标注，返回第一个视频
    if videos:
        return jsonify({'video_path': videos[0]})
    
    return jsonify({'video_path': None})

@app.route('/api/annotation_suggestions')
def annotation_suggestions():
    """获取已有标注类别的建议"""
    annotations = load_annotations()
    
    # 提取所有非空的标注，并按;分割成多个标注单元
    suggestions = set()
    for data in annotations.values():
        annotation = data.get('annotation', '').strip()
        if annotation:
            # 按;分割标注内容，去除每个标注单元的前后空格
            units = [unit.strip() for unit in annotation.split(';') if unit.strip()]
            suggestions.update(units)
    
    # 转换为列表并排序
    suggestions = sorted(list(suggestions))
    return jsonify({'suggestions': suggestions})

if __name__ == '__main__':
    # 初始化标注文件
    init_annotation_file()
    
    # 启动应用
    app.run(host='0.0.0.0', port=5001, debug=True)