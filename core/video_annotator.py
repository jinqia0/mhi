import streamlit as st
import pandas as pd
import os
import csv
from pathlib import Path
import streamlit.components.v1 as components

# 页面配置
st.set_page_config(
    page_title="视频人物交互标注",
    page_icon="🎥",
    layout="wide"
)

# 标注者配置
ANNOTATORS = {
    'annotator_1': 'manual_annotation',  # 第一轮标注者
    'annotator_2': 'manual_annotation_2',  # 第二轮标注者
    'annotator_3': 'manual_annotation_3',  # 第三轮标注者（可选）
}

# 数据源配置
DATA_SOURCES = {
    'OpenHumanVid': {
        'input_csv': 'results/openhv_1k_yolo_filtered.csv',
        'output_csv': 'results/contact_stats_main_ohv.csv',
        'video_dir': 'Dataset/OpenHumanVid',
        'path_column': 'path',
        'video_extension': '.mp4'
    },
    'Panda': {
        'input_csv': 'results/contact_stats_main.csv',
        'output_csv': 'results/contact_stats_main_panda_manual.csv',
        'video_dir': '',
        'path_column': 'video',
        'video_extension': ''  # 已包含在路径中
    }
}

def load_input_data(data_source):
    """加载输入CSV文件"""
    input_csv = DATA_SOURCES[data_source]['input_csv']
    try:
        return pd.read_csv(input_csv)
    except FileNotFoundError:
        st.error(f"找不到输入文件: {input_csv}")
        return pd.DataFrame()

def load_existing_annotations(data_source, annotator_column='manual_annotation'):
    """加载已有的手动标注结果"""
    output_csv = DATA_SOURCES[data_source]['output_csv']
    if os.path.exists(output_csv):
        try:
            # 检查文件是否为空或只有空行
            with open(output_csv, 'r') as f:
                content = f.read().strip()
            if not content:
                return {}
            
            # 尝试读取CSV
            try:
                df = pd.read_csv(output_csv)
                # 检查是否有指定的标注列
                path_column = DATA_SOURCES[data_source]['path_column']
                if annotator_column in df.columns:
                    # 返回手动标注结果
                    manual_annotations = {}
                    for _, row in df.iterrows():
                        if pd.notna(row.get(annotator_column, '')) and row.get(annotator_column, '') != '':
                            manual_annotations[row[path_column]] = row[annotator_column]
                    return manual_annotations
                elif path_column in df.columns and 'is_contact' in df.columns:
                    # 兼容旧格式
                    return dict(zip(df[path_column], df['is_contact']))
                else:
                    # 如果没有列名，假设列的顺序是固定的
                    if len(df.columns) >= 5:
                        # 根据实际列数设置列名
                        base_columns = [path_column, 'contact_frames', 'total_frames', 'contact_frame_ratio', 'is_contact']
                        if len(df.columns) == 6:
                            df.columns = base_columns + ['manual_annotation']
                        else:
                            df.columns = base_columns + list(df.columns[5:])
                        return dict(zip(df[path_column], df['is_contact']))
                    return {}
            except pd.errors.EmptyDataError:
                return {}
        except Exception as e:
            st.error(f"读取标注文件出错: {e}")
            return {}
    return {}

def save_annotation(video_path, manual_annotation, data_source, annotator_column='manual_annotation'):
    """保存手动标注结果"""
    output_csv = DATA_SOURCES[data_source]['output_csv']
    path_column = DATA_SOURCES[data_source]['path_column']
    # 检查输出文件是否存在，如果不存在则创建
    file_exists = os.path.exists(output_csv)
    
    # 如果文件存在，读取现有数据
    if file_exists:
        try:
            # 检查文件是否为空
            with open(output_csv, 'r') as f:
                content = f.read().strip()
            if not content:
                annotations = []
            else:
                df = pd.read_csv(output_csv)
                # 如果没有列名，添加列名
                if df.columns.tolist() == [str(i) for i in range(len(df.columns))]:
                    base_columns = [path_column, 'contact_frames', 'total_frames', 'contact_frame_ratio', 'is_contact']
                    if len(df.columns) == 6:
                        df.columns = base_columns + ['manual_annotation']
                    else:
                        df.columns = base_columns + list(df.columns[5:])
                elif path_column not in df.columns:
                    base_columns = [path_column, 'contact_frames', 'total_frames', 'contact_frame_ratio', 'is_contact']
                    if len(df.columns) == 6:
                        df.columns = base_columns + ['manual_annotation']
                    else:
                        df.columns = base_columns + list(df.columns[5:])
                annotations = df.to_dict('records')
        except Exception as e:
            st.error(f"读取现有标注文件出错: {e}")
            annotations = []
    else:
        annotations = []
    
    # 检查是否已有该视频的记录
    video_found = False
    for i, annotation in enumerate(annotations):
        if annotation.get(path_column) == video_path:
            # 添加指定的手动标注列，保留原有自动检测结果
            annotations[i][annotator_column] = manual_annotation
            video_found = True
            break
    
    # 如果没有找到，创建新记录（保留自动检测结果为空，只添加手动标注）
    if not video_found:
        new_annotation = {
            path_column: video_path,
            'contact_frames': '',  # 自动检测结果为空
            'total_frames': '',    # 自动检测结果为空
            'contact_frame_ratio': '',  # 自动检测结果为空
            'is_contact': '',      # 自动检测结果为空
            annotator_column: manual_annotation  # 手动标注结果
        }
        annotations.append(new_annotation)
    
    # 保存到CSV
    df = pd.DataFrame(annotations)
    # 确保列顺序，包含所有可能的标注者列
    base_columns = [path_column, 'contact_frames', 'total_frames', 'contact_frame_ratio', 'is_contact']
    annotator_columns = list(ANNOTATORS.values())  # 所有标注者列
    columns = base_columns + annotator_columns
    
    for col in columns:
        if col not in df.columns:
            df[col] = ''
    df = df[columns]
    df.to_csv(output_csv, index=False)

def add_keyboard_shortcuts():
    """添加键盘快捷键支持"""
    # JavaScript代码来处理键盘事件
    keyboard_js = """
    <script>
    function findAndClickButton(searchText) {
        // 先在当前页面查找
        let buttons = document.querySelectorAll('button');
        for (let button of buttons) {
            if (button.textContent && button.textContent.includes(searchText)) {
                console.log('Found button in current frame:', button.textContent);
                button.click();
                return true;
            }
        }
        
        // 在父页面查找
        try {
            buttons = parent.document.querySelectorAll('button');
            for (let button of buttons) {
                if (button.textContent && button.textContent.includes(searchText)) {
                    console.log('Found button in parent frame:', button.textContent);
                    button.click();
                    return true;
                }
            }
        } catch (e) {
            console.log('Cannot access parent document:', e);
        }
        
        // 在所有iframe中查找
        try {
            const iframes = parent.document.querySelectorAll('iframe');
            for (let iframe of iframes) {
                try {
                    const iframeButtons = iframe.contentDocument.querySelectorAll('button');
                    for (let button of iframeButtons) {
                        if (button.textContent && button.textContent.includes(searchText)) {
                            console.log('Found button in iframe:', button.textContent);
                            button.click();
                            return true;
                        }
                    }
                } catch (e) {
                    // Skip iframes we can't access
                }
            }
        } catch (e) {
            console.log('Cannot search in iframes:', e);
        }
        
        console.log('Button not found:', searchText);
        return false;
    }
    
    document.addEventListener('keydown', function(event) {
        // 阻止在输入框中触发快捷键
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA' || 
            event.target.contentEditable === 'true') {
            return;
        }
        
        console.log('Key pressed:', event.key);
        
        switch(event.key) {
            case '1':
                event.preventDefault();
                console.log('Triggering: 有交互');
                findAndClickButton('✅ 有交互');
                break;
            case '0':
                event.preventDefault();
                console.log('Triggering: 无交互');
                findAndClickButton('❌ 无交互');
                break;
            case 's':
            case 'S':
                event.preventDefault();
                console.log('Triggering: 跳过');
                findAndClickButton('⏭️ 跳过此视频');
                break;
        }
    });
    
    // 确保事件监听器正确绑定
    console.log('Event listener attached to document');
    
    console.log('Keyboard shortcuts initialized');
    </script>
    """
    
    # 嵌入JavaScript
    components.html(keyboard_js, height=0)

def main():
    st.title("🎥 视频人物交互标注工具")
    st.markdown("---")
    
    # 添加键盘快捷键支持
    add_keyboard_shortcuts()
    
    # 数据源选择
    st.header("📂 数据源选择")
    col1, col2 = st.columns(2)
    with col1:
        selected_source = st.selectbox(
            "选择要标注的数据源:",
            list(DATA_SOURCES.keys()),
            index=0,
            key="data_source_selector"
        )
    with col2:
        selected_annotator = st.selectbox(
            "选择标注者:",
            list(ANNOTATORS.keys()),
            index=1,  # 默认选择第二个标注者
            key="annotator_selector",
            format_func=lambda x: {
                'annotator_1': '标注者 1 (第一轮)',
                'annotator_2': '标注者 2 (第二轮)',
                'annotator_3': '标注者 3 (第三轮)'
            }.get(x, x)
        )
    
    # 获取当前标注者对应的列名
    annotator_column = ANNOTATORS[selected_annotator]
    
    # 显示当前数据源信息
    current_config = DATA_SOURCES[selected_source]
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**输入文件**: {current_config['input_csv']}")
        st.info(f"**视频目录**: {current_config['video_dir']}")
    with col2:
        st.info(f"**输出文件**: {current_config['output_csv']}")
        st.info(f"**路径列**: {current_config['path_column']}")
    
    st.markdown("---")
    
    # 加载数据
    input_df = load_input_data(selected_source)
    if input_df.empty:
        return
    
    # 检查是否需要强制刷新（在标注完成后）
    if 'force_refresh' in st.session_state and st.session_state.force_refresh:
        st.session_state.force_refresh = False
        # 清除缓存以确保重新加载
        if hasattr(st, 'cache_data'):
            st.cache_data.clear()
    
    existing_annotations = load_existing_annotations(selected_source, annotator_column)
    
    # 获取当前数据源配置
    config = DATA_SOURCES[selected_source]
    
    # 侧边栏 - 进度和控制
    with st.sidebar:
        st.header("📊 标注进度")
        total_videos = len(input_df)
        annotated_count = len(existing_annotations)
        st.metric("总视频数", total_videos)
        st.metric("已标注", annotated_count)
        st.metric("剩余", total_videos - annotated_count)
        
        if total_videos > 0:
            progress = min(annotated_count / total_videos, 1.0)  # 确保进度不超过1.0
            st.progress(progress)
            st.write(f"完成度: {progress:.1%}")
        
        st.markdown("---")
        st.header("🎯 标注说明")
        st.markdown(f"""
        **当前标注者：** {selected_annotator.replace('_', ' ').title()}
        
        **标注标准：**
        - ✅ **有交互**: 视频中人物之间有明显的身体接触、互动行为
        - ❌ **无交互**: 视频中人物独立行动，没有明显交互
        
        **⌨️ 键盘快捷键：**
        - 按 **1** = 有交互 ✅
        - 按 **0** = 无交互 ❌  
        - 按 **S** = 跳过视频 ⏭️
        """)
    
    # 主界面
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("🎬 视频标注")
        
        # 视频选择
        video_options = []
        unannotated_indices = []
        path_column = config['path_column']
        for idx, row in input_df.iterrows():
            video_path = row[path_column]
            is_annotated = video_path in existing_annotations
            status = "✅ 已标注" if is_annotated else "⏳ 待标注"
            video_options.append(f"{status} | {video_path}")
            if not is_annotated:
                unannotated_indices.append(idx)
        
        # 默认选择第一个未标注的视频
        default_index = 0
        if unannotated_indices:
            default_index = unannotated_indices[0]
        
        # 使用session state来跟踪当前选择的视频
        # 检查是否切换了标注者，如果切换了则重置索引
        if 'current_annotator' not in st.session_state:
            st.session_state.current_annotator = selected_annotator
        elif st.session_state.current_annotator != selected_annotator:
            st.session_state.current_annotator = selected_annotator
            st.session_state.current_video_index = default_index
        
        if 'current_video_index' not in st.session_state:
            st.session_state.current_video_index = default_index
        
        # 确保索引在有效范围内，并在标注状态改变后重新调整
        if st.session_state.current_video_index >= len(video_options):
            st.session_state.current_video_index = default_index
        
        # 检查当前选择的视频是否已经被标注，如果是则自动跳转到下一个未标注的视频
        current_video_path = input_df.iloc[st.session_state.current_video_index][path_column]
        if current_video_path in existing_annotations and unannotated_indices:
            # 如果当前视频已标注且还有未标注的视频，自动跳转
            st.session_state.current_video_index = unannotated_indices[0]
        
        selected_video_display = st.selectbox(
            "选择要标注的视频:",
            video_options,
            index=st.session_state.current_video_index,
            key="video_selector"
        )
        
        # 更新session state
        st.session_state.current_video_index = video_options.index(selected_video_display)
        
        if selected_video_display:
            # 提取实际的视频路径
            video_path = selected_video_display.split(" | ")[1]
            full_video_path = os.path.join(config['video_dir'], video_path + config['video_extension'])
            
            st.subheader(f"当前视频: {video_path}")
            
            # 显示视频信息
            if os.path.exists(full_video_path):
                # 显示视频，自动播放
                try:
                    with open(full_video_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                    st.video(video_bytes, autoplay=True)
                except Exception as e:
                    st.error(f"无法加载视频: {e}")
                    st.info(f"视频路径: {full_video_path}")
            else:
                st.error(f"视频文件不存在: {full_video_path}")
            
            # 显示当前标注状态
            current_annotation = existing_annotations.get(video_path, None)
            if current_annotation is not None:
                status_text = "有交互" if current_annotation == 1 else "无交互"
                st.info(f"当前标注状态: {status_text}")
    
    with col2:
        st.header("📝 标注操作")
        
        if 'selected_video_display' in locals() and selected_video_display:
            video_path = selected_video_display.split(" | ")[1]
            
            st.markdown("**请观看视频后选择:**")
            st.markdown("*💡 提示: 可使用键盘快捷键 1/0/S*")
            
            # 自动切换到下一个未标注视频的函数
            def go_to_next_video():
                current_idx = st.session_state.current_video_index
                # 重新加载当前标注者的标注结果
                current_existing_annotations = load_existing_annotations(selected_source, annotator_column)
                
                # 寻找下一个未标注的视频
                found_next = False
                for i in range(current_idx + 1, len(input_df)):
                    next_video_path = input_df.iloc[i][path_column]
                    if next_video_path not in current_existing_annotations:
                        st.session_state.current_video_index = i
                        found_next = True
                        break
                
                # 如果没有找到，从头开始寻找
                if not found_next:
                    for i in range(0, current_idx):
                        next_video_path = input_df.iloc[i][path_column]
                        if next_video_path not in current_existing_annotations:
                            st.session_state.current_video_index = i
                            found_next = True
                            break
                
                # 如果仍然没有找到，保持当前位置
                if not found_next:
                    st.info("所有视频已完成标注！")
                
                # 强制刷新页面以更新视频选择器
                st.session_state.force_refresh = True
            
            col_yes, col_no = st.columns(2)
            
            with col_yes:
                if st.button("✅ 有交互 (按1)", use_container_width=True, type="primary", key="btn_yes"):
                    save_annotation(video_path, 1, selected_source, annotator_column)
                    st.success("已标注为: 有交互")
                    go_to_next_video()
                    st.rerun()
            
            with col_no:
                if st.button("❌ 无交互 (按0)", use_container_width=True, key="btn_no"):
                    save_annotation(video_path, 0, selected_source, annotator_column)
                    st.success("已标注为: 无交互")
                    go_to_next_video()
                    st.rerun()
            
            st.markdown("---")
            
            # 快捷键提示
            st.markdown("**快捷操作:**")
            if st.button("⏭️ 跳过此视频 (按S)", use_container_width=True, key="btn_skip"):
                go_to_next_video()
                st.rerun()
            
            # 删除标注按钮
            if video_path in existing_annotations:
                if st.button("🗑️ 删除此标注", use_container_width=True, key="btn_delete"):
                    # 重新加载并删除对应记录
                    output_csv = config['output_csv']
                    if os.path.exists(output_csv):
                        try:
                            with open(output_csv, 'r') as f:
                                content = f.read().strip()
                            if content:
                                df = pd.read_csv(output_csv)
                                # 处理指定的标注列
                                if annotator_column in df.columns:
                                    df.loc[df[path_column] == video_path, annotator_column] = ''
                                else:
                                    # 兼容旧格式
                                    df = df[df[path_column] != video_path]
                                df.to_csv(output_csv, index=False)
                                st.success("标注已删除")
                                st.rerun()
                        except Exception as e:
                            st.error(f"删除标注时出错: {e}")
    
    # 底部 - 已标注结果预览
    st.markdown("---")
    st.header("📋 标注结果预览")
    
    output_csv = config['output_csv']
    if os.path.exists(output_csv):
        try:
            # 检查文件是否为空
            with open(output_csv, 'r') as f:
                content = f.read().strip()
            if not content:
                st.info("暂无标注结果")
            else:
                result_df = pd.read_csv(output_csv)
                # 如果没有列名，添加列名
                if result_df.columns.tolist() == [str(i) for i in range(len(result_df.columns))]:
                    base_columns = [path_column, 'contact_frames', 'total_frames', 'contact_frame_ratio', 'is_contact']
                    if len(result_df.columns) == 6:
                        result_df.columns = base_columns + ['manual_annotation']
                    else:
                        result_df.columns = base_columns + list(result_df.columns[5:])
                elif path_column not in result_df.columns:
                    base_columns = [path_column, 'contact_frames', 'total_frames', 'contact_frame_ratio', 'is_contact']
                    if len(result_df.columns) == 6:
                        result_df.columns = base_columns + ['manual_annotation']
                    else:
                        result_df.columns = base_columns + list(result_df.columns[5:])
                
                if not result_df.empty:
                    # 统计当前标注者的结果
                    if annotator_column in result_df.columns:
                        manual_df = result_df[result_df[annotator_column].notna() & (result_df[annotator_column] != '')]
                        total_manual = len(manual_df)
                        manual_contact = len(manual_df[manual_df[annotator_column] == 1])
                        manual_no_contact = len(manual_df[manual_df[annotator_column] == 0])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"{selected_annotator.replace('_', ' ').title()}标注数", total_manual)
                        with col2:
                            st.metric(f"{selected_annotator.replace('_', ' ').title()}-有交互", manual_contact)
                        with col3:
                            st.metric(f"{selected_annotator.replace('_', ' ').title()}-无交互", manual_no_contact)
                        
                        # 显示结果表格
                        display_df = result_df.copy()
                        
                        # 添加显示列
                        display_df['自动检测'] = display_df['is_contact'].apply(
                            lambda x: '有交互' if x == 1 else ('无交互' if x == 0 else '未检测')
                        )
                        
                        # 为所有可能的标注者添加显示列
                        display_columns = [path_column, '自动检测']
                        for annotator_key, column_name in ANNOTATORS.items():
                            if column_name in display_df.columns:
                                display_name = f"{annotator_key.replace('_', ' ').title()}"
                                display_df[display_name] = display_df[column_name].apply(
                                    lambda x: '有交互' if x == 1 else ('无交互' if x == 0 else '未标注')
                                )
                                display_columns.append(display_name)
                        
                        st.dataframe(
                            display_df[display_columns],
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        # 兼容旧格式
                        total_annotated = len(result_df)
                        has_contact = len(result_df[result_df['is_contact'] == 1])
                        no_contact = len(result_df[result_df['is_contact'] == 0])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("总标注数", total_annotated)
                        with col2:
                            st.metric("有交互", has_contact)
                        with col3:
                            st.metric("无交互", no_contact)
                        
                        # 显示结果表格
                        display_df = result_df.copy()
                        display_df['交互状态'] = display_df['is_contact'].apply(lambda x: '有交互' if x == 1 else '无交互')
                        st.dataframe(
                            display_df[[path_column, '交互状态']],
                            use_container_width=True,
                            hide_index=True
                        )
                else:
                    st.info("暂无标注结果")
        except Exception as e:
            st.error(f"读取结果文件出错: {e}")
            st.info("暂无标注结果")
    else:
        st.info("暂无标注结果")

if __name__ == "__main__":
    main()