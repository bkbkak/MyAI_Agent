# 本地多模态 AI 助手

这是一个基于 Python 的本地文件管理助手。

## 功能
1. **语义搜索论文**: 只要问它内容，它就能找到文件。
2. **自动分类**: 根据主题自动整理 PDF。
3. **以文搜图**: 描述画面，找到对应的图片。

## 安装
pip install chromadb sentence-transformers torch pypdf pillow

## 使用方法
1. 添加论文: `python main.py add_paper "/path/to/paper.pdf" --topics "AI,History"`
2. 搜索论文: `python main.py search_paper "Transformer architecture"`
3. 索引图片: `python main.py index_images "/path/to/your/photos"`
4. 搜图: `python main.py search_image "cat on the sofa"`