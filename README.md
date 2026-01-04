# 本地多模态 AI 智能文献与图像管理助手

# (Local Multimodal AI Agent)

本项目是一个基于 Python 的本地 AI 助手，利用多模态神经网络技术实现对本地文献和图像内容的**语义搜索**与**自动分类**。

---

## 1. 项目简介 (Project Introduction)

传统的文件管理依赖于文件名搜索，而本项目通过提取文件的深度特征（Embeddings），让用户可以用自然语言直接描述内容进行检索。本项目旨在解决海量科研文献与图像素材堆积、难以快速定位的问题，是一个典型的多模态大模型应用案例。

---

## 2. 核心功能 (Core Features)

### 2.1 智能文献管理

* **语义搜索**: 支持自然语言提问（如 "Transformer 架构图解"），系统通过理解内容返回最相关的 PDF 文件。
* **自动分类与整理**:
* 运行 `add_paper` 命令，系统将自动阅读 PDF，根据预设主题（如 CV, NLP, Robotics）将文件搬运至对应的子文件夹中。


* **文件索引**: 基于向量数据库（ChromaDB），实现秒级检索，告别全盘扫描。

### 2.2 智能图像管理

* **以文搜图**: 基于 OpenAI 的 CLIP 模型，支持通过描述（如 "海边的日落"）在本地图库中精准匹配图像，无需任何手动标签。

---

## 3. 技术选型 (Technical Stack)

为了在本地环境（特别是 **Intel Mac**）中保持高性能与低延迟，本项目采用了以下模块化方案：

* **文本模型**: `all-MiniLM-L6-v2` —— 极速、轻量的 SentenceTransformer。
* **多模态模型**: `CLIP (ViT-B-32)` —— 经典的图文对齐模型，用于图片理解。
* **向量库**: `ChromaDB` —— 开箱即用的嵌入式数据库，支持持久化存储。
* **解析工具**: `pypdf` 用于文献处理，`Pillow` 用于图像处理。

---

## 4. 环境要求 (Environment)

* **操作系统**: macOS (建议 macOS 12+), Windows, Linux
* **Python 版本**: 3.8 或以上 (推荐 3.10/3.11)
* **硬件建议**: 至少 8GB 内存

---

## 5. 使用说明 (Detailed Usage)

### 5.1 环境安装

在项目根目录下打开终端，运行：

```bash
pip install chromadb sentence-transformers torch pypdf pillow "numpy<2.0"

```

### 5.2 核心操作示例

本项目通过 `main.py` 统一入口调用：

* **添加并分类论文**：
```bash
python main.py add_paper "/path/to/paper.pdf" --topics "Computer_Vision,NLP,Robotics"

```


* **语义搜索文献**：
```bash
python main.py search_paper "What is deep learning?"

```


* **扫描并索引图片**（首次使用需运行）：
```bash
python main.py index_images "/your/image/folder"

```


* **以文搜图**：
```bash
python main.py search_image "a cute cat sitting on a chair"

```
