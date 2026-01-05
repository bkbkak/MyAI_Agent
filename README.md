# 本地多模态 AI 智能文献与图像管理助手

**Local Multimodal AI Agent for Papers & Images**

本项目是一个基于 **Python** 的本地多模态 AI 助手，利用文本与图像的深度语义表示（Embeddings），实现对本地 **科研文献（PDF）** 与 **图像素材** 的语义搜索、自动分类与高效管理。

> 目标：用「自然语言」取代「文件名 + 文件夹」，让本地资料像搜索引擎一样好用。

---

## 项目亮点 (Highlights)

* **语义级检索**：不依赖文件名，直接理解内容
* **多模态支持**：文本 + 图像统一向量空间
* **自动分类归档**：论文下载后自动整理
* **本地运行、低延迟**：无需云端 API
* **幂等设计**：可重复执行，不产生重复索引

---

## 1. 项目简介 (Project Introduction)

传统文件管理高度依赖文件名与人工分类，在科研场景下效率极低。本项目通过提取文献正文与图像内容的 **语义向量（Embeddings）**，构建本地向量索引，使用户可以：

* 用自然语言描述论文内容进行搜索
* 自动判断论文主题并分类存储
* 用一句话在本地图库中“以文搜图”

这是一个典型的 **多模态大模型落地应用（LLM / Foundation Model + 向量数据库）** 示例，适合科研人员、学生与工程实践者。

---

## 2. 核心功能 (Core Features)

### 2.1 📄 智能文献管理 (Paper Management)

#### 语义搜索

* 支持自然语言查询，如：

  * `"Transformer 架构图解"`
  * `"attention mechanism in vision"`
* 基于文献正文语义而非标题匹配，召回更精准

#### 自动分类与整理

* 运行 `add_paper` 或 `batch_organize`
* 系统自动读取 PDF 内容
* 根据预设主题（如 `Computer_Vision / NLP / Robotics`）分类
* 自动移动到对应子目录

#### 文件向量索引

* 使用 **ChromaDB** 作为本地向量数据库
* 支持持久化存储
* 秒级检索，避免全盘扫描

---

### 2.2 智能图像管理 (Image Management)

#### 以文搜图（Text-to-Image Search）

* 基于 **CLIP 多模态模型**
* 无需任何人工标签
* 示例查询：

  * `"sunset by the beach"`
  * `"a cat sitting on a laptop"`

#### 本地图库索引

* 扫描指定目录
* 提取图像 embedding
* 支持增量更新（已索引图片自动跳过）

---

## 3. 技术选型 (Technical Stack)

针对 **本地运行（尤其是 Intel Mac）** 的性能与易用性，采用如下组合：

| 模块     | 技术                 | 说明                          |
| ------ | ------------------ | --------------------------- |
| 文本模型   | `all-MiniLM-L6-v2` | SentenceTransformer，速度快、占用小 |
| 多模态模型  | `CLIP ViT-B/32`    | 经典图文对齐模型                    |
| 向量数据库  | `ChromaDB`         | 嵌入式、支持持久化                   |
| PDF 解析 | `pypdf`            | 轻量稳定                        |
| 图像处理   | `Pillow`           | 本地图像读取                      |
| 深度学习   | `PyTorch`          | 模型推理后端                      |

---

## 4. 环境要求 (Environment Requirements)

* **操作系统**：macOS / Windows / Linux

  * 推荐：macOS 12+
* **Python 版本**：3.8+

  * 推荐：3.10 / 3.11
* **硬件建议**：

  * 内存：≥ 8GB
  * CPU 即可运行（无需 GPU）

---

## 5. 安装指南 (Installation)

在项目根目录下执行：

```bash
pip install chromadb sentence-transformers torch pypdf pillow "numpy<2.0"
```

---

## 6. 使用说明 (Usage Guide)

项目通过 `main.py` 作为统一入口，所有功能均通过 **命令行参数** 调用。

---

### 6.1 📂 批量整理与去重 (Batch Organize)

扫描指定目录下的所有 PDF 文件：

* 自动解析内容
* 建立向量索引
* 根据主题分类
* 移动到 `library/papers/<topic>/`

```bash
python main.py batch_organize downloads --topics "Computer_Vision,NLP,Robotics"
```

支持重复执行（幂等设计，不会重复入库）

---

### 6.2 ➕ 添加单篇论文 (Add Single Paper)

适合日常新论文下载后的快速入库：

```bash
python main.py add_paper test3.pdf --topics "Computer_Vision,NLP,Robotics"
```

系统会：

* 自动分析内容
* 判断最相关主题
* 更新向量索引

---

### 6.3 🔎 自适应文献搜索 (Adaptive Paper Search)

使用自然语言进行语义检索：

```bash
python main.py search_paper "Transformer architecture"
```

🔧 特性说明：

* 使用 **动态相似度阈值**
* 自动决定返回文献数量
* 避免返回低相关“凑数”结果

---

### 6.4 图像索引与跨模态搜索 (Image Search)

#### 建立 / 更新图像索引

```bash
python main.py index_images test_images
```

* 自动扫描目录
* 提取 CLIP embedding
* 支持增量更新

#### 以文搜图

```bash
python main.py search_image "sunset by the beach"
```

建议使用英文描述，CLIP 对英文语义支持最佳。

---
