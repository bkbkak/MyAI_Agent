import os
import argparse
import shutil
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid

# --- 配置部分 ---
# 这里设置数据存储在当前目录下的 db 文件夹
DB_PATH = os.path.join(os.getcwd(), "db")
PAPER_DIR = os.path.join(os.getcwd(), "library", "papers")
IMAGE_DIR = os.path.join(os.getcwd(), "library", "images")

# --- 核心功能类 ---
class AIAgent:
    def __init__(self):
        print("正在初始化 AI 模型 (首次运行可能需要下载模型，请耐心等待)...")
        # 1. 加载文本模型 (处理论文)
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        # 2. 加载图像模型 (处理图片)
        self.clip_model = SentenceTransformer('clip-ViT-B-32', device='cpu')
        
        # 3. 初始化数据库
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.paper_collection = self.client.get_or_create_collection("papers")
        self.image_collection = self.client.get_or_create_collection("images")
        
        # 确保文件夹存在
        os.makedirs(PAPER_DIR, exist_ok=True)
        os.makedirs(IMAGE_DIR, exist_ok=True)
        print("初始化完成！")

    def add_paper(self, file_path, topics=None):
        """添加论文并自动分类"""
        if not os.path.exists(file_path):
            print(f"错误：找不到文件 {file_path}")
            return

        print(f"正在读取文件: {file_path}")
        # 读取 PDF 文字
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages[:5]: # 只读前5页，加快速度
                text += page.extract_text() + "\n"
        except Exception as e:
            print(f"读取 PDF 失败: {e}")
            return

        # 简单的自动分类逻辑
        final_dir = PAPER_DIR
        if topics:
            topic_list = topics.split(',')
            # 计算文本和所有主题的相似度
            doc_emb = self.text_model.encode(text)
            topic_embs = self.text_model.encode(topic_list)
            
            from sentence_transformers import util
            scores = util.cos_sim(doc_emb, topic_embs)[0]
            best_topic = topic_list[scores.argmax()]
            
            final_dir = os.path.join(PAPER_DIR, best_topic)
            os.makedirs(final_dir, exist_ok=True)
            print(f"自动归类到: {best_topic}")

        # 移动文件
        filename = os.path.basename(file_path)
        new_path = os.path.join(final_dir, filename)
        shutil.copy(file_path, new_path) # 复制文件过去

        # 存入数据库
        embedding = self.text_model.encode(text).tolist()
        self.paper_collection.add(
            documents=[text[:500]], # 只存开头部分预览
            embeddings=[embedding],
            metadatas=[{"path": new_path, "filename": filename}],
            ids=[str(uuid.uuid4())]
        )
        print(f"✅ 论文 '{filename}' 已成功录入系统！")

    def search_paper(self, query):
        """搜索论文"""
        print(f"🔍 正在搜索: {query} ...")
        query_emb = self.text_model.encode(query).tolist()
        results = self.paper_collection.query(query_embeddings=[query_emb], n_results=3)
        
        for i, meta in enumerate(results['metadatas'][0]):
            print(f"[{i+1}] {meta['filename']}")
            print(f"    路径: {meta['path']}")

    def index_images(self, folder_path):
        """扫描并索引文件夹里的图片"""
        from PIL import Image
        print(f"正在扫描图片文件夹: {folder_path}")
        count = 0
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(root, file)
                    try:
                        img = Image.open(path)
                        emb = self.clip_model.encode(img).tolist()
                        self.image_collection.add(
                            embeddings=[emb],
                            metadatas=[{"path": path}],
                            ids=[str(uuid.uuid4())]
                        )
                        count += 1
                        print(f"已索引: {file}")
                    except:
                        pass
        print(f"✅ 完成！共索引 {count} 张图片。")

    def search_image(self, query):
        """以文搜图"""
        print(f"🖼️ 正在寻找图片: {query} ...")
        # CLIP 模型的特殊之处：用文本编码器搜图片嵌入
        query_emb = self.clip_model.encode(query).tolist()
        results = self.image_collection.query(query_embeddings=[query_emb], n_results=3)
        
        for i, meta in enumerate(results['metadatas'][0]):
            print(f"[{i+1}] 路径: {meta['path']}")

# --- 命令行入口 ---
def main():
    parser = argparse.ArgumentParser(description="我的 AI 助手")
    subparsers = parser.add_subparsers(dest='command')

    # 命令1: 添加论文
    p_add = subparsers.add_parser('add_paper')
    p_add.add_argument('path')
    p_add.add_argument('--topics')

    # 命令2: 搜索论文
    p_search = subparsers.add_parser('search_paper')
    p_search.add_argument('query')

    # 命令3: 索引图片
    p_idx = subparsers.add_parser('index_images')
    p_idx.add_argument('path')

    # 命令4: 搜图
    p_img = subparsers.add_parser('search_image')
    p_img.add_argument('query')

    args = parser.parse_args()
    
    if args.command:
        agent = AIAgent()
        if args.command == 'add_paper':
            agent.add_paper(args.path, args.topics)
        elif args.command == 'search_paper':
            agent.search_paper(args.query)
        elif args.command == 'index_images':
            agent.index_images(args.path)
        elif args.command == 'search_image':
            agent.search_image(args.query)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()