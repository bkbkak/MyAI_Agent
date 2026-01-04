import os
import argparse
import shutil
import uuid
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
import chromadb
from PIL import Image

# --- 全局路径配置 ---
BASE_DIR = os.getcwd()
DB_PATH = os.path.join(BASE_DIR, "db")
PAPER_DIR = os.path.join(BASE_DIR, "library", "papers")
IMAGE_DIR = os.path.join(BASE_DIR, "library", "images")

class LocalAIAgent:
    def __init__(self):
        print("🚀 正在初始化 AI 模型 ...")
        # 1. 加载模型 (强制使用 CPU)
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.clip_model = SentenceTransformer('clip-ViT-B-32', device='cpu')
        
        # 2. 初始化向量数据库
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.paper_collection = self.client.get_or_create_collection("papers")
        self.image_collection = self.client.get_or_create_collection("images")
        
        # 3. 确保目录存在
        os.makedirs(PAPER_DIR, exist_ok=True)
        os.makedirs(IMAGE_DIR, exist_ok=True)
        print("✅ 系统就绪\n" + "="*30)

    def _extract_text_from_pdf(self, file_path):
        """提取 PDF 文本"""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages[:5]:
                text += page.extract_text() + "\n"
            return text.strip() if text.strip() else None
        except Exception as e:
            print(f"❌ 无法读取 PDF {file_path}: {e}")
            return None

    # ================= 核心功能：文献管理 =================

    def add_paper(self, file_path, topics=None):
        """添加论文：分类 + 物理移动 + 索引去重"""
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return

        text = self._extract_text_from_pdf(file_path)
        if not text:
            print(f"⚠️ 跳过空文件: {file_path}")
            return

        # 1. 语义分类
        target_dir = PAPER_DIR
        if topics:
            topic_list = [t.strip() for t in topics.split(',')]
            doc_emb = self.text_model.encode(text)
            topic_embs = self.text_model.encode(topic_list)
            scores = util.cos_sim(doc_emb, topic_embs)[0]
            best_topic = topic_list[scores.argmax()]
            target_dir = os.path.join(PAPER_DIR, best_topic)
            os.makedirs(target_dir, exist_ok=True)

        # 2. 物理整理
        filename = os.path.basename(file_path)
        new_path = os.path.join(target_dir, filename)
        if os.path.abspath(file_path) != os.path.abspath(new_path):
            shutil.copy(file_path, new_path)

        # 3. 建立索引 (Upsert 实现去重)
        # 逻辑：如果 ID (文件名) 已存在，则更新；不存在则插入。
        embedding = self.text_model.encode(text).tolist()
        self.paper_collection.upsert(
            ids=[filename], 
            embeddings=[embedding],
            metadatas=[{"path": new_path, "filename": filename}],
            documents=[text[:500]]
        )
        print(f"✅ [去重导入] 已归档: {filename} -> {os.path.basename(target_dir)}")

    def batch_organize(self, source_folder, topics):
        """批量整理"""
        print(f"📂 扫描文件夹: {source_folder}")
        files = [f for f in os.listdir(source_folder) if f.lower().endswith('.pdf')]
        if not files:
            print("❌ 未发现 PDF。")
            return
        for f in files:
            self.add_paper(os.path.join(source_folder, f), topics)
        print("✨ 批量整理完成！")

    def search_paper(self, query):
        """【自适应版】搜论文：智能过滤无关结果"""
        print(f"🔍 搜文献: '{query}'")
        query_emb = self.text_model.encode(query).tolist()
        
        results = self.paper_collection.query(
            query_embeddings=[query_emb],
            n_results=3,
            include=["metadatas", "distances"]
        )

        if not results['distances'][0]:
            print("❌ 文献库为空。")
            return

        # --- 动态阈值逻辑 (Adaptive Threshold) ---
        # 文本嵌入通常比较紧密 (0~2之间)，所以阈值容忍度设小一点
        best_score = results['distances'][0][0]
        # 策略：允许比第一名差 0.5 (距离) 以内的结果
        # 如果第一名是 0.8，那么 1.3 以内的才显示，超过 1.3 的说明差距太大
        dynamic_threshold = best_score + 0.5 

        found = False
        print(f"(最佳匹配分: {best_score:.4f} | 智能过滤线: {dynamic_threshold:.4f})")

        for i, (meta, dist) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
            if dist <= dynamic_threshold:
                print(f"[{i+1}] ✅ 匹配: {meta['filename']} (分值: {dist:.4f})")
                print(f"    📍 {meta['path']}")
                found = True
            else:
                # 过滤掉的结果（可选：注释掉下面这行以完全隐藏）
                pass # print(f"   [已过滤] {meta['filename']} 相关度太低。")

        if not found:
            print("❌ 未找到高度相关的文献。")

    # ================= 核心功能：图像管理 =================

    def index_images(self, folder_path):
        """建立图片索引 (Upsert 去重)"""
        print(f"🖼️ 正在索引图片: {folder_path}")
        count = 0
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(root, file)
                    try:
                        img = Image.open(path)
                        emb = self.clip_model.encode(img).tolist()
                        # Upsert 实现去重
                        self.image_collection.upsert(
                            ids=[file], 
                            embeddings=[emb],
                            metadatas=[{"path": path}]
                        )
                        count += 1
                    except:
                        pass
        print(f"✅ [去重导入] 完成！当前共处理 {count} 张图片。")

    def search_image(self, query):
        """【自适应版】搜图片：智能过滤无关结果"""
        print(f"🖼️ 搜图: '{query}'")
        query_emb = self.clip_model.encode(query).tolist()
        
        results = self.image_collection.query(
            query_embeddings=[query_emb],
            n_results=3,
            include=["metadatas", "distances"]
        )
        
        if not results['distances'][0]:
            print("❌ 图片库为空。")
            return

        # --- 动态阈值逻辑 ---
        # CLIP 的距离分值较大 (通常 150~200)，所以容忍度给大一点
        best_score = results['distances'][0][0]
        dynamic_threshold = best_score + 10.0 

        found = False
        print(f"(最佳匹配分: {best_score:.2f} | 智能过滤线: {dynamic_threshold:.2f})")

        for i, (meta, dist) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
            if dist <= dynamic_threshold:
                print(f"[{i+1}] ✅ 匹配: {meta['path']} (分值: {dist:.4f})")
                found = True
            
        if not found:
            print("❌ 未找到匹配图片。")

# ================= 命令行入口 =================

def main():
    parser = argparse.ArgumentParser(description="Local AI Agent")
    subparsers = parser.add_subparsers(dest='command')

    p_add = subparsers.add_parser('add_paper')
    p_add.add_argument('path')
    p_add.add_argument('--topics')

    p_batch = subparsers.add_parser('batch_organize')
    p_batch.add_argument('folder')
    p_batch.add_argument('--topics')

    p_search = subparsers.add_parser('search_paper')
    p_search.add_argument('query')

    p_idx = subparsers.add_parser('index_images')
    p_idx.add_argument('path')

    p_img = subparsers.add_parser('search_image')
    p_img.add_argument('query')

    args = parser.parse_args()
    
    if args.command:
        agent = LocalAIAgent()
        if args.command == 'add_paper':
            agent.add_paper(args.path, args.topics)
        elif args.command == 'batch_organize':
            agent.batch_organize(args.folder, args.topics)
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
