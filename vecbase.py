"""
- 替换JSON文件存储为专业向量数据库(FAISS、Pinecone等)
- 实现高效的向量索引系统
- 添加元数据过滤功能
- 支持向量更新与删除操作
- 实现批量处理接口
- 添加数据库连接池管理
- 实现事务支持确保数据一致性
- 添加备份与恢复机制
- 优化数据库性能
"""
import os
from typing import Dict, List, Optional, Tuple, Union
import json
from emb import BaseEmbeddings, ZhipuEmbedding
import numpy as np
from tqdm import tqdm
import faiss
import shutil


class VectorStore:
    def __init__(self, embedding_dim: int = 1024, index_type: str = 'IVF', document: List[str] = None) -> None:
        self.document = document or []
        self.metadata = []  # 新增元数据存储
        self.embedding_dim = embedding_dim
        self.index = self._create_index(index_type)
        self.id_counter = 0  # 用于跟踪向量ID

    def _create_index(self, index_type: str):
        if index_type == 'HNSW':
            # 使用倒排文件索引，适合大规模数据
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            return faiss.IndexIVFFlat(quantizer, self.embedding_dim, min(2, len(self.document)) if self.document else 2)
        elif index_type == 'HNSW':
            # 使用层次化近似最近邻索引，适合高维数据
            return faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            # 默认使用Flat索引（精确搜索）
            return faiss.IndexFlatL2(self.embedding_dim)

    def add_vector(self, vector: List[float], metadata: Optional[Dict] = None):
        # 添加单个向量及元数据
        vector_np = np.array(vector, dtype=np.float32).reshape(1, -1)
        self.index.add(vector_np)
        self.document.append(metadata.get('text', '') if metadata else '')
        self.metadata.append(metadata or {})
        self.id_counter += 1
        vector_id = self.id_counter - 1
        # 事务日志记录
        if hasattr(self, 'transaction_log') and self.transaction_log is not None:
            self.transaction_log.append({
                'type': 'add',
                'id': vector_id,
                'vector': vector,
                'metadata': metadata
            })
        return vector_id

    def add_vectors_batch(self, vectors: List[List[float]], metadatas: Optional[List[Dict]] = None):
        # 批量添加向量
        if metadatas is None:
            metadatas = [{} for _ in vectors]
        assert len(vectors) == len(metadatas), "向量和元数据数量必须一致"

        vectors_np = np.array(vectors, dtype=np.float32)
        # IVF索引需要先训练
        if not self.index.is_trained:
            self.index.train(vectors_np)
        self.index.add(vectors_np)
        self.document.extend([meta.get('text', '') for meta in metadatas])
        self.metadata.extend(metadatas)
        start_id = self.id_counter
        self.id_counter += len(vectors)
        return list(range(start_id, self.id_counter))

    def delete_vector(self, vector_id: int):
        # 删除向量（FAISS需要特殊处理，这里使用标记删除法）
        if 0 <= vector_id < len(self.metadata):
            # 事务日志记录
            if hasattr(self, 'transaction_log') and self.transaction_log is not None:
                self.transaction_log.append({
                    'type': 'delete',
                    'id': vector_id,
                    'vector': self.vectors[vector_id] if hasattr(self, 'vectors') else None,
                    'metadata': self.metadata[vector_id]
                })
            self.metadata[vector_id]['deleted'] = True
            return True
        return False

    def update_vector(self, vector_id: int, new_vector: List[float], new_metadata: Optional[Dict] = None):
        # 更新向量
        if 0 <= vector_id < len(self.metadata) and not self.metadata[vector_id].get('deleted', False):
            # 记录旧数据用于回滚
            old_vector = self.vectors[vector_id] if hasattr(self, 'vectors') else None
            old_metadata = self.metadata[vector_id].copy()
            # FAISS不直接支持更新，采用先删除后添加的策略
            self.delete_vector(vector_id)
            new_id = self.add_vector(new_vector, new_metadata)
            # 事务日志记录
            if hasattr(self, 'transaction_log') and self.transaction_log is not None:
                self.transaction_log.append({
                    'type': 'update',
                    'id': vector_id,
                    'new_id': new_id,
                    'old_vector': old_vector,
                    'old_metadata': old_metadata,
                    'new_vector': new_vector,
                    'new_metadata': new_metadata
                })
            return new_id
        return None

    def search(self, query_vector: List[float], k: int = 5, filter_metadata: Optional[Dict] = None):
        # 带元数据过滤的向量搜索
        query_np = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_np, k * 2)  # 获取双倍结果用于过滤

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # 无结果
                continue
            if self.metadata[idx].get('deleted', False):  # 跳过已删除向量
                continue
            # 应用元数据过滤
            if filter_metadata and not self._match_metadata(self.metadata[idx], filter_metadata):
                continue
            results.append({
                'document': self.document[idx],
                'metadata': self.metadata[idx],
                'distance': distances[0][i]
            })
            if len(results) >= k:
                break
        return results

    def _match_metadata(self, metadata: Dict, filter: Dict) -> bool:
        # 元数据过滤匹配
        for key, value in filter.items():
            if metadata.get(key) != value:
                return False
        return True

    def persist(self, path: str = 'storage'):
        # 持久化索引和元数据
        if not os.path.exists(path):
            os.makedirs(path)
        # 保存FAISS索引
        faiss.write_index(self.index, os.path.join(path, 'faiss_index.bin'))
        # 保存文档和元数据
        with open(os.path.join(path, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump({'documents': self.document, 'metadata': self.metadata, 'id_counter': self.id_counter}, f, ensure_ascii=False)

    def load(self, path: str = 'storage'):
        # 加载索引和元数据
        self.index = faiss.read_index(os.path.join(path, 'faiss_index.bin'))
        with open(os.path.join(path, 'metadata.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.document = data['documents']
            self.metadata = data['metadata']
            self.id_counter = data['id_counter']

    def backup(self, backup_path: str):
        # 备份索引和元数据
        if not os.path.exists(backup_path):
            os.makedirs(backup_path)
        # 复制当前存储的所有文件
        for filename in ['faiss_index.bin', 'metadata.json']:
            src = os.path.join('storage', filename)
            dst = os.path.join(backup_path, filename)
            if os.path.exists(src):
                shutil.copy(src, dst)
        return True

    def restore(self, backup_path: str):
        # 从备份恢复
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"备份路径不存在: {backup_path}")
        # 复制备份文件到存储目录
        for filename in ['faiss_index.bin', 'metadata.json']:
            src = os.path.join(backup_path, filename)
            dst = os.path.join('storage', filename)
            if os.path.exists(src):
                shutil.copy(src, dst)
        # 重新加载数据
        self.load()
        return True

    def begin_transaction(self):
        # 开始事务（简单实现）
        self.transaction_log = []

    def commit_transaction(self):
        # 提交事务
        self.transaction_log = None

    def rollback_transaction(self):
        # 回滚事务
        for action in reversed(self.transaction_log):
            if action['type'] == 'add':
                self.delete_vector(action['id'])
            elif action['type'] == 'update':
                self.update_vector(action['id'], action['old_vector'], action['old_metadata'])
            elif action['type'] == 'delete':
                self.add_vector(action['vector'], action['metadata'])
        self.transaction_log = None

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        
        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        return BaseEmbeddings.cosine_similarity(vector1, vector2)

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        query_vector = EmbeddingModel.get_embedding(query)
        result = np.array([self.get_similarity(query_vector, vector)
                          for vector in self.vectors])
        return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()