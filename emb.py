"""
- 添加完整的异常处理机制，包括API调用失败、网络错误等场景
- 实现嵌入结果缓存系统，避免重复计算
- 添加请求重试机制与指数退避策略
- 设计多嵌入模型的降级切换方案
- 增加输入文本验证与清洗
- 添加性能监控与日志记录
- 实现批量处理接口提升效率
- 增加配置管理系统，统一管理API密钥和参数
"""

import os
import time
import hashlib
import logging
import json
from functools import lru_cache, wraps
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pydantic_settings import BaseSettings
from pydantic import Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('embeddings.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 配置管理系统
class EmbeddingSettings(BaseSettings):
    openai_api_key: Optional[str] = Field(None, env='OPENAI_API_KEY')
    openai_base_url: Optional[str] = Field(None, env='OPENAI_BASE_URL')
    zhipuai_api_key: Optional[str] = Field(None, env='ZHIPUAI_API_KEY')
    dashscope_api_key: Optional[str] = Field(None, env='DASHSCOPE_API_KEY')
    embedding_cache_dir: str = Field('embedding_cache', env='EMBEDDING_CACHE_DIR')
    max_retries: int = Field(3, env='EMBEDDING_MAX_RETRIES')
    backoff_factor: float = Field(0.5, env='EMBEDDING_BACKOFF_FACTOR')
    timeout_seconds: int = Field(30, env='EMBEDDING_TIMEOUT_SECONDS')
    max_text_length: int = Field(8192, env='MAX_TEXT_LENGTH')

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

settings = EmbeddingSettings()

# 创建缓存目录
os.makedirs(settings.embedding_cache_dir, exist_ok=True)

# 缓存装饰器
def cache_embedding(func):
    @wraps(func)
    def wrapper(self, text: str, *args, **kwargs):
        # 生成缓存键
        cache_key = hashlib.md5(f'{text}_{self.__class__.__name__}_{args}_{kwargs}'.encode()).hexdigest()
        cache_path = os.path.join(settings.embedding_cache_dir, f'{cache_key}.json')

        # 尝试从缓存加载
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f'Loaded embedding from cache for {self.__class__.__name__}')
                    return data['embedding']
            except Exception as e:
                logger.warning(f'Cache load failed: {str(e)}')

        # 调用原始函数
        embedding = func(self, text, *args, **kwargs)

        # 保存到缓存
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({'text': text, 'embedding': embedding, 'timestamp': time.time()}, f)
            logger.info(f'Saved embedding to cache for {self.__class__.__name__}')
        except Exception as e:
            logger.warning(f'Cache save failed: {str(e)}')

        return embedding
    return wrapper

# 重试装饰器
def retry_with_backoff(func):
    @wraps(func)
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.backoff_factor, min=1, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, TimeoutError)),
        before_sleep=lambda retry_state: logger.warning(f'Retrying in {retry_state.next_action.sleep} seconds...'),
    )
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f'API call successful, elapsed time: {elapsed:.2f}s')
            return result
        except Exception as e:
            logger.error(f'API call failed: {str(e)}')
            raise
    return wrapper

# 文本验证和清洗函数
def clean_and_validate_text(text: str) -> Tuple[str, bool]:
    # 移除多余空白
    cleaned = ' '.join(text.strip().split())
    
    # 检查长度
    if len(cleaned) > settings.max_text_length:
        logger.warning(f'Text exceeds max length, truncating to {settings.max_text_length} characters')
        cleaned = cleaned[:settings.max_text_length]
        return cleaned, False
    
    # 检查是否为空
    if not cleaned:
        logger.warning('Empty text after cleaning')
        return cleaned, False
    
    return cleaned, True


class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api
    
    def get_embedding(self, text: str, model: str = None) -> List[float]:
        raise NotImplementedError
    
    def batch_get_embedding(self, texts: List[str], model: str = None) -> List[List[float]]:
        """
        批量获取嵌入向量
        :param texts: 文本列表
        :param model: 模型名称
        :return: 嵌入向量列表
        """
        embeddings = []
        for text in texts:
            embeddings.append(self.get_embedding(text, model))
        return embeddings
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude
    

class OpenAIEmbedding(BaseEmbeddings):
    """
    class for OpenAI embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from openai import OpenAI, OpenAIError
            self.OpenAIError = OpenAIError
            self.client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url
            )
    
    @cache_embedding
    @retry_with_backoff
    def get_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        if not self.is_api:
            raise NotImplementedError("OpenAIEmbedding only supports API mode")
        
        # 文本清洗和验证
        cleaned_text, is_valid = clean_and_validate_text(text)
        if not is_valid:
            logger.warning(f"Invalid text for embedding: {text[:50]}...")
            return []
        
        try:
            response = self.client.embeddings.create(
                input=[cleaned_text],
                model=model,
                timeout=settings.timeout_seconds
            )
            return response.data[0].embedding
        except self.OpenAIError as e:
            logger.error(f"OpenAI embedding error: {str(e)}")
            raise
    
    def batch_get_embedding(self, texts: List[str], model: str = "text-embedding-3-large") -> List[List[float]]:
        """优化的批量嵌入获取方法"""
        if not self.is_api:
            raise NotImplementedError("OpenAIEmbedding only supports API mode")
        
        # 清洗和验证所有文本
        cleaned_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            cleaned, is_valid = clean_and_validate_text(text)
            if is_valid:
                cleaned_texts.append(cleaned)
                valid_indices.append(i)
            else:
                logger.warning(f"Invalid text at index {i}: {text[:50]}...")
        
        # 处理空输入
        if not cleaned_texts:
            return [[] for _ in texts]
        
        try:
            response = self.client.embeddings.create(
                input=cleaned_texts,
                model=model,
                timeout=settings.timeout_seconds
            )
            
            # 将结果映射回原始顺序
            results = [[] for _ in texts]
            for i, idx in enumerate(valid_indices):
                results[idx] = response.data[i].embedding
            
            return results
        except self.OpenAIError as e:
            logger.error(f"OpenAI batch embedding error: {str(e)}")
            raise

class JinaEmbedding(BaseEmbeddings):
    """
    class for Jina embeddings
    """
    def __init__(self, path: str = 'jinaai/jina-embeddings-v2-base-zh', is_api: bool = False) -> None:
        super().__init__(path, is_api)
        try:
            self._model = self.load_model()
            self._tokenizer = self.load_tokenizer()
        except Exception as e:
            logger.error(f"Jina model initialization failed: {str(e)}")
            raise
    
    @cache_embedding
    def get_embedding(self, text: str) -> List[float]:
        # 文本清洗和验证
        cleaned_text, is_valid = clean_and_validate_text(text)
        if not is_valid:
            logger.warning(f"Invalid text for Jina embedding: {text[:50]}...")
            return []
        
        try:
            with torch.no_grad():
                inputs = self._tokenizer(cleaned_text, return_tensors='pt', padding=True, truncation=True).to(self._model.device)
                outputs = self._model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
            return embedding
        except Exception as e:
            logger.error(f"Jina embedding inference error: {str(e)}")
            raise
    
    def batch_get_embedding(self, texts: List[str]) -> List[List[float]]:
        """优化的Jina模型批量嵌入获取方法"""
        # 清洗和验证所有文本
        cleaned_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            cleaned, is_valid = clean_and_validate_text(text)
            if is_valid:
                cleaned_texts.append(cleaned)
                valid_indices.append(i)
            else:
                logger.warning(f"Invalid text at index {i}: {text[:50]}...")
        
        # 处理空输入
        if not cleaned_texts:
            return [[] for _ in texts]
        
        try:
            with torch.no_grad():
                inputs = self._tokenizer(cleaned_texts, return_tensors='pt', padding=True, truncation=True).to(self._model.device)
                outputs = self._model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).tolist()
            
            # 将结果映射回原始顺序
            results = [[] for _ in texts]
            for i, idx in enumerate(valid_indices):
                results[idx] = embeddings[i]
            
            return results
        except Exception as e:
            logger.error(f"Jina batch embedding error: {str(e)}")
            raise
    
    def load_model(self):
        import torch
        from transformers import AutoModel
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Loading Jina model to CUDA device")
            else:
                device = torch.device("cpu")
                logger.warning("CUDA not available, using CPU for Jina model")
            
            model = AutoModel.from_pretrained(
                self.path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
            ).to(device)
            model.eval()  # 设置为评估模式
            logger.info(f"Successfully loaded Jina model from {self.path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load Jina model: {str(e)}")
            raise
    
    def load_tokenizer(self):
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
            logger.info(f"Successfully loaded Jina tokenizer from {self.path}")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load Jina tokenizer: {str(e)}")
            raise

class ZhipuEmbedding(BaseEmbeddings):
    """
    class for Zhipu embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from zhipuai import ZhipuAI
            from requests.exceptions import ConnectionError as APIConnectionError, Timeout, HTTPError as APIError
            self.ZhipuErrors = (APIError, APIConnectionError, Timeout)
            self.client = ZhipuAI(api_key=settings.zhipuai_api_key)
    
    @cache_embedding
    @retry_with_backoff
    def get_embedding(self, text: str, model: str = "embedding-2") -> List[float]:
        if not self.is_api:
            raise NotImplementedError("ZhipuEmbedding only supports API mode")
        
        # 文本清洗和验证
        cleaned_text, is_valid = clean_and_validate_text(text)
        if not is_valid:
            logger.warning(f"Invalid text for Zhipu embedding: {text[:50]}...")
            return []
        
        try:
            response = self.client.embeddings.create(
                model=model,
                input=cleaned_text,
                timeout=settings.timeout_seconds
            )
            return response.data[0].embedding
        except self.ZhipuErrors as e:
            logger.error(f"Zhipu embedding error: {str(e)}")
            raise
    
    def batch_get_embedding(self, texts: List[str], model: str = "embedding-2") -> List[List[float]]:
        """优化的Zhipu批量嵌入获取方法"""
        if not self.is_api:
            raise NotImplementedError("ZhipuEmbedding only supports API mode")
        
        # 清洗和验证所有文本
        cleaned_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            cleaned, is_valid = clean_and_validate_text(text)
            if is_valid:
                cleaned_texts.append(cleaned)
                valid_indices.append(i)
            else:
                logger.warning(f"Invalid text at index {i}: {text[:50]}...")
        
        # 处理空输入
        if not cleaned_texts:
            return [[] for _ in texts]
        
        try:
            response = self.client.embeddings.create(
                model=model,
                input=cleaned_texts,
                timeout=settings.timeout_seconds
            )
            
            # 将结果映射回原始顺序
            results = [[] for _ in texts]
            for i, idx in enumerate(valid_indices):
                results[idx] = response.data[i].embedding
            
            return results
        except self.ZhipuErrors as e:
            logger.error(f"Zhipu batch embedding error: {str(e)}")
            raise

class DashscopeEmbedding(BaseEmbeddings):
    """
    class for Dashscope embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            import dashscope
            from dashscope import DashScopeAPIError, ConnectionError, Timeout
            self.DashscopeErrors = (DashScopeAPIError, ConnectionError, Timeout)
            dashscope.api_key = settings.dashscope_api_key
            self.client = dashscope.TextEmbedding
    
    @cache_embedding
    @retry_with_backoff
    def get_embedding(self, text: str, model: str='text-embedding-v1') -> List[float]:
        if not self.is_api:
            raise NotImplementedError("DashscopeEmbedding only supports API mode")
        
        # 文本清洗和验证
        cleaned_text, is_valid = clean_and_validate_text(text)
        if not is_valid:
            logger.warning(f"Invalid text for Dashscope embedding: {text[:50]}...")
            return []
        
        try:
            response = self.client.call(
                model=model,
                input=cleaned_text,
                timeout=settings.timeout_seconds
            )
            if response.status_code != 200:
                raise self.DashscopeErrors(f"API request failed with status code {response.status_code}")
            return response.output['embeddings'][0]['embedding']
        except self.DashscopeErrors as e:
            logger.error(f"Dashscope embedding error: {str(e)}")
            raise
    
    def batch_get_embedding(self, texts: List[str], model: str='text-embedding-v1') -> List[List[float]]:
        """优化的Dashscope批量嵌入获取方法"""
        if not self.is_api:
            raise NotImplementedError("DashscopeEmbedding only supports API mode")
        
        # 清洗和验证所有文本
        cleaned_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            cleaned, is_valid = clean_and_validate_text(text)
            if is_valid:
                cleaned_texts.append(cleaned)
                valid_indices.append(i)
            else:
                logger.warning(f"Invalid text at index {i}: {text[:50]}...")
        
        # 处理空输入
        if not cleaned_texts:
            return [[] for _ in texts]
        
        try:
            response = self.client.call(
                model=model,
                input=cleaned_texts,
                timeout=settings.timeout_seconds
            )
            if response.status_code != 200:
                raise self.DashscopeErrors(f"API request failed with status code {response.status_code}")
            
            # 将结果映射回原始顺序
            results = [[] for _ in texts]
            for i, idx in enumerate(valid_indices):
                results[idx] = response.output['embeddings'][i]['embedding']
            
            return results
        except self.DashscopeErrors as e:
            logger.error(f"Dashscope batch embedding error: {str(e)}")
            raise


class EmbeddingManager:
    """
    嵌入模型管理器，实现多模型降级切换功能
    """
    def __init__(self, primary_models: List[BaseEmbeddings], fallback_models: List[BaseEmbeddings] = None):
        self.primary_models = primary_models
        self.fallback_models = fallback_models or []
        self.all_models = primary_models + fallback_models
        
    def get_embedding(self, text: str, model_preferences: List[str] = None) -> List[float]:
        """
        获取文本嵌入向量，支持按优先级尝试不同模型
        :param text: 输入文本
        :param model_preferences: 模型优先级列表（类名）
        :return: 嵌入向量
        """
        # 根据优先级排序模型
        if model_preferences:
            ordered_models = []
            for pref in model_preferences:
                for model in self.all_models:
                    if model.__class__.__name__ == pref and model not in ordered_models:
                        ordered_models.append(model)
            # 添加未在偏好列表中的模型
            for model in self.all_models:
                if model not in ordered_models:
                    ordered_models.append(model)
        else:
            ordered_models = self.all_models
        
        # 尝试所有模型直到成功
        last_exception = None
        for model in ordered_models:
            try:
                logger.info(f"Attempting to get embedding with {model.__class__.__name__}")
                return model.get_embedding(text)
            except Exception as e:
                last_exception = e
                logger.warning(f"Failed to get embedding with {model.__class__.__name__}: {str(e)}")
                continue
        
        # 所有模型都失败
        logger.error(f"All embedding models failed: {str(last_exception)}")
        raise last_exception
    
    def batch_get_embedding(self, texts: List[str], model_preferences: List[str] = None) -> List[List[float]]:
        """
        批量获取文本嵌入向量，支持按优先级尝试不同模型
        :param texts: 输入文本列表
        :param model_preferences: 模型优先级列表（类名）
        :return: 嵌入向量列表
        """
        # 根据优先级排序模型
        if model_preferences:
            ordered_models = []
            for pref in model_preferences:
                for model in self.all_models:
                    if model.__class__.__name__ == pref and model not in ordered_models:
                        ordered_models.append(model)
            # 添加未在偏好列表中的模型
            for model in self.all_models:
                if model not in ordered_models:
                    ordered_models.append(model)
        else:
            ordered_models = self.all_models
        
        # 尝试所有模型直到成功
        last_exception = None
        for model in ordered_models:
            try:
                logger.info(f"Attempting batch embedding with {model.__class__.__name__}")
                return model.batch_get_embedding(texts)
            except Exception as e:
                last_exception = e
                logger.warning(f"Failed to get batch embedding with {model.__class__.__name__}: {str(e)}")
                continue
        
        # 所有模型都失败
        logger.error(f"All batch embedding models failed: {str(last_exception)}")
        raise last_exception