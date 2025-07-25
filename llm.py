"""
- 完善错误处理与异常捕获机制
- 实现对话历史管理系统
- 添加令牌使用量跟踪与限制
- 设计模型降级与故障转移策略
- 增强提示模板管理系统
- 支持流式响应输出
- 实现请求限流与并发控制
- 添加详细日志与性能监控
- 实现安全的API密钥管理
"""
import os
import time
import logging
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Union, Generator, Callable
from pydantic_settings import BaseSettings
from pydantic import Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import tiktoken
from functools import wraps
from collections import deque
import threading
from datetime import datetime

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('llm.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 配置管理系统
class LLMConfig(BaseSettings):
    # API密钥配置
    openai_api_key: Optional[str] = Field(None, env='OPENAI_API_KEY')
    openai_base_url: Optional[str] = Field(None, env='OPENAI_BASE_URL')
    dashscope_api_key: Optional[str] = Field(None, env='DASHSCOPE_API_KEY')
    
    # 模型配置
    default_model: str = Field('openai', env='DEFAULT_LLM_MODEL')
    max_tokens: int = Field(4096, env='MAX_TOKENS')
    temperature: float = Field(0.1, env='TEMPERATURE')
    timeout_seconds: int = Field(30, env='LLM_TIMEOUT_SECONDS')
    
    # 限流配置
    max_concurrent_requests: int = Field(10, env='MAX_CONCURRENT_REQUESTS')
    request_rate_limit: int = Field(60, env='REQUEST_RATE_LIMIT')  # 每分钟请求数
    
    # API密钥配置
    zhipu_api_key: Optional[str] = Field(None, env='ZHIPUAI_API_KEY')
    zhipuai_api_key: Optional[str] = Field(None, env='ZHIPUAI_API_KEY')
    
    # 重试配置
    max_retries: int = Field(3, env='LLM_MAX_RETRIES')
    backoff_factor: float = Field(0.5, env='LLM_BACKOFF_FACTOR')
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

# 加载配置
config = LLMConfig()

# 令牌计数器
class TokenCounter:
    def __init__(self):
        self.encoders = {
            'gpt-3.5-turbo': tiktoken.get_encoding('cl100k_base'),
            'gpt-4': tiktoken.get_encoding('cl100k_base'),
            'default': tiktoken.get_encoding('cl100k_base')
        }
    
    def count_tokens(self, text: str, model: str = 'default') -> int:
        """计算文本的令牌数量"""
        encoder = self.encoders.get(model, self.encoders['default'])
        return len(encoder.encode(text))
    
    def count_messages_tokens(self, messages: List[Dict], model: str = 'default') -> int:
        """计算消息列表的令牌数量"""
        if not messages:
            return 0
        
        encoder = self.encoders.get(model, self.encoders['default'])
        token_count = 0
        
        # 不同模型有不同的消息格式和令牌计算方式
        if model.startswith('gpt-'):
            for message in messages:
                token_count += 4  # 每条消息的固定开销
                for key, value in message.items():
                    token_count += self.count_tokens(value, model)
            token_count += 2  # 对话结束标记
        else:
            # 其他模型的简单计算方式
            for message in messages:
                for value in message.values():
                    token_count += self.count_tokens(str(value), model)
        
        return token_count

# 创建全局令牌计数器实例
token_counter = TokenCounter()

# 对话历史管理器
class ConversationHistory:
    def __init__(self, max_history_tokens: int = 3000, model: str = 'default'):
        self.history = []
        self.max_history_tokens = max_history_tokens
        self.model = model
    
    def add_message(self, role: str, content: str) -> None:
        """添加消息到历史记录并确保不超过令牌限制"""
        message = {'role': role, 'content': content}
        self.history.append(message)
        self._truncate_history()
    
    def _truncate_history(self) -> None:
        """如果历史记录超出令牌限制，则截断最早的消息"""
        while self.get_total_tokens() > self.max_history_tokens and len(self.history) > 1:
            # 保留系统消息（如果有），删除最早的用户/助手消息对
            if self.history[0].get('role') == 'system':
                self.history.pop(1)
                if len(self.history) > 1:
                    self.history.pop(1)
            else:
                self.history.pop(0)
                if len(self.history) > 0:
                    self.history.pop(0)
    
    def get_history(self) -> List[Dict]:
        """获取当前对话历史"""
        return self.history.copy()
    
    def get_total_tokens(self) -> int:
        """获取历史记录的总令牌数"""
        return token_counter.count_messages_tokens(self.history, self.model)
    
    def clear(self) -> None:
        """清除对话历史"""
        self.history = []
    
    def to_dict(self) -> Dict:
        """转换为字典以便存储"""
        return {
            'history': self.history,
            'max_history_tokens': self.max_history_tokens,
            'model': self.model,
            'total_tokens': self.get_total_tokens()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationHistory':
        """从字典加载对话历史"""
        history = cls(
            max_history_tokens=data.get('max_history_tokens', 3000),
            model=data.get('model', 'default')
        )
        history.history = data.get('history', [])
        return history

# 请求限流和并发控制
class RequestLimiter:
    def __init__(self, max_concurrent: int = config.max_concurrent_requests, rate_limit: int = config.request_rate_limit):
        self.semaphore = threading.Semaphore(max_concurrent)
        self.rate_limit = rate_limit
        self.request_timestamps = deque()
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        """获取请求许可，返回是否成功"""
        # 检查并发限制
        if not self.semaphore.acquire(blocking=False):
            logger.warning("Concurrent request limit exceeded")
            return False
        
        # 检查速率限制
        now = time.time()
        with self.lock:
            # 移除一小时前的时间戳
            while self.request_timestamps and now - self.request_timestamps[0] > 3600:
                self.request_timestamps.popleft()
            
            # 检查是否超过速率限制
            if len(self.request_timestamps) >= self.rate_limit:
                self.semaphore.release()
                logger.warning("Rate limit exceeded")
                return False
            
            self.request_timestamps.append(now)
        
        return True
    
    def release(self) -> None:
        """释放请求许可"""
        self.semaphore.release()
    
    def __enter__(self) -> 'RequestLimiter':
        if not self.acquire():
            raise Exception("Request limit exceeded")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

# 创建全局请求限制器实例
request_limiter = RequestLimiter()

# 重试装饰器
def llm_retry_decorator(func):
    @wraps(func)
    @retry(
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(multiplier=config.backoff_factor, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=lambda retry_state: logger.warning(f"LLM request retry in {retry_state.next_action.sleep} seconds...")
    )
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"LLM request successful, elapsed time: {elapsed:.2f}s")
            return result
        except Exception as e:
            logger.error(f"LLM request failed: {str(e)}")
            raise
    return wrapper

# 提示模板管理器
class PromptTemplateManager:
    def __init__(self):
        self.templates = {
            'RAG_PROMPT_TEMPLATE': """严格基于提供的上下文回答问题，不得使用任何外部知识或假设。如果上下文未提供相关信息或无法从中推导出答案，请直接回答"数据库中没有相关内容，无法回答该问题"。始终使用中文回答。
问题: {question}
可参考的上下文：
···
{context}
···
如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
有用的回答:""",
            'INTERNLM_PROMPT_TEMPLATE': """先对上下文内容进行客观总结，然后仅基于总结内容回答问题，不得添加任何未在上下文中明确提及的信息。如果上下文未提供相关信息，请直接回答"数据库中没有相关内容，无法回答该问题"。始终使用中文回答。
问题: {question}
可参考的上下文：
···
{context}
···
如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
有用的回答:"""
        }
    
    def get_template(self, template_name: str) -> str:
        """获取指定的提示模板"""
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template {template_name} not found")
        return template
    
    def render_template(self, template_name: str, **kwargs) -> str:
        """渲染提示模板"""
        template = self.get_template(template_name)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing template variable: {str(e)}")
            raise
    
    def add_template(self, template_name: str, template_content: str) -> None:
        """添加新的提示模板"""
        self.templates[template_name] = template_content
    
    def list_templates(self) -> List[str]:
        """列出所有可用的模板"""
        return list(self.templates.keys())

# 创建全局提示模板管理器实例
prompt_manager = PromptTemplateManager()

PROMPT_TEMPLATE = dict(
    RAG_PROMPT_TEMPALTE="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:""",
    InternLM_PROMPT_TEMPALTE="""先对上下文进行内容总结,再使用上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:"""
)


class BaseLLM:
    def __init__(
        self,
        model_name: str,
        max_tokens: int = config.max_tokens,
        temperature: float = config.temperature,
        timeout: int = config.timeout_seconds,
        max_history_tokens: int = 3000,
        **kwargs
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.conversation_history = ConversationHistory(max_history_tokens, model_name)
        self.token_counter = token_counter
        self.kwargs = kwargs
        self.logger = logger
        self.metrics = {
            'total_tokens': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_response_time': 0.0
        }

    def _preprocess_input(self, prompt: str, history: Optional[List[Dict]] = None) -> Tuple[List[Dict], int]:
        """预处理输入，合并历史记录和当前提示，并计算令牌数"""
        # 如果提供了历史记录，则使用它初始化对话历史
        if history:
            self.conversation_history.history = history
            self.conversation_history._truncate_history()

        # 添加当前用户提示
        self.conversation_history.add_message('user', prompt)
        messages = self.conversation_history.get_history()

        # 计算输入令牌数
        input_tokens = self.token_counter.count_messages_tokens(messages, self.model_name)
        self.logger.info(f"Input tokens: {input_tokens}")

        # 检查令牌限制
        if input_tokens + self.max_tokens > self.get_model_max_tokens():
            raise ValueError(f"Input tokens + max_tokens exceeds model's maximum context size: {input_tokens + self.max_tokens} > {self.get_model_max_tokens()}")

        return messages, input_tokens

    def _postprocess_output(self, response: str, input_tokens: int) -> str:
        """后处理输出，更新对话历史和令牌统计"""
        # 添加助手响应到对话历史
        self.conversation_history.add_message('assistant', response)

        # 计算输出令牌数
        output_tokens = self.token_counter.count_tokens(response, self.model_name)
        total_tokens = input_tokens + output_tokens

        # 更新指标
        self.metrics['total_tokens'] += total_tokens
        self.metrics['successful_requests'] += 1

        self.logger.info(f"Output tokens: {output_tokens}, Total tokens: {total_tokens}")
        return response

    def get_model_max_tokens(self) -> int:
        """获取模型的最大令牌限制"""
        # 默认值，子类应根据实际模型重写
        return 4096

    def generate(self, prompt: str, history: Optional[List[Dict]] = None, **kwargs) -> str:
        """生成文本响应"""
        raise NotImplementedError

    def generate_stream(self, prompt: str, history: Optional[List[Dict]] = None, **kwargs) -> Generator[str, None, None]:
        """生成流式文本响应"""
        raise NotImplementedError

    def get_conversation_history(self) -> List[Dict]:
        """获取当前对话历史"""
        return self.conversation_history.get_history()

    def clear_conversation_history(self) -> None:
        """清除对话历史"""
        self.conversation_history.clear()

    def get_metrics(self) -> Dict:
        """获取性能指标"""
        return self.metrics.copy()

    def _log_error(self, error: Exception, context: str = "") -> None:
        """记录错误信息"""
        self.metrics['failed_requests'] += 1
        self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)

class OpenAIChat(BaseLLM):
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = config.max_tokens,
        temperature: float = config.temperature,
        timeout: int = config.timeout_seconds,
        max_history_tokens: int = 3000,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            max_history_tokens=max_history_tokens,
            **kwargs
        )
        self.api_key = api_key or config.openai_api_key
        self.base_url = base_url or config.openai_base_url
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.logger.info(f"OpenAI client initialized with model: {model_name}")
        except ImportError:
            self.logger.error("openai package not installed. Please install it with 'pip install openai'")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    def get_model_max_tokens(self) -> int:
        """获取模型的最大令牌限制"""
        model_limits = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000
        }
        return model_limits.get(self.model_name, 4096)

    @llm_retry_decorator
    def generate(self, prompt: str, history: Optional[List[Dict]] = None, **kwargs) -> str:
        """生成文本响应"""
        start_time = time.time()
        try:
            with request_limiter:
                # 预处理输入
                messages, input_tokens = self._preprocess_input(prompt, history)
                
                # 调用API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=self.timeout,
                    **kwargs
                )
                
                # 提取响应内容
                if not response.choices or not response.choices[0].message:
                    raise ValueError("Empty response from OpenAI API")
                
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("No content in OpenAI response")
                
                # 后处理输出
                processed_content = self._postprocess_output(content, input_tokens)
                
                # 更新指标
                self.metrics['total_response_time'] += (time.time() - start_time)
                
                return processed_content
        except Exception as e:
            self._log_error(e, context=f"OpenAIChat.generate({self.model_name})")
            self.metrics['failed_requests'] += 1
            raise

    @llm_retry_decorator
    def generate_stream(
        self, prompt: str, history: Optional[List[Dict]] = None,** kwargs
    ) -> Generator[str, None, None]:
        """生成流式文本响应"""
        start_time = time.time()
        full_response = []
        input_tokens = 0
        
        try:
            with request_limiter:
                # 预处理输入
                messages, input_tokens = self._preprocess_input(prompt, history)
                
                # 调用API获取流式响应
                stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=self.timeout,
                    stream=True,
                    **kwargs
                )
                
                # 处理流式响应
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response.append(content)
                        yield content
                
                # 合并完整响应并后处理
                full_content = ''.join(full_response)
                self._postprocess_output(full_content, input_tokens)
                
                # 更新指标
                self.metrics['total_response_time'] += (time.time() - start_time)
                self.metrics['successful_requests'] += 1
                
                return
        except Exception as e:
            self._log_error(e, context=f"OpenAIChat.generate_stream({self.model_name})")
            self.metrics['failed_requests'] += 1
            # 如果发生错误，仍然尝试返回已生成的部分响应
            if full_response:
                yield ''.join(full_response) + f"\n\n[Warning: Response interrupted due to error: {str(e)}]"
            raise

    def check_health(self) -> bool:
        """检查模型服务健康状态"""
        try:
            response = self.client.models.retrieve(self.model_name)
            return response.id == self.model_name
        except Exception as e:
            self.logger.error(f"Health check failed for {self.model_name}: {str(e)}")
            return False

class InternLMChat(BaseLLM):
    def __init__(
        self,
        model_name: str = "internlm-chat-7b",
        max_tokens: int = config.max_tokens,
        temperature: float = config.temperature,
        timeout: int = config.timeout_seconds,
        max_history_tokens: int = 3000,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            max_history_tokens=max_history_tokens,
            **kwargs
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.loaded = False
        self.logger.info(f"Initializing InternLMChat with model: {model_name} on {self.device}")

    def get_model_max_tokens(self) -> int:
        """获取模型的最大令牌限制"""
        model_limits = {
            "internlm-chat-7b": 4096,
            "internlm-chat-10b": 4096,
            "internlm-xcomposer-7b": 4096,
            "internlm-20b": 8192
        }
        return model_limits.get(self.model_name, 4096)

    def load_model(self) -> None:
        """加载模型和分词器"""
        if self.loaded:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.logger.info(f"Loading InternLM model: {self.model_name}")
            start_time = time.time()

            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                **self.kwargs
            )

            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                **self.kwargs
            ).to(self.device).eval()

            # 启用混合精度推理（如果使用CUDA）
            if self.device == "cuda":
                self.model = torch.compile(self.model)

            load_time = time.time() - start_time
            self.logger.info(f"Model {self.model_name} loaded successfully in {load_time:.2f} seconds")
            self.loaded = True

        except ImportError as e:
            self._log_error(e, "InternLMChat.load_model - Missing dependencies")
            raise RuntimeError("Please install required packages: pip install torch transformers") from e
        except Exception as e:
            self._log_error(e, f"InternLMChat.load_model - Failed to load {self.model_name}")
            raise

    @llm_retry_decorator
    def generate(self, prompt: str, history: Optional[List[Dict]] = None,** kwargs) -> str:
        """生成文本响应"""
        start_time = time.time()
        try:
            with request_limiter:
                # 确保模型已加载
                self.load_model()

                # 预处理输入
                messages, input_tokens = self._preprocess_input(prompt, history)

                # 渲染提示模板
                formatted_prompt = prompt_manager.render_template(
                    'INTERNLM_PROMPT_TEMPLATE',
                    question=prompt,
                    context=messages[-1]['content'] if messages else ""
                )

                # 调用模型生成响应
                response, new_history = self.model.chat(
                    self.tokenizer,
                    formatted_prompt,
                    history=[(h['role'], h['content']) for h in messages[:-1]] if len(messages) > 1 else [],
                    temperature=self.temperature,
                    max_new_tokens=self.max_tokens,
                    **kwargs
                )

                # 后处理输出
                processed_response = self._postprocess_output(response, input_tokens)

                # 更新指标
                self.metrics['total_response_time'] += (time.time() - start_time)

                return processed_response

        except Exception as e:
            self._log_error(e, context=f"InternLMChat.generate({self.model_name})")
            self.metrics['failed_requests'] += 1
            raise

    @llm_retry_decorator
    def generate_stream(
        self, prompt: str, history: Optional[List[Dict]] = None,** kwargs
    ) -> Generator[str, None, None]:
        """生成流式文本响应"""
        start_time = time.time()
        full_response = []
        input_tokens = 0

        try:
            with request_limiter:
                # 确保模型已加载
                self.load_model()

                # 预处理输入
                messages, input_tokens = self._preprocess_input(prompt, history)

                # 渲染提示模板
                formatted_prompt = prompt_manager.render_template(
                    'INTERNLM_PROMPT_TEMPLATE',
                    question=prompt,
                    context=messages[-1]['content'] if messages else ""
                )

                # 调用模型生成流式响应
                for response_chunk in self.model.chat_stream(
                    self.tokenizer,
                    formatted_prompt,
                    history=[(h['role'], h['content']) for h in messages[:-1]] if len(messages) > 1 else [],
                    temperature=self.temperature,
                    max_new_tokens=self.max_tokens,
                    **kwargs
                ):
                    full_response.append(response_chunk)
                    yield response_chunk

                # 合并完整响应并后处理
                full_content = ''.join(full_response)
                self._postprocess_output(full_content, input_tokens)

                # 更新指标
                self.metrics['total_response_time'] += (time.time() - start_time)
                self.metrics['successful_requests'] += 1

                return

        except Exception as e:
            self._log_error(e, context=f"InternLMChat.generate_stream({self.model_name})")
            self.metrics['failed_requests'] += 1
            if full_response:
                yield ''.join(full_response) + f"\n\n[Warning: Response interrupted due to error: {str(e)}]"
            raise

    def check_health(self) -> bool:
        """检查模型健康状态"""
        try:
            self.load_model()
            return self.loaded
        except Exception:
            return False


class DashscopeChat(BaseLLM):
    def __init__(
        self,
        model_name: str = "qwen-plus",
        max_tokens: int = config.max_tokens,
        temperature: float = config.temperature,
        timeout: int = config.timeout_seconds,
        max_history_tokens: int = 3000,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            max_history_tokens=max_history_tokens,
            **kwargs
        )
        self.api_key = config.dashscope_api_key
        
        if not self.api_key:
            raise ValueError("Dashscope API key not provided")
        
        try:
            import dashscope
            dashscope.api_key = self.api_key
            self.logger.info(f"Dashscope client initialized with model: {model_name}")
        except ImportError:
            self.logger.error("dashscope package not installed. Please install it with 'pip install dashscope'")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize Dashscope client: {str(e)}")
            raise

    def get_model_max_tokens(self) -> int:
        """获取模型的最大令牌限制"""
        model_limits = {
            "qwen-v1": 8192,
            "qwen-plus": 8192,
            "qwen-max": 32768,
            "qwen-max-longcontext": 128000,
            "qwen-7b-chat": 4096
        }
        return model_limits.get(self.model_name, 8192)

    @llm_retry_decorator
    def generate(self, prompt: str, history: Optional[List[Dict]] = None, **kwargs) -> str:
        """生成文本响应"""
        start_time = time.time()
        try:
            import dashscope
            from dashscope import Generation
            
            with request_limiter:
                # 预处理输入
                messages, input_tokens = self._preprocess_input(prompt, history)
                
                # 调用API
                response = Generation.call(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=self.timeout,
                    **kwargs
                )
                
                # 处理响应
                if response.status_code != Generation.StatusCode.SUCCESS:
                    raise RuntimeError(f"Dashscope API error: {response.message}")
                
                if not response.output or not response.output.choices:
                    raise ValueError("Empty response from Dashscope API")
                
                content = response.output.choices[0].message.content
                if not content:
                    raise ValueError("No content in Dashscope response")
                
                # 后处理输出
                processed_content = self._postprocess_output(content, input_tokens)
                
                # 更新指标
                self.metrics['total_response_time'] += (time.time() - start_time)
                
                return processed_content
        except Exception as e:
            self._log_error(e, context=f"DashscopeChat.generate({self.model_name})")
            self.metrics['failed_requests'] += 1
            raise

    @llm_retry_decorator
    def generate_stream(
        self, prompt: str, history: Optional[List[Dict]] = None,** kwargs
    ) -> Generator[str, None, None]:
        """生成流式文本响应"""
        start_time = time.time()
        full_response = []
        input_tokens = 0
        
        try:
            import dashscope
            from dashscope import Generation
            
            with request_limiter:
                # 预处理输入
                messages, input_tokens = self._preprocess_input(prompt, history)
                
                # 调用API获取流式响应
                responses = Generation.call(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=self.timeout,
                    stream=True,
                    **kwargs
                )
                
                # 处理流式响应
                for response in responses:
                    if response.status_code != Generation.StatusCode.SUCCESS:
                        raise RuntimeError(f"Dashscope API error: {response.message}")
                    
                    if response.output and response.output.choices:
                        content = response.output.choices[0].message.content
                        if content:
                            full_response.append(content)
                            yield content
                
                # 合并完整响应并后处理
                full_content = ''.join(full_response)
                self._postprocess_output(full_content, input_tokens)
                
                # 更新指标
                self.metrics['total_response_time'] += (time.time() - start_time)
                self.metrics['successful_requests'] += 1
                
                return
        except Exception as e:
            self._log_error(e, context=f"DashscopeChat.generate_stream({self.model_name})")
            self.metrics['failed_requests'] += 1
            if full_response:
                yield ''.join(full_response) + f"\n\n[Warning: Response interrupted due to error: {str(e)}]"
            raise

    def check_health(self) -> bool:
        """检查模型服务健康状态"""
        try:
            import dashscope
            from dashscope import Generation
            
            test_prompt = "健康检查"
            response = Generation.call(
                model=self.model_name,
                messages=[{"role": "user", "content": test_prompt}],
                max_tokens=10,
                timeout=5
            )
            return response.status_code == Generation.StatusCode.SUCCESS
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False
    

class LLMManager:
    def __init__(
        self,
        model_type: str,
        model_name: str,
        model_configs: List[Dict],
        default_model_index: int = 0,
        max_history_tokens: int = 3000,
        **kwargs
    ):
        """
        初始化LLM管理器，支持模型降级与故障转移
        
        Args:
            model_configs: 模型配置列表，每个配置包含'type'和模型特定参数
            default_model_index: 默认使用的模型索引
            max_history_tokens: 对话历史的最大令牌数
        """
        self.model_configs = model_configs
        self.default_model_index = default_model_index
        self.model_type = model_type
        self.model_name = model_name
        self.max_history_tokens = max_history_tokens
        self.models = []
        self.model_health = {}
        self.conversation_history = ConversationHistory(max_history_tokens)
        self.logger = logger
        self.kwargs = kwargs
        
        # 初始化所有模型
        self._initialize_models()
        
        # 检查所有模型健康状态
        self.check_all_models_health()

    def _initialize_models(self) -> None:
        """初始化配置的所有模型"""
        model_classes = {
            'openai': OpenAIChat,
            'internlm': InternLMChat,
            'dashscope': DashscopeChat,
            'zhipu': ZhipuChat
        }
        
        for i, config in enumerate(self.model_configs):
            model_type = self.model_type
            if not model_type or model_type not in model_classes:
                self.logger.error(f"Invalid model type: {model_type} in config {i}")
                continue
            
            try:
                # 提取模型参数，排除'type'
                model_params = {k: v for k, v in config.dict().items() if k != 'type'}
                
                # 添加通用参数
                model_params.setdefault('max_history_tokens', self.max_history_tokens)
                model_params.update(self.kwargs)
                
                # 创建模型实例
                model_class = model_classes[model_type]
                model = model_class(**model_params)
                self.models.append(model)
                self.model_health[i] = True
                self.logger.info(f"Initialized model {i}: {model_type} ({model.model_name})")
            except Exception as e:
                self.logger.error(f"Failed to initialize model {i}: {str(e)}")
                self.models.append(None)
                self.model_health[i] = False

    def check_all_models_health(self) -> Dict[int, bool]:
        """检查所有模型的健康状态"""
        for i, model in enumerate(self.models):
            if model is None:
                self.model_health[i] = False
                continue
            
            try:
                self.model_health[i] = model.check_health()
                status = "healthy" if self.model_health[i] else "unhealthy"
                self.logger.info(f"Model {i} ({model.model_name}) is {status}")
            except Exception as e:
                self.logger.error(f"Error checking health for model {i}: {str(e)}")
                self.model_health[i] = False
        
        return self.model_health

    def get_healthy_models(self) -> List[int]:
        """获取所有健康模型的索引"""
        return [i for i, healthy in self.model_health.items() if healthy]

    def _get_fallback_model_index(self, start_from: int = 0) -> Optional[int]:
        """获取降级模型的索引"""
        healthy_models = self.get_healthy_models()
        
        # 首先尝试从start_from之后查找健康模型
        for i in healthy_models:
            if i >= start_from:
                return i
        
        # 如果找不到，尝试从头查找
        for i in healthy_models:
            if i < start_from:
                return i
        
        return None

    def generate(
        self,
        prompt: str,
        history: Optional[List[Dict]] = None,
        model_index: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        生成文本响应，支持指定模型或使用默认模型，自动降级
        
        Args:
            prompt: 用户提示
            history: 对话历史
            model_index: 指定使用的模型索引
            **kwargs: 额外参数
        
        Returns:
            生成的响应文本
        """
        # 如果提供了历史记录，更新对话历史
        if history:
            self.conversation_history.history = history
            
        # 确定要使用的模型索引
        current_index = model_index if model_index is not None else self.default_model_index
        
        # 检查模型是否健康
        if not self.model_health.get(current_index, False):
            self.logger.warning(f"Model {current_index} is unhealthy, trying fallback")
            current_index = self._get_fallback_model_index()
            
            if current_index is None:
                raise RuntimeError("No healthy models available")
        
        # 尝试使用当前模型生成响应
        last_exception = None
        while current_index is not None:
            model = self.models[current_index]
            if not model:
                current_index = self._get_fallback_model_index(current_index + 1)
                continue
            
            try:
                self.logger.info(f"Generating response with model {current_index}: {model.model_name}")
                response = model.generate(prompt, history,** kwargs)
                
                # 更新全局对话历史
                self.conversation_history.history = model.get_conversation_history()
                
                return response
            except Exception as e:
                last_exception = e
                self.logger.error(f"Model {current_index} generation failed: {str(e)}")
                
                # 标记模型为不健康
                self.model_health[current_index] = False
                
                # 获取下一个降级模型
                current_index = self._get_fallback_model_index(current_index + 1)
        
        raise RuntimeError(f"All models failed to generate response: {str(last_exception)}") from last_exception

    def generate_stream(
        self,
        prompt: str,
        history: Optional[List[Dict]] = None,
        model_index: Optional[int] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        生成流式文本响应，支持指定模型或使用默认模型，自动降级
        
        Args:
            prompt: 用户提示
            history: 对话历史
            model_index: 指定使用的模型索引
            **kwargs: 额外参数
        
        Returns:
            流式响应生成器
        """
        # 如果提供了历史记录，更新对话历史
        if history:
            self.conversation_history.history = history
            
        # 确定要使用的模型索引
        current_index = model_index if model_index is not None else self.default_model_index
        
        # 检查模型是否健康
        if not self.model_health.get(current_index, False):
            self.logger.warning(f"Model {current_index} is unhealthy, trying fallback")
            current_index = self._get_fallback_model_index()
            
            if current_index is None:
                raise RuntimeError("No healthy models available")
        
        # 尝试使用当前模型生成流式响应
        last_exception = None
        while current_index is not None:
            model = self.models[current_index]
            if not model:
                current_index = self._get_fallback_model_index(current_index + 1)
                continue
            
            try:
                self.logger.info(f"Generating stream with model {current_index}: {model.model_name}")
                stream = model.generate_stream(prompt, history,** kwargs)
                
                # 转发流式响应
                full_response = []
                for chunk in stream:
                    full_response.append(chunk)
                    yield chunk
                
                # 更新全局对话历史
                self.conversation_history.history = model.get_conversation_history()
                
                return
            except Exception as e:
                last_exception = e
                self.logger.error(f"Model {current_index} stream generation failed: {str(e)}")
                
                # 标记模型为不健康
                self.model_health[current_index] = False
                
                # 获取下一个降级模型
                current_index = self._get_fallback_model_index(current_index + 1)
        
        # 如果所有模型都失败，尝试返回部分响应
        if 'full_response' in locals() and full_response:
            yield ''.join(full_response) + f"\n\n[Error: All models failed. Last error: {str(last_exception)}]"
        
        raise RuntimeError(f"All models failed to generate stream: {str(last_exception)}") from last_exception

    def get_conversation_history(self) -> List[Dict]:
        """获取当前对话历史"""
        return self.conversation_history.get_history()

    def clear_conversation_history(self) -> None:
        """清除对话历史"""
        self.conversation_history.clear()
        for model in self.models:
            if model:
                model.clear_conversation_history()

    def get_model_metrics(self) -> Dict:
        """获取所有模型的指标"""
        metrics = {}
        for i, model in enumerate(self.models):
            if model:
                metrics[f"model_{i}_{model.model_name}"] = model.get_metrics()
        return metrics


class ZhipuChat(BaseLLM):
    def __init__(
        self,
        model_name: str = "glm-4",
        max_tokens: int = config.max_tokens,
        temperature: float = config.temperature,
        timeout: int = config.timeout_seconds,
        max_history_tokens: int = 3000,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            max_history_tokens=max_history_tokens,
            **kwargs
        )
        self.api_key = config.zhipuai_api_key
        
        if not self.api_key:
            raise ValueError("Zhipu API key not provided")
        
        try:
            from zhipuai import ZhipuAI
            self.client = ZhipuAI(api_key=self.api_key)
            self.logger.info(f"Zhipu client initialized with model: {model_name}")
        except ImportError:
            self.logger.error("zhipuai package not installed. Please install it with 'pip install zhipuai'")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize Zhipu client: {str(e)}")
            raise

    def get_model_max_tokens(self) -> int:
        """获取模型的最大令牌限制"""
        model_limits = {
            "glm-3-turbo": 4096,
            "glm-4": 8192,
            "glm-4v": 8192,
            "glm-4-long": 128000
        }
        return model_limits.get(self.model_name, 8192)

    @llm_retry_decorator
    def generate(self, prompt: str, history: Optional[List[Dict]] = None, **kwargs) -> str:
        """生成文本响应"""
        start_time = time.time()
        try:
            with request_limiter:
                # 预处理输入
                messages, input_tokens = self._preprocess_input(prompt, history)
                
                # 调用API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=self.timeout,
                    **kwargs
                )
                
                # 处理响应
                if not response.choices:
                    raise ValueError("Empty response from Zhipu API")
                
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("No content in Zhipu response")
                
                # 后处理输出
                processed_content = self._postprocess_output(content, input_tokens)
                
                # 更新指标
                self.metrics['total_response_time'] += (time.time() - start_time)
                
                return processed_content
        except Exception as e:
            self._log_error(e, context=f"ZhipuChat.generate({self.model_name})")
            self.metrics['failed_requests'] += 1
            raise

    @llm_retry_decorator
    def generate_stream(
        self, prompt: str, history: Optional[List[Dict]] = None,** kwargs
    ) -> Generator[str, None, None]:
        """生成流式文本响应"""
        start_time = time.time()
        full_response = []
        input_tokens = 0
        
        try:
            with request_limiter:
                # 预处理输入
                messages, input_tokens = self._preprocess_input(prompt, history)
                
                # 调用API获取流式响应
                stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=self.timeout,
                    stream=True,
                    **kwargs
                )
                
                # 处理流式响应
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response.append(content)
                        yield content
                
                # 合并完整响应并后处理
                full_content = ''.join(full_response)
                self._postprocess_output(full_content, input_tokens)
                
                # 更新指标
                self.metrics['total_response_time'] += (time.time() - start_time)
                self.metrics['successful_requests'] += 1
                
                return
        except Exception as e:
            self._log_error(e, context=f"ZhipuChat.generate_stream({self.model_name})")
            self.metrics['failed_requests'] += 1
            if full_response:
                yield ''.join(full_response) + f"\n\n[Warning: Response interrupted due to error: {str(e)}]"
            raise

    def check_health(self) -> bool:
        """检查模型服务健康状态"""
        try:
            test_prompt = "健康检查"
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": test_prompt}],
                max_tokens=10,
                timeout=5
            )
            return bool(response.choices and response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False