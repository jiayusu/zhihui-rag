"""
查询处理器模块，实现步骤2的查询转换功能
包括多查询转写、问题分解和意图识别
"""
from typing import List, Dict, Tuple
from llm import LLMManager
import logging

logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.prompt_templates = {
            "multi_query": """
你是一个查询改写专家。将以下用户问题改写成3-5个不同角度的查询，以提高向量数据库搜索效果。
用户问题: {question}
要求:
1. 保持原意不变
2. 使用不同的表述方式
3. 包含不同的关键词，对于技术问题需包含专业术语如原理、机制、架构等
4. 每个查询单独一行
输出格式: 仅返回查询列表，不包含其他内容
""",
            "question_decomposition": """
将以下复杂问题分解为2-3个简单子问题，每个子问题应能独立回答。
用户问题: {question}
输出格式: 仅返回子问题列表，每个问题单独一行
""",
            "intent_classification": """
判断以下用户问题属于哪种类型，仅返回类型标签。
类型:
- overview: 了解项目整体信息
- detail: 了解项目细节信息
用户问题: {question}
输出格式: 仅返回overview或detail
"""
        }

    def generate_multiple_queries(self, question: str) -> List[str]:
        """生成多个改写查询以提高召回率"""
        try:
            prompt = self.prompt_templates["multi_query"].format(question=question)
            response = self.llm_manager.generate(prompt)
            queries = [q.strip() for q in response.split('\n') if q.strip()]
            return queries[:5]  # 最多返回5个查询
        except Exception as e:
            logger.error(f"多查询生成失败: {str(e)}")
            return [question]  # 失败时返回原始问题

    def decompose_question(self, question: str) -> List[str]:
        """将复杂问题分解为子问题"""
        try:
            prompt = self.prompt_templates["question_decomposition"].format(question=question)
            response = self.llm_manager.generate(prompt)
            sub_questions = [q.strip() for q in response.split('\n') if q.strip()]
            return sub_questions
        except Exception as e:
            logger.error(f"问题分解失败: {str(e)}")
            return [question]  # 失败时返回原始问题

    def classify_intent(self, question: str) -> str:
        """将问题分类为整体信息(overview)或细节信息(detail)"""
        try:
            prompt = self.prompt_templates["intent_classification"].format(question=question)
            response = self.llm_manager.generate(prompt).strip().lower()
            return response if response in ['overview', 'detail'] else 'detail'  # 默认返回detail
        except Exception as e:
            logger.error(f"意图分类失败: {str(e)}")
            return 'detail'  # 失败时默认返回detail

    def process_query(self, question: str) -> Tuple[str, List[str], List[str]]:
        """完整处理流程: 分类 -> 分解 -> 生成多查询"""
        intent = self.classify_intent(question)
        sub_questions = self.decompose_question(question)
        
        # 为每个子问题生成多查询
        all_queries = []
        for sub_q in sub_questions:
            all_queries.extend(self.generate_multiple_queries(sub_q))
            
        return intent, sub_questions, all_queries

    def get_fallback_response(self, project_id: str) -> str:
        """步骤6: 搜索无结果时返回项目联系人"""
        # 在实际应用中，这里应该从数据库或配置文件读取联系人信息
        contact_info = {
            'project1': '张经理: zhang@example.com',
            'project2': '李经理: li@example.com'
        }
        return f"未找到相关信息。项目联系人: {contact_info.get(project_id, '未知')}"

    def judge_satisfaction(self, question: str, answer: str) -> Tuple[bool, List[str]]:
        """步骤7: 判断回答是否满足需求并生成相关问题"""
        prompt = f"""
判断以下回答是否充分回答了问题。如果不充分，生成3个相关问题选项。
问题: {question}
回答: {answer}
输出格式:
充分性: [yes/no]
相关问题: [问题1];[问题2];[问题3]
"""
        
        try:
            response = self.llm_manager.generate(prompt)
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            # 解析充分性
            sufficient = any(line.startswith('充分性: yes') for line in lines)
            
            # 解析相关问题
            related_questions = []
            for line in lines:
                if line.startswith('相关问题:'):
                    related_questions = [q.strip() for q in line.split('相关问题:')[1].split(';') if q.strip()]
            
            return sufficient, related_questions[:3]
        except Exception as e:
            logger.error(f"结果判断失败: {str(e)}")
            return True, []  # 默认认为充分