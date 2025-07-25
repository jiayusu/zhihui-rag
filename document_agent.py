"""
文档智能体模块，实现步骤4的多文档摘要功能
每个文档对应一个智能体，用于生成专业领域摘要
"""
from typing import List, Dict, Optional
from llm import LLMManager
import logging

logger = logging.getLogger(__name__)

class DocumentAgent:
    """单个文档的智能体，负责生成专业摘要"""
    def __init__(self, llm_manager: LLMManager, document: Dict):
        self.llm_manager = llm_manager
        self.document = document
        self.content = document.get('content', '')
        self.metadata = document.get('metadata', {})
        self.summary = None
        self.expertise = self._determine_expertise()

    def _determine_expertise(self) -> str:
        """根据文档元数据确定专业领域"""
        file_name = self.metadata.get('file_name', '').lower()
        if '技术' in file_name or 'tech' in file_name:
            return '技术文档专家'
        elif '财务' in file_name or 'finance' in file_name:
            return '财务分析专家'
        elif '市场' in file_name or 'market' in file_name:
            return '市场策略专家'
        elif '项目' in file_name or 'project' in file_name:
            return '项目管理专家'
        return '通用文档专家'

    def generate_summary(self) -> str:
        """生成文档摘要"""
        if self.summary:
            return self.summary

        prompt = f"""
你是{self.expertise}，请根据以下文档内容生成专业摘要。
文档标题: {self.metadata.get('file_name', '未命名')}
文档类型: {self.metadata.get('file_type', '未知')}

文档内容:
{self.content[:3000]}  # 限制输入长度

摘要要求:
1. 突出核心内容和关键数据
2. 结构清晰，分点说明
3. 专业术语准确
4. 长度控制在200字以内
"""

        try:
            self.summary = self.llm_manager.generate(prompt)
            return self.summary
        except Exception as e:
            logger.error(f"文档摘要生成失败: {str(e)}")
            return f"【摘要生成失败】{self.metadata.get('file_name', '文档')}核心内容概述"

class MultiDocumentAgent:
    """多文档智能体协调器，负责整合多个文档智能体的结果"""
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.document_agents = []

    def add_documents(self, documents: List[Dict]):
        """为多个文档创建智能体"""
        self.document_agents = [
            DocumentAgent(self.llm_manager, doc)
            for doc in documents
        ]
        logger.info(f"创建了 {len(self.document_agents)} 个文档智能体")

    def generate_individual_summaries(self) -> List[Dict]:
        """生成所有文档的单独摘要"""
        return [{
            'document_id': i,
            'file_name': agent.metadata.get('file_name', f'文档{i}'),
            'expertise': agent.expertise,
            'summary': agent.generate_summary()
        } for i, agent in enumerate(self.document_agents)]

    def integrate_summaries(self, individual_summaries: List[Dict]) -> str:
        """整合多个摘要为综合概述"""
        summaries_text = "\n\n".join([
            f"【{summary['expertise']}】{summary['file_name']}: {summary['summary']}"
            for summary in individual_summaries
        ])

        prompt = f"""
你是项目信息整合专家，请将以下多个文档摘要整合成一份连贯的项目概述。

各个文档摘要:
{summaries_text}

整合要求:
1. 保持各专业领域的关键信息
2. 消除重复内容
3. 按项目整体结构组织信息
4. 突出项目核心价值和特点
5. 长度控制在500字以内
"""

        try:
            return self.llm_manager.generate(prompt)
        except Exception as e:
            logger.error(f"摘要整合失败: {str(e)}")
            return summaries_text

    def process_overview_question(self, question: str) -> str:
        """处理整体信息查询"""
        # 生成所有文档摘要
        individual_summaries = self.generate_individual_summaries()

        # 整合摘要
        integrated_summary = self.integrate_summaries(individual_summaries)

        # 根据问题定制回答
        prompt = f"""
基于以下项目概述回答用户问题:

项目概述:{integrated_summary}

用户问题:{question}

回答要求:
1. 直接回应问题
2. 基于提供的概述内容
3. 保持回答简洁专业
4. 必要时引用具体文档信息
"""

        return self.llm_manager.generate(prompt)