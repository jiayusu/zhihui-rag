import logging
from vecbase import VectorStore
from utils import ReadFiles
import os
from llm import LLMManager, LLMConfig, PromptTemplateManager
from query_processor import QueryProcessor
from document_agent import MultiDocumentAgent
from emb import EmbeddingManager, ZhipuEmbedding

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # 初始化文档读取器
    import os
    logger.info("初始化文档读取器...")
    reader = ReadFiles(path="data/")
    
    # 处理文档（使用并行处理）
    logger.info("处理文档...")
    docs = []
    for root, dirs, files in os.walk('data'):
        for file in files:
            if file.endswith(('.md', '.txt', '.pdf')):
                    file_path = os.path.join(root, file)
                    logger.info(f"正在处理文件: {file_path}")
                    file_docs = reader.get_content(
                        path=file_path,
                        max_token_len=600,
                        cover_content=150,
                        max_workers=4
                    )
                    docs.extend(file_docs)
    
    if not docs:
        logger.warning("未找到任何文档，请检查data目录")
        exit(1)
    
    # 初始化嵌入模型管理器
    logger.info("初始化嵌入模型...")
    embedding_manager = EmbeddingManager(
        primary_models=[ZhipuEmbedding()],
    fallback_models=[]
    )
    
    # 删除现有存储以重建索引
    import shutil
    if os.path.exists('storage'):
        shutil.rmtree('storage')
    
    # 初始化向量存储（使用FAISS IVF索引）
    logger.info("初始化向量存储...")
    vector_store = VectorStore(
        embedding_dim=1024,  # 匹配Zhipu embedding维度
        index_type='HNSW'
    )
    
    # 开始事务
    vector_store.begin_transaction()
    
    try:
        # 提取文本和元数据
        texts = [doc['content'] for doc in docs]
        metadatas = [doc['metadata'] for doc in docs]
        
        # 批量获取嵌入向量
        logger.info("生成嵌入向量...")
        embeddings = embedding_manager.batch_get_embedding(texts)
        
        # 批量添加向量
        logger.info("存储向量数据...")
        vector_ids = vector_store.add_vectors_batch(embeddings, metadatas)
        
        logger.info(f"成功存储 {len(vector_ids)} 个向量")
        
        # 提交事务
        vector_store.commit_transaction()
        
        # 持久化到磁盘
        vector_store.persist(path='storage')
        
    except Exception as e:
        logger.error(f"向量处理出错: {str(e)}", exc_info=True)
        vector_store.rollback_transaction()
        raise
    
    # 初始化查询处理器
    logger.info("初始化查询处理器...")
    # 加载API密钥配置
    # 创建LLM配置实例
    model_config = LLMConfig(
        zhipuai_api_key=os.getenv("ZHIPUAI_API_KEY")
    )
    model_configs = [model_config]
    llm_manager = LLMManager(model_type="zhipu", model_name="glm-4", model_configs=model_configs)
    query_processor = QueryProcessor(llm_manager)
    
    # 问题处理
    question = 'git的原理是什么？'
    logger.info(f"处理问题: {question}")
    
    # 步骤2: 查询转换 - 分类、分解和多查询生成
    intent, sub_questions, all_queries = query_processor.process_query(question)
    logger.info(f"问题类型: {intent}, 生成 {len(all_queries)} 个查询")
    
    # 候选项目ID列表（示例）
    candidate_project_ids = ['project1', 'project2']  # 实际应用中应动态获取
    
    # 执行多查询搜索并合并结果
    all_results = []
    for query in all_queries[:3]:  # 最多使用3个查询
        query_embedding = embedding_manager.get_embedding(query)
        results = vector_store.search(
            query_embedding, 
            k=5,  # 增加返回结果数量以提高召回率
            filter_metadata={

            }
        )
        all_results.extend(results)
    
    # 去重结果
    unique_results = []
    seen_ids = set()
    for result in all_results:
        doc_id = result['metadata'].get('file_path', '') + str(result['metadata'].get('chunk_index', 0))
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            unique_results.append(result)
    
    # 按相关性排序并取前3个结果
    unique_results.sort(key=lambda x: x['distance'])
    results = unique_results[:5]  # 增加返回结果数量以提高召回率
    
    # 步骤6: 搜索无结果时返回项目联系人
    if not results:
        fallback_response = query_processor.get_fallback_response(candidate_project_ids[0])
        logger.warning(f"未找到相关文档: {fallback_response}")
        print(f"回答: {fallback_response}")
        exit(1)
    
    # 准备上下文
    context = "\n\n".join([result['document'] for result in results])
    
    # 步骤4: 整体信息查询的多文档智能体摘要处理
    if intent == 'overview':
        logger.info("使用多文档智能体生成项目摘要...")
        multi_doc_agent = MultiDocumentAgent(llm_manager)
        multi_doc_agent.add_documents(results)  # 添加候选项目文档
        
        # 生成各文档摘要
        individual_summaries = multi_doc_agent.generate_individual_summaries()
        
        # 整合为项目概述
        integrated_summary = multi_doc_agent.integrate_summaries(individual_summaries)
        
        # 生成针对问题的回答
        response = multi_doc_agent.process_overview_question(question)
    else:
        # 步骤5: 细节信息查询的知识库搜索结果处理
        prompt_manager = PromptTemplateManager()
        prompt = prompt_manager.render_template(
            'RAG_PROMPT_TEMPLATE',
            question=question,
            context=context
        )
        response = llm_manager.generate(prompt)
    
    # 步骤7: 判断结果是否满足需求并生成相关问题
    is_sufficient, related_questions = query_processor.judge_satisfaction(question, response)
    
    # 步骤8: 整理搜索结果
    final_answer = f"回答: {response}\n"
    if not is_sufficient:
        final_answer += "\n注意: 此回答可能不完整\n"
    if related_questions:
        final_answer += "\n相关问题:\n"
        for i, q in enumerate(related_questions, 1):
            final_answer += f"{i}. {q}\n"
    
    # 输出结果
    print(f"问题: {question}")
    print(final_answer)
    
except Exception as e:
    logger.error(f"应用程序出错: {str(e)}", exc_info=True)
    exit(1)