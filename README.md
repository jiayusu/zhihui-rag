# zhihui-rag

## 项目概述
25 夏季学习项目，基于检索增强生成(RAG)技术的知识库问答系统实现。该项目旨在通过向量检索与大语言模型结合，提供准确的问答服务，所有回答严格基于知识库内容。

## 功能特点
- 多格式文档处理：支持Markdown、文本和PDF文件的加载与解析
- 高效向量检索：使用FAISS向量数据库实现快速相似性搜索
- 智能查询扩展：自动生成多样化查询以提高召回率
- 严格知识库约束：确保回答完全来源于提供的文档内容
- 完整项目结构：模块化设计，包含文档处理、向量存储、查询处理和LLM集成

## 安装指南

### 前提条件
- Python 3.8+ 
- pip (Python包管理器)

### 安装步骤
1. 克隆仓库
```bash
git clone https://github.com/sujiayu/zhihui-rag.git
cd zhihui-rag
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

### 准备数据
将需要处理的文档放入`data/`目录下，支持以下格式：
- .md (Markdown)
- .txt (文本文件)
- .pdf (PDF文档)

### 运行应用
```bash
python demo.py
```

## 项目结构
```
zhihui-rag/
├── data/              # 文档存储目录
├── storage/           # 向量数据库存储
├── embedding_cache/   # 嵌入缓存
├── demo.py            # 主程序入口
├── document_agent.py  # 文档处理模块
├── llm.py             # 大语言模型集成
├── vecbase.py         # 向量存储管理
├── query_processor.py # 查询处理与扩展
├── utils.py           # 工具函数
├── requirements.txt   # 项目依赖
└── README.md          # 项目文档
```

## 技术栈
- **向量数据库**: FAISS
- **嵌入模型**: 未指定 (可在emb.py中配置)
- **大语言模型**: 未指定 (可在llm.py中配置)
- **文档处理**: 自定义ReadFiles类
- **编程语言**: Python 3.8+

## 常见问题
### Q: 如何添加新的文档格式支持？
A: 扩展document_agent.py中的文档解析器，添加新格式的处理逻辑

### Q: 向量搜索性能不佳怎么办？
A: 可在vecbase.py中调整索引类型(HNSW/IVF)或修改emb.py中的嵌入模型

## 贡献指南
1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开Pull Request


## 联系方式
项目维护者: [sujiayu]
邮箱: bbbbbbbb@bupt.edu.cn
