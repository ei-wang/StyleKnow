# -*- coding: utf-8 -*-
"""
GraphState 定义

LangGraph 工作流的状态类型定义
"""

from typing import TypedDict, List, Dict, Any, Optional


class UserIntent(TypedDict):
    """用户意图"""
    intent: str                     # recommend/wardrobe_add/query
    has_image: bool                # 是否有图片
    scene: str                     # 场景关键词
    user_query: str                # 原始问题
    rewritten_query: str           # 重写后的问题
    reason: str                   # 识别理由


class UploadedImageInfo(TypedDict):
    """上传的图片信息"""
    image_url: str                # 图片路径
    item_id: str                  # 存储后的物品ID
    semantics: Dict                # VLM 分析结果


class SearchTask(TypedDict):
    """搜索任务定义"""
    category: str                  # 品类 (上衣/裤子/鞋子/配饰等)
    keywords: List[str]           # 语义标签关键词 (用于向量检索)
    search_query: str            # 搜索查询字符串 (用于小红书搜索)
    reason: str                   # 搜索理由


class WardrobeSearchResult(TypedDict):
    """衣柜搜索结果"""
    category: str                 # 品类
    search_task: SearchTask       # 原始搜索任务
    items: List[dict]            # 检索到的衣物


class XHSFeedItem(TypedDict):
    """小红书内容条目"""
    note_id: str                  # 笔记ID
    title: str                    # 标题
    desc: str                     # 正文描述
    liked_count: int              # 点赞数
    images: List[str]            # 图片URL列表
    formula: str                  # 提取的穿搭公式


class GraphState(TypedDict):
    """LangGraph 工作流状态定义"""
    
    # 用户输入
    user_query: str                      # 用户原始查询
    user_images: List[str]               # 用户上传的图片列表（可选）
    
    # 对话历史（多轮对话用）
    conversation_history: List[Dict[str, str]]  # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    
    # 意图识别结果
    intent: UserIntent                   # LLM 识别后的意图
    uploaded_images: List[UploadedImageInfo]  # 已上传处理的图片
    parsed_intent: Dict[str, Any]        # 解析后的意图（scene, rewritten_query, needed_categories 等）
    
    # 搜索任务 (由 LLM 生成)
    search_tasks: List[SearchTask]       # 搜索任务列表 (按品类分配)
    
    # 检索结果 (新流程：先小红书后衣柜)
    web_formulas: List[XHSFeedItem]     # 小红书穿搭公式 (先检索)
    xhs_extracted_outfits: List[Dict]    # 小红书 VLM 提取结果
    wardrobe_results: List[WardrobeSearchResult]  # 衣柜检索结果 (后检索)
    
    # 生成结果
    draft_outfit: str                    # 生成的搭配文案（草稿）
    
    # 评估结果
    critic_feedback: str                 # 批评者的反馈
    
    # 迭代控制
    iterations: int                      # 迭代次数
    
    # 最终输出
    final_outfit: str                    # 最终确定的搭配
    
    # 兜底推荐
    fallback_recommendations: List[Dict]  # 兜底推荐列表
    missing_recommendations: List[Dict]   # 缺失品类的推荐
