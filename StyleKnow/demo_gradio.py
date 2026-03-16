# -*- coding: utf-8 -*-
"""
AI 穿搭助手 - Gradio 演示界面

展示项目的核心功能和能力：
1. 智能穿搭对话
2. 小红书灵感搜索
3. 衣橱检索
4. 用户偏好管理

使用异步调用 AsyncChatOpenAI 持久化连接，提升响应速度。
"""

import sys
import os
import asyncio

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from agent.graph import run_workflow_async
from agent.tools import search_xhs_tool, search_wardrobe_tool
from storage.db import get_wardrobe_db
from agent.ultis import init_wardrobe_db

# 初始化数据库
init_wardrobe_db()


# ========== 功能函数 ==========

async def chat_with_style_assistant_async(user_query: str, thread_id: str = "default"):
    """与穿搭助手对话（异步版本）"""
    if not user_query.strip():
        return "请输入您的穿搭需求...", ""

    try:
        result = await run_workflow_async(
            user_query=user_query,
            thread_id=thread_id
        )
        final_response = result.get("final_response", "暂无回复")
        iterations = result.get("iterations", 0)

        # 构建调试信息
        debug_info = f"迭代次数: {iterations}\n意图: {result.get('current_intent', '未知')}"

        return final_response, debug_info
    except Exception as e:
        return f"出错了: {str(e)}", ""


def chat_with_style_assistant(user_query: str, thread_id: str = "default"):
    """与穿搭助手对话（同步封装，供 Gradio 使用）"""
    if not user_query.strip():
        return "请输入您的穿搭需求...", ""

    try:
        # 使用 asyncio.run 调用异步函数
        result = asyncio.run(run_workflow_async(
            user_query=user_query,
            thread_id=thread_id
        ))
        final_response = result.get("final_response", "暂无回复")
        iterations = result.get("iterations", 0)

        # 构建调试信息
        debug_info = f"迭代次数: {iterations}\n意图: {result.get('current_intent', '未知')}"

        return final_response, debug_info
    except Exception as e:
        return f"出错了: {str(e)}", ""


def search_xhs_demo(query: str, category: str = None):
    """小红书灵感搜索演示"""
    if not query.strip():
        return "请输入搜索关键词..."
    
    try:
        result = search_xhs_tool.invoke({"query": query, "category": category})
        return result
    except Exception as e:
        return f"搜索失败: {str(e)}"


def search_wardrobe_demo(keywords: str, category: str = None):
    """衣橱检索演示"""
    if not keywords.strip():
        return "请输入检索关键词..."
    
    try:
        keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]
        result = search_wardrobe_tool.invoke({
            "keywords": keyword_list,
            "category": category if category != "不限" else None,
            "top_k": 5
        })
        return result
    except Exception as e:
        return f"检索失败: {str(e)}"


def show_wardrobe():
    """显示用户衣橱"""
    try:
        db = get_wardrobe_db()
        items = db.get_all_items()
        
        if not items:
            return "衣橱为空，请先添加衣物。"
        
        lines = ["📦 用户衣橱\n" + "=" * 40]
        
        # 按品类分组
        category_map = {}
        for item in items:
            cat = item.get("basic_info", {}).get("category", "未知")
            if cat not in category_map:
                category_map[cat] = []
            category_map[cat].append(item)
        
        for cat, items_list in category_map.items():
            lines.append(f"\n【{cat}】({len(items_list)}件)")
            for item in items_list[:5]:  # 每类最多显示5件
                name = item.get("basic_info", {}).get("name", "未命名")
                tags = item.get("semantic_tags", [])[:3]
                tags_str = "、".join(tags) if tags else "无"
                lines.append(f"  • {name} - {tags_str}")
        
        return "\n".join(lines)
    except Exception as e:
        return f"获取衣橱失败: {str(e)}"


def show_user_preferences(user_id: str = "default_user"):
    """显示用户偏好"""
    try:
        db = get_wardrobe_db()
        prefs = db.get_user_preference(user_id)
        
        if not prefs:
            return "暂无用户偏好记录。"
        
        lines = ["👤 用户偏好设置\n" + "=" * 40]
        
        default_scene = prefs.get("default_scene", "日常")
        lines.append(f"\n默认场景: {default_scene}")
        
        scenes = prefs.get("scenes", {})
        if scenes:
            for scene_name, scene_prefs in scenes.items():
                lines.append(f"\n【{scene_name}】")
                if not scene_prefs:
                    lines.append("  无偏好记录")
                for key, value in scene_prefs.items():
                    lines.append(f"  - {key}: {value}")
        
        return "\n".join(lines)
    except Exception as e:
        return f"获取偏好失败: {str(e)}"


def show_system_architecture():
    """展示系统架构"""
    return """
🏗️ AI 穿搭助手系统架构
══════════════════════════════════════

┌─────────────────────────────────────────────┐
│              用户交互层 (Gradio)             │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│              FastAPI / API Routes            │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│          LangGraph 工作流引擎                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────────┐  │
│  │ Router  │→ │ Stylist │→ │   Critic    │  │
│  │ (路由)   │  │ (穿搭师)  │  │   (评估)    │  │
│  └─────────┘  └────┬────┘  └─────────────┘  │
│                     │                        │
│              ┌──────▼──────┐                │
│              │   Tools     │                │
│              │  (工具执行)  │                │
│              └─────────────┘                │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│              工具层 (Tools)                   │
│  ┌────────────┐  ┌────────────┐              │
│  │ 小红书搜索  │  │  衣橱检索  │              │
│  └────────────┘  └────────────┘              │
│  ┌────────────┐  ┌────────────┐              │
│  │ 天气查询   │  │ 偏好管理   │              │
│  └────────────┘  └────────────┘              │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│              数据存储层 (Storage)              │
│  ┌────────────┐  ┌────────────┐              │
│  │  衣橱数据库  │  │ 推荐数据库  │              │
│  └────────────┘  └────────────┘              │
│  ┌────────────┐  ┌────────────┐              │
│  │  偏好存储   │  │ 配置存储   │              │
│  └────────────┘  └────────────┘              │
└────────────────────────────────────────────══

📌 核心特性：
• 基于 LangGraph 的 Supervisor 模式工作流
• 支持多轮对话和上下文记忆
• 混合检索（向量 + 关键词）
• 用户偏好学习与个性化推荐
• 小红书热门穿搭灵感获取
"""


# ========== Gradio 界面 ==========

def create_gradio_app():
    """创建 Gradio 应用"""
    
    # 自定义 CSS 样式
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .header-title {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .feature-card {
        padding: 15px;
        border-radius: 8px;
        background: #f8f9fa;
        border-left: 4px solid #667eea;
    }
    """
    
    with gr.Blocks(title="AI 穿搭助手 Demo", css=custom_css) as app:
        
        # 标题
        gr.Markdown("""
        <div class="header-title">
            <h1>🎨 AI 穿搭助手</h1>
            <p>基于 LLM 和多源检索的智能穿搭推荐系统</p>
        </div>
        """)
        
        with gr.Tab("💬 智能对话"):
            gr.Markdown("### 与 AI 穿搭助手对话")
            gr.Markdown("描述您的穿搭需求，如：")
            gr.Markdown("- \"明天上班穿什么好？\"")
            gr.Markdown("- \"帮我搭配一套通勤穿搭，要简约干练的\"")
            gr.Markdown("- \"周末去海边度假，推荐一下穿搭\"")
            
            with gr.Row():
                with gr.Column(scale=3):
                    chat_input = gr.Textbox(
                        label="输入您的穿搭需求",
                        placeholder="请描述您想要的穿搭...",
                        lines=3
                    )
                    thread_input = gr.Textbox(
                        label="对话线程ID（可选）",
                        value="default",
                        lines=1
                    )
                    chat_btn = gr.Button("发送", variant="primary")
                
                with gr.Column(scale=1):
                    debug_output = gr.Textbox(
                        label="调试信息",
                        lines=8,
                        interactive=False
                    )
            
            chat_output = gr.Textbox(
                label="AI 回复",
                lines=10,
                interactive=False
            )
            
            chat_btn.click(
                fn=chat_with_style_assistant,
                inputs=[chat_input, thread_input],
                outputs=[chat_output, debug_output]
            )
            
            # 回车提交
            chat_input.submit(
                fn=chat_with_style_assistant,
                inputs=[chat_input, thread_input],
                outputs=[chat_output, debug_output]
            )
        
        with gr.Tab("📱 小红书灵感"):
            gr.Markdown("### 搜索小红书热门穿搭")
            gr.Markdown("获取当前流行的穿搭灵感和趋势")
            
            with gr.Row():
                xhs_query = gr.Textbox(
                    label="搜索关键词",
                    placeholder="如：通勤穿搭 简约...",
                    lines=2
                )
                xhs_category = gr.Dropdown(
                    label="品类筛选（可选）",
                    choices=["不限", "上衣", "裤子", "裙子", "外套", "鞋子", "配饰"],
                    value="不限"
                )
            
            xhs_btn = gr.Button("搜索", variant="primary")
            xhs_output = gr.Textbox(
                label="搜索结果",
                lines=15,
                interactive=False
            )
            
            xhs_btn.click(
                fn=search_xhs_demo,
                inputs=[xhs_query, xhs_category],
                outputs=xhs_output
            )
        
        with gr.Tab("👔 我的衣橱"):
            gr.Markdown("### 检索用户衣橱")
            gr.Markdown("从个人衣物收藏中搜索符合条件的单品")
            
            with gr.Row():
                wardrobe_keywords = gr.Textbox(
                    label="检索关键词",
                    placeholder="输入关键词，用逗号分隔，如：通勤, 简约, 黑色",
                    lines=2
                )
                wardrobe_category = gr.Dropdown(
                    label="品类筛选",
                    choices=["不限", "上衣", "裤子", "裙子", "外套", "鞋子", "配饰"],
                    value="不限"
                )
            
            wardrobe_btn = gr.Button("检索", variant="primary")
            wardrobe_output = gr.Textbox(
                label="检索结果",
                lines=12,
                interactive=False
            )
            
            wardrobe_btn.click(
                fn=search_wardrobe_demo,
                inputs=[wardrobe_keywords, wardrobe_category],
                outputs=wardrobe_output
            )
            
            gr.Markdown("---")
            
            with gr.Row():
                gr.Markdown("### 查看完整衣橱")
                view_wardrobe_btn = gr.Button("显示衣橱", variant="secondary")
            
            wardrobe_view = gr.Textbox(
                label="衣橱内容",
                lines=15,
                interactive=False
            )
            
            view_wardrobe_btn.click(
                fn=show_wardrobe,
                inputs=[],
                outputs=wardrobe_view
            )
        
        with gr.Tab("⚙️ 用户偏好"):
            gr.Markdown("### 用户偏好管理")
            gr.Markdown("查看和管理用户的风格偏好设置")
            
            view_prefs_btn = gr.Button("查看用户偏好", variant="primary")
            prefs_output = gr.Textbox(
                label="用户偏好",
                lines=15,
                interactive=False
            )
            
            view_prefs_btn.click(
                fn=show_user_preferences,
                inputs=[],
                outputs=prefs_output
            )
        
        with gr.Tab("🏗️ 系统架构"):
            gr.Markdown("### 项目系统架构")
            arch_output = gr.Textbox(
                value=show_system_architecture(),
                lines=30,
                interactive=False
            )
        
        with gr.Tab("ℹ️ 关于项目"):
            gr.Markdown("""
            ## 🎨 AI 穿搭助手
            
            ### 项目简介
            这是一个基于 **LangGraph** 工作流的智能穿搭推荐系统，利用大语言模型（LLM）结合多源检索技术，为用户提供个性化的穿搭建议。
            
            ### 核心功能
            1. **智能穿搭对话** - 基于 LangGraph ReAct 模式，理解用户需求并给出搭配建议
            2. **小红书灵感搜索** - 实时获取热门穿搭趋势和灵感
            3. **衣橱智能管理** - 向量检索 + 关键词混合搜索个人衣物
            4. **偏好学习** - 记录并学习用户的风格偏好
            5. **天气辅助** - 结合天气信息给出更贴心的建议
            
            ### 技术栈
            - **LLM**: OpenAI / Claude (通过 302.ai API)
            - **工作流**: LangGraph (Supervisor 模式)
            - **向量检索**: ChromaDB
            - **API**: FastAPI
            - **前端**: Gradio
            - **数据存储**: 内存数据库 (可扩展)
            
            ### 工作流说明
            系统采用 Supervisor 模式的多 Agent 协作：
            1. **Router** - 意图识别，判断用户需求类型
            2. **Stylist** - 穿搭 Agent，调用工具生成搭配方案
            3. **Tools** - 工具执行节点（搜索、检索等）
            4. **Critic** - 评估 Agent，审核搭配方案的合理性
            """)
    
    return app


# ========== 启动入口 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("启动 AI 穿搭助手 Gradio 演示界面...")
    print("=" * 60)
    
    app = create_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
