# -*- coding: utf-8 -*-
"""
AI 穿搭助手 - 独立测试脚本

不依赖 FastAPI，直接运行 LangGraph 工作流进行测试

使用方式：
    python test_agent.py
"""

import os
import sys

# 设置项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ========== 配置加载 ==========
from storage.config_store import load_config

# 加载配置并设置环境变量
config = load_config()

# 设置环境变量供其他模块使用
os.environ["OPENAI_API_KEY"] = config.api_key
os.environ["DASHSCOPE_API_KEY"] = config.api_key
os.environ["OPENAI_MODEL"] = config.model

# 移除背景配置
os.environ["REMOVEBG_API_KEY"] = config.removebg_api_key
os.environ["BG_REMOVAL_METHOD"] = config.bg_removal_method

# 天气 API 配置
os.environ["QWEATHER_API_KEY"] = config.qweather_api_key
os.environ["QWEATHER_API_HOST"] = config.qweather_api_host

# 小红书 API 配置
os.environ["XHS_API_URL"] = config.xhs_api_url
os.environ["XHS_API_KEY"] = config.xhs_api_key


# ========== 图片处理函数 ==========
async def process_uploaded_image(image_path: str) -> dict:
    """
    处理用户上传的图片（模拟 upload API 的流程）
    
    流程：
    1. 读取图片
    2. 判断图片类型（全身照/单件）
    3. 移除背景
    4. VLM 分析语义
    5. 存储到数据库
    6. 返回处理结果
    
    Returns:
        处理后的图片信息字典
    """
    from services.segment import remove_background
    from services.vlm import detect_image_type_sync, analyze_full_body_items_sync
    from services.openai_compatible import analyze_clothes_openai
    from services.embedding import embed_image
    from storage.db import get_wardrobe_db
    from domain.clothes import ClothesSemantics
    
    print(f"\n[Upload] 开始处理图片: {image_path}")
    
    # 读取图片
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    # 判断图片类型
    image_type = detect_image_type_sync(image_bytes)
    print(f"[Upload] 图片类型: {image_type}")
    
    db = get_wardrobe_db()
    
    if image_type == "full_body":
        # 全身照：识别多件衣物
        print(f"[Upload] 检测到全身照，开始识别多件衣物...")
        semantics_list = analyze_full_body_items_sync(image_bytes)
        
        if not semantics_list:
            raise ValueError("全身照解析失败：未识别到衣物")
        
        items = []
        item_ids = []
        
        # 使用 VLM + Embedding API 生成向量
        vector = embed_image(image_bytes)
        
        for sem in semantics_list:
            storage_data = sem.to_storage_format()
            storage_data["basic_info"]["image_url"] = image_path
            storage_data["vector_embedding"] = vector
            
            item_id = db.add_item(storage_data)
            item = db.get_item(item_id)
            
            item_ids.append(item_id)
            items.append(item)
            
            print(f"[Upload] 已存储: {storage_data['basic_info']['name']} ({sem.category})")
        
        # 返回全身照处理结果
        return {
            "image_type": "full_body",
            "image_url": image_path,
            "item_ids": item_ids,
            "items": items,
            "semantics_list": [
                {
                    "category": sem.category,
                    "item": sem.item,
                    "color": sem.color_semantics,
                    "style_semantics": sem.style_semantics,
                    "season_semantics": sem.season_semantics,
                    "usage_semantics": sem.usage_semantics,
                    "material": sem.material,
                    "pattern": sem.pattern,
                    "fit": sem.fit,
                    "description": sem.description,
                    "details": sem.details
                }
                for sem in semantics_list
            ]
        }
    
    else:
        # 单件衣物
        # 移除背景
        processed_bytes = remove_background(image_bytes)
        
        # VLM 分析
        semantics: ClothesSemantics = await analyze_clothes_openai(processed_bytes)
        
        # 转换为存储格式
        storage_data = semantics.to_storage_format()
        storage_data["basic_info"]["image_url"] = image_path
        
        # 多模态向量化
        storage_data["vector_embedding"] = embed_image(processed_bytes)
        
        # 存储到数据库
        item_id = db.add_item(storage_data)
        item = db.get_item(item_id)
        
        print(f"[Upload] 已存储: {storage_data['basic_info']['name']} ({semantics.category})")
        
        return {
            "image_type": "single_item",
            "image_url": image_path,
            "item_id": item_id,
            "item": item,
            "semantics": {
                "category": semantics.category,
                "item": semantics.item,  # 具体款式，如"抓绒衣"
                "color": semantics.color_semantics,
                "style_semantics": semantics.style_semantics,  # 风格标签列表
                "season_semantics": semantics.season_semantics,  # 季节标签列表
                "usage_semantics": semantics.usage_semantics,  # 使用场景列表
                "material": semantics.material,  # 材质
                "pattern": semantics.pattern,  # 图案
                "fit": semantics.fit,  # 版型
                "description": semantics.description,
                "details": semantics.details
            }
        }


# ========== 测试用例 ==========
TEST_CASES = [
    # {
    #     "name": "基础穿搭推荐",
    #     "query": "明天去上班怎么穿？"
    # },
    # {
    #     "name": "目的地+天气+天数",
    #     "query": "准备后天去三亚玩3天，每天怎么穿搭啊"
    # },
    # {
    #     "name": "明确地点和时间",
    #     "query": "下周五去北京出差，需要穿什么衣服"
    # },
    # {
    #     "name": "场景+风格",
    #     "query": "周末约会有什么穿搭推荐吗"
    # },
    {
        "name": "带图片查询",
        "query": "这件衣服配什么裤子好看？",
        "images": ["output.png"]  # 需要实际图片路径
    }
]


def run_test(query: str, images: list = None):
    """运行单次测试"""
    print("\n" + "=" * 70)
    print(f"📋 测试: {query}")
    print("=" * 70)
    
    # 初始化数据库
    from agent.ultis import init_wardrobe_db
    init_wardrobe_db()
    
    # 处理上传的图片（如果有）
    uploaded_images = []
    
    if images:
        print(f"\n[Upload] 检测到 {len(images)} 张图片，开始处理...")
        import asyncio
        for img_path in images:
            try:
                result = asyncio.run(process_uploaded_image(img_path))
                
                # 转换为 uploaded_images 格式
                if result["image_type"] == "full_body":
                    for i, semantics in enumerate(result["semantics_list"]):
                        uploaded_images.append({
                            "image_url": result["image_url"],
                            "item_id": result["item_ids"][i],
                            "semantics": {
                                "category": semantics["category"],
                                "item": semantics.get("item", ""),
                                "color": semantics.get("color", []),
                                "style_semantics": semantics.get("style_semantics", []),
                                "season_semantics": semantics.get("season_semantics", []),
                                "usage_semantics": semantics.get("usage_semantics", []),
                                "material": semantics.get("material", ""),
                                "pattern": semantics.get("pattern", ""),
                                "fit": semantics.get("fit", ""),
                                "description": semantics.get("description", ""),
                                "details": semantics.get("details", {})
                            },
                            "image_type": "clothing"
                        })
                else:
                    uploaded_images.append({
                        "image_url": result["image_url"],
                        "item_id": result["item_id"],
                        "semantics": result["semantics"],
                        "image_type": "clothing"
                    })
                
                print(f"[Upload] 图片处理完成: {result.get('item_id') or result.get('item_ids')}")
                
            except Exception as e:
                print(f"[Upload] 图片处理失败: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"[Upload] 共处理 {len(uploaded_images)} 件衣物")
    
    try:
        # 导入工作流
        from agent.graph import run_workflow
        
        # 运行工作流
        result = run_workflow(
            user_query=query,
            user_images=images or [],
            uploaded_images=uploaded_images
        )
        
        # 打印结果
        print("\n✅ 运行成功!\n")
        
        # 意图识别结果
        if result.get("intent"):
            print("🎯 意图识别:")
            print(f"   - 类型: {result['intent'].get('intent')}")
            print(f"   - 场景: {result['intent'].get('scene')}")
            print(f"   - 重写后: {result['intent'].get('rewritten_query', '')}")
        
        # 天气信息（如果有）
        if result.get("weather_info"):
            w = result["weather_info"]
            print(f"\n🌤️ 天气信息:")
            print(f"   - 地点: {w.get('location')}")
            print(f"   - 温度: {w.get('weather', {}).get('temperature')}°C")
            print(f"   - 天气: {w.get('weather', {}).get('condition')}")
            print(f"   - 天数: {w.get('trip_days')}天")
        
        # 小红书公式
        if result.get("web_formulas"):
            print(f"\n📕 小红书公式 ({len(result['web_formulas'])} 条):")
            for i, formula in enumerate(result["web_formulas"][:3], 1):
                print(f"   {i}. {formula.get('title', '')[:40]}...")
        
        # 衣柜检索
        if result.get("wardrobe_results"):
            print(f"\n👔 衣柜检索 ({len(result['wardrobe_results'])} 个品类):")
            for r in result["wardrobe_results"]:
                items = r.get("items", [])
                print(f"   - {r.get('category')}: {len(items)} 件")
        
        # 最终搭配
        if result.get("final_outfit"):
            print(f"\n✨ 最终搭配:")
            print(result["final_outfit"])
        elif result.get("draft_outfit"):
            print(f"\n📝 搭配草稿:")
            print(result["draft_outfit"])
        
        # 评估反馈
        if result.get("critic_feedback"):
            print(f"\n💬 评估反馈:")
            print(result["critic_feedback"])
        
        return result
        
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_tests():
    """运行所有测试用例"""
    # 打印配置信息
    print("=" * 70)
    print("🚀 AI 穿搭助手 - 多 Agent 工作流测试")
    print("=" * 70)
    print(f"\n📋 当前配置:")
    print(f"   - API Base: {config.api_base}")
    print(f"   - Model: {config.model}")
    print(f"   - API Key: {config.api_key[:8]}..." if config.api_key else "   - API Key: 未设置")
    print(f"   - 天气 API: {'已设置' if config.qweather_api_key else '未设置'}")
    print(f"   - 小红书 API: {'已设置' if config.xhs_api_key else '未设置'}")
    print(f"   - remove.bg: {config.removebg_api_key[:8]}..." if config.removebg_api_key else "   - remove.bg: 未设置")
    print()
    
    # 检查 API Key
    if not config.api_key:
        print("⚠️  警告: 未设置 API Key，部分功能可能使用 Mock 模式\n")
    
    # 运行测试
    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n\n{'='*70}")
        print(f"📌 测试 {i}/{len(TEST_CASES)}: {case['name']}")
        print(f"{'='*70}")
        
        result = run_test(case["query"], case.get("images"))
        
        # 可以在这里添加结果验证逻辑
        if result:
            print(f"\n✅ 测试通过!")
        else:
            print(f"\n❌ 测试失败!")
        
        # 每次测试后暂停一下
        if i < len(TEST_CASES):
            input("\n按 Enter 继续下一个测试...")


def test_single_query():
    """交互式单次查询"""
    print("=" * 70)
    print("🚀 AI 穿搭助手 - 交互式查询")
    print("=" * 70)
    print("\n输入你的穿搭问题（或输入 'q' 退出）:\n")
    
    while True:
        query = input("👤 你: ").strip()
        
        if not query:
            continue
        if query.lower() in ['q', 'quit', 'exit']:
            print("\n👋 再见!")
            break
        
        run_test(query)
        print()


# ========== 主入口 ==========
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI 穿搭助手测试")
    parser.add_argument("--query", "-q", type=str, help="直接运行单次查询")
    parser.add_argument("--all", "-a", action="store_true", help="运行所有测试用例")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互式查询模式")
    
    args = parser.parse_args()
    
    # if args.query:
    #     # 单次查询
    #     run_test(args.query)
    # elif args.all:
    #     # 运行所有测试
    run_all_tests()
    # else:
    #     # 默认：交互式
    #     test_single_query()
