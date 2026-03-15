"""
图片上传 API
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import uuid

from services.segment import remove_background
from services.removebg import remove_background_api
from services.openai_compatible import analyze_clothes_openai
from services.vlm import detect_image_type_sync, analyze_full_body_items_sync
from services.embedding import embed_text, embed_image
from storage.config_store import load_config
from domain.clothes import ClothesSemantics
from storage.db import InMemoryWardrobeDB, get_wardrobe_db

router = APIRouter()

# 上传目录
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    上传衣物图片
    
    流程：
    1. 接收图片
    2. 根据配置使用 rembg 或 remove.bg API 去除背景
    3. 使用 LLM Vision 进行语义分析
    4. 转换为新格式并保存到数据库
    5. 返回衣物信息
    
    返回格式：
    {
        "success": true,
        "item_id": "物品ID",
        "item": {...}
    }
    """
    # 验证文件类型
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="只支持图片文件")
    
    try:
        # 读取原始图片
        raw_bytes = await file.read()
        
        # 加载配置
        config = load_config()

        # 先判断图片类型（全身照/单件衣物）
        image_type = detect_image_type_sync(raw_bytes)

        # 生成文件名并保存原图（用于全身照或追溯）
        raw_filename = f"{uuid.uuid4()}_raw.png"
        raw_filepath = UPLOAD_DIR / raw_filename
        with open(raw_filepath, "wb") as f:
            f.write(raw_bytes)
        raw_image_url = f"/uploads/{raw_filename}"

        # 全身照：多件解析 + 多条入库
        if image_type == "full_body":
            semantics_list = analyze_full_body_items_sync(raw_bytes)
            if not semantics_list:
                raise ValueError("全身照解析失败：未识别到衣物")

            db = get_wardrobe_db()
            items = []
            item_ids = []

            # 全身照共用同一张图的 embedding（如果未来做人体/单品分割，可替换为每件单品的裁剪图）
            # 使用 Qwen VLM + Embedding API 生成向量
            vector = embed_image(raw_bytes)

            for sem in semantics_list:
                storage_data = sem.to_storage_format()
                storage_data["basic_info"]["image_url"] = raw_image_url
                storage_data["vector_embedding"] = vector
                # 标记来源
                storage_data.setdefault("dynamic_metadata", {})
                storage_data["dynamic_metadata"].setdefault("source", {})
                storage_data["dynamic_metadata"]["source"].update(
                    {"image_type": "full_body", "image_url": raw_image_url}
                )

                item_id = db.add_item(storage_data)
                item = db.get_item(item_id)

                item_ids.append(item_id)
                items.append(item)

            return {
                "success": True,
                "image_type": "full_body",
                "item_ids": item_ids,
                "items": items,
                "message": f"成功从全身照中添加 {len(items)} 件衣物"
            }
        
        # 根据配置选择背景移除方式
        if config.bg_removal_method == "removebg" and config.removebg_api_key:
            # 使用 remove.bg API
            try:
                processed_bytes = await remove_background_api(
                    raw_bytes, 
                    config.removebg_api_key
                )
            except ValueError as e:
                # 如果 remove.bg 失败，回退到本地处理
                print(f"⚠️ remove.bg API 失败，回退到本地处理: {e}")
                processed_bytes = remove_background(raw_bytes)
        else:
            # 使用本地 rembg
            processed_bytes = remove_background(raw_bytes)
        
        # 使用 OpenAI 兼容 API 进行语义分析
        semantics: ClothesSemantics = await analyze_clothes_openai(processed_bytes)
        
        # 生成文件名并保存
        filename = f"{uuid.uuid4()}_nobg.png"
        filepath = UPLOAD_DIR / filename
        
        with open(filepath, "wb") as f:
            f.write(processed_bytes)
        
        # 转换为存储格式
        storage_data = semantics.to_storage_format()
        
        # 设置图片路径
        storage_data["basic_info"]["image_url"] = f"/uploads/{filename}"

        # 多模态向量化（使用 Qwen VLM + Embedding API）
        storage_data["vector_embedding"] = embed_image(processed_bytes)
        
        # 获取数据库并添加
        db = get_wardrobe_db()
        item_id = db.add_item(storage_data)
        
        # 获取完整数据
        item = db.get_item(item_id)
        
        return {
            "success": True,
            "image_type": "single_item",
            "item_id": item_id,
            "message": f"成功添加衣物: {storage_data['basic_info']['name']}",
            "item": item
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"图片分析失败: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@router.get("/items")
async def list_items(category: str = None, limit: int = 100):
    """
    获取衣物列表
    
    参数:
        category: 可选，按品类筛选
        limit: 返回数量限制
    """
    db = get_wardrobe_db()
    items = db.list_items(category=category, limit=limit)
    
    return {
        "total": len(items),
        "items": items
    }


@router.get("/items/{item_id}")
async def get_item(item_id: str):
    """获取单个衣物详情"""
    db = get_wardrobe_db()
    item = db.get_item(item_id)
    
    if not item:
        raise HTTPException(status_code=404, detail="未找到该衣物")
    
    return item
