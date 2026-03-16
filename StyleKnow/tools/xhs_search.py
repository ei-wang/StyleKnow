import requests
import httpx
from typing import List, Dict, Optional
import json

def clean_xhs_response(raw_json: dict) -> List[dict]:
    """
    清洗小红书 API 返回的原始 JSON 数据
    
    提取规则：
    1. 从 data.data.items 列表中提取每个 note
    2. 只保留字段：id, title, desc, liked_count
    3. 从 images_list 中提取前两张高清图片 URL
    
    Args:
        raw_json: 小红书 API 返回的原始 JSON 字典
    
    Returns:
        清洗后的笔记列表，每条包含：
        - id: 笔记 ID
        - title: 标题
        - desc: 正文描述
        - liked_count: 点赞数
        - images: 前两张高清图片 URL 列表
    """
    cleaned_items = []
    
    try:
        items = raw_json.get("data", {}).get("data", {}).get("items", [])
        
        for item in items:
            note = item.get("note", {})
            if not note:
                continue
            
            note_id = note.get("id", "")
            title = note.get("title", "")
            desc = note.get("desc", "")
            liked_count = note.get("liked_count", 0)
            
            # 提取前两张高清图片
            images_list = note.get("images_list", [])
            images = []
            
            for idx, img in enumerate(images_list[:2]):
                img_url = img.get("url") or img.get("url_size_large", "")
                if img_url:
                    images.append(img_url)
            
            cleaned_note = {
                "id": note_id,
                "title": title,
                "desc": desc,
                "liked_count": liked_count,
                "images": images
            }
            
            cleaned_items.append(cleaned_note)
    
    except Exception as e:
        print(f"[ERROR] 清洗小红书数据失败: {e}")
    
    print(f"[INFO] 清洗完成，共提取 {len(cleaned_items)} 篇笔记")
    return cleaned_items


# ========== 真实小红书搜索 API (302.ai) ==========
def search_xhs_notes(
    keyword: str,
    page: int = 1,
    sort_type: str = "general",
    filter_note_time: str = "不限",
    filter_note_type: str = "不限",
    search_id: str = None,
    session_id: str = None
) -> dict:
    """
    使用 302.ai API 调用小红书搜索（GET 请求）
    
    Args:
        keyword: 搜索关键词
        page: 页码 (默认 1)
        sort_type: 排序方式 (general/popularity_descending/time_descending)
        filter_note_time: 时间筛选 (不限/一天内/一周内/半年内)
        filter_note_type: 类型筛选 (不限/视频笔记/普通笔记)
        search_id: 搜索ID，翻页时需要
        session_id: 会话ID，翻页时需要
    
    Returns:
        小红书 API 响应的原始 JSON
    """
    from storage.config_store import load_config
    config = load_config()
    xhs_api_key = config.xhs_api_key
    xhs_api_url = config.xhs_api_url
    
    headers = {
        "Authorization": f"Bearer {xhs_api_key}",
        "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
        "Accept": "*/*",
        "Cache-Control": "no-cache"
    }
    
    # 构建查询参数（GET 请求）
    params = {
        "keyword": keyword,
        "sort_type": sort_type,
        "page": page,
        "filter_note_type": filter_note_type,
        "filter_note_time": filter_note_time
    }
    
    # 添加可选参数
    if search_id:
        params["search_id"] = search_id
    if session_id:
        params["session_id"] = session_id
    
    try:
        response = requests.get(xhs_api_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        print(f"[INFO] 小红书搜索成功: keyword='{keyword}', page={page}")
        return result
        
    except requests.exceptions.Timeout:
        print(f"[ERROR] 小红书 API 请求超时")
        return {"code": -1, "msg": "请求超时", "data": {"items": []}}
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 小红书 API 请求失败: {e}")
        return {"code": -1, "msg": str(e), "data": {"items": []}}


def search_xhs_notes_by_task(
    search_query: str,
    scene: str = "commute",
    top_k: int = 5
) -> Dict:
    """
    按 LLM 分配的搜索任务搜索小红书（新版本使用）
    
    小红书搜索通常是整体穿搭搜索，不是按品类分开搜索
    
    Args:
        search_query: LLM 生成的搜索关键词
        scene: 场景名称（用于参考）
        top_k: 返回数量
    
    Returns:
        搜索结果:
        {
            "search_query": str,
            "scene": str,
            "total": int,
            "notes": List[dict]
        }
    
    Examples:
        >>> search_xhs_notes_by_task("通勤穿搭 简约干练", "commute", top_k=5)
    """
    # 执行搜索
    raw_result = search_xhs_notes(
        keyword=search_query,
        sort_type="general",
        filter_note_time="半年内"
    )
    
    # 清洗数据
    notes = clean_xhs_response(raw_result)
    
    # 限制数量
    notes = notes[:top_k]
    
    return {
        "search_query": search_query,
        "scene": scene,
        "total": len(notes),
        "notes": notes
    }


# ========== 异步版本 ==========
async def search_xhs_notes_async(
    keyword: str,
    page: int = 1,
    sort_type: str = "general",
    filter_note_time: str = "不限",
    filter_note_type: str = "不限",
    search_id: str = None,
    session_id: str = None
) -> dict:
    """
    使用 302.ai API 调用小红书搜索（异步版本）

    Args:
        keyword: 搜索关键词
        page: 页码 (默认 1)
        sort_type: 排序方式
        filter_note_time: 时间筛选
        filter_note_type: 类型筛选
        search_id: 搜索ID，翻页时需要
        session_id: 会话ID，翻页时需要

    Returns:
        小红书 API 响应的原始 JSON
    """
    from storage.config_store import load_config
    config = load_config()
    xhs_api_key = config.xhs_api_key
    xhs_api_url = config.xhs_api_url

    headers = {
        "Authorization": f"Bearer {xhs_api_key}",
        "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
        "Accept": "*/*",
        "Cache-Control": "no-cache"
    }

    params = {
        "keyword": keyword,
        "sort_type": sort_type,
        "page": page,
        "filter_note_type": filter_note_type,
        "filter_note_time": filter_note_time
    }

    if search_id:
        params["search_id"] = search_id
    if session_id:
        params["session_id"] = session_id

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(xhs_api_url, params=params, headers=headers)
            response.raise_for_status()
            result = response.json()
            print(f"[INFO] 小红书搜索成功(异步): keyword='{keyword}', page={page}")
            return result

    except httpx.TimeoutException:
        print(f"[ERROR] 小红书 API 请求超时(异步)")
        return {"code": -1, "msg": "请求超时", "data": {"items": []}}
    except httpx.HTTPStatusError as e:
        print(f"[ERROR] 小红书 API HTTP错误(异步): {e}")
        return {"code": -1, "msg": str(e), "data": {"items": []}}
    except Exception as e:
        print(f"[ERROR] 小红书 API 请求失败(异步): {e}")
        return {"code": -1, "msg": str(e), "data": {"items": []}}


async def search_xhs_notes_by_task_async(
    search_query: str,
    scene: str = "commute",
    top_k: int = 5
) -> Dict:
    """
    按 LLM 分配的搜索任务搜索小红书（异步版本）

    Args:
        search_query: LLM 生成的搜索关键词
        scene: 场景名称
        top_k: 返回数量

    Returns:
        搜索结果字典
    """
    raw_result = await search_xhs_notes_async(
        keyword=search_query,
        sort_type="general",
        filter_note_time="半年内"
    )

    notes = clean_xhs_response(raw_result)
    notes = notes[:top_k]

    return {
        "search_query": search_query,
        "scene": scene,
        "total": len(notes),
        "notes": notes
    }
