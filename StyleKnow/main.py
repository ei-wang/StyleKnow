# -*- coding: utf-8 -*-
"""
AI 穿搭助手 - 主入口

FastAPI 应用入口
"""

from fastapi import FastAPI
from api.routes import router

# 创建 FastAPI 应用
app = FastAPI(
    title="AI 穿搭助手 API",
    description="基于 LLM 和多源检索的智能穿搭推荐系统",
    version="1.0.0"
)

# 注册路由
app.include_router(router, prefix="/api", tags=["main"])


# ========== 启动入口 ==========
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("启动 AI 穿搭助手 API...")
    print("=" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
