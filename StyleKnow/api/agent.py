# -*- coding: utf-8 -*-
"""
AI 穿搭助手 Agent 入口

提供统一的 Agent 运行接口
"""

from typing import Dict, Any, Optional, List

# 导入核心模块
from agent.graph import run_workflow as _run_workflow, build_workflow as _build_workflow


class FashionAgent:
    """
    穿搭推荐 Agent 主类
    
    提供简洁的接口来运行穿搭推荐工作流
    """
    
    def run(self, user_query: str, thread_id: str = "default") -> Dict[str, Any]:
        """
        运行穿搭推荐工作流
        
        Args:
            user_query: 用户查询，如 "帮我搭配一套通勤穿搭"
            thread_id: 线程ID，用于多轮对话记忆
        
        Returns:
            包含推荐结果的字典
        """
        return _run_workflow(user_query, thread_id=thread_id)
    
    def run_with_context(
        self,
        user_query: str,
        location: Optional[str] = None,
        include_weather: bool = False
    ) -> Dict[str, Any]:
        """
        带上下文的穿搭推荐
        
        Args:
            user_query: 用户查询
            location: 位置（用于天气查询）
            include_weather: 是否包含天气信息
        
        Returns:
            包含推荐结果和天气的字典
        """
        result = self.run(user_query)
        
        if include_weather and location:
            from tools.weather import get_weather_with_suggestion
            weather = get_weather_with_suggestion(location)
            result["weather"] = weather
        
        return result
    
    def get_workflow(self):
        """获取编译后的工作流"""
        return _build_workflow()


# 全局单例
_agent: Optional[FashionAgent] = None


def get_agent() -> FashionAgent:
    """获取全局 Agent 实例"""
    global _agent
    if _agent is None:
        _agent = FashionAgent()
    return _agent


def run_fashion_recommendation(user_query: str) -> Dict[str, Any]:
    """
    便捷函数：运行穿搭推荐
    
    Args:
        user_query: 用户查询
    
    Returns:
        推荐结果
    """
    agent = get_agent()
    return agent.run(user_query)


# ========== CLI 入口 ==========
if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("AI 穿搭助手 Agent")
    print("=" * 60)
    
    # 获取 Agent
    agent = get_agent()
    
    # 如果有命令行参数，执行推荐
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\n执行推荐: {query}")
        result = agent.run(query)
        print("\n结果:")
        print(result.get("final_response", "无结果"))
    else:
        # 默认测试
        print("\n运行默认测试...")
        result = agent.run("帮我搭配一套通勤穿搭")
        print("\n结果:")
        print(result.get("final_response", "无结果"))
