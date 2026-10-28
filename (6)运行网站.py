import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

BASE_DIR = Path(__file__).resolve().parent
SCORE_MODULE_PATH = BASE_DIR / "(5)计算得分.py"
HOME_TEMPLATE_PATH = BASE_DIR / "templates" / "index.html"


def load_score_module() -> ModuleType:
    if not SCORE_MODULE_PATH.exists():
        raise RuntimeError(f"未找到核心算力模型文件，请确保该文件存在: {SCORE_MODULE_PATH}")
        
    spec = importlib.util.spec_from_file_location("score_engine", SCORE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法解析评分模块结构：{SCORE_MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# 加载多维算法推理引擎
score_engine = load_score_module()
TopicRequest = score_engine.TopicRequest
EvaluationResponse = score_engine.EvaluationResponse

app = FastAPI(
    title="短视频选题评估 DSS - 智能决策支持系统",
    description="深度耦合定量计算模型与定性大语言模型推理(DeepSeek)"
)


@app.get("/", response_class=HTMLResponse, tags=["前端页面"])
async def render_home_page():
    if not HOME_TEMPLATE_PATH.exists():
        return """
        <html>
            <head><title>DSS 预警</title></head>
            <body style='font-family: sans-serif; padding: 50px; text-align: center;'>
                <h2>⚠️ 未检测到系统的前端看板网页文件 index.html</h2>
                <p>请确认项目根目录下是否存在 <code>templates/index.html</code> 目录结构及对应模板文件。</p>
            </body>
        </html>
        """
    with open(HOME_TEMPLATE_PATH, "r", encoding="utf-8") as file:
        return file.read()


@app.post("/api/v1/evaluate", tags=["决策评估"], summary="原有公式算分 + 联动 DeepSeek 给诊断优化建议")
async def api_evaluate_topic(request: TopicRequest):
    """
    网关 1：调用原有模型严格算分，并安全叠加 AI 意见列表返回给前端（确保 total_score 字段完整不丢失）
    """
    try:
        # 1. 运算原有客观定量分析模型
        response_data = score_engine.evaluate_topic_dss(request)
        
        # 2. 🧠 调用 DeepSeek 动态抽取更高水准的定性诊断，作为优化项返回给前端列表
        if hasattr(score_engine, "generate_ai_optimization_advices"):
            ai_advices = score_engine.generate_ai_optimization_advices(
                request.title, request.domain, response_data.score_breakdown
            )
            if ai_advices:
                # 融合 AI 意见与本地备份提示，依然由最外层的 optimizations 数组包裹
                response_data.optimizations = ai_advices + response_data.optimizations

        # 🔒 核心修复点：直接返回标准扁平 Pydantic 响应，让前端能 100% 成功解析到 response.total_score
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"决策引擎核心求解单元发生突发异常: {str(e)}")


@app.post("/api/v1/script", tags=["AI智能改写"], summary="多维特征重编 + 真实运行 DeepSeek 脚本生成")
async def api_generate_script(request: TopicRequest):
    """
    网关 2：一键重写脚本
    """
    try:
        if hasattr(score_engine, "generate_ai_script_via_deepseek"):
            try:
                real_llm_result = score_engine.generate_ai_script_via_deepseek(request)
                return {
                    "status": "ok",
                    "source": "DeepSeek-V3-Realtime",
                    "result": real_llm_result
                }
            except Exception as deepseek_err:
                print(f"⚠️ 真实 DeepSeek 调用失败: {str(deepseek_err)}。系统已平滑切换为安全防御型 Mock 逻辑。")

        # 🔒 兜底 Mock 机制
        mock_hook = f"你是不是也觉得{request.target_audience}在{request.domain}里面踩过太多坑了？别眨眼，看完这条视频帮你省下一大笔钱！"
        if "医保" in request.title:
            mock_hook = f"2026年最新医保重大变动！好多{request.target_audience}都不知道，原来这样能多报几千块！"

        mock_script = (
            f"【黄金开头 - {request.hook_type}】\n{mock_hook}\n\n"
            f"【核心干货阐述 - 严格遵循 {request.content_structure} 结构模型】\n"
            f"第一步：千万别盲目去柜台，先登录确认核心标签绑定状态...\n"
            f"第二步：在手机后台搜索打开互联功能...\n"
            f"第三步：严格控制在 {request.duration_sec} 秒内以契合大盘完播率。\n\n"
            f"【结尾互动转化 - 运营目标：{request.content_goal}】\n"
            f"这个新规太实用了，赶紧把它【分享】给你关心的那个人，或者先【收藏】起来免得找不到！"
        )

        return {
            "status": "ok",
            "source": "DSS-Local-Defensive-Mock",
            "result": {
                "title": f"【DSS策略调优版】{request.title}",
                "opening_hook": mock_hook,
                "full_script": mock_script,
                "shot_plan": [
                    "全景：主讲人手持手机道具语气激昂切入，上方迅速打出核心吸睛钩子词",
                    "特写：切至手机界面高清录屏，红圈高亮展示标准化关联操作指南",
                    "中景：主讲人面对镜头，手势强化“转发、收藏”核心转化交付意图"
                ]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"大模型重写单元彻底响应失败: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "(6)运行网站:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )