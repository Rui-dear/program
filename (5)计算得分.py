import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

import jieba
import mysql.connector
from pydantic import BaseModel, Field
from snownlp import SnowNLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests   # 新增：用于调用外部 API

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")


def load_env_file(path: str = ENV_PATH):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file()

# 数据库配置（请根据实际环境调校）
DB_KB_CONFIG = {
    "host": "localhost",
    "database": "dds知识库",
    "user": "root",
    "password": "Yjj2004090",
}

DB_CB_CONFIG = {
    "host": "localhost",
    "database": "dds案例库",
    "user": "root",
    "password": "Yjj2004090",
}


# ==================== Pydantic 模型定义 ====================
class TopicRequest(BaseModel):
    title: str = Field(..., description="视频拟发布标题")
    domain: str = Field("知识科普", description="赛道领域")
    content_draft: Optional[str] = Field("", description="内容草稿/提纲摘要")
    publish_platform: str = Field("抖音", description="发布平台")
    target_audience: str = Field("大学生", description="目标受众")
    content_goal: str = Field("点赞", description="内容目标")
    content_structure: str = Field("总分总", description="内容结构")
    duration_sec: int = Field(60, description="视频预计时长")
    hook_type: str = Field("痛点直击", description="黄金开头钩子类型")
    scenario: str = Field("通勤", description="典型观看场景")
    tags: List[str] = Field(default_factory=list, description="自由标签组")
    core_entities: List[str] = Field(default_factory=list, description="核心实体词")
    risk_level: str = Field("低", description="风控初始预判等级")
    
    # 答辩优化：此处的四个字段不再由前端滑块自由决定分数，而是接收前端的 AHP 微调乘子或由后端算法完全重写
    emotion: float = Field(0.35, description="情感倾向 AHP 固定权重系数")
    hot: float = Field(0.30, description="热点匹配 AHP 固定权重系数")
    history: float = Field(0.20, description="历史成功 AHP 固定权重系数")
    timeliness: float = Field(0.15, description="内容时效 AHP 固定权重系数")


class EvaluationResponse(BaseModel):
    final_score: float = Field(..., description="由四项指标结合 AHP 固定权重自动加权求得的最终决策分")
    cbr_score: float = Field(..., description="案例库推演得分")
    rbr_score: float = Field(..., description="知识库/规则引擎得分")
    cbr_weight: float = Field(..., description="CBR 动态分配权重")
    rbr_weight: float = Field(..., description="RBR 动态分配权重")
    max_similarity: float = Field(..., description="与案例库匹配的最大相似度")
    score_breakdown: Dict[str, float] = Field(..., description="四项核心指标计算得分详情 (0-100)")
    top_cases: List[Dict[str, Any]] = Field(default_factory=list, description="匹配到的 Top5 历史爆款案例")
    optimizations: List[str] = Field(default_factory=list, description="RBR 规则生成的优化建议")
    triggered_rules: List[Dict[str, Any]] = Field(default_factory=list, description="触发的触发规则列表")
    ai_generated: Dict[str, Any] = Field(default_factory=dict, description="AI 生成的优化建议及脚本预留结构")


# ==================== 算法核心层：四项指标自动计算 ====================

def calc_emotion_score(text: str) -> float:
    """
    第一项：情感倾向 (35%) -> 基于 SnowNLP 情感极性分析
    """
    if not text.strip():
        return 50.0
    try:
        s = SnowNLP(text)
        # sentiments 返回 0~1 之间的极性分，越接近 1 越积极，越接近 0 越消极
        # 短视频文案通常带有明显的情感偏向（不论极度焦虑还是极度亢奋都是高情感唤醒）
        # 这里直接转为百分制，并对极端中庸情况做保底
        score = s.sentiments * 100
        return round(score, 2)
    except Exception:
        return 60.0  # 默认保底分


def calc_hot_score(text: str) -> float:
    """
    第二项：热点匹配度 (30%) -> 基于本地行业热点词库的 Jaccard 相似度计算
    """
    # 答辩级配置：预设的平台垂直爆款热点词/高流量实体库
    HOT_WORDS = [
        "医保", "公积金", "社保", "育儿", "儿童", "AI", "高考", "考研", "新规",
        "补贴", "省钱", "内卷", "搞钱", "副业", "红利", "真相", "避坑", "大厂", "干货"
    ]
    if not text.strip():
        return 0.0
    
    # 混合用户传入的 core_entities 进行热度扩充提分
    tokens = set(jieba.lcut(text))
    hot_set = set(HOT_WORDS)
    
    intersection = tokens & hot_set
    union = tokens | hot_set
    
    if not union:
        return 50.0
        
    jaccard = len(intersection) / len(union)
    # Jaccard 通常基数极小，根据短视频标题特征，给予 600 倍系数放大，上限封顶 100，保底 50分（基础大盘热度）
    score = min(50.0 + (jaccard * 600), 100.0)
    return round(score, 2)


def calc_history_score(current_title: str) -> tuple[float, List[Dict[str, Any]]]:
    """
    第三项：历史相似成功度 (20%) -> 联通 MySQL 案例库，基于 TF-IDF + 余弦相似度计算
    """
    top_cases = []
    max_sim = 0.0
    
    try:
        conn = mysql.connector.connect(**DB_CB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        # 修改后的查询：从 Video_Case_Base 和 Video_Case_Metrics 联合查询
        cursor.execute("""
            SELECT b.CaseId as id, b.Title as title, b.Domain as domain, m.CompletionRate as completion_rate
            FROM Video_Case_Base b
            JOIN Video_Case_Metrics m ON b.CaseId = m.CaseId
            LIMIT 100
        """)
        db_cases = cursor.fetchall()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"案例库连接失败: {e}")
        # 如果数据库未连接，降级返回模拟的答辩展示数据
        db_cases = [
            {"id": 1, "title": "医保报销全流程，这样操作多报销50%", "domain": "知识科普", "completion_rate": 0.65},
            {"id": 2, "title": "2026年最新社保补缴新规解读", "domain": "财经商业", "completion_rate": 0.58},
            {"id": 3, "title": "公积金提取的隐藏秘密，千万别做错", "domain": "知识科普", "completion_rate": 0.72}
        ]

    if db_cases and current_title.strip():
        titles = [c["title"] for c in db_cases]
        corpus = [current_title] + titles
        
        # 建立 TF-IDF 矩阵并计算第一项（当前输入）与其余案例的余弦相似度
        vectorizer = TfidfVectorizer(tokenizer=jieba.lcut)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # 计算当前输入(0) 与其余所有历史爆款的相似度
        sim_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        sim_scores = sim_matrix[0]
        
        # 将相似度关联回案例中
        for idx, score in enumerate(sim_scores):
            db_cases[idx]["similarity"] = float(score)
            
        # 降序排列获取前 5 个最贴近的爆款参照物
        db_cases.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        top_cases = db_cases[:5]
        
        if top_cases:
            max_sim = top_cases[0]["similarity"]
            
    # 将最大相似度映射为百分制作为历史成功度得分，没有匹配到则给 50 分及格分
    history_score = max_sim * 100 if max_sim > 0 else 55.0
    return round(history_score, 2), top_cases


def calc_timeliness_score(text: str) -> float:
    """
    第四项：内容时效性 (15%) -> 时间衰减敏感词权重矩阵检测
    """
    TIME_WORDS = {
        "今天": 100, "刚刚": 100, "最新": 98, "新规": 95, "今年": 88,
        "2026": 95, "本月": 82, "近期": 75, "最近": 70, "长期": 45, "历史": 30
    }
    
    score = 60.0  # 大盘常规常态化选题默认时效分 (及格)
    for word, val in TIME_WORDS.items():
        if word in text:
            score = max(score, val)
            
    return round(score, 2)


# ==================== 仿真知识库规则引擎 (RBR 支撑) ====================
def evaluate_rbr_logic(request: TopicRequest) -> Dict[str, Any]:
    """
    模拟 RBR 规则库诊断，为答辩输出完备的扣分规则链条与优化建议
    """
    triggered = []
    advices = []
    rbr_score = 85.0  # 基础规则满分
    
    # 规则1：字数审查
    if len(request.title) < 12:
        rbr_score -= 10
        triggered.append({"rule_name": "标题字数过短警告", "reason": "标题字数少于12字，在短视频信息流中无法建立足够的悬念与交付预期。"})
        advices.append("建议扩充标题，补充利益点或具体受众群体。")
    elif len(request.title) > 35:
        rbr_score -= 5
        triggered.append({"rule_name": "标题过长冗余", "reason": "标题超过35字，尾部关键核心实体极易在主页被截断展示。"})
        advices.append("精简标题文字，将核心钩子向句首提炼。")
        
    # 规则2：极端词汇风控拦截
    danger_words = ["绝对", "100%", "暴富", "一定", "稳赚不赔", "不看后悔"]
    found_danger = [w for w in danger_words if w in request.title or w in request.content_draft]
    if found_danger:
        penalty = min(len(found_danger) * 8, 20)
        rbr_score -= penalty
        triggered.append({"rule_name": "平台绝对化违规词拦截", "reason": f"检测到包含高风险词汇: {found_danger}，极易引发平台限流或下架。"})
        advices.append("立刻剔除文案中过于绝对化的承诺，改用中性词或叙述性句式替代。")
        
    # 规则3：钩子结构与场景契合审查
    if request.hook_type == "痛点直击" and "？" not in request.title and "?" not in request.title:
        rbr_score -= 5
        triggered.append({"rule_name": "痛点钩子缺乏情绪发问", "reason": "声明使用了痛点直击类型钩子，但标题未包含问号，无法高效激发用户对痛点的反思。"})
        advices.append("在标题中引入问号（如：‘你是不是也遇到过...？’）来强化痛点冲击力。")
        
    if not advices:
        advices.append("当前选题框架非常符合短视频工业化爆款模型，建议立即投产。")
        
    return {
        "score": max(rbr_score, 30.0),
        "triggered_rules": triggered,
        "advices": advices
    }


# ==================== 系统主控入口：纯算法评估（不调用AI） ====================
def evaluate_topic_dss(request: TopicRequest) -> EvaluationResponse:
    """
    系统主入口：纯客观定量分析，不掺杂AI调用（AI由外部接口按需触发）
    """
    # 1. 深度调用算法层：自动计算出四项指标得分 (0 - 100)
    text_corpus = f"{request.title} {request.content_draft}"
    
    emotion_score = calc_emotion_score(text_corpus)
    hot_score = calc_hot_score(request.title)
    history_score, top_cases = calc_history_score(request.title)
    timeliness_score = calc_timeliness_score(request.title)
    
    # 组装四项维度的分值雷达映射图字典，由前端直接加载
    score_breakdown = {
        "emotion": round(emotion_score / 100.0, 2),
        "hot": round(hot_score / 100.0, 2),
        "history": round(history_score / 100.0, 2),
        "timeliness": round(timeliness_score / 100.0, 2),
    }
    
    # 2. 使用用户调节的权重计算最终得分
    w_emotion = request.emotion
    w_hot = request.hot
    w_history = request.history
    w_timeliness = request.timeliness
    
    # 归一化（防御性编程）
    total_weight = w_emotion + w_hot + w_history + w_timeliness
    if abs(total_weight - 1.0) > 0.0001:
        w_emotion /= total_weight
        w_hot /= total_weight
        w_history /= total_weight
        w_timeliness /= total_weight
    
    final_score = (
        emotion_score * w_emotion +
        hot_score * w_hot +
        history_score * w_history +
        timeliness_score * w_timeliness
    )
    final_score = max(0.0, min(100.0, final_score))
    
    # 3. 混合 CBR 案例得分与 RBR 规则库得分（用于多维混合动力系统展示）
    # CBR 核心分基于 Top5 案例的平均完播率和相似度计算
    if top_cases:
        max_sim = top_cases[0].get("similarity", 0.0)
        cbr_score = sum(c.get("completion_rate", 0.5) * 100 * c.get("similarity", 0.1) for c in top_cases) / (sum(c.get("similarity", 0.1) for c in top_cases) + 0.01)
        cbr_score = min(cbr_score * 1.3, 100.0)  # 进行尺度放大
    else:
        max_sim = 0.0
        cbr_score = 60.0
        
    rbr_result = evaluate_rbr_logic(request)
    rbr_score = rbr_result["score"]
    
    # 4. 根据相似度动态切分两套推理引擎在最终展现上的推荐权重系数
    if max_sim >= 0.70:
        w_cbr, w_rbr = 0.60, 0.40
    elif max_sim >= 0.40:
        w_cbr, w_rbr = 0.50, 0.50
    else:
        w_cbr, w_rbr = 0.30, 0.70
        
    # 5. 构建基础评估响应（不含AI结果，由外部填充）
    return EvaluationResponse(
        final_score=round(final_score, 2),
        cbr_score=round(cbr_score, 2),
        rbr_score=round(rbr_score, 2),
        cbr_weight=w_cbr,
        rbr_weight=w_rbr,
        max_similarity=round(max_sim, 2),
        score_breakdown=score_breakdown,
        top_cases=top_cases,
        optimizations=rbr_result["advices"],
        triggered_rules=rbr_result["triggered_rules"],
        ai_generated={}
    )


# ==================== AI 调用工具函数（供外部接口使用） ====================

def get_ai_config() -> tuple:
    """从环境变量读取 AI API 配置，并自动补全 Chat Completions 端点路径。"""
    api_key = os.environ.get("DDS_LLM_API_KEY", "").strip()
    api_url = os.environ.get("DDS_LLM_API_URL", "https://api.deepseek.com/chat/completions").strip()
    model_name = os.environ.get("DDS_LLM_MODEL", "deepseek-chat").strip()
    timeout = int(os.environ.get("DDS_LLM_TIMEOUT_SEC", 60))

    # 如果 URL 不以 /chat/completions 结尾，则自动补全（兼容仅写域名的情况）
    if not api_url.endswith("/chat/completions"):
        api_url = api_url.rstrip("/") + "/chat/completions"

    return api_key, api_url, model_name, timeout


def _call_llm_api(prompt_messages: List[Dict[str, str]], temperature: float = 0.5, max_tokens: int = 500) -> Optional[str]:
    """通用 LLM 调用，返回 response 文本，失败返回 None。"""
    api_key, api_url, model_name, timeout = get_ai_config()
    if not api_key or "sk-" not in api_key:
        return None

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": prompt_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"}  # 强制 JSON 输出
    }
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"⚠️ LLM API 调用异常: {e}")
    return None


def generate_ai_optimization_advices(title: str, domain: str, score_breakdown: Dict[str, float]) -> List[str]:
    """
    根据系统算分结果，生成 2~3 条优化建议（供 /api/v1/evaluate 调用）。
    score_breakdown 中各项为 0~1 的小数。
    """
    # 将小数转为百分比，便于展示
    emotion = score_breakdown.get("emotion", 0) * 100
    hot = score_breakdown.get("hot", 0) * 100
    history = score_breakdown.get("history", 0) * 100
    timeliness = score_breakdown.get("timeliness", 0) * 100

    system_prompt = (
        "你是一位资深短视频内容策略专家，精通抖音、快手等平台的流量算法与用户心理。"
        "请根据系统给出的定量评估结果，提供 2~3 条具体、可执行的选题优化建议。"
        "建议应针对各项得分短板，直接指出改进方向，并辅以实例。"
    )
    user_prompt = f"""
    【当前选题】
    - 标题：{title}
    - 赛道：{domain}
    
    【系统评估得分（满分100）】
    - 情感倾向：{emotion:.1f}
    - 热点匹配：{hot:.1f}
    - 历史相似成功度：{history:.1f}
    - 时效性：{timeliness:.1f}
    
    【任务】
    请输出 2~3 条精炼的优化建议，每条建议控制在 50 字以内。
    输出格式必须为 JSON 数组，例如：
    ["建议1：...", "建议2：...", "建议3：..."]
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    result = _call_llm_api(messages, temperature=0.5, max_tokens=300)
    if result:
        try:
            advices = json.loads(result)
            if isinstance(advices, list):
                return advices
        except:
            pass
    # 降级返回本地预设建议
    return [
        "建议：适当提高标题中的情绪唤醒度（如使用问句或感叹号）。",
        "建议：检查是否包含当前热门关键词（如'新规'、'2026'等）。"
    ]


def generate_ai_script_via_deepseek(request: TopicRequest) -> Dict[str, Any]:
    """
    根据完整选题特征，生成优化标题、口播脚本及分镜头方案。
    供 /api/v1/script 调用。
    返回的 shot_plan 保证为字符串列表，避免前端显示 [object Object]。
    """
    system_prompt = (
        "你是一个顶尖的短视频编导与洗稿重写专家。请严格基于系统输入的结构化特征限制，"
        "生成完全匹配爆款工业标准的视频文案及分镜头脚本。格式必须输出为合法 JSON。"
        "JSON 必须包含以下字段：\n"
        "- title: 优化后的标题（字符串）\n"
        "- opening_hook: 前3秒吸睛口播词（字符串）\n"
        "- full_script: 完整口播文案（字符串）\n"
        "- shot_plan: 分镜头列表，**每个元素必须为字符串**，例如：[\"镜头1：全景，主讲人...\", \"镜头2：特写，手机界面...\"]\n"
        "注意：shot_plan 中的每个元素都应当是一个简短的文字描述，不要使用对象或嵌套结构。"
    )
    user_prompt = f"""
    【决策系统输入的标准化多维选题特征】:
    - 视频标题: {request.title}
    - 赛道领域: {request.domain}
    - 内容结构约束: {request.content_structure}
    - 目标受众: {request.target_audience}
    - 黄金开头钩子类型: {request.hook_type}
    - 典型观看场景: {request.scenario}
    - 预计视频时长: {request.duration_sec}秒
    - 运营转化目标: 引导受众进行【{request.content_goal}】
    
    【严格输出格式要求】:
    请直接返回标准的 JSON 格式字典，不要带有 Markdown 标记。
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    result = _call_llm_api(messages, temperature=0.7, max_tokens=800)
    if result:
        try:
            script = json.loads(result)
            if isinstance(script, dict) and "full_script" in script:
                # ----- 规范化 shot_plan 字段 -----
                if "shot_plan" in script and isinstance(script["shot_plan"], list):
                    normalized = []
                    for item in script["shot_plan"]:
                        if isinstance(item, str):
                            normalized.append(item)
                        elif isinstance(item, dict):
                            # 如果是对象，尝试提取 '描述' 或 '镜头' 字段，否则转为 JSON 字符串
                            desc = item.get("描述") or item.get("镜头") or item.get("content") or json.dumps(item, ensure_ascii=False)
                            normalized.append(desc)
                        else:
                            normalized.append(str(item))
                    script["shot_plan"] = normalized
                else:
                    # 如果没有 shot_plan，提供默认值
                    script["shot_plan"] = [
                        "镜头1：开场，主讲人正面入画，语气激昂，字幕呈现核心观点。",
                        "镜头2：中景，展示操作步骤（如手机界面），配合画外音讲解。",
                        "镜头3：特写，强调关键数字或利益点，增强说服力。",
                        "镜头4：结尾，主讲人面对镜头，引导点赞、收藏、分享。"
                    ]
                return script
        except Exception as e:
            print(f"⚠️ 解析 DeepSeek 返回 JSON 失败，将使用 Mock 数据: {e}")

    # ----- 兜底 Mock 数据（保证 shot_plan 为字符串列表） -----
    mock_hook = f"你是不是也觉得{request.target_audience}在{request.domain}里面踩过太多坑了？别眨眼，看完这条视频帮你省下一大笔钱！"
    if "医保" in request.title:
        mock_hook = f"2026年最新医保重大变动！好多{request.target_audience}都不知道，原来这样能多报几千块！"
    
    return {
        "title": f"【DSS策略调优版】{request.title}",
        "opening_hook": mock_hook,
        "full_script": (
            f"【黄金开头 - {request.hook_type}】\n{mock_hook}\n\n"
            f"【核心干货阐述 - 严格遵循 {request.content_structure} 结构模型】\n"
            f"第一步：千万别盲目去柜台，先登录确认核心标签绑定状态...\n"
            f"第二步：在手机后台搜索打开互联功能...\n"
            f"第三步：严格控制在 {request.duration_sec} 秒内以契合大盘完播率。\n\n"
            f"【结尾互动转化 - 运营目标：{request.content_goal}】\n"
            f"这个新规太实用了，赶紧把它【分享】给你关心的那个人，或者先【收藏】起来免得找不到！"
        ),
        "shot_plan": [
            "镜头1：全景，主讲人手持手机道具，语气激昂，屏幕上方打出吸睛标题。",
            "镜头2：特写，手机录屏操作，红圈标注关键按钮，配合画外音讲解。",
            "镜头3：中景，主讲人面对镜头，手势引导互动（点赞、收藏、分享）。",
            "镜头4：结尾，主讲人微笑，屏幕显示“关注我，了解更多干货”字样。"
        ]
    }


# ==================== 测试桩 ====================
if __name__ == "__main__":
    test_payload = TopicRequest(
        title="2026年最新医保门诊报销政策大调整！原来这样操作能多省几千块？",
        domain="知识科普",
        content_draft="根据最新的规定，门诊起付线全面下调，报销比例全面提升。但是很多人不知道怎么去关联绑定自己的共济账户...",
        publish_platform="小红书",
        target_audience="宝妈",
        content_goal="收藏",
        content_structure="问题-方案",
        hook_type="痛点直击",
        scenario="午休",
        tags=["医保", "报销", "新规", "省钱"],
        core_entities=["社保局", "支付宝"]
    )
    
    print("========== 纯算法评估 ==========")
    res = evaluate_topic_dss(test_payload)
    print(f"总分: {res.final_score}")
    print(f"各项得分: {res.score_breakdown}")
    
    print("\n========== 测试 AI 优化建议 ==========")
    advices = generate_ai_optimization_advices(test_payload.title, test_payload.domain, res.score_breakdown)
    for i, a in enumerate(advices, 1):
        print(f"{i}. {a}")
    
    print("\n========== 测试 AI 脚本生成 ==========")
    script = generate_ai_script_via_deepseek(test_payload)
    print(f"优化标题: {script.get('title')}")
    print(f"开头钩子: {script.get('opening_hook')[:50]}...")
    print(f"脚本长度: {len(script.get('full_script', ''))} 字")