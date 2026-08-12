# -*- coding: utf-8 -*-
"""
ragas context_recall 评估脚本
================================================================
功能:对 test/ragas_testset_50.json 中的每条 question,调用本项目查询链路
(商品确认 → 问题缓存检测 → 三路检索 → RRF 融合 → Rerank 重排),
取【重排后的 reranked_docs】的 text 字段作为 contexts,组装 ragas 的
EvaluationDataset,计算检索阶段上下文召回率 context_recall(0~1)。

contexts 取自哪个环节:
    state["reranked_docs"] —— Rerank 节点最终输出的文档列表
    (上游:Milvus 混合检索 req_limit=10 × 2 路 → RRF k=60, max_results=10
     → Rerank 动态 TopK, 硬上限 10)
    这是系统实际提供给 LLM 的上下文,最能反映端到端检索覆盖能力。

注意:
1. 本脚本复用 main_graph.py 的节点,但构建【检索子图】(不含 answer_output),
   不生成答案,节省 LLM 调用与 reranker 之外的额外开销。
2. 默认不含 Web 搜索路(node_web_search_mcp):context_recall 衡量的是
   【知识库召回】,Web 结果不在人工标注的 reference_contexts 范围内,且依赖
   MCP 服务可用性。如需纳入,将 ENABLE_WEB_SEARCH 改为 True。
3. 每条样本使用独立 session_id 并在运行前清理该会话历史,避免污染
   真实会话与问题缓存(缓存写入只发生在有历史 assistant 消息时)。
4. 运行环境:rag_project conda 环境(需先安装 ragas,见 README/交付说明)。
5. 商品名仅中置信度(0.6~0.85)时,系统会反问候选列表:交互模式(默认)
   下打印候选并等待人工输入选择后重跑链路;--auto 模式下自动跳过该样本。
"""

import argparse
import json
import os
import sys
import time
from typing import List

# 将项目根目录加入 sys.path(脚本位于 test/ 目录)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# ragas 0.4.x 与 langchain_community 1.x 兼容垫片
# ragas/llms/base.py 硬编码导入 langchain_community 已移除的旧模型模块
# (chat_models.vertexai / llms.vertexai,0.4.x 起迁移到独立集成包)。
# 本脚本只使用 ChatOpenAI(百炼端点),这些桩类不会被实例化,
# 仅为让 ragas 的 import 链通过,且只影响当前进程。
# ---------------------------------------------------------------------------
def _install_ragas_compat_shims() -> None:
    import importlib
    import types

    shims = {
        "langchain_community.chat_models.vertexai": ["ChatVertexAI"],
        "langchain_community.llms.vertexai": ["VertexAI"],
    }
    for mod_name, attrs in shims.items():
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            # 模块不存在:创建空桩模块,并挂到父包属性上
            parent_name, _, child_name = mod_name.rpartition(".")
            mod = types.ModuleType(mod_name)
            sys.modules[mod_name] = mod
            try:
                parent = importlib.import_module(parent_name)
            except ModuleNotFoundError:
                parent = types.ModuleType(parent_name)
                sys.modules[parent_name] = parent
            setattr(parent, child_name, mod)
        # 模块存在但缺属性:补一个空桩类
        for attr in attrs:
            if getattr(mod, attr, None) is None:
                setattr(mod, attr, type(attr, (), {}))


_install_ragas_compat_shims()

from langgraph.graph import StateGraph, END
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_query_cache import node_query_cache
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_rerank import node_rerank

# 是否纳入 Web 搜索路(默认关闭,理由见文件头注释)
ENABLE_WEB_SEARCH = False

# 商品名候选反问的交互重试上限(超过则跳过该样本)
MAX_CONFIRM_RETRIES = 3

# 检索环节标注(写进结果文件,便于审计)
RETRIEVAL_PARAMS = {
    "layer": "reranked_docs",
    "embedding_req_limit": 10,   # node_search_embedding Milvus 每路请求条数
    "rrf_k": 60,
    "rrf_max_results": 10,
    "rerank_topk": "动态 TopK,硬上限 10",
    "web_search": ENABLE_WEB_SEARCH,
}


# ==================================================================
# 一、构建检索子图(商品确认 → 缓存检测 → 三路检索 → RRF → Rerank)
# ==================================================================
def build_retrieval_graph():
    """构建与 main_graph.py 路由逻辑一致的检索子图(不含答案生成)。"""
    builder = StateGraph(QueryGraphState)

    builder.add_node("node_item_name_confirm", node_item_name_confirm)
    builder.add_node("node_query_cache", node_query_cache)
    builder.add_node("node_multi_search", lambda x: x)
    builder.add_node("node_search_embedding", node_search_embedding)
    builder.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
    if ENABLE_WEB_SEARCH:
        from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
        builder.add_node("node_web_search_mcp", node_web_search_mcp)
    builder.add_node("node_join", lambda x: {})
    builder.add_node("node_rrf", node_rrf)
    builder.add_node("node_rerank", node_rerank)

    builder.set_entry_point("node_item_name_confirm")

    # 商品确认后:已有答案(反问/拒答)直接结束,否则走缓存检测
    builder.add_conditional_edges(
        "node_item_name_confirm",
        lambda state: END if state.get("answer") else "node_query_cache",
    )
    # 缓存检测后:命中直接结束,未命中走多路检索
    builder.add_conditional_edges(
        "node_query_cache",
        lambda state: END if state.get("answer") else "node_multi_search",
    )

    # 多路检索分叉
    builder.add_edge("node_multi_search", "node_search_embedding")
    builder.add_edge("node_multi_search", "node_search_embedding_hyde")
    if ENABLE_WEB_SEARCH:
        builder.add_edge("node_multi_search", "node_web_search_mcp")

    # 各路结果汇合
    builder.add_edge("node_search_embedding", "node_join")
    builder.add_edge("node_search_embedding_hyde", "node_join")
    if ENABLE_WEB_SEARCH:
        builder.add_edge("node_web_search_mcp", "node_join")

    builder.add_edge("node_join", "node_rrf")
    builder.add_edge("node_rrf", "node_rerank")
    builder.add_edge("node_rerank", END)

    return builder.compile()


def clear_session(session_id: str) -> None:
    """清理指定会话历史(评估样本之间隔离,避免缓存/历史串扰)。"""
    try:
        from app.clients.mongo_history_utils import clear_history
        clear_history(session_id)
    except Exception:
        pass


def _parse_candidates_from_answer(answer: str) -> List[str]:
    """解析商品确认反问中的候选列表,如「您是想问以下哪个产品:A、B、C?」。"""
    import re

    m = re.search(r"以下哪个产品[:：]\s*([^?？。]+)", answer)
    if not m:
        return []
    return [x.strip() for x in re.split(r"[、,，]", m.group(1)) if x.strip()]


def _ask_user_choice(index: int, candidates: List[str]) -> str:
    """打印候选列表并等待人工输入(序号或完整商品名);回车=跳过该样本。"""
    print(f"\n>>> 样本 #{index} 商品名不够明确,系统给出候选:")
    for i, c in enumerate(candidates, start=1):
        print(f"    [{i}] {c}")
    print("    请输入序号或完整商品名(直接回车跳过该样本):")
    try:
        raw = input(">>> ").strip()
    except EOFError:  # 无交互终端(如重定向)时跳过
        return ""
    if not raw:
        return ""
    if raw.isdigit() and 1 <= int(raw) <= len(candidates):
        return candidates[int(raw) - 1]
    return raw


# ==================================================================
# 二、ragas 评估配置(复用百炼 OpenAI 兼容端点)
# ==================================================================
def setup_ragas():
    """初始化 ragas LLM 与 ContextRecall 指标,兼容 0.2.x ~ 0.4.x。"""
    from langchain_openai import ChatOpenAI
    from app.conf.lm_config import lm_config

    eval_llm_wrapper = None
    try:
        from ragas.llms import LangchainLLMWrapper
        eval_llm_wrapper = LangchainLLMWrapper
    except ImportError:
        from ragas.llms.base import LangchainLLMWrapper

    eval_llm = eval_llm_wrapper(
        ChatOpenAI(
            model=lm_config.llm_model,
            base_url=lm_config.base_url,
            api_key=lm_config.api_key,
            temperature=lm_config.llm_temperature,
        )
    )

    try:
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.metrics._context_recall import ContextRecall
    except ImportError:  # ragas 0.2.x 回退
        try:
            from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
            from ragas.metrics.context_recall import ContextRecall
        except ImportError:
            from ragas import EvaluationDataset, SingleTurnSample
            from ragas.metrics import ContextRecall

    metric = ContextRecall(llm=eval_llm)
    return metric, EvaluationDataset, SingleTurnSample


# ==================================================================
# 三、样本级 reference 命中检查(辅助诊断,非 ragas 官方指标)
# ==================================================================
def reference_hit(reference_contexts, retrieved_contexts, min_len=20):
    """人工标注片段(reference_contexts)是否被系统检索上下文覆盖。
    使用双向子串包含判断,过短片段(< min_len 字符)不计,避免误判。"""
    hits = []
    for ref in reference_contexts or []:
        ref_strip = "".join(ref.split())
        if len(ref_strip) < min_len:
            continue
        matched = False
        for ctx in retrieved_contexts or []:
            ctx_strip = "".join(ctx.split())
            if ref_strip in ctx_strip or ctx_strip in ref_strip:
                matched = True
                break
        hits.append(matched)
    return hits


# ==================================================================
# 四、主流程
# ==================================================================
def run_evaluation(testset_path, out_path, limit=None, auto_mode=False):
    with open(testset_path, "r", encoding="utf-8") as f:
        samples_meta = json.load(f)
    if limit:
        samples_meta = samples_meta[:limit]

    metric, EvaluationDataset, SingleTurnSample = setup_ragas()
    graph = build_retrieval_graph()
    logger.info(f"检索子图构建完成,待评估样本: {len(samples_meta)} 条")

    samples, results = [], []
    skipped = []
    for idx, meta in enumerate(samples_meta, start=1):
        question = meta["question"]
        session_id = f"eval_recall_{meta['index']}"
        clear_session(session_id)

        # 1) 调用检索链路(商品确认 → 缓存检测 → 三路检索 → RRF → Rerank)
        #    商品名仅中置信度(0.6~0.85)时节点会反问候选列表:
        #    交互模式(默认)等待人工选择后重跑链路;--auto 模式直接跳过该样本
        final_state = None
        confirm_answer = ""
        user_choice = ""
        for attempt in range(MAX_CONFIRM_RETRIES + 1):
            query_now = question if attempt == 0 else f"{question}(具体型号:{user_choice})"
            initial_state = {
                "session_id": session_id,
                "original_query": query_now,
                "is_stream": False,
            }
            try:
                final_state = graph.invoke(initial_state)
            except Exception as e:
                logger.error(f"样本 {meta['index']} 检索链路执行失败: {e}", exc_info=True)
                skipped.append({"index": meta["index"], "reason": f"检索链路异常: {e}"})
                final_state = None
                break

            confirm_answer = final_state.get("answer") or ""
            if "以下哪个产品" not in confirm_answer:
                break  # 商品名已确认(或拒答),无需人工介入

            candidates = _parse_candidates_from_answer(confirm_answer)
            if not candidates:
                break
            if auto_mode:
                logger.warning(f"样本 {meta['index']} 触发候选反问(auto 模式跳过): {candidates}")
                break
            choice = _ask_user_choice(meta["index"], candidates)
            if not choice:
                logger.warning(f"样本 {meta['index']} 未选择候选,按跳过处理")
                break
            user_choice = choice
            clear_session(session_id)  # 清理反问历史,避免缓存/历史串扰

        if final_state is None:
            continue

        # 2) 提取最终检索上下文:重排后的 reranked_docs
        reranked = final_state.get("reranked_docs") or []
        contexts = [d.get("text") or "" for d in reranked if d.get("text")]

        # 3) 组装 ragas 样本
        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=contexts,
            reference=meta["ground_truth"],
        )
        samples.append(sample)
        results.append(
            {
                "index": meta["index"],
                "item_name": meta.get("item_name", ""),
                "question_type": meta.get("question_type", ""),
                "question": question,
                "retrieved_count": len(contexts),
                "rrf_count": len(final_state.get("rrf_chunks") or []),
                "cache_hit": bool(final_state.get("cache_hit")),
                "retrieved_titles": list(
                    dict.fromkeys(
                        d.get("title") or d.get("item_name") or "" for d in reranked
                    )
                ),
                "reference_hit": reference_hit(
                    meta.get("reference_contexts"), contexts
                ),
                "confirm_answer": confirm_answer,
                "user_choice": user_choice,
            }
        )
        logger.info(f"[{idx}/{len(samples_meta)}] 样本 {meta['index']} 完成, "
                    f"检索到 {len(contexts)} 条上下文, 缓存命中={results[-1]['cache_hit']}")

    if not samples:
        logger.error("没有可评估的样本,终止。")
        return

    # 4) 计算 context_recall(逐条评分,整体得分 = 均值)
    logger.info("开始 ragas context_recall 评分(依赖百炼 LLM,耗时较长)...")
    # ragas 0.4.x 评分入口为 single_turn_score,0.2/0.3.x 为 score
    score_fn = getattr(metric, "single_turn_score", None) or getattr(metric, "score", None)
    if score_fn is None:
        raise AttributeError("ragas 指标对象缺少评分方法(single_turn_score / score)")
    scores = []
    for i, (sample, res) in enumerate(zip(samples, results), start=1):
        try:
            score = float(score_fn(sample))
        except Exception as e:
            logger.error(f"样本 {res['index']} 评分失败: {e}")
            score = None
        scores.append(score)
        res["context_recall"] = score
        logger.info(f"[{i}/{len(samples)}] 样本 {res['index']} context_recall = {score}")

    valid_scores = [s for s in scores if s is not None]
    overall = sum(valid_scores) / len(valid_scores) if valid_scores else None

    # 5) 分组统计(按商品 / 问题类型)
    def group_mean(key):
        groups = {}
        for res in results:
            if res.get("context_recall") is None:
                continue
            groups.setdefault(res.get(key, "未知"), []).append(res["context_recall"])
        return {k: round(sum(v) / len(v), 4) for k, v in groups.items()}

    # 6) 输出与落盘
    summary = {
        "metric": "context_recall",
        "overall_context_recall": overall,
        "total_samples": len(samples_meta),
        "evaluated_samples": len(valid_scores),
        "skipped_samples": skipped,
        "retrieval_params": RETRIEVAL_PARAMS,
        "by_item_name": group_mean("item_name"),
        "by_question_type": group_mean("question_type"),
        "results": results,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"context_recall(整体): {overall}")
    print(f"评估样本数: {len(valid_scores)} / {len(samples_meta)}")
    print("按问题类型:", json.dumps(summary["by_question_type"], ensure_ascii=False))
    print("按商品(前 5):", json.dumps(
        dict(list(summary["by_item_name"].items())[:5]), ensure_ascii=False))
    print(f"详细结果已保存至: {out_path}")
    print("=" * 60)

    for res in results:
        hit_flag = "HIT" if any(res["reference_hit"]) else "MISS"
        print(f"  [{hit_flag}] #{res['index']} {res['question_type']} {res['item_name']} "
              f"recall={res['context_recall']} retrieved={res['retrieved_count']} "
              f"cache={res['cache_hit']} | {res['question'][:40]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ragas context_recall 评估")
    parser.add_argument("--testset", default=os.path.join(PROJECT_ROOT, "test", "ragas_testset_50.json"))
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "test", "ragas_results.json"))
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 条(调试用)")
    parser.add_argument("--auto", action="store_true",
                        help="无人值守模式:触发商品名候选反问时自动跳过,不等待输入")
    args = parser.parse_args()

    run_evaluation(args.testset, args.out, args.limit, auto_mode=args.auto)
