# -*- coding: utf-8 -*-
import sys
import json
from app.utils.task_utils import add_running_task, add_done_task
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import create_hybrid_search_requests, hybrid_search, get_milvus_client
from app.clients.milvus_cache_utils import QUERY_CACHE_COLLECTION, QUERY_CACHE_HIT_THRESHOLD
from app.core.logger import logger


def node_query_cache(state):
    """
    问题缓存命中检测节点（位于 node_item_name_confirm 之后、三路检索之前）
    流程：改写问题向量化 → 在问答缓存集合按商品名过滤混合检索 → 取最高相似度
    - 得分 > QUERY_CACHE_HIT_THRESHOLD（默认0.85）：判定命中，将历史 answer/image_urls/rewritten_query
      写入 state 并标记 cache_hit=True，由路由直接跳转 node_answer_output，跳过耗时三路检索
    - 未命中：原样返回 state，继续走 node_multi_search 三路并行检索
    容错：向量化失败 / Milvus异常 / 集合不存在 / item_names为空 均视为未命中，绝不阻塞主流程
    :param state: Dict - 会话状态字典，关键字段：
                  {
                      "session_id": str,        # 会话唯一标识
                      "rewritten_query": str,   # step3改写后的完整用户问题（含商品名）
                      "item_names": list[str],  # step6已确认的标准化商品名列表
                      "is_stream": bool/None    # 是否为流式响应，可选
                  }
    :return: Dict - 命中时写入 answer/image_urls/rewritten_query/cache_hit，未命中时原样返回
    """
    logger.info("---node_query_cache (问题缓存命中检测) 节点开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    query = state.get("rewritten_query")
    item_names = state.get("item_names")

    # 1. 前置校验：无改写问题或商品名时无法过滤检索，直接视为未命中
    if not query or not item_names:
        logger.warning(f"node_query_cache: rewritten_query/item_names 缺失，跳过缓存检测 (query={query!r})")
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return state

    try:
        # 2. 对改写后的问题生成 BGE-M3 稠密 + 稀疏双路向量
        logger.info(f"node_query_cache: 开始向量化查询问题: {query[:50]}")
        embeddings = generate_embeddings([query])
        dense_vec = embeddings.get("dense")[0]
        sparse_vec = embeddings.get("sparse")[0]

        # 3. 构造商品名过滤表达式（与 node_search_embedding 保持一致）
        quoted = ", ".join(f'"{v}"' for v in item_names)
        expr = f"item_name in [{quoted}]"
        logger.info(f"node_query_cache: 过滤表达式: {expr}")

        # 4. 在问答缓存集合中执行混合检索，仅取最高相似度的一条
        reqs = create_hybrid_search_requests(
            dense_vector=dense_vec,
            sparse_vector=sparse_vec,
            expr=expr,
            limit=1
        )
        client = get_milvus_client()
        if not client:
            logger.error("node_query_cache: 无法获取 Milvus 客户端，视为未命中")
            return state

        res = hybrid_search(
            client=client,
            collection_name=QUERY_CACHE_COLLECTION,
            reqs=reqs,
            ranker_weights=(0.8, 0.2),
            norm_score=True,
            limit=1,
            output_fields=["answer", "rewritten_query", "image_urls"]
        )

        # 5. 命中判定：最高相似度超过阈值则复用历史答案
        if res and len(res) > 0 and len(res[0]) > 0:
            top = res[0][0]
            score = top.get("distance")
            entity = top.get("entity") or {}
            logger.info(f"node_query_cache: 缓存检索到 top1，相似度={score:.4f}，阈值={QUERY_CACHE_HIT_THRESHOLD}")

            if score is not None and float(score) > QUERY_CACHE_HIT_THRESHOLD:
                cached_answer = (entity.get("answer") or "").strip()
                if cached_answer:
                    # 命中：复用历史答案与图片，替换改写问题（供历史落库追溯实际回答的问题）
                    state["answer"] = cached_answer
                    state["image_urls"] = _parse_image_urls(entity.get("image_urls"))
                    cached_query = (entity.get("rewritten_query") or "").strip()
                    if cached_query:
                        state["rewritten_query"] = cached_query
                    state["cache_hit"] = True
                    logger.info(f"node_query_cache: 命中缓存！score={float(score):.4f}，"
                                f"answer_len={len(cached_answer)}，images={state['image_urls']}")
                    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
                    return state
                else:
                    logger.warning("node_query_cache: 命中但缓存答案为空，视为未命中")
            else:
                logger.info("node_query_cache: 相似度未达阈值，未命中，继续三路检索")
        else:
            logger.info("node_query_cache: 缓存集合无匹配记录，未命中")

    except Exception as e:
        # 缓存集合不存在 / Milvus异常 / 向量化异常：一律降级为未命中，不阻塞主流程
        logger.warning(f"node_query_cache: 缓存检测异常，降级为未命中: {e}")

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


def _parse_image_urls(image_urls_str) -> list:
    """
    解析缓存记录中的图片URL列表（JSON字符串格式），解析失败返回空列表
    """
    if not image_urls_str:
        return []
    try:
        urls = json.loads(image_urls_str)
        return urls if isinstance(urls, list) else []
    except Exception:
        logger.warning(f"node_query_cache: 缓存图片URL解析失败: {image_urls_str[:100]}")
        return []


if __name__ == "__main__":
    # 本地测试：命中/未命中两条路径
    print("\n" + "=" * 50)
    print(">>> 启动 node_query_cache 本地测试")
    print("=" * 50)

    test_state = {
        "session_id": "test_cache_001",
        "rewritten_query": "HAK 180 烫金机如何设置烫金温度",
        "item_names": ["HAK 180 烫金机"],
        "is_stream": False
    }

    result = node_query_cache(test_state)
    print(f"\n>>> 命中标记: {result.get('cache_hit')}")
    print(f">>> 答案长度: {len(result.get('answer') or '')}")
    print(f">>> 图片列表: {result.get('image_urls')}")
    print("=" * 50)
