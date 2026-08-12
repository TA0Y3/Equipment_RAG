# -*- coding: utf-8 -*-
"""
问答历史缓存集合读写工具（基于 Milvus）
职责：管理「问题缓存」集合 qa_answer_cache 的创建、写入（去重覆盖）、容量清理，
供 node_item_name_confirm 的缓存写入步骤与 node_query_cache 命中检测节点调用。
缓存命中判定与写入均依赖环境变量配置，缺省使用默认值（不修改 .env 也可运行）。
"""
import os
import json
import time
from pymilvus import DataType
from app.clients.milvus_utils import get_milvus_client
from app.lm.embedding_utils import generate_embeddings
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.core.logger import logger

# 缓存集合名与参数：均可通过环境变量覆盖，缺省使用默认值
QUERY_CACHE_COLLECTION = os.getenv("QUERY_CACHE_COLLECTION", "qa_answer_cache")
QUERY_CACHE_HIT_THRESHOLD = float(os.getenv("QUERY_CACHE_HIT_THRESHOLD", "0.85"))
QUERY_CACHE_MAX_SIZE = int(os.getenv("QUERY_CACHE_MAX_SIZE", "20"))
# Milvus VARCHAR 字段上限为 65535 字节（中文约 3 字节/字），写入前按安全长度截断避免入库失败
_SAFE_ANSWER_LEN = 60000
_SAFE_IMAGE_LEN = 60000
_SAFE_QUERY_LEN = 4000


def ensure_cache_collection(client) -> bool:
    """
    确保缓存集合存在：不存在则创建（Schema + 双向量索引），已存在直接复用
    :param client: MilvusClient 实例
    :return: 集合是否可用
    """
    try:
        if client.has_collection(collection_name=QUERY_CACHE_COLLECTION):
            return True
        logger.info(f"缓存集合[{QUERY_CACHE_COLLECTION}]不存在，开始创建Schema和索引")
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        # 自增主键：INT64，唯一标识每条缓存记录
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
        # 会话唯一标识
        schema.add_field(field_name="session_id", datatype=DataType.VARCHAR, max_length=128)
        # 确认商品名（单值，去重键之一，检索过滤依据）
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        # 改写后的问题（去重键之一）
        schema.add_field(field_name="rewritten_query", datatype=DataType.VARCHAR, max_length=4096)
        # 历史答案正文
        schema.add_field(field_name="answer", datatype=DataType.VARCHAR, max_length=65535)
        # 图片URL列表的JSON字符串
        schema.add_field(field_name="image_urls", datatype=DataType.VARCHAR, max_length=65535)
        # 稠密向量：BGE-M3 固定 1024 维
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        # 稀疏向量：变长
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 写入时间戳（秒级），容量清理排序依据
        schema.add_field(field_name="created_ts", datatype=DataType.INT64)

        # 构建向量索引：稠密 HNSW+COSINE，稀疏倒排+IP（与现有 kb_chunks 集合保持一致）
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200}
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
        )
        client.create_collection(collection_name=QUERY_CACHE_COLLECTION, schema=schema, index_params=index_params)
        logger.info(f"缓存集合[{QUERY_CACHE_COLLECTION}]创建成功")
        return True
    except Exception as e:
        logger.error(f"缓存集合创建失败[{QUERY_CACHE_COLLECTION}]: {e}", exc_info=True)
        return False


def _load_collection(client) -> None:
    """加载集合（幂等）：搜索/删除前确保集合已加载，加载失败仅告警"""
    try:
        client.load_collection(collection_name=QUERY_CACHE_COLLECTION)
    except Exception as e:
        logger.warning(f"缓存集合加载提示: {e}")


def _find_existing_pk(client, item_name: str, rewritten_query: str):
    """
    按唯一键（item_name + rewritten_query）查询已存在记录的主键
    :return: 主键值，不存在或查询失败返回 None
    """
    try:
        expr = (f'item_name == "{escape_milvus_string(item_name)}" '
                f'and rewritten_query == "{escape_milvus_string(rewritten_query)}"')
        rows = client.query(collection_name=QUERY_CACHE_COLLECTION, filter=expr, output_fields=["pk"])
        return rows[0]["pk"] if rows else None
    except Exception as e:
        logger.warning(f"缓存去重查询失败: {e}")
        return None


def enforce_cache_capacity(client, max_size: int = None) -> None:
    """
    容量上限控制：缓存集合仅维护最近 max_size 条记录，超出后按写入时间删除最旧记录
    :param client: MilvusClient 实例
    :param max_size: 容量上限，缺省使用 QUERY_CACHE_MAX_SIZE（默认20）
    """
    max_size = max_size or QUERY_CACHE_MAX_SIZE
    try:
        rows = client.query(collection_name=QUERY_CACHE_COLLECTION, filter="pk >= 0",
                            output_fields=["pk", "created_ts"])
        if len(rows) > max_size:
            rows.sort(key=lambda r: r.get("created_ts") or 0)
            excess = rows[: len(rows) - max_size]
            client.delete(collection_name=QUERY_CACHE_COLLECTION, ids=[r["pk"] for r in excess])
            logger.info(f"缓存容量清理: 删除{len(excess)}条最旧记录，剩余{max_size}条")
    except Exception as e:
        # 容量清理失败不影响主流程，仅记录告警
        logger.warning(f"缓存容量清理失败(不影响主流程): {e}")


def write_cache_record(session_id: str, item_names, rewritten_query: str, answer: str, image_urls=None) -> bool:
    """
    写入一条问答缓存记录（BGE-M3 双路向量化 + 去重覆盖 + 容量清理）
    :param session_id: 会话唯一标识
    :param item_names: 该轮确认的商品名列表（取第一个作为缓存过滤键）
    :param rewritten_query: 该轮改写后的问题（唯一键之一）
    :param answer: 该轮助手回答正文
    :param image_urls: 该轮答案关联的图片URL列表（可选）
    :return: 是否写入成功；任何环节失败均返回 False 且不影响主流程
    """
    try:
        client = get_milvus_client()
        if not client or not item_names or not rewritten_query or not answer:
            logger.debug("缓存写入跳过: Milvus客户端或必要字段缺失")
            return False
        if not ensure_cache_collection(client):
            return False
        _load_collection(client)

        # 只取第一个确认商品名作为过滤键（命中场景以主要商品为准）
        item_name = str(item_names[0])

        # 去重覆盖：唯一键 item_name + rewritten_query，已存在则先删除旧记录再写入
        exist_pk = _find_existing_pk(client, item_name, rewritten_query)
        if exist_pk is not None:
            client.delete(collection_name=QUERY_CACHE_COLLECTION, ids=[exist_pk])
            logger.info(f"缓存去重: 删除同键旧记录 pk={exist_pk}")

        # BGE-M3 双路向量化（稠密 + 稀疏）
        embeddings = generate_embeddings([rewritten_query])

        record = {
            "session_id": session_id,
            "item_name": item_name,
            "rewritten_query": rewritten_query[:_SAFE_QUERY_LEN],
            "answer": answer[:_SAFE_ANSWER_LEN],
            "image_urls": json.dumps(image_urls or [], ensure_ascii=False)[:_SAFE_IMAGE_LEN],
            "dense_vector": embeddings["dense"][0],
            "sparse_vector": embeddings["sparse"][0],
            "created_ts": int(time.time())
        }
        client.insert(collection_name=QUERY_CACHE_COLLECTION, data=[record])
        logger.info(f"缓存写入成功: session={session_id}, item={item_name}, query={rewritten_query[:50]}")

        # 容量上限控制：超出 QUERY_CACHE_MAX_SIZE 删除最旧记录
        enforce_cache_capacity(client)
        return True
    except Exception as e:
        logger.error(f"缓存写入失败: {e}", exc_info=True)
        return False
