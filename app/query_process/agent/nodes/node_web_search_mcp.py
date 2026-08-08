# -*- coding: utf-8 -*-
import sys
import json
import asyncio
from app.utils.task_utils import add_done_task, add_running_task
from app.conf.bailian_mcp_config import mcp_config
from app.core.logger import logger

# ==================================================================================
# 改用 Streamable HTTP 传输（百炼官方推荐，比 SSE 更稳定）
# ==================================================================================
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession


async def mcp_call(query):
    """
    异步调用百炼 MCP 搜索服务（Streamable HTTP 版本）。
    """
    api_key = getattr(mcp_config, 'api_key', None)
    if not api_key:
        logger.error("[MCP] api_key 未配置")
        return None

    # Bearer 前缀（避免重复添加）
    auth_header = api_key if api_key.startswith("Bearer ") else f"Bearer {api_key}"

    # 注意：Streamable HTTP 的 URL 后缀是 /mcp，不是 /sse
    mcp_url = getattr(mcp_config, 'mcp_base_url',
                      "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp")

    try:
        logger.info(f"[MCP] 正在连接百炼 WebSearch 服务: {mcp_url}")

        # Streamable HTTP 客户端：自动管理连接生命周期，无需手动 cleanup
        async with streamablehttp_client(
            mcp_url,
            headers={"Authorization": auth_header}
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info(f"[MCP] 连接成功，正在调用工具查询: {query}")

                result = await session.call_tool(
                    name="bailian_web_search",
                    arguments={"query": query, "count": 5}
                )
                logger.info("[MCP] 工具调用完成，已获取返回结果")
                return result

    except Exception as e:
        logger.error(f"[MCP] 调用过程中发生异常: {e}", exc_info=True)
        return None


def node_web_search_mcp(state):
    """
    LangGraph 同步节点：MCP 搜索入口。
    """
    logger.info("---node_web_search_mcp 开始处理---")

    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    query = state.get("rewritten_query", "")
    if not query:
        query = state.get("original_query", "")

    docs = []

    if query:
        try:
            logger.info(f"启动异步 MCP 调用，Query: {query}")

            # 同步-异步桥接
            result = asyncio.run(mcp_call(query))

            # ==================================================================
            # 解析 Streamable HTTP 返回结果
            # ------------------------------------------------------------------
            # result.content 是 TextContent / ImageContent 等对象的列表
            # 搜索工具通常返回 TextContent，.text 字段是 JSON 字符串
            # ==================================================================
            if result and result.content:
                raw_text = ""
                for block in result.content:
                    if hasattr(block, "text"):
                        raw_text += block.text

                if raw_text:
                    try:
                        data = json.loads(raw_text)
                        pages = data.get("pages") or []

                        logger.info(f"MCP 返回原始页面数量: {len(pages)}")

                        for item in pages:
                            snippet = (item.get("snippet") or "").strip()
                            url = (item.get("url") or "").strip()
                            title = (item.get("title") or "").strip()

                            if not snippet:
                                continue

                            docs.append({"title": title, "url": url, "snippet": snippet})

                    except json.JSONDecodeError:
                        logger.error(f"MCP 返回结果解析 JSON 失败: {raw_text[:100]}...")
                else:
                    logger.warning("MCP 返回内容为空文本")
            else:
                logger.warning("MCP 返回结果为空或无效")

            logger.info(f"结构化搜索结果数量: {len(docs)}")

        except Exception as e:
            logger.error(f"MCP 搜索节点执行异常: {e}", exc_info=True)
    else:
        logger.warning("查询词为空，跳过 MCP 搜索")

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    logger.info("---node_web_search_mcp 处理结束---")

    if docs:
        return {"web_search_docs": docs}
    return {}


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print(">>> 启动 node_web_search_mcp 本地测试")
    print("=" * 50)

    test_state = {
        "session_id": "test_mcp_session",
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
        "is_stream": False
    }

    try:
        result_state = node_web_search_mcp(test_state)
        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        search_results = result_state.get('web_search_docs', [])
        print(f"搜索结果数量: {len(search_results)}")
        if search_results:
            print("首条结果预览:")
            print(json.dumps(search_results[0], indent=2, ensure_ascii=False))
        else:
            print("未获取到搜索结果")
        print("=" * 50)
    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")