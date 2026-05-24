"""LLM 封装 — DeepSeek API 对话与事件抽取。"""

import os
from openai import OpenAI


class LLM:
    """DeepSeek API 封装（OpenAI 兼容格式）。"""

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
    ):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 环境变量未设置")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat(self, system: str, messages: list[dict]) -> str:
        """发送对话请求，返回回复文本。

        API 调用失败时重试一次（共两次机会），两次都失败则抛出 RuntimeError。
        """
        full_messages = [{"role": "system", "content": system}]
        full_messages.extend(messages)

        last_error = None
        for attempt in range(2):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    max_tokens=4096,
                )
                return resp.choices[0].message.content
            except Exception as e:
                last_error = e
        raise RuntimeError(f"API 调用失败（已重试一次）: {last_error}")

    def chat_stream(self, system: str, messages: list[dict], callback):
        """发送流式对话请求，逐 chunk 回调。

        Parameters
        ----------
        callback : callable
            每个 text chunk 被调用，参数为 (chunk_text: str)
        """
        full_messages = [{"role": "system", "content": system}]
        full_messages.extend(messages)

        last_error = None
        for attempt in range(2):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    max_tokens=4096,
                    stream=True,
                )
                for chunk in resp:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        callback(delta.content)
                return
            except Exception as e:
                last_error = e
        raise RuntimeError(f"API 调用失败（已重试一次）: {last_error}")

    def chat_json(self, system: str, messages: list[dict]) -> dict:
        """发送对话请求，强制返回 JSON，解析为 dict。

        API 调用失败时重试一次，JSON 解析失败时返回 {"narration": raw, ...}。
        """
        import json

        full_messages = [{"role": "system", "content": system}]
        full_messages.extend(messages)

        last_error = None
        for attempt in range(2):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    max_tokens=4096,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content
                return self._parse_json_response(raw)
            except Exception as e:
                last_error = e
        raise RuntimeError(f"API 调用失败（已重试一次）: {last_error}")

    def _parse_json_response(self, raw: str) -> dict:
        """Parse LLM JSON response, with fallback."""
        import json

        text = raw.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Try direct parse first
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Try to extract JSON object from within text (handles LLM adding extra text)
        start_idx = text.find("{")
        if start_idx >= 0:
            # Brace-counting to find the matching closing brace
            depth = 0
            in_string = False
            escape_next = False
            end_idx = -1
            for i in range(start_idx, len(text)):
                ch = text[i]
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\':
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
            if end_idx > start_idx:
                try:
                    result = json.loads(text[start_idx:end_idx + 1])
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    pass

        # Fallback: entire response is narration
        return {
            "narration": raw,
            "tool_calls": [],
            "suggestions": [],
        }

    def extract_memory(self, dialogue: str) -> str:
        """从对话文本中抽取 1-2 条关键事件，用中文一句话概括。

        如果响应为空则返回空字符串。
        """
        system = (
            "你是一个 TRPG 跑团记录员。从以下对话中提取 1-2 条关键事件，用一句中文概括。"
            "以故事叙述视角记录，不要出现骰子数值（d100=57）、检定结果（成功/失败）、"
            "DM元对话（'GM制止'、'玩家选择'）等游戏机制内容。"
            "只记录虚构世界中实际发生了什么。如果没有关键事件则返回空字符串。"
        )
        messages = [{"role": "user", "content": dialogue}]
        result = self.chat(system=system, messages=messages)
        if not result or not result.strip():
            return ""
        return result.strip()

    def generate_search_query(self, user_input: str, context: str = "") -> str:
        """分析玩家输入，生成适用于 ChromaDB 语义搜索的查询字符串。

        提取关键实体、事件主题、时间线索，输出简洁的关键词和短语（20-50字）。
        空响应时返回空字符串，由调用方 fallback。
        """
        system = (
            "你是一个 TRPG 跑团记忆检索助手。分析玩家的输入，提取其中的关键信息，"
            "生成一个简洁的搜索查询（20-50字），用于在向量数据库中检索相关记忆。\n\n"
            "规则：\n"
            "1. 提取关键实体：人名、地名、物品名、组织名\n"
            "2. 提取事件主题：发生了什么、玩家想做什么\n"
            "3. 提取时间线索：之前、昨天、上次、过去等\n"
            "4. 输出格式：只用关键词和短语，不要完整句子，不要叙事\n"
            "5. 如果玩家输入信息量很少（如只是闲聊、简单询问），直接返回玩家原话\n\n"
            "示例：\n"
            "玩家输入：\"你还记得我们之前在码头遇到的那个商人吗？他说有线索\"\n"
            "输出：码头 商人 线索 之前相遇\n\n"
            "玩家输入：\"我想去酒馆打听一下附近有没有奇怪的事情发生\"\n"
            "输出：酒馆 打听消息 奇怪事件 附近\n\n"
            "玩家输入：\"你好\"\n"
            "输出：你好"
        )
        parts = [f"玩家输入: {user_input}"]
        if context:
            parts.append(context)
        messages = [{"role": "user", "content": "\n".join(parts)}]
        result = self.chat(system=system, messages=messages)
        if not result or not result.strip():
            return ""
        return result.strip()
