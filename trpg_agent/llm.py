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

    def extract_memory(self, dialogue: str) -> str:
        """从对话文本中抽取 1-2 条关键事件，用中文一句话概括。

        如果响应为空则返回空字符串。
        """
        system = (
            "你是一个 TRPG 跑团记录员。"
            "从以下对话中提取 1-2 条关键事件，用一句中文概括。"
            "如果没有关键事件则返回空字符串。"
        )
        messages = [{"role": "user", "content": dialogue}]
        result = self.chat(system=system, messages=messages)
        if not result or not result.strip():
            return ""
        return result.strip()
