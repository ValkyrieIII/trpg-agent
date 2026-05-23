"""TRPG Agent CLI — Rich-powered interactive role-playing interface.

Usage
-----
    python -m trpg_agent.main

Requires a ``config.yaml`` in the working directory (or ``TRPG_CONFIG``
environment variable pointing to an alternate path).
"""

from __future__ import annotations

import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from trpg_agent.game_master import GameMaster


def main() -> None:
    """Entry point: display welcome panel, then run the main interaction loop."""
    console = Console()

    # Config path (from env or default)
    config_path = os.environ.get("TRPG_CONFIG", "config.yaml")

    # ---- Check API key early and warn ----
    if not os.environ.get("DEEPSEEK_API_KEY"):
        console.print()
        console.print(
            Panel(
                "[yellow]DEEPSEEK_API_KEY 环境变量未设置。[/yellow]\n"
                "LLM 对话功能将不可用，但骰子、角色信息和事件判定可以正常使用。\n"
                "请设置环境变量后重启以获得完整功能。",
                title="[bold]提示[/bold]",
                border_style="yellow",
            )
        )
        console.print()

    # ---- Initialize GameMaster ----
    try:
        gm = GameMaster(config_path)
    except Exception as e:
        console.print(
            Panel(
                f"[red]{e}[/red]",
                title="[bold]初始化失败[/bold]",
                border_style="red",
            )
        )
        sys.exit(1)

    character_name = gm.character.name

    # ---- Welcome panel ----
    console.print(
        Panel(
            f"[bold cyan]{character_name}[/bold cyan] 已经准备好踏上冒险了。\n\n"
            "可用命令：\n"
            "  [green]/dice <表达式>[/green]  投掷骰子（如 [green]/dice 3d6[/green]）\n"
            "  [green]exit[/green] / [green]quit[/green] / [green]退出[/green]  结束游戏\n\n"
            "也可以直接输入对话内容与角色互动。",
            title="[bold]TRPG Agent[/bold]",
            subtitle=f"欢迎，{character_name}",
            border_style="bright_blue",
        )
    )

    # ---- Main interaction loop ----
    while True:
        try:
            user_input = Prompt.ask("> ")
        except (KeyboardInterrupt, EOFError):
            console.print()
            console.print("[dim]冒险结束。再见。[/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # ---- Exit commands ----
        if user_input.lower() in ("exit", "quit", "退出"):
            console.print("[dim]冒险结束。再见。[/dim]")
            break

        # ---- Slash commands ----
        if user_input.startswith("/"):
            if user_input.startswith("/dice"):
                expr = user_input[len("/dice") :].strip()
                if not expr:
                    reply = "请指定骰子表达式，例如：/dice 3d6"
                else:
                    reply = gm.process(f"掷骰 {expr}")
            else:
                reply = f"未知命令：{user_input}"
        else:
            reply = gm.process(user_input)

        # ---- Display response ----
        console.print(
            Panel(
                reply,
                title=f"[bold]{character_name}[/bold]",
                border_style="green",
            )
        )


if __name__ == "__main__":
    main()
