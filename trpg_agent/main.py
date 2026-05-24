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

# 兼容直接执行和模块运行两种方式
if __name__ == "__main__":
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from trpg_agent.game_master import GameMaster


def main() -> None:
    """Entry point: display welcome panel, then run the main interaction loop."""
    import argparse

    parser = argparse.ArgumentParser(description="TRPG Agent — AI驱动的单人跑团")
    parser.add_argument("--debug", "-d", action="store_true", help="显示每轮管线诊断信息")
    parser.add_argument("--config", "-c", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    console = Console()

    # Config path
    config_path = os.environ.get("TRPG_CONFIG", args.config)

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
        with console.status("[bold cyan]正在启动 TRPG Agent...[/bold cyan]", spinner="dots") as status:
            gm = GameMaster(
                config_path,
                debug=args.debug,
                status_fn=lambda msg: status.update(f"[bold cyan]{msg}[/bold cyan]"),
            )
    except Exception as e:
        console.print(
            Panel(
                f"[red]{e}[/red]",
                title="[bold]初始化失败[/bold]",
                border_style="red",
            )
        )
        sys.exit(1)

    player_name = gm.player.name
    world_name = gm.world.get("name", "未知世界")

    # ---- Load save or generate opening ----
    save_path = "data/save.json"
    if gm.load_save(save_path):
        opening = gm.generate_continuation()
        opening_title = "[bold]续接[/bold]"
    else:
        opening = gm.generate_opening()
        opening_title = "[bold]开场[/bold]"

    # ---- Welcome panel ----
    console.print(
        Panel(
            f"[bold cyan]{player_name}[/bold cyan]，身处[bold]{world_name}[/bold]。\n\n"
            "可用命令：\n"
            "  [green]/dice <表达式>[/green]  投掷骰子（如 [green]/dice 3d6[/green]）\n"
            "  [green]查看状态[/green]  查看角色属性和当前状态\n"
            "  [green]exit[/green] / [green]quit[/green] / [green]退出[/green]  结束冒险\n\n"
            "用括号声明行动，如 [dim](我拔出弓箭，瞄准远处的兽人)[/dim]。\n"
            "也可以直接对 NPC 说话，如 [dim]老马，来杯麦酒[/dim]。",
            title="[bold]TRPG Agent[/bold]",
            subtitle=f"你是 {player_name}",
            border_style="bright_blue",
        )
    )

    # ---- Display opening scene ----
    if opening:
        console.print(
            Panel(
                opening,
                title=opening_title,
                border_style="yellow",
            )
        )

    # ---- Main interaction loop ----
    while True:
        try:
            user_input = Prompt.ask("> ")
        except (KeyboardInterrupt, EOFError):
            console.print()
            gm.save("data/save.json")
            console.print("[dim]进度已保存。冒险结束。再见。[/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # ---- Exit commands ----
        if user_input.lower() in ("exit", "quit", "退出"):
            gm.save("data/save.json")
            console.print("[dim]进度已保存。冒险结束。再见。[/dim]")
            break

        # ---- Slash commands ----
        if user_input.startswith("/"):
            if user_input.startswith("/dice"):
                expr = user_input[len("/dice") :].strip()
                if not expr:
                    reply = "请指定骰子表达式，例如：/dice 3d6"
                else:
                    reply = gm.process(f"掷骰 {expr}", console=console)
            elif user_input.startswith("/clear_npc"):
                arg = user_input[len("/clear_npc") :].strip()
                if not arg:
                    reply = "用法：/clear_npc <NPC名称> 或 /clear_npc all"
                elif arg.lower() == "all":
                    count = gm.npc_store.clear_history()
                    reply = f"已清除全部 {count} 个 NPC 的对话记忆。"
                else:
                    count = gm.npc_store.clear_history(arg)
                    if count:
                        reply = f"已清除 [{arg}] 的对话记忆。"
                    else:
                        reply = f"未找到 NPC [{arg}]。"
            else:
                reply = f"未知命令：{user_input}"
        else:
            reply = gm.process(user_input, console=console)

        # ---- Check game over ----
        if gm._game_over:
            console.print(
                Panel(
                    reply,
                    title="[bold red]冒险终结[/bold red]",
                    border_style="red",
                )
            )
            console.print("[dim]角色已死亡。冒险结束。[/dim]")
            break

        # ---- Display response (suggestions / world events) ----
        if reply and reply.strip():
            console.print(
                Panel(
                    reply.strip(),
                    title="[bold]建议[/bold]" if not reply.startswith("【世界模拟】") else "[bold]世界事件[/bold]",
                    border_style="green",
                )
            )
        else:
            # 流式叙述已完成，打印空行分隔
            console.print()


if __name__ == "__main__":
    main()
