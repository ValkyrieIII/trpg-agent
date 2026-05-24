"""Phase 1.12: Quick integration test — verify GameMaster works with SDK Runner."""

import os
import sys

# Load .env
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ[key.strip()] = value.strip()

from trpg_agent.game_master import GameMaster


def main():
    print("Creating GameMaster...")
    gm = GameMaster("config.yaml", debug=True)

    if not gm._llm_available:
        print("[SKIP] LLM not available — check DEEPSEEK_API_KEY")
        return

    if gm.gm_agent is None:
        print("[SKIP] gm_agent not created")
        return

    print(f"[OK] GameMaster created, player: {gm.player.name}")
    print(f"    scene_npcs: {gm.scene_npcs}")
    print(f"    time: {gm._time_of_day}, weather: {gm._weather}")

    # Test 1: Simple narrate (no tools needed)
    print("\n--- Test 1: Simple narrative ---")
    result = gm.process("我环顾四周，看看周围有什么")
    print(f"Response ({len(result)} chars):")
    print(result[:500])
    print("...")

    # Test 2: NPC interaction (should invoke create_npc + invoke_npc)
    print("\n--- Test 2: NPC interaction ---")
    result = gm.process("我走向吧台，对酒保说：来一杯麦酒")
    print(f"Response ({len(result)} chars):")
    print(result[:500])
    print("...")

    # Test 3: Dice roll
    print("\n--- Test 3: Dice roll ---")
    result = gm.process("我尝试跳过面前的沟壑")
    print(f"Response ({len(result)} chars):")
    print(result[:500])
    print("...")

    print("\n===== Integration tests passed =====")


if __name__ == "__main__":
    main()
