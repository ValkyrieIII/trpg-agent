interface StatePanelProps {
  playerState: {
    hp: number
    max_hp: number
    emotion: string
    trust: number
    stamina: string
  }
  npcs: Array<{ name: string; hp: number; max_hp: number }>
}

export default function StatePanel({ playerState: state, npcs }: StatePanelProps) {
  const emotionLabel = (e: string) => {
    if (e === 'calm') return '平静'
    if (e === 'wary') return '警惕'
    if (e === 'hostile') return '敌对'
    return e
  }

  const staminaLabel = (s: string) => {
    if (s === 'fresh') return '充沛'
    if (s === 'tired') return '疲惫'
    if (s === 'exhausted') return '力竭'
    return s
  }

  return (
    <div className="space-y-4">
      {/* Player */}
      <div>
        <h3 className="text-xs text-brass-500 font-bold uppercase tracking-wider mb-2">
          角色
        </h3>
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-coal-400">HP</span>
            <span className="text-coal-200">{state.hp}/{state.max_hp}</span>
          </div>
          <div className="h-1.5 bg-coal-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-blood-600 rounded-full transition-all duration-300"
              style={{ width: `${Math.max(0, (state.hp / state.max_hp) * 100)}%` }}
            />
          </div>

          <div className="flex items-center justify-between text-xs mt-2">
            <span className="text-coal-400">情绪</span>
            <span className="text-coal-200">{emotionLabel(state.emotion)}</span>
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className="text-coal-400">体力</span>
            <span className="text-coal-200">{staminaLabel(state.stamina)}</span>
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className="text-coal-400">信任</span>
            <span className="text-coal-200">{Math.round(state.trust * 100)}%</span>
          </div>
        </div>
      </div>

      {/* NPCs */}
      {npcs.length > 0 && (
        <div>
          <h3 className="text-xs text-brass-500 font-bold uppercase tracking-wider mb-2">
            NPC
          </h3>
          <div className="space-y-2">
            {npcs.map(npc => (
              <div key={npc.name} className="text-xs space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-coal-300">{npc.name}</span>
                  <span className="text-coal-400">{npc.hp}/{npc.max_hp} HP</span>
                </div>
                <div className="h-1 bg-coal-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blood-700/60 rounded-full transition-all duration-300"
                    style={{ width: `${Math.max(0, (npc.hp / npc.max_hp) * 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
