import { useState, useEffect } from 'react'
import { useGameStore, NPCInfo } from '../store/gameStore';
import { apiFetch, apiPost, apiGet } from '../api';

interface ModalNPCProps {
  onClose: () => void
}

export default function ModalNPC({ onClose }: ModalNPCProps) {
  const { npcs, setNpcs } = useGameStore()
  const [newNpcDesc, setNewNpcDesc] = useState('')
  const [isCreating, setIsCreating] = useState(false)
  const [createResult, setCreateResult] = useState('')

  useEffect(() => {
    apiGet('/api/npcs')
      .then((r) => r.json())
      .then((data) => {
        if (data.npcs) setNpcs(data.npcs)
      })
      .catch(() => {})
  }, [])

  const handleCreate = async () => {
    if (!newNpcDesc.trim() || isCreating) return
    setIsCreating(true)
    setCreateResult('')

    try {
      const res = await apiPost('/api/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: `!npc ${newNpcDesc}` }),
      })
      const data = await res.json()
      setCreateResult(data.result || data.error || '创建完成')

      // Refresh NPC list
      const refetch = await fetch('/api/npcs')
      const npcData = await refetch.json()
      if (npcData.npcs) setNpcs(npcData.npcs)

      setNewNpcDesc('')
    } catch (err) {
      setCreateResult(`创建失败: ${err instanceof Error ? err.message : '未知错误'}`)
    } finally {
      setIsCreating(false)
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal-content w-full max-w-2xl mx-auto mt-20 max-h-[70vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-coal-700">
          <h2 className="text-xl font-serif text-brass-400">NPC 管理</h2>
          <button onClick={onClose} className="text-coal-400 hover:text-coal-200 text-2xl leading-none">
            ×
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* NPC list */}
          <div>
            <h3 className="text-sm text-coal-400 mb-2">当前场景 NPC</h3>
            {npcs.length === 0 ? (
              <p className="text-coal-500 text-sm italic">场景中暂无 NPC</p>
            ) : (
              <div className="space-y-2">
                {npcs.map((npc: NPCInfo, i) => (
                  <div
                    key={i}
                    className="p-3 bg-coal-800 rounded border border-coal-700"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-coal-200 font-medium">{npc.name}</span>
                      <span className="text-xs text-coal-400">
                        HP: {npc.hp}/{npc.max_hp}
                      </span>
                    </div>
                    <p className="text-xs text-coal-400 mt-1">
                      情绪: {npc.emotion} | 信任: {Math.round(npc.trust * 100)}%
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Create NPC */}
          <div className="decorative-border p-4">
            <h3 className="text-sm text-coal-300 mb-2">创建新 NPC</h3>
            <p className="text-xs text-coal-500 mb-3">
              用自然语言描述，系统会自动生成完整的角色卡
            </p>
            <div className="flex gap-2">
              <input
                type="text"
                value={newNpcDesc}
                onChange={(e) => setNewNpcDesc(e.target.value)}
                placeholder="例如：一个在码头工作的老渔民，性格豪爽..."
                className="input-field flex-1"
                onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
              />
              <button
                onClick={handleCreate}
                disabled={isCreating || !newNpcDesc.trim()}
                className="btn-primary disabled:opacity-50"
              >
                {isCreating ? '创建中...' : '创建'}
              </button>
            </div>
            {createResult && (
              <p className="mt-2 text-sm text-brass-300">{createResult}</p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
