import { useState, useEffect } from 'react'
import { useGameStore, KnowledgeEntry } from '../store/gameStore'

interface ModalKnowledgeProps {
  onClose: () => void
}

export default function ModalKnowledge({ onClose }: ModalKnowledgeProps) {
  const { knowledge, setKnowledge } = useGameStore()
  const [newKnowledge, setNewKnowledge] = useState('')
  const [isAdding, setIsAdding] = useState(false)
  const [addResult, setAddResult] = useState('')
  const [filterCategory, setFilterCategory] = useState('')

  useEffect(() => {
    fetch('/api/knowledge')
      .then((r) => r.json())
      .then((data) => {
        if (data.knowledge) {
          setKnowledge(data.knowledge.map((k: any, i: number) => ({
            id: `k-${i}`,
            content: k.content || k,
            category: k.category || '未分类',
            known_by: k.known_by || '所有人',
          })))
        }
      })
      .catch(() => {})
  }, [])

  const handleAdd = async () => {
    if (!newKnowledge.trim() || isAdding) return
    setIsAdding(true)
    setAddResult('')

    try {
      const res = await fetch('/api/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: `!world ${newKnowledge}` }),
      })
      const data = await res.json()
      setAddResult(data.result || data.error || '添加完成')
      setNewKnowledge('')

      // Refresh
      const refetch = await fetch('/api/knowledge')
      const kData = await refetch.json()
      if (kData.knowledge) {
        setKnowledge(kData.knowledge.map((k: any, i: number) => ({
          id: `k-${i}`,
          content: k.content || k,
          category: k.category || '未分类',
          known_by: k.known_by || '所有人',
        })))
      }
    } catch (err) {
      setAddResult(`添加失败: ${err instanceof Error ? err.message : '未知错误'}`)
    } finally {
      setIsAdding(false)
    }
  }

  const filteredKnowledge = filterCategory
    ? knowledge.filter((k) => k.category === filterCategory)
    : knowledge

  const categories = [...new Set(knowledge.map((k) => k.category))]

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal-content w-full max-w-2xl mx-auto mt-20 max-h-[70vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-coal-700">
          <h2 className="text-xl font-serif text-brass-400">世界观设置</h2>
          <button onClick={onClose} className="text-coal-400 hover:text-coal-200 text-2xl leading-none">
            ×
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Filter */}
          {categories.length > 1 && (
            <div className="flex gap-2 flex-wrap">
              <button
                onClick={() => setFilterCategory('')}
                className={`px-2 py-1 text-xs rounded ${
                  !filterCategory ? 'bg-brass-700 text-coal-50' : 'bg-coal-800 text-coal-400'
                }`}
              >
                全部
              </button>
              {categories.map((cat) => (
                <button
                  key={cat}
                  onClick={() => setFilterCategory(cat)}
                  className={`px-2 py-1 text-xs rounded ${
                    filterCategory === cat ? 'bg-brass-700 text-coal-50' : 'bg-coal-800 text-coal-400'
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>
          )}

          {/* Knowledge list */}
          <div className="space-y-2">
            {filteredKnowledge.length === 0 ? (
              <p className="text-coal-500 text-sm italic">暂无世界观知识</p>
            ) : (
              filteredKnowledge.map((k) => (
                <div
                  key={k.id}
                  className="p-3 bg-coal-800 rounded border border-coal-700"
                >
                  <p className="text-sm text-coal-200">{k.content}</p>
                  <div className="flex gap-2 mt-2 text-xs text-coal-500">
                    <span>分类: {k.category}</span>
                    <span>可见: {k.known_by}</span>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Add knowledge */}
          <div className="decorative-border p-4">
            <h3 className="text-sm text-coal-300 mb-2">添加世界观知识</h3>
            <p className="text-xs text-coal-500 mb-3">
              用自然语言描述世界设定，系统会自动提炼为知识条目
            </p>
            <div className="flex gap-2">
              <textarea
                value={newKnowledge}
                onChange={(e) => setNewKnowledge(e.target.value)}
                placeholder="例如：贝克兰德的东区是贫民窟，治安很差..."
                className="input-field flex-1 min-h-[80px] resize-y"
              />
            </div>
            <button
              onClick={handleAdd}
              disabled={isAdding || !newKnowledge.trim()}
              className="btn-primary mt-2 disabled:opacity-50"
            >
              {isAdding ? '添加中...' : '添加'}
            </button>
            {addResult && (
              <p className="mt-2 text-sm text-brass-300">{addResult}</p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
