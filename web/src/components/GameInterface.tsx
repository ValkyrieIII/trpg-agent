import { useState, useRef, useEffect } from 'react'
import { useGameStore, Message } from '../store/gameStore';
import { apiFetch, apiPost, apiGet } from '../api';
import StatePanel from './StatePanel'
import ModalNPC from './ModalNPC'
import ModalKnowledge from './ModalKnowledge'
import ModalSettings from './ModalSettings'

type ModalType = 'npc' | 'knowledge' | 'settings' | null

export default function GameInterface() {
  const { messages, addMessage, suggestions, setSuggestions, playerState, setPlayerState, npcs, setNpcs, knowledge, setKnowledge, isLoading, setIsLoading, error, setError, streamingText, isStreaming, startStreaming, appendStreaming, stopStreaming, agentStatus, setAgentStatus, currentTool, setCurrentTool } = useGameStore()
  const [inputValue, setInputValue] = useState('')
  const [activeModal, setActiveModal] = useState<ModalType>(null)
  const [gameOverPending, setGameOverPending] = useState(false)
  const [gameOverCause, setGameOverCause] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingText])

  // Fetch player state on mount
  useEffect(() => {
    apiGet('/api/status')
      .then((r) => r.json())
      .then((data) => {
        if (data.state) setPlayerState(data.state)
        if (data.npcs) setNpcs(data.npcs)
      })
      .catch(() => {})
  }, [])

  const sendAction = async (action: string) => {
    if (!action.trim() || isLoading) return

    addMessage('player', action)
    setInputValue('')
    setIsLoading(true)
    setError(null)
    startStreaming()

    try {
      const res = await apiPost('/api/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
      })

      if (!res.ok) {
        const errText = await res.text()
        throw new Error(`请求失败 (${res.status}): ${errText}`)
      }

      const reader = res.body?.getReader()
      const decoder = new TextDecoder()
      let fullText = ''
      let finalData: any = null

      if (reader) {
        let buffer = ''
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            const trimmed = line.trim()
            if (!trimmed || !trimmed.startsWith('data: ')) continue

            const data = trimmed.slice(6)
            if (data === '[DONE]') continue

            try {
              const parsed = JSON.parse(data)
              if (parsed.done) {
                finalData = parsed
              } else if (parsed.type) {
                // Status event from Agent
                if (parsed.type === 'thinking_start') {
                  setAgentStatus('thinking')
                } else if (parsed.type === 'thinking_end') {
                  setAgentStatus('idle')
                } else if (parsed.type === 'tool_call_start') {
                  setAgentStatus('tool_calling')
                  setCurrentTool({ tool: parsed.tool, display: parsed.display, result: null })
                } else if (parsed.type === 'tool_call_end') {
                  setAgentStatus('idle')
                } else if (parsed.type === 'npc_message') {
                  // NPC dialogue — add as a separate NPC message bubble
                  if (parsed.npc_name && parsed.npc_text) {
                    addMessage('npc', parsed.npc_text, parsed.npc_name)
                  }
                }
              } else if (parsed.text) {
                fullText += parsed.text
                appendStreaming(parsed.text)
              }
            } catch {
              // Raw text chunk
              fullText += data
              appendStreaming(data)
            }
          }
        }
      }

      stopStreaming()

      // GM 消息 — narration 已由后端从 JSON 中提取，不含建议行
      if (fullText) {
        addMessage('gm', fullText)
      }

      // 更新建议 — 由后端 JSON suggestions 字段提供
      if (finalData?.suggestions?.length > 0) {
        setSuggestions(finalData.suggestions)
      }

      // Update player state
      if (finalData?.state) {
        setPlayerState(finalData.state)
      }

      // Update NPCs
      if (finalData?.npcs) {
        setNpcs(finalData.npcs)
      }

      // Game over confirmation
      if (finalData?.gameOverPending) {
        setGameOverPending(true)
        setGameOverCause(finalData.gameOverCause || '未知原因')
      }
    } catch (err) {
      stopStreaming()
      setError(err instanceof Error ? err.message : '未知错误')
    } finally {
      setIsLoading(false)
      setAgentStatus('idle')
      setCurrentTool(null)
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    sendAction(inputValue)
  }

  const handleConfirmGameOver = async () => {
    setGameOverPending(false)
    try {
      const res = await apiPost('/api/confirm_game_over')
      const data = await res.json()
      if (data.result) addMessage('gm', data.result)
    } catch {
      setError('确认失败')
    }
  }

  const handleCancelGameOver = async () => {
    setGameOverPending(false)
    try {
      const res = await apiPost('/api/cancel_game_over')
      const data = await res.json()
      if (data.result) addMessage('gm', data.result)
    } catch {
      setError('取消失败')
    }
  }

  return (
    <div className="h-screen flex flex-col bg-coal-950">
      {/* Top bar */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-coal-700 bg-coal-900/80 backdrop-blur">
        <h1 className="text-xl font-serif text-brass-400 glow-text">TRPG Agent</h1>
        <div className="flex gap-2">
          <button onClick={() => setActiveModal('npc')} className="btn-secondary text-sm">
            NPC
          </button>
          <button onClick={() => setActiveModal('knowledge')} className="btn-secondary text-sm">
            世界观
          </button>
          <button onClick={() => setActiveModal('settings')} className="btn-secondary text-sm">
            设置
          </button>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Chat area */}
        <div className="flex-1 flex flex-col">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            {isStreaming && streamingText && (
              <div className="chat-bubble-gm animate-fade-in">
                <p className="whitespace-pre-wrap">{streamingText}</p>
                <span className="animate-typing-cursor text-brass-400" />
              </div>
            )}
            {isLoading && !streamingText && agentStatus === 'thinking' && (
              <div className="flex items-center gap-2 text-coal-400">
                <span className="text-brass-400 font-serif">GM</span>
                <span className="thinking-dots">正在思考</span>
              </div>
            )}
            {isLoading && agentStatus === 'tool_calling' && currentTool && (
              <div className="flex items-center gap-2 text-brass-500 font-mono text-sm animate-fade-in">
                <span className="w-2 h-2 rounded-full bg-brass-500 animate-pulse" />
                <span>调用 {currentTool.display}</span>
              </div>
            )}
            {isLoading && !streamingText && agentStatus === 'idle' && (
              <div className="flex items-center gap-2 text-coal-400 animate-pulse">
                <span className="text-brass-400 font-serif">GM</span>
                <span>正在思考...</span>
              </div>
            )}
            {error && (
              <div className="p-4 bg-blood-900/30 border border-blood-700/50 rounded text-blood-300">
                {error}
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Suggestions */}
          {suggestions.length > 0 && (
            <div className="px-6 pb-3">
              <div className="decorative-border p-3">
                <p className="text-xs text-coal-400 mb-2">建议行动</p>
                <div className="flex flex-wrap gap-2">
                  {suggestions.map((s, i) => (
                    <button
                      key={i}
                      onClick={() => sendAction(s)}
                      className="px-3 py-1.5 bg-coal-800 hover:bg-coal-700 border border-coal-600 hover:border-brass-600 rounded text-sm text-coal-200 transition-all duration-200"
                    >
                      {s.replace(/^\d+\.\s*/, '')}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Input */}
          <form onSubmit={handleSubmit} className="p-4 border-t border-coal-700 bg-coal-900/50">
            <div className="flex gap-3">
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="描述你的行动..."
                className="input-field flex-1"
                disabled={isLoading}
                autoFocus
              />
              <button
                type="submit"
                disabled={isLoading || !inputValue.trim()}
                className="btn-primary disabled:opacity-50"
              >
                发送
              </button>
            </div>
          </form>
        </div>

        {/* State panel */}
        {playerState && (
          <div className="w-72 border-l border-coal-700 bg-coal-900/50 overflow-y-auto p-4">
            <StatePanel playerState={playerState} npcs={npcs} />
          </div>
        )}
      </div>

      {/* Modals */}
      {activeModal === 'npc' && (
        <ModalNPC onClose={() => setActiveModal(null)} />
      )}
      {activeModal === 'knowledge' && (
        <ModalKnowledge onClose={() => setActiveModal(null)} />
      )}
      {activeModal === 'settings' && (
        <ModalSettings onClose={() => setActiveModal(null)} />
      )}

      {/* Game over confirmation dialog */}
      {gameOverPending && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="decorative-border p-6 max-w-md mx-4 bg-coal-900">
            <h2 className="text-xl font-serif text-blood-400 mb-3">游戏结束</h2>
            <p className="text-coal-200 mb-2">{gameOverCause || '角色已死亡'}</p>
            <p className="text-coal-400 text-sm mb-6">确认后将清空所有存档和记忆，不可恢复。</p>
            <div className="flex gap-3">
              <button
                onClick={handleConfirmGameOver}
                className="flex-1 py-2 bg-blood-700/30 border border-blood-600 text-blood-300 rounded hover:bg-blood-700/50 transition"
              >
                确认结束
              </button>
              <button
                onClick={handleCancelGameOver}
                className="flex-1 py-2 bg-coal-700 border border-coal-600 text-coal-300 rounded hover:bg-coal-600 transition"
              >
                取消
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function MessageBubble({ message }: { message: Message }) {
  if (message.role === 'player') {
    return (
      <div className="chat-bubble-player animate-slide-up">
        <p className="whitespace-pre-wrap text-coal-200">{message.content}</p>
      </div>
    )
  }

  if (message.role === 'npc') {
    return (
      <div className="animate-fade-in flex items-start gap-3">
        <div className="flex-shrink-0 w-9 h-9 rounded-full bg-gradient-to-br from-emerald-800 to-emerald-950 border border-emerald-600/40 flex items-center justify-center text-emerald-300 text-sm font-bold shadow-lg">
          {(message.npcName || '?').charAt(0)}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs text-emerald-400 font-medium mb-1">{message.npcName || 'NPC'}</p>
          <div className="bg-emerald-950/40 border border-emerald-700/30 rounded-lg rounded-tl-none px-4 py-2.5">
            <p className="whitespace-pre-wrap text-emerald-100 leading-relaxed">{message.content}</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="chat-bubble-gm animate-fade-in">
      <p className="whitespace-pre-wrap text-coal-100 leading-relaxed">{message.content}</p>
    </div>
  )
}
