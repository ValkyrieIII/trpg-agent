import { useEffect, useRef } from 'react'
import { useGameStore } from '../store/gameStore'

interface ModalSettingsProps {
  onClose: () => void
}

export default function ModalSettings({ onClose }: ModalSettingsProps) {
  const {
    worldSimulationEnabled,
    setWorldSimulationEnabled,
    debugMode,
    setDebugMode,
    debugLogs,
    appendDebugLogs,
    clearDebugLogs,
    setGameStarted,
    clearMessages,
    setSuggestions,
    setPlayerState,
    setNpcs,
    setIsLoading,
    setError,
  } = useGameStore()

  const logEndRef = useRef<HTMLDivElement>(null)
  const pollingTimer = useRef<ReturnType<typeof setInterval>>()

  // 调试模式开启时轮询后端日志
  useEffect(() => {
    if (debugMode) {
      pollingTimer.current = setInterval(async () => {
        try {
          const res = await fetch('/api/debug')
          const data = await res.json()
          if (data.logs?.length > 0) {
            appendDebugLogs(data.logs)
          }
        } catch {
          // 后端未响应，忽略
        }
      }, 2000)
    } else {
      clearInterval(pollingTimer.current)
    }
    return () => clearInterval(pollingTimer.current)
  }, [debugMode])

  // 日志自动滚动到底部
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [debugLogs])

  const handleToggleDebug = async () => {
    try {
      const res = await fetch('/api/debug/toggle', { method: 'POST' })
      const data = await res.json()
      setDebugMode(data.debug)
      if (data.debug) {
        clearDebugLogs()
      }
    } catch {
      setDebugMode(!debugMode)
    }
  }

  const handleSave = async () => {
    try {
      const res = await fetch('/api/save', { method: 'POST' })
      const data = await res.json()
      if (data.error) {
        alert(`保存失败: ${data.error}`)
      } else {
        alert('存档已保存')
      }
    } catch {
      alert('保存失败')
    }
  }

  const handleLoad = async () => {
    try {
      setIsLoading(true)
      const res = await fetch('/api/load', { method: 'POST' })
      const data = await res.json()
      if (data.error) {
        alert(data.error)
      } else {
        if (data.state) setPlayerState(data.state)
        if (data.npcs) setNpcs(data.npcs)
        setSuggestions([])
        alert('存档已加载')
        onClose()
      }
    } catch {
      alert('加载失败')
    } finally {
      setIsLoading(false)
    }
  }

  const handleNewGame = () => {
    if (confirm('确定要开始新游戏吗？当前进度将丢失。')) {
      clearMessages()
      setSuggestions([])
      setGameStarted(false)
      onClose()
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal-content w-full max-w-md mx-auto mt-4 max-h-[90vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-coal-700 shrink-0">
          <h2 className="text-xl font-serif text-brass-400">游戏设置</h2>
          <button onClick={onClose} className="text-coal-400 hover:text-coal-200 text-2xl leading-none">
            ×
          </button>
        </div>

        {/* Scrollable content */}
        <div className="overflow-y-auto p-4 space-y-6">
          {/* World Simulation */}
          <div>
            <h3 className="text-sm text-coal-300 mb-2">世界模拟</h3>
            <div className="flex items-center justify-between p-3 bg-coal-800 rounded border border-coal-700">
              <div>
                <p className="text-sm text-coal-200">自动 NPC 行动</p>
                <p className="text-xs text-coal-500">NPC 会在适当时机自主行动</p>
              </div>
              <button
                onClick={() => setWorldSimulationEnabled(!worldSimulationEnabled)}
                className={`w-12 h-6 rounded-full transition-all duration-200 ${
                  worldSimulationEnabled ? 'bg-brass-600' : 'bg-coal-600'
                }`}
              >
                <div
                  className={`w-5 h-5 rounded-full bg-coal-50 shadow transition-transform duration-200 ${
                    worldSimulationEnabled ? 'translate-x-6' : 'translate-x-0.5'
                  }`}
                />
              </button>
            </div>
          </div>

          {/* Debug Mode */}
          <div>
            <h3 className="text-sm text-coal-300 mb-2">调试模式</h3>
            <div className="flex items-center justify-between p-3 bg-coal-800 rounded border border-coal-700">
              <div>
                <p className="text-sm text-coal-200">启用 GM 调试日志</p>
                <p className="text-xs text-coal-500">显示 LLM 调用、工具执行、记忆检索等诊断信息</p>
              </div>
              <button
                onClick={handleToggleDebug}
                className={`w-12 h-6 rounded-full transition-all duration-200 ${
                  debugMode ? 'bg-brass-600' : 'bg-coal-600'
                }`}
              >
                <div
                  className={`w-5 h-5 rounded-full bg-coal-50 shadow transition-transform duration-200 ${
                    debugMode ? 'translate-x-6' : 'translate-x-0.5'
                  }`}
                />
              </button>
            </div>

            {/* Debug log panel */}
            {debugMode && (
              <div className="mt-3 p-3 bg-coal-950 border border-coal-700 rounded max-h-48 overflow-y-auto">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-brass-500 font-mono">DEBUG LOG</span>
                  <button
                    onClick={clearDebugLogs}
                    className="text-xs text-coal-500 hover:text-coal-300"
                  >
                    清空
                  </button>
                </div>
                {debugLogs.length === 0 ? (
                  <p className="text-xs text-coal-500 italic">等待日志输出...</p>
                ) : (
                  <div className="space-y-1">
                    {debugLogs.map((log, i) => (
                      <p key={i} className="text-xs text-coal-300 font-mono whitespace-pre-wrap leading-relaxed">
                        {log}
                      </p>
                    ))}
                    <div ref={logEndRef} />
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Save/Load */}
          <div>
            <h3 className="text-sm text-coal-300 mb-2">存档管理</h3>
            <div className="space-y-2">
              <button onClick={handleSave} className="btn-primary w-full">
                保存进度
              </button>
              <button onClick={handleLoad} className="btn-secondary w-full">
                加载存档
              </button>
            </div>
          </div>

          {/* New Game */}
          <div>
            <button onClick={handleNewGame} className="w-full py-2 text-blood-400 hover:text-blood-300 text-sm border border-blood-800 hover:border-blood-600 rounded transition-all">
              开始新游戏
            </button>
          </div>

          {/* Debug info */}
          <div className="p-3 bg-coal-800/50 rounded border border-coal-700">
            <p className="text-xs text-coal-500">
              TRPG Agent v0.2.0
            </p>
            <p className="text-xs text-coal-500 mt-1">
              AI GM · 单人跑团 · 流式输出
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
