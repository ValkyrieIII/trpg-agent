import { useState, useEffect, useRef } from 'react'
import { useGameStore } from '../store/gameStore'

const LOADING_STEPS = [
  '连接 LLM...',
  '加载角色卡...',
  '初始化 NPC 存储...',
  '初始化记忆存储...',
  '初始化知识库...',
  '加载世界知识...',
  '正在生成开场叙述...',
]

export default function StartScreen() {
  const { setGameStarted, setIsLoading, isLoading, error, setError, addMessage, setSuggestions, setPlayerState, setNpcs } = useGameStore()
  const [mode, setMode] = useState<'new' | 'load'>('new')
  const [worldView, setWorldView] = useState('')
  const [npcSetup, setNpcSetup] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [loadingStep, setLoadingStep] = useState(0)
  const loadingTimer = useRef<ReturnType<typeof setInterval>>()

  // 加载时轮换进度描述
  useEffect(() => {
    if (isLoading) {
      setLoadingStep(0)
      loadingTimer.current = setInterval(() => {
        setLoadingStep((prev) => (prev < LOADING_STEPS.length - 1 ? prev + 1 : prev))
      }, 1500)
    } else {
      clearInterval(loadingTimer.current)
    }
    return () => clearInterval(loadingTimer.current)
  }, [isLoading])

  const handleStart = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const res = await fetch('/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          worldview: worldView || undefined,
          npc_setup: npcSetup || undefined,
        }),
      })

      if (!res.ok) {
        const errText = await res.text()
        throw new Error(`启动失败 (${res.status}): ${errText}`)
      }

      const data = await res.json()
      if (data.error) {
        throw new Error(data.error)
      }

      // 如果有开场叙述，添加到消息列表
      if (data.opening) {
        addMessage('gm', data.opening)
      }

      // 建议由后端 JSON suggestions 字段提供
      if (data.suggestions?.length > 0) {
        setSuggestions(data.suggestions)
      }

      // 设置玩家状态
      if (data.state) {
        setPlayerState(data.state)
      }

      // 设置 NPC
      if (data.npcs) {
        setNpcs(data.npcs)
      }

      // 进入游戏界面
      setGameStarted(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : '未知错误')
    } finally {
      setIsLoading(false)
    }
  }

  const handleLoad = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const res = await fetch('/api/load', {
        method: 'POST',
      })

      if (!res.ok) {
        const errText = await res.text()
        throw new Error(`加载失败 (${res.status}): ${errText}`)
      }

      const data = await res.json()
      if (data.error) {
        throw new Error(data.error)
      }

      if (data.loaded) {
        if (data.state) {
          setPlayerState(data.state)
        }
        if (data.npcs) {
          setNpcs(data.npcs)
        }
        setGameStarted(true)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '未知错误')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-coal-950">
      {/* Background decoration */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-brass-900/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blood-900/10 rounded-full blur-3xl" />
      </div>

      <div className="relative z-10 w-full max-w-lg px-6">
        {/* Title */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-serif text-brass-400 glow-text mb-2">
            TRPG Agent
          </h1>
          <p className="text-coal-400 text-lg italic">自定义你的世界</p>
          <div className="mt-4 w-24 h-px bg-gradient-to-r from-transparent via-brass-600 to-transparent mx-auto" />
        </div>

        {/* Mode selection */}
        <div className="decorative-border p-6">
          <div className="flex gap-3 mb-6">
            <button
              onClick={() => setMode('new')}
              disabled={isLoading}
              className={`flex-1 py-2 rounded text-center transition-all duration-200 border ${
                mode === 'new'
                  ? 'bg-brass-700/20 border-brass-600 text-brass-300'
                  : 'bg-coal-800 border-coal-600 text-coal-400 hover:border-brass-700/50'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              新游戏
            </button>
            <button
              onClick={() => setMode('load')}
              disabled={isLoading}
              className={`flex-1 py-2 rounded text-center transition-all duration-200 border ${
                mode === 'load'
                  ? 'bg-brass-700/20 border-brass-600 text-brass-300'
                  : 'bg-coal-800 border-coal-600 text-coal-400 hover:border-brass-700/50'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              加载存档
            </button>
          </div>

          {mode === 'new' ? (
            <div className="space-y-4">
              {/* Worldview input */}
              <div>
                <label className="block text-sm text-coal-300 mb-1">
                  世界观设定 <span className="text-coal-500">(可选，留空使用默认)</span>
                </label>
                <textarea
                  value={worldView}
                  onChange={(e) => setWorldView(e.target.value)}
                  placeholder="例：蒸汽朋克世界，第五纪1350年，鲁恩王国首都贝克兰德..."
                  className="input-field w-full h-20 resize-none text-sm"
                  disabled={isLoading}
                />
              </div>

              {/* NPC setup */}
              <div>
                <label className="block text-sm text-coal-300 mb-1">
                  初始 NPC <span className="text-coal-500">(可选，用自然语言描述)</span>
                </label>
                <textarea
                  value={npcSetup}
                  onChange={(e) => setNpcSetup(e.target.value)}
                  placeholder="例：我的妹妹艾莉西亚，温柔但倔强；邻居老马，退休的蒸汽工程师..."
                  className="input-field w-full h-16 resize-none text-sm"
                  disabled={isLoading}
                />
              </div>

              {/* Advanced toggle */}
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="text-xs text-coal-500 hover:text-brass-500 transition-colors"
              >
                {showAdvanced ? '收起高级选项' : '高级选项'}
              </button>

              {showAdvanced && (
                <div className="p-3 bg-coal-800/50 rounded border border-coal-700 text-xs text-coal-400">
                  <p>提示：世界观和 NPC 留空时，将使用 config.yaml 中的默认设定。</p>
                  <p>可用自然语言描述任意世界观和角色，系统会自动生成。</p>
                </div>
              )}

              <button
                onClick={handleStart}
                disabled={isLoading}
                className="w-full py-3 btn-primary text-lg disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? LOADING_STEPS[loadingStep] : '开始冒险'}
              </button>

              {/* Loading progress bar */}
              {isLoading && (
                <div className="mt-3">
                  <div className="h-1 bg-coal-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-brass-600 rounded-full transition-all duration-1000 animate-pulse"
                      style={{ width: `${((loadingStep + 1) / LOADING_STEPS.length) * 100}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-4">
              <p className="text-sm text-coal-400 text-center">
                加载最近的存档继续游戏
              </p>
              <button
                onClick={handleLoad}
                disabled={isLoading}
                className="w-full py-3 btn-primary text-lg disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? '加载中...' : '加载存档'}
              </button>
            </div>
          )}

          {error && (
            <div className="mt-4 p-3 bg-blood-900/30 border border-blood-700/50 rounded text-blood-300 text-sm">
              {error}
            </div>
          )}
        </div>

        {/* Footer */}
        <p className="text-center text-coal-500 text-sm mt-8">
          用括号声明行动，如 (我拔出弓箭，瞄准远处的兽人)
        </p>
      </div>
    </div>
  )
}
