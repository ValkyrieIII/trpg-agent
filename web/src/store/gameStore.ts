import { create } from 'zustand'

export interface PlayerState {
  hp: number
  max_hp: number
  emotion: string
  trust: number
  stamina: string
  madness?: number // deprecated, kept for backward compat with old saves
}

export interface Message {
  id: string
  role: 'gm' | 'player'
  content: string
  timestamp: number
}

export interface NPCInfo {
  name: string
  emotion: string
  trust: number
  hp: number
  max_hp: number
}

export interface KnowledgeEntry {
  id: string
  content: string
  category: string
  known_by: string
}

export type AgentStatus = 'idle' | 'thinking' | 'tool_calling'

export interface ToolCallInfo {
  tool: string
  display: string
  result: string | null
}

interface GameState {
  // Messages
  messages: Message[]
  addMessage: (role: 'gm' | 'player', content: string) => void
  clearMessages: () => void

  // Streaming
  isStreaming: boolean
  streamingText: string
  startStreaming: () => void
  appendStreaming: (text: string) => void
  stopStreaming: () => void

  // Agent status (thinking / tool calling)
  agentStatus: AgentStatus
  currentTool: ToolCallInfo | null
  setAgentStatus: (status: AgentStatus) => void
  setCurrentTool: (tool: ToolCallInfo | null) => void

  // Suggestions
  suggestions: string[]
  setSuggestions: (suggestions: string[]) => void

  // Player state
  playerState: PlayerState | null
  setPlayerState: (state: PlayerState) => void

  // NPCs
  npcs: NPCInfo[]
  setNpcs: (npcs: NPCInfo[]) => void

  // Knowledge
  knowledge: KnowledgeEntry[]
  setKnowledge: (knowledge: KnowledgeEntry[]) => void
  addKnowledge: (entry: KnowledgeEntry) => void
  removeKnowledge: (id: string) => void

  // Game status
  gameStarted: boolean
  setGameStarted: (started: boolean) => void
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
  error: string | null
  setError: (error: string | null) => void

  // World simulation
  worldSimulationEnabled: boolean
  setWorldSimulationEnabled: (enabled: boolean) => void

  // Debug
  debugMode: boolean
  setDebugMode: (enabled: boolean) => void
  debugLogs: string[]
  appendDebugLogs: (logs: string[]) => void
  clearDebugLogs: () => void
}

export const useGameStore = create<GameState>((set) => ({
  messages: [],
  addMessage: (role, content) =>
    set((state) => ({
      messages: [
        ...state.messages,
        { id: `${Date.now()}-${Math.random()}`, role, content, timestamp: Date.now() },
      ],
    })),
  clearMessages: () => set({ messages: [] }),

  isStreaming: false,
  streamingText: '',
  startStreaming: () => set({ isStreaming: true, streamingText: '' }),
  appendStreaming: (text) =>
    set((state) => ({ streamingText: state.streamingText + text })),
  stopStreaming: () => set({ isStreaming: false }),

  suggestions: [],
  setSuggestions: (suggestions) => set({ suggestions }),

  playerState: null,
  setPlayerState: (playerState) => set({ playerState }),

  npcs: [],
  setNpcs: (npcs) => set({ npcs }),

  knowledge: [],
  setKnowledge: (knowledge) => set({ knowledge }),
  addKnowledge: (entry) =>
    set((state) => ({ knowledge: [...state.knowledge, entry] })),
  removeKnowledge: (id) =>
    set((state) => ({ knowledge: state.knowledge.filter((k) => k.id !== id) })),

  gameStarted: false,
  setGameStarted: (gameStarted) => set({ gameStarted }),

  isLoading: false,
  setIsLoading: (isLoading) => set({ isLoading }),

  error: null,
  setError: (error) => set({ error }),

  agentStatus: 'idle',
  currentTool: null,
  setAgentStatus: (agentStatus) => set({ agentStatus }),
  setCurrentTool: (currentTool) => set({ currentTool }),

  worldSimulationEnabled: true,
  setWorldSimulationEnabled: (worldSimulationEnabled) => set({ worldSimulationEnabled }),

  debugMode: false,
  setDebugMode: (debugMode) => set({ debugMode }),
  debugLogs: [],
  appendDebugLogs: (logs) =>
    set((state) => ({ debugLogs: [...state.debugLogs, ...logs] })),
  clearDebugLogs: () => set({ debugLogs: [] }),
}))
