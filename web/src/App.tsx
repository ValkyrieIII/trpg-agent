import { useState, useEffect } from 'react'
import { useGameStore } from './store/gameStore'
import GameInterface from './components/GameInterface'
import StartScreen from './components/StartScreen'

function App() {
  const gameStarted = useGameStore((s) => s.gameStarted)

  return (
    <div className="min-h-screen bg-coal-950">
      {gameStarted ? <GameInterface /> : <StartScreen />}
    </div>
  )
}

export default App
