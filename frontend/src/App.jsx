import { useRef, useState } from 'react'
import PromptForm from './components/PromptForm.jsx'
import GeneratedPrompt from './components/GeneratedPrompt.jsx'
import ErrorMessage from './components/ErrorMessage.jsx'
import LoadingIndicator from './components/LoadingIndicator.jsx'
import { generatePrompt } from './api.js'
import { validateScenario } from './validation.js'
import { DEFAULT_MODEL } from './models.js'
import './App.css'

export default function App() {
  const [model, setModel] = useState(DEFAULT_MODEL)
  const [scenario, setScenario] = useState('')
  const [validationError, setValidationError] = useState(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [generatedPrompt, setGeneratedPrompt] = useState(null)
  const [generatedModel, setGeneratedModel] = useState(null)
  const [apiError, setApiError] = useState(null)
  const inFlightRef = useRef(false)

  const handleScenarioChange = (value) => {
    setScenario(value)
    if (validationError) setValidationError(null)
  }

  const handleSubmit = async () => {
    if (inFlightRef.current) return // prevent duplicate submissions

    const error = validateScenario(scenario)
    if (error) {
      setValidationError(error)
      return
    }
    setValidationError(null)

    inFlightRef.current = true
    setIsGenerating(true)
    setApiError(null)
    setGeneratedPrompt(null)
    setGeneratedModel(null)

    try {
      const result = await generatePrompt(model, scenario)
      setGeneratedPrompt(result.generatedPrompt)
      setGeneratedModel(result.model)
    } catch (err) {
      setApiError(err?.message || 'Generation failed. Please try again.')
    } finally {
      inFlightRef.current = false
      setIsGenerating(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Prompt Generator</h1>
        <p>
          Describe your scenario and pick a model. You will get a structured,
          ready-to-use prompt for another AI system — not the answer to the
          scenario itself.
        </p>
      </header>

      <main>
        <PromptForm
          model={model}
          scenario={scenario}
          validationError={validationError}
          isGenerating={isGenerating}
          onModelChange={setModel}
          onScenarioChange={handleScenarioChange}
          onSubmit={handleSubmit}
        />

        {isGenerating && <LoadingIndicator />}
        <ErrorMessage message={apiError} />
        <GeneratedPrompt prompt={generatedPrompt} model={generatedModel} />
      </main>

      <footer className="app-footer">
        Powered by Cloudflare Workers (Python) and Workers AI.
      </footer>
    </div>
  )
}
