import { useEffect, useRef, useState } from 'react'

async function copyText(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text)
    return
  }
  const helper = document.createElement('textarea')
  helper.value = text
  helper.setAttribute('readonly', '')
  helper.style.position = 'fixed'
  helper.style.opacity = '0'
  document.body.appendChild(helper)
  helper.select()
  const ok = document.execCommand('copy')
  document.body.removeChild(helper)
  if (!ok) {
    throw new Error('Copy command failed')
  }
}

export default function GeneratedPrompt({ prompt, model }) {
  const [copyState, setCopyState] = useState(null) // 'success' | 'error' | null
  const timeoutRef = useRef(null)

  useEffect(() => {
    setCopyState(null)
  }, [prompt])

  useEffect(() => () => clearTimeout(timeoutRef.current), [])

  if (!prompt) return null

  const handleCopy = async () => {
    try {
      await copyText(prompt)
      setCopyState('success')
    } catch {
      setCopyState('error')
    }
    clearTimeout(timeoutRef.current)
    timeoutRef.current = setTimeout(() => setCopyState(null), 3000)
  }

  return (
    <section className="generated-prompt" aria-labelledby="generated-heading">
      <div className="generated-header">
        <h2 id="generated-heading">Generated Prompt</h2>
        <button type="button" className="copy-button" onClick={handleCopy}>
          Copy Prompt
        </button>
      </div>
      <p className="generated-model">
        Generated with <span className="model-id">{model}</span>. Paste this
        prompt into the AI system of your choice.
      </p>
      <output className="prompt-output" htmlFor="scenario-input">
        {prompt}
      </output>
      <div className="copy-status" aria-live="polite">
        {copyState === 'success' && 'Prompt copied to clipboard.'}
        {copyState === 'error' &&
          'Could not copy automatically. Select the prompt text and copy manually.'}
      </div>
    </section>
  )
}
