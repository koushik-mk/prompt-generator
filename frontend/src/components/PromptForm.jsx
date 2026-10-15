import { MODELS } from '../models.js'
import { MAX_SCENARIO_LENGTH } from '../validation.js'

export default function PromptForm({
  model,
  scenario,
  validationError,
  isGenerating,
  onModelChange,
  onScenarioChange,
  onSubmit,
}) {
  // Enabled unless a generation is running; submitting an empty scenario
  // surfaces an inline validation error instead of being silently blocked.
  const canSubmit = !isGenerating

  return (
    <form
      className="prompt-form"
      onSubmit={(event) => {
        event.preventDefault()
        onSubmit()
      }}
      noValidate
    >
      <div className="field">
        <label htmlFor="model-select">AI model</label>
        <select
          id="model-select"
          value={model}
          onChange={(event) => onModelChange(event.target.value)}
          disabled={isGenerating}
        >
          {MODELS.map((option) => (
            <option key={option.id} value={option.id}>
              {option.name}
            </option>
          ))}
        </select>
      </div>

      <div className="field">
        <label htmlFor="scenario-input">Describe your scenario</label>
        <textarea
          id="scenario-input"
          value={scenario}
          onChange={(event) => onScenarioChange(event.target.value)}
          placeholder="Example: I need an email to apologize for a delayed product shipment."
          rows={6}
          maxLength={MAX_SCENARIO_LENGTH}
          aria-describedby="scenario-help scenario-counter"
          aria-invalid={validationError ? 'true' : undefined}
          disabled={isGenerating}
        />
        <div id="scenario-help" className="help-text">
          Describe the task you want an AI to perform. You will receive an
          optimized prompt for that task, not the task result itself.
        </div>
        <div id="scenario-counter" className="counter" aria-live="polite">
          {scenario.length}/{MAX_SCENARIO_LENGTH} characters
        </div>
        {validationError && (
          <p className="field-error" role="alert">
            {validationError}
          </p>
        )}
      </div>

      <button type="submit" className="generate-button" disabled={!canSubmit}>
        {isGenerating ? 'Generating…' : 'Generate Prompt'}
      </button>
    </form>
  )
}
