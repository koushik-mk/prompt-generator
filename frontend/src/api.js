export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

export class ApiError extends Error {
  constructor(message, code, status) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
  }
}

/**
 * Calls POST /generate-prompt and returns { model, generatedPrompt }.
 * Throws ApiError with a user-readable message on any failure.
 */
export async function generatePrompt(model, scenario) {
  let response
  try {
    response = await fetch(`${API_BASE_URL}/generate-prompt`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, scenario: scenario.trim() }),
    })
  } catch {
    throw new ApiError(
      'Could not reach the Prompt Generator service. Check your connection and try again.',
      'NETWORK_ERROR',
      0,
    )
  }

  let payload = null
  try {
    payload = await response.json()
  } catch {
    payload = null
  }

  if (!response.ok) {
    const message =
      payload?.error?.message ?? `Request failed with status ${response.status}. Please try again later.`
    throw new ApiError(message, payload?.error?.code ?? 'UNKNOWN_ERROR', response.status)
  }

  const generated = payload?.generatedPrompt
  if (typeof generated !== 'string' || generated.trim() === '') {
    throw new ApiError(
      'The service returned an unusable response. Please try again.',
      'INVALID_AI_RESPONSE',
      response.status,
    )
  }

  return { model: payload.model, generatedPrompt: generated }
}
