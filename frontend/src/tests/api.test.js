import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, generatePrompt } from '../api.js'

function jsonResponse(status, body) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  })
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('generatePrompt', () => {
  it('sends model and trimmed scenario and returns the generated prompt', async () => {
    const fetchMock = vi.fn(() =>
      jsonResponse(200, { model: 'llama8b', generatedPrompt: 'Role: test prompt' }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const result = await generatePrompt('llama8b', '  Write an email.  ')

    expect(result).toEqual({ model: 'llama8b', generatedPrompt: 'Role: test prompt' })
    const [url, options] = fetchMock.mock.calls[0]
    expect(url).toBe('/generate-prompt')
    expect(options.method).toBe('POST')
    expect(options.headers['Content-Type']).toBe('application/json')
    expect(JSON.parse(options.body)).toEqual({
      model: 'llama8b',
      scenario: 'Write an email.',
    })
  })

  it('maps a backend error response to a readable ApiError', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        jsonResponse(429, {
          error: {
            code: 'AI_QUOTA_EXCEEDED',
            message: 'The AI usage limit has been reached. Please try again after the usage allowance resets.',
          },
        }),
      ),
    )

    await expect(generatePrompt('llama8b', 'Write an email.')).rejects.toMatchObject({
      name: 'ApiError',
      code: 'AI_QUOTA_EXCEEDED',
      status: 429,
      message:
        'The AI usage limit has been reached. Please try again after the usage allowance resets.',
    })
  })

  it('falls back to a generic message when the error body is not JSON', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve({
          ok: false,
          status: 500,
          json: () => Promise.reject(new Error('not json')),
        }),
      ),
    )

    const error = await generatePrompt('llama8b', 'Write an email.').catch((e) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect(error.message).toContain('Request failed with status 500')
  })

  it('reports network failures as a readable error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))),
    )

    await expect(generatePrompt('llama8b', 'Write an email.')).rejects.toMatchObject({
      code: 'NETWORK_ERROR',
    })
  })

  it('rejects an empty generatedPrompt in a 200 response', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => jsonResponse(200, { model: 'llama8b', generatedPrompt: '  ' })),
    )

    await expect(generatePrompt('llama8b', 'Write an email.')).rejects.toMatchObject({
      code: 'INVALID_AI_RESPONSE',
    })
  })
})
