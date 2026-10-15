import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from '../App.jsx'

const SCENARIO = 'I need an email to apologize for a delayed product shipment.'

function jsonResponse(status, body) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  }
}

let fetchMock

beforeEach(() => {
  fetchMock = vi.fn()
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('App', () => {
  it('renders the heading, description, and both model options', () => {
    render(<App />)
    expect(
      screen.getByRole('heading', { level: 1, name: /prompt generator/i }),
    ).toBeInTheDocument()
    expect(screen.getByLabelText(/ai model/i)).toBeInTheDocument()

    const options = screen.getAllByRole('option')
    expect(options).toHaveLength(2)
    expect(options[0]).toHaveValue('llama8b')
    expect(options[1]).toHaveValue('llama1b')
    expect(options[0].textContent).toMatch(/Llama 3\.1 8B/i)
    expect(options[1].textContent).toMatch(/Llama 3\.2 1B/i)
  })

  it('accepts scenario input', async () => {
    const user = userEvent.setup()
    render(<App />)
    const textarea = screen.getByLabelText(/describe your scenario/i)
    await user.type(textarea, SCENARIO)
    expect(textarea).toHaveValue(SCENARIO)
  })

  it('shows a validation error and does not call the API for an empty scenario', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('button', { name: /generate prompt/i }))

    expect(
      await screen.findByRole('alert'),
    ).toHaveTextContent(/please describe the task/i)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('shows the loading state, disables the button, and displays the generated prompt', async () => {
    let resolveFetch
    fetchMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveFetch = resolve
        }),
    )
    const user = userEvent.setup()
    render(<App />)

    await user.type(screen.getByLabelText(/describe your scenario/i), SCENARIO)
    await user.click(screen.getByRole('button', { name: /generate prompt/i }))

    // Loading state visible, controls disabled.
    expect(await screen.findByText(/generating your prompt/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /generating/i })).toBeDisabled()
    expect(screen.getByLabelText(/describe your scenario/i)).toBeDisabled()

    resolveFetch(
      jsonResponse(200, {
        model: 'llama8b',
        generatedPrompt: 'Role: Act as an experienced customer support specialist…',
      }),
    )

    expect(await screen.findByText(/customer support specialist/i)).toBeInTheDocument()
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /generate prompt/i })).toBeEnabled(),
    )
    expect(screen.queryByText(/generating your prompt/i)).not.toBeInTheDocument()

    // The request carried the selected model and scenario.
    const [url, options] = fetchMock.mock.calls[0]
    expect(url).toBe('/generate-prompt')
    expect(JSON.parse(options.body)).toEqual({ model: 'llama8b', scenario: SCENARIO })
  })

  it('prevents duplicate submissions while a request is in flight', async () => {
    let resolveFetch
    fetchMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveFetch = resolve
        }),
    )
    const user = userEvent.setup()
    render(<App />)
    await user.type(screen.getByLabelText(/describe your scenario/i), SCENARIO)

    const button = screen.getByRole('button', { name: /generate prompt/i })
    await user.click(button)
    // Button is disabled during generation; forced clicks must not refire.
    button.disabled = false
    await user.click(button).catch(() => {})
    expect(fetchMock).toHaveBeenCalledTimes(1)

    resolveFetch(
      jsonResponse(200, { model: 'llama8b', generatedPrompt: 'Role: test' }),
    )
    await screen.findByText(/role: test/i)
  })

  it('sends the selected model when the second option is chosen', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(200, { model: 'llama1b', generatedPrompt: 'Role: test prompt' }),
    )
    const user = userEvent.setup()
    render(<App />)
    await user.selectOptions(screen.getByLabelText(/ai model/i), 'llama1b')
    await user.type(screen.getByLabelText(/describe your scenario/i), SCENARIO)
    await user.click(screen.getByRole('button', { name: /generate prompt/i }))

    await screen.findByText(/role: test prompt/i)
    expect(JSON.parse(fetchMock.mock.calls[0][1].body).model).toBe('llama1b')
  })

  it('displays a readable error message when the API returns an error', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(429, {
        error: {
          code: 'AI_QUOTA_EXCEEDED',
          message:
            'The AI usage limit has been reached. Please try again after the usage allowance resets.',
        },
      }),
    )
    const user = userEvent.setup()
    render(<App />)
    await user.type(screen.getByLabelText(/describe your scenario/i), SCENARIO)
    await user.click(screen.getByRole('button', { name: /generate prompt/i }))

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent(/usage limit has been reached/i)
    // The form is usable again.
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /generate prompt/i })).toBeEnabled(),
    )
  })

  it('copies the generated prompt to the clipboard', async () => {
    const user = userEvent.setup()
    // Override after setup(): user-event installs its own clipboard stub.
    const writeText = vi.fn(() => Promise.resolve())
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText },
      configurable: true,
      writable: true,
    })
    fetchMock.mockResolvedValue(
      jsonResponse(200, { model: 'llama8b', generatedPrompt: 'Role: copy me' }),
    )
    render(<App />)
    await user.type(screen.getByLabelText(/describe your scenario/i), SCENARIO)
    await user.click(screen.getByRole('button', { name: /generate prompt/i }))

    await screen.findByText(/role: copy me/i)
    await user.click(screen.getByRole('button', { name: /copy prompt/i }))

    await waitFor(() => expect(writeText).toHaveBeenCalledWith('Role: copy me'))
    expect(await screen.findByText(/copied to clipboard/i)).toBeInTheDocument()
  })

  it('handles clipboard failure gracefully', async () => {
    const user = userEvent.setup()
    // Override after setup(): user-event installs its own clipboard/execCommand stubs.
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText: vi.fn(() => Promise.reject(new Error('denied'))) },
      configurable: true,
      writable: true,
    })
    document.execCommand = vi.fn(() => false)
    fetchMock.mockResolvedValue(
      jsonResponse(200, { model: 'llama8b', generatedPrompt: 'Role: copy me' }),
    )
    render(<App />)
    await user.type(screen.getByLabelText(/describe your scenario/i), SCENARIO)
    await user.click(screen.getByRole('button', { name: /generate prompt/i }))

    await screen.findByText(/role: copy me/i)
    await user.click(screen.getByRole('button', { name: /copy prompt/i }))

    expect(await screen.findByText(/copy manually/i)).toBeInTheDocument()
  })
})
