export const MAX_SCENARIO_LENGTH = 5000

/**
 * Frontend validation for user experience. The backend is authoritative
 * and applies the same rules independently.
 *
 * Returns an error message string, or null when the scenario is valid.
 */
export function validateScenario(scenario) {
  if (typeof scenario !== 'string' || scenario.trim() === '') {
    return 'Please describe the task you need a prompt for.'
  }
  if (scenario.trim().length > MAX_SCENARIO_LENGTH) {
    return `Please enter a scenario containing no more than ${MAX_SCENARIO_LENGTH} characters.`
  }
  return null
}
