export default function LoadingIndicator() {
  return (
    <div className="loading-indicator" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      Generating your prompt…
    </div>
  )
}
