export default function ErrorMessage({ message }) {
  if (!message) return null
  return (
    <div className="error-message" role="alert">
      <strong>Something went wrong:</strong> {message}
    </div>
  )
}
