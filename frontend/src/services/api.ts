
// Base URL of our FastAPI backend.
const API_BASE_URL = 'http://localhost:8000/api/v1'

// Fetch retrieval and index statistics from the backend.
export async function getEvaluationStats() {
  // Send a GET request to the evaluation statistics endpoint.
  const response = await fetch(
    `${API_BASE_URL}/evaluation/stats`
  )

  // Check whether the backend returned a successful response.
  if (!response.ok) {
    throw new Error('Failed to fetch evaluation statistics')
  }

  // Convert the JSON response into a JavaScript object.
  return response.json()
}

// Fetch all documents from the backend.
export async function getDocuments() {
  // Send a GET request to the documents endpoint.
  const response = await fetch(
    `${API_BASE_URL}/documents/`
  )

  // Check whether the backend returned a successful response.
  if (!response.ok) {
    throw new Error('Failed to fetch documents')
  }

  // Convert the JSON response into a JavaScript array.
  return response.json()
}

// Upload a document to the FastAPI backend.
export async function uploadDocument(file: File) {
  // FormData is used to send files through an HTTP request.
  const formData = new FormData()

  // Add the selected file using the field name expected by FastAPI.
  formData.append('file', file)

  // Send the file to the document upload endpoint.
  const response = await fetch(
    `${API_BASE_URL}/documents/upload`,
    {
      method: 'POST',
      body: formData,
    }
  )

  // Check whether the upload was successful.
  if (!response.ok) {
    throw new Error('Failed to upload document')
  }

  // Convert the backend response into a JavaScript object.
  return response.json()
}

