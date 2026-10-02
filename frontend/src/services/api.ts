// Base URL of our FastAPI backend.
// Using a relative path allows the frontend to work locally and on AWS.
const API_BASE_URL = '/api/v1'


// ============================================================
// EVALUATION / STATISTICS
// ============================================================

// Fetch retrieval and index statistics from the backend.
export async function getEvaluationStats() {
  const response = await fetch(
    `${API_BASE_URL}/evaluation/stats`
  )

  if (!response.ok) {
    throw new Error(
      'Failed to fetch evaluation statistics'
    )
  }

  return response.json()
}


// ============================================================
// DOCUMENTS
// ============================================================

// Fetch all documents from the backend.
export async function getDocuments() {
  const response = await fetch(
    `${API_BASE_URL}/documents/`
  )

  if (!response.ok) {
    throw new Error('Failed to fetch documents')
  }

  return response.json()
}


// Upload a document to the FastAPI backend.
export async function uploadDocument(file: File) {
  const formData = new FormData()

  formData.append('file', file)

  const response = await fetch(
    `${API_BASE_URL}/documents/upload`,
    {
      method: 'POST',
      body: formData,
    }
  )

  if (!response.ok) {
    throw new Error('Failed to upload document')
  }

  return response.json()
}


// Delete a document from the FastAPI backend.
export async function deleteDocument(
  filename: string
) {
  const response = await fetch(
    `${API_BASE_URL}/documents/${encodeURIComponent(filename)}`,
    {
      method: 'DELETE',
    }
  )

  if (!response.ok) {
    throw new Error('Failed to delete document')
  }

  return response.json()
}


// Reindex an existing document using the backend pipeline.
export async function reindexDocument(
  filename: string
) {
  const response = await fetch(
    `${API_BASE_URL}/documents/${encodeURIComponent(filename)}/reindex`,
    {
      method: 'POST',
    }
  )

  if (!response.ok) {
    throw new Error('Failed to reindex document')
  }

  return response.json()
}


// ============================================================
// CHAT / RESEARCH
// ============================================================

// Send a question to the conversational RAG pipeline.
export async function sendChatMessage(
  question: string,
  sessionId: string,
) {
  const response = await fetch(
    `${API_BASE_URL}/chat/`,
    {
      method: 'POST',

      // Tell the backend that we are sending JSON.
      headers: {
        'Content-Type': 'application/json',
      },

      // Send the question and conversation session ID.
      body: JSON.stringify({
        question,
        session_id: sessionId,
      }),
    }
  )

  if (!response.ok) {
    throw new Error('Failed to send chat message')
  }

  return response.json()
}


// Fetch the conversation history for a session.
export async function getConversation(
  sessionId: string
) {
  const response = await fetch(
    `${API_BASE_URL}/chat/conversations/${encodeURIComponent(sessionId)}`
  )

  if (!response.ok) {
    throw new Error('Failed to fetch conversation')
  }

  return response.json()
}


// Delete the conversation history for a session.
export async function deleteConversation(
  sessionId: string
) {
  const response = await fetch(
    `${API_BASE_URL}/chat/conversations/${encodeURIComponent(sessionId)}`,
    {
      method: 'DELETE',
    }
  )

  if (!response.ok) {
    throw new Error('Failed to delete conversation')
  }

  return response.json()
}


// ============================================================
// SEARCH
// ============================================================

// Search the indexed document collection.
export async function searchDocuments(
  query: string,
  limit: number = 5,
) {
  const response = await fetch(
    `${API_BASE_URL}/search/`,
    {
      method: 'POST',

      // Tell FastAPI that we are sending JSON.
      headers: {
        'Content-Type': 'application/json',
      },

      // Send the search query and result limit.
      body: JSON.stringify({
        query,
        limit,
      }),
    }
  )

  if (!response.ok) {
    throw new Error('Failed to search documents')
  }

  return response.json()
}


// Fetch retrieval/index statistics.
export async function getSearchStats() {
  const response = await fetch(
    `${API_BASE_URL}/search/stats`
  )

  if (!response.ok) {
    throw new Error(
      'Failed to fetch search statistics'
    )
  }

  return response.json()
}


// Run the retrieval evaluation benchmark.
export async function runEvaluation() {
  const response = await fetch(`${API_BASE_URL}/evaluation/run`)

  if (!response.ok) {
    throw new Error('Failed to run evaluation')
  }

  return response.json()
}