//
// Import React hooks for state and running code when the component loads.
import { useEffect, useState } from 'react'

// Import the API functions used to fetch backend statistics.
import { getEvaluationStats, getDocuments } from '../services/api'

function Dashboard() {
  // Dashboard is the main overview page of our application.

  // Store the number of chunks in the vector database.
  const [vectorChunks, setVectorChunks] = useState(0)

  // Store the number of chunks in the BM25 index.
  const [bm25Chunks, setBm25Chunks] = useState(0)

  // Store the number of documents stored in PostgreSQL.
  const [documentCount, setDocumentCount] = useState(0)

  // Store whether the statistics are still being loaded.
  const [loading, setLoading] = useState(true)

  // Run this code when the Dashboard component is loaded.
  useEffect(() => {
    // Fetch the latest retrieval statistics from our FastAPI backend.
    getEvaluationStats()
      .then((data) => {
        // Store the vector database chunk count in React state.
        setVectorChunks(data.retrieval.vector_chunks)

        // Store the BM25 chunk count in React state.
        setBm25Chunks(data.retrieval.bm25_chunks)
      })
      .catch((error) => {
        // Display an error if the statistics request fails.
        console.error('Failed to load evaluation stats:', error)
      })

    // Fetch all documents from the FastAPI backend.
    getDocuments()
      .then((documents) => {
        // The number of documents is the length of the returned array.
        setDocumentCount(documents.length)
      })
      .catch((error) => {
        // Display an error if the documents request fails.
        console.error('Failed to load documents:', error)
      })
      .finally(() => {
        // Stop showing the loading state after the request finishes.
        setLoading(false)
      })
  }, [])

  return (
    <section className="dashboard">

      {/* Page heading and short description. */}
      <div className="dashboard-header">
        <h1>Research Dashboard</h1>
        <p>
          Overview of your documents and AI research workspace.
        </p>
      </div>

      {/* Summary cards showing important system information. */}
      <div className="dashboard-cards">

        {/* Number of documents currently stored. */}
        <div className="dashboard-card">
          <h3>Documents</h3>
          <p>
            {loading ? '...' : documentCount}
          </p>
        </div>

        {/* Number of chunks available for vector retrieval. */}
        <div className="dashboard-card">
          <h3>Vector Chunks</h3>
          <p>
            {loading ? '...' : vectorChunks}
          </p>
        </div>

        {/* Number of chunks available for BM25 keyword retrieval. */}
        <div className="dashboard-card">
          <h3>BM25 Chunks</h3>
          <p>
            {loading ? '...' : bm25Chunks}
          </p>
        </div>

      </div>

    </section>
  )
}

// Make the Dashboard component available for other files to import.
export default Dashboard