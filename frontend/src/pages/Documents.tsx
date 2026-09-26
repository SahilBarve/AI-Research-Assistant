
// Import React hooks for storing data and running code when the page loads.
import { useEffect, useState } from 'react'

// Import the API functions used to communicate with our FastAPI backend.
import {
  getDocuments,
  uploadDocument,
} from '../services/api'

// Define the structure of a document returned by the backend.
interface Document {
  id: string
  filename: string
  file_type: string
  file_size: number
  status: string
  chunk_count: number
  created_at: string
}

function Documents() {
  // Store the list of documents returned by the backend.
  const [documents, setDocuments] = useState<Document[]>([])

  // Store whether the documents are still being loaded.
  const [loading, setLoading] = useState(true)

  // Store the file selected by the user.
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  // Store whether a document is currently being uploaded.
  const [uploading, setUploading] = useState(false)

  // Fetch all documents from the backend.
  const loadDocuments = () => {
    // Show the loading state while documents are being fetched.
    setLoading(true)

    getDocuments()
      .then((data) => {
        // Store the returned documents in React state.
        setDocuments(data)
      })
      .catch((error) => {
        // Display an error if the request fails.
        console.error('Failed to load documents:', error)
      })
      .finally(() => {
        // Stop showing the loading state.
        setLoading(false)
      })
  }

  // Run this code when the Documents page is loaded.
  useEffect(() => {
    // Load the current documents from the backend.
    loadDocuments()
  }, [])

  // Handle the file selected from the file picker.
  const handleFileChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    // Get the first selected file.
    const file = event.target.files?.[0] ?? null

    // Store the selected file in React state.
    setSelectedFile(file)
  }

  // Upload the selected file to the backend.
  const handleUpload = async () => {
    // Do nothing if the user hasn't selected a file.
    if (!selectedFile) {
      return
    }

    try {
      // Show the uploading state.
      setUploading(true)

      // Send the selected file to FastAPI.
      await uploadDocument(selectedFile)

      // Clear the selected file after a successful upload.
      setSelectedFile(null)

      // Refresh the document list so the new document appears.
      loadDocuments()
    } catch (error) {
      // Display an error if the upload fails.
      console.error('Failed to upload document:', error)
    } finally {
      // Stop showing the uploading state.
      setUploading(false)
    }
  }

  return (
    <section className="documents">

      {/* Page heading and description. */}
      <div className="documents-header">
        <h1>Documents</h1>
        <p>
          Manage the documents used by your AI research assistant.
        </p>
      </div>

      {/* Document upload section. */}
      <div className="document-upload">

        {/* File picker allows the user to select a document. */}
        <input
          type="file"
          accept=".pdf,.docx,.txt,.md,.markdown"
          onChange={handleFileChange}
        />

        {/* Upload the selected document. */}
        <button
          type="button"
          onClick={handleUpload}
          disabled={!selectedFile || uploading}
        >
          {uploading ? 'Uploading...' : 'Upload Document'}
        </button>

      </div>

      {/* Show a loading message while the documents are being fetched. */}
      {loading && <p>Loading documents...</p>}

      {/* Show a message when no documents exist. */}
      {!loading && documents.length === 0 && (
        <p>No documents found.</p>
      )}

      {/* Display all documents returned by the backend. */}
      {!loading && documents.length > 0 && (
        <div className="documents-list">

          {documents.map((document) => (
            <div
              className="document-card"
              key={document.id}
            >

              {/* Document filename. */}
              <h3>{document.filename}</h3>

              {/* Basic information about the document. */}
              <p>Type: {document.file_type}</p>

              {/* Number of chunks created during indexing. */}
              <p>Chunks: {document.chunk_count}</p>

              {/* Current processing status. */}
              <p>Status: {document.status}</p>

            </div>
          ))}

        </div>
      )}

    </section>
  )
}

// Make the Documents page available to other files.
export default Documents

