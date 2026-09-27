// Import React hooks for loading and managing conversation data.
import { useEffect, useState } from 'react'

// Import API functions used by the conversations page.
import {
  getConversation,
  deleteConversation,
} from '../services/api'


// ============================================================
// TYPES
// ============================================================

// Define the structure of a citation returned by the backend.
interface Citation {
  id: number
  source: string
  page_number: number
  chunk_id: number
  text: string
}

// Define the structure of a conversation message.
interface Message {
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
}


// ============================================================
// COMPONENT
// ============================================================

function Conversations() {

  // Store the messages belonging to the current conversation.
  const [messages, setMessages] = useState<Message[]>([])

  // Store the current browser session ID.
  const [sessionId, setSessionId] = useState('')

  // Track whether the conversation is being loaded.
  const [loading, setLoading] = useState(true)

  // Store an error message if something goes wrong.
  const [error, setError] = useState('')


  // ----------------------------------------------------------
  // LOAD CURRENT CONVERSATION
  // ----------------------------------------------------------

  useEffect(() => {

    // Get the session ID created by the Research page.
    const storedSessionId = localStorage.getItem(
      'research_session_id'
    )

    // If there is no session yet, there is nothing to load.
    if (!storedSessionId) {
      setLoading(false)
      return
    }

    // Store the session ID in React state.
    setSessionId(storedSessionId)

    // Load the conversation from the FastAPI backend.
    getConversation(storedSessionId)
      .then((data) => {

        // Convert backend messages into our frontend format.
        const history: Message[] = data.messages.map(
          (message: any) => ({
            role: message.role,
            content: message.content,
            citations: message.citations ?? [],
          })
        )

        // Store the conversation messages.
        setMessages(history)
      })
      .catch((error) => {

        console.error(
          'Failed to load conversation:',
          error
        )

        setError(
          'Failed to load the current conversation.'
        )
      })
      .finally(() => {

        // Stop showing the loading state.
        setLoading(false)
      })

  }, [])


  // ----------------------------------------------------------
  // DELETE CONVERSATION
  // ----------------------------------------------------------

  const handleDeleteConversation = async () => {

    // There is nothing to delete if there is no session.
    if (!sessionId) {
      return
    }

    // Ask the user for confirmation before deleting.
    const confirmed = window.confirm(
      'Are you sure you want to delete this conversation?'
    )

    if (!confirmed) {
      return
    }

    try {

      setLoading(true)
      setError('')

      // Delete the conversation from backend memory.
      await deleteConversation(sessionId)

      // Remove the session ID from browser storage.
      localStorage.removeItem(
        'research_session_id'
      )

      // Clear the displayed messages.
      setMessages([])

      // Clear the session ID from React state.
      setSessionId('')

    } catch (error) {

      console.error(
        'Failed to delete conversation:',
        error
      )

      setError(
        'Failed to delete the conversation.'
      )

    } finally {

      setLoading(false)
    }
  }


  // ----------------------------------------------------------
  // OPEN RESEARCH WORKSPACE
  // ----------------------------------------------------------

  const handleOpenResearch = () => {

    // Navigate back to the Research workspace.
    window.location.href = '/research'
  }


  // ==========================================================
  // UI
  // ==========================================================

  return (
    <section className="conversations-page">

      {/* Page heading. */}
      <div className="conversations-header">

        <div>

          <h1>
            Conversations
          </h1>

          <p>
            View and manage your current research conversation.
          </p>

        </div>

        {/* Open the Research workspace. */}
        <button
          type="button"
          className="conversation-primary-button"
          onClick={handleOpenResearch}
        >
          Open Research
        </button>

      </div>


      {/* Error message. */}
      {error && (
        <div className="conversation-error">
          {error}
        </div>
      )}


      {/* Loading state. */}
      {loading && (
        <div className="conversation-empty">

          <h2>
            Loading conversation...
          </h2>

          <p>
            Retrieving your conversation history.
          </p>

        </div>
      )}


      {/* No conversation exists. */}
      {!loading && !error && !sessionId && (

        <div className="conversation-empty">

          <div className="conversation-empty-icon">
            💬
          </div>

          <h2>
            No active conversation
          </h2>

          <p>
            Start a research conversation to see your
            messages here.
          </p>

          <button
            type="button"
            className="conversation-primary-button"
            onClick={handleOpenResearch}
          >
            Start Research
          </button>

        </div>
      )}


      {/* Display the active conversation. */}
      {!loading && sessionId && (

        <div className="conversation-card">

          {/* Conversation header. */}
          <div className="conversation-card-header">

            <div>

              <span className="conversation-label">
                ACTIVE CONVERSATION
              </span>

              <h2>
                Research Conversation
              </h2>

              <p className="conversation-session-id">
                Session: {sessionId}
              </p>

            </div>

            {/* Delete the current conversation. */}
            <button
              type="button"
              className="conversation-delete-button"
              onClick={handleDeleteConversation}
            >
              Delete Conversation
            </button>

          </div>


          {/* Conversation statistics. */}
          <div className="conversation-stats">

            <div className="conversation-stat">

              <span>
                Messages
              </span>

              <strong>
                {messages.length}
              </strong>

            </div>


            <div className="conversation-stat">

              <span>
                User Questions
              </span>

              <strong>
                {
                  messages.filter(
                    (message) =>
                      message.role === 'user'
                  ).length
                }
              </strong>

            </div>


            <div className="conversation-stat">

              <span>
                AI Responses
              </span>

              <strong>
                {
                  messages.filter(
                    (message) =>
                      message.role === 'assistant'
                  ).length
                }
              </strong>

            </div>

          </div>


          {/* Conversation history. */}
          <div className="conversation-history">

            <div className="conversation-history-header">

              <h3>
                Conversation History
              </h3>

            </div>


            {/* Show this when the conversation has no messages. */}
            {messages.length === 0 && (

              <div className="conversation-no-messages">

                <p>
                  This conversation does not contain
                  any messages yet.
                </p>

                <button
                  type="button"
                  className="conversation-primary-button"
                  onClick={handleOpenResearch}
                >
                  Start Asking Questions
                </button>

              </div>
            )}


            {/* Render every conversation message. */}
            {messages.map((message, index) => (

              <div
                className={`conversation-message ${
                  message.role === 'user'
                    ? 'conversation-user-message'
                    : 'conversation-assistant-message'
                }`}
                key={index}
              >

                {/* Display who sent the message. */}
                <div className="conversation-message-role">

                  {message.role === 'user'
                    ? 'You'
                    : 'AI Research Assistant'}

                </div>


                {/* Display the message content. */}
                <div className="conversation-message-content">
                  {message.content}
                </div>


                {/* Display citations belonging to an AI response. */}
                {message.role === 'assistant' &&
                  message.citations &&
                  message.citations.length > 0 && (

                    <div className="conversation-citations">

                      <strong>
                        Sources: {message.citations.length}
                      </strong>

                      <div>

                        {/* Render each citation. */}
                        {message.citations.map((citation) => (

                          <span
                            className="conversation-citation"
                            key={citation.id}
                          >

                            [{citation.id}] {citation.source}
                            {' '}· Page {citation.page_number}

                          </span>

                        ))}

                      </div>

                    </div>
                  )}

              </div>

            ))}

          </div>

        </div>
      )}

    </section>
  )
}

export default Conversations