// Import React hooks for managing the chat state.
import { useEffect, useState } from 'react'

// Import the API functions used by the research workspace.
import {
  sendChatMessage,
  getConversation,
  deleteConversation,
} from '../services/api'

// Generate a session ID that works across different browsers.
const createSessionId = (): string => {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID()
  }

  // Fallback for browsers/environments without crypto.randomUUID.
  return `${Date.now()}-${Math.random().toString(36).slice(2, 11)}`
}
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


// Define the structure of a message displayed in the conversation.
interface Message {
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
}


// ============================================================
// COMPONENT
// ============================================================

function Research() {

  // Store all messages in the current conversation.
  const [messages, setMessages] = useState<Message[]>([])

  // Store the question currently being typed.
  const [question, setQuestion] = useState('')

  // Track whether the assistant is generating an answer.
  const [loading, setLoading] = useState(false)

  // Store any error message shown to the user.
  const [error, setError] = useState('')


  // ----------------------------------------------------------
  // SESSION ID
  // ----------------------------------------------------------

  // Get the existing session ID or create a new one.
  const getSessionId = () => {

    const storedSessionId = localStorage.getItem(
      'research_session_id'
    )

    if (storedSessionId) {
      return storedSessionId
    }

    // Generate a unique ID for a new conversation.
    const newSessionId = createSessionId()

    localStorage.setItem(
      'research_session_id',
      newSessionId
    )

    return newSessionId
  }


  // ----------------------------------------------------------
  // LOAD CONVERSATION
  // ----------------------------------------------------------

  // Load the previous conversation when the page opens.
  useEffect(() => {

    const sessionId = getSessionId()

    getConversation(sessionId)
      .then((data) => {

        // Convert backend messages into our frontend format.
        const history: Message[] = data.messages.map(
          (message: any) => ({
            role: message.role,
            content: message.content,
            citations: message.citations ?? [],
          })
        )

        setMessages(history)
      })
      .catch((error) => {
        console.error(
          'Failed to load conversation:',
          error
        )
      })

  }, [])


  // ----------------------------------------------------------
  // SEND MESSAGE
  // ----------------------------------------------------------

  const handleSend = async () => {

    // Do not send empty questions.
    if (!question.trim() || loading) {
      return
    }

    const userQuestion = question.trim()

    // Clear previous errors.
    setError('')

    // Display the user's message immediately.
    setMessages((previousMessages) => [
      ...previousMessages,
      {
        role: 'user',
        content: userQuestion,
      },
    ])

    // Clear the input box.
    setQuestion('')

    try {

      setLoading(true)

      const sessionId = getSessionId()

      // Send the question to the FastAPI RAG pipeline.
      const response = await sendChatMessage(
        userQuestion,
        sessionId
      )

      // Add the assistant response to the conversation.
      setMessages((previousMessages) => [
        ...previousMessages,
        {
          role: 'assistant',
          content: response.answer,
          citations: response.citations ?? [],
        },
      ])

    } catch (error) {

      console.error(
        'Failed to send message:',
        error
      )

      setError(
        'Something went wrong while generating the answer.'
      )

    } finally {

      setLoading(false)
    }
  }


  // ----------------------------------------------------------
  // KEYBOARD HANDLING
  // ----------------------------------------------------------

  // Allow the user to press Enter to send the question.
  const handleKeyDown = (
    event: React.KeyboardEvent<HTMLTextAreaElement>
  ) => {

    if (
      event.key === 'Enter' &&
      !event.shiftKey
    ) {

      event.preventDefault()

      handleSend()
    }
  }


  // ----------------------------------------------------------
  // NEW CONVERSATION
  // ----------------------------------------------------------

  const handleNewConversation = async () => {

    try {

      const sessionId = getSessionId()

      // Delete the conversation from backend memory.
      await deleteConversation(sessionId)

    } catch (error) {

      console.error(
        'Failed to delete conversation:',
        error
      )

    }

    // Create a completely new session.
    const newSessionId = createSessionId()

    localStorage.setItem(
      'research_session_id',
      newSessionId
    )

    // Clear the current messages.
    setMessages([])

    setError('')
  }


  // ==========================================================
  // UI
  // ==========================================================

  return (
    <section className="research">

      {/* Page heading. */}
      <div className="research-header">

        <div>
          <h1>Research Workspace</h1>

          <p>
            Ask questions about your indexed documents.
          </p>
        </div>

        {/* Start a fresh conversation. */}
        <button
          type="button"
          className="new-conversation-button"
          onClick={handleNewConversation}
        >
          New Conversation
        </button>

      </div>


      {/* Main chat container. */}
      <div className="research-container">

        {/* Conversation messages. */}
        <div className="conversation">

          {messages.length === 0 && !loading && (

            <div className="empty-research">

              <h2>Start your research</h2>

              <p>
                Ask a question about the documents
                you've uploaded.
              </p>

              <div className="research-examples">

                <span>
                  What are the main challenges discussed?
                </span>

                <span>
                  Summarize the key findings.
                </span>

                <span>
                  What recommendations are provided?
                </span>

              </div>

            </div>
          )}


          {/* Render every conversation message. */}
          {messages.map((message, index) => (

            <div
              className={`message ${
                message.role === 'user'
                  ? 'user-message'
                  : 'assistant-message'
              }`}
              key={index}
            >

              <div className="message-role">
                {message.role === 'user'
                  ? 'You'
                  : 'AI Research Assistant'}
              </div>

              <div className="message-content">
                {message.content}
              </div>


              {/* Display citations returned by the backend. */}
              {message.role === 'assistant' &&
                message.citations &&
                message.citations.length > 0 && (

                <div className="citations">

                  <h3>Sources</h3>

                  {message.citations.map(
                    (citation) => (

                    <div
                      className="citation-card"
                      key={citation.id}
                    >

                      <div className="citation-header">

                        <strong>
                          [{citation.id}]
                        </strong>

                        <span>
                          {citation.source}
                        </span>

                        <span>
                          Page {citation.page_number}
                        </span>

                        <span>
                          Chunk {citation.chunk_id}
                        </span>

                      </div>

                      <p>
                        {citation.text}
                      </p>

                    </div>
                  ))}

                </div>
              )}

            </div>
          ))}


          {/* Loading indicator while Ollama generates. */}
          {loading && (

            <div className="message assistant-message">

              <div className="message-role">
                AI Research Assistant
              </div>

              <div className="loading-message">
                Researching your documents...
              </div>

            </div>
          )}

        </div>


        {/* Error message. */}
        {error && (
          <div className="chat-error">
            {error}
          </div>
        )}


        {/* Question input area. */}
        <div className="chat-input-container">

          <textarea
            value={question}
            onChange={(event) =>
              setQuestion(event.target.value)
            }
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents..."
            maxLength={1000}
            rows={2}
            disabled={loading}
          />

          <button
            type="button"
            onClick={handleSend}
            disabled={!question.trim() || loading}
            className="send-button"
          >
            {loading ? 'Researching...' : 'Send'}
          </button>

        </div>

      </div>

    </section>
  )
}

export default Research