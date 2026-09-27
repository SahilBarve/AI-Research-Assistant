// Import React hooks for managing the search state.
import { useState } from 'react'

// Import the API function used to search the backend.
import { searchDocuments } from '../services/api'


// ============================================================
// TYPES
// ============================================================

// Define the structure of a search result returned by FastAPI.
interface SearchResult {
  chunk_id: string
  text: string
  source: string | null
  page_number: number | null
  score: number
}


// Define the structure of the complete search response.
interface SearchResponse {
  query: string
  results: SearchResult[]
  total_results: number
}


// ============================================================
// COMPONENT
// ============================================================

function Search() {

  // Store the text entered into the search box.
  const [query, setQuery] = useState('')

  // Store the number of results requested.
  const [limit, setLimit] = useState(5)

  // Store the search results returned by the backend.
  const [searchResponse, setSearchResponse] =
    useState<SearchResponse | null>(null)

  // Track whether a search request is running.
  const [loading, setLoading] = useState(false)

  // Store an error message when the request fails.
  const [error, setError] = useState('')


  // ==========================================================
  // SEARCH
  // ==========================================================

  const handleSearch = async () => {

    // Do not send empty search queries.
    if (!query.trim() || loading) {
      return
    }

    try {

      // Show the loading state.
      setLoading(true)

      // Clear any previous error.
      setError('')

      // Send the query to the FastAPI search endpoint.
      const response = await searchDocuments(
        query.trim(),
        limit
      )

      // Store the returned search results.
      setSearchResponse(response)

    } catch (error) {

      console.error(
        'Failed to search documents:',
        error
      )

      setError(
        'Something went wrong while searching the documents.'
      )

      setSearchResponse(null)

    } finally {

      // Stop the loading state.
      setLoading(false)
    }
  }


  // ==========================================================
  // KEYBOARD HANDLING
  // ==========================================================

  // Allow Enter to execute the search.
  const handleKeyDown = (
    event: React.KeyboardEvent<HTMLInputElement>
  ) => {

    if (event.key === 'Enter') {
      handleSearch()
    }
  }


  // ==========================================================
  // UI
  // ==========================================================

  return (
    <section className="search-page">

      {/* Page heading. */}
      <div className="search-header">

        <h1>Search</h1>

        <p>
          Search your indexed documents using hybrid retrieval
          and semantic reranking.
        </p>

      </div>


      {/* Search controls. */}
      <div className="search-controls">

        <input
          type="text"
          value={query}
          onChange={(event) =>
            setQuery(event.target.value)
          }
          onKeyDown={handleKeyDown}
          placeholder="Search your documents..."
          maxLength={1000}
        />


        {/* Number of results to retrieve. */}
        <select
          value={limit}
          onChange={(event) =>
            setLimit(Number(event.target.value))
          }
        >
          <option value={5}>5 results</option>
          <option value={10}>10 results</option>
          <option value={20}>20 results</option>
          <option value={50}>50 results</option>
        </select>


        {/* Execute the search. */}
        <button
          type="button"
          onClick={handleSearch}
          disabled={!query.trim() || loading}
          className="search-button"
        >
          {loading ? 'Searching...' : 'Search'}
        </button>

      </div>


      {/* Error message. */}
      {error && (
        <div className="search-error">
          {error}
        </div>
      )}


      {/* Search results. */}
      {searchResponse && (

        <div className="search-results">

          {/* Result summary. */}
          <div className="search-results-header">

            <div>
              <h2>Search Results</h2>

              <p>
                {searchResponse.total_results}{' '}
                result
                {searchResponse.total_results !== 1
                  ? 's'
                  : ''}{' '}
                for "
                {searchResponse.query}"
              </p>
            </div>

          </div>


          {/* Show an empty state when nothing was found. */}
          {searchResponse.results.length === 0 && (

            <div className="search-empty">
              <h3>No results found</h3>

              <p>
                Try using different keywords or a more
                specific question.
              </p>
            </div>
          )}


          {/* Render every retrieved result. */}
          <div className="search-result-list">

            {searchResponse.results.map(
              (result, index) => (

              <article
                className="search-result-card"
                key={`${result.chunk_id}-${index}`}
              >

                {/* Result header. */}
                <div className="search-result-top">

                  <div className="result-number">
                    #{index + 1}
                  </div>

                  <div className="result-score">
                    Score:{' '}
                    {result.score.toFixed(4)}
                  </div>

                </div>


                {/* Source metadata. */}
                <div className="result-metadata">

                  {result.source && (
                    <span>
                      📄 {result.source}
                    </span>
                  )}

                  {result.page_number !== null && (
                    <span>
                      Page {result.page_number}
                    </span>
                  )}

                  <span>
                    Chunk {result.chunk_id}
                  </span>

                </div>


                {/* Retrieved chunk text. */}
                <div className="result-text">
                  {result.text}
                </div>

              </article>
            ))}
          </div>

        </div>
      )}

    </section>
  )
}

export default Search