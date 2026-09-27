// Import React hooks for managing evaluation data and UI state.
import { useState } from 'react'

// Import the API function used to run the evaluation benchmark.
import { runEvaluation } from '../services/api'


// ============================================================
// TYPES
// ============================================================

// Define the metrics returned by the evaluation service.
interface Metrics {
  precision_at_5: number
  recall_at_5: number
  ndcg_at_5: number

  precision_at_10: number
  recall_at_10: number
  ndcg_at_10: number

  precision_at_20: number
  recall_at_20: number
  ndcg_at_20: number

  mrr: number
}


// Define latency statistics.
interface Latency {
  average_ms: number
  min_ms: number
  max_ms: number
}


// Define the structure of each reranker experiment.
interface RerankerExperiment {
  candidate_limit: number
  rerank_limit: number
  metrics: Metrics
  latency: Latency
}


// Define the complete evaluation response.
interface EvaluationResult {
  dataset_size: number

  configuration: {
    retrieval_limit: number
    rrf_limit: number
    rerank_limit: number
    reranker_candidate_limits: number[]
    metric_k_values: number[]
  }

  metrics: {
    dense: Metrics
    bm25: Metrics
    rrf: Metrics
  }

  reranker_experiment: {
    candidates_10: RerankerExperiment
    candidates_15: RerankerExperiment
    candidates_20: RerankerExperiment
  }

  latency: {
    dense: Latency
    bm25: Latency
    rrf: Latency
    pipeline: Latency
  }
}


// ============================================================
// HELPERS
// ============================================================

// Convert a decimal metric such as 0.3229 into 32.29%.
const formatPercentage = (value: number) => {
  return `${(value * 100).toFixed(2)}%`
}


// Format latency values in milliseconds.
const formatLatency = (value: number) => {
  return `${value.toFixed(2)} ms`
}


// ============================================================
// COMPONENT
// ============================================================

function Evaluation() {

  // Store the evaluation result returned by the backend.
  const [evaluation, setEvaluation] =
    useState<EvaluationResult | null>(null)

  // Track whether the benchmark is currently running.
  const [loading, setLoading] = useState(false)

  // Store any evaluation error.
  const [error, setError] = useState('')


  // ----------------------------------------------------------
  // RUN EVALUATION
  // ----------------------------------------------------------

  const handleRunEvaluation = async () => {

    try {

      setLoading(true)
      setError('')

      // Run the actual retrieval benchmark.
      const result = await runEvaluation()

      // Store the returned evaluation data.
      setEvaluation(result)

    } catch (error) {

      console.error(
        'Failed to run evaluation:',
        error
      )

      setError(
        'Failed to run the evaluation benchmark.'
      )

    } finally {

      setLoading(false)
    }
  }


  // ==========================================================
  // UI
  // ==========================================================

  return (
    <section className="evaluation-page">

      {/* Page heading. */}
      <div className="evaluation-header">

        <div>

          <h1>
            Evaluation
          </h1>

          <p>
            Measure the performance of the retrieval pipeline
            using the labeled evaluation dataset.
          </p>

        </div>


        {/* Run the benchmark. */}
        <button
          type="button"
          className="evaluation-run-button"
          onClick={handleRunEvaluation}
          disabled={loading}
        >
          {loading
            ? 'Running Evaluation...'
            : 'Run Evaluation'}
        </button>

      </div>


      {/* Explanation card shown before the first evaluation. */}
      {!evaluation && !loading && !error && (

        <div className="evaluation-intro">

          <div className="evaluation-intro-icon">
            📊
          </div>

          <h2>
            Retrieval Evaluation
          </h2>

          <p>
            Run the benchmark to compare dense retrieval,
            BM25, hybrid RRF retrieval, and Cross-Encoder
            reranking.
          </p>

          <p className="evaluation-intro-note">
            The benchmark uses the project's manually
            labeled evaluation dataset.
          </p>

        </div>
      )}


      {/* Loading state. */}
      {loading && (

        <div className="evaluation-loading">

          <div className="evaluation-spinner">
            ⟳
          </div>

          <h2>
            Running benchmark...
          </h2>

          <p>
            Evaluating dense retrieval, BM25, RRF, and
            Cross-Encoder reranking.
          </p>

        </div>
      )}


      {/* Error message. */}
      {error && (

        <div className="evaluation-error">
          {error}
        </div>
      )}


      {/* Evaluation results. */}
      {evaluation && !loading && (

        <div className="evaluation-results">

          {/* ==================================================
              OVERVIEW
              ================================================== */}

          <div className="evaluation-overview">

            <div className="evaluation-overview-card">

              <span>
                Dataset Size
              </span>

              <strong>
                {evaluation.dataset_size}
              </strong>

              <small>
                labeled queries
              </small>

            </div>


            <div className="evaluation-overview-card">

              <span>
                RRF Candidates
              </span>

              <strong>
                {evaluation.configuration.rrf_limit}
              </strong>

              <small>
                candidates
              </small>

            </div>


            <div className="evaluation-overview-card">

              <span>
                Final Results
              </span>

              <strong>
                {evaluation.configuration.rerank_limit}
              </strong>

              <small>
                returned after reranking
              </small>

            </div>


            <div className="evaluation-overview-card">

              <span>
                Reranker Tests
              </span>

              <strong>
                {evaluation.configuration
                  .reranker_candidate_limits.length}
              </strong>

              <small>
                candidate depths
              </small>

            </div>

          </div>


          {/* ==================================================
              BASELINE COMPARISON
              ================================================== */}

          <div className="evaluation-section">

            <div className="evaluation-section-header">

              <div>

                <h2>
                  Retrieval Comparison
                </h2>

                <p>
                  Performance of the main retrieval strategies.
                </p>

              </div>

            </div>


            <div className="evaluation-table-wrapper">

              <table className="evaluation-table">

                <thead>

                  <tr>

                    <th>
                      Method
                    </th>

                    <th>
                      Precision@5
                    </th>

                    <th>
                      Recall@5
                    </th>

                    <th>
                      NDCG@5
                    </th>

                    <th>
                      MRR
                    </th>

                  </tr>

                </thead>


                <tbody>

                  <tr>

                    <td>
                      <strong>
                        Dense
                      </strong>
                    </td>

                    <td>
                      {formatPercentage(
                        evaluation.metrics.dense.precision_at_5
                      )}
                    </td>

                    <td>
                      {formatPercentage(
                        evaluation.metrics.dense.recall_at_5
                      )}
                    </td>

                    <td>
                      {formatPercentage(
                        evaluation.metrics.dense.ndcg_at_5
                      )}
                    </td>

                    <td>
                      {formatPercentage(
                        evaluation.metrics.dense.mrr
                      )}
                    </td>

                  </tr>


                  <tr>

                    <td>
                      <strong>
                        BM25
                      </strong>
                    </td>

                    <td>
                      {formatPercentage(
                        evaluation.metrics.bm25.precision_at_5
                      )}
                    </td>

                    <td>
                      {formatPercentage(
                        evaluation.metrics.bm25.recall_at_5
                      )}
                    </td>

                    <td>
                      {formatPercentage(
                        evaluation.metrics.bm25.ndcg_at_5
                      )}
                    </td>

                    <td>
                      {formatPercentage(
                        evaluation.metrics.bm25.mrr
                      )}
                    </td>

                  </tr>


                  <tr>

                    <td>
                      <strong>
                        Hybrid RRF
                      </strong>
                    </td>

                    <td>
                      {formatPercentage(
                        evaluation.metrics.rrf.precision_at_5
                      )}
                    </td>

                    <td>
                      {formatPercentage(
                        evaluation.metrics.rrf.recall_at_5
                      )}
                    </td>

                    <td>
                      {formatPercentage(
                        evaluation.metrics.rrf.ndcg_at_5
                      )}
                    </td>

                    <td>
                      {formatPercentage(
                        evaluation.metrics.rrf.mrr
                      )}
                    </td>

                  </tr>

                </tbody>

              </table>

            </div>

          </div>


          {/* ==================================================
              RERANKER EXPERIMENT
              ================================================== */}

          <div className="evaluation-section">

            <div className="evaluation-section-header">

              <div>

                <h2>
                  Cross-Encoder Reranker
                </h2>

                <p>
                  Effect of changing the number of RRF
                  candidates passed to the reranker.
                </p>

              </div>

            </div>


            <div className="evaluation-table-wrapper">

              <table className="evaluation-table">

                <thead>

                  <tr>

                    <th>
                      Candidates
                    </th>

                    <th>
                      Precision@5
                    </th>

                    <th>
                      Recall@5
                    </th>

                    <th>
                      NDCG@5
                    </th>

                    <th>
                      MRR
                    </th>

                    <th>
                      Avg Latency
                    </th>

                  </tr>

                </thead>


                <tbody>

                  {[10, 15, 20].map(
                    (candidateCount) => {

                      const experiment =
                        evaluation.reranker_experiment[
                          `candidates_${candidateCount}` as
                            | 'candidates_10'
                            | 'candidates_15'
                            | 'candidates_20'
                        ]

                      return (

                        <tr key={candidateCount}>

                          <td>
                            <strong>
                              {candidateCount}
                            </strong>
                          </td>

                          <td>
                            {formatPercentage(
                              experiment.metrics
                                .precision_at_5
                            )}
                          </td>

                          <td>
                            {formatPercentage(
                              experiment.metrics
                                .recall_at_5
                            )}
                          </td>

                          <td>
                            {formatPercentage(
                              experiment.metrics
                                .ndcg_at_5
                            )}
                          </td>

                          <td>
                            {formatPercentage(
                              experiment.metrics.mrr
                            )}
                          </td>

                          <td>
                            {formatLatency(
                              experiment.latency.average_ms
                            )}
                          </td>

                        </tr>
                      )
                    }
                  )}

                </tbody>

              </table>

            </div>

          </div>


          {/* ==================================================
              LATENCY
              ================================================== */}

          <div className="evaluation-section">

            <div className="evaluation-section-header">

              <div>

                <h2>
                  Retrieval Latency
                </h2>

                <p>
                  Measured latency during the evaluation run.
                </p>

              </div>

            </div>


            <div className="evaluation-latency-grid">

              <div className="evaluation-latency-card">

                <span>
                  Dense
                </span>

                <strong>
                  {formatLatency(
                    evaluation.latency.dense.average_ms
                  )}
                </strong>

                <small>
                  average
                </small>

              </div>


              <div className="evaluation-latency-card">

                <span>
                  BM25
                </span>

                <strong>
                  {formatLatency(
                    evaluation.latency.bm25.average_ms
                  )}
                </strong>

                <small>
                  average
                </small>

              </div>


              <div className="evaluation-latency-card">

                <span>
                  Hybrid RRF
                </span>

                <strong>
                  {formatLatency(
                    evaluation.latency.rrf.average_ms
                  )}
                </strong>

                <small>
                  average
                </small>

              </div>


              <div className="evaluation-latency-card">

                <span>
                  Full Pipeline
                </span>

                <strong>
                  {formatLatency(
                    evaluation.latency.pipeline.average_ms
                  )}
                </strong>

                <small>
                  average
                </small>

              </div>

            </div>

          </div>


          {/* ==================================================
              CONFIGURATION
              ================================================== */}

          <div className="evaluation-section">

            <div className="evaluation-section-header">

              <div>

                <h2>
                  Evaluation Configuration
                </h2>

                <p>
                  Parameters used during this benchmark run.
                </p>

              </div>

            </div>


            <div className="evaluation-config-grid">

              <div>
                <span>
                  Retrieval Limit
                </span>

                <strong>
                  {evaluation.configuration.retrieval_limit}
                </strong>
              </div>


              <div>
                <span>
                  RRF Limit
                </span>

                <strong>
                  {evaluation.configuration.rrf_limit}
                </strong>
              </div>


              <div>
                <span>
                  Rerank Limit
                </span>

                <strong>
                  {evaluation.configuration.rerank_limit}
                </strong>
              </div>


              <div>
                <span>
                  Metric K Values
                </span>

                <strong>
                  {evaluation.configuration.metric_k_values.join(
                    ', '
                  )}
                </strong>
              </div>

            </div>

          </div>

        </div>
      )}

    </section>
  )
}

export default Evaluation