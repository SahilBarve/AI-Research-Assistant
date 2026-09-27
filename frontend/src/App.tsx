
// Import the routing components used to define application pages.
import { Routes, Route } from 'react-router-dom'

// Import the Sidebar component.
import Sidebar from './components/Sidebar'

// Import the Dashboard page.
import Dashboard from './components/Dashboard'

// Import the Documents page.
import Documents from './pages/Documents'

// Import the application's CSS styles.
import './App.css'

// Import the Research page.
import Research from './pages/Research'

// Import the Search page.
import Search from './pages/Search'

// Import the Conversations page.
import Conversations from './pages/Conversations'

// Import the Evaluation page.
import Evaluation from './pages/Evaluation'

function App() {
  return (
    // Main container for the entire application.
    <div className="app">

      {/* Navigation sidebar. */}
      <Sidebar />

      {/* Main content area. */}
      <main className="main-content">

        {/* Define which page should be displayed for each URL. */}
        <Routes>

          {/* Dashboard is displayed at the root URL. */}
          <Route path="/" element={<Dashboard />} />

          {/* Documents page is displayed at /documents. */}
          <Route path="/documents" element={<Documents />} />
          {/* Research workspace is displayed at /research. */}
          <Route path="/research" element={<Research />} />

          {/* Search page is displayed at /search. */}
            <Route
              path="/search"
              element={<Search />}
            />

            {/* Conversations page is displayed at /conversations. */}
            <Route
              path="/conversations"
              element={<Conversations />}
            />

            {/* Evaluation page is displayed at /evaluation. */}
              <Route
                path="/evaluation"
                element={<Evaluation />}
              />
        </Routes>

      </main>

    </div>
  )
}

export default App

