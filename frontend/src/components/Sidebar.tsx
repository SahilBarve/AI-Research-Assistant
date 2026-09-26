
// Import Link from React Router for navigation between application pages.
import { Link } from 'react-router-dom'

function Sidebar() {
  // This component contains the application's main navigation.
  return (
    <aside className="sidebar">

      {/* Application name displayed at the top of the sidebar. */}
      <h2>AI Research Assistant</h2>

      {/* Navigation links for the different sections of our application. */}
      <nav>

        {/* Navigate to the Dashboard page. */}
        <Link to="/">Dashboard</Link>

        {/* Navigate to the Documents page. */}
        <Link to="/documents">Documents</Link>

        {/* These pages will be added as we build them. */}
        <Link to="/research">Research</Link>
        <Link to="/search">Search</Link>
        <Link to="/conversations">Conversations</Link>
        <Link to="/evaluation">Evaluation</Link>

      </nav>

    </aside>
  )
}

export default Sidebar

