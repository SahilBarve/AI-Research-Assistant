
// Import StrictMode from React for additional development checks.
import { StrictMode } from 'react'

// Import the function used to render our React application.
import { createRoot } from 'react-dom/client'

// Import BrowserRouter so our application can use React Router.
import { BrowserRouter } from 'react-router-dom'

// Import the main App component.
import App from './App.tsx'

// Import the application's global CSS.
import './index.css'

// Find the HTML element where React should render our application.
createRoot(document.getElementById('root')!).render(

  // StrictMode helps identify potential problems during development.
  <StrictMode>

    {/* BrowserRouter enables routing throughout our React application. */}
    <BrowserRouter>

      {/* Render the main application inside the router. */}
      <App />

    </BrowserRouter>

  </StrictMode>,
)

