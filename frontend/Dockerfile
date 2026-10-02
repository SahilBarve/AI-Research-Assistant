# Build the React application
FROM node:22-alpine AS builder

# Set the working directory inside the container
WORKDIR /app

# Copy package files first so Docker can cache dependencies
COPY package*.json ./

# Install frontend dependencies
RUN npm ci

# Copy the frontend source code
COPY . .

# Create the production React build
RUN npm run build


# Serve the production build using Nginx
FROM nginx:alpine

# Copy the React build into Nginx's web directory
COPY --from=builder /app/dist /usr/share/nginx/html

# Replace the default Nginx configuration
COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf

# Expose the HTTP port
EXPOSE 80

# Start Nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]