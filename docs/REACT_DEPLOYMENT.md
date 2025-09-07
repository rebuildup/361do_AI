# React Frontend Deployment Guide

This document describes the deployment setup for the React frontend integrated with the existing FastAPI backend.

## Overview

The deployment integrates a React frontend with the existing FastAPI backend using:

- **Multi-stage Docker build** for React frontend and Python backend
- **Nginx** as reverse proxy and static file server
- **Supervisor** for process management
- **Docker Compose** for orchestration

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React App     │    │     Nginx       │    │   FastAPI       │
│   (Frontend)    │───▶│  (Reverse Proxy)│───▶│   (Backend)     │
│   Port: 3000    │    │   Port: 80      │    │   Port: 8000    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## File Structure

```
project/
├── frontend/                    # React frontend
│   ├── src/                    # Source code
│   ├── dist/                   # Build output (generated)
│   ├── package.json            # Dependencies
│   └── vite.config.ts          # Build configuration
├── docker/
│   ├── nginx/
│   │   └── integrated.conf     # Nginx configuration
│   ├── supervisor/
│   │   └── supervisord.conf    # Process management
│   └── entrypoint-integrated.sh # Startup script
├── Dockerfile.integrated       # Multi-stage Docker build
├── docker-compose.yml          # Orchestration
└── src/advanced_agent/
    └── interfaces/
        └── fastapi_app.py      # FastAPI with static serving
```

## Build Process

### 1. Frontend Build

The React frontend is built using Vite:

```bash
cd frontend
npm install
npm run build
```

This generates optimized static files in `frontend/dist/`.

### 2. Docker Multi-Stage Build

The `Dockerfile.integrated` uses a multi-stage build:

1. **Stage 1**: Build React frontend with Node.js
2. **Stage 2**: Setup Python backend dependencies
3. **Stage 3**: Production runtime with both frontend and backend

### 3. Static File Integration

The built React app is:

1. Copied to `/app/static` in the Docker container
2. Served by Nginx for static assets
3. Proxied through FastAPI for API routes

## Configuration

### Nginx Configuration

Located in `docker/nginx/integrated.conf`:

- **Static Files**: Serves React build from `/app/static`
- **API Proxy**: Routes `/api/*` and `/v1/*` to FastAPI
- **React Router**: Fallback to `index.html` for SPA routing
- **WebSocket**: Support for streaming responses

### FastAPI Integration

The FastAPI app (`src/advanced_agent/interfaces/fastapi_app.py`) includes:

```python
# Static file serving for React frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
```

### Supervisor Configuration

Manages multiple processes:

- **FastAPI**: Backend API server
- **Nginx**: Web server and reverse proxy
- **Health Check**: Service monitoring

## Deployment

### Development Mode

For development with hot reload:

```bash
# Start backend only
docker-compose up ai-agent

# Start frontend dev server separately
cd frontend
npm run dev
```

### Production Mode

For integrated production deployment:

```bash
# Build and start all services
docker-compose up --build

# Or build specific service
docker-compose build ai-agent
docker-compose up ai-agent
```

### Environment Variables

Key environment variables in `docker-compose.yml`:

```yaml
environment:
  - FRONTEND_ENABLED=true # Enable React frontend
  - NGINX_ENABLED=true # Enable Nginx
  - LOG_LEVEL=INFO # Logging level
  - ENVIRONMENT=production # Environment mode
```

## Testing

### Frontend Build Test

```bash
cd frontend
npm run build
npm run test:run
```

### Integration Test

```bash
./scripts/test-integration.sh
```

### Docker Build Test

```bash
docker build -f Dockerfile.integrated -t ai-agent-test .
docker run --rm -p 80:80 ai-agent-test
```

## Troubleshooting

### Common Issues

1. **Build Failures**

   - Check Node.js version (requires 18+)
   - Verify all dependencies are installed
   - Check TypeScript compilation errors

2. **Static Files Not Loading**

   - Verify build output in `frontend/dist/`
   - Check Nginx configuration
   - Ensure FastAPI static mounting

3. **API Routes Not Working**

   - Check Nginx proxy configuration
   - Verify FastAPI is running on port 8000
   - Check CORS settings

4. **React Router Issues**
   - Ensure Nginx fallback configuration
   - Check `try_files` directive
   - Verify `@fallback` location block

### Debugging

1. **Check Container Logs**

   ```bash
   docker-compose logs ai-agent
   docker-compose logs -f ai-agent  # Follow logs
   ```

2. **Access Container**

   ```bash
   docker-compose exec ai-agent bash
   ```

3. **Check Service Status**

   ```bash
   docker-compose exec ai-agent supervisorctl status
   ```

4. **Test Endpoints**
   ```bash
   curl http://localhost/health        # Health check
   curl http://localhost/v1/models     # API endpoint
   curl http://localhost/             # React app
   ```

## Performance Optimization

### Frontend Optimizations

- **Code Splitting**: Automatic chunk splitting by route/feature
- **Tree Shaking**: Unused code elimination
- **Asset Optimization**: Image and CSS optimization
- **Caching**: Proper cache headers for static assets

### Backend Optimizations

- **Static File Caching**: Nginx handles static files
- **API Caching**: Redis for API response caching
- **Compression**: Gzip compression for text assets

### Docker Optimizations

- **Multi-stage Build**: Smaller production image
- **Layer Caching**: Optimized Dockerfile layer order
- **Health Checks**: Proper container health monitoring

## Security Considerations

### Frontend Security

- **Content Security Policy**: Configured in Nginx
- **XSS Protection**: React's built-in protection
- **HTTPS**: SSL/TLS termination at Nginx

### Backend Security

- **CORS**: Proper CORS configuration
- **Rate Limiting**: API rate limiting in Nginx
- **Input Validation**: FastAPI request validation

## Monitoring

### Health Checks

- **Container Health**: Docker health checks
- **Service Health**: Supervisor process monitoring
- **Application Health**: Custom health endpoints

### Logging

- **Nginx Logs**: Access and error logs
- **FastAPI Logs**: Application logs
- **Supervisor Logs**: Process management logs

### Metrics

- **Performance**: Response times and throughput
- **Errors**: Error rates and types
- **Resources**: CPU, memory, and disk usage

## Maintenance

### Updates

1. **Frontend Updates**

   ```bash
   cd frontend
   npm update
   npm run build
   docker-compose build ai-agent
   ```

2. **Backend Updates**
   ```bash
   pip install -r requirements.txt
   docker-compose build ai-agent
   ```

### Backup

- **Database**: Regular database backups
- **Configuration**: Version control for configs
- **Logs**: Log rotation and archival

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review container logs
3. Test individual components
4. Check configuration files

## References

- [Vite Documentation](https://vitejs.dev/)
- [React Documentation](https://react.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [Docker Documentation](https://docs.docker.com/)
