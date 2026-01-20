"""FastAPI application entry point for FrankAsset backend.

This module initializes the FastAPI application, includes routers,
configures CORS, sets up exception handlers, and starts the server.
"""

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

import sys
from pathlib import Path as PathLib

# Add current directory to path for imports
sys.path.insert(0, str(PathLib(__file__).resolve().parent))

from logger.custom_logger import CustomLogger
from exception.custom_exception import CustomException

logger = CustomLogger().get_logger(__file__)
from src.api.routes import router
from utils.config import config


# Create FastAPI app
app = FastAPI(
    title="FrankAsset Backend API",
    description="Agentic AI backend for brand classification using Google ADK",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers

@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException):
    """Handle CustomException instances.
    
    Args:
        request: FastAPI request object
        exc: CustomException instance
        
    Returns:
        JSONResponse: JSON error response
    """
    logger.error(
        "CustomException raised",
        error_message=str(exc),
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "detail": exc.error_message
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTPException instances.
    
    Args:
        request: FastAPI request object
        exc: HTTPException instance
        
    Returns:
        JSONResponse: JSON error response
    """
    logger.warning(
        "HTTPException raised",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions.
    
    Args:
        request: FastAPI request object
        exc: Exception instance
        
    Returns:
        JSONResponse: JSON error response
    """
    logger.error(
        "Unhandled exception",
        error_type=type(exc).__name__,
        error_message=str(exc),
        path=request.url.path,
        method=request.method,
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred"
        }
    )


# Include routers
app.include_router(router)


# Startup and shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup.
    
    This function is called when the FastAPI application starts.
    """
    try:
        logger.info(
            "FastAPI application starting",
            app_title=app.title,
            version=app.version
        )
        
        # Validate configuration
        try:
            config.validate()
            logger.info("Configuration validated successfully")
        except ValueError as e:
            logger.warning("Configuration validation warning", error=str(e))
        
        # Initialize database connection (will be created on first use)
        # The routes module handles lazy initialization
        
        logger.info("FastAPI application started successfully")
    
    except Exception as e:
        logger.error("Startup error", error=str(e), exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown.
    
    This function is called when the FastAPI application shuts down.
    """
    try:
        logger.info("FastAPI application shutting down")
        
        # Close database connections
        from src.api.routes import db_manager
        if db_manager:
            db_manager.close()
            logger.info("Database connections closed")
        
        logger.info("FastAPI application shutdown complete")
    
    except Exception as e:
        logger.error("Shutdown error", error=str(e), exc_info=True)


# Root endpoint

@app.get("/")
async def root():
    """Root endpoint.
    
    Returns:
        dict: API information
    """
    return {
        "message": "FrankAsset Backend API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health"
    }


# Health check endpoint

@app.get("/health")
async def health_check():
    """Health check endpoint.
    
    Returns:
        dict: Health status
    """
    try:
        # Check database connection
        from src.api.routes import get_db_manager
        db = get_db_manager()
        
        # Simple query to test connection
        db.conn.execute("SELECT 1")
        
        return {
            "status": "healthy",
            "database": "connected",
            "message": "All systems operational"
        }
    except Exception as e:
        logger.warning("Health check failed", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


# Run application

if __name__ == "__main__":
    """Run the FastAPI application using uvicorn.
    
    Usage:
        # Using Python directly (if venv is activated)
        python main.py
        
        # Using uv run (recommended with UV)
        uv run python main.py
        
        # Or run uvicorn directly
        uvicorn main:app --reload
        
        # Or using uv run with uvicorn
        uv run uvicorn main:app --reload
    """
    logger.info("Starting FastAPI server...")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload in development
        log_level="info"
    )
