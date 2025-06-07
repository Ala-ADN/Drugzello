#!/bin/bash

# Docker management script for Drugzello backend

case "$1" in
    "build")
        echo "Building Docker containers..."
        docker-compose build
        ;;
    "up")
        echo "Starting services..."
        docker-compose up -d
        echo "Backend available at: http://localhost:8000"
        echo "MLflow UI available at: http://localhost:5000"
        ;;
    "down")
        echo "Stopping services..."
        docker-compose down
        ;;
    "logs")
        service=${2:-backend}
        echo "Showing logs for $service..."
        docker-compose logs -f $service
        ;;
    "shell")
        echo "Opening shell in backend container..."
        docker-compose exec backend /bin/bash
        ;;
    "test")
        echo "Running tests in container..."
        docker-compose exec backend python -m pytest tests/ -v
        ;;
    "lint")
        echo "Running linting..."
        docker-compose exec backend black src/ tests/ scripts/
        docker-compose exec backend flake8 src/ tests/ scripts/
        ;;
    "install")
        echo "Installing new dependencies..."
        docker-compose exec backend pip install $2
        ;;
    "restart")
        echo "Restarting services..."
        docker-compose restart
        ;;
    *)
        echo "Usage: $0 {build|up|down|logs|shell|test|lint|install|restart}"
        echo ""
        echo "Commands:"
        echo "  build    - Build Docker containers"
        echo "  up       - Start all services"
        echo "  down     - Stop all services"
        echo "  logs     - Show logs (optional: service name)"
        echo "  shell    - Open shell in backend container"
        echo "  test     - Run tests"
        echo "  lint     - Run code formatting and linting"
        echo "  install  - Install a new package"
        echo "  restart  - Restart services"
        exit 1
        ;;
esac
