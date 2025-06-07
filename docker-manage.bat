@echo off

if "%1"=="build" (
    echo Building Docker containers...
    docker-compose build
    goto :eof
)

if "%1"=="up" (
    echo Starting services...
    docker-compose up -d
    echo Backend available at: http://localhost:8000
    echo MLflow UI available at: http://localhost:5000
    goto :eof
)

if "%1"=="down" (
    echo Stopping services...
    docker-compose down
    goto :eof
)

if "%1"=="logs" (
    if "%2"=="" (
        set service=backend
    ) else (
        set service=%2
    )
    echo Showing logs for %service%...
    docker-compose logs -f %service%
    goto :eof
)

if "%1"=="shell" (
    echo Opening shell in backend container...
    docker-compose exec backend /bin/bash
    goto :eof
)

if "%1"=="test" (
    echo Running tests in container...
    docker-compose exec backend python -m pytest tests/ -v
    goto :eof
)

if "%1"=="lint" (
    echo Running linting...
    docker-compose exec backend black src/ tests/ scripts/
    docker-compose exec backend flake8 src/ tests/ scripts/
    goto :eof
)

if "%1"=="install" (
    echo Installing new dependencies...
    docker-compose exec backend pip install %2
    goto :eof
)

if "%1"=="restart" (
    echo Restarting services...
    docker-compose restart
    goto :eof
)

echo Usage: %0 {build^|up^|down^|logs^|shell^|test^|lint^|install^|restart}
echo.
echo Commands:
echo   build    - Build Docker containers
echo   up       - Start all services
echo   down     - Stop all services
echo   logs     - Show logs (optional: service name)
echo   shell    - Open shell in backend container
echo   test     - Run tests
echo   lint     - Run code formatting and linting
echo   install  - Install a new package
echo   restart  - Restart services
