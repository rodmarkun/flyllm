#!/bin/bash
# FlyLLM Monitoring Stack
# Starts Prometheus and Grafana for monitoring LLM operations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== FlyLLM Monitoring Stack ==="
echo ""

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check for Docker Compose
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

case "${1:-up}" in
    up|start)
        echo "Starting monitoring stack..."
        $COMPOSE_CMD up -d
        echo ""
        echo "Monitoring stack started!"
        echo ""
        echo "  Grafana:    http://localhost:3000  (admin/admin)"
        echo "  Prometheus: http://localhost:9091"
        echo ""
        echo "The FlyLLM dashboard is pre-loaded in Grafana."
        echo ""
        echo "Make sure your FlyLLM application exposes Prometheus metrics at port 9090."
        echo "See examples/metrics.rs for how to set this up."
        ;;
    down|stop)
        echo "Stopping monitoring stack..."
        $COMPOSE_CMD down
        echo "Monitoring stack stopped."
        ;;
    logs)
        $COMPOSE_CMD logs -f
        ;;
    status)
        $COMPOSE_CMD ps
        ;;
    clean)
        echo "Stopping and removing all data..."
        $COMPOSE_CMD down -v
        echo "Cleaned up."
        ;;
    *)
        echo "Usage: $0 {up|down|logs|status|clean}"
        echo ""
        echo "Commands:"
        echo "  up, start  - Start the monitoring stack"
        echo "  down, stop - Stop the monitoring stack"
        echo "  logs       - View container logs"
        echo "  status     - Show container status"
        echo "  clean      - Stop and remove all data"
        exit 1
        ;;
esac
