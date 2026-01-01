# FlyLLM Monitoring Stack

This directory contains a ready-to-use Prometheus + Grafana setup for monitoring your FlyLLM applications.

## Quick Start

1. **Start the monitoring stack:**
   ```bash
   cd monitoring
   ./start.sh up
   ```

2. **Run your FlyLLM application** (or the example):
   ```bash
   OPENAI_API_KEY=your-key cargo run --example metrics --features metrics
   ```

3. **Open Grafana:**
   - URL: http://localhost:3000
   - Login: `admin` / `admin`
   - The FlyLLM dashboard is pre-loaded and ready to use

## Requirements

- Docker
- Docker Compose

## Services

| Service    | Port | URL                    |
|------------|------|------------------------|
| Grafana    | 3000 | http://localhost:3000  |
| Prometheus | 9091 | http://localhost:9091  |

Your application should expose metrics on port **9090** (the default in the example).

## Configuring Your Application

Add `metrics-exporter-prometheus` to your project:

```toml
[dependencies]
flyllm = { version = "0.4.1", features = ["metrics"] }
metrics-exporter-prometheus = "0.16"
```

Then set up the Prometheus exporter in your application:

```rust
use metrics_exporter_prometheus::PrometheusBuilder;

// Expose metrics at http://0.0.0.0:9090/metrics
PrometheusBuilder::new()
    .with_http_listener(([0, 0, 0, 0], 9090))
    .install()
    .expect("Failed to install Prometheus exporter");

// Optional: describe metrics for better Prometheus discovery
flyllm::describe_metrics();

// Your FlyLLM manager setup...
let manager = LlmManager::builder()
    // ...
    .build()
    .await?;
```

See `examples/metrics.rs` for a complete working example.

## Commands

```bash
./start.sh up      # Start the stack
./start.sh down    # Stop the stack
./start.sh logs    # View logs
./start.sh status  # Check container status
./start.sh clean   # Stop and remove all data (including Prometheus history)
```

## Customizing Prometheus Targets

Edit `prometheus.yml` to add or modify scrape targets:

```yaml
scrape_configs:
  - job_name: 'flyllm'
    static_configs:
      - targets: ['host.docker.internal:9090']
```

If your application runs in Docker, use the container name instead of `host.docker.internal`.

## Available Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `llm_requests_total` | Counter | Total LLM requests |
| `llm_request_duration_seconds` | Summary | Request latency (with quantiles) |
| `llm_tokens_prompt_total` | Counter | Prompt tokens consumed |
| `llm_tokens_completion_total` | Counter | Completion tokens generated |
| `llm_errors_total` | Counter | Errors by type |
| `llm_retries_total` | Counter | Retry attempts |
| `llm_rate_limits_total` | Counter | Rate limit events |

## Dashboard Panels

The pre-loaded Grafana dashboard includes:

**Summary Stats (top row):**
- Total Requests
- Total Tokens
- Total Errors
- Average Latency

**Time Series Charts:**
- Request Rate (by provider/model)
- Request Latency (p50/p95)
- Token Usage (prompt vs completion)
- Error Rate by Type
- Retry Rate
- Rate Limit Events
- Requests by Task

The dashboard supports filtering by Provider, Model, and Task using the dropdowns at the top.
