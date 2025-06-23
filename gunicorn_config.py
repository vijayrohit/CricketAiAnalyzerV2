"""Gunicorn configuration for cricket analysis application."""

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
workers = 1
worker_class = "sync"
worker_connections = 1000
timeout = 120  # Increased timeout for video processing
keepalive = 2

# Restart workers
max_requests = 1000
max_requests_jitter = 50
preload_app = False

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"

# Process naming
proc_name = "cricket_analysis"

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
keyfile = None
certfile = None

# Application
reload = True
reload_engine = "auto"