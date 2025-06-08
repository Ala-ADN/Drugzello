#!/bin/bash

# Script to generate self-signed SSL certificates for development
# For production, use Let's Encrypt or Azure-provided certificates

set -e  # Exit on any error

# Configuration
DOMAIN=${1:-"drugzello.local"}
SSL_DIR="./backend/nginx/ssl"
DAYS_VALID=365

# Print header
echo "====================================="
echo "Drugzello SSL Certificate Generator"
echo "Domain: $DOMAIN"
echo "====================================="
echo

# Create SSL directory if it doesn't exist
mkdir -p $SSL_DIR

# Generate a private key and certificate signing request
echo "Generating private key and CSR..."
openssl req -newkey rsa:2048 -nodes \
  -keyout $SSL_DIR/server.key \
  -out $SSL_DIR/server.csr \
  -subj "/CN=$DOMAIN/O=Drugzello/C=US"

# Generate self-signed certificate
echo "Generating self-signed certificate..."
openssl x509 -req -days $DAYS_VALID \
  -in $SSL_DIR/server.csr \
  -signkey $SSL_DIR/server.key \
  -out $SSL_DIR/server.crt \
  -extfile <(echo -e "subjectAltName=DNS:$DOMAIN,DNS:www.$DOMAIN")

# Clean up CSR
rm $SSL_DIR/server.csr

# Set proper permissions
chmod 600 $SSL_DIR/server.key
chmod 644 $SSL_DIR/server.crt

echo
echo "SSL certificate generated successfully!"
echo "Certificate: $SSL_DIR/server.crt"
echo "Private Key: $SSL_DIR/server.key"
echo "Valid for: $DAYS_VALID days"
