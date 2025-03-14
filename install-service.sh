#!/bin/bash

# Copy service file to systemd
sudo cp crypto-bot.service /etc/systemd/system/

# Reload systemd daemon
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable crypto-bot.service

# Start the service
sudo systemctl start crypto-bot.service

echo "Service installed and started!"
echo "Check status with: sudo systemctl status crypto-bot.service"
echo "View logs with: sudo journalctl -u crypto-bot.service -f"
