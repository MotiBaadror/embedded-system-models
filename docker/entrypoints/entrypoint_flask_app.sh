#!/bin/bash

# Set the environment variables
export FLASK_APP=app.py
#export FLASK_ENV=production


# Run the Flask app
flask run --host 0.0.0.0 --port 8000 --debug