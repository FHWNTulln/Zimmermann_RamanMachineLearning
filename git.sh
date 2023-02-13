#!/bin/bash

# Get commit message from user input
read -p "Enter commit message: " message

# Add all changes and commit with user message
git add -A
git commit -m "$message"

# Push changes to remote repository
git push
