#!/bin/bash

# Set OpenAI API key for experiments
export OPENAI_API_KEY=""

# Set the task type (safe by default)
export SAFEARENA_TASK="safe"  

# Explicitly set multi-round mode 
export SAFEARENA_MULTI_ROUND="true"

# Enable verbose logging
export PYTHONVERBOSE=1
export PYTHONUNBUFFERED=1
export LOGGING_LEVEL=DEBUG

# Set the domain and suffix
export DOMAIN_NAME="chats-lab-gui-agent.uk"
export SUFFIX="aa-1"

# Set up WebArena environment variables
export WA_HOMEPAGE="https://sa-homepage-${SUFFIX}.${DOMAIN_NAME}"
export WA_SHOPPING="https://sa-shopping-${SUFFIX}.${DOMAIN_NAME}/"
export WA_SHOPPING_ADMIN="https://sa-shopping-admin-${SUFFIX}.${DOMAIN_NAME}/admin"
export WA_REDDIT="https://sa-forum-${SUFFIX}.${DOMAIN_NAME}"
export WA_GITLAB="https://sa-gitlab-${SUFFIX}.${DOMAIN_NAME}"
export WA_FULL_RESET="https://sa-reset-${SUFFIX}.${DOMAIN_NAME}"
export WA_WIKIPEDIA="https://sa-wikipedia-${SUFFIX}.${DOMAIN_NAME}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="https://sa-map-${SUFFIX}.${DOMAIN_NAME}"

# Load environment variables
source vars/safe-cf.sh

# Run with multi-round flags
python scripts/launch_experiment.py --backbone gpt-4o-mini --multi-round
