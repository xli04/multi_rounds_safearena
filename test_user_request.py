#!/usr/bin/env python3
"""
Simplified test for the User Request action.

This script focuses only on testing if the User Request action works
without using an agent.
"""

import os
import logging
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger("test_user_request")

# Import necessary modules
from safearena.custom_actions import UserRequestAction, Action, ActionError

def test_user_request_action():
    """Test the basic functionality of the UserRequestAction class"""
    logger.info("Testing UserRequestAction...")
    
    # Test creating different user request actions
    test_cases = [
        ("username", None),
        ("password", "Please enter your password"),
        ("email", "We need your email to continue"),
        ("credentials", "Please provide your login information")
    ]
    
    for request_type, message in test_cases:
        # Create the action
        if message:
            action = UserRequestAction(request_type, message)
            logger.info(f"Created action: user_request({request_type}, {message})")
        else:
            action = UserRequestAction(request_type)
            logger.info(f"Created action: user_request({request_type})")
        
        # Check string representation
        logger.info(f"String representation: {action}")
        
        # Test executing the action
        try:
            # Empty context for testing
            context = {}
            result = action.execute(context)
            logger.info(f"Execution result: {result}")
            
            # Check expected values in result
            assert result["success"] == True, "Expected success to be True"
            assert request_type in result["message"], f"Expected message to contain {request_type}"
            assert result["request_type"] == request_type, f"Expected request_type to be {request_type}"
            
            logger.info(f"✅ Test passed for user_request({request_type})")
        except Exception as e:
            logger.error(f"❌ Test failed for user_request({request_type}): {e}")
    
    logger.info("All UserRequestAction tests completed")

def main():
    logger.info("Starting simplified User Request action test")
    
    try:
        # Test the UserRequestAction directly
        test_user_request_action()
        
        logger.info("All tests completed successfully!")
    except Exception as e:
        logger.error(f"Error in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 