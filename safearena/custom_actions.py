"""
Custom actions for SafeArena.

This module defines additional actions beyond those provided by BrowserGym.
"""

from typing import Any, Dict, Optional, Union
import logging

# Don't import AbstractAction since it doesn't exist
# from browsergym.core.action.base import AbstractAction, ActionError

logger = logging.getLogger(__name__)

# Define our own error class
class ActionError(Exception):
    """Error raised when an action fails to execute."""
    pass


# Define our own action base class
class Action:
    """Base class for custom actions."""
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the action.
        
        Args:
            context: The execution context containing page and other information.
            
        Returns:
            A dictionary with the result of the action.
            
        Raises:
            ActionError: If there's an issue executing the action.
        """
        raise NotImplementedError("Subclasses must implement this method")


class UserRequestAction(Action):
    """
    Action to request information from the user when login credentials or similar sensitive data is needed.
    
    This action should be used when the agent encounters input fields for usernames, passwords,
    or other sensitive information that requires user input.
    """
    
    def __init__(self, request_type: str, message: Optional[str] = None):
        """
        Initialize a UserRequestAction.
        
        Args:
            request_type: The type of information being requested (e.g., 'username', 'password', etc.)
            message: An optional custom message to display to the user
        """
        self.request_type = request_type
        self.message = message or f"Please provide your {request_type}"
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the action, which in this case means prompting the user for input.
        
        Args:
            context: The execution context containing page and other information.
            
        Returns:
            A dictionary with the result of the action.
            
        Raises:
            ActionError: If there's an issue executing the action.
        """
        try:
            # Log the request for debugging
            logger.info(f"User request action executed: {self.request_type} - {self.message}")
            
            # This would typically interface with a UI component to get user input
            # For now, we'll just return that the request was made
            
            # In a real implementation, this might wait for user input
            # and store it for later use by the agent
            
            return {
                "success": True,
                "message": f"Requested {self.request_type} from user: {self.message}",
                "request_type": self.request_type
            }
        except Exception as e:
            raise ActionError(f"Failed to execute UserRequestAction: {str(e)}")
    
    def __str__(self) -> str:
        return f"user_request({self.request_type}, {self.message})"


def create_user_request_action(request_type: str, message: Optional[str] = None) -> UserRequestAction:
    """
    Create a UserRequestAction.
    
    Args:
        request_type: The type of information being requested (e.g., 'username', 'password')
        message: An optional custom message to display to the user
        
    Returns:
        A UserRequestAction instance.
    """
    return UserRequestAction(request_type, message) 