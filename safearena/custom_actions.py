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
        Execute the action, which in this case means providing account credentials for tasks that don't require auto-login.
        
        Args:
            context: The execution context containing page and other information.
            
        Returns:
            A dictionary with the result of the action.
            
        Raises:
            ActionError: If there's an issue executing the action.
        """
        try:
            # Enhanced logging for debugging
            logger.info(f"🔑 USER_REQUEST action triggered: {self.request_type}")
            logger.info(f"📝 Message: {self.message}")
            
            # Import the real account credentials
            try:
                from .helper.helper import ACCOUNTS
                logger.info("✅ Successfully imported ACCOUNTS from helper.helper")
                logger.info(f"🏪 Available sites: {list(ACCOUNTS.keys())}")
            except ImportError as e:
                logger.error(f"❌ Could not import ACCOUNTS: {e}")
                logger.warning("🔄 Using fallback credentials")
                ACCOUNTS = {}
            
            # Determine which site we're on based on context
            current_url = context.get('url', '') if context else ''
            logger.info(f"🌐 Current URL: {current_url}")
            
            # Enhanced site detection
            site_type = self._detect_site_from_context(context, current_url)
            logger.info(f"🎯 Detected site type: {site_type}")
            
            # Get real credentials or use smart defaults
            credential_value = self._get_credential_for_site(site_type, self.request_type, ACCOUNTS)
            
            # Enhanced response with more information
            response = {
                "success": True,
                "action_type": "user_request",
                "request_type": self.request_type,
                "message": f"✅ Providing {self.request_type} for {site_type}",
                "user_provided_value": credential_value,
                "credential_value": credential_value,  # Also provide as credential_value for compatibility
                "site_detected": site_type,
                "current_url": current_url,
                "source": "real_accounts" if (site_type in ACCOUNTS and self.request_type.lower() in ['username', 'password', 'user', 'pass', 'pwd', 'login', 'email']) else "fallback",
                "simulated": True,  # Flag to indicate this was automated
                "accounts_available": list(ACCOUNTS.keys()),
                "timestamp": context.get('timestamp', 'unknown')
            }
            
            # Log the successful response
            logger.info(f"🎉 USER_REQUEST successful: Provided {self.request_type} for {site_type}")
            logger.info(f"📊 Credential source: {response['source']}")
            
            return response
            
        except Exception as e:
            error_msg = f"Failed to execute UserRequestAction: {str(e)}"
            logger.error(f"❌ {error_msg}")
            raise ActionError(error_msg)
    
    def _detect_site_from_context(self, context: Dict[str, Any], current_url: str) -> str:
        """
        Enhanced site detection with better logging and fallback mechanisms.
        
        Args:
            context: The execution context
            current_url: The current URL
            
        Returns:
            str: The detected site type (reddit, gitlab, shopping, etc.)
        """
        logger.info(f"🔍 Detecting site from URL: {current_url}")
        
        # Check URL patterns to detect the site
        if 'forum' in current_url or 'reddit' in current_url or '/f/' in current_url:
            logger.info("🎯 Detected: Reddit (forum patterns)")
            return 'reddit'
        elif 'gitlab' in current_url or 'byteblaze' in current_url:
            logger.info("🎯 Detected: GitLab")
            return 'gitlab'  
        elif 'shopping' in current_url or 'magento' in current_url:
            # Distinguish between regular shopping and admin
            if 'admin' in current_url:
                logger.info("🎯 Detected: Shopping Admin")
                return 'shopping_admin'
            else:
                logger.info("🎯 Detected: Shopping")
                return 'shopping'
        
        # Try to detect from environment variables as fallback
        import os
        try:
            if 'WA_REDDIT' in os.environ and os.environ['WA_REDDIT'] in current_url:
                logger.info("🎯 Detected: Reddit (from WA_REDDIT env var)")
                return 'reddit'
            elif 'WA_GITLAB' in os.environ and os.environ['WA_GITLAB'] in current_url:
                logger.info("🎯 Detected: GitLab (from WA_GITLAB env var)")
                return 'gitlab'
            elif 'WA_SHOPPING' in os.environ:
                if 'WA_SHOPPING_ADMIN' in os.environ and os.environ['WA_SHOPPING_ADMIN'] in current_url:
                    logger.info("🎯 Detected: Shopping Admin (from WA_SHOPPING_ADMIN env var)")
                    return 'shopping_admin'
                elif os.environ['WA_SHOPPING'] in current_url:
                    logger.info("🎯 Detected: Shopping (from WA_SHOPPING env var)")
                    return 'shopping'
        except Exception as e:
            logger.debug(f"⚠️ Error checking environment variables: {e}")
        
        # If no URL provided or detection fails, default to reddit (most common)
        if not current_url:
            logger.info("⚠️ No URL provided, defaulting to reddit for credentials")
            return 'reddit'
        
        # Default fallback
        logger.warning(f"⚠️ Could not detect site from URL: {current_url}, defaulting to reddit")
        return 'reddit'  # Default to reddit since it's most commonly used
    
    def _get_credential_for_site(self, site_type: str, request_type: str, accounts: Dict) -> str:
        """
        Enhanced credential retrieval with better logging and error handling.
        
        Args:
            site_type: The detected site (reddit, gitlab, shopping, etc.)
            request_type: The type of credential requested (username, password, etc.)
            accounts: The ACCOUNTS dictionary with real credentials
            
        Returns:
            str: The appropriate credential value
        """
        logger.info(f"🔑 Getting {request_type} for {site_type}")
        
        # Map request types to account keys
        if request_type.lower() in ['username', 'user', 'login', 'email']:
            credential_key = 'username'
        elif request_type.lower() in ['password', 'pass', 'pwd']:
            credential_key = 'password'
        elif request_type.lower() == 'credentials':
            # Return username:password format
            if site_type in accounts:
                username = accounts[site_type].get('username', 'unknown_user')
                password = accounts[site_type].get('password', 'unknown_pass')
                result = f"{username}:{password}"
                logger.info(f"✅ Providing credentials pair for {site_type}")
                return result
            else:
                logger.warning(f"⚠️ No accounts found for {site_type}, using fallback credentials")
                return "demo_user:demo_password"
        else:
            # For other request types, use fallback
            credential_key = 'username'  # Default to username
            logger.info(f"🔄 Unknown request type '{request_type}', defaulting to username")
        
        # Get the real credential if available
        if site_type in accounts and credential_key in accounts[site_type]:
            real_credential = accounts[site_type][credential_key]
            logger.info(f"✅ Using REAL {credential_key} for {site_type}: {real_credential[:3]}***")
            return real_credential
        
        # Fallback to demo data if real credentials not available
        fallback_responses = {
            "username": "demo_user",
            "password": "demo_password",
            "email": "demo@example.com", 
            "name": "Demo User",
            "phone_number": "+1-555-0123",
            "address": "123 Demo Street, Demo City, DC 12345",
            "payment_info": "**** **** **** 1234",
            "birthday": "01/01/1990",
            "social_security": "***-**-1234"
        }
        
        fallback_value = fallback_responses.get(request_type.lower(), f"demo_{request_type}")
        logger.warning(f"⚠️ Using FALLBACK {request_type} for {site_type}: {fallback_value}")
        return fallback_value
    
    def __str__(self) -> str:
        return f"user_request({self.request_type})"
    
    def __repr__(self) -> str:
        return f"UserRequestAction(request_type='{self.request_type}', message='{self.message}')"


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