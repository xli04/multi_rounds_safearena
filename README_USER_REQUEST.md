# User Request Action for SafeArena

This feature adds a "User Request" action allowing agents to request information from users when encountering forms requiring sensitive data.

## Purpose

- Provides a safe way for agents to handle login forms and sensitive inputs
- Prevents agents from guessing credentials
- Enables interactive agent-user information exchange

## Implementation

- `UserRequestAction` class in `safearena/custom_actions.py`
- Custom `Action` base class and `ActionError` exception

## Action Format

```
user_request(username)  # Request username
user_request(password)  # Request password
user_request(info_type, custom message)  # Request with custom message
```

## Return Value

```python
{
    "success": True,
    "message": "Requested <type> from user: <message>",
    "request_type": "<type>"
}
```

## Usage in Agent Prompts

Include instructions like:

```
When encountering forms requiring sensitive information, use the user_request action:
- user_request(username)
- user_request(password, Please provide your password)
- user_request(email, We need your email address)
``` 