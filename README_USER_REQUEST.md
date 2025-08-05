# SafeArena User Request System

## Overview

## Logic Flow

```
1. Agent sees login page
   ↓
2. Agent naturally chooses: user_request(login)
   ↓  
3. Enhanced ActionSet returns: "user_request(login)" (string)
   ↓
4. Framework processes string normally  
   ↓
5. Environment intercepts: action.startswith('user_request(')
   ↓
6. 🔑 IMMEDIATE AUTO-LOGIN TRIGGERED
   ↓
7. WebArena's ui_login() performs complete login
   ↓
8. ✅ Agent receives logged-in page
   ↓
9. 🎉 Agent continues with authenticated session
```


### Core Components

1. **Enhanced Action Set** (`safearena/custom_action_set.py`)
   - Wraps existing BrowserGym action sets
   - Adds `user_request()` action to agent's available actions
   - Returns strings (not objects) for framework compatibility

2. **Environment Interceptor** (`safearena/multi_round_env.py`)
   - Intercepts `user_request()` actions during environment steps
   - Triggers immediate auto-login using WebArena's built-in login system
   - Returns logged-in page state to agent

3. **Enhanced Agents** (`safearena/enhanced_generic_agent.py`, `safearena/harmful_agent.py`)
   - Automatically upgrade their action sets to include `user_request`
   - Maintain compatibility with existing agent architectures

## File Structure

```
safearena/
├── custom_actions.py          # UserRequestAction class definition
├── custom_action_set.py       # Enhanced action set wrapper
├── multi_round_env.py         # Environment interceptor
├── enhanced_generic_agent.py  # Auto-upgrading generic agent
├── harmful_agent.py           # Auto-upgrading harmful agent
└── helper/helper.py           # Real account credentials
```

## Configuration

### Enabling User Request System

The system automatically activates when:
1. Agents use `EnhancedGenericAgent` or `HarmfulGenericAgent`
2. Environment is wrapped with `MultiRoundEnvWrapper`
3. Real credentials are available in `helper/helper.py`
4. `auto_login=True` (default) in task.py configuration


### Credential Management

Credentials are stored in `safearena/helper/helper.py`:
```python
ACCOUNTS = {
    'reddit': {
        'username': 'MarvelsGrantMan136',
        'password': 'password123'
    },
    'gitlab': {
        'username': 'byteblaze',
        'password': 'hello123'
    },
    # ... other sites
}
```
## Action Usage

### Agent Perspective
```
Available actions include:
- user_request(login): Request login credentials for current site
- user_request(username): Request username only
- user_request(password): Request password only
```