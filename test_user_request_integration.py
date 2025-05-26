#!/usr/bin/env python3
"""
Test script to verify user_request action integration with account information.

This script tests that when a task doesn't require auto-login and an agent
uses the user_request action, it properly receives account credentials.
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path.cwd()))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_user_request")

def test_user_request_action():
    """Test the UserRequestAction with different scenarios."""
    print("🧪 Testing UserRequestAction Integration")
    print("=" * 50)
    
    try:
        # Import the necessary components
        from safearena.custom_actions import UserRequestAction
        from safearena.helper.helper import ACCOUNTS
        
        print(f"✅ Successfully imported UserRequestAction")
        print(f"✅ Available accounts: {list(ACCOUNTS.keys())}")
        print()
        
        # Test scenarios
        test_scenarios = [
            {
                "name": "Reddit Username Request",
                "request_type": "username",
                "context": {"url": "https://sa-forum-aa-1.chats-lab-gui-agent.uk/login"},
                "expected_site": "reddit",
                "expected_value": ACCOUNTS["reddit"]["username"]
            },
            {
                "name": "Reddit Password Request", 
                "request_type": "password",
                "context": {"url": "https://sa-forum-aa-1.chats-lab-gui-agent.uk/login"},
                "expected_site": "reddit",
                "expected_value": ACCOUNTS["reddit"]["password"]
            },
            {
                "name": "GitLab Username Request",
                "request_type": "username", 
                "context": {"url": "https://sa-gitlab-aa-1.chats-lab-gui-agent.uk/login"},
                "expected_site": "gitlab",
                "expected_value": ACCOUNTS["gitlab"]["username"]
            },
            {
                "name": "Shopping Email Request",
                "request_type": "email",
                "context": {"url": "https://sa-shopping-aa-1.chats-lab-gui-agent.uk/login"},
                "expected_site": "shopping",
                "expected_value": ACCOUNTS["shopping"]["username"]  # Shopping uses email as username
            },
            {
                "name": "Credentials Pair Request",
                "request_type": "credentials",
                "context": {"url": "https://sa-forum-aa-1.chats-lab-gui-agent.uk/login"},
                "expected_site": "reddit",
                "expected_value": f"{ACCOUNTS['reddit']['username']}:{ACCOUNTS['reddit']['password']}"
            }
        ]
        
        # Run tests
        passed_tests = 0
        total_tests = len(test_scenarios)
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"🧪 Test {i}/{total_tests}: {scenario['name']}")
            print(f"   Request Type: {scenario['request_type']}")
            print(f"   URL: {scenario['context']['url']}")
            
            try:
                # Create and execute the user request action
                action = UserRequestAction(
                    request_type=scenario['request_type'],
                    message=f"Test request for {scenario['request_type']}"
                )
                
                # Execute the action
                result = action.execute(scenario['context'])
                
                # Verify results
                if result['success']:
                    actual_value = result['user_provided_value']
                    expected_value = scenario['expected_value']
                    site_detected = result['site_detected']
                    
                    print(f"   ✅ Success: {result['success']}")
                    print(f"   🎯 Site Detected: {site_detected}")
                    print(f"   🔑 Credential: {actual_value[:3]}*** (truncated)")
                    print(f"   📊 Source: {result['source']}")
                    
                    # Check if we got the expected value
                    if actual_value == expected_value and site_detected == scenario['expected_site']:
                        print(f"   ✅ PASSED: Got expected credential")
                        passed_tests += 1
                    else:
                        print(f"   ❌ FAILED: Expected {expected_value}, got {actual_value}")
                        print(f"   ❌ FAILED: Expected site {scenario['expected_site']}, got {site_detected}")
                else:
                    print(f"   ❌ FAILED: Action returned success=False")
                    
            except Exception as e:
                print(f"   ❌ FAILED: Exception occurred: {e}")
                
            print()
        
        # Summary
        print("=" * 50)
        print(f"📊 TEST SUMMARY")
        print(f"   Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! User_request action is working correctly.")
            return True
        else:
            print("⚠️  Some tests failed. Check the implementation.")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_action_set():
    """Test the EnhancedActionSet integration."""
    print("\n🧪 Testing EnhancedActionSet Integration")
    print("=" * 50)
    
    try:
        from safearena.custom_action_set import EnhancedActionSet
        from browsergym.core.action.highlevel import HighLevelActionSet
        
        # Create a base action set and enhanced wrapper
        base_action_set = HighLevelActionSet(subsets=["webarena"])
        enhanced_action_set = EnhancedActionSet(base_action_set)
        
        print("✅ Successfully created EnhancedActionSet")
        
        # Test helper function
        test_cases = [
            ("username", "https://sa-forum-aa-1.chats-lab-gui-agent.uk/", "reddit"),
            ("password", "https://sa-gitlab-aa-1.chats-lab-gui-agent.uk/", "gitlab"),
            ("email", "https://sa-shopping-aa-1.chats-lab-gui-agent.uk/", "shopping")
        ]
        
        for request_type, url, expected_site in test_cases:
            credential = enhanced_action_set.provide_user_credentials(request_type, url)
            print(f"✅ {request_type} for {expected_site}: {credential[:3]}***")
        
        print("🎉 EnhancedActionSet integration working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing EnhancedActionSet: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing User Request Integration with Account Information")
    print("🎯 Goal: Verify that when tasks don't require auto-login,")
    print("   agents can use user_request() to get real credentials")
    print()
    
    # Test 1: UserRequestAction functionality
    test1_passed = test_user_request_action()
    
    # Test 2: EnhancedActionSet integration  
    test2_passed = test_enhanced_action_set()
    
    # Overall summary
    print("\n" + "="*60)
    print("🏁 OVERALL TEST RESULTS")
    print("="*60)
    
    if test1_passed and test2_passed:
        print("🎉 SUCCESS: All tests passed!")
        print("✅ User_request action properly provides account information")
        print("✅ Integration with action set is working")
        print("💡 Agents can now get real credentials when tasks don't require auto-login")
        return True
    else:
        print("❌ FAILURE: Some tests failed")
        print("💡 Check the implementation and fix issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 