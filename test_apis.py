#!/usr/bin/env python3
"""
Test script to verify API connections
Run this before using the main script to ensure all APIs are working
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import main script modules
sys.path.insert(0, str(Path(__file__).parent))

from ai_response_generator import (
    Config, ClaudeClient, ChatGPTClient, 
    GeminiClient, GrokClient
)


async def test_api(client_name: str, client, test_prompt: str, model_info: str = None):
    """Test a single API client"""
    print(f"\nTesting {client_name}...")
    if model_info:
        print(f"   Model: {model_info}")
    try:
        response = await client.generate_response(test_prompt)
        if response.startswith("Error"):
            print(f"❌ {client_name}: {response}")
            return False
        else:
            print(f"✅ {client_name}: Success!")
            print(f"   Response preview: {response[:100]}...")
            return True
    except Exception as e:
        print(f"❌ {client_name}: {str(e)}")
        return False


async def main():
    """Test all configured APIs"""
    print("AI Response Generator - API Test")
    print("=" * 40)
    
    # Load configuration
    config = Config()
    
    print("\nConfiguration Summary:")
    print("-" * 40)
    for key, value in config.config.items():
        if key.endswith("_api_key") and value:
            print(f"✓ {key}: Configured")
        elif key.endswith("_model") and value:
            print(f"  {key}: {value}")
    print("-" * 40)
    
    # Test prompt
    test_prompt = "Reply with a single sentence confirming you received this test message."
    
    # Track results
    results = {}
    
    # Test Claude
    if config.config.get("claude_api_key"):
        model = config.config.get("claude_model", "claude-3-sonnet-20240229")
        client = ClaudeClient(config.config["claude_api_key"], model)
        results["Claude"] = await test_api("Claude", client, test_prompt, model)
    else:
        print("\n⚠️  Claude: No API key configured")
        results["Claude"] = None
    
    # Test ChatGPT
    if config.config.get("openai_api_key"):
        model = config.config.get("openai_model", "gpt-4-turbo-preview")
        client = ChatGPTClient(config.config["openai_api_key"], model)
        results["ChatGPT"] = await test_api("ChatGPT", client, test_prompt, model)
    else:
        print("\n⚠️  ChatGPT: No API key configured")
        results["ChatGPT"] = None
    
    # Test Gemini
    if config.config.get("gemini_api_key"):
        model = config.config.get("gemini_model", "gemini-pro")
        client = GeminiClient(config.config["gemini_api_key"], model)
        results["Gemini"] = await test_api("Gemini", client, test_prompt, model)
    else:
        print("\n⚠️  Gemini: No API key configured")
        results["Gemini"] = None
    
    # Test Grok
    if config.config.get("grok_api_key"):
        model = config.config.get("grok_model", "grok-beta")
        client = GrokClient(
            config.config["grok_api_key"],
            config.config.get("grok_api_url", "https://api.x.ai/v1/chat/completions"),
            model
        )
        results["Grok"] = await test_api("Grok", client, test_prompt, model)
    else:
        print("\n⚠️  Grok: No API key configured")
        results["Grok"] = None
    
    # Summary
    print("\n" + "=" * 40)
    print("Summary:")
    print("-" * 40)
    
    working = sum(1 for v in results.values() if v is True)
    configured = sum(1 for v in results.values() if v is not None)
    total = len(results)
    
    for api, status in results.items():
        if status is True:
            print(f"✅ {api}: Working")
        elif status is False:
            print(f"❌ {api}: Failed")
        else:
            print(f"⚠️  {api}: Not configured")
    
    print("-" * 40)
    print(f"APIs working: {working}/{configured} configured ({total} total)")
    
    if working == 0:
        print("\n❌ No APIs are working. Please check your configuration.")
        return False
    elif working < configured:
        print("\n⚠️  Some APIs are not working. Check the errors above.")
        return True
    else:
        print("\n✅ All configured APIs are working correctly!")
        return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
