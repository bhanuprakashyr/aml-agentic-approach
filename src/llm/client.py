"""
Unified LLM Client

Supports OpenAI, Anthropic, and Ollama with consistent interface.
"""

import os
import json
import requests
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Structured response from LLM."""
    content: str
    model: str
    provider: str
    usage: Optional[Dict] = None


class LLMClient:
    """
    Unified LLM client supporting multiple providers.
    
    Providers:
        - openai: GPT-4, GPT-4o, GPT-3.5-turbo
        - anthropic: Claude 3.5 Sonnet, Claude 3 Opus
        - ollama: Llama 3, Mistral, etc. (local)
    
    Usage:
        client = LLMClient(provider="openai")
        response = client.call("Analyze this transaction...")
        print(response.content)
    """
    
    def __init__(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ):
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set default models per provider
        self.default_models = {
            "openai": "gpt-4o",
            "anthropic": "claude-3-5-sonnet-20241022",
            "ollama": "llama3.2"
        }
        self.model = model or self.default_models.get(self.provider, "gpt-4o")
        
        # Validate provider
        if self.provider not in self.default_models:
            raise ValueError(f"Unknown provider: {provider}. Use: openai, anthropic, ollama")
    
    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None
    ) -> LLMResponse:
        """
        Send prompt to LLM and get response.
        
        Args:
            prompt: User message/prompt
            system_prompt: Optional system instruction
            model: Override default model
            
        Returns:
            LLMResponse with content and metadata
        """
        model = model or self.model
        
        if self.provider == "openai":
            return self._call_openai(prompt, system_prompt, model)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt, system_prompt, model)
        elif self.provider == "ollama":
            return self._call_ollama(prompt, system_prompt, model)
    
    def _call_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str
    ) -> LLMResponse:
        """Call OpenAI API."""
        try:
            from openai import OpenAI
            client = OpenAI()
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=model,
                provider="openai",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        except Exception as e:
            return LLMResponse(
                content=f"OpenAI Error: {str(e)}",
                model=model,
                provider="openai"
            )
    
    def _call_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str
    ) -> LLMResponse:
        """Call Anthropic Claude API."""
        try:
            import anthropic
            client = anthropic.Anthropic()
            
            kwargs = {
                "model": model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            
            response = client.messages.create(**kwargs)
            
            return LLMResponse(
                content=response.content[0].text,
                model=model,
                provider="anthropic",
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            )
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        except Exception as e:
            return LLMResponse(
                content=f"Anthropic Error: {str(e)}",
                model=model,
                provider="anthropic"
            )
    
    def _call_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: str
    ) -> LLMResponse:
        """Call local Ollama server."""
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                return LLMResponse(
                    content=data.get("response", "No response"),
                    model=model,
                    provider="ollama",
                    usage={
                        "eval_count": data.get("eval_count", 0),
                        "eval_duration": data.get("eval_duration", 0)
                    }
                )
            else:
                return LLMResponse(
                    content=f"Ollama Error: {response.status_code} - {response.text}",
                    model=model,
                    provider="ollama"
                )
        except requests.exceptions.ConnectionError:
            return LLMResponse(
                content="Error: Ollama not running. Start with: ollama serve",
                model=model,
                provider="ollama"
            )
        except Exception as e:
            return LLMResponse(
                content=f"Ollama Error: {str(e)}",
                model=model,
                provider="ollama"
            )
    
    def test_connection(self) -> bool:
        """Test if LLM provider is accessible."""
        response = self.call("Reply with 'OK' if you can read this.")
        return "OK" in response.content or "ok" in response.content.lower()
