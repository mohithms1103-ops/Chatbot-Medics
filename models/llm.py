from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import streamlit as st # type: ignore
from groq import Groq # type: ignore
import openai # type: ignore
import google.generativeai as genai # type: ignore
from config.config import config

class BaseLLM(ABC):
    """Abstract base class for LLM models"""
    
    @abstractmethod
    def generate_response(self, messages: List[Dict], **kwargs) -> str:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass

class GroqLLM(BaseLLM):
    """Groq LLM implementation"""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.model = model or config.DEFAULT_MODEL
        self.client = None
        
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                st.error(f"Failed to initialize Groq client: {e}")
    
    def is_available(self) -> bool:
        return self.client is not None and self.api_key is not None
    
    def generate_response(self, messages: List[Dict], **kwargs) -> str:
        """Generate response using Groq API"""
        if not self.is_available():
            return "Groq client not available. Please check your API key."
        
        try:
            # Extract parameters with defaults
            temperature = kwargs.get('temperature', config.TEMPERATURE)
            max_tokens = kwargs.get('max_tokens', config.MAX_TOKENS)
            model = kwargs.get('model', self.model)
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=config.TOP_P,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model
        
        if self.api_key:
            openai.api_key = self.api_key
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    def generate_response(self, messages: List[Dict], **kwargs) -> str:
        """Generate response using OpenAI API"""
        if not self.is_available():
            return "OpenAI client not available. Please check your API key."
        
        try:
            temperature = kwargs.get('temperature', config.TEMPERATURE)
            max_tokens = kwargs.get('max_tokens', config.MAX_TOKENS)
            model = kwargs.get('model', self.model)
            
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

class GeminiLLM(BaseLLM):
    """Google Gemini LLM implementation"""
    
    def __init__(self, api_key: str = None, model: str = "gemini-pro"):
        self.api_key = api_key or config.GOOGLE_API_KEY
        self.model = model
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
            except Exception as e:
                st.error(f"Failed to initialize Gemini client: {e}")
                self.client = None
        else:
            self.client = None
    
    def is_available(self) -> bool:
        return self.client is not None and self.api_key is not None
    
    def generate_response(self, messages: List[Dict], **kwargs) -> str:
        """Generate response using Gemini API"""
        if not self.is_available():
            return "Gemini client not available. Please check your API key."
        
        try:
            # Convert messages to Gemini format
            prompt = self._convert_messages_to_prompt(messages)
            
            temperature = kwargs.get('temperature', config.TEMPERATURE)
            max_tokens = kwargs.get('max_tokens', config.MAX_TOKENS)
            
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _convert_messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert OpenAI-style messages to Gemini prompt format"""
        prompt_parts = []
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"Human: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)

class LLMManager:
    """Manager class to handle multiple LLM providers"""
    
    def __init__(self):
        self.providers = {
            'groq': GroqLLM(),
            'openai': OpenAILLM(),
            'gemini': GeminiLLM()
        }
        self.default_provider = 'groq'
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers"""
        return [name for name, provider in self.providers.items() if provider.is_available()]
    
    def get_provider(self, provider_name: str = None) -> BaseLLM:
        """Get LLM provider by name"""
        if provider_name is None:
            provider_name = self.default_provider
        
        if provider_name not in self.providers:
            st.warning(f"Provider {provider_name} not found. Using default.")
            provider_name = self.default_provider
        
        return self.providers[provider_name]
    
    def generate_response(self, messages: List[Dict], provider: str = None, **kwargs) -> str:
        """Generate response using specified provider"""
        llm = self.get_provider(provider)
        
        if not llm.is_available():
            # Try to find an available provider
            available_providers = self.get_available_providers()
            if available_providers:
                llm = self.get_provider(available_providers[0])
                st.info(f"Using {available_providers[0]} as fallback provider.")
            else:
                return "No LLM providers available. Please check your API keys."
        
        return llm.generate_response(messages, **kwargs)

# Global LLM manager instance
llm_manager = LLMManager()